use std::sync::{Arc, Mutex, OnceLock};

use windows::Win32::Graphics::Direct3D12::*;

use crate::resource::HostTexture;

/// A host texture's states after the last submission.
pub(crate) struct TextureStates {
    count: OnceLock<u32>,
    committed: Mutex<Subresources<u32>>,
}

impl TextureStates {
    pub(crate) fn new(state: u32) -> Self {
        Self {
            count: OnceLock::new(),
            committed: Mutex::new(Subresources::Uniform(state)),
        }
    }
}

impl HostTexture {
    fn subresource_count(&self) -> u32 {
        *self.states.count.get_or_init(|| {
            // SAFETY: a plain query on the live host resource.
            let desc = unsafe { self.texture.GetDesc() };
            let array_size = if desc.Dimension == D3D12_RESOURCE_DIMENSION_TEXTURE3D {
                1 // The depth slices belong to a single subresource at each mip.
            } else {
                u32::from(desc.DepthOrArraySize)
            };
            let mut info = D3D12_FEATURE_DATA_FORMAT_INFO {
                Format: desc.Format,
                PlaneCount: 1,
            };
            if let Some(host) = super::host() {
                // SAFETY: a feature query with the structure matching its feature.
                let _ = unsafe {
                    host.device.CheckFeatureSupport(
                        D3D12_FEATURE_FORMAT_INFO,
                        (&raw mut info).cast(),
                        size_of::<D3D12_FEATURE_DATA_FORMAT_INFO>() as u32,
                    )
                };
            }
            u32::from(desc.MipLevels) * array_size * u32::from(info.PlaneCount.max(1))
        })
    }

    #[cfg(any(test, feature = "diagnostics"))]
    pub(crate) fn state(&self, subresource: u32) -> u32 {
        let states = self.states.committed.lock().unwrap_or_else(|e| e.into_inner());
        match &*states {
            Subresources::Uniform(state) => *state,
            Subresources::Split(states) => states[subresource as usize],
        }
    }

    /// Visits the selected committed states, using one whole-texture entry
    /// when the texture is uniform and the whole texture was requested.
    pub(super) fn visit_states(&self, subresource: u32, mut visit: impl FnMut(u32, u32)) {
        let states = self.states.committed.lock().unwrap_or_else(|e| e.into_inner());
        match &*states {
            Subresources::Uniform(state) => visit(subresource, *state),
            Subresources::Split(states)
                if subresource == D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES =>
            {
                for (index, &state) in states.iter().enumerate() {
                    visit(index as u32, state);
                }
            }
            Subresources::Split(states) => visit(subresource, states[subresource as usize]),
        }
    }
}

/// Most textures are used as a whole. Allocate individual entries only
/// when a command needs to distinguish their subresources.
enum Subresources<T> {
    Uniform(T),
    Split(Box<[T]>),
}

impl<T: Copy> Subresources<T> {
    fn split(&mut self, count: usize) -> &mut [T] {
        if let Self::Uniform(value) = self {
            *self = Self::Split(vec![*value; count].into_boxed_slice());
        }
        match self {
            Self::Split(states) => states,
            Self::Uniform(_) => unreachable!(),
        }
    }
}

#[derive(Clone, Copy)]
struct Use {
    first: u32,
    last: u32,
}

pub(super) struct RecordedTexture {
    pub(super) copy: Arc<HostTexture>,
    states: Subresources<Option<Use>>,
}

impl RecordedTexture {
    pub(super) fn new(copy: Arc<HostTexture>) -> Self {
        Self {
            copy,
            states: Subresources::Uniform(None),
        }
    }

    /// Records a state for the selected subresources. A first use names
    /// the state the prelude must provide, so no inline barrier is needed.
    pub(super) fn record(
        &mut self,
        subresource: u32,
        wanted: u32,
        mut emit: impl FnMut(&ID3D12Resource, u32, u32, u32),
    ) {
        let mut record = |index, state: &mut Option<Use>| {
            let used = state.get_or_insert(Use {
                first: wanted,
                last: wanted,
            });
            let before = std::mem::replace(&mut used.last, wanted);
            if before != wanted {
                emit(&self.copy.texture, index, before, wanted);
            }
        };
        if subresource == D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES {
            match &mut self.states {
                Subresources::Uniform(state) => record(subresource, state),
                Subresources::Split(states) => {
                    for (index, state) in states.iter_mut().enumerate() {
                        record(index as u32, state);
                    }
                }
            }
            return;
        }
        let count = self.copy.subresource_count();
        if subresource >= count {
            return;
        }
        if count == 1 && let Subresources::Uniform(state) = &mut self.states {
            record(subresource, state);
        } else if !matches!(self.states, Subresources::Uniform(Some(used)) if used.last == wanted) {
            record(
                subresource,
                &mut self.states.split(count as usize)[subresource as usize],
            );
        }
    }

    #[cfg(test)]
    pub(super) fn last(&self, subresource: u32) -> Option<u32> {
        match &self.states {
            Subresources::Uniform(state) => *state,
            Subresources::Split(states) => states[subresource as usize],
        }
        .map(|used| used.last)
    }

    /// Fills in the list's initial states and commits its final states in
    /// submission order. Subresources the list did not use keep their state.
    pub(super) fn reconcile(&self, mut emit: impl FnMut(u32, u32, u32)) {
        let mut real = self.copy.states.committed.lock().unwrap_or_else(|e| e.into_inner());
        let mut apply = |index, state: &mut u32, used: Use| {
            if *state != used.first {
                emit(index, *state, used.first);
            }
            *state = used.last;
        };
        match (&self.states, &mut *real) {
            (Subresources::Uniform(None), _) => {}
            (Subresources::Uniform(Some(used)), Subresources::Uniform(state)) => {
                apply(D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES, state, *used);
            }
            (Subresources::Uniform(Some(used)), Subresources::Split(states)) => {
                for (index, state) in states.iter_mut().enumerate() {
                    apply(index as u32, state, *used);
                }
                *real = Subresources::Uniform(used.last);
            }
            (Subresources::Split(uses), _) => {
                let states = real.split(uses.len());
                for (index, (state, used)) in states.iter_mut().zip(uses.iter()).enumerate() {
                    if let Some(used) = used {
                        apply(index as u32, state, *used);
                    }
                }
                let first = states[0];
                if states.iter().all(|&state| state == first) {
                    *real = Subresources::Uniform(first);
                }
            }
        }
    }
}
