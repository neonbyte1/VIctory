//! Host textures placed in heaps made ahead instead of committed one by
//! one: a new shot brings hundreds of textures, and the host spent most of
//! a frame making them (300 textures, 430 MB on the RX 6700 XT: committed
//! 255 ms; one heap 32 ms, then placed 3 ms).
//!
//! Textures without render-target, depth or unordered-access use, up to a
//! quarter of a heap, go into heaps of [`heap_bytes`] bytes, first fit
//! (with `VICTORY_VRAM_FIT=1`, the fullest heap first). A texture's range
//! is free again once nothing else holds its resource (heap slots keep
//! views' resources until the running submission has run, lists until they
//! are reset), so no view or list sees its memory reused. Empty heaps past
//! [`SPARE`] are released.

use std::collections::BTreeMap;
use std::sync::Mutex;

use windows::Win32::Graphics::Direct3D12::*;
use windows::core::Interface;

/// Bytes of one heap: 64 MiB, or 16 MiB on cards with a small budget
/// (`crate::vram_fit`).
fn heap_bytes() -> u64 {
    crate::vram_fit::heap_bytes()
}

/// Empty heaps kept for the next textures.
const SPARE: usize = 2;
/// Placement alignment of textures that are not render targets.
const ALIGN: u64 = 64 << 10;

/// Where a texture lies: heap, offset and bytes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct Placement {
    heap: u64,
    offset: u64,
    size: u64,
}

/// A heap, its free ranges (offset → bytes) and the ranges handed out
/// (offset → bytes and the presented frame they were placed in, freed or
/// not: given back only once nothing holds their texture).
struct Heap {
    heap: ID3D12Heap,
    id: u64,
    free: BTreeMap<u64, u64>,
    placed: BTreeMap<u64, (u64, u64)>,
    /// Under a quarter full while the heaps hold too much free space
    /// ([`drain_heaps`]): new textures go elsewhere if they can, and the
    /// ones in it are given up.
    draining: bool,
}

impl Heap {
    fn used(&self) -> u64 {
        heap_bytes() - self.free.values().sum::<u64>()
    }
}

/// Heaps made and released (diagnostics, [`log`]).
static HEAPS_MADE: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
static HEAPS_RELEASED: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

#[derive(Default)]
struct Heaps {
    heaps: Vec<Heap>,
    next: u64,
    /// Ranges whose texture may still be held elsewhere, with a holder of
    /// our own.
    pending: Vec<(ID3D12Resource, Placement)>,
}

// SAFETY: D3D12 heaps and resources are free-threaded; used under the lock.
unsafe impl Send for Heaps {}

static HEAPS: Mutex<Option<Heaps>> = Mutex::new(None);

/// The first range of `free` that fits `size` bytes at `align`: its
/// offset, taken out of `free`.
fn take(free: &mut BTreeMap<u64, u64>, size: u64, align: u64) -> Option<u64> {
    let (&start, &len, at) = free.iter().find_map(|(s, l)| {
        let at = s.next_multiple_of(align);
        (at + size <= s + l).then_some((s, l, at))
    })?;
    free.remove(&start);
    if at > start {
        free.insert(start, at - start);
    }
    if at + size < start + len {
        free.insert(at + size, start + len - at - size);
    }
    Some(at)
}

/// Gives `[offset, offset + size)` back to `free`, joined with its
/// neighbours.
fn give(free: &mut BTreeMap<u64, u64>, offset: u64, size: u64) {
    let (mut start, mut end) = (offset, offset + size);
    if let Some((&s, &l)) = free.range(..start).next_back()
        && s + l == start
    {
        free.remove(&s);
        start = s;
    }
    if let Some(&l) = free.get(&end) {
        free.remove(&end);
        end += l;
    }
    free.insert(start, end - start);
}

/// Whether a resource has no holder but the one passed.
fn last_holder(resource: &ID3D12Resource) -> bool {
    // SAFETY: a live COM object; one reference taken and given back.
    unsafe {
        let raw = resource.as_raw();
        let unknown = &*(raw as *const *const windows::core::IUnknown_Vtbl);
        ((**unknown).AddRef)(raw);
        ((**unknown).Release)(raw) == 1
    }
}

impl Heaps {
    /// Frees the pending ranges nothing else holds now, and releases empty
    /// heaps past [`SPARE`].
    fn reap(&mut self) {
        let mut i = 0;
        while i < self.pending.len() {
            if last_holder(&self.pending[i].0) {
                let (_, p) = self.pending.swap_remove(i);
                if let Some(h) = self.heaps.iter_mut().find(|h| h.id == p.heap) {
                    give(&mut h.free, p.offset, p.size);
                    h.placed.remove(&p.offset);
                }
            } else {
                i += 1;
            }
        }
        let empty = |h: &Heap| h.free.get(&0) == Some(&heap_bytes());
        let mut spare = 0;
        self.heaps.retain(|h| {
            if !empty(h) {
                return true;
            }
            spare += 1;
            let keep = spare <= SPARE;
            if !keep {
                HEAPS_RELEASED.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            }
            keep
        });
    }

    /// A range of `size` bytes: in a heap with room, else in a new heap.
    fn allocate(
        &mut self,
        device: &ID3D12Device,
        size: u64,
        align: u64,
    ) -> Option<(ID3D12Heap, Placement)> {
        // With `VICTORY_VRAM_FIT`, the fullest heap that fits first: a
        // long-lived texture then keeps a heap that is full anyway, and
        // sparse heaps empty out and are released. Heaps being drained
        // last ([`drain_heaps`]).
        let mut order: Vec<usize> = (0..self.heaps.len()).collect();
        let fullest = crate::vram_fit::on();
        order.sort_by_key(|&i| {
            let h = &self.heaps[i];
            (
                h.draining,
                if fullest {
                    h.free.values().sum::<u64>()
                } else {
                    0
                },
            )
        });
        let frame = crate::queue::presented();
        for i in order {
            let h = &mut self.heaps[i];
            if let Some(offset) = take(&mut h.free, size, align) {
                h.placed.insert(offset, (size, frame));
                return Some((
                    h.heap.clone(),
                    Placement {
                        heap: h.id,
                        offset,
                        size,
                    },
                ));
            }
        }
        let desc = D3D12_HEAP_DESC {
            SizeInBytes: heap_bytes(),
            Properties: D3D12_HEAP_PROPERTIES {
                Type: D3D12_HEAP_TYPE_DEFAULT,
                ..Default::default()
            },
            Alignment: ALIGN,
            Flags: D3D12_HEAP_FLAG_ALLOW_ONLY_NON_RT_DS_TEXTURES,
        };
        let mut heap: Option<ID3D12Heap> = None;
        // SAFETY: plain creation, our out pointer.
        unsafe { device.CreateHeap(&desc, &mut heap) }.ok()?;
        let heap = heap?;
        crate::host::name(&heap, || "texture heap".into());
        let id = self.next;
        self.next += 1;
        let mut free = BTreeMap::new();
        free.insert(0, heap_bytes());
        let offset = take(&mut free, size, align)?;
        HEAPS_MADE.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        self.heaps.push(Heap {
            heap: heap.clone(),
            id,
            free,
            placed: BTreeMap::from([(offset, (size, frame))]),
            draining: false,
        });
        Some((
            heap,
            Placement {
                heap: id,
                offset,
                size,
            },
        ))
    }
}

/// Whether a texture of `desc` goes into a heap: no render-target, depth or
/// unordered-access use, and at most a quarter of a heap.
pub(crate) fn fits(device: &ID3D12Device, desc: &D3D12_RESOURCE_DESC) -> bool {
    const USES: i32 = D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET.0
        | D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL.0
        | D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS.0
        | D3D12_RESOURCE_FLAG_ALLOW_SIMULTANEOUS_ACCESS.0;
    if desc.Dimension == D3D12_RESOURCE_DIMENSION_BUFFER
        || desc.Flags.0 & USES != 0
        || desc.SampleDesc.Count > 1
    {
        return false;
    }
    // SAFETY: a plain query.
    let info = unsafe { device.GetResourceAllocationInfo(0, &[*desc]) };
    info.SizeInBytes != u64::MAX && info.SizeInBytes <= heap_bytes() / 4 && info.Alignment <= ALIGN
}

/// Room for a texture of `desc` in a heap ([`fits`] first): the heap and
/// the placement, which the texture gives back with [`free`].
pub(crate) fn place(
    device: &ID3D12Device,
    desc: &D3D12_RESOURCE_DESC,
) -> Option<(ID3D12Heap, Placement)> {
    // SAFETY: a plain query.
    let info = unsafe { device.GetResourceAllocationInfo(0, &[*desc]) };
    let size = info.SizeInBytes.next_multiple_of(ALIGN);
    let mut heaps = HEAPS.lock().unwrap_or_else(|e| e.into_inner());
    let heaps = heaps.get_or_insert_with(Heaps::default);
    heaps.reap();
    heaps.allocate(device, size, info.Alignment.max(ALIGN))
}

/// A range [`place`] gave whose texture could not be made there.
pub(crate) fn unplace(p: Placement) {
    let mut heaps = HEAPS.lock().unwrap_or_else(|e| e.into_inner());
    if let Some(h) = heaps
        .as_mut()
        .and_then(|hs| hs.heaps.iter_mut().find(|h| h.id == p.heap))
    {
        give(&mut h.free, p.offset, p.size);
        h.placed.remove(&p.offset);
    }
}

/// The texture at `p` is let go of: its range is free again once nothing
/// else holds `resource` (checked at the next placements and passes).
pub(crate) fn free(resource: ID3D12Resource, p: Placement) {
    let mut heaps = HEAPS.lock().unwrap_or_else(|e| e.into_inner());
    if let Some(h) = heaps.as_mut() {
        h.pending.push((resource, p));
        h.reap();
    }
}

/// Bytes of heap space free for textures (after freeing what nothing
/// holds now): the card's memory they take is not in use.
pub(crate) fn free_bytes() -> u64 {
    let mut heaps = HEAPS.lock().unwrap_or_else(|e| e.into_inner());
    let Some(h) = heaps.as_mut() else { return 0 };
    h.reap();
    h.heaps.iter().flat_map(|h| h.free.values()).sum()
}

/// Heaps of textures held, and their used and free bytes (after freeing
/// what nothing holds now), for the statistics line.
pub(crate) fn stats() -> (usize, u64, u64) {
    let mut heaps = HEAPS.lock().unwrap_or_else(|e| e.into_inner());
    let Some(h) = heaps.as_mut() else {
        return (0, 0, 0);
    };
    h.reap();
    let free: u64 = h.heaps.iter().flat_map(|h| h.free.values()).sum();
    (
        h.heaps.len(),
        h.heaps.len() as u64 * heap_bytes() - free,
        free,
    )
}

/// Share of the card's budget the heaps may hold free before sparse heaps
/// drain ([`drain_heaps`]).
const DRAIN_FREE: f64 = 0.02;

/// With `VICTORY_HEAP_DRAIN=1` and more free space in the heaps than
/// [`DRAIN_FREE`] of the card's budget (`budget_mb`): the heaps under a
/// quarter full are marked draining, and their ids returned (their
/// textures are given up, `crate::resource`). Otherwise none drains.
pub(crate) fn drain_heaps(budget_mb: f64) -> Option<std::collections::HashSet<u64>> {
    let mut heaps = HEAPS.lock().unwrap_or_else(|e| e.into_inner());
    let h = heaps.as_mut()?;
    h.reap();
    let free: u64 = h.heaps.iter().flat_map(|h| h.free.values()).sum();
    let over = crate::vram_fit::drain() && free as f64 > budget_mb * 1e6 * DRAIN_FREE;
    let ids = h.mark_draining(over);
    (!ids.is_empty()).then_some(ids)
}

impl Heaps {
    /// With `over`, the heaps holding something but under a quarter full
    /// drain (their ids); without, none does.
    fn mark_draining(&mut self, over: bool) -> std::collections::HashSet<u64> {
        let mut ids = std::collections::HashSet::new();
        for heap in &mut self.heaps {
            let used = heap.used();
            heap.draining = over && used > 0 && used * 4 < heap_bytes();
            if heap.draining {
                ids.insert(heap.id);
            }
        }
        ids
    }
}

/// Textures given up to drain heaps, and their bytes (diagnostics).
pub(crate) static DRAINED: [std::sync::atomic::AtomicU64; 2] =
    [const { std::sync::atomic::AtomicU64::new(0) }; 2];

/// The heaps line, with the frame-rate statistics: heaps by how full they
/// are and the free space in each class; what pins the heaps under a
/// quarter full (textures still ours, by how many frames ago they were
/// placed; ranges let go of but held elsewhere); heaps made and released,
/// and textures drained, per frame over `frames`.
#[cfg(feature = "diagnostics")]
pub(crate) fn log(frames: f64) {
    use std::sync::atomic::Ordering::Relaxed;
    let frame = crate::queue::presented();
    let mut fill = [(0usize, 0u64); 4];
    let (mut ages, mut live_bytes, mut held, mut held_bytes) = (Vec::new(), 0u64, 0usize, 0u64);
    {
        let mut heaps = HEAPS.lock().unwrap_or_else(|e| e.into_inner());
        let Some(h) = heaps.as_mut() else { return };
        h.reap();
        let pending: std::collections::HashSet<(u64, u64)> =
            h.pending.iter().map(|(_, p)| (p.heap, p.offset)).collect();
        for heap in &h.heaps {
            let used = heap.used();
            let class = ((used * 4) / heap_bytes()).min(3) as usize;
            fill[class].0 += 1;
            fill[class].1 += heap_bytes() - used;
            if class > 0 {
                continue;
            }
            for (offset, &(size, made)) in &heap.placed {
                if pending.contains(&(heap.id, *offset)) {
                    held += 1;
                    held_bytes += size;
                } else {
                    ages.push(frame.saturating_sub(made));
                    live_bytes += size;
                }
            }
        }
    }
    ages.sort_unstable();
    let at = |q: usize| ages.get(ages.len() * q / 100).copied().unwrap_or(0);
    let per = |n: u64| n as f64 / frames.max(1.0);
    let mb = |b: u64| b as f64 / 1e6;
    let (made, released) = (HEAPS_MADE.swap(0, Relaxed), HEAPS_RELEASED.swap(0, Relaxed));
    let drained = [DRAINED[0].swap(0, Relaxed), DRAINED[1].swap(0, Relaxed)];
    tracing::info!(
        "heaps of textures by fill: under 25 % {} ({:.0} MB free), 25-50 % {} ({:.0} MB), 50-75 % {} ({:.0} MB), \
         75 % and over {} ({:.0} MB); pinning those under 25 %: {} textures of ours ({:.0} MB, placed {} / {} frames ago, \
         median / 90th percentile), {} let go of and held elsewhere ({:.0} MB); heaps made {:.2}, released {:.2} per frame; \
         drained (VICTORY_HEAP_DRAIN={}) {:.1} textures ({:.1} MB) per frame",
        fill[0].0,
        mb(fill[0].1),
        fill[1].0,
        mb(fill[1].1),
        fill[2].0,
        mb(fill[2].1),
        fill[3].0,
        mb(fill[3].1),
        ages.len(),
        mb(live_bytes),
        at(50),
        at(90),
        held,
        mb(held_bytes),
        per(made),
        per(released),
        u8::from(crate::vram_fit::drain()),
        per(drained[0]),
        mb(drained[1]) / frames.max(1.0),
    );
}

impl Placement {
    /// Its offset in its heap.
    pub(crate) fn offset(&self) -> u64 {
        self.offset
    }

    /// Its heap's id ([`drain_heaps`]).
    pub(crate) fn heap_id(&self) -> u64 {
        self.heap
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A texture's range is not handed out again while anything else holds
    /// its resource (a view's slot, a list), and is once nothing does.
    #[test]
    fn a_range_is_reused_only_once_nothing_holds_its_texture() {
        use windows::Win32::Graphics::Dxgi::Common::*;
        let Some(host) = crate::host::host() else {
            return; // no host GPU here
        };
        let desc = D3D12_RESOURCE_DESC {
            Dimension: D3D12_RESOURCE_DIMENSION_TEXTURE2D,
            Width: 256,
            Height: 256,
            DepthOrArraySize: 1,
            MipLevels: 9,
            Format: DXGI_FORMAT_BC1_UNORM,
            SampleDesc: DXGI_SAMPLE_DESC {
                Count: 1,
                Quality: 0,
            },
            ..Default::default()
        };
        assert!(fits(&host.device, &desc));
        let mut heaps = Heaps::default();
        let (heap, a) = heaps.allocate(&host.device, ALIGN, ALIGN).unwrap();
        let mut texture: Option<ID3D12Resource> = None;
        // SAFETY: a range of the heap, our out pointer.
        unsafe {
            host.device.CreatePlacedResource(
                &heap,
                a.offset(),
                &desc,
                D3D12_RESOURCE_STATE_COMMON,
                None,
                &mut texture,
            )
        }
        .unwrap();
        let texture = texture.unwrap();
        let held = texture.clone();
        heaps.pending.push((texture, a));
        heaps.reap();
        let (_, b) = heaps.allocate(&host.device, ALIGN, ALIGN).unwrap();
        assert_ne!(b, a, "still held: not handed out");
        drop(held);
        heaps.reap();
        let (_, c) = heaps.allocate(&host.device, ALIGN, ALIGN).unwrap();
        assert_eq!(c, a, "nothing holds it: free again");
    }

    /// A heap holding little drains while there is too much free space:
    /// it is marked, and new ranges go to a heap that does not drain while
    /// one has room; without too much free space nothing drains.
    #[test]
    fn sparse_heaps_drain_and_new_ranges_avoid_them() {
        let Some(host) = crate::host::host() else {
            return; // no host GPU here
        };
        let mut heaps = Heaps::default();
        // A whole heap, so that a little goes into a second one; then two
        // fifths of the first given back: 60 % full.
        let (_, big) = heaps.allocate(&host.device, heap_bytes(), ALIGN).unwrap();
        let (_, small) = heaps.allocate(&host.device, ALIGN, ALIGN).unwrap();
        assert_ne!(small.heap, big.heap);
        give(&mut heaps.heaps[0].free, 0, heap_bytes() / 5 * 2);
        let ids = heaps.mark_draining(true);
        assert_eq!(ids, [small.heap].into(), "under a quarter full: drains");
        let (_, next) = heaps.allocate(&host.device, ALIGN, ALIGN).unwrap();
        assert_eq!(next.heap, big.heap, "new ranges avoid the heap draining");
        assert!(heaps.mark_draining(false).is_empty());
        assert!(heaps.heaps.iter().all(|h| !h.draining));
    }

    #[test]
    fn ranges_are_taken_first_fit_and_joined_when_given_back() {
        let k = 64 << 10;
        let mut free = BTreeMap::from([(0, 10 * k)]);
        let a = take(&mut free, 3 * k, k).unwrap();
        let b = take(&mut free, 2 * k, k).unwrap();
        let c = take(&mut free, 4 * k, k).unwrap();
        assert_eq!((a, b, c), (0, 3 * k, 5 * k));
        assert_eq!(take(&mut free, 2 * k, k), None, "one 64 KiB range left");
        give(&mut free, b, 2 * k);
        assert_eq!(
            take(&mut free, 3 * k, k),
            None,
            "two ranges of 2 and 1, not joined"
        );
        give(&mut free, c, 4 * k);
        assert_eq!(
            free,
            BTreeMap::from([(3 * k, 7 * k)]),
            "joined with both neighbours"
        );
        give(&mut free, a, 3 * k);
        assert_eq!(free, BTreeMap::from([(0, 10 * k)]));
        // An aligned range inside a free one keeps the padding before it.
        let mut free = BTreeMap::from([(k, 10 * k)]);
        assert_eq!(take(&mut free, k, 4 * k), Some(4 * k));
        assert_eq!(free, BTreeMap::from([(k, 3 * k), (5 * k, 6 * k)]));
    }
}
