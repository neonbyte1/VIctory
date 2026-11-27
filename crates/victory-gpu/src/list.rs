//! Command allocators and command lists.
//!
//! The title's inlined code writes packets into a list only while its put
//! pointer is below a limit, and calls the list's method otherwise. Our
//! lists keep the pointer and every limit at zero, so every command reaches
//! the table as a call.
//!
//! Each list records onto a host command list as its methods are called
//! (decision 0007); draw state lives in `crate::draw`. Methods not
//! written yet log once with their caller, so a run lists what to do next.
#![allow(clippy::not_unsafe_ptr_arg_deref)]

use std::mem::offset_of;

use crate::object::{
    self, ALLOCATOR, E_INVALIDARG, HRESULT, Iid, LIST, Object, S_OK, Table, This, guid,
};
use crate::queue::Written;
use crate::resource::PAGEABLE_IIDS;

const ALLOCATOR_IIDS: [Iid; 4] = [
    guid(
        0x6102dee4,
        0xaf59,
        0x4b09,
        [0xb9, 0x99, 0xb4, 0x4d, 0x73, 0xf0, 0x9b, 0x24],
    ),
    PAGEABLE_IIDS[0],
    PAGEABLE_IIDS[1],
    PAGEABLE_IIDS[2],
];

/// The list interfaces up to `ID3D12GraphicsCommandList6` (the table has
/// all their slots): those whose ids are the same as on PC, the console's
/// own List5 and List6 (Grand Theft Auto VI asks for List6), then the
/// device child's.
const LIST_IIDS: [Iid; 10] = [
    guid(
        0x7116d91c,
        0xe7e4,
        0x47ce,
        [0xb8, 0xc6, 0xec, 0x81, 0x68, 0xf4, 0x37, 0xe5],
    ),
    guid(
        0x5b160d0f,
        0xac1b,
        0x4185,
        [0x8b, 0xa8, 0xb3, 0xae, 0x42, 0xa5, 0xa4, 0x55],
    ),
    guid(
        0x553103fb,
        0x1fe7,
        0x4557,
        [0xbb, 0x38, 0x94, 0x6d, 0x7d, 0x0e, 0x7c, 0xa7],
    ),
    guid(
        0x38c3e585,
        0xff17,
        0x412c,
        [0x91, 0x50, 0x4f, 0xc6, 0xf9, 0xd7, 0x2a, 0x28],
    ),
    guid(
        0x6fda83a7,
        0xb84c,
        0x4e38,
        [0x9a, 0xc8, 0xc7, 0xbd, 0x22, 0x01, 0x6b, 0x3d],
    ),
    guid(
        0x8754318e,
        0xd3a9,
        0x4541,
        [0x98, 0xcf, 0x64, 0x5b, 0x50, 0xdc, 0x48, 0x74],
    ),
    guid(
        0x55050859,
        0x4024,
        0x474c,
        [0x98, 0xcf, 0x64, 0x5b, 0x51, 0xdc, 0x48, 0x74],
    ),
    guid(
        0xc3827890,
        0x4024,
        0x474c,
        [0x98, 0xcf, 0x64, 0x5b, 0x51, 0xdc, 0x48, 0x74],
    ),
    PAGEABLE_IIDS[1],
    PAGEABLE_IIDS[2],
];

#[repr(C)]
struct Allocator {
    object: Object,
    kind: u32,
}

/// The layout the title's inlined code reads.
#[repr(C)]
struct List {
    object: Object,
    kind: u32,
    /// The put pointer, then the plain, draw, API and unbiased API limits:
    /// all zero, so every inline check fails.
    put: [u64; 5],
    reserved: [u32; 64],
    /// State the inline code keeps for building packets; unused by us.
    graphics_root_headers: [u32; 128],
    compute_root_headers: [u32; 128],
    pipeline_shadow: [u32; 8],
    /// Newer titles' lists are larger: Grand Theft Auto VI's state flush
    /// (title+0x32786) copies a pipeline's 32 bytes to +0x544..+0x564 and
    /// tests +0x548, so its layout runs past 0x560. Its size is unknown;
    /// this keeps our fields clear of whatever the title writes there.
    newer: [u32; 256],
    /// Ours, after the console layout: memory handed out by
    /// [`allocate_upload`], kept until the list is reset; the host list the
    /// calls record into (decision 0007), and the draw state the title set.
    uploads: std::sync::Mutex<Vec<Upload>>,
    /// The host lists the calls record into, in order: one, and one more
    /// for every wait the title records after commands ([`wait`]). Shared
    /// with the queues that hold them until their waits hold
    /// (`crate::hold`), so they outlive a list the title releases first.
    #[cfg(windows)]
    hosts: std::sync::Mutex<Vec<std::sync::Arc<crate::host::HostList>>>,
    /// Which of them the calls record into.
    #[cfg(windows)]
    segment: std::sync::atomic::AtomicUsize,
    #[cfg(windows)]
    state: std::sync::Mutex<crate::draw::State>,
}
const _: () = assert!(offset_of!(List, put) == 0x18);
const _: () = assert!(offset_of!(List, pipeline_shadow) + 32 == 0x560);

/// Runs `f` with the host list and draw state of a title list, if there
/// is a host GPU.
#[cfg(windows)]
pub(crate) fn with_state(this: This, f: impl FnOnce(&crate::host::HostList, &mut crate::draw::State)) {
    let Some(list) = host_list(this) else { return };
    // SAFETY: the title passes one of our lists.
    let state = unsafe { &(*this.cast::<List>()).state };
    f(list, &mut state.lock().unwrap_or_else(|e| e.into_inner()));
}

/// The host list a title list's calls record into (its current segment),
/// if there is a host GPU.
#[cfg(windows)]
pub(crate) fn host_list<'a>(this: This) -> Option<&'a crate::host::HostList> {
    if this.is_null() {
        return None;
    }
    // SAFETY: the title passes one of our lists.
    let list = unsafe { &*this.cast::<List>() };
    let k = list.segment.load(std::sync::atomic::Ordering::Relaxed);
    let hosts = list.hosts.lock().unwrap_or_else(|e| e.into_inner());
    let current: *const crate::host::HostList = std::sync::Arc::as_ptr(hosts.get(k)?);
    // SAFETY: the list holds this host list until it is reset or released
    // (an `Arc` does not move it), and the title records into a list from
    // one thread at a time.
    Some(unsafe { &*current })
}

/// A title list's host lists, in order: what a submission of it runs.
#[cfg(windows)]
pub(crate) fn segments(this: This) -> Vec<std::sync::Arc<crate::host::HostList>> {
    if this.is_null() {
        return Vec::new();
    }
    // SAFETY: the title passes one of our lists.
    let list = unsafe { &*this.cast::<List>() };
    let k = list.segment.load(std::sync::atomic::Ordering::Relaxed);
    let hosts = list.hosts.lock().unwrap_or_else(|e| e.into_inner());
    hosts.iter().take(k + 1).cloned().collect()
}

/// Waits recorded after commands: the title list goes on in a new host
/// list, which runs once the wait holds.
static CUTS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// Lists reset or released while a queue still held some of their host
/// lists: D3D12 lets a list be reset as soon as it is submitted (Grand
/// Theft Auto VI does, 0-21 times a world frame).
static EARLY_RESETS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// Cuts and early resets since the last call (the frame line).
pub(crate) fn take_cuts() -> (u64, u64) {
    use std::sync::atomic::Ordering::Relaxed;
    (CUTS.swap(0, Relaxed), EARLY_RESETS.swap(0, Relaxed))
}

/// `Wait32BitValueX(address, comparison, reference, mask, flags)` and
/// `Wait64BitValueX(address, comparison, reference, flags)`: the GPU waits
/// until the value compares true. Recorded at the start of the current host
/// list, or, after commands, at the start of a new one the rest of the
/// title list records into (the queue runs each once its waits hold,
/// `crate::hold`); bottom-of-pipe writes recorded before it land when the
/// part before it has run.
extern "system" fn wait32(this: This, address: u64, comparison: u32, reference: u32, mask: u32, _flags: u32) {
    wait(this, crate::hold::ValueWait { address, bytes: 4, comparison, reference: u64::from(reference), mask: u64::from(mask) });
}
pub(crate) extern "system" fn wait64(this: This, address: u64, comparison: u32, reference: u64, _flags: u32) {
    wait(this, crate::hold::ValueWait { address, bytes: 8, comparison, reference, mask: u64::MAX });
}

fn wait(this: This, wait: crate::hold::ValueWait) {
    #[cfg(windows)]
    {
        let Some(current) = host_list(this) else { return };
        if !current.is_empty() {
            let Some(next) = next_segment(this) else { return };
            CUTS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            current.close();
            // SAFETY: the title passes one of our lists.
            let list = unsafe { &*this.cast::<List>() };
            list.state.lock().unwrap_or_else(|e| e.into_inner()).cut(next);
            next.defer_wait(wait);
            return;
        }
        current.defer_wait(wait);
    }
    #[cfg(not(windows))]
    let _ = (this, wait);
}

/// Moves a title list's recording on to its next host list (made, or one
/// it held since an earlier recording, reset then): that host list.
#[cfg(windows)]
fn next_segment<'a>(this: This) -> Option<&'a crate::host::HostList> {
    // SAFETY: the title passes one of our lists.
    let list = unsafe { &*this.cast::<List>() };
    let k = list.segment.load(std::sync::atomic::Ordering::Relaxed) + 1;
    let mut hosts = list.hosts.lock().unwrap_or_else(|e| e.into_inner());
    if hosts.len() <= k {
        let made = crate::host::host()?.new_list().ok()?;
        hosts.push(std::sync::Arc::new(made));
    }
    list.segment.store(k, std::sync::atomic::Ordering::Relaxed);
    let next: *const crate::host::HostList = std::sync::Arc::as_ptr(&hosts[k]);
    drop(hosts);
    // SAFETY: as in `host_list`.
    Some(unsafe { &*next })
}

/// After the title list's last submission: its host lists start again,
/// and a host list a queue still holds (`crate::hold`) is left to the
/// queue and replaced. The last one left so, which should keep what the
/// title list owned.
#[cfg(windows)]
fn restart(this: This) -> Option<std::sync::Arc<crate::host::HostList>> {
    // SAFETY: the title passes one of our lists.
    let list = unsafe { &*this.cast::<List>() };
    let used = list.segment.swap(0, std::sync::atomic::Ordering::Relaxed);
    let mut hosts = list.hosts.lock().unwrap_or_else(|e| e.into_inner());
    let mut left = None;
    for (k, h) in hosts.iter_mut().enumerate() {
        if std::sync::Arc::strong_count(h) > 1 {
            EARLY_RESETS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            tracing::debug!(segment = k, "a list reset while a queue still holds part of it: that part keeps its own host list");
            let spare = SPARE.lock().unwrap_or_else(|e| e.into_inner()).pop();
            match spare.map(Ok).or_else(|| crate::host::host().map(|host| host.new_list())) {
                Some(Ok(fresh)) => left = Some(std::mem::replace(h, std::sync::Arc::new(fresh))),
                _ => continue,
            }
        } else if k <= used {
            h.reset();
        }
    }
    left
}

/// Host lists [`restart`] left to a queue, back once they ran: the next
/// ones it hands out (a host list and its allocators are not free to make;
/// Grand Theft Auto VI resets up to 21 lists a frame while they are held).
#[cfg(windows)]
static SPARE: std::sync::Mutex<Vec<crate::host::HostList>> = std::sync::Mutex::new(Vec::new());

/// A host list a queue has run: one no title list holds any more is reset
/// and kept for [`restart`].
#[cfg(windows)]
pub(crate) fn recycle(list: std::sync::Arc<crate::host::HostList>) {
    const KEPT: usize = 64;
    if let Ok(list) = std::sync::Arc::try_unwrap(list) {
        list.reset();
        let mut spare = SPARE.lock().unwrap_or_else(|e| e.into_inner());
        if spare.len() < KEPT {
            spare.push(list);
        }
    }
}

/// What a title list owned that the host list a queue still holds, if any,
/// may read: its upload memory (list 0x4a0) and its draw state's dynamic
/// descriptor slots and title-memory page table. Freed once that host list
/// (the list's last held one, so the last to run) is gone.
#[cfg(windows)]
fn leave(to: Option<std::sync::Arc<crate::host::HostList>>, uploads: Vec<Upload>, state: crate::draw::State) {
    struct Kept(#[allow(dead_code)] Vec<Upload>, #[allow(dead_code)] crate::draw::State);
    // SAFETY: only dropped, with the host list, after it has run.
    unsafe impl Send for Kept {}
    if let Some(held) = to {
        let kept = Kept(uploads, state);
        held.defer_after(move || {
            let _ = &kept;
        });
    }
}

pub(crate) extern "system" fn create_allocator(
    _: This,
    kind: u32,
    iid: *const Iid,
    out: *mut This,
) -> HRESULT {
    let allocator = Box::new(Allocator {
        object: Object::new(&ALLOCATOR_FUNCTIONS),
        kind,
    });
    object::hand_out(ALLOCATOR, allocator, iid, &ALLOCATOR_IIDS, out)
}

/// Lists start open, as in D3D12.
pub(crate) extern "system" fn create_list(
    _: This,
    _nodes: u32,
    kind: u32,
    allocator: This,
    pipeline: This,
    iid: *const Iid,
    out: *mut This,
) -> HRESULT {
    new_list(kind, allocator, pipeline, iid, out)
}

/// The console's list creation: type, flags, node mask and barrier batch
/// size in a description.
#[repr(C)]
#[derive(Debug)]
pub(crate) struct ListDesc {
    kind: u32,
    flags: u32,
    nodes: u32,
    barrier_batch: u32,
}

pub(crate) extern "system" fn create_list_x(
    _: This,
    desc: *const ListDesc,
    allocator: This,
    pipeline: This,
    iid: *const Iid,
    out: *mut This,
) -> HRESULT {
    // SAFETY: the title passes a description (null is tolerated).
    let Some(desc) = (unsafe { desc.as_ref() }) else {
        return E_INVALIDARG;
    };
    tracing::debug!(?desc, "console command list");
    new_list(desc.kind, allocator, pipeline, iid, out)
}

fn new_list(
    kind: u32,
    allocator: This,
    pipeline: This,
    iid: *const Iid,
    out: *mut This,
) -> HRESULT {
    if allocator.is_null() {
        return E_INVALIDARG;
    }
    tracing::debug!(kind, "command list");
    let list = Box::new(List {
        object: if crate::device::newer() {
            Object::new(&NEWER_LIST_FUNCTIONS)
        } else {
            Object::new(&LIST_FUNCTIONS)
        },
        kind,
        put: [0; 5],
        reserved: [0; 64],
        graphics_root_headers: [0; 128],
        compute_root_headers: [0; 128],
        pipeline_shadow: [0; 8],
        newer: [0; 256],
        uploads: Default::default(),
        #[cfg(windows)]
        hosts: std::sync::Mutex::new(
            crate::host::host()
                .and_then(|h| match h.new_list() {
                    Ok(list) => Some(std::sync::Arc::new(list)),
                    Err(e) => {
                        tracing::warn!(error = %e, "no host command list");
                        None
                    }
                })
                .into_iter()
                .collect(),
        ),
        #[cfg(windows)]
        segment: Default::default(),
        #[cfg(windows)]
        state: {
            let mut state = crate::draw::State::default();
            state.set_pipeline(pipeline);
            std::sync::Mutex::new(state)
        },
    });
    let hr = object::hand_out(LIST, list, iid, &LIST_IIDS, out);
    if hr == S_OK {
        crate::pipeline::bound(pipeline);
    }
    hr
}

extern "system" fn allocator_query(this: This, iid: *const Iid, out: *mut This) -> HRESULT {
    let found = object::is(iid, &ALLOCATOR_IIDS).then_some(this);
    object::answer(ALLOCATOR, iid, found, out)
}
extern "system" fn allocator_release(this: This) -> u32 {
    object::release_boxed::<Allocator>(this)
}
/// Nothing is recorded, so there is nothing to free.
extern "system" fn allocator_reset(_: This) -> HRESULT {
    S_OK
}

extern "system" fn list_query(this: This, iid: *const Iid, out: *mut This) -> HRESULT {
    object::answer(LIST, iid, object::is(iid, &LIST_IIDS).then_some(this), out)
}
extern "system" fn list_release(this: This) -> u32 {
    // A queue may still hold some of its host lists; they are its own, and
    // the last of them takes what the list owned.
    #[cfg(windows)]
    if !this.is_null() && object::refs(this) == 1 {
        // SAFETY: the title passes one of our lists.
        let list = unsafe { &*this.cast::<List>() };
        let held = list.hosts.lock().unwrap_or_else(|e| e.into_inner()).iter().rfind(|h| std::sync::Arc::strong_count(h) > 1).cloned();
        if held.is_some() {
            EARLY_RESETS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            tracing::debug!("a list released while a queue still holds part of it: that part runs on its own");
            let uploads = std::mem::take(&mut *list.uploads.lock().unwrap_or_else(|e| e.into_inner()));
            let state = std::mem::take(&mut *list.state.lock().unwrap_or_else(|e| e.into_inner()));
            leave(held, uploads, state);
        }
    }
    object::release_boxed::<List>(this)
}
extern "system" fn list_type(this: This) -> u32 {
    // SAFETY: `this` is one of our lists.
    unsafe { (*this.cast::<List>()).kind }
}
extern "system" fn close(this: This) -> HRESULT {
    #[cfg(windows)]
    if let Some(host) = host_list(this) {
        host.close();
    }
    S_OK
}
extern "system" fn reset(this: This, allocator: This, pipeline: This) -> HRESULT {
    if allocator.is_null() {
        return E_INVALIDARG;
    }
    // A queue may still hold what the list recorded (`restart`, `leave`).
    // SAFETY: the title passes one of our lists.
    let uploads = unsafe { &(*this.cast::<List>()).uploads };
    let uploads = std::mem::take(&mut *uploads.lock().unwrap_or_else(|e| e.into_inner()));
    #[cfg(windows)]
    {
        let held = restart(this);
        // SAFETY: the title passes one of our lists.
        let mut state = unsafe { &(*this.cast::<List>()).state }.lock().unwrap_or_else(|e| e.into_inner());
        let old = std::mem::take(&mut *state);
        state.reset(pipeline);
        drop(state);
        leave(held, uploads, old);
    }
    #[cfg(not(windows))]
    drop(uploads);
    crate::pipeline::bound(pipeline);
    #[cfg(not(windows))]
    let _ = this;
    S_OK
}

/// A resource barrier as titles pass it: type, flags, then a resource and
/// either a subresource and two states (transition) or a second resource
/// (aliasing). The same layout as the host's.
#[repr(C)]
#[derive(Clone, Copy)]
pub(crate) struct Barrier {
    kind: u32,
    flags: u32,
    first: This,
    second: u64,
    last: u64,
}

/// The barrier on host resources, or `None` if it names a resource the host
/// GPU does not have (buffers so far).
#[cfg(windows)]
fn host_barrier(list: &crate::host::HostList, b: &Barrier) -> Option<Barrier> {
    use windows::core::Interface;
    let host = |r: This| crate::resource::host_resource(r).map(|h| h.as_raw().cast_const().cast());
    match b.kind {
        0 => {
            // A split barrier's first half: its second half transitions.
            const BEGIN_ONLY: u32 = 1;
            if b.flags & BEGIN_ONLY != 0 {
                return None;
            }
            // shortcut: whole resources, from the state the list tracked
            // (`HostList::transition`); subresources are not told apart.
            let wanted = crate::host::texture_state(b.last as u32);
            let (texture, before) = list.transition(b.first, wanted)?;
            Some(Barrier {
                kind: 0,
                flags: 0,
                first: texture.as_raw().cast_const().cast(),
                second: 0xffff_ffff | (u64::from(before) << 32),
                last: u64::from(crate::host::legal_state(&texture, wanted)),
            })
        }
        1 => Some(Barrier {
            first: if b.first.is_null() {
                b.first
            } else {
                host(b.first)?
            },
            second: if b.second == 0 {
                0
            } else {
                host(b.second as This)? as u64
            },
            ..*b
        }),
        2 => Some(Barrier {
            first: if b.first.is_null() {
                b.first
            } else {
                host(b.first)?
            },
            ..*b
        }),
        _ => None,
    }
}

/// An unordered access barrier on every resource: earlier shader writes
/// finish before later work reads or writes.
const UAV_BARRIER: Barrier = Barrier {
    kind: 2,
    flags: 0,
    first: std::ptr::null(),
    second: 0,
    last: 0,
};

/// Returns the console's barrier "actions" (none).
extern "system" fn barrier(this: This, count: u32, barriers: *const Barrier) -> u32 {
    #[cfg(windows)]
    if let Some(list) = host_list(this)
        && !barriers.is_null()
    {
        // SAFETY: the title passes `count` barriers.
        let barriers = unsafe { std::slice::from_raw_parts(barriers, count as usize) };
        #[cfg(feature = "diagnostics")]
        list.note_op(|| format!("barrier kinds {:?}", barriers.iter().map(|b| b.kind).collect::<Vec<_>>()));
        let mut host: Vec<Barrier> = barriers
            .iter()
            .filter_map(|b| host_barrier(list, b))
            .collect();
        // Buffers have host copies, not resources the barrier can name:
        // a barrier on one waits for every shader write before it.
        if barriers.iter().any(|b| {
            b.kind != 1 && !b.first.is_null() && crate::resource::buffer_bytes(b.first).is_some()
        }) {
            host.push(UAV_BARRIER);
        }
        if !host.is_empty() {
            use windows::Win32::Graphics::Direct3D12::D3D12_RESOURCE_BARRIER;
            const _: () = assert!(size_of::<Barrier>() == size_of::<D3D12_RESOURCE_BARRIER>());
            // SAFETY: the same layout; borrowed resource pointers, never
            // released through this slice.
            let host = unsafe {
                std::slice::from_raw_parts(
                    host.as_ptr().cast::<D3D12_RESOURCE_BARRIER>(),
                    host.len(),
                )
            };
            // SAFETY: an open list and live resources.
            unsafe { list.list.ResourceBarrier(host) };
        }
    }
    0
}

/// Moves a title texture's host copy to `state` in the list, if needed.
#[cfg(windows)]
pub(crate) fn ensure_state(list: &crate::host::HostList, resource: This, state: u32) {
    use windows::Win32::Graphics::Direct3D12::D3D12_RESOURCE_STATES as State;
    if let Some((texture, before)) = list.transition(resource, state) {
        let state = crate::host::legal_state(&texture, state);
        let barrier = crate::host::transition(&texture, State(before as i32), State(state as i32));
        // SAFETY: an open list and a live texture.
        unsafe { list.list.ResourceBarrier(&[barrier]) };
    }
}

#[cfg(windows)]
const COPY_DEST: u32 = 0x400;
#[cfg(windows)]
const COPY_SOURCE: u32 = 0x800;

/// `len` bytes at `offset` of a title buffer, if they are inside it.
#[cfg(windows)]
fn buffer_range(buffer: This, offset: u64, len: u64) -> Option<*mut u8> {
    let (data, size) = crate::resource::buffer_bytes(buffer)?;
    let end = offset.checked_add(len)?;
    if end > size {
        tracing::warn!(offset, len, size, "copy outside its buffer");
        return None;
    }
    // SAFETY: inside the buffer, checked above.
    Some(unsafe { data.add(offset as usize) })
}

/// Buffer to buffer: both are host memory, copied at submission.
extern "system" fn copy_buffer_region(
    this: This,
    dst: This,
    dst_offset: u64,
    src: This,
    src_offset: u64,
    len: u64,
) {
    #[cfg(windows)]
    if let Some(list) = host_list(this)
        && let Some(to) = buffer_range(dst, dst_offset, len)
        && let Some(from) = buffer_range(src, src_offset, len)
    {
        list.defer_copy(to, from, len as usize);
    }
    #[cfg(not(windows))]
    let _ = (this, dst, dst_offset, src, src_offset, len);
}

extern "system" fn copy_resource(this: This, dst: This, src: This) {
    #[cfg(windows)]
    {
        let Some(list) = host_list(this) else {
            return;
        };
        if let (Some(to), Some(from)) = (
            crate::resource::host_resource(dst),
            crate::resource::host_resource(src),
        ) {
            if crate::texture::title_written(&from) {
                crate::texture::upload(list, src, &from);
            }
            ensure_state(list, dst, COPY_DEST);
            ensure_state(list, src, COPY_SOURCE);
            crate::texture::host_writes(list, dst);
            #[cfg(feature = "diagnostics")]
            list.note_op(|| format!("copy resource {src:?} to {dst:?}"));
            // SAFETY: an open list and live textures.
            unsafe { list.list.CopyResource(&to, &from) };
        } else if let (Some((to, a)), Some((from, b))) = (
            crate::resource::buffer_bytes(dst),
            crate::resource::buffer_bytes(src),
        ) {
            list.defer_copy(to, from, a.min(b) as usize);
        }
    }
    #[cfg(not(windows))]
    let _ = (this, dst, src);
}

extern "system" fn copy_resource_x(this: This, dst: This, src: This, _flags: u32) {
    copy_resource(this, dst, src);
}

/// A copy location as titles pass it, laid out as the host's: a resource,
/// then a subresource index (type 0) or a placed footprint (type 1).
#[repr(C)]
#[derive(Clone, Copy)]
pub(crate) struct CopyLocation {
    resource: This,
    kind: u32,
    footprint: crate::resource::PlacedFootprint,
}

/// Texture to texture on the GPU; buffer to texture through an upload
/// buffer filled at submission; texture to buffer through readback memory
/// ([`read_back`]).
extern "system" fn copy_texture_region(
    this: This,
    dst: *const CopyLocation,
    x: u32,
    y: u32,
    z: u32,
    src: *const CopyLocation,
    source_box: *const [u32; 6],
) {
    #[cfg(windows)]
    {
        use windows::Win32::Graphics::Direct3D12::{D3D12_BOX, D3D12_TEXTURE_COPY_LOCATION};
        use windows::core::Interface;
        const FOOTPRINT: u32 = 1;
        // SAFETY: the title passes both locations (null is tolerated).
        let (Some(list), Some(&dst), Some(&src)) =
            (host_list(this), unsafe { dst.as_ref() }, unsafe {
                src.as_ref()
            })
        else {
            return;
        };
        let Some(to) = crate::resource::host_resource(dst.resource) else {
            if dst.kind == FOOTPRINT && src.kind != FOOTPRINT {
                read_back(list, &dst, [x, y, z], &src, source_box);
            }
            return;
        };
        let mut from = src;
        let _upload; // keeps the source alive while this call records it
        #[cfg(feature = "diagnostics")]
        let mut read = None;
        if src.kind == FOOTPRINT {
            let (Some(host), Some(bytes)) = (crate::host::host(), src.footprint.bytes()) else {
                return;
            };
            let Some(data) = buffer_range(src.resource, src.footprint.offset, bytes) else {
                return;
            };
            let Some((buffer, staging)) = list.upload(host, bytes) else {
                return;
            };
            // The list owns this mapped D3D12 upload buffer. Wine can report
            // its address as MEM_FREE, so use the host-upload path.
            list.defer_fill(staging, data, bytes as usize);
            from.resource = buffer.as_raw().cast_const().cast();
            from.footprint.offset = 0;
            #[cfg(feature = "diagnostics")]
            {
                // SAFETY: a live buffer.
                read = Some((unsafe { buffer.GetGPUVirtualAddress() }, bytes));
            }
            _upload = buffer;
        } else {
            let Some(texture) = crate::resource::host_resource(src.resource) else {
                return;
            };
            if crate::texture::title_written(&texture) {
                crate::texture::upload(list, src.resource, &texture);
            }
            ensure_state(list, src.resource, COPY_SOURCE);
            from.resource = texture.as_raw().cast_const().cast();
        }
        ensure_state(list, dst.resource, COPY_DEST);
        crate::texture::host_writes(list, dst.resource);
        let mut to_location = dst;
        to_location.resource = to.as_raw().cast_const().cast();
        let host_box = (!source_box.is_null()).then(|| source_box.cast::<D3D12_BOX>());
        #[cfg(feature = "diagnostics")]
        {
            let text = || {
                // SAFETY: a live texture; the box is the title's or null.
                let (d, b) = unsafe { (to.GetDesc(), host_box.map(|b| *b)) };
                format!(
                    "copy to {:?} {}x{} f{} sub/kind {} at {x},{y},{z} from {:?} kind {} footprint {:?} box {b:?}",
                    dst.resource,
                    d.Width,
                    d.Height,
                    d.Format.0,
                    dst.kind,
                    src.resource,
                    src.kind,
                    (src.kind == FOOTPRINT).then_some(src.footprint)
                )
            };
            match read {
                Some(range) => list.note_read(text, range),
                None => list.note_op(text),
            }
        }
        const _: () =
            assert!(size_of::<CopyLocation>() == size_of::<D3D12_TEXTURE_COPY_LOCATION>());
        // SAFETY: an open list; both locations in the host's layout, naming
        // live resources; the box is the title's (same layout).
        unsafe {
            list.list.CopyTextureRegion(
                (&raw const to_location).cast(),
                x,
                y,
                z,
                (&raw const from).cast(),
                host_box,
            )
        };
    }
    #[cfg(not(windows))]
    let _ = (this, dst, x, y, z, src, source_box);
}

/// Texture read backs since the last frame-rate line: how many, bytes
/// (diagnostics).
#[cfg(feature = "diagnostics")]
pub(crate) static READ_BACKS: [std::sync::atomic::AtomicU64; 2] = [const { std::sync::atomic::AtomicU64::new(0) }; 2];

/// (texture, buffer) pairs read back so far, each logged once (diagnostics).
#[cfg(feature = "diagnostics")]
static READ_BACK_PAIRS: std::sync::Mutex<Vec<(usize, usize)>> = std::sync::Mutex::new(Vec::new());

/// Texture to buffer (a read back): the region is copied on the host into
/// readback memory laid out as the title's footprint, and once the list has
/// run its rows go into the title's buffer (the title reads them on the CPU
/// after its fence).
#[cfg(windows)]
fn read_back(list: &crate::host::HostList, dst: &CopyLocation, at: [u32; 3], src: &CopyLocation, source_box: *const [u32; 6]) {
    use windows::Win32::Graphics::Direct3D12::{
        D3D12_BOX, D3D12_RESOURCE_DIMENSION_TEXTURE3D, D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL,
    };
    use windows::core::Interface;
    macro_rules! refused {
        ($why:literal) => {{
            static ONCE: std::sync::Once = std::sync::Once::new();
            ONCE.call_once(|| tracing::warn!(concat!("texture to buffer copy refused: ", $why)));
            return;
        }};
    }
    let (Some(host), Some(texture)) = (crate::host::host(), crate::resource::host_resource(src.resource)) else {
        refused!("no host texture");
    };
    // SAFETY: the title's box (same layout as the host's) or null; a live texture.
    let (b, desc) = unsafe { (source_box.cast::<D3D12_BOX>().as_ref(), texture.GetDesc()) };
    let size = match b {
        Some(b) => [b.right.saturating_sub(b.left), b.bottom.saturating_sub(b.top), b.back.saturating_sub(b.front)],
        None => {
            // A subresource index shares the footprint's first dword.
            let mip = src.footprint.offset as u32 % u32::from(desc.MipLevels.max(1));
            let depth = if desc.Dimension == D3D12_RESOURCE_DIMENSION_TEXTURE3D { u32::from(desc.DepthOrArraySize) } else { 1 };
            [((desc.Width >> mip) as u32).max(1), (desc.Height >> mip).max(1), (depth >> mip).max(1)]
        }
    };
    // The host takes footprints of 256-byte pitches only (as the console).
    if !dst.footprint.row_pitch.is_multiple_of(256) {
        refused!("a row pitch not a multiple of 256");
    }
    // shortcut: depth and stencil planes copy out in typeless formats of
    // their own; refused until a title is seen reading one back.
    if desc.Flags.0 & D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL.0 != 0 {
        refused!("a depth or stencil texture");
    }
    // The host copies into a footprint only within a buffer that holds all
    // of it, so readback memory spans the footprint, not just the region.
    let (Some(rows), Some(len)) = (dst.footprint.rows(at, size), dst.footprint.bytes()) else {
        refused!("a region outside its footprint");
    };
    let (Some(title), Some((buffer, data))) = (buffer_range(dst.resource, dst.footprint.offset, len), list.readback(host, len)) else {
        return;
    };
    if crate::texture::title_written(&texture) {
        crate::texture::upload(list, src.resource, &texture);
    }
    ensure_state(list, src.resource, COPY_SOURCE);
    let (mut to, mut from) = (*dst, *src);
    to.resource = buffer.as_raw().cast_const().cast();
    to.footprint.offset = 0;
    from.resource = texture.as_raw().cast_const().cast();
    #[cfg(feature = "diagnostics")]
    {
        let text = || {
            let mem = crate::resource::texture_memory(src.resource).map_or(0, |(at, _)| at as usize);
            format!(
                "read back {:?} (mem {mem:#x}, {}x{} f{} sub {}) {size:?} at {at:?} into {:?}+{:#x}, {len} bytes",
                src.resource,
                desc.Width,
                desc.Height,
                desc.Format.0,
                src.footprint.offset as u32,
                dst.resource,
                dst.footprint.offset
            )
        };
        READ_BACKS[0].fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        READ_BACKS[1].fetch_add(len, std::sync::atomic::Ordering::Relaxed);
        let pair = (src.resource as usize, dst.resource as usize);
        let mut seen = READ_BACK_PAIRS.lock().unwrap_or_else(|e| e.into_inner());
        if !seen.contains(&pair) {
            seen.push(pair);
            tracing::info!("a texture read back (first of this pair): {}", text());
        }
        list.note_op(text);
    }
    // SAFETY: an open list; both locations in the host's layout, naming live
    // resources; the region lies in the footprint (checked).
    unsafe {
        list.list.CopyTextureRegion(
            (&raw const to).cast(),
            at[0],
            at[1],
            at[2],
            (&raw const from).cast(),
            b.map(std::ptr::from_ref),
        )
    };
    let (data, title) = (data as usize, title as usize);
    list.defer_after(move || {
        for &(at, n) in &rows {
            let (at, n) = (at as usize, n as usize);
            let _writing = crate::watch::writing(title + at, n);
            crate::host::write_mapped((title + at) as *mut u8, (data + at) as *const u8, n);
        }
    });
}

#[allow(clippy::too_many_arguments)] // the method's own arguments
extern "system" fn copy_texture_region_x(
    this: This,
    dst: *const CopyLocation,
    x: u32,
    y: u32,
    z: u32,
    src: *const CopyLocation,
    source_box: *const [u32; 6],
    _flags: u32,
) {
    copy_texture_region(this, dst, x, y, z, src, source_box);
}

extern "system" fn resolve(
    this: This,
    dst: This,
    dst_sub: u32,
    src: This,
    src_sub: u32,
    format: u32,
) {
    #[cfg(windows)]
    if let Some(list) = host_list(this)
        && let (Some(to), Some(from)) = (
            crate::resource::host_resource(dst),
            crate::resource::host_resource(src),
        )
    {
        use windows::Win32::Graphics::Direct3D12::{
            D3D12_RESOLVE_MODE_AVERAGE, ID3D12GraphicsCommandList1,
        };
        use windows::Win32::Graphics::Dxgi::Common::DXGI_FORMAT;
        use windows::core::Interface;
        const RESOLVE_DEST: u32 = 0x1000;
        const RESOLVE_SOURCE: u32 = 0x2000;
        ensure_state(list, dst, RESOLVE_DEST);
        ensure_state(list, src, RESOLVE_SOURCE);
        crate::texture::host_writes(list, dst);
        #[cfg(feature = "diagnostics")]
        list.note_op(|| format!("resolve {src:?} to {dst:?} format {format}"));
        // SAFETY: an open list and live textures; the region is inside both.
        unsafe {
            let format = DXGI_FORMAT(format as i32);
            if let Some(rect) = resolve_region(to.GetDesc(), dst_sub, from.GetDesc(), src_sub) {
                if let Ok(list1) = list.list.cast::<ID3D12GraphicsCommandList1>() {
                    // The console clips a larger source to the destination.
                    list1.ResolveSubresourceRegion(
                        &to,
                        dst_sub,
                        0,
                        0,
                        &from,
                        src_sub,
                        Some(&rect),
                        format,
                        D3D12_RESOLVE_MODE_AVERAGE,
                    );
                }
            } else {
                list.list
                    .ResolveSubresource(&to, dst_sub, &from, src_sub, format);
            }
        }
    }
    #[cfg(not(windows))]
    let _ = (this, dst, dst_sub, src, src_sub, format);
}

/// Crop rectangle for `ResolveSubresourceRegion` when the selected subresource
/// mips differ in extent, or `None` when a full `ResolveSubresource` matches.
#[cfg(windows)]
fn resolve_region(
    dst: windows::Win32::Graphics::Direct3D12::D3D12_RESOURCE_DESC,
    dst_sub: u32,
    src: windows::Win32::Graphics::Direct3D12::D3D12_RESOURCE_DESC,
    src_sub: u32,
) -> Option<windows::Win32::Foundation::RECT> {
    let extent = |d: windows::Win32::Graphics::Direct3D12::D3D12_RESOURCE_DESC, subresource: u32| {
        // Mips are the low-order component of an array or plane subresource index.
        let mip = subresource % u32::from(d.MipLevels);
        ((d.Width >> mip).max(1), (d.Height >> mip).max(1))
    };
    let (dst_width, dst_height) = extent(dst, dst_sub);
    let (src_width, src_height) = extent(src, src_sub);
    if (dst_width, dst_height) == (src_width, src_height) {
        None
    } else {
        Some(windows::Win32::Foundation::RECT {
            left: 0,
            top: 0,
            right: dst_width.min(src_width) as i32,
            bottom: dst_height.min(src_height) as i32,
        })
    }
}

#[repr(C)]
pub(crate) struct Rect {
    left: i32,
    top: i32,
    right: i32,
    bottom: i32,
}

extern "system" fn clear_rtv(
    this: This,
    handle: u64,
    color: *const [f32; 4],
    count: u32,
    rects: *const Rect,
) {
    #[cfg(windows)]
    {
        let (Some(list), Some(record)) = (host_list(this), crate::descriptor::rtv_record(handle))
        else {
            return;
        };
        let (Some(texture), Some(host)) = (
            crate::resource::host_resource(record.resource),
            crate::host::host(),
        ) else {
            return;
        };
        // SAFETY: the title passes four floats (null is tolerated).
        let Some(color) = (unsafe { color.as_ref() }) else {
            return;
        };
        let rects = (count > 0 && !rects.is_null()).then(|| {
            // SAFETY: the title passes `count` rectangles, laid out as the host's.
            unsafe { std::slice::from_raw_parts(rects.cast(), count as usize) }
        });
        const RENDER_TARGET: u32 = 0x4;
        let rtv = host.rtv(Some(&texture), record.desc().as_ref());
        // SAFETY: an open list, a fresh view of a live texture.
        unsafe {
            if let Some((_, before)) = list.transition(record.resource, RENDER_TARGET) {
                use windows::Win32::Graphics::Direct3D12::D3D12_RESOURCE_STATES as State;
                list.list.ResourceBarrier(&[crate::host::transition(
                    &texture,
                    State(before as i32),
                    State(RENDER_TARGET as i32),
                )]);
            }
            list.list.ClearRenderTargetView(rtv, color, rects);
        }
    }
    #[cfg(not(windows))]
    let _ = (this, handle, color, count, rects);
}

/// `ClearDepthStencilView(handle, flags, depth, stencil, count, rects)`;
/// flags: 1 depth, 2 stencil.
extern "system" fn clear_dsv(
    this: This,
    handle: u64,
    flags: u32,
    depth: f32,
    stencil: u8,
    count: u32,
    rects: *const Rect,
) {
    #[cfg(windows)]
    {
        use windows::Win32::Graphics::Direct3D12::D3D12_CLEAR_FLAGS;
        let (Some(list), Some(record)) = (host_list(this), crate::descriptor::dsv_record(handle))
        else {
            return;
        };
        let (Some(texture), Some(host)) = (
            crate::resource::host_resource(record.resource),
            crate::host::host(),
        ) else {
            return;
        };
        let rects = (count > 0 && !rects.is_null()).then(|| {
            // SAFETY: the title passes `count` rectangles, laid out as the host's.
            unsafe { std::slice::from_raw_parts(rects.cast(), count as usize) }
        });
        const DEPTH_WRITE: u32 = 0x10;
        ensure_state(list, record.resource, DEPTH_WRITE);
        let dsv = host.dsv(&texture, record.depth_desc().as_ref());
        // SAFETY: an open list, a fresh view of a live texture.
        unsafe {
            list.list
                .ClearDepthStencilView(dsv, D3D12_CLEAR_FLAGS(flags as i32 & 3), depth, stencil, rects)
        };
    }
    #[cfg(not(windows))]
    let _ = (this, handle, flags, depth, stencil, count, rects);
}

/// `CopyMemoryToMemoryX(destination, source, size, flags)`: GPU addresses
/// are title memory, copied at submission like buffer copies (if mapped).
extern "system" fn copy_memory(this: This, dst: u64, src: u64, len: u64, _flags: u32) {
    #[cfg(windows)]
    if let Some(list) = host_list(this)
        && dst != 0
        && src != 0
    {
        let len = len as usize;
        list.defer(move || {
            if crate::texture::committed(src as *const u8, len)
                && crate::texture::committed(dst as *const u8, len)
            {
                // SAFETY: both ranges are committed title memory.
                unsafe { std::ptr::copy(src as *const u8, dst as *mut u8, len) };
            }
        });
    }
    #[cfg(not(windows))]
    let _ = (this, dst, src, len);
}

/// `DriverHintX(feature, value)`: no hints for the host.
extern "system" fn driver_hint(_: This, _feature: u64, _value: u64) -> u64 {
    0
}

/// `DispatchX(x, y, z, flags)`: a dispatch with console flags, which the
/// host has no use for (the ray-tracing builder dispatches this way).
extern "system" fn dispatch_x(this: This, x: u32, y: u32, z: u32, _flags: u32) {
    #[cfg(windows)]
    state::dispatch(this, x, y, z);
    #[cfg(not(windows))]
    {
        let _ = (this, x, y, z);
        count_dispatch();
    }
}

/// `BeginQuery(heap, type, index)`: recorded into the host list, in order
/// with the draws, for the types the host has (not timestamps, which only
/// end, nor a query running already).
extern "system" fn begin_query(this: This, heap: This, kind: u32, index: u32) {
    #[cfg(windows)]
    if let Some(list) = host_list(this)
        && let Some(host) = crate::query::host(heap, kind, index, 1)
        && list.begin_query(&host.heap, kind, index)
    {
        host.began(index);
    }
    #[cfg(not(windows))]
    let _ = (this, heap, kind, index);
}

/// `EndQuery(heap, type, index)`: takes a timestamp, or ends a query begun
/// in the same list; an end without its begin is dropped, so the query
/// has no host result.
extern "system" fn end_query(this: This, heap: This, kind: u32, index: u32) {
    #[cfg(windows)]
    if let Some(list) = host_list(this)
        && let Some(host) = crate::query::host(heap, kind, index, 1)
        && let Some(ended) = list.end_query(&host.heap, kind, index)
    {
        host.ended(index, ended);
    }
    #[cfg(not(windows))]
    let _ = (this, heap, kind, index);
}

/// `ResolveQueryData(heap, type, first, count, buffer, offset)`: writes
/// each query's result into the buffer once the list has run. Queries that
/// ran on the host GPU give its results ([`host_resolve`]); the others,
/// and every query of types the host lacks, stand-in values: timestamps
/// (type 2) the GPU clock then, occlusion counts (0) a million samples
/// passed and binary occlusion (1) visible, so nothing is culled by a
/// query that did not run. Results and stand-ins are written in the order
/// resolved, so the last resolve to an address is what the title reads.
/// shortcut: the console's other query types and result layouts are not
/// known; they get zeros, 16 bytes per query.
/// shortcut: the results reach title memory when the submission has run;
/// GPU commands later in the same submission that read the buffer (through
/// its host copy) see its earlier bytes.
extern "system" fn resolve_queries(
    this: This,
    heap: This,
    kind: u32,
    first: u32,
    count: u32,
    buffer: This,
    offset: u64,
) {
    use crate::query::{BINARY_OCCLUSION, OCCLUSION, PIPELINE_STATISTICS, TIMESTAMP};
    let Some((data, size)) = crate::resource::buffer_bytes(buffer) else {
        return;
    };
    let (what, bytes) = match kind {
        OCCLUSION => (Written::Value(1 << 20), 8),
        BINARY_OCCLUSION => (Written::Value(1), 8),
        TIMESTAMP => (Written::GpuTimestamp, 8),
        PIPELINE_STATISTICS => (Written::Value(0), 88),
        _ => (Written::Value(0), 16),
    };
    let Some(end) = offset.checked_add(u64::from(count) * bytes) else {
        return;
    };
    if end > size {
        return;
    }
    #[cfg(windows)]
    let measured = host_resolve(this, heap, kind, first, count, bytes);
    #[cfg(not(windows))]
    let (measured, _) = (None::<(Vec<bool>, usize)>, (heap, first));
    for i in 0..u64::from(count) {
        // Where the host's result for this query will be, if it has one.
        let result = measured
            .as_ref()
            .filter(|(ran, _)| ran[i as usize])
            .map(|(_, results)| results + (i * bytes) as usize);
        // Wider results are written as 8-byte pieces.
        for piece in 0..bytes / 8 {
            // SAFETY: inside the buffer, checked above.
            let at = unsafe { data.add((offset + i * bytes + piece * 8) as usize) };
            let what = match result {
                Some(raw) if kind == TIMESTAMP => Written::HostTimestamp(raw),
                Some(raw) => Written::HostValue(raw + (piece * 8) as usize),
                None if piece == 0 => what,
                None => Written::Value(0),
            };
            defer_write(this, at as u64, what, 8);
        }
    }
}

/// Resolves on the host GPU the queries of `[first, first + count)` whose
/// last run there was of type `kind` (`bytes` per result): which ones, and
/// where the results will be once the list has run, query `first`'s first
/// (those of queries not resolved are not written). None if none is.
///
/// The results go into read-back memory of the list, never into a title
/// buffer's host copy: the copy is in whatever state the title's shaders
/// and indirect executions left it in, which a resolve cannot rely on.
#[cfg(windows)]
fn host_resolve(this: This, heap: This, kind: u32, first: u32, count: u32, bytes: u64) -> Option<(Vec<bool>, usize)> {
    use windows::Win32::Graphics::Direct3D12::D3D12_QUERY_TYPE;
    let (list, host) = (host_list(this)?, crate::query::host(heap, kind, first, count)?);
    let measured: Vec<bool> = (first..first + count).map(|i| host.measured(i, kind)).collect();
    // Runs of measured queries: (first index from `first`, count).
    let mut runs: Vec<(u32, u32)> = Vec::new();
    for (i, _) in measured.iter().enumerate().filter(|(_, m)| **m) {
        match runs.last_mut() {
            Some((from, n)) if *from + *n == i as u32 => *n += 1,
            _ => runs.push((i as u32, 1)),
        }
    }
    if runs.is_empty() {
        return None;
    }
    let (back, base, results) = list.query_readback(u64::from(count) * bytes)?;
    list.keep(&host.heap);
    #[cfg(feature = "diagnostics")]
    list.note_op(|| format!("resolve {count} queries of type {kind} from {first}"));
    for &(from, n) in &runs {
        let offset = base + u64::from(from) * bytes;
        // SAFETY: an open list, a live heap kept until the list is reset,
        // and the list's read-back memory.
        unsafe { list.list.ResolveQueryData(&host.heap, D3D12_QUERY_TYPE(kind as i32), first + from, n, &back, offset) };
    }
    Some((measured, results as usize))
}

/// IUnknown 3, object 4, device child 1, allocator 2.
static ALLOCATOR_FUNCTIONS: Table<24> = {
    let mut f = object::d3d12_object(object::missing_table::<ALLOCATOR, 24>());
    f[0] = allocator_query as *const ();
    f[2] = allocator_release as *const ();
    f[7] = object::parent_device as *const (); // 0x38
    f[8] = allocator_reset as *const (); // 0x40
    f[9] = object::ignore as *const (); // 0x48 allocation callback
    Table(f)
};

extern "system" fn set_pipeline(this: This, pipeline: This) {
    crate::pipeline::bound(pipeline);
    #[cfg(windows)]
    with_state(this, |_, state| state.set_pipeline(pipeline));
    #[cfg(not(windows))]
    let _ = this;
}

/// Pipeline syncs: waits for earlier work and cache flushes, at the top or
/// bottom of the pipe `(flags, control, address, size)`. Host lists order
/// their own work and every submission is waited for, so there is nothing
/// to do.
extern "system" fn sync(_: This, _flags: u32, _control: u32, _address: u64, _size: u64) {}

/// Reset with console flags `(allocator, pipeline, flags)`; the flags only
/// tune the console's command memory.
extern "system" fn reset_x(this: This, allocator: This, pipeline: This, _flags: u32) -> HRESULT {
    reset(this, allocator, pipeline)
}

/// Newer lists' 0x4a0 `(size, alignment) -> pointer`: memory the list
/// owns until it is reset, which the CPU writes and the GPU reads at the
/// same address; null on failure. Grand Theft Auto VI's BVH builder puts
/// its instance descriptions there (and wrote through our old
/// `E_NOTIMPL` as a pointer).
extern "system" fn allocate_upload(this: This, size: u64, alignment: u64) -> *mut u8 {
    let align = usize::try_from(alignment).unwrap_or(0).max(16);
    let Some(len) = usize::try_from(size).ok().and_then(|s| s.checked_add(align)) else {
        return std::ptr::null_mut();
    };
    if this.is_null() || !align.is_power_of_two() {
        return std::ptr::null_mut();
    }
    let Some(block) = Upload::new(len) else {
        return std::ptr::null_mut();
    };
    let start = block.start();
    // SAFETY: the offset is below `align`, so `size` bytes fit after it.
    let at = unsafe { start.add(start.align_offset(align)) };
    // SAFETY: the title passes one of our lists.
    let uploads = unsafe { &(*this.cast::<List>()).uploads };
    uploads.lock().unwrap_or_else(|e| e.into_inner()).push(block);
    at
}

/// Memory [`allocate_upload`] hands out, zeroed, freed when the list is
/// reset. On Windows it lies in [4 GiB, 2^40), where translated shaders
/// find title memory by address (`crate::memory`; the console's GPU
/// addresses have 40 bits), and is a buffer of its own to the host copies
/// (`crate::shadow`), so its copy is exactly it.
struct Upload {
    #[cfg(windows)]
    at: usize,
    #[cfg(windows)]
    len: usize,
    #[cfg(not(windows))]
    block: Vec<u8>,
}

impl Upload {
    #[cfg(windows)]
    fn new(len: usize) -> Option<Upload> {
        use windows_sys::Win32::System::Memory::{
            MEM_ADDRESS_REQUIREMENTS, MEM_COMMIT, MEM_EXTENDED_PARAMETER, MEM_RESERVE,
            MemExtendedParameterAddressRequirements, PAGE_READWRITE, VirtualAlloc2,
        };
        const MEM_WRITE_WATCH: u32 = 0x20_0000;
        let mut requirements = MEM_ADDRESS_REQUIREMENTS {
            LowestStartingAddress: (1usize << 32) as *mut _,
            HighestEndingAddress: ((1usize << 40) - 1) as *mut _,
            Alignment: 0,
        };
        let mut parameter = MEM_EXTENDED_PARAMETER::default();
        parameter.Anonymous1._bitfield = MemExtendedParameterAddressRequirements as u64;
        parameter.Anonymous2.Pointer = (&raw mut requirements).cast();
        // Watched like the title's pools: copies take only written pages.
        // SAFETY: a fresh allocation; the parameter points at `requirements`,
        // which outlives the call.
        let at = unsafe {
            VirtualAlloc2(
                std::ptr::null_mut(),
                std::ptr::null(),
                len,
                MEM_RESERVE | MEM_COMMIT | MEM_WRITE_WATCH,
                PAGE_READWRITE,
                &mut parameter,
                1,
            )
        } as usize;
        if at == 0 {
            return None;
        }
        crate::shadow::register(at, at as u64, len as u64);
        UPLOADS[0].fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        UPLOADS[1].fetch_add(len as u64, std::sync::atomic::Ordering::Relaxed);
        Some(Upload { at, len })
    }

    #[cfg(not(windows))]
    fn new(len: usize) -> Option<Upload> {
        Some(Upload {
            block: vec![0; len],
        })
    }

    fn start(&self) -> *mut u8 {
        #[cfg(windows)]
        return self.at as *mut u8;
        #[cfg(not(windows))]
        return self.block.as_ptr().cast_mut();
    }
}

#[cfg(windows)]
impl Drop for Upload {
    fn drop(&mut self) {
        use windows_sys::Win32::System::Memory::{MEM_RELEASE, VirtualFree};
        crate::shadow::unregister(self.at, self.at as u64);
        UPLOADS[0].fetch_sub(1, std::sync::atomic::Ordering::Relaxed);
        UPLOADS[1].fetch_sub(self.len as u64, std::sync::atomic::Ordering::Relaxed);
        // Not guarded any more: memory allocated here next is new.
        crate::watch::title_memory_remapped(self.at, self.len);
        // SAFETY: our allocation; lists that still use its copy hold that.
        unsafe { VirtualFree(self.at as *mut _, 0, MEM_RELEASE) };
    }
}

// SAFETY: plain memory, handed out once.
#[cfg(windows)]
unsafe impl Send for Upload {}

/// Live list upload blocks and their bytes (the statistics).
#[cfg(windows)]
pub(crate) static UPLOADS: [std::sync::atomic::AtomicU64; 2] =
    [std::sync::atomic::AtomicU64::new(0), std::sync::atomic::AtomicU64::new(0)];

/// Draw state the host has no use for (root signatures: ours come with
/// the pipeline; depth bounds, depth order and shading rate hints): any
/// arguments, nothing returned.
extern "system" fn ignored_state(_: This) {}

extern "system" fn depth_bounds(this: This, min: f32, max: f32) {
    #[cfg(windows)]
    with_state(this, |list, s| s.set_depth_bounds(list, min, max));
    #[cfg(not(windows))]
    let _ = (this, min, max);
}

/// The title's draw state (decision 0007, step 3), kept for its draws or
/// set on the host list at once. Without a host GPU these do nothing.
#[cfg(windows)]
mod state {
    use super::{This, with_state};
    use crate::draw::{IndexView, VertexView};
    use windows::Win32::Foundation::RECT;
    use windows::Win32::Graphics::Direct3D12::D3D12_VIEWPORT;

    /// `count` items at `items`, or none.
    fn items<'a, T>(items: *const T, count: u32) -> &'a [T] {
        if items.is_null() || count == 0 {
            return &[];
        }
        // SAFETY: the title passes `count` items.
        unsafe { std::slice::from_raw_parts(items, count as usize) }
    }

    pub(super) extern "system" fn topology(this: This, topology: u32) {
        with_state(this, |list, s| s.set_topology(list, topology));
    }
    pub(super) extern "system" fn viewports(
        this: This,
        count: u32,
        viewports: *const D3D12_VIEWPORT,
    ) {
        with_state(this, |list, s| s.viewports(list, items(viewports, count)));
    }
    pub(super) extern "system" fn scissors(this: This, count: u32, rects: *const RECT) {
        with_state(this, |list, s| s.scissors(list, items(rects, count)));
    }
    pub(super) extern "system" fn blend_factor(this: This, factor: *const [f32; 4]) {
        // SAFETY: the title passes four floats or null.
        with_state(this, |list, s| s.blend_factor(list, unsafe { factor.as_ref() }));
    }
    pub(super) extern "system" fn stencil_ref(this: This, reference: u32) {
        with_state(this, |list, s| s.stencil_ref(list, reference));
    }
    pub(super) extern "system" fn heaps(this: This, count: u32, heaps: *const This) {
        with_state(this, |_, s| s.set_heaps(items(heaps, count)));
    }
    /// `ClearUnorderedAccessViewUint(GPU handle, CPU handle, resource,
    /// values, rectangle count, rectangles)`.
    pub(super) extern "system" fn clear_unordered_uint(
        this: This,
        _gpu: u64,
        cpu: u64,
        _resource: This,
        values: *const [u32; 4],
        count: u32,
        rects: *const RECT,
    ) {
        clear_unordered(this, cpu, values, false, items(rects, count));
    }
    /// `ClearUnorderedAccessViewFloat`, the same with four floats.
    pub(super) extern "system" fn clear_unordered_float(
        this: This,
        _gpu: u64,
        cpu: u64,
        _resource: This,
        values: *const [u32; 4],
        count: u32,
        rects: *const RECT,
    ) {
        clear_unordered(this, cpu, values, true, items(rects, count));
    }
    fn clear_unordered(this: This, cpu: u64, values: *const [u32; 4], float: bool, rects: &[RECT]) {
        // SAFETY: the title passes four values (null is tolerated).
        let Some(&values) = (unsafe { values.as_ref() }) else {
            return;
        };
        with_state(this, |list, s| {
            s.clear_unordered(list, cpu, values, float, rects);
        });
    }
    pub(super) extern "system" fn compute_table(this: This, slot: u32, handle: u64) {
        with_state(this, |_, s| s.set_table(true, slot, handle));
    }
    pub(super) extern "system" fn graphics_table(this: This, slot: u32, handle: u64) {
        with_state(this, |_, s| s.set_table(false, slot, handle));
    }
    pub(super) extern "system" fn compute_constant(this: This, slot: u32, value: u32, offset: u32) {
        with_state(this, |_, s| s.set_constants(true, slot, &[value], offset));
    }
    pub(super) extern "system" fn graphics_constant(
        this: This,
        slot: u32,
        value: u32,
        offset: u32,
    ) {
        with_state(this, |_, s| s.set_constants(false, slot, &[value], offset));
    }
    pub(super) extern "system" fn compute_constants(
        this: This,
        slot: u32,
        count: u32,
        values: *const u32,
        offset: u32,
    ) {
        with_state(this, |_, s| {
            s.set_constants(true, slot, items(values, count), offset)
        });
    }
    pub(super) extern "system" fn graphics_constants(
        this: This,
        slot: u32,
        count: u32,
        values: *const u32,
        offset: u32,
    ) {
        with_state(this, |_, s| {
            s.set_constants(false, slot, items(values, count), offset)
        });
    }
    /// Root constant buffer, shader resource and unordered access views.
    pub(super) extern "system" fn compute_view(this: This, slot: u32, address: u64) {
        with_state(this, |_, s| s.set_view(true, slot, address));
    }
    pub(super) extern "system" fn graphics_view(this: This, slot: u32, address: u64) {
        with_state(this, |_, s| s.set_view(false, slot, address));
    }
    pub(super) extern "system" fn index_buffer(this: This, view: *const IndexView) {
        // SAFETY: the title passes a view or null.
        with_state(this, |_, s| s.set_index_buffer(unsafe { view.as_ref() }));
    }
    pub(super) extern "system" fn vertex_buffers(
        this: This,
        start: u32,
        count: u32,
        views: *const VertexView,
    ) {
        with_state(this, |_, s| {
            if views.is_null() {
                s.clear_vertex_buffers(start, count);
            } else {
                s.set_vertex_buffers(start, items(views, count));
            }
        });
    }
    /// `(count, handles, one range, depth)`: with one range, the handles
    /// follow the first at the render target view size (128 bytes).
    pub(super) extern "system" fn targets(
        this: This,
        count: u32,
        handles: *const u64,
        one_range: i32,
        depth: *const u64,
    ) {
        let handles: Vec<u64> = if one_range != 0 && !handles.is_null() {
            // SAFETY: the title passes the first handle.
            let first = unsafe { *handles };
            (0..u64::from(count)).map(|i| first + 128 * i).collect()
        } else {
            items(handles, count).to_vec()
        };
        // SAFETY: the title passes a handle or null.
        let depth = unsafe { depth.as_ref() }.copied().unwrap_or(0);
        with_state(this, |list, s| s.set_targets(list, &handles, depth));
    }
    pub(super) extern "system" fn draw(
        this: This,
        vertices: u32,
        instances: u32,
        first: u32,
        first_instance: u32,
    ) {
        super::count_draw();
        with_state(this, |list, s| {
            s.draw(list, vertices, instances, first, first_instance)
        });
    }
    pub(super) extern "system" fn dispatch(this: This, x: u32, y: u32, z: u32) {
        super::count_dispatch();
        with_state(this, |list, s| s.dispatch(list, x, y, z));
    }
    /// `DispatchMesh(x, y, z)` (the PC method of List6); the title's mesh
    /// programs take `x` groups only.
    pub(super) extern "system" fn dispatch_mesh(this: This, x: u32, y: u32, z: u32) {
        super::count_draw();
        if y * z != 1 {
            static ONCE: std::sync::Once = std::sync::Once::new();
            ONCE.call_once(|| tracing::warn!(x, y, z, "a mesh dispatch with y or z groups"));
        }
        with_state(this, |list, s| s.dispatch_mesh(list, x));
    }
    /// `DispatchMeshX(x, flags, indirect arguments, payload)`.
    /// shortcut: dispatches whose group count is in GPU memory (indirect)
    /// are dropped; make them ExecuteIndirect when the title uses them.
    pub(super) extern "system" fn dispatch_mesh_x(
        this: This,
        x: u32,
        _flags: u64,
        indirect: u64,
        _payload: u64,
    ) {
        super::count_draw();
        if indirect != 0 {
            static ONCE: std::sync::Once = std::sync::Once::new();
            ONCE.call_once(|| tracing::warn!("indirect mesh dispatches dropped"));
            if let Some(list) = super::host_list(this) {
                list.missed_draw();
            }
            return;
        }
        with_state(this, |list, s| s.dispatch_mesh(list, x));
    }
    pub(super) extern "system" fn draw_indexed(
        this: This,
        indices: u32,
        instances: u32,
        first: u32,
        base: i32,
        first_instance: u32,
    ) {
        super::count_draw();
        with_state(this, |list, s| {
            s.draw_indexed(list, indices, instances, first, base, first_instance)
        });
    }
}

/// Without a host GPU every draw state call is ignored.
#[cfg(not(windows))]
mod state {
    pub(super) use super::dispatch;
    pub(super) use super::draw;
    pub(super) use super::draw as draw_indexed;
    pub(super) use super::draw as dispatch_mesh;
    pub(super) use super::draw as dispatch_mesh_x;
    pub(super) use super::ignored_state as blend_factor;
    pub(super) use super::ignored_state as clear_unordered_float;
    pub(super) use super::ignored_state as clear_unordered_uint;
    pub(super) use super::ignored_state as compute_constant;
    pub(super) use super::ignored_state as compute_constants;
    pub(super) use super::ignored_state as compute_table;
    pub(super) use super::ignored_state as compute_view;
    pub(super) use super::ignored_state as graphics_constant;
    pub(super) use super::ignored_state as graphics_constants;
    pub(super) use super::ignored_state as graphics_table;
    pub(super) use super::ignored_state as graphics_view;
    pub(super) use super::ignored_state as heaps;
    pub(super) use super::ignored_state as index_buffer;
    pub(super) use super::ignored_state as scissors;
    pub(super) use super::ignored_state as stencil_ref;
    pub(super) use super::ignored_state as targets;
    pub(super) use super::ignored_state as topology;
    pub(super) use super::ignored_state as vertex_buffers;
    pub(super) use super::ignored_state as viewports;
}

fn count_draw() {
    DRAWS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
}

/// Draws of the kinds the host does not run yet (indirect, mesh), and
/// dispatches, counted per frame for the log.
/// shortcut: dropped until their shaders translate (decision 0007).
#[cfg(not(windows))]
extern "system" fn draw(_: This) {
    count_draw();
}

/// `ExecuteIndirect(signature, max count, arguments, offset, count buffer,
/// offset)`. Grand Theft Auto VI's signatures (title+0x2479466, 2026-11-24)
/// hold one argument each: draw (stride 16), indexed draw (20), dispatch
/// (12), mesh dispatch (12), and the console's single-pass dispatch, type
/// 105 (stride 12: the three group counts; its flags live in the
/// signature, the title passes none), run as a plain dispatch. They run
/// through the host's indirect execution (`State::execute_indirect`).
/// shortcut: signatures that change root arguments are dropped (logged
/// once per signature).
extern "system" fn execute_indirect(
    this: This,
    signature: This,
    max: u32,
    arguments: This,
    offset: u64,
    count: This,
    count_offset: u64,
) {
    count_draw();
    let Some((stride, args)) = crate::signature::layout(signature) else {
        return;
    };
    let address = crate::resource::buffer_bytes(arguments).map(|(p, _)| p as u64 + offset);
    let counter = crate::resource::buffer_bytes(count).map(|(p, _)| p as u64 + count_offset);
    let mut ran = false;
    #[cfg(windows)]
    if let ([[kind, ..]], Some(address)) = (args, address)
        && (count.is_null() || counter.is_some())
    {
        const DISPATCH: u32 = 2;
        const DISPATCH_SINGLE: u32 = 105;
        let kind = if *kind == DISPATCH_SINGLE { DISPATCH } else { *kind };
        with_state(this, |list, s| {
            ran = s.execute_indirect(list, (kind, stride, max), address, counter);
        });
    } else {
        #[cfg(all(windows, feature = "diagnostics"))]
        crate::pipeline::note_why(format!(
            "indirect signature {args:?}: arguments in a buffer {}, count in a buffer {}",
            address.is_some(),
            count.is_null() || counter.is_some()
        ));
    }
    if ran {
        return;
    }
    #[cfg(windows)]
    if let Some(list) = host_list(this) {
        list.missed_draw();
    }
    INDIRECT.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    static SEEN: std::sync::Mutex<Vec<usize>> = std::sync::Mutex::new(Vec::new());
    let mut seen = SEEN.lock().unwrap_or_else(|e| e.into_inner());
    if !seen.contains(&(signature as usize)) {
        seen.push(signature as usize);
        tracing::info!(
            stride,
            ?args,
            max,
            address = ?address.map(|a| format!("{a:#x}")),
            counted = !count.is_null(),
            "indirect execution dropped (first time for this signature)"
        );
    }
}

/// Indirect executions dropped since the last call.
pub(crate) fn take_indirect() -> u64 {
    INDIRECT.swap(0, std::sync::atomic::Ordering::Relaxed)
}
static INDIRECT: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
fn count_dispatch() {
    DISPATCHES.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
}
#[cfg(not(windows))]
extern "system" fn dispatch(_: This) {
    count_dispatch();
}
static DRAWS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
static DISPATCHES: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// Draws (any kind, mesh and indirect included) and compute dispatches
/// recorded since the last call.
pub(crate) fn take_counts() -> (u64, u64) {
    use std::sync::atomic::Ordering::Relaxed;
    (DRAWS.swap(0, Relaxed), DISPATCHES.swap(0, Relaxed))
}


/// An enhanced barrier group (`ID3D12GraphicsCommandList7::Barrier`):
/// barrier type (0 global, 1 texture, 2 buffer), barrier count, and
/// pointer to the barrier array.
#[repr(C)]
#[derive(Clone, Copy)]
#[cfg_attr(not(windows), allow(dead_code))]
pub(crate) struct BarrierGroup {
    kind: u32,
    count: u32,
    barriers: *const u8,
}

/// An enhanced texture barrier (`D3D12_TEXTURE_BARRIER`).
#[repr(C)]
#[derive(Clone, Copy)]
#[allow(dead_code)]
struct TextureBarrier {
    sync_before: u32,
    sync_after: u32,
    access_before: u32,
    access_after: u32,
    layout_before: u32,
    layout_after: u32,
    resource: This,
    subresources: [u32; 6],
    flags: u32,
    _pad: u32,
}

#[cfg(windows)]
const _: () = {
    use windows::Win32::Graphics::Direct3D12::{
        D3D12_BARRIER_GROUP, D3D12_BARRIER_SUBRESOURCE_RANGE, D3D12_TEXTURE_BARRIER,
    };
    assert!(size_of::<BarrierGroup>() == size_of::<D3D12_BARRIER_GROUP>());
    assert!(size_of::<TextureBarrier>() == size_of::<D3D12_TEXTURE_BARRIER>());
    assert!(offset_of!(TextureBarrier, resource) == offset_of!(D3D12_TEXTURE_BARRIER, pResource));
    assert!(offset_of!(TextureBarrier, subresources) == offset_of!(D3D12_TEXTURE_BARRIER, Subresources));
    assert!(size_of::<[u32; 6]>() == size_of::<D3D12_BARRIER_SUBRESOURCE_RANGE>());
    assert!(offset_of!(D3D12_BARRIER_SUBRESOURCE_RANGE, IndexOrFirstMipLevel) == 0);
    assert!(offset_of!(D3D12_BARRIER_SUBRESOURCE_RANGE, NumMipLevels) == 4);
    assert!(offset_of!(D3D12_BARRIER_SUBRESOURCE_RANGE, FirstArraySlice) == 8);
    assert!(offset_of!(D3D12_BARRIER_SUBRESOURCE_RANGE, NumArraySlices) == 12);
    assert!(offset_of!(D3D12_BARRIER_SUBRESOURCE_RANGE, FirstPlane) == 16);
    assert!(offset_of!(D3D12_BARRIER_SUBRESOURCE_RANGE, NumPlanes) == 20);
};

/// Whether an enhanced texture barrier's `D3D12_BARRIER_SUBRESOURCE_RANGE`
/// covers the whole `texture` (else a partial subresource transition must
/// not clobber sibling mips/slices/planes on our whole-resource tracker).
#[cfg(windows)]
fn is_whole_resource_barrier(
    subresources: &[u32; 6],
    texture: &windows::Win32::Graphics::Direct3D12::ID3D12Resource,
) -> bool {
    use windows::Win32::Graphics::Direct3D12::{
        D3D12_FEATURE_DATA_FORMAT_INFO, D3D12_FEATURE_FORMAT_INFO,
        D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES, D3D12_RESOURCE_DIMENSION_TEXTURE3D,
    };
    // SAFETY: `texture` is a live D3D12 resource.
    let desc = unsafe { texture.GetDesc() };
    let planes = crate::host::host().map_or(1u32, |host| {
        let mut info = D3D12_FEATURE_DATA_FORMAT_INFO {
            Format: desc.Format,
            PlaneCount: 0,
        };
        // SAFETY: `info` has the exact layout expected by `D3D12_FEATURE_FORMAT_INFO`.
        let ok = unsafe {
            host.device.CheckFeatureSupport(
                D3D12_FEATURE_FORMAT_INFO,
                (&raw mut info).cast(),
                size_of_val(&info) as u32,
            )
        }
        .is_ok();
        if ok {
            u32::from(info.PlaneCount).max(1)
        } else {
            1
        }
    });
    let mips = u32::from(desc.MipLevels).max(1);
    let slices = if desc.Dimension == D3D12_RESOURCE_DIMENSION_TEXTURE3D {
        1
    } else {
        u32::from(desc.DepthOrArraySize).max(1)
    };
    let [first_mip_or_index, num_mips, first_slice, num_slices, first_plane, num_planes] =
        *subresources;
    if num_mips == 0 {
        first_mip_or_index == D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES
            || (first_mip_or_index == 0 && mips == 1 && slices == 1 && planes == 1)
    } else {
        first_mip_or_index == 0
            && num_mips >= mips
            && first_slice == 0
            && num_slices >= slices
            && first_plane == 0
            && num_planes >= planes
    }
}

/// Maps an enhanced barrier `D3D12_BARRIER_LAYOUT` value to the legacy host
/// `D3D12_RESOURCE_STATES` texture state bitmask, or `None` for `UNDEFINED`
/// and video-only layouts.
#[cfg_attr(not(windows), allow(dead_code))]
fn enhanced_layout_state(layout: u32) -> Option<u32> {
    Some(match layout {
        // COMMON / PRESENT and queue-specific COMMON
        0 | 18 | 24 | 30 => 0,
        // GENERIC_READ and queue-specific GENERIC_READ (texture bits only)
        1 | 19 | 25 => 0x40 | 0x80 | 0x800,
        // RENDER_TARGET
        2 => 0x4,
        // UNORDERED_ACCESS and queue-specific UNORDERED_ACCESS
        3 | 20 | 26 => 0x8,
        // DEPTH_STENCIL_WRITE
        4 => 0x10,
        // DEPTH_STENCIL_READ
        5 => 0x20,
        // SHADER_RESOURCE and queue-specific SHADER_RESOURCE
        6 | 21 | 27 => 0x40 | 0x80,
        // COPY_SOURCE and queue-specific COPY_SOURCE
        7 | 22 | 28 => 0x800,
        // COPY_DEST and queue-specific COPY_DEST
        8 | 23 | 29 => 0x400,
        // RESOLVE_SOURCE
        9 => 0x2000,
        // RESOLVE_DEST
        10 => 0x1000,
        // SHADING_RATE_SOURCE
        11 => 0x100_0000,
        // UNDEFINED (u32::MAX), video-only (12..=17), or unknown
        _ => return None,
    })
}

/// The enhanced barriers `(group count, groups)`: translates texture
/// layout transitions onto the host list's tracked texture states and
/// emits a global UAV barrier when shader UAV ordering or non-texture
/// groups require it.
extern "system" fn barrier_groups(this: This, count: u32, groups: *const BarrierGroup) {
    if count == 0 || groups.is_null() {
        return;
    }
    #[cfg(windows)]
    if let Some(list) = host_list(this) {
        use windows::Win32::Graphics::Direct3D12::{
            D3D12_RESOURCE_BARRIER, D3D12_RESOURCE_BARRIER_0, D3D12_RESOURCE_BARRIER_FLAG_NONE,
            D3D12_RESOURCE_BARRIER_TYPE_UAV, D3D12_RESOURCE_STATES as State,
            D3D12_RESOURCE_UAV_BARRIER,
        };
        use windows::core::Interface;
        const TEXTURE: u32 = 1;
        const SYNC_SPLIT: u32 = 0x8000_0000;
        const ACCESS_UAV: u32 = 0x10;
        const STATE_UAV: u32 = 0x8;

        // SAFETY: the title passes `count` barrier groups.
        let groups = unsafe { std::slice::from_raw_parts(groups, count as usize) };
        // Which kinds of barriers (0 global, 1 texture, 2 buffer) and how many.
        #[cfg(feature = "diagnostics")]
        list.note_op(|| {
            format!(
                "barrier groups {:?}",
                groups.iter().map(|g| (g.kind, g.count)).collect::<Vec<_>>()
            )
        });
        let mut out: Vec<D3D12_RESOURCE_BARRIER> = Vec::new();
        let mut seen_transitions: Vec<usize> = Vec::new();
        let mut need_global_uav = false;

        for g in groups {
            if g.kind == TEXTURE {
                if g.count == 0 || g.barriers.is_null() {
                    continue;
                }
                // SAFETY: a texture barrier group passes `g.count` `TextureBarrier` entries.
                let barriers = unsafe {
                    std::slice::from_raw_parts(
                        g.barriers.cast::<TextureBarrier>(),
                        g.count as usize,
                    )
                };
                for b in barriers {
                    if b.sync_after == SYNC_SPLIT || b.resource.is_null() {
                        continue;
                    }
                    let wanted = enhanced_layout_state(b.layout_after);
                    let whole = crate::resource::host_resource(b.resource)
                        .is_some_and(|tex| is_whole_resource_barrier(&b.subresources, &tex));
                    let mut emitted_transition = false;
                    if whole
                        && let Some(wanted) = wanted
                        && let Some((texture, before)) = list.transition(b.resource, wanted)
                    {
                        let after = crate::host::legal_state(&texture, wanted);
                        if before != after {
                            let tex_key = texture.as_raw() as usize;
                            if seen_transitions.contains(&tex_key) {
                                // SAFETY: an open list and live resources.
                                unsafe { list.list.ResourceBarrier(&out) };
                                out.clear();
                                seen_transitions.clear();
                            }
                            seen_transitions.push(tex_key);
                            out.push(crate::host::transition(
                                &texture,
                                State(before as i32),
                                State(after as i32),
                            ));
                            emitted_transition = true;
                        }
                    }
                    if ((b.access_before | b.access_after) & ACCESS_UAV != 0)
                        || (wanted == Some(STATE_UAV) && !emitted_transition)
                    {
                        need_global_uav = true;
                    }
                }
            } else if g.count > 0 {
                need_global_uav = true;
            }
        }

        if need_global_uav {
            out.push(D3D12_RESOURCE_BARRIER {
                Type: D3D12_RESOURCE_BARRIER_TYPE_UAV,
                Flags: D3D12_RESOURCE_BARRIER_FLAG_NONE,
                Anonymous: D3D12_RESOURCE_BARRIER_0 {
                    UAV: std::mem::ManuallyDrop::new(D3D12_RESOURCE_UAV_BARRIER {
                        pResource: std::mem::ManuallyDrop::new(None),
                    }),
                },
            });
        }
        if !out.is_empty() {
            // SAFETY: an open list and live resources.
            unsafe { list.list.ResourceBarrier(&out) };
        }
    }
    #[cfg(not(windows))]
    let _ = this;
}

/// The bottom-of-pipe value writes `(address, value, flush, flags)`, 32 and
/// 64 bits: done when the list has run, as the GPU would. Flag 1 writes the
/// GPU timestamp instead (64-bit form), flag 2 the system timestamp.
extern "system" fn write32(this: This, address: u64, value: u32, _flush: u32, _flags: u32) {
    defer_write(this, address, Written::Value(u64::from(value)), 4);
}
pub(crate) extern "system" fn write64(this: This, address: u64, value: u64, _flush: u32, flags: u32) {
    const GPU_TIMESTAMP: u32 = 1;
    const SYSTEM_TIMESTAMP: u32 = 2;
    let what = match flags {
        GPU_TIMESTAMP => Written::GpuTimestamp,
        SYSTEM_TIMESTAMP => Written::SystemTimestamp,
        _ => Written::Value(value),
    };
    defer_write(this, address, what, 8);
}

fn defer_write(this: This, address: u64, what: Written, bytes: usize) {
    if address == 0 {
        return;
    }
    #[cfg(windows)]
    if let Some(list) = host_list(this) {
        list.defer_write(address as *mut u8, what, bytes);
        return;
    }
    let _ = this;
    // Without a host GPU nothing runs later: write now.
    what.write(address as *mut u8, bytes);
}

/// The list methods by their slot in older titles' lists (the 2020-era
/// layout): IUnknown 3, object 4, device child 1, list 1, graphics list 96
/// (59 as on PC, then console ones), then versions 1-6 (6 + 1 + 1 + 11 + 3
/// + 13): 140 slots.
const METHODS: [(usize, *const ()); 52] = [
    (0, list_query as *const ()),
    (2, list_release as *const ()),
    (7, object::parent_device as *const ()),      // 0x38
    (8, list_type as *const ()),                  // 0x40
    (9, close as *const ()),                      // 0x48
    (10, reset as *const ()),                     // 0x50
    (12, state::draw as *const ()),               // 0x60 instanced
    (13, state::draw_indexed as *const ()),       // 0x68 indexed
    (14, state::dispatch as *const ()),           // 0x70
    (15, copy_buffer_region as *const ()),        // 0x78
    (16, copy_texture_region as *const ()),       // 0x80
    (17, copy_resource as *const ()),             // 0x88
    (19, resolve as *const ()),                   // 0x98
    (20, state::topology as *const ()),           // 0xa0
    (21, state::viewports as *const ()),          // 0xa8
    (22, state::scissors as *const ()),           // 0xb0
    (23, state::blend_factor as *const ()),       // 0xb8
    (24, state::stencil_ref as *const ()),        // 0xc0
    (25, set_pipeline as *const ()),              // 0xc8
    (26, barrier as *const ()),                   // 0xd0
    (32, state::heaps as *const ()),              // 0x100
    (33, ignored_state as *const ()),             // 0x108 compute root signature
    (34, ignored_state as *const ()),             // 0x110 graphics root signature
    (35, state::compute_table as *const ()),      // 0x118
    (36, state::graphics_table as *const ()),     // 0x120
    (37, state::compute_constant as *const ()),   // 0x128
    (38, state::graphics_constant as *const ()),  // 0x130
    (39, state::compute_constants as *const ()),  // 0x138
    (40, state::graphics_constants as *const ()), // 0x140
    (41, state::compute_view as *const ()),       // 0x148 CBV
    (42, state::graphics_view as *const ()),      // 0x150 CBV
    (43, state::compute_view as *const ()),       // 0x158 SRV
    (44, state::graphics_view as *const ()),      // 0x160 SRV
    (45, state::compute_view as *const ()),       // 0x168 UAV
    (46, state::graphics_view as *const ()),      // 0x170 UAV
    (47, state::index_buffer as *const ()),       // 0x178
    (48, state::vertex_buffers as *const ()),     // 0x180
    (50, state::targets as *const ()),            // 0x190
    (51, clear_dsv as *const ()),                 // 0x198
    (52, clear_rtv as *const ()),                 // 0x1a0
    (59, execute_indirect as *const ()),          // 0x1d8 indirect
    (60, copy_memory as *const ()),               // 0x1e0
    (68, depth_bounds as *const ()),              // 0x220
    (69, write32 as *const ()),                   // 0x228
    (70, write64 as *const ()),                   // 0x230
    (71, wait32 as *const ()),                    // 0x238 (newer 0x248)
    (72, wait64 as *const ()),                    // 0x240 (newer 0x250)
    (73, ignored_state as *const ()),             // 0x248 pixel shader depth order
    (83, copy_texture_region_x as *const ()),     // 0x298
    (84, copy_resource_x as *const ()),           // 0x2a0
    (124, ignored_state as *const ()),            // 0x3e0 shading rate
    (127, state::dispatch_mesh as *const ()),     // 0x3f8 mesh dispatch
];
/// The rest of [`METHODS`], kept apart only to stay readable.
const MORE_METHODS: [(usize, *const ()); 12] = [
    (53, state::clear_unordered_uint as *const ()), // 0x1a8
    (54, state::clear_unordered_float as *const ()), // 0x1b0
    (55, ignored_state as *const ()),               // 0x1b8 DiscardResource
    (28, begin_query as *const ()),                 // 0xe0 BeginQuery
    (29, end_query as *const ()),                   // 0xe8 EndQuery
    (30, resolve_queries as *const ()),             // 0xf0
    (78, dispatch_x as *const ()),                  // 0x270
    (98, state::dispatch_mesh_x as *const ()),      // 0x310
    (99, driver_hint as *const ()),                 // 0x318
    // Pipeline syncs at the top and bottom of the pipe, 0x408 and 0x410.
    (129, sync as *const ()),
    (130, sync as *const ()),
    (137, reset_x as *const ()), // 0x448
];

/// Newer titles' lists have two more methods in the console part of the
/// base list, after the clears (0x198/0x1a0 as before) and before
/// ExecuteIndirect, which moved from 0x1d8 to 0x1e8: every slot from 0x1d8
/// on is two later (taken here as starting exactly at 0x1d8). Inferred from
/// Grand Theft Auto VI's call shapes: ExecuteIndirect 0x1e8, bottom-of-pipe
/// writes 0x238/0x240, waits 0x248/0x250, depth bounds 0x230, shading rate
/// 0x3f0/0x400, syncs 0x418/0x420, ResetX 0x458 (asserts on failure).
const fn newer_slot(slot: usize) -> usize {
    if slot >= 59 { slot + 2 } else { slot }
}

/// Methods only newer titles' lists have. 0x490: the enhanced barrier
/// `(group count, groups)`, with 16-byte groups `(type, count, barriers)`
/// and 40-byte buffer barriers (title+0x95e0 builds them). 0x470 `(u64)`,
/// called from Grand Theft Auto VI's state flush with its result unused.
/// shortcut: 0x470's purpose is unknown; ignored until a title shows it
/// matters.
const NEWER_METHODS: [(usize, *const ()); 3] = [
    (142, ignored_state as *const ()),
    (146, barrier_groups as *const ()),
    (148, allocate_upload as *const ()),
];

const fn list_table(newer: bool) -> [*const (); 160] {
    let mut f = object::d3d12_object(object::missing_table::<LIST, 160>());
    let mut i = 0;
    while i < METHODS.len() + MORE_METHODS.len() {
        let (slot, method) = if i < METHODS.len() {
            METHODS[i]
        } else {
            MORE_METHODS[i - METHODS.len()]
        };
        f[if newer { newer_slot(slot) } else { slot }] = method;
        i += 1;
    }
    let mut i = 0;
    while newer && i < NEWER_METHODS.len() {
        f[NEWER_METHODS[i].0] = NEWER_METHODS[i].1;
        i += 1;
    }
    f
}

/// Every slot up to the limit exists and logs, for the methods newer
/// titles add.
static LIST_FUNCTIONS: Table<160> = Table(list_table(false));
static NEWER_LIST_FUNCTIONS: Table<160> = Table(list_table(true));

#[cfg(all(test, windows))]
pub(crate) fn has_newer_table(list: This) -> bool {
    // SAFETY: one of our lists; its first field is the table. The device
    // tests can switch layouts, so use this list's table.
    let table = unsafe { *list.cast::<*const *const ()>() };
    table == NEWER_LIST_FUNCTIONS.0.as_ptr()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::object::slot;
    use std::sync::{
        Arc,
        atomic::{AtomicUsize, Ordering},
    };
    use tracing::{Event, Metadata, Subscriber, span};

    struct PipelineBinds(Arc<AtomicUsize>);

    impl Subscriber for PipelineBinds {
        fn enabled(&self, metadata: &Metadata<'_>) -> bool {
            metadata.is_event()
                && metadata.target() == "victory_gpu::pipeline"
                && metadata.fields().field("pipeline").is_some()
                && metadata.fields().field("frame").is_some()
        }

        fn new_span(&self, _: &span::Attributes<'_>) -> span::Id {
            unreachable!()
        }

        fn record(&self, _: &span::Id, _: &span::Record<'_>) {}
        fn record_follows_from(&self, _: &span::Id, _: &span::Id) {}
        fn enter(&self, _: &span::Id) {}
        fn exit(&self, _: &span::Id) {}

        fn event(&self, event: &Event<'_>) {
            if self.enabled(event.metadata()) {
                self.0.fetch_add(1, Ordering::Relaxed);
            }
        }
    }

    #[test]
    fn lists_bind_the_initial_pipeline_on_creation() {
        use crate::pipeline::{ComputeHead, create_compute};

        // A synthetic compute description matching the public ABI.
        #[repr(C)]
        struct ComputeDesc {
            root: This,
            code: *const u8,
            size: usize,
        }

        let null = std::ptr::null();
        let mut allocator = null;
        assert_eq!(
            create_allocator(null, 0, &ALLOCATOR_IIDS[0], &mut allocator),
            S_OK
        );
        let desc = ListDesc {
            kind: 0,
            flags: 0,
            nodes: 0,
            barrier_batch: 0,
        };
        let bytes = [0xa5, 0xb6, 0xc7];
        let compute = ComputeDesc {
            root: null,
            code: bytes.as_ptr(),
            size: bytes.len(),
        };
        let compute = std::ptr::from_ref(&compute).cast::<ComputeHead>();
        let release = |o: This| slot::<extern "system" fn(This) -> u32>(o, 0x10)(o);
        let binds = Arc::new(AtomicUsize::new(0));
        let subscriber = tracing::Dispatch::new(PipelineBinds(binds.clone()));

        tracing::dispatcher::with_default(&subscriber, || {
            for console in [false, true] {
                let create = |pipeline, out| {
                    if console {
                        create_list_x(null, &desc, allocator, pipeline, &LIST_IIDS[1], out)
                    } else {
                        create_list(null, 0, 0, allocator, pipeline, &LIST_IIDS[1], out)
                    }
                };
                let mut pipeline = null;
                let mut list = null;
                assert_eq!(
                    create_compute(null, compute, &PAGEABLE_IIDS[0], &mut pipeline),
                    S_OK
                );
                let before = binds.load(Ordering::Relaxed);

                assert_eq!(create(null, &mut list), S_OK);
                assert_eq!(binds.load(Ordering::Relaxed), before);
                assert_eq!(release(list), 0);

                assert_eq!(create(pipeline, &mut list), S_OK);
                assert_eq!(binds.load(Ordering::Relaxed), before + 1);
                assert_eq!(release(list), 0);
                assert_eq!(release(pipeline), 0);
            }
        });
        assert_eq!(release(allocator), 0);
    }

    /// ClearUnorderedAccessViewUint/Float and DiscardResource come before
    /// the methods newer lists insert: the AMD upscaler calls them there.
    #[test]
    fn unordered_access_clears_have_the_same_slots_in_both_layouts() {
        for table in [&LIST_FUNCTIONS, &NEWER_LIST_FUNCTIONS] {
            assert_eq!(table.0[0x1a8 / 8], state::clear_unordered_uint as *const ());
            assert_eq!(
                table.0[0x1b0 / 8],
                state::clear_unordered_float as *const ()
            );
            assert_eq!(table.0[0x1b8 / 8], ignored_state as *const ());
        }
    }

    #[cfg(windows)]
    #[test]
    fn resolve_uses_the_selected_mip_dimensions() {
        use windows::Win32::Foundation::RECT;
        use windows::Win32::Graphics::Direct3D12::*;
        use windows::Win32::Graphics::Dxgi::Common::*;
        use windows::core::Interface;

        struct Held(This);
        impl Drop for Held {
            fn drop(&mut self) {
                slot::<extern "system" fn(This) -> u32>(self.0, 0x10)(self.0);
            }
        }

        let Some(host) = crate::host::host() else {
            return;
        };
        let device = &host.device;
        let format = DXGI_FORMAT_R8G8B8A8_UNORM;
        let null = std::ptr::null();
        let iid = ID3D12Resource::IID;
        let heap = D3D12_HEAP_PROPERTIES {
            Type: D3D12_HEAP_TYPE_DEFAULT,
            ..Default::default()
        };
        let make = |width: u32, mips, slices, samples| {
            let desc = D3D12_RESOURCE_DESC {
                Dimension: D3D12_RESOURCE_DIMENSION_TEXTURE2D,
                Width: u64::from(width),
                Height: width,
                MipLevels: mips,
                DepthOrArraySize: slices,
                Format: format,
                SampleDesc: DXGI_SAMPLE_DESC {
                    Count: samples,
                    Quality: 0,
                },
                Flags: D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET,
                ..Default::default()
            };
            let mut resource = null;
            assert_eq!(
                crate::resource::create_committed(
                    null,
                    (&raw const heap).cast(),
                    0,
                    (&raw const desc).cast(),
                    D3D12_RESOURCE_STATE_RENDER_TARGET.0 as u32,
                    null.cast(),
                    (&raw const iid).cast(),
                    &mut resource,
                ),
                S_OK
            );
            Held(resource)
        };
        let source_pixel = |x: u32, y: u32| {
            [
                (x * 64) as u8,
                (y * 64) as u8,
                (32 + x * 16 + y * 4) as u8,
                255,
            ]
        };

        for (size, mips, slice, mip, extent) in [
            (4, 2, 0, 1, 2),
            (8, 3, 0, 2, 2),
            (8, 2, 0, 1, 4),
            (4, 2, 1, 1, 2),
            (4, 0, 0, 1, 2),
        ] {
            let mut allocator = null;
            assert_eq!(
                create_allocator(null, 0, &ALLOCATOR_IIDS[0], &mut allocator),
                S_OK
            );
            let allocator = Held(allocator);
            let mut list = null;
            assert_eq!(
                create_list(null, 0, 0, allocator.0, null, &LIST_IIDS[1], &mut list),
                S_OK
            );
            let list = Held(list);
            let recorded = host_list(list.0).expect("host command list");
            let src = make(4, 1, 1, 4);
            let dst = make(size, mips, slice + 1, 1);
            let src_tex = crate::resource::host_resource(src.0).expect("4x MSAA source");
            let dst_tex = crate::resource::host_resource(dst.0).expect("single-sample destination");
            // SAFETY: live host resources. GetDesc reports the actual mip count.
            let (src_desc, dst_desc) = unsafe { (src_tex.GetDesc(), dst_tex.GetDesc()) };
            let actual_mips = dst_desc.MipLevels;
            assert_eq!(actual_mips, if mips == 0 { 3 } else { mips });
            let first = u32::from(slice) * u32::from(actual_mips);
            let expected_region = (extent != 4).then_some(RECT {
                left: 0,
                top: 0,
                right: extent as i32,
                bottom: extent as i32,
            });
            assert_eq!(
                resolve_region(dst_desc, first + mip, src_desc, 0),
                expected_region,
                "{size}x{size} mips={mips} slice={slice} mip={mip}"
            );
            let _ = host.log_messages();

            let source_rtv = host.rtv(&src_tex, None);
            for y in 0..4 {
                for x in 0..4 {
                    let rect = RECT {
                        left: x,
                        top: y,
                        right: x + 1,
                        bottom: y + 1,
                    };
                    let color = source_pixel(x as u32, y as u32).map(|v| f32::from(v) / 255.0);
                    // SAFETY: the one-pixel rect lies inside the live multisample target.
                    unsafe {
                        recorded
                            .list
                            .ClearRenderTargetView(source_rtv, &color, Some(&[rect]))
                    };
                }
            }
            for (target_mip, color) in [(0, [0.0, 1.0, 1.0, 1.0]), (mip, [1.0, 0.0, 1.0, 1.0])] {
                let (dimension, view) = if slice == 0 {
                    (
                        D3D12_RTV_DIMENSION_TEXTURE2D,
                        D3D12_RENDER_TARGET_VIEW_DESC_0 {
                            Texture2D: D3D12_TEX2D_RTV {
                                MipSlice: target_mip,
                                PlaneSlice: 0,
                            },
                        },
                    )
                } else {
                    (
                        D3D12_RTV_DIMENSION_TEXTURE2DARRAY,
                        D3D12_RENDER_TARGET_VIEW_DESC_0 {
                            Texture2DArray: D3D12_TEX2D_ARRAY_RTV {
                                MipSlice: target_mip,
                                FirstArraySlice: u32::from(slice),
                                ArraySize: 1,
                                PlaneSlice: 0,
                            },
                        },
                    )
                };
                let desc = D3D12_RENDER_TARGET_VIEW_DESC {
                    Format: format,
                    ViewDimension: dimension,
                    Anonymous: view,
                };
                let rtv = host.rtv(&dst_tex, Some(&desc));
                // SAFETY: the view is live and its subresource begins in RENDER_TARGET.
                unsafe { recorded.list.ClearRenderTargetView(rtv, &color, None) };
            }
            assert_eq!(host.log_messages(), None);

            let render_target = D3D12_RESOURCE_STATE_RENDER_TARGET.0 as u32;
            ensure_state(recorded, src.0, render_target);
            ensure_state(recorded, dst.0, render_target);
            resolve(list.0, dst.0, first + mip, src.0, 0, format.0 as u32);
            assert_eq!(host.log_messages(), None);

            ensure_state(recorded, dst.0, D3D12_RESOURCE_STATE_COPY_SOURCE.0 as u32);
            let reads = [first, first + mip].map(|sub| {
                // SAFETY: the list is open and the destination is in COPY_SOURCE.
                unsafe { read_resolve_test_mip(device, &recorded.list, &dst_tex, sub) }
            });
            assert_eq!(close(list.0), S_OK);
            host.execute(&[recorded]);
            assert_eq!(host.log_messages(), None);

            for ((buffer, fp), (dim, resolved)) in
                reads.into_iter().zip([(size, false), (extent, true)])
            {
                let f = fp.Footprint;
                assert_eq!((f.Width, f.Height), (dim, dim));
                let mut pixels = std::ptr::null_mut();
                // SAFETY: submission has finished and all offsets stay inside the footprint.
                unsafe {
                    buffer.Map(0, None, Some(&mut pixels)).unwrap();
                    for y in 0..dim {
                        for x in 0..dim {
                            let offset = fp.Offset as usize + (y * f.RowPitch + x * 4) as usize;
                            let expected = if resolved {
                                source_pixel(x, y)
                            } else {
                                [0, 255, 255, 255]
                            };
                            let actual =
                                std::slice::from_raw_parts(pixels.cast::<u8>().add(offset), 4);
                            assert_eq!(
                                actual, expected,
                                "{size}x{size} mips={mips} slice={slice} mip={mip} at ({x},{y})"
                            );
                        }
                    }
                    buffer.Unmap(0, None);
                }
            }
        }
    }

    #[cfg(windows)]
    unsafe fn read_resolve_test_mip(
        device: &windows::Win32::Graphics::Direct3D12::ID3D12Device,
        list: &windows::Win32::Graphics::Direct3D12::ID3D12GraphicsCommandList,
        texture: &windows::Win32::Graphics::Direct3D12::ID3D12Resource,
        subresource: u32,
    ) -> (
        windows::Win32::Graphics::Direct3D12::ID3D12Resource,
        windows::Win32::Graphics::Direct3D12::D3D12_PLACED_SUBRESOURCE_FOOTPRINT,
    ) {
        use windows::Win32::Graphics::Direct3D12::*;
        use windows::Win32::Graphics::Dxgi::Common::DXGI_SAMPLE_DESC;
        let mut footprint = D3D12_PLACED_SUBRESOURCE_FOOTPRINT::default();
        let mut total = 0;
        // SAFETY: a live single-sample texture and an open list with the mip in COPY_SOURCE.
        unsafe {
            device.GetCopyableFootprints(
                &texture.GetDesc(),
                subresource,
                1,
                0,
                Some(&mut footprint),
                None,
                None,
                Some(&mut total),
            );
            let mut buffer = None;
            device
                .CreateCommittedResource(
                    &D3D12_HEAP_PROPERTIES {
                        Type: D3D12_HEAP_TYPE_READBACK,
                        ..Default::default()
                    },
                    D3D12_HEAP_FLAG_NONE,
                    &D3D12_RESOURCE_DESC {
                        Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
                        Width: total,
                        Height: 1,
                        DepthOrArraySize: 1,
                        MipLevels: 1,
                        SampleDesc: DXGI_SAMPLE_DESC {
                            Count: 1,
                            Quality: 0,
                        },
                        Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
                        ..Default::default()
                    },
                    D3D12_RESOURCE_STATE_COPY_DEST,
                    None,
                    &mut buffer,
                )
                .unwrap();
            let buffer: ID3D12Resource = buffer.unwrap();
            let to = D3D12_TEXTURE_COPY_LOCATION {
                pResource: std::mem::transmute_copy(&buffer),
                Type: D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT,
                Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 {
                    PlacedFootprint: footprint,
                },
            };
            let from = D3D12_TEXTURE_COPY_LOCATION {
                pResource: std::mem::transmute_copy(texture),
                Type: D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX,
                Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 {
                    SubresourceIndex: subresource,
                },
            };
            list.CopyTextureRegion(&to, 0, 0, 0, &from, None);
            (buffer, footprint)
        }
    }

    #[test]
    fn every_inline_command_falls_back_to_the_table() {
        let null = std::ptr::null();
        let mut allocator = null;
        assert_eq!(
            create_allocator(null, 0, &ALLOCATOR_IIDS[0], &mut allocator),
            S_OK
        );
        let mut list = null;
        let hr = create_list(null, 0, 0, allocator, null, &LIST_IIDS[1], &mut list);
        assert_eq!(hr, S_OK);
        // What the title's inline draw checks: put < draw limit.
        // SAFETY: the list is 0x560 bytes.
        let (put, draw_limit, api_limit) = unsafe {
            let at = |o: usize| list.cast::<u8>().add(o).cast::<u64>().read();
            (at(0x18), at(0x28), at(0x30))
        };
        assert!(put >= draw_limit && put >= api_limit);
        // The console's ID3D12GraphicsCommandList6, as Grand Theft Auto VI asks.
        let mut same = null;
        assert_eq!(list_query(list, &LIST_IIDS[7], &mut same), S_OK);
        assert_eq!(same, list);
        assert_eq!(slot::<extern "system" fn(This) -> u32>(list, 0x10)(list), 1);
        assert_eq!(slot::<extern "system" fn(This) -> u32>(list, 0x40)(list), 0);
        assert_eq!(
            slot::<extern "system" fn(This) -> HRESULT>(list, 0x48)(list),
            S_OK
        );
        // List-owned upload memory: aligned, writable, below 2^40 (where
        // shaders find title memory), gone at reset.
        let p = allocate_upload(list, 40, 0x100);
        assert!(!p.is_null() && (p as usize).is_multiple_of(0x100));
        #[cfg(windows)]
        assert!((1 << 32..1 << 40).contains(&(p as usize)), "{p:p}");
        // SAFETY: 40 bytes were handed out.
        unsafe { p.write_bytes(0xab, 40) };
        assert!(allocate_upload(list, 8, 0x30).is_null(), "not a power of two");
        assert_eq!(reset(list, allocator, null), S_OK);
        // SAFETY: the list is ours.
        assert!(unsafe { &(*list.cast::<List>()).uploads }.lock().unwrap().is_empty());
        let release = |o: This| slot::<extern "system" fn(This) -> u32>(o, 0x10)(o);
        assert_eq!(release(list), 0);
        assert_eq!(release(allocator), 0);
    }

    /// An offset near `u64::MAX` must not overflow and write outside the
    /// query destination buffer.
    #[test]
    fn resolve_queries_rejects_overflowing_offset() {
        let null = std::ptr::null();
        let mut allocator = null;
        assert_eq!(
            create_allocator(null, 0, &ALLOCATOR_IIDS[0], &mut allocator),
            S_OK
        );
        let mut list = null;
        assert_eq!(
            create_list(null, 0, 0, allocator, null, &LIST_IIDS[1], &mut list),
            S_OK
        );
        let buffer = crate::resource::test_buffer(64);
        // Used to overflow `offset + count * 8` (panic in debug, wrap in release).
        resolve_queries(list, null, 0, 0, 2, buffer, u64::MAX - 8);
        let release = |o: This| slot::<extern "system" fn(This) -> u32>(o, 0x10)(o);
        for object in [buffer, list, allocator] {
            assert_eq!(release(object), 0);
        }
    }

    fn new_newer_list(allocator: This) -> This {
        let mut out = std::ptr::null();
        let list = Box::new(List {
            object: Object::new(&NEWER_LIST_FUNCTIONS),
            kind: 0,
            put: [0; 5],
            reserved: [0; 64],
            graphics_root_headers: [0; 128],
            compute_root_headers: [0; 128],
            pipeline_shadow: [0; 8],
            newer: [0; 256],
            uploads: Default::default(),
            #[cfg(windows)]
            hosts: std::sync::Mutex::new(
                crate::host::host()
                    .and_then(|h| h.new_list().ok())
                    .map(Arc::new)
                    .into_iter()
                    .collect(),
            ),
            #[cfg(windows)]
            segment: AtomicUsize::new(0),
            #[cfg(windows)]
            state: Default::default(),
        });
        let hr = object::hand_out(LIST, list, &LIST_IIDS[1], &LIST_IIDS, &mut out);
        assert_eq!(hr, S_OK);
        let _ = allocator;
        out
    }

    #[test]
    fn enhanced_layout_mapping_and_abi() {
        assert_eq!(size_of::<BarrierGroup>(), 16);
        assert_eq!(size_of::<TextureBarrier>(), 64);

        let expected: &[(u32, Option<u32>)] = &[
            (0, Some(0)),
            (1, Some(0x40 | 0x80 | 0x800)),
            (2, Some(0x4)),
            (3, Some(0x8)),
            (4, Some(0x10)),
            (5, Some(0x20)),
            (6, Some(0x40 | 0x80)),
            (7, Some(0x800)),
            (8, Some(0x400)),
            (9, Some(0x2000)),
            (10, Some(0x1000)),
            (11, Some(0x100_0000)),
            (12, None),
            (13, None),
            (14, None),
            (15, None),
            (16, None),
            (17, None),
            (18, Some(0)),
            (19, Some(0x40 | 0x80 | 0x800)),
            (20, Some(0x8)),
            (21, Some(0x40 | 0x80)),
            (22, Some(0x800)),
            (23, Some(0x400)),
            (24, Some(0)),
            (25, Some(0x40 | 0x80 | 0x800)),
            (26, Some(0x8)),
            (27, Some(0x40 | 0x80)),
            (28, Some(0x800)),
            (29, Some(0x400)),
            (30, Some(0)),
            (31, None),
            (u32::MAX, None),
        ];
        for &(layout, state) in expected {
            assert_eq!(enhanced_layout_state(layout), state, "layout {layout}");
        }

        #[cfg(windows)]
        {
            use windows::Win32::Graphics::Direct3D12::*;
            assert_eq!(size_of::<BarrierGroup>(), size_of::<D3D12_BARRIER_GROUP>());
            assert_eq!(
                size_of::<TextureBarrier>(),
                size_of::<D3D12_TEXTURE_BARRIER>()
            );
            assert_eq!(
                offset_of!(TextureBarrier, resource),
                offset_of!(D3D12_TEXTURE_BARRIER, pResource)
            );
            let generic_read = (D3D12_RESOURCE_STATE_ALL_SHADER_RESOURCE
                | D3D12_RESOURCE_STATE_COPY_SOURCE)
                .0 as u32;
            let legacy: &[(D3D12_BARRIER_LAYOUT, Option<u32>)] = &[
                (
                    D3D12_BARRIER_LAYOUT_COMMON,
                    Some(D3D12_RESOURCE_STATE_COMMON.0 as u32),
                ),
                (
                    D3D12_BARRIER_LAYOUT_PRESENT,
                    Some(D3D12_RESOURCE_STATE_PRESENT.0 as u32),
                ),
                (
                    D3D12_BARRIER_LAYOUT_DIRECT_QUEUE_COMMON,
                    Some(D3D12_RESOURCE_STATE_COMMON.0 as u32),
                ),
                (
                    D3D12_BARRIER_LAYOUT_COMPUTE_QUEUE_COMMON,
                    Some(D3D12_RESOURCE_STATE_COMMON.0 as u32),
                ),
                (
                    D3D12_BARRIER_LAYOUT_VIDEO_QUEUE_COMMON,
                    Some(D3D12_RESOURCE_STATE_COMMON.0 as u32),
                ),
                (D3D12_BARRIER_LAYOUT_GENERIC_READ, Some(generic_read)),
                (
                    D3D12_BARRIER_LAYOUT_DIRECT_QUEUE_GENERIC_READ,
                    Some(generic_read),
                ),
                (
                    D3D12_BARRIER_LAYOUT_COMPUTE_QUEUE_GENERIC_READ,
                    Some(generic_read),
                ),
                (
                    D3D12_BARRIER_LAYOUT_RENDER_TARGET,
                    Some(D3D12_RESOURCE_STATE_RENDER_TARGET.0 as u32),
                ),
                (
                    D3D12_BARRIER_LAYOUT_UNORDERED_ACCESS,
                    Some(D3D12_RESOURCE_STATE_UNORDERED_ACCESS.0 as u32),
                ),
                (
                    D3D12_BARRIER_LAYOUT_DIRECT_QUEUE_UNORDERED_ACCESS,
                    Some(D3D12_RESOURCE_STATE_UNORDERED_ACCESS.0 as u32),
                ),
                (
                    D3D12_BARRIER_LAYOUT_COMPUTE_QUEUE_UNORDERED_ACCESS,
                    Some(D3D12_RESOURCE_STATE_UNORDERED_ACCESS.0 as u32),
                ),
                (
                    D3D12_BARRIER_LAYOUT_DEPTH_STENCIL_WRITE,
                    Some(D3D12_RESOURCE_STATE_DEPTH_WRITE.0 as u32),
                ),
                (
                    D3D12_BARRIER_LAYOUT_DEPTH_STENCIL_READ,
                    Some(D3D12_RESOURCE_STATE_DEPTH_READ.0 as u32),
                ),
                (
                    D3D12_BARRIER_LAYOUT_SHADER_RESOURCE,
                    Some(D3D12_RESOURCE_STATE_ALL_SHADER_RESOURCE.0 as u32),
                ),
                (
                    D3D12_BARRIER_LAYOUT_DIRECT_QUEUE_SHADER_RESOURCE,
                    Some(D3D12_RESOURCE_STATE_ALL_SHADER_RESOURCE.0 as u32),
                ),
                (
                    D3D12_BARRIER_LAYOUT_COMPUTE_QUEUE_SHADER_RESOURCE,
                    Some(D3D12_RESOURCE_STATE_ALL_SHADER_RESOURCE.0 as u32),
                ),
                (
                    D3D12_BARRIER_LAYOUT_COPY_SOURCE,
                    Some(D3D12_RESOURCE_STATE_COPY_SOURCE.0 as u32),
                ),
                (
                    D3D12_BARRIER_LAYOUT_DIRECT_QUEUE_COPY_SOURCE,
                    Some(D3D12_RESOURCE_STATE_COPY_SOURCE.0 as u32),
                ),
                (
                    D3D12_BARRIER_LAYOUT_COMPUTE_QUEUE_COPY_SOURCE,
                    Some(D3D12_RESOURCE_STATE_COPY_SOURCE.0 as u32),
                ),
                (
                    D3D12_BARRIER_LAYOUT_COPY_DEST,
                    Some(D3D12_RESOURCE_STATE_COPY_DEST.0 as u32),
                ),
                (
                    D3D12_BARRIER_LAYOUT_DIRECT_QUEUE_COPY_DEST,
                    Some(D3D12_RESOURCE_STATE_COPY_DEST.0 as u32),
                ),
                (
                    D3D12_BARRIER_LAYOUT_COMPUTE_QUEUE_COPY_DEST,
                    Some(D3D12_RESOURCE_STATE_COPY_DEST.0 as u32),
                ),
                (
                    D3D12_BARRIER_LAYOUT_RESOLVE_SOURCE,
                    Some(D3D12_RESOURCE_STATE_RESOLVE_SOURCE.0 as u32),
                ),
                (
                    D3D12_BARRIER_LAYOUT_RESOLVE_DEST,
                    Some(D3D12_RESOURCE_STATE_RESOLVE_DEST.0 as u32),
                ),
                (
                    D3D12_BARRIER_LAYOUT_SHADING_RATE_SOURCE,
                    Some(D3D12_RESOURCE_STATE_SHADING_RATE_SOURCE.0 as u32),
                ),
                (D3D12_BARRIER_LAYOUT_UNDEFINED, None),
                (D3D12_BARRIER_LAYOUT_VIDEO_DECODE_READ, None),
                (D3D12_BARRIER_LAYOUT_VIDEO_DECODE_WRITE, None),
                (D3D12_BARRIER_LAYOUT_VIDEO_PROCESS_READ, None),
                (D3D12_BARRIER_LAYOUT_VIDEO_PROCESS_WRITE, None),
                (D3D12_BARRIER_LAYOUT_VIDEO_ENCODE_READ, None),
                (D3D12_BARRIER_LAYOUT_VIDEO_ENCODE_WRITE, None),
            ];
            for &(layout, state) in legacy {
                assert_eq!(enhanced_layout_state(layout.0 as u32), state, "{layout:?}");
            }
        }

        // Null / zero-count / null-pointer groups are safe no-ops.
        let null = std::ptr::null();
        let mut allocator = null;
        assert_eq!(
            create_allocator(null, 0, &ALLOCATOR_IIDS[0], &mut allocator),
            S_OK
        );
        let list = new_newer_list(allocator);
        type BarrierFn = extern "system" fn(This, u32, *const BarrierGroup);
        let call_barrier = slot::<BarrierFn>(list, 0x490);
        call_barrier(null, 1, std::ptr::null());
        call_barrier(list, 0, std::ptr::null());
        call_barrier(list, 1, std::ptr::null());
        let null_ptr_group = BarrierGroup {
            kind: 1,
            count: 2,
            barriers: std::ptr::null(),
        };
        let zero_count_group = BarrierGroup {
            kind: 1,
            count: 0,
            barriers: (&raw const null_ptr_group).cast(),
        };
        let null_tex_barrier = TextureBarrier {
            sync_before: 0,
            sync_after: 0,
            access_before: 0,
            access_after: 0,
            layout_before: 0,
            layout_after: 7,
            resource: null,
            subresources: [0; 6],
            flags: 0,
            _pad: 0,
        };
        let null_res_group = BarrierGroup {
            kind: 1,
            count: 1,
            barriers: (&raw const null_tex_barrier).cast(),
        };
        let groups = [null_ptr_group, zero_count_group, null_res_group];
        call_barrier(list, groups.len() as u32, groups.as_ptr());
        assert_eq!(
            slot::<extern "system" fn(This) -> HRESULT>(list, 0x48)(list),
            S_OK
        );
        let release = |o: This| slot::<extern "system" fn(This) -> u32>(o, 0x10)(o);
        assert_eq!(release(list), 0);
        assert_eq!(release(allocator), 0);
    }

    #[cfg(windows)]
    #[derive(Clone, Default)]
    struct D3d12ErrorCapture(std::sync::Arc<std::sync::Mutex<Vec<String>>>);

    #[cfg(windows)]
    impl tracing::Subscriber for D3d12ErrorCapture {
        fn enabled(&self, metadata: &tracing::Metadata<'_>) -> bool {
            *metadata.level() <= tracing::Level::WARN
        }
        fn new_span(&self, _: &tracing::span::Attributes<'_>) -> tracing::span::Id {
            tracing::span::Id::from_u64(1)
        }
        fn record(&self, _: &tracing::span::Id, _: &tracing::span::Record<'_>) {}
        fn record_follows_from(&self, _: &tracing::span::Id, _: &tracing::span::Id) {}
        fn event(&self, event: &tracing::Event<'_>) {
            use windows::Win32::Graphics::Direct3D12::D3D12_MESSAGE_SEVERITY_ERROR;
            struct Visitor {
                severity: Option<i64>,
                message: Option<String>,
            }
            impl tracing::field::Visit for Visitor {
                fn record_i64(&mut self, field: &tracing::field::Field, value: i64) {
                    if field.name() == "severity" {
                        self.severity = Some(value);
                    }
                }
                fn record_debug(
                    &mut self,
                    field: &tracing::field::Field,
                    value: &dyn std::fmt::Debug,
                ) {
                    if field.name() == "message" {
                        self.message = Some(format!("{value:?}"));
                    }
                }
            }
            let mut v = Visitor {
                severity: None,
                message: None,
            };
            event.record(&mut v);
            if let (Some(sev), Some(msg)) = (v.severity, v.message)
                && msg.starts_with("D3D12:")
                && sev <= i64::from(D3D12_MESSAGE_SEVERITY_ERROR.0)
            {
                self.0.lock().unwrap().push(msg);
            }
        }
        fn enter(&self, _: &tracing::span::Id) {}
        fn exit(&self, _: &tracing::span::Id) {}
    }

    #[cfg(windows)]
    #[test]
    fn enhanced_texture_barrier_transitions_and_propagates_across_lists() {
        use std::sync::atomic::Ordering;
        use windows::Win32::Graphics::Direct3D12::*;
        use windows::Win32::Graphics::Dxgi::Common::{
            DXGI_FORMAT_R8G8B8A8_UNORM, DXGI_SAMPLE_DESC,
        };

        let Some(host) = crate::host::host() else {
            return;
        };
        // Clear any messages from earlier tests before capturing.
        let _ = host.log_messages();

        let errors = D3d12ErrorCapture::default();
        let subscriber = errors.clone();
        tracing::subscriber::with_default(subscriber, || {
            let null = std::ptr::null();
            let resource_iid = guid(
                0x696442be,
                0xa72e,
                0x4059,
                [0xbc, 0x79, 0x5b, 0x5c, 0x98, 0x04, 0x0f, 0xad],
            );
            let queue_iid = guid(
                0x0ec870a6,
                0x5d7e,
                0x4c22,
                [0x8c, 0xfc, 0x5b, 0xaa, 0xe0, 0x76, 0x16, 0xed],
            );

            assert_eq!(size_of::<D3D12_HEAP_PROPERTIES>(), 20);
            assert_eq!(size_of::<D3D12_RESOURCE_DESC>(), 56);
            let heap = D3D12_HEAP_PROPERTIES {
                Type: D3D12_HEAP_TYPE_DEFAULT,
                ..Default::default()
            };
            let tex_desc = D3D12_RESOURCE_DESC {
                Dimension: D3D12_RESOURCE_DIMENSION_TEXTURE2D,
                Alignment: 0,
                Width: 4,
                Height: 4,
                DepthOrArraySize: 1,
                MipLevels: 1,
                Format: DXGI_FORMAT_R8G8B8A8_UNORM,
                SampleDesc: DXGI_SAMPLE_DESC {
                    Count: 1,
                    Quality: 0,
                },
                Layout: D3D12_TEXTURE_LAYOUT_UNKNOWN,
                Flags: D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET
                    | D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
            };
            let mut tex = null;
            assert_eq!(
                crate::resource::create_committed(
                    null,
                    (&raw const heap).cast(),
                    0,
                    (&raw const tex_desc).cast(),
                    0,
                    null.cast(),
                    &resource_iid,
                    &mut tex,
                ),
                S_OK
            );
            let host_tex = crate::resource::host_resource(tex).unwrap();

            let mut rtv_slot = [0u64; 16];
            let rtv = rtv_slot.as_mut_ptr() as u64;
            crate::descriptor::create_rtv(null, tex, null.cast(), rtv);

            let mut queue = null;
            let queue_desc = [0u32; 4];
            assert_eq!(
                crate::queue::create(null, queue_desc.as_ptr().cast(), &queue_iid, &mut queue),
                S_OK
            );

            let mut allocator1 = null;
            assert_eq!(
                create_allocator(null, 0, &ALLOCATOR_IIDS[0], &mut allocator1),
                S_OK
            );
            let list1 = new_newer_list(allocator1);

            let readback_heap = D3D12_HEAP_PROPERTIES {
                Type: D3D12_HEAP_TYPE_READBACK,
                ..Default::default()
            };
            let readback_desc = D3D12_RESOURCE_DESC {
                Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
                Width: 1024,
                Height: 1,
                DepthOrArraySize: 1,
                MipLevels: 1,
                SampleDesc: DXGI_SAMPLE_DESC {
                    Count: 1,
                    Quality: 0,
                },
                Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
                ..Default::default()
            };
            let mut readback: Option<ID3D12Resource> = None;
            // SAFETY: plain creation of a 1 KiB readback buffer.
            unsafe {
                host.device
                    .CreateCommittedResource(
                        &readback_heap,
                        D3D12_HEAP_FLAG_NONE,
                        &readback_desc,
                        D3D12_RESOURCE_STATE_COPY_DEST,
                        None,
                        &mut readback,
                    )
                    .unwrap();
            }
            let readback = readback.unwrap();

            let record_copy_to_readback = |list: This| {
                let host_list = host_list(list).unwrap();
                let dst = D3D12_TEXTURE_COPY_LOCATION {
                    pResource: std::mem::ManuallyDrop::new(Some(readback.clone())),
                    Type: D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT,
                    Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 {
                        PlacedFootprint: D3D12_PLACED_SUBRESOURCE_FOOTPRINT {
                            Offset: 0,
                            Footprint: D3D12_SUBRESOURCE_FOOTPRINT {
                                Format: DXGI_FORMAT_R8G8B8A8_UNORM,
                                Width: 4,
                                Height: 4,
                                Depth: 1,
                                RowPitch: 256,
                            },
                        },
                    },
                };
                let src = D3D12_TEXTURE_COPY_LOCATION {
                    pResource: std::mem::ManuallyDrop::new(Some(host_tex.clone())),
                    Type: D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX,
                    Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 {
                        SubresourceIndex: 0,
                    },
                };
                // SAFETY: an open host list and live resources.
                unsafe { host_list.list.CopyTextureRegion(&dst, 0, 0, 0, &src, None) };
            };

            let read_pixel = || -> [u8; 4] {
                let mut mapped = std::ptr::null_mut();
                let read_range = D3D12_RANGE { Begin: 0, End: 4 };
                let written_range = D3D12_RANGE { Begin: 0, End: 0 };
                // SAFETY: execution waited for completion; readback buffer is 1 KiB.
                unsafe {
                    readback
                        .Map(0, Some(&read_range), Some(&mut mapped))
                        .unwrap();
                    let pixel = *(mapped.cast::<[u8; 4]>());
                    readback.Unmap(0, Some(&written_range));
                    pixel
                }
            };

            type BarrierFn = extern "system" fn(This, u32, *const BarrierGroup);
            type ExecuteFn = extern "system" fn(This, u32, *const This);
            let execute = slot::<ExecuteFn>(queue, 0x50);
            let tex_barrier = |sync_before: D3D12_BARRIER_SYNC,
                               sync_after: D3D12_BARRIER_SYNC,
                               access_before: D3D12_BARRIER_ACCESS,
                               access_after: D3D12_BARRIER_ACCESS,
                               layout_before: D3D12_BARRIER_LAYOUT,
                               layout_after: D3D12_BARRIER_LAYOUT|
             -> TextureBarrier {
                TextureBarrier {
                    sync_before: sync_before.0 as u32,
                    sync_after: sync_after.0 as u32,
                    access_before: access_before.0 as u32,
                    access_after: access_after.0 as u32,
                    layout_before: layout_before.0 as u32,
                    layout_after: layout_after.0 as u32,
                    resource: tex,
                    subresources: [0xffff_ffff, 0, 0, 0, 0, 0],
                    flags: 0,
                    _pad: 0,
                }
            };

            let unorm = |c: [u8; 4]| c.map(|b| f32::from(b) / 255.0);

            // 1. Basic transition: RENDER_TARGET -> COPY_SOURCE without ensure_state.
            clear_rtv(list1, rtv, &unorm([64, 128, 191, 255]), 0, null.cast());
            let b1 = tex_barrier(
                D3D12_BARRIER_SYNC_RENDER_TARGET,
                D3D12_BARRIER_SYNC_COPY,
                D3D12_BARRIER_ACCESS_RENDER_TARGET,
                D3D12_BARRIER_ACCESS_COPY_SOURCE,
                D3D12_BARRIER_LAYOUT_RENDER_TARGET,
                D3D12_BARRIER_LAYOUT_COPY_SOURCE,
            );
            let g1 = BarrierGroup {
                kind: D3D12_BARRIER_TYPE_TEXTURE.0 as u32,
                count: 1,
                barriers: (&raw const b1).cast(),
            };
            slot::<BarrierFn>(list1, 0x490)(list1, 1, &g1);
            record_copy_to_readback(list1);
            assert_eq!(close(list1), S_OK);
            execute(queue, 1, &list1);
            assert!(host.log_messages().is_none());
            assert!(errors.0.lock().unwrap().is_empty());
            assert_eq!(read_pixel(), [64, 128, 191, 255]);

            // 2. Split-begin skipped, split-end transitions.
            assert_eq!(reset(list1, allocator1, null), S_OK);
            clear_rtv(list1, rtv, &unorm([128, 64, 255, 255]), 0, null.cast());
            let b_split_begin = tex_barrier(
                D3D12_BARRIER_SYNC_RENDER_TARGET,
                D3D12_BARRIER_SYNC_SPLIT,
                D3D12_BARRIER_ACCESS_RENDER_TARGET,
                D3D12_BARRIER_ACCESS_NO_ACCESS,
                D3D12_BARRIER_LAYOUT_RENDER_TARGET,
                D3D12_BARRIER_LAYOUT_COPY_SOURCE,
            );
            let g_split_begin = BarrierGroup {
                kind: D3D12_BARRIER_TYPE_TEXTURE.0 as u32,
                count: 1,
                barriers: (&raw const b_split_begin).cast(),
            };
            slot::<BarrierFn>(list1, 0x490)(list1, 1, &g_split_begin);
            // Must still be in RENDER_TARGET so a raw RTV clear succeeds cleanly.
            // SAFETY: an open host list and live texture in RENDER_TARGET.
            unsafe {
                host_list(list1).unwrap().list.ClearRenderTargetView(
                    host.rtv(&host_tex, None),
                    &unorm([255, 128, 64, 255]),
                    None,
                );
            }
            let b_split_end = tex_barrier(
                D3D12_BARRIER_SYNC_SPLIT,
                D3D12_BARRIER_SYNC_COPY,
                D3D12_BARRIER_ACCESS_NO_ACCESS,
                D3D12_BARRIER_ACCESS_COPY_SOURCE,
                D3D12_BARRIER_LAYOUT_RENDER_TARGET,
                D3D12_BARRIER_LAYOUT_COPY_SOURCE,
            );
            let g_split_end = BarrierGroup {
                kind: D3D12_BARRIER_TYPE_TEXTURE.0 as u32,
                count: 1,
                barriers: (&raw const b_split_end).cast(),
            };
            slot::<BarrierFn>(list1, 0x490)(list1, 1, &g_split_end);
            record_copy_to_readback(list1);
            assert_eq!(close(list1), S_OK);
            execute(queue, 1, &list1);
            assert!(host.log_messages().is_none());
            assert!(errors.0.lock().unwrap().is_empty());
            assert_eq!(read_pixel(), [255, 128, 64, 255]);

            // 3. Duplicate-texture transitions in one Barrier() call + UAV ordering.
            assert_eq!(reset(list1, allocator1, null), S_OK);
            clear_rtv(list1, rtv, &unorm([0, 255, 128, 255]), 0, null.cast());
            let dup = [
                tex_barrier(
                    D3D12_BARRIER_SYNC_RENDER_TARGET,
                    D3D12_BARRIER_SYNC_COMPUTE_SHADING,
                    D3D12_BARRIER_ACCESS_RENDER_TARGET,
                    D3D12_BARRIER_ACCESS_UNORDERED_ACCESS,
                    D3D12_BARRIER_LAYOUT_RENDER_TARGET,
                    D3D12_BARRIER_LAYOUT_UNORDERED_ACCESS,
                ),
                tex_barrier(
                    D3D12_BARRIER_SYNC_COMPUTE_SHADING,
                    D3D12_BARRIER_SYNC_COPY,
                    D3D12_BARRIER_ACCESS_UNORDERED_ACCESS,
                    D3D12_BARRIER_ACCESS_COPY_SOURCE,
                    D3D12_BARRIER_LAYOUT_UNORDERED_ACCESS,
                    D3D12_BARRIER_LAYOUT_COPY_SOURCE,
                ),
            ];
            let g_dup = BarrierGroup {
                kind: D3D12_BARRIER_TYPE_TEXTURE.0 as u32,
                count: dup.len() as u32,
                barriers: dup.as_ptr().cast(),
            };
            slot::<BarrierFn>(list1, 0x490)(list1, 1, &g_dup);
            record_copy_to_readback(list1);
            assert_eq!(close(list1), S_OK);
            execute(queue, 1, &list1);
            assert!(host.log_messages().is_none());
            assert!(errors.0.lock().unwrap().is_empty());
            assert_eq!(read_pixel(), [0, 255, 128, 255]);

            // 4. Cross-list propagation via HostList::prelude.
            assert_eq!(reset(list1, allocator1, null), S_OK);
            let mut allocator2 = null;
            assert_eq!(
                create_allocator(null, 0, &ALLOCATOR_IIDS[0], &mut allocator2),
                S_OK
            );
            let list2 = new_newer_list(allocator2);

            clear_rtv(list1, rtv, &unorm([191, 64, 128, 255]), 0, null.cast());
            let b_to_srv = tex_barrier(
                D3D12_BARRIER_SYNC_RENDER_TARGET,
                D3D12_BARRIER_SYNC_ALL_SHADING,
                D3D12_BARRIER_ACCESS_RENDER_TARGET,
                D3D12_BARRIER_ACCESS_SHADER_RESOURCE,
                D3D12_BARRIER_LAYOUT_RENDER_TARGET,
                D3D12_BARRIER_LAYOUT_SHADER_RESOURCE,
            );
            let g_to_srv = BarrierGroup {
                kind: D3D12_BARRIER_TYPE_TEXTURE.0 as u32,
                count: 1,
                barriers: (&raw const b_to_srv).cast(),
            };
            slot::<BarrierFn>(list1, 0x490)(list1, 1, &g_to_srv);
            assert_eq!(close(list1), S_OK);
            execute(queue, 1, &list1);
            assert_eq!(
                crate::resource::host_copy(tex)
                    .unwrap()
                    .state
                    .load(Ordering::Relaxed),
                0x40 | 0x80
            );

            let b_to_copy = tex_barrier(
                D3D12_BARRIER_SYNC_ALL_SHADING,
                D3D12_BARRIER_SYNC_COPY,
                D3D12_BARRIER_ACCESS_SHADER_RESOURCE,
                D3D12_BARRIER_ACCESS_COPY_SOURCE,
                D3D12_BARRIER_LAYOUT_SHADER_RESOURCE,
                D3D12_BARRIER_LAYOUT_COPY_SOURCE,
            );
            let g_to_copy = BarrierGroup {
                kind: D3D12_BARRIER_TYPE_TEXTURE.0 as u32,
                count: 1,
                barriers: (&raw const b_to_copy).cast(),
            };
            slot::<BarrierFn>(list2, 0x490)(list2, 1, &g_to_copy);
            record_copy_to_readback(list2);
            assert_eq!(close(list2), S_OK);
            execute(queue, 1, &list2);
            assert!(host.log_messages().is_none());
            assert_eq!(
                crate::resource::host_copy(tex)
                    .unwrap()
                    .state
                    .load(Ordering::Relaxed),
                0x800
            );
            assert!(errors.0.lock().unwrap().is_empty());
            assert_eq!(read_pixel(), [191, 64, 128, 255]);

            let release = |o: This| slot::<extern "system" fn(This) -> u32>(o, 0x10)(o);
            assert_eq!(release(list2), 0);
            assert_eq!(release(allocator2), 0);
            assert_eq!(release(list1), 0);
            assert_eq!(release(allocator1), 0);
            assert_eq!(release(queue), 0);
            assert_eq!(release(tex), 0);
        });
    }

    /// Partial-subresource enhanced barriers (such as targeting mip 1 of a
    /// two-mip texture) must be skipped on the whole-resource tracker so they
    /// never move mip 0 out of `RENDER_TARGET` before a subsequent mip-0 clear.
    #[cfg(windows)]
    #[test]
    fn enhanced_texture_barrier_skips_partial_subresource_ranges() {
        use windows::Win32::Graphics::Direct3D12::*;
        use windows::Win32::Graphics::Dxgi::Common::{
            DXGI_FORMAT_R8G8B8A8_UNORM, DXGI_SAMPLE_DESC,
        };

        let Some(host) = crate::host::host() else {
            return;
        };
        let _ = host.log_messages();
        let errors = D3d12ErrorCapture::default();
        let subscriber = errors.clone();
        tracing::subscriber::with_default(subscriber, || {
            let null = std::ptr::null();
            let resource_iid = guid(
                0x696442be,
                0xa72e,
                0x4059,
                [0xbc, 0x79, 0x5b, 0x5c, 0x98, 0x04, 0x0f, 0xad],
            );
            let queue_iid = guid(
                0x0ec870a6,
                0x5d7e,
                0x4c22,
                [0x8c, 0xfc, 0x5b, 0xaa, 0xe0, 0x76, 0x16, 0xed],
            );

            let heap = D3D12_HEAP_PROPERTIES {
                Type: D3D12_HEAP_TYPE_DEFAULT,
                ..Default::default()
            };
            let tex_desc = D3D12_RESOURCE_DESC {
                Dimension: D3D12_RESOURCE_DIMENSION_TEXTURE2D,
                Alignment: 0,
                Width: 4,
                Height: 4,
                DepthOrArraySize: 1,
                MipLevels: 2,
                Format: DXGI_FORMAT_R8G8B8A8_UNORM,
                SampleDesc: DXGI_SAMPLE_DESC {
                    Count: 1,
                    Quality: 0,
                },
                Layout: D3D12_TEXTURE_LAYOUT_UNKNOWN,
                Flags: D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET,
            };
            let mut tex = null;
            assert_eq!(
                crate::resource::create_committed(
                    null,
                    (&raw const heap).cast(),
                    0,
                    (&raw const tex_desc).cast(),
                    0,
                    null.cast(),
                    &resource_iid,
                    &mut tex,
                ),
                S_OK
            );
            let mut rtv_slot = [0u64; 16];
            let rtv = rtv_slot.as_mut_ptr() as u64;
            crate::descriptor::create_rtv(null, tex, null.cast(), rtv);

            let mut queue = null;
            let queue_desc = [0u32; 4];
            assert_eq!(
                crate::queue::create(null, queue_desc.as_ptr().cast(), &queue_iid, &mut queue),
                S_OK
            );
            let mut allocator = null;
            assert_eq!(
                create_allocator(null, 0, &ALLOCATOR_IIDS[0], &mut allocator),
                S_OK
            );
            let list = new_newer_list(allocator);

            let unorm = |c: [u8; 4]| c.map(|b| f32::from(b) / 255.0);
            clear_rtv(list, rtv, &unorm([0, 255, 0, 255]), 0, null.cast());

            type BarrierFn = extern "system" fn(This, u32, *const BarrierGroup);
            type ExecuteFn = extern "system" fn(This, u32, *const This);
            let call_barrier = slot::<BarrierFn>(list, 0x490);
            let partial_barriers = [
                // Flat subresource index 1 (mip 1)
                TextureBarrier {
                    sync_before: D3D12_BARRIER_SYNC_RENDER_TARGET.0 as u32,
                    sync_after: D3D12_BARRIER_SYNC_COPY.0 as u32,
                    access_before: D3D12_BARRIER_ACCESS_RENDER_TARGET.0 as u32,
                    access_after: D3D12_BARRIER_ACCESS_COPY_SOURCE.0 as u32,
                    layout_before: D3D12_BARRIER_LAYOUT_RENDER_TARGET.0 as u32,
                    layout_after: D3D12_BARRIER_LAYOUT_COPY_SOURCE.0 as u32,
                    resource: tex,
                    subresources: [1, 0, 0, 0, 0, 0],
                    flags: 0,
                    _pad: 0,
                },
                // Explicit range covering only mip 1
                TextureBarrier {
                    sync_before: D3D12_BARRIER_SYNC_RENDER_TARGET.0 as u32,
                    sync_after: D3D12_BARRIER_SYNC_COPY.0 as u32,
                    access_before: D3D12_BARRIER_ACCESS_RENDER_TARGET.0 as u32,
                    access_after: D3D12_BARRIER_ACCESS_COPY_SOURCE.0 as u32,
                    layout_before: D3D12_BARRIER_LAYOUT_RENDER_TARGET.0 as u32,
                    layout_after: D3D12_BARRIER_LAYOUT_COPY_SOURCE.0 as u32,
                    resource: tex,
                    subresources: [1, 1, 0, 1, 0, 1],
                    flags: 0,
                    _pad: 0,
                },
            ];
            let group = BarrierGroup {
                kind: D3D12_BARRIER_TYPE_TEXTURE.0 as u32,
                count: partial_barriers.len() as u32,
                barriers: partial_barriers.as_ptr().cast(),
            };
            call_barrier(list, 1, &group);

            // Must still be in RENDER_TARGET (0x4) so a subsequent mip-0 clear does not fail.
            assert!(
                host_list(list).unwrap().transition(tex, 0x4).is_none(),
                "partial mip-1 barrier clobbered whole-resource RENDER_TARGET state"
            );
            clear_rtv(list, rtv, &unorm([0, 128, 255, 255]), 0, null.cast());
            assert_eq!(close(list), S_OK);
            slot::<ExecuteFn>(queue, 0x50)(queue, 1, &list);

            assert!(host.log_messages().is_none());
            assert!(errors.0.lock().unwrap().is_empty());

            let release = |o: This| slot::<extern "system" fn(This) -> u32>(o, 0x10)(o);
            assert_eq!(release(list), 0);
            assert_eq!(release(allocator), 0);
            assert_eq!(release(queue), 0);
            assert_eq!(release(tex), 0);
        });
    }

    /// Premise check (fails on `main`, passes here): heap-indexed shaders do
    /// not trigger `list::ensure_state` on render-target textures (`title_written`
    /// is false), so reading a cleared RTV through a heap SRV relies on the
    /// enhanced barrier transitioning `RENDER_TARGET -> SHADER_RESOURCE`.
    #[cfg(windows)]
    #[test]
    fn enhanced_texture_barrier_transitions_render_target_for_heap_srv_read() {
        use std::sync::atomic::Ordering;
        use windows::Win32::Graphics::Direct3D::Fxc::D3DCompile;
        use windows::Win32::Graphics::Direct3D::ID3DBlob;
        use windows::Win32::Graphics::Direct3D12::*;
        use windows::Win32::Graphics::Dxgi::Common::{
            DXGI_FORMAT_R8G8B8A8_UNORM, DXGI_SAMPLE_DESC,
        };
        use windows::core::s;

        let Some(host) = crate::host::host() else {
            return;
        };
        let _ = host.log_messages();
        let errors = D3d12ErrorCapture::default();
        let subscriber = errors.clone();
        tracing::subscriber::with_default(subscriber, || {
            let null = std::ptr::null();
            let resource_iid = guid(
                0x696442be,
                0xa72e,
                0x4059,
                [0xbc, 0x79, 0x5b, 0x5c, 0x98, 0x04, 0x0f, 0xad],
            );
            let queue_iid = guid(
                0x0ec870a6,
                0x5d7e,
                0x4c22,
                [0x8c, 0xfc, 0x5b, 0xaa, 0xe0, 0x76, 0x16, 0xed],
            );
            let heap_iid = guid(
                0x8efb471d,
                0x616c,
                0x4f49,
                [0x90, 0xf7, 0x12, 0x7b, 0xb7, 0x63, 0xfa, 0x51],
            );

            let heap_props = D3D12_HEAP_PROPERTIES {
                Type: D3D12_HEAP_TYPE_DEFAULT,
                ..Default::default()
            };
            let tex_desc = D3D12_RESOURCE_DESC {
                Dimension: D3D12_RESOURCE_DIMENSION_TEXTURE2D,
                Alignment: 0,
                Width: 4,
                Height: 4,
                DepthOrArraySize: 1,
                MipLevels: 1,
                Format: DXGI_FORMAT_R8G8B8A8_UNORM,
                SampleDesc: DXGI_SAMPLE_DESC {
                    Count: 1,
                    Quality: 0,
                },
                Layout: D3D12_TEXTURE_LAYOUT_UNKNOWN,
                Flags: D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET,
            };
            let mut tex = null;
            assert_eq!(
                crate::resource::create_committed(
                    null,
                    (&raw const heap_props).cast(),
                    0,
                    (&raw const tex_desc).cast(),
                    0,
                    null.cast(),
                    &resource_iid,
                    &mut tex,
                ),
                S_OK
            );

            let mut rtv_slot = [0u64; 16];
            let rtv = rtv_slot.as_mut_ptr() as u64;
            crate::descriptor::create_rtv(null, tex, null.cast(), rtv);

            let heap_desc: [u32; 8] = [0, 4, 1, 0, 0, 0, 0, 0];
            let mut title_heap_ptr = null;
            assert_eq!(
                crate::descriptor::create_heap(
                    null,
                    heap_desc.as_ptr().cast(),
                    &heap_iid,
                    &mut title_heap_ptr
                ),
                S_OK
            );
            let title_heap = crate::descriptor::heap(title_heap_ptr);
            crate::descriptor::create_srv(null, tex, std::ptr::null(), title_heap.start);
            let heap_host = title_heap.host().expect("host view heap");

            let hlsl = r#"
#define RS "DescriptorTable(SRV(t0)), UAV(u0)"
Texture2D<float4> src : register(t0);
RWByteAddressBuffer dst : register(u0);
[RootSignature(RS)]
[numthreads(1, 1, 1)]
void main() {
    uint4 px = (uint4)round(src.Load(int3(0, 0, 0)) * 255.0);
    dst.Store4(0, px);
}
"#;
            let (mut code, mut compile_errs): (Option<ID3DBlob>, Option<ID3DBlob>) = (None, None);
            // SAFETY: HLSL source and literals outlive the call.
            unsafe {
                D3DCompile(
                    hlsl.as_ptr().cast(),
                    hlsl.len(),
                    s!("heap_srv_barrier"),
                    None,
                    None,
                    s!("main"),
                    s!("cs_5_1"),
                    0,
                    0,
                    &mut code,
                    Some(&mut compile_errs),
                )
                .unwrap();
            }
            let code = code.unwrap();
            // SAFETY: compiled shader blob bytes.
            let bytes = unsafe {
                std::slice::from_raw_parts(code.GetBufferPointer().cast::<u8>(), code.GetBufferSize())
            };
            // SAFETY: root signature and compute PSO compiled above.
            let (root, pso) = unsafe {
                let root: ID3D12RootSignature = host.device.CreateRootSignature(0, bytes).unwrap();
                let desc = D3D12_COMPUTE_PIPELINE_STATE_DESC {
                    pRootSignature: std::mem::transmute_copy(&root),
                    CS: D3D12_SHADER_BYTECODE {
                        pShaderBytecode: bytes.as_ptr().cast(),
                        BytecodeLength: bytes.len(),
                    },
                    ..Default::default()
                };
                let pso: ID3D12PipelineState = host.device.CreateComputePipelineState(&desc).unwrap();
                (root, pso)
            };

            let (out_buf, out_ptr) = host.shared_buffer(256).expect("shared buffer");
            // SAFETY: 256 bytes mapped write-back buffer.
            unsafe { std::ptr::write_bytes(out_ptr, 0, 256) };
            // SAFETY: `out_buf` is a live D3D12 resource.
            let out_gpu = unsafe { out_buf.GetGPUVirtualAddress() };

            let mut queue = null;
            let queue_desc = [0u32; 4];
            assert_eq!(
                crate::queue::create(null, queue_desc.as_ptr().cast(), &queue_iid, &mut queue),
                S_OK
            );
            let mut allocator = null;
            assert_eq!(
                create_allocator(null, 0, &ALLOCATOR_IIDS[0], &mut allocator),
                S_OK
            );
            let list = new_newer_list(allocator);

            let unorm = |c: [u8; 4]| c.map(|b| f32::from(b) / 255.0);
            clear_rtv(list, rtv, &unorm([64, 128, 192, 255]), 0, null.cast());

            type BarrierFn = extern "system" fn(This, u32, *const BarrierGroup);
            type ExecuteFn = extern "system" fn(This, u32, *const This);
            let b = TextureBarrier {
                sync_before: D3D12_BARRIER_SYNC_RENDER_TARGET.0 as u32,
                sync_after: D3D12_BARRIER_SYNC_COMPUTE_SHADING.0 as u32,
                access_before: D3D12_BARRIER_ACCESS_RENDER_TARGET.0 as u32,
                access_after: D3D12_BARRIER_ACCESS_SHADER_RESOURCE.0 as u32,
                layout_before: D3D12_BARRIER_LAYOUT_RENDER_TARGET.0 as u32,
                layout_after: D3D12_BARRIER_LAYOUT_SHADER_RESOURCE.0 as u32,
                resource: tex,
                subresources: [D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES, 0, 0, 0, 0, 0],
                flags: 0,
                _pad: 0,
            };
            let group = BarrierGroup {
                kind: D3D12_BARRIER_TYPE_TEXTURE.0 as u32,
                count: 1,
                barriers: (&raw const b).cast(),
            };
            slot::<BarrierFn>(list, 0x490)(list, 1, &group);

            let hlist = host_list(list).unwrap();
            // Dispatch reading slot 0 from the heap directly (no `ensure_state`).
            // SAFETY: open list with live shader-visible heap, root signature, PSO, and UAV.
            unsafe {
                hlist
                    .list
                    .SetDescriptorHeaps(&[Some(heap_host.heap.clone())]);
                hlist.list.SetComputeRootSignature(&root);
                hlist.list.SetComputeRootDescriptorTable(0, heap_host.gpu(0));
                hlist.list.SetComputeRootUnorderedAccessView(1, out_gpu);
                hlist.list.SetPipelineState(&pso);
                hlist.list.Dispatch(1, 1, 1);
            }
            assert!(
                hlist.transition(tex, 0x40 | 0x80).is_none(),
                "texture stayed in RENDER_TARGET instead of transitioning to SHADER_RESOURCE"
            );

            assert_eq!(close(list), S_OK);
            slot::<ExecuteFn>(queue, 0x50)(queue, 1, &list);

            assert!(host.log_messages().is_none());
            assert!(errors.0.lock().unwrap().is_empty());
            assert_eq!(
                crate::resource::host_copy(tex)
                    .unwrap()
                    .state
                    .load(Ordering::Relaxed),
                0x40 | 0x80
            );
            // SAFETY: GPU execution finished and `out_ptr` is mapped for 256 bytes.
            let got = unsafe { std::slice::from_raw_parts(out_ptr.cast::<u32>(), 4) };
            assert_eq!(got, &[64, 128, 192, 255]);

            let release = |o: This| slot::<extern "system" fn(This) -> u32>(o, 0x10)(o);
            assert_eq!(release(list), 0);
            assert_eq!(release(allocator), 0);
            assert_eq!(release(queue), 0);
            assert_eq!(release(title_heap_ptr), 0);
            assert_eq!(release(tex), 0);
        });
    }
}
