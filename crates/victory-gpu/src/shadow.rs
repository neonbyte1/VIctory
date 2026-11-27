//! Host copies of title buffers (decision 0007, step 3 and "Copies of title
//! memory"). Title buffers are title memory, which the host GPU cannot
//! read, so a buffer a draw uses gets a persistent host buffer of its size
//! on first use (system memory shaders may also write), and the ranges each
//! submission reads (vertex, index and constant data, buffer views) are
//! brought up to date from title memory before it runs: the 64 KiB pages
//! the title wrote since they were last copied ([`crate::watch`]: read-only
//! guards and the runtime's reports), all of it where the memory is not the
//! runtime's. A title GPU address inside the buffer is the same offset
//! in its copy. Ranges shaders write go back to title memory after the
//! submission ([`Shadow::gpu_writes`]).
//!
//! The ranges brought up to date are those draws are known to read (vertex
//! and index buffer views, constant buffers in tables, root descriptors,
//! buffer views in tables) and, for shaders that index the heap
//! themselves, every buffer range the heap's views reach
//! (`crate::bindless`).

use std::collections::BTreeMap;
use std::sync::atomic::{AtomicU32, Ordering as AtomicOrdering};
use std::sync::{Arc, Mutex};

use windows::Win32::Graphics::Direct3D12::ID3D12Resource;

use crate::host::{HostList, host};
use crate::watch::PAGE;

fn merge_writeback(title: &mut [u8], host: &mut [u8], old: &[u8]) -> bool {
    let mut changed = 0;
    for ((title_byte, host_byte), old_byte) in title.iter_mut().zip(host).zip(old) {
        let mask = 0u8.wrapping_sub(u8::from(*host_byte != *old_byte));
        let delta = (*title_byte ^ *host_byte) & mask;
        *title_byte ^= delta;
        *host_byte = *title_byte;
        changed |= delta;
    }
    changed != 0
}

#[test]
fn writeback_merge_matches_bytewise_oracle() {
    for len in [0, 1, 63, 64, 65, 193, 65_536] {
        let old = vec![0x11; len];
        for stride in [1, 5, 67, len + 1] {
            let mut title = old.clone();
            let mut host = old.clone();
            for index in (0..len).step_by(3) {
                title[index] = 0x22;
            }
            for index in (0..len).step_by(stride) {
                host[index] = if index % 2 == 0 { 0x22 } else { 0x33 };
            }
            let expected: Vec<_> = host
                .iter()
                .zip(&old)
                .zip(&title)
                .map(|((&gpu, &previous), &cpu)| if gpu == previous { cpu } else { gpu })
                .collect();
            let changed = expected != title;
            assert_eq!(merge_writeback(&mut title, &mut host, &old), changed);
            assert_eq!(title, expected);
            assert_eq!(host, expected);
        }
    }
}

/// CPU-side backing for VRAM-resident [`Shadow`] copies (`alloc_zeroed` pages
/// demand-committed by the OS, avoiding extra D3D12 `L0` heap resources).
pub(crate) struct CpuMem {
    ptr: *mut u8,
    layout: Option<std::alloc::Layout>,
    _shared: Option<ID3D12Resource>,
}

// SAFETY: page-backed process memory owned via Arc; written/read by Shadow's synchronized paths.
unsafe impl Send for CpuMem {}
// SAFETY: as above.
unsafe impl Sync for CpuMem {}

impl CpuMem {
    fn new(size: u64) -> Option<Arc<Self>> {
        let layout = std::alloc::Layout::from_size_align(size.max(1) as usize, 64).ok()?;
        // SAFETY: non-zero layout; zeroed by allocator.
        let ptr = unsafe { std::alloc::alloc_zeroed(layout) };
        if ptr.is_null() {
            None
        } else {
            Some(Arc::new(Self { ptr, layout: Some(layout), _shared: None }))
        }
    }

    fn wrap_shared(res: ID3D12Resource) -> Arc<Self> {
        Arc::new(Self {
            ptr: std::ptr::null_mut(),
            layout: None,
            _shared: Some(res),
        })
    }

    fn ptr(&self) -> *mut u8 {
        self.ptr
    }
}

impl Drop for CpuMem {
    fn drop(&mut self) {
        if let Some(layout) = self.layout {
            // SAFETY: allocated with `layout` via `alloc_zeroed`.
            unsafe { std::alloc::dealloc(self.ptr, layout) };
        }
    }
}

pub(crate) struct VramCopy {
    pub(crate) dst: ID3D12Resource,
    pub(crate) dst_off: u64,
    pub(crate) src_ptr: usize,
    pub(crate) len: u64,
    pub(crate) _cpu: Arc<CpuMem>,
}

static VRAM_UPLOADS: Mutex<Vec<VramCopy>> = Mutex::new(Vec::new());
static VRAM_BYTES: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

fn try_vram_shadow(size: u64, max_single: u64) -> Option<ID3D12Resource> {
    if size > max_single {
        return None;
    }
    let h = host()?;
    let budget = h.vram_shadow_budget();
    if budget == 0 {
        return None;
    }
    VRAM_BYTES
        .fetch_update(
            std::sync::atomic::Ordering::Relaxed,
            std::sync::atomic::Ordering::Relaxed,
            |cur| cur.checked_add(size).filter(|&next| next <= budget),
        )
        .ok()?;
    let Some(buf) = h.default_uav_buffer(size) else {
        VRAM_BYTES.fetch_sub(size, std::sync::atomic::Ordering::Relaxed);
        return None;
    };
    Some(buf)
}

struct Backing {
    buffer: ID3D12Resource,
    cpu: Arc<CpuMem>,
    vram: bool,
    alloc_size: u64,
    data: *mut u8,
    gpu: u64,
}

fn alloc_backing(size: u64, max_vram_single: u64, arena_owned: bool, label: impl FnOnce() -> String) -> Option<Backing> {
    let h = host()?;
    let (buffer, cpu, data, vram) = if let Some(v) = try_vram_shadow(size, max_vram_single) {
        let Some(cpu) = CpuMem::new(size) else {
            VRAM_BYTES.fetch_sub(size, std::sync::atomic::Ordering::Relaxed);
            return None;
        };
        let data = cpu.ptr();
        (v, cpu, data, true)
    } else {
        let (ram, data) = h.shared_buffer(size)?;
        (ram.clone(), CpuMem::wrap_shared(ram), data, false)
    };
    crate::host::name(&buffer, label);
    // SAFETY: a live buffer.
    let gpu = unsafe { buffer.GetGPUVirtualAddress() };
    Some(Backing {
        buffer,
        cpu,
        vram,
        alloc_size: if arena_owned { 0 } else { size },
        data,
        gpu,
    })
}

pub(crate) fn take_vram_uploads() -> Vec<VramCopy> {
    std::mem::take(&mut *VRAM_UPLOADS.lock().unwrap_or_else(|e| e.into_inner()))
}

/// A title buffer's host copy.
pub(crate) struct Shadow {
    pub(crate) buffer: ID3D12Resource,
    cpu: Arc<CpuMem>,
    vram: bool,
    alloc_size: u64,
    data: *mut u8,
    gpu: u64,
    /// Where the title buffer starts and how long it is.
    start: u64,
    len: u64,
    /// The pages changed since the copy took them (none: the memory is not
    /// the runtime's, copied whole each time).
    tracked: Option<crate::watch::Tracked>,
    /// What each submission not yet completed does with the copy, by
    /// submission (`crate::host`'s owner numbers, in order): the ranges its
    /// shaders may write (title memory there is older than the copy) and
    /// its snapshot of them ([`Sub`]). Submissions in flight together write
    /// disjoint ranges of a copy, or both by marks (`crate::host::Host::submit`).
    subs: Mutex<Vec<Sub>>,
    /// Its first bit in the buffer views' written-pages bitmap, one per
    /// 64 KiB title page it touches (tracked memory only): writes through
    /// views of it are marked there ([`Shadow::mark_entry`]).
    marks: Option<u32>,
    /// Where it starts in `buffer` (0 but for copies sharing a buffer), and
    /// the run of the shared buffer it holds ([`Arena`]), given back when it
    /// goes.
    base: u64,
    slice: Option<Slice>,
}

/// One submission's use of a copy's ranges its shaders write.
#[derive(Default)]
struct Sub {
    owner: u64,
    /// Declared ranges `(start, end, marked)` (title addresses) not copied
    /// back yet.
    pending: std::collections::HashSet<(usize, usize, bool)>,
    /// The copy's bytes before the submission ran where its shaders write
    /// unmarked: what the GPU wrote differs. Disjoint segments (offset,
    /// bytes) in order.
    segments: Vec<(usize, Vec<u8>)>,
    /// The 64 KiB pieces (by title address) already compared.
    compared: std::collections::HashSet<usize>,
}

impl Sub {
    /// Whether its shaders may write page `page` (a title address on a
    /// page boundary) through marked ranges.
    fn marks_page(&self, page: usize) -> bool {
        self.pending.iter().any(|&(a, e, marked)| marked && a < page + PAGE && page < e)
    }
}

/// `subs`' entry for `owner`, made if it has none.
fn sub_of(subs: &mut Vec<Sub>, owner: u64) -> &mut Sub {
    match subs.iter().position(|s| s.owner == owner) {
        Some(i) => &mut subs[i],
        None => {
            subs.push(Sub { owner, ..Sub::default() });
            subs.last_mut().expect("just pushed")
        }
    }
}

/// Snapshot memory kept for reuse (already touched, so no page faults).
static POOL: Mutex<Vec<Vec<u8>>> = Mutex::new(Vec::new());

/// An empty buffer with room for `n` bytes, from the pool if it has one.
fn pooled(n: usize) -> Vec<u8> {
    let mut pool = POOL.lock().unwrap_or_else(|e| e.into_inner());
    let fit = pool
        .iter()
        .enumerate()
        .filter(|(_, v)| v.capacity() >= n)
        .min_by_key(|(_, v)| v.capacity())
        .map(|(i, _)| i);
    let mut v = match fit {
        Some(i) => pool.swap_remove(i),
        None => Vec::with_capacity(n),
    };
    v.clear();
    v
}

/// Returns a snapshot buffer to the pool, which keeps the largest up to
/// [`POOL_BYTES`] in all: a world submission snapshots several copies, and
/// fresh buffers cost a page fault per 4 KiB touched (millions per frame).
fn give_back(v: Vec<u8>) {
    let mut pool = POOL.lock().unwrap_or_else(|e| e.into_inner());
    pool.push(v);
    pool.sort_by_key(|v| std::cmp::Reverse(v.capacity()));
    let mut total = 0;
    pool.retain(|v| {
        total += v.capacity();
        total <= POOL_BYTES
    });
}

/// Snapshot memory kept for reuse.
const POOL_BYTES: usize = 512 << 20;

// SAFETY: D3D12 resources are free-threaded; `data` is a persistent mapping
// written only on the submitting thread.
unsafe impl Send for Shadow {}
// SAFETY: as above.
unsafe impl Sync for Shadow {}

impl Shadow {
    /// The host GPU address of a title address inside the buffer.
    pub(crate) fn gpu(&self, address: u64) -> u64 {
        self.gpu + (address - self.start)
    }

    /// Bytes from a title address to the buffer's end.
    pub(crate) fn left(&self, address: u64) -> u64 {
        (self.start + self.len).saturating_sub(address)
    }

    /// `n` dwords of the copy at title address `address` (tests).
    #[cfg(test)]
    pub(crate) fn read_host(&self, address: u64, n: usize) -> Vec<u32> {
        let at = (address - self.start) as usize;
        assert!(at + 4 * n <= self.len as usize);
        if self.vram
            && let Some(h) = host()
        {
            h.flush_test_vram_uploads();
            // SAFETY: `self.data.add(at)` is valid for `4 * n` bytes.
            h.test_readback_buffer(&self.buffer, self.base + at as u64, unsafe { self.data.add(at) }, (4 * n) as u64);
        }
        // SAFETY: inside the mapping (checked); the GPU is done with it.
        unsafe { std::slice::from_raw_parts(self.data.add(at).cast::<u32>(), n).to_vec() }
    }

    /// Writes `bytes` into the copy at title address `address`, as shaders
    /// do (tests).
    #[cfg(test)]
    pub(crate) fn write_host(&self, address: u64, bytes: &[u8]) {
        let at = (address - self.start) as usize;
        assert!(at + bytes.len() <= self.len as usize);
        // SAFETY: inside the mapping (checked).
        unsafe { std::ptr::copy_nonoverlapping(bytes.as_ptr(), self.data.add(at), bytes.len()) };
        if self.vram
            && let Some(h) = host()
        {
            h.flush_test_vram_uploads();
            h.test_upload_buffer(&self.buffer, self.base + at as u64, bytes.as_ptr(), bytes.len() as u64);
        }
    }

    /// Submission `owner`'s snapshot segments, as (offset, bytes) (tests).
    #[cfg(test)]
    pub(crate) fn segments(&self, owner: u64) -> Vec<(usize, usize)> {
        self.subs
            .lock()
            .unwrap()
            .iter()
            .filter(|s| s.owner == owner)
            .flat_map(|s| s.segments.iter().map(|(at, bytes)| (*at, bytes.len())))
            .collect()
    }

    /// Up to 64 bytes of the copy at title address `address`, as floats, and
    /// which copy it is (diagnostics: what shaders see there).
    #[cfg(feature = "diagnostics")]
    pub(crate) fn peek(&self, address: u64, len: u64) -> String {
        let at = address.saturating_sub(self.start) as usize;
        let n = (len.min(64) as usize).min((self.len as usize).saturating_sub(at)) / 4;
        // SAFETY: inside the mapping (checked); a racing GPU write only makes
        // the logged value stale.
        let v = unsafe { std::slice::from_raw_parts(self.data.add(at).cast::<f32>(), n) };
        format!("copy {:#x}+{:#x} {v:?}", self.start, self.len)
    }

    /// Offset of a title address from the buffer's start.
    pub(crate) fn offset(&self, address: u64) -> u64 {
        address - self.start + self.base
    }

    /// Where the title memory it copies starts, and its length.
    pub(crate) fn range(&self) -> (u64, u64) {
        (self.start, self.len)
    }

    /// After a submission whose shaders wrote title memory at addresses they
    /// computed (`crate::memory`), somewhere in the 64 KiB page at `page`:
    /// the bytes of the page (cut to the copy) that differ from title memory
    /// go back into it, when the title has not written the page since the
    /// copy took it (then title memory is the copy as it was before the
    /// submission, and what differs is what the GPU wrote). Returns the
    /// bytes written; a page the title wrote meanwhile keeps the title's
    /// bytes (logged once).
    // shortcut: a page the copy holds only in part (at its ends) has no
    // "copied" mark and is compared without the title-write check.
    pub(crate) fn copy_back_page(&self, page: usize) -> usize {
        let (start, end) = (self.start as usize, (self.start + self.len) as usize);
        let (from, to) = (page.max(start), (page + PAGE).min(end));
        if from >= to {
            return 0;
        }
        if let Some(t) = &self.tracked
            && t.changed(page / PAGE)
        {
            static ONCE: std::sync::Once = std::sync::Once::new();
            ONCE.call_once(|| {
                tracing::warn!(
                    page = format_args!("{page:#x}"),
                    "the title and translated shaders both wrote a page of title memory in one submission: the title's bytes kept (logged once)"
                )
            });
            return 0;
        }
        let n = to - from;
        // SAFETY: inside the mapping; only the submitting thread writes it.
        let host = unsafe { std::slice::from_raw_parts(self.data.add(from - start), n) };
        let mut title = vec![0u8; n];
        if !crate::host::read_into(from as *const u8, &mut title) || title == host {
            return 0;
        }
        {
            let _writing = crate::watch::writing(from, n);
            crate::host::write_mapped(from as *mut u8, host.as_ptr(), n);
        }
        // Our write is not the title's: the copy holds the page as it is.
        if let Some(t) = &self.tracked {
            t.taken(&[page / PAGE]);
        }
        n
    }

    /// Brings `len` bytes at `address` up to date when `list` is submitted,
    /// and keeps the copy alive until the list is reset.
    pub(crate) fn refresh(self: &Arc<Self>, list: &HostList, address: u64, len: u64) {
        let len = len.min(self.start + self.len - address);
        list.defer_refresh(self, address as usize, len as usize);
    }

    fn new(b: Backing, start: u64, len: u64) -> Arc<Shadow> {
        Shadow::new_at(b, start, len, None)
    }

    /// [`Shadow::new`] of a copy in a run of a shared buffer (`b.data` and
    /// `b.gpu` are the run's).
    fn new_at(b: Backing, start: u64, len: u64, slice: Option<Slice>) -> Arc<Shadow> {
        let tracked = crate::watch::track_copy(start as usize, len as usize, true);
        let marks = tracked.as_ref().and_then(|_| marks().take(pages(start, len)));
        let shadow = Arc::new(Shadow {
            buffer: b.buffer,
            cpu: b.cpu,
            vram: b.vram,
            alloc_size: b.alloc_size,
            data: b.data,
            gpu: b.gpu,
            start,
            len,
            tracked,
            subs: Mutex::default(),
            marks,
            base: slice.map_or(0, |s| s.offset),
            slice,
        });
        if let Some(base) = marks {
            self::marks().owners.insert(base, Arc::downgrade(&shadow));
        }
        shadow
    }

    /// The view-mark entry (`victory_shader::emit`, "Write marks") of a
    /// view of this copy whose first byte is at title address `start`: the
    /// bit of that byte's page and its offset in the page. None where
    /// writes through it are not marked (memory not tracked, or no bits
    /// left): such views declare what they write ([`Shadow::gpu_writes`]).
    pub(crate) fn mark_entry(&self, start: u64) -> Option<(u32, u32)> {
        let base = self.marks?;
        let page = (start >> 16).checked_sub(self.start >> 16)?;
        Some((base + u32::try_from(page).ok()?, (start & 0xFFFF) as u32))
    }

    /// Whether writes through views of this copy are marked.
    pub(crate) fn marked(&self) -> bool {
        self.marks.is_some()
    }

    /// `[address, address + len)` cut to the buffer, as offsets in the copy.
    fn inside(&self, address: usize, len: usize) -> (usize, usize) {
        let (start, end) = (self.start as usize, (self.start + self.len) as usize);
        let from = address.clamp(start, end);
        (from - start, (address + len).clamp(from, end) - start)
    }

    /// Shaders of `list` may write `[address, address + len)` of the copy
    /// (unordered access): after the list's submission has run, the range
    /// is copied back into title memory, where the title reads it; until
    /// then no refresh copies title memory over it. Call after the
    /// [`Shadow::refresh`] that gives the shaders the title's data.
    ///
    /// shortcut: what shaders wrote is found by keeping the range's bytes
    /// before the submission (a copy per declared range and submission);
    /// a GPU-side dirty mask would save that if big ranges make it slow.
    ///
    /// `marked`: the shaders mark the pages they write through this copy's
    /// views (`victory_shader::emit`, "Write marks"): no snapshot then, only
    /// the pages they marked are compared ([`Shadow::copy_back_marked`]).
    pub(crate) fn gpu_writes(self: &Arc<Self>, list: &HostList, address: u64, len: u64, marked: bool) {
        let len = len.min(self.left(address));
        list.defer_gpu_write(self, address as usize, len as usize, marked && self.marked());
    }

    /// At submission `owner`'s preparing, after the range's refresh: a
    /// submitted list's shaders write this range. Its bytes are kept to tell
    /// later what the GPU wrote: per copy and submission, segments covering
    /// the ranges the submission declares (the world declares overlapping
    /// views of its 174 MB heap dozens of times; one segment over all of
    /// them copied most of the heap), in reused memory (fresh snapshots cost
    /// ~2.3 million page faults per world frame). A submission in flight
    /// meanwhile writes no unmarked range overlapping these
    /// (`crate::host::Host::submit`), so the copy holds what this one's
    /// shaders will find there.
    ///
    /// With `marked`, nothing is kept: the pages are armed instead, so that
    /// a title write there before the copy-back keeps the page's bytes from
    /// before it ([`crate::watch::Tracked::arm`]).
    pub(crate) fn writing(&self, owner: u64, address: usize, len: usize, marked: bool) {
        let marked = marked && self.tracked.is_some();
        let mut subs = self.subs.lock().unwrap_or_else(|e| e.into_inner());
        let sub = sub_of(&mut subs, owner);
        sub.pending.insert((address, address + len, marked));
        if marked && let Some(t) = &self.tracked {
            t.arm(address, address + len);
            return;
        }
        let (from, to) = self.inside(address, len);
        if from >= to {
            return;
        }
        // SAFETY: inside the mapping.
        let host = |a: usize, b: usize| unsafe { std::slice::from_raw_parts(self.data.add(a), b - a) };
        let segments = &mut sub.segments;
        let mut i = segments.partition_point(|(at, bytes)| at + bytes.len() <= from);
        let mut cur = from;
        #[cfg(feature = "diagnostics")]
        let started = std::time::Instant::now();
        #[cfg(feature = "diagnostics")]
        let mut added = 0usize;
        while cur < to {
            let next_start = segments.get(i).map_or(to, |(at, _)| (*at).min(to));
            if cur < next_start {
                let mut gap = pooled(next_start - cur);
                gap.extend_from_slice(host(cur, next_start));
                #[cfg(feature = "diagnostics")]
                {
                    added += next_start - cur;
                }
                segments.insert(i, (cur, gap));
                i += 1;
                cur = next_start;
            } else {
                let seg_end = segments[i].0 + segments[i].1.len();
                cur = cur.max(seg_end);
                i += 1;
            }
        }
        #[cfg(feature = "diagnostics")]
        if added > 0 {
            crate::copystat::add(crate::copystat::C::SnapshotUs, started.elapsed().as_micros() as u64);
            crate::copystat::add(crate::copystat::C::SnapshotBytes, added as u64);
            crate::copystat::snapshot(self.start, self.len, added as u64);
        }
    }

    /// `[address, address + len)` expanded outward to 64 KiB page
    /// boundaries and cut to the copy (`[start, start + len)`): whole pages
    /// so [`crate::watch::Tracked::take`] guards them for
    /// [`Shadow::copy_back_marked`].
    pub(crate) fn page_range(&self, address: u64, len: u64) -> (u64, u64) {
        let end = self.start.saturating_add(self.len);
        let from = address.clamp(self.start, end);
        let to = address.saturating_add(len).clamp(from, end);
        if from >= to {
            return (from, 0);
        }
        let page = PAGE as u64;
        let from = (from & !(page - 1)).max(self.start);
        let to = to.checked_next_multiple_of(page).unwrap_or(end).min(end);
        (from, to - from)
    }

    /// After submission `owner` ran: copies the bytes its shaders wrote back
    /// into title memory (unmapped pages left alone), and only those: the
    /// title's threads write the same buffers while the GPU runs (its
    /// command memory shares the big GPU heap with what shaders write), and
    /// putting back the copy's older bytes there broke the title's own
    /// records (a fence list read as -1, title+0x4e9b54a). Returns the bytes
    /// written.
    pub(crate) fn copy_back(&self, owner: u64, address: usize, len: usize) -> usize {
        self.copy_back_or_not(owner, address, len, true)
    }

    /// After submission `owner` ran: what its shaders wrote in the range
    /// stays in the copy and does not go into title memory (they computed
    /// it from memory with no copy, `crate::memory`).
    pub(crate) fn keep_writes(&self, owner: u64, address: usize, len: usize) {
        self.copy_back_or_not(owner, address, len, false);
    }

    fn copy_back_or_not(&self, owner: u64, address: usize, len: usize, back: bool) -> usize {
        #[cfg(feature = "diagnostics")]
        let _site = crate::watch::at_site(crate::watch::Site::Declared);
        let mut subs = self.subs.lock().unwrap_or_else(|e| e.into_inner());
        let Some(i) = subs.iter().position(|s| s.owner == owner) else {
            return 0;
        };
        let Sub {
            pending,
            segments,
            compared,
            ..
        } = &mut subs[i];
        let mut written = 0;
        let (from, to) = self.inside(address, len);
        let start = self.start as usize;
        if back && from < to {
            #[cfg(feature = "diagnostics")]
            let _timed = crate::watch::Timed(std::time::Instant::now(), crate::copystat::C::CompareUs);
            // By 64 KiB piece: the copy compared with its bytes before the
            // submission, title memory read only where the GPU wrote.
            const PIECE: usize = 64 << 10;
            let mut title = pooled(PIECE);
            title.resize(PIECE, 0);
            let mut synced = Vec::new();
            let mut seg_idx = segments.partition_point(|(at, bytes)| at + bytes.len() <= from);
            let mut at = from;
            while at < to && seg_idx < segments.len() {
                let piece_len = (PIECE - (start + at) % PIECE).min(to - at);
                let piece_end = at + piece_len;
                // A whole piece is compared once per submission, whichever
                // declared ranges overlap it: then what the GPU wrote there
                // went back.
                if piece_len == PIECE && !compared.insert(start + at) {
                    at = piece_end;
                    continue;
                }
                while at < piece_end && seg_idx < segments.len() {
                    let (taken, before) = &segments[seg_idx];
                    if *taken >= piece_end {
                        break;
                    }
                    if *taken > at {
                        at = *taken;
                    }
                    let seg_end = taken + before.len();
                    if at >= seg_end {
                        seg_idx += 1;
                        continue;
                    }
                    let n = (piece_end - at).min(seg_end - at);
                    // SAFETY: inside the mapping; only the submitting thread
                    // writes it.
                    let host = unsafe { std::slice::from_raw_parts_mut(self.data.add(at), n) };
                    let old = &before[at - taken..at - taken + n];
                    let title = &mut title[..n];
                    #[cfg(feature = "diagnostics")]
                    crate::copystat::add(crate::copystat::C::CompareBytes, n as u64);
                    if host != old && crate::host::read_into((start + at) as *const u8, title) {
                        // Title memory with the GPU's bytes put in; the copy
                        // takes the title's bytes in turn.
                        if merge_writeback(title, host, old) {
                            let _writing = crate::watch::writing(start + at, n);
                            crate::host::write_mapped((start + at) as *mut u8, title.as_ptr(), n);
                            written += n;
                        }
                        synced.push((start + at) / PAGE);
                    }
                    at += n;
                }
                at = piece_end;
            }
            give_back(title);
            // Pages made equal to title memory are current in the copy, so
            // our own write is not taken for the title's; others keep what
            // they had (a title write there shows at the next refresh).
            let (first, last) = ((start + from).div_ceil(PAGE), (start + to) / PAGE);
            if let Some(t) = &self.tracked {
                let whole: Vec<usize> = synced.into_iter().filter(|p| (first..last).contains(p)).collect();
                t.taken(&whole);
            }
        }
        // Every step of the submission that declared the range is done with
        // it, and its snapshot with its last range; the pages are disarmed
        // once no submission's shaders write the copy.
        pending.retain(|&(a, e, _)| (a, e) != (address, address + len));
        if pending.is_empty() {
            for (_, bytes) in segments.drain(..) {
                give_back(bytes);
            }
            subs.swap_remove(i);
            if subs.is_empty()
                && let Some(t) = &self.tracked
            {
                t.disarm();
            }
        }
        written
    }

    /// After a submission whose shaders marked the 64 KiB title page at
    /// `page` as written through views of this copy: the bytes they changed
    /// go into title memory, and the copy takes the title's others. What
    /// they changed is what differs from the copy's bytes before the
    /// submission: title memory where the title has not written the page
    /// since the copy took it, else what the page held when the title first
    /// wrote it ([`crate::watch::Tracked::saved`], if the page was armed).
    /// Returns the bytes written. A page in `unchanged` (by its address in
    /// the copy: a copy in video memory read back as the copy held it,
    /// `crate::host`) was not changed by shaders: nothing goes back, and
    /// title memory is not read. `owner`: the submission completing; a later
    /// one in flight whose shaders may mark the page too keeps it armed, so
    /// that a title write from now on keeps the bytes this one left.
    // shortcut: a page the title wrote that was not armed (a write through
    // the heap's views no list declared) keeps the title's bytes, losing
    // the shaders' there (counted).
    pub(crate) fn copy_back_marked(&self, owner: u64, page: usize, unchanged: &std::collections::HashSet<usize>) -> usize {
        let Some(t) = &self.tracked else {
            return 0;
        };
        let written = self.copy_back_marked_page(t, page, unchanged);
        self.rearm_for_later(t, owner, &[page]);
        written
    }

    /// After submission `owner`'s copy-back of marked `pages`: those a
    /// later submission in flight may mark too are armed again, so that a
    /// title write from now on keeps the bytes this one left.
    fn rearm_for_later(&self, t: &crate::watch::Tracked, owner: u64, pages: &[usize]) {
        let subs = self.subs.lock().unwrap_or_else(|e| e.into_inner());
        for &page in pages {
            if subs.iter().any(|s| s.owner != owner && s.marks_page(page)) {
                t.rearm(page / PAGE);
            }
        }
    }

    fn copy_back_marked_page(
        &self,
        t: &crate::watch::Tracked,
        page: usize,
        unchanged: &std::collections::HashSet<usize>,
    ) -> usize {
        let (start, end) = (self.start as usize, (self.start + self.len) as usize);
        let (from, to) = (page.max(start), (page + PAGE).min(end));
        if from >= to {
            return 0;
        }
        // Our own write below must not save the page for this copy again.
        t.unarm(page / PAGE);
        let saved = t.saved(page / PAGE);
        if saved.is_none() && t.changed(page / PAGE) {
            #[cfg(feature = "diagnostics")]
            crate::copystat::add(crate::copystat::C::MarkedKept, 1);
            return 0;
        }
        let n = to - from;
        // SAFETY: inside the mapping; only the submitting thread writes it.
        let host = unsafe { std::slice::from_raw_parts_mut(self.data.add(from - start), n) };
        // Shaders changed nothing: as below with no difference found.
        let gpu_unchanged = unchanged.contains(&(host.as_ptr() as usize));
        let mut title = TITLE.with(|t| std::mem::take(&mut *t.borrow_mut()));
        title.resize(n, 0);
        let give_back = |title: Vec<u8>| TITLE.with(|t| *t.borrow_mut() = title);
        if !gpu_unchanged && !crate::host::read_into(from as *const u8, &mut title) {
            give_back(title);
            return 0;
        }
        let written = match &saved {
            _ if gpu_unchanged => None,
            // The title's bytes are the copy's from before: what differs is
            // the shaders'.
            None => (*host != title[..]).then_some(host.as_ptr()),
            Some(before) => {
                let before = &before[from - page..to - page];
                merge_writeback(&mut title, host, before).then_some(title.as_ptr())
            }
        };
        if let Some(bytes) = written {
            let _writing = crate::watch::writing(from, n);
            crate::host::write_mapped(from as *mut u8, bytes, n);
        }
        give_back(title);
        // Equal to title memory now: taken, unless it holds the page in part.
        if from == page && to == page + PAGE {
            t.taken(&[page / PAGE]);
        }
        written.map_or(0, |_| n)
    }

    /// Whether the copy tracks its memory and holds the title page at
    /// `page` whole: such pages go back in runs ([`copy_back_marks`]).
    fn holds_whole(&self, page: usize) -> bool {
        self.tracked.is_some() && page >= self.start as usize && page + PAGE <= (self.start + self.len) as usize
    }

    /// [`Shadow::copy_back_marked`] of `pages`, consecutive 64 KiB title
    /// pages the copy holds whole (ascending), in one lease of the guards
    /// ([`copy_back_run`]), for submission `owner` (re-arming as there).
    pub(crate) fn copy_back_marked_run(
        &self,
        owner: u64,
        pages: &[usize],
        unchanged: &std::collections::HashSet<usize>,
    ) -> RunBack {
        let Some(t) = &self.tracked else {
            return RunBack::default();
        };
        debug_assert!(pages.iter().all(|&p| self.holds_whole(p)));
        // SAFETY: whole pages inside the mapping; only the submitting
        // thread writes it, and not meanwhile.
        let host = unsafe { std::slice::from_raw_parts(self.data.add(pages[0] - self.start as usize), pages.len() * PAGE) };
        let back = copy_back_run(t, pages, host, unchanged);
        self.rearm_for_later(t, owner, pages);
        back
    }

    /// Whether the title wrote none of the copy's memory since the copy took
    /// it (tracked memory only): [`Shadow::update`] would copy nothing.
    pub(crate) fn clean(&self) -> bool {
        self.tracked.as_ref().is_some_and(crate::watch::Tracked::clean)
    }

    /// Its tracker's bit: `crate::watch::clean_bit` of it is
    /// [`Shadow::clean`] (none for memory that is not tracked: never clean).
    pub(crate) fn dirty_bit(&self) -> Option<u32> {
        self.tracked.as_ref().and_then(crate::watch::Tracked::bit)
    }

    /// At submission `owner`'s preparing: copies what changed of `[address,
    /// address + len)` since it was last copied (all of it if the memory is
    /// not write-watched). Returns the bytes copied, or `Err(o)`: submission
    /// `o`, still in flight, declared that its shaders write where this
    /// would copy title memory over the copy (pages the title changed under
    /// its ranges, or memory copied whole there). What the GPU writes there
    /// reaches title memory only when `o` completes, so `o` completes first
    /// and this is asked again: nothing was copied.
    pub(crate) fn update(&self, owner: u64, address: usize, len: usize) -> Result<usize, u64> {
        if self.clean() {
            return Ok(0);
        }
        let (start, end) = (self.start as usize, (self.start + self.len) as usize);
        let (from, to) = (address.max(start), (address + len).min(end));
        if from >= to {
            return Ok(0);
        }
        // This submission's own ranges, and those of others in flight.
        let (mut busy, mut others) = (Vec::new(), Vec::new());
        for s in self.subs.lock().unwrap_or_else(|e| e.into_inner()).iter() {
            for &(a, e, _) in s.pending.iter().filter(|&&(a, e, _)| a < to && e > from) {
                if s.owner == owner {
                    busy.push((a, e));
                } else {
                    others.push((a, e, s.owner));
                }
            }
        }
        if self.tracked.is_some() {
            return self.update_tracked(owner, from, to, !busy.is_empty(), &others);
        }
        if let Some(o) = others.iter().map(|r| r.2).max() {
            return Err(o);
        }
        if !busy.is_empty() {
            // Around ranges shaders are writing: the gaps, each as below
            // (in the world most submissions have such ranges; copying the
            // gaps whole cost ~1 s per frame). Under them nothing: what the
            // title wrote there is not known.
            busy.sort_unstable();
            let (mut at, mut bytes) = (from, 0);
            for (a, e) in busy {
                let (a, e) = (a.max(from), e.min(to));
                if a > at {
                    bytes += self.update_untracked(at, a);
                }
                at = at.max(e);
            }
            if at < to {
                bytes += self.update_untracked(at, to);
            }
            return Ok(bytes);
        }
        Ok(self.update_untracked(from, to))
    }

    /// [`Shadow::update`] of `[from, to)` (inside the buffer) where the
    /// memory is tracked: the pages the title wrote since they were taken,
    /// in one [`crate::watch::Tracked::take`], so that a page wholly in the
    /// range is guarded even where shaders write part of it (split at those
    /// ranges it stayed changed, and [`Shadow::copy_back_marked`] dropped
    /// what they wrote).
    ///
    /// `busy`: a list of this submission (`owner`) submitted earlier
    /// declared that shaders write in the range. None of its shaders has run
    /// yet: the pages the title wrote are copied as anywhere else. Skipped,
    /// the title's constants there stayed stale for the whole submission (a
    /// dispatch read its sizes as zeros and looped until the device was
    /// removed). The same bytes go into the snapshot `writing` took, so
    /// copy-back neither takes them for the GPU's nor puts them back over a
    /// later title write.
    ///
    /// `others`: ranges `(start, end, owner)` submissions in flight declared
    /// here. A changed page under one, or memory copied whole there, is not
    /// taken: `Err` with the latest such owner (decided under the tracker's
    /// lock, so a page the title writes meanwhile is either seen here or
    /// left for a later take).
    fn update_tracked(
        &self,
        owner: u64,
        from: usize,
        to: usize,
        busy: bool,
        others: &[(usize, usize, u64)],
    ) -> Result<usize, u64> {
        let Some(t) = &self.tracked else {
            return Ok(0);
        };
        let untracked = t.untracked(from, to);
        if let Some(o) = others
            .iter()
            .filter(|&&(a, e, _)| untracked.iter().any(|&(x, y)| a < y && x < e))
            .map(|r| r.2)
            .max()
        {
            return Err(o);
        }
        let forbidden: Vec<(usize, usize, u64)> = others.iter().map(|&(a, e, o)| (a / PAGE, e.div_ceil(PAGE), o)).collect();
        #[cfg(feature = "diagnostics")]
        let started = std::time::Instant::now();
        let (runs, unmapped) = t.take_split_unless(from, to, &forbidden)?;
        #[cfg(feature = "diagnostics")]
        crate::copystat::add(crate::copystat::C::WatchUs, started.elapsed().as_micros() as u64);
        let start = self.start as usize;
        let mut subs = busy.then(|| self.subs.lock().unwrap_or_else(|e| e.into_inner()));
        let mut sync_before = |a: usize, b: usize| {
            let Some(sub) = subs.as_mut().and_then(|s| s.iter_mut().find(|s| s.owner == owner)) else {
                return;
            };
            let (a, b) = (a - start, b - start);
            for (at, bytes) in sub.segments.iter_mut() {
                let (s, e) = ((*at).max(a), (*at + bytes.len()).min(b));
                if s < e {
                    // SAFETY: inside the mapping.
                    let host = unsafe { std::slice::from_raw_parts(self.data.add(s), e - s) };
                    bytes[s - *at..e - *at].copy_from_slice(host);
                }
            }
        };
        let mut copied = 0;
        for (a, b) in runs {
            copied += self.copy(a, b);
            sync_before(a, b);
        }
        for (a, b) in unmapped {
            // SAFETY: inside the mapping (`take_split` cuts to the buffer).
            unsafe { self.data.add(a - start).write_bytes(0, b - a) };
            self.queue_vram_upload((a - start) as u64, (b - a) as u64);
            sync_before(a, b);
            #[cfg(feature = "diagnostics")]
            crate::copystat::add(crate::copystat::C::Unmapped, (b - a) as u64);
        }
        // Parts on memory that is not the runtime's: whole, as below.
        for (a, b) in untracked {
            #[cfg(feature = "diagnostics")]
            let started = std::time::Instant::now();
            copied += self.copy(a, b);
            sync_before(a, b);
            #[cfg(feature = "diagnostics")]
            {
                use crate::copystat::{C, add};
                add(C::WholeBytes, (b - a) as u64);
                add(C::WholeUs, started.elapsed().as_micros() as u64);
            }
        }
        #[cfg(feature = "diagnostics")]
        {
            use crate::copystat::{C, add};
            add(C::Watches, 1);
            add(C::Checked, (to - from) as u64);
            add(C::Unchanged, (to - from).saturating_sub(copied) as u64);
        }
        Ok(copied)
    }

    /// [`Shadow::update`] of untracked `[from, to)` (inside the buffer).
    fn update_untracked(&self, from: usize, to: usize) -> usize {
        #[cfg(feature = "diagnostics")]
        let started = std::time::Instant::now();
        self.copy(from, to);
        #[cfg(feature = "diagnostics")]
        {
            use crate::copystat::{C, add};
            add(C::WholeBytes, (to - from) as u64);
            add(C::WholeUs, started.elapsed().as_micros() as u64);
        }
        to - from
    }

    pub(crate) fn vram_readback(&self) -> Option<(usize, u64, u64, u64)> {
        self.vram.then_some((self.data as usize, self.base, self.start, self.len))
    }

    fn queue_vram_upload(&self, off: u64, len: u64) {
        if self.vram && len > 0 {
            VRAM_UPLOADS
                .lock()
                .unwrap_or_else(|e| e.into_inner())
                .push(VramCopy {
                    dst: self.buffer.clone(),
                    dst_off: self.base + off,
                    // SAFETY: `off` is inside `[0, self.len)` of `self.data`.
                    src_ptr: unsafe { self.data.add(off as usize) } as usize,
                    len,
                    _cpu: self.cpu.clone(),
                });
        }
    }

    /// Copies title memory `[from, to)`, cut to the buffer, into the copy.
    fn copy(&self, from: usize, to: usize) -> usize {
        let (start, end) = (self.start as usize, (self.start + self.len) as usize);
        let (from, to) = (from.max(start), to.min(end));
        // SAFETY: inside the mapping (`from` and `to` are within the buffer).
        let at = unsafe { self.data.add(from - start) };
        crate::host::copy_mapped(at, from as *const u8, to - from, false);
        self.queue_vram_upload((from - start) as u64, (to - from) as u64);
        if let Some(t) = &self.tracked {
            // SAFETY: `to - from` bytes inside the mapping, just copied.
            t.refreshed(from, unsafe { std::slice::from_raw_parts(at, to - from) });
        }
        to - from
    }
}

impl Drop for Shadow {
    fn drop(&mut self) {
        if let Some(slice) = self.slice {
            arena().give_back(slice);
        } else if self.vram && self.alloc_size > 0 {
            VRAM_BYTES.fetch_sub(self.alloc_size, std::sync::atomic::Ordering::Relaxed);
        }
        if let Some(base) = self.marks {
            let mut m = marks();
            m.owners.remove(&base);
            m.give_back(base, pages(self.start, self.len));
        }
    }
}

/// The 64 KiB title pages `[start, start + len)` touches.
fn pages(start: u64, len: u64) -> u32 {
    ((start + len.max(1) - 1) >> 16).saturating_sub(start >> 16) as u32 + 1
}

/// The bits of the buffer views' written-pages bitmap (`victory_shader::
/// emit`, "Write marks"): a run per tracked copy, a bit per title page it
/// touches, for as long as it lives.
#[derive(Default)]
struct Marks {
    /// Free runs below `end`, by first bit: their length.
    free: BTreeMap<u32, u32>,
    /// Bits handed out at some time: those past it are clear.
    end: u32,
    /// The copy holding each run, by its first bit.
    owners: BTreeMap<u32, std::sync::Weak<Shadow>>,
}

static MARKS: Mutex<Marks> = Mutex::new(Marks {
    free: BTreeMap::new(),
    end: 0,
    owners: BTreeMap::new(),
});

fn marks() -> std::sync::MutexGuard<'static, Marks> {
    MARKS.lock().unwrap_or_else(|e| e.into_inner())
}

impl Marks {
    /// A run of `n` bits: its first. None when the bitmap is full (the last
    /// bit takes the marks of offsets past it, so it is never handed out).
    fn take(&mut self, n: u32) -> Option<u32> {
        if let Some((&at, &len)) = self.free.iter().find(|&(_, &len)| len >= n) {
            self.free.remove(&at);
            if len > n {
                self.free.insert(at + n, len - n);
            }
            return Some(at);
        }
        if self.end + n < victory_shader::emit::VIEW_WRITE_BITS {
            self.end += n;
            return Some(self.end - n);
        }
        static ONCE: std::sync::Once = std::sync::Once::new();
        ONCE.call_once(|| tracing::warn!("no bits left in the views' written-pages bitmap: copies made now declare their shaders' writes"));
        None
    }

    /// The run of `n` bits at `at` is free again (joined to its neighbours).
    fn give_back(&mut self, mut at: u32, mut n: u32) {
        if let Some((&before, &len)) = self.free.range(..at).next_back()
            && before + len == at
        {
            self.free.remove(&before);
            (at, n) = (before, n + len);
        }
        if let Some(len) = self.free.remove(&(at + n)) {
            n += len;
        }
        self.free.insert(at, n);
    }
}

/// The buffer views' written-pages bitmap: the buffer, its mapping (bits
/// as dwords) and host GPU address; where the host has video memory, the
/// copy there that shaders mark in, with zeros to clear its write marks
/// from (`(copy, zeros)`).
struct Bitmap {
    _buffer: ID3D12Resource,
    vram: Option<(ID3D12Resource, ID3D12Resource)>,
    data: usize,
    gpu: u64,
}

// SAFETY: D3D12 resources are free-threaded; write marks in system memory
// are read and cleared only with no other submission in flight
// (`crate::host`), those in video memory are each submission's own
// ([`capture_write_marks`]); read marks may be read while one may still set
// more (taken by a later call).
unsafe impl Send for Bitmap {}
// SAFETY: as above.
unsafe impl Sync for Bitmap {}

static BITMAPS: [std::sync::OnceLock<Option<Bitmap>>; 2] = [const { std::sync::OnceLock::new() }; 2];

fn existing_bitmap(memory: bool) -> Option<&'static Bitmap> {
    BITMAPS[usize::from(memory)].get().and_then(Option::as_ref)
}

/// The bitmap of other shaders, or (`memory`) of shaders that use title
/// memory, whose marks go back only while it is not withheld
/// (`crate::memory::withholding`).
fn bitmap(memory: bool) -> Option<&'static Bitmap> {
    BITMAPS[usize::from(memory)]
        .get_or_init(|| {
            // Written pages, then one read mark per heap index.
            let bits = victory_shader::emit::VIEW_READS + victory_shader::emit::HEAP_VIEWS;
            let bytes = u64::from(bits / 8);
            let h = host()?;
            let (buffer, data) = h.shared_buffer(bytes)?;
            crate::host::name(&buffer, || "views' written pages and read slots".into());
            // SAFETY: our mapped buffer of `bytes` bytes.
            unsafe { std::ptr::write_bytes(data, 0, bytes as usize) };
            let vram = (!h.is_uma())
                .then(|| Some((h.default_uav_buffer(bytes)?, h.default_uav_buffer(u64::from(WRITE_WORDS) * 4)?)))
                .flatten();
            // SAFETY: a live buffer.
            let gpu = unsafe { vram.as_ref().map_or(&buffer, |v| &v.0).GetGPUVirtualAddress() };
            Some(Bitmap {
                _buffer: buffer,
                vram,
                data: data as usize,
                gpu,
            })
        })
        .as_ref()
}

/// The read marks of a bitmap, one bit per heap slot. Shaders of a
/// submission still in flight may set more (atomic ORs) while the CPU reads
/// them, which is harmless: what is not seen yet is seen by a later call.
/// The GPU's ORs are not atomic with the CPU's operations, though, so marks
/// are cleared only with no submission that marks reads in flight
/// ([`take_reads`] with `clear`, [`reads_since`]); as atomics, the CPU's own
/// accesses are not a data race.
fn read_marks(b: &'static Bitmap) -> &'static [AtomicU32] {
    let words = (victory_shader::emit::HEAP_VIEWS / 32) as usize;
    let first = (victory_shader::emit::VIEW_READS / 32) as usize;
    // SAFETY: the bitmap's read marks, `words` dwords from `first`, mapped
    // for good (the bitmap is never released); `AtomicU32` has `u32`'s
    // layout, and every access to them is atomic.
    unsafe { std::slice::from_raw_parts((b.data as *const AtomicU32).add(first), words) }
}
/// Whether a bitmap has a copy in video memory, which shaders mark in.
pub(crate) fn has_vram_bitmaps() -> bool {
    [false, true]
        .into_iter()
        .filter_map(existing_bitmap)
        .any(|b| b.vram.is_some())
}

/// Whether each submission's write marks are its own: every bitmap shaders
/// mark in is in video memory, where [`capture_write_marks`] takes them
/// apart per submission. Else shaders mark in system memory, and the marks
/// are taken only with nothing in flight after the submission
/// (`crate::host`).
pub(crate) fn captures_write_marks() -> bool {
    let mut maps = [false, true].into_iter().filter_map(existing_bitmap).peekable();
    maps.peek().is_some() && maps.all(|b| b.vram.is_some())
}

/// Dwords of a bitmap's write marks.
const WRITE_WORDS: u32 = victory_shader::emit::VIEW_WRITE_BITS / 32;

/// Bytes of the write marks that copies use now (whole dwords): every mark
/// a submission prepared now can set but those past the bitmap's end, which
/// land in its last dword and are no copy's.
pub(crate) fn write_marks_bytes() -> u64 {
    (marks().end.div_ceil(32).min(WRITE_WORDS - 1) * 4).into()
}

/// In the list run before a marking submission's own (`crate::host`): the
/// write marks in video memory cleared (the `bytes` in use and the last
/// dword; the last marking submission's were captured after it), and the
/// read marks as the CPU left them, so that clears of slots it forgot reach
/// the copy the shaders mark in.
pub(crate) fn sync_bitmaps_pre(list: &windows::Win32::Graphics::Direct3D12::ID3D12GraphicsCommandList, bytes: u64) {
    let reads = u64::from(victory_shader::emit::VIEW_READS / 8);
    let last = u64::from(WRITE_WORDS - 1) * 4;
    for b in [false, true].into_iter().filter_map(existing_bitmap) {
        if let Some((vram, zeros)) = &b.vram {
            // SAFETY: live bitmap buffers: the read marks from `reads` on,
            // the write marks (and the zeros) below it.
            unsafe {
                list.CopyBufferRegion(vram, reads, &b._buffer, reads, u64::from(victory_shader::emit::HEAP_VIEWS / 8));
                if bytes > 0 {
                    list.CopyBufferRegion(vram, 0, zeros, 0, bytes);
                }
                list.CopyBufferRegion(vram, last, zeros, last, 4);
            }
        }
    }
}

/// In the list run after a marking submission's own: its read marks into
/// system memory, where the CPU takes and clears them.
pub(crate) fn sync_bitmaps_post(list: &windows::Win32::Graphics::Direct3D12::ID3D12GraphicsCommandList) {
    let reads = u64::from(victory_shader::emit::VIEW_READS / 8);
    for b in [false, true].into_iter().filter_map(existing_bitmap) {
        if let Some((vram, _)) = &b.vram {
            // SAFETY: live bitmap buffers, the read marks from `reads` on.
            unsafe { list.CopyBufferRegion(&b._buffer, reads, vram, reads, u64::from(victory_shader::emit::HEAP_VIEWS / 8)) };
        }
    }
}

/// In the list run after a marking submission's own, before its fence
/// value: the write marks its shaders set in video memory (`bytes` of
/// each bitmap, [`write_marks_bytes`] when its list before cleared them),
/// copied into read-back staging `staging` from `at` on. Returns where
/// each bitmap's went (others, title memory); [`WriteMarks::staged`] reads
/// them once the submission has run. The next marking submission clears
/// them ([`sync_bitmaps_pre`]), so each submission's marks are its own,
/// whatever is in flight before or after it.
pub(crate) fn capture_write_marks(
    list: &windows::Win32::Graphics::Direct3D12::ID3D12GraphicsCommandList,
    staging: &ID3D12Resource,
    mut at: u64,
    bytes: u64,
) -> [Option<u64>; 2] {
    let mut placed = [None; 2];
    for (i, memory) in [false, true].into_iter().enumerate() {
        if bytes > 0
            && let Some((vram, _)) = existing_bitmap(memory).and_then(|b| b.vram.as_ref())
        {
            // SAFETY: live buffers; the caller made `bytes` room at `at`.
            unsafe { list.CopyBufferRegion(staging, at, vram, 0, bytes) };
            placed[i] = Some(at);
            at += bytes.next_multiple_of(16);
        }
    }
    placed
}

/// Room [`capture_write_marks`] needs in staging for `bytes` per bitmap.
pub(crate) fn capture_bytes(bytes: u64) -> u64 {
    let maps = [false, true]
        .into_iter()
        .filter_map(existing_bitmap)
        .filter(|b| b.vram.is_some())
        .count() as u64;
    maps * bytes.next_multiple_of(16)
}

/// One submission's write marks, as dwords per bitmap (others, shaders that
/// use title memory).
#[derive(Default)]
pub(crate) struct WriteMarks([Vec<u32>; 2]);

impl WriteMarks {
    /// What [`capture_write_marks`] put in staging mapped at `data` (`placed`
    /// as it returned, `bytes` each), with the marks shaders set in system
    /// memory, which are taken and cleared ([`WriteMarks::take_ram`]).
    ///
    /// # Safety
    /// The submission has run; `data` maps the staging, holding `bytes` at
    /// each placed offset.
    pub(crate) unsafe fn staged(data: usize, placed: [Option<u64>; 2], bytes: u64) -> WriteMarks {
        let mut marks = WriteMarks::take_ram();
        for (into, at) in marks.0.iter_mut().zip(placed) {
            let Some(at) = at else { continue };
            // SAFETY: as the caller says.
            let words = unsafe { std::slice::from_raw_parts((data + at as usize) as *const u32, (bytes / 4) as usize) };
            if into.len() < words.len() {
                into.resize(words.len(), 0);
            }
            for (w, &x) in into.iter_mut().zip(words) {
                *w |= x;
            }
        }
        marks
    }

    /// The write marks shaders set in the bitmaps in system memory (hosts
    /// with no copies of them in video memory, tests), cleared there. With
    /// copies in video memory none are set there.
    pub(crate) fn take_ram() -> WriteMarks {
        let used = (write_marks_bytes() / 4) as usize;
        let mut marks = WriteMarks::default();
        for (into, memory) in marks.0.iter_mut().zip([false, true]) {
            let Some(b) = existing_bitmap(memory) else { continue };
            // SAFETY: the mapped bitmap's write marks; the GPU is done with
            // them (`crate::host` takes them with nothing in flight after
            // the submission where shaders mark in system memory).
            let bits = unsafe { std::slice::from_raw_parts_mut(b.data as *mut u32, WRITE_WORDS as usize) };
            // Marks of offsets past the bitmap land in its last bit, no copy's.
            bits[WRITE_WORDS as usize - 1] = 0;
            if bits[..used].iter().any(|&w| w != 0) {
                *into = bits[..used].iter_mut().map(std::mem::take).collect();
            }
        }
        marks
    }

    /// The marked bits, of both bitmaps or (`memory`) of one.
    fn bits(&self, memory: Option<bool>) -> Vec<u32> {
        let mut bits = Vec::new();
        for (i, words) in self.0.iter().enumerate() {
            if memory.is_some_and(|m| usize::from(m) != i) {
                continue;
            }
            for (w, &word) in words.iter().enumerate() {
                let mut x = word;
                while x != 0 {
                    bits.push((w * 32) as u32 + x.trailing_zeros());
                    x &= x - 1;
                }
            }
        }
        bits
    }
}

/// The pages of copies in video memory that `taken` names, as read-backs
/// (buffer, offset, mirror, bytes).
pub(crate) fn vram_marked_reads(taken: &WriteMarks) -> Vec<(ID3D12Resource, u64, usize, u64)> {
    let mut marked = taken.bits(None);
    if marked.is_empty() {
        return Vec::new();
    }
    marked.sort_unstable();
    marked.dedup();
    let m = marks();
    let mut seen = std::collections::HashSet::new();
    let mut reads = Vec::new();
    for n in marked {
        let Some((&base, copy)) = m.owners.range(..=n).next_back() else {
            continue;
        };
        let Some(copy) = copy.upgrade() else {
            continue;
        };
        let k = n - base;
        if k >= pages(copy.start, copy.len) {
            continue;
        }
        let Some((cpu_ptr, vram_base, start, buf_len)) = copy.vram_readback() else {
            continue;
        };
        let page = ((copy.start >> 16) + u64::from(k)) << 16;
        if !seen.insert((Arc::as_ptr(&copy) as usize, page)) {
            continue;
        }
        let s = page.max(start);
        let e = (page + PAGE as u64).min(start + buf_len);
        if e > s {
            let off = s - start;
            reads.push((
                copy.buffer.clone(),
                vram_base + off,
                cpu_ptr + off as usize,
                e - s,
            ));
        }
    }
    reads
}

/// The host GPU address of the buffer views' written-pages bitmap, for the
/// root UAV of shaders that mark their writes (`memory`: shaders that use
/// title memory, see [`bitmap`]).
pub(crate) fn view_writes(memory: bool) -> Option<u64> {
    bitmap(memory).map(|b| b.gpu)
}

/// Heap slots whose texture views shaders read: the marks seen so far (a
/// slot's stays set in the bitmap until [`forget_read`]), and the slots to
/// clear at the next [`take_reads`].
static READS: Mutex<(Vec<u32>, Vec<u32>)> = Mutex::new((Vec::new(), Vec::new()));

/// After a submission: the heap slots its shaders first read texture views
/// at (`victory_shader::emit`, "Read marks"), each once until
/// [`forget_read`]. With `clear` (no submission that marks reads is in
/// flight), first clears the slots forgotten since the last such call; else
/// they wait for one.
pub(crate) fn take_reads(clear: bool) -> Vec<u32> {
    let words = (victory_shader::emit::HEAP_VIEWS / 32) as usize;
    // Both bitmaps: shaders that use title memory mark their reads in theirs.
    let maps: Vec<&[AtomicU32]> = [false, true]
        .into_iter()
        .filter_map(bitmap)
        .map(read_marks)
        .collect();
    if maps.is_empty() {
        return Vec::new();
    }
    let mut reads = READS.lock().unwrap_or_else(|e| e.into_inner());
    let (seen, forgotten) = &mut *reads;
    seen.resize(words, 0);
    let forgotten = if clear {
        std::mem::take(forgotten)
    } else {
        Vec::new()
    };
    for slot in forgotten {
        let (w, bit) = ((slot / 32) as usize, 1u32 << (slot % 32));
        if w < words {
            maps.iter().for_each(|bits| {
                bits[w].fetch_and(!bit, AtomicOrdering::AcqRel);
            });
            seen[w] &= !bit;
        }
    }
    let mut new = Vec::new();
    for (w, s) in seen.iter_mut().enumerate() {
        let mut fresh = maps
            .iter()
            .fold(0, |x, bits| x | bits[w].load(AtomicOrdering::Acquire))
            & !*s;
        *s |= fresh;
        while fresh != 0 {
            new.push(w as u32 * 32 + fresh.trailing_zeros());
            fresh &= fresh - 1;
        }
    }
    new
}

/// The heap slots shaders read texture views at since the last call (both
/// bitmaps), whose marks are cleared: what the title used (eviction stamps,
/// `crate::resource::evict`). With no submission that marks reads in
/// flight, after [`take_reads`] has seen every mark: its own record of
/// slots seen keeps a slot read again from being reported as new.
pub(crate) fn reads_since() -> Vec<u32> {
    let mut slots = Vec::new();
    for bits in [false, true].into_iter().filter_map(bitmap).map(read_marks) {
        for (w, word) in bits.iter().enumerate() {
            // Most words are clear (no submission marks meanwhile).
            if word.load(AtomicOrdering::Relaxed) == 0 {
                continue;
            }
            let mut x = word.swap(0, AtomicOrdering::AcqRel);
            while x != 0 {
                slots.push(w as u32 * 32 + x.trailing_zeros());
                x &= x - 1;
            }
        }
    }
    slots.sort_unstable();
    slots.dedup();
    slots
}

/// Heap slot `slot` now holds a texture view left null: its next read is
/// marked again.
pub(crate) fn forget_read(slot: u32) {
    READS.lock().unwrap_or_else(|e| e.into_inner()).1.push(slot);
}

/// After submission `owner`: every page its shaders marked as written
/// through views (`taken`, its own) goes back (consecutive whole pages of a
/// copy in runs under one lease of the guards, [`copy_back_run`]; others
/// page by page, [`Shadow::copy_back_marked`]), but for pages in `withheld`
/// (ranges of copies that lists whose writes stay out of title memory
/// declared) and, with `memory_withheld`, pages that shaders using title
/// memory marked. Before the copy-backs of
/// declared ranges, which disarm the copies. `unchanged`: pages of copies
/// in video memory shaders left as they were ([`Shadow::copy_back_marked`]).
// shortcut: marks do not tell lists apart: a page that a withheld list and
// another both wrote through views goes back unless the withheld list
// declared it.
pub(crate) fn copy_back_marks(
    owner: u64,
    taken: &WriteMarks,
    withheld: &[(Arc<Shadow>, usize, usize)],
    memory_withheld: bool,
    unchanged: &std::collections::HashSet<usize>,
) {
    let (mut marked, held) = if memory_withheld {
        (taken.bits(Some(false)), taken.bits(Some(true)))
    } else {
        (taken.bits(None), Vec::new())
    };
    #[cfg(feature = "diagnostics")]
    let _timed = crate::watch::Timed(std::time::Instant::now(), crate::copystat::C::MarkedUs);
    marked.sort_unstable();
    marked.dedup();
    let page_of = |m: &Marks, n: u32| {
        let (&base, copy) = m.owners.range(..=n).next_back()?;
        let copy = copy.upgrade()?;
        let k = n - base;
        (k < pages(copy.start, copy.len)).then(|| {
            let page = (((copy.start >> 16) + u64::from(k)) << 16) as usize;
            (copy, page)
        })
    };
    // Held back: dropped from the copies (`crate::memory::drop_withheld`).
    if !held.is_empty() {
        let m = marks();
        for (_, page) in held.into_iter().filter_map(|n| page_of(&m, n)) {
            crate::memory::drop_withheld(page, PAGE);
        }
    }
    if marked.is_empty() {
        return;
    }
    let pages: Vec<(Arc<Shadow>, usize)> = {
        let m = marks();
        marked.iter().filter_map(|&n| page_of(&m, n)).collect()
    };
    let pages: Vec<(Arc<Shadow>, usize)> = pages
        .into_iter()
        .filter(|(copy, page)| {
            let page = *page;
            !withheld
                .iter()
                .any(|(c, a, n)| Arc::ptr_eq(c, copy) && *a < page + PAGE && page < a + n)
        })
        .collect();
    #[cfg(feature = "diagnostics")]
    let counts = crate::watch::thread_counts();
    let items: Vec<(usize, usize, bool)> = pages
        .iter()
        .map(|(copy, page)| (Arc::as_ptr(copy) as usize, *page, runs_on() && copy.holds_whole(*page)))
        .collect();
    for (range, run) in mark_runs(&items, RUN_PAGES) {
        let copy = &pages[range.start].0;
        // Census (diagnostics): the pages' memory, a pool (private,
        // write-watched) or a view (a section's pages), and the protection
        // changes they cost.
        #[cfg(feature = "diagnostics")]
        let _kind = MarkKind::new(pages[range.start].1, range.len() as u64);
        if !run {
            #[cfg(feature = "diagnostics")]
            let _site = crate::watch::at_site(crate::watch::Site::MarkedPage);
            let written = copy.copy_back_marked(owner, pages[range.start].1, unchanged);
            #[cfg(feature = "diagnostics")]
            {
                crate::copystat::add(crate::copystat::C::MarkedPages, 1);
                crate::copystat::add(crate::copystat::C::MarkedBytes, written as u64);
            }
            #[cfg(not(feature = "diagnostics"))]
            let _ = written;
            continue;
        }
        let run: Vec<usize> = pages[range].iter().map(|&(_, page)| page).collect();
        let back = copy.copy_back_marked_run(owner, &run, unchanged);
        #[cfg(feature = "diagnostics")]
        {
            use crate::copystat::{C, add};
            add(C::MarkedPages, run.len() as u64);
            add(C::MarkedBytes, (back.written * PAGE) as u64);
            add(C::MarkedKept, back.kept as u64);
            add(C::MarkRuns, 1);
            add(C::MarkRunPages, run.len() as u64);
            add(C::MarkDiffRuns, back.diff_runs as u64);
            add(C::MarkDiffBytes, back.diff_bytes as u64);
            add(C::MarkMisses, back.misses as u64);
            add(C::MarkMerged, back.merged as u64);
            add(C::MarkBridged, back.bridged as u64);
            add(C::MarkThrough, back.through as u64);
            for (c, us) in [C::MarkPreUs, C::MarkWaitUs, C::MarkProtectUs, C::MarkWriteUs, C::MarkVerifyUs]
                .into_iter()
                .zip(back.us)
            {
                add(c, us);
            }
        }
        #[cfg(not(feature = "diagnostics"))]
        let _ = back;
    }
    #[cfg(feature = "diagnostics")]
    {
        let (rounds, protects) = crate::watch::thread_counts();
        crate::copystat::add(crate::copystat::C::MarkLeases, rounds - counts.0);
        crate::copystat::add(crate::copystat::C::MarkProtects, protects - counts.1);
    }
}

/// Counts a range of marked pages and the protection changes it cost by
/// the memory it lies in, when dropped (diagnostics census).
#[cfg(feature = "diagnostics")]
struct MarkKind(bool, u64, u64);

#[cfg(feature = "diagnostics")]
impl MarkKind {
    fn new(page: usize, pages: u64) -> Self {
        use windows_sys::Win32::System::Memory::GetWriteWatch;
        // A pool is write-watched (asking about one page, no reset: cheap,
        // where `VirtualQuery` walks the runtime's big pools).
        let (mut found, mut count, mut granularity) = (0usize, 1usize, 0u32);
        // SAFETY: room for one answer; no reset, so nothing changes.
        let pool = unsafe {
            GetWriteWatch(0, page as *const _, PAGE, (&raw mut found).cast(), &mut count, &mut granularity) == 0
        };
        Self(!pool, pages, crate::watch::thread_counts().1)
    }
}

#[cfg(feature = "diagnostics")]
impl Drop for MarkKind {
    fn drop(&mut self) {
        use crate::copystat::{C, add};
        let protects = crate::watch::thread_counts().1 - self.2;
        let (pages, calls) = if self.0 {
            (C::MarkViewPages, C::MarkViewProtects)
        } else {
            (C::MarkPoolPages, C::MarkPoolProtects)
        };
        add(pages, self.1);
        add(calls, protects);
    }
}

/// Whether marked pages go back in runs ([`copy_back_run`]); off
/// (`VICTORY_MARK_RUNS=0`), page by page as before (comparison runs).
fn runs_on() -> bool {
    static ON: std::sync::LazyLock<bool> =
        std::sync::LazyLock::new(|| std::env::var("VICTORY_MARK_RUNS").as_deref() != Ok("0"));
    *ON
}

/// Most pages a run of marked pages goes back in (2 MiB): the guard lock is
/// held for the run's writes and checks, and the title's faults wait.
const RUN_PAGES: usize = 32;

/// Most byte runs a page's changes go back in; past it the closest join.
const DIFF_RUNS: usize = 64;

/// The marked pages `items` (copy, page address, whether the copy holds the
/// page whole and tracks it), in their order, cut into what goes back
/// together: runs of consecutive whole pages of one copy, at most `max`
/// long (`true`), and other pages alone (`false`).
fn mark_runs(items: &[(usize, usize, bool)], max: usize) -> Vec<(std::ops::Range<usize>, bool)> {
    let mut out = Vec::new();
    let mut i = 0;
    while i < items.len() {
        let (copy, _, whole) = items[i];
        let mut j = i + 1;
        while whole
            && j < items.len()
            && j - i < max
            && items[j].0 == copy
            && items[j].2
            && items[j].1 == items[j - 1].1 + PAGE
        {
            j += 1;
        }
        out.push((i..j, whole));
        i = j;
    }
    out
}

/// Appends to `out` the byte runs `[a, b)` (offsets) where `new` differs
/// from `old` (equal lengths), found by aligned 8-byte words: consecutive
/// differing words make one run, cut at its ends to the bytes that differ.
/// Past `max` runs the closest join, the bytes between them too.
fn diff_runs(new: &[u8], old: &[u8], max: usize, out: &mut Vec<(u32, u32)>) {
    debug_assert_eq!(new.len(), old.len());
    const CHUNK: usize = 256;
    let (first, n) = (out.len(), new.len());
    let mut at = 0;
    while at < n {
        let end = (at + CHUNK).min(n);
        if new[at..end] != old[at..end] {
            for w in (at..end).step_by(8) {
                let we = (w + 8).min(end);
                let (a, b) = if we - w == 8 {
                    let word = |s: &[u8]| u64::from_le_bytes(s[w..we].try_into().expect("8 bytes"));
                    let x = word(new) ^ word(old);
                    if x == 0 {
                        continue;
                    }
                    (w + x.trailing_zeros() as usize / 8, w + 8 - x.leading_zeros() as usize / 8)
                } else {
                    let differs = |k: &usize| new[*k] != old[*k];
                    let Some(a) = (w..we).find(differs) else {
                        continue;
                    };
                    (a, (w..we).rfind(differs).unwrap_or(a) + 1)
                };
                match out[first..].last_mut() {
                    // The word before differs too.
                    Some(r) if w >= 8 && r.1 as usize > w - 8 => r.1 = b as u32,
                    _ => out.push((a as u32, b as u32)),
                }
            }
        }
        at = end;
    }
    let mut gap = 16;
    while out.len() - first > max {
        let mut k = first;
        for j in first + 1..out.len() {
            if out[j].0 - out[k].1 <= gap {
                out[k].1 = out[j].1;
            } else {
                k += 1;
                out[k] = out[j];
            }
        }
        out.truncate(k + 1);
        gap *= 2;
    }
}

thread_local! {
    /// One buffer per thread for a title page, not a fresh zeroed 64 KiB
    /// for each (thousands of marked pages a world frame; a fresh
    /// allocation's pages fault in again). Every byte of it in use is read
    /// from the title first.
    static TITLE: std::cell::RefCell<Vec<u8>> = const { std::cell::RefCell::new(Vec::new()) };
}

/// What a run of marked pages did ([`copy_back_run`]).
#[derive(Debug, Default, PartialEq)]
pub(crate) struct RunBack {
    /// Pages written into title memory; byte runs and bytes written.
    pub(crate) written: usize,
    pub(crate) diff_runs: usize,
    pub(crate) diff_bytes: usize,
    /// Pages written that the title wrote too while open: left changed.
    pub(crate) misses: usize,
    /// Pages opened only to bridge the run's written pages; pages written
    /// through the runtime's own mapping of them (no protection change).
    pub(crate) bridged: usize,
    pub(crate) through: usize,
    /// Pages the title wrote since the copy took them: gone back by their
    /// saved bytes and left changed; kept the title's bytes (none saved).
    pub(crate) merged: usize,
    pub(crate) kept: usize,
    /// Protection changes asked.
    pub(crate) protects: u64,
    /// Microseconds (diagnostics): reading and comparing before the lease,
    /// waiting for it, changing protections, writing, checking after.
    #[cfg(feature = "diagnostics")]
    pub(crate) us: [u64; 5],
}

#[cfg(test)]
type RunHook = Option<Box<dyn Fn(u8, usize)>>;

#[cfg(test)]
thread_local! {
    /// Test hook: called with 0 after the pages are read (before the
    /// lease) and 1 once they are open (before the writes), with the run's
    /// first page.
    pub(crate) static RUN_HOOK: std::cell::RefCell<RunHook> = const { std::cell::RefCell::new(None) };
}

#[cfg(test)]
fn run_hook(at: u8, page: usize) {
    RUN_HOOK.with(|h| {
        if let Some(f) = &*h.borrow() {
            f(at, page);
        }
    });
}

/// The copy-back of marked pages `pages` (consecutive whole 64 KiB title
/// pages of one copy tracked by `t`, ascending; `host` the copy's bytes of
/// them) in one [`crate::watch::Lease`]: what [`Shadow::copy_back_marked`]
/// does page by page, with one round of the guard lock and one protection
/// change each way for the run instead of three rounds and two changes a
/// page.
///
/// Per page, what the shaders changed is what differs from the copy's bytes
/// before the submission, and only those bytes go back (by 8-byte words, cut
/// to the bytes at a run's ends):
/// - The title has not written it since the copy took it (guarded): its
///   bytes are the copy's from before. They are read before the lease; the
///   lease finds it still guarded, so no title write landed since
///   (`crate::watch::Lease`). After the writes it is guarded again and
///   taken only if title memory then equals the copy: a title write on the
///   open page (no fault) either missed the bytes written, and differs, or
///   hit them, and is the last word. Taken, it is what the copy holds.
/// - The title wrote it (it saved what the page held at its first write,
///   [`crate::watch::Tracked::saved`]): the changes against those bytes go
///   back over the title's, and the page stays changed: the next refresh
///   takes the title's bytes, which no write of ours overwrote but where the
///   shaders changed the same bytes. A page found written under the lease
///   (a fault came between the read and the lease) goes this way too: it
///   was armed, so its first write saved it.
/// - The title wrote it and nothing was saved (not armed): it keeps the
///   title's bytes, losing the shaders' (counted), as page by page.
/// - A page in `unchanged` (the shaders changed nothing, `crate::host`):
///   nothing goes back, and if guarded it is taken unread.
/// - Pages between those written are opened and guarded with them, so one
///   protection change covers the run each way: open ones as they are, and
///   guarded ones only where this copy alone holds them
///   ([`crate::watch::Lease::alone`]); those are checked as the written
///   ones, and marked changed if the title wrote them while open.
/// - Pages a section backs (the runtime's physical-page views) are written
///   through the runtime's own writable mapping of the section
///   ([`crate::watch::writable_alias`]) and never opened: the title's
///   mappings stay guarded throughout, so a title write meanwhile faults,
///   waits for the lease and lands after ours, and the page is taken with
///   no check.
///
/// A title write faulting while the lease is held lands after it, so after
/// our writes. The one write of the title's that ours can overwrite lands
/// on an open page, before ours, on bytes of an 8-byte word the shaders
/// changed: a race of the title with the shaders on those bytes, which the
/// console orders either way; page by page the whole page was written.
fn copy_back_run(
    t: &crate::watch::Tracked,
    pages: &[usize],
    host: &[u8],
    unchanged: &std::collections::HashSet<usize>,
) -> RunBack {
    debug_assert_eq!(host.len(), pages.len() * PAGE);
    /// Per page: byte runs `[a, b)` of `diffs` to write; the title had not
    /// written it (`Clean`) or had (`Merge`); it keeps the title's bytes
    /// (`Kept`); title memory is not readable (`Skip`, as page by page).
    #[derive(Clone, Copy)]
    enum Plan {
        Clean(usize, usize),
        Merge(usize, usize),
        Kept,
        Skip,
    }
    fn page_in(host: &[u8], i: usize) -> &[u8] {
        &host[i * PAGE..(i + 1) * PAGE]
    }
    let host_page = |i: usize| page_in(host, i);
    let first = pages[0] / PAGE;
    let mut diffs: Vec<(u32, u32)> = Vec::new();
    let mut title = TITLE.with(|b| std::mem::take(&mut *b.borrow_mut()));
    title.resize(PAGE, 0);
    let merge = |i: usize, before: &[u8], diffs: &mut Vec<(u32, u32)>| {
        let s = diffs.len();
        diff_runs(host_page(i), before, DIFF_RUNS, diffs);
        Plan::Merge(s, diffs.len())
    };
    #[cfg(feature = "diagnostics")]
    let mut clock = std::time::Instant::now();
    // Adds the time since the last lap to `us[k]` (diagnostics).
    #[cfg(feature = "diagnostics")]
    let mut lap = |us: &mut [u64; 5], k: usize| {
        let now = std::time::Instant::now();
        us[k] += (now - clock).as_micros() as u64;
        clock = now;
    };
    #[cfg(feature = "diagnostics")]
    let mut us = [0u64; 5];
    // Before the lease: what the shaders changed of each page.
    let mut plans: Vec<Plan> = Vec::with_capacity(pages.len());
    for (i, &page) in pages.iter().enumerate() {
        let plan = if let Some(before) = t.saved(page / PAGE) {
            merge(i, &before, &mut diffs)
        } else if t.changed(page / PAGE) {
            Plan::Kept
        } else if unchanged.contains(&(host_page(i).as_ptr() as usize)) {
            Plan::Clean(diffs.len(), diffs.len())
        } else if crate::host::read_into(page as *const u8, &mut title) {
            let s = diffs.len();
            diff_runs(host_page(i), &title, DIFF_RUNS, &mut diffs);
            Plan::Clean(s, diffs.len())
        } else {
            Plan::Skip
        };
        plans.push(plan);
    }
    // Pages a section backs (the runtime's physical-page views) are
    // written through the runtime's own writable mapping of them: the
    // title's stays guarded, so no protection changes and no check after
    // (asked before the lease: the runtime's lock is taken around calls in
    // here).
    let alias: Vec<Option<usize>> = pages.iter().map(|&page| crate::watch::writable_alias(page)).collect();
    #[cfg(test)]
    run_hook(0, pages[0]);
    #[cfg(feature = "diagnostics")]
    lap(&mut us, 0);
    let mut lease = t.lease();
    #[cfg(feature = "diagnostics")]
    lap(&mut us, 1);
    // Our writes below must not save the pages for this copy (unarmed under
    // the lease: a first title write before it still saved them). A page
    // the title wrote since it was read goes back by what that write saved.
    for (i, &page) in pages.iter().enumerate() {
        let pn = page / PAGE;
        t.unarm(pn);
        let again = match plans[i] {
            Plan::Clean(..) => !lease.guarded(pn) || t.changed(pn),
            Plan::Kept => true,
            Plan::Merge(..) | Plan::Skip => false,
        };
        if again {
            plans[i] = match t.saved(pn) {
                Some(before) => merge(i, &before, &mut diffs),
                None => Plan::Kept,
            };
        }
    }
    let mut back = RunBack::default();
    let written: Vec<usize> = pages
        .iter()
        .zip(&plans)
        .filter_map(|(&page, plan)| match *plan {
            Plan::Clean(a, b) | Plan::Merge(a, b) if a < b => Some(page / PAGE),
            _ => None,
        })
        .collect();
    let (through, opened): (Vec<usize>, Vec<usize>) = written.iter().partition(|&&pn| alias[pn - first].is_some());
    back.through = through.len();
    // Between the pages opened, those not written are opened too, so that
    // one protection change covers the run each way (the shaders leave
    // many pages between those they change as they were): pages open
    // already, and guarded ones this copy alone holds, checked after.
    let bridge: Vec<usize> = match (opened.first(), opened.last()) {
        (Some(&lo), Some(&hi)) => (lo + 1..hi)
            .filter(|&pn| match plans[pn - first] {
                Plan::Clean(a, b) | Plan::Merge(a, b) if a < b => false,
                Plan::Skip => false,
                _ => !lease.guarded(pn) || lease.alone(pn),
            })
            .collect(),
        _ => Vec::new(),
    };
    lease.will_write(&through);
    back.protects += lease.open(&opened, &bridge);
    #[cfg(feature = "diagnostics")]
    lap(&mut us, 2);
    #[cfg(test)]
    run_hook(1, pages[0]);
    for &pn in &written {
        let i = pn - first;
        let (Plan::Clean(a, b) | Plan::Merge(a, b)) = plans[i] else {
            continue;
        };
        for &(x, y) in &diffs[a..b] {
            let (x, y) = (x as usize, y as usize);
            let to = alias[i].unwrap_or(pn * PAGE) + x;
            crate::host::write_mapped(to as *mut u8, host_page(i)[x..].as_ptr(), y - x);
            back.diff_bytes += y - x;
        }
        back.diff_runs += b - a;
        back.written += 1;
    }
    #[cfg(feature = "diagnostics")]
    lap(&mut us, 3);
    // The pages the title had not written: guarded again, and taken where
    // title memory holds what the copy does (those opened, checked).
    let clean: Vec<usize> = pages
        .iter()
        .zip(&plans)
        .filter(|(_, plan)| matches!(plan, Plan::Clean(..)))
        .map(|(&page, _)| page / PAGE)
        .collect();
    let (guarded, calls) = lease.guard(&clean);
    back.protects += calls;
    #[cfg(feature = "diagnostics")]
    lap(&mut us, 2);
    let (mut taken, mut changed) = (Vec::with_capacity(guarded.len()), Vec::new());
    for pn in guarded {
        let i = pn - first;
        let Plan::Clean(a, b) = plans[i] else {
            continue;
        };
        let bridged = bridge.binary_search(&pn).is_ok();
        back.bridged += usize::from(bridged);
        let opened = (a < b && alias[i].is_none()) || bridged;
        if !opened || crate::host::read_into((pn * PAGE) as *const u8, &mut title) && title[..] == *host_page(i) {
            taken.push(pn);
        } else {
            back.misses += 1;
            if bridged {
                changed.push(pn);
            }
        }
    }
    lease.written(&changed);
    lease.taken(&taken);
    drop(lease);
    #[cfg(feature = "diagnostics")]
    {
        lap(&mut us, 4);
        back.us = us;
    }
    TITLE.with(|b| *b.borrow_mut() = title);
    for plan in &plans {
        match plan {
            Plan::Merge(..) => back.merged += 1,
            Plan::Kept => back.kept += 1,
            _ => {}
        }
    }
    back
}

/// The copy of [`Flight`]'s bottom-of-pipe writes: none, so every copy and
/// every read of title memory crosses them.
pub(crate) const LABEL: usize = usize::MAX;

/// Title memory that submissions in flight write and that reaches title
/// memory only once each completes (`crate::host::Host::finish`), by
/// submission: ranges `(start, end, copy)` their shaders declared as written
/// through a copy (its address; marked ranges as whole pages, as their
/// copy-back writes them) and their bottom-of-pipe writes ([`LABEL`]).
/// The copy itself holds what its shaders wrote for later submissions; any
/// other reader of that memory (another copy of it, a title memory read at
/// submission, a texture over it) would take title memory without it.
#[derive(Default)]
pub(crate) struct Flight {
    owners: Vec<Flying>,
}

/// One submission's part of [`Flight`]: its ranges sorted by start, and
/// the largest end up to each.
struct Flying {
    owner: u64,
    ranges: Vec<(usize, usize, usize)>,
    ends: Vec<usize>,
}

impl Flight {
    /// Submission `owner` writes `ranges` until it [`Flight::land`]s.
    pub(crate) fn fly(&mut self, owner: u64, mut ranges: Vec<(usize, usize, usize)>) {
        ranges.retain(|r| r.0 < r.1);
        if ranges.is_empty() {
            return;
        }
        // One run per copy where its ranges overlap or touch.
        ranges.sort_unstable_by_key(|&(a, _, c)| (c, a));
        let mut merged: Vec<(usize, usize, usize)> = Vec::with_capacity(ranges.len());
        for (a, e, c) in ranges {
            match merged.last_mut() {
                Some((_, end, copy)) if *copy == c && a <= *end => *end = (*end).max(e),
                _ => merged.push((a, e, c)),
            }
        }
        merged.sort_unstable();
        let ends = merged
            .iter()
            .scan(0, |max, r| {
                *max = r.1.max(*max);
                Some(*max)
            })
            .collect();
        self.owners.push(Flying { owner, ranges: merged, ends });
    }

    /// Submission `owner` completed: what it wrote is in title memory.
    pub(crate) fn land(&mut self, owner: u64) {
        self.owners.retain(|o| o.owner != owner);
    }

    /// The latest submission in flight that writes title memory in `[start,
    /// end)` other than through copy `copy`, if any, and whether what it
    /// writes there is a bottom-of-pipe write ([`LABEL`]).
    pub(crate) fn crossing(&self, start: usize, end: usize, copy: usize) -> Option<(u64, bool)> {
        let mut latest: Option<(u64, bool)> = None;
        for Flying { owner, ranges, ends } in &self.owners {
            if latest.is_some_and(|l| l.0 >= *owner) {
                continue;
            }
            let mut i = ranges.partition_point(|r| r.0 < end);
            while i > 0 && ends[i - 1] > start {
                i -= 1;
                let (a, e, c) = ranges[i];
                if a < end && start < e && c != copy {
                    latest = Some((*owner, c == LABEL));
                    break;
                }
            }
        }
        latest
    }

    /// Whether no submission in flight writes anything.
    pub(crate) fn is_empty(&self) -> bool {
        self.owners.is_empty()
    }
}

static FLIGHT: Mutex<Flight> = Mutex::new(Flight { owners: Vec::new() });

/// Whether [`FLIGHT`] holds anything: asked first by the many lookups of a
/// submission's preparing.
static FLYING: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

fn flight() -> std::sync::MutexGuard<'static, Flight> {
    FLIGHT.lock().unwrap_or_else(|e| e.into_inner())
}

/// [`Flight::fly`] of the submissions in flight.
pub(crate) fn fly(owner: u64, ranges: Vec<(usize, usize, usize)>) {
    let mut f = flight();
    f.fly(owner, ranges);
    FLYING.store(!f.is_empty(), AtomicOrdering::Release);
}

/// [`Flight::land`] of the submissions in flight.
pub(crate) fn land(owner: u64) {
    let mut f = flight();
    f.land(owner);
    FLYING.store(!f.is_empty(), AtomicOrdering::Release);
}

/// [`Flight::crossing`] of the submissions in flight. Asked while
/// preparing a submission, under the submission lock, which their
/// completions take too: the answer holds until it is let go.
pub(crate) fn crossing(start: usize, end: usize, copy: usize) -> Option<(u64, bool)> {
    if !FLYING.load(AtomicOrdering::Acquire) {
        return None;
    }
    flight().crossing(start, end, copy)
}

struct Entry {
    end: u64,
    shadow: Option<Arc<Shadow>>,
}

/// Title buffers by (start address, resource), with their copy once made.
static BUFFERS: Mutex<BTreeMap<(u64, usize), Entry>> = Mutex::new(BTreeMap::new());

fn buffers() -> std::sync::MutexGuard<'static, BTreeMap<(u64, usize), Entry>> {
    BUFFERS.lock().unwrap_or_else(|e| e.into_inner())
}

/// Counts copies made and title buffers released: [`all`] may differ when
/// it changed.
static GENERATION: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// The generation of [`all`]: read it before calling that.
pub(crate) fn generation() -> u64 {
    GENERATION.load(std::sync::atomic::Ordering::Acquire)
}

fn changed() {
    GENERATION.fetch_add(1, std::sync::atomic::Ordering::Release);
}

/// A title buffer exists at `start` (`len` bytes).
pub(crate) fn register(resource: usize, start: u64, len: u64) {
    buffers().insert(
        (start, resource),
        Entry {
            end: start + len,
            shadow: None,
        },
    );
}

/// The title released it; lists that still hold its copy keep it alive.
pub(crate) fn unregister(resource: usize, start: u64) {
    if buffers().remove(&(start, resource)).is_some_and(|e| e.shadow.is_some()) {
        changed();
    }
}

/// The copy of the innermost title buffer holding `[address, address +
/// len)`, made on first use.
pub(crate) fn find(address: u64, len: u64) -> Option<Arc<Shadow>> {
    let end = address.checked_add(len.max(1))?;
    let mut map = buffers();
    let key = map
        .range(..=(address, usize::MAX))
        .rev()
        .take(64)
        .find(|(_, e)| e.end >= end)
        .map(|(k, _)| *k);
    let Some(key) = key else {
        drop(map);
        return loose(address, end);
    };
    let entry = map.get_mut(&key)?;
    if entry.shadow.is_none() {
        let len = entry.end - key.0;
        // Rounded up so that views ending in the buffer's last 256 bytes fit.
        // Shaders write some buffers (unordered access), so not upload memory.
        let alloc_size = len.next_multiple_of(1 << 16);
        let backing = alloc_backing(alloc_size, 128 << 20, false, || {
            format!("title buffer {:#x}+{len:#x}", key.0)
        })?;
        if len >= 16 << 20 {
            tracing::info!(
                start = format_args!("{:#x}", key.0),
                len,
                "large buffer copied to the host"
            );
        } else {
            tracing::debug!(
                start = format_args!("{:#x}", key.0),
                len,
                "buffer copied to the host"
            );
        }
        entry.shadow = Some(Shadow::new(backing, key.0, len));
        changed();
    }
    entry.shadow.clone()
}

/// Titles also use memory outside any buffer object as GPU memory (Grand
/// Theft Auto VI's interface vertices, from its own allocator). Such memory
/// gets copies of the aligned windows it touches.
/// shortcut: copies of windows are never freed; key them to the title's
/// allocations if memory use grows.
const WINDOW: u64 = 4 << 20;

/// Copies of title memory outside buffers, by (start, length).
static LOOSE: Mutex<BTreeMap<(u64, u64), Arc<Shadow>>> = Mutex::new(BTreeMap::new());

fn loose(address: u64, end: u64) -> Option<Arc<Shadow>> {
    let start = address & !(WINDOW - 1);
    let len = end.next_multiple_of(WINDOW) - start;
    let mut map = LOOSE.lock().unwrap_or_else(|e| e.into_inner());
    if let Some(shadow) = map.get(&(start, len)) {
        return Some(shadow.clone());
    }
    let backing = alloc_backing(len, 128 << 20, false, || {
        format!("title memory {start:#x}+{len:#x}")
    })?;
    tracing::debug!(
        start = format_args!("{start:#x}"),
        len,
        "loose memory copied to the host"
    );
    let shadow = Shadow::new(backing, start, len);
    map.insert((start, len), shadow.clone());
    changed();
    Some(shadow)
}

/// The copy a buffer view of `len` bytes at `address`, whose elements are
/// `element` bytes, reads: the innermost title buffer's ([`find`]), unless
/// the view does not start on an element of that copy. A host view counts
/// its first element from its buffer's start; a console view starts at any
/// address, and the title places structured views at ring positions inside
/// its big GPU heap (the meshlet culling's 456-byte constants, the colour
/// bakes' parameter blocks: refused, their slots read zeros, every meshlet
/// culled). Such a view gets a copy of its own bytes, starting at its
/// address (`true` with it), which lives while a heap slot made from the
/// view or a list holds it.
pub(crate) fn for_view(address: u64, len: u64, element: u64) -> Option<(Arc<Shadow>, bool)> {
    let shadow = find(address, len)?;
    if element == 0 || shadow.offset(address).is_multiple_of(element) {
        return Some((shadow, false));
    }
    #[cfg(feature = "diagnostics")]
    VIEWS[0].fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    exact(address, len, element).map(|s| (s, true))
}

/// Asks for, and copies made of, views off their copy's elements, the
/// copies' bytes and the largest (diagnostics).
#[cfg(feature = "diagnostics")]
static VIEWS: [std::sync::atomic::AtomicU64; 4] = [const { std::sync::atomic::AtomicU64::new(0) }; 4];

/// Logs [`VIEWS`] per frame over `frames`.
#[cfg(feature = "diagnostics")]
pub(crate) fn log_views(frames: f64) {
    use std::sync::atomic::Ordering::Relaxed;
    let [asks, made, bytes, largest] = [0, 1, 2, 3].map(|i| VIEWS[i].swap(0, Relaxed));
    let live = EXACT
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .1
        .values()
        .filter(|w| w.strong_count() > 0)
        .count();
    let (chunks, held) = {
        let a = arena();
        // SAFETY: live buffers.
        let size = |c: &Chunk| unsafe { c.buffer.GetDesc() }.Width;
        let held: u64 = a.chunks.iter().map(|c| size(c) - c.free.values().sum::<u64>()).sum();
        (a.chunks.len(), held)
    };
    if asks > 0 {
        tracing::info!(
            "buffer views off their copy's elements: {:.0} asked for/frame, {:.0} copies made/frame ({:.1} KB), largest {largest} bytes, {live} live in {chunks} shared buffers ({:.1} MB held)",
            asks as f64 / frames,
            made as f64 / frames,
            bytes as f64 / frames / 1e3,
            held as f64 / 1e6
        );
    }
}

/// Copies of single views' bytes ([`for_view`]), by (start, length), held
/// by the heap slots made from them and the lists that use them: a copy is
/// found again for as long as a slot shows it, so draws refresh the copy
/// the GPU reads (a capped cache dropped copies slots still showed, and a
/// new copy of the same bytes took the refreshes). With the size the map
/// is next pruned of dead entries at.
#[allow(clippy::type_complexity)]
static EXACT: Mutex<(usize, BTreeMap<(u64, u64, u64), std::sync::Weak<Shadow>>)> = Mutex::new((0, BTreeMap::new()));

fn exact(address: u64, len: u64, element: u64) -> Option<Arc<Shadow>> {
    let mut guard = EXACT.lock().unwrap_or_else(|e| e.into_inner());
    let (prune_at, map) = &mut *guard;
    if let Some(shadow) = map.get(&(address, len, element)).and_then(std::sync::Weak::upgrade) {
        return Some(shadow);
    }
    // Ring positions change every frame: entries die by the hundred.
    if map.len() >= *prune_at {
        map.retain(|_, w| w.strong_count() > 0);
        *prune_at = (2 * map.len()).max(1024);
    }
    // A run of a shared buffer, its start a whole number of elements in.
    let (backing, slice) = arena().take(len, element)?;
    let shadow = Shadow::new_at(backing, address, len, Some(slice));
    map.insert((address, len, element), Arc::downgrade(&shadow));
    #[cfg(feature = "diagnostics")]
    {
        use std::sync::atomic::Ordering::Relaxed;
        VIEWS[1].fetch_add(1, Relaxed);
        VIEWS[2].fetch_add(len, Relaxed);
        VIEWS[3].fetch_max(len, Relaxed);
    }
    Some(shadow)
}

/// A run of an [`Arena`] buffer: the buffer, where the run starts, its
/// bytes.
#[derive(Clone, Copy)]
struct Slice {
    chunk: usize,
    offset: u64,
    len: u64,
}

/// Shared buffers for the copies of single views ([`exact`]): they are
/// small (~2 KB on average) and come and go by the hundred a frame, and a
/// buffer of their own each cost a kernel call to make and one to free on
/// the title's threads and the submitting one (300-820 a frame), and 64 KiB
/// of memory each (~5,000 live).
struct Chunk {
    buffer: ID3D12Resource,
    cpu: Arc<CpuMem>,
    vram: bool,
    data: usize,
    gpu: u64,
    free: BTreeMap<u64, u64>,
}

#[derive(Default)]
struct Arena {
    /// Each buffer, its mapping and host address, and its free runs
    /// (offset → bytes).
    chunks: Vec<Chunk>,
}

/// Bytes of each arena buffer (a larger copy gets one of its size).
const CHUNK: u64 = 16 << 20;

static ARENA: Mutex<Arena> = Mutex::new(Arena { chunks: Vec::new() });

fn arena() -> std::sync::MutexGuard<'static, Arena> {
    ARENA.lock().unwrap_or_else(|e| e.into_inner())
}

impl Arena {
    /// A run of `len` bytes starting at a multiple of `element` (and of 256)
    /// in a shared buffer: the buffer, the run's mapping and host address.
    fn take(&mut self, len: u64, element: u64) -> Option<(Backing, Slice)> {
        let gcd = |mut a: u64, mut b: u64| {
            while b != 0 {
                (a, b) = (b, a % b);
            }
            a
        };
        let align = 256 / gcd(256, element.max(1)) * element.max(1);
        let len = len.max(1).next_multiple_of(4);
        let fits = |free: &BTreeMap<u64, u64>| {
            free.iter().find_map(|(&at, &n)| {
                let start = at.next_multiple_of(align);
                (start + len <= at + n).then_some((at, n, start))
            })
        };
        let found = self.chunks.iter().enumerate().find_map(|(c, chunk)| fits(&chunk.free).map(|f| (c, f)));
        let (chunk, (at, n, start)) = match found {
            Some(found) => found,
            None => {
                let size = CHUNK.max((len + align).next_multiple_of(1 << 16));
                let b = alloc_backing(size, CHUNK, true, || format!("title views {size:#x}"))?;
                let free = BTreeMap::from([(0, size)]);
                self.chunks.push(Chunk {
                    buffer: b.buffer,
                    cpu: b.cpu,
                    vram: b.vram,
                    data: b.data as usize,
                    gpu: b.gpu,
                    free,
                });
                let c = self.chunks.len() - 1;
                (c, fits(&self.chunks[c].free)?)
            }
        };
        let Chunk { buffer, cpu, vram, data, gpu, free } = &mut self.chunks[chunk];
        free.remove(&at);
        if start > at {
            free.insert(at, start - at);
        }
        if at + n > start + len {
            free.insert(start + len, at + n - start - len);
        }
        let slice = Slice { chunk, offset: start, len };
        let backing = Backing {
            buffer: buffer.clone(),
            cpu: cpu.clone(),
            vram: *vram,
            alloc_size: 0,
            // SAFETY: `start` is within the chunk's allocation.
            data: unsafe { (*data as *mut u8).add(start as usize) },
            gpu: *gpu + start,
        };
        Some((backing, slice))
    }

    /// A run goes back, joined to its free neighbours.
    fn give_back(&mut self, slice: Slice) {
        let Some(chunk) = self.chunks.get_mut(slice.chunk) else { return };
        let free = &mut chunk.free;
        let (mut at, mut n) = (slice.offset, slice.len);
        if let Some((&before, &m)) = free.range(..at).next_back()
            && before + m == at
        {
            free.remove(&before);
            (at, n) = (before, n + m);
        }
        if let Some(m) = free.remove(&(at + n)) {
            n += m;
        }
        free.insert(at, n);
    }
}

/// Every host copy of title memory made so far (buffers and windows).
pub(crate) fn all() -> Vec<Arc<Shadow>> {
    let mut all: Vec<Arc<Shadow>> = buffers().values().filter_map(|e| e.shadow.clone()).collect();
    all.extend(LOOSE.lock().unwrap_or_else(|e| e.into_inner()).values().cloned());
    all
}

/// For device-removed reports: the copy of title memory whose host range
/// starts closest below host GPU address `va`, and where `va` is in it.
pub(crate) fn near(va: u64) -> String {
    let mut all: Vec<(u64, u64, u64)> = buffers()
        .values()
        .filter_map(|e| e.shadow.as_deref().map(|s| (s.gpu, s.start, s.len)))
        .collect();
    all.extend(
        LOOSE
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .values()
            .map(|s| (s.gpu, s.start, s.len)),
    );
    let below = all.iter().filter(|c| c.0 <= va).max();
    let above = all.iter().filter(|c| c.0 > va).min();
    let mut text = match below {
        Some(&(gpu, start, len)) => format!(
            "; the nearest title memory copy below it: {start:#x}+{len:#x} at host {gpu:#x}, offset {:#x}",
            va - gpu
        ),
        None => "; no title memory copy below it".into(),
    };
    if let Some(&(gpu, start, len)) = above {
        text.push_str(&format!(
            "; the nearest above: {start:#x}+{len:#x} at host {gpu:#x} ({} copies in all)",
            all.len()
        ));
    }
    text
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;

    /// A shader's read mark of heap slot `slot` in the bitmap of shaders
    /// that use title memory (`memory`) or of the others.
    pub(crate) fn mark_read(memory: bool, slot: u32) {
        let marks = read_marks(bitmap(memory).expect("a host GPU"));
        marks[(slot / 32) as usize].fetch_or(1 << (slot % 32), AtomicOrdering::AcqRel);
    }

    /// Records into `list` what a shader writing page `page` (from the
    /// copy's first) through a marked view of `copy` leaves in the write
    /// marks: the dword holding the page's bit set to that bit alone, in the
    /// bitmap shaders mark in (video memory, or system memory without). None
    /// where the copy's writes are not marked.
    pub(crate) fn mark_on_gpu(list: &HostList, copy: &Shadow, page: u32) -> Option<()> {
        let n = copy.marks? + page;
        let h = host()?;
        let b = bitmap(false)?;
        let (up, data) = h.upload_buffer(4)?;
        // SAFETY: a mapped upload buffer of 4 bytes; an open list, live
        // buffers, a dword inside the bitmap.
        unsafe {
            data.cast::<u32>().write_volatile(1 << (n % 32));
            let into = b.vram.as_ref().map_or(&b._buffer, |v| &v.0);
            list.list.CopyBufferRegion(into, u64::from(n / 32) * 4, &up, 0, 4);
        }
        list.keep(&up);
        Some(())
    }

    /// The title writes a constant under a range an earlier list of the
    /// submission declared as written by shaders: the refresh before the
    /// submission runs still brings it to the copy, a shader's write still
    /// goes back, and a title write made after the refresh is not undone.
    #[test]
    fn title_writes_under_a_busy_range_are_copied() {
        let p = crate::watch::tests::pool(2);
        let len = 2 * PAGE;
        register(p, p as u64, len as u64);
        let Some(copy) = find(p as u64, len as u64) else {
            unregister(p, p as u64);
            return; // no host GPU here
        };
        let title = |at: usize| (p + at) as *mut u8;
        let host = |at: usize| copy.read_host((p + at) as u64 & !3, 1)[0].to_le_bytes()[at % 4];
        copy.update(0, p, len).unwrap();
        // An earlier list declares all of it as written by shaders.
        copy.writing(0, p, len, false);
        // SAFETY: (all title reads and writes below) bytes of our pool.
        unsafe { title(PAGE + 20).write_volatile(0x55) };
        assert!(copy.update(0, p + PAGE + 16, 16).unwrap() > 0, "the title's write is copied");
        assert_eq!(host(PAGE + 20), 0x55, "the title's constant reaches the copy");
        // The shaders run: one byte written. The title writes meanwhile.
        copy.write_host((p + 10) as u64, &[0xaa]);
        // SAFETY: a byte of our pool.
        unsafe { title(PAGE + 20).write_volatile(0x66) };
        copy.copy_back(0, p, len);
        // SAFETY: bytes of our pool.
        unsafe {
            assert_eq!(title(10).read_volatile(), 0xaa, "the shader's write goes back");
            assert_eq!(title(PAGE + 20).read_volatile(), 0x66, "the title's later write stays");
        }
        unregister(p, p as u64);
    }

    /// A title buffer over a pool and, after it, a plain allocation (memory
    /// that is not the runtime's): the pool is tracked and guarded; the
    /// plain allocation's bytes are copied at each use and never guarded,
    /// so a value the title keeps there reaches the copy, and again after
    /// the title changes it.
    #[test]
    fn bytes_off_the_runtimes_memory_are_copied_at_each_use() {
        use crate::watch::tests::{MEM_WRITE_WATCH, free};
        use windows_sys::Win32::System::Memory::{
            MEM_COMMIT, MEM_RELEASE, MEM_RESERVE, MEMORY_BASIC_INFORMATION, PAGE_READONLY, PAGE_READWRITE,
            VirtualAlloc, VirtualFree, VirtualQuery,
        };
        // A pool page and a plain one right after it, in two pages found free
        // (other tests allocate meanwhile: tried again if taken).
        let (pool, plain) = (0..8)
            .find_map(|_| {
                // SAFETY: a fresh reservation, released at once; then
                // allocations where it was, freed below.
                unsafe {
                    let p = VirtualAlloc(std::ptr::null(), 2 * PAGE, MEM_RESERVE, PAGE_READWRITE) as usize;
                    VirtualFree(p as *mut _, 0, MEM_RELEASE);
                    let at = |q: usize, kind: u32| {
                        VirtualAlloc(q as *const _, PAGE, MEM_RESERVE | MEM_COMMIT | kind, PAGE_READWRITE) as usize
                    };
                    let (pool, plain) = (at(p, MEM_WRITE_WATCH), at(p + PAGE, 0));
                    if p != 0 && pool == p && plain == p + PAGE {
                        return Some((pool, plain));
                    }
                    [pool, plain].into_iter().filter(|&q| q != 0).for_each(free);
                    None
                }
            })
            .expect("two free pages");
        let len = 2 * PAGE;
        register(pool, pool as u64, len as u64);
        let Some(copy) = find(pool as u64, len as u64) else {
            unregister(pool, pool as u64);
            [pool, plain].into_iter().for_each(free);
            return; // no host GPU here
        };
        let word = (plain + 64) as *mut u32;
        let protection = |p: usize| {
            // SAFETY: plain data filled in by the query.
            let mut info: MEMORY_BASIC_INFORMATION = unsafe { std::mem::zeroed() };
            // SAFETY: a valid out pointer of the right size.
            unsafe { VirtualQuery(p as *const _, &mut info, size_of_val(&info)) };
            info.Protect
        };
        // SAFETY: a word of our plain allocation.
        unsafe { word.write_volatile(0x1234) };
        copy.update(0, pool, len).unwrap();
        assert_eq!(copy.read_host(word as u64, 1), [0x1234]);
        assert!(!copy.clean(), "never clean: part of it is not tracked");
        // SAFETY: as above.
        unsafe { word.write_volatile(0x5678) };
        copy.update(0, pool, len).unwrap();
        assert_eq!(copy.read_host(word as u64, 1), [0x5678], "the title's change reaches the copy");
        assert_eq!(protection(pool), PAGE_READONLY, "the pool is guarded");
        assert_eq!(protection(plain), PAGE_READWRITE, "the plain allocation is not");
        unregister(pool, pool as u64);
        // The pages stay allocated (as the other tests' pools): a copy some
        // page table still holds must not name memory reused meanwhile.
    }

    /// Runs of the shared buffers start a whole number of elements (and of
    /// 256 bytes) in, do not overlap, and join again when given back.
    #[test]
    fn arena_runs_are_aligned_to_their_elements_and_join_again() {
        if host().is_none() {
            return; // no host GPU here
        }
        let mut arena = Arena::default();
        let (_, a) = arena.take(100, 456).unwrap();
        let (_, b) = arena.take(5000, 16).unwrap();
        let (_, c) = arena.take(4, 4).unwrap();
        assert!(a.offset % 456 == 0 && a.offset % 256 == 0 && b.offset % 256 == 0);
        let mut runs = [a, b, c].map(|s| (s.offset, s.offset + s.len));
        runs.sort_unstable();
        assert!(runs.windows(2).all(|w| w[0].1 <= w[1].0), "{runs:?}");
        for s in [b, a, c] {
            arena.give_back(s);
        }
        assert_eq!(arena.chunks.len(), 1);
        assert_eq!(arena.chunks[0].free.len(), 1, "one free run again");
        // Larger than a buffer: one of its own.
        let (_, d) = arena.take(CHUNK + 1, 4).unwrap();
        assert_eq!((arena.chunks.len(), d.chunk), (2, 1));
    }

    /// Ranges shaders write far apart are kept apart (not the copy between
    /// them); what the GPU wrote goes back once, whichever overlapping
    /// ranges declared it, and nothing else does.
    #[test]
    fn snapshots_cover_the_declared_ranges_and_writes_go_back_once() {
        let p = crate::watch::tests::pool(4);
        let len = 4 * PAGE;
        register(p, p as u64, len as u64);
        let Some(copy) = find(p as u64, len as u64) else {
            unregister(p, p as u64);
            return; // no host GPU here
        };
        copy.update(0, p, len).unwrap();
        let bytes = |s: &Shadow| s.segments(0).iter().map(|(_, n)| n).sum::<usize>();
        copy.writing(0, p + 16, 64, false);
        copy.writing(0, p + 3 * PAGE, 100, false);
        assert_eq!(bytes(&copy), 164, "two segments");
        copy.writing(0, p + PAGE, 2 * PAGE, false);
        copy.writing(0, p + PAGE, PAGE, false);
        copy.writing(0, p + 80, 8, false);
        assert_eq!(bytes(&copy), 72 + 2 * PAGE + 100, "merged where they overlap or touch");
        // The GPU writes inside two ranges.
        copy.write_host((p + 20) as u64, &[1, 2, 3]);
        copy.write_host((p + PAGE + 5) as u64, &[9]);
        assert_eq!(copy.copy_back(0, p + 16, 64), 64, "the range's part of the piece");
        assert_eq!(copy.copy_back(0, p + PAGE, 2 * PAGE), PAGE, "the piece written");
        assert_eq!(copy.copy_back(0, p + PAGE, PAGE), 0, "compared once");
        assert_eq!(copy.copy_back(0, p + 3 * PAGE, 100), 0);
        assert_eq!(copy.copy_back(0, p + 80, 8), 0);
        // SAFETY: our pool.
        let title = |at: usize| unsafe { (at as *const u8).read_volatile() };
        assert_eq!([title(p + 20), title(p + 21), title(p + 22), title(p + PAGE + 5)], [1, 2, 3, 9]);
        assert_eq!(bytes(&copy), 0, "the snapshot goes with the last range");
        unregister(p, p as u64);
    }

    /// Pages shaders marked (`copy_back_marks`, `victory_shader::emit`
    /// "Write marks") go back by the bytes the GPU changed. The title writes
    /// a marked page after the copy took it (the guard saves its bytes from
    /// before) and a page nobody marked: the shader's byte and the title's
    /// both stay in the first; the second keeps the title's, and the copy's
    /// byte there (no mark) is not put back. The bits are cleared.
    #[test]
    fn marked_pages_go_back_by_the_bytes_the_gpu_changed() {
        let p = crate::watch::tests::pool(3);
        let len = 3 * PAGE;
        register(p, p as u64, len as u64);
        let (Some(copy), Some(b)) = (find(p as u64, len as u64), bitmap(false)) else {
            unregister(p, p as u64);
            return; // no host GPU here
        };
        let base = copy.marks.expect("tracked memory gets mark bits") as usize;
        assert_eq!(copy.mark_entry(p as u64 + PAGE as u64 + 8), Some((base as u32 + 1, 8)));
        copy.update(0, p, len).unwrap();
        copy.writing(0, p, len, true);
        assert!(copy.segments(0).is_empty(), "no snapshot");
        // The shaders write a byte in each page; the title writes pages 1
        // and 2 meanwhile.
        copy.write_host((p + 10) as u64, &[0xaa]);
        copy.write_host((p + PAGE + 20) as u64, &[0xbb]);
        copy.write_host((p + 2 * PAGE + 30) as u64, &[0xcc]);
        let title = |at: usize| (p + at) as *mut u8;
        // SAFETY: (title writes and reads below) bytes of our pool.
        unsafe {
            title(PAGE + 40).write_volatile(0x11);
            title(2 * PAGE + 50).write_volatile(0x22);
        }
        // The shaders marked pages 0 and 1.
        let bits = b.data as *mut u32;
        for n in [base, base + 1] {
            // SAFETY: a dword of the mapped bitmap.
            unsafe { *bits.add(n / 32) |= 1 << (n % 32) };
        }
        copy_back_marks(0, &WriteMarks::take_ram(), &[], false, &Default::default());
        // SAFETY: bytes of our pool.
        let read = |at: usize| unsafe { title(at).read_volatile() };
        assert_eq!(read(10), 0xaa, "the shader's byte, page 0");
        assert_eq!((read(PAGE + 20), read(PAGE + 40)), (0xbb, 0x11), "both writers' bytes, page 1");
        assert_eq!((read(2 * PAGE + 30), read(2 * PAGE + 50)), (0, 0x22), "page 2 not marked");
        // SAFETY: dwords of the mapped bitmap.
        let left: u32 = (base / 32..=(base + 1) / 32).map(|w| unsafe { *bits.add(w) }).fold(0, |a, w| a | w);
        assert_eq!(left, 0, "bits cleared");
        assert_eq!(copy.copy_back(0, p, len), 0, "nothing kept to compare");
        unregister(p, p as u64);
    }

    /// Marked pages go back one after another through one buffer for the
    /// title's bytes: a whole page the shaders filled, then the copy's last
    /// page, held in part, which the title and a shader both wrote. None of
    /// the first page's bytes reach the second.
    #[test]
    fn a_partial_marked_page_after_a_whole_one_takes_none_of_its_bytes() {
        let p = crate::watch::tests::pool(2);
        let len = PAGE + PAGE / 2;
        register(p, p as u64, len as u64);
        let (Some(copy), Some(b)) = (find(p as u64, len as u64), bitmap(false)) else {
            unregister(p, p as u64);
            return; // no host GPU here
        };
        let base = copy.marks.expect("tracked memory gets mark bits") as usize;
        copy.update(0, p, len).unwrap();
        copy.writing(0, p, len, true);
        // The shaders fill page 0 and write a byte of page 1; the title
        // writes another byte of page 1 meanwhile.
        copy.write_host(p as u64, &vec![0xee; PAGE]);
        copy.write_host((p + PAGE + 20) as u64, &[0xbb]);
        let title = |at: usize| (p + at) as *mut u8;
        // SAFETY: (title writes and reads below) bytes of our pool.
        unsafe { title(PAGE + 40).write_volatile(0x11) };
        let bits = b.data as *mut u32;
        for n in [base, base + 1] {
            // SAFETY: a dword of the mapped bitmap.
            unsafe { *bits.add(n / 32) |= 1 << (n % 32) };
        }
        copy_back_marks(0, &WriteMarks::take_ram(), &[], false, &Default::default());
        // SAFETY: bytes of our pool.
        let read = |at: usize| unsafe { title(at).read_volatile() };
        assert!((0..PAGE).all(|at| read(at) == 0xee), "page 0 went back whole");
        assert_eq!((read(PAGE + 20), read(PAGE + 40)), (0xbb, 0x11), "both writers' bytes, page 1");
        let stale = (PAGE..len).filter(|&at| at != PAGE + 20 && at != PAGE + 40).find(|&at| read(at) != 0);
        assert_eq!(stale, None, "a byte of page 1 the first page's buffer left");
        unregister(p, p as u64);
    }

    /// Marks of shaders using title memory stay out of title memory while
    /// their results are held back, and are cleared; the other shaders'
    /// marks of the same submission go back.
    #[test]
    fn marks_of_shaders_using_title_memory_are_held_back() {
        let p = crate::watch::tests::pool(2);
        let len = 2 * PAGE;
        register(p, p as u64, len as u64);
        let (Some(copy), Some(others), Some(memory)) = (find(p as u64, len as u64), bitmap(false), bitmap(true)) else {
            unregister(p, p as u64);
            return; // no host GPU here
        };
        let base = copy.marks.expect("tracked memory gets mark bits") as usize;
        copy.update(0, p, len).unwrap();
        copy.writing(0, p, len, true);
        copy.write_host((p + 10) as u64, &[0xaa]);
        copy.write_host((p + PAGE + 10) as u64, &[0xbb]);
        // Page 0 marked by another shader, page 1 by one using title memory.
        for (b, n) in [(others, base), (memory, base + 1)] {
            // SAFETY: a dword of the mapped bitmap.
            unsafe { *(b.data as *mut u32).add(n / 32) |= 1 << (n % 32) };
        }
        copy_back_marks(0, &WriteMarks::take_ram(), &[], true, &Default::default());
        // SAFETY: bytes of our pool.
        let read = |at: usize| unsafe { ((p + at) as *const u8).read_volatile() };
        assert_eq!((read(10), read(PAGE + 10)), (0xaa, 0), "page 1 held back");
        // SAFETY: a dword of the mapped bitmap.
        assert_eq!(unsafe { *(memory.data as *const u32).add((base + 1) / 32) } & 1 << ((base + 1) % 32), 0, "cleared");
        unregister(p, p as u64);
    }

    /// What a frame withheld leaves every host copy of its memory at the
    /// frame's end (run textures9's shape): in frame N a program using title
    /// memory writes a record into the copies and the frame misses, so it
    /// stays out of title memory but in the copies for the GPU; in frame
    /// N+1, which misses nothing, a program reading the record must see
    /// title memory's bytes, not frame N's, in a title buffer's copy and in
    /// an inner buffer's copy of the same pages alike.
    #[test]
    fn what_a_frame_withholds_leaves_the_copies_at_its_end() {
        let p = crate::watch::tests::pool(2);
        let len = 2 * PAGE;
        // SAFETY: a byte of our pool.
        unsafe { ((p + 16) as *mut u8).write_volatile(0x11) };
        register(p, p as u64, len as u64);
        register(p + 1, p as u64, PAGE as u64);
        let (Some(outer), Some(inner)) = (find(p as u64, len as u64), find(p as u64 + 16, 16)) else {
            unregister(p + 1, p as u64);
            unregister(p, p as u64);
            return; // no host GPU here
        };
        assert!(!Arc::ptr_eq(&outer, &inner));
        let byte = |c: &Shadow| c.read_host(p as u64 + 16, 1)[0] & 0xff;
        for c in [&outer, &inner] {
            let (start, n) = c.range();
            c.update(0, start as usize, n as usize).unwrap();
            c.writing(0, start as usize, n as usize, false);
            c.write_host(p as u64 + 16, &[0x22]);
            c.keep_writes(0, start as usize, n as usize);
        }
        crate::memory::drop_withheld(p + 16, 4);
        assert_eq!((byte(&outer), byte(&inner)), (0x22, 0x22), "frame N: the GPU's copies keep it");
        crate::memory::log_frame();
        for c in [&outer, &inner] {
            let (start, n) = c.range();
            c.update(0, start as usize, n as usize).unwrap();
        }
        assert_eq!((byte(&outer), byte(&inner)), (0x11, 0x11), "frame N+1: title memory's bytes");
        // SAFETY: as above.
        assert_eq!(unsafe { ((p + 16) as *const u8).read_volatile() }, 0x11, "title memory untouched");
        unregister(p + 1, p as u64);
        unregister(p, p as u64);
    }

    /// A partial refresh of an armed page updates its saved bytes too (both
    /// under and beside a pending marked range): copy-back neither puts the
    /// refreshed byte back over a later title write nor drops a shader write
    /// back to the pre-fault value.
    #[test]
    fn partial_refresh_under_marked_range_updates_saved_bytes() {
        let p = crate::watch::tests::pool(2);
        let len = 2 * PAGE;
        register(p, p as u64, len as u64);
        let Some(copy) = find(p as u64, len as u64) else {
            unregister(p, p as u64);
            return; // no host GPU here
        };
        let write = |at: usize, v: u8| {
            // SAFETY: a byte of our pool.
            unsafe { ((p + at) as *mut u8).write_volatile(v) };
        };
        copy.update(0, p, len).unwrap();
        // Arms both pages: page 0 is wholly busy; page 1 past +16 is free.
        copy.writing(0, p, PAGE + 16, true);
        for (at, v) in [(20, 0x55), (24, 0x44), (40, 0x11), (PAGE + 20, 0x55)] {
            write(at, v);
        }
        assert!(copy.update(0, p + 16, 16).unwrap() > 0 && copy.update(0, p + PAGE + 16, 16).unwrap() > 0);
        copy.write_host((p + 10) as u64, &[0xaa]);
        copy.write_host((p + 24) as u64, &[0x00]);
        copy.write_host((p + PAGE + 10) as u64, &[0xbb]);
        write(20, 0x66);
        write(PAGE + 20, 0x77);
        copy.copy_back_marked(0, p, &Default::default());
        copy.copy_back_marked(0, p + PAGE, &Default::default());
        // SAFETY: bytes of our pool.
        let read = |at: usize| unsafe { ((p + at) as *const u8).read_volatile() };
        assert_eq!(
            [read(10), read(20), read(24), read(40)],
            [0xaa, 0x66, 0x00, 0x11]
        );
        assert_eq!([read(PAGE + 10), read(PAGE + 20)], [0xbb, 0x77]);
        copy.copy_back(0, p, PAGE + 16);
        unregister(p, p as u64);
    }

    /// A marked page read back as the copy held it (shaders left it as it
    /// was, `crate::host`) goes back with no look at title memory: the
    /// copy's bytes are not written over title memory though they differ.
    /// Not named so, the same page goes back by what differs.
    #[test]
    fn a_marked_page_shaders_left_does_not_go_back() {
        let p = crate::watch::tests::pool(1);
        register(p, p as u64, PAGE as u64);
        let Some(copy) = find(p as u64, PAGE as u64) else {
            unregister(p, p as u64);
            return; // no host GPU here
        };
        copy.update(0, p, PAGE).unwrap();
        copy.writing(0, p, PAGE, true);
        copy.write_host((p + 8) as u64, &[0xaa]);
        // SAFETY: a byte of our pool.
        let read = || unsafe { ((p + 8) as *const u8).read_volatile() };
        let unchanged = std::collections::HashSet::from([copy.data as usize]);
        assert_eq!(copy.copy_back_marked(0, p, &unchanged), 0);
        assert_eq!(read(), 0, "title memory not written");
        assert_eq!(copy.copy_back_marked(0, p, &Default::default()), PAGE);
        assert_eq!(read(), 0xaa, "not named unchanged: the difference goes back");
        copy.copy_back(0, p, PAGE);
        unregister(p, p as u64);
    }

    /// Mark bits are handed out in runs and come back joined.
    #[test]
    fn mark_runs_are_reused_and_joined() {
        let mut m = Marks::default();
        assert_eq!((m.take(3), m.take(2)), (Some(0), Some(3)));
        m.give_back(0, 3);
        assert_eq!(m.take(1), Some(0), "first fit");
        m.give_back(3, 2);
        assert_eq!(m.free.iter().collect::<Vec<_>>(), [(&1, &4)], "3-4 joins the free 1-2");
        m.give_back(0, 1);
        assert_eq!(m.free.iter().collect::<Vec<_>>(), [(&0, &5)], "and 0 joins them");
        assert_eq!(m.take(5), Some(0));
        assert_eq!(pages(0x1_fff0, 0x20), 2, "a range across a page boundary");
        assert_eq!(pages(0x2_0000, 0x1_0000), 1);
    }

    /// A shader write into a VRAM-backed shadow that sets a `VIEW_WRITES` bit
    /// without a `Step::GpuWrite` still reaches title memory on `Host::execute`.
    #[test]
    fn vram_marked_view_writes_go_back_to_title() {
        let Some(host) = host() else {
            return; // no host GPU here
        };
        let p = crate::watch::tests::pool(1);
        register(p, p as u64, PAGE as u64);
        let Some(copy) = find(p as u64, PAGE as u64) else {
            unregister(p, p as u64);
            return;
        };
        let base = copy.marks.expect("tracked memory gets mark bits") as usize;
        let Some(b) = bitmap(false) else {
            unregister(p, p as u64);
            return;
        };
        copy.update(0, p, PAGE).unwrap();
        let list = host.new_list().unwrap();
        // The shader's write, on the GPU into the copy (with copies in video
        // memory the mirror holds what the copy held before the submission).
        let (up, up_ptr) = host.upload_buffer(8).expect("upload bit and word");
        // SAFETY: mapped 8-byte upload buffer.
        unsafe { up_ptr.cast::<[u32; 2]>().write_volatile([1u32 << (base % 32), 0x1337_c0de]) };
        list.keep(&up);
        // Set the VIEW_WRITES mark bit directly on GPU (or RAM when UMA) without Step::GpuWrite.
        let word_off = ((base / 32) * 4) as u64;
        // SAFETY: open command list and valid buffer regions.
        unsafe {
            list.list.CopyBufferRegion(&copy.buffer, copy.base, &up, 4, 4);
            if let Some((vram, _)) = &b.vram {
                list.list.CopyBufferRegion(vram, word_off, &up, 0, 4);
            } else {
                list.list.CopyBufferRegion(&b._buffer, word_off, &up, 0, 4);
            }
        }
        list.close();
        host.execute(&[&list]);
        // SAFETY: our pool page.
        let title_word = unsafe { (p as *const u32).read_volatile() };
        assert_eq!(title_word, 0x1337_c0de);
    }
    /// Overlapping `writing` calls only snapshot uncovered gaps, and a
    /// `copy_back` across multiple segments and 64 KiB chunks merges GPU writes
    /// without clobbering concurrent title writes.
    #[test]
    fn overlapping_snapshots_never_recopy_and_cross_segment_copy_back() {
        let p = crate::watch::tests::pool(2);
        let len = 2 * PAGE;
        register(p, p as u64, len as u64);
        let Some(copy) = find(p as u64, len as u64) else {
            unregister(p, p as u64);
            return; // no host GPU here
        };
        let title = |at: usize| (p + at) as *mut u8;
        // SAFETY: bytes of our pool.
        unsafe {
            title(10).write_volatile(0x01);
            title(PAGE - 4).write_volatile(0x02);
            title(PAGE + 4).write_volatile(0x03);
            title(PAGE + 100).write_volatile(0x04);
        }
        copy.update(0, p, len).unwrap();
        copy.writing(0, p, PAGE - 16, false);
        copy.writing(0, p + PAGE + 16, PAGE - 16, false);
        copy.writing(0, p + 128, 2 * PAGE - 256, false);
        copy.writing(0, p, len, false);
        copy.writing(0, p, PAGE, false);
        {
            let segs = copy.segments(0);
            for w in segs.windows(2) {
                assert!(w[0].0 + w[0].1 <= w[1].0, "segments must stay sorted and disjoint");
            }
            let total: usize = segs.iter().map(|(_, n)| n).sum();
            assert_eq!(total, len, "overlapping writing calls must not re-copy bytes");
        }
        copy.write_host((p + 10) as u64, &[0xaa]);
        copy.write_host((p + PAGE - 4) as u64, &[0xbb]);
        copy.write_host((p + PAGE + 4) as u64, &[0xcc]);
        // SAFETY: a byte of our pool.
        unsafe { title(PAGE + 100).write_volatile(0x55) };
        copy.copy_back(0, p, len);
        // SAFETY: bytes of our pool.
        let read = |at: usize| unsafe { title(at).read_volatile() };
        assert_eq!(read(10), 0xaa);
        assert_eq!(read(PAGE - 4), 0xbb);
        assert_eq!(read(PAGE + 4), 0xcc);
        assert_eq!(read(PAGE + 100), 0x55);
        // A second overlapping copy_back on a split 64 KiB piece must skip it.
        // SAFETY: a byte of our pool.
        unsafe { title(PAGE - 4).write_volatile(0x77) };
        assert_eq!(copy.copy_back(0, p, PAGE), 0);
        assert_eq!(read(PAGE - 4), 0x77);
        unregister(p, p as u64);
    }

    /// What submissions in flight write: a range read through another copy
    /// (or none) crosses them, through the same copy it does not; a
    /// bottom-of-pipe write crosses every reader; the latest submission
    /// crossing is named; a range nested in a long one is found; nothing
    /// crosses once its submission has landed.
    #[test]
    fn flight_names_the_latest_writer_of_memory_read_elsewhere() {
        let mut f = Flight::default();
        assert!(f.is_empty());
        // Submission 3 writes [0x1000, 0x9000) through copy 1 (twice, merged)
        // and a label at 0x20000; 5 writes [0x4000, 0x5000) through copy 2.
        f.fly(3, vec![(0x1000, 0x6000, 1), (0x5000, 0x9000, 1), (0x2_0000, 0x2_0008, LABEL), (0x10, 0x10, 1)]);
        f.fly(5, vec![(0x4000, 0x5000, 2)]);
        f.fly(6, Vec::new());
        assert_eq!(f.crossing(0x8000, 0x8100, 1), None, "the same copy");
        assert_eq!(f.crossing(0x8000, 0x8100, 2), Some((3, false)), "another copy");
        assert_eq!(f.crossing(0x8000, 0x8100, 0), Some((3, false)), "a title read");
        assert_eq!(f.crossing(0x4800, 0x4900, 9), Some((5, false)), "the latest");
        assert_eq!(f.crossing(0x4800, 0x4900, 2), Some((3, false)), "the latest through another copy");
        assert_eq!(f.crossing(0x9000, 0x1_0000, 2), None, "ends are exclusive");
        assert_eq!(f.crossing(0x1_fff8, 0x2_0001, 1), Some((3, true)), "a bottom-of-pipe write");
        assert_eq!(f.crossing(0x10, 0x20, 2), None, "empty ranges are dropped");
        f.land(5);
        assert_eq!(f.crossing(0x4800, 0x4900, 9), Some((3, false)));
        f.land(3);
        assert_eq!(f.crossing(0, usize::MAX, 9), None);
        assert!(f.is_empty(), "nothing in flight writes anything");
    }

    /// Two submissions in flight on one copy: each keeps its own snapshot;
    /// a refresh for the later one refuses pages the title changed under
    /// the earlier one's range (naming it) until that one went back, and
    /// takes them after; what each wrote goes back with the title's write.
    #[test]
    fn submissions_in_flight_keep_their_own_snapshots() {
        let p = crate::watch::tests::pool(2);
        let len = 2 * PAGE;
        register(p, p as u64, len as u64);
        let Some(copy) = find(p as u64, len as u64) else {
            unregister(p, p as u64);
            return; // no host GPU here
        };
        let title = |at: usize| (p + at) as *mut u8;
        copy.update(1, p, len).unwrap();
        // Submission 1 writes [p, p + 64) and marks page 1; its shader
        // writes a byte.
        copy.writing(1, p, 64, false);
        copy.writing(1, p + PAGE, PAGE, true);
        copy.write_host((p + 8) as u64, &[0xaa]);
        // Submission 2 is prepared meanwhile; the title wrote page 0.
        // SAFETY: (title writes and reads below) bytes of our pool.
        unsafe { title(200).write_volatile(0x55) };
        assert_eq!(copy.update(2, p, PAGE), Err(1), "page 0 changed under submission 1's range");
        assert_eq!(copy.update(2, p + PAGE, PAGE), Ok(0), "page 1 unchanged");
        copy.writing(2, p + 128, 64, false);
        assert_eq!(copy.segments(1), [(0, 64)]);
        assert_eq!(copy.segments(2), [(128, 64)]);
        // Submission 1 completes.
        assert_eq!(copy.copy_back(1, p, 64), 64);
        copy.copy_back(1, p + PAGE, PAGE);
        assert!(copy.segments(1).is_empty());
        assert!(copy.update(2, p, PAGE).unwrap() > 0, "now taken");
        let host = |at: usize| copy.read_host((p + at) as u64 & !3, 1)[0].to_le_bytes()[at % 4];
        assert_eq!((host(8), host(200)), (0xaa, 0x55), "the copy holds both");
        // Submission 2's shader writes; it completes.
        copy.write_host((p + 130) as u64, &[0xcc]);
        copy.copy_back(2, p + 128, 64);
        // SAFETY: as above.
        let read = |at: usize| unsafe { title(at).read_volatile() };
        assert_eq!((read(8), read(130), read(200)), (0xaa, 0xcc, 0x55));
        assert!(copy.segments(2).is_empty());
        unregister(p, p as u64);
    }

    /// A page two submissions in flight mark: the earlier one's copy-back
    /// leaves it armed for the later one, so a title write between them
    /// keeps both the title's byte and the later shaders' (unarmed, the page
    /// changed with nothing saved and the later shaders' byte was dropped).
    #[test]
    fn a_page_marked_by_a_later_submission_stays_armed() {
        let p = crate::watch::tests::pool(1);
        register(p, p as u64, PAGE as u64);
        let Some(copy) = find(p as u64, PAGE as u64) else {
            unregister(p, p as u64);
            return; // no host GPU here
        };
        if !copy.marked() {
            unregister(p, p as u64);
            return; // no mark bits left
        }
        copy.update(1, p, PAGE).unwrap();
        copy.writing(1, p, PAGE, true);
        copy.writing(2, p, PAGE, true);
        copy.write_host((p + 10) as u64, &[0xaa]);
        assert_eq!(copy.copy_back_marked(1, p, &Default::default()), PAGE);
        // SAFETY: (title writes and reads below) bytes of our pool.
        unsafe { ((p + 40) as *mut u8).write_volatile(0x11) };
        copy.write_host((p + 20) as u64, &[0xbb]);
        assert_eq!(copy.copy_back_marked(2, p, &Default::default()), PAGE);
        // SAFETY: as above.
        let read = |at: usize| unsafe { ((p + at) as *const u8).read_volatile() };
        assert_eq!((read(10), read(20), read(40)), (0xaa, 0xbb, 0x11));
        copy.copy_back(1, p, PAGE);
        copy.copy_back(2, p, PAGE);
        unregister(p, p as u64);
    }
}

/// Marked pages going back in runs ([`copy_back_run`]): run building, the
/// bytes written, and the title's writes around a run. No GPU: a tracker
/// over a pool and a host buffer stand for the copy. (They take the guard
/// lock and fault: run them by exact name in Wine, as `watch::`'s.)
#[cfg(test)]
mod run_tests {
    use super::*;
    use crate::watch::tests::{free, pool};

    fn pattern(k: usize) -> u8 {
        (k * 7 + 3) as u8
    }

    /// Fills `len` bytes of title memory at `p` with [`pattern`].
    fn fill(p: usize, len: usize) {
        // SAFETY: bytes of our pool or view, read-write (not taken yet).
        let title = unsafe { std::slice::from_raw_parts_mut(p as *mut u8, len) };
        title.iter_mut().enumerate().for_each(|(k, b)| *b = pattern(k));
    }

    fn title_bytes(p: usize, len: usize) -> Vec<u8> {
        let mut v = vec![0; len];
        assert!(crate::host::read_into(p as *const u8, &mut v));
        v
    }

    fn pages_at(p: usize, n: usize) -> Vec<usize> {
        (0..n).map(|k| p + k * PAGE).collect()
    }

    fn hook(f: impl Fn(u8, usize) + 'static) {
        RUN_HOOK.with(|h| *h.borrow_mut() = Some(Box::new(f)));
    }

    fn unhook() {
        RUN_HOOK.with(|h| *h.borrow_mut() = None);
    }

    /// Runs are consecutive whole pages of one copy, cut at `max`; other
    /// pages go alone.
    #[test]
    fn mark_runs_join_consecutive_whole_pages_of_one_copy() {
        let items = [
            (1, 0, true),
            (1, PAGE, true),
            (1, 2 * PAGE, true),
            (1, 4 * PAGE, true),
            (2, 5 * PAGE, true),
            (1, 6 * PAGE, false),
            (1, 7 * PAGE, true),
            (1, 8 * PAGE, true),
        ];
        assert_eq!(
            mark_runs(&items, 2),
            [(0..2, true), (2..3, true), (3..4, true), (4..5, true), (5..6, false), (6..8, true)]
        );
        assert_eq!(mark_runs(&items, 32)[0], (0..3, true));
        assert!(mark_runs(&[], 32).is_empty());
    }

    /// The byte runs cover every changed byte, start and end on one, and
    /// hold no aligned 8-byte word that did not change; past the cap the
    /// closest join and still cover them.
    #[test]
    fn diff_runs_cover_the_changed_bytes_by_words() {
        let old: Vec<u8> = (0..PAGE).map(pattern).collect();
        let cases: Vec<Vec<usize>> = vec![
            vec![],
            vec![0],
            vec![PAGE - 1],
            vec![5, 6, 7, 8, 9],
            vec![9, 17],
            vec![3, 40, 41, 255, 256, 257, 1000],
            (100..300).collect(),
            (0..PAGE).step_by(4).collect(),
        ];
        for changed in cases {
            let mut new = old.clone();
            for &k in &changed {
                new[k] ^= 0xff;
            }
            // Runs of another page before stay.
            let mut out = vec![(1, 2)];
            diff_runs(&new, &old, usize::MAX, &mut out);
            assert_eq!(out[0], (1, 2));
            let runs = &out[1..];
            let inside = |k: usize| runs.iter().any(|&(a, b)| (a as usize..b as usize).contains(&k));
            assert!(changed.iter().all(|&k| inside(k)), "{changed:?}: {runs:?}");
            assert!(runs.windows(2).all(|r| r[0].1 < r[1].0), "{runs:?}");
            for &(a, b) in runs {
                let (a, b) = (a as usize, b as usize);
                assert!(new[a] != old[a] && new[b - 1] != old[b - 1], "{changed:?}: {runs:?}");
                for w in (a / 8 * 8..b).step_by(8) {
                    assert!(new[w..w + 8] != old[w..w + 8], "an unchanged word in a run: {runs:?}");
                }
            }
        }
        let mut new = old.clone();
        for k in (0..PAGE).step_by(64) {
            new[k] ^= 1;
        }
        let mut out = Vec::new();
        diff_runs(&new, &old, 4, &mut out);
        assert!(out.len() <= 4, "{}", out.len());
        assert!((0..PAGE).step_by(64).all(|k| out.iter().any(|&(a, b)| (a as usize..b as usize).contains(&k))));
    }

    /// A run writes what the shaders changed, with one protection change
    /// each way, and takes its pages: guarded again, a title write counts.
    #[test]
    fn a_run_writes_back_what_the_shaders_changed_and_takes_its_pages() {
        let p = pool(4);
        let t = crate::watch::track_copy(p, 4 * PAGE, true).expect("pool memory is tracked");
        fill(p, 4 * PAGE);
        let mut host = title_bytes(p, 4 * PAGE);
        assert!(!t.take(p, p + 4 * PAGE).is_empty());
        // The shaders: bytes of page 0, one of page 1, most of 2, all of 3.
        host[100..200].fill(0xaa);
        host[PAGE + 5] ^= 1;
        host[2 * PAGE + 8..3 * PAGE - 8].fill(0x11);
        host[3 * PAGE..].fill(0x55);
        let back = copy_back_run(&t, &pages_at(p, 4), &host, &Default::default());
        assert_eq!(title_bytes(p, 4 * PAGE), host, "title memory holds the copy");
        assert_eq!((back.written, back.misses, back.merged, back.kept), (4, 0, 0, 0));
        assert_eq!(back.diff_runs, 4);
        assert_eq!(back.protects, 2, "one protection change each way for the run");
        assert!((0..4).all(|k| !t.changed(p / PAGE + k)), "taken");
        // SAFETY: a byte of our pool, guarded again: the handler opens it.
        unsafe { ((p + PAGE + 8) as *mut u8).write_volatile(9) };
        assert!(t.changed(p / PAGE + 1) && !t.changed(p / PAGE));
        drop(t);
        free(p);
    }

    /// Pages between those a run writes are opened and guarded with them
    /// (one protection change each way), checked after: one the title
    /// wrote while open stays changed, one it did not is taken.
    #[test]
    fn a_run_bridges_the_pages_it_does_not_write_and_checks_them() {
        let p = pool(4);
        let t = crate::watch::track_copy(p, 4 * PAGE, true).expect("pool memory is tracked");
        fill(p, 4 * PAGE);
        let mut host = title_bytes(p, 4 * PAGE);
        assert!(!t.take(p, p + 4 * PAGE).is_empty());
        // The shaders change pages 0 and 3; 1 and 2 are bridged. The title
        // writes page 2 while it is open.
        host[100] ^= 1;
        host[3 * PAGE + 100] ^= 1;
        hook(move |at, _| {
            if at == 1 {
                // SAFETY: a byte of our pool, open for the run.
                unsafe { ((p + 2 * PAGE + 64) as *mut u8).write_volatile(0x5c) };
            }
        });
        let back = copy_back_run(&t, &pages_at(p, 4), &host, &Default::default());
        unhook();
        assert_eq!((back.written, back.bridged, back.misses, back.protects), (2, 2, 1, 2));
        let title = title_bytes(p, 4 * PAGE);
        assert_eq!((title[100], title[3 * PAGE + 100], title[2 * PAGE + 64]), (host[100], host[3 * PAGE + 100], 0x5c));
        assert_eq!([0, 1, 2, 3].map(|k| t.changed(p / PAGE + k)), [false, false, true, false]);
        // Guarded again, the bridged page counts a write.
        // SAFETY: a byte of our pool.
        unsafe { ((p + PAGE + 8) as *mut u8).write_volatile(9) };
        assert!(t.changed(p / PAGE + 1));
        drop(t);
        free(p);
    }

    /// A page another tracker holds too is not bridged: a title write there
    /// must fault, for that tracker.
    #[test]
    fn a_page_another_tracker_holds_is_not_bridged() {
        let p = pool(3);
        let t = crate::watch::track_copy(p, 3 * PAGE, true).expect("pool memory is tracked");
        let other = crate::watch::track(p + PAGE, PAGE, true).expect("tracked");
        fill(p, 3 * PAGE);
        let mut host = title_bytes(p, 3 * PAGE);
        assert!(!t.take(p, p + 3 * PAGE).is_empty() && !other.take(p, p + 3 * PAGE).is_empty());
        host[100] ^= 1;
        host[2 * PAGE + 100] ^= 1;
        let back = copy_back_run(&t, &pages_at(p, 3), &host, &Default::default());
        assert_eq!((back.written, back.bridged, back.protects), (2, 0, 4));
        assert!(other.clean(), "the other tracker's page stayed guarded");
        // SAFETY: a byte of our pool, guarded: the handler opens it.
        unsafe { ((p + PAGE + 8) as *mut u8).write_volatile(9) };
        assert!(!other.clean() && t.changed(p / PAGE + 1));
        drop((t, other));
        free(p);
    }

    /// Pages the shaders left as they were go back unread and are taken,
    /// with no protection change.
    #[test]
    fn a_run_of_pages_the_shaders_left_changes_no_protection() {
        let p = pool(2);
        let t = crate::watch::track_copy(p, 2 * PAGE, true).expect("pool memory is tracked");
        fill(p, 2 * PAGE);
        let host = title_bytes(p, 2 * PAGE);
        assert!(!t.take(p, p + 2 * PAGE).is_empty());
        let unchanged = [host.as_ptr() as usize, host[PAGE..].as_ptr() as usize].into_iter().collect();
        let back = copy_back_run(&t, &pages_at(p, 2), &host, &unchanged);
        assert_eq!((back.written, back.protects), (0, 0));
        assert!(!t.changed(p / PAGE) && !t.changed(p / PAGE + 1));
        drop(t);
        free(p);
    }

    /// A title write landing on a page the run opened (no fault), before
    /// our writes, stays: only the shaders' bytes are written, and the page
    /// stays changed, so the next refresh takes the title's byte.
    #[test]
    fn a_title_write_while_a_run_is_open_stays_and_leaves_its_page_changed() {
        let p = pool(2);
        let t = crate::watch::track_copy(p, 2 * PAGE, true).expect("pool memory is tracked");
        fill(p, 2 * PAGE);
        let mut host = title_bytes(p, 2 * PAGE);
        assert!(!t.take(p, p + 2 * PAGE).is_empty());
        host[100..200].fill(0xaa);
        host[PAGE + 300] ^= 1;
        hook(move |at, _| {
            if at == 1 {
                // SAFETY: a byte of our pool, open for the run.
                unsafe { ((p + 5000) as *mut u8).write_volatile(0x5c) };
            }
        });
        let back = copy_back_run(&t, &pages_at(p, 2), &host, &Default::default());
        unhook();
        let title = title_bytes(p, 2 * PAGE);
        assert_eq!(title[5000], 0x5c, "the title's write stays");
        assert_eq!(&title[100..200], &host[100..200], "the shaders' bytes went back");
        assert_eq!(&title[PAGE..], &host[PAGE..]);
        assert_eq!((back.written, back.misses), (2, 1));
        assert!(t.changed(p / PAGE) && !t.changed(p / PAGE + 1), "only the page the title wrote stays changed");
        drop(t);
        free(p);
    }

    /// A title write after the run read the page and before its lease (it
    /// faults, and the page was armed, so its bytes were saved) turns the
    /// page into one the title wrote: the shaders' bytes go back by the
    /// saved bytes, the title's stays, and the page stays changed. Not
    /// armed, nothing was saved: the page keeps the title's bytes.
    #[test]
    fn a_title_write_between_the_read_and_the_lease_goes_back_by_saved_bytes() {
        for armed in [true, false] {
            let p = pool(2);
            let t = crate::watch::track_copy(p, 2 * PAGE, true).expect("pool memory is tracked");
            fill(p, 2 * PAGE);
            let mut host = title_bytes(p, 2 * PAGE);
            assert!(!t.take(p, p + 2 * PAGE).is_empty());
            if armed {
                t.arm(p, p + 2 * PAGE);
            }
            host[100..200].fill(0xaa);
            hook(move |at, _| {
                if at == 0 {
                    // SAFETY: a byte of our pool, guarded: the handler opens it.
                    unsafe { ((p + 300) as *mut u8).write_volatile(0x5c) };
                }
            });
            let back = copy_back_run(&t, &pages_at(p, 2), &host, &Default::default());
            unhook();
            let title = title_bytes(p, 2 * PAGE);
            assert_eq!(title[300], 0x5c, "the title's write stays");
            if armed {
                assert_eq!((back.merged, back.kept, back.written), (1, 0, 1));
                assert_eq!(&title[100..200], &host[100..200], "the shaders' bytes went back");
            } else {
                assert_eq!((back.merged, back.kept, back.written), (0, 1, 0));
                assert!((100..200).all(|k| title[k] == pattern(k)), "the title's page as it was");
            }
            assert!(t.changed(p / PAGE) && !t.changed(p / PAGE + 1));
            drop(t);
            free(p);
        }
    }

    /// A title write through another mapping of a page the run writes
    /// faults (that mapping stays guarded) and waits for the lease: it
    /// lands after ours, on the very bytes the shaders changed, and the page
    /// counts as changed again after the run took it.
    #[test]
    fn a_title_write_faulting_under_the_lease_lands_after_ours() {
        let (a, b) = crate::watch::tests::twin_views(1);
        let t = crate::watch::track_copy(a, PAGE, true).expect("views are tracked");
        fill(a, PAGE);
        let mut host = title_bytes(a, PAGE);
        assert!(!t.take(a, a + PAGE).is_empty());
        host[100..108].fill(0xaa);
        let writer: Arc<Mutex<Option<std::thread::JoinHandle<()>>>> = Arc::default();
        let slot = writer.clone();
        hook(move |at, _| {
            if at == 1 {
                *slot.lock().unwrap() = Some(std::thread::spawn(move || {
                    // SAFETY: a byte of the other view, guarded: the
                    // handler waits for the lease, then opens it.
                    unsafe { ((b + 100) as *mut u8).write_volatile(0x5c) };
                }));
                std::thread::sleep(std::time::Duration::from_millis(200));
            }
        });
        let back = copy_back_run(&t, &[a], &host, &Default::default());
        unhook();
        writer.lock().unwrap().take().expect("the writer ran").join().unwrap();
        assert_eq!((back.written, back.misses), (1, 0));
        let title = title_bytes(a, PAGE);
        assert_eq!(title[100], 0x5c, "the title's write lands last");
        assert_eq!(&title[101..108], &host[101..108], "the shaders' other bytes went back");
        assert!(t.changed(a / PAGE), "the page changed after the run took it");
    }

    /// Three views of one fresh section of `pages` pages: the title's
    /// (`a`), another title view reported as showing the same pages (`b`),
    /// and the runtime's own (`c`), which copy-back is told to write
    /// through for `a`'s pages.
    fn three_views(pages: usize) -> (usize, usize, usize) {
        use std::sync::atomic::{AtomicUsize, Ordering::Relaxed};
        use windows_sys::Win32::System::Memory::{CreateFileMappingW, FILE_MAP_WRITE, MapViewOfFile, PAGE_READWRITE};
        static AT: [AtomicUsize; 3] = [const { AtomicUsize::new(0) }; 3];
        let len = pages * PAGE;
        // SAFETY: a pagefile section mapped three times; the test keeps them.
        let [a, b, c] = unsafe {
            let section = CreateFileMappingW(-1isize as _, std::ptr::null(), PAGE_READWRITE, 0, len as u32, std::ptr::null());
            assert!(!section.is_null());
            [(); 3].map(|()| MapViewOfFile(section, FILE_MAP_WRITE, 0, 0, len).Value as usize)
        };
        assert!(a != 0 && b != 0 && c != 0);
        for k in (0..len).step_by(PAGE) {
            crate::watch::title_memory_aliased(b + k, &[a + k]);
        }
        AT[0].store(a, Relaxed);
        AT[1].store(len, Relaxed);
        AT[2].store(c, Relaxed);
        crate::watch::set_writable_alias(|at| {
            let (a, len, c) = (AT[0].load(Relaxed), AT[1].load(Relaxed), AT[2].load(Relaxed));
            (a..a + len).contains(&at).then(|| at - a + c)
        });
        (a, b, c)
    }

    /// Pages the runtime can map for us go back through its own mapping:
    /// no protection change, the title's mappings stay guarded (a later
    /// write there counts), and nothing is checked after.
    #[test]
    fn a_run_writes_view_pages_through_our_own_mapping() {
        let (a, b, _) = three_views(3);
        let t = crate::watch::track_copy(a, 3 * PAGE, true).expect("views are tracked");
        fill(a, 3 * PAGE);
        let mut host = title_bytes(a, 3 * PAGE);
        assert!(!t.take(a, a + 3 * PAGE).is_empty());
        host[100..200].fill(0xaa);
        host[2 * PAGE + 7] ^= 1;
        let back = copy_back_run(&t, &pages_at(a, 3), &host, &Default::default());
        assert_eq!((back.written, back.through, back.protects, back.misses), (2, 2, 0, 0));
        assert_eq!(title_bytes(a, 3 * PAGE), host, "title memory holds the copy");
        assert!((0..3).all(|k| !t.changed(a / PAGE + k)), "taken");
        // SAFETY: a byte of the title's other view, still guarded.
        unsafe { ((b + 2 * PAGE + 8) as *mut u8).write_volatile(9) };
        assert!(t.changed(a / PAGE + 2) && !t.changed(a / PAGE));
        // SAFETY: a byte of the title's view, still guarded.
        unsafe { ((a + 8) as *mut u8).write_volatile(9) };
        assert!(t.changed(a / PAGE));
    }

    /// A title write while a run writes through our own mapping faults (the
    /// title's mappings stay guarded) and waits for the lease: it lands
    /// after ours, on the very bytes the shaders changed, and the page
    /// counts as changed after the run took it. Through the other title
    /// mapping too.
    #[test]
    fn a_title_write_during_a_write_through_our_mapping_lands_after_it() {
        let (a, b, _) = three_views(2);
        let t = crate::watch::track_copy(a, 2 * PAGE, true).expect("views are tracked");
        fill(a, 2 * PAGE);
        let mut host = title_bytes(a, 2 * PAGE);
        assert!(!t.take(a, a + 2 * PAGE).is_empty());
        host[100..108].fill(0xaa);
        host[PAGE + 100..PAGE + 108].fill(0xbb);
        let writers: Arc<Mutex<Vec<std::thread::JoinHandle<()>>>> = Arc::default();
        let slot = writers.clone();
        hook(move |at, _| {
            if at == 1 {
                let mut w = slot.lock().unwrap();
                // SAFETY: a byte of a title view, guarded: the handler
                // waits for the lease, then opens it.
                w.push(std::thread::spawn(move || unsafe { ((a + 100) as *mut u8).write_volatile(0x5c) }));
                // SAFETY: a byte of the same view's other mapping, guarded
                // the same way.
                w.push(std::thread::spawn(move || unsafe { ((b + PAGE + 100) as *mut u8).write_volatile(0x5d) }));
                std::thread::sleep(std::time::Duration::from_millis(200));
            }
        });
        let back = copy_back_run(&t, &pages_at(a, 2), &host, &Default::default());
        unhook();
        writers.lock().unwrap().drain(..).for_each(|w| w.join().unwrap());
        assert_eq!((back.written, back.through, back.protects, back.misses), (2, 2, 0, 0));
        let title = title_bytes(a, 2 * PAGE);
        assert_eq!((title[100], title[PAGE + 100]), (0x5c, 0x5d), "the title's writes land last");
        assert_eq!((&title[101..108], &title[PAGE + 101..PAGE + 108]), (&host[101..108], &host[PAGE + 101..PAGE + 108]));
        assert!(t.changed(a / PAGE) && t.changed(a / PAGE + 1), "both pages changed after the run took them");
    }
}
