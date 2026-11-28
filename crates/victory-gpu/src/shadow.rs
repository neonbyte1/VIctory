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
use std::sync::atomic::{AtomicU64, Ordering};
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

/// A title buffer's host copy.
pub(crate) struct Shadow {
    pub(crate) buffer: ID3D12Resource,
    data: *mut u8,
    gpu: u64,
    /// Where the title buffer starts and how long it is.
    start: u64,
    len: u64,
    /// The pages changed since the copy took them (none: the memory is not
    /// the runtime's, copied whole each time).
    tracked: Option<crate::watch::Tracked>,
    /// Ranges shaders of a submission not yet copied back may be writing:
    /// title memory there is older than the copy.
    pending: Mutex<crate::fx::Set<(usize, usize)>>,
    /// The copy's bytes before the submission ran where its shaders write:
    /// what the GPU wrote differs. Disjoint segments (offset, bytes) in
    /// order, and the 64 KiB pieces (by title address) already compared.
    before: Mutex<Snapshot>,
    /// Title ranges `[from, to)` whose copies [`Shadow::copy_queued`]
    /// queued, for [`Shadow::queued_made`] (tracked memory only).
    queued: Mutex<Vec<(usize, usize)>>,
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

/// Snapshot segments (offset, bytes) and the pieces compared.
type Snapshot = (Vec<(usize, Vec<u8>)>, crate::fx::Set<usize>);

/// Snapshot memory kept for reuse (already touched, so no page faults).
static POOL: Mutex<Vec<Vec<u8>>> = Mutex::new(Vec::new());

/// Capacity of idle snapshots retained in the pool, in bytes.
#[cfg(feature = "diagnostics")]
pub(crate) fn idle_snapshot_bytes() -> Option<usize> {
    let pool = POOL.try_lock().ok()?;
    Some(pool.iter().map(Vec::capacity).sum())
}

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
// written on the submitting thread, and by the refresh workers only inside
// `crate::copy_pool::run`, which the flush waits for before going on.
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
    }

    /// Marks the page of title address `address` as written through a view
    /// of the copy, as shaders do (tests).
    #[cfg(test)]
    pub(crate) fn mark_written(&self, address: u64) {
        if let (Some((bit, _)), Some(b)) = (self.mark_entry(address), bitmap(false)) {
            // SAFETY: a dword of the mapped bitmap; no submission runs.
            unsafe { *(b.data as *mut u32).add(bit as usize / 32) |= 1 << (bit % 32) };
        }
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
        let written = {
            let _writing = crate::watch::writing(from, n);
            crate::host::write_mapped(from as *mut u8, host.as_ptr(), n)
        };
        // The writing guard marked this page changed. Clear that mark only
        // when the copy and title memory are equal after the write.
        if written == n && let Some(t) = &self.tracked {
            t.taken(&[page / PAGE]);
        }
        written
    }

    /// Brings `len` bytes at `address` up to date when `list` is submitted,
    /// and keeps the copy alive until the list is reset.
    pub(crate) fn refresh(self: &Arc<Self>, list: &HostList, address: u64, len: u64) {
        let len = len.min(self.start + self.len - address);
        list.defer_refresh(self, address as usize, len as usize);
    }

    fn new(buffer: ID3D12Resource, data: *mut u8, gpu: u64, start: u64, len: u64) -> Arc<Shadow> {
        Shadow::new_at(buffer, data, gpu, start, len, None)
    }

    /// [`Shadow::new`] of a copy in a run of a shared buffer (`data` and
    /// `gpu` are the run's).
    fn new_at(buffer: ID3D12Resource, data: *mut u8, gpu: u64, start: u64, len: u64, slice: Option<Slice>) -> Arc<Shadow> {
        let tracked = crate::watch::track_copy(start as usize, len as usize, true);
        let marks = tracked.as_ref().and_then(|_| marks().take(pages(start, len)));
        let shadow = Arc::new(Shadow {
            buffer,
            data,
            gpu,
            start,
            len,
            tracked,
            pending: Mutex::default(),
            before: Mutex::default(),
            queued: Mutex::default(),
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

    /// Whether every page of this copy has a mark bit below the bitmap's last
    /// dword (which holds the overflow sink bit and is cleared whole in
    /// [`copy_back_marks`]).
    pub(crate) fn marks_complete(&self) -> bool {
        self.marks
            .is_some_and(|base| marks_below_last_word(base, pages(self.start, self.len)))
    }

    /// Marks the 64 KiB title page at `page` in the written-pages bitmap
    /// (`memory`: shaders using title memory), as a shader write does (tests).
    #[cfg(test)]
    pub(crate) fn mark_page_for_test(&self, page: usize, memory: bool) {
        let (bit, _) = self
            .mark_entry(page as u64)
            .expect("tracked copy with mark bits");
        let b = bitmap(memory).expect("written-pages bitmap");
        // SAFETY: `bit` is a valid mark bit inside the mapped bitmap.
        unsafe { *(b.data as *mut u32).add((bit / 32) as usize) |= 1 << (bit % 32) };
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

    /// At submission, after the range's refresh: a submitted list's shaders
    /// write this range. Its bytes are kept to tell later what the GPU wrote:
    /// per copy and submission, segments covering the ranges the submission
    /// declares (the world declares overlapping views of its 174 MB heap
    /// dozens of times; one segment over all of them copied most of the
    /// heap), in reused memory (fresh snapshots cost ~2.3 million page
    /// faults per world frame).
    ///
    /// With `marked`, nothing is kept: the pages are armed instead, so that
    /// a title write there before the copy-back keeps the page's bytes from
    /// before it ([`crate::watch::Tracked::arm`]).
    pub(crate) fn writing(&self, address: usize, len: usize, marked: bool) {
        self.pending
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .insert((address, address + len));
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
        let mut before = self.before.lock().unwrap_or_else(|e| e.into_inner());
        let segments = &mut before.0;
        // The segments the range overlaps or touches.
        let first = segments.partition_point(|(at, bytes)| at + bytes.len() < from);
        let last = first + segments[first..].iter().take_while(|(at, _)| *at <= to).count();
        if let [(at, bytes)] = &segments[first..last]
            && *at <= from
            && to <= at + bytes.len()
        {
            return;
        }
        let start = segments.get(first).filter(|_| first < last).map_or(from, |s| s.0.min(from));
        let end = segments[first..last].last().map_or(to, |(at, bytes)| (at + bytes.len()).max(to));
        #[cfg(feature = "diagnostics")]
        let _timed = crate::watch::Timed(std::time::Instant::now(), crate::copystat::C::SnapshotUs);
        #[cfg(feature = "diagnostics")]
        crate::copystat::add(crate::copystat::C::SnapshotBytes, (end - start) as u64);
        #[cfg(feature = "diagnostics")]
        crate::copystat::snapshot(self.start, self.len, (end - start) as u64);
        // Parts newly covered from the copy, the rest as it was taken.
        let mut grown = pooled(end - start);
        let mut at = start;
        for (taken, bytes) in segments.drain(first..last) {
            grown.extend_from_slice(host(at, taken));
            grown.extend_from_slice(&bytes);
            at = taken + bytes.len();
            give_back(bytes);
        }
        grown.extend_from_slice(host(at, end));
        segments.insert(first, (start, grown));
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

    /// After the submission ran: copies the bytes its shaders wrote back
    /// into title memory (unmapped pages left alone), and only those: the
    /// title's threads write the same buffers while the GPU runs (its
    /// command memory shares the big GPU heap with what shaders write), and
    /// putting back the copy's older bytes there broke the title's own
    /// records (a fence list read as -1, title+0x4e9b54a). Returns the bytes
    /// written.
    pub(crate) fn copy_back(&self, address: usize, len: usize) -> usize {
        self.copy_back_or_not(address, len, true)
    }

    /// After the submission ran: what its shaders wrote in the range stays
    /// in the copy and does not go into title memory (they computed it
    /// from memory with no copy, `crate::memory`).
    pub(crate) fn keep_writes(&self, address: usize, len: usize) {
        self.copy_back_or_not(address, len, false);
    }

    fn copy_back_or_not(&self, address: usize, len: usize, back: bool) -> usize {
        let mut snapshot = self.before.lock().unwrap_or_else(|e| e.into_inner());
        let (segments, compared) = &mut *snapshot;
        let mut written = 0;
        let (from, to) = self.inside(address, len);
        let start = self.start as usize;
        if let Some((taken, before)) = segments
            .iter_mut()
            .find(|(at, bytes)| *at <= from && to <= at + bytes.len())
            && back
            && from < to
        {
            let before = &mut before[from - *taken..to - *taken];
            #[cfg(feature = "diagnostics")]
            let _timed = crate::watch::Timed(std::time::Instant::now(), crate::copystat::C::CompareUs);
            // By 64 KiB piece: the copy compared with its bytes before the
            // submission, title memory read only where the GPU wrote.
            const PIECE: usize = 64 << 10;
            let mut title = vec![0u8; PIECE];
            let mut synced = Vec::new();
            let mut at = from;
            while at < to {
                let n = (PIECE - (start + at) % PIECE).min(to - at);
                // SAFETY: inside the mapping; only the submitting thread
                // writes it.
                let host = unsafe { std::slice::from_raw_parts_mut(self.data.add(at), n) };
                let old = &mut before[at - from..at - from + n];
                let title = &mut title[..n];
                // A whole piece is compared once per submission, whichever
                // declared ranges overlap it: then what the GPU wrote there
                // went back.
                if n == PIECE && !compared.insert(start + at) {
                    at += n;
                    continue;
                }
                #[cfg(feature = "diagnostics")]
                crate::copystat::add(crate::copystat::C::CompareBytes, n as u64);
                if host != old && crate::host::read_into((start + at) as *const u8, title) {
                    // Title memory with the GPU's bytes put in; the copy
                    // takes the title's bytes in turn.
                    let synchronized = if merge_writeback(title, host, old) {
                        let _writing = crate::watch::writing(start + at, n);
                        let stored = crate::host::write_mapped((start + at) as *mut u8, title.as_ptr(), n);
                        written += stored;
                        stored == n
                    } else {
                        true
                    };
                    if synchronized {
                        synced.push((start + at) / PAGE);
                    }
                }
                at += n;
            }
            // Pages made equal to title memory are current in the copy, so
            // our own write is not taken for the title's; others keep what
            // they had (a title write there shows at the next refresh).
            let (first, last) = ((start + from).div_ceil(PAGE), (start + to) / PAGE);
            if let Some(t) = &self.tracked {
                let whole: Vec<usize> = synced.into_iter().filter(|p| (first..last).contains(p)).collect();
                t.taken(&whole);
            }
        }
        // Every submitted step that declared the range is done with it (one
        // submission runs at a time), and the snapshot with the last range.
        let mut pending = self.pending.lock().unwrap_or_else(|e| e.into_inner());
        pending.remove(&(address, address + len));
        if pending.is_empty() {
            compared.clear();
            for (_, bytes) in segments.drain(..) {
                give_back(bytes);
            }
            if let Some(t) = &self.tracked {
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
    /// Returns the bytes written.
    // shortcut: a page the title wrote that was not armed (a write through
    // the heap's views no list declared) keeps the title's bytes, losing
    // the shaders' there (counted).
    pub(crate) fn copy_back_marked(&self, page: usize) -> usize {
        let Some(t) = &self.tracked else {
            return 0;
        };
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
        let mut title = vec![0u8; n];
        if !crate::host::read_into(from as *const u8, &mut title) {
            return 0;
        }
        let written = match &saved {
            // The title's bytes are the copy's from before: what differs is
            // the shaders'.
            None => (*host != title[..]).then_some(host.as_ptr()),
            Some(before) => {
                let before = &before[from - page..to - page];
                merge_writeback(&mut title, host, before).then_some(title.as_ptr())
            }
        };
        let delivered = if let Some(bytes) = written {
            let _writing = crate::watch::writing(from, n);
            crate::host::write_mapped(from as *mut u8, bytes, n)
        } else {
            0
        };
        // Only a complete delivery makes the whole copy current.
        if (written.is_none() || delivered == n) && from == page && to == page + PAGE {
            t.taken(&[page / PAGE]);
        }
        // So is the snapshot of declared ranges over the page: their
        // copy-backs must not take the title's bytes for the GPU's.
        self.sync_snapshot(from - start, host);
        delivered
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

    /// Advances any snapshot segments covering `[offset, offset + data.len())`
    /// to `data`, so a later declared copy-back neither takes those bytes for
    /// the GPU's nor puts them back over a later title write.
    fn sync_snapshot(&self, offset: usize, data: &[u8]) {
        let mut before = self.before.lock().unwrap_or_else(|e| e.into_inner());
        let end = offset + data.len();
        for (at, bytes) in before.0.iter_mut() {
            let (s, e) = ((*at).max(offset), (*at + bytes.len()).min(end));
            if s < e {
                bytes[s - *at..e - *at].copy_from_slice(&data[s - offset..e - offset]);
            }
        }
    }

    /// At submission: copies what changed of `[address, address + len)`
    /// since it was last copied (all of it if the memory is not
    /// write-watched). Returns the bytes copied.
    pub(crate) fn update(&self, address: usize, len: usize) -> usize {
        if self.clean() {
            return 0;
        }
        let (start, end) = (self.start as usize, (self.start + self.len) as usize);
        let (from, to) = (address.max(start), (address + len).min(end));
        if from >= to {
            return 0;
        }
        let mut busy: Vec<(usize, usize)> = self
            .pending
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .iter()
            .copied()
            .filter(|&(a, e)| a < to && e > from)
            .collect();
        if self.tracked.is_some() {
            return self.update_tracked(from, to, !busy.is_empty());
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
            return bytes;
        }
        self.update_untracked(from, to)
    }

    /// [`Shadow::update`] of `[from, to)` (inside the buffer) where the
    /// memory is tracked: the pages the title wrote since they were taken,
    /// in one [`crate::watch::Tracked::take`], so that a page wholly in the
    /// range is guarded even where shaders write part of it (split at those
    /// ranges it stayed changed, and [`Shadow::copy_back_marked`] dropped
    /// what they wrote).
    ///
    /// `busy`: a list submitted earlier declared that shaders write in the
    /// range. A submission waits for the one before it, so none of this
    /// submission's shaders has run yet: the pages the title wrote are
    /// copied as anywhere else. Skipped, the title's constants there stayed
    /// stale for the whole submission (a dispatch read its sizes as zeros
    /// and looped until the device was removed). The same bytes go into the
    /// snapshot `writing` took, so copy-back neither takes them for the
    /// GPU's nor puts them back over a later title write.
    fn update_tracked(&self, from: usize, to: usize, busy: bool) -> usize {
        let Some(t) = &self.tracked else {
            return 0;
        };
        #[cfg(feature = "diagnostics")]
        let started = std::time::Instant::now();
        let (runs, unmapped) = t.take_split(from, to);
        #[cfg(feature = "diagnostics")]
        crate::copystat::add(crate::copystat::C::WatchUs, started.elapsed().as_micros() as u64);
        let start = self.start as usize;
        let mut before = busy.then(|| self.before.lock().unwrap_or_else(|e| e.into_inner()));
        let sync_before = |before: Option<&mut Snapshot>, a: usize, b: usize| {
            let Some(before) = before else { return };
            let (a, b) = (a - start, b - start);
            for (at, bytes) in before.0.iter_mut() {
                let (s, e) = ((*at).max(a), (*at + bytes.len()).min(b));
                if s < e {
                    // SAFETY: inside the mapping.
                    let host = unsafe { std::slice::from_raw_parts(self.data.add(s), e - s) };
                    bytes[s - *at..e - *at].copy_from_slice(host);
                }
            }
        };
        // A snapshot over the run takes the copied bytes at once; with none
        // there (marked ranges keep none) the copy may be queued.
        let refresh = |before: Option<&mut Snapshot>, a: usize, b: usize| -> usize {
            let (ra, rb) = (a - start, b - start);
            match before {
                Some(before)
                    if before.0.iter().any(|(at, bytes)| *at < rb && ra < *at + bytes.len()) =>
                {
                    let copied = self.copy(a, b);
                    sync_before(Some(before), a, b);
                    copied
                }
                _ => self.copy_queued(a, b),
            }
        };
        let mut copied = 0;
        for (a, b) in runs {
            copied += refresh(before.as_deref_mut(), a, b);
        }
        for (a, b) in unmapped {
            // SAFETY: inside the mapping (`take_split` cuts to the buffer).
            unsafe { self.data.add(a - start).write_bytes(0, b - a) };
            sync_before(before.as_deref_mut(), a, b);
            #[cfg(feature = "diagnostics")]
            crate::copystat::add(crate::copystat::C::Unmapped, (b - a) as u64);
        }
        // Parts on memory that is not the runtime's: whole, as below.
        for (a, b) in t.untracked(from, to) {
            #[cfg(feature = "diagnostics")]
            let started = std::time::Instant::now();
            copied += refresh(before.as_deref_mut(), a, b);
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
        copied
    }

    /// [`Shadow::update`] of untracked `[from, to)` (inside the buffer).
    fn update_untracked(&self, from: usize, to: usize) -> usize {
        #[cfg(feature = "diagnostics")]
        let started = std::time::Instant::now();
        self.copy_queued(from, to);
        #[cfg(feature = "diagnostics")]
        {
            use crate::copystat::{C, add};
            add(C::WholeBytes, (to - from) as u64);
            add(C::WholeUs, started.elapsed().as_micros() as u64);
        }
        to - from
    }

    /// As [`Shadow::copy`], queued where the caller collects a
    /// submission's refreshes ([`crate::copy_pool::queued`]): for copies
    /// nothing reads back before they are made. What a queued copy took
    /// goes into the saved bytes of armed pages once it is made
    /// ([`Shadow::queued_made`]).
    fn copy_queued(&self, from: usize, to: usize) -> usize {
        let (start, end) = (self.start as usize, (self.start + self.len) as usize);
        let (from, to) = (from.max(start), to.min(end));
        // SAFETY: inside the mapping (`from` and `to` are within the buffer).
        let at = unsafe { self.data.add(from - start) };
        if !crate::copy_pool::copy(at, from as *const u8, to - from) {
            if let Some(t) = &self.tracked {
                // SAFETY: `to - from` bytes inside the mapping, just copied.
                t.refreshed(from, unsafe { std::slice::from_raw_parts(at, to - from) });
            }
        } else if self.tracked.is_some() {
            self.queued.lock().unwrap_or_else(|e| e.into_inner()).push((from, to));
        }
        to - from
    }

    /// After [`crate::copy_pool::run`] made the copies
    /// [`Shadow::copy_queued`] queued: what they took goes into the saved
    /// bytes of the armed pages they cover, as [`Shadow::copy`]'s does.
    pub(crate) fn queued_made(&self) {
        let Some(t) = &self.tracked else {
            return;
        };
        let made = std::mem::take(&mut *self.queued.lock().unwrap_or_else(|e| e.into_inner()));
        let start = self.start as usize;
        for (from, to) in made {
            // SAFETY: `to - from` bytes inside the mapping, copied.
            let bytes =
                unsafe { std::slice::from_raw_parts(self.data.add(from - start), to - from) };
            t.refreshed(from, bytes);
        }
    }

    /// The stale check of this copy ([`check_stale`]): every page its
    /// tracker calls unchanged and that it holds in tracked memory is
    /// compared with title memory. Passed by: pages the title writes
    /// meanwhile (a guard fault marks them before the write lands, so a
    /// difference is asked about again after the compare), pages with
    /// nothing mapped, and pages whose withheld shader writes stay in the
    /// copies until the frame's end
    /// (`crate::memory::withheld_until_frame_end`). Returns the pages
    /// compared and, per page that differed, its address and the bytes that
    /// did; with `fix` such pages are copied again.
    #[cfg(feature = "diagnostics")]
    pub(crate) fn stale_pages(&self, fix: bool) -> (u64, Vec<(usize, usize)>) {
        let Some(t) = &self.tracked else {
            return (0, Vec::new());
        };
        if !self
            .pending
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .is_empty()
        {
            return (0, Vec::new()); // shaders may be writing it
        }
        let (start, end) = (self.start as usize, (self.start + self.len) as usize);
        let (mut compared, mut differ) = (0, Vec::new());
        let mut title = vec![0u8; PAGE];
        for page in start / PAGE..end.div_ceil(PAGE) {
            let (from, to) = ((page * PAGE).max(start), ((page + 1) * PAGE).min(end));
            if t.changed(page)
                || !t.untracked(from, to).is_empty()
                || crate::memory::withheld_until_frame_end(from, to)
            {
                continue;
            }
            let title = &mut title[..to - from];
            if !crate::host::read_into(from as *const u8, title) {
                continue;
            }
            compared += 1;
            // SAFETY: inside the mapping; only the submitting thread writes it.
            let ours =
                unsafe { std::slice::from_raw_parts(self.data.add(from - start), to - from) };
            if ours == &title[..] || t.changed(page) {
                continue;
            }
            differ.push((
                from,
                ours.iter()
                    .zip(title.iter())
                    .filter(|(a, b)| a != b)
                    .count(),
            ));
            if fix {
                self.copy(from, to);
            }
        }
        (compared, differ)
    }

    /// Copies title memory `[from, to)`, cut to the buffer, into the copy.
    fn copy(&self, from: usize, to: usize) -> usize {
        let (start, end) = (self.start as usize, (self.start + self.len) as usize);
        let (from, to) = (from.max(start), to.min(end));
        // SAFETY: inside the mapping (`from` and `to` are within the buffer).
        let at = unsafe { self.data.add(from - start) };
        crate::host::copy_mapped(at, from as *const u8, to - from, false);
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

/// Whether `page_count` consecutive mark bits starting at `base` all lie below
/// the written-pages bitmap's last dword (which is cleared whole because of the
/// overflow sink bit).
fn marks_below_last_word(base: u32, page_count: u32) -> bool {
    u64::from(base) + u64::from(page_count) <= u64::from(victory_shader::emit::VIEW_WRITE_BITS - 32)
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
/// as dwords) and host GPU address.
struct Bitmap {
    _buffer: ID3D12Resource,
    data: usize,
    gpu: u64,
}

// SAFETY: D3D12 resources are free-threaded; the mapping is read and cleared
// only between submissions, which run one at a time.
unsafe impl Send for Bitmap {}
// SAFETY: as above.
unsafe impl Sync for Bitmap {}

/// The bitmap of other shaders, or (`memory`) of shaders that use title
/// memory, whose marks go back only while it is not withheld
/// (`crate::memory::withholding`).
fn bitmap(memory: bool) -> Option<&'static Bitmap> {
    static BITMAPS: [std::sync::OnceLock<Option<Bitmap>>; 2] = [const { std::sync::OnceLock::new() }; 2];
    BITMAPS[usize::from(memory)]
        .get_or_init(|| {
            // Written pages, then one read mark per heap index.
            let bits = victory_shader::emit::VIEW_READS + victory_shader::emit::HEAP_VIEWS;
            let bytes = u64::from(bits / 8);
            let (buffer, data) = host()?.shared_buffer(bytes)?;
            crate::host::name(&buffer, || "views' written pages and read slots".into());
            // SAFETY: our mapped buffer of `bytes` bytes.
            unsafe { std::ptr::write_bytes(data, 0, bytes as usize) };
            // SAFETY: a live buffer.
            let gpu = unsafe { buffer.GetGPUVirtualAddress() };
            Some(Bitmap {
                _buffer: buffer,
                data: data as usize,
                gpu,
            })
        })
        .as_ref()
}

/// The host GPU address of the buffer views' written-pages bitmap, for the
/// root UAV of shaders that mark their writes (`memory`: shaders that use
/// title memory, see [`bitmap`]).
pub(crate) fn view_writes(memory: bool) -> Option<u64> {
    bitmap(memory).map(|b| b.gpu)
}

/// Heap slots whose texture views shaders read: the marks seen so far (a
/// slot's stays set in the bitmap until [`forget_read`]), and the slots to
/// clear before the next submission runs ([`clear_forgotten_reads`]).
static READS: Mutex<(Vec<u32>, Vec<u32>)> = Mutex::new((Vec::new(), Vec::new()));

/// The read marks of both bitmaps (shaders that use title memory mark their
/// reads in theirs), when no submission runs.
fn read_marks() -> Vec<&'static mut [u32]> {
    let words = (victory_shader::emit::HEAP_VIEWS / 32) as usize;
    let first = (victory_shader::emit::VIEW_READS / 32) as usize;
    [false, true]
        .into_iter()
        .filter_map(bitmap)
        // SAFETY: the mapped bitmap's read marks; the GPU is done with them
        // (one submission runs at a time).
        .map(|b| unsafe { std::slice::from_raw_parts_mut((b.data as *mut u32).add(first), words) })
        .collect()
}

/// The marks of slots forgotten since go, from the bitmaps and the seen.
fn clear_forgotten(maps: &mut [&mut [u32]], (seen, forgotten): &mut (Vec<u32>, Vec<u32>)) {
    let words = (victory_shader::emit::HEAP_VIEWS / 32) as usize;
    seen.resize(words, 0);
    for slot in forgotten.drain(..) {
        let (w, bit) = ((slot / 32) as usize, 1u32 << (slot % 32));
        if w < words {
            maps.iter_mut().for_each(|bits| bits[w] &= !bit);
            seen[w] &= !bit;
        }
    }
}

/// Before a submission: the marks of slots forgotten since the last one go
/// now (a placed texture written, one given up), so the reads of the
/// submission about to run are the ones [`take_reads`] finds after it.
pub(crate) fn clear_forgotten_reads() {
    let mut maps = read_marks();
    if !maps.is_empty() {
        let mut reads = READS.lock().unwrap_or_else(|e| e.into_inner());
        clear_forgotten(&mut maps, &mut reads);
    }
}

/// After a submission: the heap slots its shaders first read texture views
/// at (`victory_shader::emit`, "Read marks"), each once until
/// [`forget_read`]. Clears the slots forgotten while it ran first.
pub(crate) fn take_reads() -> Vec<u32> {
    let mut maps = read_marks();
    if maps.is_empty() {
        return Vec::new();
    }
    let mut reads = READS.lock().unwrap_or_else(|e| e.into_inner());
    clear_forgotten(&mut maps, &mut reads);
    let mut new = Vec::new();
    // Slots past every view heap's are no heap's (`descriptor::view_slots`):
    // their marks are left for a larger heap to find.
    let used = (crate::descriptor::view_slots().div_ceil(32) as usize).min(reads.0.len());
    for (w, s) in reads.0[..used].iter_mut().enumerate() {
        let mut fresh = maps.iter().fold(0, |x, bits| x | bits[w]) & !*s;
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
/// `crate::resource::evict`). Between submissions, after [`take_reads`]
/// has seen every mark: its own record of slots seen keeps a slot read
/// again from being reported as new.
pub(crate) fn reads_since() -> Vec<u32> {
    let words = (victory_shader::emit::HEAP_VIEWS / 32) as usize;
    let first = (victory_shader::emit::VIEW_READS / 32) as usize;
    let mut slots = Vec::new();
    for b in [false, true].into_iter().filter_map(bitmap) {
        // SAFETY: the mapped bitmap's read marks; no submission runs.
        let bits = unsafe { std::slice::from_raw_parts_mut((b.data as *mut u32).add(first), words) };
        // As in `take_reads`: no further than any view heap's slots.
        let used = (crate::descriptor::view_slots().div_ceil(32) as usize).min(words);
        for (w, word) in bits[..used].iter_mut().enumerate() {
            let mut x = std::mem::take(word);
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

/// The marked bits of `bits` below `end` (and so the bits themselves
/// cleared); marks of offsets past the bitmap land in its last bit, which no
/// copy owns.
fn drain_marks(bits: &mut [u32], end: u32) -> Vec<u32> {
    if let Some(last) = bits.last_mut() {
        *last &= !(1 << 31);
    }
    let used = (end as usize).div_ceil(32).min(bits.len());
    let mut marked = Vec::new();
    for (w, word) in bits[..used].iter_mut().enumerate() {
        let mut x = std::mem::take(word);
        while x != 0 {
            marked.push((w * 32) as u32 + x.trailing_zeros());
            x &= x - 1;
        }
    }
    marked
}

/// Threads that put marked pages back, at most (the title's own threads
/// run meanwhile).
const PAGE_THREADS: usize = 6;
/// Marked pages worth a thread more: fewer go back on the submitting one.
const PAGES_PER_THREAD: usize = 8;

/// Runs `f` on every `(copy, page)`, on several threads when there are many:
/// a 64 KiB page costs 55 to 80 microseconds (two reads, a compare, writes
/// and a protection change), and a world frame marks 2,000. The copies of
/// one page keep their order, on one thread.
fn for_each_by_page(mut pages: Vec<(Arc<Shadow>, usize)>, f: impl Fn(&Arc<Shadow>, usize) + Sync) {
    pages.sort_by_key(|&(_, page)| page);
    let threads = (pages.len() / PAGES_PER_THREAD).min(PAGE_THREADS);
    if threads < 2 {
        for (copy, page) in &pages {
            f(copy, *page);
        }
        return;
    }
    let groups: Vec<&[(Arc<Shadow>, usize)]> = pages.chunk_by(|a, b| a.1 == b.1).collect();
    let next = std::sync::atomic::AtomicUsize::new(0);
    let work = || {
        while let Some(group) = groups.get(next.fetch_add(1, std::sync::atomic::Ordering::Relaxed)) {
            for (copy, page) in *group {
                f(copy, *page);
            }
        }
    };
    std::thread::scope(|scope| {
        for _ in 1..threads {
            scope.spawn(work);
        }
        work();
    });
}

/// After a submission: every page its shaders marked as written through
/// views goes back ([`Shadow::copy_back_marked`]), except pages in
/// `withheld` (ranges of copies that lists whose writes stay out of title
/// memory declared) and, with `memory_withheld`, pages that shaders using
/// title memory marked; both kinds of withheld marked pages are dropped from
/// the copies (`crate::memory::drop_withheld`), and the bitmaps are cleared.
/// Before the copy-backs of declared ranges, which disarm the copies.
// shortcut: marks do not tell lists apart: a page that a withheld list and
// another both wrote through views goes back unless the withheld list
// declared it.
pub(crate) fn copy_back_marks(withheld: &[(Arc<Shadow>, usize, usize)], memory_withheld: bool) {
    let (mut marked, mut held) = (Vec::new(), Vec::new());
    for memory in [false, true] {
        let Some(b) = bitmap(memory) else {
            continue;
        };
        let words = (victory_shader::emit::VIEW_WRITE_BITS / 32) as usize;
        // SAFETY: the mapped bitmap; the GPU is done with it (one submission
        // runs at a time).
        let bits = unsafe { std::slice::from_raw_parts_mut(b.data as *mut u32, words) };
        let into = if memory && memory_withheld { &mut held } else { &mut marked };
        into.extend(drain_marks(bits, marks().end));
    }
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
    let mut dropped = Vec::new();
    for_each_by_page(pages, |copy, page| {
        if withheld
            .iter()
            .any(|(c, a, n)| Arc::ptr_eq(c, copy) && *a < page + PAGE && page < a + n)
        {
            dropped.push(page);
            return;
        }
        let written = copy.copy_back_marked(page);
        #[cfg(feature = "diagnostics")]
        {
            crate::copystat::add(crate::copystat::C::MarkedPages, 1);
            crate::copystat::add(crate::copystat::C::MarkedBytes, written as u64);
        }
        #[cfg(not(feature = "diagnostics"))]
        let _ = written;
    });
    // Dropped after the marked copy-backs above: an immediate retake on a
    // shared page would otherwise mark another copy's page changed before its
    // own copy-back runs.
    dropped.sort_unstable();
    dropped.dedup();
    for page in dropped {
        crate::memory::drop_withheld(page, PAGE);
    }
}

struct Entry {
    end: u64,
    shadow: Option<Arc<Shadow>>,
}

struct Buffers {
    map: BTreeMap<(u64, usize), Entry>,
    /// Live registration count per buffer size, keeping `max_size()` exact
    /// in O(log n) across inserts, in-place replacements, and removals.
    sizes: BTreeMap<u64, usize>,
}

impl Buffers {
    const fn new() -> Self {
        Self {
            map: BTreeMap::new(),
            sizes: BTreeMap::new(),
        }
    }

    fn max_size(&self) -> u64 {
        self.sizes.keys().next_back().copied().unwrap_or(0)
    }

    fn insert(&mut self, resource: usize, start: u64, len: u64) {
        if let Some(prev) = self.map.insert(
            (start, resource),
            Entry {
                end: start + len,
                shadow: None,
            },
        ) {
            let old_len = prev.end - start;
            if old_len == len {
                return;
            }
            self.decrement_size(old_len);
        }
        *self.sizes.entry(len).or_insert(0) += 1;
    }

    fn remove(&mut self, resource: usize, start: u64) -> Option<Entry> {
        let removed = self.map.remove(&(start, resource))?;
        self.decrement_size(removed.end - start);
        Some(removed)
    }

    fn decrement_size(&mut self, len: u64) {
        if let std::collections::btree_map::Entry::Occupied(mut entry) = self.sizes.entry(len) {
            if *entry.get() > 1 {
                *entry.get_mut() -= 1;
            } else {
                entry.remove();
            }
        }
    }
}

/// Title buffers by (start address, resource), with their copy once made.
static BUFFERS: Mutex<Buffers> = Mutex::new(Buffers::new());

fn buffers() -> std::sync::MutexGuard<'static, Buffers> {
    BUFFERS.lock().unwrap_or_else(|e| e.into_inner())
}

/// The longest title buffer registered so far (it stays after the buffer is
/// released): a buffer starting further below an address than this cannot
/// hold it. Changed under `BUFFERS`' lock.
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

/// Counts title buffers made and released: which buffer is the innermost
/// one holding some bytes ([`find`]) may differ when it changed.
static OWNERS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// The count of [`OWNERS`].
pub(crate) fn owners() -> u64 {
    OWNERS.load(std::sync::atomic::Ordering::Acquire)
}

/// A title buffer exists at `start` (`len` bytes).
pub(crate) fn register(resource: usize, start: u64, len: u64) {
    buffers().insert(resource, start, len);
    // After the buffers changed: a count read before is looked at again.
    OWNERS.fetch_add(1, std::sync::atomic::Ordering::Release);
}

/// The title released it; lists that still hold its copy keep it alive.
pub(crate) fn unregister(resource: usize, start: u64) {
    let removed = buffers().remove(resource, start);
    OWNERS.fetch_add(1, std::sync::atomic::Ordering::Release);
    if removed.is_some_and(|e| e.shadow.is_some()) {
        changed();
    }
}

fn find_buffer(
    map: &BTreeMap<(u64, usize), Entry>,
    max_size: u64,
    address: u64,
    end: u64,
) -> Option<(u64, usize)> {
    let lowest = end.saturating_sub(max_size);
    if lowest > address {
        return None;
    }
    map.range((lowest, 0)..=(address, usize::MAX))
        .rev()
        .find(|(_, e)| e.end >= end)
        .map(|(k, _)| *k)
}

/// The copy of the innermost title buffer holding `[address, address +
/// len)`, made on first use.
pub(crate) fn find(address: u64, len: u64) -> Option<Arc<Shadow>> {
    let end = address.checked_add(len.max(1))?;
    let mut state = buffers();
    let Some(key) = find_buffer(&state.map, state.max_size(), address, end) else {
        drop(state);
        return loose(address, end);
    };
    let entry = state.map.get_mut(&key)?;
    if entry.shadow.is_none() {
        let len = entry.end - key.0;
        // Rounded up so that views ending in the buffer's last 256 bytes fit.
        // Shaders write some buffers (unordered access), so not upload memory.
        let (buffer, data) = host()?.shared_buffer(len.next_multiple_of(1 << 16))?;
        crate::host::name(&buffer, || format!("title buffer {:#x}+{len:#x}", key.0));
        // SAFETY: a live buffer.
        let gpu = unsafe { buffer.GetGPUVirtualAddress() };
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
        entry.shadow = Some(Shadow::new(buffer, data, gpu, key.0, len));
        changed();
    }
    entry.shadow.clone()
}

/// Titles also use memory outside any buffer object as GPU memory (Grand
/// Theft Auto VI's interface vertices, from its own allocator). Such memory
/// gets copies of the aligned windows it touches, 4 MiB by default.
/// `VICTORY_LOOSE_WINDOW_KIB` (any power of two) sets another size:
/// lookups while Grand Theft Auto VI loads its world touched about
/// 30% of the 4 MiB windows' bytes, so 256 KiB windows hold GBs less host
/// memory, but in the open world they copied about ten times the bytes per
/// frame and frames took longer.
/// shortcut: copies of windows are never freed; key them to the title's
/// allocations if memory use grows.
fn window() -> u64 {
    static WINDOW: std::sync::OnceLock<u64> = std::sync::OnceLock::new();
    *WINDOW.get_or_init(|| window_from(std::env::var("VICTORY_LOOSE_WINDOW_KIB").ok().as_deref()))
}

/// The window size `VICTORY_LOOSE_WINDOW_KIB` asks for, in bytes; 4 MiB
/// when it is unset or not a power of two (sizes far from the default are
/// the user's experiment).
fn window_from(kib: Option<&str>) -> u64 {
    kib.and_then(|v| v.trim().parse::<u64>().ok())
        .filter(|k| k.is_power_of_two())
        .and_then(|k| k.checked_mul(1 << 10))
        .unwrap_or(4 << 20)
}

/// Copies of title memory outside buffers, by (start, length).
static LOOSE: Mutex<BTreeMap<(u64, u64), Arc<Shadow>>> = Mutex::new(BTreeMap::new());

/// The longest window in `LOOSE`. Changed under its lock.
static LOOSE_SPAN: AtomicU64 = AtomicU64::new(0);

/// Below this nothing is ever mapped (the null page and its guard area).
const LOWEST_MAPPED: u64 = 0x10000;

/// The window copy holding `[address, end)` among `windows` (by start and
/// length, none longer than `span`), the innermost if several do.
fn enclosing_window<V>(windows: &BTreeMap<(u64, u64), V>, span: u64, address: u64, end: u64) -> Option<&V> {
    let lowest = end.saturating_sub(span);
    if lowest > address {
        return None;
    }
    windows
        .range((lowest, 0)..=(address, u64::MAX))
        .rev()
        .find(|&(&(s, l), _)| s.saturating_add(l) >= end)
        .map(|(_, v)| v)
}

/// Windows a covering window answered instead of a new copy (diagnostics).
#[cfg(windows)]
pub(crate) static REUSES: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// The loose windows by count and bytes (diagnostics): the shared segment's
/// own content; the copies are never freed, so what this reaches is what
/// the segment carries until the process ends.
#[cfg(windows)]
pub(crate) fn loose_stats() -> (usize, u64) {
    let map = LOOSE.lock().unwrap_or_else(|e| e.into_inner());
    let bytes = map.iter().map(|((_, len), _)| *len).sum();
    (map.len(), bytes)
}

/// The window copy for `[address, end)` outside buffers: one already made
/// that holds it (a range spanning two windows has one of its own, which
/// later ranges in it share), else a new one. None for memory that is not
/// committed (a stale or null address): a window of zeros would not help.
fn loose(address: u64, end: u64) -> Option<Arc<Shadow>> {
    if address < LOWEST_MAPPED {
        return None;
    }
    let window = window();
    let start = address & !(window - 1);
    let len = end.checked_next_multiple_of(window)? - start;
    let mut map = LOOSE.lock().unwrap_or_else(|e| e.into_inner());
    if let Some(shadow) = enclosing_window(&map, LOOSE_SPAN.load(Ordering::Relaxed), address, end) {
        return Some(shadow.clone());
    }
    // A window that already covers [start, start + len) serves it without
    // another host copy: the title asks for runs of neighbouring windows
    // (a mesh pass's constants, then its vertices), and one big window
    // covers each of them; measured, most asks were covered this way and
    // the copies they would have made were the shared segment's bulk.
    // Candidates start at or before ours; the first whose end reaches
    // past ours wins (nearest start first, from the map's end).
    let covering = map
        .range(..(start, u64::MAX))
        .rev()
        .find(|((at, l), _)| at + *l >= start + len);
    if let Some((_, shadow)) = covering {
        #[cfg(windows)]
        REUSES.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        tracing::debug!(
            start = format_args!("{start:#x}"),
            len,
            "loose memory covered by a wider window"
        );
        return Some(shadow.clone());
    }
    if !crate::texture::committed(address as *const u8, 1) {
        return None;
    }
    let (buffer, data) = host()?.shared_buffer(len)?;
    crate::host::name(&buffer, || format!("title memory {start:#x}+{len:#x}"));
    // SAFETY: a live buffer.
    let gpu = unsafe { buffer.GetGPUVirtualAddress() };
    tracing::debug!(
        start = format_args!("{start:#x}"),
        len,
        "loose memory copied to the host"
    );
    let shadow = Shadow::new(buffer, data, gpu, start, len);
    LOOSE_SPAN.fetch_max(len, Ordering::Relaxed);
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
        let size = |c: &(ID3D12Resource, usize, u64, BTreeMap<u64, u64>)| unsafe { c.0.GetDesc() }.Width;
        let held: u64 = a.chunks.iter().map(|c| size(c) - c.3.values().sum::<u64>()).sum();
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

/// `VICTORY_STALE_CHECK` (diagnostics, after ka1qux's check of the same
/// name): once per presented frame, before a submission refreshes anything,
/// one host copy in N (round robin over the copies) has the pages its tracker
/// calls unchanged compared with title memory. A page that differs holds a
/// write the tracking missed: the title's through a path that neither faults
/// nor reports, or the shaders' through a view nobody declared or marked.
/// The counts and the first pages that differ are logged with the frame
/// statistics ([`log_stale`]); the target is none.
///
/// Its value: N (`1` checks every copy every frame); `fix`, every copy, and
/// the pages that differ are copied; anything else, one copy in 64; unset,
/// empty or `0`, off.
#[cfg(feature = "diagnostics")]
fn stale_setting(value: Option<&str>) -> Option<(u64, bool)> {
    match value {
        None | Some("" | "0") => None,
        Some("fix") => Some((1, true)),
        Some(n) => Some((n.parse().ok().filter(|&n| n > 0).unwrap_or(64), false)),
    }
}

#[cfg(feature = "diagnostics")]
fn stale_check() -> Option<(u64, bool)> {
    static ON: std::sync::OnceLock<Option<(u64, bool)>> = std::sync::OnceLock::new();
    *ON.get_or_init(|| {
        let on = stale_setting(std::env::var("VICTORY_STALE_CHECK").ok().as_deref());
        if let Some((every, fix)) = on {
            tracing::warn!(
                every,
                fix,
                "VICTORY_STALE_CHECK: pages of host copies the write tracking calls unchanged are compared with title memory"
            );
        }
        on
    })
}

/// What [`check_stale`] found since the last [`log_stale`]: copies checked,
/// pages compared, pages and bytes that differed, the first pages that did
/// (addresses, at most 10); and pages that differed in all.
#[cfg(feature = "diagnostics")]
#[derive(Default)]
struct Stale {
    copies: u64,
    pages: u64,
    differed: u64,
    bytes: u64,
    first: Vec<usize>,
    in_all: u64,
}

#[cfg(feature = "diagnostics")]
static STALE: Mutex<Stale> = Mutex::new(Stale {
    copies: 0,
    pages: 0,
    differed: 0,
    bytes: 0,
    first: Vec::new(),
    in_all: 0,
});

/// At the start of a submission (under the submission lock: no copy-back or
/// refresh runs, nothing is armed or pending): with `VICTORY_STALE_CHECK`,
/// the first submission of each presented frame checks its share of the
/// copies ([`Shadow::stale_pages`]).
#[cfg(feature = "diagnostics")]
pub(crate) fn check_stale() {
    use std::sync::atomic::{AtomicU64, Ordering::Relaxed};
    let Some((every, fix)) = stale_check() else {
        return;
    };
    static FRAME: AtomicU64 = AtomicU64::new(u64::MAX);
    let frame = crate::queue::presented();
    if FRAME.swap(frame, Relaxed) == frame {
        return;
    }
    for (k, copy) in all().iter().enumerate() {
        if !(k as u64).wrapping_add(frame).is_multiple_of(every) {
            continue;
        }
        let (pages, differ) = copy.stale_pages(fix);
        let mut s = STALE.lock().unwrap_or_else(|e| e.into_inner());
        s.copies += 1;
        s.pages += pages;
        s.differed += differ.len() as u64;
        s.in_all += differ.len() as u64;
        s.bytes += differ.iter().map(|&(_, n)| n as u64).sum::<u64>();
        let room = 10usize.saturating_sub(s.first.len());
        s.first.extend(differ.iter().map(|&(at, _)| at).take(room));
    }
}

/// Logs what [`check_stale`] found over the last `frames` frames, if on.
#[cfg(feature = "diagnostics")]
pub(crate) fn log_stale(frames: f64) {
    let Some((every, _)) = stale_check() else {
        return;
    };
    let s = {
        let mut s = STALE.lock().unwrap_or_else(|e| e.into_inner());
        let in_all = s.in_all;
        let taken = std::mem::take(&mut *s);
        s.in_all = in_all;
        taken
    };
    let first: Vec<String> = s.first.iter().map(|a| format!("{a:#x}")).collect();
    tracing::info!(
        "stale check over {frames:.0} frames (one copy in {every} a frame): {} copies, {} pages the write tracking \
         called unchanged compared with title memory, {} of them differed ({} bytes), {} in all; first: {}",
        s.copies,
        s.pages,
        s.differed,
        s.bytes,
        s.in_all,
        first.join(", ")
    );
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
    let (buffer, data, gpu, slice) = arena().take(len, element)?;
    let shadow = Shadow::new_at(buffer, data, gpu, address, len, Some(slice));
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
#[derive(Default)]
struct Arena {
    /// Each buffer, its mapping and host address, and its free runs
    /// (offset → bytes).
    chunks: Vec<(ID3D12Resource, usize, u64, BTreeMap<u64, u64>)>,
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
    fn take(&mut self, len: u64, element: u64) -> Option<(ID3D12Resource, *mut u8, u64, Slice)> {
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
        let found = self.chunks.iter().enumerate().find_map(|(c, chunk)| fits(&chunk.3).map(|f| (c, f)));
        let (chunk, (at, n, start)) = match found {
            Some(found) => found,
            None => {
                let size = CHUNK.max((len + align).next_multiple_of(1 << 16));
                let (buffer, data) = host()?.shared_buffer(size)?;
                crate::host::name(&buffer, || format!("title views {size:#x}"));
                // SAFETY: a live buffer.
                let gpu = unsafe { buffer.GetGPUVirtualAddress() };
                let free = BTreeMap::from([(0, size)]);
                self.chunks.push((buffer, data as usize, gpu, free));
                let c = self.chunks.len() - 1;
                (c, fits(&self.chunks[c].3)?)
            }
        };
        let (buffer, data, gpu, free) = &mut self.chunks[chunk];
        free.remove(&at);
        if start > at {
            free.insert(at, start - at);
        }
        if at + n > start + len {
            free.insert(start + len, at + n - start - len);
        }
        let slice = Slice { chunk, offset: start, len };
        Some((buffer.clone(), (*data + start as usize) as *mut u8, *gpu + start, slice))
    }

    /// A run goes back, joined to its free neighbours.
    fn give_back(&mut self, slice: Slice) {
        let Some((_, _, _, free)) = self.chunks.get_mut(slice.chunk) else { return };
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
    let mut all: Vec<Arc<Shadow>> = buffers().map.values().filter_map(|e| e.shadow.clone()).collect();
    all.extend(LOOSE.lock().unwrap_or_else(|e| e.into_inner()).values().cloned());
    all
}

/// For device-removed reports: the copy of title memory whose host range
/// starts closest below host GPU address `va`, and where `va` is in it.
pub(crate) fn near(va: u64) -> String {
    let mut all: Vec<(u64, u64, u64)> = buffers()
        .map
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
    use windows_sys::Win32::System::Memory::{
        MEM_COMMIT, MEM_RESERVE, PAGE_READONLY, PAGE_READWRITE, VirtualAlloc, VirtualProtect,
    };

    const OS_PAGE: usize = 4 << 10;

    struct TitlePage(usize);

    impl Drop for TitlePage {
        fn drop(&mut self) {
            crate::watch::tests::free(self.0);
        }
    }

    impl TitlePage {
        fn protect_readonly(&self, offset: usize) {
            assert!(offset <= PAGE - OS_PAGE);
            let mut previous = 0;
            // SAFETY: a committed page in the fixture's allocation.
            let ok = unsafe {
                VirtualProtect((self.0 + offset) as _, OS_PAGE, PAGE_READONLY, &mut previous)
            };
            assert_ne!(ok, 0);
        }

        fn byte(&self, offset: usize) -> u8 {
            assert!(offset < PAGE);
            // SAFETY: the fixture's committed pages remain readable.
            unsafe { ((self.0 + offset) as *const u8).read_volatile() }
        }
    }

    /// A fully delivered computed-address write becomes current in a watched
    /// shadow. A later partial declared write must keep that shadow dirty.
    #[test]
    fn copyback_acknowledges_only_delivered_watched_pages() {
        let Some(host) = host() else { return };
        let title = TitlePage(crate::watch::tests::pool(1));
        let at = title.0;
        let (buffer, data) = host.shared_buffer(PAGE as u64).expect("native shadow");
        let copy = Shadow::new(buffer, data, 0, at as u64, PAGE as u64);
        let tracked = copy.tracked.as_ref().expect("watched title allocation");
        assert_eq!(copy.update(at, PAGE), PAGE);
        assert!(!tracked.changed(at / PAGE));

        copy.write_host((at + 8) as u64, &[0x31]);
        assert_eq!(copy.copy_back_page(at), PAGE);
        assert_eq!(title.byte(8), 0x31);
        assert!(!tracked.changed(at / PAGE));

        copy.writing(at, PAGE, false);
        copy.write_host((at + 8) as u64, &[0x62]);
        copy.write_host((at + OS_PAGE + 8) as u64, &[0x73]);
        // Hand the page back to the title, then apply a different protection
        // that belongs to the test, not to the write-watch tracker.
        drop(crate::watch::writing(at, PAGE));
        title.protect_readonly(OS_PAGE);
        assert_eq!(copy.copy_back(at, PAGE), PAGE - OS_PAGE);
        assert_eq!([title.byte(8), title.byte(OS_PAGE + 8)], [0x62, 0]);
        assert!(tracked.changed(at / PAGE));
    }

    #[test]
    fn marked_copyback_acknowledges_only_complete_pages() {
        let Some(host) = host() else { return };
        let title = TitlePage(crate::watch::tests::pool(1));
        let at = title.0;
        let (buffer, data) = host.shared_buffer(PAGE as u64).expect("native shadow");
        let copy = Shadow::new(buffer, data, 0, at as u64, PAGE as u64);
        let tracked = copy.tracked.as_ref().expect("watched title allocation");
        assert_eq!(copy.update(at, PAGE), PAGE);
        copy.writing(at, PAGE, true);
        drop(crate::watch::writing(at, PAGE));
        title.protect_readonly(OS_PAGE);

        copy.write_host((at + 8) as u64, &[0x31]);
        copy.write_host((at + OS_PAGE + 8) as u64, &[0x42]);
        assert_eq!(copy.copy_back_marked(at), PAGE - OS_PAGE);
        assert_eq!([title.byte(8), title.byte(OS_PAGE + 8)], [0x31, 0]);
        assert!(tracked.changed(at / PAGE));
    }

    #[test]
    fn computed_address_copyback_counts_only_delivered_spans() {
        let Some(host) = host() else { return };
        // SAFETY: a committed private allocation owned by TitlePage.
        let at = unsafe {
            VirtualAlloc(std::ptr::null(), PAGE, MEM_COMMIT | MEM_RESERVE, PAGE_READWRITE)
        } as usize;
        assert_ne!(at, 0);
        let title = TitlePage(at);
        let (buffer, data) = host.shared_buffer(PAGE as u64).expect("native shadow");
        let copy = Shadow::new(buffer, data, 0, at as u64, PAGE as u64);
        assert!(copy.tracked.is_none());
        assert_eq!(copy.update(at, PAGE), PAGE);
        copy.write_host((at + 8) as u64, &[0x42]);
        copy.write_host((at + OS_PAGE + 8) as u64, &[0x53]);
        title.protect_readonly(OS_PAGE);
        assert_eq!(copy.copy_back_page(at), PAGE - OS_PAGE);
        assert_eq!([title.byte(8), title.byte(OS_PAGE + 8)], [0x42, 0]);
    }
    /// The copy holding a view's first byte holds it to that copy's end:
    /// a lookup of that whole range finds the same copy, in an inner buffer
    /// inside an outer one, in the outer one, and in loose memory in no
    /// buffer. Constant views are looked up once so (`draw`'s `readable`).
    #[test]
    fn the_copy_holding_a_views_first_byte_holds_it_to_its_end() {
        if host().is_none() {
            return;
        }
        let same = |address: u64, size: u64| {
            let first = find(address, 1).expect("a copy");
            let len = size.min(first.left(address));
            let whole = find(address, len).expect("a copy");
            assert!(Arc::ptr_eq(&first, &whole), "{address:#x}+{size}");
        };
        let memory = vec![0u8; 4 * PAGE];
        let base = memory.as_ptr() as u64;
        let page = PAGE as u64;
        // Resource ids of their own (the addresses are the memory's).
        let (outer, inner) = (base as usize | 1, base as usize | 2);
        register(outer, base, 4 * page);
        register(inner, base + page, page);
        // In the outer buffer; in the inner one; past the inner one's end.
        for (address, size) in [(base + 16, 10_000), (base + page + 256, 10_000), (base + 2 * page - 100, 10_000)] {
            same(address, size);
        }
        unregister(inner, base + page);
        unregister(outer, base);
        drop(memory);
        // Loose memory: copies of the windows it touches.
        let loose = vec![0u8; 2 * PAGE];
        let address = loose.as_ptr() as u64 + 64;
        same(address, 1 << 20);
        same(address, 64);
    }

    /// A shader's read mark of heap slot `slot` in the bitmap of shaders
/// that use title memory (`memory`) or of the others.
pub(crate) fn mark_read(memory: bool, slot: u32) {
    let b = bitmap(memory).expect("a host GPU");
    let bit = victory_shader::emit::VIEW_READS + slot;
    // SAFETY: a bit of the mapped bitmap; no submission runs (the caller
    // holds the submission lock).
    unsafe { *(b.data as *mut u32).add((bit / 32) as usize) |= 1 << (bit % 32) };
}

    /// Records a GPU write setting `slot`'s read mark in the bitmap, as a
    /// shader reading that heap slot does during a submission.
    pub(crate) fn record_mark_read(list: &crate::host::HostList, memory: bool, slot: u32) {
        let b = bitmap(memory).expect("a host GPU");
        let (src, data) = host().unwrap().upload_buffer(4).unwrap();
        // SAFETY: our 4-byte upload buffer.
        unsafe { data.cast::<u32>().write_unaligned(1u32 << (slot % 32)) };
        let offset = u64::from(victory_shader::emit::VIEW_READS / 8) + u64::from(slot / 32) * 4;
        // SAFETY: an open list, live buffers, 4 bytes inside both.
        unsafe { list.list.CopyBufferRegion(&b._buffer, offset, &src, 0, 4) };
        list.keep(&src);
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
        copy.update(p, len);
        // An earlier list declares all of it as written by shaders.
        copy.writing(p, len, false);
        // SAFETY: (all title reads and writes below) bytes of our pool.
        unsafe { title(PAGE + 20).write_volatile(0x55) };
        assert!(copy.update(p + PAGE + 16, 16) > 0, "the title's write is copied");
        assert_eq!(host(PAGE + 20), 0x55, "the title's constant reaches the copy");
        // The shaders run: one byte written. The title writes meanwhile.
        copy.write_host((p + 10) as u64, &[0xaa]);
        // SAFETY: a byte of our pool.
        unsafe { title(PAGE + 20).write_volatile(0x66) };
        copy.copy_back(p, len);
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
        copy.update(pool, len);
        assert_eq!(copy.read_host(word as u64, 1), [0x1234]);
        assert!(!copy.clean(), "never clean: part of it is not tracked");
        // SAFETY: as above.
        unsafe { word.write_volatile(0x5678) };
        copy.update(pool, len);
        assert_eq!(copy.read_host(word as u64, 1), [0x5678], "the title's change reaches the copy");
        assert_eq!(protection(pool), PAGE_READONLY, "the pool is guarded");
        assert_eq!(protection(plain), PAGE_READWRITE, "the plain allocation is not");
        unregister(pool, pool as u64);
        // The pages stay allocated (as the other tests' pools): a copy some
        // page table still holds must not name memory reused meanwhile.
    }

    /// A loose window wider than a later ask covers it: the narrower ask
    /// reuses the wider copy instead of making another host buffer for
    /// memory a copy already spans (the shared segment's bulk was these).
    #[test]
    fn a_wider_loose_window_covers_a_later_smaller_ask() {
        use windows_sys::Win32::System::Memory::{
            MEM_COMMIT, MEM_RELEASE, MEM_RESERVE, PAGE_READWRITE, VirtualAlloc, VirtualFree,
        };
        // One window-aligned pair of pages inside a reservation that
        // guarantees the addresses, found free again (other tests allocate
        // meanwhile: tried again if taken).
        let pair = (0..8)
            .find_map(|_| {
                // SAFETY: a fresh reservation, released at once; then the
                // two window-aligned allocations where it was, kept (as the
                // other tests' pools: their copies must not name memory a
                // later test reuses).
                unsafe {
                    let p = VirtualAlloc(std::ptr::null(), 3 * window() as usize, MEM_RESERVE, PAGE_READWRITE) as usize;
                    if p == 0 {
                        return None;
                    }
                    VirtualFree(p as *mut _, 0, MEM_RELEASE);
                    let aligned = p.next_multiple_of(window() as usize);
                    let at = |q: usize| {
                        VirtualAlloc(q as *const _, window() as usize, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE) as usize
                    };
                    let (one, two) = (at(aligned), at(aligned + window() as usize));
                    if one == aligned && two == aligned + window() as usize {
                        return Some((one, two));
                    }
                    [one, two].into_iter().filter(|&q| q != 0).for_each(|q| {
                        VirtualFree(q as *mut _, 0, MEM_RELEASE);
                    });
                    None
                }
            })
            .expect("a window-aligned pair of pages");
        let (one, two) = (pair.0 as u64, pair.1 as u64);
        let Some(wide) = find(one, 2 * window()) else {
            return; // no host GPU here
        };
        let before = REUSES.load(std::sync::atomic::Ordering::Relaxed);
        let narrow = find(two, window());
        let Some(narrow) = narrow else {
            return; // no host GPU here (the wide one already was)
        };
        assert!(
            std::sync::Arc::ptr_eq(&wide, &narrow),
            "the narrower ask reuses the covering window"
        );
        assert_eq!(
            REUSES.load(std::sync::atomic::Ordering::Relaxed),
            before + 1,
            "the reuse is counted"
        );
        // An ask the wide window does not reach still gets its own copy.
        let beyond = find(two, 2 * window());
        if let Some(beyond) = beyond {
            assert!(
                !std::sync::Arc::ptr_eq(&wide, &beyond),
                "an ask past the covering window makes its own"
            );
            assert_eq!(
                REUSES.load(std::sync::atomic::Ordering::Relaxed),
                before + 1,
                "no reuse for an ask past the window"
            );
        }
    }

    /// The stale check (`VICTORY_STALE_CHECK`) finds a page the title wrote
    /// with its guard opened behind the tracking's back (a write nothing
    /// sees or reports), and only it: a write the guard saw counts as
    /// changed and is passed by, and so is a copy shaders may be writing.
    /// With `fix` the page is copied, and the next check finds nothing.
    #[cfg(feature = "diagnostics")]
    #[test]
    fn the_stale_check_finds_writes_the_tracking_missed() {
        use windows_sys::Win32::System::Memory::{PAGE_READWRITE, VirtualProtect};
        let p = crate::watch::tests::pool(3);
        let len = 3 * PAGE;
        register(p, p as u64, len as u64);
        let Some(copy) = find(p as u64, len as u64) else {
            unregister(p, p as u64);
            return; // no host GPU here
        };
        copy.update(p, len);
        assert_eq!(copy.stale_pages(false), (3, Vec::new()), "nothing missed");
        // SAFETY: a byte of our pool, guarded: the handler marks the page.
        unsafe { ((p + 10) as *mut u8).write_volatile(1) };
        let mut old = 0;
        // SAFETY: a page of our pool; only its protection changes.
        let opened =
            unsafe { VirtualProtect((p + PAGE) as *const _, PAGE, PAGE_READWRITE, &mut old) };
        assert_ne!(opened, 0);
        // SAFETY: two bytes of our pool, now writable with no fault.
        unsafe { ((p + PAGE + 20) as *mut u16).write_volatile(0x0203) };
        let found = (2, vec![(p + PAGE, 2)]);
        assert_eq!(
            copy.stale_pages(false),
            found,
            "page 0 changed, page 1 missed"
        );
        assert_eq!(copy.stale_pages(true), found);
        assert_eq!(copy.stale_pages(false), (2, Vec::new()), "copied");
        assert_eq!(
            copy.read_host((p + PAGE + 20) as u64, 1)[0] & 0xffff,
            0x0203
        );
        copy.writing(p, len, false);
        assert_eq!(
            copy.stale_pages(false),
            (0, Vec::new()),
            "shaders may be writing it"
        );
        copy.copy_back(p, len);
        unregister(p, p as u64);
    }

    /// `VICTORY_STALE_CHECK`'s values: off, one copy in N, every copy with
    /// the pages that differ copied, one in 64 for anything else.
    #[cfg(feature = "diagnostics")]
    #[test]
    fn stale_check_settings() {
        for off in [None, Some(""), Some("0")] {
            assert_eq!(stale_setting(off), None);
        }
        assert_eq!(stale_setting(Some("1")), Some((1, false)));
        assert_eq!(stale_setting(Some("16")), Some((16, false)));
        assert_eq!(stale_setting(Some("fix")), Some((1, true)));
        assert_eq!(stale_setting(Some("on")), Some((64, false)));
    }

    /// Runs of the shared buffers start a whole number of elements (and of
    /// 256 bytes) in, do not overlap, and join again when given back.
    #[test]
    fn arena_runs_are_aligned_to_their_elements_and_join_again() {
        if host().is_none() {
            return; // no host GPU here
        }
        let mut arena = Arena::default();
        let (_, _, _, a) = arena.take(100, 456).unwrap();
        let (_, _, _, b) = arena.take(5000, 16).unwrap();
        let (_, _, _, c) = arena.take(4, 4).unwrap();
        assert!(a.offset % 456 == 0 && a.offset % 256 == 0 && b.offset % 256 == 0);
        let mut runs = [a, b, c].map(|s| (s.offset, s.offset + s.len));
        runs.sort_unstable();
        assert!(runs.windows(2).all(|w| w[0].1 <= w[1].0), "{runs:?}");
        for s in [b, a, c] {
            arena.give_back(s);
        }
        assert_eq!(arena.chunks.len(), 1);
        assert_eq!(arena.chunks[0].3.len(), 1, "one free run again");
        // Larger than a buffer: one of its own.
        let (_, _, _, d) = arena.take(CHUNK + 1, 4).unwrap();
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
        copy.update(p, len);
        let bytes = |s: &Shadow| s.before.lock().unwrap().0.iter().map(|(_, b)| b.len()).sum::<usize>();
        copy.writing(p + 16, 64, false);
        copy.writing(p + 3 * PAGE, 100, false);
        assert_eq!(bytes(&copy), 164, "two segments");
        copy.writing(p + PAGE, 2 * PAGE, false);
        copy.writing(p + PAGE, PAGE, false);
        copy.writing(p + 80, 8, false);
        assert_eq!(bytes(&copy), 72 + 2 * PAGE + 100, "merged where they overlap or touch");
        // The GPU writes inside two ranges.
        copy.write_host((p + 20) as u64, &[1, 2, 3]);
        copy.write_host((p + PAGE + 5) as u64, &[9]);
        assert_eq!(copy.copy_back(p + 16, 64), 64, "the range's part of the piece");
        assert_eq!(copy.copy_back(p + PAGE, 2 * PAGE), PAGE, "the piece written");
        assert_eq!(copy.copy_back(p + PAGE, PAGE), 0, "compared once");
        assert_eq!(copy.copy_back(p + 3 * PAGE, 100), 0);
        assert_eq!(copy.copy_back(p + 80, 8), 0);
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
        // No submission takes the marks set below meanwhile.
        let _quiet = crate::host::tests::no_submissions();
        let p = crate::watch::tests::pool(3);
        let len = 3 * PAGE;
        register(p, p as u64, len as u64);
        let (Some(copy), Some(b)) = (find(p as u64, len as u64), bitmap(false)) else {
            unregister(p, p as u64);
            return; // no host GPU here
        };
        let base = copy.marks.expect("tracked memory gets mark bits") as usize;
        assert_eq!(copy.mark_entry(p as u64 + PAGE as u64 + 8), Some((base as u32 + 1, 8)));
        copy.update(p, len);
        copy.writing(p, len, true);
        assert!(copy.before.lock().unwrap().0.is_empty(), "no snapshot");
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
        copy_back_marks(&[], false);
        // SAFETY: bytes of our pool.
        let read = |at: usize| unsafe { title(at).read_volatile() };
        assert_eq!(read(10), 0xaa, "the shader's byte, page 0");
        assert_eq!((read(PAGE + 20), read(PAGE + 40)), (0xbb, 0x11), "both writers' bytes, page 1");
        assert_eq!((read(2 * PAGE + 30), read(2 * PAGE + 50)), (0, 0x22), "page 2 not marked");
        // SAFETY: dwords of the mapped bitmap.
        let left: u32 = (base / 32..=(base + 1) / 32).map(|w| unsafe { *bits.add(w) }).fold(0, |a, w| a | w);
        assert_eq!(left, 0, "bits cleared");
        assert_eq!(copy.copy_back(p, len), 0, "nothing kept to compare");
        unregister(p, p as u64);
    }

    /// Marks of shaders using title memory stay out of title memory while
    /// their results are held back, and are cleared; the other shaders'
    /// marks of the same submission go back.
    #[test]
    fn marks_of_shaders_using_title_memory_are_held_back() {
        // No submission takes the marks set below meanwhile.
        let _quiet = crate::host::tests::no_submissions();
        let p = crate::watch::tests::pool(2);
        let len = 2 * PAGE;
        register(p, p as u64, len as u64);
        let (Some(copy), Some(others), Some(memory)) = (find(p as u64, len as u64), bitmap(false), bitmap(true)) else {
            unregister(p, p as u64);
            return; // no host GPU here
        };
        let base = copy.marks.expect("tracked memory gets mark bits") as usize;
        copy.update(p, len);
        copy.writing(p, len, true);
        copy.write_host((p + 10) as u64, &[0xaa]);
        copy.write_host((p + PAGE + 10) as u64, &[0xbb]);
        // Page 0 marked by another shader, page 1 by one using title memory.
        for (b, n) in [(others, base), (memory, base + 1)] {
            // SAFETY: a dword of the mapped bitmap.
            unsafe { *(b.data as *mut u32).add(n / 32) |= 1 << (n % 32) };
        }
        copy_back_marks(&[], true);
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
            c.update(start as usize, n as usize);
            c.writing(start as usize, n as usize, false);
            c.write_host(p as u64 + 16, &[0x22]);
            c.keep_writes(start as usize, n as usize);
        }
        crate::memory::drop_withheld(p + 16, 4);
        assert_eq!((byte(&outer), byte(&inner)), (0x22, 0x22), "frame N: the GPU's copies keep it");
        crate::memory::log_frame();
        for c in [&outer, &inner] {
            let (start, n) = c.range();
            c.update(start as usize, n as usize);
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
        // No submission works through the copies meanwhile.
        let _quiet = crate::host::tests::no_submissions();
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
        copy.update(p, len);
        // Arms both pages: page 0 is wholly busy; page 1 past +16 is free.
        copy.writing(p, PAGE + 16, true);
        for (at, v) in [(20, 0x55), (24, 0x44), (40, 0x11), (PAGE + 20, 0x55)] {
            write(at, v);
        }
        assert!(copy.update(p + 16, 16) > 0 && copy.update(p + PAGE + 16, 16) > 0);
        copy.write_host((p + 10) as u64, &[0xaa]);
        copy.write_host((p + 24) as u64, &[0x00]);
        copy.write_host((p + PAGE + 10) as u64, &[0xbb]);
        write(20, 0x66);
        write(PAGE + 20, 0x77);
        copy.copy_back_marked(p);
        copy.copy_back_marked(p + PAGE);
        // SAFETY: bytes of our pool.
        let read = |at: usize| unsafe { ((p + at) as *const u8).read_volatile() };
        assert_eq!(
            [read(10), read(20), read(24), read(40)],
            [0xaa, 0x66, 0x00, 0x11]
        );
        assert_eq!([read(PAGE + 10), read(PAGE + 20)], [0xbb, 0x77]);
        copy.copy_back(p, PAGE + 16);
        unregister(p, p as u64);
    }

    /// Many marked pages, in two copies of the same memory (a buffer and
    /// one placed inside it): every shader write reaches title memory.
    #[test]
    fn many_marked_pages_all_go_back() {
        const PAGES: usize = 96;
        let p = crate::watch::tests::pool(PAGES);
        let len = PAGES * PAGE;
        register(p, p as u64, len as u64);
        register(p + 1, (p + len / 2) as u64, (len / 2) as u64);
        let (Some(outer), Some(inner), Some(b)) = (
            find(p as u64, len as u64),
            find((p + len / 2) as u64, (len / 2) as u64),
            bitmap(false),
        ) else {
            unregister(p + 1, (p + len / 2) as u64);
            unregister(p, p as u64);
            return; // no host GPU here
        };
        outer.update(p, len);
        inner.update(p + len / 2, len / 2);
        let words = b.data as *mut u32;
        let mark = |n: usize| {
            // SAFETY: a dword of the mapped bitmap.
            unsafe { *words.add(n / 32) |= 1 << (n % 32) };
        };
        // Even pages through the outer copy, odd pages of the upper half
        // through the inner one.
        let mut written = Vec::new();
        for page in 0..PAGES {
            let at = p + page * PAGE + 10;
            let (copy, first) = if page >= PAGES / 2 && page % 2 == 1 {
                (&inner, PAGES / 2)
            } else {
                (&outer, 0)
            };
            copy.writing(p + page * PAGE, PAGE, true);
            copy.write_host(at as u64, &[page as u8 + 1]);
            mark(copy.marks.expect("tracked memory gets mark bits") as usize + page - first);
            written.push((at, page as u8 + 1));
        }
        copy_back_marks(&[], false);
        for (at, value) in written {
            // SAFETY: a byte of our pool.
            assert_eq!(unsafe { (at as *const u8).read_volatile() }, value, "{at:#x}");
        }
        outer.copy_back(p, len);
        inner.copy_back(p + len / 2, len / 2);
        unregister(p + 1, (p + len / 2) as u64);
        unregister(p, p as u64);
    }

    /// A marked copy-back advances the snapshot of the declared range over
    /// the same page with the copy, so that range's copy-back does not put
    /// the title's bytes it took back over a title write made since.
    #[test]
    fn a_declared_copy_back_after_a_marked_one_keeps_newer_title_writes() {
        let p = crate::watch::tests::pool(1);
        register(p, p as u64, PAGE as u64);
        let Some(copy) = find(p as u64, PAGE as u64) else {
            unregister(p, p as u64);
            return; // no host GPU here
        };
        // SAFETY: bytes of our pool.
        let write = |at: usize, v: u8| unsafe { ((p + at) as *mut u8).write_volatile(v) };
        // SAFETY: bytes of our pool.
        let read = |at: usize| unsafe { ((p + at) as *const u8).read_volatile() };
        copy.update(p, PAGE);
        copy.writing(p, PAGE, true);
        copy.writing(p, PAGE, false);
        // A shader writes byte 10; the title writes byte 40 meanwhile.
        copy.write_host((p + 10) as u64, &[0xaa]);
        write(40, 0x11);
        assert!(copy.copy_back_marked(p) > 0);
        // The title writes both bytes again before the declared copy-back.
        write(10, 0x22);
        write(40, 0x23);
        assert_eq!(copy.copy_back(p, PAGE), 0);
        assert_eq!([read(10), read(40)], [0x22, 0x23]);
        unregister(p, p as u64);
    }

    /// Only the last bit of the bitmap is the sink for offsets past it; the
    /// bits before it in its last dword belong to copies.
    #[test]
    fn marks_in_the_last_dword_are_read_but_the_sink_bit_is_not() {
        let last = victory_shader::emit::VIEW_WRITE_BITS - 1;
        let mut bits = vec![0u32; (victory_shader::emit::VIEW_WRITE_BITS / 32) as usize];
        let w = bits.len() - 1;
        bits[w] = (1 << 31) | (1 << 30) | 1;
        assert_eq!(drain_marks(&mut bits, last), [last - 31, last - 1]);
        assert_eq!(bits[w], 0, "the sink bit is cleared too");
    }

    /// As above with the refreshes queued ([`crate::copy_pool::queued`], as
    /// `Batch::flush` does): once made, what they took goes into the saved
    /// bytes too ([`Shadow::queued_made`]).
    #[test]
    fn queued_partial_refresh_under_marked_range_updates_saved_bytes() {
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
        copy.update(p, len);
        copy.writing(p, PAGE + 16, true);
        for (at, v) in [(20, 0x55), (24, 0x44), (40, 0x11), (PAGE + 20, 0x55)] {
            write(at, v);
        }
        let (copied, pieces) = crate::copy_pool::queued(|| {
            copy.update(p + 16, 16) + copy.update(p + PAGE + 16, 16)
        });
        assert!(copied > 0 && !pieces.is_empty(), "queued");
        crate::copy_pool::run(pieces);
        copy.queued_made();
        copy.write_host((p + 10) as u64, &[0xaa]);
        copy.write_host((p + 24) as u64, &[0x00]);
        copy.write_host((p + PAGE + 10) as u64, &[0xbb]);
        write(20, 0x66);
        write(PAGE + 20, 0x77);
        copy.copy_back_marked(p);
        copy.copy_back_marked(p + PAGE);
        // SAFETY: bytes of our pool.
        let read = |at: usize| unsafe { ((p + at) as *const u8).read_volatile() };
        assert_eq!(
            [read(10), read(20), read(24), read(40)],
            [0xaa, 0x66, 0x00, 0x11]
        );
        assert_eq!([read(PAGE + 10), read(PAGE + 20)], [0xbb, 0x77]);
        copy.copy_back(p, PAGE + 16);
        unregister(p, p as u64);
    }

    /// Unmapping another mapping of an armed page leaves the copy's mapping
    /// guarded and clean: the shaders' marked writes still go back, and a
    /// title write to the surviving mapping still saves the page's old bytes
    /// so both writers survive.
    #[test]
    fn remapping_an_alias_does_not_drop_marked_writes_on_the_surviving_copy() {
        let (a, b) = crate::watch::tests::twin_views(2);
        let len = 2 * PAGE;
        register(a, a as u64, len as u64);
        let Some(copy) = find(a as u64, len as u64) else {
            unregister(a, a as u64);
            return; // no host GPU here
        };
        copy.update(a, len);
        copy.writing(a, len, true);
        // The shaders write a byte in each page; the runtime unmaps the
        // other mapping of both pages while the submission runs.
        copy.write_host((a + 10) as u64, &[0xaa]);
        copy.write_host((a + PAGE + 20) as u64, &[0xbb]);
        crate::watch::title_memory_remapped(b, len);
        // The title then writes page 1 through the surviving mapping: it
        // must still be guarded so the page's old bytes are saved first.
        // SAFETY: a byte of our surviving mapping.
        unsafe { ((a + PAGE + 40) as *mut u8).write_volatile(0x11) };
        assert!(
            copy.copy_back_marked(a) > 0,
            "page 0's shader write goes back"
        );
        assert!(
            copy.copy_back_marked(a + PAGE) > 0,
            "page 1's shader write goes back"
        );
        // SAFETY: bytes of our surviving mapping.
        let read = |at: usize| unsafe { ((a + at) as *const u8).read_volatile() };
        assert_eq!(read(10), 0xaa, "page 0 kept the shader's write");
        assert_eq!(
            (read(PAGE + 20), read(PAGE + 40)),
            (0xbb, 0x11),
            "page 1 kept both writers' bytes"
        );
        copy.copy_back(a, len);
        unregister(a, a as u64);
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

    const MIB: u64 = 1 << 20;

    fn entry(start: u64, len: u64) -> Entry {
        Entry {
            end: start + len,
            shadow: None,
        }
    }

    /// A 256 MiB parent with `children` 64 KiB buffers placed in its first
    /// part, as a heap's placed buffers.
    fn heap(base: u64, children: u64) -> BTreeMap<(u64, usize), Entry> {
        let mut map = BTreeMap::new();
        map.insert((base, 1), entry(base, 256 * MIB));
        for i in 0..children {
            map.insert((base + i * 0x10000, 100 + i as usize), entry(base + i * 0x10000, 0x10000));
        }
        map
    }

    #[test]
    fn a_parent_is_found_past_many_children() {
        let base = 0x10_0000_0000;
        let map = heap(base, 200);
        let span = 256 * MIB;
        // Past all 200 children (and the 64 entries looked at before).
        let at = base + 200 * 0x10000 + 0x100;
        assert_eq!(find_buffer(&map, span, at, at + 16), Some((base, 1)));
        let first_64 = map.range(..=(at, usize::MAX)).rev().take(64).find(|(_, e)| e.end >= at + 16);
        assert!(first_64.is_none(), "the parent is more than 64 entries back");
        // Near the parent's end.
        let at = base + 256 * MIB - 16;
        assert_eq!(find_buffer(&map, span, at, at + 16), Some((base, 1)));
        // Inside a child: the child, the innermost.
        let at = base + 150 * 0x10000 + 8;
        assert_eq!(find_buffer(&map, span, at, at + 8), Some((base + 150 * 0x10000, 250)));
        // Across two children: only the parent holds it.
        let at = base + 151 * 0x10000 - 8;
        assert_eq!(find_buffer(&map, span, at, at + 16), Some((base, 1)));
        // Past the parent, and before it.
        let at = base + 256 * MIB;
        assert_eq!(find_buffer(&map, span, at, at + 1), None);
        assert_eq!(find_buffer(&map, span, base - 16, base + 16), None);
        // Longer than any buffer: none, without looking.
        assert_eq!(find_buffer(&map, span, base, base + span + 1), None);
        assert_eq!(find_buffer(&map, 0, base, base + 1), None);
    }

    #[test]
    fn registered_buffers_widen_the_search() {
        // High above anything a test maps; resource numbers no live
        // resource has.
        let base = 0x7f00_0000_0000;
        let len = 512 * MIB;
        register(1, base, len);
        for i in 0..100u64 {
            register(2 + i as usize, base + i * 0x1000, 0x1000);
        }
        let at = base + len - 64;
        let found = find_buffer(&buffers().map, buffers().max_size(), at, at + 64);
        for i in 0..100u64 {
            unregister(2 + i as usize, base + i * 0x1000);
        }
        unregister(1, base);
        assert_eq!(found, Some((base, 1)));
        let state = buffers();
        assert!(state.map.range((base, 0)..=(base + len, usize::MAX)).next().is_none());
    }

    #[test]
    fn find_reaches_covering_buffer_past_many_inner_buffers() {
        let p = crate::watch::tests::pool(4);
        let base = p as u64;
        let outer_len = (4 * PAGE) as u64;

        register(p, base, outer_len);
        for i in 0..128 {
            register(p + 1 + i, base + (i as u64) * 256, 128);
        }

        let target = base + (3 * PAGE) as u64;
        {
            let state = buffers();
            assert_eq!(
                find_buffer(&state.map, state.max_size(), target, target + 64),
                Some((base, p))
            );
        }
        if let Some(outer) = find(target, 64) {
            assert_eq!(outer.range(), (base, outer_len));
        }

        for i in 0..128 {
            unregister(p + 1 + i, base + (i as u64) * 256);
        }
        unregister(p, base);
    }

    #[test]
    fn find_stops_at_max_size_bound_and_tightens_on_unregister() {
        let mut state = Buffers::new();
        let base = 0x7e00_0000_0000_u64;
        let huge = 64 << 20;
        let small = 256;

        state.insert(1, base, huge);
        state.insert(2, base + 0x1000, small);
        assert_eq!(state.max_size(), huge);
        assert_eq!(
            find_buffer(&state.map, state.max_size(), base + 0x2000, base + 0x2040),
            Some((base, 1))
        );

        // Removing the largest buffer tightens `max_size` to the remaining buffers.
        assert!(state.remove(1, base).is_some());
        assert_eq!(state.max_size(), small);
        assert_eq!(
            find_buffer(&state.map, state.max_size(), base + 0x2000, base + 0x2040),
            None
        );

        // Entries starting further back than `max_size` are not visited on a miss.
        let mut stale = BTreeMap::new();
        stale.insert(
            (base, 3),
            Entry {
                end: base + 1024,
                shadow: None,
            },
        );
        assert_eq!(find_buffer(&stale, small, base + 512, base + 576), None);

        // Also exercise global `register` / `unregister` tightening `max_size`.
        register(0x7e00_0001, base, huge);
        assert!(buffers().max_size() >= huge);
        unregister(0x7e00_0001, base);
        assert!(buffers().max_size() < huge);
    }

    #[test]
    fn max_size_tracks_equal_sizes_replacements_and_descending_removals() {
        let mut state = Buffers::new();
        let base = 0x7d00_0000_0000_u64;
        assert_eq!(state.max_size(), 0);

        // Multiple equal maximum-size buffers keep `max_size()` pinned until the
        // last one is removed, then fall back cleanly to the next largest size.
        state.insert(1, base, 512);
        state.insert(2, base + 0x1000, 4096);
        state.insert(3, base + 0x2000, 4096);
        state.insert(4, base + 0x3000, 4096);
        assert_eq!(state.max_size(), 4096);

        assert!(state.remove(2, base + 0x1000).is_some());
        assert_eq!(state.max_size(), 4096);
        assert!(state.remove(3, base + 0x2000).is_some());
        assert_eq!(state.max_size(), 4096);
        assert!(state.remove(4, base + 0x3000).is_some());
        assert_eq!(state.max_size(), 512);

        // In-place replacement: equal size is a no-op, growing raises the max,
        // shrinking lowers it immediately to the exact remaining maximum.
        state.insert(5, base + 0x4000, 2048);
        assert_eq!(state.max_size(), 2048);
        state.insert(5, base + 0x4000, 2048);
        assert_eq!(state.max_size(), 2048);
        state.insert(5, base + 0x4000, 8192);
        assert_eq!(state.max_size(), 8192);
        state.insert(5, base + 0x4000, 1024);
        assert_eq!(state.max_size(), 1024);

        // Removing distinct sizes in strictly descending order steps down to 0.
        state.insert(6, base + 0x5000, 256);
        for (resource, addr, next_max) in [
            (5, base + 0x4000, 512),
            (1, base, 256),
            (6, base + 0x5000, 0),
        ] {
            assert!(state.remove(resource, addr).is_some());
            assert_eq!(state.max_size(), next_max);
        }
        assert!(state.map.is_empty());
        assert!(state.sizes.is_empty());
        assert_eq!(state.max_size(), 0);
    }

    /// Overlapping copy-backs within a single submission must advance the
    /// saved `before` snapshot after `merge_writeback` so a second overlapping
    /// copy-back does not mistake earlier CPU writes (copied into `host`) for
    /// GPU writes and clobber newer CPU writes.
    #[test]
    fn overlapping_copy_backs_do_not_clobber_newer_cpu_writes() {
        let cases = [
            ("partial_then_wider", (0, 0x1000), (0, 0x2000), 0),
            ("partial_then_whole_page", (0x1000, 0x1000), (0, PAGE), 0x1000),
        ];
        for (label, (off1, len1), (off2, len2), probe) in cases {
            let p = crate::watch::tests::pool(2);
            let len = 2 * PAGE;
            let write_title = |at: usize, v: u8| {
                // SAFETY: bytes inside our allocated test pool.
                unsafe { ((p + at) as *mut u8).write_volatile(v) };
            };
            let read_title = |at: usize| {
                // SAFETY: bytes inside our allocated test pool.
                unsafe { ((p + at) as *const u8).read_volatile() }
            };
            write_title(probe, 0x11);
            write_title(probe + 1, 0x11);
            register(p, p as u64, len as u64);
            let Some(copy) = find(p as u64, len as u64) else {
                unregister(p, p as u64);
                return; // no host GPU here
            };
            copy.update(p, len);
            copy.writing(p + off1, len1, false);
            copy.writing(p + off2, len2, false);
            // GPU writes byte `probe`; CPU concurrently writes byte `probe + 1`.
            copy.write_host((p + probe) as u64, &[0xAA]);
            write_title(probe + 1, 0x22);
            copy.copy_back(p + off1, len1);
            assert_eq!(read_title(probe), 0xAA, "{label}: first GPU writeback");
            assert_eq!(read_title(probe + 1), 0x22, "{label}: first CPU write preserved");

            // Title writes a newer byte to `probe + 1` before the overlapping
            // second copy-back of the same submission runs.
            write_title(probe + 1, 0x33);
            copy.copy_back(p + off2, len2);
            assert_eq!(read_title(probe), 0xAA, "{label}: GPU write stays");
            assert_eq!(
                read_title(probe + 1),
                0x33,
                "{label}: newer CPU write must not be clobbered by stale before snapshot"
            );
            unregister(p, p as u64);
        }
    }

    /// Copies whose mark bits reach the bitmap's last dword (cleared whole for
    /// the overflow sink bit) are not treated as having complete marks.
    #[test]
    fn marks_below_last_word_excludes_the_sink_dword() {
        let limit = victory_shader::emit::VIEW_WRITE_BITS - 32;
        assert!(marks_below_last_word(0, 1));
        assert!(
            marks_below_last_word(limit - 1, 1),
            "last bit before final dword"
        );
        assert!(
            marks_below_last_word(limit - 4, 4),
            "run ending just before final dword"
        );
        assert!(!marks_below_last_word(limit, 1), "first bit of final dword");
        assert!(
            !marks_below_last_word(limit - 1, 2),
            "run crossing into final dword"
        );
        assert!(!marks_below_last_word(u32::MAX, 1), "overflow-safe");
    }
}
