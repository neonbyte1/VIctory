//! Which title memory changed since a host copy took it (decision 0007,
//! "Copies of title memory" and "Change tracking"). Each host copy of title
//! memory (`crate::shadow`) and each texture's staging (`crate::texture`)
//! tracks its memory ([`Tracked`]): the 64 KiB pages it has not taken since
//! they last changed. Taking pages makes their blocks read-only (guards);
//! the first write to a guarded block faults into our vectored handler,
//! which gives write access back and marks the block changed in every copy
//! holding it. Writers that do not fault say so: our own and the kernel's
//! writes into title memory ([`writing`], before they write), and the
//! runtime mapping or unmapping memory ([`title_memory_remapped`]). So
//! bringing a copy up to date costs what changed since, not how much memory
//! it covers: no system call for memory nobody wrote.
//!
//! Only the runtime's GPU memory is tracked: its write-watched pools, its
//! physical-page views and their placeholders. Other memory a copy takes
//! (the C heap, where the kernel writes system call results into buffers a
//! guard would make fail) is copied whole every time.
//!
//! Blocks that cannot be made read-only (placeholders, nothing mapped)
//! read as zeros and count as taken: only a mapping changes them, which the
//! runtime reports.
//!
//! Physical pages may be mapped at several addresses (the title fills
//! pages through a pool view and maps them into its heaps), so a write at
//! one is a change at all of them: the runtime says where else memory is
//! mapped whenever it maps it ([`title_memory_aliased`]); taking a page
//! guards every address showing it, and a write at any of them marks the
//! trackers at all. (Asking the runtime at each take instead waited for
//! its lock, which the title's page mappings hold: 4 s per world frame.)
//! DirectStorage writing pages through views of the runtime's own reports
//! the addresses showing them as written.
//!
//! shortcut: a system call writing into a guarded block fails; the known
//! ones (file reads, DirectStorage) open their blocks first. Memory freed
//! and allocated again while a tracker holds it is not seen.

use std::collections::{BTreeMap, BTreeSet};
use std::sync::atomic::{AtomicU8, AtomicU32, AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

/// The unit tracked and copied: 64 KiB, the console's page. Finer units
/// made the checks of the world's big buffers cost more than the copies
/// they saved.
pub(crate) const PAGE: usize = 64 << 10;

/// Per page (its number folded onto the table), a count bumped under the
/// guard lock whenever the page stops being guarded: bytes read of a guarded
/// page without the lock are still its bytes if, back under the lock, the
/// page is guarded and its count did not move (pages sharing a count only
/// make the check fail more often). Write faults read the bytes they save
/// this way: the guard lock is not held across a 64 KiB read.
static OPENED: [AtomicU32; OPENED_COUNTS] = [const { AtomicU32::new(0) }; OPENED_COUNTS];
const OPENED_COUNTS: usize = 1 << 12;

fn opened(page: usize) -> &'static AtomicU32 {
    &OPENED[page % OPENED_COUNTS]
}

/// The pages a tracker has not taken since they changed, what it saves of
/// pages it armed, its bit in [`DIRTY`] (none: always taken as dirty), and
/// whether it is a host buffer copy's (the only trackers [`retake`] marks).
struct Changed(Mutex<BTreeSet<usize>>, Mutex<Saved>, Option<u32>, bool);

/// A bit per tracker, set while it holds changed pages: set and cleared
/// under the tracker's lock, read without one ([`Tracked::clean`]). Dense,
/// so that a submission asking about every copy its heaps reach (~14,000)
/// reads a few KB instead of a cache line per copy.
static DIRTY: [AtomicU64; DIRTY_WORDS] = [const { AtomicU64::new(0) }; DIRTY_WORDS];
const DIRTY_WORDS: usize = 1 << 14;

/// Bits of [`DIRTY`] given back, and the first never handed out.
static FREE_BITS: Mutex<(Vec<u32>, u32)> = Mutex::new((Vec::new(), 0));

fn dirty_bit() -> Option<u32> {
    let mut free = FREE_BITS.lock().unwrap_or_else(|e| e.into_inner());
    if let Some(bit) = free.0.pop() {
        return Some(bit);
    }
    let bit = free.1;
    (bit < (DIRTY_WORDS * 64) as u32).then(|| {
        free.1 += 1;
        bit
    })
}

fn set_dirty(bit: Option<u32>, dirty: bool) {
    if let Some(bit) = bit {
        let (word, mask) = (&DIRTY[bit as usize / 64], 1u64 << (bit % 64));
        if dirty {
            word.fetch_or(mask, Ordering::Release);
        } else {
            word.fetch_and(!mask, Ordering::Release);
        }
    }
}

/// Whether the tracker with bit `bit` holds no changed page (see
/// [`Tracked::clean`]).
pub(crate) fn clean_bit(bit: Option<u32>) -> bool {
    bit.is_some_and(|bit| DIRTY[bit as usize / 64].load(Ordering::Acquire) & 1 << (bit % 64) == 0)
}

impl Drop for Changed {
    fn drop(&mut self) {
        if let Some(bit) = self.2 {
            set_dirty(Some(bit), false);
            FREE_BITS.lock().unwrap_or_else(|e| e.into_inner()).0.push(bit);
        }
    }
}

/// Pages (page numbers) whose bytes a tracker keeps when they are first
/// written after it took them, and those bytes: while shaders may be
/// writing its copy of them, what it held before (`crate::shadow`'s
/// copy-back of marked pages), with what the copy took of the page since
/// laid over them ([`Tracked::refreshed`]).
#[derive(Default)]
struct Saved {
    /// The armed pages as runs `[first, last)`, sorted, apart and not
    /// touching (runs armed that meet merge): a root view declares its
    /// copy from its address to the end, the world's 174 MB heap many
    /// times a submission, and a set entry per page cost 30 microseconds a
    /// declaration; descriptor tables arm many sub-ranges end to end, and
    /// the write handler checks a page under the guards' lock.
    armed: Vec<(usize, usize)>,
    /// Pages of those runs that are not armed any more.
    unarmed: BTreeSet<usize>,
    bytes: std::collections::HashMap<usize, Box<[u8]>>,
}

impl Saved {
    /// Whether page `page` is armed.
    fn armed(&self, page: usize) -> bool {
        // The first run ending after `page` is the only one that can hold it.
        let i = self.armed.partition_point(|&(_, last)| last <= page);
        self.armed.get(i).is_some_and(|&(first, _)| first <= page) && !self.unarmed.contains(&page)
    }

    /// Arms pages `[first, last)`.
    fn arm(&mut self, first: usize, last: usize) {
        if first >= last {
            return;
        }
        if !self.unarmed.is_empty() {
            let pages: Vec<usize> = self.unarmed.range(first..last).copied().collect();
            for page in pages {
                self.unarmed.remove(&page);
            }
        }
        // Runs `[lo, hi)` overlap or touch the new one: it replaces them.
        let lo = self.armed.partition_point(|&(_, b)| b < first);
        let hi = self.armed.partition_point(|&(a, _)| a <= last);
        if lo + 1 == hi && self.armed[lo].0 <= first && last <= self.armed[lo].1 {
            return;
        }
        let run = if lo < hi {
            (self.armed[lo].0.min(first), self.armed[hi - 1].1.max(last))
        } else {
            (first, last)
        };
        self.armed.splice(lo..hi, [run]);
    }
}

impl Changed {
    /// Page `page` was taken: what it saved of it is stale.
    fn taken(&self, page: usize) {
        self.1.lock().unwrap_or_else(|e| e.into_inner()).bytes.remove(&page);
    }

    /// Page `page` is about to change for the first time since it was
    /// taken: its bytes now, if armed for it; those of `read` if they are
    /// this page's (read before, still its bytes), else read here.
    fn save(&self, page: usize, read: &mut Read) {
        let mut saved = self.1.lock().unwrap_or_else(|e| e.into_inner());
        if !saved.armed(page) || saved.bytes.contains_key(&page) {
            return;
        }
        // Read before (the fault handler, without the guard lock), or here
        // with no look at lazy pages: called under the guard lock, which that
        // look takes (a lazy page has no access, and the read fails: no bytes
        // saved rather than a deadlock; the callers give lazy pages their
        // bytes before they mark them).
        let bytes = match read.take_if(|(p, _)| *p == page) {
            Some((_, bytes)) => bytes,
            None => match page_bytes(page) {
                Some(bytes) => bytes,
                None => return,
            },
        };
        saved.bytes.insert(page, bytes);
        #[cfg(feature = "diagnostics")]
        crate::copystat::add(crate::copystat::C::SavedPages, 1);
    }

    /// Whether a first write to page `page` would have it saved now.
    fn would_save(&self, page: usize) -> bool {
        let changed = self.0.lock().unwrap_or_else(|e| e.into_inner());
        let changed = changed.contains(&page);
        let saved = self.1.lock().unwrap_or_else(|e| e.into_inner());
        !changed && saved.armed(page) && !saved.bytes.contains_key(&page)
    }
}

/// Bytes read ahead of a save: (page, its bytes) (see [`Changed::save`]).
type Read = Option<(usize, Box<[u8]>)>;

/// Page `page`'s bytes now (None if it cannot be read).
fn page_bytes(page: usize) -> Option<Box<[u8]>> {
    let mut bytes = vec![0u8; PAGE].into_boxed_slice();
    // A save of the title's bytes, not a reader of what shaders wrote.
    #[cfg(feature = "diagnostics")]
    let _own = crate::census::as_kind(crate::census::Kind::Own, 0);
    // With no look at lazy pages: callers may hold the guard lock.
    crate::host::read_raw((page * PAGE) as *const u8, &mut bytes).then_some(bytes)
}

/// Byte runs `[start, end)` in order.
pub(crate) type Runs = Vec<(usize, usize)>;

/// Title memory a host copy or texture staging takes: bytes `[start, end)`.
pub(crate) struct Tracked {
    start: usize,
    end: usize,
    changed: Arc<Changed>,
    /// Parts of the range on pages that are not the runtime's memory (in
    /// order): not tracked, copied whole at each use ([`Tracked::untracked`]).
    untracked: Vec<(usize, usize)>,
}

struct Guards {
    /// Blocks (pages) we made read-only.
    protected: BTreeSet<usize>,
    /// The trackers holding each page.
    holders: BTreeMap<usize, Vec<Arc<Changed>>>,
    /// Pages `[first, last)` of writes the guards do not see that are under
    /// way ([`Writing`]): not guarded until they end, or the write fails.
    busy: Vec<(usize, usize)>,
    /// Pages showing the same physical pages elsewhere, both ways, as the
    /// runtime told when it mapped them: a write at one marks the others,
    /// with no question to the runtime (whose lock its page mappings hold
    /// for long).
    alias_of: BTreeMap<usize, Vec<usize>>,
    /// View pages whose title bytes are older than a host copy's
    /// ([`Lease::lazify`]): no access until the title or our host first
    /// touches them.
    lazy: BTreeMap<usize, LazyPage>,
    /// Pages being written through views the guards do not see
    /// ([`unseen_writing`]), with how many such writes each: never made lazy.
    held: BTreeMap<usize, u32>,
    /// Tracked pages written this frame and the frame before (diagnostics).
    #[cfg(feature = "diagnostics")]
    written: BTreeSet<usize>,
    #[cfg(feature = "diagnostics")]
    written_before: BTreeSet<usize>,
}

static GUARDS: Mutex<Guards> = Mutex::new(Guards {
    protected: BTreeSet::new(),
    holders: BTreeMap::new(),
    busy: Vec::new(),
    alias_of: BTreeMap::new(),
    lazy: BTreeMap::new(),
    held: BTreeMap::new(),
    #[cfg(feature = "diagnostics")]
    written: BTreeSet::new(),
    #[cfg(feature = "diagnostics")]
    written_before: BTreeSet::new(),
});

#[cfg(not(feature = "diagnostics"))]
type GuardsHeld = std::sync::MutexGuard<'static, Guards>;

#[cfg(not(feature = "diagnostics"))]
fn guards() -> GuardsHeld {
    GUARDS.lock().unwrap_or_else(|e| e.into_inner())
}

#[cfg(feature = "diagnostics")]
type GuardsHeld = Held;

/// The guard lock, timed while held by the line that took it
/// (diagnostics: [`held_times`]), and counted for the thread taking it
/// ([`thread_counts`]).
#[cfg(feature = "diagnostics")]
#[track_caller]
fn guards() -> Held {
    let line = std::panic::Location::caller().line();
    let guard = GUARDS.lock().unwrap_or_else(|e| e.into_inner());
    COUNTS.with(|c| c[0].set(c[0].get() + 1));
    Held(Some(guard), std::time::Instant::now(), line)
}

#[cfg(feature = "diagnostics")]
std::thread_local! {
    /// Per thread (diagnostics): rounds of the guard lock taken, and
    /// protection changes made (`VirtualProtect` calls). No destructor, so
    /// the fault handler may touch them.
    static COUNTS: [std::cell::Cell<u64>; 2] = const { [std::cell::Cell::new(0), std::cell::Cell::new(0)] };
    /// The caller the thread's protection changes count for ([`Site`]).
    static SITE: std::cell::Cell<u8> = const { std::cell::Cell::new(0) };
}

/// Who asked for a protection change (diagnostics census, [`site_counts`]):
/// the outermost entry that named itself on the thread (0: none did).
#[cfg(feature = "diagnostics")]
#[derive(Clone, Copy)]
pub(crate) enum Site {
    /// The title's write fault ([`install`]'s handler).
    Handler = 1,
    /// [`writing`] and its drop: kernel reads and DirectStorage.
    Writing,
    WritingDrop,
    Remapped,
    Aliased,
    TrackerDrop,
    /// [`Tracked::take_split`] of a host buffer copy (refresh), of other
    /// trackers (textures, heaps).
    TakeCopy,
    TakeOther,
    /// [`Tracked::taken`] other than below.
    Taken,
    /// `crate::shadow`: marked pages page by page (its `writing` and
    /// `taken`), declared ranges' copy-back.
    MarkedPage,
    Declared,
    /// [`Lease::open`], [`Lease::guard`].
    LeaseOpen,
    LeaseGuard,
    /// [`Lease::lazify`]: view pages losing access until their first touch.
    Lazify,
}

#[cfg(feature = "diagnostics")]
const SITES: usize = 15;

#[cfg(feature = "diagnostics")]
pub(crate) const SITE_NAMES: [&str; SITES] = [
    "other",
    "fault handler",
    "kernel writes (writing)",
    "kernel writes ended",
    "remapped",
    "aliased",
    "tracker dropped",
    "refresh take (copies)",
    "take (textures, heaps)",
    "taken (other)",
    "marked page by page",
    "declared ranges back",
    "run open",
    "run guard",
    "made lazy",
];

/// Per [`Site`]: protection changes, pages they cover, range changes that
/// failed and went page by page.
#[cfg(feature = "diagnostics")]
static SITE_COUNTS: [[AtomicU64; 3]; SITES] = [const { [const { AtomicU64::new(0) }; 3] }; SITES];

/// Names the caller of the protection changes the thread makes until the
/// result drops, unless an outer one named itself (diagnostics).
#[cfg(feature = "diagnostics")]
pub(crate) struct AtSite(Option<u8>);

#[cfg(feature = "diagnostics")]
pub(crate) fn at_site(site: Site) -> AtSite {
    SITE.with(|s| {
        if s.get() == 0 {
            s.set(site as u8);
            AtSite(Some(0))
        } else {
            AtSite(None)
        }
    })
}

#[cfg(feature = "diagnostics")]
impl Drop for AtSite {
    fn drop(&mut self) {
        if let Some(old) = self.0 {
            SITE.with(|s| s.set(old));
        }
    }
}

#[cfg(feature = "diagnostics")]
fn count_protect(calls: u64, pages: u64) {
    COUNTS.with(|c| c[1].set(c[1].get() + calls));
    let site = SITE.with(std::cell::Cell::get) as usize;
    SITE_COUNTS[site][0].fetch_add(calls, Ordering::Relaxed);
    SITE_COUNTS[site][1].fetch_add(pages, Ordering::Relaxed);
}

#[cfg(feature = "diagnostics")]
fn count_fallback() {
    let site = SITE.with(std::cell::Cell::get) as usize;
    SITE_COUNTS[site][2].fetch_add(1, Ordering::Relaxed);
}

/// Per [`Site`] since the last call: protection changes, pages, failed
/// range changes (diagnostics).
#[cfg(feature = "diagnostics")]
pub(crate) fn site_counts() -> [[u64; 3]; SITES] {
    std::array::from_fn(|s| std::array::from_fn(|k| SITE_COUNTS[s][k].swap(0, Ordering::Relaxed)))
}

/// This thread's rounds of the guard lock and protection changes so far
/// (diagnostics): the caller subtracts two readings.
#[cfg(feature = "diagnostics")]
pub(crate) fn thread_counts() -> (u64, u64) {
    COUNTS.with(|c| (c[0].get(), c[1].get()))
}

#[cfg(feature = "diagnostics")]
struct Held(Option<std::sync::MutexGuard<'static, Guards>>, std::time::Instant, u32);

#[cfg(feature = "diagnostics")]
impl std::ops::Deref for Held {
    type Target = Guards;
    fn deref(&self) -> &Guards {
        self.0.as_ref().expect("held until dropped")
    }
}

#[cfg(feature = "diagnostics")]
impl std::ops::DerefMut for Held {
    fn deref_mut(&mut self) -> &mut Guards {
        self.0.as_mut().expect("held until dropped")
    }
}

/// Per line taking the guard lock: the line, microseconds held in all, the
/// longest hold (no allocation: the fault handler takes the lock).
#[cfg(feature = "diagnostics")]
static HELD: [[std::sync::atomic::AtomicU64; 3]; 16] =
    [const { [const { std::sync::atomic::AtomicU64::new(0) }; 3] }; 16];

#[cfg(feature = "diagnostics")]
impl Drop for Held {
    fn drop(&mut self) {
        use std::sync::atomic::Ordering::Relaxed;
        let us = self.1.elapsed().as_micros() as u64;
        drop(self.0.take());
        let line = u64::from(self.2);
        if let Some(slot) = HELD
            .iter()
            .find(|s| s[0].load(Relaxed) == line || s[0].compare_exchange(0, line, Relaxed, Relaxed).is_ok())
        {
            slot[1].fetch_add(us, Relaxed);
            slot[2].fetch_max(us, Relaxed);
        }
    }
}

/// Since the last call, per line of this file taking the guard lock: (line,
/// milliseconds held, longest hold in milliseconds), most held first.
#[cfg(feature = "diagnostics")]
pub(crate) fn held_times() -> Vec<(u64, f64, f64)> {
    use std::sync::atomic::Ordering::Relaxed;
    let mut out: Vec<(u64, f64, f64)> = HELD
        .iter()
        .filter(|s| s[0].load(Relaxed) != 0)
        .map(|s| (s[0].load(Relaxed), s[1].swap(0, Relaxed) as f64 / 1e3, s[2].swap(0, Relaxed) as f64 / 1e3))
        .collect();
    out.sort_by(|a, b| b.1.total_cmp(&a.1));
    out
}

/// (page, other page) for the pages (page numbers) showing the same memory
/// as `pages` elsewhere.
fn aliases(g: &Guards, pages: &[usize]) -> Vec<(usize, usize)> {
    pages
        .iter()
        .flat_map(|&p| g.alias_of.get(&p).into_iter().flatten().map(move |&a| (p, a)))
        .collect()
}

/// Guards the other mappings of taken pages, returning those that still
/// have an unguarded mapping (cannot be taken yet).
fn guard_aliases(g: &mut Guards, pairs: &[(usize, usize)]) -> BTreeSet<usize> {
    if pairs.is_empty() {
        return BTreeSet::new();
    }
    let others: Vec<usize> = pairs.iter().map(|&(_, a)| a).collect();
    let guarded: BTreeSet<usize> = protect(g, &others).into_iter().collect();
    pairs
        .iter()
        .filter(|(_, a)| !guarded.contains(a))
        .map(|&(p, _)| p)
        .collect()
}

/// Tracks `[start, start + len)` if it is the runtime's GPU memory (every
/// page changed at first); `None` for other memory, and with `views` false
/// for anything but its write-watched pools (physical-page views are
/// written through other views showing the same pages: the title fills
/// streamed textures and video planes through a pool view and maps the
/// pages into its tiled heap, where a guard sees none of it).
///
/// Only the pages of the range that are such memory are tracked; the rest is
/// never guarded and is copied whole at each use, as memory with no tracker.
pub(crate) fn track(start: usize, len: usize, views: bool) -> Option<Tracked> {
    track_as(start, len, views, false)
}

/// [`track`] for a host buffer copy: what shaders write lands in it, so a
/// [`retake`] of its memory marks it.
pub(crate) fn track_copy(start: usize, len: usize, views: bool) -> Option<Tracked> {
    track_as(start, len, views, true)
}

fn track_as(start: usize, len: usize, views: bool, copy: bool) -> Option<Tracked> {
    if len == 0 {
        return None;
    }
    let end = start + len;
    let pages = runtime_pages(start, end, views);
    if pages.is_empty() {
        return None;
    }
    install();
    // The rest of the range, as byte runs.
    let mut untracked: Vec<(usize, usize)> = Vec::new();
    let mut at = start;
    for &page in pages.iter().chain([&usize::MAX]) {
        let (a, b) = ((page.saturating_mul(PAGE)).max(start).min(end), (page.saturating_add(1).saturating_mul(PAGE)).min(end));
        if at < a {
            untracked.push((at, a));
        }
        at = at.max(b);
    }
    // A tracker with untracked parts is never clean: no bit.
    let bit = if untracked.is_empty() { dirty_bit() } else { None };
    set_dirty(bit, true);
    let changed = Arc::new(Changed(Mutex::new(pages.iter().copied().collect()), Mutex::default(), bit, copy));
    let mut g = guards();
    for &page in &pages {
        g.holders.entry(page).or_default().push(changed.clone());
    }
    drop(g);
    Some(Tracked {
        start,
        end,
        changed,
        untracked,
    })
}

/// The pages of `[start, end)` that are the runtime's GPU memory: a
/// write-watched pool, or (with `views`) anything but plain committed memory
/// and images (views, placeholders, nothing mapped yet). Decided for every
/// region the range covers, and a page touching any other memory is left
/// out: a range can run past its allocation (a 4 MiB window of memory
/// outside buffers, `crate::shadow`), and when the first page alone decided,
/// a window starting in free memory had the process heap's pages after it
/// guarded; the heap's own code then faulted while holding its lock, and the
/// guard handler waited for the guard lock, whose holder waited for the heap
/// lock (the crate's tests hung).
///
/// shortcut: memory free or reserved when the range is tracked counts as the
/// runtime's; another allocator committing it before the first take would
/// have it guarded.
fn runtime_pages(start: usize, end: usize, views: bool) -> Vec<usize> {
    use windows_sys::Win32::System::Memory::{
        GetWriteWatch, MEM_COMMIT, MEM_IMAGE, MEM_PRIVATE, MEMORY_BASIC_INFORMATION, VirtualQuery,
    };
    let watched = |at: usize, len: usize| {
        let (mut found, mut count, mut granularity) = (0usize, 1usize, 0u32);
        // SAFETY: room for one answer; no reset, so nothing changes.
        unsafe { GetWriteWatch(0, at as *const _, len, (&raw mut found).cast(), &mut count, &mut granularity) == 0 }
    };
    let (first, last) = (start / PAGE, end.div_ceil(PAGE));
    // One pool holds it all (the common case; asking `VirtualQuery` costs
    // milliseconds in the runtime's big pools).
    if watched(start & !(4096 - 1), end - (start & !(4096 - 1))) {
        return (first..last).collect();
    }
    let (mut runtime, mut other) = (std::collections::BTreeSet::new(), std::collections::BTreeSet::new());
    let mut at = start & !(4096 - 1);
    while at < end {
        // SAFETY: plain data filled in by the query.
        let mut info: MEMORY_BASIC_INFORMATION = unsafe { std::mem::zeroed() };
        // SAFETY: a valid out pointer of the right size.
        if unsafe { VirtualQuery(at as *const _, &mut info, size_of_val(&info)) } == 0 {
            break;
        }
        let next = (info.BaseAddress as usize + info.RegionSize).max(at + 1).min(end);
        let ours = watched(at, next - at)
            || views && info.Type != MEM_IMAGE && !(info.State == MEM_COMMIT && info.Type == MEM_PRIVATE);
        (if ours { &mut runtime } else { &mut other }).extend(at / PAGE..next.div_ceil(PAGE));
        at = next;
    }
    runtime.difference(&other).copied().collect()
}

impl Drop for Tracked {
    /// Pages no tracker holds any more get write access back, so that the
    /// guard set never names memory freed and allocated again; unless another
    /// mapping of the page is held: the guard here is that tracker's too (it
    /// was set when that tracker took its page), and with it opened a write
    /// here went unseen there.
    fn drop(&mut self) {
        #[cfg(feature = "diagnostics")]
        let _site = at_site(Site::TrackerDrop);
        let mut g = guards();
        // Lazy pages of its copy got their bytes as the copy went
        // ([`Tracked::resolve_all`]); any left keep title memory's.
        if any_lazy(self.start / PAGE, self.end.div_ceil(PAGE)) {
            let mine: Vec<usize> = g
                .lazy
                .range(self.start / PAGE..self.end.div_ceil(PAGE))
                .filter(|(_, l)| Arc::ptr_eq(&l.owner, &self.changed))
                .map(|(&p, _)| p)
                .collect();
            for page in mine {
                forget(&mut g, page);
            }
        }
        let mut free = Vec::new();
        for page in self.start / PAGE..self.end.div_ceil(PAGE) {
            if let Some(holders) = g.holders.get_mut(&page) {
                holders.retain(|c| !Arc::ptr_eq(c, &self.changed));
                if holders.is_empty() {
                    g.holders.remove(&page);
                    free.push(page);
                }
            }
        }
        for page in free {
            let watched_elsewhere = g.alias_of.get(&page).is_some_and(|o| o.iter().any(|p| g.holders.contains_key(p)));
            if !watched_elsewhere {
                open(&mut g, page);
            }
        }
    }
}

/// Gives write access back to page `page` if we guarded it.
fn open(g: &mut Guards, page: usize) -> bool {
    use windows_sys::Win32::System::Memory::{PAGE_READWRITE, VirtualProtect};
    if !g.protected.remove(&page) {
        return false;
    }
    opened(page).fetch_add(1, Ordering::Relaxed);
    let mut old = 0;
    // SAFETY: a block we made read-only (failing harmlessly if it is gone).
    unsafe { VirtualProtect((page * PAGE) as *const _, PAGE, PAGE_READWRITE, &mut old) };
    #[cfg(feature = "diagnostics")]
    {
        crate::copystat::add(crate::copystat::C::OpenPages, 1);
        count_protect(1, 1);
    }
    true
}

/// Pages of `[first, last)` guarded now have their bytes changed without
/// being opened: a write the guards do not see (`writing`, a remap, an alias
/// write, a copy-back through another mapping). Their count goes up, so a
/// write fault's read of one before the lock is discarded ([`OPENED`]): the
/// page may be guarded again by the time the lock is back, its bytes not
/// the ones read.
fn note_opened(g: &Guards, first: usize, last: usize) {
    for page in g.protected.range(first..last) {
        opened(*page).fetch_add(1, Ordering::Relaxed);
    }
}

impl Tracked {
    /// Takes the changed pages of `[from, to)`: guards them (writes from
    /// now on mark them again) and returns the byte ranges to copy, cut to
    /// `[from, to)`, in order. A page whose part in the tracked memory lies
    /// only partly in `[from, to)` is copied in part and stays changed.
    pub(crate) fn take(&self, from: usize, to: usize) -> Vec<(usize, usize)> {
        let (copy, zero) = self.take_split(from, to);
        let mut runs: Vec<(usize, usize)> = copy.into_iter().chain(zero).collect();
        runs.sort_unstable();
        let mut merged: Vec<(usize, usize)> = Vec::with_capacity(runs.len());
        for (a, b) in runs {
            match merged.last_mut() {
                Some((_, end)) if *end == a => *end = b,
                _ => merged.push((a, b)),
            }
        }
        merged
    }

    /// [`Tracked::take`], the runs on pages with nothing mapped apart: they
    /// read as zeros, with no copy to try (each such page cost a dozen
    /// failing kernel reads, 7-30 ms a frame in run perf4).
    pub(crate) fn take_split(&self, from: usize, to: usize) -> (Runs, Runs) {
        match self.take_split_unless(from, to, &[]) {
            Ok(taken) => taken,
            Err(_) => unreachable!("nothing is forbidden"),
        }
    }

    /// [`Tracked::take_split`], unless a changed page of `[from, to)` lies
    /// in one of `forbidden`'s page runs `[first, last)`: then nothing is
    /// taken, and the result is the largest key among the runs holding such
    /// pages. Decided on the pages changed when asked: one written after
    /// is left changed for a later take.
    pub(crate) fn take_split_unless(&self, from: usize, to: usize, forbidden: &[(usize, usize, u64)]) -> Result<(Runs, Runs), u64> {
        #[cfg(feature = "diagnostics")]
        let _site = at_site(if self.changed.3 { Site::TakeCopy } else { Site::TakeOther });
        let (from, to) = (from.max(self.start), to.min(self.end));
        if from >= to {
            return Ok((Vec::new(), Vec::new()));
        }
        let pages: Vec<usize> = {
            #[cfg(feature = "diagnostics")]
            let waited = Timed(std::time::Instant::now(), crate::copystat::C::GuardWaitUs);
            let changed = self.changed.0.lock().unwrap_or_else(|e| e.into_inner());
            #[cfg(feature = "diagnostics")]
            drop(waited);
            changed.range(from / PAGE..to.div_ceil(PAGE)).copied().collect()
        };
        if pages.is_empty() {
            return Ok((Vec::new(), Vec::new()));
        }
        if let Some(key) = forbidden
            .iter()
            .filter(|&&(first, last, _)| pages.iter().any(|p| (first..last).contains(p)))
            .map(|r| r.2)
            .max()
        {
            return Err(key);
        }
        let mut whole: Vec<usize> = pages
            .iter()
            .copied()
            .filter(|&p| (p * PAGE).max(self.start) >= from && ((p + 1) * PAGE).min(self.end) <= to)
            .collect();
        // Every address showing the same pages is guarded too. A write in
        // between marks them changed again; the copy, made after, has it.
        #[cfg(feature = "diagnostics")]
        let waited = Timed(std::time::Instant::now(), crate::copystat::C::GuardWaitUs);
        let mut g = guards();
        let mut changed = self.changed.0.lock().unwrap_or_else(|e| e.into_inner());
        #[cfg(feature = "diagnostics")]
        drop(waited);
        #[cfg(feature = "diagnostics")]
        let asked = Timed(std::time::Instant::now(), crate::copystat::C::AliasUs);
        let others = aliases(&g, &whole);
        let unguarded = guard_aliases(&mut g, &others);
        #[cfg(feature = "diagnostics")]
        drop(asked);
        whole.retain(|p| !unguarded.contains(p));
        let mut unmapped = Vec::new();
        for page in protect_noting(&mut g, &whole, &mut unmapped) {
            changed.remove(&page);
            self.changed.taken(page);
        }
        set_dirty(self.changed.2, !changed.is_empty());
        drop(changed);
        drop(g);
        let (mut copy, mut zero): (Runs, Runs) = (Vec::new(), Vec::new());
        for p in pages {
            let (a, b) = ((p * PAGE).max(from), ((p + 1) * PAGE).min(to));
            let runs = if unmapped.contains(&p) { &mut zero } else { &mut copy };
            match runs.last_mut() {
                Some((_, end)) if *end == a => *end = b,
                _ => runs.push((a, b)),
            }
        }
        Ok((copy, zero))
    }

    /// Whether page `page` (a page number) changed since it was taken.
    pub(crate) fn changed(&self, page: usize) -> bool {
        self.changed
            .0
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .contains(&page)
    }

    /// The caller made pages (page numbers) of its copy equal to title
    /// memory itself (what shaders wrote went back): taken and guarded.
    pub(crate) fn taken(&self, pages: &[usize]) {
        #[cfg(feature = "diagnostics")]
        let _site = at_site(Site::Taken);
        let mut g = guards();
        let mut changed = self.changed.0.lock().unwrap_or_else(|e| e.into_inner());
        let others = aliases(&g, pages);
        let unguarded = guard_aliases(&mut g, &others);
        let pages: Vec<usize> = pages
            .iter()
            .copied()
            .filter(|p| !unguarded.contains(p))
            .collect();
        for page in protect(&mut g, &pages) {
            changed.remove(&page);
            self.changed.taken(page);
        }
        set_dirty(self.changed.2, !changed.is_empty());
    }

    /// Whether no page changed since it was taken: then [`Tracked::take`]
    /// finds nothing, and asking costs no lock. A write racing this answer
    /// is found by the next asking, as one racing a take.
    pub(crate) fn clean(&self) -> bool {
        clean_bit(self.changed.2)
    }

    /// The tracker's bit for [`clean_bit`], for callers that ask about
    /// many trackers at once without touching each (none: never clean).
    pub(crate) fn bit(&self) -> Option<u32> {
        self.changed.2
    }

    /// The parts of `[from, to)` that are not tracked (see
    /// [`Tracked::untracked`] field): to be copied whole at each use.
    pub(crate) fn untracked(&self, from: usize, to: usize) -> Vec<(usize, usize)> {
        self.untracked
            .iter()
            .map(|&(a, b)| (a.max(from), b.min(to)))
            .filter(|&(a, b)| a < b)
            .collect()
    }

    /// Shaders may write the copy's bytes `[from, to)` until [`Tracked::disarm`]:
    /// pages there written next after they were taken have their bytes kept
    /// first ([`Tracked::saved`]).
    pub(crate) fn arm(&self, from: usize, to: usize) {
        let (from, to) = (from.max(self.start), to.min(self.end));
        if from < to {
            self.changed.1.lock().unwrap_or_else(|e| e.into_inner()).arm(from / PAGE, to.div_ceil(PAGE));
        }
    }

    /// Page `page` (a page number) is armed again after [`Tracked::unarm`]:
    /// shaders of a later submission may write it, and the copy-back just
    /// made it equal to title memory, so its next first write keeps its
    /// bytes.
    pub(crate) fn rearm(&self, page: usize) {
        self.changed.1.lock().unwrap_or_else(|e| e.into_inner()).unarmed.remove(&page);
    }

    /// Page `page` (a page number) needs no saving any more: its copy-back
    /// is under way.
    pub(crate) fn unarm(&self, page: usize) {
        let mut saved = self.changed.1.lock().unwrap_or_else(|e| e.into_inner());
        if saved.armed(page) {
            saved.unarmed.insert(page);
        }
    }

    /// No shader writes the copy any more: nothing is armed or kept.
    pub(crate) fn disarm(&self) {
        let mut saved = self.changed.1.lock().unwrap_or_else(|e| e.into_inner());
        saved.armed.clear();
        saved.unarmed.clear();
        saved.bytes.clear();
    }

    /// The bytes page `page` (a page number) held when it was first written
    /// since it was taken, if it was armed then (and so changed), with what
    /// the copy took of it since: the copy's bytes before the submission.
    pub(crate) fn saved(&self, page: usize) -> Option<Box<[u8]>> {
        self.changed.1.lock().unwrap_or_else(|e| e.into_inner()).bytes.remove(&page)
    }

    /// The copy took `bytes` at title address `from`: any saved bytes of pages
    /// it covers take them too, so marked copy-back neither takes a partial
    /// refresh for the GPU's nor drops a shader write back to the pre-fault byte.
    pub(crate) fn refreshed(&self, from: usize, bytes: &[u8]) {
        if bytes.is_empty() {
            return;
        }
        let mut saved = self.changed.1.lock().unwrap_or_else(|e| e.into_inner());
        if saved.bytes.is_empty() {
            return;
        }
        let to = from + bytes.len();
        for page in from / PAGE..to.div_ceil(PAGE) {
            if let Some(before) = saved.bytes.get_mut(&page) {
                let (a, b) = (from.max(page * PAGE), to.min((page + 1) * PAGE));
                before[a - page * PAGE..b - page * PAGE]
                    .copy_from_slice(&bytes[a - from..b - from]);
            }
        }
    }

    /// The copy goes (its bytes with it): its lazy pages get them now.
    pub(crate) fn resolve_all(&self) {
        let (first, last) = (self.start / PAGE, self.end.div_ceil(PAGE));
        if !any_lazy(first, last) {
            return;
        }
        let mut g = guards();
        let mine: Vec<usize> = g
            .lazy
            .range(first..last)
            .filter(|(_, l)| Arc::ptr_eq(&l.owner, &self.changed))
            .map(|(&p, _)| p)
            .collect();
        for page in mine {
            resolve(&mut g, page, Touch::Host);
        }
    }
}

/// One hold of the guard lock for the caller's own writes into a run of a
/// tracker's pages (`crate::shadow`'s copy-back of marked pages): one round
/// of the lock and one protection change each way per run of consecutive
/// pages, where [`writing`] and [`Tracked::taken`] cost three rounds and
/// two changes a page (2,829 pages a world frame, mostly waiting for the
/// lock, `CpuAnatomy` §3).
///
/// What it guarantees against the title's threads writing meanwhile:
/// - A page [`Lease::guarded`] reports guarded took no title write since it
///   was guarded: a write there faults, and the handler waits for this lock
///   before it lets the write land. Bytes the caller read from it before the
///   lease are still title memory's.
/// - A title write faulting while the lease is held (on a guarded page, or
///   on another mapping of an opened one, which stay guarded) lands after
///   the lease ends, so after the caller's writes; the handler then marks the
///   page changed after [`Lease::taken`] cleared it.
/// - A title write on a page [`Lease::open`] opened lands without a fault,
///   before or after the caller's: the caller writes only the bytes it
///   changes, and takes the page only if, once guarded again, title memory
///   is what the copy holds. Otherwise the page stays changed (marked by
///   `open`), and the next refresh takes the title's bytes.
/// - Pages opened only to bridge a run (not written) are ones this tracker
///   alone holds and nothing else shows: checked the same way once guarded
///   again, and marked changed ([`Lease::written`]) if the title wrote them.
///   An armed other tracker would have needed its bytes saved first; there
///   is none.
///
/// Lock order as elsewhere: the guard lock, then a tracker's change set,
/// then its saved bytes.
pub(crate) struct Lease<'a> {
    g: GuardsHeld,
    tracked: &'a Tracked,
}

impl Tracked {
    /// The guard lock, held for the caller's writes into this tracker's
    /// pages until dropped ([`Lease`]).
    pub(crate) fn lease(&self) -> Lease<'_> {
        Lease {
            g: guards(),
            tracked: self,
        }
    }
}

/// The consecutive pages of `pages` (ascending, no repeats) as runs
/// `[first, last)`.
fn page_runs(pages: &[usize]) -> impl Iterator<Item = (usize, usize)> + '_ {
    let mut i = 0;
    std::iter::from_fn(move || {
        let first = *pages.get(i)?;
        let mut last = first + 1;
        i += 1;
        while pages.get(i) == Some(&last) {
            last += 1;
            i += 1;
        }
        Some((first, last))
    })
}

impl Lease<'_> {
    /// Whether page `page` is guarded: the title wrote nothing there since
    /// it was, and a write of its now waits for the lease.
    pub(crate) fn guarded(&self, page: usize) -> bool {
        self.g.protected.contains(&page)
    }

    /// Whether this tracker alone holds page `page` and nothing else shows
    /// it: a title write there that no fault sees concerns no other tracker.
    pub(crate) fn alone(&self, page: usize) -> bool {
        !self.g.alias_of.contains_key(&page)
            && self
                .g
                .holders
                .get(&page)
                .is_some_and(|h| h.iter().all(|c| Arc::ptr_eq(c, &self.tracked.changed)))
    }

    /// Before the caller writes `write` (ascending, no repeats): marks them
    /// changed in every tracker holding them or another mapping of them
    /// (armed ones keep their bytes first, as for [`writing`]; the caller
    /// unarms its own), then gives write access back to them and to
    /// `bridge` (pages between them it will not write, unmarked: guarded
    /// ones it alone holds, [`Lease::alone`], or ones open already), one
    /// protection change per run of consecutive pages of both. The other
    /// mappings stay guarded: the write goes through this one, and the
    /// title's through them waits for the lease. Returns the protection
    /// changes.
    pub(crate) fn open(&mut self, write: &[usize], bridge: &[usize]) -> u64 {
        #[cfg(feature = "diagnostics")]
        let _site = at_site(Site::LeaseOpen);
        use windows_sys::Win32::System::Memory::{PAGE_READWRITE, VirtualProtect};
        let g = &mut *self.g;
        for (first, last) in page_runs(write) {
            mark_changed(g, first, last, true, &mut None);
        }
        for (_, other) in aliases(g, write) {
            mark_changed(g, other, other + 1, true, &mut None);
        }
        let mut all: Vec<usize> = write.iter().chain(bridge).copied().collect();
        all.sort_unstable();
        all.dedup();
        let mut calls = 0;
        for (first, last) in page_runs(&all) {
            if g.protected.range(first..last).next().is_none() {
                continue;
            }
            let mut old = 0;
            calls += 1;
            #[cfg(feature = "diagnostics")]
            count_protect(1, (last - first) as u64);
            // SAFETY: title memory of the copy; only its protection changes
            // (pages open already stay so).
            let opened = unsafe {
                VirtualProtect((first * PAGE) as *const _, (last - first) * PAGE, PAGE_READWRITE, &mut old) != 0
            };
            if opened {
                let guarded: Vec<usize> = g.protected.range(first..last).copied().collect();
                #[cfg(feature = "diagnostics")]
                crate::copystat::add(crate::copystat::C::OpenPages, guarded.len() as u64);
                for page in guarded {
                    // Out of `protected`, and its count goes up: a write
                    // fault that read the page before the lock drops what it
                    // read ([`OPENED`]'s check), also if the lease is over
                    // and the page guarded again by the time it looks.
                    g.protected.remove(&page);
                    self::opened(page).fetch_add(1, Ordering::Relaxed);
                }
            } else {
                // Across allocations: page by page.
                #[cfg(feature = "diagnostics")]
                count_fallback();
                for page in first..last {
                    calls += u64::from(open(g, page));
                }
            }
        }
        calls
    }

    /// Pages `pages` (ascending, no repeats) that [`Lease::open`] bridged
    /// changed while open (a title write no fault saw): changed in every
    /// tracker holding them or another mapping of them.
    pub(crate) fn written(&mut self, pages: &[usize]) {
        let g = &mut *self.g;
        for (first, last) in page_runs(pages) {
            mark_changed(g, first, last, false, &mut None);
        }
        for (_, other) in aliases(g, pages) {
            mark_changed(g, other, other + 1, false, &mut None);
        }
    }

    /// Before the caller writes `pages` (ascending, no repeats) through its
    /// own writable mapping of their memory ([`writable_alias`]): marks
    /// them changed in every tracker holding them or another mapping of
    /// them (armed ones keep their bytes first), as [`Lease::open`], but
    /// changes no protection. The title's mappings stay guarded: its writes
    /// there fault and wait for the lease, so they land after the caller's.
    pub(crate) fn will_write(&mut self, pages: &[usize]) {
        let g = &mut *self.g;
        for (first, last) in page_runs(pages) {
            mark_changed(g, first, last, true, &mut None);
        }
        for (_, other) in aliases(g, pages) {
            mark_changed(g, other, other + 1, true, &mut None);
        }
        // These stay guarded while their bytes change: a write fault reading
        // one before the lock must not keep what it read ([`note_opened`]).
        for (first, last) in page_runs(pages) {
            note_opened(g, first, last);
        }
    }

    /// After the caller's writes: guards `pages` (ascending, no repeats)
    /// again, one protection change per run of consecutive pages, with every
    /// other mapping of them. Returns the pages guarded now, leaving out
    /// those busy with a write the guards do not see, with nothing mapped, or
    /// with another mapping that cannot be guarded; and the protection
    /// changes asked.
    pub(crate) fn guard(&mut self, pages: &[usize]) -> (Vec<usize>, u64) {
        #[cfg(feature = "diagnostics")]
        let _site = at_site(Site::LeaseGuard);
        let g = &mut *self.g;
        let fresh: Vec<usize> = pages
            .iter()
            .copied()
            .filter(|&p| !g.protected.contains(&p) && !g.busy.iter().any(|&(a, b)| (a..b).contains(&p)))
            .collect();
        let calls = page_runs(&fresh).count() as u64;
        let mut unmapped = Vec::new();
        let mut taken = protect_noting(g, pages, &mut unmapped);
        taken.retain(|p| !unmapped.contains(p));
        let others = aliases(g, &taken);
        let unguarded = guard_aliases(g, &others);
        taken.retain(|p| !unguarded.contains(p));
        (taken, calls)
    }

    /// Pages `pages`, guarded (by [`Lease::guard`]), hold in the copy what
    /// title memory holds: taken, their saved bytes stale.
    pub(crate) fn taken(&mut self, pages: &[usize]) {
        let c = &self.tracked.changed;
        let mut changed = c.0.lock().unwrap_or_else(|e| e.into_inner());
        for &page in pages {
            changed.remove(&page);
            c.taken(page);
        }
        set_dirty(c.2, !changed.is_empty());
    }

    /// Instead of writing them back now, leaves `pages` lazy (view pages
    /// this tracker holds whole, whose copy shaders changed, and that title
    /// memory holds as before the submission: guarded and not changed since
    /// the copy took them). Each loses all access at every address showing
    /// it; its bytes stay in the copy (`src`) until the first touch of the
    /// title (a fault, read or write) or of our host ([`reading`],
    /// [`writing`], [`unseen_writing`], the runtime mapping or unmapping it,
    /// the copy going: [`Tracked::resolve_all`]), which writes them through
    /// the runtime's own mapping of the page (`alias`) while every title
    /// mapping still has no access, then guards it as any taken page
    /// ([`resolve`]). Until then shaders may write the copy again: a page
    /// written again before any touch is never copied in between (the
    /// caller sees it with [`Lease::lazy_own`]). Every other tracker holding
    /// the page or another mapping of it counts it changed (armed ones keep
    /// title memory's bytes first, as for a write).
    ///
    /// Returns the pages made lazy (ascending); the others are the caller's
    /// to write now. Left out: pages changed, not guarded, under a write the
    /// guards do not see ([`writing`], [`unseen_writing`]), lazy already,
    /// past [`LAZY_LIMIT`], or shown at another address that is not guarded,
    /// is so, or lies in this tracker's range too (two copies of one
    /// physical page in one copy).
    pub(crate) fn lazify(&mut self, pages: &[LazyView]) -> Vec<usize> {
        use windows_sys::Win32::System::Memory::{PAGE_NOACCESS, PAGE_READONLY};
        #[cfg(feature = "diagnostics")]
        let _site = at_site(Site::Lazify);
        let Lease { g, tracked } = self;
        let g = &mut **g;
        let owner = tracked.changed.clone();
        let own = tracked.start / PAGE..tracked.end.div_ceil(PAGE);
        #[cfg(feature = "diagnostics")]
        crate::copystat::add(crate::copystat::C::LazyOffered, pages.len() as u64);
        let offered: BTreeSet<usize> = pages.iter().map(|v| v.page).collect();
        let ok: Vec<&LazyView> = {
            let changed = owner.0.lock().unwrap_or_else(|e| e.into_inner());
            let blocked = |g: &Guards, q: usize| {
                q >= LAZY_LIMIT
                    || g.busy.iter().any(|&(a, b)| (a..b).contains(&q))
                    || g.held.contains_key(&q)
                    || g.lazy.contains_key(&q)
            };
            pages
                .iter()
                .filter(|v| {
                    let p = v.page;
                    let why = if blocked(g, p) || !own.contains(&p) {
                        Refused::Other
                    } else if changed.contains(&p) {
                        Refused::Changed
                    } else if !g.protected.contains(&p) {
                        Refused::Unguarded
                    } else if g.alias_of.get(&p).is_some_and(|others| {
                        others.iter().any(|&q| {
                            blocked(g, q) || !g.protected.contains(&q) || own.contains(&q) || offered.contains(&q)
                        })
                    }) {
                        Refused::Alias
                    } else {
                        return true;
                    };
                    refused(why);
                    false
                })
                .collect()
        };
        // Armed trackers keep title memory's bytes while they can be read.
        for v in &ok {
            mark_changed_but(g, v.page, &owner);
            for q in g.alias_of.get(&v.page).cloned().unwrap_or_default() {
                mark_changed(g, q, q + 1, true, &mut None);
            }
        }
        let entry = |v: &LazyView| LazyPage {
            src: v.src,
            alias: v.alias,
            owner: owner.clone(),
            #[cfg(any(test, feature = "diagnostics"))]
            print: print(v.alias),
        };
        let make = |g: &mut Guards, q: usize, e: LazyPage| {
            g.protected.remove(&q);
            g.lazy.insert(q, e);
            set_lazy_bit(q, true);
        };
        let mut made = Vec::with_capacity(ok.len());
        // Pages shown elsewhere one by one: at all addresses or none.
        let (alone, shown): (Vec<&LazyView>, Vec<&LazyView>) =
            ok.into_iter().partition(|v| !g.alias_of.contains_key(&v.page));
        for v in shown {
            let group: Vec<usize> = std::iter::once(v.page).chain(g.alias_of[&v.page].iter().copied()).collect();
            let hidden = group.iter().take_while(|&&q| set_protection(q, q + 1, PAGE_NOACCESS)).count();
            if hidden < group.len() {
                for &q in &group[..hidden] {
                    set_protection(q, q + 1, PAGE_READONLY);
                }
                refused(Refused::Other);
                continue;
            }
            let e = entry(v);
            for q in group {
                make(g, q, e.clone());
            }
            made.push(v.page);
        }
        // The others in runs of consecutive pages: one change each.
        let mut i = 0;
        while i < alone.len() {
            let mut j = i + 1;
            while j < alone.len() && alone[j].page == alone[j - 1].page + 1 {
                j += 1;
            }
            // Across views (one mapping per page, as the GPU's page
            // mappings make them) a range fails: page by page then.
            let whole = j - i > 1 && set_protection(alone[i].page, alone[j - 1].page + 1, PAGE_NOACCESS);
            for v in &alone[i..j] {
                if whole || set_protection(v.page, v.page + 1, PAGE_NOACCESS) {
                    make(g, v.page, entry(v));
                    made.push(v.page);
                } else {
                    refused(Refused::Other);
                }
            }
            i = j;
        }
        made.sort_unstable();
        #[cfg(feature = "diagnostics")]
        crate::copystat::add(crate::copystat::C::LazyMade, made.len() as u64);
        made
    }

    /// Whether page `page` (a page number) is lazy for this tracker with
    /// its copy's bytes at `src`: shaders wrote it again before anyone
    /// touched it, and what goes back at the first touch is in the copy.
    pub(crate) fn lazy_own(&self, page: usize, src: usize) -> bool {
        self.g
            .lazy
            .get(&page)
            .is_some_and(|l| l.src == src && Arc::ptr_eq(&l.owner, &self.tracked.changed))
    }

    /// Page `page`, if lazy, gets its copy's bytes now ([`resolve`]; our
    /// host's touch).
    pub(crate) fn resolve(&mut self, page: usize) -> bool {
        resolve(&mut self.g, page, Touch::Host)
    }
}

/// Makes `pages` read-only where not already (in runs), and returns those
/// now guarded or with nothing mapped (taken either way); a page that is
/// mapped but cannot be guarded is left out (it stays changed).
fn protect(g: &mut Guards, pages: &[usize]) -> Vec<usize> {
    protect_noting(g, pages, &mut Vec::new())
}

/// [`protect`], also listing in `unmapped` the pages taken because nothing
/// is mapped there.
fn protect_noting(g: &mut Guards, pages: &[usize], unmapped_pages: &mut Vec<usize>) -> Vec<usize> {
    #[cfg(feature = "diagnostics")]
    let counted = |calls: u64, pages: u64| {
        crate::copystat::add(crate::copystat::C::ProtectCalls, calls);
        crate::copystat::add(crate::copystat::C::ProtectPages, pages);
    };
    use windows_sys::Win32::System::Memory::{
        MEM_COMMIT, MEMORY_BASIC_INFORMATION, PAGE_READONLY, VirtualProtect, VirtualQuery,
    };
    #[cfg(feature = "diagnostics")]
    let _timed = Timed(std::time::Instant::now(), crate::copystat::C::ProtectUs);
    let set = |from: usize, to: usize| {
        #[cfg(feature = "diagnostics")]
        count_protect(1, (to - from) as u64);
        let mut old = 0;
        // SAFETY: title memory; only its protection changes.
        unsafe { VirtualProtect((from * PAGE) as *const _, (to - from) * PAGE, PAGE_READONLY, &mut old) != 0 }
    };
    let unmapped = |page: usize| {
        // SAFETY: plain data filled in by the query.
        let mut info: MEMORY_BASIC_INFORMATION = unsafe { std::mem::zeroed() };
        // SAFETY: a valid out pointer of the right size.
        let n = unsafe { VirtualQuery((page * PAGE) as *const _, &mut info, size_of_val(&info)) };
        n == 0 || info.State != MEM_COMMIT
    };
    let mut taken = Vec::with_capacity(pages.len());
    let busy = |page: usize| g.busy.iter().any(|&(a, b)| (a..b).contains(&page));
    let pages: Vec<usize> = pages.iter().copied().filter(|&p| !busy(p)).collect();
    let mut i = 0;
    // A lazy page has no access at all: guarded as it is (made read-only,
    // it would show its old bytes).
    let guarded = |g: &Guards, p: usize| g.protected.contains(&p) || g.lazy.contains_key(&p);
    while i < pages.len() {
        if guarded(g, pages[i]) {
            taken.push(pages[i]);
            i += 1;
            continue;
        }
        // A run of consecutive pages not guarded yet.
        let first = i;
        while i + 1 < pages.len() && pages[i + 1] == pages[i] + 1 && !guarded(g, pages[i + 1]) {
            i += 1;
        }
        let run = &pages[first..=i];
        i += 1;
        #[cfg(feature = "diagnostics")]
        counted(1, run.len() as u64);
        if set(run[0], run[run.len() - 1] + 1) {
            g.protected.extend(run);
            taken.extend(run);
            continue;
        }
        #[cfg(feature = "diagnostics")]
        if run.len() > 1 {
            count_fallback();
        }
        for &page in run {
            if set(page, page + 1) {
                g.protected.insert(page);
                taken.push(page);
            } else if unmapped(page) {
                taken.push(page);
                unmapped_pages.push(page);
            }
        }
    }
    taken
}

/// Pages `[first, last)` changed, and so the pages showing the same memory
/// elsewhere: gives write access back where we guarded them and marks them
/// in every tracker holding them. True if one of `[first, last)` was
/// guarded. Lazy pages among them get their copy's bytes first (`touch`
/// touched them): the write lands on those.
fn changed(g: &mut Guards, first: usize, last: usize, save: bool, read: &mut Read, touch: Touch) -> bool {
    let others: Vec<usize> = g.alias_of.range(first..last).flat_map(|(_, o)| o.iter().copied()).collect();
    resolve_range(g, first, last, touch);
    for &page in &others {
        resolve(g, page, touch);
    }
    let guarded = g.protected.range(first..last).next().is_some();
    #[cfg(feature = "diagnostics")]
    {
        let held = others.iter().filter(|p| g.holders.contains_key(p)).count();
        crate::copystat::add(crate::copystat::C::AliasMarks, held as u64);
    }
    // Every page's bytes are saved before any is opened: once a page is
    // open, a write by another thread lands without a fault, and a save
    // after it would keep that write as the page's old bytes.
    mark_changed(g, first, last, save, read);
    for &page in &others {
        mark_changed(g, page, page + 1, save, read);
    }
    open_guarded(g, first, last);
    for page in others {
        open_guarded(g, page, page + 1);
    }
    guarded
}

/// Gives write access back to the pages of `[first, last)` we guarded.
fn open_guarded(g: &mut Guards, first: usize, last: usize) {
    let guarded: Vec<usize> = g.protected.range(first..last).copied().collect();
    for page in guarded {
        open(g, page);
    }
}

/// Marks `[first, last)` changed in the trackers holding its pages. With
/// `save`, before the write lands: trackers armed for a page they had taken
/// keep its bytes.
fn mark_changed(g: &mut Guards, first: usize, last: usize, save: bool, read: &mut Read) {
    for (&page, holders) in g.holders.range(first..last) {
        for c in holders {
            let first_write = {
                let mut set = c.0.lock().unwrap_or_else(|e| e.into_inner());
                set_dirty(c.2, true);
                set.insert(page)
            };
            if first_write && save {
                c.save(page, read);
            }
        }
        #[cfg(feature = "diagnostics")]
        g.written.insert(page);
    }
}

/// [`mark_changed`] (saving) of page `page` in every tracker holding it but
/// `except`.
fn mark_changed_but(g: &mut Guards, page: usize, except: &Arc<Changed>) {
    let Some(holders) = g.holders.get(&page) else {
        return;
    };
    for c in holders.iter().filter(|c| !Arc::ptr_eq(c, except)) {
        let first_write = {
            let mut set = c.0.lock().unwrap_or_else(|e| e.into_inner());
            set_dirty(c.2, true);
            set.insert(page)
        };
        if first_write {
            c.save(page, &mut None);
        }
    }
    #[cfg(feature = "diagnostics")]
    g.written.insert(page);
}

/// At each presented frame (diagnostics): counts the tracked pages written
/// in it, and those written the frame before too; every 32nd frame, checks
/// that lazy pages still have no access (a protection change we did not
/// make would show their old bytes).
#[cfg(feature = "diagnostics")]
pub(crate) fn frame_done() {
    let mut g = guards();
    let written = std::mem::take(&mut g.written);
    let again = written.intersection(&g.written_before).count();
    crate::copystat::add(crate::copystat::C::WrittenPages, written.len() as u64);
    crate::copystat::add(crate::copystat::C::WrittenAgain, again as u64);
    g.written_before = written;
    static FRAMES: AtomicUsize = AtomicUsize::new(0);
    if FRAMES.fetch_add(1, Ordering::Relaxed).is_multiple_of(32) && !g.lazy.is_empty() {
        use windows_sys::Win32::System::Memory::PAGE_NOACCESS;
        let open: Vec<String> = g
            .lazy
            .keys()
            .filter(|&&p| protection(p * PAGE) != Some(PAGE_NOACCESS))
            .map(|&p| format!("{:#x}", p * PAGE))
            .collect();
        if !open.is_empty() {
            tracing::warn!(count = open.len(), first = ?&open[..open.len().min(16)], "lazy pages lost their no-access guard");
        }
    }
}

/// A write into title memory the guards do not see, under way: its pages
/// stay open (a tracker taking them meanwhile would guard them, and the
/// kernel's write would fail) until it is dropped, which marks them changed
/// once more.
#[must_use = "the write must happen while this lives"]
pub struct Writing {
    first: usize,
    last: usize,
}

impl Drop for Writing {
    fn drop(&mut self) {
        #[cfg(feature = "diagnostics")]
        let _site = at_site(Site::WritingDrop);
        let mut g = guards();
        if let Some(i) = g.busy.iter().position(|&r| r == (self.first, self.last)) {
            g.busy.swap_remove(i);
        }
        changed(&mut g, self.first, self.last, false, &mut None, Touch::Host);
    }
}

/// Before our own write into title memory `[at, at + len)` through the
/// kernel (which does not fault into our handler) or another write the
/// guards do not see: opens its guarded blocks, marks them changed and
/// keeps them open while the result lives. The runtime calls it (as
/// `kernel_will_write`) around the title's kernel reads into its own
/// memory and DirectStorage's writes to addresses.
pub fn writing(at: usize, len: usize) -> Writing {
    #[cfg(feature = "diagnostics")]
    let _site = at_site(Site::Writing);
    // The length is the title's (`fread`'s size times count, saturated).
    let (first, last) = (at / PAGE, at.saturating_add(len).div_ceil(PAGE));
    #[cfg(feature = "diagnostics")]
    crate::census::written(at, len, crate::census::Writer::Kernel);
    let mut g = guards();
    g.busy.push((first, last));
    changed(&mut g, first, last, true, &mut None, Touch::Host);
    note_opened(&g, first, last);
    Writing { first, last }
}

/// The runtime mapped or unmapped title memory `[at, at + len)`: what it
/// shows is new, and read-write.
pub fn title_memory_remapped(at: usize, len: usize) {
    #[cfg(feature = "diagnostics")]
    let _site = at_site(Site::Remapped);
    let (first, last) = (at / PAGE, at.saturating_add(len).div_ceil(PAGE));
    #[cfg(feature = "diagnostics")]
    crate::census::gone(at, len);
    let mut g = guards();
    // What it showed is gone already: nothing to keep. A lazy page there
    // gets its copy's bytes first (the runtime reports an unmap before it
    // unmaps: the physical page keeps them for the views still showing it).
    changed(&mut g, first, last, false, &mut None, Touch::Host);
    // What it shows now is new: bytes read of those pages before are stale.
    note_opened(&g, first, last);
    // What it showed is not what it shows now: its links go, both ways.
    let stale: Vec<usize> = g.alias_of.range(first..last).map(|(&p, _)| p).collect();
    for page in stale {
        for other in g.alias_of.remove(&page).unwrap_or_default() {
            if let Some(links) = g.alias_of.get_mut(&other) {
                links.retain(|&p| p != page);
                if links.is_empty() {
                    g.alias_of.remove(&other);
                }
            }
        }
    }
}

/// The runtime's answer to where title page `page` (its address) can be
/// written without its guards: the address of the same page in a writable
/// mapping of the runtime's own that no tracker holds and the title does
/// not use, or `None` (memory that is not a section's: the runtime's
/// pools). A function pointer, 0 for none ([`set_writable_alias`]).
static WRITABLE_ALIAS: AtomicU64 = AtomicU64::new(0);

/// The runtime tells how to find its own writable mapping of title pages
/// that sections back (its physical-page views): copy-back of marked pages
/// writes there instead of opening the title's guarded mapping
/// (`crate::shadow`'s runs). The function must not call into this module
/// (it is asked with no guard lock held, but its own lock is taken while
/// the runtime maps pages, which calls in here).
pub fn set_writable_alias(f: fn(usize) -> Option<usize>) {
    WRITABLE_ALIAS.store(f as usize as u64, Ordering::Release);
}

/// See [`set_writable_alias`].
pub(crate) fn writable_alias(page: usize) -> Option<usize> {
    match WRITABLE_ALIAS.load(Ordering::Acquire) {
        0 => None,
        f => {
            // SAFETY: stored from a `fn(usize) -> Option<usize>`.
            let f = unsafe { std::mem::transmute::<usize, fn(usize) -> Option<usize>>(f as usize) };
            f(page)
        }
    }
}

// Lazy view pages (`VICTORY_LAZY_VIEWS=1`). Of the view pages shaders mark,
// most are marked again by the next submissions before the title or our host
// touches them (run lazyprobe-u1: 89 % of the pages that could be watched),
// so their copy-back is work thrown away. A lazy page skips it: title memory
// keeps its bytes from before the submission behind no access at all, the
// copy keeps what shaders wrote, and the first touch puts the copy's bytes in
// through the runtime's own mapping of the page's section while every title
// mapping still has no access, so no reader ever sees the old bytes (the
// window that crashed run lazy1-u1, `perf/lazy-copyback`, where the bytes
// went in through the title's mapping after it was opened).
//
// Who touches title memory, and how each finds a lazy page's bytes first:
// - in-process reads and writes (the title, the runtime's code, our host's
//   direct loads and stores: labels, fences, `CopyMemoryToMemoryX`) fault
//   into the handler, which resolves the page before the access retries;
// - our host through the kernel: `crate::host::read_into` and `copy_mapped`
//   (source) call [`reading`], their destinations and other kernel writes
//   [`writing`] (both resolve), `crate::texture::committed` (asked before a
//   read) calls [`reading`];
// - the runtime's own views of physical pages (DirectStorage destinations)
//   hold [`unseen_writing`] while they write (resolves, and no page there
//   goes lazy meanwhile);
// - the runtime mapping or unmapping a page ([`title_memory_aliased`],
//   [`title_memory_remapped`], reported before an unmap);
// - the copy going (`crate::shadow`'s drop: [`Tracked::resolve_all`]); what
//   it wrote being withheld ([`forget_lazy`]: title memory keeps its bytes).
//
// shortcut: a system call reading title memory on the title's behalf (a file
// write from a lazy page) fails as a kernel write into a guarded page does;
// no title call is known to pass GPU-written view memory to one.

/// A view page whose title bytes are older than a host copy's (an entry per
/// address showing it, the same for all).
#[derive(Clone)]
struct LazyPage {
    /// The copy's 64 KiB of the page (alive while the copy is: it gives its
    /// lazy pages their bytes before it goes).
    src: usize,
    /// The page in the runtime's own writable mapping of its section
    /// ([`writable_alias`]), which the title does not see.
    alias: usize,
    owner: Arc<Changed>,
    /// [`print`] of the title bytes left there (checked at the first touch
    /// and by [`verify_lazy`]).
    #[cfg(any(test, feature = "diagnostics"))]
    print: u64,
}

/// A page (page number) offered to [`Lease::lazify`]: where its copy keeps
/// its 64 KiB, and the page in the runtime's own mapping of its section.
pub(crate) struct LazyView {
    pub(crate) page: usize,
    pub(crate) src: usize,
    pub(crate) alias: usize,
}

/// Whether marked view pages go back lazily (`VICTORY_LAZY_VIEWS=1`): 1 on,
/// 0 off, `u8::MAX` not asked yet. Read by the fault handler.
static LAZY_VIEWS: AtomicU8 = AtomicU8::new(u8::MAX);

/// Whether marked view pages go back lazily ([`Lease::lazify`]).
pub(crate) fn lazy_views() -> bool {
    match LAZY_VIEWS.load(Ordering::Relaxed) {
        u8::MAX => {
            let on = std::env::var("VICTORY_LAZY_VIEWS").as_deref() == Ok("1");
            LAZY_VIEWS.store(u8::from(on), Ordering::Relaxed);
            on
        }
        mode => mode == 1,
    }
}

/// Tests: lazy view pages on or off.
#[cfg(test)]
pub(crate) fn set_lazy_views(on: bool) {
    LAZY_VIEWS.store(u8::from(on), Ordering::Relaxed);
}

/// Pages that may be lazy: those below 2^44 (title memory is below 2^40).
const LAZY_LIMIT: usize = 1 << 28;

/// Lazy pages now ([`Guards::lazy`]'s length, changed under the guard lock).
static LAZY_LIVE: AtomicUsize = AtomicUsize::new(0);

/// Lazy pages now (diagnostics).
#[cfg(feature = "diagnostics")]
pub(crate) fn lazy_live() -> usize {
    LAZY_LIVE.load(Ordering::Relaxed)
}

/// A bit per page below [`LAZY_LIMIT`], set while it is lazy: readers and
/// the fault handler ask it without the guard lock. 32 MiB of address
/// space, committed by the system where touched.
static LAZY_BITS: std::sync::LazyLock<Box<[AtomicU64]>> = std::sync::LazyLock::new(|| {
    let n = LAZY_LIMIT / 64;
    let layout = std::alloc::Layout::array::<AtomicU64>(n).expect("a small layout");
    // SAFETY: a non-zero layout; zeroed `AtomicU64`s are valid.
    let at = unsafe { std::alloc::alloc_zeroed(layout) }.cast::<AtomicU64>();
    if at.is_null() {
        std::alloc::handle_alloc_error(layout);
    }
    // SAFETY: `n` zeroed elements allocated with the global allocator.
    unsafe { Box::from_raw(std::ptr::slice_from_raw_parts_mut(at, n)) }
});

fn lazy_bit(page: usize) -> bool {
    page < LAZY_LIMIT
        && LAZY_LIVE.load(Ordering::Acquire) != 0
        && LAZY_BITS[page / 64].load(Ordering::Acquire) & 1 << (page % 64) != 0
}

/// Whether page `page` (a page number) may be lazy now (no lock: a page
/// made lazy or given its bytes meanwhile is the caller's to check again
/// under the guard lock).
pub(crate) fn maybe_lazy(page: usize) -> bool {
    lazy_bit(page)
}

/// Whether a page of `[first, last)` is lazy.
fn any_lazy(first: usize, last: usize) -> bool {
    if LAZY_LIVE.load(Ordering::Acquire) == 0 {
        return false;
    }
    let (mut page, last) = (first.min(LAZY_LIMIT), last.min(LAZY_LIMIT));
    while page < last {
        let word = page / 64;
        let (lo, hi) = (page % 64, (last - word * 64).min(64));
        let mask = if hi - lo == 64 { u64::MAX } else { ((1u64 << (hi - lo)) - 1) << lo };
        if LAZY_BITS[word].load(Ordering::Acquire) & mask != 0 {
            return true;
        }
        page = (word + 1) * 64;
    }
    false
}

fn set_lazy_bit(page: usize, lazy: bool) {
    let (word, mask) = (&LAZY_BITS[page / 64], 1u64 << (page % 64));
    if lazy {
        word.fetch_or(mask, Ordering::Release);
        LAZY_LIVE.fetch_add(1, Ordering::AcqRel);
    } else {
        word.fetch_and(!mask, Ordering::Release);
        LAZY_LIVE.fetch_sub(1, Ordering::AcqRel);
    }
}

/// Who touched a lazy page first.
#[derive(Clone, Copy)]
enum Touch {
    TitleRead,
    TitleWrite,
    /// Our host: its reads and writes of title memory through the kernel,
    /// DirectStorage's views, the runtime mapping memory, a copy going.
    Host,
}

/// Why [`Lease::lazify`] left a page out (diagnostics).
#[derive(Clone, Copy)]
enum Refused {
    Changed,
    Unguarded,
    Alias,
    /// Under a write the guards do not see, lazy already, past the limit,
    /// not the tracker's, or a protection change failed.
    Other,
}

fn refused(why: Refused) {
    #[cfg(feature = "diagnostics")]
    crate::copystat::add(
        match why {
            Refused::Changed => crate::copystat::C::LazyRefusedChanged,
            Refused::Unguarded => crate::copystat::C::LazyRefusedUnguarded,
            Refused::Alias => crate::copystat::C::LazyRefusedAlias,
            Refused::Other => crate::copystat::C::LazyRefusedOther,
        },
        1,
    );
    #[cfg(not(feature = "diagnostics"))]
    let _ = why;
}

/// Sets the protection of pages `[first, last)` (for lazy pages).
fn set_protection(first: usize, last: usize, protection: u32) -> bool {
    use windows_sys::Win32::System::Memory::VirtualProtect;
    #[cfg(feature = "diagnostics")]
    {
        crate::copystat::add(crate::copystat::C::LazyProtects, 1);
        count_protect(1, (last - first) as u64);
    }
    let mut old = 0;
    // SAFETY: title memory; only its protection changes (failing harmlessly
    // if nothing is mapped there).
    unsafe { VirtualProtect((first * PAGE) as *const _, (last - first) * PAGE, protection, &mut old) != 0 }
}

/// Page `page`'s first touch since it went lazy (by `touch`): the copy's
/// bytes go into title memory through the runtime's own mapping while
/// every title mapping still has no access (a reader there faults, waits
/// for the guard lock and retries once they are in), then those mappings
/// are guarded as any page a copy took, so that a title write marks it
/// changed again. False if it was not lazy.
fn resolve(g: &mut Guards, page: usize, touch: Touch) -> bool {
    let Some((lazy, pages)) = take_lazy(g, page) else {
        return false;
    };
    #[cfg(feature = "diagnostics")]
    let _timed = Timed(std::time::Instant::now(), crate::copystat::C::LazyResolveUs);
    // The title bytes are still those we left: no writer went past us.
    #[cfg(any(test, feature = "diagnostics"))]
    check(page, lazy.alias, lazy.print);
    // SAFETY: the copy's 64 KiB, alive while the copy is; the runtime's own
    // read-write mapping of the page, alive while a title mapping shows it
    // (which reports its unmap first, resolving it).
    unsafe { std::ptr::copy_nonoverlapping(lazy.src as *const u8, lazy.alias as *mut u8, PAGE) };
    for p in pages {
        guard_again(g, p, &lazy.owner);
    }
    #[cfg(feature = "diagnostics")]
    crate::copystat::add(
        match touch {
            Touch::TitleRead => crate::copystat::C::LazyTitleReads,
            Touch::TitleWrite => crate::copystat::C::LazyTitleWrites,
            Touch::Host => crate::copystat::C::LazyHostTouches,
        },
        1,
    );
    #[cfg(not(feature = "diagnostics"))]
    let _ = touch;
    true
}

/// [`resolve`] of the lazy pages of `[first, last)`.
fn resolve_range(g: &mut Guards, first: usize, last: usize, touch: Touch) {
    if !any_lazy(first, last) {
        return;
    }
    let pages: Vec<usize> = g.lazy.range(first..last).map(|(&p, _)| p).collect();
    for page in pages {
        resolve(g, page, touch);
    }
}

/// Takes lazy page `page` out of [`Guards::lazy`] with the pages showing it
/// elsewhere (they went lazy with it): its entry and all those pages.
fn take_lazy(g: &mut Guards, page: usize) -> Option<(LazyPage, Vec<usize>)> {
    let lazy = g.lazy.remove(&page)?;
    let mut pages = vec![page];
    for other in g.alias_of.get(&page).cloned().unwrap_or_default() {
        if g.lazy.get(&other).is_some_and(|l| l.alias == lazy.alias) {
            g.lazy.remove(&other);
            pages.push(other);
        }
    }
    Some((lazy, pages))
}

/// A lazy page `page` (out of [`Guards::lazy`]) is a guarded page again: its
/// copy holds title memory as it now is, and a title write faults. One
/// that cannot be guarded (nothing mapped any more) is changed in `owner`
/// instead (its next refresh takes title memory).
fn guard_again(g: &mut Guards, page: usize, owner: &Arc<Changed>) {
    use windows_sys::Win32::System::Memory::PAGE_READONLY;
    if set_protection(page, page + 1, PAGE_READONLY) {
        g.protected.insert(page);
    } else {
        mark_owner(g, page, owner);
    }
    set_lazy_bit(page, false);
}

/// Page `page` counts as changed in `owner` if it holds it.
fn mark_owner(g: &Guards, page: usize, owner: &Arc<Changed>) {
    if g.holders.get(&page).is_some_and(|h| h.iter().any(|c| Arc::ptr_eq(c, owner))) {
        owner.0.lock().unwrap_or_else(|e| e.into_inner()).insert(page);
        set_dirty(owner.2, true);
    }
}

/// Lazy page `page` does not go back: title memory keeps its bytes, guarded
/// again, and its copy counts it changed (its next refresh takes them).
fn forget(g: &mut Guards, page: usize) {
    if let Some((lazy, pages)) = take_lazy(g, page) {
        for &p in &pages {
            guard_again(g, p, &lazy.owner);
            mark_owner(g, p, &lazy.owner);
        }
        #[cfg(feature = "diagnostics")]
        crate::copystat::add(crate::copystat::C::LazyDropped, 1);
    }
}

/// What shaders wrote into host copies of `[at, at + len)` stays out of
/// title memory (`crate::memory::drop_withheld`, [`retake`]): lazy pages
/// there do not go back either ([`forget`]).
// shortcut: a lazy page shaders write again in a submission whose writes are
// withheld loses the earlier submission's bytes too (they were only in the
// copy); copy-back makes no page lazy while writes are withheld, so this
// takes a miss right after a lazy copy-back.
pub(crate) fn forget_lazy(at: usize, len: usize) {
    let (first, last) = (at / PAGE, at.saturating_add(len).div_ceil(PAGE));
    if !any_lazy(first, last) {
        return;
    }
    let mut g = guards();
    let pages: Vec<usize> = g.lazy.range(first..last).map(|(&p, _)| p).collect();
    for page in pages {
        forget(&mut g, page);
    }
}

/// Before our host reads title memory `[at, at + len)` through the kernel
/// (which faults into no handler: a page with no access fails the read):
/// its lazy pages get their copies' bytes.
pub(crate) fn reading(at: usize, len: usize) {
    let (first, last) = (at / PAGE, at.saturating_add(len).div_ceil(PAGE));
    if !any_lazy(first, last) {
        return;
    }
    let mut g = guards();
    resolve_range(&mut g, first, last, Touch::Host);
}

/// Writes into title pages through views the guards do not see (the
/// runtime's own views of physical pages: DirectStorage destinations), under
/// way: none of the pages goes lazy until it is dropped. The writer reports
/// the pages written afterwards as before ([`writing`]).
#[must_use = "the write must happen while this lives"]
pub struct UnseenWriting(Vec<usize>);

impl Drop for UnseenWriting {
    fn drop(&mut self) {
        if self.0.is_empty() {
            return;
        }
        let mut g = guards();
        for page in &self.0 {
            if let Some(n) = g.held.get_mut(page) {
                *n -= 1;
                if *n == 0 {
                    g.held.remove(page);
                }
            }
        }
    }
}

/// Before writes into title pages through views the guards do not see, at
/// the title addresses `shown()` gives (asked only while view pages go back
/// lazily: nothing to do otherwise): lazy pages there get their copies'
/// bytes first (the write lands on those, not under them), and none goes
/// lazy while the result lives.
pub fn unseen_writing(shown: impl FnOnce() -> Vec<usize>) -> UnseenWriting {
    if !lazy_views() {
        return UnseenWriting(Vec::new());
    }
    let pages: Vec<usize> = shown().iter().map(|&at| at / PAGE).collect();
    if pages.is_empty() {
        return UnseenWriting(pages);
    }
    let mut g = guards();
    for &page in &pages {
        resolve(&mut g, page, Touch::Host);
        *g.held.entry(page).or_default() += 1;
    }
    UnseenWriting(pages)
}

/// What [`print`] reads: every word (`VICTORY_LAZY_VERIFY=full`, tests) or
/// one in 64 (one per 512 bytes, the default: an eighth of the cache lines).
#[cfg(any(test, feature = "diagnostics"))]
fn print_step() -> usize {
    static STEP: std::sync::LazyLock<usize> = std::sync::LazyLock::new(|| {
        if cfg!(test) || std::env::var("VICTORY_LAZY_VERIFY").as_deref() == Ok("full") { 1 } else { 64 }
    });
    *STEP
}

/// A fingerprint of the 64 KiB at `at` (the runtime's own mapping of a
/// page: readable whatever the title's mappings allow).
#[cfg(any(test, feature = "diagnostics"))]
fn print(at: usize) -> u64 {
    // SAFETY: a page of the runtime's own read-write mapping.
    let words = unsafe { std::slice::from_raw_parts(at as *const u64, PAGE / 8) };
    words
        .iter()
        .step_by(print_step())
        .fold(0xcbf2_9ce4_8422_2325u64, |h, &w| (h ^ w).wrapping_mul(0x0100_0000_01b3))
}

/// Lazy pages checked, and those whose title bytes were not what we left
/// (a writer that went past us: its bytes are lost at the first touch);
/// the last such pages (page numbers). No allocation: checked in the
/// fault handler.
#[cfg(any(test, feature = "diagnostics"))]
static CHECKS: [AtomicUsize; 2] = [const { AtomicUsize::new(0) }; 2];
#[cfg(any(test, feature = "diagnostics"))]
static MISSED: [AtomicUsize; 8] = [const { AtomicUsize::new(0) }; 8];

/// Checks lazy page `page` (title address `page * PAGE`): its bytes, read
/// through `alias`, still print as `print`.
#[cfg(any(test, feature = "diagnostics"))]
fn check(page: usize, alias: usize, was: u64) {
    CHECKS[0].fetch_add(1, Ordering::Relaxed);
    if print(alias) != was {
        let n = CHECKS[1].fetch_add(1, Ordering::Relaxed);
        MISSED[n % MISSED.len()].store(page, Ordering::Relaxed);
    }
}

/// Tests: lazy pages checked and those found written behind our back so far.
#[cfg(test)]
pub(crate) fn lazy_checks() -> (usize, usize) {
    (CHECKS[0].load(Ordering::Relaxed), CHECKS[1].load(Ordering::Relaxed))
}

/// Since the last call (diagnostics): lazy pages checked, those found
/// written behind our back, and the last of those (title addresses).
#[cfg(feature = "diagnostics")]
pub(crate) fn take_lazy_checks() -> (usize, usize, Vec<String>) {
    let (checks, missed) = (CHECKS[0].swap(0, Ordering::Relaxed), CHECKS[1].swap(0, Ordering::Relaxed));
    let last = MISSED
        .iter()
        .take(missed.min(MISSED.len()))
        .map(|p| format!("{:#x}", p.load(Ordering::Relaxed) * PAGE))
        .collect();
    (checks, missed, last)
}

/// At each completion (diagnostics): checks up to `n` lazy pages, in turn,
/// for bytes a writer put into title memory past us (each page is also
/// checked at its first touch, before the copy's bytes go in).
#[cfg(feature = "diagnostics")]
pub(crate) fn verify_lazy(n: usize) {
    static NEXT: AtomicUsize = AtomicUsize::new(0);
    if LAZY_LIVE.load(Ordering::Relaxed) == 0 {
        return;
    }
    let g = guards();
    let from = NEXT.load(Ordering::Relaxed);
    let mut last = from;
    let mut seen = std::collections::HashSet::new();
    for (&page, lazy) in g.lazy.range(from..).chain(g.lazy.range(..from)).take(n) {
        // A page shown at several addresses once.
        if seen.insert(lazy.alias) {
            check(page, lazy.alias, lazy.print);
        }
        last = page + 1;
    }
    NEXT.store(last, Ordering::Relaxed);
}

/// The protection of the page at `at` now, if mapped.
fn protection(at: usize) -> Option<u32> {
    use windows_sys::Win32::System::Memory::{MEMORY_BASIC_INFORMATION, VirtualQuery};
    // SAFETY: plain data filled in by the query.
    let mut info: MEMORY_BASIC_INFORMATION = unsafe { std::mem::zeroed() };
    // SAFETY: a valid out pointer of the right size.
    (unsafe { VirtualQuery(at as *const _, &mut info, size_of_val(&info)) } != 0).then_some(info.Protect)
}

/// Logs (at most four times) an access fault at `at` we pass on while pages
/// are lazy: whether it is one (a fault we should have taken).
fn log_lazy_fault(at: usize) {
    static LOGGED: AtomicUsize = AtomicUsize::new(0);
    if LAZY_LIVE.load(Ordering::Relaxed) == 0 || LOGGED.fetch_add(1, Ordering::Relaxed) >= 4 {
        return;
    }
    tracing::warn!(
        at = format_args!("{at:#x}"),
        lazy = lazy_bit(at / PAGE),
        protection = ?protection(at),
        lazy_now = LAZY_LIVE.load(Ordering::Relaxed),
        "an access fault passed on while view pages are lazy"
    );
}

/// Whether `at` is committed memory a read does not fault on now.
fn readable(at: usize) -> bool {
    use windows_sys::Win32::System::Memory::{
        MEM_COMMIT, MEMORY_BASIC_INFORMATION, PAGE_GUARD, PAGE_NOACCESS, VirtualQuery,
    };
    // SAFETY: plain data filled in by the query.
    let mut info: MEMORY_BASIC_INFORMATION = unsafe { std::mem::zeroed() };
    // SAFETY: a valid out pointer of the right size.
    let n = unsafe { VirtualQuery(at as *const _, &mut info, size_of_val(&info)) };
    n != 0 && info.State == MEM_COMMIT && info.Protect != 0 && info.Protect & (PAGE_NOACCESS | PAGE_GUARD) == 0
}

/// What shaders wrote into host copies of `[at, at + len)` is dropped:
/// every host buffer copy holding the memory takes it from title memory
/// again at its next refresh (the pages count as changed in each, aliases
/// too). Title memory itself did not change, so its guards stay (opened page
/// by page, they cost ~15 % of the submitting thread at each frame end, run
/// perf15), and other trackers (textures' staging, heaps) are left alone:
/// shaders' writes never reach them (marked, textures over withheld ranges
/// were uploaded again for nothing: 19 % of the uploads, run evict3).
pub(crate) fn retake(at: usize, len: usize) {
    let (first, last) = (at / PAGE, at.saturating_add(len).div_ceil(PAGE));
    forget_lazy(at, len);
    let g = guards();
    let others: Vec<usize> = g.alias_of.range(first..last).flat_map(|(_, o)| o.iter().copied()).collect();
    mark(&g, first, last);
    for page in others {
        mark(&g, page, page + 1);
    }
}

/// Pages `[first, last)` count as changed in every host buffer copy's
/// tracker holding them.
fn mark(g: &Guards, first: usize, last: usize) {
    for (&page, holders) in g.holders.range(first..last) {
        for c in holders.iter().filter(|c| c.3) {
            let mut set = c.0.lock().unwrap_or_else(|e| e.into_inner());
            set_dirty(c.2, true);
            set.insert(page);
        }
    }
}

/// The runtime mapped the page at `at` where `others` show it already: the
/// pages are linked both ways, and if we guard it there, it is guarded here
/// too (a write through the new mapping must not go unseen by the trackers
/// there).
pub fn title_memory_aliased(at: usize, others: &[usize]) {
    #[cfg(feature = "diagnostics")]
    let _site = at_site(Site::Aliased);
    #[cfg(feature = "diagnostics")]
    crate::census::aliased(at, others);
    let mut g = guards();
    let page = at / PAGE;
    // The new mapping shows the physical page: where it is lazy already,
    // it gets its copy's bytes now.
    for &other in others {
        resolve(&mut g, other / PAGE, Touch::Host);
    }
    for &other in others {
        for (x, y) in [(other / PAGE, page), (page, other / PAGE)] {
            let links = g.alias_of.entry(x).or_default();
            if !links.contains(&y) {
                links.push(y);
            }
        }
    }
    // If the new mapping cannot be guarded (a write the guards do not see is
    // under way through it), the page is changed; its bytes now may hold that
    // write already, so nothing is saved of them.
    if others.iter().any(|&o| g.protected.contains(&(o / PAGE)))
        && protect(&mut g, &[page]).is_empty()
    {
        changed(&mut g, page, page + 1, false, &mut None, Touch::Host);
        note_opened(&g, page, page + 1);
    }
}

/// The other page numbers showing the same memory as page `page`
/// (diagnostics: the owner census).
#[cfg(feature = "diagnostics")]
pub(crate) fn mappings(page: usize) -> Vec<usize> {
    guards().alias_of.get(&page).cloned().unwrap_or_default()
}

/// Registers the handler that takes writes to guarded blocks and the first
/// touch of lazy pages, once, ahead of the crash reporter's.
fn install() {
    use windows_sys::Win32::Foundation::{EXCEPTION_ACCESS_VIOLATION, NTSTATUS};
    use windows_sys::Win32::System::Diagnostics::Debug::{
        AddVectoredExceptionHandler, EXCEPTION_CONTINUE_EXECUTION, EXCEPTION_CONTINUE_SEARCH,
        EXCEPTION_POINTERS,
    };
    unsafe extern "system" fn on_write(info: *mut EXCEPTION_POINTERS) -> i32 {
        // SAFETY: the system passes the exception being dispatched.
        let r = unsafe { &*(*info).ExceptionRecord };
        const READ: usize = 0;
        const WRITE: usize = 1;
        if r.ExceptionCode != EXCEPTION_ACCESS_VIOLATION as NTSTATUS || r.NumberParameters < 2 {
            return EXCEPTION_CONTINUE_SEARCH;
        }
        let (kind, at) = (r.ExceptionInformation[0], r.ExceptionInformation[1]);
        if kind == READ {
            // Only lazy pages fault on reads (no allocation here, and no
            // lock unless the page's bit is set).
            if LAZY_LIVE.load(Ordering::Acquire) == 0 && LAZY_VIEWS.load(Ordering::Relaxed) != 1 {
                return EXCEPTION_CONTINUE_SEARCH;
            }
            let given = lazy_bit(at / PAGE) && {
                #[cfg(feature = "diagnostics")]
                let _timed = Timed(std::time::Instant::now(), crate::copystat::C::HandlerUs);
                resolve(&mut guards(), at / PAGE, Touch::TitleRead)
            };
            // Not lazy: given its bytes by another thread since it faulted,
            // or not ours.
            if given || readable(at) {
                return EXCEPTION_CONTINUE_EXECUTION;
            }
            log_lazy_fault(at);
            return EXCEPTION_CONTINUE_SEARCH;
        }
        if kind != WRITE {
            return EXCEPTION_CONTINUE_SEARCH;
        }
        #[cfg(feature = "diagnostics")]
        let _site = at_site(Site::Handler);
        #[cfg(feature = "diagnostics")]
        let _timed = Timed(std::time::Instant::now(), crate::copystat::C::HandlerUs);
        // Held across both checks: another thread may have opened the block
        // since this one faulted, and a copy may have guarded it again since
        // (then it is ours again, not read-only for some other reason).
        let page = at / PAGE;
        #[cfg(feature = "diagnostics")]
        crate::census::written(at, 1, crate::census::Writer::Title(r.ExceptionAddress as usize));
        let mut g = guards();
        // A tracker armed for the page keeps its bytes before the write: they
        // are read without the lock, the page guarded meanwhile, and kept
        // only if it stayed guarded throughout ([`OPENED`]); else they are
        // read under it, as before.
        let mut read = None;
        let saves = |g: &Guards| {
            g.holders
                .get(&page)
                .is_some_and(|h| h.iter().any(|c| c.would_save(page)))
        };
        if g.protected.contains(&page) && saves(&g) {
            let before = opened(page).load(Ordering::Relaxed);
            drop(g);
            let bytes = page_bytes(page);
            g = guards();
            if g.protected.contains(&page) && opened(page).load(Ordering::Relaxed) == before {
                read = bytes.map(|b| (page, b));
            }
        }
        if changed(&mut g, page, page + 1, true, &mut read, Touch::TitleWrite) {
            #[cfg(feature = "diagnostics")]
            crate::copystat::add(crate::copystat::C::GuardFaults, 1);
            return EXCEPTION_CONTINUE_EXECUTION;
        }
        if writable(at) {
            return EXCEPTION_CONTINUE_EXECUTION;
        }
        drop(g);
        log_lazy_fault(at);
        EXCEPTION_CONTINUE_SEARCH
    }
    static ONCE: std::sync::Once = std::sync::Once::new();
    // SAFETY: the handler only touches our guard state and protections.
    ONCE.call_once(|| unsafe {
        AddVectoredExceptionHandler(1, Some(on_write));
    });
}

/// Whether `at` is committed read-write memory now.
fn writable(at: usize) -> bool {
    use windows_sys::Win32::System::Memory::{
        MEM_COMMIT, MEMORY_BASIC_INFORMATION, PAGE_READWRITE, VirtualQuery,
    };
    // SAFETY: plain data filled in by the query.
    let mut info: MEMORY_BASIC_INFORMATION = unsafe { std::mem::zeroed() };
    // SAFETY: a valid out pointer of the right size.
    let n = unsafe { VirtualQuery(at as *const _, &mut info, size_of_val(&info)) };
    n != 0 && info.State == MEM_COMMIT && info.Protect == PAGE_READWRITE
}

/// Adds the time since it was made to a counter when dropped (diagnostics).
#[cfg(feature = "diagnostics")]
pub(crate) struct Timed(pub(crate) std::time::Instant, pub(crate) crate::copystat::C);

#[cfg(feature = "diagnostics")]
impl Drop for Timed {
    fn drop(&mut self) {
        let c = std::mem::replace(&mut self.1, crate::copystat::C::Submissions);
        crate::copystat::add(c, self.0.elapsed().as_micros() as u64);
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use windows_sys::Win32::System::Memory::{
        MEM_COMMIT, MEM_RELEASE, MEM_RESERVE, PAGE_READWRITE, VirtualAlloc, VirtualFree,
    };
    pub(crate) const MEM_WRITE_WATCH: u32 = 0x20_0000;

    /// A write-watched allocation of `pages` pages (as the runtime's pools),
    /// on a 64 KiB boundary (the allocation granularity).
    pub(crate) fn pool(pages: usize) -> usize {
        // SAFETY: a fresh allocation, freed by the test.
        let p = unsafe {
            VirtualAlloc(
                std::ptr::null(),
                pages * PAGE,
                MEM_RESERVE | MEM_COMMIT | MEM_WRITE_WATCH,
                PAGE_READWRITE,
            )
        } as usize;
        assert!(p != 0 && p.is_multiple_of(PAGE));
        p
    }

    /// Sets its flag when dropped, also while a failed assertion unwinds:
    /// writer threads looping until the flag stop, and the test fails
    /// instead of keeping `thread::scope` waiting for them.
    struct StopOnDrop<'a>(&'a std::sync::atomic::AtomicBool);

    impl Drop for StopOnDrop<'_> {
        fn drop(&mut self) {
            self.0.store(true, std::sync::atomic::Ordering::Relaxed);
        }
    }

    pub(crate) fn free(p: usize) {
        // SAFETY: an allocation of ours.
        unsafe { VirtualFree(p as *mut _, 0, MEM_RELEASE) };
    }

    /// Everything is changed at first; after taking, only what is written
    /// (by faults, or reported) is, and every tracker of a page sees it.
    #[test]
    fn writes_mark_every_tracker_of_their_page() {
        let p = pool(4);
        let a = track(p, 4 * PAGE, true).expect("pool memory is tracked");
        let b = track(p + PAGE, 2 * PAGE, true).expect("tracked");
        assert!(!a.clean() && !b.clean(), "changed at first");
        assert_eq!(a.take(p, p + 4 * PAGE), [(p, p + 4 * PAGE)]);
        assert!(a.clean() && !b.clean());
        assert_eq!(b.take(p, p + 4 * PAGE), [(p + PAGE, p + 3 * PAGE)]);
        assert!(a.take(p, p + 4 * PAGE).is_empty() && b.take(p, p + 4 * PAGE).is_empty());
        assert!(a.clean() && b.clean());
        // SAFETY: committed pages of ours, now read-only: the handler opens
        // the block; the second write needs no fault.
        unsafe {
            ((p + PAGE + 8) as *mut u8).write_volatile(1);
            ((p + PAGE + 9) as *mut u8).write_volatile(2);
        }
        assert!(!a.clean() && !b.clean(), "a write is seen by every tracker");
        assert_eq!(a.take(p, p + 4 * PAGE), [(p + PAGE, p + 2 * PAGE)]);
        assert!(a.clean());
        // `a` guarded it again; `b` still has it changed.
        assert!(b.changed(p / PAGE + 1));
        assert_eq!(b.take(p, p + 4 * PAGE), [(p + PAGE, p + 2 * PAGE)]);
        // SAFETY: as above: a fault again, for both.
        unsafe { ((p + PAGE + 10) as *mut u8).write_volatile(3) };
        assert!(a.changed(p / PAGE + 1) && b.changed(p / PAGE + 1));
        assert_eq!(a.take(p, p + 4 * PAGE), [(p + PAGE, p + 2 * PAGE)]);
        drop(writing(p + 3 * PAGE + 5, 1));
        assert_eq!(a.take(p, p + 4 * PAGE), [(p + 3 * PAGE, p + 4 * PAGE)]);
        title_memory_remapped(p, PAGE);
        assert!(a.changed(p / PAGE) && !a.changed(p / PAGE + 1));
        drop((a, b));
        free(p);
    }

    /// A range running from free memory over another allocation into a pool
    /// (a 4 MiB window of memory outside buffers) tracks the pool's pages
    /// and leaves the other allocation alone: taking guards the pool only.
    #[test]
    fn a_range_over_foreign_memory_guards_only_the_runtimes() {
        use windows_sys::Win32::System::Memory::{MEMORY_BASIC_INFORMATION, PAGE_READONLY, VirtualQuery};
        const WINDOW: usize = 4 << 20;
        // In a free 4 MiB-aligned window: plain committed memory (as the
        // process heap's) 1 MiB in, a pool 2 MiB in. Other tests allocate
        // meanwhile, so a window taken under our feet is tried again.
        let (base, foreign, pool) = (0..8)
            .find_map(|_| {
                // SAFETY: a fresh reservation, released at once; then
                // allocations in the window it left free, freed below.
                unsafe {
                    let p = VirtualAlloc(std::ptr::null(), 2 * WINDOW, MEM_RESERVE, PAGE_READWRITE) as usize;
                    VirtualFree(p as *mut _, 0, MEM_RELEASE);
                    let base = p.next_multiple_of(WINDOW);
                    let at = |offset: usize, kind: u32| {
                        VirtualAlloc((base + offset) as *const _, PAGE, MEM_RESERVE | MEM_COMMIT | kind, PAGE_READWRITE) as usize
                    };
                    let (foreign, pool) = (at(1 << 20, 0), at(2 << 20, MEM_WRITE_WATCH));
                    if p != 0 && foreign == base + (1 << 20) && pool == base + (2 << 20) {
                        return Some((base, foreign, pool));
                    }
                    [foreign, pool].into_iter().filter(|&q| q != 0).for_each(free);
                    None
                }
            })
            .expect("a free window");
        let t = track(base, WINDOW, true).expect("the pool is tracked");
        let taken = t.take(base, base + WINDOW);
        assert!(taken.iter().any(|&(a, b)| a <= pool && pool + PAGE <= b), "{taken:x?}");
        assert!(!taken.iter().any(|&(a, b)| a < foreign + PAGE && foreign < b), "{taken:x?}");
        let protection = |p: usize| {
            // SAFETY: plain data filled in by the query.
            let mut info: MEMORY_BASIC_INFORMATION = unsafe { std::mem::zeroed() };
            // SAFETY: a valid out pointer of the right size.
            unsafe { VirtualQuery(p as *const _, &mut info, size_of_val(&info)) };
            info.Protect
        };
        assert_eq!(protection(pool), PAGE_READONLY, "the pool is guarded");
        assert_eq!(protection(foreign), PAGE_READWRITE, "the other allocation is not");
        drop(t);
        free(foreign);
        free(pool);
    }

    /// A tracker's dirty bit goes back when it goes, and the next tracker
    /// starts dirty on it (everything changed at first).
    #[test]
    fn a_new_tracker_starts_dirty_on_a_bit_given_back() {
        let p = pool(1);
        let a = track(p, PAGE, true).expect("tracked");
        assert_eq!(a.take(p, p + PAGE).len(), 1);
        assert!(a.clean() && clean_bit(a.bit()));
        drop(a);
        let b = track(p, PAGE, true).expect("tracked");
        assert!(!b.clean() && !clean_bit(b.bit()));
        assert!(!clean_bit(None), "no bit: never clean");
        drop(b);
        free(p);
    }

    /// A page with nothing mapped is taken as zeros: `take_split` lists it
    /// apart from the pages to copy.
    #[test]
    fn unmapped_pages_are_taken_apart() {
        // SAFETY: a fresh write-watched reservation of two pages, the first
        // committed; freed below.
        let p = unsafe {
            let p = VirtualAlloc(std::ptr::null(), 2 * PAGE, MEM_RESERVE | MEM_WRITE_WATCH, PAGE_READWRITE) as usize;
            assert_ne!(p, 0);
            assert_ne!(VirtualAlloc(p as *const _, PAGE, MEM_COMMIT, PAGE_READWRITE) as usize, 0);
            p
        };
        let t = track(p, 2 * PAGE, true).expect("tracked");
        assert_eq!(t.take_split(p, p + 2 * PAGE), (vec![(p, p + PAGE)], vec![(p + PAGE, p + 2 * PAGE)]));
        assert!(t.take(p, p + 2 * PAGE).is_empty(), "both taken");
        drop(t);
        free(p);
    }

    /// A page only partly in the range asked is copied in part and stays
    /// changed; edge pages count whole when the tracked part is covered.
    #[test]
    fn partial_pages_stay_changed() {
        let p = pool(3);
        let t = track(p + 100, 2 * PAGE, true).expect("tracked");
        assert_eq!(t.take(p + 100, p + PAGE + 10), [(p + 100, p + PAGE + 10)]);
        assert!(!t.changed(p / PAGE) && t.changed(p / PAGE + 1));
        assert!(!t.clean(), "a page taken in part stays changed");
        assert_eq!(t.take(p, p + 3 * PAGE), [(p + PAGE, p + 2 * PAGE + 100)]);
        assert!(t.take(p, p + 3 * PAGE).is_empty());
        assert!(t.clean());
        drop(t);
        free(p);
    }

    /// A take refused for a changed page under a forbidden run takes
    /// nothing and names the run's largest key; changed pages outside them
    /// are taken as usual, and a forbidden page that is not changed does
    /// not refuse it. A page rearmed after unarming keeps its bytes again.
    #[test]
    fn takes_refuse_changed_pages_under_forbidden_runs() {
        let p = pool(3);
        let t = track(p, 3 * PAGE, true).expect("tracked");
        let first = p / PAGE;
        let forbidden = [(first + 1, first + 2, 7), (first + 1, first + 3, 9), (first + 5, first + 6, 11)];
        assert_eq!(t.take_split_unless(p, p + 3 * PAGE, &forbidden), Err(9), "all three changed at first");
        assert!(t.changed(first) && t.changed(first + 1), "nothing taken");
        assert_eq!(t.take_split_unless(p, p + PAGE, &forbidden), Ok((vec![(p, p + PAGE)], Vec::new())), "page 0 is not forbidden");
        assert_eq!(t.take(p + PAGE, p + 3 * PAGE).len(), 1);
        // SAFETY: a byte of our pool's first page.
        unsafe { (p as *mut u8).write_volatile(1) };
        assert_eq!(t.take_split_unless(p, p + 3 * PAGE, &forbidden), Ok((vec![(p, p + PAGE)], Vec::new())), "pages 1 and 2 unchanged");
        // Armed, unarmed (a copy-back under way), rearmed: the next first
        // write keeps the page's bytes.
        t.arm(p, p + PAGE);
        t.unarm(first);
        t.rearm(first);
        // SAFETY: as above.
        unsafe { (p as *mut u8).write_volatile(2) };
        assert_eq!(t.saved(first).map(|b| b[0]), Some(1), "rearmed: its bytes before the write kept");
        t.disarm();
        drop(t);
        free(p);
    }

    /// Pages of the ranges armed keep their bytes from before the first
    /// write after they were taken, whichever overlapping ranges armed
    /// them; a page unarmed keeps none until it is armed again, and none
    /// does once the tracker is disarmed.
    #[test]
    fn armed_ranges_keep_their_pages_bytes_from_before_the_first_write() {
        let p = pool(4);
        let t = track(p, 4 * PAGE, true).expect("pool memory is tracked");
        let all = || t.take(p, p + 4 * PAGE);
        // SAFETY: (every write below) a byte of our pool; a guarded page
        // faults into the handler, which saves it first where it is armed.
        let write = |page: usize, value: u8| unsafe { ((p + page * PAGE + 8) as *mut u8).write_volatile(value) };
        let saved = |page: usize| t.saved(p / PAGE + page).map(|bytes| bytes[8]);
        all();
        write(0, 1);
        assert_eq!(saved(0), None, "nothing armed yet");
        all();
        // Bytes 100 of page 0 to 1 of page 2: pages 0 to 2, page 3 not.
        t.arm(p + 100, p + 2 * PAGE + 1);
        t.arm(p + PAGE, p + 2 * PAGE);
        t.arm(p + PAGE + 5, p + 2 * PAGE + 1);
        t.unarm(p / PAGE + 1);
        for page in 0..4 {
            write(page, 2);
        }
        assert_eq!([saved(0), saved(1), saved(2), saved(3)], [Some(1), None, Some(0), None]);
        assert_eq!(saved(0), None, "handed out once");
        all();
        // Armed again by a range holding it.
        t.arm(p + PAGE, p + 2 * PAGE);
        write(1, 3);
        write(1, 4);
        assert_eq!(saved(1), Some(2), "from before the first write");
        t.disarm();
        all();
        write(2, 5);
        assert_eq!(saved(2), None, "disarmed");
        drop(t);
        free(p);
    }

    /// Runs armed that touch or overlap merge into one, so a page's check
    /// stays a search however many sub-ranges are armed; pages unarmed in
    /// them stay unarmed until a range holding them is armed again.
    #[test]
    fn adjacent_and_overlapping_armed_runs_coalesce() {
        let mut s = Saved::default();
        // Adjacent: [2k, 2k + 2) end to end, in order and out of it.
        for k in 0..200 {
            s.arm(2 * k, 2 * k + 2);
        }
        assert_eq!(s.armed, [(0, 400)]);
        let mut s = Saved::default();
        for k in (0..200).rev() {
            s.arm(2 * k, 2 * k + 2);
        }
        assert_eq!(s.armed, [(0, 400)]);
        // Overlapping, nested and disjoint.
        let mut s = Saved::default();
        s.arm(10, 20);
        s.arm(15, 25);
        assert_eq!(s.armed, [(10, 25)]);
        s.arm(12, 14);
        assert_eq!(s.armed, [(10, 25)], "nested");
        s.arm(40, 50);
        s.arm(0, 5);
        assert_eq!(s.armed, [(0, 5), (10, 25), (40, 50)], "disjoint, sorted");
        s.arm(5, 10);
        assert_eq!(s.armed, [(0, 25), (40, 50)], "touching both sides");
        s.arm(30, 35);
        s.arm(20, 45);
        assert_eq!(s.armed, [(0, 50)], "spanning several");
        s.arm(60, 70);
        s.arm(55, 55);
        assert_eq!(s.armed, [(0, 50), (60, 70)], "an empty run arms nothing");
        assert!(s.armed(0) && s.armed(49) && !s.armed(50) && !s.armed(59));
        assert!(s.armed(60) && s.armed(69) && !s.armed(70) && !s.armed(1000));
        // Unarmed pages stay out of a merged run until armed again.
        s.unarmed.insert(45);
        s.unarmed.insert(65);
        s.arm(50, 60);
        assert_eq!(s.armed, [(0, 70)]);
        assert!(!s.armed(45) && !s.armed(65) && s.armed(55));
        s.arm(44, 46);
        assert!(s.armed(45) && !s.armed(65));
        s.arm(0, 70);
        assert!(s.armed(65) && s.unarmed.is_empty());
    }

    /// Plain committed memory (the C heap) is not tracked; reserved memory
    /// is, reads as taken, and changes when the runtime maps it.
    #[test]
    fn heap_memory_is_not_tracked_and_placeholders_wait_for_mappings() {
        let plain = vec![0u8; 2 * PAGE];
        assert!(track(plain.as_ptr() as usize, plain.len(), true).is_none());
        // SAFETY: a reservation of ours, freed below.
        let r = unsafe { VirtualAlloc(std::ptr::null(), 2 * PAGE, MEM_RESERVE, PAGE_READWRITE) } as usize;
        assert_ne!(r, 0);
        let t = track(r, 2 * PAGE, true).expect("reserved memory is tracked");
        assert_eq!(t.take(r, r + 2 * PAGE).len(), 1, "copied once (as zeros)");
        assert!(t.take(r, r + 2 * PAGE).is_empty());
        title_memory_remapped(r, PAGE);
        assert_eq!(t.take(r, r + 2 * PAGE), [(r, r + PAGE)]);
        drop(t);
        // SAFETY: our reservation.
        unsafe { VirtualFree(r as *mut _, 0, MEM_RELEASE) };
    }

    /// A kernel read into a guarded view block fails; opened first, it
    /// lands and the block counts as changed.
    #[test]
    fn kernel_reads_into_guarded_blocks_after_opening() {
        use std::io::{Read, Seek, Write};
        use windows_sys::Win32::System::Memory::{
            CreateFileMappingW, FILE_MAP_WRITE, MapViewOfFile, UnmapViewOfFile,
        };
        // SAFETY: a pagefile section of two blocks, mapped read-write.
        let view = unsafe {
            let section = CreateFileMappingW(
                -1isize as _,
                std::ptr::null(),
                PAGE_READWRITE,
                0,
                (2 * PAGE) as u32,
                std::ptr::null(),
            );
            assert!(!section.is_null());
            MapViewOfFile(section, FILE_MAP_WRITE, 0, 0, 2 * PAGE).Value as usize
        };
        assert_ne!(view, 0);
        assert!(track(view, 2 * PAGE, false).is_none(), "views only where asked");
        let t = track(view, 2 * PAGE, true).expect("views are tracked");
        assert_eq!(t.take(view, view + 2 * PAGE), [(view, view + 2 * PAGE)]); // now read-only
        let mut file = tempfile::tempfile().unwrap();
        file.write_all(b"from the disk").unwrap();
        // `ReadFile` inside: the kernel writes the view.
        let read = |file: &mut std::fs::File| {
            file.rewind().unwrap();
            // SAFETY: 13 bytes of our view, nothing else refers to them.
            let into = unsafe { std::slice::from_raw_parts_mut((view + 100) as *mut u8, 13) };
            file.read(into).is_ok_and(|n| n == 13)
        };
        assert!(!read(&mut file), "the kernel does not fault into our handler");
        let write = writing(view + 100, 13);
        // A tracker taking the page meanwhile leaves it open.
        assert_eq!(t.take(view, view + 2 * PAGE), [(view, view + PAGE)]);
        assert!(read(&mut file));
        drop(write);
        // SAFETY: the bytes just read.
        assert_eq!(unsafe { std::slice::from_raw_parts((view + 100) as *const u8, 13) }, b"from the disk");
        assert_eq!(t.take(view, view + 2 * PAGE), [(view, view + PAGE)]);
        drop(t);
        // SAFETY: our view.
        unsafe { UnmapViewOfFile(windows_sys::Win32::System::Memory::MEMORY_MAPPED_VIEW_ADDRESS { Value: view as *mut _ }) };
    }

    /// `pages` pages of a fresh pagefile section mapped twice (as the
    /// title's pool view and tiled heap show the same physical pages),
    /// reported as the runtime does when it maps them.
    pub(crate) fn twin_views(pages: usize) -> (usize, usize) {
        use windows_sys::Win32::System::Memory::{CreateFileMappingW, FILE_MAP_WRITE, MapViewOfFile};
        let len = pages * PAGE;
        // SAFETY: a pagefile section mapped read-write twice; the test
        // keeps both views.
        let (a, b) = unsafe {
            let section = CreateFileMappingW(
                -1isize as _,
                std::ptr::null(),
                PAGE_READWRITE,
                0,
                len as u32,
                std::ptr::null(),
            );
            assert!(!section.is_null());
            (
                MapViewOfFile(section, FILE_MAP_WRITE, 0, 0, len).Value as usize,
                MapViewOfFile(section, FILE_MAP_WRITE, 0, 0, len).Value as usize,
            )
        };
        assert!(a != 0 && b != 0);
        for k in (0..len).step_by(PAGE) {
            title_memory_aliased(b + k, &[a + k]);
        }
        (a, b)
    }

    /// A write through another mapping of the same pages marks the tracker
    /// (the guard there faults, the runtime says where else it is mapped);
    /// so does a write the runtime reports at the other mapping.
    #[test]
    fn writes_through_another_mapping_mark_the_tracker() {
        let (a, b) = twin_views(2);
        let t = track(a, 2 * PAGE, true).expect("views are tracked");
        assert_eq!(t.take(a, a + 2 * PAGE), [(a, a + 2 * PAGE)]);
        // SAFETY: the second view of the pages, guarded by the take.
        unsafe { ((b + PAGE + 4) as *mut u32).write_volatile(7) };
        assert_eq!(t.take(a, a + 2 * PAGE), [(a + PAGE, a + 2 * PAGE)]);
        // SAFETY: the tracked view, which shows the write.
        assert_eq!(unsafe { ((a + PAGE + 4) as *const u32).read_volatile() }, 7);
        drop(writing(b, 8));
        assert_eq!(t.take(a, a + 2 * PAGE), [(a, a + PAGE)]);
        assert!(t.take(a, a + 2 * PAGE).is_empty());
        // A third mapping made later starts guarded, so its writes count.
        let (c, _) = twin_views(1);
        title_memory_aliased(c, &[a]);
        // SAFETY: a view of ours (another section: only the tracking
        // takes it for `a`'s pages here).
        unsafe { (c as *mut u32).write_volatile(1) };
        assert_eq!(t.take(a, a + 2 * PAGE), [(a, a + PAGE)]);
    }

    /// A tracker of one mapping going away leaves the guard its page shares
    /// with a tracker of another mapping, so a write there still counts (it
    /// was opened, and the pages it went unseen in stayed clean).
    #[test]
    fn a_tracker_dropped_on_one_mapping_leaves_the_other_guarded() {
        let (a, b) = twin_views(1);
        let t = track(a, PAGE, true).expect("views are tracked");
        let u = track(b, PAGE, true).expect("views are tracked");
        assert_eq!(t.take(a, a + PAGE), [(a, a + PAGE)]);
        assert_eq!(u.take(b, b + PAGE), [(b, b + PAGE)]);
        drop(u);
        // SAFETY: the second view, guarded for `t`'s page too.
        unsafe { ((b + 4) as *mut u32).write_volatile(7) };
        assert_eq!(t.take(a, a + PAGE), [(a, a + PAGE)]);
    }

    /// A retake has every copy of the memory, and of its other mappings,
    /// take it again, and leaves it guarded: a later write still counts.
    #[test]
    fn a_retake_marks_the_copies_and_keeps_the_guard() {
        use windows_sys::Win32::System::Memory::{MEMORY_BASIC_INFORMATION, PAGE_READONLY, VirtualQuery};
        let protection = |at: usize| {
            // SAFETY: plain data filled in by the query.
            let mut info: MEMORY_BASIC_INFORMATION = unsafe { std::mem::zeroed() };
            // SAFETY: a valid out pointer of the right size.
            unsafe { VirtualQuery(at as *const _, &mut info, size_of_val(&info)) };
            info.Protect
        };
        let (a, b) = twin_views(2);
        let t = track_copy(a, 2 * PAGE, true).expect("views are tracked");
        let u = track_copy(b, 2 * PAGE, true).expect("views are tracked");
        assert_eq!(t.take(a, a + 2 * PAGE), [(a, a + 2 * PAGE)]);
        assert_eq!(u.take(b, b + 2 * PAGE), [(b, b + 2 * PAGE)]);
        assert!(t.clean() && u.clean());
        retake(a + PAGE + 16, 32);
        // Marked in both copies, the second mapping's too, still guarded.
        assert!(!t.clean() && !u.clean());
        assert!(t.changed((a + PAGE) / PAGE) && u.changed((b + PAGE) / PAGE));
        assert!(!t.changed(a / PAGE));
        assert_eq!(protection(a + PAGE), PAGE_READONLY);
        assert_eq!(protection(b + PAGE), PAGE_READONLY);
        // Taken again (the copies re-copy that page), then guarded and clean.
        assert_eq!(t.take(a, a + 2 * PAGE), [(a + PAGE, a + 2 * PAGE)]);
        assert_eq!(u.take(b, b + 2 * PAGE), [(b + PAGE, b + 2 * PAGE)]);
        assert!(t.clean() && u.clean());
        assert_eq!(protection(a + PAGE), PAGE_READONLY);
        // SAFETY: the second view, guarded all along.
        unsafe { ((b + PAGE + 8) as *mut u32).write_volatile(3) };
        assert_eq!(t.take(a, a + 2 * PAGE), [(a + PAGE, a + 2 * PAGE)]);
        assert_eq!(u.take(b, b + 2 * PAGE), [(b + PAGE, b + 2 * PAGE)]);
        // A retake while marked and guarded, then a drop: nothing breaks.
        retake(a, 2 * PAGE);
        drop(u);
        assert_eq!(t.take(a, a + 2 * PAGE), [(a, a + 2 * PAGE)]);
        assert!(t.clean());
    }

    /// A retake marks host buffer copies holding the memory, not other
    /// trackers of it (a texture's staging): shaders' writes reach only the
    /// copies. A title write still marks both.
    #[test]
    fn a_retake_marks_buffer_copies_alone() {
        let p = pool(2);
        let copy = track_copy(p, 2 * PAGE, true).expect("pool memory is tracked");
        let texture = track(p, 2 * PAGE, true).expect("pool memory is tracked");
        assert_eq!(copy.take(p, p + 2 * PAGE), [(p, p + 2 * PAGE)]);
        assert_eq!(texture.take(p, p + 2 * PAGE), [(p, p + 2 * PAGE)]);
        retake(p + PAGE, 16);
        assert!(copy.changed(p / PAGE + 1), "the copy takes it again");
        assert!(!texture.changed(p / PAGE + 1) && texture.clean(), "the texture's tracker is left alone");
        assert_eq!(copy.take(p, p + 2 * PAGE), [(p + PAGE, p + 2 * PAGE)]);
        // SAFETY: a page of our pool, guarded: the handler opens it.
        unsafe { ((p + PAGE + 8) as *mut u8).write_volatile(1) };
        assert!(copy.changed(p / PAGE + 1) && texture.changed(p / PAGE + 1), "a title write marks both");
    }

    /// A write under way on another mapping of a page keeps the page open
    /// there, so taking the page (or marking it taken) through its tracked
    /// mapping must leave it changed and keep what was saved of it; aliasing
    /// a busy mapping to a guarded one must mark the guarded page changed.
    #[test]
    fn a_busy_alias_leaves_the_page_changed_and_keeps_saved_bytes() {
        let (a, b) = twin_views(1);
        let t = track(a, PAGE, true).expect("views are tracked");
        assert_eq!(t.take(a, a + PAGE), [(a, a + PAGE)]);
        t.arm(a, a + PAGE);
        let write = writing(b, 16);
        assert!(t.changed(a / PAGE));
        assert_eq!(t.take(a, a + PAGE), [(a, a + PAGE)]);
        assert!(t.changed(a / PAGE), "take must leave the page changed");
        t.taken(&[a / PAGE]);
        assert!(t.changed(a / PAGE), "taken must leave the page changed");
        // SAFETY: the second view of the page, kept open by `write`.
        unsafe { (b as *mut u32).write_volatile(0x1234) };
        assert_eq!(t.take(a, a + PAGE), [(a, a + PAGE)]);
        drop(write);
        let saved = t.saved(a / PAGE).expect("saved bytes survive");
        assert_eq!(&saved[..4], &[0, 0, 0, 0]);
        assert_eq!(t.take(a, a + PAGE), [(a, a + PAGE)]);
        assert!(t.take(a, a + PAGE).is_empty(), "taken once the write ended");
        let (c, _) = twin_views(1);
        let busy_c = writing(c, 16);
        title_memory_aliased(c, &[a]);
        assert!(
            t.changed(a / PAGE),
            "aliasing a busy page marks the tracked one"
        );
        drop(busy_c);
    }

    /// A write or mapping that runs past the end of the address space (a
    /// title's `fread` with a huge count) covers up to the end; the sum used
    /// to overflow. Up there no test has guarded blocks.
    #[test]
    fn ranges_past_the_address_space_end_do_not_overflow() {
        let top = usize::MAX - 16;
        drop(writing(top, PAGE));
        title_memory_remapped(top, usize::MAX);
    }

    /// Lazy view pages on while it lives, as before after (and no earlier
    /// test's misses: copy-back makes no page lazy while writes are
    /// withheld).
    pub(crate) struct LazyOn(bool);

    impl LazyOn {
        pub(crate) fn set() -> LazyOn {
            let before = lazy_views();
            set_lazy_views(true);
            crate::memory::forget_misses();
            LazyOn(before)
        }
    }

    impl Drop for LazyOn {
        fn drop(&mut self) {
            set_lazy_views(self.0);
        }
    }

    /// `n` title views of one fresh pagefile section of `pages` pages,
    /// reported as showing the same pages (as the runtime's physical-page
    /// views), and the runtime's own view of it, through which lazy pages
    /// get their bytes. The test keeps them mapped.
    pub(crate) fn section_views(pages: usize, n: usize) -> (Vec<usize>, usize) {
        use windows_sys::Win32::System::Memory::{CreateFileMappingW, FILE_MAP_WRITE, MapViewOfFile};
        let len = pages * PAGE;
        // SAFETY: a pagefile section mapped `n + 1` times, never unmapped.
        let views: Vec<usize> = unsafe {
            let section = CreateFileMappingW(-1isize as _, std::ptr::null(), PAGE_READWRITE, 0, len as u32, std::ptr::null());
            assert!(!section.is_null());
            (0..=n).map(|_| MapViewOfFile(section, FILE_MAP_WRITE, 0, 0, len).Value as usize).collect()
        };
        assert!(views.iter().all(|&v| v != 0));
        for i in 1..n {
            for k in (0..len).step_by(PAGE) {
                let shown: Vec<usize> = views[..i].iter().map(|v| v + k).collect();
                title_memory_aliased(views[i] + k, &shown);
            }
        }
        (views[..n].to_vec(), views[n])
    }

    /// `writable_alias` answers `own` for title memory `[title, title + len)`
    /// (the runtime's own view of the pages a title view shows).
    pub(crate) fn own_alias(title: usize, len: usize, own: usize) {
        static AT: [AtomicUsize; 3] = [const { AtomicUsize::new(0) }; 3];
        AT[0].store(title, Ordering::Relaxed);
        AT[1].store(len, Ordering::Relaxed);
        AT[2].store(own, Ordering::Relaxed);
        set_writable_alias(|at| {
            let (title, len, own) = (AT[0].load(Ordering::Relaxed), AT[1].load(Ordering::Relaxed), AT[2].load(Ordering::Relaxed));
            (title..title + len).contains(&at).then(|| at - title + own)
        });
    }

    /// A byte of title memory read through the kernel with no look at lazy
    /// pages: none while the page has no access.
    pub(crate) fn raw(at: usize) -> Option<u8> {
        let mut b = [0u8];
        crate::host::read_raw(at as *const u8, &mut b).then_some(b[0])
    }

    /// A byte read in process (a fault gives a lazy page its bytes).
    fn byte(at: usize) -> u8 {
        // SAFETY: a byte of a test's view, mapped for the whole test.
        unsafe { (at as *const u8).read_volatile() }
    }

    /// The lazy offer of page `k` of title view `a`, its copy's bytes in
    /// `copy` (a page each) and the runtime's view at `own`.
    fn offer(a: usize, k: usize, copy: &[u8], own: usize) -> LazyView {
        LazyView {
            page: a / PAGE + k,
            src: copy.as_ptr() as usize + k * PAGE,
            alias: own + k * PAGE,
        }
    }

    /// A lazy view page has no access and title memory keeps its bytes (the
    /// runtime's own view shows them); the title's first read faults, the
    /// copy's bytes go in and the page is guarded as taken, so a later write
    /// faults once and marks it changed. Another tracker holding the page
    /// counts it changed from the start; a page not made lazy stays guarded.
    #[test]
    fn a_lazy_view_page_gets_the_copys_bytes_at_the_titles_first_read() {
        use windows_sys::Win32::System::Memory::{PAGE_NOACCESS, PAGE_READONLY};
        let _on = LazyOn::set();
        let missed = lazy_checks().1;
        let (v, own) = section_views(2, 1);
        let a = v[0];
        let t = track_copy(a, 2 * PAGE, true).expect("views are tracked");
        let u = track(a + PAGE, PAGE, true).expect("views are tracked");
        t.take(a, a + 2 * PAGE);
        u.take(a, a + 2 * PAGE);
        let copy = vec![0x5au8; 2 * PAGE];
        let page = a / PAGE + 1;
        assert_eq!(t.lease().lazify(&[offer(a, 1, &copy, own)]), [page]);
        assert!(u.changed(page) && !t.changed(page), "the other copy is older now");
        assert_eq!(protection(a + PAGE), Some(PAGE_NOACCESS));
        assert_eq!(raw(a + PAGE + 8), None, "no access");
        assert_eq!(byte(own + PAGE + 8), 0, "title memory keeps its bytes");
        assert_eq!(protection(a), Some(PAGE_READONLY), "page 0 not lazy");
        assert_eq!(byte(a + PAGE + 8), 0x5a, "the first read faults and finds the copy's bytes");
        assert!(!maybe_lazy(page));
        assert_eq!(protection(a + PAGE), Some(PAGE_READONLY), "guarded as taken");
        assert!(!t.changed(page) && t.take(a, a + 2 * PAGE).is_empty(), "the copy holds it as it is");
        // SAFETY: a byte of the view, read-only: the handler opens it.
        unsafe { ((a + PAGE + 8) as *mut u8).write_volatile(1) };
        assert!(t.changed(page), "a write faults as on any taken page");
        assert_eq!((byte(a + PAGE + 8), byte(a + PAGE + 9)), (1, 0x5a));
        assert_eq!(lazy_checks().1, missed, "nothing wrote behind our back");
        drop((t, u));
    }

    /// Title threads reading a lazy page at once all find the copy's bytes:
    /// they go in while the page has no access (one reader gives them, the
    /// others wait for the guard lock and retry). Bytes put in after opening
    /// the page lost this race (run lazy1-u1's crash, `perf/lazy-copyback`).
    #[test]
    fn racing_first_reads_of_a_lazy_view_page_all_find_the_copys_bytes() {
        let _on = LazyOn::set();
        for round in 0..20u8 {
            let (v, own) = section_views(1, 1);
            let a = v[0];
            let t = track_copy(a, PAGE, true).expect("views are tracked");
            t.take(a, a + PAGE);
            let copy = vec![0x40 | round; PAGE];
            assert_eq!(t.lease().lazify(&[offer(a, 0, &copy, own)]).len(), 1);
            let read: Vec<u8> = std::thread::scope(|s| {
                let readers: Vec<_> = (0..8).map(|k| s.spawn(move || byte(a + 512 * k))).collect();
                readers.into_iter().map(|r| r.join().unwrap()).collect()
            });
            assert_eq!(read, [0x40 | round; 8], "round {round}");
            drop(t);
        }
    }

    /// The title's first touch is a write: the copy's bytes go in under it,
    /// and an armed tracker keeps them as the page's bytes from before.
    #[test]
    fn a_write_to_a_lazy_view_page_lands_on_the_copys_bytes() {
        let _on = LazyOn::set();
        let (v, own) = section_views(1, 1);
        let a = v[0];
        let t = track_copy(a, PAGE, true).expect("views are tracked");
        t.take(a, a + PAGE);
        t.arm(a, a + PAGE);
        let copy = vec![0x77u8; PAGE];
        assert_eq!(t.lease().lazify(&[offer(a, 0, &copy, own)]).len(), 1);
        // SAFETY: a byte of the view, without access: the handler gives it.
        unsafe { ((a + 4) as *mut u8).write_volatile(1) };
        assert_eq!((raw(a + 4), raw(a + 8)), (Some(1), Some(0x77)), "the copy's bytes around the write");
        assert!(t.changed(a / PAGE));
        assert_eq!(t.saved(a / PAGE).as_deref(), Some(&copy[..]), "saved: the copy's bytes, before the write");
        drop(t);
    }

    /// Our host's touches give lazy pages the copy's bytes first: another
    /// copy's refresh (it takes the page as it is, then reads it through the
    /// kernel), a kernel read, a kernel write, a copy from it, the check
    /// before a read, the runtime unmapping it, the copy going.
    #[test]
    fn our_own_touches_give_lazy_view_pages_their_bytes_first() {
        use windows_sys::Win32::System::Memory::{PAGE_NOACCESS, PAGE_READONLY};
        let _on = LazyOn::set();
        let (v, own) = section_views(6, 1);
        let a = v[0];
        let t = track_copy(a, 6 * PAGE, true).expect("views are tracked");
        let u = track(a, PAGE, true).expect("views are tracked");
        t.take(a, a + 6 * PAGE);
        u.take(a, a + PAGE);
        let copy: Vec<u8> = (0..6 * PAGE).map(|i| (i / PAGE) as u8 + 1).collect();
        let offers: Vec<LazyView> = (0..6).map(|k| offer(a, k, &copy, own)).collect();
        assert_eq!(t.lease().lazify(&offers).len(), 6);
        assert_eq!(u.take(a, a + PAGE), [(a, a + PAGE)]);
        assert_eq!(protection(a), Some(PAGE_NOACCESS), "taking a lazy page leaves it without access");
        let mut bytes = [0u8; 4];
        assert!(crate::host::read_into((a + 16) as *const u8, &mut bytes));
        assert_eq!(bytes, [1; 4], "a kernel read finds the copy's bytes");
        assert_eq!(protection(a), Some(PAGE_READONLY));
        let w = writing(a + PAGE + 10, 1);
        crate::host::write_mapped((a + PAGE + 10) as *mut u8, [9u8].as_ptr(), 1);
        drop(w);
        assert_eq!((raw(a + PAGE + 10), raw(a + PAGE + 11)), (Some(9), Some(2)), "a kernel write lands on them");
        let mut moved = [0u8; 16];
        crate::host::copy_mapped(moved.as_mut_ptr(), (a + 2 * PAGE) as *const u8, 16, false);
        assert_eq!(moved, [3; 16], "a copy from title memory");
        assert!(crate::texture::committed((a + 3 * PAGE) as *const u8, 8));
        assert_eq!(raw(a + 3 * PAGE), Some(4), "the check before a read");
        title_memory_remapped(a + 4 * PAGE, PAGE);
        assert_eq!(raw(a + 4 * PAGE), Some(5), "unmapped: the physical page keeps them");
        t.resolve_all();
        assert_eq!(raw(a + 5 * PAGE + 5), Some(6), "the copy going");
        assert!((0..6).all(|k| !maybe_lazy(a / PAGE + k)));
        drop((t, u));
    }

    /// A page shown at two addresses goes lazy at both: the other mapping's
    /// tracker counts it changed, a read there gives the copy's bytes to
    /// both, guarded again together.
    #[test]
    fn a_lazy_view_page_shown_elsewhere_is_lazy_at_every_address() {
        use windows_sys::Win32::System::Memory::{PAGE_NOACCESS, PAGE_READONLY};
        let _on = LazyOn::set();
        let (v, own) = section_views(1, 2);
        let (a, b) = (v[0], v[1]);
        let t = track_copy(a, PAGE, true).expect("views are tracked");
        let u = track(b, PAGE, true).expect("views are tracked");
        t.take(a, a + PAGE);
        u.take(b, b + PAGE);
        let copy = vec![6u8; PAGE];
        assert_eq!(t.lease().lazify(&[offer(a, 0, &copy, own)]), [a / PAGE]);
        assert!(u.changed(b / PAGE) && !t.changed(a / PAGE));
        assert_eq!((protection(a), protection(b)), (Some(PAGE_NOACCESS), Some(PAGE_NOACCESS)));
        assert_eq!(byte(b + 3), 6, "a read at the other address");
        assert_eq!((protection(a), protection(b)), (Some(PAGE_READONLY), Some(PAGE_READONLY)));
        assert!(!maybe_lazy(a / PAGE) && !maybe_lazy(b / PAGE));
        assert_eq!(raw(a + 3), Some(6), "the same physical page");
        drop((t, u));
    }

    /// Pages a tracker cannot make lazy go back at once: one the title wrote
    /// since it was taken, one under a kernel write, one the runtime writes
    /// through its own view, one lazy already, one shown elsewhere where a
    /// write of the runtime's is under way.
    #[test]
    fn view_pages_that_cannot_be_lazy_are_left_out() {
        let _on = LazyOn::set();
        let (v, own) = section_views(5, 1);
        let a = v[0];
        let t = track_copy(a, 5 * PAGE, true).expect("views are tracked");
        t.take(a, a + 5 * PAGE);
        let copy = vec![1u8; 5 * PAGE];
        // SAFETY: a byte of the view, read-only: the handler opens it.
        unsafe { (a as *mut u8).write_volatile(1) };
        let busy = writing(a + PAGE, 1);
        let held = unseen_writing(|| vec![a + 2 * PAGE]);
        assert_eq!(t.lease().lazify(&[offer(a, 4, &copy, own)]).len(), 1);
        let offers: Vec<LazyView> = (0..5).map(|k| offer(a, k, &copy, own)).collect();
        assert_eq!(t.lease().lazify(&offers), [a / PAGE + 3]);
        drop((busy, held));
        t.resolve_all();
        // Shown elsewhere, written there through the runtime's own view.
        let (w, own2) = section_views(1, 2);
        let s = track_copy(w[0], PAGE, true).expect("views are tracked");
        s.take(w[0], w[0] + PAGE);
        let held = unseen_writing(|| vec![w[1]]);
        assert!(s.lease().lazify(&[offer(w[0], 0, &copy, own2)]).is_empty());
        drop(held);
        assert_eq!(s.lease().lazify(&[offer(w[0], 0, &copy, own2)]).len(), 1, "once it is done");
        s.resolve_all();
        drop((t, s));
    }

    /// The runtime writes a lazy page through its own view (DirectStorage's
    /// destinations, `xmem::page_window`): the page gets the copy's bytes
    /// first, stays out of lazy until the write is done, and the runtime's
    /// bytes, reported as written after, win where they land.
    #[test]
    fn a_runtime_write_to_a_lazy_view_page_lands_on_the_copys_bytes() {
        let _on = LazyOn::set();
        let (v, own) = section_views(1, 1);
        let a = v[0];
        let t = track_copy(a, PAGE, true).expect("views are tracked");
        t.take(a, a + PAGE);
        let copy = vec![0x33u8; PAGE];
        assert_eq!(t.lease().lazify(&[offer(a, 0, &copy, own)]).len(), 1);
        let held = unseen_writing(|| vec![a]);
        assert!(!maybe_lazy(a / PAGE), "given its bytes first");
        assert_eq!(raw(a + 100), Some(0x33));
        assert!(t.lease().lazify(&[offer(a, 0, &copy, own)]).is_empty(), "not lazy while written");
        // SAFETY: the runtime's own view, read-write.
        unsafe { std::ptr::write_bytes(own as *mut u8, 0x77, 16) };
        drop(held);
        drop(writing(a, PAGE));
        assert_eq!((raw(a), raw(a + 15), raw(a + 16)), (Some(0x77), Some(0x77), Some(0x33)));
        assert!(t.changed(a / PAGE), "the copy takes the runtime's bytes next");
        drop(t);
    }

    /// A write into a lazy page that goes past us (through the runtime's own
    /// view, unreported) is caught at the first touch: its title bytes are
    /// not those we left.
    #[test]
    fn a_write_behind_a_lazy_view_pages_back_is_caught() {
        let _on = LazyOn::set();
        let (v, own) = section_views(1, 1);
        let a = v[0];
        let t = track_copy(a, PAGE, true).expect("views are tracked");
        t.take(a, a + PAGE);
        let copy = vec![2u8; PAGE];
        assert_eq!(t.lease().lazify(&[offer(a, 0, &copy, own)]).len(), 1);
        let (checks, missed) = lazy_checks();
        // SAFETY: the runtime's own view, read-write.
        unsafe { ((own + 40) as *mut u8).write_volatile(0x99) };
        assert_eq!(byte(a + 40), 2, "the copy's bytes go in over it");
        assert_eq!(lazy_checks(), (checks + 1, missed + 1));
        drop(t);
    }

    /// What shaders wrote and the submission withheld stays out of title
    /// memory: a lazy page there is guarded again with the title's bytes,
    /// and the copy takes them again.
    #[test]
    fn a_withheld_lazy_view_page_keeps_the_titles_bytes() {
        use windows_sys::Win32::System::Memory::PAGE_READONLY;
        let _on = LazyOn::set();
        let (v, own) = section_views(1, 1);
        let a = v[0];
        let t = track_copy(a, PAGE, true).expect("views are tracked");
        t.take(a, a + PAGE);
        let copy = vec![3u8; PAGE];
        assert_eq!(t.lease().lazify(&[offer(a, 0, &copy, own)]).len(), 1);
        retake(a, PAGE);
        assert_eq!(protection(a), Some(PAGE_READONLY));
        assert_eq!(raw(a), Some(0), "the title's bytes");
        assert!(t.changed(a / PAGE), "the copy takes them again");
        drop(t);
    }

    /// Measurement, not a check (`--ignored`): per 64 KiB page under this
    /// process's Wine, a guard cycle (one fault into our handler, guarded
    /// again by a take) against the kernel's write watch (a write faulting
    /// inside Wine, found and reset by `GetWriteWatch`), and protection
    /// changes page by page against one per range. Spinning threads stand
    /// for the title's (a read-only change interrupts every CPU running the
    /// process).
    #[test]
    #[ignore = "measurement"]
    fn guards_against_write_watch_costs() {
        use windows_sys::Win32::System::Memory::{GetWriteWatch, PAGE_READONLY, ResetWriteWatch, VirtualProtect};
        const PAGES: usize = 1024;
        let stop = Arc::new(std::sync::atomic::AtomicBool::new(false));
        let spinners: Vec<_> = (0..8)
            .map(|_| {
                let stop = stop.clone();
                std::thread::spawn(move || {
                    while !stop.load(Ordering::Relaxed) {
                        std::hint::spin_loop();
                    }
                })
            })
            .collect();
        let us = |t: std::time::Instant| t.elapsed().as_secs_f64() * 1e6 / PAGES as f64;
        // SAFETY: (every write below) a byte of our pool.
        let touch = |p: usize| (0..PAGES).for_each(|k| unsafe { ((p + k * PAGE + 8) as *mut u8).write_volatile(1) });
        // Guards: take (one change for the range), a fault a page, take again.
        let p = pool(PAGES);
        let t = track(p, PAGES * PAGE, false).expect("tracked");
        touch(p);
        let at = std::time::Instant::now();
        t.take(p, p + PAGES * PAGE);
        let first_take = us(at);
        let at = std::time::Instant::now();
        touch(p);
        let faults = us(at);
        let at = std::time::Instant::now();
        t.take(p, p + PAGES * PAGE);
        let take = us(at);
        // Every other page written: the take guards a range a page.
        // SAFETY: as above.
        (0..PAGES).step_by(2).for_each(|k| unsafe { ((p + k * PAGE + 8) as *mut u8).write_volatile(2) });
        let at = std::time::Instant::now();
        t.take(p, p + PAGES * PAGE);
        let take_split = us(at) * 2.0;
        drop(t);
        free(p);
        // Write watch: reset, a write a page, ask and reset.
        let p = pool(PAGES);
        touch(p);
        // SAFETY: our pool.
        unsafe { ResetWriteWatch(p as *const _, PAGES * PAGE) };
        let at = std::time::Instant::now();
        touch(p);
        let watch_faults = us(at);
        let mut found = vec![0usize; PAGES * 16];
        let (mut count, mut granularity) = (found.len(), 0u32);
        let at = std::time::Instant::now();
        // SAFETY: room for every 4 KiB page of the pool.
        let ok = unsafe {
            GetWriteWatch(1, p as *const _, PAGES * PAGE, found.as_mut_ptr().cast(), &mut count, &mut granularity)
        };
        let watch_get = us(at);
        assert_eq!(ok, 0);
        // Protection changes: page by page, then one for all.
        let at = std::time::Instant::now();
        for k in 0..PAGES {
            let mut old = 0;
            // SAFETY: our pool.
            unsafe { VirtualProtect((p + k * PAGE) as *const _, PAGE, PAGE_READONLY, &mut old) };
        }
        let per_page = us(at);
        let mut old = 0;
        // SAFETY: our pool.
        unsafe { VirtualProtect(p as *const _, PAGES * PAGE, PAGE_READWRITE, &mut old) };
        let at = std::time::Instant::now();
        // SAFETY: our pool.
        unsafe { VirtualProtect(p as *const _, PAGES * PAGE, PAGE_READONLY, &mut old) };
        let one = at.elapsed().as_secs_f64() * 1e6;
        // SAFETY: our pool.
        unsafe { VirtualProtect(p as *const _, PAGES * PAGE, PAGE_READWRITE, &mut old) };
        free(p);
        // A sweep: 1 GiB of pool, 1,000 pages written, asked and reset in
        // one call (what a per-submission journal would do).
        const BIG: usize = 16384;
        let p = pool(BIG);
        // Populated, as the title's pools are: every 4 KiB page written once.
        // SAFETY: bytes of our pool.
        (0..BIG * PAGE).step_by(4096).for_each(|at| unsafe { ((p + at) as *mut u8).write_volatile(1) });
        // SAFETY: our pool.
        unsafe { ResetWriteWatch(p as *const _, BIG * PAGE) };
        // SAFETY: bytes of our pool.
        (0..BIG).step_by(16).for_each(|k| unsafe { ((p + k * PAGE) as *mut u8).write_volatile(1) });
        let mut found = vec![0usize; 4096];
        let (mut swept, mut granularity) = (found.len(), 0u32);
        let at = std::time::Instant::now();
        // SAFETY: room for 4,096 answers.
        unsafe { GetWriteWatch(1, p as *const _, BIG * PAGE, found.as_mut_ptr().cast(), &mut swept, &mut granularity) };
        let sweep = at.elapsed().as_secs_f64() * 1e6;
        let (mut count2, mut granularity) = (found.len(), 0u32);
        let at = std::time::Instant::now();
        // SAFETY: as above.
        unsafe { GetWriteWatch(1, p as *const _, BIG * PAGE, found.as_mut_ptr().cast(), &mut count2, &mut granularity) };
        let sweep_clean = at.elapsed().as_secs_f64() * 1e6;
        free(p);
        stop.store(true, Ordering::Relaxed);
        spinners.into_iter().for_each(|s| s.join().unwrap());
        println!(
            "per page, {PAGES} pages: guards: first take {first_take:.2} us, fault {faults:.2} us, take of all {take:.2} us, \
             take of every other {take_split:.2} us; write watch: fault {watch_faults:.2} us, get and reset {watch_get:.2} us \
             ({count} found); read-only page by page {per_page:.2} us, one call for all {one:.0} us; \
             sweep of 1 GiB populated: {sweep:.0} us with {swept} found, {sweep_clean:.0} us with {count2}"
        );
    }

    /// Writers on several threads and a taker copying what it takes: at the
    /// end, every page the taker's copy disagrees with title memory on is
    /// still changed in its tracker (no write lost between a take's guard
    /// and its copy), and no write faulted past the handler (the process
    /// lives).
    #[test]
    fn concurrent_writes_and_takes_lose_no_write() {
        use std::sync::atomic::{AtomicBool, Ordering::Relaxed};
        const PAGES: usize = 32;
        let p = pool(PAGES);
        let trackers: Vec<Tracked> = (0..PAGES / 4)
            .map(|i| track_copy(p + i * 4 * PAGE, 4 * PAGE, true).expect("tracked"))
            .collect();
        let mut copy = vec![0u32; PAGES * PAGE / 4];
        let stop = AtomicBool::new(false);
        std::thread::scope(|s| {
            let _stop = StopOnDrop(&stop);
            for w in 0..12u32 {
                let stop = &stop;
                s.spawn(move || {
                    let mut x = 0x2545_f491_u32.wrapping_mul(w + 1);
                    while !stop.load(Relaxed) {
                        x ^= x << 13;
                        x ^= x >> 17;
                        x ^= x << 5;
                        let at = p + (x as usize % PAGES) * PAGE + (x as usize >> 8 & 0xFFFC);
                        // SAFETY: the pool's pages; a guarded one faults
                        // into the handler, which opens it.
                        unsafe { (at as *mut u32).write_volatile(x | 1) };
                    }
                });
            }
            // Enough rounds to catch a lost write or a stray fault in release
            // (a guard race fell to it in 2 of 10 runs); fewer in debug.
            for round in 0..if cfg!(debug_assertions) { 200 } else { 3000 } {
                for (i, t) in trackers.iter().enumerate() {
                    if (round + i) % 3 == 0 {
                        t.arm(t.start, t.end);
                    }
                    for (a, b) in t.take(t.start, t.end) {
                        for at in (a..b).step_by(4) {
                            // SAFETY: the pool's pages, readable.
                            copy[(at - p) / 4] = unsafe { (at as *const u32).read_volatile() };
                        }
                    }
                    if round % 7 == 0 {
                        t.taken(&[t.start / PAGE]);
                        // The caller made its copy equal to memory first.
                        for at in (t.start..t.start + PAGE).step_by(4) {
                            // SAFETY: as above.
                            copy[(at - p) / 4] = unsafe { (at as *const u32).read_volatile() };
                        }
                    }
                    t.disarm();
                }
            }
        });
        for t in &trackers {
            for page in t.start / PAGE..t.end / PAGE {
                let base = page * PAGE;
                // SAFETY: the pool's pages, no writer left.
                let memory = unsafe { std::slice::from_raw_parts(base as *const u32, PAGE / 4) };
                let held = &copy[(base - p) / 4..(base - p) / 4 + PAGE / 4];
                assert!(
                    memory == held || t.changed(page),
                    "page {} differs from its copy and is not changed",
                    page - p / PAGE
                );
            }
        }
        drop(trackers);
        free(p);
    }

    /// Armed trackers and writers faulting on their pages on several
    /// threads: whatever a tracker saved of a page is the page as it was
    /// taken (the copy's bytes then), never bytes a write put there after.
    #[test]
    fn saved_bytes_are_the_page_as_taken_under_concurrent_faults() {
        use std::sync::atomic::{AtomicBool, Ordering::Relaxed};
        const PAGES: usize = 16;
        let p = pool(PAGES);
        let trackers: Vec<Tracked> = (0..PAGES / 4)
            .map(|i| track_copy(p + i * 4 * PAGE, 4 * PAGE, true).expect("tracked"))
            .collect();
        // The copy: each page's bytes as last taken.
        let mut copy = vec![0u8; PAGES * PAGE];
        let stop = AtomicBool::new(false);
        let (mut checked, mut saved) = (0usize, 0usize);
        std::thread::scope(|s| {
            let _stop = StopOnDrop(&stop);
            for w in 0..12u32 {
                let stop = &stop;
                s.spawn(move || {
                    let mut x = 0x2545_f491_u32.wrapping_mul(w + 1);
                    while !stop.load(Relaxed) {
                        x ^= x << 13;
                        x ^= x >> 17;
                        x ^= x << 5;
                        let at = p + (x as usize % PAGES) * PAGE + (x as usize >> 8 & 0xFFFC);
                        // SAFETY: the pool's pages; a guarded one faults
                        // into the handler, which opens it.
                        unsafe { (at as *mut u32).write_volatile(x | 1) };
                    }
                });
            }
            for _ in 0..if cfg!(debug_assertions) { 200 } else { 2000 } {
                for t in &trackers {
                    for (a, b) in t.take(t.start, t.end) {
                        // SAFETY: the pool's pages, readable.
                        let bytes = unsafe { std::slice::from_raw_parts(a as *const u8, b - a) };
                        copy[a - p..b - p].copy_from_slice(bytes);
                    }
                    t.arm(t.start, t.end);
                }
                // A write of ours too: a page saved each round, however few
                // of the writers' faults land between arming and checking.
                for t in &trackers {
                    // SAFETY: the pool's pages; a guarded one faults.
                    unsafe { ((t.start + 64) as *mut u32).write_volatile(1) };
                }
                std::thread::yield_now();
                for t in &trackers {
                    for page in t.start / PAGE..t.end / PAGE {
                        checked += 1;
                        if let Some(bytes) = t.saved(page) {
                            saved += 1;
                            let at = page * PAGE - p;
                            let page = page - p / PAGE;
                            assert!(
                                *bytes == copy[at..at + PAGE],
                                "page {page} saved other bytes than taken"
                            );
                        }
                    }
                    t.disarm();
                }
            }
        });
        assert!(saved > 0, "no page saved of {checked}: nothing was checked");
        drop(trackers);
        free(p);
    }

    /// What the title's threads wait for when they write guarded pages
    /// while a submission takes them again (issue #400: about 940 faults a
    /// frame at about 450 µs each, mostly waiting for the guard lock).
    ///
    /// `writers` threads write one dword at a time into a hot set of
    /// `HOT` pages of a write-watched pool; a submission thread, in rounds,
    /// arms every other tracker (as a root view's declaration does) and
    /// takes every tracker, which guards the written pages again. A write
    /// slower than 2 µs counts as a fault. Per writer count: faults a
    /// second, their median and 99th percentile, and a take round's median.
    /// A kernel writer thread meanwhile opens 8-page ranges with [`writing`]
    /// every 200 µs (a file read, DirectStorage): its median and 99th
    /// percentile, the opening and the marking after.
    ///
    /// `cargo test --release -p victory-gpu --lib guard_lock_contention_bench
    /// -- --ignored --nocapture --test-threads=1` (`GUARD_BENCH_MS` sets the
    /// time per writer count, 1,500 by default; `GUARD_BENCH_SPIN_US` has
    /// each writer work that long between writes, 0 by default;
    /// `GUARD_BENCH_NO_ARM` arms nothing).
    #[test]
    #[ignore = "a benchmark: run alone, in release"]
    fn guard_lock_contention_bench() {
        use std::sync::atomic::{AtomicBool, Ordering::Relaxed};
        use std::time::{Duration, Instant};
        const HOT: usize = 256;
        const PER_TRACKER: usize = 16;
        let millis: u64 = std::env::var("GUARD_BENCH_MS")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(1500);
        let spin = Duration::from_micros(
            std::env::var("GUARD_BENCH_SPIN_US")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(0),
        );
        let p = pool(HOT);
        let trackers: Vec<Tracked> = (0..HOT / PER_TRACKER)
            .map(|i| {
                track_copy(p + i * PER_TRACKER * PAGE, PER_TRACKER * PAGE, true)
                    .expect("pool memory is tracked")
            })
            .collect();
        let median = |v: &mut Vec<f64>| {
            v.sort_by(f64::total_cmp);
            v.get(v.len() / 2).copied().unwrap_or(0.0)
        };
        let p99 = |v: &[f64]| v.get(v.len() * 99 / 100).copied().unwrap_or(0.0);
        println!(
            "{HOT} hot pages, {} trackers of {PER_TRACKER} pages, every other armed; {millis} ms per row",
            trackers.len()
        );
        println!(
            "{:>7} {:>10} {:>12} {:>12} {:>14} {:>14} {:>12}",
            "writers",
            "faults/s",
            "fault p50",
            "fault p99",
            "take round p50",
            "8-page write p50",
            "p99"
        );
        for writers in [1usize, 2, 4, 8, 15] {
            for t in &trackers {
                t.take(t.start, t.end);
            }
            let stop = AtomicBool::new(false);
            let (faults, mut rounds, mut kernel) = std::thread::scope(|s| {
                let handles: Vec<_> = (0..writers)
                    .map(|w| {
                        let stop = &stop;
                        s.spawn(move || {
                            let mut slow = Vec::new();
                            let mut x = 0x9e37_79b9_u32.wrapping_mul(w as u32 + 1);
                            while !stop.load(Relaxed) {
                                x ^= x << 13;
                                x ^= x >> 17;
                                x ^= x << 5;
                                let at =
                                    p + (x as usize % HOT) * PAGE + (x as usize >> 16 & 0x3FFC);
                                let t0 = Instant::now();
                                // SAFETY: the pool's committed pages; a guarded
                                // one faults into the handler, which opens it.
                                unsafe { (at as *mut u32).write_volatile(x) };
                                let us = t0.elapsed().as_secs_f64() * 1e6;
                                let until = Instant::now() + spin;
                                while Instant::now() < until {
                                    std::hint::spin_loop();
                                }
                                if us > 2.0 {
                                    slow.push(us);
                                }
                            }
                            slow
                        })
                    })
                    .collect();
                // A kernel write (a file read, DirectStorage) over 8 pages
                // every 200 µs: what `writing` and its drop take, together.
                let kernel = {
                    let stop = &stop;
                    s.spawn(move || {
                        let mut took = Vec::new();
                        let mut x = 0x85eb_ca6b_u32;
                        while !stop.load(Relaxed) {
                            x ^= x << 13;
                            x ^= x >> 17;
                            x ^= x << 5;
                            let at = p + (x as usize % (HOT - 8)) * PAGE;
                            let t0 = Instant::now();
                            let open = writing(at, 8 * PAGE);
                            // SAFETY: the pool's pages, kept open by `open`.
                            unsafe { std::ptr::write_bytes(at as *mut u8, x as u8, 8 * PAGE) };
                            drop(open);
                            took.push(t0.elapsed().as_secs_f64() * 1e6);
                            std::thread::sleep(Duration::from_micros(200));
                        }
                        took
                    })
                };
                let mut rounds = Vec::new();
                let end = Instant::now() + Duration::from_millis(millis);
                while Instant::now() < end {
                    let t0 = Instant::now();
                    for (i, t) in trackers.iter().enumerate() {
                        if i % 2 == 0 && std::env::var_os("GUARD_BENCH_NO_ARM").is_none() {
                            t.arm(t.start, t.end);
                        }
                        t.take(t.start, t.end);
                    }
                    rounds.push(t0.elapsed().as_secs_f64() * 1e6);
                    for t in trackers.iter().step_by(2) {
                        t.disarm();
                    }
                    std::thread::sleep(Duration::from_micros(500));
                }
                stop.store(true, Relaxed);
                let faults: Vec<f64> = handles
                    .into_iter()
                    .flat_map(|h| h.join().unwrap())
                    .collect();
                (faults, rounds, kernel.join().unwrap())
            });
            let mut faults = faults;
            let rate = faults.len() as f64 / (millis as f64 / 1e3);
            let p50 = median(&mut faults);
            let kernel_p50 = median(&mut kernel);
            println!(
                "{writers:>7} {rate:>10.0} {p50:>10.1}us {:>10.1}us {:>12.1}us {kernel_p50:>12.1}us {:>10.1}us",
                p99(&faults),
                median(&mut rounds),
                p99(&kernel)
            );
        }
        drop(trackers);
        free(p);
    }
}
