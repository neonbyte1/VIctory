//! Diagnostics (`VICTORY_OWNER_CENSUS=1`): who touches a page shaders wrote
//! first, after its copy-back. Nothing here changes what is copied or
//! guarded; it only counts, so that the copy-back can be judged by who
//! needs its bytes (SyncDesign, stage 1).
//!
//! A page is *offered* where a design that leaves what shaders wrote with
//! the copy that wrote it (until something needs the bytes) could do so: a
//! whole 64 KiB title page of a copy in video memory, a section's page (the
//! runtime has its own mapping of it), that went back clean (the title had
//! not written it since the copy took it) and that the shaders changed.
//! It is watched at every address showing the same physical page (as
//! `crate::watch` knows them), so a write through another mapping is seen
//! too. Its first touch after that ends the entry, counted by what touched
//! it and whether at the copy's address or another mapping:
//! - the same copy marking it again (the copy-back was for no one), or
//!   another copy marking it (at its own address);
//! - a reader that could be given the copy's bytes instead of title memory:
//!   a refresh of another copy, a texture's staging, a title copy or fill
//!   at submission, another read of title memory through the kernel (at
//!   those, the copy's bytes are compared with title memory, which under
//!   today's copy-back must hold the same);
//! - a writer that would need the bytes in title memory first: a title
//!   write fault, our copy-back of another copy, the HLE's kernel writes
//!   (file reads, DirectStorage) and other kernel writes;
//! - the page or the copy going (a remap, the copy dropped).
//!
//! A refresh of the copy itself means something marked its page changed
//! that none of the above saw: counted apart, with samples.
//!
//! Not seen: reads of title memory in process (the title's, and our own
//! direct reads), which only an access fault would show.

use std::collections::{BTreeMap, HashMap};
use std::sync::Mutex;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering::Relaxed};

use crate::watch::PAGE;

/// Whether the census runs (`VICTORY_OWNER_CENSUS=1`, read once).
pub(crate) fn on() -> bool {
    static ON: std::sync::LazyLock<bool> =
        std::sync::LazyLock::new(|| std::env::var("VICTORY_OWNER_CENSUS").as_deref() == Ok("1"));
    *ON
}

/// What reads or writes title memory on this thread now ([`as_kind`]).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum Kind {
    /// Any other access through the kernel (`crate::host::read_into`,
    /// `crate::watch::writing`).
    Other,
    /// A refresh of the copy at the address given ([`as_kind`]'s `copy`).
    Refresh,
    /// A texture's staging.
    Texture,
    /// A title copy or fill at submission.
    TitleCopy,
    /// Our own bookkeeping (copy-back, saves): reads are not consumers,
    /// writes are our copy-back.
    Own,
    /// The HLE: a file read into title memory.
    File,
    /// The HLE: DirectStorage writing an address, or physical pages through
    /// the runtime's own views.
    StorageAddress,
    StoragePages,
}

thread_local! {
    static KIND: std::cell::Cell<(Kind, usize)> = const { std::cell::Cell::new((Kind::Other, 0)) };
}

/// Sets this thread's reader or writer until dropped ([`as_kind`]).
pub struct AsKind((Kind, usize));

impl Drop for AsKind {
    fn drop(&mut self) {
        KIND.with(|k| k.set(self.0));
    }
}

/// Accesses of title memory on this thread are `kind`'s (for a refresh, of
/// the copy at `copy`) until the result is dropped.
pub(crate) fn as_kind(kind: Kind, copy: usize) -> AsKind {
    AsKind(KIND.with(|k| k.replace((kind, copy))))
}

/// What the HLE writes into title memory ([`hle_writes`]).
pub enum HleWrite {
    /// A file read.
    File,
    /// DirectStorage to an address.
    StorageAddress,
    /// DirectStorage to physical pages, through the runtime's own views
    /// (reported where the pages show once written).
    StoragePages,
}

/// The HLE's kernel writes on this thread are `what` until dropped.
pub fn hle_writes(what: HleWrite) -> AsKind {
    as_kind(
        match what {
            HleWrite::File => Kind::File,
            HleWrite::StorageAddress => Kind::StorageAddress,
            HleWrite::StoragePages => Kind::StoragePages,
        },
        0,
    )
}

/// An offered page: the copy (its address, title range) and the copy's
/// bytes of it, and every page number showing it.
#[derive(Clone)]
struct Entry {
    copy: usize,
    range: (usize, usize),
    mirror: usize,
    shown: Vec<usize>,
}

#[derive(Default)]
struct State {
    /// Offered pages by their copy's page number.
    entries: HashMap<usize, Entry>,
    /// Every page number showing an offered page: the copy's page number.
    at: BTreeMap<usize, usize>,
    /// What ended pages other than marked again (diagnostics samples).
    samples: Vec<Sample>,
}

/// One page's end, for the log.
struct Sample {
    what: N,
    page: usize,
    range: (usize, usize),
    at: usize,
    by: usize,
    by_range: (usize, usize),
}

const SAMPLES: usize = 24;

static STATE: std::sync::LazyLock<Mutex<State>> = std::sync::LazyLock::new(Mutex::default);
/// How many entries (read without the lock: nothing to look up when 0).
static LIVE: AtomicUsize = AtomicUsize::new(0);

/// What the counts count ([`line`]).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum N {
    Offered,
    Again,
    OtherCopy,
    OtherCopyElsewhere,
    Refresh,
    RefreshElsewhere,
    RefreshOwn,
    Texture,
    TitleCopy,
    HostRead,
    Elsewhere,
    TitleWrite,
    TitleWriteElsewhere,
    OwnWrite,
    OwnWriteElsewhere,
    FileWrite,
    StorageAddress,
    StoragePages,
    KernelWrite,
    KernelElsewhere,
    Gone,
    Equal,
    Differ,
}
const NS: usize = 23;
static COUNTS: [AtomicU64; NS] = [const { AtomicU64::new(0) }; NS];

fn count(n: N) {
    COUNTS[n as usize].fetch_add(1, Relaxed);
}

fn state() -> std::sync::MutexGuard<'static, State> {
    STATE.lock().unwrap_or_else(|e| e.into_inner())
}

impl State {
    /// Ends the entry showing at page number `page`; with its own page
    /// number and whether `page` is another mapping of it.
    fn take(&mut self, page: usize) -> Option<(usize, Entry, bool)> {
        let home = *self.at.get(&page)?;
        let e = self.entries.remove(&home)?;
        for p in &e.shown {
            self.at.remove(p);
        }
        LIVE.fetch_sub(1, Relaxed);
        Some((home, e, page != home))
    }

    fn sample(&mut self, what: N, home: usize, e: &Entry, at: usize, by: usize, by_range: (usize, usize)) {
        if self.samples.len() < SAMPLES {
            self.samples.push(Sample {
                what,
                page: home,
                range: e.range,
                at,
                by,
                by_range,
            });
        }
    }

    /// The offered pages showing in `[at, at + len)`.
    fn found(&self, at: usize, len: usize) -> Vec<usize> {
        self.at.range(at / PAGE..at.saturating_add(len).div_ceil(PAGE)).map(|(&k, _)| k).collect()
    }
}

/// Page `page` (a page number) of the copy at `copy` (of title memory
/// `range`, start and length) went back clean after its shaders changed
/// it; `mirror` is the copy's 64 KiB of it, valid until [`drop_copy`] of
/// the copy; `others` the other page numbers showing it.
pub(crate) fn offer(page: usize, copy: usize, range: (usize, usize), mirror: usize, others: &[usize]) {
    let mut s = state();
    // A page shown here offered before ends (by another copy at another
    // mapping, which `marked` saw first unless its copy-back went page by
    // page).
    for &p in std::iter::once(&page).chain(others) {
        if let Some((home, e, elsewhere)) = s.take(p) {
            let n = if e.copy == copy { N::Again } else if elsewhere { N::OtherCopyElsewhere } else { N::OtherCopy };
            count(n);
            if n != N::Again {
                s.sample(n, home, &e, p, copy, range);
            }
        }
    }
    let shown: Vec<usize> = std::iter::once(page).chain(others.iter().copied()).collect();
    for &p in &shown {
        s.at.insert(p, page);
    }
    s.entries.insert(
        page,
        Entry {
            copy,
            range,
            mirror,
            shown,
        },
    );
    LIVE.fetch_add(1, Relaxed);
    count(N::Offered);
}

/// The copy at `copy` (of title memory `range`) marked `page` (a page
/// number) again: before its copy-back.
pub(crate) fn marked(page: usize, copy: usize, range: (usize, usize)) {
    if LIVE.load(Relaxed) == 0 {
        return;
    }
    let mut s = state();
    if let Some((home, e, elsewhere)) = s.take(page) {
        let n = if e.copy == copy && !elsewhere {
            N::Again
        } else if elsewhere {
            N::OtherCopyElsewhere
        } else {
            N::OtherCopy
        };
        count(n);
        if n != N::Again {
            s.sample(n, home, &e, page, copy, range);
        }
    }
}

/// Title memory `[at, at + len)` is read by this thread's reader
/// ([`as_kind`]): the offered pages there end, counted by reader, and their
/// copy's bytes are compared with title memory.
pub(crate) fn touch(at: usize, len: usize) {
    if LIVE.load(Relaxed) == 0 || len == 0 {
        return;
    }
    let (kind, reader) = KIND.with(|k| k.get());
    if kind == Kind::Own {
        return;
    }
    let mut s = state();
    for page in s.found(at, len) {
        let Some((home, e, elsewhere)) = s.take(page) else {
            continue;
        };
        let n = match kind {
            Kind::Refresh if e.copy == reader && !elsewhere => N::RefreshOwn,
            Kind::Refresh if elsewhere => N::RefreshElsewhere,
            Kind::Refresh => N::Refresh,
            Kind::Texture => N::Texture,
            Kind::TitleCopy => N::TitleCopy,
            _ if elsewhere => N::Elsewhere,
            _ => N::HostRead,
        };
        count(n);
        if matches!(n, N::RefreshOwn | N::Elsewhere | N::RefreshElsewhere) {
            s.sample(n, home, &e, page * PAGE, reader, (0, 0));
        }
        // Under the lock: the copy's bytes live until `drop_copy` of it,
        // which waits for it.
        count(if same_as_title(home, e.mirror) { N::Equal } else { N::Differ });
    }
}

/// Whether title page `page` holds the 64 KiB at `mirror` (false if it
/// cannot be read).
fn same_as_title(page: usize, mirror: usize) -> bool {
    thread_local! {
        static TITLE: std::cell::RefCell<Vec<u8>> = const { std::cell::RefCell::new(Vec::new()) };
    }
    let _own = as_kind(Kind::Own, 0);
    TITLE.with(|t| {
        let mut t = t.borrow_mut();
        t.resize(PAGE, 0);
        // SAFETY: the copy's bytes of a whole page, alive (see `touch`).
        let mirror = unsafe { std::slice::from_raw_parts(mirror as *const u8, PAGE) };
        crate::host::read_into((page * PAGE) as *const u8, &mut t) && t[..] == *mirror
    })
}

/// Who writes title memory without our copy's bytes in it first.
#[derive(Clone, Copy)]
pub(crate) enum Writer {
    /// A title write fault, at the instruction given.
    Title(usize),
    /// A write through the kernel (`crate::watch::writing`): ours or the
    /// HLE's, by this thread's [`as_kind`].
    Kernel,
}

/// `writer` is about to write title memory `[at, at + len)` (or has, for
/// writes through the runtime's own views).
pub(crate) fn written(at: usize, len: usize, writer: Writer) {
    if LIVE.load(Relaxed) == 0 || len == 0 {
        return;
    }
    let kind = KIND.with(|k| k.get().0);
    let mut s = state();
    for page in s.found(at, len) {
        let Some((home, e, elsewhere)) = s.take(page) else {
            continue;
        };
        let (n, by) = match writer {
            Writer::Title(rip) => (if elsewhere { N::TitleWriteElsewhere } else { N::TitleWrite }, rip),
            Writer::Kernel => (
                match kind {
                    Kind::Own if elsewhere => N::OwnWriteElsewhere,
                    Kind::Own => N::OwnWrite,
                    Kind::File => N::FileWrite,
                    Kind::StorageAddress => N::StorageAddress,
                    Kind::StoragePages => N::StoragePages,
                    _ if elsewhere => N::KernelElsewhere,
                    _ => N::KernelWrite,
                },
                0,
            ),
        };
        count(n);
        s.sample(n, home, &e, page * PAGE, by, (0, 0));
    }
}

/// What title memory `[at, at + len)` shows is new (remapped).
pub(crate) fn gone(at: usize, len: usize) {
    if LIVE.load(Relaxed) == 0 || len == 0 {
        return;
    }
    let mut s = state();
    for page in s.found(at, len) {
        if s.take(page).is_some() {
            count(N::Gone);
        }
    }
}

/// The runtime mapped page `at` where `others` show it already: if one of
/// those shows an offered page, it shows at `at` too.
pub(crate) fn aliased(at: usize, others: &[usize]) {
    if LIVE.load(Relaxed) == 0 {
        return;
    }
    let mut s = state();
    let Some(home) = others.iter().find_map(|o| s.at.get(&(o / PAGE)).copied()) else {
        return;
    };
    let page = at / PAGE;
    if s.at.insert(page, home).is_none()
        && let Some(e) = s.entries.get_mut(&home)
    {
        e.shown.push(page);
    }
}

/// The copy at `copy` goes: its pages end (before its bytes are freed).
pub(crate) fn drop_copy(copy: usize) {
    if LIVE.load(Relaxed) == 0 {
        return;
    }
    let mut s = state();
    let found: Vec<usize> = s.entries.iter().filter(|(_, e)| e.copy == copy).map(|(&k, _)| k).collect();
    for page in found {
        s.take(page);
        count(N::Gone);
    }
}

/// Where `at` is, for the log: the module and offset of an instruction, or
/// the region of title memory.
fn place(at: usize, code: bool) -> String {
    use windows_sys::Win32::System::LibraryLoader::{
        GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS, GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT, GetModuleFileNameW,
        GetModuleHandleExW,
    };
    use windows_sys::Win32::System::Memory::{MEMORY_BASIC_INFORMATION, VirtualQuery};
    if code {
        let mut module = std::ptr::null_mut();
        // SAFETY: a lookup of the module holding an address; no reference
        // taken.
        let found = unsafe {
            GetModuleHandleExW(
                GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS | GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
                at as *const u16,
                &mut module,
            )
        } != 0;
        if !found {
            return format!("{at:#x}");
        }
        let mut name = [0u16; 260];
        // SAFETY: a module handle just found and a sized buffer.
        let n = unsafe { GetModuleFileNameW(module, name.as_mut_ptr(), name.len() as u32) } as usize;
        let name = String::from_utf16_lossy(&name[..n]);
        let name = name.rsplit(['\\', '/']).next().unwrap_or(&name);
        return format!("{name}+{:#x}", at - module as usize);
    }
    // SAFETY: plain data filled in by the query.
    let mut info: MEMORY_BASIC_INFORMATION = unsafe { std::mem::zeroed() };
    // SAFETY: a valid out pointer of the right size.
    unsafe { VirtualQuery(at as *const _, &mut info, size_of_val(&info)) };
    format!(
        "{at:#x} (type {:#x}, allocation {:#x}, region {:#x}+{:#x})",
        info.Type, info.AllocationBase as usize, info.BaseAddress as usize, info.RegionSize
    )
}

/// The census lines per frame over `frames` frames (counts and samples
/// reset), if on.
pub(crate) fn lines(frames: f64) -> Vec<String> {
    if !on() {
        return Vec::new();
    }
    let c: Vec<f64> = COUNTS.iter().map(|a| a.swap(0, Relaxed) as f64 / frames).collect();
    let n = |k: N| c[k as usize];
    let mut out = vec![format!(
        "  owner census per frame: {:.1} offered (whole view pages of copies in video memory gone back clean), {} live; \
         first touch: {:.1} marked again by the same copy, {:.1} by another copy here, {:.1} by another copy at another \
         mapping; could be served the copy's bytes: {:.1} refresh of another copy here, {:.1} at another mapping, {:.1} \
         texture staging, {:.1} title copies and fills, {:.1} other reads here, {:.1} at another mapping; need title memory \
         first: title write faults {:.1} here, {:.1} at another mapping; our copy-back writes {:.1} here, {:.1} at \
         another mapping; HLE writes: {:.1} file reads, {:.1} DirectStorage to addresses, {:.1} DirectStorage to pages; \
         other kernel writes {:.1} here, {:.1} at another mapping; {:.1} gone (remapped, copy dropped); refresh of the \
         same copy (unseen writer) {:.1}; served bytes equal title memory {:.1}, differ {:.1}",
        n(N::Offered),
        LIVE.load(Relaxed),
        n(N::Again),
        n(N::OtherCopy),
        n(N::OtherCopyElsewhere),
        n(N::Refresh),
        n(N::RefreshElsewhere),
        n(N::Texture),
        n(N::TitleCopy),
        n(N::HostRead),
        n(N::Elsewhere),
        n(N::TitleWrite),
        n(N::TitleWriteElsewhere),
        n(N::OwnWrite),
        n(N::OwnWriteElsewhere),
        n(N::FileWrite),
        n(N::StorageAddress),
        n(N::StoragePages),
        n(N::KernelWrite),
        n(N::KernelElsewhere),
        n(N::Gone),
        n(N::RefreshOwn),
        n(N::Equal),
        n(N::Differ),
    )];
    let samples = std::mem::take(&mut state().samples);
    for s in samples {
        let by = match s.what {
            N::TitleWrite | N::TitleWriteElsewhere => format!("instruction {}", place(s.by, true)),
            N::OtherCopy | N::OtherCopyElsewhere => format!("copy {:#x} of {:#x}+{:#x}", s.by, s.by_range.0, s.by_range.1),
            N::RefreshOwn | N::Refresh | N::RefreshElsewhere => format!("refresh of copy {:#x}", s.by),
            _ => String::new(),
        };
        out.push(format!(
            "  owner census sample: {:?} of page {:#x} (copy of {:#x}+{:#x}) at {} {by}",
            s.what,
            s.page * PAGE,
            s.range.0,
            s.range.1,
            place(s.at, false),
        ));
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn counts() -> [u64; NS] {
        std::array::from_fn(|i| COUNTS[i].load(Relaxed))
    }

    #[test]
    fn the_first_touch_of_an_offered_page_is_counted_once_by_kind_and_mapping() {
        let p = crate::watch::tests::pool(4);
        let (page, next, far) = (p / PAGE, p / PAGE + 1, p / PAGE + 3);
        // SAFETY: the pool's pages, ours.
        unsafe { std::ptr::write_bytes(p as *mut u8, 7, 4 * PAGE) };
        let mirror = vec![7u8; PAGE];
        let other = vec![9u8; PAGE];
        let (copy, reader, range) = (0x1000usize, 0x2000usize, (p, 2 * PAGE));
        let before = counts();
        let delta = |n: N| counts()[n as usize] - before[n as usize];

        // A refresh of another copy reads it: served, bytes equal; a
        // second read is not counted.
        offer(page, copy, range, mirror.as_ptr() as usize, &[]);
        {
            let _k = as_kind(Kind::Refresh, reader);
            touch(p, 2 * PAGE);
            touch(p, PAGE);
        }
        assert_eq!((delta(N::Refresh), delta(N::Equal), delta(N::Differ)), (1, 1, 0));

        // Bytes that differ from title memory are told apart.
        offer(next, copy, range, other.as_ptr() as usize, &[]);
        crate::host::read_into((next * PAGE) as *const u8, &mut [0u8; 4]);
        assert_eq!((delta(N::HostRead), delta(N::Differ)), (1, 1));

        // Our own reads are not consumers.
        offer(page, copy, range, mirror.as_ptr() as usize, &[far]);
        {
            let _k = as_kind(Kind::Own, 0);
            touch(p, PAGE);
        }
        assert_eq!(delta(N::HostRead), 1);

        // Marked again by its copy; then by another copy at another
        // mapping of the page.
        marked(page, copy, range);
        offer(page, copy, range, mirror.as_ptr() as usize, &[far]);
        marked(far, reader, (far * PAGE, PAGE));
        assert_eq!((delta(N::Again), delta(N::OtherCopyElsewhere)), (1, 1));

        // A title write at another mapping; one here.
        offer(page, copy, range, mirror.as_ptr() as usize, &[far]);
        written(far * PAGE + 5, 1, Writer::Title(0));
        offer(page, copy, range, mirror.as_ptr() as usize, &[far]);
        written(p + 5, 1, Writer::Title(0));
        assert_eq!((delta(N::TitleWriteElsewhere), delta(N::TitleWrite)), (1, 1));

        // Our copy-back writes through the kernel at another mapping; the
        // HLE's DirectStorage writes pages; a mapping made later is watched.
        offer(page, copy, range, mirror.as_ptr() as usize, &[far]);
        {
            let _k = as_kind(Kind::Own, 0);
            written(far * PAGE, PAGE, Writer::Kernel);
        }
        offer(page, copy, range, mirror.as_ptr() as usize, &[]);
        aliased(far * PAGE, &[p]);
        {
            let _k = as_kind(Kind::StoragePages, 0);
            written(far * PAGE, PAGE, Writer::Kernel);
        }
        assert_eq!((delta(N::OwnWriteElsewhere), delta(N::StoragePages)), (1, 1));

        // A remap, the copy going.
        offer(page, copy, range, mirror.as_ptr() as usize, &[far]);
        gone(far * PAGE, PAGE);
        offer(next, copy, range, mirror.as_ptr() as usize, &[]);
        drop_copy(copy);
        assert_eq!((delta(N::Gone), delta(N::Offered)), (2, 10));
        assert!(state().at.range(page..far + 1).next().is_none());
        crate::watch::tests::free(p);
    }
}
