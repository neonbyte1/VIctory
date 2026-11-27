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
//! Its first touch after that ends the entry, counted by what touched it:
//! - the same copy marking it again (the copy-back was for no one), or
//!   another copy marking it;
//! - a reader that could be given the copy's bytes instead of title memory:
//!   a refresh of another copy, a texture's staging, a title copy or fill
//!   at submission, another read of title memory through the kernel (at
//!   those, the copy's bytes are compared with title memory, which under
//!   today's copy-back must hold the same);
//! - a writer that would need the bytes in title memory first: a title
//!   write fault, our kernel writes (`crate::watch::writing`);
//! - the page or the copy going (a remap, the copy dropped).
//!
//! Not seen: reads of title memory in process (the title's, and our own
//! direct reads), which only an access fault would show.

use std::collections::BTreeMap;
use std::sync::Mutex;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering::Relaxed};

use crate::watch::PAGE;

/// Whether the census runs (`VICTORY_OWNER_CENSUS=1`, read once).
pub(crate) fn on() -> bool {
    static ON: std::sync::LazyLock<bool> =
        std::sync::LazyLock::new(|| std::env::var("VICTORY_OWNER_CENSUS").as_deref() == Ok("1"));
    *ON
}

/// What reads title memory on this thread now ([`as_kind`]).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum Kind {
    /// Any other read through the kernel (`crate::host::read_into`).
    HostRead,
    /// A refresh of the copy at the address given ([`as_kind`]'s `copy`).
    Refresh,
    /// A texture's staging.
    Texture,
    /// A title copy or fill at submission.
    TitleCopy,
    /// Our own bookkeeping (copy-back, saves): not a consumer.
    Own,
}

thread_local! {
    static KIND: std::cell::Cell<(Kind, usize)> = const { std::cell::Cell::new((Kind::HostRead, 0)) };
}

/// Sets this thread's reader until dropped ([`as_kind`]).
pub(crate) struct AsKind((Kind, usize));

impl Drop for AsKind {
    fn drop(&mut self) {
        KIND.with(|k| k.set(self.0));
    }
}

/// Reads of title memory on this thread are `kind`'s (for a refresh, of
/// the copy at `copy`) until the result is dropped.
pub(crate) fn as_kind(kind: Kind, copy: usize) -> AsKind {
    AsKind(KIND.with(|k| k.replace((kind, copy))))
}

/// An offered page: the copy (its address) and the copy's bytes of it.
#[derive(Clone, Copy)]
struct Entry {
    copy: usize,
    mirror: usize,
}

/// Offered pages by page number.
static PAGES: Mutex<BTreeMap<usize, Entry>> = Mutex::new(BTreeMap::new());
/// How many (read without the lock: nothing to look up when 0).
static LIVE: AtomicUsize = AtomicUsize::new(0);

/// What the counts count ([`line`]).
#[derive(Clone, Copy)]
enum N {
    Offered,
    Again,
    OtherCopy,
    Refresh,
    RefreshOwn,
    Texture,
    TitleCopy,
    HostRead,
    TitleWrite,
    KernelWrite,
    Gone,
    Equal,
    Differ,
}
const NS: usize = 13;
static COUNTS: [AtomicU64; NS] = [const { AtomicU64::new(0) }; NS];

fn count(n: N) {
    COUNTS[n as usize].fetch_add(1, Relaxed);
}

fn pages() -> std::sync::MutexGuard<'static, BTreeMap<usize, Entry>> {
    PAGES.lock().unwrap_or_else(|e| e.into_inner())
}

/// Page `page` (a page number) of the copy at `copy` went back clean after
/// its shaders changed it; `mirror` is the copy's 64 KiB of it, valid until
/// [`drop_copy`] of the copy.
pub(crate) fn offer(page: usize, copy: usize, mirror: usize) {
    let mut p = pages();
    if p.insert(page, Entry { copy, mirror }).is_none() {
        LIVE.fetch_add(1, Relaxed);
    }
    count(N::Offered);
}

/// The copy at `copy` marked `page` (a page number) again: before its
/// copy-back.
pub(crate) fn marked(page: usize, copy: usize) {
    if LIVE.load(Relaxed) == 0 {
        return;
    }
    if let Some(e) = take(&mut pages(), page) {
        count(if e.copy == copy { N::Again } else { N::OtherCopy });
    }
}

fn take(p: &mut BTreeMap<usize, Entry>, page: usize) -> Option<Entry> {
    let e = p.remove(&page)?;
    LIVE.fetch_sub(1, Relaxed);
    Some(e)
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
    let mut p = pages();
    let found: Vec<(usize, Entry)> = p.range(at / PAGE..at.saturating_add(len).div_ceil(PAGE)).map(|(&k, &e)| (k, e)).collect();
    for (page, e) in found {
        take(&mut p, page);
        count(match kind {
            Kind::Refresh if e.copy == reader => N::RefreshOwn,
            Kind::Refresh => N::Refresh,
            Kind::Texture => N::Texture,
            Kind::TitleCopy => N::TitleCopy,
            Kind::HostRead | Kind::Own => N::HostRead,
        });
        // Under the lock: the copy's bytes live until `drop_copy` of it,
        // which waits for it.
        count(if same_as_title(page, e.mirror) { N::Equal } else { N::Differ });
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
    /// A title write fault.
    Title,
    /// Our or the kernel's write (`crate::watch::writing`).
    Kernel,
}

/// `writer` is about to write title memory `[at, at + len)`.
pub(crate) fn written(at: usize, len: usize, writer: Writer) {
    ended(at, len, match writer {
        Writer::Title => N::TitleWrite,
        Writer::Kernel => N::KernelWrite,
    });
}

/// What title memory `[at, at + len)` shows is new (remapped).
pub(crate) fn gone(at: usize, len: usize) {
    ended(at, len, N::Gone);
}

fn ended(at: usize, len: usize, n: N) {
    if LIVE.load(Relaxed) == 0 || len == 0 {
        return;
    }
    let mut p = pages();
    let found: Vec<usize> = p.range(at / PAGE..at.saturating_add(len).div_ceil(PAGE)).map(|(&k, _)| k).collect();
    for page in found {
        take(&mut p, page);
        count(n);
    }
}

/// The copy at `copy` goes: its pages end (before its bytes are freed).
pub(crate) fn drop_copy(copy: usize) {
    if LIVE.load(Relaxed) == 0 {
        return;
    }
    let mut p = pages();
    let found: Vec<usize> = p.iter().filter(|(_, e)| e.copy == copy).map(|(&k, _)| k).collect();
    for page in found {
        take(&mut p, page);
        count(N::Gone);
    }
}

/// The census line per frame over `frames` frames (counts reset), if on.
pub(crate) fn line(frames: f64) -> Option<String> {
    if !on() {
        return None;
    }
    let c: Vec<f64> = COUNTS.iter().map(|a| a.swap(0, Relaxed) as f64 / frames).collect();
    let n = |k: N| c[k as usize];
    Some(format!(
        "  owner census per frame: {:.1} offered (whole view pages of copies in video memory gone back clean), {} live; \
         first touch: {:.1} marked again by the same copy, {:.1} by another copy; could be served the copy's bytes: \
         {:.1} refresh of another copy, {:.1} refresh of the same copy, {:.1} texture staging, {:.1} title copies and \
         fills, {:.1} other reads; need title memory first: {:.1} title write faults, {:.1} our kernel writes; {:.1} gone \
         (remapped, copy dropped); served bytes equal title memory {:.1}, differ {:.1}",
        n(N::Offered),
        LIVE.load(Relaxed),
        n(N::Again),
        n(N::OtherCopy),
        n(N::Refresh),
        n(N::RefreshOwn),
        n(N::Texture),
        n(N::TitleCopy),
        n(N::HostRead),
        n(N::TitleWrite),
        n(N::KernelWrite),
        n(N::Gone),
        n(N::Equal),
        n(N::Differ),
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn counts() -> [u64; NS] {
        std::array::from_fn(|i| COUNTS[i].load(Relaxed))
    }

    #[test]
    fn the_first_touch_of_an_offered_page_is_counted_once_by_kind() {
        let p = crate::watch::tests::pool(2);
        let (page, next) = (p / PAGE, p / PAGE + 1);
        // SAFETY: the pool's two pages, ours.
        unsafe { std::ptr::write_bytes(p as *mut u8, 7, 2 * PAGE) };
        let mirror = vec![7u8; PAGE];
        let other = vec![9u8; PAGE];
        let (copy, reader) = (0x1000usize, 0x2000usize);
        let before = counts();
        let delta = |n: N| counts()[n as usize] - before[n as usize];

        // A refresh of another copy reads it: served, bytes equal; a
        // second read is not counted.
        offer(page, copy, mirror.as_ptr() as usize);
        {
            let _k = as_kind(Kind::Refresh, reader);
            touch(p, 2 * PAGE);
            touch(p, PAGE);
        }
        assert_eq!((delta(N::Refresh), delta(N::Equal), delta(N::Differ)), (1, 1, 0));

        // Bytes that differ from title memory are told apart.
        offer(next, copy, other.as_ptr() as usize);
        crate::host::read_into((next * PAGE) as *const u8, &mut [0u8; 4]);
        assert_eq!((delta(N::HostRead), delta(N::Differ)), (1, 1));

        // Our own reads are not consumers.
        offer(page, copy, mirror.as_ptr() as usize);
        {
            let _k = as_kind(Kind::Own, 0);
            touch(p, PAGE);
        }
        assert_eq!(delta(N::HostRead), 1);

        // Marked again by its copy, then by another one.
        marked(page, copy);
        offer(page, copy, mirror.as_ptr() as usize);
        marked(page, reader);
        assert_eq!((delta(N::Again), delta(N::OtherCopy)), (1, 1));

        // A title write, a remap, the copy going.
        offer(page, copy, mirror.as_ptr() as usize);
        written(p + 5, 1, Writer::Title);
        offer(page, copy, mirror.as_ptr() as usize);
        gone(p, PAGE);
        offer(next, copy, mirror.as_ptr() as usize);
        drop_copy(copy);
        assert_eq!((delta(N::TitleWrite), delta(N::Gone), delta(N::Offered)), (1, 2, 7));
        assert!(pages().range(page..next + 1).next().is_none());
        crate::watch::tests::free(p);
    }
}
