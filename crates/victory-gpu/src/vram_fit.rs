//! `VICTORY_VRAM_FIT=1`: video memory sized from the adapter's budget
//! (DXGI `QueryVideoMemoryInfo`) instead of fixed numbers (fork issue #15).
//!
//! - Host copies of title memory may take video memory past the fixed
//!   quarter of the budget (`crate::host::Host::vram_shadow_budget`) while
//!   the card's use stays under [`TARGET`] of its budget: the room is read
//!   once a presented frame ([`frame`]) and each copy made on it takes its
//!   bytes off ([`take`]). Copies are never given back, so the target stays
//!   under the texture eviction's mark (`crate::resource`, 90 %): textures
//!   of a new area are not evicted to make room for copies.
//! - Heaps of textures (`crate::placement`) are 16 MiB instead of 64, and
//!   the fullest heap that fits is taken first: a long-lived texture keeps
//!   a small heap, not a large one, and sparse heaps empty out and are
//!   released (the 5090's 64 MiB heaps hold ~2.9 GB free, fork issue #15).
//!
//! With the switch off the room is still kept and counted (what the
//! switch would have put in video memory), and [`log`] prints the line.

use std::sync::LazyLock;
use std::sync::atomic::{AtomicI64, AtomicU64, Ordering::Relaxed};

/// Whether the switch is on (`VICTORY_VRAM_FIT=1`, read once).
pub(crate) fn on() -> bool {
    static ON: LazyLock<bool> = LazyLock::new(|| {
        let on = crate::switch::on("VICTORY_VRAM_FIT");
        tracing::info!(
            on,
            "video memory fit: copies and texture heaps sized from the card's budget (VICTORY_VRAM_FIT)"
        );
        on
    });
    *ON
}

/// Share of the card's budget host copies may fill it up to.
pub(crate) const TARGET: f64 = 0.85;

/// Bytes of one heap of textures: 64 MiB, 16 MiB with the switch.
pub(crate) fn heap_bytes() -> u64 {
    if on() { 16 << 20 } else { 64 << 20 }
}

/// Bytes copies may still take in video memory past the fixed quarter:
/// [`TARGET`] of the budget less the card's use, at the last presented
/// frame, less what copies took since.
static ROOM: AtomicI64 = AtomicI64::new(0);

/// Counts: copies the fixed quarter refused that took the room (or, with
/// the switch off, would have) and their bytes; those the room refused
/// too, and their bytes.
#[derive(Clone, Copy)]
enum N {
    Roomed,
    RoomedBytes,
    Refused,
    RefusedBytes,
}

static COUNTED: [AtomicU64; 4] = [const { AtomicU64::new(0) }; 4];
static TOTALS: [AtomicU64; 4] = [const { AtomicU64::new(0) }; 4];

fn count(what: N, n: u64) {
    COUNTED[what as usize].fetch_add(n, Relaxed);
    TOTALS[what as usize].fetch_add(n, Relaxed);
}

/// Once a presented frame: the card's local use and budget (MB, as
/// `crate::host::video_memory` gives them) set the copies' room.
pub(crate) fn frame(used_mb: f64, budget_mb: f64) {
    ROOM.store(((budget_mb * TARGET - used_mb) * 1e6) as i64, Relaxed);
}

/// A copy of `size` bytes the fixed quarter refused: whether it goes to
/// video memory on the room (taking its bytes off it). With the switch off
/// the room is taken all the same and counted, and the answer is no.
pub(crate) fn take(size: u64) -> bool {
    let size = size as i64;
    let roomed = ROOM
        .fetch_update(Relaxed, Relaxed, |r| (r >= size).then_some(r - size))
        .is_ok();
    if roomed {
        count(N::Roomed, 1);
        count(N::RoomedBytes, size as u64);
    } else {
        count(N::Refused, 1);
        count(N::RefusedBytes, size as u64);
    }
    roomed && on()
}

/// A copy [`take`] gave room to could not be made there: the room back.
pub(crate) fn give_back(size: u64) {
    ROOM.fetch_add(size as i64, Relaxed);
}

/// The line, with the frame-rate statistics: the card's budget and use,
/// what the copies, the heaps of textures and the committed textures take
/// of it, and the rest; what copies took the room for (or would have, with
/// the switch off).
#[cfg(feature = "diagnostics")]
pub(crate) fn log(frames: f64, used_mb: f64, budget_mb: f64) {
    let c: [u64; 4] = std::array::from_fn(|i| COUNTED[i].swap(0, Relaxed));
    let t = |n: N| TOTALS[n as usize].load(Relaxed);
    let f = |n: N| c[n as usize] as f64 / frames.max(1.0);
    let mb = |b: f64| b / 1e6;
    let copies = crate::shadow::vram_bytes() as f64;
    let (_, system) = crate::shadow::system_copies();
    let (heaps, heap_used, heap_free) = crate::placement::stats();
    let heap_total = (heap_used + heap_free) as f64;
    let committed = crate::resource::texture_counts()
        .1
        .saturating_sub(heap_used) as f64;
    let rest = used_mb * 1e6 - copies - heap_total - committed;
    let (asked_vram, asked_system) = crate::shadow::take_asked();
    tracing::info!(
        "video memory fit (VICTORY_VRAM_FIT={}): {used_mb:.0} MB used of a {budget_mb:.0} MB budget, \
         copies' target {:.0} MB, room {:.0} MB; copies {:.0} MB in video memory, {:.0} MB in system memory \
         (asked for per frame by lists: {:.0} MB of those in video memory, {:.0} MB of those in system memory); \
         past the fixed quarter per frame: {:.1} copies took the room ({:.1} MB), {:.1} did not fit it ({:.1} MB); \
         since the start {} ({:.0} MB) and {} ({:.0} MB); heaps of textures {heaps} of {} MiB, {:.0} MB ({:.0} MB free); \
         committed textures {:.0} MB; the rest {:.0} MB",
        u8::from(on()),
        budget_mb * TARGET,
        mb(ROOM.load(Relaxed) as f64),
        mb(copies),
        mb(system as f64),
        mb(asked_vram as f64) / frames.max(1.0),
        mb(asked_system as f64) / frames.max(1.0),
        f(N::Roomed),
        mb(f(N::RoomedBytes)),
        f(N::Refused),
        mb(f(N::RefusedBytes)),
        t(N::Roomed),
        mb(t(N::RoomedBytes) as f64),
        t(N::Refused),
        mb(t(N::RefusedBytes) as f64),
        heap_bytes() >> 20,
        mb(heap_total),
        mb(heap_free as f64),
        mb(committed),
        mb(rest),
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The room is the target less the use, each copy taking its bytes;
    /// one that does not fit takes nothing.
    #[test]
    fn copies_take_the_room_until_it_is_gone() {
        frame(800.0, 1000.0); // 850 MB target: 50 MB of room
        assert_eq!(ROOM.load(Relaxed), 50_000_000);
        let _ = take(30_000_000);
        assert_eq!(ROOM.load(Relaxed), 20_000_000);
        let _ = take(30_000_000);
        assert_eq!(ROOM.load(Relaxed), 20_000_000, "did not fit: nothing taken");
        give_back(30_000_000);
        assert_eq!(ROOM.load(Relaxed), 50_000_000);
        frame(900.0, 1000.0);
        assert!(ROOM.load(Relaxed) < 0, "over the target: no room");
        assert!(!take(1));
    }
}
