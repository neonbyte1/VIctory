//! Early release of held queue waits (`VICTORY_EARLY_RELEASE=1`, off by
//! default; it acts only with `VICTORY_GPU_OVERLAP`).
//!
//! A title queue's segment waiting for a memory value that another queue's
//! list writes at its bottom of pipe (`crate::hold`) is released, without
//! the switch, once that value is in title memory: `Host::finish` writes it
//! after the writer has run on the host GPU and its results went back. But
//! every host submission goes to one in-order host queue: a segment
//! submitted after the writer runs after it on the GPU, whatever its waits
//! read. For the GPU's order, the value the writer *will* store is enough.
//!
//! So each submission in flight registers its bottom-of-pipe value writes
//! ([`fly`], at the end of `Host::run`, under the submission lock), which
//! leave once `Host::finish` has written them to title memory ([`land`]).
//! The hold pump's waits read them over title memory ([`read`]). Nothing
//! here writes title memory: what the title's CPU reads (labels, fences)
//! still lands in `Host::finish`, after the copy-back, in submission order.
//! The released segment's preparing reads title memory the writer has not
//! written back yet: `Host::settle` completes the writer first where it
//! does (`crate::shadow::crossing`).
//!
//! The overlay is refused, and the wait reads title memory as without the
//! switch (counted once per word and writes in flight, [`Refused`]):
//! - the word does not hold what it held when the writes in flight were
//!   registered or the last of them landed (the title's CPU wrote it);
//! - a submission in flight at or after the last value writer also writes
//!   the word through a copy (declared or marked ranges): its bytes there
//!   are not known before it completes;
//! - a barrier is in flight at or after it (it writes undeclared memory);
//! - the last write is a timestamp (only plain values are known ahead);
//! - writes in flight of another address or size overlap the word.

use std::collections::BTreeMap;
use std::sync::Mutex;
use std::sync::atomic::{AtomicBool, Ordering};

use crate::queue::Written;

/// Whether held waits may read the values of writes in flight
/// (`VICTORY_EARLY_RELEASE=1`); logged once.
pub(crate) fn on() -> bool {
    static ON: std::sync::LazyLock<bool> = std::sync::LazyLock::new(|| {
        let on = crate::switch::on("VICTORY_EARLY_RELEASE");
        if on {
            tracing::info!("held queue waits read the values of bottom-of-pipe writes in flight (VICTORY_EARLY_RELEASE)");
        }
        on
    });
    *ON
}

/// Whether barriers are relaxed (`VICTORY_EARLY_BARRIERS=1`); logged once.
/// A barrier is then prepared with earlier submissions still in flight (its
/// preparing settles them where it reads what they write back), and work
/// after a submission that names the title memory it writes (a texture read
/// back) no longer makes a barrier (`crate::host`'s `alone`, `barrier`).
pub(crate) fn barriers() -> bool {
    static ON: std::sync::LazyLock<bool> = std::sync::LazyLock::new(|| {
        let on = crate::switch::on("VICTORY_EARLY_BARRIERS");
        if on {
            tracing::info!(
                "barriers are prepared with earlier submissions in flight; named work after is no barrier \
                 (VICTORY_EARLY_BARRIERS)"
            );
        }
        on
    });
    *ON
}

/// Why a wait reads title memory though a value write to its word is in
/// flight.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Refused {
    /// The word changed since the writes in flight were registered.
    Cpu,
    /// A submission at or after the last value writer writes it through a
    /// copy.
    Writer,
    /// A barrier is in flight at or after the last value writer.
    Barrier,
    /// The last write is a timestamp.
    Stamp,
    /// Writes of another address or size overlap the word.
    Size,
}

/// What a wait reads at a word.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Answer {
    /// No value write in flight: title memory.
    Memory,
    /// The value the last write in flight stores.
    Value(u64),
    /// Title memory, and why; `true` the first time for these writes.
    Refused(Refused, bool),
}

/// The low `bytes` bytes of a value.
fn low(value: u64, bytes: usize) -> u64 {
    if bytes >= 8 { value } else { value & ((1u64 << (8 * bytes)) - 1) }
}

/// A word value writes in flight go to.
struct Word {
    /// The size of the first write (4 or 8); `mixed`: a later one differs.
    bytes: usize,
    mixed: bool,
    /// What title memory holds before the writes in flight: as read when
    /// the first was registered, then the value of each as it lands (none
    /// after a timestamp, or unreadable memory).
    base: Option<u64>,
    /// The writes in flight, in submission order: (submission, what).
    writes: Vec<(u64, Written)>,
    /// A refusal was counted since the writes last changed.
    refused: bool,
}

/// The value writes of the submissions in flight, and their barriers.
#[derive(Default)]
pub(crate) struct Overlay {
    words: BTreeMap<usize, Word>,
    barriers: Vec<u64>,
}

impl Overlay {
    pub(crate) const fn new() -> Self {
        Overlay {
            words: BTreeMap::new(),
            barriers: Vec::new(),
        }
    }

    /// Submission `owner` (a barrier or not) is in flight, with its
    /// bottom-of-pipe writes `(address, what, bytes)` in the order
    /// `Host::finish` makes them; `read` gives title memory (address,
    /// bytes) for words no write in flight goes to yet.
    pub(crate) fn fly(
        &mut self,
        owner: u64,
        barrier: bool,
        writes: &[(usize, Written, usize)],
        read: &dyn Fn(usize, usize) -> Option<u64>,
    ) {
        if barrier {
            self.barriers.push(owner);
        }
        for &(at, what, bytes) in writes {
            let word = self.words.entry(at).or_insert_with(|| Word {
                bytes,
                mixed: false,
                base: read(at, bytes).map(|v| low(v, bytes)),
                writes: Vec::new(),
                refused: false,
            });
            word.mixed |= word.bytes != bytes;
            word.writes.push((owner, what));
            word.refused = false;
        }
    }

    /// Submission `owner` completed: its writes are in title memory.
    pub(crate) fn land(&mut self, owner: u64) {
        self.barriers.retain(|&b| b != owner);
        self.words.retain(|_, w| {
            if let Some(i) = w.writes.iter().rposition(|&(o, _)| o == owner) {
                w.base = match w.writes[i].1 {
                    Written::Value(v) => Some(low(v, w.bytes)),
                    Written::GpuTimestamp | Written::SystemTimestamp => None,
                };
                w.writes.retain(|&(o, _)| o != owner);
                w.refused = false;
            }
            !w.writes.is_empty()
        });
    }

    /// What a wait for `bytes` at `address` reads: `memory` gives title
    /// memory (address, bytes), `writer` the latest submission in flight
    /// writing `[start, end)` through a copy.
    pub(crate) fn value(
        &mut self,
        address: usize,
        bytes: usize,
        memory: &dyn Fn(usize, usize) -> Option<u64>,
        writer: &dyn Fn(usize, usize) -> Option<u64>,
    ) -> Answer {
        let Overlay { words, barriers } = self;
        let end = address.saturating_add(bytes);
        let mut near = words
            .range_mut(address.saturating_sub(7)..end)
            .filter(|(at, w)| **at + w.bytes > address);
        let Some((&at, word)) = near.next() else {
            return Answer::Memory;
        };
        let alone = near.next().is_none();
        let judged = (|| {
            if !alone || at != address || word.bytes != bytes || word.mixed {
                return Err(Refused::Size);
            }
            let &(owner, what) = word.writes.last().ok_or(Refused::Size)?;
            let (Written::Value(value), Some(base)) = (what, word.base) else {
                return Err(Refused::Stamp);
            };
            if barriers.iter().any(|&b| b >= owner) {
                return Err(Refused::Barrier);
            }
            if writer(address, end).is_some_and(|o| o >= owner) {
                return Err(Refused::Writer);
            }
            if memory(address, bytes).map(|v| low(v, bytes)) != Some(base) {
                return Err(Refused::Cpu);
            }
            Ok(low(value, bytes))
        })();
        match judged {
            Ok(value) => Answer::Value(value),
            Err(why) => Answer::Refused(why, !std::mem::replace(&mut word.refused, true)),
        }
    }

    pub(crate) fn is_empty(&self) -> bool {
        self.words.is_empty() && self.barriers.is_empty()
    }
}

static OVERLAY: Mutex<Overlay> = Mutex::new(Overlay::new());

/// Whether [`OVERLAY`] holds anything: asked first by every held wait.
static FLYING: AtomicBool = AtomicBool::new(false);

fn overlay() -> std::sync::MutexGuard<'static, Overlay> {
    OVERLAY.lock().unwrap_or_else(|e| e.into_inner())
}

/// Title memory as a wait reads it: the low `bytes` of the word at
/// `address`, if readable.
fn title(address: usize, bytes: usize) -> Option<u64> {
    let mut value = [0u8; 8];
    crate::host::read_into(address as *const u8, &mut value[..bytes.min(8)]).then(|| u64::from_le_bytes(value))
}

/// [`Overlay::fly`] of the submissions in flight (`Host::run`, with the
/// switch on and submissions staying in flight).
pub(crate) fn fly(owner: u64, barrier: bool, writes: &[(usize, Written, usize)]) {
    if !barrier && writes.is_empty() {
        return;
    }
    let mut o = overlay();
    o.fly(owner, barrier, writes, &title);
    FLYING.store(!o.is_empty(), Ordering::Release);
    #[cfg(feature = "diagnostics")]
    crate::copystat::add(crate::copystat::C::EarlyWrites, writes.len() as u64);
}

/// [`Overlay::land`] of the submissions in flight (`Host::finish`, once
/// its bottom-of-pipe writes are in title memory).
pub(crate) fn land(owner: u64) {
    if !FLYING.load(Ordering::Acquire) {
        return;
    }
    let mut o = overlay();
    o.land(owner);
    FLYING.store(!o.is_empty(), Ordering::Release);
}

/// The value a held wait reads (`crate::queue`'s pump, with the switch
/// on): that of the last value write in flight to the word, else title
/// memory (`memory`).
pub(crate) fn read(address: u64, bytes: u8, memory: &dyn Fn(u64, u8) -> Option<u64>) -> Option<u64> {
    if FLYING.load(Ordering::Acquire) {
        let answer = overlay().value(address as usize, usize::from(bytes), &title, &|start, end| {
            crate::shadow::crossing(start, end, crate::shadow::LABEL).map(|c| c.0)
        });
        match answer {
            Answer::Value(value) => return Some(value),
            #[cfg(feature = "diagnostics")]
            Answer::Refused(why, true) => {
                use crate::copystat::{C, add};
                add(
                    match why {
                        Refused::Cpu => C::EarlyRefusedCpu,
                        Refused::Writer => C::EarlyRefusedWriter,
                        Refused::Barrier => C::EarlyRefusedBarrier,
                        Refused::Stamp => C::EarlyRefusedStamp,
                        Refused::Size => C::EarlyRefusedSize,
                    },
                    1,
                );
            }
            _ => {}
        }
    }
    memory(address, bytes)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::cell::RefCell;
    use std::collections::HashMap;

    const L: usize = 0x1000;

    /// Title memory of 8-byte words, and the latest writer through a copy.
    #[derive(Default)]
    struct Title {
        words: HashMap<usize, u64>,
        writer: Option<u64>,
    }

    impl Title {
        fn ask(&self, o: &mut Overlay, at: usize, bytes: usize) -> Answer {
            o.value(at, bytes, &|a, _| Some(*self.words.get(&a).unwrap_or(&0)), &|_, _| self.writer)
        }
        fn fly(&self, o: &mut Overlay, owner: u64, barrier: bool, writes: &[(usize, Written, usize)]) {
            o.fly(owner, barrier, writes, &|a, _| Some(*self.words.get(&a).unwrap_or(&0)));
        }
    }

    /// A wait reads the value the last submission in flight stores, while
    /// title memory keeps its own until the writer lands (`Host::finish`
    /// writes it, then lands it): the title's CPU sees the label only then.
    #[test]
    fn waits_read_the_last_value_in_flight_and_title_memory_keeps_its_own() {
        let mut title = Title::default();
        title.words.insert(L, 4);
        let mut o = Overlay::new();
        assert_eq!(title.ask(&mut o, L, 8), Answer::Memory, "nothing in flight");
        title.fly(&mut o, 5, false, &[(L, Written::Value(5), 8)]);
        title.fly(&mut o, 6, false, &[(L, Written::Value(6), 8), (L + 8, Written::Value(1), 8)]);
        assert_eq!(title.ask(&mut o, L, 8), Answer::Value(6), "the later writer's value");
        assert_eq!(title.words[&L], 4, "title memory untouched");
        assert_eq!(title.ask(&mut o, L + 0x100, 8), Answer::Memory, "another word");
        // Submission 5 completes: its label lands, then leaves the overlay.
        title.words.insert(L, 5);
        assert_eq!(title.ask(&mut o, L, 8), Answer::Refused(Refused::Cpu, true), "written, not landed yet: memory");
        o.land(5);
        assert_eq!(title.ask(&mut o, L, 8), Answer::Value(6), "the base is 5 now");
        title.words.insert(L, 6);
        title.words.insert(L + 8, 1);
        o.land(6);
        assert!(o.is_empty());
        assert_eq!(title.ask(&mut o, L, 8), Answer::Memory);
    }

    /// A 4-byte wait reads the low half of a 4-byte write; other sizes and
    /// overlapping words are refused.
    #[test]
    fn only_a_write_of_the_waits_own_word_answers() {
        let title = Title::default();
        let mut o = Overlay::new();
        title.fly(&mut o, 1, false, &[(L, Written::Value(0x1_0000_0007), 4)]);
        assert_eq!(title.ask(&mut o, L, 4), Answer::Value(7));
        assert_eq!(title.ask(&mut o, L, 8), Answer::Refused(Refused::Size, true), "wider wait");
        assert_eq!(title.ask(&mut o, L - 4, 8), Answer::Refused(Refused::Size, false), "overlapping, counted once");
        assert_eq!(title.ask(&mut o, L + 4, 4), Answer::Memory, "the word after");
        title.fly(&mut o, 2, false, &[(L, Written::Value(8), 8)]);
        assert_eq!(title.ask(&mut o, L, 4), Answer::Refused(Refused::Size, true), "mixed sizes");
    }

    /// The title's CPU writing the word after the writes in flight were
    /// registered: the wait reads title memory (as without the switch).
    #[test]
    fn a_word_the_cpu_wrote_meanwhile_reads_title_memory() {
        let mut title = Title::default();
        let mut o = Overlay::new();
        title.fly(&mut o, 1, false, &[(L, Written::Value(9), 8)]);
        title.words.insert(L, 0x77);
        assert_eq!(title.ask(&mut o, L, 8), Answer::Refused(Refused::Cpu, true));
        assert_eq!(title.ask(&mut o, L, 8), Answer::Refused(Refused::Cpu, false), "counted once");
    }

    /// Writers whose value is not known ahead are refused: a timestamp, a
    /// later (or the same) submission writing the word through a copy, a
    /// barrier at or after the writer. An earlier copy writer is not: its
    /// copy-back lands before the label.
    #[test]
    fn writers_not_known_ahead_are_refused() {
        let mut title = Title::default();
        let mut o = Overlay::new();
        title.fly(&mut o, 3, false, &[(L, Written::GpuTimestamp, 8)]);
        assert_eq!(title.ask(&mut o, L, 8), Answer::Refused(Refused::Stamp, true));
        o.land(3);
        assert_eq!(title.ask(&mut o, L, 8), Answer::Memory);
        title.words.insert(L + 8, 1);
        title.fly(&mut o, 5, false, &[(L + 8, Written::Value(2), 8)]);
        title.writer = Some(4);
        assert_eq!(title.ask(&mut o, L + 8, 8), Answer::Value(2), "an earlier copy writer");
        for writer in [5, 6] {
            title.writer = Some(writer);
            assert_eq!(title.ask(&mut o, L + 8, 8), Answer::Refused(Refused::Writer, writer == 5), "copy writer {writer}");
        }
        title.writer = None;
        title.fly(&mut o, 7, true, &[]);
        assert_eq!(title.ask(&mut o, L + 8, 8), Answer::Refused(Refused::Barrier, false), "a barrier after");
        o.land(5);
        o.land(7);
        assert!(o.is_empty());
        title.fly(&mut o, 8, true, &[(L, Written::Value(1), 8)]);
        assert_eq!(title.ask(&mut o, L, 8), Answer::Refused(Refused::Barrier, true), "the writer is one");
    }

    /// With the switch's pieces together (`crate::hold`'s pump reading
    /// [`read`], `Host::run` flying, `Host::finish` writing then landing):
    /// a segment waiting for a label another queue's piece writes runs
    /// right after that piece, before its completion, with title memory
    /// unchanged until the completion; a wait on a label of a refused writer
    /// (a barrier) stays held until its completion.
    #[test]
    fn a_waiting_segment_runs_behind_its_writer_before_the_label_lands() {
        use crate::hold::{Held, Item, ValueWait, World};
        let p = crate::watch::tests::pool(1);
        let word = |at: usize| (p + at) as *mut u64;
        // SAFETY: (here and below) words of our pool.
        unsafe { word(8).write_volatile(1) };
        let ge = |at: usize, reference| ValueWait {
            address: (p + at) as u64,
            bytes: 8,
            comparison: 7, // GREATER_EQUAL
            reference,
            mask: u64::MAX,
        };
        const WRITER: u64 = u64::MAX / 4;
        // "draw" writes label 0; "pages", a barrier (it writes through the
        // heap's views), writes label 8.
        let writes = |name: &str| match name {
            "draw" => vec![(p, Written::Value(2), 8)],
            "pages" => vec![(p + 8, Written::Value(2), 8)],
            _ => Vec::new(),
        };
        let log = RefCell::new(Vec::<String>::new());
        let in_flight = RefCell::new(Vec::<(u64, Vec<(usize, Written, usize)>)>::new());
        let pump = |held: &mut Held<&'static str>| {
            let memory = |a: u64, b: u8| title(a as usize, usize::from(b));
            let overlaid = |a: u64, b: u8| read(a, b, &memory);
            let mut run = |q: usize, segments: Vec<&'static str>| {
                let owner = WRITER + in_flight.borrow().len() as u64;
                let w: Vec<_> = segments.iter().flat_map(|s| writes(s)).collect();
                fly(owner, segments.contains(&"pages"), &w);
                in_flight.borrow_mut().push((owner, w));
                log.borrow_mut().push(format!("q{q}: {}", segments.join(" ")));
            };
            held.pump(&mut World {
                read: &overlaid,
                memory: Some(&memory),
                fence: &|_| 0,
                run: &mut run,
                signal: &mut |_, _, _| {},
            })
        };
        // `Host::finish`: the labels, then the overlay lets go.
        let complete = || {
            for (owner, w) in in_flight.borrow_mut().drain(..) {
                for (at, what, bytes) in w {
                    what.write(at as *mut u8, bytes);
                }
                land(owner);
            }
        };
        let mut held = Held::default();
        held.push(2, Item::Segment("depth", vec![ge(0, 2)]));
        held.push(1, Item::Segment("draw", vec![]));
        let pumped = pump(&mut held);
        assert_eq!(*log.borrow(), ["q1: draw", "q2: depth"], "released behind its writer");
        assert_eq!(pumped.early, 1);
        // SAFETY: as above.
        assert_eq!(unsafe { word(0).read_volatile() }, 0, "the label is not in title memory yet");
        complete();
        // SAFETY: as above.
        assert_eq!(unsafe { word(0).read_volatile() }, 2, "it lands with the completion");
        // A barrier's label: the wait reads title memory.
        held.push(2, Item::Segment("late", vec![ge(8, 2)]));
        held.push(1, Item::Segment("pages", vec![]));
        let pumped = pump(&mut held);
        assert_eq!((pumped.early, held.holds_work(2)), (0, true), "held behind a barrier's label");
        complete();
        let pumped = pump(&mut held);
        assert_eq!((pumped.early, held.holds_work(2)), (0, false), "released by the landed value");
        assert!(!FLYING.load(Ordering::Acquire));
        crate::watch::tests::free(p);
    }

    /// The settle: a segment released on a label in flight is prepared with
    /// its writer still in flight; where its preparing reads what the writer
    /// writes back (its declared range, the label's word), the registry
    /// names the writer to complete first (`Host::settle`), and memory it
    /// does not write needs nothing.
    #[test]
    fn a_released_segment_settles_its_writer_where_it_reads_its_writes() {
        const WRITER: u64 = u64::MAX / 4 + 100;
        let (range, label) = (0x10_0000, 0x20_0000);
        let mut o = Overlay::new();
        o.fly(WRITER, false, &[(label, Written::Value(3), 8)], &|_, _| Some(2));
        crate::shadow::fly(WRITER, vec![(range, range + 0x100, 1), (label, label + 8, crate::shadow::LABEL)]);
        assert_eq!(
            o.value(label, 8, &|_, _| Some(2), &|s, e| crate::shadow::crossing(s, e, crate::shadow::LABEL).map(|c| c.0)),
            Answer::Value(3),
            "released early"
        );
        assert_eq!(crate::shadow::crossing(range + 0x80, range + 0x88, 2), Some((WRITER, false)), "its range: Alias settle");
        assert_eq!(crate::shadow::crossing(label, label + 8, 2), Some((WRITER, true)), "its label: Label settle");
        assert_eq!(crate::shadow::crossing(range + 0x100, range + 0x200, 2), None, "elsewhere: none");
        crate::shadow::land(WRITER);
        o.land(WRITER);
        assert_eq!(crate::shadow::crossing(label, label + 8, 2), None);
    }
}
