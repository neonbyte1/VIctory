//! Work a queue holds back (decision 0007, "Queues wait on memory values").
//!
//! The console's lists wait on memory values (`Wait32BitValueX`,
//! `Wait64BitValueX`) that lists on other queues write at their end
//! (`Write32/64BitValueBottomOfPipeX`): Grand Theft Auto VI's four compute
//! queues start each list with a wait for the frame's value of a word that
//! a graphics list writes, and some lists wait again after their first
//! commands. Our submissions all go to one host queue, which runs them in
//! order, and the memory values a list writes land once it has completed,
//! so a queue here is a FIFO of what the title submitted to it: segments of
//! lists (a list is cut where it waits after commands, `crate::list`), its
//! Signals and its fence Waits, in their order. The head of a queue runs
//! once its waits hold; consecutive heads that may run go to the host GPU
//! together (a piece); after every piece the heads of all queues are tested
//! again until none can move. Order within a queue never changes, only the
//! interleaving across queues. A queue's PresentX behind held work is
//! held too, as an action run in its place (`crate::queue`).

use std::collections::VecDeque;
use std::time::{Duration, Instant};

/// A wait for a memory value, as a list records it: the value at `address`
/// (`bytes` 4 or 8, the 32-bit form masked) compared with `reference` by a
/// D3D12 comparison function.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct ValueWait {
    pub(crate) address: u64,
    pub(crate) bytes: u8,
    pub(crate) comparison: u32,
    pub(crate) reference: u64,
    pub(crate) mask: u64,
}

/// D3D12 comparison functions (`D3D12_COMPARISON_FUNC`).
const NEVER: u32 = 1;
const LESS: u32 = 2;
const EQUAL: u32 = 3;
const LESS_EQUAL: u32 = 4;
const GREATER: u32 = 5;
const NOT_EQUAL: u32 = 6;
const GREATER_EQUAL: u32 = 7;
const ALWAYS: u32 = 8;

impl ValueWait {
    /// Whether it holds with `read` giving the memory's value: `None` for a
    /// comparison we do not know (counted by the caller, taken as holding)
    /// and for memory that cannot be read (taken as not holding).
    pub(crate) fn holds(&self, read: &dyn Fn(u64, u8) -> Option<u64>) -> Option<bool> {
        if !(NEVER..=ALWAYS).contains(&self.comparison) {
            return None;
        }
        let value = read(self.address, self.bytes)? & self.mask;
        let reference = self.reference;
        Some(match self.comparison {
            NEVER => false,
            LESS => value < reference,
            EQUAL => value == reference,
            LESS_EQUAL => value <= reference,
            GREATER => value > reference,
            NOT_EQUAL => value != reference,
            GREATER_EQUAL => value >= reference,
            _ => true,
        })
    }
}

/// What a queue holds, in its order.
pub(crate) enum Item<S> {
    /// A list segment and the waits at its start.
    Segment(S, Vec<ValueWait>),
    /// A fence Signal: the fence and its value.
    Signal(usize, u64),
    /// A fence Wait: until the fence reaches the value.
    FenceWait(usize, u64),
    /// Done once the work before it has been given to the host GPU (a
    /// present of the frame it drew).
    Then(Box<dyn FnOnce() + Send>),
}

/// What a pump did, for the frame's accounting.
#[derive(Default, Debug, PartialEq, Eq)]
pub(crate) struct Pumped {
    /// Pieces given to the host GPU, and the segments in them.
    pub(crate) pieces: u32,
    pub(crate) segments: u32,
    /// Of those, segments that had waited (not run when submitted): their
    /// queue and how long.
    pub(crate) waited: Vec<(usize, Duration)>,
    /// Waits with a comparison we do not know (taken as holding).
    pub(crate) unknown: u32,
    /// Segments released on values of writes in flight (`crate::early`)
    /// that title memory alone would have held.
    pub(crate) early: u32,
}

/// Every queue's held work.
pub(crate) struct Held<S> {
    queues: Vec<(usize, Fifo<S>)>,
}

/// A queue's items, each with when it was pushed and whether it is fresh
/// (not yet found waiting).
type Fifo<S> = VecDeque<(Item<S>, Instant, bool)>;

impl<S> Default for Held<S> {
    fn default() -> Self {
        Self::new()
    }
}

impl<S> Held<S> {
    pub(crate) const fn new() -> Self {
        Held { queues: Vec::new() }
    }
}

/// What the pump needs from the outside world.
pub(crate) struct World<'a, S> {
    /// A memory value (address, bytes).
    pub(crate) read: &'a dyn Fn(u64, u8) -> Option<u64>,
    /// Title memory alone (address, bytes), when `read` gives the values
    /// of writes in flight over it (`crate::early`): then the segments of a
    /// queue gathered so far run before a later one's waits are tested, so
    /// that they read what those write (upstream PR #3381), and segments
    /// released only by such values are counted.
    pub(crate) memory: Option<&'a dyn Fn(u64, u8) -> Option<u64>>,
    /// A fence's value.
    pub(crate) fence: &'a dyn Fn(usize) -> u64,
    /// Runs a piece (a queue, its segments in order); it completes later,
    /// its bottom-of-pipe writes then (the queue's Signals given to
    /// `signal` meanwhile wait for it).
    pub(crate) run: &'a mut dyn FnMut(usize, Vec<S>),
    /// Signals a fence behind a queue's work (queue, fence, value).
    pub(crate) signal: &'a mut dyn FnMut(usize, usize, u64),
}

impl<S> Held<S> {
    /// Appends `item` to `queue`'s FIFO, marked fresh: whether it waited is
    /// decided by its first test.
    pub(crate) fn push(&mut self, queue: usize, item: Item<S>) {
        let now = Instant::now();
        match self.queues.iter_mut().find(|(q, _)| *q == queue) {
            Some((_, items)) => items.push_back((item, now, true)),
            None => self.queues.push((queue, VecDeque::from([(item, now, true)]))),
        }
    }

    /// Runs every queue's heads that may run, until none can.
    pub(crate) fn pump(&mut self, w: &mut World<S>) -> Pumped {
        let mut done = Pumped::default();
        loop {
            let mut moved = false;
            for (queue, items) in &mut self.queues {
                let mut piece = Vec::new();
                loop {
                    let ready = match items.front_mut() {
                        None => false,
                        Some((Item::Segment(_, waits), since, fresh)) => {
                            if w.memory.is_some() && !waits.is_empty() && !piece.is_empty() {
                                (w.run)(*queue, std::mem::take(&mut piece));
                                done.pieces += 1;
                            }
                            let mut ok = true;
                            for wait in waits.iter() {
                                match wait.holds(w.read) {
                                    Some(true) => {}
                                    Some(false) => ok = false,
                                    None => done.unknown += 1,
                                }
                            }
                            if !ok {
                                *fresh = false;
                            } else if !*fresh {
                                done.waited.push((*queue, since.elapsed()));
                            }
                            if ok
                                && let Some(memory) = w.memory
                                && waits.iter().any(|wait| wait.holds(memory) == Some(false))
                            {
                                done.early += 1;
                            }
                            ok
                        }
                        // A Signal, action or Wait keeps its place: the
                        // piece before it runs first.
                        Some((Item::Signal(..) | Item::Then(_), ..)) => {
                            if !piece.is_empty() {
                                (w.run)(*queue, std::mem::take(&mut piece));
                                done.pieces += 1;
                            }
                            true
                        }
                        Some((Item::FenceWait(fence, value), _, fresh)) => {
                            let ok = (w.fence)(*fence) >= *value;
                            if !ok {
                                *fresh = false;
                            }
                            ok
                        }
                    };
                    if !ready {
                        break;
                    }
                    moved = true;
                    match items.pop_front().map(|(item, ..)| item) {
                        Some(Item::Segment(s, _)) => {
                            piece.push(s);
                            done.segments += 1;
                        }
                        Some(Item::Signal(fence, value)) => (w.signal)(*queue, fence, value),
                        Some(Item::Then(then)) => then(),
                        Some(Item::FenceWait(..)) | None => {}
                    }
                }
                if !piece.is_empty() {
                    (w.run)(*queue, piece);
                    done.pieces += 1;
                }
            }
            if !moved {
                break;
            }
        }
        self.queues.retain(|(_, items)| !items.is_empty());
        done
    }

    /// The queues still holding work: (queue, items, how long the head
    /// waited, the head's waits that do not hold).
    pub(crate) fn blocked(&self, read: &dyn Fn(u64, u8) -> Option<u64>) -> Vec<(usize, usize, Duration, Vec<ValueWait>)> {
        self.queues
            .iter()
            .filter_map(|(q, items)| {
                let (head, since, _) = items.front()?;
                let unmet = match head {
                    Item::Segment(_, waits) => waits.iter().filter(|w| w.holds(read) == Some(false)).copied().collect(),
                    _ => Vec::new(),
                };
                Some((*q, items.len(), since.elapsed(), unmet))
            })
            .collect()
    }

    /// Whether no queue holds anything.
    pub(crate) fn is_empty(&self) -> bool {
        self.queues.is_empty()
    }

    /// Whether `queue` holds anything.
    pub(crate) fn holds_work(&self, queue: usize) -> bool {
        self.queues.iter().any(|(q, items)| *q == queue && !items.is_empty())
    }

    /// Releases work held longer than `limit` at the heads of blocked
    /// queues, as if its waits held: the count of segments forced.
    // shortcut: a safety valve against waits nothing will satisfy (a write
    // we do not emulate), which would hold the title's fences back until
    // its 30 s watchdog ends the run; remove once runs show it never fires.
    pub(crate) fn force_older_than(&mut self, limit: Duration) -> Vec<(usize, Vec<ValueWait>)> {
        let mut forced = Vec::new();
        for (q, items) in &mut self.queues {
            if let Some((Item::Segment(_, waits), since, _)) = items.front_mut()
                && since.elapsed() >= limit
            {
                forced.push((*q, std::mem::take(waits)));
            }
        }
        forced
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::cell::RefCell;
    use std::collections::HashMap;

    /// Memory, fences and a log of what ran, for a recording executor.
    /// With `early`, what a piece writes is pending (in flight) until
    /// [`Fake::complete`], and waits read it over memory (`crate::early`).
    #[derive(Default)]
    struct Fake {
        memory: HashMap<u64, u64>,
        fences: HashMap<usize, u64>,
        log: Vec<String>,
        /// Bottom-of-pipe writes of each segment: (address, value).
        writes: HashMap<&'static str, Vec<(u64, u64)>>,
        early: bool,
        pending: HashMap<u64, u64>,
    }

    impl Fake {
        /// What is in flight completes: its writes land in memory.
        fn complete(&mut self) {
            let pending: Vec<(u64, u64)> = self.pending.drain().collect();
            self.memory.extend(pending);
        }
    }

    fn ge(address: u64, reference: u64) -> ValueWait {
        ValueWait { address, bytes: 8, comparison: GREATER_EQUAL, reference, mask: u64::MAX }
    }

    fn pump(held: &mut Held<&'static str>, fake: &RefCell<Fake>) -> Pumped {
        let memory = |a: u64, _| Some(*fake.borrow().memory.get(&a).unwrap_or(&0));
        let read = |a: u64, b| fake.borrow().pending.get(&a).copied().or_else(|| memory(a, b));
        let fence = |f: usize| *fake.borrow().fences.get(&f).unwrap_or(&0);
        let mut run = |q: usize, segments: Vec<&'static str>| {
            let mut f = fake.borrow_mut();
            for s in &segments {
                for (a, v) in f.writes.get(s).cloned().unwrap_or_default() {
                    if f.early {
                        f.pending.insert(a, v);
                    } else {
                        f.memory.insert(a, v);
                    }
                }
            }
            f.log.push(format!("q{q}: {}", segments.join(" ")));
        };
        let mut signal = |_: usize, fence: usize, value: u64| {
            let mut f = fake.borrow_mut();
            f.fences.insert(fence, value);
            f.log.push(format!("signal {fence} {value}"));
        };
        let early = fake.borrow().early;
        held.pump(&mut World {
            read: &read,
            memory: if early { Some(&memory) } else { None },
            fence: &fence,
            run: &mut run,
            signal: &mut signal,
        })
    }

    /// With early release, a segment's waits are tested after the segments
    /// before it in its queue ran: they read what those will write (in
    /// flight), not the stale value in memory (upstream PR #3381's case: a
    /// label starts at 1, the first segment sets it to 0, the second waits
    /// for 1).
    #[test]
    fn with_early_release_a_wait_reads_what_earlier_segments_of_its_queue_will_write() {
        let fake = RefCell::new(Fake { early: true, ..Fake::default() });
        fake.borrow_mut().memory.insert(0x100, 1);
        fake.borrow_mut().writes.insert("reset", vec![(0x100, 0)]);
        let eq = ValueWait { address: 0x100, bytes: 8, comparison: EQUAL, reference: 1, mask: u64::MAX };
        let mut held = Held::default();
        held.push(1, Item::Segment("reset", vec![]));
        held.push(1, Item::Segment("after", vec![eq]));
        let p = pump(&mut held, &fake);
        assert_eq!(fake.borrow().log, ["q1: reset"], "the wait reads the 0 in flight");
        assert_eq!((p.pieces, p.early), (1, 0));
        fake.borrow_mut().complete();
        pump(&mut held, &fake);
        assert!(held.holds_work(1), "0 landed");
        fake.borrow_mut().memory.insert(0x100, 1);
        let p = pump(&mut held, &fake);
        assert_eq!(fake.borrow().log, ["q1: reset", "q1: after"]);
        assert_eq!(p.early, 0, "released by memory");
    }

    /// With early release, the compute queue's list waiting for a value the
    /// graphics queue's list writes runs right after that list, with the
    /// value still in flight (counted), and memory gets it only when it
    /// completes; a segment with no waits joins the piece before it.
    #[test]
    fn with_early_release_a_wait_runs_behind_its_writer_in_flight() {
        let fake = RefCell::new(Fake { early: true, ..Fake::default() });
        fake.borrow_mut().writes.insert("draw", vec![(0x100, 7)]);
        let mut held = Held::default();
        held.push(2, Item::Segment("depth", vec![ge(0x100, 7)]));
        held.push(2, Item::Segment("more", vec![]));
        held.push(1, Item::Segment("draw", vec![]));
        let p = pump(&mut held, &fake);
        assert_eq!(fake.borrow().log, ["q1: draw", "q2: depth more"]);
        assert_eq!((p.pieces, p.segments, p.early), (2, 3, 1));
        assert_eq!(fake.borrow().memory.get(&0x100), None, "not in memory before it completes");
        fake.borrow_mut().complete();
        assert_eq!(fake.borrow().memory.get(&0x100), Some(&7));
    }

    #[test]
    fn comparisons_test_the_masked_value() {
        let read = |_, _| Some(0x1_0000_0005);
        let w = |comparison, reference| ValueWait { address: 8, bytes: 4, comparison, reference, mask: 0xffff_ffff };
        let expect = [(NEVER, false), (LESS, false), (EQUAL, true), (LESS_EQUAL, true), (GREATER, false), (NOT_EQUAL, false), (GREATER_EQUAL, true), (ALWAYS, true)];
        for (cmp, holds) in expect {
            assert_eq!(w(cmp, 5).holds(&read), Some(holds), "comparison {cmp}");
        }
        assert_eq!(w(LESS, 6).holds(&read), Some(true));
        assert_eq!(w(GREATER, 4).holds(&read), Some(true));
        assert_eq!(w(0, 5).holds(&read), None, "unknown comparison");
        assert_eq!(w(9, 5).holds(&read), None);
        assert_eq!(w(EQUAL, 5).holds(&|_, _| None), None, "unreadable memory");
    }

    /// The compute queue's list waits for a value the graphics queue's list
    /// writes at its end: it runs after it.
    #[test]
    fn a_list_waits_for_another_queues_write() {
        let fake = RefCell::new(Fake::default());
        fake.borrow_mut().writes.insert("draw", vec![(0x100, 7)]);
        let mut held = Held::default();
        held.push(2, Item::Segment("depth", vec![ge(0x100, 7)]));
        let p = pump(&mut held, &fake);
        assert_eq!((p.pieces, p.segments), (0, 0), "held: the value is not there");
        assert!(held.holds_work(2));
        held.push(1, Item::Segment("draw", vec![]));
        let p = pump(&mut held, &fake);
        assert_eq!(fake.borrow().log, ["q1: draw", "q2: depth"]);
        assert_eq!((p.pieces, p.segments, p.waited.len()), (2, 2, 1));
        assert_eq!(p.waited[0].0, 2);
        assert!(!held.holds_work(2));
    }

    #[test]
    fn a_wait_that_holds_defers_nothing() {
        let fake = RefCell::new(Fake::default());
        fake.borrow_mut().memory.insert(0x100, 9);
        let mut held = Held::default();
        held.push(2, Item::Segment("a", vec![ge(0x100, 7)]));
        held.push(2, Item::Segment("b", vec![]));
        let p = pump(&mut held, &fake);
        assert_eq!(fake.borrow().log, ["q2: a b"], "one piece");
        assert_eq!((p.pieces, p.waited.len()), (1, 0));
    }

    /// A list's second segment waits for a value the other queue writes
    /// after reading what the first segment wrote: the cut is what lets
    /// both make progress.
    #[test]
    fn chained_waits_across_two_queues() {
        let fake = RefCell::new(Fake::default());
        {
            let mut f = fake.borrow_mut();
            f.writes.insert("c1", vec![(0x10, 1)]);
            f.writes.insert("g1", vec![(0x20, 1)]);
            f.writes.insert("c2", vec![(0x30, 1)]);
        }
        let mut held = Held::default();
        // Compute: [wait 0x08][c1, writes 0x10][wait 0x20][c2, writes 0x30].
        held.push(2, Item::Segment("c1", vec![ge(0x08, 1)]));
        held.push(2, Item::Segment("c2", vec![ge(0x20, 1)]));
        // Graphics: [g0, writes 0x08], [wait 0x10][g1, writes 0x20], [wait 0x30][g2].
        held.push(1, Item::Segment("g0", vec![]));
        fake.borrow_mut().writes.insert("g0", vec![(0x08, 1)]);
        held.push(1, Item::Segment("g1", vec![ge(0x10, 1)]));
        held.push(1, Item::Segment("g2", vec![ge(0x30, 1)]));
        pump(&mut held, &fake);
        assert_eq!(fake.borrow().log, ["q1: g0", "q2: c1", "q1: g1", "q2: c2", "q1: g2"]);
        assert!(held.blocked(&|_, _| Some(0)).is_empty());
    }

    /// Order within a queue never changes: a later segment whose waits hold
    /// does not pass a held one.
    #[test]
    fn a_queue_keeps_its_order() {
        let fake = RefCell::new(Fake::default());
        let mut held = Held::default();
        held.push(2, Item::Segment("first", vec![ge(0x100, 1)]));
        held.push(2, Item::Segment("second", vec![]));
        pump(&mut held, &fake);
        assert!(fake.borrow().log.is_empty());
        fake.borrow_mut().memory.insert(0x100, 1);
        pump(&mut held, &fake);
        assert_eq!(fake.borrow().log, ["q2: first second"]);
    }

    /// A Signal behind held work is delivered after it: the title's fence
    /// (and its memory) does not move before the work ran.
    #[test]
    fn a_signal_waits_for_the_work_before_it() {
        let fake = RefCell::new(Fake::default());
        let mut held = Held::default();
        held.push(2, Item::Segment("work", vec![ge(0x100, 1)]));
        held.push(2, Item::Signal(5, 3));
        held.push(2, Item::Segment("after", vec![]));
        pump(&mut held, &fake);
        assert_eq!(fake.borrow().fences.get(&5), None, "not signalled yet");
        fake.borrow_mut().memory.insert(0x100, 1);
        pump(&mut held, &fake);
        assert_eq!(fake.borrow().log, ["q2: work", "signal 5 3", "q2: after"]);
    }

    /// An action behind held work (a present) is done once that work has
    /// run, before the work behind it.
    #[test]
    fn an_action_waits_for_the_work_before_it() {
        let log = std::sync::Arc::new(std::sync::Mutex::new(Vec::new()));
        let memory = RefCell::new(0);
        let mut held = Held::default();
        held.push(2, Item::Segment("frame", vec![ge(0x100, 1)]));
        let shown = log.clone();
        held.push(
            2,
            Item::Then(Box::new(move || {
                shown.lock().unwrap().push("present".to_string())
            })),
        );
        held.push(2, Item::Segment("next", vec![]));
        let pump = |held: &mut Held<&'static str>| {
            let read = |_, _| Some(*memory.borrow());
            let mut run = |q: usize, segments: Vec<&'static str>| {
                log.lock()
                    .unwrap()
                    .push(format!("q{q}: {}", segments.join(" ")))
            };
            held.pump(&mut World {
                read: &read,
                memory: None,
                fence: &|_| 0,
                run: &mut run,
                signal: &mut |_, _, _| {},
            })
        };
        pump(&mut held);
        assert!(
            log.lock().unwrap().is_empty(),
            "the present waits for the frame"
        );
        *memory.borrow_mut() = 1;
        pump(&mut held);
        assert_eq!(*log.lock().unwrap(), ["q2: frame", "present", "q2: next"]);
        assert!(!held.holds_work(2));
    }

    #[test]
    fn a_fence_wait_holds_its_queue() {
        let fake = RefCell::new(Fake::default());
        let mut held = Held::default();
        held.push(2, Item::FenceWait(5, 2));
        held.push(2, Item::Segment("after", vec![]));
        pump(&mut held, &fake);
        assert!(fake.borrow().log.is_empty());
        held.push(1, Item::Signal(5, 2));
        pump(&mut held, &fake);
        assert_eq!(fake.borrow().log, ["signal 5 2", "q2: after"]);
    }

    /// The frame ends with work held: it stays held (and is reported), and
    /// the next frame's write releases it.
    #[test]
    fn work_held_at_the_frames_end_stays_held() {
        let fake = RefCell::new(Fake::default());
        let mut held = Held::default();
        held.push(2, Item::Segment("late", vec![ge(0x100, 2)]));
        held.push(2, Item::Signal(5, 1));
        pump(&mut held, &fake);
        let read = |a: u64, _| Some(*fake.borrow().memory.get(&a).unwrap_or(&0));
        let blocked = held.blocked(&read);
        assert_eq!(blocked.len(), 1);
        assert_eq!((blocked[0].0, blocked[0].1, blocked[0].3.clone()), (2, 2, vec![ge(0x100, 2)]));
        fake.borrow_mut().memory.insert(0x100, 2);
        let p = pump(&mut held, &fake);
        assert_eq!(p.waited.len(), 1);
        assert_eq!(fake.borrow().log, ["q2: late", "signal 5 1"]);
    }

    #[test]
    fn unknown_comparisons_are_counted_and_do_not_hold_work() {
        let fake = RefCell::new(Fake::default());
        let mut held = Held::default();
        let odd = ValueWait { address: 0x100, bytes: 4, comparison: 12, reference: 1, mask: 0xffff_ffff };
        held.push(2, Item::Segment("odd", vec![odd]));
        let p = pump(&mut held, &fake);
        assert_eq!((p.segments, p.unknown), (1, 1));
    }

    #[test]
    fn the_valve_releases_only_old_heads() {
        let fake = RefCell::new(Fake::default());
        let mut held = Held::default();
        held.push(2, Item::Segment("stuck", vec![ge(0x100, 1)]));
        assert!(held.force_older_than(Duration::from_secs(60)).is_empty());
        let forced = held.force_older_than(Duration::ZERO);
        assert_eq!(forced, [(2, vec![ge(0x100, 1)])]);
        pump(&mut held, &fake);
        assert_eq!(fake.borrow().log, ["q2: stuck"]);
    }
}
