//! A submission's completion split around the submission lock
//! (`VICTORY_PIPELINE=1`, off by default): the completion thread runs its
//! middle (`Host::finish_middle`: read-backs into the mirrors, the pages
//! shaders marked copied back) without the lock, beside the next preparing,
//! and its beginning and end under it.
//!
//! The split completion is the one `Host::finish` does in one go at the
//! moment its end runs: everything a lock holder does that could see or
//! change what the middle touches settles it first, and a settle of it waits
//! for the middle and runs the end then, as a settle of a submission in
//! flight completes it there.
//! - The middle touches the bytes, snapshots and trackers of the copies its
//!   marks name and of those its lists declare writes through, and title
//!   memory over them: their whole ranges are flown as busy
//!   (`crate::shadow::fly_busy`), which every reader of title memory and
//!   every copy, its own included, crosses (the existing crossings of a
//!   preparing: refreshes, title copies and fills, textures). Steps that
//!   touch copies without a crossing settle it themselves where they meet
//!   them (`Step::GpuWrite`, `Step::Heap`, `Step::Memory`) or always
//!   (`Step::Host`, arbitrary host work).
//! - Completions stay in order: only the completion thread starts a middle,
//!   one at a time, and every other completion ends it first
//!   ([`Host::end_completing`]: `Host::complete_while`, `Host::settle`,
//!   `Host::hold_submissions`, the completion thread's next).
//! - Labels, title fences and the landing of the submission's ranges are in
//!   its end, after its middle; a title Signal behind it waits there too.
//! - Where the lists' probes named pages of copies only a page table knows,
//!   the footprint is not known: that completion runs whole under the lock.

use super::*;

/// Whether the completion thread runs a completion's middle off the
/// submission lock (`VICTORY_PIPELINE=1`; off by default, logged once):
/// only where submissions complete after they return ([`overlap`]).
pub(crate) fn pipeline() -> bool {
    static ON: std::sync::LazyLock<bool> = std::sync::LazyLock::new(|| {
        let on = crate::switch::on("VICTORY_PIPELINE");
        if on {
            tracing::info!(
                "the completion thread copies back off the submission lock, beside the next \
                 preparing (VICTORY_PIPELINE)"
            );
        }
        on
    });
    #[cfg(test)]
    match tests::PIPELINE.load(Ordering::Acquire) {
        1 => return false,
        2 => return overlap(),
        _ => {}
    }
    *ON && overlap()
}

/// A completion whose middle runs off the submission lock: the record (its
/// lists, signals, what its end releases) waits in `Submitted::completing`.
pub(super) struct Completing {
    pub(super) record: InFlight,
    begun: Begun,
    done: std::sync::Arc<Handoff>,
}

/// Set once the middle is done.
#[derive(Default)]
struct Handoff {
    done: Mutex<bool>,
    ended: std::sync::Condvar,
}

/// `took` of copying back (`COPY_BACK`), and with diagnostics by who did it.
pub(super) fn copied_back(took: std::time::Duration) {
    let us = took.as_micros() as u64;
    COPY_BACK[0].fetch_add(us, Ordering::Relaxed);
    #[cfg(feature = "diagnostics")]
    crate::copystat::add(
        match WHO.with(std::cell::Cell::get) {
            Who::Thread => crate::copystat::C::BackThreadUs,
            Who::Settle => crate::copystat::C::BackSettleUs,
            Who::Waiting => crate::copystat::C::BackWaitUs,
            Who::Other => crate::copystat::C::BackOtherUs,
        },
        us,
    );
}

/// Who copies back on this thread now (diagnostics).
#[derive(Clone, Copy)]
#[cfg_attr(not(feature = "diagnostics"), allow(dead_code))]
pub(super) enum Who {
    /// The completion thread.
    Thread,
    /// A settle of a preparing.
    Settle,
    /// The waits before a title submission is prepared.
    Waiting,
    /// Anything else: `execute`, `complete_all`, `hold_submissions`.
    Other,
}

#[cfg(feature = "diagnostics")]
thread_local! {
    static WHO: std::cell::Cell<Who> = const { std::cell::Cell::new(Who::Other) };
    /// The thread prepares a submission (`Host::run`): its waits for a
    /// middle are not overlap.
    static PREPARING: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

/// Copying back on this thread counts as `who`'s until the result goes
/// (diagnostics; nothing without).
pub(super) fn who(who: Who) -> impl Drop {
    struct Was(#[cfg(feature = "diagnostics")] Who);
    impl Drop for Was {
        fn drop(&mut self) {
            #[cfg(feature = "diagnostics")]
            WHO.with(|w| w.set(self.0));
        }
    }
    #[cfg(feature = "diagnostics")]
    return Was(WHO.with(|w| w.replace(who)));
    #[cfg(not(feature = "diagnostics"))]
    {
        let _ = who;
        Was()
    }
}

/// When a preparing and a middle started, if they run (diagnostics): the
/// time both run is the overlap.
#[cfg(feature = "diagnostics")]
struct Spans {
    preparing: Option<Instant>,
    middle: Option<Instant>,
}

#[cfg(feature = "diagnostics")]
static SPANS: Mutex<Spans> = Mutex::new(Spans {
    preparing: None,
    middle: None,
});

/// A preparing (`preparing`) or a middle starts (`on`) or ends: on an end,
/// the time since both ran is overlap (diagnostics).
#[cfg(feature = "diagnostics")]
fn span(preparing: bool, on: bool) {
    let mut s = SPANS.lock().unwrap_or_else(|e| e.into_inner());
    let Spans {
        preparing: p,
        middle: m,
    } = &mut *s;
    let (mine, other) = if preparing { (p, *m) } else { (m, *p) };
    let now = Instant::now();
    if on {
        *mine = Some(now);
    } else if let Some(start) = mine.take()
        && let Some(other) = other
    {
        let both = now.saturating_duration_since(start.max(other));
        crate::copystat::add(crate::copystat::C::PipeOverlapUs, both.as_micros() as u64);
    }
}

/// `Host::run` starts or ends preparing a submission (diagnostics: the
/// overlap of middles with it).
pub(super) fn preparing(on: bool) {
    #[cfg(feature = "diagnostics")]
    {
        PREPARING.with(|p| p.set(on));
        span(true, on);
    }
    let _ = on;
}

impl Host {
    /// The completion thread's completion of the first submission in flight
    /// (reached) with `VICTORY_PIPELINE`: its beginning under the lock, its
    /// middle without it where its footprint is known (flown as busy, the
    /// record parked in `Submitted::completing`), its end under the lock
    /// again unless a lock holder ended it meanwhile. Returns the lock held.
    pub(super) fn complete_split<'a>(
        &'a self,
        mut submitted: std::sync::MutexGuard<'a, Submitted>,
    ) -> std::sync::MutexGuard<'a, Submitted> {
        self.end_completing(&mut submitted);
        let mut record = submitted.flight.pop_front().expect("a first");
        let lists = std::mem::take(&mut record.lists);
        let runs: Vec<&HostList> = lists.iter().map(|l| &**l).collect();
        let (middle, begun) = self.finish_begin(&mut record, &runs, Waited::Completion);
        drop(runs);
        record.lists = lists;
        let Some(footprint) = middle.footprint(&begun.back) else {
            #[cfg(feature = "diagnostics")]
            crate::copystat::add(crate::copystat::C::PipeWhole, 1);
            self.finish_middle(middle);
            let quiet = !submitted.flight.iter().any(|r| r.marks);
            self.complete_end(record, begun, quiet);
            return submitted;
        };
        let owner = record.owner;
        crate::shadow::fly_busy(owner, footprint);
        let done = std::sync::Arc::new(Handoff::default());
        submitted.completing = Some(Completing {
            record,
            begun,
            done: done.clone(),
        });
        drop(submitted);
        #[cfg(test)]
        tests::hold_middle(owner);
        let _ = owner;
        #[cfg(feature = "diagnostics")]
        let started = {
            crate::copystat::add(crate::copystat::C::PipeSplit, 1);
            span(false, true);
            Instant::now()
        };
        self.finish_middle(middle);
        #[cfg(feature = "diagnostics")]
        {
            span(false, false);
            crate::copystat::add(
                crate::copystat::C::PipeMiddleUs,
                started.elapsed().as_micros() as u64,
            );
        }
        *done.done.lock().unwrap_or_else(|e| e.into_inner()) = true;
        done.ended.notify_all();
        let mut submitted = self.lock_submitted();
        if submitted
            .completing
            .as_ref()
            .is_some_and(|c| std::sync::Arc::ptr_eq(&c.done, &done))
        {
            self.end_completing(&mut submitted);
        }
        submitted
    }

    /// Ends the completion whose middle runs off the lock, if one does: waits
    /// for its middle (the lock held: the middle takes none a holder can
    /// hold while it waits here), then runs its end. Whether there was one.
    pub(super) fn end_completing(&self, submitted: &mut Submitted) -> bool {
        let Some(c) = submitted.completing.take() else {
            return false;
        };
        {
            let mut done = c.done.done.lock().unwrap_or_else(|e| e.into_inner());
            if !*done {
                #[cfg(feature = "diagnostics")]
                let (waiting, preparing) = {
                    let preparing = PREPARING.with(std::cell::Cell::get);
                    if preparing {
                        span(true, false);
                    }
                    (Instant::now(), preparing)
                };
                while !*done {
                    done = c.done.ended.wait(done).unwrap_or_else(|e| e.into_inner());
                }
                #[cfg(feature = "diagnostics")]
                {
                    use crate::copystat::{C, add};
                    add(C::PipeWaits, 1);
                    add(C::PipeWaitUs, waiting.elapsed().as_micros() as u64);
                    if preparing {
                        span(true, true);
                    }
                }
            }
        }
        let quiet = !submitted.flight.iter().any(|r| r.marks);
        self.complete_end(c.record, c.begun, quiet);
        true
    }

    /// [`Host::finish_end`] of `record`, then its lists recycled.
    fn complete_end(&self, mut record: InFlight, begun: Begun, quiet: bool) {
        let lists = std::mem::take(&mut record.lists);
        let runs: Vec<&HostList> = lists.iter().map(|l| &**l).collect();
        self.finish_end(record, begun, &runs, quiet);
        drop(runs);
        lists.into_iter().for_each(crate::list::recycle);
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;

    /// [`pipeline`] for tests: 0 as the environment says, 1 off, 2 on.
    pub(crate) static PIPELINE: std::sync::atomic::AtomicU8 = std::sync::atomic::AtomicU8::new(0);

    /// The pipeline on (`on`) or off until the result goes.
    pub(crate) fn pipeline_for(on: bool) -> impl Drop {
        struct Reset;
        impl Drop for Reset {
            fn drop(&mut self) {
                PIPELINE.store(0, Ordering::Release);
            }
        }
        PIPELINE.store(if on { 2 } else { 1 }, Ordering::Release);
        Reset
    }

    /// A gate the completion thread stops at before a middle off the lock:
    /// the owner it stopped for, and whether it may go.
    #[derive(Default)]
    struct Hold {
        at: Mutex<(Option<u64>, bool)>,
        moved: std::sync::Condvar,
    }

    static HOLD: Mutex<Option<std::sync::Arc<Hold>>> = Mutex::new(None);

    /// The completion thread, about to run submission `owner`'s middle off
    /// the lock: stops there while a test holds it (5 s at most).
    pub(super) fn hold_middle(owner: u64) {
        let Some(hold) = HOLD.lock().unwrap_or_else(|e| e.into_inner()).clone() else {
            return;
        };
        let mut at = hold.at.lock().unwrap_or_else(|e| e.into_inner());
        at.0 = Some(owner);
        hold.moved.notify_all();
        let deadline = Instant::now() + Duration::from_secs(5);
        while !at.1 && Instant::now() < deadline {
            at = hold
                .moved
                .wait_timeout(at, Duration::from_millis(50))
                .unwrap_or_else(|e| e.into_inner())
                .0;
        }
    }

    /// Holds the next middle off the lock until the result goes.
    struct Held(std::sync::Arc<Hold>);

    impl Held {
        fn new() -> Self {
            let hold = std::sync::Arc::new(Hold::default());
            *HOLD.lock().unwrap_or_else(|e| e.into_inner()) = Some(hold.clone());
            Held(hold)
        }

        /// The submission whose middle the completion thread stopped at, once
        /// it has (5 s at most).
        fn stopped(&self) -> Option<u64> {
            let deadline = Instant::now() + Duration::from_secs(5);
            let mut at = self.0.at.lock().unwrap_or_else(|e| e.into_inner());
            while at.0.is_none() && Instant::now() < deadline {
                at = self
                    .0
                    .moved
                    .wait_timeout(at, Duration::from_millis(50))
                    .unwrap_or_else(|e| e.into_inner())
                    .0;
            }
            at.0
        }

        /// Lets the middle go `ms` from now, from another thread.
        fn go_after(&self, ms: u64) -> std::thread::JoinHandle<()> {
            let hold = self.0.clone();
            std::thread::spawn(move || {
                std::thread::sleep(Duration::from_millis(ms));
                let mut at = hold.at.lock().unwrap_or_else(|e| e.into_inner());
                at.1 = true;
                hold.moved.notify_all();
            })
        }
    }

    impl Drop for Held {
        fn drop(&mut self) {
            HOLD.lock().unwrap_or_else(|e| e.into_inner()).take();
            let mut at = self.0.at.lock().unwrap_or_else(|e| e.into_inner());
            at.1 = true;
            self.0.moved.notify_all();
        }
    }

    /// The busy footprint is crossed by every copy, its own too, and by
    /// title reads; it lands with the submission.
    #[test]
    fn a_busy_footprint_is_crossed_by_every_copy_until_it_lands() {
        let _queues = crate::queue::tests::TITLE_QUEUES
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        let owner = u64::MAX - 7;
        crate::shadow::fly_busy(owner, vec![(0x7000_0000, 0x7001_0000)]);
        assert_eq!(crate::shadow::busy_owner(), Some(owner));
        assert_eq!(crate::shadow::busy(0x7000_fff0, 0x7001_0010), Some(owner));
        assert_eq!(crate::shadow::busy(0x7001_0000, 0x7002_0000), None);
        let copy = 0x1234_5678;
        assert_eq!(
            crate::shadow::crossing(0x7000_0100, 0x7000_0200, copy).map(|c| c.0),
            Some(owner),
            "any copy crosses it"
        );
        assert_eq!(
            crate::shadow::crossing(0x7000_0100, 0x7000_0200, 0).map(|c| c.0),
            Some(owner),
            "title reads cross it"
        );
        crate::shadow::land(owner);
        assert_eq!(crate::shadow::busy_owner(), None);
        assert_eq!(crate::shadow::busy(0x7000_0000, 0x7001_0000), None);
        assert_eq!(
            crate::shadow::crossing(0x7000_0100, 0x7000_0200, copy),
            None
        );
    }

    /// A copy in video memory of a fresh pool of `pages` pages that takes
    /// write marks, and the pool; none without one.
    fn marked_copy(pages: usize) -> Option<(usize, std::sync::Arc<crate::shadow::Shadow>)> {
        let p = crate::watch::tests::pool(pages);
        let len = (pages * crate::watch::PAGE) as u64;
        crate::shadow::register(p, p as u64, len);
        match crate::shadow::find(p as u64, len)
            .filter(|c| c.vram_readback().is_some() && c.marked())
            .filter(|_| crate::shadow::captures_write_marks())
        {
            Some(copy) => Some((p, copy)),
            None => {
                release(p, None);
                None
            }
        }
    }

    fn release(p: usize, copy: Option<std::sync::Arc<crate::shadow::Shadow>>) {
        crate::shadow::unregister(p, p as u64);
        drop(copy);
        crate::watch::tests::free(p);
    }

    /// A list that reads `copy` whole and whose "shader" writes `value` at
    /// title address `at` of it through a marked view (the page's mark set
    /// on the GPU).
    fn marking_writer(
        host: &Host,
        copy: &std::sync::Arc<crate::shadow::Shadow>,
        at: usize,
        value: u32,
    ) -> HostList {
        let list = host.new_list().unwrap();
        let (start, len) = copy.range();
        copy.refresh(&list, start, len);
        list.marking();
        copy.gpu_writes(&list, start, len, true);
        crate::shadow::tests::mark_on_gpu(
            &list,
            copy,
            ((at as u64 - start) / crate::watch::PAGE as u64) as u32,
        )
        .expect("marked");
        let (up, data) = host.upload_buffer(4).expect("an upload buffer");
        // SAFETY: a mapped upload buffer of 4 bytes; an open list, live buffers.
        unsafe {
            data.cast::<u32>().write_volatile(value);
            list.list
                .CopyBufferRegion(&copy.buffer, copy.offset(at as u64), &up, 0, 4);
        }
        list.keep(&up);
        list
    }

    fn word(at: usize) -> u32 {
        // SAFETY: a word of a test's pool.
        unsafe { (at as *const u32).read_volatile() }
    }

    /// With the pipeline, a submission's marked page goes back off the lock;
    /// a later submission that reads that page at submission (a title copy)
    /// while the copy-back is held waits for it and copies the bytes that
    /// went back, not the page as it was.
    #[test]
    fn a_title_read_of_a_page_going_back_waits_for_its_bytes() {
        let Some(host) = host() else { return };
        if !overlap() {
            return; // VICTORY_GPU_OVERLAP=0
        }
        let _queues = crate::queue::tests::TITLE_QUEUES
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        let _on = pipeline_for(true);
        let Some((p, a)) = marked_copy(1) else { return };
        let q = crate::watch::tests::pool(1);
        let held = Held::new();
        let writer = marking_writer(host, &a, p + 16, 0x5151_0001);
        writer.close();
        host.submit(0, vec![std::sync::Arc::new(writer)]);
        let owner = held
            .stopped()
            .expect("the completion thread stopped at the middle");
        assert_eq!(crate::shadow::busy_owner(), Some(owner), "busy meanwhile");
        assert_ne!(word(p + 16), 0x5151_0001, "not back before its middle");
        let reader = host.new_list().unwrap();
        reader.defer_copy(q as *mut u8, (p + 16) as *const u8, 4);
        reader.close();
        let going = held.go_after(300);
        let started = Instant::now();
        host.submit(0, vec![std::sync::Arc::new(reader)]);
        let took = started.elapsed();
        going.join().unwrap();
        assert!(
            took >= Duration::from_millis(250),
            "the reader waited for the copy-back ({took:?})"
        );
        assert_eq!(
            word(q),
            0x5151_0001,
            "the reader copied the bytes that went back"
        );
        host.complete_all();
        assert_eq!(crate::shadow::busy_owner(), None, "landed");
        assert_eq!(word(p + 16), 0x5151_0001);
        drop(held);
        crate::watch::tests::free(q);
        release(p, Some(a));
    }

    /// While a submission's copy-back runs off the lock, a later preparing
    /// that touches its copy waits for it even where no range in flight says
    /// so (a page the title left alone, away from the declared range: its
    /// copy is busy), and one touching another copy does not.
    #[test]
    fn a_preparing_touching_a_copy_going_back_waits_and_others_do_not() {
        let Some(host) = host() else { return };
        if !overlap() {
            return; // VICTORY_GPU_OVERLAP=0
        }
        let _queues = crate::queue::tests::TITLE_QUEUES
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        let _on = pipeline_for(true);
        let page = crate::watch::PAGE;
        let vram = |pages: usize| {
            let p = crate::watch::tests::pool(pages);
            let len = (pages * page) as u64;
            crate::shadow::register(p, p as u64, len);
            match crate::shadow::find(p as u64, len).filter(|c| c.vram_readback().is_some()) {
                Some(copy) => Some((p, copy)),
                None => {
                    release(p, None);
                    None
                }
            }
        };
        let Some((p, a)) = vram(2) else { return };
        let Some((q, b)) = vram(1) else {
            return release(p, Some(a));
        };
        // An unmarked declared range on the first page of `a`.
        let writer = host.new_list().unwrap();
        let (start, len) = a.range();
        a.refresh(&writer, start, len);
        a.gpu_writes(&writer, (p + 16) as u64, 4, false);
        let (up, data) = host.upload_buffer(4).expect("an upload buffer");
        // SAFETY: a mapped upload buffer of 4 bytes; an open list, live buffers.
        unsafe {
            data.cast::<u32>().write_volatile(0x8484_0004);
            writer
                .list
                .CopyBufferRegion(&a.buffer, a.offset((p + 16) as u64), &up, 0, 4);
        }
        writer.keep(&up);
        writer.close();
        let held = Held::new();
        host.submit(0, vec![std::sync::Arc::new(writer)]);
        let owner = held
            .stopped()
            .expect("the completion thread stopped at the middle");
        assert_eq!(crate::shadow::busy_owner(), Some(owner));
        let refresh = |copy: &std::sync::Arc<crate::shadow::Shadow>, at: usize| {
            let list = host.new_list().unwrap();
            copy.refresh(&list, at as u64, page as u64);
            list.close();
            let started = Instant::now();
            host.submit(0, vec![std::sync::Arc::new(list)]);
            started.elapsed()
        };
        let other = refresh(&b, q);
        assert!(
            other < Duration::from_millis(250),
            "another copy's refresh did not wait ({other:?})"
        );
        assert_eq!(crate::shadow::busy_owner(), Some(owner), "still busy");
        let going = held.go_after(300);
        let same = refresh(&a, p + page);
        going.join().unwrap();
        assert!(
            same >= Duration::from_millis(250),
            "the busy copy's second page waited for its copy-back ({same:?})"
        );
        assert_eq!(crate::shadow::busy_owner(), None, "settled");
        host.complete_all();
        assert_eq!(word(p + 16), 0x8484_0004);
        drop(held);
        release(q, Some(b));
        release(p, Some(a));
    }

    /// A submission's bottom-of-pipe write (its label) reaches title memory
    /// only after its marked page went back off the lock: a title thread
    /// that sees the label finds the page's new bytes, and while the
    /// copy-back is held neither is there.
    #[test]
    fn a_label_follows_its_copy_back_off_the_lock() {
        let Some(host) = host() else { return };
        if !overlap() {
            return; // VICTORY_GPU_OVERLAP=0
        }
        let _queues = crate::queue::tests::TITLE_QUEUES
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        let _on = pipeline_for(true);
        let Some((p, a)) = marked_copy(1) else { return };
        let q = crate::watch::tests::pool(1);
        let held = Held::new();
        let writer = marking_writer(host, &a, p + 16, 0x6262_0002);
        writer.defer_write(q as *mut u8, crate::queue::Written::Value(1), 4);
        writer.close();
        let watcher = std::thread::spawn(move || {
            let deadline = Instant::now() + Duration::from_secs(10);
            while word(q) != 1 {
                if Instant::now() > deadline {
                    return None;
                }
                std::hint::spin_loop();
            }
            Some(word(p + 16))
        });
        host.submit(0, vec![std::sync::Arc::new(writer)]);
        held.stopped()
            .expect("the completion thread stopped at the middle");
        std::thread::sleep(Duration::from_millis(100));
        assert_eq!(word(q), 0, "no label while the copy-back is held");
        assert_ne!(word(p + 16), 0x6262_0002, "nor the page");
        drop(held);
        let seen = watcher.join().unwrap();
        assert_eq!(
            seen,
            Some(0x6262_0002),
            "the label came after the page went back"
        );
        host.complete_all();
        crate::watch::tests::free(q);
        release(p, Some(a));
    }

    /// A page remap (`Host::hold_submissions`) waits for a copy-back running
    /// off the lock: no title page is remapped under its writes.
    #[test]
    fn holding_submissions_waits_for_a_copy_back_off_the_lock() {
        let Some(host) = host() else { return };
        if !overlap() {
            return; // VICTORY_GPU_OVERLAP=0
        }
        let _queues = crate::queue::tests::TITLE_QUEUES
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        let _on = pipeline_for(true);
        let Some((p, a)) = marked_copy(1) else { return };
        let held = Held::new();
        let writer = marking_writer(host, &a, p + 16, 0x7373_0003);
        writer.close();
        host.submit(0, vec![std::sync::Arc::new(writer)]);
        held.stopped()
            .expect("the completion thread stopped at the middle");
        let going = held.go_after(300);
        let started = Instant::now();
        let quiet = host.hold_submissions();
        let took = started.elapsed();
        assert!(quiet.completing.is_none(), "ended");
        assert_eq!(word(p + 16), 0x7373_0003, "back before the hold");
        drop(quiet);
        going.join().unwrap();
        assert!(
            took >= Duration::from_millis(250),
            "the hold waited for the copy-back ({took:?})"
        );
        host.complete_all();
        drop(held);
        release(p, Some(a));
    }

    /// Guards after a copy-back off the lock are as after one under it: a
    /// second submission marking the same copy, prepared while the first's
    /// copy-back is held, waits for it (its snapshot and arming of the copy);
    /// both pages go back; and a title write after them reaches the copy at
    /// the next refresh while the shaders' bytes stay (pipeline off, then
    /// on).
    #[test]
    fn guards_after_a_copy_back_off_the_lock_are_as_under_it() {
        let Some(host) = host() else { return };
        if !overlap() {
            return; // VICTORY_GPU_OVERLAP=0
        }
        let _queues = crate::queue::tests::TITLE_QUEUES
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        for on in [false, true] {
            let _on = pipeline_for(on);
            let Some((p, a)) = marked_copy(2) else { return };
            let page = crate::watch::PAGE;
            let held = on.then(Held::new);
            let first = marking_writer(host, &a, p + 16, 0x1000_0001);
            first.close();
            host.submit(0, vec![std::sync::Arc::new(first)]);
            if let Some(held) = &held {
                held.stopped()
                    .expect("the completion thread stopped at the middle");
            }
            let going = held.as_ref().map(|h| h.go_after(300));
            let second = marking_writer(host, &a, p + page + 16, 0x2000_0002);
            second.close();
            let started = Instant::now();
            host.submit(0, vec![std::sync::Arc::new(second)]);
            let took = started.elapsed();
            if let Some(going) = going {
                going.join().unwrap();
                assert!(
                    took >= Duration::from_millis(250),
                    "the second waited for the first's copy-back ({took:?})"
                );
            }
            host.complete_all();
            drop(held);
            assert_eq!(
                (word(p + 16), word(p + page + 16)),
                (0x1000_0001, 0x2000_0002),
                "both went back (pipeline {on})"
            );
            // SAFETY: a word of our pool (guarded: the write marks the page).
            unsafe { ((p + 64) as *mut u32).write_volatile(0x3333) };
            let reader = host.new_list().unwrap();
            let (start, len) = a.range();
            a.refresh(&reader, start, len);
            reader.close();
            host.submit(0, vec![std::sync::Arc::new(reader)]);
            host.complete_all();
            assert_eq!(
                (
                    a.read_host((p + 16) as u64, 1)[0],
                    a.read_host((p + 64) as u64, 1)[0],
                    a.read_host((p + page + 16) as u64, 1)[0],
                ),
                (0x1000_0001, 0x3333, 0x2000_0002),
                "the title's later write reached the copy, the shaders' bytes stayed \
                 (pipeline {on})"
            );
            release(p, Some(a));
        }
    }
}
