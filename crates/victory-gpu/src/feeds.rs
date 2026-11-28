//! What feeds the presented image, while pipelines are being made.
//!
//! A pass dropped for want of its pipeline leaves what it writes as it
//! was, and every pass that runs after it and reads that reads stale data.
//! When the pass that writes the presented image is dropped,
//! `crate::shown` holds the frame and its pipeline goes first. When a pass
//! further back is dropped, the image is written, but from stale inputs:
//! Grand Theft Auto VI's world on a cold driver cache stayed black for
//! ~50 s after the final upscale was made, until the compute pass that
//! resolves the lighting (`CS_RaytracingResolve`) was, made 6 s after a
//! worker took it, 47 s after its first dispatch.
//!
//! So while pipelines are queued, each frame notes what the passes that run
//! read and write (the views their code uses, their targets) and what the
//! dropped ones would have written. At the present, the walk back from
//! the presented images through passes that ran finds the dropped passes
//! whose writes they read, nearest first, and the first [`HURRY_MAX`] of
//! their pipelines go to the head of the workers' queue.
//!
//! What it must not find is the world's content. Reads are kept per pass,
//! not per pipeline: a pipeline many passes share (a blit) would join
//! every one of their inputs. And a resource with more than [`FAN_IN_MAX`]
//! dropped writers is not followed to them: that is a target hundreds of
//! draws fill (depth, a geometry buffer, a shadow map), missing some of its
//! objects, not one a single pass leaves wholly stale. Taking those (666
//! pipelines in one round) put them ahead of the pipelines the queue would
//! have made next, and the world went dark again for a minute.
//!
//! Resources only, by title object: buffers and views a shader indexes in
//! the heap itself are not followed. A frame's passes are taken in the
//! order they were recorded on any thread, so a pass may be counted as
//! feeding the image that runs after the read: that only moves a pipeline
//! ahead in the queue.

use std::cell::RefCell;
use std::collections::{HashMap, HashSet, VecDeque};
use std::sync::{LazyLock, Mutex};

use crate::pipeline::Waited;

/// Notes a frame keeps at most (writes and reads): the graph is cleared at
/// every present, and a frame of Grand Theft Auto VI's world has ~5,000
/// draws; a title that stops presenting while it records does not grow it
/// past this.
const NOTES_MAX: usize = 1 << 20;

/// Dropped writers of one resource, at most, for the walk to take them: a
/// pass a frame has only one or a few of (lighting, a resolve) leaves the
/// resource stale; hundreds (geometry into depth or a geometry buffer)
/// leave objects missing.
const FAN_IN_MAX: usize = 4;

/// Pipelines put first per present, at most (nearest the image first).
const HURRY_MAX: usize = 8;

/// The frame's passes: per resource, the passes that ran and wrote it and
/// the dropped pipelines that would have; what each pass that ran read;
/// the dropped pipelines (`T`: what to hurry); `notes` counts the entries.
struct Graph<T> {
    ran: HashMap<usize, Vec<u32>>,
    dropped_writers: HashMap<usize, HashSet<usize>>,
    reads: Vec<Vec<usize>>,
    dropped: HashMap<usize, T>,
    notes: usize,
}

impl<T> Default for Graph<T> {
    fn default() -> Self {
        Graph {
            ran: HashMap::new(),
            dropped_writers: HashMap::new(),
            reads: Vec::new(),
            dropped: HashMap::new(),
            notes: 0,
        }
    }
}

impl<T> Graph<T> {
    /// A pass ran, reading `reads` and writing `writes`. A null descriptor
    /// (`0`) names no resource and is left out of both.
    ///
    /// A pass that wrote nothing is not noted at all: no walk back from a
    /// presented image reaches it, so its reads would only grow `reads`
    /// (a title that stops presenting while its pipelines are queued kept
    /// `NOTES_MAX` passes' worth of them, none counted).
    fn ran(&mut self, reads: &[usize], writes: &[usize]) {
        if self.notes >= NOTES_MAX || !writes.iter().any(|&r| r != 0) {
            return;
        }
        let pass = self.reads.len() as u32;
        let mut noted = 0;
        for &r in writes.iter().filter(|&&r| r != 0) {
            self.ran.entry(r).or_default().push(pass);
            noted += 1;
        }
        let mut reads = reads.to_vec();
        reads.retain(|&r| r != 0);
        self.notes += noted + reads.len();
        self.reads.push(reads);
    }

    /// A pass of `pipeline` (`what`) was dropped; it would have written
    /// `writes` (a null descriptor names nothing).
    fn dropped(&mut self, pipeline: usize, what: impl FnOnce() -> T, writes: &[usize]) {
        if self.notes >= NOTES_MAX || !writes.iter().any(|&r| r != 0) {
            return;
        }
        for &r in writes.iter().filter(|&&r| r != 0) {
            if self.dropped_writers.entry(r).or_default().insert(pipeline) {
                self.notes += 1;
            }
        }
        self.dropped.entry(pipeline).or_insert_with(what);
    }

    /// The dropped pipelines whose writes reach `images` through passes
    /// that ran, nearest first; resources with more than [`FAN_IN_MAX`]
    /// dropped writers give none.
    fn feeding(&self, images: &[usize]) -> Vec<&T> {
        let (mut seen, mut walked, mut found) = (HashSet::new(), HashSet::new(), HashSet::new());
        let mut out = Vec::new();
        let mut next: VecDeque<usize> = images.iter().copied().collect();
        while let Some(r) = next.pop_front() {
            if !seen.insert(r) {
                continue;
            }
            if let Some(d) = self.dropped_writers.get(&r)
                && d.len() <= FAN_IN_MAX
            {
                let mut d: Vec<usize> = d.iter().copied().filter(|p| found.insert(*p)).collect();
                d.sort_unstable();
                out.extend(d.iter().filter_map(|p| self.dropped.get(p)));
            }
            for &pass in self.ran.get(&r).into_iter().flatten() {
                if walked.insert(pass) {
                    next.extend(&self.reads[pass as usize]);
                }
            }
        }
        out
    }
}

static GRAPH: LazyLock<Mutex<Graph<Waited>>> = LazyLock::new(Mutex::default);

/// What the pass this thread is binding reads and writes through views,
/// and whether it is noted at all.
#[derive(Default)]
struct Pass {
    on: bool,
    reads: Vec<usize>,
    writes: Vec<usize>,
}

thread_local! {
    static PASS: RefCell<Pass> = RefCell::default();
}

/// Whether passes are noted: while pipelines are queued.
fn tracking() -> bool {
    crate::pipeline::making() > 0
}

/// A pass starts binding its views.
pub(crate) fn begin() {
    PASS.with_borrow_mut(|p| {
        p.on = tracking();
        p.reads.clear();
        p.writes.clear();
    });
}

/// The pass being bound reads (or, `write`, writes) `resource` through a
/// view. A null descriptor names no resource (`Graph` would ignore it).
pub(crate) fn view(resource: usize, write: bool) {
    PASS.with_borrow_mut(|p| {
        if p.on && resource != 0 {
            if write { &mut p.writes } else { &mut p.reads }.push(resource);
        }
    });
}

/// The pass just bound runs, also writing `targets` and `depth` (0: none).
pub(crate) fn ran(targets: impl IntoIterator<Item = usize>, depth: usize) {
    PASS.with_borrow_mut(|p| {
        if !p.on {
            return;
        }
        p.on = false;
        p.writes.extend(targets);
        p.writes.extend((depth != 0).then_some(depth));
        let mut g = GRAPH.lock().unwrap_or_else(|e| e.into_inner());
        g.ran(&p.reads, &p.writes);
    });
}

/// A pass of `pipeline` (`waited`: its pending host pipeline) was dropped
/// while that is being made; it would have written `writes`.
pub(crate) fn dropped(pipeline: usize, waited: &Waited, writes: &[usize]) {
    if tracking() {
        let mut g = GRAPH.lock().unwrap_or_else(|e| e.into_inner());
        g.dropped(pipeline, || waited.clone(), writes);
    }
}

/// The title presents `images`: the pending pipelines of the frame's
/// dropped passes that feed them, nearest first, at most [`HURRY_MAX`];
/// the frame's notes are cleared.
pub(crate) fn presented(images: &[usize]) -> Vec<Waited> {
    let g = std::mem::take(&mut *GRAPH.lock().unwrap_or_else(|e| e.into_inner()));
    if g.dropped.is_empty() {
        return Vec::new();
    }
    g.feeding(images)
        .into_iter()
        .filter(|w| crate::pipeline::may_hurry(w))
        .take(HURRY_MAX)
        .cloned()
        .collect()
}

#[cfg(test)]
mod tests {
    use super::{FAN_IN_MAX, Graph, NOTES_MAX};

    /// Frames of a deferred renderer while pipelines are made. In the
    /// first, the lighting resolve and the shadow pass are dropped; the
    /// composite, the temporal pass and the upscale ran: the walk finds the
    /// resolve, and not the shadows (whose reader did not run) nor the
    /// dropped particles nothing presented reads. Once the resolve runs,
    /// the next frame's walk finds the shadows behind it.
    #[test]
    fn the_dropped_passes_a_presented_image_reads_are_found_nearest_first() {
        let (gbuffer, shadow, lit, colour, history, taa, image, particles) =
            (1, 2, 3, 4, 5, 6, 7, 8);
        let frame = |resolve_ran: bool| {
            let mut g = Graph::default();
            g.ran(&[], &[gbuffer]);
            g.dropped(11, || "shadows", &[shadow]);
            if resolve_ran {
                g.ran(&[gbuffer, shadow], &[lit]);
            } else {
                g.dropped(12, || "resolve", &[lit]);
            }
            g.ran(&[lit], &[colour]);
            g.ran(&[colour, history], &[taa, history]);
            g.ran(&[taa], &[image]);
            g.dropped(16, || "particles", &[particles]);
            g
        };
        assert_eq!(frame(false).feeding(&[image]), [&"resolve"]);
        assert_eq!(frame(true).feeding(&[image]), [&"shadows"]);
        assert_eq!(frame(false).feeding(&[particles]), [&"particles"]);
        assert!(frame(false).feeding(&[42]).is_empty());
    }

    /// Two dropped passes at different distances: the nearer comes first.
    #[test]
    fn nearer_feeders_come_first() {
        let mut g = Graph::default();
        g.ran(&[2, 3], &[9]);
        g.ran(&[5], &[2]);
        g.dropped(6, || "far", &[5]);
        g.dropped(7, || "near", &[3]);
        assert_eq!(g.feeding(&[9]), [&"near", &"far"]);
    }

    /// The pass writing the image itself, dropped, is found; a dropped
    /// pass that writes nothing known is not noted.
    #[test]
    fn the_image_s_own_writer_and_writeless_passes() {
        let mut g = Graph::default();
        g.dropped(1, || "upscale", &[7]);
        g.dropped(2, || "unknown", &[]);
        assert_eq!(g.feeding(&[7]), [&"upscale"]);
        assert!(!g.dropped.contains_key(&2));
    }

    /// Reads feeding each other in a cycle (a history read and written by
    /// one pass) end the walk.
    #[test]
    fn a_cycle_ends() {
        let mut g = Graph::default();
        g.ran(&[5], &[5, 7]);
        g.dropped(2, || "feeder", &[5]);
        assert_eq!(g.feeding(&[7]), [&"feeder"]);
    }

    /// A target many dropped draws write (depth, a geometry buffer) is not
    /// followed to them; the one pass that leaves the lighting stale is.
    #[test]
    fn a_target_many_dropped_draws_write_is_not_followed() {
        let (depth, lit, image) = (1, 2, 3);
        let mut g = Graph::default();
        for draw in 100..100 + FAN_IN_MAX + 1 {
            g.dropped(draw, || "geometry", &[depth]);
        }
        g.dropped(10, || "resolve", &[lit]);
        g.ran(&[depth, lit], &[image]);
        assert_eq!(g.feeding(&[image]), [&"resolve"]);
    }

    /// One pipeline used by two passes (a blit): the walk follows only the
    /// pass that writes what the image reads, not the other's inputs.
    #[test]
    fn a_shared_pipeline_s_passes_are_followed_apart() {
        let (bloom, image, unrelated, unread) = (1, 2, 3, 4);
        let mut g = Graph::default();
        g.ran(&[bloom], &[image]);
        g.ran(&[unread], &[unrelated]);
        g.dropped(9, || "unread's writer", &[unread]);
        assert!(g.feeding(&[image]).is_empty());
    }

    /// A null descriptor (`resource == 0`) names nothing: a pass binding
    /// one joins no other pass that did, so no unrelated dropped writer
    /// reaches the presented image through it (the reviewer's repro:
    /// `["unrelated_dropped"]` before the fix).
    #[test]
    fn a_null_view_feeds_nothing() {
        let (unrelated_tex, image) = (1, 2);
        let mut g = Graph::default();
        g.dropped(99, || "unrelated_dropped", &[unrelated_tex]);
        g.ran(&[unrelated_tex], &[0]);
        g.ran(&[0], &[image]);
        assert!(g.feeding(&[image]).is_empty());
        assert_eq!(g.ran.keys().copied().collect::<Vec<_>>(), [image]);
        assert_eq!(g.reads, [Vec::<usize>::new()], "the null read is not kept");
    }

    /// A pass that wrote nothing is not noted: 10,000 of them (a title
    /// that stops presenting while its pipelines are queued) leave no
    /// unreachable reads behind.
    #[test]
    fn a_pass_that_wrote_nothing_is_not_noted() {
        let mut g = Graph::<&str>::default();
        for _ in 0..10_000 {
            g.ran(&[7], &[]);
        }
        assert!(g.reads.is_empty() && g.notes == 0 && g.ran.is_empty());
    }

    /// A full frame takes no more notes.
    #[test]
    fn notes_are_capped() {
        let mut g = Graph::default();
        g.ran(&[2, 2], &[3]);
        assert_eq!(g.notes, 3);
        g.notes = NOTES_MAX;
        g.dropped(4, || "late", &[3]);
        g.ran(&[6], &[7]);
        assert!(g.dropped.is_empty() && !g.ran.contains_key(&7));
    }
}
