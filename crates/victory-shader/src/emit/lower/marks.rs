//! Which lanes of a wave set a write or read mark (module notes of `emit`,
//! "Write marks", "Read marks").

use super::super::bitcode::{Attr, Bin, V, pred};
use super::super::shader::flags;
use super::Lower;

// Test-only: lower the shipped per-lane marks, with no leader election,
// so a GPU test can compare both paths bit for bit. Thread-local because
// lowering runs on the test's thread and tests run concurrently.
#[cfg(test)]
thread_local! {
    pub(crate) static MARK_LEAD_OFF: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

/// Lower `f` with per-lane marks and restore the setting afterwards.
#[cfg(test)]
pub(crate) fn per_lane_marks<T>(f: impl FnOnce() -> T) -> T {
    MARK_LEAD_OFF.with(|off| {
        let old = off.replace(true);
        let result = f();
        off.set(old);
        result
    })
}

impl Lower<'_> {
    /// Whether this lane sets mark bit `n` (its load and atomic go to the
    /// bitmap in system memory): lanes with the first active lane's bit
    /// leave it to that lane. Most lanes of a wave mark the same page, so
    /// one load crosses the bus instead of 32. In pixel shaders the lane
    /// that leads is the first that is no helper: a helper's writes are
    /// dropped.
    pub(super) fn mark_lead(&mut self, n: V) -> Option<V> {
        #[cfg(test)]
        if MARK_LEAD_OFF.with(|off| off.get()) {
            return None;
        }
        self.s.flags |= flags::WAVE_OPS;
        let c = self.cur;
        let (i1, i32t) = (self.s.t.i1, self.s.t.i32);
        let (f, first) = if self.s.pixel {
            let helper = self
                .s
                .op(c, "isHelperLane.i1", Attr::ReadOnly, i1, 221, &[]);
            let lane = self
                .s
                .op(c, "waveGetLaneIndex", Attr::ReadNone, i32t, 111, &[]);
            // A helper's key is its lane with a high bit: the minimum is the
            // first non-helper, or (all helpers) a helper no lane matches,
            // still a lane to read at (writes there are dropped anyway).
            let flag = self.s.m.i32(1 << 30);
            let flagged = self.s.m.bin(c, Bin::Or, lane, flag);
            let key = self.s.m.select(c, helper, flagged, lane);
            // Operation 2, unsigned: the minimum.
            let (min, unsigned) = (self.s.m.i8(2), self.s.m.i8(1));
            let lead = self.s.op(
                c,
                "waveActiveOp.i32",
                Attr::NoUnwind,
                i32t,
                119,
                &[key, min, unsigned],
            );
            let low = self.s.m.i32((1 << 30) - 1);
            let at = self.s.m.bin(c, Bin::And, lead, low);
            let f = self
                .s
                .op(c, "waveReadLaneAt.i32", Attr::NoUnwind, i32t, 117, &[n, at]);
            (f, self.s.m.cmp(c, pred::EQ, lane, lead))
        } else {
            let f = self
                .s
                .op(c, "waveReadLaneFirst.i32", Attr::NoUnwind, i32t, 118, &[n]);
            (
                f,
                self.s
                    .op(c, "waveIsFirstLane", Attr::NoUnwind, i1, 110, &[]),
            )
        };
        let differ = self.s.m.cmp(c, pred::NE, n, f);
        Some(self.s.m.bin(c, Bin::Or, differ, first))
    }
}
