use super::super::bitcode::{Attr, Bin, V, pred};
use super::Lower;

impl<'a> Lower<'a> {
    /// `Op::AddF` prefix sum over active host lanes from cross-lane reads
    /// instead of `wavePrefixOp.f32`, which on AMD Radeon drivers corrupts a
    /// lane whose own term is NaN, infinity or large enough to round away
    /// earlier terms.
    pub(super) fn float_prefix_sum(&mut self, x: V, id: V, inclusive: bool) -> V {
        let c = self.cur;
        let i32t = self.s.t.i32;
        let me = self.lane_index();
        let t = self.s.m.bool(true);
        let four = self.s.m.named_struct("dx.types.fouri32", vec![i32t; 4]);
        let ballot = self
            .s
            .op(c, "waveActiveBallot", Attr::NoUnwind, four, 116, &[t]);
        let (words, rounds) = match self.s.wave_size {
            Some(32) => (1, 5),
            Some(64) => (2, 6),
            _ => (4, 7),
        };
        let (z, one, k5, k31, none) = (
            self.s.m.i32(0),
            self.s.m.i32(1),
            self.s.m.i32(5),
            self.s.m.i32(31),
            self.s.m.i32(u32::MAX),
        );
        let word_idx = self.s.m.bin(c, Bin::LShr, me, k5);
        let bit_idx = self.s.m.bin(c, Bin::And, me, k31);
        let low_mask = self.s.m.bin(c, Bin::Shl, one, bit_idx);
        let low_mask = self.s.m.bin(c, Bin::Sub, low_mask, one);
        let mut prev = none;
        for j in 0..words {
            let w = self.s.m.extract(c, ballot, j);
            let m = if words == 1 {
                low_mask
            } else {
                let kj = self.s.m.i32(j);
                let below = self.s.m.cmp(c, pred::IULT, kj, word_idx);
                let same = self.s.m.cmp(c, pred::EQ, kj, word_idx);
                let m = self.s.m.select(c, same, low_mask, z);
                self.s.m.select(c, below, none, m)
            };
            let mw = self.s.m.bin(c, Bin::And, w, m);
            let clz = self
                .s
                .op(c, "unaryBits.i32", Attr::ReadNone, i32t, 33, &[mw]);
            let mut cand = self.s.m.bin(c, Bin::Sub, k31, clz);
            if j > 0 {
                let base = self.s.m.i32(32 * j);
                cand = self.s.m.bin(c, Bin::Or, cand, base);
            }
            let has = self.s.m.cmp(c, pred::NE, mw, z);
            prev = self.s.m.select(c, has, cand, prev);
        }
        let (mut acc, mut p) = if inclusive {
            (self.as_float(x), prev)
        } else {
            let has = self.s.m.cmp(c, pred::NE, prev, none);
            let at = self.s.m.select(c, has, prev, me);
            let px = self.read_lane(x, at);
            let pp = self.read_lane(prev, at);
            let x0 = self.s.m.select(c, has, px, id);
            let has2 = self.s.m.cmp(c, pred::NE, pp, none);
            let p0 = self.s.m.select(c, has2, prev, none);
            (self.as_float(x0), p0)
        };
        for _ in 0..rounds {
            let has = self.s.m.cmp(c, pred::NE, p, none);
            let at = self.s.m.select(c, has, p, me);
            let ai = self.as_int(acc);
            let other = self.read_lane(ai, at);
            let other = self.as_float(other);
            let next = self.read_lane(p, at);
            let sum = self.s.m.bin(c, Bin::Add, other, acc);
            acc = self.s.m.select(c, has, sum, acc);
            p = self.s.m.select(c, has, next, none);
        }
        acc
    }

}
