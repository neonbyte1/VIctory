use super::{Bin, Cast, Lower, V, pred};

impl Lower<'_> {
    /// The portion lost to rounding in the f32 sum `a + b` (TwoSum).
    pub(super) fn f32_sum_error(&mut self, a: V, b: V, sum: V) -> V {
        let c = self.cur;
        let m = &mut self.s.m;
        let b_hi = m.bin(c, Bin::Sub, sum, a);
        let a_hi = m.bin(c, Bin::Sub, sum, b_hi);
        let a_lo = m.bin(c, Bin::Sub, a, a_hi);
        let b_lo = m.bin(c, Bin::Sub, b, b_hi);
        m.bin(c, Bin::Add, a_lo, b_lo)
    }

    /// Return an f32 intermediate suitable for `narrow16`. If f32 rounding
    /// moves the sum away from zero, step back one f32 value before truncating.
    /// Every f16 boundary sits on the f32 grid, so stepping across a boundary
    /// only corrects a sum that rounded onto it from the wrong side.
    pub(super) fn half_add_sub(&mut self, op: Bin, a: V, b: V) -> V {
        let c = self.cur;
        let sum = self.s.m.bin(c, op, a, b);
        if self.mode16 != Some(3) {
            return sum;
        }
        let addend = if op == Bin::Sub {
            let neg_zero = self.s.m.f32(0x8000_0000);
            self.s.m.bin(c, Bin::Sub, neg_zero, b)
        } else {
            b
        };
        let error = self.f32_sum_error(a, addend, sum);
        let sum_bits = self.as_int(sum);
        let error_bits = self.as_int(error);
        let m = &mut self.s.m;
        let sign_mask = m.i32(0x8000_0000);
        let sign = m.bin(c, Bin::And, sum_bits, sign_mask);
        let relative_error = m.bin(c, Bin::Xor, error_bits, sign);
        let relative_error = self.as_float(relative_error);
        let m = &mut self.s.m;
        let zero = m.f32(0);
        // A non-finite sum produces a NaN error, so the ordered comparison
        // also preserves the original infinity or NaN, including its sign.
        let inward = m.cmp(c, pred::OLT, relative_error, zero);
        let step = m.cast(c, Cast::ZExt, inward, self.s.t.i32);
        let adjusted = m.bin(c, Bin::Sub, sum_bits, step);
        self.as_float(adjusted)
    }
}
