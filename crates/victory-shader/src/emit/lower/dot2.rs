//! `v_dot2_f32_f16` and `v_dot2c_f32_f16`: two f16 products and an f32
//! addend, summed with one rounding as RDNA2 does.

use super::{Lower, Why};
use crate::emit::bitcode::{Attr, Bin, V, pred};
use crate::gfx10::{Fields, Operand};
use crate::ir::{Inst, Reg};

impl Lower<'_> {
    pub(super) fn dot2_f32_f16(&mut self, inst: &Inst, guard: Option<V>) -> Result<(), Why> {
        let Fields::Valu(v) = &inst.fields else {
            unreachable!()
        };
        let name = inst.name();
        let c = self.cur;
        // D.f32 = S0.f16[0] * S1.f16[0] + S0.f16[1] * S1.f16[1] + S2.f32;
        // v_dot2c (VOP2) accumulates into its destination instead of
        // S2. Each f16 and each of their products is exact in f32,
        // and RDNA2 rounds the exact sum of the products and the
        // addend once (measured on gfx1036).
        let packed = name == "v_dot2_f32_f16";
        if v.omod != 0
            || v.sdwa.is_some()
            || packed && v.neg_hi >> 2 & 1 != 0
            || !packed && (v.neg != 0 || v.abs != 0 || v.opsel != 0)
        {
            return Err("modifiers on a 2-component f16 dot product");
        }
        let f32t = self.s.t.f32;
        let mut products = Vec::with_capacity(2);
        for high in [false, true] {
            let mut xy = Vec::with_capacity(2);
            for k in 0..2 {
                // VOP3P picks each half with op_sel/op_sel_hi; VOP2
                // takes the low half for the low product.
                let h = if packed {
                    self.packed_src(inst, k, high, true)?
                } else {
                    if matches!(v.src.get(k), Some(Operand::Const(_))) {
                        // shortcut: how VOP2 widens an inline constant to a
                        // pair of halves is not measured.
                        return Err("an inline constant of v_dot2c_f32_f16");
                    }
                    let x = self.src(inst, k)?;
                    let x = if high {
                        let k16 = self.s.m.i32(16);
                        self.s.m.bin(c, Bin::LShr, x, k16)
                    } else {
                        x
                    };
                    let m = self.s.m.i32(0xFFFF);
                    self.s.m.bin(c, Bin::And, x, m)
                };
                let mut x = self
                    .s
                    .op(c, "legacyF16ToF32", Attr::ReadNone, f32t, 131, &[h]);
                let neg = if high { v.neg_hi } else { v.neg };
                if packed && neg >> k & 1 != 0 {
                    let z = self.s.m.f32(0x8000_0000);
                    x = self.s.m.bin(c, Bin::Sub, z, x);
                }
                xy.push(x);
            }
            products.push(self.s.m.bin(c, Bin::Mul, xy[0], xy[1]));
        }
        let acc = if packed {
            // An f32 source: neg_lo bit 2 negates it.
            self.fsrc(inst, 2)?
        } else {
            let Some(Operand::Vector { reg, .. }) = v.dst else {
                return Err("an accumulator");
            };
            let acc = inst.read(Reg::V(reg)).ok_or("an accumulator")?;
            match self.floats.get(&acc) {
                Some(&x) => x,
                None => {
                    let x = self.i32_of(acc)?;
                    self.as_float(x)
                }
            }
        };
        // write_f applies the clamp.
        let r = self.sum3_rounded_once(products[0], products[1], acc);
        self.write_f(inst, r, guard)
    }

    /// Knuth's TwoSum: `(s, e)` with `s` the f32 sum of `a` and `b` and
    /// `e` its rounding error, so `s + e == a + b` exactly (finite inputs).
    fn two_sum(&mut self, a: V, b: V) -> (V, V) {
        let c = self.cur;
        let s = self.s.m.bin(c, Bin::Add, a, b);
        let bb = self.s.m.bin(c, Bin::Sub, s, a);
        let aa = self.s.m.bin(c, Bin::Sub, s, bb);
        let da = self.s.m.bin(c, Bin::Sub, a, aa);
        let db = self.s.m.bin(c, Bin::Sub, b, bb);
        let e = self.s.m.bin(c, Bin::Add, da, db);
        (s, e)
    }

    /// `a + b + d` rounded to f32 once, as a fused sum: Boldo and
    /// Melquiond's correctly rounded sum of three ("Emulation of FMA and
    /// correctly rounded sums: proved algorithms using rounding to odd",
    /// IEEE TC 2008). The two TwoSum errors are added with rounding to
    /// odd, so the last addition rounds as the exact sum would. An
    /// infinite or NaN term gives the plain sum.
    /// shortcut: an f32 denormal along the way (the addend, a partial sum
    /// or the result) follows the host's denormal mode, not the console's,
    /// which reads it as zero ([`Self::ftz`]).
    fn sum3_rounded_once(&mut self, a: V, b: V, d: V) -> V {
        let c = self.cur;
        let (uh, ul) = self.two_sum(b, d);
        let (th, tl) = self.two_sum(a, uh);
        // v = tl + ul rounded to odd: where the sum is inexact and its last
        // bit is even, one step toward the exact value (the error's side).
        let (v, e) = self.two_sum(tl, ul);
        let (vi, ei) = (self.as_int(v), self.as_int(e));
        let one = self.s.m.i32(1);
        let lsb = self.s.m.bin(c, Bin::And, vi, one);
        let zero = self.s.m.i32(0);
        let even = self.s.m.cmp(c, pred::EQ, lsb, zero);
        let fzero = self.s.m.f32(0);
        let inexact = self.s.m.cmp(c, pred::UNE, e, fzero);
        let bump = self.s.m.bin(c, Bin::And, even, inexact);
        // Same sign: away from zero (+1 on the magnitude bits), else toward.
        let signs = self.s.m.bin(c, Bin::Xor, vi, ei);
        let apart = self.s.m.cmp(c, pred::SLT, signs, zero);
        let minus_one = self.s.m.i32(u32::MAX);
        let step = self.s.m.select(c, apart, minus_one, one);
        let odd = self.s.m.bin(c, Bin::Add, vi, step);
        let odd = self.as_float(odd);
        let v = self.s.m.select(c, bump, odd, v);
        // A zero tail leaves th as it is, -0 included (th + +0 would be +0).
        let z = self.s.m.bin(c, Bin::Add, th, v);
        let tail = self.s.m.cmp(c, pred::OEQ, v, fzero);
        let z = self.s.m.select(c, tail, th, z);
        // Where th is infinite or NaN the TwoSum errors are NaN, and th is
        // the sum (for a dot product of halves, whose products stay below
        // 2^32, that takes an infinite or NaN term).
        let ti = self.as_int(th);
        let abs = self.s.m.i32(0x7FFF_FFFF);
        let mag = self.s.m.bin(c, Bin::And, ti, abs);
        let inf = self.s.m.i32(0x7F80_0000);
        let finite = self.s.m.cmp(c, pred::IULT, mag, inf);
        self.s.m.select(c, finite, z, th)
    }
}
