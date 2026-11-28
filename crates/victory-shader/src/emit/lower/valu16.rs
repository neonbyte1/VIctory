//! 16-bit integer VALU helpers extracted from `lower.rs` (`v_mad_{u32_u16,i32_i16}`
//! and `v_{min3,max3,med3}_{i16,u16}`).

use super::{Lower, Why};
use crate::emit::bitcode::{Bin, V, pred};
use crate::gfx10::{Fields, Operand};
use crate::ir::{Inst, Reg};

impl Lower<'_> {
    pub(super) fn mad_32_16(
        &mut self,
        inst: &Inst,
        name: &str,
        guard: Option<V>,
    ) -> Result<(), Why> {
        let Fields::Valu(v) = &inst.fields else {
            unreachable!()
        };
        if v.omod != 0 {
            return Err("integer output modifiers");
        }
        if v.abs != 0 || v.neg != 0 {
            return Err("integer operand modifiers");
        }
        let Some(Operand::Vector { reg, .. }) = v.dst else {
            return Err("a result without a register");
        };
        // 16-bit integer operands decode inline float constants as f32
        // bit patterns (not f16), then select low/high half via op_sel.
        let (mut a, mut bb) = (
            self.packed_src(inst, 0, false, false)?,
            self.packed_src(inst, 1, false, false)?,
        );
        let acc = self.src(inst, 2)?;
        let c = self.cur;
        let signed = name == "v_mad_i32_i16";
        if signed {
            let k16 = self.s.m.i32(16);
            let ta = self.s.m.bin(c, Bin::Shl, a, k16);
            a = self.s.m.bin(c, Bin::AShr, ta, k16);
            let tb = self.s.m.bin(c, Bin::Shl, bb, k16);
            bb = self.s.m.bin(c, Bin::AShr, tb, k16);
        }
        let prod = self.s.m.bin(c, Bin::Mul, a, bb);
        let sum = self.s.m.bin(c, Bin::Add, prod, acc);
        let r = if !v.clamp {
            sum
        } else if signed {
            let k0 = self.s.m.i32(0);
            let pa_xor = self.s.m.bin(c, Bin::Xor, prod, acc);
            let same_sign = self.s.m.cmp(c, pred::SGE, pa_xor, k0);
            let ps_xor = self.s.m.bin(c, Bin::Xor, prod, sum);
            let flipped = self.s.m.cmp(c, pred::SLT, ps_xor, k0);
            let ov = self.s.m.bin(c, Bin::And, same_sign, flipped);
            let pos = self.s.m.cmp(c, pred::SGE, prod, k0);
            let (imin, imax) = (self.s.m.i32(i32::MIN as u32), self.s.m.i32(i32::MAX as u32));
            let sat = self.s.m.select(c, pos, imax, imin);
            self.s.m.select(c, ov, sat, sum)
        } else {
            let ov = self.s.m.cmp(c, pred::IULT, sum, acc);
            let umax = self.s.m.i32(u32::MAX);
            self.s.m.select(c, ov, umax, sum)
        };
        self.write(inst, Reg::V(reg), r, guard);
        Ok(())
    }

    pub(super) fn min_max_med3_16(&mut self, inst: &Inst, name: &str) -> Result<V, Why> {
        // 16-bit integer operands decode inline float constants as f32
        // bit patterns (not f16), then select low/high half via op_sel.
        let (mut a, mut b, mut d) = (
            self.packed_src(inst, 0, false, false)?,
            self.packed_src(inst, 1, false, false)?,
            self.packed_src(inst, 2, false, false)?,
        );
        let c = self.cur;
        let k16 = self.s.m.i32(16);
        let sext = |l: &mut Self, x: V| {
            let t = l.s.m.bin(c, Bin::Shl, x, k16);
            l.s.m.bin(c, Bin::AShr, t, k16)
        };
        let signed = name.ends_with("_i16");
        if signed {
            a = sext(self, a);
            b = sext(self, b);
            d = sext(self, d);
        }
        let (gt, lt) = if signed {
            (pred::SGT, pred::SLT)
        } else {
            (pred::IUGT, pred::IULT)
        };
        let pick = |l: &mut Self, p: u8, x: V, y: V| {
            let t = l.s.m.cmp(c, p, x, y);
            l.s.m.select(c, t, x, y)
        };
        Ok(match name {
            "v_min3_i16" | "v_min3_u16" => {
                let t = pick(self, lt, a, b);
                pick(self, lt, t, d)
            }
            "v_max3_i16" | "v_max3_u16" => {
                let t = pick(self, gt, a, b);
                pick(self, gt, t, d)
            }
            _ => {
                let hi = pick(self, gt, a, b);
                let lo = pick(self, lt, a, b);
                let t = pick(self, lt, hi, d);
                pick(self, gt, t, lo)
            }
        })
    }
}
