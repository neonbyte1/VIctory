//! Double-precision VALU instructions.
//!
//! A double is a VGPR (or SGPR) pair on the console. DXIL builds one from
//! two dwords with `makeDouble` and takes it apart with `splitDouble`, and
//! has IEEE `fadd`, `fmul`, `fma`, `fmax` and `fmin` for it, which is what
//! these instructions need. Instructions with no DXIL double form (the
//! rounding, square root, reciprocal and `ldexp` families) stay refused.
//!
//! `fma` and the conversions between a double and an integer are D3D11.1
//! double extensions, which the shader flags must declare; `makeDouble`,
//! `splitDouble`, the float conversions and the other arithmetic are not.

use super::{Attr, Bin, Cast, Fields, Inst, Lower, NULL, Operand, Reg, V, Why, flags};

/// The inline float constants 240-248 as doubles: 0.5, -0.5, 1, -1, 2, -2,
/// 4, -4 and AMD's 1/(2 pi).
const INLINE_FLOATS: [u64; 9] = [
    0x3fe0_0000_0000_0000,
    0xbfe0_0000_0000_0000,
    0x3ff0_0000_0000_0000,
    0xbff0_0000_0000_0000,
    0x4000_0000_0000_0000,
    0xc000_0000_0000_0000,
    0x4010_0000_0000_0000,
    0xc010_0000_0000_0000,
    0x3fc4_5f30_6dc9_c882,
];

/// `dx.op.splitDouble`'s result: the low and the high dword.
const SPLIT_DOUBLE: &str = "dx.types.splitdouble";

impl Lower<'_> {
    /// The dwords of a register pair read as a double, each XORed with a
    /// zero the host compiler cannot prove (the lane index shifted right
    /// by 31).
    ///
    /// AMD's D3D12 compiler (RDNA2 and RDNA3) gets a double wrong whose
    /// dwords are uniform and sit at an odd offset of one load, such as
    /// dwords 1 and 2 of the root constants: it reads the low dword as
    /// both halves. Built from these values instead, the double comes out
    /// right there; WARP and NVIDIA give the same results either way. The
    /// cost is one XOR per dword.
    fn unfolded_pair(&mut self, (lo, hi): (V, V)) -> (V, V) {
        self.s.flags |= flags::WAVE_OPS;
        let e = self.s.entry;
        let i32t = self.s.t.i32;
        let lane = self
            .s
            .op(e, "waveGetLaneIndex", Attr::ReadOnly, i32t, 111, &[]);
        let k31 = self.s.m.i32(31);
        let zero = self.s.m.bin(e, Bin::LShr, lane, k31);
        let c = self.cur;
        (
            self.s.m.bin(c, Bin::Xor, lo, zero),
            self.s.m.bin(c, Bin::Xor, hi, zero),
        )
    }

    /// The low and high dwords of 64-bit float source `k`, with its abs and
    /// neg modifiers applied to the sign bit (abs first).
    fn fparts64(&mut self, inst: &Inst, k: usize) -> Result<(V, V), Why> {
        let Fields::Valu(v) = &inst.fields else {
            unreachable!()
        };
        if v.sdwa.is_some() || v.dpp.is_some() {
            return Err("64-bit operand modifiers");
        }
        let op = *v.src.get(k).ok_or("a missing operand")?;
        let (lo, mut hi) = match op {
            Operand::Vector { reg, .. } => {
                let pair = self.read_pair(inst, Reg::V(reg), Reg::V(reg + 1))?;
                self.unfolded_pair(pair)
            }
            Operand::Scalar { code, .. } if code != NULL => {
                let pair = self.read_pair(inst, Reg::S(code), Reg::S(code + 1))?;
                self.unfolded_pair(pair)
            }
            Operand::Const(code @ 240..=248) => {
                let bits = INLINE_FLOATS[usize::from(code - 240)];
                (self.s.m.i32(bits as u32), self.s.m.i32((bits >> 32) as u32))
            }
            // A double literal is the high dword; the low one is zero.
            Operand::Literal(bits) => (self.s.m.i32(0), self.s.m.i32(bits)),
            _ => {
                // An integer constant, sign-extended.
                let lo = self.operand(inst, &op)?;
                let hi = if matches!(op, Operand::Const(193..=208)) {
                    u32::MAX
                } else {
                    0
                };
                (lo, self.s.m.i32(hi))
            }
        };
        let block = self.cur;
        if v.abs >> k & 1 != 0 {
            let mask = self.s.m.i32(0x7fff_ffff);
            hi = self.s.m.bin(block, Bin::And, hi, mask);
        }
        if v.neg >> k & 1 != 0 {
            let sign = self.s.m.i32(0x8000_0000);
            hi = self.s.m.bin(block, Bin::Xor, hi, sign);
        }
        Ok((lo, hi))
    }

    /// Source `k` of a 64-bit float operation as a double.
    pub(super) fn fsrc64(&mut self, inst: &Inst, k: usize) -> Result<V, Why> {
        let (lo, hi) = self.fparts64(inst, k)?;
        Ok(self.make_double(lo, hi))
    }

    fn make_double(&mut self, lo: V, hi: V) -> V {
        self.s.flags |= flags::DOUBLES;
        let ty = self.s.m.double();
        self.s.op(
            self.cur,
            "makeDouble.f64",
            Attr::ReadNone,
            ty,
            101,
            &[lo, hi],
        )
    }

    fn split_double(&mut self, d: V) -> (V, V) {
        self.s.flags |= flags::DOUBLES;
        let i32t = self.s.t.i32;
        let pair = self.s.m.named_struct(SPLIT_DOUBLE, vec![i32t, i32t]);
        let c = self.cur;
        let r = self
            .s
            .op(c, "splitDouble.f64", Attr::ReadNone, pair, 102, &[d]);
        (self.s.m.extract(c, r, 0), self.s.m.extract(c, r, 1))
    }

    /// Writes a double to the VGPR pair `dst`.
    fn write_f64(&mut self, inst: &Inst, d: V, guard: Option<V>) -> Result<(), Why> {
        let Fields::Valu(v) = &inst.fields else {
            unreachable!()
        };
        if v.clamp || v.omod != 0 {
            return Err("64-bit result modifiers");
        }
        let Some(Operand::Vector { reg, .. }) = v.dst else {
            return Err("a 64-bit result outside a VGPR");
        };
        let (lo, hi) = self.split_double(d);
        self.write(inst, Reg::V(reg), lo, guard);
        self.write(inst, Reg::V(reg + 1), hi, guard);
        Ok(())
    }

    /// `v_max_f64` and `v_min_f64`. DXIL's `fmax`/`fmin` return the other
    /// operand for a NaN, as the hardware does; of two zeros, `max` is +0
    /// unless both are -0 and `min` is -0 unless neither is (as for f32).
    fn max_min64(&mut self, inst: &Inst, max: bool) -> Result<V, Why> {
        let c = self.cur;
        let ((lo_a, hi_a), (lo_b, hi_b)) = (self.fparts64(inst, 0)?, self.fparts64(inst, 1)?);
        let (a, b) = (self.make_double(lo_a, hi_a), self.make_double(lo_b, hi_b));
        let ty = self.s.m.double();
        let op = if max { 35 } else { 36 };
        let r = self.s.op(c, "binary.f64", Attr::ReadNone, ty, op, &[a, b]);
        let (lo, hi) = (
            self.s.m.bin(c, Bin::Or, lo_a, lo_b),
            self.s.m.bin(c, Bin::Or, hi_a, hi_b),
        );
        let magnitude = self.s.m.i32(0x7fff_ffff);
        let hi = self.s.m.bin(c, Bin::And, hi, magnitude);
        let bits = self.s.m.bin(c, Bin::Or, lo, hi);
        let zero = self.s.m.i32(0);
        let zeros = self.s.m.cmp(c, super::pred::EQ, bits, zero);
        let join = if max { Bin::And } else { Bin::Or };
        let sign = self.s.m.bin(c, join, hi_a, hi_b);
        let sign_bit = self.s.m.i32(0x8000_0000);
        let sign = self.s.m.bin(c, Bin::And, sign, sign_bit);
        let signed_zero = self.make_double(zero, sign);
        Ok(self.s.m.select(c, zeros, signed_zero, r))
    }

    /// The instructions of this module, or `None` for any other.
    pub(super) fn valu_f64(
        &mut self,
        inst: &Inst,
        name: &str,
        guard: Option<V>,
    ) -> Option<Result<(), Why>> {
        matches!(
            name,
            "v_add_f64"
                | "v_mul_f64"
                | "v_fma_f64"
                | "v_max_f64"
                | "v_min_f64"
                | "v_cvt_f64_f32"
                | "v_cvt_f64_i32"
                | "v_cvt_f64_u32"
                | "v_cvt_i32_f64"
                | "v_cvt_u32_f64"
        )
        .then(|| self.double_op(inst, name, guard))
    }

    fn double_op(&mut self, inst: &Inst, name: &str, guard: Option<V>) -> Result<(), Why> {
        let c = self.cur;
        let double = |l: &mut Self| l.s.m.double();
        match name {
            "v_add_f64" | "v_mul_f64" => {
                let (a, b) = (self.fsrc64(inst, 0)?, self.fsrc64(inst, 1)?);
                let op = if name == "v_add_f64" {
                    Bin::Add
                } else {
                    Bin::Mul
                };
                let r = self.s.m.bin(c, op, a, b);
                self.write_f64(inst, r, guard)
            }
            "v_fma_f64" => {
                let (a, b, d) = (
                    self.fsrc64(inst, 0)?,
                    self.fsrc64(inst, 1)?,
                    self.fsrc64(inst, 2)?,
                );
                let ty = double(self);
                self.s.flags |= flags::DOUBLE_EXTENSIONS;
                let r = self
                    .s
                    .op(c, "tertiary.f64", Attr::ReadNone, ty, 47, &[a, b, d]);
                self.write_f64(inst, r, guard)
            }
            "v_max_f64" | "v_min_f64" => {
                let r = self.max_min64(inst, name == "v_max_f64")?;
                self.write_f64(inst, r, guard)
            }
            "v_cvt_f64_f32" => {
                let a = self.fsrc(inst, 0)?;
                let ty = double(self);
                let r = self.s.m.cast(c, Cast::FpExt, a, ty);
                self.write_f64(inst, r, guard)
            }
            "v_cvt_f64_i32" | "v_cvt_f64_u32" => {
                let a = self.src(inst, 0)?;
                let ty = double(self);
                let op = if name == "v_cvt_f64_i32" {
                    Cast::SiToFp
                } else {
                    Cast::UiToFp
                };
                self.s.flags |= flags::DOUBLES | flags::DOUBLE_EXTENSIONS;
                let r = self.s.m.cast(c, op, a, ty);
                self.write_f64(inst, r, guard)
            }
            _ => {
                // v_cvt_i32_f64 and v_cvt_u32_f64: D3D saturates and turns
                // NaN into 0, as the hardware does.
                let a = self.fsrc64(inst, 0)?;
                let op = if name == "v_cvt_u32_f64" {
                    Cast::FpToUi
                } else {
                    Cast::FpToSi
                };
                self.s.flags |= flags::DOUBLE_EXTENSIONS;
                let r = self.s.m.cast(c, op, a, self.s.t.i32);
                let Fields::Valu(v) = &inst.fields else {
                    unreachable!()
                };
                if v.clamp || v.omod != 0 {
                    return Err("integer output modifiers");
                }
                let Some(Operand::Vector { reg, .. }) = v.dst else {
                    return Err("a result outside a VGPR");
                };
                self.write(inst, Reg::V(reg), r, guard);
                Ok(())
            }
        }
    }
}
