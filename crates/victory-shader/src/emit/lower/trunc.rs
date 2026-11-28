use super::{Bin, Fields, Inst, Lower, NULL, Operand, Reg, V, Why, pred};

impl Lower<'_> {
    /// Truncate binary64 by clearing the fraction below the units bit.
    pub(super) fn trunc_f64(&mut self, inst: &Inst, guard: Option<V>) -> Result<(), Why> {
        let Fields::Valu(v) = &inst.fields else {
            unreachable!()
        };
        if v.clamp || v.omod != 0 || v.sdwa.is_some() || v.dpp.is_some() {
            return Err("binary64 truncation modifiers");
        }
        let Some(Operand::Vector { reg, .. }) = v.dst else {
            return Err("a binary64 destination pair");
        };
        let next = reg
            .checked_add(1)
            .ok_or("a missing destination high word")?;
        let source = *v.src.first().ok_or("a missing binary64 source")?;
        let pair = |l: &mut Self, high: Reg| -> Result<(V, V), Why> {
            let high = inst.read(high).ok_or("an unread high word")?;
            Ok((l.operand(inst, &source)?, l.i32_of(high)?))
        };
        let (lo, mut hi) = match source {
            Operand::Vector { reg, .. } => pair(
                self,
                Reg::V(reg.checked_add(1).ok_or("a missing high word")?),
            )?,
            Operand::Scalar { code, .. } if code != NULL => pair(
                self,
                Reg::S(code.checked_add(1).ok_or("a missing high word")?),
            )?,
            Operand::Literal(high) => (self.s.m.i32(0), self.s.m.i32(high)),
            Operand::Const(code @ 240..=248) => {
                let value = crate::resource::inline(code).ok_or("an unknown inline float")?;
                let bits = f64::from(f32::from_bits(value)).to_bits();
                (self.s.m.i32(bits as u32), self.s.m.i32((bits >> 32) as u32))
            }
            _ => {
                let high = if matches!(source, Operand::Const(193..=208)) {
                    u32::MAX
                } else {
                    0
                };
                (self.operand(inst, &source)?, self.s.m.i32(high))
            }
        };
        let c = self.cur;
        let sign = self.s.m.i32(0x8000_0000);
        if v.abs & 1 != 0 {
            let mask = self.s.m.i32(0x7fff_ffff);
            hi = self.s.m.bin(c, Bin::And, hi, mask);
        }
        if v.neg & 1 != 0 {
            hi = self.s.m.bin(c, Bin::Xor, hi, sign);
        }

        let k20 = self.s.m.i32(20);
        let exp_mask = self.s.m.i32(0x7ff);
        let exp = self.s.m.bin(c, Bin::LShr, hi, k20);
        let exp = self.s.m.bin(c, Bin::And, exp, exp_mask);
        let bias = self.s.m.i32(1023);
        let low_dword_boundary = self.s.m.i32(1023 + 20);
        let already_integral = self.s.m.i32(1023 + 52);
        let zero = self.s.m.i32(0);
        let ones = self.s.m.i32(u32::MAX);
        let k31 = self.s.m.i32(31);
        let keep_mask = |l: &mut Self, boundary| {
            let bits = l.s.m.bin(c, Bin::Sub, boundary, exp);
            let shift = l.s.m.bin(c, Bin::And, bits, k31);
            l.s.m.bin(c, Bin::Shl, ones, shift)
        };

        // Once all fractional bits are in the low dword, its mask applies
        // until there are none left. A shift of 32 clears the whole word.
        let low_mask = keep_mask(self, already_integral);
        let low_rounded = self.s.m.bin(c, Bin::And, lo, low_mask);
        let low_kept = self.s.m.cmp(c, pred::IUGT, exp, low_dword_boundary);
        let low_rounded = self.s.m.select(c, low_kept, low_rounded, zero);
        let no_fraction = self.s.m.cmp(c, pred::IUGE, exp, already_integral);
        let low_result = self.s.m.select(c, no_fraction, lo, low_rounded);

        let high_mask = keep_mask(self, low_dword_boundary);
        let high_rounded = self.s.m.bin(c, Bin::And, hi, high_mask);
        let high_needs_rounding = self.s.m.cmp(c, pred::IULT, exp, low_dword_boundary);
        let high_rounded = self.s.m.select(c, high_needs_rounding, high_rounded, hi);
        let signed_zero = self.s.m.bin(c, Bin::And, hi, sign);
        let below_one = self.s.m.cmp(c, pred::IULT, exp, bias);
        let high_result = self.s.m.select(c, below_one, signed_zero, high_rounded);

        self.write(inst, Reg::V(reg), low_result, guard);
        self.write(inst, Reg::V(next), high_result, guard);
        Ok(())
    }
}
