use super::{Attr, Bin, Fields, Inst, Lower, NULL, Operand, Reg, V, Why, pred};

impl Lower<'_> {
    /// A binary64 operand as two dwords, with source modifiers on its sign.
    fn f64_words(&mut self, inst: &Inst) -> Result<(V, V), Why> {
        let Fields::Valu(v) = &inst.fields else {
            unreachable!()
        };
        if v.sdwa.is_some() || v.dpp.is_some() {
            return Err("binary64 operand modifiers");
        }
        let source = *v.src.first().ok_or("a missing binary64 source")?;
        let (low, mut high) = match source {
            Operand::Vector { reg, .. } => {
                let next = reg.checked_add(1).ok_or("a missing high word")?;
                let next = inst.read(Reg::V(next)).ok_or("an unread high word")?;
                (self.operand(inst, &source)?, self.i32_of(next)?)
            }
            Operand::Scalar { code, .. } if code != NULL => {
                let next = code.checked_add(1).ok_or("a missing high word")?;
                let next = inst.read(Reg::S(next)).ok_or("an unread high word")?;
                (self.operand(inst, &source)?, self.i32_of(next)?)
            }
            Operand::Literal(high) => (self.s.m.i32(0), self.s.m.i32(high)),
            Operand::Const(code @ 240..=248) => {
                let bits = if code == 248 {
                    0x3fc4_5f30_6dc9_c882
                } else {
                    let value = crate::resource::inline(code).ok_or("an unknown inline float")?;
                    f64::from(f32::from_bits(value)).to_bits()
                };
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
        if v.abs & 1 != 0 {
            let mask = self.s.m.i32(0x7fff_ffff);
            high = self.s.m.bin(self.cur, Bin::And, high, mask);
        }
        if v.neg & 1 != 0 {
            let sign = self.s.m.i32(0x8000_0000);
            high = self.s.m.bin(self.cur, Bin::Xor, high, sign);
        }
        Ok((low, high))
    }

    /// Binary64 frexp on the encoded words. Keep subnormal inputs and NaN
    /// payloads entirely in the integer domain.
    pub(super) fn frexp64(&mut self, inst: &Inst, guard: Option<V>) -> Result<(), Why> {
        let Fields::Valu(v) = &inst.fields else {
            unreachable!()
        };
        if v.clamp || v.omod != 0 {
            return Err("binary64 frexp output modifiers");
        }
        let (low, high) = self.f64_words(inst)?;
        let c = self.cur;
        let zero = self.s.m.i32(0);
        let k20 = self.s.m.i32(20);
        let exp_mask = self.s.m.i32(0x7ff);
        let frac_mask = self.s.m.i32(0x000f_ffff);
        let wide = self.s.m.i32(32);
        let exp = self.s.m.bin(c, Bin::LShr, high, k20);
        let exp = self.s.m.bin(c, Bin::And, exp, exp_mask);
        let fraction = self.s.m.bin(c, Bin::And, high, frac_mask);
        let fraction_is_zero = self.s.m.cmp(c, pred::EQ, fraction, zero);
        let all_fraction = self.s.m.bin(c, Bin::Or, fraction, low);
        let all_fraction_is_zero = self.s.m.cmp(c, pred::EQ, all_fraction, zero);
        let exp_is_zero = self.s.m.cmp(c, pred::EQ, exp, zero);
        let exp_is_max = self.s.m.cmp(c, pred::EQ, exp, exp_mask);
        let is_zero = self.s.m.bin(c, Bin::And, exp_is_zero, all_fraction_is_zero);
        let special = self.s.m.bin(c, Bin::Or, is_zero, exp_is_max);
        let high_leading = self.s.op(
            c,
            "unaryBits.i32",
            Attr::ReadNone,
            self.s.t.i32,
            33,
            &[fraction],
        );
        let low_leading = self
            .s
            .op(c, "unaryBits.i32", Attr::ReadNone, self.s.t.i32, 33, &[low]);
        let low_leading = self.s.m.bin(c, Bin::Add, low_leading, wide);
        let leading = self
            .s
            .m
            .select(c, fraction_is_zero, low_leading, high_leading);
        // The leading 1 of a subnormal must move to bit 52. For a
        // nonzero fraction the shift is in 1..=52.
        let eleven = self.s.m.i32(11);
        let shift = self.s.m.bin(c, Bin::Sub, leading, eleven);

        if inst.name() == "v_frexp_exp_i32_f64" {
            let bias = self.s.m.i32(1022);
            let min_normal = self.s.m.i32((-1021_i32) as u32);
            let normal = self.s.m.bin(c, Bin::Sub, exp, bias);
            let subnormal = self.s.m.bin(c, Bin::Sub, min_normal, shift);
            let result = self.s.m.select(c, exp_is_zero, subnormal, normal);
            let result = self.s.m.select(c, special, zero, result);
            return self.write_i(inst, result, guard);
        }

        let Some(Operand::Vector { reg, .. }) = v.dst else {
            return Err("a binary64 result outside a VGPR");
        };
        let next = reg.checked_add(1).ok_or("a missing high result word")?;
        let m = &mut self.s.m;
        let word_mask = m.i32(31);
        let sign_mask = m.i32(0x8000_0000);
        let mantissa_exp = m.i32(0x3fe0_0000);
        // Mask both shifts to keep the dword operations defined at 32 bits.
        let short = m.cmp(c, pred::IULT, shift, wide);
        let word_shift = m.bin(c, Bin::And, shift, word_mask);
        let inverse_shift = m.bin(c, Bin::Sub, wide, word_shift);
        let inverse_shift = m.bin(c, Bin::And, inverse_shift, word_mask);
        let low_shifted = m.bin(c, Bin::Shl, low, word_shift);
        let carry = m.bin(c, Bin::LShr, low, inverse_shift);
        let high_shifted = m.bin(c, Bin::Shl, fraction, word_shift);
        let joined = m.bin(c, Bin::Or, high_shifted, carry);
        let subnormal_low = m.select(c, short, low_shifted, zero);
        let subnormal_high = m.select(c, short, joined, low_shifted);
        let subnormal_high = m.bin(c, Bin::And, subnormal_high, frac_mask);
        let result_low = m.select(c, exp_is_zero, subnormal_low, low);
        let result_high = m.select(c, exp_is_zero, subnormal_high, fraction);
        let sign = m.bin(c, Bin::And, high, sign_mask);
        let result_high = m.bin(c, Bin::Or, result_high, mantissa_exp);
        let result_high = m.bin(c, Bin::Or, result_high, sign);
        let result_low = m.select(c, special, low, result_low);
        let result_high = m.select(c, special, high, result_high);
        self.write(inst, Reg::V(reg), result_low, guard);
        self.write(inst, Reg::V(next), result_high, guard);
        Ok(())
    }
}
