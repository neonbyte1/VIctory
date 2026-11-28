use super::{Bin, Cast, Fields, Inst, Lower, NULL, Operand, Reg, V, Why, pred};

impl Lower<'_> {
    /// Ceil or floor a binary64 value in its two 32-bit register words.
    pub(super) fn round_directed_f64(
        &mut self,
        inst: &Inst,
        guard: Option<V>,
        ceil: bool,
    ) -> Result<(), Why> {
        let Fields::Valu(v) = &inst.fields else {
            unreachable!()
        };
        if v.clamp || v.omod != 0 || v.sdwa.is_some() || v.dpp.is_some() {
            return Err("binary64 rounding modifiers");
        }
        let Some(Operand::Vector { reg, .. }) = v.dst else {
            return Err("a binary64 destination pair");
        };
        let dst_hi = reg
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
                let bits = if code == 248 {
                    0x3fc4_5f30_6dc9_c882 // Binary64 1/(2π).
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
        let c = self.cur;
        let m = &mut self.s.m;
        let i32t = self.s.t.i32;
        let zero = m.i32(0);
        let ones = m.i32(u32::MAX);
        let sign_mask = m.i32(0x8000_0000);
        let mag_mask = m.i32(0x7fff_ffff);
        if v.abs & 1 != 0 {
            hi = m.bin(c, Bin::And, hi, mag_mask);
        }
        if v.neg & 1 != 0 {
            hi = m.bin(c, Bin::Xor, hi, sign_mask);
        }
        let sign = m.bin(c, Bin::And, hi, sign_mask);
        let away = m.cmp(c, if ceil { pred::EQ } else { pred::NE }, sign, zero);

        let mag_hi = m.bin(c, Bin::And, hi, mag_mask);
        let k20 = m.i32(20);
        let exponent = m.bin(c, Bin::LShr, mag_hi, k20);
        let integral_exponent = m.i32(1023 + 52);
        let integral = m.cmp(c, pred::IUGE, exponent, integral_exponent);
        let n = m.bin(c, Bin::Sub, integral_exponent, exponent);
        let (k31, k32) = (m.i32(31), m.i32(32));
        let clears_low = m.cmp(c, pred::IUGE, n, k32);
        let clears_high_bits = m.cmp(c, pred::IUGT, n, k32);
        let shift = m.bin(c, Bin::And, n, k31);
        let keep = m.bin(c, Bin::Shl, ones, shift);
        let discard = m.bin(c, Bin::Xor, keep, ones);

        // Add ones in the discarded bits when rounding away from zero.
        // Then clearing those bits advances to the next integer only if
        // the fraction was nonzero.
        let low_bias = m.select(c, clears_low, ones, discard);
        let low_bias = m.select(c, away, low_bias, zero);
        let sum_lo = m.bin(c, Bin::Add, lo, low_bias);
        let carry = m.cmp(c, pred::IULT, sum_lo, lo);
        let carry = m.cast(c, Cast::ZExt, carry, i32t);
        let high_increments = m.bin(c, Bin::And, away, clears_high_bits);
        let high_bias = m.select(c, high_increments, discard, zero);
        let sum_hi = m.bin(c, Bin::Add, hi, high_bias);
        let sum_hi = m.bin(c, Bin::Add, sum_hi, carry);
        let round_lo = m.bin(c, Bin::And, sum_lo, keep);
        let round_lo = m.select(c, clears_low, zero, round_lo);
        let round_hi = m.bin(c, Bin::And, sum_hi, keep);
        let round_hi = m.select(c, clears_high_bits, round_hi, sum_hi);

        // Below one, the result is signed zero or signed one.
        let magnitude = m.bin(c, Bin::Or, mag_hi, lo);
        let nonzero = m.cmp(c, pred::NE, magnitude, zero);
        let to_one = m.bin(c, Bin::And, away, nonzero);
        let one_hi = m.i32(0x3ff0_0000);
        let small_hi = m.select(c, to_one, one_hi, zero);
        let small_hi = m.bin(c, Bin::Or, small_hi, sign);
        let bias = m.i32(1023);
        let below_one = m.cmp(c, pred::IULT, exponent, bias);
        let round_hi = m.select(c, below_one, small_hi, round_hi);

        let result_lo = m.select(c, integral, lo, round_lo);
        let result_hi = m.select(c, integral, hi, round_hi);
        self.write(inst, Reg::V(reg), result_lo, guard);
        self.write(inst, Reg::V(dst_hi), result_hi, guard);
        Ok(())
    }
}
