use super::{Bin, Cast, Fields, Inst, Lower, NULL, Operand, Reg, V, Why, pred};

impl Lower<'_> {
    /// Round a binary64 register pair without requiring host doubles or i64.
    pub(super) fn round_ne_f64(&mut self, inst: &Inst, guard: Option<V>) -> Result<(), Why> {
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
        let m = &mut self.s.m;
        let i32t = self.s.t.i32;
        let zero = m.i32(0);
        let one = m.i32(1);
        let k20 = m.i32(20);
        let k31 = m.i32(31);
        let k32 = m.i32(32);
        let ones = m.i32(u32::MAX);
        let sign_mask = m.i32(0x8000_0000);
        let mag_mask = m.i32(0x7fff_ffff);
        if v.abs & 1 != 0 {
            hi = m.bin(c, Bin::And, hi, mag_mask);
        }
        if v.neg & 1 != 0 {
            hi = m.bin(c, Bin::Xor, hi, sign_mask);
        }

        let mag_hi = m.bin(c, Bin::And, hi, mag_mask);
        let exponent = m.bin(c, Bin::LShr, mag_hi, k20);
        let integral_exponent = m.i32(1023 + 52);
        let integral = m.cmp(c, pred::IUGE, exponent, integral_exponent);
        let n = m.bin(c, Bin::Sub, integral_exponent, exponent);

        // Add 2^(n-1) - 1 plus the units bit, then clear the n fraction
        // bits. The units bit for exponent 1023 is implicit, and the low
        // bit of that exponent field is also one.
        let units_in_hi = m.cmp(c, pred::IUGE, n, k32);
        let fraction_in_hi = m.cmp(c, pred::IUGT, n, k32);
        let shift = m.bin(c, Bin::And, n, k31);
        let units_word = m.select(c, units_in_hi, hi, lo);
        let parity = m.bin(c, Bin::LShr, units_word, shift);
        let parity = m.bin(c, Bin::And, parity, one);

        let half_shift = m.bin(c, Bin::Sub, n, one);
        let half_shift = m.bin(c, Bin::And, half_shift, k31);
        let half = m.bin(c, Bin::Shl, one, half_shift);
        let half_less_one = m.bin(c, Bin::Sub, half, one);
        let lo_bias = m.bin(c, Bin::Add, half_less_one, parity);
        let sum_lo = m.bin(c, Bin::Add, lo, lo_bias);
        let carry = m.cmp(c, pred::IULT, sum_lo, lo);
        let carry = m.cast(c, Cast::ZExt, carry, i32t);

        // When the half bit is in the high word, the low word contributes
        // a carry to half - 1 if it was nonzero or the units bit was odd.
        let lo_nonzero = m.cmp(c, pred::NE, lo, zero);
        let sticky = m.cast(c, Cast::ZExt, lo_nonzero, i32t);
        let sticky_or_odd = m.bin(c, Bin::Or, sticky, parity);
        let hi_bias = m.bin(c, Bin::Add, half_less_one, sticky_or_odd);
        let hi_bias = m.select(c, fraction_in_hi, hi_bias, carry);
        let sum_hi = m.bin(c, Bin::Add, hi, hi_bias);

        let keep_mask = m.bin(c, Bin::Shl, ones, shift);
        let round_lo = m.bin(c, Bin::And, sum_lo, keep_mask);
        let round_lo = m.select(c, units_in_hi, zero, round_lo);
        let round_hi = m.bin(c, Bin::And, sum_hi, keep_mask);
        let round_hi = m.select(c, fraction_in_hi, round_hi, sum_hi);

        // Magnitudes below one round to signed zero through the exact
        // halfway point, and signed one above it.
        let half_hi = m.i32(0x3fe0_0000);
        let one_hi = m.i32(0x3ff0_0000);
        let above_half_hi = m.cmp(c, pred::IUGT, mag_hi, half_hi);
        let at_half_hi = m.cmp(c, pred::EQ, mag_hi, half_hi);
        let half_with_tail = m.bin(c, Bin::And, at_half_hi, lo_nonzero);
        let above_half = m.bin(c, Bin::Or, above_half_hi, half_with_tail);
        let small_hi = m.select(c, above_half, one_hi, zero);
        let sign = m.bin(c, Bin::And, hi, sign_mask);
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
