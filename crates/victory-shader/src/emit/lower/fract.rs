//! Binary64 fractional parts using two dwords.

use super::{Attr, Bin, Cast, Fields, Inst, Lower, NULL, Operand, Reg, Shader, V, Why, pred};
use crate::emit::bitcode::Module;

impl Lower<'_> {
    /// Extract the binary64 fraction without requiring native doubles or int64.
    pub(super) fn fract_f64(&mut self, inst: &Inst, guard: Option<V>) -> Result<(), Why> {
        let Fields::Valu(v) = &inst.fields else {
            unreachable!()
        };
        if v.clamp || v.omod != 0 || v.sdwa.is_some() || v.dpp.is_some() {
            return Err("binary64 fraction modifiers");
        }
        let Some(Operand::Vector { reg, .. }) = v.dst else {
            return Err("a binary64 destination pair");
        };
        let dst_high = reg
            .checked_add(1)
            .ok_or("a missing destination high word")?;
        let source = *v.src.first().ok_or("a missing binary64 source")?;
        let read_pair = |l: &mut Self, high: Reg| -> Result<(V, V), Why> {
            let high = inst.read(high).ok_or("an unread high word")?;
            Ok((l.operand(inst, &source)?, l.i32_of(high)?))
        };
        let (low, mut high) = match source {
            Operand::Vector { reg, .. } => read_pair(
                self,
                Reg::V(reg.checked_add(1).ok_or("a missing high word")?),
            )?,
            Operand::Scalar { code, .. } if code != NULL => read_pair(
                self,
                Reg::S(code.checked_add(1).ok_or("a missing high word")?),
            )?,
            Operand::Literal(high) => (self.s.m.i32(0), self.s.m.i32(high)),
            Operand::Const(code @ 240..=248) => {
                let bits = if code == 248 {
                    0x3fc4_5f30_6dc9_c882 // Binary64 1/(2 pi).
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
        let i32t = self.s.t.i32;
        let first_bit =
            |s: &mut Shader, op, word| s.op(c, "unaryBits.i32", Attr::ReadNone, i32t, op, &[word]);
        let m = &mut self.s.m;
        let zero = m.i32(0);
        let one = m.i32(1);
        let ones = m.i32(u32::MAX);
        let sign_mask = m.i32(0x8000_0000);
        let magnitude_mask = m.i32(0x7fff_ffff);
        let mantissa_mask = m.i32(0x000f_ffff);
        let hidden_bit = m.i32(0x0010_0000);
        let magnitude_high = m.bin(c, Bin::And, high, magnitude_mask);
        if v.abs & 1 != 0 {
            high = magnitude_high;
        }
        if v.neg & 1 != 0 {
            high = m.bin(c, Bin::Xor, high, sign_mask);
        }
        let sign = m.bin(c, Bin::And, high, sign_mask);
        let is_negative = m.cmp(c, pred::NE, sign, zero);
        let shift20 = m.i32(20);
        let exponent = m.bin(c, Bin::LShr, magnitude_high, shift20);
        let exponent_one = m.i32(1023);
        let exponent_half = m.i32(1022);
        let exponent_integral = m.i32(1075);
        let small = m.cmp(c, pred::IULT, exponent, exponent_half);
        let integral = m.cmp(c, pred::IUGE, exponent, exponent_integral);
        let mantissa_high = m.bin(c, Bin::And, high, mantissa_mask);
        let significand_high = m.bin(c, Bin::Or, mantissa_high, hidden_bit);

        // For |x| >= 1/2, the fractional field uses at most 53 bits.
        // For smaller negative values, round |x| to units of 2^-53 before
        // subtracting it from 1.0. The guard, sticky and parity select ties to even.
        let bits53 = m.i32(53);
        let tail_bits = m.bin(c, Bin::Sub, exponent_integral, exponent);
        let tail_bits = m.select(c, small, bits53, tail_bits);
        let tail_bits = m.select(c, integral, zero, tail_bits);
        let guard_shift = m.i32(1021);
        let guard_shift = m.bin(c, Bin::Sub, guard_shift, exponent);
        let guard_in_range = m.cmp(c, pred::IULT, guard_shift, bits53);
        let guard_shift = m.select(c, guard_in_range, guard_shift, bits53);
        let (guard_low, guard_high) =
            Self::shift_fraction_words(m, c, low, significand_high, guard_shift, false);
        let (scaled_low, scaled_high) =
            Self::shift_fraction_words(m, c, guard_low, guard_high, one, false);
        let guard_bit = m.bin(c, Bin::And, guard_low, one);
        let odd = m.bin(c, Bin::And, scaled_low, one);
        let low_nonzero = m.cmp(c, pred::NE, low, zero);
        let first_low = first_bit(&mut self.s, 32, low);
        let first_high = first_bit(&mut self.s, 32, significand_high);
        let m = &mut self.s.m;
        let bits32 = m.i32(32);
        let first_high = m.bin(c, Bin::Add, first_high, bits32);
        let first = m.select(c, low_nonzero, first_low, first_high);
        let sticky = m.cmp(c, pred::IULT, first, guard_shift);
        let sticky = m.cast(c, Cast::ZExt, sticky, i32t);
        let round_up = m.bin(c, Bin::Or, sticky, odd);
        let round_up = m.bin(c, Bin::And, round_up, guard_bit);
        let rounded_low = m.bin(c, Bin::Add, scaled_low, round_up);
        let carry = m.cmp(c, pred::IULT, rounded_low, scaled_low);
        let carry = m.cast(c, Cast::ZExt, carry, i32t);
        let rounded_high = m.bin(c, Bin::Add, scaled_high, carry);
        let significand_low = m.select(c, small, rounded_low, low);
        let significand_high = m.select(c, small, rounded_high, significand_high);

        // For negative x, complementing modulo the width of its fractional
        // field computes 1 - |fraction| and also keeps integer results at zero.
        let borrow = m.cmp(c, pred::NE, significand_low, zero);
        let borrow = m.cast(c, Cast::ZExt, borrow, i32t);
        let negated_low = m.bin(c, Bin::Sub, zero, significand_low);
        let negated_high = m.bin(c, Bin::Sub, zero, significand_high);
        let negated_high = m.bin(c, Bin::Sub, negated_high, borrow);
        let fraction_low = m.select(c, is_negative, negated_low, significand_low);
        let fraction_high = m.select(c, is_negative, negated_high, significand_high);
        let low_full = m.cmp(c, pred::IUGE, tail_bits, bits32);
        let bits31 = m.i32(31);
        let tail_shift = m.bin(c, Bin::And, tail_bits, bits31);
        let tail_mask = m.bin(c, Bin::Shl, ones, tail_shift);
        let tail_mask = m.bin(c, Bin::Xor, tail_mask, ones);
        let low_mask = m.select(c, low_full, ones, tail_mask);
        let high_mask = m.select(c, low_full, tail_mask, zero);
        let fraction_low = m.bin(c, Bin::And, fraction_low, low_mask);
        let fraction_high = m.bin(c, Bin::And, fraction_high, high_mask);

        // Normalize the fixed-point fraction and apply the binary64 exponent bias.
        let leading_low = first_bit(&mut self.s, 33, fraction_low);
        let leading_high = first_bit(&mut self.s, 33, fraction_high);
        let m = &mut self.s.m;
        let bits63 = m.i32(63);
        let msb_low = m.bin(c, Bin::Sub, bits31, leading_low);
        let msb_high = m.bin(c, Bin::Sub, bits63, leading_high);
        let high_nonzero = m.cmp(c, pred::NE, fraction_high, zero);
        let msb = m.select(c, high_nonzero, msb_high, msb_low);
        let bits52 = m.i32(52);
        let shift = m.bin(c, Bin::Sub, bits52, msb);
        let (normal_low, normal_high) =
            Self::shift_fraction_words(m, c, fraction_low, fraction_high, shift, true);
        let normal_high = m.bin(c, Bin::And, normal_high, mantissa_mask);
        let adjust = m.bin(c, Bin::Sub, msb, tail_bits);
        let normal_exp = m.bin(c, Bin::Add, exponent_one, adjust);
        let normal_exp = m.bin(c, Bin::Shl, normal_exp, shift20);
        let normal_high = m.bin(c, Bin::Or, normal_high, normal_exp);

        // Negative values too small to change the rounded 1.0 still return
        // the largest value below one. Exact integers and -0 return +0.
        let input_magnitude = m.bin(c, Bin::Or, magnitude_high, low);
        let input_nonzero = m.cmp(c, pred::NE, input_magnitude, zero);
        let small_negative = m.bin(c, Bin::And, small, is_negative);
        let round_to_limit = m.bin(c, Bin::And, small_negative, input_nonzero);
        let limit_high = m.i32(0x3fef_ffff);
        let empty_low = m.select(c, round_to_limit, ones, zero);
        let empty_high = m.select(c, round_to_limit, limit_high, zero);
        let fraction = m.bin(c, Bin::Or, fraction_low, fraction_high);
        let has_fraction = m.cmp(c, pred::NE, fraction, zero);
        let result_low = m.select(c, has_fraction, normal_low, empty_low);
        let result_high = m.select(c, has_fraction, normal_high, empty_high);

        // Positive inputs already below one include subnormals and need no
        // arithmetic. The IEEE nonfinite cases produce a quiet NaN.
        let below_one = m.cmp(c, pred::IULT, exponent, exponent_one);
        let is_positive = m.cmp(c, pred::EQ, sign, zero);
        let keep = m.bin(c, Bin::And, is_positive, below_one);
        let result_low = m.select(c, keep, low, result_low);
        let result_high = m.select(c, keep, magnitude_high, result_high);
        let inf_exp = m.i32(0x7ff);
        let nonfinite = m.cmp(c, pred::EQ, exponent, inf_exp);
        let quiet = m.i32(0x0008_0000);
        let nan_high = m.bin(c, Bin::Or, high, quiet);
        let result_low = m.select(c, nonfinite, low, result_low);
        let result_high = m.select(c, nonfinite, nan_high, result_high);

        self.write(inst, Reg::V(reg), result_low, guard);
        self.write(inst, Reg::V(dst_high), result_high, guard);
        Ok(())
    }

    /// Shift a pair of words without any IR shift count reaching 32.
    fn shift_fraction_words(
        m: &mut Module,
        c: usize,
        low: V,
        high: V,
        count: V,
        left: bool,
    ) -> (V, V) {
        let (near, far, shift, cross) = if left {
            (low, high, Bin::Shl, Bin::LShr)
        } else {
            (high, low, Bin::LShr, Bin::Shl)
        };
        let zero = m.i32(0);
        let bits31 = m.i32(31);
        let bits32 = m.i32(32);
        let large = m.cmp(c, pred::IUGE, count, bits32);
        let offset = m.bin(c, Bin::And, count, bits31);
        let shifted = m.bin(c, shift, near, offset);
        let from_far = m.bin(c, shift, far, offset);
        let carry_shift = m.bin(c, Bin::Sub, zero, offset);
        let carry_shift = m.bin(c, Bin::And, carry_shift, bits31);
        let carry = m.bin(c, cross, near, carry_shift);
        let needs_carry = m.cmp(c, pred::NE, offset, zero);
        let carry = m.select(c, needs_carry, carry, zero);
        let into_far = m.bin(c, Bin::Or, from_far, carry);
        let near = m.select(c, large, zero, shifted);
        let far = m.select(c, large, shifted, into_far);
        if left { (near, far) } else { (far, near) }
    }
}
