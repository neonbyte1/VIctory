//! Buffer loads of a table in the program's code: a descriptor built
//! around `s_getpc_b64` whose records are the code's own words.

use super::{Lower, Why};
use crate::emit::bitcode::{Bin, V, pred};
use crate::ir::{Inst, Reg};

/// A code-table load as `buffer` decoded it: the descriptor's stride bits,
/// records and `OOB_SELECT`, the byte address parts, and the result.
pub(super) struct CodeLoad {
    pub bits: u32,
    pub records: u32,
    pub oob: u32,
    pub base: u32,
    pub off: V,
    pub extra: Option<V>,
    pub index: Option<V>,
    /// A byte or a half: its width and whether signed.
    pub sub: Option<(u32, bool)>,
    pub count: u32,
    pub reg: u8,
}

impl Lower<'_> {
    pub(super) fn code_table_buffer_load(
        &mut self,
        inst: &Inst,
        load: &CodeLoad,
        guard: Option<V>,
    ) -> Result<(), Why> {
        let CodeLoad {
            bits,
            records,
            oob,
            base,
            off,
            extra,
            index,
            sub,
            count,
            reg,
        } = *load;
        let c = self.cur;
        let mut at = self.s.m.i32(base);
        at = self.s.m.bin(c, Bin::Add, at, off);
        if let Some(x) = extra {
            at = self.s.m.bin(c, Bin::Add, at, x);
        }
        if let Some(ix) = index {
            let stride = self.s.m.i32(bits >> 16 & 0x3FFF);
            let row = self.s.m.bin(c, Bin::Mul, ix, stride);
            at = self.s.m.bin(c, Bin::Add, at, row);
        }
        let bytes = sub.map_or(4, |(width, _)| width / 8);
        for k in 0..count {
            // In range: a value, or a constant answer.
            let ok: Result<V, bool> = match (oob, index) {
                (1, Some(ix)) => {
                    let n = self.s.m.i32(records);
                    Ok(self.s.m.cmp(c, pred::IULT, ix, n))
                }
                (1 | 2, _) => Err(records != 0),
                // Multi-dword loads are checked per dword.
                _ => match records.checked_sub(4 * k + bytes) {
                    Some(last) => {
                        let last = self.s.m.i32(last);
                        Ok(self.s.m.cmp(c, pred::IULE, off, last))
                    }
                    None => Err(false),
                },
            };
            let zero = self.s.m.i32(0);
            let x = match (ok, sub) {
                (Err(false), _) => zero,
                (_, None) => {
                    let four = self.s.m.i32(4 * k);
                    let byte = self.s.m.bin(c, Bin::Add, at, four);
                    self.code_dword(byte)?
                }
                (_, Some((width, signed))) => {
                    let (not3, three) = (self.s.m.i32(!3), self.s.m.i32(3));
                    let aligned = self.s.m.bin(c, Bin::And, at, not3);
                    let w = self.code_dword(aligned)?;
                    let byte = self.s.m.bin(c, Bin::And, at, three);
                    let sh = self.s.m.bin(c, Bin::Shl, byte, three);
                    let mut x = self.s.m.bin(c, Bin::LShr, w, sh);
                    if width > 8 {
                        // A halfword at byte 3 takes its high byte from the
                        // next dword: (next << 1) << (31 - sh) is next << (32 - sh),
                        // and 0 at sh 0 with no shift by 32.
                        let four = self.s.m.i32(4);
                        let at_next = self.s.m.bin(c, Bin::Add, aligned, four);
                        let next = self.code_dword(at_next)?;
                        let one = self.s.m.i32(1);
                        let next = self.s.m.bin(c, Bin::Shl, next, one);
                        let k31 = self.s.m.i32(31);
                        let rest = self.s.m.bin(c, Bin::Sub, k31, sh);
                        let hi = self.s.m.bin(c, Bin::Shl, next, rest);
                        x = self.s.m.bin(c, Bin::Or, x, hi);
                    }
                    let left = self.s.m.i32(32 - width);
                    let x = self.s.m.bin(c, Bin::Shl, x, left);
                    let shr = if signed { Bin::AShr } else { Bin::LShr };
                    self.s.m.bin(c, shr, x, left)
                }
            };
            let x = match ok {
                Ok(ok) => self.s.m.select(c, ok, x, zero),
                Err(_) => x,
            };
            self.write(inst, Reg::V(reg + k as u8), x, guard);
        }
        Ok(())
    }
}
