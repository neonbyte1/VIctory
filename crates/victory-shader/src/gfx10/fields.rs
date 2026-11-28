//! Operand fields of GFX10 instructions, by encoding.
//!
//! Layouts follow AMD's RDNA ISA reference. Register ranges carry their size
//! in dwords, which depends on the opcode (read from its name) and, for lane
//! masks (VCC, EXEC, compare results), on the wave size.

use super::{Encoding, Instruction};

/// A source or destination of an instruction.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Operand {
    /// Scalar registers by operand code: 0-105 `s0`-`s105`, 106/107
    /// `vcc_lo`/`vcc_hi`, 108-123 `ttmp0`-`ttmp15`, 124 `m0`, 125 `null`,
    /// 126/127 `exec_lo`/`exec_hi`.
    Scalar {
        code: u8,
        dwords: u8,
    },
    Vector {
        reg: u8,
        dwords: u8,
    },
    /// Inline constant or read-only special source, by operand code
    /// (128-192: 0-64, 193-208: -1 to -16, 240-248: floats, 251-254: vccz,
    /// execz, scc, lds_direct).
    Const(u8),
    /// The 32-bit literal that follows the instruction.
    Literal(u32),
    /// Interpolation attribute and channel (0-3 = x-w); `high` selects the
    /// upper f16 half.
    Attr {
        index: u8,
        chan: u8,
        high: bool,
    },
}

/// Operand code of `null` (no register; reads as 0).
pub const NULL: u8 = 125;
const VCC: u8 = 106;
const EXEC: u8 = 126;

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Cache {
    pub glc: bool,
    pub slc: bool,
    pub dlc: bool,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Fields {
    /// SOP1, SOP2, SOPC, SOPK; `imm` is SOPK's 16-bit immediate.
    Sop {
        dst: Option<Operand>,
        src: Vec<Operand>,
        imm: Option<u16>,
    },
    /// The 16-bit immediate: branch offset in dwords, wait counts, message...
    Sopp {
        imm: u16,
    },
    Smem {
        data: Option<Operand>,
        base: Operand,
        /// Signed byte offset, added to `soffset`.
        offset: i32,
        soffset: Operand,
        cache: Cache,
    },
    Valu(Valu),
    Vintrp {
        dst: Operand,
        /// The i/j VGPR, or for `v_interp_mov_f32` the parameter (0 `p10`,
        /// 1 `p20`, 2 `p0`) as [`Operand::Const`].
        src: Operand,
        attr: Operand,
    },
    Ds {
        dst: Option<Operand>,
        addr: Option<Operand>,
        data: Vec<Operand>,
        offset0: u8,
        offset1: u8,
        gds: bool,
    },
    /// MUBUF, and MTBUF with `format` set.
    Buffer {
        /// Loaded, stored, or the atomic operand (and result with `glc`).
        data: Option<Operand>,
        addr: Operand,
        rsrc: Operand,
        soffset: Operand,
        offset: u16,
        offen: bool,
        idxen: bool,
        format: Option<u8>,
        lds: bool,
        tfe: bool,
        cache: Cache,
    },
    Image {
        data: Operand,
        /// One range, or one VGPR per address with the NSA form.
        addr: Vec<Operand>,
        rsrc: Operand,
        sampler: Option<Operand>,
        dmask: u8,
        dim: u8,
        unorm: bool,
        r128: bool,
        a16: bool,
        d16: bool,
        tfe: bool,
        lwe: bool,
        cache: Cache,
    },
    /// FLAT, GLOBAL and SCRATCH.
    Flat {
        dst: Option<Operand>,
        addr: Option<Operand>,
        data: Option<Operand>,
        saddr: Option<Operand>,
        offset: i16,
        lds: bool,
        cache: Cache,
    },
    Exp {
        target: u8,
        /// Per channel; `None` when disabled.
        src: [Option<Operand>; 4],
        compr: bool,
        done: bool,
        vm: bool,
    },
}

/// VOP1, VOP2, VOPC, VOP3 and VOP3P, with their SDWA and DPP forms.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Valu {
    /// A VGPR range, an SGPR (`v_readlane_b32`...) or a lane mask (compares).
    pub dst: Option<Operand>,
    /// Carry-out or the second result of VOP3B (`v_add_co_u32`, `v_div_scale_f32`...).
    pub carry: Option<Operand>,
    /// In operand order, including the implicit VCC of VOP2 `v_cndmask_b32`
    /// and carry-in, and the literal of `v_madmk`/`v_madak` forms.
    pub src: Vec<Operand>,
    /// Bit i applies to `src[i]`. For VOP3P, `neg` is the low half's.
    pub neg: u8,
    pub abs: u8,
    /// VOP3P: negate the high half (`v_fma_mix*`: absolute value).
    pub neg_hi: u8,
    /// VOP3: bit 3 is the destination. VOP3P with `opsel_hi`: the halves used.
    pub opsel: u8,
    pub opsel_hi: u8,
    pub clamp: bool,
    /// Output modifier: 1 ×2, 2 ×4, 3 ÷2.
    pub omod: u8,
    pub sdwa: Option<Sdwa>,
    pub dpp: Option<Dpp>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Sdwa {
    /// Selects: 0-3 byte, 4-5 word, 6 dword. Compares have no `dst_sel`.
    pub dst_sel: u8,
    /// 0 pad with zeros, 1 sign-extend, 2 preserve.
    pub dst_unused: u8,
    pub src_sel: [u8; 2],
    /// Bit i sign-extends `src[i]`.
    pub sext: u8,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Dpp {
    /// DPP16: `ctrl` is the 9-bit permutation (quad_perm, row shifts...).
    Row {
        ctrl: u16,
        row_mask: u8,
        bank_mask: u8,
        bound_ctrl: bool,
        fi: bool,
    },
    /// DPP8: eight 3-bit lane selects.
    Lanes { sel: u32, fi: bool },
}

impl Instruction {
    /// Decodes the operand fields. `wave64` sizes lane masks.
    pub fn fields(&self, wave64: bool) -> Fields {
        let name = self.name().unwrap_or("");
        let w = |i: usize| self.words().get(i).copied().unwrap_or(0);
        let (w0, w1) = (w(0), w(1));
        let bit = |word: u32, n: u32| word >> n & 1 != 0;
        let cache = |glc, slc, dlc| Cache { glc, slc, dlc };
        match self.encoding {
            Encoding::Sop2 | Encoding::Sop1 | Encoding::Sopc => {
                let (dst, src0, src1) = salu_dwords(name);
                let sdst = scalar((w0 >> 16 & 0x7F) as u8, dst);
                let ssrc = |shift: u32, dwords| source(w0 >> shift & 0xFF, dwords, w1);
                let (dst, src) = match self.encoding {
                    Encoding::Sop2 => (Some(sdst), vec![ssrc(0, src0), ssrc(8, src1)]),
                    Encoding::Sopc => (None, vec![ssrc(0, src0), ssrc(8, src1)]),
                    _ => match name {
                        "s_getpc_b64" => (Some(sdst), vec![]),
                        "s_setpc_b64" | "s_rfe_b64" => (None, vec![ssrc(0, src0)]),
                        _ => (Some(sdst), vec![ssrc(0, src0)]),
                    },
                };
                Fields::Sop {
                    dst,
                    src,
                    imm: None,
                }
            }
            Encoding::Sopk => {
                let sdst = scalar((w0 >> 16 & 0x7F) as u8, 1 + u8::from(name == "s_call_b64"));
                let (dst, src) = match name {
                    "s_setreg_imm32_b32" => (None, vec![Operand::Literal(w1)]),
                    "s_version" => (None, vec![]),
                    n if n.starts_with("s_cmpk_")
                        || n.starts_with("s_waitcnt_")
                        || n == "s_setreg_b32" =>
                    {
                        (None, vec![sdst])
                    }
                    _ => (Some(sdst), vec![]),
                };
                Fields::Sop {
                    dst,
                    src,
                    imm: Some(w0 as u16),
                }
            }
            Encoding::Sopp => Fields::Sopp { imm: w0 as u16 },
            Encoding::Smem => {
                let sdata = (w0 >> 6 & 0x7F) as u8;
                // s_atc_probe*: SDATA is a 7-bit immediate, not an SGPR. The
                // buffer form's resource is 4 dwords. GLC and DLC are not operands.
                let probe = matches!(name, "s_atc_probe" | "s_atc_probe_buffer");
                let data = if probe {
                    Some(Operand::Literal(u32::from(sdata)))
                } else {
                    match name {
                        "s_gl1_inv" | "s_dcache_inv" | "s_dcache_wb" => None,
                        n if n.starts_with("s_dcache_discard") => None,
                        "s_memtime" | "s_memrealtime" => Some(scalar(sdata, 2)),
                        _ => Some(scalar(sdata, data_dwords(name))),
                    }
                };
                let base =
                    2 + 2 * u8::from(name.starts_with("s_buffer_") || name == "s_atc_probe_buffer");
                Fields::Smem {
                    data,
                    base: scalar(((w0 & 0x3F) << 1) as u8, base),
                    offset: (w1 << 11) as i32 >> 11,
                    soffset: scalar((w1 >> 25) as u8, 1),
                    cache: if probe {
                        Cache::default()
                    } else {
                        cache(bit(w0, 16), false, bit(w0, 14))
                    },
                }
            }
            Encoding::Vop1 | Encoding::Vop2 | Encoding::Vopc | Encoding::Vop3 | Encoding::Vop3p => {
                Fields::Valu(self.valu(name, if wave64 { 2 } else { 1 }))
            }
            Encoding::Vintrp => {
                let vsrc = (w0 & 0xFF) as u8;
                Fields::Vintrp {
                    dst: vector(w0 >> 18, 1),
                    src: if name == "v_interp_mov_f32" {
                        Operand::Const(vsrc)
                    } else {
                        vector(vsrc.into(), 1)
                    },
                    attr: Operand::Attr {
                        index: (w0 >> 10 & 0x3F) as u8,
                        chan: (w0 >> 8 & 3) as u8,
                        high: false,
                    },
                }
            }
            Encoding::Ds => {
                let ty = data_dwords(name);
                let pair = ds_pair(name);
                let returns = name.contains("read")
                    || name.contains("_rtn")
                    || matches!(
                        name,
                        "ds_swizzle_b32"
                            | "ds_permute_b32"
                            | "ds_bpermute_b32"
                            | "ds_append"
                            | "ds_consume"
                            | "ds_ordered_count"
                    );
                let no_addr = name.starts_with("ds_gws")
                    || name.contains("addtid")
                    || matches!(name, "ds_nop" | "ds_append" | "ds_consume");
                let data = if name.contains("read")
                    || name.contains("src2")
                    || matches!(
                        name,
                        "ds_swizzle_b32"
                            | "ds_nop"
                            | "ds_append"
                            | "ds_consume"
                            | "ds_ordered_count"
                            | "ds_gws_sema_v"
                            | "ds_gws_sema_p"
                            | "ds_gws_sema_release_all"
                    ) {
                    0
                } else if pair
                    || name.contains("cmpst")
                    || name.contains("mskor")
                    || name == "ds_wrap_rtn_b32"
                {
                    // ds_wrap_rtn_b32: DATA0 is subtracted, DATA1 added.
                    2
                } else {
                    1
                };
                // The operand of `ds_gws_init`, `ds_gws_sema_br` and
                // `ds_gws_barrier` is in the address field (LLVM: "data0 is
                // encoded as addr").
                let first = if name.starts_with("ds_gws") { 0 } else { 8 };
                Fields::Ds {
                    dst: returns.then(|| vector(w1 >> 24, ty * (1 + u8::from(pair)))),
                    addr: (!no_addr).then(|| vector(w1 & 0xFF, 1)),
                    data: (0..data)
                        .map(|i| vector(w1 >> (first + 8 * i) & 0xFF, ty))
                        .collect(),
                    offset0: w0 as u8,
                    offset1: (w0 >> 8) as u8,
                    gds: bit(w0, 17),
                }
            }
            Encoding::Mubuf | Encoding::Mtbuf => {
                let (offen, idxen, tfe) = (bit(w0, 12), bit(w0, 13), bit(w1, 23));
                let mtbuf = self.encoding == Encoding::Mtbuf;
                Fields::Buffer {
                    data: (!name.ends_with("_inv"))
                        .then(|| vector(w1 >> 8 & 0xFF, data_dwords(name) + u8::from(tfe))),
                    addr: vector(w1 & 0xFF, 1 + u8::from(offen && idxen)),
                    rsrc: scalar(((w1 >> 16 & 0x1F) << 2) as u8, 4),
                    soffset: source(w1 >> 24, 1, 0),
                    offset: (w0 & 0xFFF) as u16,
                    offen,
                    idxen,
                    format: mtbuf.then_some((w0 >> 19 & 0x7F) as u8),
                    lds: !mtbuf && bit(w0, 16),
                    tfe,
                    cache: cache(bit(w0, 14), bit(w1, 22), bit(w0, 15)),
                }
            }
            Encoding::Mimg => {
                let (dim, dmask, a16, d16) = (
                    (w0 >> 3 & 7) as u8,
                    (w0 >> 8 & 0xF) as u8,
                    bit(w1, 30),
                    bit(w1, 31),
                );
                let (r128, tfe, lwe) = (bit(w0, 15), bit(w0, 16), bit(w0, 17));
                let mut data = if name.starts_with("image_gather4") {
                    4
                } else {
                    dmask.count_ones().max(1) as u8
                };
                if d16 {
                    data = data.div_ceil(2);
                }
                let count = image_addr_dwords(name, dim, a16);
                let addr = if w0 >> 1 & 3 == 0 {
                    vec![vector(w1 & 0xFF, count)]
                } else {
                    // NSA: one VGPR per address, the first in dword 1, the rest
                    // in the extra dwords.
                    (0..usize::from(count))
                        .map(|i| match i {
                            0 => vector(w1 & 0xFF, 1),
                            i => vector(w(2 + (i - 1) / 4) >> (8 * ((i - 1) % 4)) & 0xFF, 1),
                        })
                        .collect()
                };
                let sampled = ["image_sample", "image_gather4", "image_get_lod"]
                    .iter()
                    .any(|p| name.starts_with(p));
                Fields::Image {
                    data: vector(w1 >> 8 & 0xFF, data + u8::from(tfe || lwe)),
                    addr,
                    rsrc: scalar(((w1 >> 16 & 0x1F) << 2) as u8, if r128 { 4 } else { 8 }),
                    sampler: sampled.then(|| scalar(((w1 >> 21 & 0x1F) << 2) as u8, 4)),
                    dmask,
                    dim,
                    unorm: bit(w0, 12),
                    r128,
                    a16,
                    d16,
                    tfe,
                    lwe,
                    cache: cache(bit(w0, 13), bit(w0, 25), bit(w0, 7)),
                }
            }
            Encoding::Flat | Encoding::Global | Encoding::Scratch => {
                let saddr = (w1 >> 16 & 0x7F) as u8;
                // GFX10.3 scratch with neither a VGPR nor an SGPR address
                // (ST mode) holds EXEC_HI's code in the SGPR field: no
                // register is read (LLVM: `off, off`).
                let st = self.encoding == Encoding::Scratch && saddr == 0x7F;
                let glc = bit(w0, 16);
                let n = data_dwords(name);
                let (load, atomic) = (name.contains("_load_"), name.contains("_atomic_"));
                let addr = match self.encoding {
                    Encoding::Flat => Some(vector(w1 & 0xFF, 2)),
                    Encoding::Global => Some(vector(w1 & 0xFF, 1 + u8::from(saddr == NULL))),
                    // With an SGPR base, scratch takes no VGPR.
                    _ => (saddr == NULL).then(|| vector(w1 & 0xFF, 1)),
                };
                let ret = if atomic && name.contains("cmpswap") {
                    n / 2
                } else {
                    n
                };
                // A global or scratch load into LDS (`lds`) has no VGPR result.
                let to_lds = self.encoding != Encoding::Flat && bit(w0, 13);
                Fields::Flat {
                    dst: (load && !to_lds || atomic && glc).then(|| vector(w1 >> 24, ret)),
                    addr,
                    data: (!load).then(|| vector(w1 >> 8 & 0xFF, n)),
                    saddr: (self.encoding != Encoding::Flat).then(|| {
                        let code = if st { NULL } else { saddr };
                        scalar(code, 1 + u8::from(self.encoding == Encoding::Global))
                    }),
                    offset: match self.encoding {
                        // GFX10 FLAT offsets are 11-bit and unsigned: the
                        // hardware ignores bit 11.
                        Encoding::Flat => (w0 & 0x7FF) as i16,
                        _ => ((w0 << 4) as i16) >> 4,
                    },
                    lds: bit(w0, 13),
                    cache: cache(glc, bit(w0, 17), bit(w0, 12)),
                }
            }
            Encoding::Exp => {
                let compr = bit(w0, 10);
                Fields::Exp {
                    target: (w0 >> 4 & 0x3F) as u8,
                    src: std::array::from_fn(|i| {
                        let slot = if compr { i / 2 } else { i };
                        bit(w0, i as u32).then(|| vector(w1 >> (8 * slot) & 0xFF, 1))
                    }),
                    compr,
                    done: bit(w0, 11),
                    vm: bit(w0, 12),
                }
            }
        }
    }

    fn valu(&self, name: &str, lanes: u8) -> Valu {
        // No operands; the console compiler stores a shader hash in their fields.
        if matches!(name, "v_nop" | "v_pipeflush" | "v_clrexcp") {
            return Valu::default();
        }
        let w = |i: usize| self.words().get(i).copied().unwrap_or(0);
        let w0 = w(0);
        let (dst_n, src_n) = valu_dwords(name);
        let compare = match self.encoding {
            Encoding::Vopc => true,
            Encoding::Vop3 => self.opcode < 0x100,
            _ => false,
        };
        let vop3b = VOP3B.contains(&name);
        let carry_in = name.ends_with("_co_ci_u32") || name == "v_cndmask_b32";
        let mut v = Valu::default();
        match self.encoding {
            Encoding::Vop1 | Encoding::Vop2 | Encoding::Vopc => {
                let ext = w(1);
                let mut src0 = w0 & 0x1FF;
                let vsrc1 = 256 + (w0 >> 9 & 0xFF);
                let mut src1 = vsrc1;
                match src0 {
                    0xF9 => {
                        let sdwa = Sdwa {
                            dst_sel: (ext >> 8 & 7) as u8,
                            dst_unused: (ext >> 11 & 3) as u8,
                            src_sel: [(ext >> 16 & 7) as u8, (ext >> 24 & 7) as u8],
                            sext: (ext >> 19 & 1 | ext >> 26 & 2) as u8,
                        };
                        // The s0/s1 bits select an SGPR or constant instead of a VGPR.
                        src0 = (ext & 0xFF) | if ext >> 23 & 1 == 0 { 256 } else { 0 };
                        if ext >> 31 == 1 {
                            src1 = vsrc1 - 256;
                        }
                        v.neg = (ext >> 20 & 1 | ext >> 27 & 2) as u8;
                        v.abs = (ext >> 21 & 1 | ext >> 28 & 2) as u8;
                        if compare {
                            if ext >> 15 & 1 == 1 && !name.starts_with("v_cmpx") {
                                v.dst = Some(scalar((ext >> 8 & 0x7F) as u8, lanes));
                            }
                            v.sdwa = Some(Sdwa {
                                dst_sel: 6,
                                dst_unused: 0,
                                ..sdwa
                            });
                        } else {
                            v.clamp = ext >> 13 & 1 == 1;
                            v.omod = (ext >> 14 & 3) as u8;
                            v.sdwa = Some(sdwa);
                        }
                    }
                    0xFA => {
                        src0 = 256 + (ext & 0xFF);
                        v.neg = (ext >> 20 & 1 | ext >> 21 & 2) as u8;
                        v.abs = (ext >> 21 & 1 | ext >> 22 & 2) as u8;
                        v.dpp = Some(Dpp::Row {
                            ctrl: (ext >> 8 & 0x1FF) as u16,
                            row_mask: (ext >> 28) as u8,
                            bank_mask: (ext >> 24 & 0xF) as u8,
                            bound_ctrl: ext >> 19 & 1 == 1,
                            fi: ext >> 18 & 1 == 1,
                        });
                    }
                    0xE9 | 0xEA => {
                        v.dpp = Some(Dpp::Lanes {
                            sel: ext >> 8,
                            fi: src0 == 0xEA,
                        });
                        src0 = 256 + (ext & 0xFF);
                    }
                    _ => {}
                }
                let op0 = source(src0, src_n[0], ext);
                let op1 = source(src1, src_n[1], ext);
                match self.encoding {
                    Encoding::Vopc => {
                        let mask = if name.starts_with("v_cmpx") {
                            EXEC
                        } else {
                            VCC
                        };
                        v.dst = v.dst.or(Some(scalar(mask, lanes)));
                        v.src = vec![op0, op1];
                    }
                    Encoding::Vop1 => {
                        v.dst = Some(valu_dst(name, w0 >> 17 & 0xFF, dst_n));
                        v.src = vec![op0];
                    }
                    _ => {
                        v.dst = Some(vector(w0 >> 17 & 0xFF, dst_n));
                        v.src = match name {
                            n if n.ends_with("mk_f32") || n.ends_with("mk_f16") => {
                                vec![op0, Operand::Literal(ext), op1]
                            }
                            n if n.ends_with("ak_f32") || n.ends_with("ak_f16") => {
                                vec![op0, op1, Operand::Literal(ext)]
                            }
                            _ => vec![op0, op1],
                        };
                        if carry_in {
                            v.src.push(scalar(VCC, lanes));
                        }
                        if name.ends_with("_co_ci_u32") {
                            v.carry = Some(scalar(VCC, lanes));
                        }
                    }
                }
            }
            Encoding::Vop3 | Encoding::Vop3p => {
                let w1 = w(1);
                let field = |i: u32| w1 >> (9 * i) & 0x1FF;
                let count = match self.opcode {
                    _ if self.encoding == Encoding::Vop3p => {
                        if name.starts_with("v_pk_")
                            && !name.contains("_fma_")
                            && !name.contains("_mad_")
                        {
                            2
                        } else {
                            3
                        }
                    }
                    0x000..0x100 => 2,
                    0x100..0x140 => 2 + usize::from(carry_in),
                    0x180..0x200 => 1,
                    _ if VOP3_TWO_SOURCES.contains(&name) => 2,
                    _ => 3,
                };
                v.src = (0..count)
                    .map(|i| {
                        let code = field(i as u32);
                        if i == 0 && name.starts_with("v_interp_") {
                            Operand::Attr {
                                index: (code & 0x3F) as u8,
                                chan: (code >> 6 & 3) as u8,
                                high: code >> 8 & 1 == 1,
                            }
                        } else if i == 2 && carry_in {
                            source(code, lanes, w(2))
                        } else {
                            source(code, src_n[i], w(2))
                        }
                    })
                    .collect();
                v.clamp = w0 >> 15 & 1 == 1;
                v.neg = (w1 >> 29) as u8;
                if self.encoding == Encoding::Vop3p {
                    v.neg_hi = (w0 >> 8 & 7) as u8;
                    v.opsel = (w0 >> 11 & 7) as u8;
                    v.opsel_hi = (w1 >> 27 & 3 | w0 >> 12 & 4) as u8;
                } else {
                    v.omod = (w1 >> 27 & 3) as u8;
                    if vop3b {
                        v.carry = Some(scalar((w0 >> 8 & 0x7F) as u8, lanes));
                    } else {
                        v.abs = (w0 >> 8 & 7) as u8;
                        v.opsel = (w0 >> 11 & 0xF) as u8;
                    }
                }
                let vdst = w0 & 0xFF;
                v.dst = Some(if compare {
                    // GFX10+ v_cmpx_* in VOP3 writes EXEC and ignores w0[7:0].
                    let code = if name.starts_with("v_cmpx") {
                        EXEC
                    } else {
                        vdst as u8
                    };
                    scalar(code, lanes)
                } else {
                    valu_dst(name, vdst, dst_n)
                });
            }
            _ => unreachable!("not a VALU encoding"),
        }
        v
    }
}

/// VOP3-only opcodes that take two sources (the rest take three).
const VOP3_TWO_SOURCES: &[&str] = &[
    "v_add_f64",
    "v_mul_f64",
    "v_min_f64",
    "v_max_f64",
    "v_ldexp_f64",
    "v_mul_lo_u32",
    "v_mul_hi_u32",
    "v_mul_lo_i32",
    "v_mul_hi_i32",
    "v_trig_preop_f64",
    "v_lshlrev_b64",
    "v_lshrrev_b64",
    "v_ashrrev_i64",
    "v_add_nc_u16",
    "v_sub_nc_u16",
    "v_mul_lo_u16",
    "v_lshrrev_b16",
    "v_ashrrev_i16",
    "v_max_u16",
    "v_max_i16",
    "v_min_u16",
    "v_min_i16",
    "v_add_nc_i16",
    "v_sub_nc_i16",
    "v_add_co_u32",
    "v_sub_co_u32",
    "v_subrev_co_u32",
    "v_pack_b32_f16",
    "v_cvt_pknorm_i16_f16",
    "v_cvt_pknorm_u16_f16",
    "v_lshlrev_b16",
    "v_interp_p1ll_f16",
    "v_readlane_b32",
    "v_writelane_b32",
    "v_ldexp_f32",
    "v_bfm_b32",
    "v_bcnt_u32_b32",
    "v_mbcnt_lo_u32_b32",
    "v_mbcnt_hi_u32_b32",
    "v_cvt_pknorm_i16_f32",
    "v_cvt_pknorm_u16_f32",
    "v_cvt_pk_u16_u32",
    "v_cvt_pk_i16_i32",
    "v_sub_nc_i32",
    "v_add_nc_i32",
];

/// VOP3 opcodes whose bits 14:8 hold a scalar destination instead of `abs`/`opsel`.
const VOP3B: &[&str] = &[
    "v_add_co_u32",
    "v_sub_co_u32",
    "v_subrev_co_u32",
    "v_add_co_ci_u32",
    "v_sub_co_ci_u32",
    "v_subrev_co_ci_u32",
    "v_div_scale_f32",
    "v_div_scale_f64",
    "v_mad_u64_u32",
    "v_mad_i64_i32",
];

fn scalar(code: u8, dwords: u8) -> Operand {
    Operand::Scalar { code, dwords }
}

fn vector(reg: u32, dwords: u8) -> Operand {
    Operand::Vector {
        reg: reg as u8,
        dwords,
    }
}

/// A 9-bit source field (8-bit scalar fields are the same codes below 256).
fn source(code: u32, dwords: u8, literal: u32) -> Operand {
    match code {
        0..=127 => scalar(code as u8, dwords),
        255 => Operand::Literal(literal),
        256.. => vector(code - 256, dwords),
        _ => Operand::Const(code as u8),
    }
}

/// VALU destinations that are SGPRs.
fn valu_dst(name: &str, field: u32, dwords: u8) -> Operand {
    match name {
        "v_readlane_b32" | "v_readfirstlane_b32" => scalar(field as u8, 1),
        _ => vector(field, dwords),
    }
}

/// LDS opcodes on two addresses (`ds_read2_b32`...), with `offset0`/`offset1`.
pub(super) fn ds_pair(name: &str) -> bool {
    name.split('_').any(|t| {
        ["read2", "write2", "wrxchg2"]
            .iter()
            .any(|p| t.starts_with(p))
    })
}

/// Dwords of a type suffix such as `b64` or `f32`.
fn type_dwords(token: &str) -> u8 {
    match token {
        "b64" | "u64" | "i64" | "f64" => 2,
        _ => 1,
    }
}

/// (sdst, ssrc0, ssrc1) dwords of a scalar ALU opcode.
fn salu_dwords(name: &str) -> (u8, u8, u8) {
    match name {
        "s_lshl_b64" | "s_lshr_b64" | "s_ashr_i64" | "s_bfe_u64" | "s_bfe_i64"
        | "s_bitcmp0_b64" | "s_bitcmp1_b64" => (2, 2, 1),
        "s_bfm_b64" | "s_bitset0_b64" | "s_bitset1_b64" | "s_bitreplicate_b64_b32" => (2, 1, 1),
        "s_bcnt0_i32_b64" | "s_bcnt1_i32_b64" | "s_ff0_i32_b64" | "s_ff1_i32_b64"
        | "s_flbit_i32_b64" | "s_flbit_i32_i64" => (1, 2, 2),
        _ => {
            let n = type_dwords(name.rsplit('_').next().unwrap_or(""));
            (n, n, n)
        }
    }
}

/// (vdst, [src0, src1, src2]) dwords of a VALU opcode.
fn valu_dwords(name: &str) -> (u8, [u8; 3]) {
    match name {
        "v_lshlrev_b64" | "v_lshrrev_b64" | "v_ashrrev_i64" => (2, [1, 2, 2]),
        "v_ldexp_f64" | "v_trig_preop_f64" | "v_cmp_class_f64" | "v_cmpx_class_f64" => {
            (2, [2, 1, 2])
        }
        "v_mad_u64_u32" | "v_mad_i64_i32" => (2, [1, 1, 2]),
        "v_qsad_pk_u16_u8" | "v_mqsad_pk_u16_u8" => (2, [2, 1, 2]),
        "v_mqsad_u32_u8" => (4, [2, 1, 4]),
        "v_frexp_exp_i32_f64" => (1, [2; 3]),
        _ => {
            let mut parts = name.split('_');
            // `v_cvt_<dst type>_<src type>`, and nothing past it.
            match (
                parts.next(),
                parts.next(),
                parts.next(),
                parts.next(),
                parts.next(),
            ) {
                (Some(_), Some("cvt"), Some(dst), Some(src), None) => {
                    (type_dwords(dst), [type_dwords(src); 3])
                }
                _ => {
                    let n = type_dwords(name.rsplit('_').next().unwrap_or(""));
                    (n, [n; 3])
                }
            }
        }
    }
}

/// Dwords of loaded, stored or atomic data, from the opcode name.
fn data_dwords(name: &str) -> u8 {
    let mut n: u8 = 1;
    for token in name.split('_') {
        n = match token {
            "dwordx2" | "x2" | "xy" | "b64" | "u64" | "i64" | "f64" => 2,
            "dwordx3" | "xyz" | "b96" => 3,
            "dwordx4" | "xyzw" | "b128" => 4,
            "dwordx8" => 8,
            "dwordx16" => 16,
            _ => n,
        };
    }
    if name.contains("cmpswap") {
        n *= 2;
    }
    if name.contains("format_d16") {
        n = n.div_ceil(2);
    }
    n
}

/// Address dwords of an image opcode: offset, bias, compare, derivatives,
/// coordinates, then LOD/clamp/mip. `a16` packs two coordinate components per
/// dword; the `_g16` form packs the gradients instead, and `a16` alone leaves
/// them at one dword each (RDNA2 supports both, so they are independent).
fn image_addr_dwords(name: &str, dim: u8, a16: bool) -> u8 {
    // (coordinates, derivatives per direction) for 1D 2D 3D CUBE 1D_ARRAY
    // 2D_ARRAY 2D_MSAA 2D_MSAA_ARRAY.
    const DIMS: [(u8, u8); 8] = [
        (1, 1),
        (2, 2),
        (3, 3),
        (3, 2),
        (2, 1),
        (3, 2),
        (3, 2),
        (4, 2),
    ];
    let (coords, grads) = DIMS[usize::from(dim & 7)];
    match name {
        "image_bvh_intersect_ray" => return if a16 { 8 } else { 11 },
        "image_bvh64_intersect_ray" => return if a16 { 9 } else { 12 },
        "image_get_resinfo" => return 1,
        _ => {}
    }
    let (mut whole, mut derivs, mut packed): (u8, u8, u8) = (0, 0, coords);
    if let Some(rest) = name
        .strip_prefix("image_sample")
        .or_else(|| name.strip_prefix("image_gather4"))
    {
        for token in rest.split('_') {
            match token {
                "o" | "b" | "c" => whole += 1,
                "d" | "cd" => derivs = grads,
                "l" | "cl" => packed += 1,
                _ => {}
            }
        }
    } else if name.contains("_mip") {
        packed += 1;
    }
    // Two gradients per direction, and the packed forms round each direction's
    // pair up to a whole dword.
    let gradients = if name.contains("_g16") {
        2 * derivs.div_ceil(2)
    } else {
        2 * derivs
    };
    let coordinates = if a16 { packed.div_ceil(2) } else { packed };
    whole + gradients + coordinates
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gfx10::opcodes;

    fn table(encoding: Encoding) -> &'static [(u16, &'static str)] {
        match encoding {
            Encoding::Sop2 => opcodes::SOP2,
            Encoding::Sopk => opcodes::SOPK,
            Encoding::Sop1 => opcodes::SOP1,
            Encoding::Sopc => opcodes::SOPC,
            Encoding::Sopp => opcodes::SOPP,
            Encoding::Smem => opcodes::SMEM,
            Encoding::Vop2 => opcodes::VOP2,
            Encoding::Vop1 => opcodes::VOP1,
            Encoding::Vopc => opcodes::VOPC,
            Encoding::Vop3 => opcodes::VOP3,
            Encoding::Vop3p => opcodes::VOP3P,
            Encoding::Vintrp => opcodes::VINTRP,
            Encoding::Ds => opcodes::DS,
            Encoding::Mubuf => opcodes::MUBUF,
            Encoding::Mtbuf => opcodes::MTBUF,
            Encoding::Mimg => opcodes::MIMG,
            Encoding::Flat => opcodes::FLAT,
            Encoding::Global => opcodes::GLOBAL,
            Encoding::Scratch => opcodes::SCRATCH,
            Encoding::Exp => &[],
        }
    }

    fn opcode(encoding: Encoding, name: &str) -> u16 {
        table(encoding)
            .iter()
            .find(|(_, n)| *n == name)
            .unwrap_or_else(|| panic!("{name} is not {encoding:?}"))
            .0
    }

    fn inst_raw(encoding: Encoding, opcode: u16, words: &[u32]) -> Instruction {
        let mut dwords = [0u32; Instruction::MAX_WORDS];
        dwords[..words.len()].copy_from_slice(words);
        Instruction {
            offset: 0,
            encoding,
            opcode,
            name: crate::gfx10::name(encoding, opcode),
            words: dwords,
            len: words.len() as u8,
        }
    }

    fn inst(encoding: Encoding, name: &str, words: &[u32]) -> Instruction {
        inst_raw(encoding, opcode(encoding, name), words)
    }

    /// VOP1/VOP2/VOPC dw0: source 0, then the opcode (VOP1) or src1 (VOP2) and
    /// the destination.
    fn vop(vdst: u32, field: u32, src0: u32) -> u32 {
        src0 | field << 9 | vdst << 17
    }

    /// VOP3/VOP3P dw0: destination, then abs/neg_hi, opsel, clamp and opcode.
    fn dw0(opcode: u16, vdst: u32, hi: u32, sel: u32, clamp: bool) -> u32 {
        vdst | hi << 8 | sel << 11 | u32::from(clamp) << 15 | u32::from(opcode) << 16
    }

    /// VOP3/VOP3P dw1: three 9-bit sources, then omod/opsel_hi and neg.
    fn dw1(src: [u32; 3], hi: u32, neg: u32) -> u32 {
        src[0] | src[1] << 9 | src[2] << 18 | hi << 27 | neg << 29
    }

    fn sgpr(code: u8, dwords: u8) -> Operand {
        Operand::Scalar { code, dwords }
    }

    fn vgpr(reg: u8, dwords: u8) -> Operand {
        Operand::Vector { reg, dwords }
    }

    fn valu_raw(encoding: Encoding, opcode: u16, words: &[u32], wave64: bool) -> Valu {
        let Fields::Valu(v) = inst_raw(encoding, opcode, words).fields(wave64) else {
            panic!("not a VALU instruction");
        };
        v
    }

    fn valu(encoding: Encoding, name: &str, words: &[u32], wave64: bool) -> Valu {
        valu_raw(encoding, opcode(encoding, name), words, wave64)
    }

    #[test]
    fn source_codes() {
        assert_eq!(source(0, 1, 0), sgpr(0, 1));
        assert_eq!(source(127, 4, 0), sgpr(127, 4));
        // Inline integers 0-64, then -1 to -16.
        assert_eq!(source(128, 1, 0), Operand::Const(128));
        assert_eq!(source(192, 1, 0), Operand::Const(192));
        assert_eq!(source(193, 1, 0), Operand::Const(193));
        assert_eq!(source(208, 1, 0), Operand::Const(208));
        // Floats and the read-only special sources.
        assert_eq!(source(240, 1, 0), Operand::Const(240));
        assert_eq!(source(254, 1, 0), Operand::Const(254));
        // 255 takes the literal that follows; 256 and up are VGPRs.
        assert_eq!(source(255, 2, 0xDEAD_BEEF), Operand::Literal(0xDEAD_BEEF));
        assert_eq!(source(256, 1, 0), vgpr(0, 1));
        assert_eq!(source(511, 2, 0), vgpr(255, 2));
    }

    #[test]
    fn scalar_alu_widths() {
        assert_eq!(salu_dwords("s_mov_b32"), (1, 1, 1));
        assert_eq!(salu_dwords("s_mov_b64"), (2, 2, 2));
        assert_eq!(salu_dwords("s_and_b64"), (2, 2, 2));
        assert_eq!(salu_dwords("s_lshl_b64"), (2, 2, 1));
        assert_eq!(salu_dwords("s_bfm_b64"), (2, 1, 1));
        assert_eq!(salu_dwords("s_bitreplicate_b64_b32"), (2, 1, 1));
        assert_eq!(salu_dwords("s_bcnt0_i32_b64"), (1, 2, 2));
        assert_eq!(salu_dwords("s_flbit_i32_i64"), (1, 2, 2));
        assert_eq!(salu_dwords("s_cmp_eq_u64"), (2, 2, 2));
        assert_eq!(salu_dwords("s_sext_i32_i8"), (1, 1, 1));
    }

    #[test]
    fn vector_alu_widths() {
        assert_eq!(valu_dwords("v_add_f32"), (1, [1, 1, 1]));
        assert_eq!(valu_dwords("v_add_f64"), (2, [2, 2, 2]));
        assert_eq!(valu_dwords("v_lshlrev_b64"), (2, [1, 2, 2]));
        assert_eq!(valu_dwords("v_ldexp_f64"), (2, [2, 1, 2]));
        assert_eq!(valu_dwords("v_mad_u64_u32"), (2, [1, 1, 2]));
        assert_eq!(valu_dwords("v_mqsad_u32_u8"), (4, [2, 1, 4]));
        assert_eq!(valu_dwords("v_frexp_exp_i32_f64"), (1, [2, 2, 2]));
        assert_eq!(valu_dwords("v_cvt_f32_f16"), (1, [1, 1, 1]));
        assert_eq!(valu_dwords("v_cvt_f64_f32"), (2, [1, 1, 1]));
        assert_eq!(valu_dwords("v_cvt_f32_f64"), (1, [2, 2, 2]));
        assert_eq!(valu_dwords("v_cvt_pkrtz_f16_f32"), (1, [1, 1, 1]));
    }

    #[test]
    fn data_widths() {
        assert_eq!(data_dwords("buffer_load_ubyte"), 1);
        assert_eq!(data_dwords("buffer_load_dword"), 1);
        assert_eq!(data_dwords("buffer_load_dwordx2"), 2);
        assert_eq!(data_dwords("flat_load_dwordx3"), 3);
        assert_eq!(data_dwords("flat_load_dwordx4"), 4);
        assert_eq!(data_dwords("global_load_sbyte"), 1);
        assert_eq!(data_dwords("global_load_b64"), 2);
        assert_eq!(data_dwords("global_load_b96"), 3);
        assert_eq!(data_dwords("global_load_b128"), 4);
        assert_eq!(data_dwords("ds_write_b128"), 4);
        // A compare-and-swap moves both the comparison and the replacement.
        assert_eq!(data_dwords("buffer_atomic_cmpswap_x2"), 4);
        assert_eq!(data_dwords("s_atomic_cmpswap_x2"), 4);
        // Packed 16-bit textels halve the data.
        assert_eq!(data_dwords("tbuffer_load_format_xy"), 2);
        assert_eq!(data_dwords("tbuffer_load_format_d16_xyzw"), 2);
    }

    #[test]
    fn lds_pairs() {
        assert!(ds_pair("ds_read2_b32"));
        assert!(ds_pair("ds_write2_b64"));
        assert!(ds_pair("ds_wrxchg2st64_rtn_b32"));
        assert!(!ds_pair("ds_read_b64"));
        assert!(!ds_pair("ds_write_b128"));
    }

    #[test]
    fn image_address_dwords() {
        // (coordinates, derivatives per direction) for each dimension:
        // 1D 2D 3D CUBE 1D_ARRAY 2D_ARRAY 2D_MSAA 2D_MSAA_ARRAY.
        let cases: &[(&str, u8, bool, u8)] = &[
            ("image_sample", 1, false, 2),
            ("image_sample", 1, true, 1),
            ("image_sample", 0, true, 1),
            ("image_sample", 2, true, 2),
            // LOD, bias, compare and offsets are whole dwords; a16 leaves them
            // alone and packs only the coordinates.
            ("image_sample_l", 1, false, 3),
            ("image_sample_l", 1, true, 2),
            ("image_sample_l", 2, true, 2),
            ("image_sample_b", 1, false, 3),
            ("image_sample_b", 1, true, 2),
            ("image_sample_c", 1, true, 2),
            ("image_sample_c_l", 3, true, 3),
            ("image_sample_o", 2, true, 3),
            ("image_sample_lz", 1, false, 2),
            ("image_gather4", 3, false, 3),
            ("image_gather4", 3, true, 2),
            ("image_gather4_l", 1, true, 2),
            ("image_get_lod", 1, true, 1),
            // Explicit derivatives: 2 * components, packed only by g16.
            ("image_sample_d", 0, false, 3),
            ("image_sample_d", 1, false, 6),
            ("image_sample_d", 1, true, 5),
            ("image_sample_d", 2, false, 9),
            ("image_sample_d_g16", 1, false, 4),
            ("image_sample_d_g16", 1, true, 3),
            ("image_sample_d_g16", 2, false, 7),
            ("image_sample_d_g16", 2, true, 6),
            ("image_sample_c_d", 1, true, 6),
            ("image_sample_c_d_g16", 1, false, 5),
            ("image_sample_c_d_cl", 3, true, 7),
            // Coarse derivatives and their compare form.
            ("image_sample_cd", 1, true, 5),
            ("image_sample_c_cd", 1, true, 6),
            ("image_sample_cd_cl", 1, false, 7),
            ("image_sample_cd_cl", 1, true, 6),
            // Loads, stores, multi-sample and mip chains.
            ("image_load", 1, false, 2),
            ("image_load", 1, true, 1),
            ("image_load_mip", 1, true, 2),
            ("image_store_mip", 0, false, 2),
            ("image_store_mip", 0, true, 1),
            ("image_msaa_load", 6, false, 3),
            ("image_msaa_load", 6, true, 2),
            ("image_msaa_load", 7, true, 2),
            ("image_atomic_add", 1, true, 1),
            ("image_get_resinfo", 1, false, 1),
            ("image_get_resinfo", 1, true, 1),
            ("image_bvh_intersect_ray", 0, false, 11),
            ("image_bvh_intersect_ray", 0, true, 8),
            ("image_bvh64_intersect_ray", 0, false, 12),
            ("image_bvh64_intersect_ray", 0, true, 9),
        ];
        for &(name, dim, a16, expected) in cases {
            assert_eq!(
                image_addr_dwords(name, dim, a16),
                expected,
                "{name} dim {dim} a16 {a16}"
            );
        }
    }

    #[test]
    fn scalar_instructions() {
        // s_and_b64 s[2:3], exec, s[4:5]
        let Fields::Sop { dst, src, imm } =
            inst(Encoding::Sop2, "s_and_b64", &[(2 << 16) | (4 << 8) | 126]).fields(false)
        else {
            panic!("not SOP");
        };
        assert_eq!(dst, Some(sgpr(2, 2)));
        assert_eq!(src, [sgpr(126, 2), sgpr(4, 2)]);
        assert_eq!(imm, None);

        // s_add_u32 s0, s1, literal
        let Fields::Sop { dst, src, .. } =
            inst(Encoding::Sop2, "s_add_u32", &[255 << 8 | 1, 0xDEAD_BEEF]).fields(false)
        else {
            panic!("not SOP");
        };
        assert_eq!(dst, Some(sgpr(0, 1)));
        assert_eq!(src, [sgpr(1, 1), Operand::Literal(0xDEAD_BEEF)]);

        // s_getpc_b64 writes the pair; s_setpc_b64 only reads one.
        let Fields::Sop { dst, src, .. } =
            inst(Encoding::Sop1, "s_getpc_b64", &[4 << 16]).fields(false)
        else {
            panic!("not SOP");
        };
        assert_eq!(dst, Some(sgpr(4, 2)));
        assert!(src.is_empty());
        let Fields::Sop { dst, src, .. } = inst(Encoding::Sop1, "s_setpc_b64", &[4]).fields(false)
        else {
            panic!("not SOP");
        };
        assert_eq!(dst, None);
        assert_eq!(src, [sgpr(4, 2)]);

        let Fields::Sop { dst, src, .. } =
            inst(Encoding::Sopc, "s_cmp_eq_u64", &[3 << 8]).fields(false)
        else {
            panic!("not SOP");
        };
        assert!(dst.is_none());
        assert_eq!(src, [sgpr(0, 2), sgpr(3, 2)]);

        // SOPK: the destination field becomes a source for compares.
        let Fields::Sop { dst, src, imm } =
            inst(Encoding::Sopk, "s_cmpk_lg_u32", &[(5 << 16) | 0x1234]).fields(false)
        else {
            panic!("not SOP");
        };
        assert_eq!(dst, None);
        assert_eq!(src, [sgpr(5, 1)]);
        assert_eq!(imm, Some(0x1234));

        let Fields::Sop { dst, imm, .. } =
            inst(Encoding::Sopk, "s_call_b64", &[(8 << 16) | 4]).fields(false)
        else {
            panic!("not SOP");
        };
        assert_eq!(dst, Some(sgpr(8, 2)));
        assert_eq!(imm, Some(4));

        let Fields::Sop { dst, src, imm } =
            inst(Encoding::Sopk, "s_setreg_imm32_b32", &[0x10, 1]).fields(false)
        else {
            panic!("not SOP");
        };
        assert_eq!(dst, None);
        assert_eq!(src, [Operand::Literal(1)]);
        assert_eq!(imm, Some(0x10));

        let Fields::Sop { dst, .. } =
            inst(Encoding::Sopk, "s_movk_i32", &[(6 << 16) | 0xFFFF]).fields(false)
        else {
            panic!("not SOP");
        };
        assert_eq!(dst, Some(sgpr(6, 1)));

        let op = u32::from(opcode(Encoding::Sopp, "s_waitcnt"));
        let Fields::Sopp { imm } =
            inst(Encoding::Sopp, "s_waitcnt", &[(op << 16) | 0xFC7F]).fields(false)
        else {
            panic!("not SOPP");
        };
        assert_eq!(imm, 0xFC7F);
    }

    #[test]
    fn scalar_memory() {
        // s_load_dwordx2 s[4:5], s[8:9], s3 offset:-8 glc dlc
        let w0 = 4 | (4 << 6) | (1 << 14) | (1 << 16);
        let w1 = 0x1F_FFF8 | (3 << 25);
        let Fields::Smem {
            data,
            base,
            offset,
            soffset,
            cache,
        } = inst(Encoding::Smem, "s_load_dwordx2", &[w0, w1]).fields(false)
        else {
            panic!("not SMEM");
        };
        assert_eq!(data, Some(sgpr(4, 2)));
        assert_eq!(base, sgpr(8, 2));
        assert_eq!(offset, -8);
        assert_eq!(soffset, sgpr(3, 1));
        assert_eq!(
            cache,
            Cache {
                glc: true,
                slc: false,
                dlc: true
            }
        );

        // The offset is 21 bits, signed.
        let Fields::Smem { offset, .. } =
            inst(Encoding::Smem, "s_load_dword", &[0, 0x0F_FFFF]).fields(false)
        else {
            panic!("not SMEM");
        };
        assert_eq!(offset, 1_048_575);
        let Fields::Smem { offset, .. } =
            inst(Encoding::Smem, "s_load_dword", &[0, 0x100_000]).fields(false)
        else {
            panic!("not SMEM");
        };
        assert_eq!(offset, -1_048_576);

        // Buffer descriptors are four SGPRs; memtime returns a pair.
        let Fields::Smem { data, base, .. } =
            inst(Encoding::Smem, "s_buffer_load_dwordx4", &[1 | (20 << 6), 0]).fields(false)
        else {
            panic!("not SMEM");
        };
        assert_eq!(data, Some(sgpr(20, 4)));
        assert_eq!(base, sgpr(2, 4));
        let Fields::Smem { data, .. } =
            inst(Encoding::Smem, "s_memtime", &[12 << 6, 0]).fields(false)
        else {
            panic!("not SMEM");
        };
        assert_eq!(data, Some(sgpr(12, 2)));
        let Fields::Smem { data, .. } = inst(Encoding::Smem, "s_dcache_inv", &[0, 0]).fields(false)
        else {
            panic!("not SMEM");
        };
        assert_eq!(data, None);
    }

    #[test]
    fn vop1_sources_and_destinations() {
        let v = valu(Encoding::Vop1, "v_mov_b32", &[vop(3, 0, 256 + 5)], false);
        assert_eq!(v.dst, Some(vgpr(3, 1)));
        assert_eq!(v.src, [vgpr(5, 1)]);

        let v = valu(
            Encoding::Vop1,
            "v_readfirstlane_b32",
            &[vop(5, 0, 256 + 2)],
            false,
        );
        assert_eq!(v.dst, Some(sgpr(5, 1)));
        assert_eq!(v.src, [vgpr(2, 1)]);

        // No operands: the console compiler keeps a shader hash in the fields.
        let v = valu(Encoding::Vop1, "v_nop", &[0xFFFF_FFFF, 0xFFFF_FFFF], false);
        assert_eq!(v, Valu::default());
    }

    #[test]
    fn sdwa_fields() {
        // dst_sel 3, dst_unused 1, clamp, omod 2, src0_sel 1, sext 3, neg 1,
        // abs 2, src1_sel 4; src0 is an SGPR (S0), src1 a VGPR.
        let ext = 0x2C99_AB08;
        let v = valu(Encoding::Vop2, "v_add_f32", &[vop(1, 5, 0xF9), ext], false);
        assert_eq!(v.dst, Some(vgpr(1, 1)));
        assert_eq!(v.src, [sgpr(8, 1), vgpr(5, 1)]);
        let sdwa = v.sdwa.expect("SDWA");
        assert_eq!((sdwa.dst_sel, sdwa.dst_unused), (3, 1));
        assert_eq!(sdwa.src_sel, [1, 4]);
        assert_eq!(sdwa.sext, 3);
        assert_eq!((v.neg, v.abs), (1, 2));
        assert!(v.clamp);
        assert_eq!(v.omod, 2);

        // Bit 31 makes src1 an SGPR: the 8-bit field is the register.
        let v = valu(
            Encoding::Vop2,
            "v_add_f32",
            &[vop(1, 6, 0xF9), (1 << 31) | 5],
            false,
        );
        assert_eq!(v.src, [vgpr(5, 1), sgpr(6, 1)]);
    }

    #[test]
    fn dpp_permutations() {
        let ext = 4
            | (0x101 << 8)
            | (1 << 18)
            | (1 << 19)
            | (1 << 20)
            | (1 << 21)
            | (1 << 22)
            | (1 << 23)
            | (0xA << 24)
            | (5 << 28);
        let v = valu(Encoding::Vop2, "v_add_f32", &[vop(2, 6, 0xFA), ext], false);
        assert_eq!(v.src, [vgpr(4, 1), vgpr(6, 1)]);
        assert_eq!((v.neg, v.abs), (3, 3));
        assert_eq!(
            v.dpp,
            Some(Dpp::Row {
                ctrl: 0x101,
                row_mask: 5,
                bank_mask: 0xA,
                bound_ctrl: true,
                fi: true,
            })
        );

        let ext = 4 | (0x12_3456 << 8);
        let v = valu(Encoding::Vop2, "v_add_f32", &[vop(2, 6, 0xE9), ext], false);
        assert_eq!(
            v.dpp,
            Some(Dpp::Lanes {
                sel: 0x12_3456,
                fi: false
            })
        );
        let v = valu(Encoding::Vop2, "v_add_f32", &[vop(2, 6, 0xEA), ext], false);
        assert_eq!(
            v.dpp,
            Some(Dpp::Lanes {
                sel: 0x12_3456,
                fi: true
            })
        );
    }

    #[test]
    fn vop2_implicit_operands() {
        let v = valu(Encoding::Vop2, "v_cndmask_b32", &[vop(1, 2, 0)], false);
        assert_eq!(v.src, [sgpr(0, 1), vgpr(2, 1), sgpr(106, 1)]);
        assert_eq!(v.carry, None);
        let v = valu(Encoding::Vop2, "v_cndmask_b32", &[vop(1, 2, 0)], true);
        assert_eq!(v.src[2], sgpr(106, 2));

        let v = valu(
            Encoding::Vop2,
            "v_add_co_ci_u32",
            &[vop(3, 4, 256 + 1)],
            true,
        );
        assert_eq!(v.src, [vgpr(1, 1), vgpr(4, 1), sgpr(106, 2)]);
        assert_eq!(v.carry, Some(sgpr(106, 2)));

        // v_madmk takes the literal second, v_madak third.
        let v = valu(
            Encoding::Vop2,
            "v_madmk_f32",
            &[vop(1, 3, 256 + 2), 0x3F00_0000],
            false,
        );
        assert_eq!(
            v.src,
            [vgpr(2, 1), Operand::Literal(0x3F00_0000), vgpr(3, 1)]
        );
        let v = valu(
            Encoding::Vop2,
            "v_madak_f32",
            &[vop(1, 3, 256 + 2), 0x3F00_0000],
            false,
        );
        assert_eq!(
            v.src,
            [vgpr(2, 1), vgpr(3, 1), Operand::Literal(0x3F00_0000)]
        );
    }

    #[test]
    fn compares_write_lane_masks() {
        let words = [vop(0, 3, 240)];
        let v = valu(Encoding::Vopc, "v_cmp_gt_f32", &words, false);
        assert_eq!(v.dst, Some(sgpr(106, 1)));
        assert_eq!(v.src, [Operand::Const(240), vgpr(3, 1)]);
        let v = valu(Encoding::Vopc, "v_cmp_gt_f32", &words, true);
        assert_eq!(v.dst, Some(sgpr(106, 2)));
        let v = valu(Encoding::Vopc, "v_cmpx_gt_f32", &words, false);
        assert_eq!(v.dst, Some(sgpr(126, 1)));
    }

    #[test]
    fn vop3_modifiers() {
        let op = opcode(Encoding::Vop3, "v_add_f64");
        let v = valu(
            Encoding::Vop3,
            "v_add_f64",
            &[dw0(op, 2, 2, 0xA, true), dw1([0x101, 0x102, 0], 1, 3)],
            false,
        );
        assert_eq!(v.dst, Some(vgpr(2, 2)));
        assert_eq!(v.src, [vgpr(1, 2), vgpr(2, 2)]);
        assert_eq!((v.neg, v.abs, v.opsel), (3, 2, 0xA));
        assert!(v.clamp);
        assert_eq!(v.omod, 1);

        // A VOP3-encoded compare writes an SGPR.
        let op = opcode(Encoding::Vopc, "v_cmp_gt_f32");
        let v = valu_raw(
            Encoding::Vop3,
            op,
            &[dw0(op, 6, 0, 0, false), dw1([0x101, 0x102, 0], 0, 0)],
            true,
        );
        assert_eq!(v.dst, Some(sgpr(6, 2)));
        assert_eq!(v.src, [vgpr(1, 1), vgpr(2, 1)]);

        // VOP3B: bits 14:8 hold the carry-out SGPR instead of abs/opsel.
        let op = opcode(Encoding::Vop3, "v_div_scale_f32");
        let w0 = 1 | (6 << 8) | (u32::from(op) << 16);
        let v = valu(
            Encoding::Vop3,
            "v_div_scale_f32",
            &[w0, dw1([0x101, 0x102, 0x103], 0, 0)],
            true,
        );
        assert_eq!(v.carry, Some(sgpr(6, 2)));
        assert_eq!(v.src, [vgpr(1, 1), vgpr(2, 1), vgpr(3, 1)]);
        assert_eq!((v.abs, v.opsel), (0, 0));

        // The literal of a nine-bit source field is the third dword.
        let v = valu(
            Encoding::Vop3,
            "v_div_scale_f32",
            &[
                1 | (u32::from(op) << 16),
                dw1([0x101, 0x102, 255], 0, 0),
                0x4120_0000,
            ],
            false,
        );
        assert_eq!(v.src[2], Operand::Literal(0x4120_0000));

        // v_readlane writes an SGPR; v_interp reads an attribute in src0.
        let op = opcode(Encoding::Vop3, "v_readlane_b32");
        let v = valu(
            Encoding::Vop3,
            "v_readlane_b32",
            &[dw0(op, 5, 0, 0, false), dw1([0x102, 3, 0], 0, 0)],
            false,
        );
        assert_eq!(v.dst, Some(sgpr(5, 1)));
        assert_eq!(v.src, [vgpr(2, 1), sgpr(3, 1)]);

        let op = opcode(Encoding::Vop3, "v_interp_p1ll_f16");
        let code = 5 | (2 << 6) | (1 << 8);
        let v = valu(
            Encoding::Vop3,
            "v_interp_p1ll_f16",
            &[dw0(op, 1, 0, 0, false), dw1([code, 0x104, 0], 0, 0)],
            false,
        );
        assert_eq!(
            v.src,
            [
                Operand::Attr {
                    index: 5,
                    chan: 2,
                    high: true
                },
                vgpr(4, 1)
            ]
        );
    }

    #[test]
    fn vop3p_halves() {
        let op = opcode(Encoding::Vop3p, "v_pk_add_f16");
        let v = valu(
            Encoding::Vop3p,
            "v_pk_add_f16",
            &[
                dw0(op, 1, 7, 5, false) | (1 << 14),
                dw1([0x101, 0x102, 0], 2, 5),
            ],
            false,
        );
        assert_eq!(v.src, [vgpr(1, 1), vgpr(2, 1)]);
        assert_eq!((v.neg, v.neg_hi, v.opsel, v.opsel_hi), (5, 7, 5, 6));

        let op = opcode(Encoding::Vop3p, "v_pk_fma_f16");
        let v = valu(
            Encoding::Vop3p,
            "v_pk_fma_f16",
            &[dw0(op, 1, 0, 0, false), dw1([0x101, 0x102, 0x103], 0, 0)],
            false,
        );
        assert_eq!(v.src.len(), 3);

        let op = opcode(Encoding::Vop3p, "v_fma_mix_f32");
        let v = valu(
            Encoding::Vop3p,
            "v_fma_mix_f32",
            &[dw0(op, 1, 0, 0, false), dw1([0x101, 0x102, 0x103], 0, 0)],
            false,
        );
        assert_eq!(v.src.len(), 3);
    }

    #[test]
    fn interpolation() {
        let op = u32::from(opcode(Encoding::Vintrp, "v_interp_p1_f32"));
        let w0 = (0x32 << 26) | (op << 16) | (2 << 18) | (5 << 10) | (1 << 8) | 4;
        let Fields::Vintrp { dst, src, attr } =
            inst(Encoding::Vintrp, "v_interp_p1_f32", &[w0]).fields(false)
        else {
            panic!("not VINTRP");
        };
        assert_eq!(dst, vgpr(2, 1));
        assert_eq!(src, vgpr(4, 1));
        assert_eq!(
            attr,
            Operand::Attr {
                index: 5,
                chan: 1,
                high: false
            }
        );

        // The destination is the full 8-bit field at 25:18.
        let w0 = (0x32 << 26) | (op << 16) | (1 << 25) | (2 << 18);
        let Fields::Vintrp { dst, .. } =
            inst(Encoding::Vintrp, "v_interp_p1_f32", &[w0]).fields(false)
        else {
            panic!("not VINTRP");
        };
        assert_eq!(dst, vgpr(130, 1));

        // v_interp_mov's second field is a parameter, not a VGPR.
        let Fields::Vintrp { src, .. } = inst(
            Encoding::Vintrp,
            "v_interp_mov_f32",
            &[(0x32 << 26) | (1 << 8) | 1],
        )
        .fields(false) else {
            panic!("not VINTRP");
        };
        assert_eq!(src, Operand::Const(1));
    }

    #[test]
    fn lds_operands() {
        // ds_read2_b32 v[6:7], v1 offset0:2 offset1:3 gds
        let Fields::Ds {
            dst,
            addr,
            data,
            offset0,
            offset1,
            gds,
        } = inst(
            Encoding::Ds,
            "ds_read2_b32",
            &[2 | (3 << 8) | (1 << 17), (6 << 24) | 1],
        )
        .fields(false)
        else {
            panic!("not DS");
        };
        assert_eq!(dst, Some(vgpr(6, 2)));
        assert_eq!(addr, Some(vgpr(1, 1)));
        assert!(data.is_empty());
        assert_eq!((offset0, offset1, gds), (2, 3, true));

        // A 128-bit store is one four-dword range, not four fields.
        let Fields::Ds { dst, data, .. } =
            inst(Encoding::Ds, "ds_write_b128", &[0, (2 << 8) | 1]).fields(false)
        else {
            panic!("not DS");
        };
        assert!(dst.is_none());
        assert_eq!(data, [vgpr(2, 4)]);

        // ds_append returns a value but takes no address.
        let Fields::Ds {
            dst, addr, data, ..
        } = inst(Encoding::Ds, "ds_append", &[0, 3 << 24]).fields(false)
        else {
            panic!("not DS");
        };
        assert_eq!(dst, Some(vgpr(3, 1)));
        assert!(addr.is_none());
        assert!(data.is_empty());

        // GWS operations have neither an address nor data.
        let Fields::Ds {
            dst, addr, data, ..
        } = inst(Encoding::Ds, "ds_gws_sema_v", &[0, 0]).fields(false)
        else {
            panic!("not DS");
        };
        assert!(dst.is_none() && addr.is_none() && data.is_empty());

        let Fields::Ds {
            dst, addr, data, ..
        } = inst(
            Encoding::Ds,
            "ds_add_rtn_u32",
            &[0, (4 << 24) | (6 << 8) | 5],
        )
        .fields(false)
        else {
            panic!("not DS");
        };
        assert_eq!(dst, Some(vgpr(4, 1)));
        assert_eq!(addr, Some(vgpr(5, 1)));
        assert_eq!(data, [vgpr(6, 1)]);

        // Compare-and-swap and the two-address forms take both data fields.
        let Fields::Ds { data, .. } =
            inst(Encoding::Ds, "ds_cmpst_b32", &[0, (3 << 16) | (2 << 8) | 1]).fields(false)
        else {
            panic!("not DS");
        };
        assert_eq!(data, [vgpr(2, 1), vgpr(3, 1)]);
        let Fields::Ds { dst, data, .. } = inst(
            Encoding::Ds,
            "ds_write2_b64",
            &[0, (3 << 16) | (2 << 8) | 1],
        )
        .fields(false) else {
            panic!("not DS");
        };
        assert!(dst.is_none());
        assert_eq!(data, [vgpr(2, 2), vgpr(3, 2)]);
    }

    #[test]
    fn buffer_operands() {
        let Fields::Buffer {
            data,
            addr,
            rsrc,
            soffset,
            offset,
            offen,
            idxen,
            format,
            lds,
            tfe,
            cache,
        } = inst(
            Encoding::Mubuf,
            "buffer_load_dwordx4",
            &[0xE034_7010, 0x0102_0402],
        )
        .fields(false)
        else {
            panic!("not a buffer instruction");
        };
        assert_eq!(data, Some(vgpr(4, 4)));
        assert_eq!(addr, vgpr(2, 2));
        assert_eq!(rsrc, sgpr(8, 4));
        assert_eq!(soffset, sgpr(1, 1));
        assert_eq!(offset, 16);
        assert!(offen && idxen && !lds && !tfe);
        assert_eq!(format, None);
        assert_eq!(
            cache,
            Cache {
                glc: true,
                slc: false,
                dlc: false
            }
        );

        // TFE returns one more dword; lds loads into shared memory.
        let Fields::Buffer { data, tfe, .. } = inst(
            Encoding::Mubuf,
            "buffer_load_dword",
            &[0, (1 << 23) | (4 << 8) | 2],
        )
        .fields(false) else {
            panic!("not a buffer instruction");
        };
        assert!(tfe);
        assert_eq!(data, Some(vgpr(4, 2)));
        let Fields::Buffer { lds, .. } =
            inst(Encoding::Mubuf, "buffer_load_dword", &[1 << 16, 0]).fields(false)
        else {
            panic!("not a buffer instruction");
        };
        assert!(lds);

        // MTBUF's unified format; format_d16 halves 3d data.
        let Fields::Buffer {
            data,
            addr,
            rsrc,
            soffset,
            offset,
            format,
            lds,
            ..
        } = inst(
            Encoding::Mtbuf,
            "tbuffer_load_format_d16_xyzw",
            &[
                (17 << 19) | 16 | (1 << 12) | (1 << 13),
                (3 << 24) | (4 << 8) | 1 | (2 << 16),
            ],
        )
        .fields(false)
        else {
            panic!("not a buffer instruction");
        };
        assert_eq!(format, Some(17));
        assert_eq!(offset, 16);
        assert!(!lds);
        assert_eq!(data, Some(vgpr(4, 2)));
        assert_eq!(addr, vgpr(1, 2));
        assert_eq!(rsrc, sgpr(8, 4));
        assert_eq!(soffset, sgpr(3, 1));

        // Stores have no destination; the stored dwords are the data.
        let Fields::Buffer { data, .. } =
            inst(Encoding::Mubuf, "buffer_store_dwordx2", &[0, (4 << 8) | 1]).fields(false)
        else {
            panic!("not a buffer instruction");
        };
        assert_eq!(data, Some(vgpr(4, 2)));
    }

    #[test]
    fn image_operands() {
        // image_sample_lz v[8:9], [v1,v5], s[16:23], s[4:7] dmask:0x7 d16:
        // three channels pack into two dwords, and the NSA form lists the
        // addresses one VGPR each.
        let Fields::Image {
            data,
            addr,
            rsrc,
            sampler,
            dmask,
            dim,
            unorm,
            r128,
            a16,
            d16,
            tfe,
            lwe,
            cache,
        } = inst(
            Encoding::Mimg,
            "image_sample_lz",
            &[0xF09C_070A, 0x8024_0801, 0x0000_0005],
        )
        .fields(false)
        else {
            panic!("not an image instruction");
        };
        assert_eq!(data, vgpr(8, 2));
        assert_eq!(addr, [vgpr(1, 1), vgpr(5, 1)]);
        assert_eq!(rsrc, sgpr(16, 8));
        assert_eq!(sampler, Some(sgpr(4, 4)));
        assert_eq!((dmask, dim), (7, 1));
        assert!(!unorm && !r128 && !a16 && d16 && !tfe && !lwe);
        assert_eq!(cache, Cache::default());

        // Most flags at once: dim 3D has three coordinates and six gradients;
        // the coordinates pack with a16 while the gradients stay whole dwords.
        let w0 = 0xF000_0000
            | 0x0F00
            | 0x10
            | (1 << 25)
            | (1 << 16)
            | (1 << 15)
            | (1 << 13)
            | (1 << 12)
            | (1 << 7);
        let w1 = 0xC062_0401;
        let Fields::Image {
            data,
            addr,
            rsrc,
            sampler,
            dim,
            unorm,
            r128,
            a16,
            d16,
            tfe,
            lwe,
            cache,
            ..
        } = inst(Encoding::Mimg, "image_sample_d", &[w0, w1]).fields(false)
        else {
            panic!("not an image instruction");
        };
        assert_eq!(dim, 2);
        assert!(unorm && r128 && a16 && d16 && tfe && !lwe);
        assert_eq!(
            cache,
            Cache {
                glc: true,
                slc: true,
                dlc: true
            }
        );
        assert_eq!(rsrc, sgpr(8, 4));
        assert_eq!(sampler, Some(sgpr(12, 4)));
        assert_eq!(data, vgpr(4, 3));
        assert_eq!(addr, [vgpr(1, 8)]);

        // lwe appends a dword of its own.
        let w0 = 0xF000_0000 | 0x10 | (1 << 8) | (1 << 17);
        let Fields::Image { data, lwe, .. } =
            inst(Encoding::Mimg, "image_load", &[w0, 0x0002_0401]).fields(false)
        else {
            panic!("not an image instruction");
        };
        assert!(lwe);
        assert_eq!(data, vgpr(4, 2));

        // The NSA address dwords carry four VGPRs each.
        let w0 = 0xF000_0000 | 0x6 | 0x10 | 0x0F00;
        let Fields::Image { addr, .. } = inst(
            Encoding::Mimg,
            "image_sample_d",
            &[w0, 0x0002_0401, 0x0403_0201, 0x0807_0605, 0],
        )
        .fields(false) else {
            panic!("not an image instruction");
        };
        assert_eq!(
            addr,
            [
                vgpr(1, 1),
                vgpr(1, 1),
                vgpr(2, 1),
                vgpr(3, 1),
                vgpr(4, 1),
                vgpr(5, 1),
                vgpr(6, 1),
                vgpr(7, 1),
                vgpr(8, 1)
            ]
        );

        // Loads take no sampler.
        let Fields::Image { sampler, .. } =
            inst(Encoding::Mimg, "image_load", &[0, 0]).fields(false)
        else {
            panic!("not an image instruction");
        };
        assert_eq!(sampler, None);
    }

    #[test]
    fn flat_global_and_scratch() {
        // flat_load_dwordx3 v[4:6], v[2:3] inst_offset:2047, every cache flag
        let w0 = 0xDC00_0000 | 2047 | (1 << 12) | (1 << 13) | (1 << 16) | (1 << 17);
        let Fields::Flat {
            dst,
            addr,
            data,
            offset,
            lds,
            cache,
            ..
        } = inst(Encoding::Flat, "flat_load_dwordx3", &[w0, 0x0400_0402]).fields(false)
        else {
            panic!("not FLAT");
        };
        assert_eq!(dst, Some(vgpr(4, 3)));
        assert_eq!(addr, Some(vgpr(2, 2)));
        assert_eq!(data, None);
        assert_eq!(offset, 2047);
        assert!(lds);
        assert_eq!(
            cache,
            Cache {
                glc: true,
                slc: true,
                dlc: true
            }
        );

        // GLOBAL's offset is 12 bits, signed.
        let Fields::Flat { offset, .. } = inst(
            Encoding::Global,
            "global_load_dwordx2",
            &[0xDC00_0FFF, (4 << 8) | 2],
        )
        .fields(false) else {
            panic!("not GLOBAL");
        };
        assert_eq!(offset, -1);
        let Fields::Flat { offset, .. } =
            inst(Encoding::Global, "global_load_dwordx2", &[0xDC00_07FF, 0]).fields(false)
        else {
            panic!("not GLOBAL");
        };
        assert_eq!(offset, 2047);

        // A null saddr adds a VGPR to the address; scratch always has 32 bits.
        let Fields::Flat {
            addr, saddr, data, ..
        } = inst(
            Encoding::Global,
            "global_load_dwordx2",
            &[0, (125 << 16) | (4 << 8) | 2],
        )
        .fields(false)
        else {
            panic!("not GLOBAL");
        };
        assert_eq!(addr, Some(vgpr(2, 2)));
        assert_eq!(saddr, Some(sgpr(125, 2)));
        assert_eq!(data, None);

        let Fields::Flat { addr, saddr, .. } = inst(
            Encoding::Scratch,
            "scratch_load_dword",
            &[0, (125 << 16) | (4 << 8) | 1],
        )
        .fields(false) else {
            panic!("not SCRATCH");
        };
        assert_eq!(addr, Some(vgpr(1, 1)));
        assert_eq!(saddr, Some(sgpr(125, 1)));
        let Fields::Flat { addr, saddr, .. } = inst(
            Encoding::Scratch,
            "scratch_load_dword",
            &[0, (3 << 16) | (4 << 8) | 1],
        )
        .fields(false) else {
            panic!("not SCRATCH");
        };
        assert_eq!(addr, None);
        assert_eq!(saddr, Some(sgpr(3, 1)));

        // A store has no destination; a returning atomic keeps the old value.
        let Fields::Flat { dst, data, .. } =
            inst(Encoding::Global, "global_store_dword", &[0, (4 << 8) | 2]).fields(false)
        else {
            panic!("not GLOBAL");
        };
        assert!(dst.is_none());
        assert_eq!(data, Some(vgpr(4, 1)));
        let Fields::Flat { dst, data, .. } = inst(
            Encoding::Global,
            "global_atomic_cmpswap_x2",
            &[1 << 16, (6 << 24) | (4 << 8) | 1],
        )
        .fields(false) else {
            panic!("not GLOBAL");
        };
        assert_eq!(dst, Some(vgpr(6, 2)));
        assert_eq!(data, Some(vgpr(4, 4)));
    }

    #[test]
    fn exports() {
        let Fields::Exp {
            target,
            src,
            compr,
            done,
            vm,
        } = inst_raw(Encoding::Exp, 0, &[0xF800_1C0F, 0x0000_0302]).fields(false)
        else {
            panic!("not EXP");
        };
        assert_eq!(target, 0);
        assert_eq!(
            src,
            [
                Some(vgpr(2, 1)),
                Some(vgpr(2, 1)),
                Some(vgpr(3, 1)),
                Some(vgpr(3, 1))
            ]
        );
        assert!(compr && done && vm);

        let Fields::Exp {
            target,
            src,
            compr,
            done,
            vm,
        } = inst_raw(
            Encoding::Exp,
            0,
            &[
                0xF800_0000 | 0b0101 | (12 << 4),
                4 | (5 << 8) | (6 << 16) | (7 << 24),
            ],
        )
        .fields(false)
        else {
            panic!("not EXP");
        };
        assert_eq!(target, 12);
        assert_eq!(src, [Some(vgpr(4, 1)), None, Some(vgpr(6, 1)), None]);
        assert!(!compr && !done && !vm);
    }
}
