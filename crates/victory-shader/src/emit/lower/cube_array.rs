use std::collections::{HashMap, HashSet};

use crate::gfx10::{Fields, NULL, Operand};
use crate::ir::{Inst, Program, Reg, Value};
use crate::resource::{Address, Resources, Role, Site, Source};

/// Cube image sites (`dim` 3) that view a `TextureCubeArray` rather than a
/// `TextureCube`. GFX10 MIMG has no array bit for cubes: the compiler encodes
/// the face operand as `8 * slice + face`, and only `TextureCubeArray` reads
/// channel 2 (the face count) of `image_get_resinfo`. Any cube site sharing a
/// descriptor with an array site takes the array kind as well.
pub(super) fn cube_arrays(p: &Program, res: &Resources) -> HashSet<Site> {
    let mut defs = HashMap::new();
    let mut phis = HashMap::new();
    let mut live = HashSet::new();
    for b in &p.blocks {
        for phi in &b.phis {
            phis.insert(phi.value, phi.args.as_slice());
        }
        for inst in &b.insts {
            for &(_, r) in &inst.reads {
                live.insert(r);
            }
            for &(_, w) in &inst.writes {
                defs.insert(w, inst);
            }
        }
    }
    let mut work: Vec<Value> = live.iter().copied().collect();
    while let Some(v) = work.pop() {
        if let Some(args) = phis.get(&v) {
            for &a in *args {
                if phis.contains_key(&a) && live.insert(a) {
                    work.push(a);
                }
            }
        }
    }

    struct Ctx<'a> {
        defs: HashMap<Value, &'a Inst>,
        phis: HashMap<Value, &'a [Value]>,
    }
    impl Ctx<'_> {
        fn reg_val(inst: &Inst, op: &Operand) -> Option<Value> {
            match *op {
                Operand::Vector { reg, .. } => inst.read(Reg::V(reg)),
                Operand::Scalar { code, .. } if code != NULL => inst.read(Reg::S(code)),
                _ => None,
            }
        }

        fn op_bits(&self, inst: &Inst, op: &Operand, depth: u8) -> Option<u32> {
            match *op {
                Operand::Literal(x) => Some(x),
                Operand::Const(c) => crate::resource::inline(c),
                Operand::Scalar { code: NULL, .. } => Some(0),
                _ => self.val_bits(Self::reg_val(inst, op)?, depth),
            }
        }

        fn val_bits(&self, v: Value, depth: u8) -> Option<u32> {
            if depth >= 16 {
                return None;
            }
            if let Some(args) = self.phis.get(&v) {
                let mut it = args
                    .iter()
                    .copied()
                    .filter(|&a| a != v)
                    .map(|a| self.val_bits(a, depth + 1));
                let first = it.next()??;
                return it.all(|x| x == Some(first)).then_some(first);
            }
            let inst = *self.defs.get(&v)?;
            match (&inst.fields, inst.name()) {
                (Fields::Sop { imm: Some(imm), .. }, "s_movk_i32") => {
                    Some(i32::from(*imm as i16) as u32)
                }
                (Fields::Sop { src, .. }, "s_mov_b32") => {
                    self.op_bits(inst, src.first()?, depth + 1)
                }
                (Fields::Valu(vf), "v_mov_b32")
                    if vf.sdwa.is_none() && vf.dpp.is_none() && !vf.clamp && vf.omod == 0 =>
                {
                    let mut x = self.op_bits(inst, vf.src.first()?, depth + 1)?;
                    if vf.abs & 1 != 0 {
                        x &= 0x7FFF_FFFF;
                    }
                    if vf.neg & 1 != 0 {
                        x ^= 0x8000_0000;
                    }
                    Some(x)
                }
                _ => None,
            }
        }

        fn fsrc_const(&self, inst: &Inst, k: usize) -> Option<f32> {
            let Fields::Valu(vf) = &inst.fields else {
                return None;
            };
            if vf.sdwa.is_some() || vf.dpp.is_some() {
                return None;
            }
            let mut x = self.op_bits(inst, vf.src.get(k)?, 0)?;
            if vf.abs >> k & 1 != 0 {
                x &= 0x7FFF_FFFF;
            }
            if vf.neg >> k & 1 != 0 {
                x ^= 0x8000_0000;
            }
            Some(f32::from_bits(x))
        }

        fn hsrc_const(&self, inst: &Inst, k: usize) -> Option<u32> {
            let Fields::Valu(vf) = &inst.fields else {
                return None;
            };
            if vf.sdwa.is_some() || vf.dpp.is_some() {
                return None;
            }
            let op = vf.src.get(k)?;
            let mut h = match *op {
                Operand::Const(c @ 240..=248) => [
                    0x3800, 0xB800, 0x3C00, 0xBC00, 0x4000, 0xC000, 0x4400, 0xC400, 0x3118,
                ][usize::from(c - 240)],
                _ => {
                    let x = self.op_bits(inst, op, 0)?;
                    let sh = if vf.opsel >> k & 1 != 0 { 16 } else { 0 };
                    (x >> sh) & 0xFFFF
                }
            };
            if vf.abs >> k & 1 != 0 {
                h &= 0x7FFF;
            }
            if vf.neg >> k & 1 != 0 {
                h ^= 0x8000;
            }
            Some(h)
        }

        fn is_array_face(
            &self,
            v: Value,
            half: Option<bool>,
            seen: &mut HashSet<(Value, Option<bool>)>,
        ) -> bool {
            if !seen.insert((v, half)) {
                return false;
            }
            if let Some(w) = self.val_bits(v, 0) {
                return match half {
                    None => {
                        let f = f32::from_bits(w);
                        f.is_finite() && f >= 8.0
                    }
                    Some(hi) => {
                        let h = (w >> if hi { 16 } else { 0 }) & 0xFFFF;
                        (0x4800..=0x7BFF).contains(&h)
                    }
                };
            }
            if let Some(args) = self.phis.get(&v) {
                return args.iter().any(|&a| self.is_array_face(a, half, seen));
            }
            let Some(inst) = self.defs.get(&v) else {
                return false;
            };
            let Fields::Valu(vf) = &inst.fields else {
                return false;
            };
            if vf.sdwa.is_some() || vf.dpp.is_some() {
                return false;
            }
            let name = inst.name();
            let src_face = |k: usize, h: Option<bool>, seen: &mut HashSet<_>| {
                vf.src
                    .get(k)
                    .and_then(|o| Self::reg_val(inst, o))
                    .is_some_and(|x| self.is_array_face(x, h, seen))
            };
            if name == "v_mov_b32" && !vf.clamp && vf.omod == 0 && vf.abs == 0 && vf.neg == 0 {
                return src_face(0, half, seen);
            }
            if name == "v_cndmask_b32" && vf.abs == 0 && vf.neg == 0 {
                return src_face(0, half, seen) || src_face(1, half, seen);
            }
            if let Some(hi) = half {
                let dst_hi = vf.opsel >> 3 & 1 != 0;
                return match name {
                    "v_cvt_f16_f32" if dst_hi == hi => src_face(0, None, seen),
                    "v_cvt_pkrtz_f16_f32" => src_face(usize::from(hi), None, seen),
                    "v_pack_b32_f16" if vf.abs == 0 && vf.neg == 0 => {
                        let k = usize::from(hi);
                        src_face(k, Some(vf.opsel >> k & 1 != 0), seen)
                    }
                    "v_fma_f16" | "v_mad_f16" | "v_fmac_f16" | "v_mac_f16" | "v_fmamk_f16"
                    | "v_fmaak_f16" | "v_mul_f16"
                        if dst_hi == hi && !vf.clamp =>
                    {
                        [0, 1]
                            .into_iter()
                            .any(|k| self.hsrc_const(inst, k) == Some(0x4800))
                    }
                    "v_add_f16" if dst_hi == hi && !vf.clamp => [0, 1].into_iter().any(|k| {
                        self.hsrc_const(inst, k)
                            .is_some_and(|h| (0x4800..=0x7BFF).contains(&h))
                            || src_face(k, Some(vf.opsel >> k & 1 != 0), seen)
                    }),
                    _ => false,
                };
            }
            let scale = [1.0f32, 2.0, 4.0, 0.5][usize::from(vf.omod & 3)];
            match name {
                "v_cvt_f32_f16" if !vf.clamp && vf.omod == 0 && vf.abs == 0 && vf.neg == 0 => {
                    src_face(0, Some(vf.opsel & 1 != 0), seen)
                }
                "v_mul_f32" | "v_mul_legacy_f32" if !vf.clamp => [0, 1]
                    .into_iter()
                    .any(|k| self.fsrc_const(inst, k).is_some_and(|x| x * scale == 8.0)),
                "v_add_f32" if !vf.clamp && vf.omod == 0 => [0, 1].into_iter().any(|k| {
                    self.fsrc_const(inst, k)
                        .is_some_and(|f| f.is_finite() && f >= 8.0)
                        || (vf.abs >> k & 1 == 0 && vf.neg >> k & 1 == 0 && src_face(k, None, seen))
                }),
                "v_fma_f32" | "v_fmaak_f32" | "v_fmamk_f32" | "v_fmac_f32" | "v_mad_f32"
                | "v_madak_f32" | "v_madmk_f32" | "v_mac_f32" | "v_mad_legacy_f32"
                | "v_mac_legacy_f32"
                    if !vf.clamp =>
                {
                    let addend = if matches!(name, "v_fmac_f32" | "v_mac_f32" | "v_mac_legacy_f32")
                    {
                        vf.dst.as_ref().and_then(|d| Self::reg_val(inst, d))
                    } else if vf.abs & 4 == 0 && vf.neg & 4 == 0 {
                        vf.src.get(2).and_then(|o| Self::reg_val(inst, o))
                    } else {
                        None
                    };
                    [0, 1]
                        .into_iter()
                        .any(|k| self.fsrc_const(inst, k).is_some_and(|x| x * scale == 8.0))
                        || (vf.omod == 0
                            && addend.is_some_and(|a| self.is_array_face(a, None, seen)))
                }
                "v_fma_mix_f32" if !vf.clamp && vf.omod == 0 => [0, 1].into_iter().any(|k| {
                    let Some(op) = vf.src.get(k) else {
                        return false;
                    };
                    let Some(w) = self.op_bits(inst, op, 0) else {
                        return false;
                    };
                    let neg = vf.neg >> k & 1 != 0;
                    let abs = vf.neg_hi >> k & 1 != 0;
                    if vf.opsel_hi >> k & 1 != 0 {
                        let sh = if vf.opsel >> k & 1 != 0 { 16 } else { 0 };
                        let mut h = (w >> sh) & 0xFFFF;
                        if abs {
                            h &= 0x7FFF;
                        }
                        if neg {
                            h ^= 0x8000;
                        }
                        h == 0x4800
                    } else {
                        let mut x = w;
                        if abs {
                            x &= 0x7FFF_FFFF;
                        }
                        if neg {
                            x ^= 0x8000_0000;
                        }
                        f32::from_bits(x) == 8.0
                    }
                }),
                _ => false,
            }
        }
    }

    let ctx = Ctx { defs, phis };
    let scalar: HashMap<Site, Address> = res.scalar.iter().copied().collect();
    let descs: HashMap<Site, &Source> = res
        .descriptors
        .iter()
        .filter(|d| d.role == Role::Resource)
        .map(|d| (d.site, &d.source))
        .collect();
    let mut sites: Vec<(Site, Option<&Source>, bool)> = Vec::new();
    for (b, block) in p.blocks.iter().enumerate() {
        for (i, inst) in block.insts.iter().enumerate() {
            let Fields::Image {
                data,
                addr,
                dmask,
                dim: 3,
                a16,
                ..
            } = &inst.fields
            else {
                continue;
            };
            let site = (b, i);
            let name = inst.name();
            let mut direct = false;
            if name.starts_with("image_sample") || name.starts_with("image_gather") {
                let (mut whole, mut derivs) = (0usize, 0usize);
                if let Some(rest) = name
                    .strip_prefix("image_sample")
                    .or_else(|| name.strip_prefix("image_gather4"))
                {
                    for token in rest.split('_') {
                        match token {
                            "o" | "b" | "c" => whole += 1,
                            "d" | "cd" => derivs = 2,
                            _ => {}
                        }
                    }
                }
                let mut regs = Vec::new();
                for a in addr {
                    if let Operand::Vector { reg, dwords } = *a {
                        regs.extend((0..dwords).map(|k| reg + k));
                    }
                }
                let (idx, half) = if *a16 {
                    // `image` indexes `regs` by `(first + grads + k) / 2`.
                    let pos = whole + 2 * derivs + 2;
                    (pos / 2, Some(pos % 2 == 1))
                } else {
                    (whole + 2 * derivs + 2, None)
                };
                if let Some(&r) = regs.get(idx)
                    && let Some(fv) = inst.read(Reg::V(r))
                {
                    direct = ctx.is_array_face(fv, half, &mut HashSet::new());
                }
            } else if name == "image_get_resinfo" {
                let m = if *dmask == 0 { 1 } else { *dmask };
                if m & 4 != 0
                    && let Operand::Vector { reg, .. } = *data
                {
                    let k = (m & 3).count_ones() as u8;
                    direct = inst
                        .write(Reg::V(reg + k))
                        .is_some_and(|v| live.contains(&v));
                }
            }
            sites.push((site, descs.get(&site).copied(), direct));
        }
    }

    let leaf_addr = |s: &Source| match s {
        Source::Loaded { load, first } => scalar.get(load).map(|a| {
            (
                false,
                Address {
                    offset: a.offset.wrapping_add(4 * u32::from(*first)),
                    ..*a
                },
            )
        }),
        Source::Fetched { load, first } => scalar.get(load).map(|a| {
            (
                true,
                Address {
                    offset: a.offset.wrapping_add(4 * u32::from(*first)),
                    ..*a
                },
            )
        }),
        _ => None,
    };
    let same = |a: &Source, b: &Source| {
        a.leaves().into_iter().any(|la| {
            b.leaves()
                .into_iter()
                .any(|lb| la == lb || leaf_addr(la).is_some() && leaf_addr(la) == leaf_addr(lb))
        })
    };
    let mut out = HashSet::new();
    for &(site, src, direct) in &sites {
        if direct
            || src.is_some_and(|s| {
                sites
                    .iter()
                    .any(|&(_, os, od)| od && os.is_some_and(|o| same(s, o)))
            })
        {
            out.insert(site);
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::super::tests::{S_ENDPGM, lower_with, texture_and_uav_tables, texture_tables};
    use super::super::{Parts, lower};
    use crate::native::{HwStage, NativeProgram, StageIo};
    use crate::{io, ir, resource, structure};

    #[test]
    fn cube_array_encodings_and_linked_resinfo_lower() {
        // Common prologue: SRV at s[8:15] (table 0 offset 0), sampler at
        // s[16:19] (table 1 offset 0), raw UAV at s[20:23] (table 0 offset 64).
        let head = [
            0xBE84_0300u32, // s_mov_b32 s4, s0
            0xBE85_0380,    // s_mov_b32 s5, 0
            0xBE86_0301,    // s_mov_b32 s6, s1
            0xBE87_0380,    // s_mov_b32 s7, 0
            0xF40C_0202,
            0xFA00_0000, // s_load_dwordx8 s[8:15], s[4:5], 0
            0xF408_0403,
            0xFA00_0000, // s_load_dwordx4 s[16:19], s[6:7], 0
            0xF408_0502,
            0xFA00_0040, // s_load_dwordx4 s[20:23], s[4:5], 64
            0x7E02_02FF,
            0x3FC0_0000, // v_mov_b32 v1, 1.5
            0x7E04_02FF,
            0x3FC0_0000, // v_mov_b32 v2, 1.5
        ];
        let store_v4 = [
            0xE070_0000u32,
            0x8005_0400, // buffer_store_dword v4, off, s[20:23], 0
            S_ENDPGM,
        ];
        let cases: &[&[u32]] = &[
            // Constant face >= 8.0 (layer 1, face 2 = 10.0).
            &[
                0x7E06_02FF,
                0x4120_0000, // v_mov_b32 v3, 10.0
                0xF09C_0118,
                0x0082_0401, // image_sample_lz v4, v[1:3], s[8:15], s[16:19] dmask:1 dim:CUBE
            ],
            // SGPR 8.0 multiplier in v_fma_f32, followed by image_get_resinfo
            // on a second load of the same table slot.
            &[
                0xBE98_03FF,
                0x4100_0000, // s_mov_b32 s24, 8.0
                0x7E06_0282, // v_mov_b32 v3, 2
                0xD54B_0003,
                0x04CE_3000, // v_fma_f32 v3, v0, s24, v3
                0xF09C_0118,
                0x0082_0401, // image_sample_lz v4, v[1:3], s[8:15], s[16:19] dmask:1 dim:CUBE
                0xF40C_0702,
                0xFA00_0000, // s_load_dwordx8 s[28:35], s[4:5], 0
                0x7E0A_0280, // v_mov_b32 v5, 0
                0xF038_0418,
                0x0007_0605, // image_get_resinfo v6, v5, s[28:35] dmask:4 dim:CUBE
                0x1008_0C04, // v_add_f32 v4, v4, v6
            ],
            // v_fmamk_f32 by 8.0, then image_sample_l with explicit LOD in v4.
            &[
                0x7E06_02F2, // v_mov_b32 v3, 1.0
                0x5806_0700,
                0x4100_0000, // v_fmamk_f32 v3, v0, 8.0, v3
                0x7E08_0280, // v_mov_b32 v4, 0
                0xF090_0118,
                0x0082_0401, // image_sample_l v4, v[1:4], s[8:15], s[16:19] dmask:1 dim:CUBE
            ],
            // v_fmac_f32 by 8.0 with comparison reference before coordinates
            // (image_sample_c_lz).
            &[
                0x7E08_02F2, // v_mov_b32 v4, 1.0
                0x5608_00FF,
                0x4100_0000, // v_fmac_f32 v4, 8.0, v0
                0x7E06_02FF,
                0x3FC0_0000, // v_mov_b32 v3, 1.5 (ref=v1, s=v2, t=v3, face=v4)
                0xF0BC_0118,
                0x0082_0401, // image_sample_c_lz v4, v[1:4], s[8:15], s[16:19] dmask:1 dim:CUBE
            ],
            // v_mul_f32 with omod *2 (4.0 * 2 = 8.0) + v_add_f32.
            &[
                0xD508_0003,
                0x0801_EC00, // v_mul_f32 v3, v0, 4.0 mul:2
                0x0606_06F4, // v_add_f32 v3, 2.0, v3
                0xF09C_0118,
                0x0082_0401, // image_sample_lz v4, v[1:3], s[8:15], s[16:19] dmask:1 dim:CUBE
            ],
            // a16 coordinates: v_cvt_pkrtz_f16_f32 + v_cvt_f16_f32 of 8.0 * slice + face.
            &[
                0x5E02_0501, // v_cvt_pkrtz_f16_f32 v1, v1, v2
                0x7E06_02F4, // v_mov_b32 v3, 2.0
                0x5606_00FF,
                0x4100_0000, // v_fmac_f32 v3, 8.0, v0
                0x7E04_1503, // v_cvt_f16_f32 v2, v3
                0xF09C_0118,
                0x4082_0401, // image_sample_lz v4, v[1:2], s[8:15], s[16:19] dmask:1 dim:CUBE a16
            ],
        ];
        for body in cases {
            let words: Vec<u32> = head.iter().chain(*body).chain(&store_v4).copied().collect();
            let (layout, used) = texture_and_uav_tables();
            let l = lower_with(&words, layout, used).unwrap();
            #[cfg(windows)]
            if let Ok(mut v) = crate::emit::validator::Validator::load(
                crate::emit::validator::Validator::default_path().as_deref(),
            ) {
                let mut dxil = l.dxil.clone();
                v.validate(&mut dxil).expect("DXIL validation failed");
            }
            let _ = l;
        }

        // Pixel-stage `image_sample_b` (implicit LOD + bias before coordinates)
        // with `v_madak_f32` by 8.0.
        let ps_words = [
            0xBE84_0300u32, // s_mov_b32 s4, s0
            0xBE85_0380,    // s_mov_b32 s5, 0
            0xBE86_0301,    // s_mov_b32 s6, s1
            0xBE87_0380,    // s_mov_b32 s7, 0
            0xF40C_0202,
            0xFA00_0000, // s_load_dwordx8 s[8:15], s[4:5], 0
            0xF408_0403,
            0xFA00_0000, // s_load_dwordx4 s[16:19], s[6:7], 0
            0xBE98_03FF,
            0x4100_0000, // s_mov_b32 s24, 8.0
            0x7E00_02F2, // v_mov_b32 v0, 1.0 (slice)
            0x7E02_0280, // v_mov_b32 v1, 0.0 (bias)
            0x7E04_02FF,
            0x3FC0_0000, // v_mov_b32 v2, 1.5 (s)
            0x7E06_02FF,
            0x3FC0_0000, // v_mov_b32 v3, 1.5 (t)
            0x4208_0018,
            0x4000_0000, // v_madak_f32 v4, s24, v0, 2.0
            0xF094_0118,
            0x0082_0401, // image_sample_b v4, v[1:4], s[8:15], s[16:19] dmask:1 dim:CUBE
            0xF800_1801,
            0x0000_0004, // exp mrt0 v4, off, off, off done vm
            S_ENDPGM,
        ];
        let (layout, used) = texture_tables();
        let ps_native = NativeProgram {
            stage: HwStage::Ps,
            wave64: false,
            vgprs: 16,
            layout,
            used,
            io: StageIo::Ps {
                inputs: vec![],
                input_addr: 0,
                db_shader_control: 0,
                patches: vec![],
            },
            code: ps_words.iter().flat_map(|w| w.to_le_bytes()).collect(),
        };
        let p = ir::build(&ps_native.code, false, 16).unwrap();
        let st = structure::structure(&p).unwrap();
        let res = resource::track(&p, &ps_native.layout, HwStage::Ps, Some(&st)).unwrap();
        let io = io::raise(&p, &ps_native).unwrap();
        let ps_l = lower(&Parts {
            native: &ps_native,
            p: &p,
            st: &st,
            io: &io,
            res: &res,
            inputs: vec![],
            outputs: vec![super::super::SigElement {
                name: "SV_Target".into(),
                index: 0,
                kind: 16,
                comp: 9,
                interp: 0,
                row: Some(0),
                col: 0,
                cols: 4,
                usage: 0,
                half: false,
            }],
            prims: vec![],
            mesh_stride: 0,
            target: crate::emit::Target::ANY,
            view_marks: false,
        })
        .unwrap();
        #[cfg(windows)]
        if let Ok(mut v) = crate::emit::validator::Validator::load(
            crate::emit::validator::Validator::default_path().as_deref(),
        ) {
            let mut dxil = ps_l.dxil.clone();
            v.validate(&mut dxil).expect("PS DXIL validation failed");
        }
        let _ = ps_l;
    }
}
