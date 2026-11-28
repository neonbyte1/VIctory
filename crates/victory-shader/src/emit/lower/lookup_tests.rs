use super::*;
use crate::native::{ComputeIo, HwStage, NativeProgram, ResourceType, RootParam, RootType, StageIo, UsedResource};
use crate::{io, ir, resource, structure};

const S_ENDPGM: u32 = 0xBF81_0000;

fn root(root: RootType, user_data: u8, slot: u8) -> RootParam {
    RootParam {
        root,
        table_offset_dwords: 0,
        sgprs: if root == RootType::Descriptor { 2 } else { 1 },
        base_register: 0,
        user_data,
        slot,
    }
}


#[test]
fn def_tables_and_done_export_match_linear_scans() {
    use std::collections::{HashMap, HashSet};

    use super::Lower;
    use crate::ir::{Reg, Value};

    // Wave64 shader with entry pair s[0:1], multi-write s_mov_b64 s[6:7],
    // a join phi pair, and a whole-float conversion.
    let cs_words = [
        0xBE86_04C1u32, // s_mov_b64 s[6:7], -1
        0x7E04_0D00,    // v_cvt_f32_u32 v2, v0
        0xBF06_8000,    // s_cmp_eq_u32 s0, 0
        0xBF85_0001,    // s_cbranch_scc1 -> join
        0xBE86_0400,    // s_mov_b64 s[6:7], s[0:1]
        0xBF07_0706,    // join: s_cmp_lg_u32 s6, s7
        0xBF85_0001,    // s_cbranch_scc1 -> end
        0xBE84_0380,    // s_mov_b32 s4, 0
        S_ENDPGM,
    ];
    let cs_native = NativeProgram {
        stage: HwStage::Cs,
        wave64: true,
        vgprs: 256,
        layout: vec![RootParam { sgprs: 2, ..root(RootType::Constants, 0, 0) }],
        used: vec![],
        io: StageIo::Cs(ComputeIo { rsrc2: 1 << 15 | 2 << 1, threads: [64, 1, 1] }),
        code: cs_words.iter().flat_map(|w| w.to_le_bytes()).collect(),
    };
    let p = ir::build(&cs_native.code, true, 256).unwrap();
    let st = structure::structure(&p).unwrap();
    let res = resource::track(&p, &cs_native.layout, cs_native.stage, Some(&st)).unwrap();
    let io = io::raise(&p, &cs_native).unwrap();
    let parts = Parts {
        native: &cs_native, p: &p, st: &st, io: &io, res: &res,
        inputs: vec![], outputs: vec![], prims: vec![], mesh_stride: 0, target: crate::emit::Target::ANY, view_marks: false,
    };
    let bad = HashMap::new();
    let l = Lower::new(&parts, &bad);

    let scan_site = |v: Value| {
        p.blocks.iter().enumerate().find_map(|(b, blk)| {
            blk.insts.iter().enumerate()
                .find(|(_, i)| i.writes.iter().any(|&(_, w)| w == v))
                .map(|(i, _)| (b, i))
        })
    };
    let scan_block = |v: Value| {
        p.blocks.iter().enumerate().find_map(|(b, blk)| {
            (blk.phis.iter().any(|phi| phi.value == v)
                || blk.insts.iter().any(|i| i.writes.iter().any(|&(_, w)| w == v)))
            .then_some(b)
        })
    };
    for v in p.inputs.iter().map(|&(_, v)| v)
        .chain(p.blocks.iter().flat_map(|b| b.phis.iter().map(|phi| phi.value)))
        .chain(p.blocks.iter().flat_map(|b| b.insts.iter().flat_map(|i| i.writes.iter().map(|&(_, v)| v))))
        .chain([Value(u32::MAX)])
    {
        assert_eq!(l.def_block(v), scan_block(v));
        assert_eq!(l.def_site(v), scan_site(v));
        assert_eq!(l.def_inst(v).map(std::ptr::from_ref), scan_site(v).map(|(b, i)| std::ptr::from_ref(&p.blocks[b].insts[i])));
    }
    let cvt = p.blocks[0].insts.iter().find(|i| i.name() == "v_cvt_f32_u32").unwrap().write(Reg::V(2)).unwrap();
    assert_eq!(l.whole(cvt, &mut HashSet::new()), Some(true));

    // Pixel shader: block 0 has no export, block 1 non-done export, block 2 done export.
    let ps_words = [
        0x7E00_0280u32, 0xBF06_8000, 0xBF85_0002,
        0xF800_100F, 0x0000_0000, // exp mrt0 (done = false)
        0xF800_180F, 0x0000_0000, // exp mrt0 done
        S_ENDPGM,
    ];
    let ps_native = NativeProgram {
        stage: HwStage::Ps, wave64: false, vgprs: 256,
        layout: vec![root(RootType::Constants, 0, 0)], used: vec![],
        io: StageIo::Ps { inputs: vec![], input_addr: 0, db_shader_control: 0, patches: vec![] },
        code: ps_words.iter().flat_map(|w| w.to_le_bytes()).collect(),
    };
    let ps_p = ir::build(&ps_native.code, false, 256).unwrap();
    let ps_st = structure::structure(&ps_p).unwrap();
    let ps_res = resource::track(&ps_p, &ps_native.layout, ps_native.stage, Some(&ps_st)).unwrap();
    let ps_io = io::raise(&ps_p, &ps_native).unwrap();
    let target0 = super::SigElement {
        name: "SV_Target".into(), index: 0, kind: 16, comp: 9, interp: 0,
        row: Some(0), col: 0, cols: 4, usage: 0, half: false,
    };
    let ps_parts = Parts {
        native: &ps_native, p: &ps_p, st: &ps_st, io: &ps_io, res: &ps_res,
        inputs: vec![], outputs: vec![target0], prims: vec![], mesh_stride: 0, target: crate::emit::Target::ANY, view_marks: false,
    };
    let ps_l = Lower::new(&ps_parts, &bad);
    for b in 0..=ps_p.blocks.len() {
        let scan_done = ps_p.blocks.get(b).is_some_and(|blk| {
            (0..blk.insts.len()).any(|i| ps_l.exports.get(&(b, i)).is_some_and(|e| e.done))
        });
        assert_eq!(ps_l.has_done_export(b), scan_done, "block {b}");
    }
}

#[test]
#[ignore = "benchmark"]
fn bench_o1_lookups_large_shader() {
    // Large single-block pixel shader: 256 table descriptor loads +
    // 2,048 wave64 ALU/mask/cvt ops + done MRT0 export.
    let mut words = vec![0xBE84_0300u32, 0xBE85_0380, 0x7E00_0280];
    for _ in 0..256 {
        words.extend_from_slice(&[
            0xF408_0202, 0xFA00_0000, // s_load_dwordx4 s[8:11], s[4:5], null
            0xE030_2000, 0x8002_0200, // buffer_load_dword v2, v0, s[8:11], 0 idxen
            0x4A00_0102,              // v_add_nc_u32 v0, v2, v0
        ]);
    }
    for _ in 0..1024 {
        words.push(0x7E04_0D00); // v_cvt_f32_u32 v2, v0
        words.push(0x0600_0102); // v_add_f32 v0, v2, v0
    }
    words.extend_from_slice(&[0xF800_180F, 0x0000_0000, S_ENDPGM]);
    let native = NativeProgram {
        stage: HwStage::Ps, wave64: false, vgprs: 256,
        layout: vec![root(RootType::Table, 0, 0)],
        used: vec![UsedResource {
            kind: ResourceType::Srv, root: RootType::Table, table_offset_dwords: 0,
            lower_bound: 0, stride: 4, dwords: 4, user_data: 0, slot: 0,
        }],
        io: StageIo::Ps { inputs: vec![], input_addr: 0, db_shader_control: 0, patches: vec![] },
        code: words.iter().flat_map(|w| w.to_le_bytes()).collect(),
    };
    let p = ir::build(&native.code, false, 256).unwrap();
    let st = structure::structure(&p).unwrap();
    let io = io::raise(&p, &native).unwrap();
    let target0 = || super::SigElement {
        name: "SV_Target".into(), index: 0, kind: 16, comp: 9, interp: 0,
        row: Some(0), col: 0, cols: 4, usage: 0, half: false,
    };
    let res = resource::track(&p, &native.layout, native.stage, Some(&st)).unwrap();
    let _ = lower(&Parts {
        native: &native, p: &p, st: &st, io: &io, res: &res,
        inputs: vec![], outputs: vec![target0()], prims: vec![], mesh_stride: 0, target: crate::emit::Target::ANY, view_marks: false,
    }).unwrap();

    let t0 = std::time::Instant::now();
    for _ in 0..10 {
        let r = resource::track(&p, &native.layout, native.stage, Some(&st)).unwrap();
        std::hint::black_box(r.descriptors.len());
    }
    let track_us = t0.elapsed().as_secs_f64() * 1e6 / 10.0;

    let t1 = std::time::Instant::now();
    for _ in 0..10 {
        let l = lower(&Parts {
            native: &native, p: &p, st: &st, io: &io, res: &res,
            inputs: vec![], outputs: vec![target0()], prims: vec![], mesh_stride: 0, target: crate::emit::Target::ANY, view_marks: false,
        }).unwrap();
        std::hint::black_box(l.dxil.len());
    }
    let lower_us = t1.elapsed().as_secs_f64() * 1e6 / 10.0;
    eprintln!("o1_lookups (256 loads + 2048 ALU, {} insts): track={track_us:.1} us, lower={lower_us:.1} us", p.blocks[0].insts.len());
}
