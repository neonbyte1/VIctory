//! End-to-end tests of synthetic GFX10 VALU shaders lowered to DXIL, validated
//! by `dxil.dll` when present, and executed on the host GPU and WARP.
//!
//! Covers RDNA2 hardware semantics that differ from naive DXIL lowering:
//! `v_med3_f32`/`v_med3_f16` with NaN operands, `v_fract_f32`/`v_fract_f16`
//! near `1.0`, turn-range reduction for `v_sin_*`/`v_cos_*`, and denormal
//! operands of `v_mul_legacy_f32`.

use windows::Win32::Graphics::{Direct3D::*, Direct3D12::*, Dxgi::*};

use super::container::SigElement;

use super::host::{Dispatch, Draw, Host, Root, Stride, View};
use super::{HostShader, lower, validator};
use crate::native::{
    ComputeIo, HwStage, NativeProgram, ResourceType, RootParam, RootType, StageIo, Usage,
    UsedResource, VsInput,
};
use crate::{io, ir, range, resource, structure};

const S_ENDPGM: u32 = 0xBF81_0000;

/// SOP1 `op sdst, src0`.
const fn sop1(op: u8, sdst: u8, src0: u8) -> u32 {
    0xBE80_0000 | (sdst as u32) << 16 | (op as u32) << 8 | src0 as u32
}

/// SOP1 `s_mov_b32 sdst, src0`.
const fn s_mov_b32(sdst: u8, src0: u8) -> u32 {
    sop1(0x03, sdst, src0)
}

/// SOP2 `op sdst, src0, src1`.
const fn sop2(op: u8, sdst: u8, src0: u8, src1: u8) -> u32 {
    0x8000_0000
        | (op as u32 & 0x7F) << 23
        | (sdst as u32 & 0x7F) << 16
        | (src1 as u32) << 8
        | src0 as u32
}

/// SOPK `op sdst, simm16`; the compares read `sdst` as their operand.
const fn sopk(op: u8, sdst: u8, simm16: u16) -> u32 {
    0xB000_0000 | (op as u32 & 0x1F) << 23 | (sdst as u32 & 0x7F) << 16 | simm16 as u32
}

/// SMEM `s_load_dwordx4 s[dst:dst+3], s[base:base+1], 0` (soffset = null).
const fn s_load_dwordx4(dst: u8, base: u8) -> [u32; 2] {
    [
        0x3D << 26 | 0x02 << 18 | (dst as u32) << 6 | (base as u32 >> 1),
        125 << 25,
    ]
}

/// VOP1 `op vdst, src0` (`src0`: 0..105 = SGPR, 256..511 = VGPR).
const fn vop1(op: u16, vdst: u8, src0: u16) -> u32 {
    0x7E00_0000 | (vdst as u32) << 17 | (op as u32) << 9 | (src0 as u32 & 0x1FF)
}

/// VOP3 `op vdst, src0, src1, src2` (no modifiers).
const fn vop3(op: u16, vdst: u8, src0: u16, src1: u16, src2: u16) -> [u32; 2] {
    [
        0x35 << 26 | (op as u32 & 0x3FF) << 16 | vdst as u32,
        (src0 as u32 & 0x1FF) | (src1 as u32 & 0x1FF) << 9 | (src2 as u32 & 0x1FF) << 18,
    ]
}

/// MUBUF `buffer_store_dword vdata, off, s[srsrc:srsrc+3], 0 offset:byte_off`.
const fn buffer_store_dword(vdata: u8, srsrc: u8, byte_off: u16) -> [u32; 2] {
    [
        0x38 << 26 | 0x1C << 18 | (byte_off as u32 & 0xFFF),
        0x80 << 24 | ((srsrc as u32 >> 2) & 0x1F) << 16 | (vdata as u32) << 8,
    ]
}

/// DS `ds_swizzle_b32 vdst, vsrc offset:offset`.
const fn ds_swizzle_b32(vdst: u8, vsrc: u8, offset: u16) -> [u32; 2] {
    [
        0x36 << 26 | 0x35 << 18 | offset as u32,
        (vdst as u32) << 24 | vsrc as u32,
    ]
}

/// MUBUF `buffer_store_dword vdata, voff, s[srsrc:srsrc+3], 0 offen`.
const fn buffer_store_dword_offen(vdata: u8, voff: u8, srsrc: u8) -> [u32; 2] {
    [
        0x38 << 26 | 0x1C << 18 | 1 << 12,
        0x80 << 24 | ((srsrc as u32 >> 2) & 0x1F) << 16 | (vdata as u32) << 8 | voff as u32,
    ]
}

const fn sgpr(n: u8) -> u16 {
    n as u16
}

const fn vgpr(n: u8) -> u16 {
    256 + n as u16
}

/// Builds and lowers a 1-thread synthetic RDNA2 compute shader that receives
/// 4 dwords of root constants in `s0..s3`, runs `body` (leaving its result in
/// `v0`), and stores `v0` to dword 0 of the raw UAV at heap index 0.
fn build_valu_cs(body: &[u32]) -> HostShader {
    try_build_valu_cs(body).expect("build")
}

fn try_build_valu_cs(body: &[u32]) -> Result<HostShader, crate::ShaderError> {
    let mut words = Vec::new();
    // s[4:5] = descriptor table address (heap index 0 -> byte offset 0).
    words.push(s_mov_b32(5, 0x80)); // inline constant 0
    words.extend_from_slice(&s_load_dwordx4(8, 4));
    // Copy root constants s0..s3 into v1..v4 so VALU instructions operate on
    // VGPRs just as they do in real shaders.
    for i in 0..4u8 {
        words.push(vop1(0x001, i + 1, sgpr(i))); // v_mov_b32
    }
    words.extend_from_slice(body);
    words.extend_from_slice(&buffer_store_dword(0, 8, 0));
    words.push(S_ENDPGM);
    try_lower_valu_cs(&words, 16, 1)
}

fn try_lower_valu_cs(
    words: &[u32],
    vgprs: u16,
    threads_x: u32,
) -> Result<HostShader, crate::ShaderError> {
    let code: Vec<u8> = words.iter().flat_map(|w| w.to_le_bytes()).collect();
    let native = NativeProgram {
        stage: HwStage::Cs,
        wave64: false,
        vgprs,
        layout: vec![
            RootParam {
                root: RootType::Constants,
                table_offset_dwords: 0,
                sgprs: 4,
                base_register: 0,
                user_data: 0,
                slot: 0,
            },
            RootParam {
                root: RootType::Table,
                table_offset_dwords: 0,
                sgprs: 1,
                base_register: 0,
                user_data: 4,
                slot: 1,
            },
        ],
        used: vec![UsedResource {
            kind: ResourceType::Uav,
            root: RootType::Table,
            table_offset_dwords: 0,
            lower_bound: 0,
            stride: 0,
            dwords: 4,
            user_data: 4,
            slot: 1,
        }],
        io: StageIo::Cs(ComputeIo {
            rsrc2: 5 << 1, // 5 user SGPRs: s0..s4
            threads: [threads_x, 1, 1],
        }),
        code,
    };

    lower_valu_cs(&native)
}

fn lower_valu_cs(native: &NativeProgram) -> Result<HostShader, crate::ShaderError> {
    let (p, st) = range::build(&native.code, native.wave64, native.vgprs)?;
    let st = st?;
    let res = resource::track(&p, &native.layout, native.stage, Some(&st))?;
    let io = io::raise(&p, native)?;
    let lowered = lower::lower(&lower::Parts {
        native,
        p: &p,
        st: &st,
        io: &io,
        res: &res,
        inputs: Vec::new(),
        outputs: Vec::new(),
        prims: Vec::new(),
        mesh_stride: 0,
        target: super::Target::ANY,
        view_marks: false,
    })?;

    let shader = HostShader {
        stage: HwStage::Cs,
        wave_size: lowered.wave_size,
        dxil: lowered.dxil,
        user_data: lowered.user_data,
        samplers: lowered.samplers,
        static_samplers: Vec::new(),
        root_cbvs: lowered.root_cbvs,
        root_srvs: lowered.root_srvs,
        root_uavs: lowered.root_uavs,
        memory: lowered.memory,
        memory_writes: lowered.memory_writes,
        view_addresses: lowered.view_addresses,
        heap: lowered.heap,
        heap_writes: lowered.heap_writes,
        view_marks: lowered.view_marks,
    };

    match validator::Validator::load(validator::Validator::default_path().as_deref()) {
        Ok(mut v) => {
            let mut copy = shader.dxil.clone();
            let ours = copy[4..20].to_vec();
            v.validate(&mut copy).expect("DXIL validation failed");
            assert_eq!(copy[4..20], ours[..], "container hash mismatch");
        }
        Err(e) if std::env::var_os("VICTORY_DXIL").is_some() => {
            panic!("could not load VICTORY_DXIL: {e}")
        }
        Err(_) => {}
    }

    Ok(shader)
}

/// One GPU test at a time. With these tests on the hardware adapter and WARP
/// from several test threads at once, now and then none of their dispatches
/// ever completes and the test binary hangs; and two GPU tests at once on
/// WARP can end the test process with an access violation. Other GPU test
/// modules take this lock too.
pub(super) static ONE_AT_A_TIME: std::sync::Mutex<()> = std::sync::Mutex::new(());

/// Runs `shader` on every available SM 6.6 host (hardware GPU and WARP) for
/// each 4-dword input set and invokes `check(backend, case_index, inputs, result_dword)`.
fn run_cases(
    shader: &HostShader,
    cases: &[[u32; 4]],
    mut check: impl FnMut(&str, usize, [u32; 4], u32),
) {
    run_cases_with_outputs(shader, cases, 1, |backend, i, inputs, output| {
        check(backend, i, inputs, output[0]);
    });
}

fn run_cases_with_outputs(
    shader: &HostShader,
    cases: &[[u32; 4]],
    output_dwords: u32,
    mut check: impl FnMut(&str, usize, [u32; 4], &[u32]),
) {
    let _one = ONE_AT_A_TIME.lock().unwrap_or_else(|e| e.into_inner());
    let mut ran = 0;
    for (label, warp) in [("hardware", false), ("warp", true)] {
        let host = match Host::new(warp) {
            Ok(host) => host,
            Err(e) => {
                eprintln!("no SM 6.6 {label} device ({e}); skipped");
                continue;
            }
        };
        let (root, pso) = host
            .compute_pipeline(shader)
            .unwrap_or_else(|e| panic!("{label}: compute_pipeline failed: {e}"));
        for (i, &inputs) in cases.iter().enumerate() {
            let mut ud = vec![0u32; shader.user_data_dwords().max(1) as usize];
            for u in &shader.user_data {
                ud[u.index as usize] = match (u.root, u.slot) {
                    (RootType::Constants, 0) => inputs[u.dword as usize],
                    (RootType::Table, 1) => 0,
                    other => panic!("unexpected user data {other:?}"),
                };
            }
            let out = host
                .dispatch(
                    &root,
                    &pso,
                    &Dispatch {
                        roots: vec![(0, Root::Constants(ud))],
                        views: &[View::RawUav(output_dwords)],
                        samplers: &[],
                        groups: [1, 1, 1],
                    },
                )
                .unwrap_or_else(|e| panic!("{label}: dispatch failed: {e}"));
            check(label, i, inputs, &out[0]);
        }
        ran += 1;
    }
    if ran == 0 {
        crate::emit::host::no_device("no SM 6.6 device", &[]);
    }
}

#[test]
fn scalar_ff0_reads_the_source_width_and_writes_only_one_sgpr() {
    use ir::Reg::S;

    for (opcode, name, reads) in [
        (0x11, "s_ff0_i32_b32", &[S(0)][..]),
        (0x12, "s_ff0_i32_b64", &[S(0), S(1)][..]),
    ] {
        let code = [sop1(opcode, 16, 0), S_ENDPGM]
            .map(u32::to_le_bytes)
            .concat();
        let program = ir::build(&code, false, 16).unwrap();
        let inst = &program.blocks[0].insts[0];
        assert_eq!(inst.name(), name);
        assert!(
            inst.reads
                .iter()
                .map(|&(reg, _)| reg)
                .eq(reads.iter().copied())
        );
        assert!(inst.writes.iter().map(|&(reg, _)| reg).eq([S(16)]));
    }
}

#[test]
fn scalar_ff0_finds_the_lowest_clear_bit_and_preserves_scc() {
    let cases32 = [
        (0, 0, 0),
        (1, 0, 1),
        (0x0000_FFFF, 0, 16),
        (0x7FFF_FFFF, 0, 31),
        (u32::MAX, 0, u32::MAX),
    ];
    let cases64 = [
        (0, 0, 0),
        (0x0000_FFFF, 0, 16),
        (u32::MAX, 0, 32),
        (u32::MAX, 0x0000_FFFF, 48),
        (u32::MAX, 0x7FFF_FFFF, 63),
        (u32::MAX, u32::MAX, u32::MAX),
    ];
    for (opcode, cases) in [(0x11, &cases32[..]), (0x12, &cases64[..])] {
        // Toggle the result's top bit when SCC is set to check both in one dword.
        let shader = build_valu_cs(&[
            0xBF06_8102,              // s_cmp_eq_u32 s2, 1
            sop1(opcode, 16, 0),      // s_ff0 s16, s0 (s[0:1] for b64)
            sop2(0x0a, 18, 255, 128), // s_cselect_b32 s18, 0x80000000, 0
            0x8000_0000,
            sop2(0x12, 16, 16, 18), // s_xor_b32 s16, s16, s18
            vop1(1, 0, sgpr(16)),   // v_mov_b32 v0, s16
        ]);
        let container = crate::Container::parse(&shader.dxil).unwrap();
        let features = container.part(crate::FourCC::SFI0).unwrap();
        assert_eq!(
            u64::from_le_bytes(features.data.try_into().unwrap()) & 0x8000,
            0,
            "opcode {opcode:#x} requires Int64"
        );

        let inputs: Vec<_> = cases
            .iter()
            .flat_map(|&(lo, hi, _)| [[lo, hi, 0, 0], [lo, hi, 1, 0]])
            .collect();
        run_cases(&shader, &inputs, |backend, i, [lo, hi, scc, _], got| {
            let expected = cases[i / 2].2 ^ (scc << 31);
            assert_eq!(
                got, expected,
                "{backend}, opcode {opcode:#x}, input {hi:08x}:{lo:08x}, SCC {scc}"
            );
        });
    }
}

fn assert_no_int64(shader: &HostShader) {
    let container = crate::Container::parse(&shader.dxil).unwrap();
    let features = container.part(crate::FourCC::SFI0).unwrap();
    let features = u64::from_le_bytes(features.data.try_into().unwrap());
    assert_eq!(
        features & 0x8000,
        0,
        "signed no-carry arithmetic requires INT64"
    );
}

#[test]
fn v_add_sub_nc_i32_wrap_and_saturate() {
    for (op, add) in [(0x37f, true), (0x376, false)] {
        let inputs = [
            [20, -3, 0, 0],
            [-20, -3, 0, 0],
            [i32::MAX, if add { 1 } else { -1 }, 0, 0],
            [i32::MIN, if add { -1 } else { 1 }, 0, 0],
            [i32::MIN, if add { i32::MAX } else { i32::MIN }, 0, 0],
            [i32::MAX, 0, 0, 0],
            [i32::MIN, 0, 0, 0],
        ]
        .map(|input| input.map(|x| x as u32));
        for clamp in [false, true] {
            let mut body = vop3(op, 0, vgpr(1), vgpr(2), 0);
            body[0] |= u32::from(clamp) << 15;
            let shader = build_valu_cs(&body);
            assert_no_int64(&shader);
            run_cases(&shader, &inputs, |backend, i, [a, b, ..], got| {
                let (a, b) = (a as i32, b as i32);
                let want = match (add, clamp) {
                    (true, false) => a.wrapping_add(b),
                    (true, true) => a.saturating_add(b),
                    (false, false) => a.wrapping_sub(b),
                    (false, true) => a.saturating_sub(b),
                };
                assert_eq!(
                    got, want as u32,
                    "{backend}: VOP3 {op:#x}, clamp {clamp}, case {i}"
                );
            });
        }
    }
}

#[test]
fn v_add_sub_nc_i16_wrap_and_saturate_with_op_sel() {
    let old = 0xA5A5_5A5A;
    for (op, add) in [(0x30d, true), (0x30e, false)] {
        let operands = [
            (20, -3),
            (-20, -3),
            (i16::MAX, if add { 1 } else { -1 }),
            (i16::MIN, if add { -1 } else { 1 }),
            (i16::MAX, 0),
            (i16::MIN, 0),
        ];
        for opsel in [0, 0b0001, 0b0010, 0b1011] {
            let a_high = (opsel & 1) * 16;
            let b_high = (opsel >> 1 & 1) * 16;
            let dst_high = (opsel >> 3 & 1) * 16;
            let inputs = operands.map(|(a, b)| {
                [
                    (u32::from(a as u16) << a_high) | (0x1234 << (16 - a_high)),
                    (u32::from(b as u16) << b_high) | (0x5678 << (16 - b_high)),
                    old,
                    0,
                ]
            });
            for clamp in [false, true] {
                let mut inst = vop3(op, 0, vgpr(1), vgpr(2), 0);
                inst[0] |= u32::from(clamp) << 15 | opsel << 11;
                let shader = build_valu_cs(&[vop1(0x001, 0, vgpr(3)), inst[0], inst[1]]);
                assert_no_int64(&shader);
                run_cases(&shader, &inputs, |backend, i, _, got| {
                    let (a, b) = operands[i];
                    let result = match (add, clamp) {
                        (true, false) => a.wrapping_add(b),
                        (true, true) => a.saturating_add(b),
                        (false, false) => a.wrapping_sub(b),
                        (false, true) => a.saturating_sub(b),
                    };
                    let want =
                        (old & !(0xFFFF << dst_high)) | (u32::from(result as u16) << dst_high);
                    assert_eq!(
                        got, want,
                        "{backend}: VOP3 {op:#x}, clamp {clamp}, op_sel {opsel:#06b}, case {i}"
                    );
                });
            }
        }
    }
}

#[test]
fn quad_sad_vop3_tracks_register_dwords_and_rejects_modifiers() {
    for (op, dwords) in [(0x172, 2), (0x173, 2), (0x175, 4)] {
        let [lo, hi] = vop3(op, 8, vgpr(1), vgpr(3), vgpr(4));
        let code: Vec<_> = [lo, hi, S_ENDPGM]
            .into_iter()
            .flat_map(u32::to_le_bytes)
            .collect();
        let program = ir::build(&code, false, 16).unwrap();
        let inst = &program.blocks[0].insts[0];
        for reg in 1..4 + dwords {
            assert!(inst.read(ir::Reg::V(reg)).is_some(), "{op:#x}: v{reg}");
        }
        for reg in 8..8 + dwords {
            assert!(inst.write(ir::Reg::V(reg)).is_some(), "{op:#x}: v{reg}");
        }
        assert!(inst.write(ir::Reg::V(8 + dwords)).is_none());

        // ABS on src0, NEG on src2, OMOD, and OP_SEL, respectively.
        for (lo_bits, hi_bits) in [(1 << 8, 0), (0, 1 << 31), (0, 1 << 27), (1 << 11, 0)] {
            let refused = try_build_valu_cs(&[lo | lo_bits, hi | hi_bits, vop1(1, 0, vgpr(8))]);
            assert!(
                matches!(
                    refused,
                    Err(crate::ShaderError::Unsupported {
                        what: "modifiers on an integer quad SAD operation",
                        ..
                    })
                ),
                "{op:#x}: {:?}",
                refused.err()
            );
        }
    }
}

#[test]
fn quad_sad_vop3_wraps_or_saturates_every_window() {
    let data = [
        u32::from_le_bytes([7, 200, 19, 88]),
        u32::from_le_bytes([251, 0, 47, 222]),
    ];
    let reference = u32::from_le_bytes([0, 100, 240, 0]);
    // These reference bytes yield SADs [416, 684, 42, 526] unmasked and
    // [321, 233, 23, 391] masked, traversing both source dwords.
    let variants: [(_, &[_], &[_], &[_]); 3] = [
        (
            0x172,
            &[0x000C_FFFA, 0xFF14_FFFE],
            &[0x02B8_019A, 0x0122_0028],
            &[0x02B8_FFFF, 0xFFFF_FFFF],
        ),
        (
            0x173,
            &[0x000C_FFFA, 0xFF14_FFFE],
            &[0x00F5_013B, 0x009B_0015],
            &[0x00F5_FFFF, 0xFFFF_FFFF],
        ),
        (
            0x175,
            &[0xFFFF_FFF0, 42, 0xFFFF_FFFE, 0x7FFF_FFA0],
            &[0x131, 0x113, 0x15, 0x8000_0127],
            &[u32::MAX, 0x113, u32::MAX, 0x8000_0127],
        ),
    ];
    for (op, acc, wrapped, saturated) in variants {
        let dwords = acc.len() as u8;
        for (clamp, expected) in [(false, wrapped), (true, saturated)] {
            // v4 already holds the first accumulator dword from user data.
            let mut body = Vec::new();
            for (i, &word) in acc.iter().enumerate().skip(1) {
                body.extend([vop1(1, 4 + i as u8, 255), word]);
            }
            // Alias the destination and accumulator to exercise SSA reads
            // of the original dwords. The harness stores v0 at offset 0.
            let [lo, hi] = vop3(op, 4, vgpr(1), vgpr(3), vgpr(4));
            body.extend([lo | u32::from(clamp) << 15, hi, vop1(1, 0, vgpr(4))]);
            for i in 1..dwords {
                body.extend(buffer_store_dword(4 + i, 8, u16::from(i) * 4));
            }
            let shader = build_valu_cs(&body);
            let container = crate::Container::parse(&shader.dxil).unwrap();
            let features = container.part(crate::FourCC::SFI0).unwrap();
            let features = u64::from_le_bytes(features.data.try_into().unwrap());
            assert_eq!(features & 0x8000, 0, "{op:#x}: Int64Ops required");

            let cases = [
                [data[0], data[1], reference, acc[0]],
                [data[0], data[1] ^ 0xFF00_0000, reference, acc[0]],
            ];
            run_cases_with_outputs(&shader, &cases, u32::from(dwords), |backend, i, _, got| {
                assert_eq!(got, expected, "{backend}: {op:#x}, clamp={clamp}, case {i}");
            });
        }
    }
}

#[test]
fn scalar_quad_masks_pack_nibbles_and_set_scc() {
    let pack_words = |opcode: u32| {
        [
            0xBE90_0000 | opcode << 8,   // s_quadmask s16, s0 (pairs for b64)
            sop2(0x0a, 18, 129, 128),  // s_cselect_b32 s18, 1, 0
            sop2(0x1e, 18, 18, 144),   // s_lshl_b32 s18, s18, 16
            sop2(0x10, 16, 16, 18),    // s_or_b32 s16, s16, s18
            vop1(1, 0, sgpr(16)),      // v_mov_b32 v0, s16
        ]
    };
    let cases = [
        ([0, 0, 0, 0], [0, 0]),
        ([0x8000_0420, 0x0100_0001, 0, 0], [0x1_0086, 0x1_4186]),
        ([0, 0x8000_0000, 0, 0], [0, 0x1_8000]),
        ([u32::MAX, u32::MAX, 0, 0], [0x1_00ff, 0x1_ffff]),
    ];
    let inputs = cases.map(|(input, _)| input);
    for (wide, opcode) in [0x2c, 0x2d].into_iter().enumerate() {
        let shader = build_valu_cs(&pack_words(opcode));
        run_cases(&shader, &inputs, |backend, i, _, got| {
            assert_eq!(got, cases[i].1[wide], "{backend}, opcode {opcode:#x}");
        });
    }
    let high = build_valu_cs(&[
        0xBE90_2D00, // s_quadmask_b64 s[16:17], s[0:1]
        vop1(1, 0, sgpr(17)),
    ]);
    run_cases(&high, &inputs, |backend, _, _, got| {
        assert_eq!(got, 0, "{backend}: high dword");
    });
}

/// A uniform branch joins a mask restored from a VGPR lane with a directly
/// computed mask. The two cutoffs fall on opposite sides of the wave boundary.
#[test]
fn wave32_spilled_mask_at_join() {
    use windows::Win32::Graphics::Direct3D12::{
        D3D12_FEATURE_D3D12_OPTIONS1, D3D12_FEATURE_DATA_D3D12_OPTIONS1,
    };

    const THREADS: u32 = 64;
    let mut words = vec![s_mov_b32(5, 0x80)];
    words.extend(s_load_dwordx4(8, 4));
    words.extend([
        vop1(0x001, 10, vgpr(0)), // keep the thread ID
        vop1(0x001, 0, 0x81),     // unselected lanes store 1
    ]);
    words.extend(vop3(0x0C4, 6, 128 + 40, vgpr(10), 0)); // s6 = thread ID < 40
    words.extend(vop3(0x361, 20, sgpr(6), 0x81, 0)); // spill s6 into v20 lane 1
    words.extend(vop3(0x0C4, 14, 128 + 24, vgpr(10), 0)); // s14 = thread ID < 24
    words.extend([
        0xBF06_8000, // s_cmp_eq_u32 s0, 0
        0xBF84_0003, // s_cbranch_scc0 to the direct mask
    ]);
    words.extend(vop3(0x360, 12, vgpr(20), 0x81, 0)); // restore s12
    words.extend([
        0xBF82_0001,              // s_branch to the join
        s_mov_b32(12, 14),        // direct mask on the other edge
        0xBE90_3C0C,              // s_and_saveexec_b32 s16, s12
        vop1(0x001, 0, 0x85),     // selected lanes store 5
        sop2(0x10, 126, 126, 16), // s_or_b32 exec_lo, exec_lo, s16
        0x3412_1482,              // v_lshlrev_b32 v9, 2, v10
    ]);
    let [lo, hi] = buffer_store_dword(0, 8, 0);
    words.extend([lo | 1 << 12, hi | 9, S_ENDPGM]); // offen, byte address in v9
    let shader = try_lower_valu_cs(&words, 32, THREADS).expect("lower Wave32 mask join");
    let branch = shader
        .user_data
        .iter()
        .find(|u| u.root == RootType::Constants && u.slot == 0 && u.dword == 0)
        .expect("the shader reads the branch selector")
        .index as usize;
    let mut data = vec![0; shader.user_data_dwords().max(1) as usize];

    let _guard = ONE_AT_A_TIME.lock().unwrap_or_else(|e| e.into_inner());
    for (label, warp) in [("hardware", false), ("WARP", true)] {
        let host = match Host::new(warp) {
            Ok(host) => host,
            Err(e) => {
                eprintln!("no SM 6.6 {label} device ({e}); skipped");
                continue;
            }
        };
        let mut waves = D3D12_FEATURE_DATA_D3D12_OPTIONS1::default();
        // SAFETY: the feature query writes the matching options structure.
        unsafe {
            host.device.CheckFeatureSupport(
                D3D12_FEATURE_D3D12_OPTIONS1,
                (&raw mut waves).cast(),
                size_of_val(&waves) as u32,
            )
        }
        .unwrap_or_else(|e| panic!("{label}: querying wave support failed: {e}"));
        if !waves.WaveOps.as_bool()
            || !(waves.WaveLaneCountMin..=waves.WaveLaneCountMax).contains(&32)
        {
            eprintln!("{label}: Wave32 unavailable; skipping GPU readback");
            continue;
        }
        let (root, pipeline) = host
            .compute_pipeline(&shader)
            .unwrap_or_else(|e| panic!("{label}: compute_pipeline failed: {e}"));
        for (selector, cutoff) in [(0, 40), (1, 24)] {
            data[branch] = selector;
            let out = host
                .dispatch(
                    &root,
                    &pipeline,
                    &Dispatch {
                        roots: vec![(0, Root::Constants(data.clone()))],
                        views: &[View::RawUav(THREADS)],
                        samplers: &[],
                        groups: [1, 1, 1],
                    },
                )
                .unwrap_or_else(|e| panic!("{label}: dispatch failed: {e}"));
            assert_eq!(out[0].len(), THREADS as usize);
            for (thread, &value) in out[0].iter().enumerate() {
                let expected = if thread < cutoff { 5 } else { 1 };
                assert_eq!(
                    value, expected,
                    "{label}: branch {selector}, thread {thread}"
                );
            }
            eprintln!("{label}: branch {selector}, all {THREADS} thread values matched");
        }
    }
}

fn is_f16_nan(bits: u16) -> bool {
    (bits & 0x7FFF) > 0x7C00
}

#[test]
fn a_branch_to_fallthrough_keeps_both_values_at_a_join() {
    let shader = build_valu_cs(&[
        0xBF06_0100,            // s_cmp_eq_u32 s0, s1
        0xBF85_0002,            // s_cbranch_scc1 -> the s2 = 1 path
        s_mov_b32(2, 128),      // s2 = 0
        0xBF82_0002,            // s_branch -> the join
        s_mov_b32(2, 129),      // s2 = 1
        0xBF85_0000,            // s_cbranch_scc1 -> the following instruction
        vop1(0x001, 0, sgpr(2)), // join: v0 = s2
    ]);
    run_cases(
        &shader,
        &[[7, 7, 0, 0], [7, 5, 0, 0]],
        |backend, _, inputs, actual| {
            assert_eq!(actual, u32::from(inputs[0] == inputs[1]), "{backend}");
        },
    );
}

#[test]
fn v_fma_f16_rounds_once_and_preserves_other_half() {
    use super::host::Stride;
    use windows::Win32::Graphics::{Direct3D12::*, Dxgi::*};

    let _one = ONE_AT_A_TIME.lock().unwrap_or_else(|e| e.into_inner());
    let Ok(host) = Host::new(false) else {
        eprintln!("no SM 6.6 device; skipped");
        return;
    };
    // SAFETY: plain device queries with correctly typed output storage.
    unsafe {
        let factory: IDXGIFactory4 = CreateDXGIFactory2(DXGI_CREATE_FACTORY_FLAGS(0)).unwrap();
        let adapter: IDXGIAdapter1 = factory
            .EnumAdapterByLuid(host.device.GetAdapterLuid())
            .unwrap();
        let desc = adapter.GetDesc1().unwrap();
        let name = String::from_utf16_lossy(&desc.Description);
        let mut options = D3D12_FEATURE_DATA_D3D12_OPTIONS1::default();
        host.device
            .CheckFeatureSupport(
                D3D12_FEATURE_D3D12_OPTIONS1,
                (&raw mut options).cast(),
                size_of_val(&options) as u32,
            )
            .unwrap();
        eprintln!(
            "DXGI adapter: {}; lanes {}..{}; shader Wave32 (cs_6_6)",
            name.trim_end_matches('\0'),
            options.WaveLaneCountMin,
            options.WaveLaneCountMax
        );
    }

    let mut words = vec![s_mov_b32(1, 0x80), s_mov_b32(3, 0x80)];
    words.extend(s_load_dwordx4(8, 0)); // input SRV table at s0
    words.extend(s_load_dwordx4(12, 2)); // output UAV table at s2
    // buffer_load_dwordx4 v[1:4], off, s[8:11], 0
    words.extend([0xE038_0000, 0x8002_0100]);
    words.push(vop1(1, 0, vgpr(4)));
    words.push(vop1(1, 5, vgpr(4)));
    words.extend(vop3(0x34B, 0, vgpr(1), vgpr(2), vgpr(3)));
    let mut high = vop3(0x34B, 5, vgpr(1), vgpr(2), vgpr(3));
    high[0] |= 1 << 14; // destination op_sel: high half
    words.extend(high);
    words.extend(buffer_store_dword(0, 12, 0));
    words.extend(buffer_store_dword(5, 12, 4));
    words.push(S_ENDPGM);

    let tables = [(ResourceType::Srv, 0, 0), (ResourceType::Uav, 2, 1)];
    let shader = lower_valu_cs(&NativeProgram {
        stage: HwStage::Cs,
        wave64: false,
        vgprs: 8,
        layout: tables
            .iter()
            .map(|&(_, user_data, slot)| RootParam {
                root: RootType::Table,
                table_offset_dwords: 0,
                sgprs: 1,
                base_register: 0,
                user_data,
                slot,
            })
            .collect(),
        used: tables
            .iter()
            .map(|&(kind, user_data, slot)| UsedResource {
                kind,
                root: RootType::Table,
                table_offset_dwords: 0,
                lower_bound: 0,
                stride: 0,
                dwords: 4,
                user_data,
                slot,
            })
            .collect(),
        io: StageIo::Cs(ComputeIo {
            rsrc2: 3 << 1,
            threads: [1, 1, 1],
        }),
        code: words.iter().flat_map(|w| w.to_le_bytes()).collect(),
    })
    .expect("build");
    if let Ok(dir) = std::env::var("VICTORY_EMIT_DUMP") {
        std::fs::write(
            std::path::Path::new(&dir).join("v_fma_f16.dxil"),
            &shader.dxil,
        )
        .unwrap();
    }
    if let Some(path) = std::env::var_os("VICTORY_DXIL") {
        validator::Validator::load(Some(std::path::Path::new(&path)))
            .unwrap()
            .validate(&mut shader.dxil.clone())
            .unwrap();
        eprintln!("DXIL validated");
    }
    let (root, pso) = host.compute_pipeline(&shader).unwrap();
    let mut ud = vec![0; shader.user_data_dwords() as usize];
    for u in &shader.user_data {
        ud[u.index as usize] = u32::from(u.slot) * 32; // console descriptor byte offset
    }
    let seed = 0xABCD_1234;
    let cases = [
        ([0x3E00, 0x6C03, 0x0400, seed], 0x6E05), // 1.5 * 4108 + 2^-14
        ([0x3E00, 0x6C01, 0x8400, seed], 0x6E01), // 1.5 * 4100 - 2^-14
        ([0x3E00, 0x4000, 0x3C00, seed], 0x4400), // 1.5 * 2 + 1
        ([0x3E00, 0x6C03, 0, seed], 0x6E04),      // exact tie, lower even
        ([0x3E00, 0x6C01, 0, seed], 0x6E02),      // exact tie, upper even
        ([0xBE00, 0x6C03, 0x8400, seed], 0xEE05), // negative midpoint witness
        ([0x0400, 0x1000, 0, seed], 0),           // tie at half the least subnormal
        ([0x0400, 0x1000, 1, seed], 2),           // tie between subnormals
        ([0x7BFF, 0x3C00, 0x4C00, seed], 0x7C00), // overflow midpoint
    ];
    let mut results = Vec::new();
    for (data, half) in cases {
        let out = host
            .dispatch(
                &root,
                &pso,
                &Dispatch {
                    roots: vec![(0, Root::Constants(ud.clone()))],
                    views: &[
                        View::Buffer {
                            data: &data,
                            stride: Stride::Raw,
                        },
                        View::RawUav(2),
                    ],
                    samplers: &[],
                    groups: [1, 1, 1],
                },
            )
            .unwrap();
        let expected = [0xABCD_0000 | half, (half << 16) | 0x1234];
        eprintln!(
            "inputs {data:08x?}: got {:08x?}, expected {expected:08x?}",
            out[0]
        );
        results.push((out[0].clone(), expected));
    }
    for (got, expected) in results {
        assert_eq!(got, expected);
    }
}

fn f16_to_f32(bits: u16) -> f32 {
    let sign = u32::from(bits >> 15) << 31;
    let exp = (bits >> 10) & 0x1F;
    let mant = bits & 0x3FF;
    let f32_bits = match exp {
        0 => {
            if mant == 0 {
                sign
            } else {
                let mut m = u32::from(mant);
                let mut e = 127 - 14;
                while (m & 0x400) == 0 {
                    m <<= 1;
                    e -= 1;
                }
                sign | (e << 23) | ((m & 0x3FF) << 13)
            }
        }
        0x1F => sign | (0xFF << 23) | (u32::from(mant) << 13),
        _ => sign | ((u32::from(exp) + 112) << 23) | (u32::from(mant) << 13),
    };
    f32::from_bits(f32_bits)
}

/// MTBUF `op vdata, off, s[srsrc:srsrc+3], 0 format:format` (at byte 0).
const fn mtbuf(op: u8, vdata: u8, srsrc: u8, format: u8) -> [u32; 2] {
    [
        0x3A << 26 | (format as u32 & 0x7F) << 19 | (op as u32 & 7) << 16,
        0x80 << 24
            | (op as u32 >> 3 & 1) << 21
            | (srsrc as u32 >> 2 & 0x1F) << 16
            | (vdata as u32) << 8,
    ]
}

/// Stores `v1` to the buffer, then reads it back into `v0` with the tbuffer
/// load `op` in buffer format `format`.
fn tbuffer_load_of_v1(op: u8, format: u8) -> HostShader {
    let mut body = buffer_store_dword(1, 8, 0).to_vec();
    body.extend(mtbuf(op, 0, 8, format));
    build_valu_cs(&body)
}

#[test]
fn shift_add_and_align_take_sign_bit_modifiers() {
    // VOP3 abs / neg on v_lshl_add_u32, v_add_lshl_u32, v_alignbit_b32 and
    // v_alignbyte_b32 clear / flip a source's bit 31, as on a float
    // (RDNA2: boykopovar/AnyPS5#25964 on gfx1036 and gfx1201, #25799).
    // (op, NEG bits, ABS bits, inputs v1..v3, result)
    let cases: &[(u16, u32, u32, [u32; 3], u32)] = &[
        (0x346, 4, 0, [5, 2, 1], 0x8000_0015),          // lshl_add: neg(src2)
        (0x347, 0, 1, [0x8000_0003, 1, 0], 4),          // add_lshl: abs(src0)
        (0x14E, 2, 0, [0, 1, 31], 1),                   // alignbit: neg(src1)
        (0x14F, 2, 0, [0, 0x00FF_FFFF, 3], 0x80),       // alignbyte: neg(src1)
        (0x14E, 0, 2, [0, 0x8000_0000, 31], 0),         // alignbit: abs(src1)
        // abs and neg on one source: abs clears bit 31, then neg sets it
        // (-|x|, as on a float); the other order would give 4 and 0x15
        (0x347, 1, 1, [3, 1, 0], 0x8000_0004),          // add_lshl: -|src0|
        (0x346, 4, 4, [5, 2, 0x8000_0001], 0x8000_0015), // lshl_add: -|src2|
    ];
    for &(op, neg, abs, [a, b, c], want) in cases {
        let mut w = vop3(op, 0, vgpr(1), vgpr(2), vgpr(3));
        w[0] |= abs << 8;
        w[1] |= neg << 29;
        let shader = build_valu_cs(&w);
        run_cases(&shader, &[[a, b, c, 0]], |backend, _, _, got| {
            assert_eq!(got, want, "{backend}: op {op:#x} neg {neg} abs {abs} of {a:#x}, {b:#x}, {c:#x} = {got:#010x}, want {want:#010x}");
        });
    }
}

#[test]
fn tbuffer_d16_loads_pack_two_channels_per_dword() {
    // tbuffer_load_format_d16_x and tbuffer_load_format_d16_xy.
    let (x, xy) = (8, 9);
    // (opcode, format, dword stored, dword loaded)
    let cases: &[(u8, u8, u32, u32)] = &[
        // 16_16_FLOAT: the f16 pair (1.0, 2.0) as it was stored.
        (xy, 29, 0x4000_3C00, 0x4000_3C00),
        // 16_16_UINT: the 16-bit integers as they were stored.
        (xy, 27, 0xBEEF_1234, 0xBEEF_1234),
        // 16_16_UNORM: 0.0 and 1.0, as f16.
        (xy, 23, 0xFFFF_0000, 0x3C00_0000),
        // 16_FLOAT, one channel: the high half is 0.
        (x, 13, 0x4000_3C00, 0x0000_3C00),
    ];
    for &(op, format, stored, want) in cases {
        let shader = tbuffer_load_of_v1(op, format);
        run_cases(&shader, &[[stored, 0, 0, 0]], |backend, _, _, got| {
            assert_eq!(
                got, want,
                "{backend}: d16 opcode {op} format {format} of {stored:#010x} = {got:#010x}, want {want:#010x}",
            );
        });
    }
}

#[test]
fn tbuffer_d16_loads_saturate_32_bit_integers() {
    // tbuffer_load_format_d16_x of a 32-bit integer: RDNA2 saturates it to
    // 16 bits (UINT to 0xFFFF, SINT to -32768..32767) rather than keeping
    // its low half (measured on gfx1031 and gfx1036). Only the low half of
    // the result is compared: what an odd last channel leaves in the high
    // half is a separate question.
    let x = 8;
    // (format, dword stored, low half loaded)
    let cases: &[(u8, u32, u32)] = &[
        (20, 70000, 0xFFFF),              // 32_UINT: past 0xFFFF
        (20, 5, 5),                       // in range: as stored
        (21, 70000, 0x7FFF),              // 32_SINT: past 32767
        (21, (-65536i32) as u32, 0x8000), // below -32768
        (21, (-5i32) as u32, 0xFFFB),     // in range: the low half
    ];
    for &(format, stored, want) in cases {
        let shader = tbuffer_load_of_v1(x, format);
        run_cases(&shader, &[[stored, 0, 0, 0]], |backend, _, _, got| {
            assert_eq!(got & 0xFFFF, want, "{backend}: d16_x format {format} of {stored:#x} = {got:#010x}, want low half {want:#06x}");
        });
    }
}

#[test]
fn d16_normalized_format_loads_do_not_require_int64() {
    for unorm in [1, 7, 14, 23, 30, 37, 44, 50, 56, 65] {
        for format in [unorm, unorm + 1] {
            let shader = tbuffer_load_of_v1(11, format);
            let container = crate::Container::parse(&shader.dxil).unwrap();
            let flags = container.part(crate::FourCC::SFI0).unwrap().data;
            let flags = u64::from_le_bytes(flags.try_into().unwrap());
            assert_eq!(flags & 0x8000, 0, "format {format} requires Int64ShaderOps");
        }
    }
}

#[test]
fn tbuffer_d16_loads_of_16_bit_norms_round_once() {
    // tbuffer_load_format_d16_x of 16_UNORM / 16_SNORM: RDNA2 rounds the
    // normalized value to f16 once, to nearest even (gfx1031 for buffer
    // format loads, xcl-dev/xcl#1488 XCL-BT-020; gfx1036 for images,
    // boykopovar/AnyPS5#22750). The expected halves are the exact value
    // rounded once. Only the low half is compared.
    const UNORM: [u16; 64] = [0x0000, 0xFFFF, 0xFFAF, 0xFFEF, 0x0801, 0x0803, 0x0805, 0x0807, 0x8000, 0x7FFF, 0x1234, 0x4321, 0xEBBE, 0x78FA, 0xE20F, 0xF2BF, 0x4275, 0x8398, 0xBD51, 0x1BC1, 0xB57A, 0x7378, 0x64F0, 0x9631, 0xA93D, 0x1078, 0xE750, 0x620A, 0x03BF, 0x5971, 0x9EFA, 0xA070, 0xC8CB, 0x2DF2, 0xF2DD, 0xEEA9, 0x028F, 0xD55A, 0x6C19, 0x23A3, 0xE272, 0x61DF, 0x186A, 0xB09A, 0x0F62, 0xAD5F, 0x89C7, 0xDB80, 0x92C3, 0x507A, 0x6FF9, 0x5CFD, 0xF790, 0xCFC4, 0xB827, 0xE334, 0x70BB, 0xF609, 0x5ADA, 0x4EDA, 0xDEC6, 0xCA73, 0xAAE7, 0x0E6C];
    const UNORM_F16: [u16; 64] = [0x0000, 0x3C00, 0x3BFD, 0x3BFF, 0x2801, 0x2802, 0x2803, 0x2804, 0x3800, 0x3800, 0x2C8D, 0x3432, 0x3B5E, 0x3790, 0x3B10, 0x3B96, 0x3427, 0x381D, 0x39EB, 0x2EF0, 0x39AC, 0x3738, 0x364F, 0x38B2, 0x394A, 0x2C1E, 0x3B3B, 0x3621, 0x237E, 0x3597, 0x38F8, 0x3904, 0x3A46, 0x31BE, 0x3B97, 0x3B75, 0x211E, 0x3AAB, 0x36C2, 0x3074, 0x3B14, 0x361E, 0x2E1B, 0x3985, 0x2BB1, 0x396B, 0x384E, 0x3ADC, 0x3896, 0x3508, 0x3700, 0x35D0, 0x3BBD, 0x3A7E, 0x39C1, 0x3B1A, 0x370C, 0x3BB0, 0x35AE, 0x34EE, 0x3AF6, 0x3A54, 0x3957, 0x2B36];
    const SNORM: [u16; 64] = [0x0000, 0x7FFF, 0x8000, 0x8001, 0x7FF7, 0x8009, 0x0401, 0x0403, 0xFBFF, 0xC000, 0x4000, 0x0001, 0x0669, 0x5B5B, 0x93FA, 0x9DB5, 0xAA7E, 0x4B72, 0x6920, 0xC24D, 0x6F99, 0x4FDA, 0x507E, 0xE531, 0x1C88, 0x0D1A, 0x3151, 0xDB97, 0xDA66, 0xADB0, 0xF832, 0xE808, 0xFE62, 0xD349, 0xEFCC, 0xFC89, 0xFF9D, 0xAEB5, 0x3CAB, 0xA475, 0x1B07, 0xFCAC, 0x2B11, 0x067A, 0xA023, 0x872F, 0xAC57, 0x775A, 0x3629, 0x315A, 0xA1B8, 0x6C88, 0xA7C3, 0xD5E3, 0xECDC, 0x4E5C, 0x6CEB, 0xFB6D, 0x0ED9, 0x00B7, 0x0A51, 0x205A, 0x490A, 0x246B];
    const SNORM_F16: [u16; 64] = [0x0000, 0x3C00, 0xBC00, 0xBC00, 0x3BFF, 0xBBFF, 0x2801, 0x2803, 0xA801, 0xB800, 0x3800, 0x0200, 0x2A69, 0x39B6, 0xBAC0, 0xBA25, 0xB958, 0x38B7, 0x3A92, 0xB7B6, 0x3AFA, 0x38FE, 0x3908, 0xB2B4, 0x3322, 0x2E8D, 0x362A, 0xB48D, 0xB4B3, 0xB925, 0xABCE, 0xB1FE, 0xA278, 0xB597, 0xB00D, 0xA6EE, 0x9A30, 0xB915, 0x3795, 0xB9B9, 0x32C2, 0xA6A8, 0x3562, 0x2A7A, 0xB9FE, 0xBB8D, 0xB93B, 0x3B76, 0x36C5, 0x362B, 0xB9E5, 0x3AC9, 0xB984, 0xB544, 0xB0C9, 0x38E6, 0x3ACF, 0xA893, 0x2F6D, 0x1DB8, 0x2D29, 0x340B, 0x3891, 0x348D];
    let x = 8;
    let mut bad = Vec::new();
    for (format, codes, want) in [(7u8, &UNORM, &UNORM_F16), (8, &SNORM, &SNORM_F16)] {
        let shader = tbuffer_load_of_v1(x, format);
        let inputs: Vec<[u32; 4]> = codes.iter().map(|&c| [u32::from(c), 0, 0, 0]).collect();
        run_cases(&shader, &inputs, |backend, i, _, got| {
            if got & 0xFFFF != u32::from(want[i]) {
                bad.push(format!("{backend}: format {format} code {:#06x} = {:#06x}, want {:#06x}", codes[i], got & 0xFFFF, want[i]));
            }
        });
    }
    assert!(bad.is_empty(), "{} differ:
{}", bad.len(), bad.join("
"));
}

#[test]
fn normalized_buffer_formats_load_their_ends_exactly() {
    // tbuffer_load_format_x.
    let x = 0;
    // (format, dword stored, value loaded)
    let cases: &[(u8, u32, f32)] = &[
        (1, 0xFF, 1.0),   // 8_UNORM
        (7, 0xFFFF, 1.0), // 16_UNORM
        (2, 0x7F, 1.0),   // 8_SNORM: the top, the one above the lowest, the lowest
        (2, 0x81, -1.0),
        (2, 0x80, -1.0),
        (8, 0x7FFF, 1.0), // 16_SNORM
        (8, 0x8001, -1.0),
        (8, 0x8000, -1.0),
    ];
    for &(format, stored, want) in cases {
        let shader = tbuffer_load_of_v1(x, format);
        run_cases(&shader, &[[stored, 0, 0, 0]], |backend, _, _, got| {
            assert_eq!(
                got,
                want.to_bits(),
                "{backend}: format {format} of {stored:#x} = {got:#010x}, want {want}",
            );
        });
    }
}

/// Codes of the normalized formats sampled below: every 8-bit code; for
/// 16 bits a stride, the RX 6700 XT's division misses (257, 261, ...) and
/// the ends.
fn normalized_codes(bits: u32) -> Vec<u32> {
    if bits <= 10 {
        return (0..1 << bits).collect();
    }
    let edges = [1, 2, 3, 255, 256, 257, 261, 265, 269, 511, 512, 1000, 0x7FFF, 0x8000, 0x8001, 0xFFFE];
    (0..=0xFFFF).step_by(257).chain(edges).collect()
}

/// Every normalized code loads as the float nearest its exact quotient,
/// as the console's buffer unit converts it, not as the host divides (the
/// RX 6700 XT gave 125 of the 8-bit codes one ulp high and 16-bit codes
/// such as 257 one ulp low): unsigned and signed, 8, 10 (the x channel of
/// 2_10_10_10) and 16 bits, the signed formats' two lowest codes -1.0.
#[test]
fn normalized_buffer_formats_load_correctly_rounded_quotients() {
    let x = 0;
    // (format, bits, signed): 8_UNORM, 16_UNORM, 8_SNORM, 16_SNORM,
    // 2_10_10_10_UNORM and _SNORM.
    for (format, bits, signed) in [(1u8, 8u32, false), (7, 16, false), (2, 8, true), (8, 16, true), (50, 10, false), (51, 10, true)] {
        let shader = tbuffer_load_of_v1(x, format);
        let cases: Vec<[u32; 4]> = normalized_codes(bits).iter().map(|&a| [a, 0, 0, 0]).collect();
        run_cases(&shader, &cases, |backend, _, [a, ..], got| {
            let want = if signed {
                let top = (1i32 << (bits - 1)) - 1;
                let s = ((a << (32 - bits)) as i32) >> (32 - bits);
                (s.max(-top) as f32 / top as f32).to_bits()
            } else {
                (a as f32 / ((1u32 << bits) - 1) as f32).to_bits()
            };
            let sign = if signed { "signed" } else { "unsigned" };
            assert_eq!(got, want, "{backend}: {bits}-bit {sign} normalized {a:#x} loads as {got:#010x}, want {want:#010x}");
        });
    }
}

/// A title truncates `x * (2^n - 1)` back to the code it stored (the
/// material index a mesh program reads from a vertex colour's alpha, as
/// `v_mul_legacy_f32` by 255.0 then `v_cvt_u32_f32`): every code comes back
/// as itself, on hardware and WARP, which a quotient one ulp low breaks.
#[test]
fn normalized_codes_truncate_back_to_themselves() {
    let x = 0;
    for (format, bits) in [(1u8, 8u32), (7, 16)] {
        let scale = (((1u32 << bits) - 1) as f32).to_bits();
        let mut body = buffer_store_dword(1, 8, 0).to_vec();
        body.extend(mtbuf(x, 0, 8, format));
        // v_mul_legacy_f32 v0, literal, v0; v_cvt_u32_f32 v0, v0.
        body.extend([0x07 << 25 | 0xFF, scale]);
        body.push(vop1(0x07, 0, vgpr(0)));
        let shader = build_valu_cs(&body);
        let cases: Vec<[u32; 4]> = normalized_codes(bits).iter().map(|&a| [a, 0, 0, 0]).collect();
        run_cases(&shader, &cases, |backend, _, [a, ..], got| {
            assert_eq!(got, a, "{backend}: {bits}-bit code {a} comes back as {got}");
        });
    }
}

#[test]
fn packed_add_selects_halves_of_inline_constants() {
    // Add the inline constant 1/(2*pi) to v1 (zero) to read back its two halves.
    // The same GFX10 source field (248) has a different bit pattern for a
    // packed floating-point and a packed integer instruction.
    let cases = [
        // v_pk_add_f16 v0, 0x3118, v1
        ([0xcc0f4000, 0x1a0202f8], 0x0000_3118),
        // v_pk_add_f16 v0, 0x3118, v1 op_sel_hi:[0,1] (use low twice)
        ([0xcc0f4000, 0x120202f8], 0x3118_3118),
        // v_pk_add_u16 v0, 0x3e22f983, v1
        ([0xcc0a4000, 0x1a0202f8], 0x3e22_f983),
    ];
    for (words, expected) in cases {
        let shader = build_valu_cs(&words);
        run_cases(&shader, &[[0; 4]], |backend, _, _, got| {
            assert_eq!(got, expected, "{backend}: {words:08x?}");
        });
    }
}

#[test]
fn v_med3_f32_matches_rdna2_with_nan() {
    // v_med3_f32 v0, v1, v2, v3 (VOP3 opcode 0x157)
    let shader = build_valu_cs(&vop3(0x157, 0, vgpr(1), vgpr(2), vgpr(3)));
    let nan = f32::NAN.to_bits();
    let f = f32::to_bits;

    let cases: &[([u32; 4], Option<f32>)] = &[
        // Normal medians (all orderings).
        ([f(1.0), f(2.0), f(3.0), 0], Some(2.0)),
        ([f(3.0), f(1.0), f(2.0), 0], Some(2.0)),
        ([f(-5.0), f(0.0), f(5.0), 0], Some(0.0)),
        ([f(1.0), f(1.0), f(2.0), 0], Some(1.0)),
        // One NaN: RDNA2 returns min3 of the non-NaN operands.
        ([f(1.0), f(2.0), nan, 0], Some(1.0)),
        ([nan, f(1.0), f(0.0), 0], Some(0.0)),
        ([nan, f(0.0), f(1.0), 0], Some(0.0)),
        ([f(2.0), nan, f(1.0), 0], Some(1.0)),
        ([f(-3.0), nan, f(4.0), 0], Some(-3.0)),
        // Two NaNs: returns the sole non-NaN operand.
        ([nan, f(5.0), nan, 0], Some(5.0)),
        ([nan, nan, f(-2.0), 0], Some(-2.0)),
        ([f(7.0), nan, nan, 0], Some(7.0)),
        // Three NaNs: returns NaN.
        ([nan, nan, nan, 0], None),
    ];

    let inputs: Vec<[u32; 4]> = cases.iter().map(|c| c.0).collect();
    run_cases(&shader, &inputs, |backend, i, inp, got| {
        let expect = cases[i].1;
        let got_f = f32::from_bits(got);
        match expect {
            Some(want) => assert_eq!(
                got_f,
                want,
                "{backend}: v_med3_f32({:?}, {:?}, {:?}) = {got_f} ({got:#010x}), want {want}",
                f32::from_bits(inp[0]),
                f32::from_bits(inp[1]),
                f32::from_bits(inp[2]),
            ),
            None => assert!(
                got_f.is_nan(),
                "{backend}: v_med3_f32(NaN, NaN, NaN) = {got_f} ({got:#010x}), want NaN",
            ),
        }
    });
}

fn check_half_fma_nearest_even(body: [u32; 2]) {
    let cases = [
        ([0x6600, 0x3956, 0x0400, 0], 0x6401), // 1024.5 plus 2^-14
        ([0x6600, 0x3956, 0, 0], 0x6400), // exact tie, even
        ([0xe600, 0x3956, 0x8400, 0], 0xe401), // negative midpoint plus a negative tail
        ([0x6700, 0x3894, 0x8400, 0], 0x6401), // 1025.5 minus 2^-14
        ([0x6700, 0x3894, 0, 0], 0x6402), // exact tie, even
    ];
    let inputs = cases.map(|(input, _)| input);
    run_cases(&build_valu_cs(&body), &inputs, |backend, i, _, got| {
        assert_eq!(got & 0xffff, cases[i].1, "{backend}: case {i}");
    });
}

#[test]
fn half_fma_rounds_once_to_nearest_even() {
    check_half_fma_nearest_even(vop3(0x34b, 0, vgpr(1), vgpr(2), vgpr(3)));
}

#[test]
fn half_fmac_rounds_once_to_nearest_even() {
    let fmac = (0x036 << 25) | (2 << 9) | u32::from(vgpr(1));
    check_half_fma_nearest_even([vop1(0x001, 0, vgpr(3)), fmac]);
}

#[test]
fn fused_half_keeps_subnormal_operands() {
    let cases = [
        ([0x7e00_0001, 0xfc00_fc00, 0x5640_5152, 0], Some(0xfc00)),
        ([0x0000_fc00, 0x0000_0001, 0x5640_5152, 0], Some(0xfc00)),
        ([0x0000_8001, 0x0000_fc00, 0x5640_5152, 0], Some(0x7c00)),
        ([0x0000_0400, 0x0000_fc00, 0x5640_5152, 0], Some(0xfc00)),
        ([0x0000_0001, 0x0000_6800, 0x5640_0000, 0], Some(0x0800)),
        ([0x0000_0000, 0x0000_3c00, 0x5640_0001, 0], Some(0x0001)),
        ([0x0000_8000, 0x0000_fc00, 0x5640_5152, 0], None),
    ];
    let inputs: Vec<_> = cases.iter().map(|&(words, _)| words).collect();
    let initial = vop1(0x001, 0, vgpr(3));
    let fmac = (0x036 << 25) | (2 << 9) | u32::from(vgpr(1));
    let [fma, sources] = vop3(0x34b, 0, vgpr(1), vgpr(2), vgpr(3));

    for (form, program) in [("fmac", &[fmac][..]), ("fma", &[fma, sources][..])] {
        for (rounding, mode) in [("rne", &[][..]), ("rtz", &[0xba80_0881, 3][..])] {
            let body = [mode, &[initial], program].concat();
            let shader = build_valu_cs(&body);
            run_cases(&shader, &inputs, |device, i, _, actual| {
                assert_eq!(actual >> 16, 0x5640, "{device} {form} {rounding} row {i}");
                let value = actual & 0xffff;
                if let Some(expected) = cases[i].1 {
                    assert_eq!(value, expected, "{device} {form} {rounding} row {i}");
                } else {
                    assert!(value & 0x7fff > 0x7c00, "{device} {form} {rounding} row {i}: {value:04x}");
                }
            });
        }
    }
}

#[test]
fn half_fma_rounds_once_toward_zero() {
    let [op, src] = vop3(0x34b, 0, vgpr(1), vgpr(2), vgpr(3));
    let shader = build_valu_cs(&[0xba80_0881, 3, op, src]); // MODE.FP_ROUND16_64 = RTZ
    let cases = [
        ([0x6800, 0x3c00, 0x8400, 0], 0x67ff), // 2048 minus 2^-14
        ([0x0400, 0xbc00, 0x6800, 0], 0x67ff), // same sum with the addend larger
        ([0xe800, 0x3c00, 0x0400, 0], 0xe7ff), // negative mirror
        ([0x6800, 0x3c00, 0, 0], 0x6800),
    ];
    let inputs = cases.map(|(input, _)| input);
    run_cases(&shader, &inputs, |backend, i, _, got| {
        assert_eq!(got & 0xffff, cases[i].1, "{backend}: case {i}");
    });
}

#[test]
fn half_fused_nonfinite_results_survive_rounding() {
    const F16_NAN: u32 = 0x7e00;
    let cases = [
        ([0x7c00, 0x3c00, 0x3c00, 0], 0x7c00),  // +inf + 1
        ([0x3c00, 0xfc00, 0x3c00, 0], 0xfc00),  // -inf + 1
        ([0x3c00, 0xbc00, 0x7c00, 0], 0x7c00),  // -1 + inf
        ([0x3c00, 0x3c00, 0xfc00, 0], 0xfc00),  // 1 - inf
        ([0x7c00, 0x3c00, 0xfc00, 0], F16_NAN), // inf - inf
        ([0x7c00, 0, 0x3c00, 0], F16_NAN),      // inf * 0
    ];
    let inputs = cases.map(|(input, _)| input);
    let fmac = (0x036 << 25) | (2 << 9) | u32::from(vgpr(1));
    let forms = [
        ("FMA", vop3(0x34b, 0, vgpr(1), vgpr(2), vgpr(3))),
        ("FMAC", [vop1(0x001, 0, vgpr(3)), fmac]),
    ];

    for (form, words) in forms {
        for (mode, prefix) in [("nearest", &[][..]), ("toward zero", &[0xba80_0881, 3])] {
            let mut body = prefix.to_vec();
            body.extend_from_slice(&words);
            run_cases(&build_valu_cs(&body), &inputs, |backend, i, _, got| {
                let actual = got & 0xffff;
                let want = cases[i].1;
                if want == F16_NAN {
                    assert_eq!(
                        actual & F16_NAN,
                        F16_NAN,
                        "{backend}: {form} {mode} case {i}: expected a quiet NaN, got {actual:#06x}"
                    );
                } else {
                    assert_eq!(actual, want, "{backend}: {form} {mode} infinity case {i}");
                }
            });
        }
    }
}

#[test]
fn half_fma_clamps_in_both_rounding_modes() {
    let [op, src] = vop3(0x34b, 0, vgpr(1), vgpr(2), vgpr(3));
    // Expected results in nearest-even and toward-zero order.
    let cases = [
        ([0x0001, 0x8001, 0x3c00, 0], [0x3c00, 0x3bff]), // 1 - 2^-48
        ([0x3c01, 0x3c00, 0x8001, 0], [0x3c00; 2]),      // 1 + 2^-10 - 2^-24
        ([0x3c00, 0x3c00, 0, 0], [0x3c00; 2]),           // exactly 1
        ([0x4000, 0x3c00, 0, 0], [0x3c00; 2]),           // 2.0
        ([0x7bff, 0x3c00, 0, 0], [0x3c00; 2]),           // 65504.0
        ([0xbc00, 0x3c00, 0, 0], [0; 2]),                // -1.0
        ([0x7c00, 0x3c00, 0x3c00, 0], [0x3c00; 2]),      // +inf
        ([0xfc00, 0x3c00, 0x3c00, 0], [0; 2]),           // -inf
        ([0x7e00, 0x3c00, 0, 0], [0; 2]),                // NaN
    ];
    let inputs = cases.map(|(input, _)| input);
    for (i, (label, mode)) in [("nearest-even", 0), ("toward zero", 3)].into_iter().enumerate() {
        let shader = build_valu_cs(&[0xba80_0881, mode, op | 1 << 15, src]);
        run_cases(&shader, &inputs, |backend, case, input, got| {
            assert_eq!(
                got & 0xffff,
                cases[case].1[i],
                "{backend}: {label} clamped case {case}, inputs {input:x?}"
            );
        });
    }
}

/// RDNA2 orders -0 below +0 in its maximums and minimums; DXIL's FMax and
/// FMin compare and select, which can give either zero.
#[test]
fn max_and_min_order_negative_zero_below_positive_zero() {
    let (pz, nz) = (0_u32, 0x8000_0000_u32);
    // Each case has a -0 and a +0 among its first two operands.
    let cases = [
        [nz, pz, nz, 0],
        [pz, nz, nz, 0],
        [nz, pz, pz, 0],
        [pz, nz, pz, 0],
    ];
    let ops = [
        ("v_max_f32", 0x110, pz),
        ("v_min_f32", 0x10F, nz),
        ("v_max3_f32", 0x154, pz),
        ("v_min3_f32", 0x151, nz),
    ];
    for (name, op, want) in ops {
        let shader = build_valu_cs(&vop3(op, 0, vgpr(1), vgpr(2), vgpr(3)));
        run_cases(&shader, &cases, |backend, _, inp, got| {
            assert_eq!(got, want, "{backend}: {name} {inp:x?}");
        });
    }
}

/// The ISA defines `v_med3_f32` over `v_min_f32`/`v_max_f32`, which order -0
/// below +0 (the same order as [`max_and_min_order_negative_zero_below_positive_zero`]),
/// so of three zeros the median is the sign two of them share. DXIL's plain
/// FMin/FMax chain gives either zero.
#[test]
fn med3_orders_negative_zero_below_positive_zero() {
    let (pz, nz) = (0_u32, 0x8000_0000_u32);
    let f = f32::to_bits;
    let cases: &[([u32; 4], u32)] = &[
        ([nz, pz, nz, 0], nz),      // -0, +0, -0
        ([pz, nz, pz, 0], pz),      // +0, -0, +0
        ([nz, pz, pz, 0], pz),      // -0, +0, +0
        ([pz, nz, nz, 0], nz),      // +0, -0, -0
        ([nz, nz, nz, 0], nz),      // -0, -0, -0
        ([pz, pz, pz, 0], pz),      // +0, +0, +0
        ([nz, pz, f(1.0), 0], pz),  // -0 < +0 < 1
        ([nz, f(-1.0), pz, 0], nz), // -1 < -0 < +0
        ([pz, f(1.0), nz, 0], pz),  // -0 < +0 < 1
    ];
    let shader = build_valu_cs(&vop3(0x157, 0, vgpr(1), vgpr(2), vgpr(3)));
    let inputs: Vec<[u32; 4]> = cases.iter().map(|c| c.0).collect();
    run_cases(&shader, &inputs, |backend, i, inp, got| {
        let want = cases[i].1;
        assert_eq!(
            got, want,
            "{backend}: v_med3_f32({:#010x}, {:#010x}, {:#010x}) = {got:#010x}, want {want:#010x}",
            inp[0], inp[1], inp[2],
        );
    });

    // The same in halves: -0 is 0x8000, +0 is 0x0000 (VOP3 opcode 0x357).
    let (hpz, hnz) = (0x0000_u32, 0x8000_u32);
    let (h1, hn1) = (0x3C00_u32, 0xBC00_u32); // 1.0, -1.0
    let cases16: &[([u32; 4], u32)] = &[
        ([hnz, hpz, hnz, 0], hnz),
        ([hpz, hnz, hpz, 0], hpz),
        ([hnz, hpz, hpz, 0], hpz),
        ([hpz, hnz, hnz, 0], hnz),
        ([hnz, hpz, h1, 0], hpz),
        ([hnz, hn1, hpz, 0], hnz),
        ([hpz, h1, hnz, 0], hpz),
    ];
    let shader = build_valu_cs(&vop3(0x357, 0, vgpr(1), vgpr(2), vgpr(3)));
    let inputs: Vec<[u32; 4]> = cases16.iter().map(|c| c.0).collect();
    run_cases(&shader, &inputs, |backend, i, inp, got| {
        let want = cases16[i].1 as u16;
        assert_eq!(
            got as u16, want,
            "{backend}: v_med3_f16({:#06x}, {:#06x}, {:#06x}) = {got:#06x}, want {want:#06x}",
            inp[0], inp[1], inp[2],
        );
    });
}
/// The f16 forms go through the same `max_min` helper, after widening to f32.
#[test]
fn f16_max_and_min_order_negative_zero_below_positive_zero() {
    let (pz, nz) = (0_u32, 0x8000_u32);
    let cases = [
        [nz, pz, nz, 0],
        [pz, nz, nz, 0],
        [nz, pz, pz, 0],
        [pz, nz, pz, 0],
    ];
    let ops = [
        ("v_max_f16", 0x139, pz),
        ("v_min_f16", 0x13A, nz),
        ("v_max3_f16", 0x354, pz),
        ("v_min3_f16", 0x351, nz),
    ];
    for (name, op, want) in ops {
        let shader = build_valu_cs(&vop3(op, 0, vgpr(1), vgpr(2), vgpr(3)));
        run_cases(&shader, &cases, |backend, _, inp, got| {
            // The f16 result is the low half of v0.
            assert_eq!(got & 0xFFFF, want, "{backend}: {name} {inp:x?}");
        });
    }
}

#[test]
fn v_min3_max3_med3_integer_forms_match_rdna2() {
    // v_{min3,max3,med3}_{i32,u32} v0, v1, v2, v3 (VOP3 0x152..0x159).
    type Pick = fn([u32; 3]) -> u32;
    let ops: [(&str, u16, Pick); 6] = [
        ("v_min3_i32", 0x152, |t| {
            t.into_iter().min_by_key(|&x| x as i32).unwrap()
        }),
        ("v_min3_u32", 0x153, |t| t.into_iter().min().unwrap()),
        ("v_max3_i32", 0x155, |t| {
            t.into_iter().max_by_key(|&x| x as i32).unwrap()
        }),
        ("v_max3_u32", 0x156, |t| t.into_iter().max().unwrap()),
        ("v_med3_i32", 0x158, |mut t| {
            t.sort_by_key(|&x| x as i32);
            t[1]
        }),
        ("v_med3_u32", 0x159, |mut t| {
            t.sort_unstable();
            t[1]
        }),
    ];
    // Every ordering of three values, with the sign bit set on some so that
    // the signed and unsigned answers differ.
    let values = [0u32, 1, 0x7FFF_FFFF, 0x8000_0000, 0xFFFF_FFFF];
    let mut cases = Vec::new();
    for a in values {
        for b in values {
            for c in values {
                cases.push([a, b, c, 0]);
            }
        }
    }
    for (name, op, pick) in ops {
        let shader = build_valu_cs(&vop3(op, 0, vgpr(1), vgpr(2), vgpr(3)));
        run_cases(&shader, &cases, |backend, _, inp, got| {
            let want = pick([inp[0], inp[1], inp[2]]);
            assert_eq!(
                got, want,
                "{backend}: {name}({:#010x}, {:#010x}, {:#010x}) = {got:#010x}, want {want:#010x}",
                inp[0], inp[1], inp[2],
            );
        });
    }
}

/// Three-XOR swaps on a descriptor table SGPR (via `vcc_lo`, then `s_xor_b64`)
/// preserve its root origin across lowering and run on hardware and WARP.
#[test]
fn three_xor_swap_tracks_table_through_vcc_and_64bit_pair() {
    let mut words = vec![
        s_mov_b32(12, 4),         // s12 = s4 (descriptor table)
        s_mov_b32(106, 193),      // vcc_lo = -1
        sop2(0x12, 12, 12, 106),  // s_xor_b32 s12, s12, vcc_lo
        sop2(0x12, 106, 12, 106), // s_xor_b32 vcc_lo, s12, vcc_lo
        sop2(0x12, 12, 12, 106),  // s_xor_b32 s12, s12, vcc_lo
        s_mov_b32(14, 106),       // s14 = vcc_lo (descriptor table)
        s_mov_b32(15, 0x80),      // s15 = 0
        s_mov_b32(16, 0x80),      // s16 = 0
        s_mov_b32(17, 0x80),      // s17 = 0
        sop2(0x13, 14, 14, 16),   // s_xor_b64 s[14:15], s[14:15], s[16:17]
        sop2(0x13, 16, 14, 16),   // s_xor_b64 s[16:17], s[14:15], s[16:17]
        sop2(0x13, 14, 14, 16),   // s_xor_b64 s[14:15], s[14:15], s[16:17]
    ];
    words.extend_from_slice(&s_load_dwordx4(8, 16));
    words.push(vop1(0x001, 0, sgpr(0))); // v_mov_b32 v0, s0
    words.extend_from_slice(&buffer_store_dword(0, 8, 0));
    words.push(S_ENDPGM);

    let code: Vec<u8> = words.iter().flat_map(|w| w.to_le_bytes()).collect();
    let native = NativeProgram {
        stage: HwStage::Cs,
        wave64: false,
        vgprs: 16,
        layout: vec![
            RootParam {
                root: RootType::Constants,
                table_offset_dwords: 0,
                sgprs: 4,
                base_register: 0,
                user_data: 0,
                slot: 0,
            },
            RootParam {
                root: RootType::Table,
                table_offset_dwords: 0,
                sgprs: 1,
                base_register: 0,
                user_data: 4,
                slot: 1,
            },
        ],
        used: vec![UsedResource {
            kind: ResourceType::Uav,
            root: RootType::Table,
            table_offset_dwords: 0,
            lower_bound: 0,
            stride: 0,
            dwords: 4,
            user_data: 4,
            slot: 1,
        }],
        io: StageIo::Cs(ComputeIo {
            rsrc2: 5 << 1,
            threads: [1, 1, 1],
        }),
        code,
    };

    let (p, st) = range::build(&native.code, native.wave64, native.vgprs).unwrap();
    let st = st.unwrap();
    let res = resource::track(&p, &native.layout, native.stage, Some(&st)).unwrap();
    let io = io::raise(&p, &native).unwrap();
    let lowered = lower::lower(&lower::Parts {
        native: &native,
        p: &p,
        st: &st,
        io: &io,
        res: &res,
        inputs: Vec::new(),
        outputs: Vec::new(),
        prims: Vec::new(),
        mesh_stride: 0,
        target: crate::emit::Target::ANY,
        view_marks: false,
    })
    .unwrap();

    let shader = HostShader {
        stage: HwStage::Cs,
        wave_size: lowered.wave_size,
        dxil: lowered.dxil,
        user_data: lowered.user_data,
        samplers: lowered.samplers,
        static_samplers: Vec::new(),
        root_cbvs: lowered.root_cbvs,
        root_srvs: lowered.root_srvs,
        root_uavs: lowered.root_uavs,
        memory: lowered.memory,
        memory_writes: lowered.memory_writes,
        view_addresses: lowered.view_addresses,
        heap: lowered.heap,
        heap_writes: lowered.heap_writes,
        view_marks: lowered.view_marks,
    };
    if let Ok(mut v) = validator::Validator::load(validator::Validator::default_path().as_deref()) {
        let mut copy = shader.dxil.clone();
        v.validate(&mut copy).expect("DXIL validation failed");
    }

    run_cases(&shader, &[[0xCAFE_BEEF, 0, 0, 0]], |backend, _, _, got| {
        assert_eq!(got, 0xCAFE_BEEF, "{backend}");
    });
}

#[test]
fn v_med3_f16_matches_rdna2_with_nan() {
    // v_med3_f16 v0, v1, v2, v3 (VOP3 opcode 0x357)
    let shader = build_valu_cs(&vop3(0x357, 0, vgpr(1), vgpr(2), vgpr(3)));
    let h0 = 0x0000u32; // 0.0
    let h1 = 0x3C00u32; // 1.0
    let h2 = 0x4000u32; // 2.0
    let h3 = 0x4200u32; // 3.0
    let h5 = 0x4500u32; // 5.0
    let hn2 = 0xC000u32; // -2.0
    let nan = 0x7E00u32;

    let cases: &[([u32; 4], Option<u16>)] = &[
        ([h1, h2, h3, 0], Some(h2 as u16)),
        ([hn2, h0, h3, 0], Some(h0 as u16)),
        ([h1, h2, nan, 0], Some(h1 as u16)),
        ([nan, h1, h0, 0], Some(h0 as u16)),
        ([nan, h0, h1, 0], Some(h0 as u16)),
        ([h2, nan, h1, 0], Some(h1 as u16)),
        ([nan, h5, nan, 0], Some(h5 as u16)),
        ([nan, nan, nan, 0], None),
    ];

    let inputs: Vec<[u32; 4]> = cases.iter().map(|c| c.0).collect();
    run_cases(&shader, &inputs, |backend, i, inp, got| {
        let expect = cases[i].1;
        let got_h = got as u16;
        match expect {
            Some(want) => assert_eq!(
                got_h, want,
                "{backend}: v_med3_f16({:#06x}, {:#06x}, {:#06x}) = {got_h:#06x}, want {want:#06x}",
                inp[0], inp[1], inp[2],
            ),
            None => assert!(
                is_f16_nan(got_h),
                "{backend}: v_med3_f16(NaN, NaN, NaN) = {got_h:#06x}, want NaN",
            ),
        }
    });
}

/// `v_cmp_lg_*` is "less or greater": ordered and not equal, so false when
/// either operand is a NaN (RDNA2 ISA; the same on gfx1036 hardware).
#[test]
fn v_cmp_lg_is_false_for_nan() {
    // v_cmp_lg_{f32,f16}_e64 s20, v1, v2 (VOPC 0x005 / 0x0cd in VOP3);
    // v_cndmask_b32_e64 v0, 0, 1, s20 (VOP3 0x101)
    let nan = f32::NAN.to_bits();
    let (inf, ninf) = (f32::INFINITY.to_bits(), f32::NEG_INFINITY.to_bits());
    let f = f32::to_bits;
    let f32_cases: &[([u32; 4], u32)] = &[
        ([f(1.0), f(2.0), 0, 0], 1),
        ([f(2.0), f(1.0), 0, 0], 1),
        ([f(1.0), f(1.0), 0, 0], 0),
        ([f(0.0), f(-0.0), 0, 0], 0),
        ([inf, ninf, 0, 0], 1),
        ([inf, inf, 0, 0], 0),
        ([nan, f(1.0), 0, 0], 0),
        ([f(1.0), nan, 0, 0], 0),
        ([nan, nan, 0, 0], 0),
        ([0x7F80_0001, f(0.0), 0, 0], 0), // a signalling NaN
    ];
    let f16_cases: &[([u32; 4], u32)] = &[
        ([0x3C00, 0x4000, 0, 0], 1),
        ([0x4000, 0x3C00, 0, 0], 1),
        ([0x3C00, 0x3C00, 0, 0], 0),
        ([0x0000, 0x8000, 0, 0], 0),
        ([0x7C00, 0xFC00, 0, 0], 1),
        ([0x7E00, 0x3C00, 0, 0], 0),
        ([0x3C00, 0x7E00, 0, 0], 0),
        ([0x7E00, 0x7E00, 0, 0], 0),
        ([0x7C01, 0x0000, 0, 0], 0),           // a signalling NaN
        ([0xABCD_3C00, 0x1234_7E00, 0, 0], 0), // only the low halves are read
    ];
    for (op, what, cases) in [
        (0x005u16, "v_cmp_lg_f32", f32_cases),
        (0x0cd, "v_cmp_lg_f16", f16_cases),
    ] {
        let mut body = vop3(op, 20, vgpr(1), vgpr(2), 0x80).to_vec();
        body.extend_from_slice(&vop3(0x101, 0, 0x80, 0x81, sgpr(20)));
        let shader = build_valu_cs(&body);
        let inputs: Vec<[u32; 4]> = cases.iter().map(|c| c.0).collect();
        run_cases(&shader, &inputs, |backend, i, inp, got| {
            let want = cases[i].1;
            assert_eq!(
                got, want,
                "{backend}: {what}({:#010x}, {:#010x}) = {got}, want {want}",
                inp[0], inp[1],
            );
        });
    }
}

/// `v_div_fixup_f16` as the RDNA2 ISA lists it; `None` for a NaN.
fn div_fixup_f16_reference(q: u16, den: u16, num: u16) -> Option<u16> {
    let nan = |x: u16| is_f16_nan(x);
    let abs = |x: u16| x & 0x7FFF;
    let sign = (den ^ num) & 0x8000;
    if nan(num) || nan(den) {
        return None;
    }
    if (abs(den) == 0 && abs(num) == 0) || (abs(den) == 0x7C00 && abs(num) == 0x7C00) {
        return None;
    }
    if abs(den) == 0 || abs(num) == 0x7C00 {
        return Some(sign | 0x7C00);
    }
    if abs(den) == 0x7C00 || abs(num) == 0 {
        return Some(sign);
    }
    // An all-ones exponent overflows, so a NaN quotient gives infinity
    // (gfx1036 hardware).
    if abs(q) >= 0x7C00 {
        return Some(sign | 0x7C00);
    }
    Some(sign | abs(q))
}

#[test]
fn v_div_fixup_f16_follows_the_isa_special_cases() {
    // v_mov_b32 v0, 0; v_div_fixup_f16 v0, v1, v2, v3 (VOP3 opcode 0x35f):
    // v1 the quotient, v2 the denominator, v3 the numerator.
    let mut body = vec![vop1(0x001, 0, 128)];
    body.extend_from_slice(&vop3(0x35f, 0, vgpr(1), vgpr(2), vgpr(3)));
    let shader = build_valu_cs(&body);
    // 0, -0, 1, -1.5, 2^-24, max finite, inf, -inf, NaN, -NaN.
    let values = [
        0x0000u16, 0x8000, 0x3C00, 0xBE00, 0x0001, 0x7BFF, 0x7C00, 0xFC00, 0x7E00, 0xFE01,
    ];
    let mut cases = Vec::new();
    for &q in &[0x4248u16, 0xC248, 0x7E00, 0x7C01] {
        for &den in &values {
            for &num in &values {
                cases.push([u32::from(q), u32::from(den), u32::from(num), 0]);
            }
        }
    }
    run_cases(&shader, &cases, |backend, _, inp, got| {
        let (q, den, num) = (inp[0] as u16, inp[1] as u16, inp[2] as u16);
        let got_h = got as u16;
        assert_eq!(got >> 16, 0, "{backend}: the high half changed");
        match div_fixup_f16_reference(q, den, num) {
            Some(want) => assert_eq!(
                got_h, want,
                "{backend}: v_div_fixup_f16({q:#06x}, {den:#06x}, {num:#06x}) = {got_h:#06x}, \
                 want {want:#06x}",
            ),
            None => assert!(
                is_f16_nan(got_h),
                "{backend}: v_div_fixup_f16({q:#06x}, {den:#06x}, {num:#06x}) = {got_h:#06x}, \
                 want NaN",
            ),
        }
    });
}

/// 64-bit integer compares order by the high dword (signed for i64), then
/// by the low one. AMD's D3D12 compiler runs an ordered i64 `icmp` as if the
/// dwords were swapped (RX 9070 and gfx1036 alike), so the lowering compares
/// the dwords; the values are picked so that the two orders disagree.
#[test]
fn v_cmp_u64_and_i64_order_by_the_high_dword() {
    type Order = fn(u64, u64) -> bool;
    let ops: [(u16, &str, Order); 12] = [
        (0x0a1, "v_cmp_lt_i64", |a, b| (a as i64) < (b as i64)),
        (0x0a2, "v_cmp_eq_i64", |a, b| a == b),
        (0x0a3, "v_cmp_le_i64", |a, b| (a as i64) <= (b as i64)),
        (0x0a4, "v_cmp_gt_i64", |a, b| (a as i64) > (b as i64)),
        (0x0a5, "v_cmp_ne_i64", |a, b| a != b),
        (0x0a6, "v_cmp_ge_i64", |a, b| (a as i64) >= (b as i64)),
        (0x0e1, "v_cmp_lt_u64", |a, b| a < b),
        (0x0e2, "v_cmp_eq_u64", |a, b| a == b),
        (0x0e3, "v_cmp_le_u64", |a, b| a <= b),
        (0x0e4, "v_cmp_gt_u64", |a, b| a > b),
        (0x0e5, "v_cmp_ne_u64", |a, b| a != b),
        (0x0e6, "v_cmp_ge_u64", |a, b| a >= b),
    ];
    let values: [u64; 10] = [
        0,
        1,
        0xFFFF_FFFF,
        0x1_0000_0000,
        0x1_FFFF_FFFF,
        0x0000_0040_C010_FBFF,
        0x0001_8000_0000_0040,
        0x7FFF_FFFF_FFFF_FFFF,
        0x8000_0000_0000_0001,
        0xFFFF_FFFF_0000_0000,
    ];
    let inputs: Vec<[u32; 4]> = values
        .iter()
        .flat_map(|&a| values.iter().map(move |&b| (a, b)))
        .map(|(a, b)| [a as u32, (a >> 32) as u32, b as u32, (b >> 32) as u32])
        .collect();
    for (op, name, order) in ops {
        // op s20, v[1:2], v[3:4] and, sharing v2, op s20, v[1:2], v[2:3];
        // v_cndmask_b32_e64 v0, 0, 1, s20
        for second in [3, 2] {
            let mut body = vop3(op, 20, vgpr(1), vgpr(second), 0x80).to_vec();
            body.extend_from_slice(&vop3(0x101, 0, 0x80, 0x81, sgpr(20)));
            let shader = build_valu_cs(&body);
            let at = usize::from(second) - 1;
            run_cases(&shader, &inputs, |backend, _, inp, got| {
                let a = u64::from(inp[0]) | u64::from(inp[1]) << 32;
                let b = u64::from(inp[at]) | u64::from(inp[at + 1]) << 32;
                let want = u32::from(order(a, b));
                assert_eq!(got, want, "{backend}: {name}({a:#018x}, {b:#018x})");
            });
        }
    }
}

#[test]
fn v_cvt_off_f32_i4_uses_the_signed_low_nibble() {
    // v_cvt_off_f32_i4 v0, v1 (VOP1 opcode 0x00e)
    let shader = build_valu_cs(&[vop1(0x00e, 0, vgpr(1))]);
    let nibble = |n: u32| (((n as i32) << 28) >> 28) as f32 / 16.0;

    let mut cases: Vec<u32> = (0..16).collect();
    // Bits above the low nibble are ignored.
    cases.extend([0xABCD_EF05, 0xFFFF_FFF8, 0x1234_5670, 0x8000_000F, 0x0000_00F7]);
    let inputs: Vec<[u32; 4]> = cases.iter().map(|&x| [x, 0, 0, 0]).collect();
    run_cases(&shader, &inputs, |backend, i, inp, got| {
        let want = nibble(cases[i]).to_bits();
        assert_eq!(
            got, want,
            "{backend}: v_cvt_off_f32_i4({:#010x}) = {got:#010x} ({:?}), want {want:#010x}",
            inp[0],
            f32::from_bits(got),
        );
    });
    // Spot-check the reference itself: 0x5 -> 0.3125, 0x8 -> -0.5, 0xF -> -0.0625.
    assert_eq!(nibble(0xABCD_EF05), 0.3125);
    assert_eq!(nibble(8), -0.5);
    assert_eq!(nibble(0xF), -0.0625);
}

#[test]
fn v_fract_f32_never_returns_one() {
    // v_fract_f32 v0, v1 (VOP1 opcode 0x020)
    let shader = build_valu_cs(&[vop1(0x020, 0, vgpr(1))]);
    let f = f32::to_bits;
    let below_one = 0x3F7F_FFFFu32;

    let cases: &[(u32, Option<u32>)] = &[
        // Tiny negatives: must clamp strictly below 1.0 (not 0x3f800000).
        (f(-1e-10), Some(below_one)),
        (f(-1e-8), Some(below_one)),
        (f(-2.0f32.powi(-25)), Some(below_one)),
        (f(-2.0f32.powi(-24)), Some(below_one)),
        // Normal values.
        (f(0.0), Some(f(0.0))),
        (f(-0.0), Some(f(0.0))),
        (f(0.25), Some(f(0.25))),
        (f(3.75), Some(f(0.75))),
        (f(-0.25), Some(f(0.75))),
        (f(-3.25), Some(f(0.75))),
        (f(1.0), Some(f(0.0))),
        (f(-1.0), Some(f(0.0))),
        // Non-finite inputs yield NaN.
        (f(f32::NAN), None),
        (f(f32::INFINITY), None),
        (f(f32::NEG_INFINITY), None),
    ];

    let inputs: Vec<[u32; 4]> = cases.iter().map(|&(x, _)| [x, 0, 0, 0]).collect();
    run_cases(&shader, &inputs, |backend, i, inp, got| {
        let expect = cases[i].1;
        match expect {
            Some(want) => assert_eq!(
                got,
                want,
                "{backend}: v_fract_f32({:?} / {:#010x}) = {got:#010x} ({:?}), want {want:#010x}",
                f32::from_bits(inp[0]),
                inp[0],
                f32::from_bits(got),
            ),
            None => assert!(
                f32::from_bits(got).is_nan(),
                "{backend}: v_fract_f32({:?}) = {got:#010x}, want NaN",
                f32::from_bits(inp[0]),
            ),
        }
    });
}

#[test]
fn v_fract_f16_never_returns_one() {
    // v_fract_f16 v0, v1 (VOP1 opcode 0x05f)
    let shader = build_valu_cs(&[vop1(0x05f, 0, vgpr(1))]);
    let below_one = 0x3BFFu16; // 1.0 - 2^-11

    assert_eq!(f16_to_f32(0x9400), -2.0f32.powi(-10));
    assert_eq!(f16_to_f32(0x9000), -2.0f32.powi(-11));
    assert_eq!(f16_to_f32(0x8C00), -2.0f32.powi(-12));

    let cases: &[(u16, Option<u16>)] = &[
        // Negatives in (-2^-11, 0): must clamp to 0x3BFF, never 0x3C00 (1.0).
        (0x8C00, Some(below_one)), // -2^-12
        (0x8400, Some(below_one)), // -2^-14
        (0x8001, Some(below_one)), // -2^-24 (smallest negative f16 subnormal)
        (0x9000, Some(below_one)), // -2^-11 (exact 1 - 2^-11 = 0x3BFF)
        (0x9400, Some(0x3BFE)),    // -2^-10 (exact 1 - 2^-10 = 0x3BFE)
        // Normal values.
        (0x0000, Some(0x0000)), // 0.0 -> 0.0
        (0x3400, Some(0x3400)), // 0.25 -> 0.25
        (0xB400, Some(0x3A00)), // -0.25 -> 0.75
        (0x4300, Some(0x3800)), // 3.5 -> 0.5
        (0x3C00, Some(0x0000)), // 1.0 -> 0.0
        // Non-finite inputs yield NaN.
        (0x7E00, None), // NaN
        (0x7C00, None), // +inf
        (0xFC00, None), // -inf
    ];

    let inputs: Vec<[u32; 4]> = cases
        .iter()
        .map(|&(x, _)| [u32::from(x), 0, 0, 0])
        .collect();
    run_cases(&shader, &inputs, |backend, i, inp, got| {
        let expect = cases[i].1;
        let got_h = got as u16;
        match expect {
            Some(want) => assert_eq!(
                got_h, want,
                "{backend}: v_fract_f16({:#06x}) = {got_h:#06x}, want {want:#06x}",
                inp[0],
            ),
            None => assert!(
                is_f16_nan(got_h),
                "{backend}: v_fract_f16({:#06x}) = {got_h:#06x}, want NaN",
                inp[0],
            ),
        }
    });
}

/// RDNA2's flt16_to_snorm16 / flt16_to_unorm16: clamp, scale by 32767 or
/// 65535, round the exact product to nearest even; NaN gives 0.
fn cvt_norm16_reference(bits: u16, signed: bool) -> u16 {
    let x = f16_to_f32(bits);
    if x.is_nan() {
        return 0;
    }
    let (lo, scale) = if signed {
        (-1.0, 32767.0)
    } else {
        (0.0, 65535.0)
    };
    // f64 holds the product of an f16 and a 16-bit scale exactly.
    let p = f64::from(x.clamp(lo, 1.0)) * scale;
    p.round_ties_even() as i32 as u16
}

#[test]
fn v_cvt_norm_i16_and_u16_f16_round_the_exact_product() {
    // Every 251st f16 pattern, plus the values where an f32 product would
    // round before the rounding to an integer, ties, and the edges.
    let mut bits: Vec<u16> = (0..=u16::MAX).step_by(251).collect();
    bits.extend([
        0x0000, 0x8000, 0x0001, 0x8001, 0x03FF, 0x0400, 0x3800, 0xB800, 0x3801, 0xB801, 0x37FF,
        0x3BFF, 0xBBFF, 0x3C00, 0xBC00, 0x3C01, 0x7BFF, 0x7C00, 0xFC00, 0x7E00, 0xFE00, 0x7C01,
        0x3A9B, 0xBA9B, 0x2E66, 0x1400,
    ]);
    for (op, signed, what) in [
        (0x063u16, true, "v_cvt_norm_i16_f16"),
        (0x064, false, "v_cvt_norm_u16_f16"),
    ] {
        // v_cvt_norm_*16_f16 v0, v1 (VOP1); v0's high half starts at 0.
        let shader = build_valu_cs(&[vop1(0x001, 0, 128), vop1(op, 0, vgpr(1))]);
        let inputs: Vec<[u32; 4]> = bits.iter().map(|&b| [u32::from(b), 0, 0, 0]).collect();
        run_cases(&shader, &inputs, |backend, i, _, got| {
            let want = cvt_norm16_reference(bits[i], signed);
            assert_eq!(
                got,
                u32::from(want),
                "{backend}: {what}({:#06x}) = {got:#010x}, want {want:#06x}",
                bits[i],
            );
        });
    }
}

#[test]
fn v_sin_and_v_cos_f32_reduce_large_turns() {
    let sin_shader = build_valu_cs(&[vop1(0x035, 0, vgpr(1))]);
    let cos_shader = build_valu_cs(&[vop1(0x036, 0, vgpr(1))]);
    let f = f32::to_bits;
    let half_sqrt2 = std::f32::consts::FRAC_1_SQRT_2;

    let sin_cases: &[(f32, Option<f32>)] = &[
        (0.0, Some(0.0)),
        (0.25, Some(1.0)),
        (255.0, Some(0.0)),
        (255.25, Some(1.0)),
        (255.5, Some(0.0)),
        (-255.25, Some(-1.0)),
        (10_000.125, Some(half_sqrt2)),
        (f32::NAN, None),
        (f32::INFINITY, None),
        (f32::NEG_INFINITY, None),
    ];
    let sin_inputs: Vec<[u32; 4]> = sin_cases.iter().map(|&(x, _)| [f(x), 0, 0, 0]).collect();
    run_cases(&sin_shader, &sin_inputs, |backend, i, inp, got| {
        let expect = sin_cases[i].1;
        let got_f = f32::from_bits(got);
        match expect {
            Some(want) => assert!(
                (got_f - want).abs() < 1e-5,
                "{backend}: v_sin_f32({}) = {got_f}, want {want} (err = {})",
                f32::from_bits(inp[0]),
                (got_f - want).abs(),
            ),
            None => assert!(
                got_f.is_nan(),
                "{backend}: v_sin_f32({}) = {got_f}, want NaN",
                f32::from_bits(inp[0]),
            ),
        }
    });

    let cos_cases: &[(f32, Option<f32>)] = &[
        (0.0, Some(1.0)),
        (0.25, Some(0.0)),
        (255.0, Some(1.0)),
        (255.25, Some(0.0)),
        (255.5, Some(-1.0)),
        (-255.5, Some(-1.0)),
        (10_000.125, Some(half_sqrt2)),
        (f32::NAN, None),
        (f32::INFINITY, None),
        (f32::NEG_INFINITY, None),
    ];
    let cos_inputs: Vec<[u32; 4]> = cos_cases.iter().map(|&(x, _)| [f(x), 0, 0, 0]).collect();
    run_cases(&cos_shader, &cos_inputs, |backend, i, inp, got| {
        let expect = cos_cases[i].1;
        let got_f = f32::from_bits(got);
        match expect {
            Some(want) => assert!(
                (got_f - want).abs() < 1e-5,
                "{backend}: v_cos_f32({}) = {got_f}, want {want} (err = {})",
                f32::from_bits(inp[0]),
                (got_f - want).abs(),
            ),
            None => assert!(
                got_f.is_nan(),
                "{backend}: v_cos_f32({}) = {got_f}, want NaN",
                f32::from_bits(inp[0]),
            ),
        }
    });

    // Main's `trig_turns` preserves sin(-0.0) = -0.0 and produces exact +0.0 at
    // half-turns (sin) and quarter-turns (cos).
    let exact_sin: &[(u32, u32)] = &[
        (0x8000_0000, 0x8000_0000), // sin(-0.0) = -0.0
        (f(255.0), 0),              // sin(255.0) = +0.0
        (f(255.5), 0),              // sin(255.5) = +0.0
    ];
    let exact_sin_in: Vec<[u32; 4]> = exact_sin.iter().map(|&(x, _)| [x, 0, 0, 0]).collect();
    run_cases(&sin_shader, &exact_sin_in, |backend, i, inp, got| {
        let want = exact_sin[i].1;
        assert_eq!(
            got, want,
            "{backend}: v_sin_f32({:#010x}) = {got:#010x}, want {want:#010x}",
            inp[0],
        );
    });
    let exact_cos: &[(u32, u32)] = &[
        (f(0.25), 0),   // cos(0.25) = +0.0
        (f(255.25), 0), // cos(255.25) = +0.0
    ];
    let exact_cos_in: Vec<[u32; 4]> = exact_cos.iter().map(|&(x, _)| [x, 0, 0, 0]).collect();
    run_cases(&cos_shader, &exact_cos_in, |backend, i, inp, got| {
        let want = exact_cos[i].1;
        assert_eq!(
            got, want,
            "{backend}: v_cos_f32({:#010x}) = {got:#010x}, want {want:#010x}",
            inp[0],
        );
    });
}

#[test]
fn v_cvt_u16_and_i16_f16_saturate() {
    // v_mov_b32 v0, 0; v_cvt_u16_f16 / v_cvt_i16_f16 v0, v1 (VOP1 0x052 / 0x053).
    let cases: &[(u16, u16, u16)] = &[
        // (f16 input, u16 result, i16 result)
        (0x0000, 0, 0),          // 0
        (0x8000, 0, 0),          // -0
        (0x3E00, 1, 1),          // 1.5 -> 1 (toward zero)
        (0xBE00, 0, 0xFFFF),     // -1.5 -> 0 / -1
        (0x57D0, 125, 125),      // 125.0
        (0x7800, 32768, 32767),  // 32768 saturates i16
        (0x77FF, 32752, 32752),  // 32752, the largest f16 below 32768
        (0xF800, 0, 0x8000),     // -32768 fits i16
        (0xF801, 0, 0x8000),     // -32800 saturates i16
        (0x7BFF, 65504, 32767),  // max finite f16
        (0xFBFF, 0, 0x8000),     // -65504
        (0x7C00, 0xFFFF, 32767), // +inf
        (0xFC00, 0, 0x8000),     // -inf
        (0x7E00, 0, 0),          // NaN
        (0xFE00, 0, 0),          // -NaN
        (0x0001, 0, 0),          // smallest subnormal
    ];
    for (op, what, pick) in [
        (0x052u16, "v_cvt_u16_f16", 0usize),
        (0x053, "v_cvt_i16_f16", 1),
    ] {
        let shader = build_valu_cs(&[vop1(0x001, 0, 128), vop1(op, 0, vgpr(1))]);
        let inputs: Vec<[u32; 4]> = cases.iter().map(|c| [u32::from(c.0), 0, 0, 0]).collect();
        run_cases(&shader, &inputs, |backend, i, _, got| {
            let want = if pick == 0 { cases[i].1 } else { cases[i].2 };
            assert_eq!(
                got,
                u32::from(want),
                "{backend}: {what}({:#06x}) = {got:#010x}, want {want:#06x}",
                cases[i].0,
            );
        });
    }
}

#[test]
fn v_cvt_f16_u16_and_i16_widen_the_integer_first() {
    // v_mov_b32 v0, 0; v_cvt_f16_u16 / v_cvt_f16_i16 v0, v1 (VOP1 0x050 / 0x051).
    // Only integers f16 holds exactly: how the others round is the narrowing's
    // business, not this conversion's. `None`: the result depends on rounding.
    let cases: &[(u16, Option<u16>, Option<u16>)] = &[
        // (u16 / i16 input bits, f16 of the unsigned value, f16 of the signed value)
        (0x0000, Some(0x0000), Some(0x0000)), // 0
        (0x0001, Some(0x3C00), Some(0x3C00)), // 1.0
        (0x07FF, Some(0x67FF), Some(0x67FF)), // 2047, the last integer below 2048 that f16 holds
        (0x8000, Some(0x7800), Some(0xF800)), // 32768 / -32768
        (0xFC00, Some(0x7BE0), Some(0xE400)), // 64512 / -1024
        (0xFFFE, None, Some(0xC000)),         // 65534 does not fit f16 / -2
        (0xFFFF, None, Some(0xBC00)),         // 65535 does not fit f16 / -1
    ];
    for (op, what, pick) in [
        (0x050u16, "v_cvt_f16_u16", 0usize),
        (0x051, "v_cvt_f16_i16", 1),
    ] {
        let shader = build_valu_cs(&[vop1(0x001, 0, 128), vop1(op, 0, vgpr(1))]);
        let inputs: Vec<[u32; 4]> = cases.iter().map(|c| [u32::from(c.0), 0, 0, 0]).collect();
        run_cases(&shader, &inputs, |backend, i, _, got| {
            let want = if pick == 0 { cases[i].1 } else { cases[i].2 };
            let Some(want) = want else { return };
            assert_eq!(
                got,
                u32::from(want),
                "{backend}: {what}({:#06x}) = {got:#010x}, want {want:#06x}",
                cases[i].0,
            );
        });
    }
}

#[test]
fn s_lshl_add_shifts_the_first_operand_then_adds() {
    // s_lshl{1,2,3,4}_add_u32 s6, s0, s1 (SOP2 0x2e..0x31): (s0 << n) + s1.
    let values = [
        0u32,
        1,
        3,
        0x1000_0001,
        0x7FFF_FFFF,
        0x8000_0000,
        0xFFFF_FFFF,
    ];
    let mut cases = Vec::new();
    for a in values {
        for b in values {
            cases.push([a, b, 0, 0]);
        }
    }
    for n in 1..=4u32 {
        let shader = build_valu_cs(&[sop2(0x2d + n as u8, 6, 0, 1), vop1(0x001, 0, sgpr(6))]);
        run_cases(&shader, &cases, |backend, _, inp, got| {
            let want = (inp[0] << n).wrapping_add(inp[1]);
            assert_eq!(
                got, want,
                "{backend}: s_lshl{n}_add_u32({:#010x}, {:#010x}) = {got:#010x}, want {want:#010x}",
                inp[0], inp[1],
            );
        });
    }
}
#[test]
fn v_16_bit_integer_ops_match_rdna2() {
    // v_<op>_{u,i}16 v0, v1, v2 (VOP3 0x303..0x30c, 0x314): only the low
    // halves of the sources count, and the result is the low half of v0.
    // The 16-bit shifts take a four-bit amount from the first operand.
    type Pick = fn(u16, u16) -> u16;
    let ops: [(&str, u16, Pick); 10] = [
        ("v_add_nc_u16", 0x303, u16::wrapping_add),
        ("v_sub_nc_u16", 0x304, u16::wrapping_sub),
        ("v_mul_lo_u16", 0x305, u16::wrapping_mul),
        ("v_lshrrev_b16", 0x307, |a, b| b >> (a & 15)),
        ("v_ashrrev_i16", 0x308, |a, b| {
            ((b as i16) >> (a & 15)) as u16
        }),
        ("v_max_u16", 0x309, u16::max),
        ("v_max_i16", 0x30A, |a, b| (a as i16).max(b as i16) as u16),
        ("v_min_u16", 0x30B, u16::min),
        ("v_min_i16", 0x30C, |a, b| (a as i16).min(b as i16) as u16),
        ("v_lshlrev_b16", 0x314, |a, b| b << (a & 15)),
    ];
    let values = [0u16, 1, 5, 16, 17, 0x7FFF, 0x8000, 0xFFFF];
    let mut cases = Vec::new();
    for a in values {
        for b in values {
            // Garbage in the high halves, which must not matter.
            cases.push([0xABCD_0000 | u32::from(a), 0x1234_0000 | u32::from(b), 0, 0]);
        }
    }
    for (name, op, pick) in ops {
        let shader = build_valu_cs(&[
            vop1(0x001, 0, 128),
            vop3(op, 0, vgpr(1), vgpr(2), 0)[0],
            vop3(op, 0, vgpr(1), vgpr(2), 0)[1],
        ]);
        run_cases(&shader, &cases, |backend, _, inp, got| {
            let (a, b) = (inp[0] as u16, inp[1] as u16);
            let want = pick(a, b);
            assert_eq!(
                got & 0xFFFF,
                u32::from(want),
                "{backend}: {name}({a:#06x}, {b:#06x}) = {got:#010x}, want {want:#06x}",
            );
        });
    }
}

#[test]
fn s_cmpk_compares_against_the_extended_immediate() {
    // s_cmpk_<cond>_<type> s0, imm; s_cselect_b32 s6, 1, 0 (SOP2 0x0a, on SCC);
    // v_mov_b32 v0, s6. The signed forms sign-extend the 16-bit immediate, the
    // unsigned forms zero-extend it.
    type Cond = fn(i64, i64) -> bool;
    let conds: [(&str, Cond); 6] = [
        ("eq", |a, b| a == b),
        ("lg", |a, b| a != b),
        ("gt", |a, b| a > b),
        ("ge", |a, b| a >= b),
        ("lt", |a, b| a < b),
        ("le", |a, b| a <= b),
    ];
    // An immediate below and one at the sign bit; operands around each extension.
    let imms: [u16; 2] = [0x0005, 0x8000];
    let operands: [u32; 9] = [
        0,
        4,
        5,
        6,
        0x7FFF,
        0x8000,
        0xFFFF_8000,
        0xFFFF_FFFF,
        0x8000_0000,
    ];
    for (signed, first_op) in [(true, 0x003u8), (false, 0x009)] {
        for (k, (cond, holds)) in conds.iter().enumerate() {
            for imm in imms {
                let shader = build_valu_cs(&[
                    sopk(first_op + k as u8, 0, imm),
                    sop2(0x0a, 6, 129, 128),
                    vop1(0x001, 0, sgpr(6)),
                ]);
                let name = format!("s_cmpk_{cond}_{}32", if signed { "i" } else { "u" });
                let inputs: Vec<[u32; 4]> = operands.iter().map(|&v| [v, 0, 0, 0]).collect();
                run_cases(&shader, &inputs, |backend, i, _, got| {
                    let (a, b) = if signed {
                        (i64::from(operands[i] as i32), i64::from(imm as i16))
                    } else {
                        (i64::from(operands[i]), i64::from(imm))
                    };
                    let want = u32::from(holds(a, b));
                    assert_eq!(
                        got, want,
                        "{backend}: {name}({:#010x}, {imm:#06x}) = {got}, want {want}",
                        operands[i],
                    );
                });
            }
        }
    }
}

#[test]
fn v_sin_and_v_cos_f16_reduce_large_turns() {
    let sin_shader = build_valu_cs(&[vop1(0x060, 0, vgpr(1))]);
    let cos_shader = build_valu_cs(&[vop1(0x061, 0, vgpr(1))]);

    // Sanity-check the f16 bit encodings of large turns.
    assert_eq!(f16_to_f32(0x5BF8), 255.0);
    assert_eq!(f16_to_f32(0x5BFA), 255.25);
    assert_eq!(f16_to_f32(0x5BFC), 255.5);
    assert_eq!(f16_to_f32(0x7BFF), 65504.0);

    let sin_cases: &[(u16, Option<f32>)] = &[
        (0x5BF8, Some(0.0)),  // 255.0
        (0x5BFA, Some(1.0)),  // 255.25
        (0x5BFC, Some(0.0)),  // 255.5
        (0xDBFA, Some(-1.0)), // -255.25
        (0x7BFF, Some(0.0)),  // 65504.0 (largest finite f16, exact integer)
        (0x7E00, None),       // NaN
        (0x7C00, None),       // +inf
        (0xFC00, None),       // -inf
    ];
    let sin_inputs: Vec<[u32; 4]> = sin_cases
        .iter()
        .map(|&(x, _)| [u32::from(x), 0, 0, 0])
        .collect();
    run_cases(&sin_shader, &sin_inputs, |backend, i, inp, got| {
        let expect = sin_cases[i].1;
        let got_h = got as u16;
        match expect {
            Some(want) => {
                let got_f = f16_to_f32(got_h);
                assert!(
                    (got_f - want).abs() < 1e-3,
                    "{backend}: v_sin_f16({:#06x} = {}) = {got_f} ({got_h:#06x}), want {want}",
                    inp[0],
                    f16_to_f32(inp[0] as u16),
                );
            }
            None => assert!(
                is_f16_nan(got_h),
                "{backend}: v_sin_f16({:#06x}) = {got_h:#06x}, want NaN",
                inp[0],
            ),
        }
    });

    let cos_cases: &[(u16, Option<f32>)] = &[
        (0x5BF8, Some(1.0)),  // 255.0
        (0x5BFA, Some(0.0)),  // 255.25
        (0x5BFC, Some(-1.0)), // 255.5
        (0x7BFF, Some(1.0)),  // 65504.0 (largest finite f16, exact integer)
        (0x7E00, None),       // NaN
        (0x7C00, None),       // +inf
        (0xFC00, None),       // -inf
    ];
    let cos_inputs: Vec<[u32; 4]> = cos_cases
        .iter()
        .map(|&(x, _)| [u32::from(x), 0, 0, 0])
        .collect();
    run_cases(&cos_shader, &cos_inputs, |backend, i, inp, got| {
        let expect = cos_cases[i].1;
        let got_h = got as u16;
        match expect {
            Some(want) => {
                let got_f = f16_to_f32(got_h);
                assert!(
                    (got_f - want).abs() < 1e-3,
                    "{backend}: v_cos_f16({:#06x} = {}) = {got_f} ({got_h:#06x}), want {want}",
                    inp[0],
                    f16_to_f32(inp[0] as u16),
                );
            }
            None => assert!(
                is_f16_nan(got_h),
                "{backend}: v_cos_f16({:#06x}) = {got_h:#06x}, want NaN",
                inp[0],
            ),
        }
    });
}

/// A 32-bit literal in a signed 64-bit operand is sign-extended (an
/// unsigned one's is zero-extended): -100 compared with `v[1:2]`, and
/// shifted by `s_ashr_i64`.
#[test]
fn signed_64_bit_operands_sign_extend_their_literal() {
    const MINUS_100: u32 = -100_i32 as u32;
    let check = |name: &str, body: &[u32], cases: &[([u32; 4], u32)]| {
        let shader = build_valu_cs(body);
        let inputs: Vec<[u32; 4]> = cases.iter().map(|c| c.0).collect();
        run_cases(&shader, &inputs, |backend, i, inp, got| {
            let want = cases[i].1;
            assert_eq!(got, want, "{backend}: {name} {inp:x?}");
        });
    };
    // v_cmp_*_i64 s6, v[1:2], -100 (VOP3, literal src1), then
    // v_cndmask_b32 v0, 0, 1, s6.
    let compare = |op: u16| {
        let [a, b] = vop3(op, 6, vgpr(1), 255, 0);
        let [c, d] = vop3(0x101, 0, 0x80, 0x81, sgpr(6));
        [a, b, MINUS_100, c, d]
    };
    // An i64 as the two input dwords of `v[1:2]`.
    let pair = |n: i64| [n as u32, (n >> 32) as u32, 0, 0];
    let eq = [(pair(-100), 1), ([MINUS_100, 0, 0, 0], 0), (pair(0), 0)];
    check("v_cmp_eq_i64", &compare(0x0A2), &eq);
    let lt = [(pair(-200), 1), (pair(-50), 0), (pair(0), 0)];
    check("v_cmp_lt_i64", &compare(0x0A1), &lt);
    // s_ashr_i64 s[6:7], -100, 4; v_mov_b32 v0, s7: the high dword.
    let ashr = [sop2(0x23, 6, 255, 0x84), MINUS_100, vop1(0x001, 0, sgpr(7))];
    check("s_ashr_i64", &ashr, &[([0; 4], u32::MAX)]);
}

/// `v_mad_i64_i32` sign-extends a 32-bit literal in `src2` (`S2.i64`), while
/// `v_mad_u64_u32` zero-extends it (`S2.u64`).
#[test]
fn mad_i64_i32_sign_extends_literal_addend() {
    const MINUS_100: u32 = -100_i32 as u32;
    // VOP3B `v_mad_* v[5:6], s6, v1, v2, -100`; return the high dword `v6`.
    let high_dword = |op: u16| {
        let [a, b] = vop3(op, 5, vgpr(1), vgpr(2), 255);
        [a | 6 << 8, b, MINUS_100, vop1(0x001, 0, vgpr(6))]
    };
    let check = |name: &str, op: u16, cases: &[([u32; 4], u32)]| {
        let shader = build_valu_cs(&high_dword(op));
        let inputs: Vec<[u32; 4]> = cases.iter().map(|c| c.0).collect();
        run_cases(&shader, &inputs, |backend, i, inp, got| {
            assert_eq!(got, cases[i].1, "{backend}: {name} {inp:x?}");
        });
    };
    check(
        "v_mad_i64_i32",
        0x177,
        &[
            ([0, 0, 0, 0], u32::MAX),
            ([10, 10, 0, 0], 0),
            ([(-100_000_i32) as u32, 100_000, 0, 0], 0xFFFF_FFFD),
        ],
    );
    check(
        "v_mad_u64_u32",
        0x176,
        &[([0, 0, 0, 0], 0), ([10, 10, 0, 0], 1)],
    );
}

#[test]
fn bfe_signed_and_unsigned_match_rdna2() {
    // v_bfe_i32 v0, v1, v2, v3 (VOP3 opcode 0x149)
    let v_signed = build_valu_cs(&vop3(0x149, 0, vgpr(1), vgpr(2), vgpr(3)));
    // v_bfe_u32 v0, v1, v2, v3 (VOP3 opcode 0x148)
    let v_unsigned = build_valu_cs(&vop3(0x148, 0, vgpr(1), vgpr(2), vgpr(3)));
    // s_bfe_i32 s6, s0, s1 (SOP2 opcode 0x28: s1 = off[4:0] | width[22:16]); v_mov_b32 v0, s6
    let s_signed = build_valu_cs(&[sop2(0x28, 6, 0, 1), vop1(0x001, 0, sgpr(6))]);
    // s_bfe_u32 s6, s0, s1 (SOP2 opcode 0x27: s1 = off[4:0] | width[22:16]); v_mov_b32 v0, s6
    let s_unsigned = build_valu_cs(&[sop2(0x27, 6, 0, 1), vop1(0x001, 0, sgpr(6))]);
    // SCC = (result != 0), consumed by s_cselect_b32 s7, s2, s3.
    let s_signed_scc = build_valu_cs(&[
        sop2(0x28, 6, 0, 1),
        sop2(0x0A, 7, 2, 3),
        vop1(0x001, 0, sgpr(7)),
    ]);
    let s_unsigned_scc = build_valu_cs(&[
        sop2(0x27, 6, 0, 1),
        sop2(0x0A, 7, 2, 3),
        vop1(0x001, 0, sgpr(7)),
    ]);

    let signed_cases: &[([u32; 4], u32)] = &[
        // Field crosses bit 31: extends from bit 31.
        ([0xd2f9_0159, 28, 16, 0], 0xffff_fffd),
        ([0x8000_0000, 31, 4, 0], 0xffff_ffff),
        ([0x7000_0000, 28, 8, 0], 0x0000_0007),
        // Normal fields inside 32 bits.
        ([0x0000_0080, 4, 4, 0], 0xffff_fff8),
        ([0x0000_0070, 4, 4, 0], 0x0000_0007),
        // Edge widths: 0 yields 0; v_bfe masks width to 5 bits (32 -> 0),
        // while 31 extracts all remaining bits from offset 4.
        ([0xffff_ffff, 4, 0, 0], 0),
        ([0x8000_0000, 4, 31, 0], 0xf800_0000),
    ];
    let v_in: Vec<[u32; 4]> = signed_cases.iter().map(|c| c.0).collect();
    run_cases(&v_signed, &v_in, |backend, i, inp, got| {
        let want = signed_cases[i].1;
        assert_eq!(
            got, want,
            "{backend}: v_bfe_i32({:#010x}, {}, {}) = {got:#010x}, want {want:#010x}",
            inp[0], inp[1], inp[2],
        );
    });
    // v_bfe_{i32,u32} width is 5-bit (src2[4:0]), so width=32 wraps to 0.
    for (name, shader) in [("v_bfe_i32", &v_signed), ("v_bfe_u32", &v_unsigned)] {
        run_cases(shader, &[[0x8000_0000, 4, 32, 0]], |backend, _, _, got| {
            assert_eq!(got, 0, "{backend}: {name} width 32 should mask to 0");
        });
    }

    const WHEN_NZ: u32 = 0xAAAA_5555;
    const WHEN_ZERO: u32 = 0x1111_2222;
    let pack = |off: u32, width: u32| (off & 31) | ((width & 0x7F) << 16);

    // s_bfe_i32 width is 7-bit (src1[22:16]), so widths >= 32 keep all remaining bits
    // with sign extension from bit 31 (including width=64 where width & 31 == 0).
    let mut s_cases = signed_cases.to_vec();
    s_cases.extend([
        ([0x8000_0000, 4, 32, 0], 0xf800_0000),
        ([0x8000_0000, 0, 32, 0], 0x8000_0000),
        ([0x8000_0000, 4, 64, 0], 0xf800_0000),
        ([0xd2f9_0159, 28, 127, 0], 0xffff_fffd),
        ([0x0000_000f, 4, 32, 0], 0),
    ]);
    let mut s_in: Vec<[u32; 4]> = s_cases
        .iter()
        .map(|&(inp, _)| [inp[0], pack(inp[1], inp[2]), WHEN_NZ, WHEN_ZERO])
        .collect();
    // Unused bits src1[15:5] and src1[31:23] are ignored by hardware.
    s_cases.push(([0x8000_0000, 4, 32, 0], 0xf800_0000));
    s_in.push([0x8000_0000, 0xFF80_FFE0 | pack(4, 32), WHEN_NZ, WHEN_ZERO]);
    s_cases.push(([0xffff_ffff, 4, 0, 0], 0));
    s_in.push([0xffff_ffff, 0xFF80_FFE4, WHEN_NZ, WHEN_ZERO]);
    run_cases(&s_signed, &s_in, |backend, i, inp, got| {
        let want = s_cases[i].1;
        assert_eq!(
            got, want,
            "{backend}: s_bfe_i32({:#010x}, {:#010x}) = {got:#010x}, want {want:#010x}",
            inp[0], inp[1],
        );
    });
    run_cases(&s_signed_scc, &s_in, |backend, i, inp, got| {
        let want = if s_cases[i].1 != 0 {
            WHEN_NZ
        } else {
            WHEN_ZERO
        };
        assert_eq!(
            got, want,
            "{backend}: s_bfe_i32 SCC({:#010x}, {:#010x}) = {got:#010x}, want {want:#010x}",
            inp[0], inp[1],
        );
    });

    let unsigned_cases: &[([u32; 4], u32)] = &[
        ([0xd2f9_0159, 28, 16, 0], 0x0000_000d),
        ([0x8000_0000, 31, 4, 0], 0x0000_0001),
        ([0x0000_0080, 4, 4, 0], 0x0000_0008),
        ([0xffff_ffff, 4, 0, 0], 0),
        ([0x8000_0000, 4, 31, 0], 0x0800_0000),
    ];
    let u_in: Vec<[u32; 4]> = unsigned_cases.iter().map(|c| c.0).collect();
    run_cases(&v_unsigned, &u_in, |backend, i, inp, got| {
        let want = unsigned_cases[i].1;
        assert_eq!(
            got, want,
            "{backend}: v_bfe_u32({:#010x}, {}, {}) = {got:#010x}, want {want:#010x}",
            inp[0], inp[1], inp[2],
        );
    });

    // s_bfe_u32 width is 7-bit (src1[22:16]): widths >= 32 shift without sign extension.
    let mut su_cases = unsigned_cases.to_vec();
    su_cases.extend([
        ([0x8000_0000, 4, 32, 0], 0x0800_0000),
        ([0x8000_0000, 0, 32, 0], 0x8000_0000),
        ([0x8000_0000, 4, 64, 0], 0x0800_0000),
        ([0xd2f9_0159, 28, 127, 0], 0x0000_000d),
        ([0x0000_000f, 4, 32, 0], 0),
    ]);
    let mut su_in: Vec<[u32; 4]> = su_cases
        .iter()
        .map(|&(inp, _)| [inp[0], pack(inp[1], inp[2]), WHEN_NZ, WHEN_ZERO])
        .collect();
    su_cases.push(([0x8000_0000, 4, 32, 0], 0x0800_0000));
    su_in.push([0x8000_0000, 0xFF80_FFE0 | pack(4, 32), WHEN_NZ, WHEN_ZERO]);
    su_cases.push(([0xffff_ffff, 4, 0, 0], 0));
    su_in.push([0xffff_ffff, 0xFF80_FFE4, WHEN_NZ, WHEN_ZERO]);
    run_cases(&s_unsigned, &su_in, |backend, i, inp, got| {
        let want = su_cases[i].1;
        assert_eq!(
            got, want,
            "{backend}: s_bfe_u32({:#010x}, {:#010x}) = {got:#010x}, want {want:#010x}",
            inp[0], inp[1],
        );
    });
    run_cases(&s_unsigned_scc, &su_in, |backend, i, inp, got| {
        let want = if su_cases[i].1 != 0 {
            WHEN_NZ
        } else {
            WHEN_ZERO
        };
        assert_eq!(
            got, want,
            "{backend}: s_bfe_u32 SCC({:#010x}, {:#010x}) = {got:#010x}, want {want:#010x}",
            inp[0], inp[1],
        );
    });

    // Literal src1 (`0xFF` followed by 32-bit immediate) for both s_bfe_u32 and s_bfe_i32:
    // offset=4 (with ignored bits [15:5]), width=32/64/0 (with ignored bits [31:23]).
    let literals: [(u8, u32, u32, u32, u32); 6] = [
        (0x27, 0x8020_FFE4, 0x8000_0000, 0x0800_0000, WHEN_NZ),
        (0x28, 0x8020_FFE4, 0x8000_0000, 0xf800_0000, WHEN_NZ),
        (0x27, 0x0040_001C, 0xd2f9_0159, 0x0000_000d, WHEN_NZ),
        (0x28, 0x0040_001C, 0xd2f9_0159, 0xffff_fffd, WHEN_NZ),
        (0x27, 0xFF80_FFE4, 0xffff_ffff, 0, WHEN_ZERO),
        (0x28, 0xFF80_FFE4, 0xffff_ffff, 0, WHEN_ZERO),
    ];
    for (op, lit, src0, want, want_scc) in literals {
        let val_cs = build_valu_cs(&[sop2(op, 6, 0, 255), lit, vop1(0x001, 0, sgpr(6))]);
        run_cases(&val_cs, &[[src0, 0, 0, 0]], |backend, _, _, got| {
            assert_eq!(
                got, want,
                "{backend}: literal sop2({op:#x}, {src0:#010x}, {lit:#010x}) = {got:#010x}, want {want:#010x}",
            );
        });
        let scc_cs = build_valu_cs(&[
            sop2(op, 6, 0, 255),
            lit,
            sop2(0x0A, 7, 2, 3),
            vop1(0x001, 0, sgpr(7)),
        ]);
        run_cases(
            &scc_cs,
            &[[src0, 0, WHEN_NZ, WHEN_ZERO]],
            |backend, _, _, got| {
                assert_eq!(
                    got, want_scc,
                    "{backend}: literal sop2({op:#x}) SCC({src0:#010x}, {lit:#010x}) = {got:#010x}",
                );
            },
        );
    }
}

/// Lowers a wave32 compute program of 32 threads (with `validate`,
/// validated when `dxil.dll` is there): a root constant in s0 (slot 0), view and sampler
/// tables at s4 and s6 (slots 1 and 2); `used` lists (table user SGPR,
/// kind, table offset in dwords, descriptor dwords).
fn lower_cs32(words: &[u32], used: &[(u8, ResourceType, u32, u8)], validate: bool) -> HostShader {
    let shader = try_lower_cs32(words, used).expect("lower");
    if let (true, Ok(mut v)) = (
        validate,
        validator::Validator::load(validator::Validator::default_path().as_deref()),
    ) {
        v.validate(&mut shader.dxil.clone())
            .expect("DXIL validation failed");
    }
    shader
}

/// [`lower_cs32`] without validation, returning what the lowering refuses.
fn try_lower_cs32(
    words: &[u32],
    used: &[(u8, ResourceType, u32, u8)],
) -> Result<HostShader, crate::ShaderError> {
    let param = |root, user_data, slot| RootParam {
        root,
        table_offset_dwords: 0,
        sgprs: 1,
        base_register: 0,
        user_data,
        slot,
    };
    let native = NativeProgram {
        stage: HwStage::Cs,
        wave64: false,
        vgprs: 32,
        layout: vec![
            param(RootType::Constants, 0, 0),
            param(RootType::Table, 4, 1),
            param(RootType::Table, 6, 2),
        ],
        used: used
            .iter()
            .map(
                |&(user_data, kind, table_offset_dwords, dwords)| UsedResource {
                    kind,
                    root: RootType::Table,
                    table_offset_dwords,
                    lower_bound: 0,
                    stride: 0,
                    dwords,
                    user_data,
                    slot: user_data / 2 - 1,
                },
            )
            .collect(),
        io: StageIo::Cs(ComputeIo {
            rsrc2: 7 << 1,
            threads: [32, 1, 1],
        }),
        code: words.iter().flat_map(|w| w.to_le_bytes()).collect(),
    };
    let p = ir::build(&native.code, native.wave64, native.vgprs).expect("ir::build");
    let st = structure::structure(&p).expect("structure");
    let res = resource::track(&p, &native.layout, native.stage, Some(&st)).expect("track");
    let io = io::raise(&p, &native).expect("io::raise");
    let lowered = lower::lower(&lower::Parts {
        native: &native,
        p: &p,
        st: &st,
        io: &io,
        res: &res,
        inputs: Vec::new(),
        outputs: Vec::new(),
        prims: Vec::new(),
        mesh_stride: 0,
        target: super::Target::ANY,
        view_marks: false,
    })?;
    Ok(HostShader {
        wave_size: lowered.wave_size,
        stage: HwStage::Cs,
        dxil: lowered.dxil,
        user_data: lowered.user_data,
        samplers: lowered.samplers,
        static_samplers: Vec::new(),
        root_cbvs: lowered.root_cbvs,
        root_srvs: lowered.root_srvs,
        root_uavs: lowered.root_uavs,
        memory: lowered.memory,
        memory_writes: lowered.memory_writes,
        view_addresses: lowered.view_addresses,
        heap: lowered.heap,
        heap_writes: lowered.heap_writes,
        view_marks: lowered.view_marks,
    })
}

/// User data for [`lower_cs32`]'s programs: the root constant `k`, both
/// tables from heap index 0.
fn user_data_cs32(shader: &HostShader, k: u32) -> Vec<u32> {
    let mut ud = vec![0u32; shader.user_data_dwords().max(1) as usize];
    for u in &shader.user_data {
        ud[u.index as usize] = match (u.root, u.slot) {
            (RootType::Constants, 0) => k,
            (RootType::Table, 1 | 2) => 0,
            other => panic!("unexpected user data {other:?}"),
        };
    }
    ud
}

/// Image addresses with `a16` after a whole-VGPR operand (the offset or
/// bias keeps a VGPR; the coordinates share the next) and the clamp of
/// `image_sample_d_cl` (after the coordinates). Program 1: a biased sample
/// of 16-bit coordinates whose f16 bias is the root constant's low half.
/// Program 2: a gather with an offset of 16-bit and of 32-bit coordinates
/// (equal results), and a sample with gradients and the root constant as
/// clamp. The samples read a 32x32 texture whose level `n` holds `n`, at
/// level 1 by the derivatives, with a point sampler.
#[test]
fn image_addresses_of_16_bits_and_clamps() {
    use windows::Win32::Graphics::Direct3D12::*;
    let (v0, v3, v4) = (vgpr(0), vgpr(3), vgpr(4));
    // Descriptors: s[8:15] the levels texture (heap index 0), s[20:23] the
    // raw UAV (1), s[24:31] the 4x4 texture (2), s[16:19] the sampler.
    let mut prefix = vec![s_mov_b32(5, 0x80), s_mov_b32(7, 0x80)];
    for (op, dst, base, offset) in [
        (3u32, 8u32, 4u32, 0u32),
        (2, 20, 4, 0x20),
        (3, 24, 4, 0x40),
        (2, 16, 6, 0),
    ] {
        prefix.extend([
            0x3D << 26 | op << 18 | dst << 6 | base >> 1,
            125 << 25 | offset,
        ]);
    }
    let used = [
        (4, ResourceType::Srv, 0, 8),
        (4, ResourceType::Uav, 8, 4),
        (4, ResourceType::Srv, 16, 8),
        (6, ResourceType::Sampler, 0, 4),
    ];

    // A 2x2 quad per four lanes: u = (4 + (l & 1)) / 16, v = (4 + (l >> 1 & 1)) / 16.
    let mut biased = prefix.clone();
    biased.push(0x1B << 25 | 3 << 17 | (v0 as u32 & 0xFF) << 9 | 129); // v_and_b32 v3, 1, v0
    biased.extend(vop3(0x148, 4, v0, 129, 129)); // v_bfe_u32 v4, v0, 1, 1
    biased.push(0x25 << 25 | 3 << 17 | (v3 as u32 & 0xFF) << 9 | 132); // v_add_nc_u32 v3, 4, v3
    biased.push(0x25 << 25 | 4 << 17 | (v4 as u32 & 0xFF) << 9 | 132); // v_add_nc_u32 v4, 4, v4
    biased.extend([vop1(0x006, 3, v3), vop1(0x006, 4, v4)]); // v_cvt_f32_u32
    // v_mul_f32 v3, 1/16, v3; v_mul_f32 v4, 1/16, v4
    biased.extend([
        0x08 << 25 | 3 << 17 | (v3 as u32 & 0xFF) << 9 | 255,
        0x3D80_0000,
    ]);
    biased.extend([
        0x08 << 25 | 4 << 17 | (v4 as u32 & 0xFF) << 9 | 255,
        0x3D80_0000,
    ]);
    biased.push(0x2F << 25 | 2 << 17 | (v4 as u32 & 0xFF) << 9 | v3 as u32); // v_cvt_pkrtz_f16_f32 v2, v3, v4
    biased.push(vop1(0x001, 1, sgpr(0)));
    // image_sample_b v[5:8], v[1:2], s[8:15], s[16:19] dmask:0xf dim:2D a16
    biased.extend([0xF094_0F08, 1 | 5 << 8 | 2 << 16 | 4 << 21 | 1 << 30]);
    biased.extend(buffer_store_dword(5, 20, 0));
    biased.push(S_ENDPGM);
    // The 1.6 validator refuses implicit derivatives in compute (1.8 takes
    // them); the host runs them.
    let biased = lower_cs32(&biased, &used, false);

    let mut gathers = prefix;
    // Offset (1, 0); coordinates (0.375, 0.625) as f16 in v2 and as f32 in
    // v[11:12]: image_gather4_lz_o v[13:16], v[1:2], s[24:31], s[16:19]
    // dmask:0x1 dim:2D a16, and v[17:20], v[10:12] without a16.
    gathers.extend([vop1(0x001, 1, 129), vop1(0x001, 2, 255), 0x3900_3600]);
    gathers.extend([vop1(0x001, 10, 129), vop1(0x001, 11, 255), 0x3EC0_0000]);
    gathers.extend([vop1(0x001, 12, 255), 0x3F20_0000]);
    gathers.extend([0xF15C_0108, 1 | 13 << 8 | 6 << 16 | 4 << 21 | 1 << 30]);
    gathers.extend([0xF15C_0108, 10 | 17 << 8 | 6 << 16 | 4 << 21]);
    for k in 0..8u8 {
        gathers.extend(buffer_store_dword(13 + k, 20, 4 * u16::from(k)));
    }
    // Gradients (1/16, 0) and (0, 1/16), coordinates (1/4, 1/4), clamp s0:
    // image_sample_d_cl v[27:30], v[20:26], s[8:15], s[16:19] dmask:0xf dim:2D
    gathers.extend([vop1(0x001, 20, 255), 0x3D80_0000]);
    gathers.extend([vop1(0x001, 21, 128), vop1(0x001, 22, 128)]);
    gathers.extend([vop1(0x001, 23, 255), 0x3D80_0000]);
    gathers.extend([vop1(0x001, 24, 255), 0x3E80_0000]);
    gathers.extend([vop1(0x001, 25, 255), 0x3E80_0000]);
    gathers.push(vop1(0x001, 26, sgpr(0)));
    gathers.extend([0xF08C_0F08, 20 | 27 << 8 | 2 << 16 | 4 << 21]);
    gathers.extend(buffer_store_dword(27, 20, 32));
    gathers.push(S_ENDPGM);
    let gathers = lower_cs32(&gathers, &used, true);

    let samplers = [D3D12_SAMPLER_DESC {
        Filter: D3D12_FILTER_MIN_MAG_MIP_POINT,
        AddressU: D3D12_TEXTURE_ADDRESS_MODE_CLAMP,
        AddressV: D3D12_TEXTURE_ADDRESS_MODE_CLAMP,
        AddressW: D3D12_TEXTURE_ADDRESS_MODE_CLAMP,
        MaxAnisotropy: 1,
        MaxLOD: f32::MAX,
        ..Default::default()
    }];
    let texels: Vec<[f32; 4]> = (0..16).map(|i| [i as f32, 0.0, 0.0, 1.0]).collect();
    let views = [
        View::Levels(32, 6),
        View::RawUav(9),
        View::Texture(4, 4, &texels),
    ];
    let f = |x: f32| x.to_bits();
    // Texels (2, 3), (3, 3), (3, 2), (2, 2) twice, then the level.
    let gathered = |level| [14.0, 15.0, 11.0, 10.0, 14.0, 15.0, 11.0, 10.0, level];
    let cases: [(&HostShader, u32, &[f32]); 9] = [
        (&biased, 0x3C00, &[2.0]),      // bias 1
        (&biased, 0xABCD_BC00, &[0.0]), // bias -1, high half ignored
        (&biased, 0x4000, &[3.0]),      // bias 2
        (&biased, 0, &[1.0]),           // bias 0
        (&biased, f(2.0), &[1.0]),      // low half 0
        (&gathers, 0, &gathered(1.0)),  // clamp 0
        (&gathers, f(2.0), &gathered(2.0)),
        (&gathers, f(3.0), &gathered(3.0)),
        (&gathers, f(-1.0), &gathered(1.0)),
    ];
    let _one = ONE_AT_A_TIME.lock().unwrap_or_else(|e| e.into_inner());
    for (label, warp) in [("hardware", false), ("warp", true)] {
        let Ok(host) = Host::new(warp) else {
            eprintln!("no SM 6.6 {label} device; skipped");
            continue;
        };
        for (shader, k, want) in cases {
            let (root, pso) = host
                .compute_pipeline(shader)
                .unwrap_or_else(|e| panic!("{label}: {e}"));
            let out = host
                .dispatch(
                    &root,
                    &pso,
                    &Dispatch {
                        roots: vec![(0, Root::Constants(user_data_cs32(shader, k)))],
                        views: &views,
                        samplers: &samplers,
                        groups: [1, 1, 1],
                    },
                )
                .unwrap_or_else(|e| panic!("{label}: {e}"));
            let got: Vec<f32> = out[0][..want.len()]
                .iter()
                .map(|&x| f32::from_bits(x))
                .collect();
            assert_eq!(got, want, "{label}: {k:#x}");
        }
    }
}

/// MIMG `op v[vdata], v[vaddr:...], s[8:15], s[16:19] dmask:0x1` of this
/// dimension (0 1D, 1 2D, 2 3D), opcode bit 7 in bit 0.
const fn mimg_sample(op: u32, vdata: u8, vaddr: u8, dim: u32, a16: bool) -> [u32; 2] {
    [
        0xF000_0000 | (op & 0x7F) << 18 | 1 << 8 | dim << 3 | op >> 7,
        vaddr as u32 | (vdata as u32) << 8 | 2 << 16 | 4 << 21 | (a16 as u32) << 30,
    ]
}

const GRADIENT_BINDINGS: [(u8, ResourceType, u32, u8); 3] = [
    (4, ResourceType::Srv, 0, 8),
    (4, ResourceType::Uav, 8, 4),
    (6, ResourceType::Sampler, 0, 4),
];

fn gradient_sample_program(op: u32, dim: u32, a16: bool, addr: &[u32]) -> Vec<u32> {
    // Descriptors: s[8:15] the texture (heap index 0), s[20:23] the raw UAV
    // (1), s[16:19] the sampler.
    let mut w = vec![s_mov_b32(5, 0x80), s_mov_b32(7, 0x80)];
    for (op, dst, base, offset) in [(3u32, 8u32, 4u32, 0u32), (2, 20, 4, 0x20), (2, 16, 6, 0)] {
        w.extend([
            0x3D << 26 | op << 18 | dst << 6 | base >> 1,
            125 << 25 | offset,
        ]);
    }
    for (k, &x) in addr.iter().enumerate() {
        w.extend([vop1(0x001, k as u8, 255), x]);
    }
    w.extend(mimg_sample(op, 20, 0, dim, a16));
    w.extend(buffer_store_dword(20, 20, 0));
    w.push(S_ENDPGM);
    w
}

/// Gradient samples with 16-bit (`image_sample_d_g16`, `_d_cl_g16`) and
/// 32-bit derivatives, each with 32-bit and 16-bit (`a16`) coordinates:
/// the two are independent on RDNA2, `a16` packing the coordinates and
/// clamp, `_g16` the derivatives, a direction per VGPR (LLVM's
/// `gfx10_asm_mimg.s`: 2D `image_sample_d` with a16 takes 5 VGPRs,
/// `image_sample_d_g16` without it 4). The texture is 64x16 with 5 levels,
/// texel (x, y) of level n holding 1000 n + 64 y + x, under a point
/// sampler. The gradients (1/16, 0) and (0, 19/32) span 4 and 9.5 texels:
/// level 3.25, so level 3 (5.25 with a direction's halves swapped; 32-bit
/// gradients read as 16-bit ones give level 4), where (u, v) = (0.3125,
/// 0.75) is texel (2, 1). A clamp of 4 picks level 4, texel (1, 0). A
/// comparison sample of 16-bit gradients is refused as the 32-bit one is
/// (SM 6.6 has no `SampleCmpGrad`). A volume with tagged texels also
/// checks the third derivative component in each gradient direction.
#[test]
fn gradient_samples_of_16_bits() {
    use windows::Win32::Graphics::Direct3D12::*;
    let program = gradient_sample_program;
    let used = GRADIENT_BINDINGS;
    // Gradients x then y, then the coordinates and clamp; as f16, 1/16,
    // 19/32, 0.3125, 0.75 and 4 are 0x2C00, 0x38C0, 0x3500, 0x3A00 and
    // 0x4400.
    let f = f32::to_bits;
    let addr = |g16: bool, a16: bool, cl: bool| {
        let mut a = if g16 {
            vec![0x2C00, 0x38C0_0000]
        } else {
            vec![f(1.0 / 16.0), 0, 0, f(19.0 / 32.0)]
        };
        match (a16, cl) {
            (false, false) => a.extend([f(0.3125), f(0.75)]),
            (false, true) => a.extend([f(0.3125), f(0.75), f(4.0)]),
            (true, false) => a.push(0x3A00_3500),
            (true, true) => a.extend([0x3A00_3500, 0x4400]),
        }
        a
    };
    let at = |level: u32, x: u32, y: u32| (1000 * level + 64 * y + x) as f32;
    let mut validator =
        validator::Validator::load(validator::Validator::default_path().as_deref()).ok();
    let mut lower = |label: &str, words: &[u32]| {
        let shader = try_lower_cs32(words, &used).unwrap_or_else(|e| panic!("{label}: {e}"));
        if let Some(v) = validator.as_mut() {
            v.validate(&mut shader.dxil.clone())
                .unwrap_or_else(|e| panic!("{label}: {e}"));
        }
        shader
    };
    // image_sample_d (0x22), _d_cl (0x23), and their `_g16` forms (0xA2,
    // 0xA3), 2D.
    let mut cases = Vec::new();
    for (cl, g16, a16) in (0..8).map(|k| (k & 4 != 0, k & 2 != 0, k & 1 != 0)) {
        let op = 0x22 | u32::from(cl) | u32::from(g16) << 7;
        let label = format!(
            "image_sample_d{}{}{}",
            if cl { "_cl" } else { "" },
            if g16 { "_g16" } else { "" },
            if a16 { " a16" } else { "" }
        );
        let shader = lower(&label, &program(op, 1, a16, &addr(g16, a16, cl)));
        let want = if cl { at(4, 1, 0) } else { at(3, 2, 1) };
        cases.push((label, shader, want));
    }
    // At 16x8x32, dPdx=(1/32, 0, 1/16) spans just over two texels
    // (mip 1); dPdy=(0, 1/16, -1/8) spans just over four (mip 2).
    // Only the named direction is active. Each packed direction's z
    // occupies its own VGPR, with an intentionally nonzero unused half.
    let tag = |level: u32, x: u32, y: u32, z: u32| (10000 * level + 100 * z + 16 * y + x) as f32;
    let mut volume_cases = Vec::new();
    for (dir, dx, dy, dx16, dy16, expected) in [
        (
            "dx",
            [1.0 / 32.0, 0.0, 1.0 / 16.0],
            [0.0; 3],
            [0x2800, 0x4400_2c00],
            [0, 0x4400_0000],
            tag(1, 2, 2, 9),
        ),
        (
            "dy",
            [0.0; 3],
            [0.0, 1.0 / 16.0, -1.0 / 8.0],
            [0, 0x4400_0000],
            [0x2c00_0000, 0x4400_b000],
            tag(2, 1, 1, 4),
        ),
    ] {
        for (cl, g16, a16) in (0..8).map(|k| (k & 4 != 0, k & 2 != 0, k & 1 != 0)) {
            let mut addr = Vec::new();
            if g16 {
                addr.extend(dx16);
                addr.extend(dy16);
            } else {
                addr.extend(dx.map(f));
                addr.extend(dy.map(f));
            }
            // u=11/32, v=11/16, w=19/32; clamp=3 shares w's VGPR under a16.
            if a16 {
                addr.extend([0x3980_3580, (if cl { 0x4200_0000 } else { 0 }) | 0x38c0]);
            } else {
                addr.extend([f(11.0 / 32.0), f(11.0 / 16.0), f(19.0 / 32.0)]);
                if cl {
                    addr.push(f(3.0));
                }
            }
            let label = format!("3D {dir} cl={cl} g16={g16} a16={a16}");
            let op = 0x22 | u32::from(cl) | u32::from(g16) << 7;
            let shader = lower(&label, &program(op, 2, a16, &addr));
            volume_cases.push((label, shader, if cl { tag(3, 0, 0, 2) } else { expected }));
        }
    }
    let refused = try_lower_cs32(&program(0xAA, 1, false, &[0; 5]), &used);
    let Err(crate::ShaderError::Unsupported { what, .. }) = refused else {
        panic!("translated image_sample_c_d_g16");
    };
    assert_eq!(what, "image_sample_c_d_g16");

    let samplers = [D3D12_SAMPLER_DESC {
        Filter: D3D12_FILTER_MIN_MAG_MIP_POINT,
        AddressU: D3D12_TEXTURE_ADDRESS_MODE_CLAMP,
        AddressV: D3D12_TEXTURE_ADDRESS_MODE_CLAMP,
        AddressW: D3D12_TEXTURE_ADDRESS_MODE_CLAMP,
        MaxAnisotropy: 1,
        MaxLOD: f32::MAX,
        ..Default::default()
    }];
    let mut texels = Vec::new();
    for level in 0..5 {
        for y in 0..16 >> level {
            for x in 0..64 >> level {
                texels.push([at(level, x, y), 0.0, 0.0, 1.0]);
            }
        }
    }
    let views = [View::Mipped(64, 16, 5, &texels), View::RawUav(1)];
    let mut voxels = Vec::new();
    for level in 0..4 {
        for z in 0..32 >> level {
            for y in 0..8 >> level {
                for x in 0..16 >> level {
                    voxels.push([tag(level, x, y, z), 0.0, 0.0, 1.0]);
                }
            }
        }
    }
    let volume_views = [View::Volume(16, 8, 32, 4, &voxels), View::RawUav(1)];
    let _one = ONE_AT_A_TIME.lock().unwrap_or_else(|e| e.into_inner());
    let mut ran = 0;
    for (adapter, warp) in [("hardware", false), ("warp", true)] {
        let Ok(host) = Host::new(warp) else {
            eprintln!("no SM 6.6 {adapter} device; skipped");
            continue;
        };
        for ((label, shader, want), views) in cases
            .iter()
            .map(|case| (case, &views))
            .chain(volume_cases.iter().map(|case| (case, &volume_views)))
        {
            let (root, pso) = host
                .compute_pipeline(shader)
                .unwrap_or_else(|e| panic!("{adapter} {label}: {e}"));
            let out = host
                .dispatch(
                    &root,
                    &pso,
                    &Dispatch {
                        roots: vec![(0, Root::Constants(user_data_cs32(shader, 0)))],
                        views,
                        samplers: &samplers,
                        groups: [1, 1, 1],
                    },
                )
                .unwrap_or_else(|e| panic!("{adapter} {label}: {e}"));
            assert_eq!(f32::from_bits(out[0][0]), *want, "{adapter}: {label}");
        }
        ran += 1;
    }
    assert!(ran > 0, "no SM 6.6 device");
}

/// Isotropic point samples with level and position tags on all six faces.
/// The larger face derivative is in the high f16 half, in dPdx or dPdy.
#[test]
fn cube_gradients_of_16_bits() {
    use windows::Win32::Graphics::Direct3D12::*;

    let tag = |face: u32, mip: u32, x: u32, y: u32| (10000 * face + 1000 * mip + 32 * y + x) as f32;
    let mut texels = Vec::new();
    for face in 0..6 {
        for mip in 0..4 {
            for y in 0..32 >> mip {
                for x in 0..32 >> mip {
                    texels.push([tag(face, mip, x, y), 0.0, 0.0, 1.0]);
                }
            }
        }
    }
    let views = [View::MippedCube(32, 4, &texels), View::RawUav(1)];
    let samplers = [D3D12_SAMPLER_DESC {
        Filter: D3D12_FILTER_MIN_MAG_MIP_POINT,
        AddressU: D3D12_TEXTURE_ADDRESS_MODE_CLAMP,
        AddressV: D3D12_TEXTURE_ADDRESS_MODE_CLAMP,
        AddressW: D3D12_TEXTURE_ADDRESS_MODE_CLAMP,
        MaxAnisotropy: 1,
        MaxLOD: f32::MAX,
        ..Default::default()
    }];
    let mut cases = Vec::new();
    for (face, face16) in [0, 0x3c00, 0x4000, 0x4200, 0x4400, 0x4500]
        .into_iter()
        .enumerate()
    {
        let face = face as u32;
        // (1/32, ±1/8) spans sqrt(17) base-level texels. Point sampling
        // selects mip 2. A minimum LOD of 3 selects the last level.
        let (grad32, grad16) = if face.is_multiple_of(2) {
            ([1.0 / 32.0, 1.0 / 8.0, 0.0, 0.0], [0x3000_2800, 0])
        } else {
            ([0.0, 0.0, 1.0 / 32.0, -1.0 / 8.0], [0, 0xb000_2800])
        };
        for (cl, g16, a16) in (0..8).map(|k| (k & 4 != 0, k & 2 != 0, k & 1 != 0)) {
            let mut addr = Vec::new();
            if g16 {
                addr.extend(grad16);
            } else {
                addr.extend(grad32.map(f32::to_bits));
            }
            // Cube face operands s=1+11/32, t=1+21/32, face. Under a16
            // the minimum LOD shares the upper half of the face word.
            if a16 {
                addr.extend([0x3ea0_3d60, face16 | if cl { 0x4200_0000 } else { 0 }]);
            } else {
                addr.extend([1.34375, 1.65625, face as f32].map(f32::to_bits));
                if cl {
                    addr.push(3.0f32.to_bits());
                }
            }
            let op = 0x22 | u32::from(cl) | (u32::from(g16) << 7);
            let words = gradient_sample_program(op, 3, a16, &addr);
            let shader = lower_cs32(&words, &GRADIENT_BINDINGS, true);
            let expected = if cl {
                tag(face, 3, 1, 2)
            } else {
                tag(face, 2, 2, 5)
            };
            cases.push((
                format!("face={face} cl={cl} g16={g16} a16={a16}"),
                shader,
                expected,
            ));
        }
    }

    let _one = ONE_AT_A_TIME.lock().unwrap_or_else(|e| e.into_inner());
    let mut ran = 0;
    for (adapter, warp) in [("hardware", false), ("warp", true)] {
        let Ok(host) = Host::new(warp) else {
            eprintln!("no SM 6.6 {adapter} device; skipped");
            continue;
        };
        for (label, shader, expected) in &cases {
            let (root, pso) = host.compute_pipeline(shader).unwrap();
            let out = host
                .dispatch(
                    &root,
                    &pso,
                    &Dispatch {
                        roots: vec![(0, Root::Constants(user_data_cs32(shader, 0)))],
                        views: &views,
                        samplers: &samplers,
                        groups: [1, 1, 1],
                    },
                )
                .unwrap();
            let got = f32::from_bits(out[0][0]);
            assert_eq!(got, *expected, "{adapter}: {label}");
        }
        ran += 1;
    }
    assert!(ran > 0, "no SM 6.6 device");
}

#[test]
fn resinfo_a16_reads_the_low_mip_half() {
    let _one = ONE_AT_A_TIME.lock().unwrap_or_else(|e| e.into_inner());
    let Ok(host) = Host::new(false).or_else(|_| Host::new(true)) else {
        eprintln!("no SM 6.6 adapter; skipped");
        return;
    };
    let texels = [[0.0; 4]; 42];
    let views = [View::Mipped(8, 4, 3, &texels), View::RawUav(2)];
    let used = [(4, ResourceType::Srv, 0, 8), (4, ResourceType::Uav, 8, 4)];
    let mut observed = Vec::new();
    for (a16, mip) in [(false, 1), (true, 2), (true, 0x1234_0001), (true, 0x5678_0002)] {
        // s[8:15] is the texture at heap slot 0; s[20:23] is the output at slot 1.
        let mut words = vec![
            s_mov_b32(5, 0x80),
            0xf40c_0202, 0xfa00_0000, // s_load_dwordx8 s[8:15], s[4:5], 0
            0xf408_0502, 0xfa00_0020, // s_load_dwordx4 s[20:23], s[4:5], 0x20
            vop1(1, 0, 255), mip, // v0 holds the mip operand
            0xf038_0308, 0x0002_1400 | (a16 as u32) << 30, // image_get_resinfo width/height
        ];
        words.extend(buffer_store_dword(20, 20, 0));
        words.extend(buffer_store_dword(21, 20, 4));
        words.push(S_ENDPGM);
        let shader = lower_cs32(&words, &used, true);
        let (root, pso) = host.compute_pipeline(&shader).unwrap();
        let out = host
            .dispatch(
                &root,
                &pso,
                &Dispatch {
                    roots: vec![(0, Root::Constants(user_data_cs32(&shader, 0)))],
                    views: &views,
                    samplers: &[],
                    groups: [1, 1, 1],
                },
            )
            .unwrap();
        observed.push([out[0][0], out[0][1]]);
    }
    assert_eq!(observed, [[4, 2], [2, 1], [4, 2], [2, 1]]);
}

/// MIMG `image_atomic_<op> vdata, v[vaddr:vaddr+1], s[srsrc:srsrc+7]
/// dmask dim:2D unorm` (glc returns the old value).
const fn image_atomic(op: u32, vdata: u8, vaddr: u8, srsrc: u8, dmask: u32, glc: bool) -> [u32; 2] {
    [
        0xF000_0000 | op << 18 | (glc as u32) << 13 | 1 << 12 | dmask << 8 | 1 << 3,
        vaddr as u32 | (vdata as u32) << 8 | (srsrc as u32 >> 2) << 16,
    ]
}

/// 64-bit image atomics (two dwords per value, an R32G32_UINT view, the
/// low dword in R) as the alpha prepasses use them: an unsigned maximum of
/// keys whose high dword decides, returning the old value with glc, and an
/// exchange and a compare-exchange. 32 lanes; lane `l` writes texels
/// (l, 1..=3), all lanes texel (0, 0).
#[test]
fn image_atomics_of_64_bits() {
    let (v0, v5) = (vgpr(0), vgpr(5));
    let mut words = vec![s_mov_b32(5, 0x80)];
    // s_load_dwordx8 s[8:15], s[4:5], 0: the texture's descriptor.
    words.extend([0x3D << 26 | 0x03 << 18 | 8 << 6 | 4 >> 1, 125 << 25]);
    words.extend([
        0x1B << 25 | 5 << 17 | (v0 as u32 & 0xFF) << 9 | 131, // v_and_b32 v5, 3, v0
        0x26 << 25 | 1 << 17 | (v0 as u32 & 0xFF) << 9 | 159, // v_sub_nc_u32 v1, 31, v0
        0x1A << 25 | 2 << 17 | (v5 as u32 & 0xFF) << 9 | 158, // v_lshlrev_b32 v2, 30, v5
        vop1(0x001, 14, 128),
        vop1(0x001, 15, 128),
        vop1(0x001, 3, v0),
        vop1(0x001, 4, 129),
        vop1(0x001, 8, v0),
        vop1(0x001, 9, 129),
    ]);
    // a = (31 - l) | (l & 3) << 62 into texel (0, 0) and (l, 1).
    words.extend(image_atomic(0x17, 1, 14, 8, 3, false));
    words.extend(image_atomic(0x17, 1, 3, 8, 3, false));
    // b = l | 1 << 32 into (l, 1), returning a in v[8:9].
    words.extend(image_atomic(0x17, 8, 3, 8, 3, true));
    // a swapped into (l, 2); b into (l, 3) where it compares equal to 0.
    words.push(vop1(0x001, 4, 130));
    words.extend(image_atomic(0x0F, 8, 3, 8, 3, false));
    words.extend([
        vop1(0x001, 4, 131),
        vop1(0x001, 10, v0),
        vop1(0x001, 11, 129),
        vop1(0x001, 12, 128),
        vop1(0x001, 13, 128),
    ]);
    words.extend(image_atomic(0x10, 10, 3, 8, 0xF, false));
    words.push(S_ENDPGM);
    let shader = lower_cs32(&words, &[(4, ResourceType::Uav, 0, 8)], true);

    let key = |lo: u32, hi: u32| u64::from(lo) | u64::from(hi) << 32;
    let a = |l: u32| key(31 - l, (l & 3) << 30);
    let b = |l: u32| key(l, 1);
    let mut want = vec![0u64; 32 * 4];
    want[0] = (0..32).map(a).max().unwrap();
    for l in 0..32 {
        want[32 + l as usize] = a(l).max(b(l));
        want[64 + l as usize] = a(l);
        want[96 + l as usize] = b(l);
    }
    let _one = ONE_AT_A_TIME.lock().unwrap_or_else(|e| e.into_inner());
    for (label, warp) in [("hardware", false), ("warp", true)] {
        let Ok(host) = Host::new(warp) else {
            eprintln!("no SM 6.6 {label} device; skipped");
            continue;
        };
        if !host.heap_atomics64() {
            eprintln!("{label}: no 64-bit atomics on typed heap resources; skipped");
            continue;
        }
        let (root, pso) = host
            .compute_pipeline(&shader)
            .unwrap_or_else(|e| panic!("{label}: {e}"));
        let mut ud = vec![0u32; shader.user_data_dwords().max(1) as usize];
        for u in &shader.user_data {
            assert_eq!((u.root, u.slot), (RootType::Table, 1), "user data {u:?}");
            ud[u.index as usize] = 0;
        }
        let out = host
            .dispatch(
                &root,
                &pso,
                &Dispatch {
                    roots: vec![(0, Root::Constants(ud))],
                    views: &[View::Uint2TextureUav(32, 4)],
                    samplers: &[],
                    groups: [1, 1, 1],
                },
            )
            .unwrap_or_else(|e| panic!("{label}: {e}"));
        let got: Vec<u64> = out[0].chunks(2).map(|t| key(t[0], t[1])).collect();
        assert_eq!(got, want, "{label}");
    }
}

/// SOPC `op src0, src1`.
const fn sopc(op: u8, src0: u8, src1: u8) -> u32 {
    0xBF00_0000 | (op as u32) << 16 | (src1 as u32) << 8 | src0 as u32
}

/// `s_cselect_b32 s12, 1, 0; v_mov_b32 v0, s12`: SCC as 0 or 1 in v0.
fn scc_to_v0() -> [u32; 2] {
    [sop2(0x0A, 12, 129, 128), vop1(0x001, 0, sgpr(12))]
}

#[test]
fn s_bfe_64_extracts_and_sets_scc() {
    // s_bfe_u64 / s_bfe_i64 s[6:7], s[0:1], s2 (SOP2 0x29 / 0x2a: s2 = off[5:0]
    // | width[22:16]); the result's low dword, high dword or SCC goes to v0.
    let result = |op: u8, tail: &[u32]| {
        let mut body = vec![sop2(op, 6, 0, 2)];
        body.extend_from_slice(tail);
        build_valu_cs(&body)
    };
    let lo = |op| result(op, &[vop1(0x001, 0, sgpr(6))]);
    let hi = |op| result(op, &[vop1(0x001, 0, sgpr(7))]);
    let scc = |op| result(op, &scc_to_v0());
    let field = |off: u32, width: u32| off | width << 16;
    let x = 0xFEDC_BA98_7654_3210u64;
    // (src, offset, width, signed, expected)
    let cases: &[(u64, u32, u32, bool, u64)] = &[
        (x, 32, 16, false, 0xBA98),
        (x, 32, 16, true, 0xFFFF_FFFF_FFFF_BA98),
        (x, 0, 8, true, 0x10),
        (x, 4, 40, false, 0x00A9_8765_4321),
        (x, 56, 4, true, 0xFFFF_FFFF_FFFF_FFFE),
        (x, 60, 4, false, 0xF),
        (x, 8, 0, false, 0),
        (x, 8, 0, true, 0),
        (0x0000_00F0_0000_0000, 0, 32, false, 0),
        (0x0000_00F0_0000_0000, 36, 4, true, 0xFFFF_FFFF_FFFF_FFFF),
        (0x0000_00F0_0000_0000, 36, 4, false, 0xF),
        (0x0000_00F0_0000_0000, 32, 4, true, 0),
    ];
    for &(src, off, width, signed, want) in cases {
        let op = if signed { 0x2A } else { 0x29 };
        let inputs = [[src as u32, (src >> 32) as u32, field(off, width), 0]];
        let what = format!("s_bfe_{}64({src:#018x}, {off}, {width})", ["u", "i"][signed as usize]);
        run_cases(&lo(op), &inputs, |backend, _, _, got| {
            assert_eq!(got, want as u32, "{backend}: {what} low dword = {got:#010x}");
        });
        run_cases(&hi(op), &inputs, |backend, _, _, got| {
            assert_eq!(got, (want >> 32) as u32, "{backend}: {what} high dword = {got:#010x}");
        });
        run_cases(&scc(op), &inputs, |backend, _, _, got| {
            assert_eq!(got, (want != 0) as u32, "{backend}: {what} SCC = {got}");
        });
    }
}

#[test]
fn s_bfm_b64_builds_a_mask_and_keeps_scc() {
    // s_cmp_lg_u32 s3, 0 sets SCC from s3; s_bfm_b64 s[6:7], s0, s1 (SOP2
    // 0x25: width s0[5:0], offset s1[5:0]) must leave it alone.
    let prefix = [sopc(0x07, 3, 128), sop2(0x25, 6, 0, 1)];
    let build = |tail: &[u32]| build_valu_cs(&[&prefix[..], tail].concat());
    let (lo, hi) = (
        build(&[vop1(0x001, 0, sgpr(6))]),
        build(&[vop1(0x001, 0, sgpr(7))]),
    );
    let scc = build(&scc_to_v0());
    // (width, offset, expected)
    let cases: &[(u32, u32, u64)] = &[
        (40, 8, 0x0000_FFFF_FFFF_FF00),
        (32, 8, 0x0000_00FF_FFFF_FF00),
        (0, 5, 0),
        (1, 63, 0x8000_0000_0000_0000),
        (32, 0, 0x0000_0000_FFFF_FFFF),
        (32, 32, 0xFFFF_FFFF_0000_0000),
        (63, 0, 0x7FFF_FFFF_FFFF_FFFF),
        (4, 0x80 | 2, 0x0000_0000_0000_003C), // only bits 5:0 count
    ];
    for &(width, off, want) in cases {
        let inputs = [[width, off, 0, 0]];
        run_cases(&lo, &inputs, |backend, _, _, got| {
            assert_eq!(got, want as u32, "{backend}: s_bfm_b64({width}, {off}) low = {got:#010x}");
        });
        run_cases(&hi, &inputs, |backend, _, _, got| {
            assert_eq!(
                got,
                (want >> 32) as u32,
                "{backend}: s_bfm_b64({width}, {off}) high = {got:#010x}",
            );
        });
        // SCC keeps the compare's value whether or not the mask is empty.
        for s3 in [0u32, 1] {
            run_cases(&scc, &[[width, off, 0, s3]], |backend, _, _, got| {
                assert_eq!(got, s3, "{backend}: s_bfm_b64({width}, {off}) changed SCC {s3} to {got}");
            });
        }
    }
}


#[test]
fn v_cmp_class_f16_matches_rdna2() {
    // v_cmp_class_f16_e64 s6, v1, v2 (VOP3 opcode 0x08f); v_cndmask_b32_e64 v0, 0, 1, s6
    let cmp_lo = build_valu_cs(&[
        vop3(0x08f, 6, vgpr(1), vgpr(2), 0)[0],
        vop3(0x08f, 6, vgpr(1), vgpr(2), 0)[1],
        vop3(0x101, 0, 0x80, 0x81, sgpr(6))[0],
        vop3(0x101, 0, 0x80, 0x81, sgpr(6))[1],
    ]);
    // Same with op_sel:[1,0] (high half of v1) + neg on src0 (bit 29 of w1).
    let [mut w0, mut w1] = vop3(0x08f, 6, vgpr(1), vgpr(2), 0);
    w0 |= 1 << 11; // opsel[0] = 1 (high half)
    w1 |= 1 << 29; // neg[0] = 1
    let cmp_hi_neg = build_valu_cs(&[
        w0,
        w1,
        vop3(0x101, 0, 0x80, 0x81, sgpr(6))[0],
        vop3(0x101, 0, 0x80, 0x81, sgpr(6))[1],
    ]);

    // All 10 IEEE-754 binary16 classes: bit k corresponds to class k.
    let classes: &[(u32, u32)] = &[
        (0x7C01, 0), // sNaN (+sNaN, quiet bit 0x0200 clear)
        (0xFDFF, 0), // -sNaN (sign ignored for NaNs)
        (0x7E00, 1), // +qNaN
        (0xFE01, 1), // -qNaN
        (0xFC00, 2), // -inf
        (0xBC00, 3), // -normal (-1.0)
        (0x8001, 4), // -denormal
        (0x8000, 5), // -0.0
        (0x0000, 6), // +0.0
        (0x0001, 7), // +denormal
        (0x3C00, 8), // +normal (+1.0)
        (0x7C00, 9), // +inf
    ];
    let mut cases = Vec::new();
    for &(bits, cls) in classes {
        // Exact class bit -> true; complement of class bit across all 10 bits -> false.
        cases.push(([bits, 1u32 << cls, 0, 0], 1u32));
        cases.push(([bits, 0x3FF & !(1u32 << cls), 0, 0], 0u32));
    }
    let inputs: Vec<[u32; 4]> = cases.iter().map(|c| c.0).collect();
    run_cases(&cmp_lo, &inputs, |backend, i, inp, got| {
        let want = cases[i].1;
        assert_eq!(
            got, want,
            "{backend}: v_cmp_class_f16({:#06x}, {:#05x}) = {got}, want {want}",
            inp[0], inp[1],
        );
    });

    // High half with neg: high half 0x3C00 (+1.0) negated -> -normal (class 3);
    // high half 0x7C01 (+sNaN) negated -> still sNaN (class 0, not quieted).
    let hi_cases: &[([u32; 4], u32)] = &[
        ([0x3C00_0000, 1 << 3, 0, 0], 1),
        ([0x3C00_0000, 1 << 8, 0, 0], 0),
        ([0x7C01_FFFF, 1 << 0, 0, 0], 1),
        ([0x7C01_FFFF, 1 << 1, 0, 0], 0),
    ];
    let hi_in: Vec<[u32; 4]> = hi_cases.iter().map(|c| c.0).collect();
    run_cases(&cmp_hi_neg, &hi_in, |backend, i, inp, got| {
        let want = hi_cases[i].1;
        assert_eq!(
            got, want,
            "{backend}: v_cmp_class_f16(-hi({:#010x}), {:#05x}) = {got}, want {want}",
            inp[0], inp[1],
        );
    });
}

#[test]
fn cvt_f32_f64_matches_rdna2() {
    // v_cvt_f32_f64 v0, v[1:2] (VOP1 opcode 0x00f)
    let shader = build_valu_cs(&[vop1(0x00f, 0, vgpr(1))]);

    let cases: &[(f64, f32)] = &[
        (0.0, 0.0),
        (-0.0, -0.0),
        (1.0, 1.0),
        (-1.0, -1.0),
        (std::f64::consts::PI, std::f32::consts::PI),
        (65536.5, 65536.5),
        (1.0e-30, (1.0e-30f64) as f32),
        (1.0e38, 1.0e38),
    ];

    let inputs: Vec<[u32; 4]> = cases
        .iter()
        .map(|&(d, _)| {
            let bits = d.to_bits();
            [bits as u32, (bits >> 32) as u32, 0, 0]
        })
        .collect();

    run_cases(&shader, &inputs, |backend, i, _, got| {
        let want = cases[i].1.to_bits();
        assert_eq!(
            got,
            want,
            "{}: v_cvt_f32_f64({}) = {:#010x} ({}), want {:#010x} ({})",
            backend,
            cases[i].0,
            got,
            f32::from_bits(got),
            want,
            cases[i].1,
        );
    });
}

#[test]
fn cvt_f32_f64_decodes_double_inline_constants() {
    let values = [
        0.5,
        -0.5,
        1.0,
        -1.0,
        2.0,
        -2.0,
        4.0,
        -4.0,
        f64::from_bits(0x3fc4_5f30_6dc9_c882) as f32,
    ];
    for (code, value) in (240..=248).zip(values) {
        let shader = build_valu_cs(&[vop1(0x00f, 0, code)]);
        run_cases(&shader, &[[0; 4]], |backend, _, _, got| {
            assert_eq!(
                got,
                value.to_bits(),
                "{backend}: FP64 inline constant {code} should convert to {value}"
            );
        });
    }
    // Integer inline 1 is a bit pattern, a tiny positive double.
    let integer_one = build_valu_cs(&[vop1(0x00f, 0, 129)]);
    run_cases(&integer_one, &[[0; 4]], |backend, _, _, got| {
        assert_eq!(got, 0, "{backend}: integer inline bits");
    });
}

#[test]
fn cvt_f32_f64_uses_literal_as_the_high_word() {
    for (high, value) in [
        (0x4008_0000, 3.0f32),
        (0xc00c_0000, -3.5),
        (0x8000_0000, -0.0),
    ] {
        let shader = build_valu_cs(&[vop1(0x00f, 0, 255), high]);
        run_cases(&shader, &[[0; 4]], |backend, _, _, got| {
            assert_eq!(
                got,
                value.to_bits(),
                "{backend}: FP64 literal {high:#010x} in the high dword"
            );
        });
    }
}

#[test]
fn cvt_f32_f64_applies_e64_source_and_output_modifiers() {
    const ABS: u32 = 1 << 8;
    const NEG: u32 = 1 << 29;
    const CLAMP: u32 = 1 << 15;
    const MUL_2: u32 = 1 << 27;
    for (input, abs_clamp, neg_omod, expected) in [
        (-1.25f64, ABS, 0, 1.25f32),
        (-1.25, 0, NEG, 1.25),
        (-1.25, ABS, NEG, -1.25),
        (-0.75, ABS | CLAMP, MUL_2, 1.0),
    ] {
        let mut words = vop3(0x18f, 0, vgpr(1), 0, 0);
        words[0] |= abs_clamp;
        words[1] |= neg_omod;
        let shader = build_valu_cs(&words);
        let bits = input.to_bits();
        run_cases(
            &shader,
            &[[bits as u32, (bits >> 32) as u32, 0, 0]],
            |backend, _, _, got| {
                assert_eq!(
                    got,
                    expected.to_bits(),
                    "{backend}: {words:08x?} input {input}"
                );
            },
        );
    }
}

/// The low and high dwords of a double.
fn dwords(x: f64) -> [u32; 2] {
    let bits = x.to_bits();
    [bits as u32, (bits >> 32) as u32]
}

/// `body` runs on each 4-dword input and leaves a double in `v[6:7]`; `check`
/// gets the double's bits for each backend and input.
fn run_f64_raw(body: &[u32], inputs: &[[u32; 4]], mut check: impl FnMut(&str, usize, u64)) {
    use std::collections::HashMap;
    let with = |word: u32| build_valu_cs(&[body, &[word]].concat());
    let (low, high) = (with(vop1(0x001, 0, vgpr(6))), with(vop1(0x001, 0, vgpr(7))));
    let mut lows = HashMap::new();
    run_cases(&low, inputs, |backend, i, _, got| {
        lows.insert((backend.to_string(), i), got);
    });
    run_cases(&high, inputs, |backend, i, _, got| {
        let lo = lows[&(backend.to_string(), i)];
        check(backend, i, u64::from(got) << 32 | u64::from(lo));
    });
}

/// [`run_f64_raw`] with each input two doubles, in `v[1:2]` and `v[3:4]`.
fn run_f64(body: &[u32], cases: &[[f64; 2]], check: impl FnMut(&str, usize, u64)) {
    run_f64_raw(body, &double_inputs(cases), check);
}

fn double_inputs(cases: &[[f64; 2]]) -> Vec<[u32; 4]> {
    cases
        .iter()
        .map(|&[a, b]| {
            let (a, b) = (dwords(a), dwords(b));
            [a[0], a[1], b[0], b[1]]
        })
        .collect()
}

/// Equal doubles, any NaN equal to any other.
fn same_f64(got: u64, want: f64) -> bool {
    let got = f64::from_bits(got);
    got.to_bits() == want.to_bits() || (got.is_nan() && want.is_nan())
}

const F64_CASES: &[[f64; 2]] = &[
    [1.5, 2.25],
    [-3.0, 0.1],
    [0.1, 0.2],
    [1e300, 1e300],
    [1e-200, 1e-200],
    [-0.0, 0.0],
    [0.0, 0.0],
    [-0.0, -0.0],
    [f64::NAN, 1.0],
    [1.0, f64::NAN],
    [f64::INFINITY, f64::NEG_INFINITY],
    [f64::MAX, -f64::MAX],
];

#[test]
fn f64_add_mul_and_fma_are_ieee() {
    // v_add_f64 v[6:7], v[1:2], v[3:4] (VOP3 0x164), v_mul_f64 (0x165) and
    // v_fma_f64 v[6:7], v[1:2], v[3:4], 1.0 (0x14c; inline constant 242).
    type Op = (&'static str, [u32; 2], fn(f64, f64) -> f64);
    let ops: [Op; 3] = [
        ("v_add_f64", vop3(0x164, 6, vgpr(1), vgpr(3), 0), |a, b| {
            a + b
        }),
        ("v_mul_f64", vop3(0x165, 6, vgpr(1), vgpr(3), 0), |a, b| {
            a * b
        }),
        (
            "v_fma_f64",
            vop3(0x14c, 6, vgpr(1), vgpr(3), 242),
            |a, b| a.mul_add(b, 1.0),
        ),
    ];
    for (name, words, reference) in ops {
        run_f64(&words, F64_CASES, |backend, i, got| {
            let [a, b] = F64_CASES[i];
            let want = reference(a, b);
            assert!(
                same_f64(got, want),
                "{backend}: {name}({a}, {b}) = {} ({got:#018x}), want {want}",
                f64::from_bits(got),
            );
        });
    }
}

#[test]
fn v_cmp_and_cmpx_class_f32_match_rdna2() {
    // VOP3: v_cmp_class_f32_e64 s6, v1, v2 (0x088); v_cndmask_b32_e64 v0, 0, 1, s6.
    let cmp_e64 = build_valu_cs(&[
        vop3(0x088, 6, vgpr(1), vgpr(2), 0)[0],
        vop3(0x088, 6, vgpr(1), vgpr(2), 0)[1],
        vop3(0x101, 0, 0x80, 0x81, sgpr(6))[0],
        vop3(0x101, 0, 0x80, 0x81, sgpr(6))[1],
    ]);
    // VOPC: v_cmp_class_f32_e32 vcc_lo, s0, v2; v_cndmask_b32_e64 v0, 0, 1, vcc_lo.
    let cmp_e32_sgpr = build_valu_cs(&[
        0x7D10_0400,
        vop3(0x101, 0, 0x80, 0x81, sgpr(106))[0],
        vop3(0x101, 0, 0x80, 0x81, sgpr(106))[1],
    ]);
    // v_cmpx_class_f32_e32 v1, v2 (0x098) and v_cmpx_class_f16_e32 v1, v2 (0x09f):
    // narrows EXEC directly; non-matching lanes keep 0xCAFE.
    let cmpx_cs = |vopc_word: u32| {
        build_valu_cs(&[
            s_mov_b32(16, 126),   // s_mov_b32 s16, exec_lo
            vop1(0x001, 0, 0xFF), // v_mov_b32 v0, 0xCAFE
            0x0000_CAFE,          // literal
            vopc_word,            // v_cmpx_class_* v1, v2
            vop1(0x001, 0, 0x81), // v_mov_b32 v0, 1
            s_mov_b32(126, 16),   // s_mov_b32 exec_lo, s16
        ])
    };
    let cmpx_f32 = cmpx_cs(0x7D30_0501);
    let cmpx_f16 = cmpx_cs(0x7D3E_0501);

    // All 10 IEEE-754 binary32 classes including subnormal, normal, and NaN edges:
    // denormals are never flushed and sNaNs are never quieted.
    let classes: &[(u32, u32)] = &[
        (0x7F80_0001, 0), // smallest +sNaN (quiet bit 0x0040_0000 clear)
        (0x7FBF_FFFF, 0), // largest +sNaN
        (0xFF80_0001, 0), // smallest -sNaN
        (0xFFBF_FFFF, 0), // largest -sNaN
        (0x7FC0_0000, 1), // +qNaN
        (0xFFFF_FFFF, 1), // -qNaN
        (0xFF80_0000, 2), // -inf
        (0x8080_0000, 3), // smallest -normal
        (0xBF80_0000, 3), // -1.0 (-normal)
        (0xFF7F_FFFF, 3), // largest -normal
        (0x8000_0001, 4), // smallest -denormal
        (0x807F_FFFF, 4), // largest -denormal
        (0x8000_0000, 5), // -0.0
        (0x0000_0000, 6), // +0.0
        (0x0000_0001, 7), // smallest +denormal
        (0x007F_FFFF, 7), // largest +denormal
        (0x0080_0000, 8), // smallest +normal
        (0x3F80_0000, 8), // +1.0 (+normal)
        (0x7F7F_FFFF, 8), // largest +normal
        (0x7F80_0000, 9), // +inf
    ];
    let mut cases = Vec::new();
    for &(bits, cls) in classes {
        let bit = 1u32 << cls;
        cases.push(([bits, bit, 0, 0], 1u32));
        cases.push(([bits, 0x3FF & !bit, 0, 0], 0u32));
        // Upper mask bits src1[31:10] are ignored by hardware.
        cases.push(([bits, 0xFFFF_FC00, 0, 0], 0u32));
        cases.push(([bits, 0xFFFF_FC00 | bit, 0, 0], 1u32));
        // Combined masks: NaN (0x003), +/-inf (0x204), +/-0.0 (0x060), +/-denorm (0x090).
        for mask in [0x003u32, 0x204, 0x060, 0x090, 0x000] {
            cases.push(([bits, mask, 0, 0], u32::from(mask & bit != 0)));
        }
    }
    let inputs: Vec<[u32; 4]> = cases.iter().map(|c| c.0).collect();
    for (label, cs) in [("e64", &cmp_e64), ("e32_sgpr", &cmp_e32_sgpr)] {
        run_cases(cs, &inputs, |backend, i, inp, got| {
            let want = cases[i].1;
            assert_eq!(
                got, want,
                "{backend}: v_cmp_class_f32[{label}]({:#010x}, {:#010x}) = {got}, want {want}",
                inp[0], inp[1],
            );
        });
    }

    // 32-bit literal src0 in VOPC (`0xFF` followed by raw f32 bits): +sNaN and -denormal.
    for (lit, cls) in [(0x7F80_0001u32, 0u32), (0x8000_0001u32, 4u32)] {
        let lit_cs = build_valu_cs(&[
            0x7D10_04FF,
            lit,
            vop3(0x101, 0, 0x80, 0x81, sgpr(106))[0],
            vop3(0x101, 0, 0x80, 0x81, sgpr(106))[1],
        ]);
        let lit_in = [[0, 1 << cls, 0, 0], [0, 0x3FF & !(1 << cls), 0, 0]];
        run_cases(&lit_cs, &lit_in, |backend, i, inp, got| {
            let want = u32::from(i == 0);
            assert_eq!(
                got, want,
                "{backend}: v_cmp_class_f32 literal({lit:#010x}, {:#05x}) = {got}, want {want}",
                inp[1],
            );
        });
    }

    // v_cmpx_class_f32 and v_cmpx_class_f16 EXEC narrowing across matching and non-matching masks.
    let cmpx_f32_in = [
        [0x7F80_0001, 1 << 0, 0, 0],
        [0x7F80_0001, 1 << 1, 0, 0],
        [0x8000_0001, 1 << 4, 0, 0],
        [0x8000_0001, 1 << 5, 0, 0],
        [0x3F80_0000, 0xFFFF_FC00, 0, 0],
    ];
    let cmpx_want = [1u32, 0xCAFE, 1, 0xCAFE, 0xCAFE];
    run_cases(&cmpx_f32, &cmpx_f32_in, |backend, i, inp, got| {
        assert_eq!(
            got, cmpx_want[i],
            "{backend}: v_cmpx_class_f32({:#010x}, {:#010x}) = {got:#x}, want {:#x}",
            inp[0], inp[1], cmpx_want[i],
        );
    });
    let cmpx_f16_in = [
        [0x7C01, 1 << 0, 0, 0],
        [0x7C01, 1 << 1, 0, 0],
        [0x8001, 1 << 4, 0, 0],
        [0x8001, 1 << 5, 0, 0],
        [0x3C00, 0xFFFF_FC00, 0, 0],
    ];
    run_cases(&cmpx_f16, &cmpx_f16_in, |backend, i, inp, got| {
        assert_eq!(
            got, cmpx_want[i],
            "{backend}: v_cmpx_class_f16({:#06x}, {:#010x}) = {got:#x}, want {:#x}",
            inp[0], inp[1], cmpx_want[i],
        );
    });
}

/// A compute program in an `ACCC` part, as the AMD upscaler's are: `body`
/// leaves its result in v0, stored to the raw UAV of the table at user data
/// 4 (root parameter 1). Translated from the container's bytes.
fn fspx_cs(body: &[u32]) -> HostShader {
    use crate::native::tests::{fspx, fspx_resource};
    let mut words = vec![s_mov_b32(5, 0x80)];
    words.extend_from_slice(&s_load_dwordx4(8, 4));
    words.extend_from_slice(body);
    words.extend_from_slice(&buffer_store_dword(0, 8, 0));
    words.push(S_ENDPGM);
    let code: Vec<u8> = words.iter().flat_map(|w| w.to_le_bytes()).collect();
    let part = fspx(
        &code,
        5 << 1,
        [1, 1, 1],
        true,
        &[fspx_resource(3, 3, 0, 0, 4, 1)],
    );
    let container = crate::dxbc::build(&[(crate::FourCC::ACCC, &part)]);
    super::translate(&container).expect("translate")
}

/// VOP3P dot product, with one `opsel` and `opsel_hi` bit per source.
const fn packed_dot(
    op: u8,
    vdst: u8,
    src: [u16; 3],
    opsel: u32,
    opsel_hi: u32,
    clamp: bool,
) -> [u32; 2] {
    [
        0xCC00_0000
            | (op as u32) << 16
            | (clamp as u32) << 15
            | (opsel_hi >> 2 & 1) << 14
            | (opsel & 7) << 11
            | vdst as u32,
        (opsel_hi & 3) << 27
            | (src[2] as u32 & 0x1FF) << 18
            | (src[1] as u32 & 0x1FF) << 9
            | (src[0] as u32 & 0x1FF),
    ]
}

/// The signed byte dot product plus `acc`, wrapping (RDNA2 ISA).
fn dot4_ref(a: u32, b: u32, acc: u32) -> u32 {
    (0..4).fold(acc, |s, k| {
        let (x, y) = ((a >> (8 * k)) as i8 as i32, (b >> (8 * k)) as i8 as i32);
        s.wrapping_add((x * y) as u32)
    })
}

/// The upscaler's integer dot products, from an `ACCC` container to the
/// host GPU: op_sel picks each multiplicand's low half, op_sel_hi its high
/// half (a constant folded in as `sel_hi_lo(2.0)` is 0x4000), and
/// `v_dot4c` accumulates into its destination.
///
/// The operands are made at run time, each a literal plus a word of the
/// (zeroed) UAV: the RDNA1 driver zero-extends the bytes of a multiplicand
/// it sees as a constant, though the DXIL asks for signed bytes (#1782).
#[test]
fn fspx_programs_run_with_the_hardwares_dot_products() {
    let (a, b, acc) = (0x0302_FB02u32, 0x80FF_7F01u32, 0x7FFF_FFF0u32);
    // buffer_load_dword v4, off, s[8:11], 0: the result's word, still 0.
    let mut load = vec![0x38 << 26 | 0x0C << 18, 0x80 << 24 | 2 << 16 | 4 << 8];
    for (v, x) in [(1u32, a), (2, b), (3, acc)] {
        load.extend([0x25 << 25 | v << 17 | 4 << 9 | 0xFF, x]); // v_add_nc_u32 vN, literal, v4
    }
    let cases: [(Vec<u32>, u32); 4] = [
        (
            packed_dot(0x16, 0, [vgpr(1), vgpr(2), vgpr(3)], 0, 7, false).to_vec(),
            dot4_ref(a, b, acc),
        ),
        // sel_hi_lo on an inline 2.0 (0x40000000): the halves swapped.
        (
            packed_dot(0x16, 0, [vgpr(1), 244, vgpr(3)], 0b010, 0b101, false).to_vec(),
            dot4_ref(a, 0x4000, acc),
        ),
        (
            packed_dot(0x16, 0, [vgpr(1), vgpr(2), vgpr(3)], 0b001, 0b110, false).to_vec(),
            dot4_ref(a.rotate_left(16), b, acc),
        ),
        // v_mov_b32 v0, v3; v_dot4c_i32_i8 v0, v1, v2
        (
            vec![
                vop1(0x001, 0, vgpr(3)),
                0x0d << 25 | 2 << 9 | vgpr(1) as u32,
            ],
            dot4_ref(a, b, acc),
        ),
    ];
    for (body, want) in cases {
        let mut words = load.clone();
        words.extend(&body);
        let shader = fspx_cs(&words);
        run_cases(&shader, &[[0; 4]], |backend, _, _, got| {
            assert_eq!(got, want, "{backend}: {body:08x?}");
        });
    }
}

fn check_unsigned_dot(op: u8, cases: &[([u32; 4], u32)]) {
    let inputs: Vec<_> = cases.iter().map(|&(inputs, _)| inputs).collect();
    for clamp in [false, true] {
        let body = packed_dot(op, 0, [vgpr(1), vgpr(2), vgpr(3)], 0, 7, clamp);
        let shader = build_valu_cs(&body);
        let container = crate::Container::parse(&shader.dxil).unwrap();
        let sfi0 = container.part(crate::FourCC::SFI0).unwrap().data;
        let features = u64::from_le_bytes(sfi0.try_into().unwrap());
        assert_eq!(features & 0x8000, 0, "unexpected Int64 feature");

        run_cases(&shader, &inputs, |backend, i, inputs, got| {
            let dot = cases[i].1;
            let acc = inputs[2];
            let want = if clamp {
                acc.saturating_add(dot)
            } else {
                acc.wrapping_add(dot)
            };
            assert_eq!(got, want, "{backend}: clamp={clamp}, inputs={inputs:08x?}");
        });
    }
}

#[test]
fn unsigned_byte_dot_products_wrap_or_saturate() {
    check_unsigned_dot(
        0x17,
        &[
            ([0x0403_0201, 0x0807_0605, 9, 0], 70),
            ([0xFF00_0000, 0xFF00_0000, 0, 0], 65_025),
            ([0xFF00_0000, 0xFF00_0000, u32::MAX - 1, 0], 65_025),
            ([1, 1, u32::MAX - 1, 0], 1),
            ([0, u32::MAX, u32::MAX, 0], 0),
        ],
    );
}

#[test]
fn unsigned_nibble_dot_products_wrap_or_saturate() {
    let mut cases: Vec<_> = (0..8)
        .map(|k| ([3 << (4 * k), 14 << (4 * k), 5, 0], 42))
        .collect();
    cases.extend([
        ([u32::MAX, u32::MAX, 0, 0], 1800),
        ([u32::MAX, u32::MAX, u32::MAX - 2, 0], 1800),
        ([1, 1, u32::MAX - 2, 0], 1),
        ([1, 1, u32::MAX - 1, 0], 1),
    ]);
    check_unsigned_dot(0x19, &cases);
}

#[test]
fn unsigned_packed_dot_products_use_operand_half_selectors() {
    for op in [0x17, 0x19] {
        // Swap src0's halves, duplicate src1's low half, and read src2 whole.
        let body = packed_dot(op, 0, [vgpr(1), vgpr(2), vgpr(3)], 0b101, 0, false);
        let shader = build_valu_cs(&body);
        run_cases(
            &shader,
            &[[0x0003_0002, 0x0007_0005, 0x1234_5678, 0]],
            |backend, _, _, got| {
                assert_eq!(
                    got,
                    0x1234_5678 + 3 * 5 + 2 * 5,
                    "{backend}: opcode {op:#x}"
                );
            },
        );
    }
}

#[test]
fn f64_max_and_min_order_zeros_and_skip_nan() {
    // v_min_f64 (VOP3 0x166) and v_max_f64 (0x167) v[6:7], v[1:2], v[3:4].
    let ordered = |a: f64, b: f64, max: bool| {
        if a.is_nan() {
            return b;
        }
        if b.is_nan() {
            return a;
        }
        if a == 0.0 && b == 0.0 {
            // -0 below +0.
            let negative = if max {
                a.is_sign_negative() && b.is_sign_negative()
            } else {
                a.is_sign_negative() || b.is_sign_negative()
            };
            return if negative { -0.0 } else { 0.0 };
        }
        if max { a.max(b) } else { a.min(b) }
    };
    for (name, op, max) in [("v_min_f64", 0x166, false), ("v_max_f64", 0x167, true)] {
        let words = vop3(op, 6, vgpr(1), vgpr(3), 0);
        run_f64(&words, F64_CASES, |backend, i, got| {
            let [a, b] = F64_CASES[i];
            let want = ordered(a, b, max);
            assert!(
                same_f64(got, want),
                "{backend}: {name}({a}, {b}) = {} ({got:#018x}), want {want}",
                f64::from_bits(got),
            );
        });
    }
}

#[test]
fn f64_sources_apply_abs_and_neg() {
    // v_add_f64 v[6:7], v[1:2], v[3:4] with abs (bits 8, 9) and neg (bits
    // 61, 62) per source, abs first.
    const ABS: [u32; 2] = [1 << 8, 1 << 9];
    const NEG: [u32; 2] = [1 << 29, 1 << 30];
    let cases = [[-2.5, 4.0], [3.0, -1.0], [-0.0, 0.0]];
    for (abs, neg) in [(0, 0), (1, 0), (2, 0), (0, 1), (0, 2), (3, 3), (1, 2)] {
        let mut words = vop3(0x164, 6, vgpr(1), vgpr(3), 0);
        let modify = |x: f64, k: usize| {
            let x = if abs >> k & 1 != 0 { x.abs() } else { x };
            if neg >> k & 1 != 0 { -x } else { x }
        };
        for k in 0..2 {
            if abs >> k & 1 != 0 {
                words[0] |= ABS[k];
            }
            if neg >> k & 1 != 0 {
                words[1] |= NEG[k];
            }
        }
        run_f64(&words, &cases, |backend, i, got| {
            let [a, b] = cases[i];
            let want = modify(a, 0) + modify(b, 1);
            assert!(
                same_f64(got, want),
                "{backend}: abs {abs:#b} neg {neg:#b} ({a}, {b}) = {} want {want}",
                f64::from_bits(got),
            );
        });
    }
}

#[test]
fn f64_arithmetic_reads_sgpr_pairs() {
    // v_add_f64 v[6:7], s[0:1], s[2:3]: the root constants.
    let words = vop3(0x164, 6, sgpr(0), sgpr(2), 0);
    run_f64(&words, F64_CASES, |backend, i, got| {
        let [a, b] = F64_CASES[i];
        assert!(
            same_f64(got, a + b),
            "{backend}: ({a}, {b}) = {}",
            f64::from_bits(got),
        );
    });
}

/// Inputs for [`f64_reads_uniform_pairs_at_odd_dwords`]: v[2:3] and the
/// pair it would be with its high dword lost, (v2, v2), differ in each.
const ODD_PAIR_INPUTS: [[u32; 4]; 4] = [
    [0, 0x3ff8_0000, 0x4000_0000, 0],
    [0x1111_1111, 0x3ff8_0000, 0x4000_0000, 0x4010_0000],
    [0x89ab_cdef, 0x4049_0fdb, 0xc008_0000, 0x3fd0_0000],
    // v[1:2] = 1.5 lies between v[2:3] and (v2, v2).
    [0, 0x3ff8_0000, 0x3ff0_0000, 0x4000_0000],
];

/// The 64 bits in `v[n:n+1]` of a [`build_valu_cs`] program's inputs.
fn bits_at(x: &[u32; 4], n: usize) -> u64 {
    u64::from(x[n]) << 32 | u64::from(x[n - 1])
}

fn f64_at(x: &[u32; 4], n: usize) -> f64 {
    f64::from_bits(bits_at(x, n))
}

#[test]
fn f64_reads_uniform_pairs_at_odd_dwords() {
    // FIND-21: v1..v4 are the four root constants, one load. Before the
    // lowering XORed register-pair dwords with an opaque zero, doubles at
    // dwords 1 and 2 (v[2:3]), alone or sharing a dword with another source,
    // came out of AMD's compiler with the low dword as both halves; WARP and
    // NVIDIA were right. The 64-bit integer cases were right on AMD too.
    type Wide = (&'static str, Vec<u32>, fn(&[u32; 4]) -> u64);
    let op3 = |op, a, b, c| vop3(op, 6, a, b, c).to_vec();
    let wide: [Wide; 13] = [
        (
            "v_add_f64 v[6:7], v[1:2], v[3:4]",
            op3(0x164, vgpr(1), vgpr(3), 0),
            |x| (f64_at(x, 1) + f64_at(x, 3)).to_bits(),
        ),
        (
            "v_add_f64 v[6:7], v[2:3], 1.0",
            op3(0x164, vgpr(2), 242, 0),
            |x| (f64_at(x, 2) + 1.0).to_bits(),
        ),
        (
            "v_add_f64 v[6:7], v[2:3], v[2:3]",
            op3(0x164, vgpr(2), vgpr(2), 0),
            |x| (f64_at(x, 2) + f64_at(x, 2)).to_bits(),
        ),
        (
            "v_add_f64 v[6:7], v[2:3], v[3:4]",
            op3(0x164, vgpr(2), vgpr(3), 0),
            |x| (f64_at(x, 2) + f64_at(x, 3)).to_bits(),
        ),
        (
            "v_add_f64 v[6:7], v[1:2], v[2:3]",
            op3(0x164, vgpr(1), vgpr(2), 0),
            |x| (f64_at(x, 1) + f64_at(x, 2)).to_bits(),
        ),
        (
            "v_mul_f64 v[6:7], v[1:2], v[2:3]",
            op3(0x165, vgpr(1), vgpr(2), 0),
            |x| (f64_at(x, 1) * f64_at(x, 2)).to_bits(),
        ),
        (
            "v_fma_f64 v[6:7], v[2:3], v[3:4], v[1:2]",
            op3(0x14c, vgpr(2), vgpr(3), vgpr(1)),
            |x| f64_at(x, 2).mul_add(f64_at(x, 3), f64_at(x, 1)).to_bits(),
        ),
        (
            "v_max_f64 v[6:7], v[2:3], 1.0",
            op3(0x167, vgpr(2), 242, 0),
            |x| f64_at(x, 2).max(1.0).to_bits(),
        ),
        (
            "v_min_f64 v[6:7], v[2:3], 4.0",
            op3(0x166, vgpr(2), 246, 0),
            |x| f64_at(x, 2).min(4.0).to_bits(),
        ),
        (
            "v_cvt_f64_f32 v[6:7], v2",
            vec![vop1(0x010, 6, vgpr(2))],
            |x| f64::from(f32::from_bits(x[1])).to_bits(),
        ),
        // 64-bit integer pairs.
        (
            "v_lshlrev_b64 v[6:7], v1, v[2:3]",
            op3(0x2ff, vgpr(1), vgpr(2), 0),
            |x| bits_at(x, 2) << (x[0] & 63),
        ),
        (
            "v_lshrrev_b64 v[6:7], v4, v[2:3]",
            op3(0x300, vgpr(4), vgpr(2), 0),
            |x| bits_at(x, 2) >> (x[3] & 63),
        ),
        (
            "v_mad_u64_u32 v[6:7], null, v1, v4, v[2:3]",
            {
                // VOP3B: the carry goes to null.
                let mut w = op3(0x176, vgpr(1), vgpr(4), vgpr(2));
                w[0] |= 125 << 8;
                w
            },
            |x| (u64::from(x[0]) * u64::from(x[3])).wrapping_add(bits_at(x, 2)),
        ),
    ];
    let mut wrong = Vec::new();
    for (name, words, reference) in wide {
        run_f64_raw(&words, &ODD_PAIR_INPUTS, |backend, i, got| {
            let want = reference(&ODD_PAIR_INPUTS[i]);
            let float = name.contains("f64");
            let same = if float {
                same_f64(got, f64::from_bits(want))
            } else {
                got == want
            };
            if !same {
                wrong.push(format!(
                    "{backend}: {name} on input {i}: {got:#018x}, want {want:#018x}"
                ));
            }
        });
    }

    // Results in v0: conversions and compares (through v_cndmask_b32_e64
    // v0, 0, 1, s20).
    type Narrow = (&'static str, Vec<u32>, fn(&[u32; 4]) -> u32);
    let compare = |op: u16, a: u16, b: u16| {
        let mut body = vop3(op, 20, a, b, 0x80).to_vec();
        body.extend_from_slice(&vop3(0x101, 0, 0x80, 0x81, sgpr(20)));
        body
    };
    let narrow: [Narrow; 5] = [
        (
            "v_cvt_f32_f64 v0, v[2:3]",
            vec![vop1(0x00f, 0, vgpr(2))],
            |x| (f64_at(x, 2) as f32).to_bits(),
        ),
        (
            "v_cvt_i32_f64 v0, v[2:3]",
            vec![vop1(0x003, 0, vgpr(2))],
            |x| f64_at(x, 2) as i32 as u32,
        ),
        (
            "v_cvt_u32_f64 v0, v[2:3]",
            vec![vop1(0x015, 0, vgpr(2))],
            |x| f64_at(x, 2) as u32,
        ),
        (
            "v_cmp_gt_f64 s20, v[2:3], 2.0",
            compare(0x024, vgpr(2), 244),
            |x| u32::from(f64_at(x, 2) > 2.0),
        ),
        (
            "v_cmp_lt_f64 s20, v[1:2], v[2:3]",
            compare(0x021, vgpr(1), vgpr(2)),
            |x| u32::from(f64_at(x, 1) < f64_at(x, 2)),
        ),
    ];
    for (name, body, reference) in narrow {
        let shader = build_valu_cs(&body);
        run_cases(&shader, &ODD_PAIR_INPUTS, |backend, i, _, got| {
            let want = reference(&ODD_PAIR_INPUTS[i]);
            if got != want {
                wrong.push(format!(
                    "{backend}: {name} on input {i}: {got:#010x}, want {want:#010x}"
                ));
            }
        });
    }
    for w in &wrong {
        eprintln!("{w}");
    }
    assert!(wrong.is_empty(), "{} wrong results", wrong.len());
}

#[test]
fn v_cmp_f64_matches_ieee() {
    // v_cmp_<op>_f64 s20, v[1:2], v[3:4] (VOPC 0x020-0x02f in VOP3);
    // v_cndmask_b32_e64 v0, 0, 1, s20 (VOP3 0x101)
    type Op = (&'static str, u16, fn(f64, f64) -> bool);
    fn unordered(a: f64, b: f64) -> bool {
        a.is_nan() || b.is_nan()
    }
    let ops: [Op; 16] = [
        ("f", 0x020, |_, _| false),
        ("lt", 0x021, |a, b| a < b),
        ("eq", 0x022, |a, b| a == b),
        ("le", 0x023, |a, b| a <= b),
        ("gt", 0x024, |a, b| a > b),
        ("lg", 0x025, |a, b| !unordered(a, b) && a != b),
        ("ge", 0x026, |a, b| a >= b),
        ("o", 0x027, |a, b| !unordered(a, b)),
        ("u", 0x028, unordered),
        // "Not" forms are true for a NaN.
        ("nge", 0x029, |a, b| unordered(a, b) || a < b),
        ("nlg", 0x02a, |a, b| unordered(a, b) || a == b),
        ("ngt", 0x02b, |a, b| unordered(a, b) || a <= b),
        ("nle", 0x02c, |a, b| unordered(a, b) || a > b),
        ("neq", 0x02d, |a, b| unordered(a, b) || a != b),
        ("nlt", 0x02e, |a, b| unordered(a, b) || a >= b),
        ("tru", 0x02f, |_, _| true),
    ];
    let inputs = double_inputs(F64_CASES);
    for (name, op, reference) in ops {
        let mut body = vop3(op, 20, vgpr(1), vgpr(3), 0x80).to_vec();
        body.extend_from_slice(&vop3(0x101, 0, 0x80, 0x81, sgpr(20)));
        let shader = build_valu_cs(&body);
        run_cases(&shader, &inputs, |backend, i, _, got| {
            let [a, b] = F64_CASES[i];
            let want = u32::from(reference(a, b));
            assert_eq!(got, want, "{backend}: v_cmp_{name}_f64({a}, {b})");
        });
    }
}

#[test]
fn f64_conversions_match_rdna2() {
    // v_cvt_f64_i32 v[6:7], v1 (VOP1 0x004) and v_cvt_f64_u32 (0x016).
    let ints = [0u32, 1, 5, 0x7FFF_FFFF, 0x8000_0000, 0xFFFF_FFFB, u32::MAX];
    let inputs: Vec<[u32; 4]> = ints.iter().map(|&x| [x, 0, 0, 0]).collect();
    run_f64_raw(&[vop1(0x004, 6, vgpr(1))], &inputs, |backend, i, got| {
        let want = f64::from(ints[i] as i32);
        assert_eq!(
            got,
            want.to_bits(),
            "{backend}: v_cvt_f64_i32({:#x})",
            ints[i]
        );
    });
    run_f64_raw(&[vop1(0x016, 6, vgpr(1))], &inputs, |backend, i, got| {
        let want = f64::from(ints[i]);
        assert_eq!(
            got,
            want.to_bits(),
            "{backend}: v_cvt_f64_u32({:#x})",
            ints[i]
        );
    });

    // v_cvt_f64_f32 v[6:7], v1 (VOP1 0x010).
    let floats = [
        1.5f32,
        -0.0,
        0.1,
        f32::MAX,
        f32::INFINITY,
        f32::NEG_INFINITY,
        f32::NAN,
        f32::MIN_POSITIVE,
    ];
    let inputs: Vec<[u32; 4]> = floats.iter().map(|x| [x.to_bits(), 0, 0, 0]).collect();
    run_f64_raw(&[vop1(0x010, 6, vgpr(1))], &inputs, |backend, i, got| {
        let want = f64::from(floats[i]);
        assert!(
            same_f64(got, want),
            "{backend}: v_cvt_f64_f32({})",
            floats[i]
        );
    });

    // v_cvt_i32_f64 v0, v[1:2] (VOP1 0x003) and v_cvt_u32_f64 (0x015)
    // saturate and turn NaN into 0.
    let doubles = [
        0.0,
        1.9,
        -1.9,
        2.5,
        3e9,
        -3e9,
        5e9,
        4e9,
        f64::NAN,
        f64::INFINITY,
        f64::NEG_INFINITY,
    ];
    let inputs: Vec<[u32; 4]> = doubles
        .iter()
        .map(|&x| {
            let d = dwords(x);
            [d[0], d[1], 0, 0]
        })
        .collect();
    let to_i32 = build_valu_cs(&[vop1(0x003, 0, vgpr(1))]);
    run_cases(&to_i32, &inputs, |backend, i, _, got| {
        let want = doubles[i] as i32 as u32;
        assert_eq!(got, want, "{backend}: v_cvt_i32_f64({})", doubles[i]);
    });
    let to_u32 = build_valu_cs(&[vop1(0x015, 0, vgpr(1))]);
    run_cases(&to_u32, &inputs, |backend, i, _, got| {
        let want = doubles[i] as u32;
        assert_eq!(got, want, "{backend}: v_cvt_u32_f64({})", doubles[i]);
    });
}

/// `v_pk_fmac_f16` (VOP2): each half of the destination plus the product
/// of the sources' halves.
#[test]
fn packed_fmac_accumulates_each_half() {
    let shader = build_valu_cs(&[
        vop1(0x001, 0, vgpr(3)),              // v_mov_b32 v0, v3
        0x3C << 25 | 2 << 9 | vgpr(1) as u32, // v_pk_fmac_f16 v0, v1, v2
    ]);
    // (2.0, 1.0) * (3.0, 4.0) + (1.0, 0.5) = (7.0, 4.5)
    run_cases(
        &shader,
        &[[0x4000_3C00, 0x4200_4400, 0x3C00_3800, 0]],
        |backend, _, _, got| {
            assert_eq!(got, 0x4700_4480, "{backend}");
        },
    );
}

/// `v_pk_fmac_f16_dpp` (VOP2 + DPP16): `src0` comes from the DPP-selected
/// lane in both packed f16 halves, not the executing lane's own VGPR.
#[test]
fn packed_fmac_dpp_reads_the_swizzled_lane() {
    use super::bitcode::{Attr, Term};
    use super::container::{self, SigElement};
    use super::host::Draw;
    use super::shader::Shader;
    use crate::dxbc::{Container, FourCC};
    use crate::native::StageIo;

    // Pixel shader on a 2x2 quad: v0 = SV_Position.x (0.5 / 1.5), v1 = y.
    // v2 = pack_f16(lo = x + 2y, hi = x + 4y):
    //   (0,0): (1.5, 2.5)  (1,0): (2.5, 3.5)
    //   (0,1): (3.5, 6.5)  (1,1): (4.5, 7.5)
    // quad_perm:[1,0,3,2] swaps horizontal neighbours before multiplying by
    // (lo = 4.0, hi = 2.0) and adding (lo = 0.5, hi = 1.0).
    let words: [u32; 16] = [
        0x08 << 25 | 2 << 17 | 1 << 9 | 244,  // v_mul_f32 v2, 2.0, v1
        0x03 << 25 | 2 << 17 | 2 << 9 | 256,  // v_add_f32 v2, v0, v2
        0x08 << 25 | 3 << 17 | 1 << 9 | 246,  // v_mul_f32 v3, 4.0, v1
        0x03 << 25 | 3 << 17 | 3 << 9 | 256,  // v_add_f32 v3, v0, v3
        0x2F << 25 | 2 << 17 | 3 << 9 | 258,  // v_cvt_pkrtz_f16_f32 v2, v2, v3
        0x7E06_02FF,                          // v_mov_b32 v3, 0x4000_4400
        0x4000_4400,                          //   (hi = 2.0, lo = 4.0)
        0x7E08_02FF,                          // v_mov_b32 v4, 0x3C00_3800
        0x3C00_3800,                          //   (hi = 1.0, lo = 0.5)
        0x3C << 25 | 4 << 17 | 3 << 9 | 0xFA, // v_pk_fmac_f16_dpp v4, v2, v3
        0xFF00_B102,                          //   quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf
        vop1(0x00B, 5, vgpr(4)),              // v_cvt_f32_f16 v5, v4 (low half)
        0x16 << 25 | 6 << 17 | 4 << 9 | 144,  // v_lshrrev_b32 v6, 16, v4
        vop1(0x00B, 6, vgpr(6)),              // v_cvt_f32_f16 v6, v6 (high half)
        0xF800_180F,                          // exp mrt0 v5, v6, v5, v6 done vm
        0x0605_0605,
    ];
    let mut code: Vec<u8> = words.iter().flat_map(|w| w.to_le_bytes()).collect();
    code.extend_from_slice(&S_ENDPGM.to_le_bytes());

    let native = NativeProgram {
        stage: HwStage::Ps,
        wave64: false,
        vgprs: 8,
        layout: Vec::new(),
        used: Vec::new(),
        io: StageIo::Ps {
            inputs: Vec::new(),
            input_addr: (1 << 8) | (1 << 9),
            db_shader_control: 0,
            patches: Vec::new(),
        },
        code,
    };
    let sig_el = |name: &str, kind, interp| SigElement {
        name: name.into(),
        index: 0,
        kind,
        comp: 9,
        interp,
        row: Some(0),
        col: 0,
        cols: 4,
        usage: 0xF,
        half: false,
    };
    let isg1 = container::signature(&[sig_el("SV_Position", 3, 4)], false);
    let osg1 = container::signature(&[sig_el("SV_Target", 16, 0)], false);
    let container_bytes = crate::dxbc::build(&[(FourCC::ISG1, &isg1), (FourCC::OSG1, &osg1)]);
    let c = Container::parse(&container_bytes).expect("parse container");
    let p = ir::build(&native.code, native.wave64, native.vgprs).expect("ir");
    let st = structure::structure(&p).expect("structure");
    let res = resource::track(&p, &native.layout, native.stage, Some(&st)).expect("track");
    let io = io::raise(&p, &native).expect("io");
    let ps = super::emit(&c, &native, &p, &st, &io, &res, crate::emit::Target::ANY).expect("emit");
    let mut s = Shader::new(false);
    let b = s.entry;
    for col in 0..4u8 {
        let load_args = [s.m.i32(0), s.m.i32(0), s.m.i8(col), s.m.undef(s.t.i32)];
        let v = s.op(b, "loadInput.f32", Attr::ReadNone, s.t.f32, 4, &load_args);
        let store_args = [s.m.i32(0), s.m.i32(0), s.m.i8(col), v];
        s.op(
            b,
            "storeOutput.f32",
            Attr::NoUnwind,
            s.t.void,
            5,
            &store_args,
        );
    }
    s.inputs = vec![sig_el("POSITION", 0, 0)];
    s.outputs = vec![sig_el("SV_Position", 3, 0)];
    s.m.terminate(b, Term::Ret);
    let vs = HostShader {
        stage: HwStage::Vs,
        wave_size: None,
        dxil: s.finish(),
        user_data: Vec::new(),
        samplers: Vec::new(),
        static_samplers: Vec::new(),
        root_cbvs: Vec::new(),
        root_srvs: Vec::new(),
        root_uavs: Vec::new(),
        memory: false,
        memory_writes: false,
        view_addresses: false,
        heap: false,
        heap_writes: false,
        view_marks: false,
    };
    if let Ok(mut v) = validator::Validator::load(validator::Validator::default_path().as_deref()) {
        v.validate(&mut vs.dxil.clone())
            .expect("VS DXIL validation failed");
        v.validate(&mut ps.dxil.clone())
            .expect("DXIL validation failed");
    }

    let vertices = [
        -1.0f32, -1.0, 0.5, 1.0, //
        -1.0, 3.0, 0.5, 1.0, //
        3.0, -1.0, 0.5, 1.0,
    ];
    let want = [
        [10.5f32, 8.0, 10.5, 8.0], // (0,0) reads (1,0): (2.5*4+0.5, 3.5*2+1)
        [6.5, 6.0, 6.5, 6.0],      // (1,0) reads (0,0): (1.5*4+0.5, 2.5*2+1)
        [18.5, 16.0, 18.5, 16.0],  // (0,1) reads (1,1): (4.5*4+0.5, 7.5*2+1)
        [14.5, 14.0, 14.5, 14.0],  // (1,1) reads (0,1): (3.5*4+0.5, 6.5*2+1)
    ];

    let _one = ONE_AT_A_TIME.lock().unwrap_or_else(|e| e.into_inner());
    for (label, warp) in [("hardware", false), ("warp", true)] {
        let host = match Host::new(warp) {
            Ok(host) => host,
            Err(e) => {
                eprintln!("no SM 6.6 {label} device ({e}); skipped");
                continue;
            }
        };
        let (root, pso) = host
            .pipeline(&vs, &ps)
            .unwrap_or_else(|e| panic!("{label}: pipeline failed: {e}"));
        let texels = host
            .draw(
                &root,
                &pso,
                &Draw {
                    roots: Vec::new(),
                    textures: &[],
                    samplers: &[],
                    vertices: &vertices,
                    stride: 16,
                    size: (2, 2),
                    mesh_groups: 0,
                },
            )
            .unwrap_or_else(|e| panic!("{label}: draw failed: {e}"));
        assert_eq!(
            texels, want,
            "{label}: v_pk_fmac_f16_dpp quad_perm:[1,0,3,2]"
        );
    }
}

/// `src_scc` reads SCC as 0 or 1, in scalar and vector instructions.
#[test]
fn src_scc_reads_as_zero_or_one() {
    for (compare, want) in [(0xBF06_8181u32, 1), (0xBF06_8281, 0)] {
        // s_cmp_eq_u32 1, 1 (or 2); s_mov_b32 s12, src_scc; v_mov_b32 v0, s12;
        // v_add_nc_u32 v0, src_scc, v0 (VOP2 0x25)
        let shader = build_valu_cs(&[
            compare,
            s_mov_b32(12, 253),
            vop1(0x001, 0, sgpr(12)),
            0x25 << 25 | u32::from(vgpr(0) as u8) << 9 | 253,
        ]);
        run_cases(&shader, &[[0; 4]], |backend, _, _, got| {
            assert_eq!(got, 2 * want, "{backend}");
        });
    }
}

#[test]
fn s_sext_i32_has_single_dword_operands_and_preserves_scc_in_ir() {
    for (word, name) in [
        (0xBE86_1900, "s_sext_i32_i8"),
        (0xBE86_1A00, "s_sext_i32_i16"),
    ] {
        let words = [0xBF06_8181, word, s_mov_b32(7, 253), S_ENDPGM];
        let code: Vec<u8> = words.into_iter().flat_map(u32::to_le_bytes).collect();
        let p = ir::build(&code, false, 16).unwrap();
        let insts = &p.blocks[0].insts;
        let sext = &insts[1];
        assert_eq!(sext.name(), name);
        assert_eq!(sext.reads.len(), 1);
        assert!(sext.read(ir::Reg::S(0)).is_some());
        assert_eq!(sext.writes.len(), 1);
        assert!(sext.write(ir::Reg::S(6)).is_some());
        let scc = insts[0].write(ir::Reg::Scc).unwrap();
        assert_eq!(insts[2].read(ir::Reg::Scc), Some(scc));
    }
}

#[test]
fn s_sext_i32_lowers_without_int64_and_preserves_scc_on_gpu() {
    let cases = [
        (
            0xBE86_1900,
            [
                (0x0000_0000, 0),
                (0x1234_567F, 0x0000_007F),
                (0x0000_0080, 0xFFFF_FF80),
                (0x1234_56FF, 0xFFFF_FFFF),
                (0xFFFF_FF01, 0x0000_0001),
            ],
        ),
        (
            0xBE86_1A00,
            [
                (0x0000_0000, 0),
                (0xABCD_7FFF, 0x0000_7FFF),
                (0x0000_8000, 0xFFFF_8000),
                (0x1234_FFFF, 0xFFFF_FFFF),
                (0xFFFF_0001, 0x0000_0001),
            ],
        ),
    ];
    // The shader builder validates and checks the container hash when the DLL loads.
    if let Some(path) = std::env::var_os("VICTORY_DXIL") {
        validator::Validator::load(Some(std::path::Path::new(&path)))
            .expect("load configured DXIL validator");
    }
    for (word, pairs) in cases {
        let inputs: Vec<_> = pairs
            .iter()
            .flat_map(|&(input, _)| [[input, 0, 0, 0], [input, 1, 0, 0]])
            .collect();
        // s_cmp_eq_u32 s1, 1; s_sext_i32_* s6, s0; v_mov_b32 v0, s6 or src_scc.
        for src in [sgpr(6), 253] {
            let shader = build_valu_cs(&[0xBF06_8101, word, vop1(1, 0, src)]);
            let container = crate::Container::parse(&shader.dxil).unwrap();
            let features = container.part(crate::FourCC::SFI0).unwrap();
            let flags = u64::from_le_bytes(features.data.try_into().unwrap());
            assert_eq!(flags & 0x8000, 0, "Int64 feature for {word:#010x}");
            run_cases(&shader, &inputs, |backend, i, input, got| {
                let want = if src == sgpr(6) {
                    pairs[i / 2].1
                } else {
                    input[1]
                };
                assert_eq!(
                    got, want,
                    "{backend}: {word:#010x}, src {src}, input {input:x?}"
                );
            });
        }
    }
}

/// A register array indexed through M0 whose upper registers hold
/// descriptor words, which have no host value: the indices are clamped
/// (`v_med3_u32`, `v_and_b32`), so the moves reach only v8-v11, and the
/// program runs. Unclamped, the read could reach the descriptor words and
/// the program is refused.
#[test]
fn relative_moves_reach_only_what_their_clamped_index_can() {
    let body = |second_index: u32| {
        let mut w = Vec::new();
        for k in 0..4u8 {
            w.push(vop1(0x001, 8 + k, 138 + u16::from(k))); // v_mov_b32 v8+k, 10+k
            w.push(vop1(0x001, 12 + k, sgpr(8 + k))); // v_mov_b32 v12+k, s8+k
        }
        w.extend(vop3(0x159, 5, vgpr(1), 128, 131)); // v_med3_u32 v5, v1, 0, 3
        w.push(vop1(0x002, 12, vgpr(5))); // v_readfirstlane_b32 s12, v5
        w.push(s_mov_b32(124, 12)); // s_mov_b32 m0, s12
        w.push(vop1(0x042, 8, vgpr(3))); // v_movreld_b32 v8, v3: v[8 + m0] = v3
        w.push(second_index); // v6 = the second index
        w.push(vop1(0x002, 12, vgpr(6))); // v_readfirstlane_b32 s12, v6
        w.push(s_mov_b32(124, 12));
        w.push(vop1(0x043, 0, vgpr(8))); // v_movrels_b32 v0, v8: v0 = v[8 + m0]
        w
    };
    // v_and_b32 v6, 3, v2
    let shader = build_valu_cs(&body(0x1B << 25 | 6 << 17 | 2 << 9 | 131));
    let cases = [
        [0, 0, 77, 0],
        [5, 3, 77, 0],
        [1, 2, 77, 0],
        [u32::MAX, 7, 77, 0],
        [2, 1, 77, 0],
        [3, 0, 77, 0],
    ];
    run_cases(&shader, &cases, |backend, _, [a, b, c, _], got| {
        let (i, j) = (a.min(3), b & 3);
        let want = if i == j { c } else { 10 + j };
        assert_eq!(got, want, "{backend}: a {a:#x} b {b:#x}");
    });
    // v_mov_b32 v6, v2
    let refused = try_build_valu_cs(&body(vop1(0x001, 6, vgpr(2))));
    assert!(
        matches!(
            refused,
            Err(crate::ShaderError::Unsupported {
                what: "a descriptor's address or format read as data",
                ..
            })
        ),
        "{:?}",
        refused.err()
    );
}

/// Indexed vector loads from a root or table CBV, stored to a raw UAV.
fn build_indexed_cbv_cs(
    load_op: u32,
    byte_offset: u16,
    dynamic: bool,
    stride: Option<u16>,
) -> HostShader {
    let store_op = load_op + 16; // The matching dword width in the MUBUF store opcodes.
    let mut words = vec![s_mov_b32(3, 0x80)];
    let input_root = if let Some(stride) = stride {
        words.extend([
            s_mov_b32(8, 0), // s[8:11]: build a descriptor around the root s[0:1]
            0x8809_ff01,     // s_or_b32 s9, s1, encoded stride
            u32::from(stride) << 16,
            s_mov_b32(10, 0xc1), // unlimited records
            0xbe8b_03ff,         // s_mov_b32 s11, format word
            0x1101_4fac,
        ]);
        RootType::Descriptor
    } else {
        words.push(s_mov_b32(1, 0x80));
        words.extend(s_load_dwordx4(8, 0));
        RootType::Table
    };
    words.extend(s_load_dwordx4(12, 2));
    // v0/v1 hold the input row and optional byte offset, v2 the output offset.
    words.push(vop1(1, 1, sgpr(4)));
    words.extend([
        0x3404_0084,         // v_lshlrev_b32 v2, 4, v0
        vop1(1, 3, sgpr(5)), // v_mov_b32 v3, s5; group ID
        0x3406_0689,         // v_lshlrev_b32 v3, 9, v3; output group base
        0x4a04_0503,         // v_add_nc_u32 v2, v3, v2
        0xE000_2000 | (load_op << 18) | (u32::from(dynamic) << 12) | u32::from(byte_offset),
        0x8002_0400, // buffer_load_dword[xN] v4..., v0 or v[0:1], s[8:11]
        0xE000_1000 | (store_op << 18),
        0x8003_0402, // buffer_store_dword[xN] v4..., v2, s[12:15], 0 offen
        S_ENDPGM,
    ]);
    let native = NativeProgram {
        stage: HwStage::Cs,
        wave64: false,
        vgprs: 8,
        layout: vec![
            RootParam {
                root: input_root,
                table_offset_dwords: 0,
                sgprs: if stride.is_some() { 2 } else { 1 },
                base_register: 0,
                user_data: 0,
                slot: 0,
            },
            RootParam {
                root: RootType::Table,
                table_offset_dwords: 0,
                sgprs: 1,
                base_register: 0,
                user_data: 2,
                slot: 1,
            },
            RootParam {
                root: RootType::Constants,
                table_offset_dwords: 0,
                sgprs: 1,
                base_register: 0,
                user_data: 4,
                slot: 2,
            },
        ],
        used: [
            (ResourceType::Cbv, input_root, 0, 0),
            (ResourceType::Uav, RootType::Table, 2, 1),
        ]
        .into_iter()
        .map(|(kind, root, user_data, slot)| UsedResource {
            kind,
            user_data,
            slot,
            root,
            table_offset_dwords: 0,
            lower_bound: 0,
            stride: 0,
            dwords: 4,
        })
        .collect(),
        io: StageIo::Cs(ComputeIo {
            rsrc2: 5 << 1 | 1 << 7,
            threads: [32, 1, 1],
        }),
        code: words.iter().flat_map(|w| w.to_le_bytes()).collect(),
    };
    lower_valu_cs(&native).expect("CBV load shader")
}

#[test]
fn indexed_cbv_loads_match_their_byte_addresses() {
    let _one = ONE_AT_A_TIME.lock().unwrap_or_else(|e| e.into_inner());
    let input: Vec<u32> = (0u32..288)
        .map(|i| (i + 1).wrapping_mul(0x91de_2f71).rotate_left(i % 32))
        .collect();
    // Repeat the width/offset matrix for the current raw table path and for root
    // descriptors. Stride 4 changes the component within each row per lane.
    let mut variants = Vec::new();
    for stride in [None, Some(4), Some(16), Some(32)] {
        for (op, count) in [(12, 1), (13, 2), (15, 3), (14, 4)] {
            for (offset, dynamic) in (0..=28).step_by(4).map(|o| (o, false)).chain([(0, true)]) {
                variants.push((
                    count,
                    offset,
                    dynamic,
                    stride,
                    build_indexed_cbv_cs(op, offset, dynamic, stride),
                ));
            }
        }
    }
    let mut ran = false;

    for (label, warp) in [("hardware", false), ("warp", true)] {
        let Ok(host) = Host::new(warp) else {
            eprintln!("no SM 6.6 {label} device; skipped");
            continue;
        };
        ran = true;
        for (count, offset, dynamic, stride, shader) in &variants {
            assert_eq!(
                shader.root_cbvs.as_slice(),
                if stride.is_some() { &[0u8] } else { &[][..] }
            );
            let (root, pso) = host.compute_pipeline(shader).expect("compute pipeline");
            let byte_offsets: &[u32] = if *dynamic { &[0, 4, 12, 16] } else { &[0] };
            for &dynamic_offset in byte_offsets {
                let mut ud = vec![0u32; shader.user_data_dwords() as usize];
                for u in &shader.user_data {
                    ud[u.index as usize] = match (u.root, u.slot) {
                        (RootType::Table, 0) => 0,
                        (RootType::Table, 1) => 32,
                        (RootType::Constants, 2) => dynamic_offset,
                        other => panic!("unexpected user data {other:?}"),
                    };
                }
                let mut roots = vec![(0, Root::Constants(ud))];
                if stride.is_some() {
                    roots.push((
                        1,
                        Root::View {
                            srv: false,
                            data: input.clone(),
                        },
                    ));
                }
                let actual = host
                    .dispatch(
                        &root,
                        &pso,
                        &Dispatch {
                            roots,
                            views: &[
                                View::Buffer {
                                    data: &input,
                                    stride: Stride::Raw,
                                },
                                View::RawUav(256),
                            ],
                            samplers: &[],
                            groups: [2, 1, 1],
                        },
                    )
                    .expect("compute dispatch");
                let dword = (usize::from(*offset) + dynamic_offset as usize) / 4;
                assert_eq!(actual[0].len(), 256);
                for (group_index, group) in actual[0].as_chunks::<128>().0.iter().enumerate() {
                    for (lane, got) in group.as_chunks::<4>().0.iter().enumerate() {
                        let first = lane * usize::from(stride.unwrap_or(16) / 4) + dword;
                        assert_eq!(
                            &got[..*count],
                            &input[first..first + count],
                            "{label}: {count} dwords at stride {stride:?}, byte offset {} + {dynamic_offset}, group {group_index}, lane {lane}",
                            offset
                        );
                    }
                }
            }
        }
    }
    assert!(ran, "neither hardware nor WARP SM 6.6 device available");
}

/// One wave of 32 lanes holding 1..=32 (the input of the ISA's examples)
/// through `ds_swizzle_b32` with `offset`: what each lane reads.
fn swizzle_32(offset: u16) -> Vec<Vec<u32>> {
    let mut words = vec![s_mov_b32(5, 0x80)];
    words.extend_from_slice(&s_load_dwordx4(8, 4));
    words.push(vop2(0x25, 1, 129, 0)); // v_add_nc_u32 v1, 1, v0
    words.extend_from_slice(&ds_swizzle_b32(2, 1, offset));
    words.push(vop2(0x1A, 3, 130, 0)); // v_lshlrev_b32 v3, 2, v0
    words.extend_from_slice(&buffer_store_dword_offen(2, 3, 8));
    words.push(S_ENDPGM);
    let shader = try_lower_valu_cs(&words, 16, 32).expect("build");

    let mut out = Vec::new();
    for (label, warp) in [("hardware", false), ("warp", true)] {
        let Ok(host) = Host::new(warp) else {
            eprintln!("no SM 6.6 {label} device; skipped");
            continue;
        };
        let (root, pso) = match host.compute_pipeline(&shader) {
            Ok(p) => p,
            // A host without 32-lane waves cannot run it.
            Err(e) if warp => {
                eprintln!("{label}: no pipeline ({e}); skipped");
                continue;
            }
            Err(e) => panic!("{label}: compute_pipeline failed: {e}"),
        };
        let ud = vec![0u32; shader.user_data_dwords().max(1) as usize];
        let uavs = host
            .dispatch(
                &root,
                &pso,
                &Dispatch {
                    roots: vec![(0, Root::Constants(ud))],
                    views: &[View::RawUav(32)],
                    samplers: &[],
                    groups: [1, 1, 1],
                },
            )
            .unwrap_or_else(|e| panic!("{label}: dispatch failed: {e}"));
        out.push(uavs[0].clone());
    }
    if out.is_empty() {
        eprintln!("no SM 6.6 device with 32-lane waves; skipped");
    }
    out
}

/// The basic modes: a quad permutation and a 32-lane and/or/xor mapping.
#[test]
fn ds_swizzle_basic_modes() {
    // offset[15]: lanes of each quad read lanes 3, 2, 1, 0 of it.
    let reversed: Vec<u32> = (0..32).map(|i| (i ^ 3) + 1).collect();
    // and 0x1f, or 0, xor 0x10: the halves swap.
    let swapped: Vec<u32> = (0..32).map(|i| (i ^ 16) + 1).collect();
    for (offset, want) in [(0x801B, reversed), (0x401F, swapped)] {
        for got in swizzle_32(offset) {
            assert_eq!(got, want, "offset {offset:#06x}");
        }
    }
}

/// Offsets from 0xE000: the FFT orders, as the ISA lists them for inputs
/// 1..=0x20 (RDNA2 ISA, 12.13.1).
#[test]
fn ds_swizzle_fft_mode() {
    let cases: [(u16, [u32; 32]); 3] = [
        (
            0xE000,
            [
                0x01, 0x11, 0x09, 0x19, 0x05, 0x15, 0x0d, 0x1d, 0x03, 0x13, 0x0b, 0x1b, 0x07, 0x17,
                0x0f, 0x1f, 0x02, 0x12, 0x0a, 0x1a, 0x06, 0x16, 0x0e, 0x1e, 0x04, 0x14, 0x0c, 0x1c,
                0x08, 0x18, 0x10, 0x20,
            ],
        ),
        (
            0xE010,
            [
                0x01, 0x09, 0x05, 0x0d, 0x03, 0x0b, 0x07, 0x0f, 0x02, 0x0a, 0x06, 0x0e, 0x04, 0x0c,
                0x08, 0x10, 0x11, 0x19, 0x15, 0x1d, 0x13, 0x1b, 0x17, 0x1f, 0x12, 0x1a, 0x16, 0x1e,
                0x14, 0x1c, 0x18, 0x20,
            ],
        ),
        // 0x1f: no swizzle.
        (0xE01F, std::array::from_fn(|i| i as u32 + 1)),
    ];
    for (offset, want) in cases {
        for got in swizzle_32(offset) {
            assert_eq!(got, want, "offset {offset:#06x}");
        }
    }
    // The other bits of an FFT offset do not matter.
    for got in swizzle_32(0xFFF0) {
        assert_eq!(got, cases[1].1, "offset 0xfff0");
    }
}

/// Offsets from 0xC000 to 0xDFFF: the wave's 32 lanes rotate left by
/// offset[9:5], or right with offset[10] (RDNA2 ISA, 12.13.1).
#[test]
fn ds_swizzle_rotate_mode() {
    let rotated = |by: u32| -> Vec<u32> { (0..32).map(|i| (i + by) % 32 + 1).collect() };
    for (offset, by) in [
        (0xC000 | 1 << 5, 1),            // the ISA's example: {2, 3, ... 0x20, 1}
        (0xC000 | 1 << 10 | 1 << 5, 31), // and {0x20, 1, ... 0x1f}
        (0xC000, 0),
        (0xC000 | 5 << 5, 5),
        (0xC000 | 1 << 10 | 5 << 5, 27),
        (0xD800 | 31 << 5, 31), // bits 11 and 12 are not read
    ] {
        for got in swizzle_32(offset) {
            assert_eq!(got, rotated(by), "offset {offset:#06x}");
        }
    }
}

#[test]
fn v_mul_legacy_f32_treats_denormals_as_zero() {
    // v_mul_legacy_f32 v0, v1, v2 (VOP2 opcode 0x007 in its VOP3 encoding)
    let shader = build_valu_cs(&vop3(0x107, 0, vgpr(1), vgpr(2), vgpr(0)));
    let f = f32::to_bits;
    let (denorm, neg_denorm, min_normal) = (0x0000_0001u32, 0x8040_0000u32, 0x0080_0000u32);

    let cases: &[(u32, u32, u32)] = &[
        // Zero or denormal times anything, non-finite included, is +0.
        (f(0.0), f(f32::INFINITY), f(0.0)),
        (f(-0.0), f(f32::NAN), f(0.0)),
        (f(f32::NAN), f(0.0), f(0.0)),
        (denorm, f(f32::INFINITY), f(0.0)),
        (f(f32::NAN), neg_denorm, f(0.0)),
        (denorm, f(2.0f32.powi(100)), f(0.0)),
        // The smallest normal is not zero.
        (min_normal, f(2.0), f(f32::MIN_POSITIVE * 2.0)),
        // Ordinary products.
        (f(1.5), f(-4.0), f(-6.0)),
        (f(3.0), f(0.5), f(1.5)),
    ];

    let inputs: Vec<[u32; 4]> = cases.iter().map(|&(a, b, _)| [a, b, 0, 0]).collect();
    run_cases(&shader, &inputs, |backend, i, inp, got| {
        let want = cases[i].2;
        assert_eq!(
            got, want,
            "{backend}: v_mul_legacy_f32({:#010x}, {:#010x}) = {got:#010x}, want {want:#010x}",
            inp[0], inp[1],
        );
    });
}

fn build_wave32_cs(body: &[u32]) -> HostShader {
    let mut words = Vec::new();
    // s[4:5] = descriptor table address (heap index 0 -> byte offset 0).
    words.push(s_mov_b32(5, 0x80)); // inline constant 0
    words.extend_from_slice(&s_load_dwordx4(8, 4));
    words.extend_from_slice(body);
    // v2 = v0 * 4 (per-lane byte offset into the UAV), store v1 at out[v0].
    words.push((0x01A << 25) | (2 << 17) | 0x82); // v_lshlrev_b32 v2, 2, v0
    words.extend_from_slice(&[
        0x38 << 26 | 0x1C << 18 | (1 << 12), // buffer_store_dword ... offen
        0x80 << 24 | ((8 >> 2) << 16) | (1 << 8) | 2,
    ]);
    words.push(S_ENDPGM);

    let code: Vec<u8> = words.iter().flat_map(|w| w.to_le_bytes()).collect();
    let native = NativeProgram {
        stage: HwStage::Cs,
        wave64: false,
        vgprs: 8,
        layout: vec![
            RootParam {
                root: RootType::Constants,
                table_offset_dwords: 0,
                sgprs: 4,
                base_register: 0,
                user_data: 0,
                slot: 0,
            },
            RootParam {
                root: RootType::Table,
                table_offset_dwords: 0,
                sgprs: 1,
                base_register: 0,
                user_data: 4,
                slot: 1,
            },
        ],
        used: vec![UsedResource {
            kind: ResourceType::Uav,
            root: RootType::Table,
            table_offset_dwords: 0,
            lower_bound: 0,
            stride: 0,
            dwords: 4,
            user_data: 4,
            slot: 1,
        }],
        io: StageIo::Cs(ComputeIo {
            rsrc2: 5 << 1, // 5 user SGPRs: s0..s4
            threads: [32, 1, 1],
        }),
        code,
    };

    let p = ir::build(&native.code, native.wave64, native.vgprs).expect("ir::build");
    let st = structure::structure(&p).expect("structure");
    let res =
        resource::track(&p, &native.layout, native.stage, Some(&st)).expect("resource::track");
    let io = io::raise(&p, &native).expect("io::raise");
    let lowered = lower::lower(&lower::Parts {
        native: &native,
        p: &p,
        st: &st,
        io: &io,
        res: &res,
        inputs: Vec::new(),
        outputs: Vec::new(),
        prims: Vec::new(),
        mesh_stride: 0,
        target: crate::emit::Target::ANY,
        view_marks: false,
    })
    .expect("lower");

    let shader = HostShader {
        stage: HwStage::Cs,
        wave_size: lowered.wave_size,
        dxil: lowered.dxil,
        user_data: lowered.user_data,
        samplers: lowered.samplers,
        static_samplers: Vec::new(),
        root_cbvs: lowered.root_cbvs,
        root_srvs: lowered.root_srvs,
        root_uavs: lowered.root_uavs,
        memory: lowered.memory,
        memory_writes: lowered.memory_writes,
        view_addresses: lowered.view_addresses,
        view_marks: lowered.view_marks,
        heap: lowered.heap,
        heap_writes: lowered.heap_writes,
    };

    if let Ok(mut v) = validator::Validator::load(validator::Validator::default_path().as_deref()) {
        let mut copy = shader.dxil.clone();
        let ours = copy[4..20].to_vec();
        v.validate(&mut copy).expect("DXIL validation failed");
        assert_eq!(copy[4..20], ours[..], "container hash mismatch");
    }

    shader
}

fn dispatch_wave32(
    name: &str,
    shader: &HostShader,
    fixed_wave: bool,
    cases: &[([u32; 4], Vec<u32>)],
) {
    let _one = ONE_AT_A_TIME.lock().unwrap_or_else(|e| e.into_inner());
    let mut ran = 0;
    for (label, warp) in [("hardware", false), ("warp", true)] {
        let Ok(host) = Host::new(warp) else {
            eprintln!("{name}: no SM 6.6 {label} device; skipped");
            continue;
        };
        ran += 1;
        let (root, pso) = match host.compute_pipeline(shader) {
            Ok(p) => p,
            // WARP does not accept every fixed WaveSize requirement.
            Err(e)
                if (warp || std::env::var("VICTORY_WARP").is_ok_and(|v| v == "1"))
                    && fixed_wave =>
            {
                eprintln!("{name}: skipped on {label} ({e})");
                continue;
            }
            Err(e) => panic!("{name} ({label}): compute_pipeline failed: {e}"),
        };
        for (i, (inputs, want)) in cases.iter().enumerate() {
            let mut ud = vec![0u32; shader.user_data_dwords().max(1) as usize];
            for u in &shader.user_data {
                ud[u.index as usize] = match (u.root, u.slot) {
                    (RootType::Constants, 0) => inputs[u.dword as usize],
                    (RootType::Table, 1) => 0,
                    other => panic!("unexpected user data {other:?}"),
                };
            }
            let out = host
                .dispatch(
                    &root,
                    &pso,
                    &Dispatch {
                        roots: vec![(0, Root::Constants(ud))],
                        views: &[View::RawUav(32)],
                        samplers: &[],
                        groups: [1, 1, 1],
                    },
                )
                .unwrap_or_else(|e| panic!("{name} ({label}): dispatch failed: {e}"));
            assert_eq!(&out[0], want, "{name} ({label}): case {i} ({inputs:?})");
        }
    }
    assert!(
        ran > 0,
        "{name}: neither hardware nor WARP SM 6.6 device available"
    );
}

#[test]
fn lane_mask_phi_joins_execute_correctly() {
    // Branch join: s0 == 0 sets s2 = 0; otherwise s2 = (v0 < s1).
    let branch = build_wave32_cs(&[
        0xBF06_8000, // 0: s_cmp_eq_u32 s0, 0
        0xBF84_0002, // 4: s_cbranch_scc0 -> 16 (otherwise s2 = (v0 < s1))
        0xBE82_0380, // 8: s_mov_b32 s2, 0
        0xBF82_0002, // 12: s_branch -> 24
        0xD4C4_0002,
        u32::from(sgpr(1)) | (u32::from(vgpr(0)) << 9), // 16: v_cmp_gt_u32 s2, s1, v0 (v0 < s1)
        0xD501_0001,
        (0x8A) | (0x94 << 9) | (2 << 18), // 24: v_cndmask_b32 v1, 10, 20, s2
    ]);
    dispatch_wave32(
        "branch_zero_join",
        &branch,
        false,
        &[
            ([0, 16, 0, 0], vec![10; 32]),
            (
                [1, 16, 0, 0],
                (0..32).map(|t| if t < 16 { 20 } else { 10 }).collect(),
            ),
        ],
    );

    // Branch join with a partial constant mask: s0 == 0 sets s2 = 0xF; otherwise s2 = (v0 < s1).
    let partial_constant = build_wave32_cs(&[
        0xBF06_8000, // 0: s_cmp_eq_u32 s0, 0
        0xBF84_0002, // 4: s_cbranch_scc0 -> 16 (otherwise s2 = (v0 < s1))
        0xBE82_038F, // 8: s_mov_b32 s2, 15
        0xBF82_0002, // 12: s_branch -> 24
        0xD4C4_0002,
        u32::from(sgpr(1)) | (u32::from(vgpr(0)) << 9), // 16: v_cmp_gt_u32 s2, s1, v0 (v0 < s1)
        0xD501_0001,
        (0x8A) | (0x94 << 9) | (2 << 18), // 24: v_cndmask_b32 v1, 10, 20, s2
    ]);
    dispatch_wave32(
        "partial_constant_join",
        &partial_constant,
        true,
        &[
            (
                [0, 16, 0, 0],
                (0..32).map(|t| if t < 4 { 20 } else { 10 }).collect(),
            ),
            (
                [1, 16, 0, 0],
                (0..32).map(|t| if t < 16 { 20 } else { 10 }).collect(),
            ),
        ],
    );

    // Loop accumulator from 0: iterates i = 0..s0, reading phi s2 inside the
    // loop, accumulating `s2 |= (v0 == i)`, then reading final s2 after exit.
    // v1 = (inside last iter: 20 if v0 < s0 - 1 else 10) + (after loop: 30 if v0 < s0 else 10).
    let loop_shader = build_wave32_cs(&[
        0xBE82_0380, // 0: s_mov_b32 s2, 0
        0xBE86_0380, // 4: s_mov_b32 s6, 0
        0xD501_0001,
        (0x8A) | (0x94 << 9) | (2 << 18), // 8: v_cndmask_b32 v1, 10, 20, s2
        0x7D84_0006,                      // 16: v_cmp_eq_u32 vcc, s6, v0
        0x8802_026A,                      // 20: s_or_b32 s2, vcc_lo, s2
        0x8006_8106,                      // 24: s_add_u32 s6, s6, 1
        0xBF0A_0006,                      // 28: s_cmp_lt_u32 s6, s0
        0xBF85_FFF9,                      // 32: s_cbranch_scc1 -> 8
        0xD501_0003,
        (0x8A) | (0x9E << 9) | (2 << 18), // 36: v_cndmask_b32 v3, 10, 30, s2
        (0x025 << 25) | (1 << 17) | (3 << 9) | 257, // 44: v_add_nc_u32 v1, v1, v3
    ]);
    dispatch_wave32(
        "loop_accumulator",
        &loop_shader,
        false,
        &[(
            [5, 0, 0, 0],
            (0..32)
                .map(|t| {
                    if t < 4 {
                        50
                    } else if t == 4 {
                        40
                    } else {
                        20
                    }
                })
                .collect(),
        )],
    );
}

#[test]
fn lane_mask_phi_whole_read_and_readlane_execute_correctly() {
    let pack_whole_and_lanes = |body: &mut Vec<u32>| {
        // s6 = s2 & exec_lo (whole joined mask as a 32-bit integer)
        body.push(sop2(0x0E, 6, 2, 0x7E));
        // v2 = s2 ? 1 : 0; s12 = lane 0 of v2; s13 = lane 31 of v2
        body.extend_from_slice(&vop3(0x101, 2, 0x80, 0x81, sgpr(2)));
        body.extend_from_slice(&vop3(0x360, 12, vgpr(2), 0x80, 0));
        body.extend_from_slice(&vop3(0x360, 13, vgpr(2), 0x9F, 0));
        // out[t] = (t == 1) ? s12 : (t == 2) ? s13 : s6
        body.push(vop1(0x01, 1, sgpr(6)));
        body.push(vop1(0x01, 3, sgpr(12)));
        body.push(0x7D84_0081); // v_cmp_eq_u32 vcc_lo, 1, v0
        body.extend_from_slice(&vop3(0x101, 1, vgpr(1), vgpr(3), sgpr(0x6A)));
        body.push(vop1(0x01, 3, sgpr(13)));
        body.push(0x7D84_0082); // v_cmp_eq_u32 vcc_lo, 2, v0
        body.extend_from_slice(&vop3(0x101, 1, vgpr(1), vgpr(3), sgpr(0x6A)));
    };
    let expected = |mask: u32| -> Vec<u32> {
        let mut out = vec![mask; 32];
        out[1] = mask & 1;
        out[2] = (mask >> 31) & 1;
        out
    };

    // 1. Per-lane join where given(m, e) == Bool::TRUE inside the taken arm
    // (`s2 = vcc_lo` under `v0 < s0`), then read as a whole mask and via `v_readlane_b32`.
    let mut vcc_words = vec![
        0x7D88_0000,            // v_cmp_gt_u32 vcc_lo, s0, v0 (v0 < s0)
        s_mov_b32(2, 0x80),     // s2 = 0
        0xBE87_3C6A,            // s_and_saveexec_b32 s7, vcc_lo
        0xBF88_0001,            // s_cbranch_execz +1
        s_mov_b32(2, 0x6A),     // s2 = vcc_lo
        s_mov_b32(0x7E, 7),     // exec_lo = s7
    ];
    pack_whole_and_lanes(&mut vcc_words);
    let vcc_shader = build_wave32_cs(&vcc_words);
    let mask_of = |n: u32| u32::try_from(1u64.checked_shl(n).map_or(0u64, |v| v - 1)).unwrap_or(u32::MAX);
    let vcc_cases: Vec<([u32; 4], Vec<u32>)> = [0, 1, 7, 16, 31, 32]
        .into_iter()
        .map(|n| {
            let m = if n >= 32 { u32::MAX } else { mask_of(n) };
            ([n, 0, 0, 0], expected(m))
        })
        .collect();
    dispatch_wave32("per_lane_vcc_whole_mask_and_readlane", &vcc_shader, true, &vcc_cases);

    // 2. Per-lane join assigning a distinct sub-mask (`v0 < s1`) under `v0 < s0`.
    let mut sub_words = vec![
        0x7D88_0000,                            // v_cmp_gt_u32 vcc_lo, s0, v0 (v0 < s0)
        s_mov_b32(2, 0x80),                     // s2 = 0
        0xBE87_3C6A,                            // s_and_saveexec_b32 s7, vcc_lo
        0xBF88_0002,                            // s_cbranch_execz +2
        0xD4C4_0002,                            // v_cmp_gt_u32 s2, s1, v0 (v0 < s1)
        u32::from(sgpr(1)) | (u32::from(vgpr(0)) << 9),
        s_mov_b32(0x7E, 7),                     // exec_lo = s7
    ];
    pack_whole_and_lanes(&mut sub_words);
    let sub_shader = build_wave32_cs(&sub_words);
    let sub_cases: Vec<([u32; 4], Vec<u32>)> = [(0, 16), (16, 0), (16, 8), (8, 24), (32, 1), (32, 32)]
        .into_iter()
        .map(|(a, b)| {
            let k = a.min(b);
            let m = if k >= 32 { u32::MAX } else { mask_of(k) };
            ([a, b, 0, 0], expected(m))
        })
        .collect();
    dispatch_wave32("per_lane_submask_whole_mask_and_readlane", &sub_shader, true, &sub_cases);
}

/// VOP2 `op vdst, src0, vsrc1` (`src0`: 0..105 = SGPR, 256..511 = VGPR).
const fn vop2(op: u8, vdst: u8, src0: u16, vsrc1: u8) -> u32 {
    (op as u32 & 0x3F) << 25 | (vdst as u32) << 17 | (vsrc1 as u32) << 9 | (src0 as u32 & 0x1FF)
}

/// VOP3P `op vdst, src0, src1, src2` with its packed modifiers: `op_sel` and
/// `op_sel_hi` pick each source's half for the low and high lanes of the
/// operation (bit k for source k), `neg_lo`/`neg_hi` negate them.
#[allow(clippy::too_many_arguments)]
const fn vop3p_mods(
    op: u8,
    vdst: u8,
    [src0, src1, src2]: [u16; 3],
    op_sel: u8,
    op_sel_hi: u8,
    neg_lo: u8,
    neg_hi: u8,
    clamp: bool,
) -> [u32; 2] {
    [
        0x33 << 26
            | (op as u32 & 0x7F) << 16
            | (clamp as u32) << 15
            | (op_sel_hi as u32 >> 2 & 1) << 14
            | (op_sel as u32 & 7) << 11
            | (neg_hi as u32 & 7) << 8
            | vdst as u32,
        (src0 as u32 & 0x1FF)
            | (src1 as u32 & 0x1FF) << 9
            | (src2 as u32 & 0x1FF) << 18
            | (op_sel_hi as u32 & 3) << 27
            | (neg_lo as u32 & 7) << 29,
    ]
}

/// Inputs `[src0, src1, src2, _]` and the f32 bits RDNA2 gives for
/// `src0.f16[0] * src1.f16[0] + src0.f16[1] * src1.f16[1] + src2.f32`: f16
/// products are exact in f32, so 65504 * 65504 does not overflow as it would
/// in f16.
const DOT2_F32_F16_CASES: &[([u32; 4], u32)] = &[
    // (1, 2) . (3, 4) + 0.5 = 11.5
    ([0x4000_3c00, 0x4400_4200, 0x3f00_0000, 0], 0x4138_0000),
    // (-1.5, 0.25) . (2, -8) - 1 = -6
    ([0x3400_be00, 0xc800_4000, 0xbf80_0000, 0], 0xc0c0_0000),
    // (65504, 65504) . (65504, 0) + 0 = 4290774016
    ([0x7bff_7bff, 0x0000_7bff, 0, 0], 0x4f7f_c004),
    // (inf, 0) . (1, 0) + 0 = inf
    ([0x0000_7c00, 0x0000_3c00, 0, 0], 0x7f80_0000),
];

#[test]
fn v_dot2_f32_f16_sums_f16_products_in_f32() {
    // v0 = v_dot2_f32_f16 v1, v2, v3 (no modifiers: op_sel_hi = 0b111)
    let body = vop3p_mods(0x13, 0, [vgpr(1), vgpr(2), vgpr(3)], 0, 0b111, 0, 0, false);
    let shader = build_valu_cs(&body);
    let inputs: Vec<[u32; 4]> = DOT2_F32_F16_CASES.iter().map(|c| c.0).collect();
    run_cases(&shader, &inputs, |backend, i, inp, got| {
        let want = DOT2_F32_F16_CASES[i].1;
        assert_eq!(
            got, want,
            "{backend}: v_dot2_f32_f16({:#010x}, {:#010x}, {:#010x}) = {got:#010x}, want {want:#010x}",
            inp[0], inp[1], inp[2],
        );
    });
}

#[test]
fn v_dot2_f32_f16_applies_op_sel_and_negation_per_half() {
    // src0's halves swapped (op_sel bit 0 set, op_sel_hi bit 0 clear), src1's
    // low half negated (neg_lo bit 1), src0's high-lane half negated (neg_hi
    // bit 0): (1, 2) . (3, 4) + 0.5 becomes 2 * -3 + -1 * 4 + 0.5 = -9.5.
    let body = vop3p_mods(
        0x13,
        0,
        [vgpr(1), vgpr(2), vgpr(3)],
        0b001,
        0b110,
        0b010,
        0b001,
        false,
    );
    let shader = build_valu_cs(&body);
    let cases: &[([u32; 4], u32)] = &[
        ([0x4000_3c00, 0x4400_4200, 0x3f00_0000, 0], 0xc118_0000),
        // (-1.5, 0.25) . (2, -8) - 1 with the same modifiers:
        // 0.25 * -2 + 1.5 * -8 - 1 = -13.5
        ([0x3400_be00, 0xc800_4000, 0xbf80_0000, 0], 0xc158_0000),
    ];
    let inputs: Vec<[u32; 4]> = cases.iter().map(|c| c.0).collect();
    run_cases(&shader, &inputs, |backend, i, inp, got| {
        let want = cases[i].1;
        assert_eq!(
            got, want,
            "{backend}: v_dot2_f32_f16 with modifiers({:#010x}, {:#010x}, {:#010x}) = {got:#010x}, want {want:#010x}",
            inp[0], inp[1], inp[2],
        );
    });
}

#[test]
fn v_dot2_f32_f16_clamps_to_zero_one() {
    let body = vop3p_mods(0x13, 0, [vgpr(1), vgpr(2), vgpr(3)], 0, 0b111, 0, 0, true);
    let shader = build_valu_cs(&body);
    let cases: &[([u32; 4], u32)] = &[
        // 11.5 clamps to 1.0, -6 to 0.0
        ([0x4000_3c00, 0x4400_4200, 0x3f00_0000, 0], 0x3f80_0000),
        ([0x3400_be00, 0xc800_4000, 0xbf80_0000, 0], 0),
    ];
    let inputs: Vec<[u32; 4]> = cases.iter().map(|c| c.0).collect();
    run_cases(&shader, &inputs, |backend, i, inp, got| {
        let want = cases[i].1;
        assert_eq!(
            got, want,
            "{backend}: v_dot2_f32_f16 clamp({:#010x}, {:#010x}, {:#010x}) = {got:#010x}, want {want:#010x}",
            inp[0], inp[1], inp[2],
        );
    });
}

#[test]
fn v_dot2c_f32_f16_accumulates_into_its_destination() {
    // v3 = v_dot2c_f32_f16 v1, v2 (v3 += v1 . v2), then v0 = v3.
    let body = [vop2(0x02, 3, vgpr(1), 2), vop1(0x001, 0, vgpr(3))];
    let shader = build_valu_cs(&body);
    let inputs: Vec<[u32; 4]> = DOT2_F32_F16_CASES.iter().map(|c| c.0).collect();
    run_cases(&shader, &inputs, |backend, i, inp, got| {
        let want = DOT2_F32_F16_CASES[i].1;
        assert_eq!(
            got, want,
            "{backend}: v_dot2c_f32_f16({:#010x}, {:#010x}) + {:#010x} = {got:#010x}, want {want:#010x}",
            inp[0], inp[1], inp[2],
        );
    });
}

#[test]
fn v_dot2c_f32_f16_refuses_inline_constants() {
    // v3 = v_dot2c_f32_f16 1.0, v2 (242) and 1, v2 (129): how VOP2 widens an
    // inline constant to a pair of halves is not measured.
    for constant in [242, 129] {
        let body = [vop2(0x02, 3, constant, 2), vop1(0x001, 0, vgpr(3))];
        let refused = try_build_valu_cs(&body);
        assert!(
            matches!(
                refused,
                Err(crate::ShaderError::Unsupported {
                    what: "an inline constant of v_dot2c_f32_f16",
                    ..
                })
            ),
            "source {constant}: {:?}",
            refused.err()
        );
    }
}

/// Inputs where rounding the sum of the two products before adding the
/// addend gives a different f32 than RDNA2, which rounds the exact
/// `p0 + p1 + addend` once (measured on gfx1036 in #465).
const DOT2_ROUNDED_ONCE_CASES: &[([u32; 4], u32)] = &[
    // (1, 2^-12) . (1, 2^-12) - 1 = 2^-24; 1 + 2^-24 rounded first gives 0.
    ([0x0c00_3c00, 0x0c00_3c00, 0xbf80_0000, 0], 0x3380_0000),
    // (1, 2^-14) . (1, 2^-14) + 2^24 = 2^24 + 1 + 2^-28: above the tie
    // between 2^24 and 2^24 + 2, so it rounds up; 1 + 2^-28 rounded first
    // lands on the tie, which rounds to even (2^24).
    ([0x0400_3c00, 0x0400_3c00, 0x4b80_0000, 0], 0x4b80_0001),
    // (-0, -0) . (1, 1) - 0 = -0: the sign of an all-negative-zero sum.
    ([0x8000_8000, 0x3c00_3c00, 0x8000_0000, 0], 0x8000_0000),
];

#[test]
fn v_dot2_f32_f16_rounds_the_exact_sum_once() {
    let body = vop3p_mods(0x13, 0, [vgpr(1), vgpr(2), vgpr(3)], 0, 0b111, 0, 0, false);
    let shader = build_valu_cs(&body);
    let inputs: Vec<[u32; 4]> = DOT2_ROUNDED_ONCE_CASES.iter().map(|c| c.0).collect();
    run_cases(&shader, &inputs, |backend, i, inp, got| {
        let want = DOT2_ROUNDED_ONCE_CASES[i].1;
        assert_eq!(
            got, want,
            "{backend}: v_dot2_f32_f16({:#010x}, {:#010x}, {:#010x}) = {got:#010x}, want {want:#010x}",
            inp[0], inp[1], inp[2],
        );
    });
}

#[test]
fn v_dot2c_f32_f16_rounds_the_exact_sum_once() {
    let body = [vop2(0x02, 3, vgpr(1), 2), vop1(0x001, 0, vgpr(3))];
    let shader = build_valu_cs(&body);
    let inputs: Vec<[u32; 4]> = DOT2_ROUNDED_ONCE_CASES.iter().map(|c| c.0).collect();
    run_cases(&shader, &inputs, |backend, i, inp, got| {
        let want = DOT2_ROUNDED_ONCE_CASES[i].1;
        assert_eq!(
            got, want,
            "{backend}: v_dot2c_f32_f16({:#010x}, {:#010x}) + {:#010x} = {got:#010x}, want {want:#010x}",
            inp[0], inp[1], inp[2],
        );
    });
}

#[test]
fn compute_group_info_and_waveid_in_workgroup_match_the_group() {
    // 1 user SGPR (table base in s0), TGID_X in s1 (bit 7), TG_SIZE in s2 (bit
    // 10), and all three thread-id VGPRs in v0..v2 (`TIDIG_COMP_CNT = 2` at
    // bits 12:11). For `[8, 4, 3]` = 96 threads: wave32 gives 3 full waves,
    // wave64 gives 2 waves (the second partial, 32 threads). Each thread `f`
    // writes `[s_tg_size, s_wave_id]` to `uav[2 * f .. 2 * f + 2]`.
    let build = |wave64: bool| -> HostShader {
        let mut words = vec![
            s_mov_b32(4, 0),
            s_mov_b32(5, 0x80),
            s_load_dwordx4(8, 4)[0],
            s_load_dwordx4(8, 4)[1],
            // s_get_waveid_in_workgroup s3
            0xF4A8_00C0,
            0xFA00_0000,
            0xBF8C_C07F, // s_waitcnt lgkmcnt(0)
        ];
        // flat = (v2 * 4 + v1) * 8 + v0; byte_off = flat * 8 in v3.
        words.extend_from_slice(&vop3(0x11A, 3, 128 + 2, vgpr(2), 0));
        words.extend_from_slice(&vop3(0x125, 3, vgpr(3), vgpr(1), 0));
        words.extend_from_slice(&vop3(0x11A, 3, 128 + 3, vgpr(3), 0));
        words.extend_from_slice(&vop3(0x125, 3, vgpr(3), vgpr(0), 0));
        words.extend_from_slice(&vop3(0x11A, 3, 128 + 3, vgpr(3), 0));
        words.push(vop1(0x001, 4, sgpr(2))); // v4 = TG_SIZE
        words.push(vop1(0x001, 5, sgpr(3))); // v5 = s_get_waveid_in_workgroup
        // buffer_store_dword v4 / v5, v3, s[8:11], 0 offen offset:0 / 4
        words.extend_from_slice(&[0xE070_1000, 0x8002_0403, 0xE070_1004, 0x8002_0503]);
        words.push(S_ENDPGM);

        let native = NativeProgram {
            stage: HwStage::Cs,
            wave64,
            vgprs: 8,
            layout: vec![RootParam {
                root: RootType::Table,
                table_offset_dwords: 0,
                sgprs: 1,
                base_register: 0,
                user_data: 0,
                slot: 0,
            }],
            used: vec![UsedResource {
                kind: ResourceType::Uav,
                root: RootType::Table,
                table_offset_dwords: 0,
                lower_bound: 0,
                stride: 0,
                dwords: 4,
                user_data: 0,
                slot: 0,
            }],
            io: StageIo::Cs(ComputeIo {
                rsrc2: 1 << 1 | 1 << 7 | 1 << 10 | 2 << 11,
                threads: [8, 4, 3],
            }),
            code: words.iter().flat_map(|w| w.to_le_bytes()).collect(),
        };
        let (p, st) = range::build(&native.code, wave64, native.vgprs).expect("range");
        let st = st.expect("structure");
        let res = resource::track(&p, &native.layout, native.stage, Some(&st)).expect("resource");
        let io = io::raise(&p, &native).expect("io");
        let lowered = lower::lower(&lower::Parts {
            native: &native,
            p: &p,
            st: &st,
            io: &io,
            res: &res,
            inputs: Vec::new(),
            outputs: Vec::new(),
            prims: Vec::new(),
            mesh_stride: 0,
            target: crate::emit::Target::ANY,
            view_marks: false,
        })
        .expect("lower");
        if let Ok(mut v) =
            validator::Validator::load(validator::Validator::default_path().as_deref())
        {
            let mut copy = lowered.dxil.clone();
            let ours = copy[4..20].to_vec();
            v.validate(&mut copy).expect("DXIL validation failed");
            assert_eq!(copy[4..20], ours[..], "container hash mismatch");
        }
        HostShader {
            stage: HwStage::Cs,
            wave_size: lowered.wave_size,
            dxil: lowered.dxil,
            user_data: lowered.user_data,
            samplers: lowered.samplers,
            static_samplers: Vec::new(),
            root_cbvs: lowered.root_cbvs,
            root_srvs: lowered.root_srvs,
            root_uavs: lowered.root_uavs,
            memory: lowered.memory,
            memory_writes: lowered.memory_writes,
            view_addresses: lowered.view_addresses,
            heap: lowered.heap,
            heap_writes: lowered.heap_writes,
            view_marks: lowered.view_marks,
        }
    };

    let shaders = [(false, 32u32, build(false)), (true, 64u32, build(true))];
    let _one = ONE_AT_A_TIME.lock().unwrap_or_else(|e| e.into_inner());
    for (label, warp) in [("hardware", false), ("warp", true)] {
        let Ok(host) = Host::new(warp) else {
            eprintln!("no SM 6.6 {label} device; skipped");
            continue;
        };
        for &(wave64, wave, ref shader) in &shaders {
            let (root, pso) = host
                .compute_pipeline(shader)
                .unwrap_or_else(|e| panic!("{label} (wave64={wave64}): {e}"));
            let ud = vec![0u32; shader.user_data_dwords().max(1) as usize];
            let out = host
                .dispatch(
                    &root,
                    &pso,
                    &Dispatch {
                        roots: vec![(0, Root::Constants(ud))],
                        views: &[View::RawUav(96 * 2)],
                        samplers: &[],
                        groups: [1, 1, 1],
                    },
                )
                .unwrap_or_else(|e| panic!("{label} (wave64={wave64}): {e}"));
            let count = 96u32.div_ceil(wave);
            for flat in 0..96u32 {
                let w = flat / wave;
                let first = if w == 0 { 1u32 << 31 } else { 0 };
                let want_info = first | (w << 6) | count;
                let got_info = out[0][(flat * 2) as usize];
                let got_wave = out[0][(flat * 2 + 1) as usize];
                assert_eq!(
                    (got_info, got_wave),
                    (want_info, w),
                    "{label} (wave64={wave64}, flat={flat})"
                );
            }
        }
    }
}

/// Rotations with a mask (offset[4:0]): the mask's bits of the lane index
/// stay, the others rotate, as the ISA's pseudocode has it. Its examples for
/// mask 1 list other lanes; the lanes here are the ones measured on RDNA2
/// hardware (gfx1036, by Lyrnc in Marvisak/VIctory#606, who gives the first
/// and last entries of each list and reports the pseudocode matching all 2048
/// offsets 0xC000-0xC7FF, and bits 11 and 12 being ignored).
#[test]
fn ds_swizzle_masked_rotations() {
    // The lane each lane reads.
    let cases: [(u16, [u32; 32]); 4] = [
        // Left by 1, mask 1: 0, 3, 2, 5, 4, 7, 6, 9, 8, ..., 28, 31, 30, 1.
        (
            0xC021,
            [
                0, 3, 2, 5, 4, 7, 6, 9, 8, 11, 10, 13, 12, 15, 14, 17, 16, 19, 18, 21, 20, 23, 22,
                25, 24, 27, 26, 29, 28, 31, 30, 1,
            ],
        ),
        // Right by 1, mask 1: 30, 1, 0, 3, 2, 5, 4, ...
        (
            0xC421,
            [
                30, 1, 0, 3, 2, 5, 4, 7, 6, 9, 8, 11, 10, 13, 12, 15, 14, 17, 16, 19, 18, 21, 20,
                23, 22, 25, 24, 27, 26, 29, 28, 31,
            ],
        ),
        // Left by 1, mask 3: 0, 1, 2, 7, 4, 5, 6, 11, ...
        (
            0xC023,
            [
                0, 1, 2, 7, 4, 5, 6, 11, 8, 9, 10, 15, 12, 13, 14, 19, 16, 17, 18, 23, 20, 21, 22,
                27, 24, 25, 26, 31, 28, 29, 30, 3,
            ],
        ),
        // Bits 11 and 12 set: as 0xC021.
        (
            0xD821,
            [
                0, 3, 2, 5, 4, 7, 6, 9, 8, 11, 10, 13, 12, 15, 14, 17, 16, 19, 18, 21, 20, 23, 22,
                25, 24, 27, 26, 29, 28, 31, 30, 1,
            ],
        ),
    ];
    for (offset, lanes) in cases {
        // Lane i holds i + 1.
        let want: Vec<u32> = lanes.iter().map(|j| j + 1).collect();
        for got in swizzle_32(offset) {
            assert_eq!(got, want, "offset {offset:#06x}");
        }
    }
}

/// RDNA2 ignores VOP3 `op_sel` on 16-bit compares (reading the low halves
/// regardless), while non-compare 16-bit ops such as `v_add_nc_u16` honour it.
#[test]
fn cmp16_ignores_op_sel_as_rdna2_does() {
    let vop3_opsel = |op: u16, dst: u8, s0: u16, s1: u16, opsel: u8| -> [u32; 2] {
        let [w0, w1] = vop3(op, dst, s0, s1, 0);
        [w0 | (u32::from(opsel) << 11), w1]
    };

    // High and low halves give opposite outcomes for every non-zero source op_sel.
    let f16_cases: &[([u32; 4], u32)] = &[
        // hi 2.0 vs 1.0, lo 1.0 < 1.5 -> true only when both read the low half
        ([0x4000_3C00, 0x3C00_3E00, 0, 0], 1),
        // hi 1.0 vs 3.0, lo 2.0 < 1.5 -> false only when both read the low half
        ([0x3C00_4000, 0x4200_3E00, 0, 0], 0),
    ];
    let u16_cases: &[([u32; 4], u32)] = &[
        // hi 10 vs 1, lo 1 < 2 -> true
        ([0x000A_0001, 0x0001_0002, 0, 0], 1),
        // hi 1 vs 20, lo 10 < 2 -> false
        ([0x0001_000A, 0x0014_0002, 0, 0], 0),
    ];
    let i16_cases: &[([u32; 4], u32)] = &[
        // hi 1 vs -2, lo -1 < 0 -> true
        ([0x0001_FFFF, 0xFFFE_0000, 0, 0], 1),
        // hi -2 vs 2, lo 1 < -1 -> false
        ([0xFFFE_0001, 0x0002_FFFF, 0, 0], 0),
    ];

    let mut cmp_shaders = Vec::new();
    for (op, what, cases) in [
        (0x0c9u16, "v_cmp_lt_f16", f16_cases),
        (0x0a9, "v_cmp_lt_u16", u16_cases),
        (0x089, "v_cmp_lt_i16", i16_cases),
    ] {
        for opsel in [0b0001u8, 0b0010, 0b0011] {
            let mut body = vop3_opsel(op, 6, vgpr(1), vgpr(2), opsel).to_vec();
            body.extend_from_slice(&vop3(0x101, 0, 0x80, 0x81, sgpr(6)));
            cmp_shaders.push((what, opsel, build_valu_cs(&body), cases));
        }
    }

    // v_cmpx_lt_f16_e64 (0x0d9) writing EXEC directly also ignores op_sel.
    let [w0, w1] = vop3_opsel(0x0d9, 126, vgpr(1), vgpr(2), 0b0001);
    let cmpx = build_valu_cs(&[
        s_mov_b32(6, 126),    // save exec_lo
        vop1(0x001, 0, 0x80), // v0 = 0
        w0,
        w1,                   // v_cmpx_lt_f16_e64 exec_lo, v1, v2 op_sel:[1,0]
        vop1(0x001, 0, 0x81), // v0 = 1 if active
        s_mov_b32(126, 6),    // restore exec_lo
    ]);

    // Control: v_add_nc_u16 (0x303) still honours op_sel.
    let [w0, w1] = vop3_opsel(0x303, 0, vgpr(1), vgpr(2), 0b0001);
    let add_hi = build_valu_cs(&[vop1(0x001, 0, 0x80), w0, w1]);

    for (what, opsel, shader, cases) in &cmp_shaders {
        let inputs: Vec<[u32; 4]> = cases.iter().map(|c| c.0).collect();
        run_cases(shader, &inputs, |backend, i, inp, got| {
            let want = cases[i].1;
            assert_eq!(
                got, want,
                "{backend}: {what} opsel={opsel:#b} ({:#010x}, {:#010x}) = {got}, want {want}",
                inp[0], inp[1],
            );
        });
    }

    let f16_in: Vec<[u32; 4]> = f16_cases.iter().map(|c| c.0).collect();
    run_cases(&cmpx, &f16_in, |backend, i, inp, got| {
        let want = f16_cases[i].1;
        assert_eq!(
            got, want,
            "{backend}: v_cmpx_lt_f16 opsel=0b1 ({:#010x}, {:#010x}) = {got}, want {want}",
            inp[0], inp[1],
        );
    });

    run_cases(
        &add_hi,
        &[[0x000A_0001, 0x0014_0002, 0, 0]],
        |backend, _, _, got| {
            assert_eq!(
                got, 12,
                "{backend}: v_add_nc_u16 op_sel:[1,0] = {got}, want 12"
            );
        },
    );
}

#[test]
fn cmp16_sdwa_preserves_word_selects() {
    // llvm-mc -mcpu=gfx1030: v_cmp_lt_{f16,i16,u16}_sdwa vcc_lo, v1, v2.
    let instructions = [
        (0x7d92_04f9, [0x4000_3c00, 0x3800_3e00, 0, 0]),
        (0x7d12_04f9, [0x0003_ffff, 0xfffe_0000, 0, 0]),
        (0x7d52_04f9, [0x0003_0001, 0x0000_0002, 0, 0]),
    ];
    let select = vop3(0x101, 0, 0x80, 0x81, sgpr(106));
    for (instruction, input) in instructions {
        // Low/low gives true. Selecting either high word gives false.
        for (extension, expected) in [(0x0404_0001, 1), (0x0405_0001, 0), (0x0504_0001, 0)] {
            let shader = build_valu_cs(&[instruction, extension, select[0], select[1]]);
            run_cases(&shader, &[input], |backend, _, _, actual| {
                assert_eq!(
                    actual, expected,
                    "{backend}: {instruction:#x} {extension:#x}"
                );
            });
        }
    }
}

/// VOP3P `op vdst, src0, src1, src2` with standard packed lane selects
/// (`opsel = 0b000`, `opsel_hi = 0b111`) and optional `clamp`.
const fn vop3p(op: u16, vdst: u8, src0: u16, src1: u16, src2: u16, clamp: bool) -> [u32; 2] {
    [
        0x33 << 26
            | (op as u32 & 0x7F) << 16
            | (clamp as u32) << 15
            | (1 << 14) // opsel_hi[2] = 1
            | vdst as u32,
        (0b11 << 27) // opsel_hi[1:0] = 0b11
            | (src0 as u32 & 0x1FF)
            | (src1 as u32 & 0x1FF) << 9
            | (src2 as u32 & 0x1FF) << 18,
    ]
}

#[test]
fn v_pk_int16_clamp_saturates_as_rdna2() {
    let pack_u = |lo: u16, hi: u16| u32::from(lo) | (u32::from(hi) << 16);
    let pack_i = |lo: i16, hi: i16| pack_u(lo as u16, hi as u16);

    // VOP3P opcodes: 0x00=mad_i16, 0x02=add_i16, 0x03=sub_i16, 0x09=mad_u16, 0x0a=add_u16, 0x0b=sub_u16.
    let add_u16 = build_valu_cs(&vop3p(0x0a, 0, vgpr(1), vgpr(2), 0, true));
    let sub_u16 = build_valu_cs(&vop3p(0x0b, 0, vgpr(1), vgpr(2), 0, true));
    let mad_u16 = build_valu_cs(&vop3p(0x09, 0, vgpr(1), vgpr(2), vgpr(3), true));
    let add_i16 = build_valu_cs(&vop3p(0x02, 0, vgpr(1), vgpr(2), 0, true));
    let sub_i16 = build_valu_cs(&vop3p(0x03, 0, vgpr(1), vgpr(2), 0, true));
    let mad_i16 = build_valu_cs(&vop3p(0x00, 0, vgpr(1), vgpr(2), vgpr(3), true));

    // add_u16 clamp: lo overflows (60000+10000 -> 65535), hi fits (100+200 -> 300)
    run_cases(
        &add_u16,
        &[
            [pack_u(60000, 100), pack_u(10000, 200), 0, 0],
            [pack_u(u16::MAX, 0), pack_u(u16::MAX, 0), 0, 0],
        ],
        |backend, i, _, got| {
            let want = [pack_u(u16::MAX, 300), pack_u(u16::MAX, 0)][i];
            assert_eq!(got, want, "{backend}: v_pk_add_u16 clamp case {i}");
        },
    );

    // sub_u16 clamp: lo underflows (5-10 -> 0), hi fits (500-200 -> 300)
    run_cases(
        &sub_u16,
        &[
            [pack_u(5, 500), pack_u(10, 200), 0, 0],
            [pack_u(0, u16::MAX), pack_u(u16::MAX, 0), 0, 0],
        ],
        |backend, i, _, got| {
            let want = [pack_u(0, 300), pack_u(0, u16::MAX)][i];
            assert_eq!(got, want, "{backend}: v_pk_sub_u16 clamp case {i}");
        },
    );

    // mad_u16 clamp: lo max*max+max -> 65535 (tests no u32 wrap!), hi 20*30+40 -> 640
    run_cases(
        &mad_u16,
        &[[
            pack_u(u16::MAX, 20),
            pack_u(u16::MAX, 30),
            pack_u(u16::MAX, 40),
            0,
        ]],
        |backend, _, _, got| {
            let want = pack_u(u16::MAX, 640);
            assert_eq!(got, want, "{backend}: v_pk_mad_u16 clamp");
        },
    );

    // add_i16 clamp: lo +overflow (30000+10000 -> 32767), hi -overflow (-30000-10000 -> -32768)
    run_cases(
        &add_i16,
        &[
            [pack_i(30000, -30000), pack_i(10000, -10000), 0, 0],
            [pack_i(100, -50), pack_i(-25, 20), 0, 0],
        ],
        |backend, i, _, got| {
            let want = [pack_i(i16::MAX, i16::MIN), pack_i(75, -30)][i];
            assert_eq!(got, want, "{backend}: v_pk_add_i16 clamp case {i}");
        },
    );

    // sub_i16 clamp: lo -overflow (-30000-10000 -> -32768), hi +overflow (30000-(-10000) -> 32767)
    run_cases(
        &sub_i16,
        &[
            [pack_i(-30000, 30000), pack_i(10000, -10000), 0, 0],
            [pack_i(0, -100), pack_i(i16::MIN, -200), 0, 0],
        ],
        |backend, i, _, got| {
            let want = [pack_i(i16::MIN, i16::MAX), pack_i(i16::MAX, 100)][i];
            assert_eq!(got, want, "{backend}: v_pk_sub_i16 clamp case {i}");
        },
    );

    // mad_i16 clamp: lo (-32768)*(-32768)+32767 -> 32767, hi (-32768)*(32767)-32768 -> -32768
    run_cases(
        &mad_i16,
        &[
            [
                pack_i(i16::MIN, i16::MIN),
                pack_i(i16::MIN, i16::MAX),
                pack_i(i16::MAX, i16::MIN),
                0,
            ],
            [pack_i(-10, 5), pack_i(20, -4), pack_i(50, -3), 0],
        ],
        |backend, i, _, got| {
            let want = [pack_i(i16::MAX, i16::MIN), pack_i(-150, -23)][i];
            assert_eq!(got, want, "{backend}: v_pk_mad_i16 clamp case {i}");
        },
    );
}

#[test]
fn v_mad_16_and_int16_clamp_match_rdna2() {
    let vop3_clamp = |op: u16, s0: u16, s1: u16, s2: u16, clamp: bool| {
        let [mut w0, w1] = vop3(op, 0, s0, s1, s2);
        if clamp {
            w0 |= 1 << 15;
        }
        [w0, w1]
    };
    let with_sentinel = |words: [u32; 2]| {
        build_valu_cs(&[
            vop1(0x001, 0, vgpr(4)), // v_mov_b32 v0, v4 (sentinel upper half)
            words[0],
            words[1],
        ])
    };

    // VOP3 opcodes: 0x303=v_add_nc_u16, 0x304=v_sub_nc_u16, 0x340=v_mad_u16, 0x35e=v_mad_i16.
    let mad_u16_wrap = with_sentinel(vop3_clamp(0x340, vgpr(1), vgpr(2), vgpr(3), false));
    let mad_i16_wrap = with_sentinel(vop3_clamp(0x35e, vgpr(1), vgpr(2), vgpr(3), false));
    let add_u16_sat = with_sentinel(vop3_clamp(0x303, vgpr(1), vgpr(2), 0, true));
    let sub_u16_sat = with_sentinel(vop3_clamp(0x304, vgpr(1), vgpr(2), 0, true));

    let sentinel = 0xBEEF_0000u32;

    // v_mad_u16 wrapping (CLAMP=1 stays refused until hardware-measured on RDNA2)
    let u_cases: &[(u16, u16, u16, u16)] = &[
        (20, 30, 40, 640),
        (300, 300, 100, (90_100u32 & 0xFFFF) as u16),
        (u16::MAX, u16::MAX, u16::MAX, 0),
    ];
    let u_in: Vec<[u32; 4]> = u_cases
        .iter()
        .map(|&(a, b, c, _)| [u32::from(a), u32::from(b), u32::from(c), sentinel])
        .collect();
    run_cases(&mad_u16_wrap, &u_in, |backend, i, _, got| {
        let want = sentinel | u32::from(u_cases[i].3);
        assert_eq!(got, want, "{backend}: v_mad_u16 wrap case {i}");
    });

    // v_mad_i16 wrapping (CLAMP=1 stays refused until hardware-measured on RDNA2)
    let i_cases: &[(i16, i16, i16, i16)] = &[
        (-10, 20, 50, -150),
        (i16::MIN, i16::MIN, i16::MAX, i16::MAX),
        (i16::MIN, i16::MAX, i16::MIN, 0),
        (200, 200, 100, (40_100i32 as i16)),
        (-200, 200, -100, (-40_100i32 as i16)),
    ];
    let i_in: Vec<[u32; 4]> = i_cases
        .iter()
        .map(|&(a, b, c, _)| [u32::from(a as u16), u32::from(b as u16), u32::from(c as u16), sentinel])
        .collect();
    run_cases(&mad_i16_wrap, &i_in, |backend, i, _, got| {
        let want = sentinel | u32::from(i_cases[i].3 as u16);
        assert_eq!(got, want, "{backend}: v_mad_i16 wrap case {i}");
    });

    for op in [0x340u16, 0x35e] {
        let err = try_build_valu_cs(&vop3_clamp(op, vgpr(1), vgpr(2), vgpr(3), true))
            .expect_err("clamped v_mad_{u16,i16} must stay refused");
        assert!(
            matches!(
                err,
                crate::ShaderError::Unsupported {
                    what: "modifiers on a 16-bit integer operation",
                    ..
                }
            ),
            "op {op:#x}: {err:?}"
        );
    }

    // v_add_nc_u16 clamp & v_sub_nc_u16 clamp
    run_cases(
        &add_u16_sat,
        &[
            [60000, 10000, 0, sentinel],
            [123, 456, 0, sentinel],
        ],
        |backend, i, _, got| {
            let want = sentinel | [u32::from(u16::MAX), 579][i];
            assert_eq!(got, want, "{backend}: v_add_nc_u16 clamp case {i}");
        },
    );
    run_cases(
        &sub_u16_sat,
        &[
            [5, 10, 0, sentinel],
            [500, 200, 0, sentinel],
        ],
        |backend, i, _, got| {
            let want = sentinel | [0, 300][i];
            assert_eq!(got, want, "{backend}: v_sub_nc_u16 clamp case {i}");
        },
    );

    // CLAMP=1 is a no-op (wrapped low 16 bits) on mul_lo, 16-bit shifts, and min/max.
    let noop_ops: &[(u16, &str, u16, u16, u16)] = &[
        (0x305, "v_mul_lo_u16", 300, 300, (90_000u32 & 0xFFFF) as u16),
        (0x314, "v_lshlrev_b16", 1, 0x8001, 0x0002),
        (0x307, "v_lshrrev_b16", 4, 0x8020, 0x0802),
        (0x308, "v_ashrrev_i16", 4, 0x8020, 0xF802),
        (0x309, "v_max_u16", 0xFFF0, 12, 0xFFF0),
        (0x30b, "v_min_u16", 0xFFF0, 12, 12),
        (0x30a, "v_max_i16", (-16i16) as u16, 12, 12),
        (0x30c, "v_min_i16", (-16i16) as u16, 12, (-16i16) as u16),
    ];
    for &(op, label, s0, s1, want_lo) in noop_ops {
        let cs = with_sentinel(vop3_clamp(op, vgpr(1), vgpr(2), 0, true));
        run_cases(
            &cs,
            &[[u32::from(s0), u32::from(s1), 0, sentinel]],
            |backend, _, _, got| {
                let want = sentinel | u32::from(want_lo);
                assert_eq!(got, want, "{backend}: {label} clamp no-op");
            },
        );
    }
}

#[test]
fn v_ldexp_f16_matches_rdna2() {
    // VOP2 `v_ldexp_f16 v0, v1, v2` (opcode 0x3b), preserving upper half from v3.
    let vop2_cs = build_valu_cs(&[
        vop1(0x001, 0, vgpr(3)),   // v_mov_b32 v0, v3 (sentinel upper half)
        vop2(0x3b, 0, vgpr(1), 2), // v_ldexp_f16 v0, v1, v2
    ]);
    // VOP3 `v_ldexp_f16 v0, |v1|, v2 clamp` (opcode 0x13b) with op_sel:[1, 0, 0, 1]
    // (reads high half of v1, low half of v2, writes high half of v0).
    let [mut w0, w1] = vop3(0x13b, 0, vgpr(1), vgpr(2), 0);
    w0 |= (1 << 8) | (1 << 11) | (1 << 14) | (1 << 15); // abs[0], opsel[0], opsel[3], clamp
    let vop3_cs = build_valu_cs(&[
        vop1(0x001, 0, vgpr(3)), // v_mov_b32 v0, v3 (sentinel lower half)
        w0,
        w1,
    ]);

    let sentinel = 0xCAFE_0000u32;
    // (f16 input, i16 exponent, expected f16 output)
    let cases: &[(u16, i16, u16)] = &[
        (0x3C00, 0, 0x3C00),         // 1.0 * 2^0 = 1.0
        (0x3E00, 3, 0x4A00),         // 1.5 * 2^3 = 12.0
        (0xC200, -2, 0xBA00),        // -3.0 * 2^-2 = -0.75
        (0x3C00, -15, 0x0200),       // 1.0 * 2^-15 = f16 subnormal 0x0200
        (0x0001, 24, 0x3C00),        // min f16 subnormal (2^-24) * 2^24 = 1.0
        (0x7BFF, 0, 0x7BFF),         // max finite f16 * 2^0 stays 65504 (no premature inf)
        (0x3C00, 16, 0x7C00),        // 1.0 * 2^16 = 65536 -> +inf (RTE overflow)
        (0xBC00, 1000, 0xFC00),      // -1.0 * 2^1000 -> -inf
        (0x0001, i16::MAX, 0x7C00),  // min subnormal * 2^32767 -> +inf
        (0x7BFF, i16::MIN, 0x0000),  // max finite * 2^-32768 -> +0.0
        (0xFBFF, i16::MIN, 0x8000),  // -max finite * 2^-32768 -> -0.0
        (0x0000, i16::MAX, 0x0000),  // +0.0 * 2^32767 = +0.0 (not NaN)
        (0x8000, i16::MAX, 0x8000),  // -0.0 * 2^32767 = -0.0 (not NaN)
        (0x7C00, i16::MIN, 0x7C00),  // +inf * 2^-32768 = +inf (not NaN)
        (0xFC00, i16::MIN, 0xFC00),  // -inf * 2^-32768 = -inf (not NaN)
    ];
    let inputs: Vec<[u32; 4]> = cases
        .iter()
        .map(|&(a, e, _)| [u32::from(a), u32::from(e as u16), sentinel, 0])
        .collect();
    run_cases(&vop2_cs, &inputs, |backend, i, _, got| {
        let (a, e, want_h) = cases[i];
        let want = sentinel | u32::from(want_h);
        assert_eq!(
            got, want,
            "{backend}: v_ldexp_f16({a:#06x}, {e}) = {got:#010x}, want {want:#010x}",
        );
    });

    // NaN stays NaN (upper half preserved).
    run_cases(&vop2_cs, &[[0x7E00, 4, sentinel, 0]], |backend, _, _, got| {
        assert_eq!(got & 0xFFFF_0000, sentinel);
        let lo = (got & 0x7FFF) as u16;
        assert!(lo > 0x7C00, "{backend}: v_ldexp_f16(NaN, 4) lo={lo:#06x}, want NaN");
    });

    // VOP3 abs[0] + opsel:[1,0,0,1] + clamp:
    // |-0.25(=0xB400)| * 2^1 = 0.5 (0x3800) into high half; |-1.0(=0xBC00)| * 2^2 = 4.0 -> clamped 1.0 (0x3C00)
    let lo_sentinel = 0x0000_1234u32;
    let vop3_cases: &[([u32; 4], u32)] = &[
        ([0xB400_0000, 1, lo_sentinel, 0], (0x3800u32 << 16) | lo_sentinel),
        ([0xBC00_0000, 2, lo_sentinel, 0], (0x3C00u32 << 16) | lo_sentinel),
    ];
    let vop3_in: Vec<[u32; 4]> = vop3_cases.iter().map(|c| c.0).collect();
    run_cases(&vop3_cs, &vop3_in, |backend, i, _, got| {
        let want = vop3_cases[i].1;
        assert_eq!(
            got, want,
            "{backend}: v_ldexp_f16_e64 case {i} = {got:#010x}, want {want:#010x}",
        );
    });
}

/// SMEM `s_load_dwordx4` (count 2) or `s_load_dwordx8` (count 3) with byte offset.
const fn smem_load(count_code: u8, dst: u8, base: u8, byte_off: u32) -> [u32; 2] {
    [
        0x3D << 26 | (count_code as u32) << 18 | (dst as u32) << 6 | (base as u32 >> 1),
        125 << 25 | (byte_off & 0x1F_FFFF),
    ]
}

/// MIMG instruction (`dim` 3 = CUBE).
const fn mimg_cube(
    op: u8,
    vdata: u8,
    vaddr: u8,
    srsrc: u8,
    ssamp: u8,
    dmask: u8,
    a16: bool,
) -> [u32; 2] {
    [
        0x3C << 26 | (op as u32) << 18 | (dmask as u32) << 8 | 3 << 3,
        (vaddr as u32)
            | (vdata as u32) << 8
            | ((srsrc as u32 >> 2) & 0x1F) << 16
            | ((ssamp as u32 >> 2) & 0x1F) << 21
            | (a16 as u32) << 30,
    ]
}

/// Builds a 1-thread compute shader with root constants `s0..s3` copied into
/// `v1..v4`, SRV 0 (`s[8:15]`) at view heap index 0, raw UAV (`s[20:23]`) at
/// view heap index 1, and point-clamp sampler (`s[16:19]`) at sampler index 0.
/// `body` writes `uav_dwords` result dwords starting at `v0`.
fn build_cube_cs(body: &[u32], uav_dwords: u8) -> HostShader {
    let mut words = Vec::new();
    // Copy user SGPR s5 (sampler table) into s6 before zeroing s5 for s[4:5].
    words.push(s_mov_b32(6, 5));
    words.push(s_mov_b32(7, 0x80));
    words.push(s_mov_b32(5, 0x80));
    words.extend_from_slice(&smem_load(0x03, 8, 4, 0));
    words.extend_from_slice(&smem_load(0x02, 16, 6, 0));
    words.extend_from_slice(&smem_load(0x02, 20, 4, 32));
    for i in 0..4u8 {
        words.push(vop1(0x001, i + 1, sgpr(i)));
    }
    words.extend_from_slice(body);
    for k in 0..uav_dwords {
        words.extend_from_slice(&buffer_store_dword(k, 20, 4 * u16::from(k)));
    }
    words.push(S_ENDPGM);

    let code: Vec<u8> = words.iter().flat_map(|w| w.to_le_bytes()).collect();
    let native = NativeProgram {
        stage: HwStage::Cs,
        wave64: false,
        vgprs: 16,
        layout: vec![
            RootParam {
                root: RootType::Constants,
                table_offset_dwords: 0,
                sgprs: 4,
                base_register: 0,
                user_data: 0,
                slot: 0,
            },
            RootParam {
                root: RootType::Table,
                table_offset_dwords: 0,
                sgprs: 1,
                base_register: 0,
                user_data: 4,
                slot: 1,
            },
            RootParam {
                root: RootType::Table,
                table_offset_dwords: 0,
                sgprs: 1,
                base_register: 0,
                user_data: 5,
                slot: 2,
            },
        ],
        used: vec![
            UsedResource {
                kind: ResourceType::Srv,
                root: RootType::Table,
                table_offset_dwords: 0,
                lower_bound: 0,
                stride: 0,
                dwords: 8,
                user_data: 4,
                slot: 1,
            },
            UsedResource {
                kind: ResourceType::Uav,
                root: RootType::Table,
                table_offset_dwords: 8,
                lower_bound: 0,
                stride: 0,
                dwords: 4,
                user_data: 4,
                slot: 1,
            },
            UsedResource {
                kind: ResourceType::Sampler,
                root: RootType::Table,
                table_offset_dwords: 0,
                lower_bound: 0,
                stride: 0,
                dwords: 4,
                user_data: 5,
                slot: 2,
            },
        ],
        io: StageIo::Cs(ComputeIo {
            rsrc2: 6 << 1,
            threads: [1, 1, 1],
        }),
        code,
    };

    let p = ir::build(&native.code, native.wave64, native.vgprs).expect("ir::build");
    let st = structure::structure(&p).expect("structure");
    let res =
        resource::track(&p, &native.layout, native.stage, Some(&st)).expect("resource::track");
    let io = io::raise(&p, &native).expect("io::raise");
    let lowered = lower::lower(&lower::Parts {
        native: &native,
        p: &p,
        st: &st,
        io: &io,
        res: &res,
        inputs: Vec::new(),
        outputs: Vec::new(),
        prims: Vec::new(),
        mesh_stride: 0,
        target: crate::emit::Target::ANY,
        view_marks: false,
    })
    .expect("lower");

    let shader = HostShader {
        stage: HwStage::Cs,
        wave_size: lowered.wave_size,
        dxil: lowered.dxil,
        user_data: lowered.user_data,
        samplers: lowered.samplers,
        static_samplers: Vec::new(),
        root_cbvs: lowered.root_cbvs,
        root_srvs: lowered.root_srvs,
        root_uavs: lowered.root_uavs,
        memory: lowered.memory,
        memory_writes: lowered.memory_writes,
        view_addresses: lowered.view_addresses,
        heap: lowered.heap,
        heap_writes: lowered.heap_writes,
        view_marks: lowered.view_marks,
    };

    if let Ok(mut v) = validator::Validator::load(validator::Validator::default_path().as_deref()) {
        let mut copy = shader.dxil.clone();
        let ours = copy[4..20].to_vec();
        v.validate(&mut copy).expect("DXIL validation failed");
        assert_eq!(copy[4..20], ours[..], "container hash mismatch");
    }

    shader
}

/// Runs `shader` on every SM 6.6 adapter (`Host::adapters`, including hardware
/// GPUs and WARP) with `srv` at heap index 0 and a raw UAV of `uav_dwords` at
/// heap index 1.
fn run_cube_cases(
    shader: &HostShader,
    srv: View,
    uav_dwords: u32,
    cases: &[[u32; 4]],
    mut check: impl FnMut(&str, usize, [u32; 4], &[u32]),
) {
    use windows::Win32::Graphics::Direct3D12::*;
    let samplers = [D3D12_SAMPLER_DESC {
        Filter: D3D12_FILTER_MIN_MAG_MIP_POINT,
        AddressU: D3D12_TEXTURE_ADDRESS_MODE_CLAMP,
        AddressV: D3D12_TEXTURE_ADDRESS_MODE_CLAMP,
        AddressW: D3D12_TEXTURE_ADDRESS_MODE_CLAMP,
        MaxAnisotropy: 1,
        MaxLOD: f32::MAX,
        ..Default::default()
    }];
    let _one = ONE_AT_A_TIME.lock().unwrap_or_else(|e| e.into_inner());
    let adapters = Host::adapters();
    assert!(
        !adapters.is_empty(),
        "no SM 6.6 hardware or WARP device available"
    );
    let views = [srv, View::RawUav(uav_dwords)];
    for (label, host) in &adapters {
        eprintln!("cube test adapter: {label}");
        let (root, pso) = host
            .compute_pipeline(shader)
            .unwrap_or_else(|e| panic!("{label}: compute_pipeline failed: {e}"));
        for (i, &inputs) in cases.iter().enumerate() {
            let mut ud = vec![0u32; shader.user_data_dwords().max(1) as usize];
            for u in &shader.user_data {
                ud[u.index as usize] = match (u.root, u.slot) {
                    (RootType::Constants, 0) => inputs[u.dword as usize],
                    (RootType::Table, 1 | 2) => 0,
                    other => panic!("unexpected user data {other:?}"),
                };
            }
            let out = host
                .dispatch(
                    &root,
                    &pso,
                    &Dispatch {
                        roots: vec![(0, Root::Constants(ud))],
                        views: &views,
                        samplers: &samplers,
                        groups: [1, 1, 1],
                    },
                )
                .unwrap_or_else(|e| panic!("{label}: dispatch failed: {e}"));
            check(label, i, inputs, &out[0]);
        }
    }
}

#[test]
fn cube_array_samples_layers_faces_and_clamped_ids() {
    // 2-cube array (12 faces, 4x4 per face): each texel encodes layer, face
    // and (x, y) in RGBA so both layer/face selection and s/t direction
    // reconstruction are checked on every face of every layer.
    let mut texels = Vec::with_capacity(12 * 16);
    for layer in 0..2u32 {
        for face in 0..6u32 {
            for y in 0..4u32 {
                for x in 0..4u32 {
                    texels.push([
                        (10 * layer + face) as f32,
                        x as f32,
                        y as f32,
                        (100 + layer) as f32,
                    ]);
                }
            }
        }
    }

    // v1 = s, v2 = t, v3 = layer, v4 = face:
    // v_fma_f32 v3, v3, 8.0, v4  (VOP3 0x14b, src1 = literal 0x4100_0000)
    // image_sample_lz v[0:3], v[1:3], s[8:15], s[16:19] dmask:0xf dim:CUBE
    let [fma_w0, fma_w1] = vop3(0x14b, 3, vgpr(3), 255, vgpr(4));
    let [img_w0, img_w1] = mimg_cube(0x27, 0, 1, 8, 16, 0xF, false);
    let shader = build_cube_cs(&[fma_w0, fma_w1, 0x4100_0000, img_w0, img_w1], 4);

    // Centre of texel (x=2, y=1) in [1, 2] face space: 1.0 + (idx + 0.5) / 4.0.
    let (s, t) = (1.625f32.to_bits(), 1.375f32.to_bits());
    let mut cases = Vec::new();
    let mut expected = Vec::new();
    for layer in 0..2u32 {
        // Faces 0..=5 plus out-of-range 6 and 7, which clamp to face 5 (-Z)
        // of the same layer on RDNA2.
        for face in 0..8u32 {
            cases.push([s, t, (layer as f32).to_bits(), (face as f32).to_bits()]);
            expected.push((layer, face.min(5)));
        }
    }
    run_cube_cases(
        &shader,
        View::CubeArray(4, 2, &texels),
        4,
        &cases,
        |backend, i, inp, got| {
            let (layer, clamped_face) = expected[i];
            let want_r = (10 * layer + clamped_face) as f32;
            let got_rgba = [got[0], got[1], got[2], got[3]].map(f32::from_bits);
            assert_eq!(
                got_rgba,
                [want_r, 2.0, 1.0, (100 + layer) as f32],
                "{backend}: layer={} face={} got {got_rgba:?}",
                f32::from_bits(inp[2]),
                f32::from_bits(inp[3]),
            );
        },
    );

    // Issue #720's literal-operand reproducer: v3 = 10.0 (layer 1, face 2 = +Y)
    // with no v_fma in the shader still infers TextureCubeArray and reads 12.0.
    let const_face = build_cube_cs(
        &[
            0x7E02_02FF,
            0x3FC0_0000, // v_mov_b32 v1, 1.5
            0x7E04_02FF,
            0x3FC0_0000, // v_mov_b32 v2, 1.5
            0x7E06_02FF,
            0x4120_0000, // v_mov_b32 v3, 10.0
            mimg_cube(0x27, 0, 1, 8, 16, 0x1, false)[0],
            mimg_cube(0x27, 0, 1, 8, 16, 0x1, false)[1],
        ],
        1,
    );
    run_cube_cases(
        &const_face,
        View::CubeArray(4, 2, &texels),
        1,
        &[[0; 4]],
        |backend, _, _, got| {
            assert_eq!(
                f32::from_bits(got[0]),
                12.0,
                "{backend}: constant face 10.0"
            );
        },
    );
}

#[test]
fn cube_samples_bare_faces_on_plain_cube_view() {
    // Plain 1-cube view (`D3D12_SRV_DIMENSION_TEXTURECUBE`): a bare face
    // operand stays `TextureCube` (kind 5) without regressing single cubes.
    let mut texels = Vec::with_capacity(6 * 16);
    for face in 0..6u32 {
        for _ in 0..16 {
            texels.push([face as f32, 0.0, 0.0, 1.0]);
        }
    }
    let [img_w0, img_w1] = mimg_cube(0x27, 0, 1, 8, 16, 0x1, false);
    let shader = build_cube_cs(&[img_w0, img_w1], 1);
    let mid = 1.5f32.to_bits();
    let cases: Vec<[u32; 4]> = (0..8u32)
        .map(|f| [mid, mid, (f as f32).to_bits(), 0])
        .collect();
    run_cube_cases(
        &shader,
        View::Cube(4, &texels),
        1,
        &cases,
        |backend, f, _, got| {
            let want = f.min(5) as f32;
            assert_eq!(
                f32::from_bits(got[0]),
                want,
                "{backend}: plain cube face {f}"
            );
        },
    );
}

#[test]
fn cube_array_gather_and_resinfo_match_rdna2() {
    // 3-cube array (18 faces, 4x4 per face): texel = 100 * layer + 10 * face + y * 4 + x.
    let mut texels = Vec::with_capacity(18 * 16);
    for layer in 0..3u32 {
        for face in 0..6u32 {
            for y in 0..4u32 {
                for x in 0..4u32 {
                    texels.push([(100 * layer + 10 * face + y * 4 + x) as f32; 4]);
                }
            }
        }
    }

    // `image_gather4_lz` (0x47, dmask 1) with `v_fma_f32 v3, v3, 8.0, v4`.
    let [fma_w0, fma_w1] = vop3(0x14b, 3, vgpr(3), 255, vgpr(4));
    let [gat_w0, gat_w1] = mimg_cube(0x47, 0, 1, 8, 16, 0x1, false);
    let gather_cs = build_cube_cs(&[fma_w0, fma_w1, 0x4100_0000, gat_w0, gat_w1], 4);
    // At s = t = 1.5 (centre of the 4x4 face, footprint across texels (1,1)..(2,2)),
    // all four gathered texels come from the requested (layer, face) and sum to
    // 4 * (100 * layer + 10 * face) + (5 + 6 + 9 + 10) = 400 * layer + 40 * face + 30.
    let mid = 1.5f32.to_bits();
    let gather_cases = [
        ([mid, mid, 0.0f32.to_bits(), 0.0f32.to_bits()], 0, 0),
        ([mid, mid, 1.0f32.to_bits(), 2.0f32.to_bits()], 1, 2),
        ([mid, mid, 2.0f32.to_bits(), 5.0f32.to_bits()], 2, 5),
    ];
    let g_in: Vec<[u32; 4]> = gather_cases.iter().map(|c| c.0).collect();
    run_cube_cases(
        &gather_cs,
        View::CubeArray(4, 3, &texels),
        4,
        &g_in,
        |backend, i, _, got| {
            let (_, layer, face) = gather_cases[i];
            let vals = [got[0], got[1], got[2], got[3]].map(f32::from_bits);
            let sum: f32 = vals.iter().sum();
            let want_sum = (400 * layer + 40 * face + 30) as f32;
            assert_eq!(
                sum, want_sum,
                "{backend}: gather4 layer={layer} face={face} got {vals:?}"
            );
        },
    );

    // Standalone `image_get_resinfo v[0:3], v1, s[8:15] dmask:0xf dim:CUBE`:
    // RDNA2 returns [width, height, 6 * cubes, levels] = [4, 4, 18, 1] for 3 cubes,
    // and [4, 4, 12, 1] for 2 cubes.
    let [res_w0, res_w1] = mimg_cube(0x0e, 0, 1, 8, 0, 0xF, false);
    let resinfo_cs = build_cube_cs(&[res_w0, res_w1], 4);
    run_cube_cases(
        &resinfo_cs,
        View::CubeArray(4, 3, &texels),
        4,
        &[[0; 4]],
        |backend, _, _, got| {
            assert_eq!(&got[..4], &[4, 4, 18, 1], "{backend}: 3-cube resinfo");
        },
    );
    run_cube_cases(
        &resinfo_cs,
        View::CubeArray(4, 2, &texels[..12 * 16]),
        4,
        &[[0; 4]],
        |backend, _, _, got| {
            assert_eq!(&got[..4], &[4, 4, 12, 1], "{backend}: 2-cube resinfo");
        },
    );
}

#[test]
fn glc_heap_uav_loads_stores_and_atomics_execute_on_gpu() {
    let _one = ONE_AT_A_TIME.lock().unwrap_or_else(|e| e.into_inner());
    // Compute shader with `RootType::ResourceHeap` (`s2`) and 2 root constants
    // (`s0..s1`), 2 threads per group (`v0` = thread id in group), dispatched
    // across 2 groups (`s3` = group id X):
    // - Heap index 0 (`0x0`): raw UAV buffer (16 dwords)
    // - Heap index 1 (`0x20`): 2x2 RGBA32F texture UAV
    // - Heap index 2 (`0x40`): 2x1 R32_UINT texture UAV
    // Exercises heap UAV inference (#724) and `IsGloballyCoherent` `0x4000`
    // (#725) across `buffer_store_dword glc`, `s_barrier`, reloaded descriptor
    // `s_buffer_load_dword glc`, `buffer_load_dword` (plain and `glc`),
    // `buffer_atomic_add glc`, `image_store glc`, `image_load glc`, and
    // `image_atomic_add glc`.
    let lshl_add = vop3(0x346, 1, sgpr(3), 0x81, vgpr(0)); // v_lshl_add_u32 v1, s3, 1, v0
    let lshl_byte = vop3(0x11A, 2, 0x82, vgpr(1), 0); // v_lshlrev_b32 v2, 2, v1
    let add_s0_v1 = vop3(0x125, 3, sgpr(0), vgpr(1), 0); // v_add_nc_u32 v3, s0, v1
    let add_v4_v5 = vop3(0x125, 4, vgpr(4), vgpr(5), 0); // v_add_nc_u32 v4, v4, v5
    let add_s6_v4 = vop3(0x125, 4, sgpr(6), vgpr(4), 0); // v_add_nc_u32 v4, s6, v4
    let words = [
        s_mov_b32(4, 2), // s4 = resource heap base (s2)
        s_mov_b32(5, 0x80),
        0xF408_0202,
        0xFA00_0000, // s_load_dwordx4 s[8:11], s[4:5], 0x0   (raw UAV @ 0)
        0xF40C_0302,
        0xFA00_0020, // s_load_dwordx8 s[12:19], s[4:5], 0x20 (RGBA32F UAV @ 1)
        0xF40C_0502,
        0xFA00_0040, // s_load_dwordx8 s[20:27], s[4:5], 0x40 (R32_UINT UAV @ 2)
        // v1 = flat index = group_id_x * 2 + thread_id_x; v2 = v1 * 4 (byte offset).
        lshl_add[0],
        lshl_add[1],
        lshl_byte[0],
        lshl_byte[1],
        // v3 = s0 + v1; store with `glc` at byte offset v2.
        add_s0_v1[0],
        add_s0_v1[1],
        0xE070_5000,
        0x8002_0302, // buffer_store_dword v3, v2, s[8:11], 0 offen glc
        0xBF8A_0000, // s_barrier
        // Reload raw UAV @ 0 into s[28:31], read dword 0 via scalar `glc` and
        // lane's own slot via vector `glc` + plain load, and write sum to slot 4 + flat.
        0xF408_0702,
        0xFA00_0000, // s_load_dwordx4 s[28:31], s[4:5], 0x0
        0xF421_018E,
        0xFA00_0000, // s_buffer_load_dword s6, s[28:31], 0x0 glc
        0xE030_5000,
        0x8007_0402, // buffer_load_dword v4, v2, s[28:31], 0 offen glc
        0xE030_1000,
        0x8007_0502, // buffer_load_dword v5, v2, s[28:31], 0 offen
        add_v4_v5[0],
        add_v4_v5[1],
        add_s6_v4[0],
        add_s6_v4[1],
        0xE070_5010,
        0x8007_0402, // buffer_store_dword v4, v2, s[28:31], 0 offen offset:16 glc
        // Atomic add s1 into dword 8 of raw UAV with `glc`.
        vop1(0x001, 6, sgpr(1)), // v_mov_b32 v6, s1
        vop1(0x001, 7, 0x80),    // v_mov_b32 v7, 0
        0xE0C8_5020,
        0x8007_0607, // buffer_atomic_add v6, v7, s[28:31], 0 offen offset:32 glc
        // Texture UAV (2x2): store (1.0, 2.0, 4.0, 0.5) at (v0, s3) with `glc`,
        // load back with `glc`, and atomic_add 5 at (s3, 0) on R32U texture UAV.
        vop1(0x001, 8, vgpr(0)), // x = thread_id_x (0..1)
        vop1(0x001, 9, sgpr(3)), // y = group_id_x (0..1)
        vop1(0x001, 10, 0xF2),   // 1.0f
        vop1(0x001, 11, 0xF4),   // 2.0f
        vop1(0x001, 12, 0xF6),   // 4.0f
        vop1(0x001, 13, 0xF0),   // 0.5f
        0xF020_2F08,
        0x0003_0A08, // image_store v[10:13], v[8:9], s[12:19] dmask:0xf 2d glc
        0xBF8A_0000, // s_barrier
        0xF000_2F08,
        0x0003_0E08, // image_load v[14:17], v[8:9], s[12:19] dmask:0xf 2d glc
        0xE070_5024,
        0x8007_0E02, // buffer_store_dword v14, v2, s[28:31], 0 offen offset:36 glc
        vop1(0x001, 18, sgpr(3)), // x = group_id_x
        vop1(0x001, 19, 0x80), // y = 0
        vop1(0x001, 20, 0x85), // 5
        0xF044_2108,
        0x0005_1412, // image_atomic_add v20, v[18:19], s[20:27] dmask:0x1 2d glc
        S_ENDPGM,
    ];
    let code: Vec<u8> = words.iter().flat_map(|w| w.to_le_bytes()).collect();
    let native = NativeProgram {
        stage: HwStage::Cs,
        wave64: false,
        vgprs: 24,
        layout: vec![
            RootParam {
                root: RootType::Constants,
                table_offset_dwords: 0,
                sgprs: 2,
                base_register: 0,
                user_data: 0,
                slot: 0,
            },
            RootParam {
                root: RootType::ResourceHeap,
                table_offset_dwords: 0,
                sgprs: 1,
                base_register: 0,
                user_data: 2,
                slot: 1,
            },
        ],
        used: Vec::new(),
        io: StageIo::Cs(ComputeIo {
            // 3 user SGPRs (s0..s2) + TGID_X enabled (bit 7 -> s3).
            rsrc2: (3 << 1) | (1 << 7),
            threads: [2, 1, 1],
        }),
        code,
    };

    let (p, st) = range::build(&native.code, native.wave64, native.vgprs).expect("range::build");
    let st = st.expect("structure");
    let res =
        resource::track(&p, &native.layout, native.stage, Some(&st)).expect("resource::track");
    let io = io::raise(&p, &native).expect("io::raise");
    let lowered = lower::lower(&lower::Parts {
        native: &native,
        p: &p,
        st: &st,
        io: &io,
        res: &res,
        inputs: Vec::new(),
        outputs: Vec::new(),
        prims: Vec::new(),
        mesh_stride: 0,
        target: crate::emit::Target::ANY,
        view_marks: false,
    })
    .expect("lower");

    if let Ok(mut v) = validator::Validator::load(validator::Validator::default_path().as_deref()) {
        let mut copy = lowered.dxil.clone();
        let ours = copy[4..20].to_vec();
        v.validate(&mut copy).expect("DXIL validation failed");
        assert_eq!(copy[4..20], ours[..], "container hash mismatch");
    }

    let shader = HostShader {
        stage: HwStage::Cs,
        wave_size: lowered.wave_size,
        dxil: lowered.dxil,
        user_data: lowered.user_data,
        samplers: lowered.samplers,
        static_samplers: Vec::new(),
        root_cbvs: lowered.root_cbvs,
        root_srvs: lowered.root_srvs,
        root_uavs: lowered.root_uavs,
        memory: lowered.memory,
        memory_writes: lowered.memory_writes,
        view_addresses: lowered.view_addresses,
        heap: lowered.heap,
        heap_writes: lowered.heap_writes,
        view_marks: lowered.view_marks,
    };

    let mut ran = 0;
    for (label, warp) in [("hardware", false), ("warp", true)] {
        let Ok(host) = Host::new(warp) else {
            eprintln!("no SM 6.6 {label} device; skipped");
            continue;
        };
        let (root, pso) = host
            .compute_pipeline(&shader)
            .unwrap_or_else(|e| panic!("{label}: compute_pipeline failed: {e}"));
        let inputs = [100u32, 7u32];
        let mut ud = vec![0u32; shader.user_data_dwords().max(1) as usize];
        for u in &shader.user_data {
            ud[u.index as usize] = inputs[u.dword as usize];
        }
        let out = host
            .dispatch(
                &root,
                &pso,
                &Dispatch {
                    roots: vec![
                        (0, Root::Constants(ud)),
                        (
                            1,
                            Root::View {
                                srv: true,
                                data: vec![4, 0, 0],
                            },
                        ),
                    ],
                    views: &[
                        View::RawUav(16),
                        View::TextureUav(2, 2),
                        View::UintTextureUav(2, 1),
                    ],
                    samplers: &[],
                    groups: [2, 1, 1],
                },
            )
            .unwrap_or_else(|e| panic!("{label}: dispatch failed: {e}"));
        // Slots 0..4 hold 100..103.
        assert_eq!(&out[0][0..4], &[100, 101, 102, 103], "{label}: raw stores");
        // Group 0 saw its own s_barrier before reading dword 0 (100):
        // slot 4 = 100 + 100 + 100 = 300, slot 5 = 101 + 101 + 100 = 302.
        assert_eq!(&out[0][4..6], &[300, 302], "{label}: reloaded glc reads");
        // Slot 8 accumulated 4 atomic adds of 7 = 28.
        assert_eq!(out[0][8], 28, "{label}: buffer_atomic_add glc");
        // Slots 9..13 hold 1.0f read back through `image_load glc`.
        assert_eq!(
            &out[0][9..13],
            &[1.0f32.to_bits(); 4],
            "{label}: image_load glc readback"
        );
        // 2x2 RGBA32F texture UAV holds (1.0, 2.0, 4.0, 0.5) at all 4 texels.
        for px in out[1].as_chunks::<4>().0 {
            assert_eq!(
                *px,
                [1.0f32, 2.0, 4.0, 0.5].map(f32::to_bits),
                "{label}: texture store glc"
            );
        }
        // 2x1 R32_UINT texture UAV: 2 threads per group * 5 = 10 at (0,0) and (1,0).
        assert_eq!(&out[2][0..2], &[10, 10], "{label}: image_atomic_add glc");
        ran += 1;
    }
    assert!(ran > 0, "neither hardware nor WARP SM 6.6 device available");
}


#[test]
fn valu16_min3_max3_med3_and_mad32_16_match_rdna2() {
    let vop3_mod =
        |op: u16, opsel: u32, clamp: bool, vdst: u8, s0: u16, s1: u16, s2: u16| -> [u32; 2] {
            let mut inst = vop3(op, vdst, s0, s1, s2);
            inst[0] |= (opsel & 0xF) << 11;
            if clamp {
                inst[0] |= 1 << 15;
            }
            inst
        };

    // Pre-seed v0 with 0xCAFE_BEEF so high/low half preservation on 16-bit writes is checked.
    let seeded = |op: u16, opsel: u32| {
        let mut code = vec![vop1(0x001, 0, 255), 0xCAFE_BEEF];
        code.extend_from_slice(&vop3_mod(op, opsel, false, 0, vgpr(1), vgpr(2), vgpr(3)));
        build_valu_cs(&code)
    };

    // 16-bit signed & unsigned min3/max3/med3 (low dst, plus high dst + op_sel[0..2]).
    let min3_i16 = seeded(0x352, 0);
    let min3_u16 = seeded(0x353, 0);
    let max3_i16 = seeded(0x355, 0);
    let max3_u16 = seeded(0x356, 0);
    let med3_i16 = seeded(0x358, 0);
    let med3_u16 = seeded(0x359, 0);
    // op_sel = 0b1101 => src0=hi, src1=lo, src2=hi, dst=hi (preserves low 0xBEEF).
    let med3_i16_hi = seeded(0x358, 0b1101);

    let cases16: &[([u32; 4], [u32; 6])] = &[
        // (a, b, c) low halves: (-10 = 0xFFF6, 5 = 0x0005, -2 = 0xFFFE)
        // signed:   min=-10 (0xFFF6), max=5 (0x0005), med=-2 (0xFFFE)
        // unsigned: min=5 (0x0005), max=0xFFFE, med=0xFFF6
        (
            [0x0000_FFF6, 0x0000_0005, 0x0000_FFFE, 0],
            [
                0xCAFE_FFF6,
                0xCAFE_0005,
                0xCAFE_0005,
                0xCAFE_FFFE,
                0xCAFE_FFFE,
                0xCAFE_FFF6,
            ],
        ),
        // (0x8000 = -32768 / 32768, 0x7FFF = 32767, 0x0000 = 0)
        (
            [0x1234_8000, 0x5678_7FFF, 0x9ABC_0000, 0],
            [
                0xCAFE_8000,
                0xCAFE_0000,
                0xCAFE_7FFF,
                0xCAFE_8000,
                0xCAFE_0000,
                0xCAFE_7FFF,
            ],
        ),
    ];
    let in16: Vec<[u32; 4]> = cases16.iter().map(|c| c.0).collect();
    run_cases(&min3_i16, &in16, |b, i, _, g| {
        assert_eq!(g, cases16[i].1[0], "{b}: v_min3_i16 case {i}");
    });
    run_cases(&min3_u16, &in16, |b, i, _, g| {
        assert_eq!(g, cases16[i].1[1], "{b}: v_min3_u16 case {i}");
    });
    run_cases(&max3_i16, &in16, |b, i, _, g| {
        assert_eq!(g, cases16[i].1[2], "{b}: v_max3_i16 case {i}");
    });
    run_cases(&max3_u16, &in16, |b, i, _, g| {
        assert_eq!(g, cases16[i].1[3], "{b}: v_max3_u16 case {i}");
    });
    run_cases(&med3_i16, &in16, |b, i, _, g| {
        assert_eq!(g, cases16[i].1[4], "{b}: v_med3_i16 case {i}");
    });
    run_cases(&med3_u16, &in16, |b, i, _, g| {
        assert_eq!(g, cases16[i].1[5], "{b}: v_med3_u16 case {i}");
    });

    // High-half op_sel test: src0.hi = -5 (0xFFFB), src1.lo = 10, src2.hi = -1 (0xFFFF) => med = -1 -> 0xFFFF_BEEF.
    let hi_in = [[0xFFFB_1234, 0x9999_000A, 0xFFFF_5678, 0]];
    run_cases(&med3_i16_hi, &hi_in, |b, _, _, g| {
        assert_eq!(g, 0xFFFF_BEEF, "{b}: v_med3_i16 opsel=0b1101");
    });

    // v_mad_u32_u16 (0x373) & v_mad_i32_i16 (0x375), wrapping + clamped + op_sel.
    let mad_u32 = build_valu_cs(&vop3_mod(0x373, 0, false, 0, vgpr(1), vgpr(2), vgpr(3)));
    let mad_u32_clamp = build_valu_cs(&vop3_mod(0x373, 0, true, 0, vgpr(1), vgpr(2), vgpr(3)));
    // op_sel = 0b0010 => src0=lo, src1=hi
    let mad_u32_hi1 = build_valu_cs(&vop3_mod(
        0x373,
        0b0010,
        false,
        0,
        vgpr(1),
        vgpr(2),
        vgpr(3),
    ));
    let mad_i32 = build_valu_cs(&vop3_mod(0x375, 0, false, 0, vgpr(1), vgpr(2), vgpr(3)));
    let mad_i32_clamp = build_valu_cs(&vop3_mod(0x375, 0, true, 0, vgpr(1), vgpr(2), vgpr(3)));

    let u32_cases: &[([u32; 4], u32, u32)] = &[
        // 300 * 400 + 50 = 120050
        ([0xDEAD_012C, 0xBEEF_0190, 50, 0], 120_050, 120_050),
        // 65535 * 65535 + 0x0002_0000 overflows u32 -> wraps to 1, clamps to u32::MAX
        (
            [0x0000_FFFF, 0x0000_FFFF, 0x0002_0000, 0],
            (65535u32 * 65535).wrapping_add(0x0002_0000),
            u32::MAX,
        ),
    ];
    let u32_in: Vec<[u32; 4]> = u32_cases.iter().map(|c| c.0).collect();
    run_cases(&mad_u32, &u32_in, |b, i, _, g| {
        assert_eq!(g, u32_cases[i].1, "{b}: v_mad_u32_u16 case {i}");
    });
    run_cases(&mad_u32_clamp, &u32_in, |b, i, _, g| {
        assert_eq!(g, u32_cases[i].2, "{b}: v_mad_u32_u16 clamp case {i}");
    });
    // src0.lo = 10, src1.hi = 20, acc = 3 => 203.
    run_cases(
        &mad_u32_hi1,
        &[[0x9999_000A, 0x0014_8888, 3, 0]],
        |b, _, _, g| {
            assert_eq!(g, 203, "{b}: v_mad_u32_u16 opsel=0b0010");
        },
    );

    let i32_cases: &[([u32; 4], u32, u32)] = &[
        // (-100) * 50 + 20 = -4980
        (
            [0x0000_FF9C, 0x0000_0032, 20, 0],
            (-4980i32) as u32,
            (-4980i32) as u32,
        ),
        // (-32768) * (-32768) + i32::MAX => +2^30 + (2^31 - 1), positive overflow -> i32::MAX
        (
            [0x0000_8000, 0x0000_8000, i32::MAX as u32, 0],
            (1i32 << 30).wrapping_add(i32::MAX) as u32,
            i32::MAX as u32,
        ),
        // (-32768) * 32767 + i32::MIN => negative overflow -> i32::MIN
        (
            [0x0000_8000, 0x0000_7FFF, i32::MIN as u32, 0],
            (-32768i32 * 32767).wrapping_add(i32::MIN) as u32,
            i32::MIN as u32,
        ),
    ];
    let i32_in: Vec<[u32; 4]> = i32_cases.iter().map(|c| c.0).collect();
    run_cases(&mad_i32, &i32_in, |b, i, _, g| {
        assert_eq!(g, i32_cases[i].1, "{b}: v_mad_i32_i16 case {i}");
    });
    run_cases(&mad_i32_clamp, &i32_in, |b, i, _, g| {
        assert_eq!(g, i32_cases[i].2, "{b}: v_mad_i32_i16 clamp case {i}");
    });

    // Inline float constant 242 (1.0f = 0x3F80_0000) on 16-bit integer ops:
    // op_sel=0 selects low half 0x0000 (not f16 0x3C00); op_sel=1 selects 0x3F80.
    let mad_u32_fconst_lo = build_valu_cs(&vop3_mod(0x373, 0, false, 0, 242, vgpr(2), vgpr(3)));
    let mad_u32_fconst_hi =
        build_valu_cs(&vop3_mod(0x373, 0b0001, false, 0, 242, vgpr(2), vgpr(3)));
    run_cases(&mad_u32_fconst_lo, &[[0, 7, 11, 0]], |b, _, _, g| {
        assert_eq!(g, 11, "{b}: v_mad_u32_u16 inline 1.0 op_sel=0");
    });
    run_cases(&mad_u32_fconst_hi, &[[0, 2, 11, 0]], |b, _, _, g| {
        assert_eq!(g, 0x3F80 * 2 + 11, "{b}: v_mad_u32_u16 inline 1.0 op_sel=1");
    });
    let max3_u16_fconst_lo = build_valu_cs(&vop3_mod(0x356, 0, false, 0, 242, vgpr(2), vgpr(3)));
    let max3_u16_fconst_hi =
        build_valu_cs(&vop3_mod(0x356, 0b0001, false, 0, 242, vgpr(2), vgpr(3)));
    run_cases(&max3_u16_fconst_lo, &[[0, 10, 20, 0]], |b, _, _, g| {
        assert_eq!(g & 0xFFFF, 20, "{b}: v_max3_u16 inline 1.0 op_sel=0");
    });
    run_cases(&max3_u16_fconst_hi, &[[0, 10, 20, 0]], |b, _, _, g| {
        assert_eq!(g & 0xFFFF, 0x3F80, "{b}: v_max3_u16 inline 1.0 op_sel=1");
    });
}

struct AdapterHost {
    name: String,
    #[allow(dead_code)]
    warp: bool,
    wave_min: u32,
    wave_max: u32,
    host: Host,
}

fn all_wave_hosts() -> Vec<AdapterHost> {
    let mut out = Vec::new();
    // SAFETY: standard DXGI adapter enumeration and D3D12 feature queries.
    unsafe {
        let Ok(factory) = CreateDXGIFactory2::<IDXGIFactory4>(DXGI_CREATE_FACTORY_FLAGS(0)) else {
            return out;
        };
        let mut idx = 0u32;
        while let Ok(adapter) = factory.EnumAdapters1(idx) {
            idx += 1;
            let Ok(desc) = adapter.GetDesc1() else {
                continue;
            };
            if (desc.Flags & DXGI_ADAPTER_FLAG_SOFTWARE.0 as u32) != 0 {
                continue;
            }
            let len = desc
                .Description
                .iter()
                .position(|&c| c == 0)
                .unwrap_or(desc.Description.len());
            let name = String::from_utf16_lossy(&desc.Description[..len]);
            if let Some(h) = open_wave_adapter(&adapter, name, false) {
                out.push(h);
            }
        }
        if let Ok(warp) = factory.EnumWarpAdapter::<IDXGIAdapter1>()
            && let Some(h) = open_wave_adapter(&warp, "D3D12 WARP".into(), true)
        {
            out.push(h);
        }
    }
    out
}

unsafe fn open_wave_adapter(
    adapter: &IDXGIAdapter1,
    name: String,
    warp: bool,
) -> Option<AdapterHost> {
    // SAFETY: valid adapter and feature structs.
    unsafe {
        let mut device: Option<ID3D12Device> = None;
        D3D12CreateDevice(adapter, D3D_FEATURE_LEVEL_12_0, &mut device).ok()?;
        let device = device?;
        let mut sm = D3D12_FEATURE_DATA_SHADER_MODEL {
            HighestShaderModel: D3D_SHADER_MODEL_6_6,
        };
        device
            .CheckFeatureSupport(
                D3D12_FEATURE_SHADER_MODEL,
                (&raw mut sm).cast(),
                size_of_val(&sm) as u32,
            )
            .ok()?;
        if sm.HighestShaderModel.0 < D3D_SHADER_MODEL_6_6.0 {
            return None;
        }
        let mut opts1 = D3D12_FEATURE_DATA_D3D12_OPTIONS1::default();
        device
            .CheckFeatureSupport(
                D3D12_FEATURE_D3D12_OPTIONS1,
                (&raw mut opts1).cast(),
                size_of_val(&opts1) as u32,
            )
            .ok()?;
        if !opts1.WaveOps.as_bool() {
            return None;
        }
        Some(AdapterHost {
            name,
            warp,
            wave_min: opts1.WaveLaneCountMin,
            wave_max: opts1.WaveLaneCountMax,
            host: Host::from_device(device),
        })
    }
}

/// Builds a multi-lane synthetic compute shader (`wave64`: 64 threads, else 32)
/// with 4 root constants in `s0..s3` and a raw UAV at heap index 0 (`s[8:11]`).
fn build_wave_cs(wave64: bool, body: &[u32]) -> HostShader {
    let wave = if wave64 { 64u32 } else { 32 };
    let mut words = Vec::new();
    words.push(s_mov_b32(5, 0x80)); // s_mov_b32 s5, 0 (s4 = table offset 0)
    words.extend_from_slice(&s_load_dwordx4(8, 4));
    words.extend_from_slice(body);
    words.push(S_ENDPGM);

    let code: Vec<u8> = words.iter().flat_map(|w| w.to_le_bytes()).collect();
    let native = NativeProgram {
        stage: HwStage::Cs,
        wave64,
        vgprs: 12,
        layout: vec![
            RootParam {
                root: RootType::Constants,
                table_offset_dwords: 0,
                sgprs: 4,
                base_register: 0,
                user_data: 0,
                slot: 0,
            },
            RootParam {
                root: RootType::Table,
                table_offset_dwords: 0,
                sgprs: 1,
                base_register: 0,
                user_data: 4,
                slot: 1,
            },
        ],
        used: vec![UsedResource {
            kind: ResourceType::Uav,
            root: RootType::Table,
            table_offset_dwords: 0,
            lower_bound: 0,
            stride: 0,
            dwords: 4,
            user_data: 4,
            slot: 1,
        }],
        io: StageIo::Cs(ComputeIo {
            rsrc2: 5 << 1, // s0..s4
            threads: [wave, 1, 1],
        }),
        code,
    };

    let p = ir::build(&native.code, wave64, native.vgprs).expect("ir::build");
    let st = structure::structure(&p).expect("structure");
    let res = resource::track(&p, &native.layout, HwStage::Cs, Some(&st)).expect("resource::track");
    let io = io::raise(&p, &native).expect("io::raise");
    let lowered = lower::lower(&lower::Parts {
        native: &native,
        p: &p,
        st: &st,
        io: &io,
        res: &res,
        inputs: Vec::new(),
        outputs: Vec::new(),
        prims: Vec::new(),
        mesh_stride: 0,
        target: crate::emit::Target::ANY,
        view_marks: false,
    })
    .expect("lower");

    let shader = HostShader {
        stage: HwStage::Cs,
        wave_size: lowered.wave_size,
        dxil: lowered.dxil,
        user_data: lowered.user_data,
        samplers: lowered.samplers,
        static_samplers: Vec::new(),
        root_cbvs: lowered.root_cbvs,
        root_srvs: lowered.root_srvs,
        root_uavs: lowered.root_uavs,
        memory: lowered.memory,
        memory_writes: lowered.memory_writes,
        view_addresses: lowered.view_addresses,
        heap: lowered.heap,
        heap_writes: lowered.heap_writes,
        view_marks: lowered.view_marks,
    };

    if let Ok(mut v) = validator::Validator::load(validator::Validator::default_path().as_deref()) {
        let mut copy = shader.dxil.clone();
        let ours = copy[4..20].to_vec();
        v.validate(&mut copy).expect("DXIL validation failed");
        assert_eq!(copy[4..20], ours[..], "container hash mismatch");
    }

    shader
}

const fn vop2_e32(op: u8, vdst: u8, src0: u16, vsrc1: u8) -> u32 {
    (op as u32 & 0x3F) << 25 | (vdst as u32) << 17 | (vsrc1 as u32) << 9 | (src0 as u32 & 0x1FF)
}

const fn buffer_store_dword_offen_at(vdata: u8, vaddr: u8, srsrc: u8, byte_off: u16) -> [u32; 2] {
    [
        0x38 << 26 | 0x1C << 18 | 1 << 12 | (byte_off as u32 & 0xFFF),
        0x80 << 24 | ((srsrc as u32 >> 2) & 0x1F) << 16 | (vdata as u32) << 8 | vaddr as u32,
    ]
}

#[test]
fn v_writelane_b32_ignores_exec_and_masks_lane() {
    for wave64 in [false, true] {
        let wave = if wave64 { 64u32 } else { 32 };
        let half = wave / 2;
        let mut b = Vec::new();
        // v1 = 0x1000 | v0; v2 = 0x2000 | v0; v3 = 0x3000 | v0
        for (dst, tag) in [(1u8, 0x1000u32), (2, 0x2000), (3, 0x3000)] {
            b.push(vop2_e32(0x1C, dst, 0xFF, 0)); // v_or_b32 vdst, literal, v0
            b.push(tag);
        }
        // Narrow EXEC to upper half (`lanes half..wave`), leaving `0..half` inactive!
        // v_cmpx_ge_u32_e32 exec, half, v0 -> wait: VOPC is src0 >= vsrc1!
        // Use VOP3 `v_cmpx_ge_u32_e64` (opcode 0x0D6): src0 = v0, src1 = half -> v0 >= half!
        b.extend_from_slice(&vop3(0x0D6, 126, vgpr(0), 128 + half as u16, 0));

        // 1. Write 0xCAFEBABE to inactive lane 0.
        b.push(0x35 << 26 | 0x361 << 16 | 1); // v_writelane_b32 v1, literal, 0
        b.push(0xFF | (128u32) << 9);
        b.push(0xCAFE_BABE);

        // 2. Control VALU: `v_add_nc_u32 v2, 0x100, v2` obeys EXEC (only `half..wave` change).
        b.push(vop2_e32(0x25, 2, 0xFF, 2));
        b.push(0x0000_0100);

        // 3. Dynamic SGPR selector `s1`: `v_writelane_b32 v1, s0, s1` (wraps modulo wave).
        b.extend_from_slice(&vop3(0x361, 1, sgpr(0), sgpr(1), 0));

        // 4. Explicit `exec_lo` read spilled into inactive lane 0: `v_writelane_b32 v3, exec_lo, 0`.
        b.extend_from_slice(&vop3(0x361, 3, sgpr(126), 128, 0));

        // 5. EXEC == 0 in straight-line code: `v_writelane_b32 v1, s2, 5` still writes lane 5!
        if wave64 {
            b.push(0xBE80_0000 | 126 << 16 | 0x04 << 8 | 128); // s_mov_b64 exec, 0
        } else {
            b.push(s_mov_b32(126, 128)); // s_mov_b32 exec_lo, 0
        }
        b.extend_from_slice(&vop3(0x361, 1, sgpr(2), 128 + 5, 0));

        // Restore full EXEC, read dynamic wrapped lane `s3` via `v_readlane_b32 s12, v1, s3`.
        if wave64 {
            b.push(0xBE80_0000 | 126 << 16 | 0x04 << 8 | 193); // s_mov_b64 exec, -1
        } else {
            b.push(s_mov_b32(126, 193)); // s_mov_b32 exec_lo, -1
        }
        b.extend_from_slice(&vop3(0x360, 12, vgpr(1), sgpr(3), 0));
        b.push(vop1(0x001, 4, sgpr(12))); // v4 = s12

        // Store 4 dwords per lane at byte offset `v0 * 16`: v1, v2, v3, v4.
        b.push(vop2_e32(0x1A, 5, 128 + 4, 0)); // v_lshlrev_b32 v5, 4, v0
        b.extend_from_slice(&buffer_store_dword_offen_at(1, 5, 8, 0));
        b.extend_from_slice(&buffer_store_dword_offen_at(2, 5, 8, 4));
        b.extend_from_slice(&buffer_store_dword_offen_at(3, 5, 8, 8));
        b.extend_from_slice(&buffer_store_dword_offen_at(4, 5, 8, 12));

        let shader = build_wave_cs(wave64, &b);
        let dyn_cases: &[[u32; 4]] = if wave64 {
            &[
                // [s0_val, s1_write_sel, s2_lane5_val, s3_read_sel]
                [0x1111_2222, 65, 0x5555_0005, 64], // 65 -> 1, 64 -> 0 (0xCAFEBABE)
                [0x3333_4444, u32::MAX, 0x5555_0005, 129], // -1 -> 63, 129 -> 1
                [0x7777_8888, 32, 0x5555_0005, u32::MAX], // 32 -> 32 (not 0!), -1 -> 63
                [0x9999_AAAA, 0x2000_0000, 0x5555_0005, 69], // 0x2000_0000 -> 0 (overwrites 0xCAFEBABE), 69 -> 5
            ]
        } else {
            &[
                [0x1111_2222, 33, 0x5555_0005, 32], // 33 -> 1, 32 -> 0 (0xCAFEBABE)
                [0x3333_4444, 65, 0x5555_0005, u32::MAX], // 65 -> 1, -1 -> 31
                [0x7777_8888, u32::MAX, 0x5555_0005, 37], // -1 -> 31, 37 -> 5
                [0x9999_AAAA, 32, 0x5555_0005, 64], // 32 -> 0 (overwrites 0xCAFEBABE), 64 -> 0
            ]
        };

        let _one = ONE_AT_A_TIME.lock().unwrap_or_else(|e| e.into_inner());
        let hosts = all_wave_hosts();
        let mut ran = 0usize;
        for h in &hosts {
            if wave < h.wave_min || wave > h.wave_max {
                eprintln!(
                    "skip Wave{wave} writelane on {} (wave range {}..={})",
                    h.name, h.wave_min, h.wave_max
                );
                continue;
            }
            let (root, pso) = h
                .host
                .compute_pipeline(&shader)
                .unwrap_or_else(|e| panic!("{}: compute_pipeline failed: {e}", h.name));
            for &[s0, s1, s2, s3] in dyn_cases {
                let mut ud = vec![0u32; shader.user_data_dwords().max(1) as usize];
                for u in &shader.user_data {
                    ud[u.index as usize] = match (u.root, u.slot) {
                        (RootType::Constants, 0) => [s0, s1, s2, s3][u.dword as usize],
                        (RootType::Table, 1) => 0,
                        other => panic!("unexpected user data {other:?}"),
                    };
                }
                let out = h
                    .host
                    .dispatch(
                        &root,
                        &pso,
                        &Dispatch {
                            roots: vec![(0, Root::Constants(ud))],
                            views: &[View::RawUav(wave * 4)],
                            samplers: &[],
                            groups: [1, 1, 1],
                        },
                    )
                    .unwrap_or_else(|e| panic!("{}: dispatch failed: {e}", h.name));
                let dwords = &out[0];
                let mut expected_v1: Vec<u32> = (0..wave).map(|l| 0x1000 | l).collect();
                expected_v1[0] = 0xCAFE_BABE;
                expected_v1[(s1 & (wave - 1)) as usize] = s0;
                expected_v1[5] = s2;
                let expected_read = expected_v1[(s3 & (wave - 1)) as usize];
                let expected_exec_lo = if wave64 { 0u32 } else { 0xFFFF_0000 };

                for l in 0..wave as usize {
                    let got_v1 = dwords[l * 4];
                    let got_v2 = dwords[l * 4 + 1];
                    let got_v3 = dwords[l * 4 + 2];
                    let got_rl = dwords[l * 4 + 3];
                    let want_v2 = if (l as u32) >= half {
                        0x2100 | l as u32
                    } else {
                        0x2000 | l as u32
                    };
                    let want_v3 = if l == 0 {
                        expected_exec_lo
                    } else {
                        0x3000 | l as u32
                    };
                    assert_eq!(
                        got_v1, expected_v1[l],
                        "{} Wave{wave} (s1={s1:#x}) lane {l} v1",
                        h.name
                    );
                    assert_eq!(got_v2, want_v2, "{} Wave{wave} lane {l} control v2", h.name);
                    assert_eq!(
                        got_v3, want_v3,
                        "{} Wave{wave} lane {l} exec_lo spill",
                        h.name
                    );
                    assert_eq!(
                        got_rl, expected_read,
                        "{} Wave{wave} (s3={s3:#x}) lane {l} readlane",
                        h.name
                    );
                }
            }
            eprintln!(
                "ok: {} [writelane/readlane Wave{wave}, wave={}..={}]",
                h.name, h.wave_min, h.wave_max
            );
            ran += 1;
        }
        if !wave64 {
            assert!(ran > 0, "no SM 6.6 adapter supported Wave32");
        }
    }
}

#[test]
fn v_readlane_and_writelane_wrap_constant_lane_selector() {
    for wave64 in [false, true] {
        let wave = if wave64 { 64u32 } else { 32 };
        let mut b = Vec::new();
        // v1 = 0x4000 | v0
        b.push(vop2_e32(0x1C, 1, 0xFF, 0));
        b.push(0x4000);
        // Inline `64` (`128 + 64 = 192`) wraps to lane 0 in both Wave32 and Wave64!
        b.extend_from_slice(&vop3(0x361, 1, sgpr(0), 192, 0));
        // Inline `-1` (`193`) wraps to `wave - 1` (`31` in Wave32, `63` in Wave64).
        b.extend_from_slice(&vop3(0x361, 1, sgpr(1), 193, 0));
        // Literal `65` wraps to lane `1` in both Wave32 and Wave64.
        b.push(0x35 << 26 | 0x361 << 16 | 1);
        b.push(sgpr(2) as u32 | 0xFF << 9);
        b.push(65);
        // Constant `v_readlane_b32`: inline `64` -> lane 0, inline `-1` -> lane `wave - 1`.
        b.extend_from_slice(&vop3(0x360, 12, vgpr(1), 192, 0));
        b.extend_from_slice(&vop3(0x360, 13, vgpr(1), 193, 0));
        b.push(vop1(0x001, 2, sgpr(12)));
        b.push(vop1(0x001, 3, sgpr(13)));

        b.push(vop2_e32(0x1A, 5, 128 + 4, 0)); // v5 = v0 * 16
        b.extend_from_slice(&buffer_store_dword_offen_at(1, 5, 8, 0));
        b.extend_from_slice(&buffer_store_dword_offen_at(2, 5, 8, 4));
        b.extend_from_slice(&buffer_store_dword_offen_at(3, 5, 8, 8));

        let shader = build_wave_cs(wave64, &b);
        let [s0, s1, s2] = [0x1357_9BDFu32, 0x2468_ACE0, 0xDEAD_C0DE];
        let _one = ONE_AT_A_TIME.lock().unwrap_or_else(|e| e.into_inner());
        for h in all_wave_hosts() {
            if wave < h.wave_min || wave > h.wave_max {
                continue;
            }
            let (root, pso) = h.host.compute_pipeline(&shader).expect("PSO");
            let mut ud = vec![0u32; shader.user_data_dwords().max(1) as usize];
            for u in &shader.user_data {
                ud[u.index as usize] = match (u.root, u.slot) {
                    (RootType::Constants, 0) => [s0, s1, s2, 0][u.dword as usize],
                    (RootType::Table, 1) => 0,
                    other => panic!("unexpected user data {other:?}"),
                };
            }
            let out = h
                .host
                .dispatch(
                    &root,
                    &pso,
                    &Dispatch {
                        roots: vec![(0, Root::Constants(ud))],
                        views: &[View::RawUav(wave * 4)],
                        samplers: &[],
                        groups: [1, 1, 1],
                    },
                )
                .expect("dispatch");
            let dwords = &out[0];
            for l in 0..wave as usize {
                let want_v1 = if l == 0 {
                    s0
                } else if l == 1 {
                    s2
                } else if l as u32 == wave - 1 {
                    s1
                } else {
                    0x4000 | l as u32
                };
                assert_eq!(dwords[l * 4], want_v1, "{} Wave{wave} lane {l} v1", h.name);
                assert_eq!(
                    dwords[l * 4 + 1],
                    s0,
                    "{} Wave{wave} lane {l} rl(64)",
                    h.name
                );
                assert_eq!(
                    dwords[l * 4 + 2],
                    s1,
                    "{} Wave{wave} lane {l} rl(-1)",
                    h.name
                );
            }
            eprintln!("ok: {} [constant wrapped selectors Wave{wave}]", h.name);
        }
    }
}

/// Pixel shader (`MaskData::Host`) reading a dynamic host lane index (`s_ff1` of
/// `EXEC` -> lowest active host lane) via `v_readlane_b32`, executed on every
/// SM 6.6 backend including D3D12 WARP (`wave=4..=4`).
#[test]
fn v_readlane_b32_host_lane_runs_on_all_backends_including_warp() {
    let vs_code: Vec<u8> = [0xF800_08CFu32, 0x0706_0504, 0xF800_0200, 0, S_ENDPGM]
        .iter()
        .flat_map(|w| w.to_le_bytes())
        .collect();
    let vs_native = NativeProgram {
        stage: HwStage::Vs,
        wave64: false,
        vgprs: 8,
        layout: Vec::new(),
        used: Vec::new(),
        io: StageIo::Vs {
            inputs: vec![VsInput {
                usage: Usage::Generic,
                row: 0,
                vgpr: 4,
                mask: 0xF,
            }],
            outputs: Vec::new(),
        },
        code: vs_code,
    };
    let vs_in = vec![SigElement {
        name: "POSITION".into(),
        index: 0,
        kind: 0,
        comp: 9,
        interp: 0,
        row: Some(0),
        col: 0,
        cols: 4,
        usage: 0xF,
        half: false,
    }];
    let vs_out = vec![SigElement {
        name: "SV_Position".into(),
        index: 0,
        kind: 3,
        comp: 9,
        interp: 4,
        row: Some(0),
        col: 0,
        cols: 4,
        usage: 0xF,
        half: false,
    }];
    let vp = ir::build(&vs_native.code, false, 8).unwrap();
    let vst = structure::structure(&vp).unwrap();
    let vres = resource::track(&vp, &[], HwStage::Vs, Some(&vst)).unwrap();
    let vio = io::raise(&vp, &vs_native).unwrap();
    let vlow = lower::lower(&lower::Parts {
        native: &vs_native,
        p: &vp,
        st: &vst,
        io: &vio,
        res: &vres,
        inputs: vs_in,
        outputs: vs_out,
        prims: Vec::new(),
        mesh_stride: 0,
        target: crate::emit::Target::ANY,
        view_marks: false,
    })
    .unwrap();
    let vs = HostShader {
        stage: HwStage::Vs,
        wave_size: vlow.wave_size,
        dxil: vlow.dxil,
        user_data: Vec::new(),
        samplers: Vec::new(),
        static_samplers: Vec::new(),
        root_cbvs: Vec::new(),
        root_srvs: Vec::new(),
        root_uavs: Vec::new(),
        memory: vlow.memory,
        memory_writes: vlow.memory_writes,
        view_addresses: vlow.view_addresses,
        heap: vlow.heap,
        heap_writes: vlow.heap_writes,
        view_marks: vlow.view_marks,
    };

    // PS: capture active mask in s6, find lowest active host lane `s7 = s_ff1_i32_b32(s6)`,
    // compute `v1 = s0 ^ rank`, and broadcast lowest active lane's `v1` (`== s0`) via
    // `v_readlane_b32 s8, v1, s7` (`MaskData::Host` dynamic lane path, unmasked).
    let mut pw = Vec::new();
    pw.push(0xBE80_0000 | 6 << 16 | 0x3D << 8 | 193); // s_or_saveexec_b32 s6, -1
    pw.push(s_mov_b32(126, 6)); // s_mov_b32 exec_lo, s6
    pw.push(0xBE80_0000 | 7 << 16 | 0x13 << 8 | 6); // s_ff1_i32_b32 s7, s6
    pw.extend_from_slice(&vop3(0x365, 0, sgpr(6), 128, 0)); // v_mbcnt_lo_u32_b32 v0, s6, 0 (rank)
    pw.push(vop2_e32(0x1D, 1, sgpr(0), 0)); // v_xor_b32 v1, s0, v0 (rank 0 holds exact s0!)
    pw.extend_from_slice(&vop3(0x360, 8, vgpr(1), sgpr(7), 0)); // v_readlane_b32 s8, v1, s7
    pw.push(vop1(0x001, 2, sgpr(8)));
    pw.push(vop1(0x001, 3, vgpr(1)));
    pw.push(0xF800_180F); // exp mrt0 v2, v3, v0, v2 done vm
    pw.push(2 | 3 << 8 | 2 << 24);
    pw.push(S_ENDPGM);

    let ps_code: Vec<u8> = pw.iter().flat_map(|w| w.to_le_bytes()).collect();
    let ps_native = NativeProgram {
        stage: HwStage::Ps,
        wave64: false,
        vgprs: 8,
        layout: vec![RootParam {
            root: RootType::Constants,
            table_offset_dwords: 0,
            sgprs: 1,
            base_register: 0,
            user_data: 0,
            slot: 0,
        }],
        used: Vec::new(),
        io: StageIo::Ps {
            inputs: Vec::new(),
            input_addr: 0,
            db_shader_control: 0,
            patches: Vec::new(),
        },
        code: ps_code,
    };
    let ps_out = vec![SigElement {
        name: "SV_Target".into(),
        index: 0,
        kind: 16,
        comp: 9,
        interp: 0,
        row: Some(0),
        col: 0,
        cols: 4,
        usage: 0xF,
        half: false,
    }];
    let pp = ir::build(&ps_native.code, false, 8).unwrap();
    let pst = structure::structure(&pp).unwrap();
    let pres = resource::track(&pp, &ps_native.layout, HwStage::Ps, Some(&pst)).unwrap();
    let pio = io::raise(&pp, &ps_native).unwrap();
    let plow = lower::lower(&lower::Parts {
        native: &ps_native,
        p: &pp,
        st: &pst,
        io: &pio,
        res: &pres,
        inputs: Vec::new(),
        outputs: ps_out,
        prims: Vec::new(),
        mesh_stride: 0,
        target: crate::emit::Target::ANY,
        view_marks: false,
    })
    .unwrap();
    let ps = HostShader {
        stage: HwStage::Ps,
        wave_size: plow.wave_size,
        dxil: plow.dxil,
        user_data: plow.user_data,
        samplers: Vec::new(),
        static_samplers: Vec::new(),
        root_cbvs: Vec::new(),
        root_srvs: Vec::new(),
        root_uavs: Vec::new(),
        memory: plow.memory,
        memory_writes: plow.memory_writes,
        view_addresses: plow.view_addresses,
        heap: plow.heap,
        heap_writes: plow.heap_writes,
        view_marks: plow.view_marks,
    };

    if let Ok(mut v) = validator::Validator::load(validator::Validator::default_path().as_deref()) {
        let mut copy = ps.dxil.clone();
        v.validate(&mut copy).expect("PS DXIL validation");
    }

    let tri: [f32; 12] = [
        -1.0, -1.0, 0.0, 1.0, //
        3.0, -1.0, 0.0, 1.0, //
        -1.0, 3.0, 0.0, 1.0,
    ];
    let want = 0x3FA0_0000u32; // 1.25f32 bits (finite float safe through RGBA32F)
    let _one = ONE_AT_A_TIME.lock().unwrap_or_else(|e| e.into_inner());
    let mut ran_any = false;
    for h in all_wave_hosts() {
        let root = h.host.root_signature(&vs, &ps).unwrap();
        let pso = h.host.pipeline_with(&root, &vs.dxil, &ps.dxil).unwrap();
        let pixels = h
            .host
            .draw(
                &root,
                &pso,
                &Draw {
                    roots: vec![
                        (0, Root::Constants(vec![0])),
                        (1, Root::Constants(vec![want])),
                    ],
                    textures: &[],
                    samplers: &[],
                    vertices: &tri,
                    stride: 16,
                    size: (2, 2),
                    mesh_groups: 0,
                },
            )
            .unwrap();
        for (idx, px) in pixels.iter().enumerate() {
            assert_eq!(
                px[0].to_bits(),
                want,
                "{}: pixel {idx} v_readlane_b32 host lane mismatch",
                h.name
            );
        }
        ran_any = true;
        eprintln!("ok: {} [PS host dynamic v_readlane_b32]", h.name);
    }
    assert!(ran_any, "expected at least one adapter to execute PS v_readlane_b32");
}

#[test]
fn v_cmp_class_f32_matches_rdna2() {
    let build_mod = |abs0: bool, neg0: bool| {
        let [mut w0, mut w1] = vop3(0x088, 6, vgpr(1), vgpr(2), 0);
        if abs0 {
            w0 |= 1 << 8;
        }
        if neg0 {
            w1 |= 1 << 29;
        }
        let sel = vop3(0x101, 0, 0x80, 0x81, sgpr(6));
        build_valu_cs(&[w0, w1, sel[0], sel[1]])
    };
    let plain = build_mod(false, false);
    let neg = build_mod(false, true);
    let abs = build_mod(true, false);
    let abs_neg = build_mod(true, true);

    // SDWA: v_cmp_class_f32_sdwa s6, -v1, v2 src0_sel:WORD_0 src1_sel:DWORD.
    // Zero-extending WORD_0 of 0xDEAD_0001 yields +denormal 0x0000_0001;
    // bitwise `neg` must flip bit 31 to -denormal 0x8000_0001 (class 4),
    // not flush under FTZ to -0.0 (class 5).
    let sdwa_w0 = 0x7C00_0000 | 0x088 << 17 | 2 << 9 | 0xF9;
    let sdwa_ext = 1 | 6 << 8 | 1 << 15 | 4 << 16 | 1 << 20 | 6 << 24;
    let sel = vop3(0x101, 0, 0x80, 0x81, sgpr(6));
    let sdwa_word0_neg = build_valu_cs(&[sdwa_w0, sdwa_ext, sel[0], sel[1]]);

    let classes: &[(u32, u32)] = &[
        (0x7F80_0001, 0), // +sNaN (quiet bit 0x0040_0000 clear)
        (0xFFBF_FFFF, 0), // -sNaN
        (0x7FC0_0000, 1), // +qNaN
        (0xFFC0_0001, 1), // -qNaN
        (0xFF80_0000, 2), // -inf
        (0xBF80_0000, 3), // -normal (-1.0)
        (0x8000_0001, 4), // -denormal
        (0x8000_0000, 5), // -0.0
        (0x0000_0000, 6), // +0.0
        (0x0000_0001, 7), // +denormal
        (0x3F80_0000, 8), // +normal (+1.0)
        (0x7F80_0000, 9), // +inf
    ];
    let classify = |bits: u32| -> u32 {
        let exp = (bits >> 23) & 0xFF;
        let mant = bits & 0x007F_FFFF;
        let sign = bits >> 31;
        if exp == 0xFF && mant != 0 {
            u32::from(mant & 0x0040_0000 != 0)
        } else {
            let pos = if exp == 0xFF {
                9
            } else if exp == 0 {
                if mant == 0 { 6 } else { 7 }
            } else {
                8
            };
            if sign != 0 { 11 - pos } else { pos }
        }
    };

    for (label, cs, xform) in [
        ("plain", &plain, (|b: u32| b) as fn(u32) -> u32),
        ("neg", &neg, |b| b ^ 0x8000_0000),
        ("abs", &abs, |b| b & 0x7FFF_FFFF),
        ("abs+neg", &abs_neg, |b| (b & 0x7FFF_FFFF) ^ 0x8000_0000),
    ] {
        let mut cases = Vec::new();
        for &(bits, base_cls) in classes {
            if label == "plain" {
                assert_eq!(classify(bits), base_cls);
            }
            let cls = classify(xform(bits));
            cases.push(([bits, 1u32 << cls, 0, 0], 1u32));
            cases.push(([bits, 0x3FF & !(1u32 << cls), 0, 0], 0u32));
        }
        let inputs: Vec<[u32; 4]> = cases.iter().map(|c| c.0).collect();
        run_cases(cs, &inputs, |backend, i, inp, got| {
            let want = cases[i].1;
            assert_eq!(
                got, want,
                "{backend}: v_cmp_class_f32[{label}]({:#010x}, {:#05x}) = {got}, want {want}",
                inp[0], inp[1],
            );
        });
    }

    let sdwa_cases: &[([u32; 4], u32)] = &[
        ([0xDEAD_0001, 1 << 4, 0, 0], 1), // -denormal
        ([0xDEAD_0001, 1 << 5, 0, 0], 0), // must not flush to -0.0
        ([0xBEEF_0000, 1 << 5, 0, 0], 1), // -0.0
    ];
    let sdwa_in: Vec<[u32; 4]> = sdwa_cases.iter().map(|c| c.0).collect();
    run_cases(&sdwa_word0_neg, &sdwa_in, |backend, i, inp, got| {
        let want = sdwa_cases[i].1;
        assert_eq!(
            got, want,
            "{backend}: v_cmp_class_f32_sdwa(-word0({:#010x}), {:#05x}) = {got}, want {want}",
            inp[0], inp[1],
        );
    });
}

#[test]
fn duplicate_branch_target_critical_edge_produces_valid_dxil() {
    // Block 0 sets s6 = 7 and skips block 1 when s0 != 0. Block 1 sets
    // s6 = 11 and ends with a conditional branch whose taken and fallthrough
    // targets are both the join block.
    let shader = build_valu_cs(&[
        s_mov_b32(6, 0x80 + 7),
        0xBF07_8000, // s_cmp_lg_u32 s0, 0
        0xBF85_0002, // s_cbranch_scc1 -> join
        s_mov_b32(6, 0x80 + 11),
        0xBF84_0000,             // s_cbranch_scc0 +0 -> join (taken == not_taken)
        vop1(0x001, 0, sgpr(6)), // v_mov_b32 v0, s6
    ]);
    let cases = [([0, 0, 0, 0], 11u32), ([1, 0, 0, 0], 7u32)];
    let inputs: Vec<[u32; 4]> = cases.iter().map(|c| c.0).collect();
    run_cases(&shader, &inputs, |backend, i, _, got| {
        assert_eq!(got, cases[i].1, "{backend}: case {i}");
    });
}

#[test]
fn v_swaprel_b32_swaps_indexed_vgprs_and_respects_exec() {
    // s0..s2 hold initial (v1, v2, v3) via build_valu_cs; s3 holds M0 (low byte = src0
    // offset, byte 2 = dst offset); v_swaprel_b32 v1, v1 swaps v[1 + M0[23:16]] and
    // v[1 + M0[7:0]] and then v0 packs the three resulting low bytes (v1 | v2<<8 | v3<<16)
    // unless s3 == 0xFFFF_FFFF, which tests EXEC == 0 preservation of all three.
    // v_swaprel_b32 v1, v1: VOP1 opcode 0x68 -> 0x7E02_D101 (vdst=1, op=0x68<<9=0xD000, src0=257)
    let swap = 0x7E00_0000 | (1u32 << 17) | (0x68u32 << 9) | 257;
    // Pack v1 + (v2 << 8) + (v3 << 16) into v0 using v_lshl_or_b32 / v_lshlrev:
    let shader = build_valu_cs(&[
        0xBE86_037E, // s_mov_b32 s6, exec_lo
        // If s3 == -1, clear exec_lo so v_swaprel_b32 must leave v1..v3 untouched.
        0xBF06_C103, // s_cmp_eq_i32 s3, -1
        0xBF84_0001, // s_cbranch_scc0 +1 -> set m0
        0xBEFE_0380, // s_mov_b32 exec_lo, 0
        0xBEFC_0303, // s_mov_b32 m0, s3
        swap,
        0xBEFE_0306, // s_mov_b32 exec_lo, s6 (restore active lanes)
        0x3404_0488, // v_lshlrev_b32 v2, 8, v2
        0x3406_0690, // v_lshlrev_b32 v3, 16, v3
        0x3800_0501, // v_or_b32 v0, v1, v2
        0x3800_0700, // v_or_b32 v0, v0, v3
    ]);

    let cases: &[([u32; 4], u32)] = &[
        // (v1, v2, v3, m0) -> packed (v1 | v2<<8 | v3<<16)
        ([0x11, 0x22, 0x33, 0x0000_0002], 0x00_11_22_33), // dst+0 (v1) <-> src+2 (v3)
        ([0x11, 0x22, 0x33, 0x0001_0002], 0x00_22_33_11), // dst+1 (v2) <-> src+2 (v3)
        ([0x11, 0x22, 0x33, 0x0001_0001], 0x00_33_22_11), // self-swap dst+1 <-> src+1
        ([0x11, 0x22, 0x33, 0xFFFF_FFFF], 0x00_33_22_11), // EXEC == 0: unchanged
    ];
    let inputs: Vec<[u32; 4]> = cases.iter().map(|c| c.0).collect();
    run_cases(&shader, &inputs, |backend, i, inp, got| {
        let want = cases[i].1;
        assert_eq!(
            got, want,
            "{backend}: v_swaprel_b32(m0={:#010x}) = {got:#010x}, want {want:#010x}",
            inp[3],
        );
    });
}

#[test]
fn v_perm_b32_matches_rdna2() {
    // v_perm_b32 v0, v1, v2, v3 (VOP3 opcode 0x344): dynamic selector in v3.
    let perm_dyn = build_valu_cs(&vop3(0x344, 0, vgpr(1), vgpr(2), vgpr(3)));
    // Inline constant selector: -1 (0xC1 = 0xFFFF_FFFF -> all bytes >= 13 -> 0xFFFF_FFFF).
    let perm_const_ff = build_valu_cs(&vop3(0x344, 0, vgpr(1), vgpr(2), 0xC1));
    // Inline constant selector: 12 (0x8C = 0x0000_000C -> byte 0 = 0x00, bytes 1..3 = byte 0 of src1).
    let perm_const_12 = build_valu_cs(&vop3(0x344, 0, vgpr(1), vgpr(2), 0x8C));

    let ref_perm = |src0: u32, src1: u32, sel: u32| -> u32 {
        let mut out = 0u32;
        for i in 0..4u32 {
            let pick = (sel >> (8 * i)) & 0xFF;
            let byte = match pick {
                0..=3 => (src1 >> (8 * pick)) & 0xFF,
                4..=7 => (src0 >> (8 * (pick - 4))) & 0xFF,
                8 => if ((src1 >> 15) & 1) != 0 { 0xFF } else { 0x00 },
                9 => if ((src1 >> 31) & 1) != 0 { 0xFF } else { 0x00 },
                10 => if ((src0 >> 15) & 1) != 0 { 0xFF } else { 0x00 },
                11 => if ((src0 >> 31) & 1) != 0 { 0xFF } else { 0x00 },
                12 => 0x00,
                _ => 0xFF,
            };
            out |= byte << (8 * i);
        }
        out
    };

    let src0 = 0x8765_4321u32; // bytes 4..7; byte 1 sign=0 (0x43), byte 3 sign=1 (0x87)
    let src1 = 0x7EAB_CD01u32; // bytes 0..3; byte 1 sign=1 (0xCD), byte 3 sign=0 (0x7E)
    let dyn_cases: &[[u32; 4]] = &[
        [src0, src1, 0x0302_0100, 0], // identity src1
        [src0, src1, 0x0706_0504, 0], // identity src0
        [src0, src1, 0x0007_0205, 0], // mixed bytes across src0 and src1
        [src0, src1, 0x0B0A_0908, 0], // sign-extension selectors 8..11
        [0x1234_8000, 0x8000_7FFF, 0x0B0A_0908, 0], // flipped signs for 8..11
        [src0, src1, 0xFF0D_0C04, 0], // constants 12 (0x00), 13 (0xFF), 255 (0xFF)
    ];
    run_cases(&perm_dyn, dyn_cases, |backend, _, inp, got| {
        let want = ref_perm(inp[0], inp[1], inp[2]);
        assert_eq!(
            got, want,
            "{backend}: v_perm_b32({:#010x}, {:#010x}, {:#010x}) = {got:#010x}, want {want:#010x}",
            inp[0], inp[1], inp[2],
        );
    });

    for &[s0, s1, sel, _] in dyn_cases {
        let mut w = vop3(0x344, 0, vgpr(1), vgpr(2), 0xFF).to_vec();
        w.push(sel);
        let perm_lit = build_valu_cs(&w);
        run_cases(&perm_lit, &[[s0, s1, 0, 0]], |backend, _, inp, got| {
            let want = ref_perm(inp[0], inp[1], sel);
            assert_eq!(
                got, want,
                "{backend}: v_perm_b32({:#010x}, {:#010x}, lit {sel:#010x}) = {got:#010x}, want {want:#010x}",
                inp[0], inp[1],
            );
        });
    }

    run_cases(&perm_const_ff, &[[src0, src1, 0, 0]], |backend, _, inp, got| {
        let want = ref_perm(inp[0], inp[1], 0xFFFF_FFFF);
        assert_eq!(
            got, want,
            "{backend}: v_perm_b32(const -1) = {got:#010x}, want {want:#010x}",
        );
    });

    run_cases(&perm_const_12, &[[src0, src1, 0, 0]], |backend, _, inp, got| {
        let want = ref_perm(inp[0], inp[1], 12);
        assert_eq!(
            got, want,
            "{backend}: v_perm_b32(const 12) = {got:#010x}, want {want:#010x}",
        );
    });
}

/// AMD RDNA2 ISA Guide (Scalar Memory Operations): `s_buffer_load_dword*`
/// ignores the two LSBs of `soffset` and immediate `offset` independently
/// before summing. With `soffset = 3` and `offset = 3`, the load must read
/// dword 0 (`(3 & !3) + (3 & !3) = 0`), not dword 1 (`(3 + 3) >> 2 = 1`).
#[test]
fn s_buffer_load_truncates_terms_independently() {
    let words = [
        s_mov_b32(5, 0x80),                         // s[4:5] = table offset 0
        0xF408_0202,                                // s_load_dwordx4 s[8:11], s[4:5], 0x0 (UAV)
        125 << 25,
        0xF408_0302,                                // s_load_dwordx4 s[12:15], s[4:5], 0x20 (SRV)
        125 << 25 | 0x20,
        0xBF8C_C07F,                                // s_waitcnt lgkmcnt(0)
        0xF420_0406,                                // s_buffer_load_dword s16, s[12:15], s0, offset:3
        0x0000_0003,
        0xF420_0446,                                // s_buffer_load_dword s17, s[12:15], s0, offset:7
        0x0000_0007,
        0xBF8C_C07F,                                // s_waitcnt lgkmcnt(0)
        vop1(0x001, 0, sgpr(16)),                   // v0 = s16
        vop1(0x001, 1, sgpr(17)),                   // v1 = s17
        buffer_store_dword(0, 8, 0)[0],
        buffer_store_dword(0, 8, 0)[1],
        buffer_store_dword(1, 8, 4)[0],
        buffer_store_dword(1, 8, 4)[1],
        S_ENDPGM,
    ];
    let code: Vec<u8> = words.iter().flat_map(|w| w.to_le_bytes()).collect();
    let native = NativeProgram {
        stage: HwStage::Cs,
        wave64: false,
        vgprs: 4,
        layout: vec![
            RootParam {
                root: RootType::Constants,
                table_offset_dwords: 0,
                sgprs: 1,
                base_register: 0,
                user_data: 0,
                slot: 0,
            },
            RootParam {
                root: RootType::Table,
                table_offset_dwords: 0,
                sgprs: 1,
                base_register: 0,
                user_data: 4,
                slot: 1,
            },
        ],
        used: vec![
            UsedResource {
                kind: ResourceType::Uav,
                root: RootType::Table,
                table_offset_dwords: 0,
                lower_bound: 0,
                stride: 0,
                dwords: 4,
                user_data: 4,
                slot: 1,
            },
            UsedResource {
                kind: ResourceType::Srv,
                root: RootType::Table,
                table_offset_dwords: 8,
                lower_bound: 0,
                stride: 0,
                dwords: 4,
                user_data: 4,
                slot: 1,
            },
        ],
        io: StageIo::Cs(ComputeIo {
            rsrc2: 5 << 1,
            threads: [1, 1, 1],
        }),
        code,
    };
    let (p, st) = range::build(&native.code, native.wave64, native.vgprs).unwrap();
    let st = st.unwrap();
    let res = resource::track(&p, &native.layout, native.stage, Some(&st)).unwrap();
    let io = io::raise(&p, &native).unwrap();
    let lowered = lower::lower(&lower::Parts {
        native: &native,
        p: &p,
        st: &st,
        io: &io,
        res: &res,
        inputs: Vec::new(),
        outputs: Vec::new(),
        prims: Vec::new(),
        mesh_stride: 0,
        target: super::Target::ANY,
        view_marks: false,
    })
    .unwrap();
    let shader = HostShader {
        stage: HwStage::Cs,
        wave_size: None,
        dxil: lowered.dxil,
        user_data: lowered.user_data,
        samplers: lowered.samplers,
        static_samplers: Vec::new(),
        root_cbvs: lowered.root_cbvs,
        root_srvs: lowered.root_srvs,
        root_uavs: lowered.root_uavs,
        memory: lowered.memory,
        memory_writes: lowered.memory_writes,
        view_addresses: lowered.view_addresses,
        heap: lowered.heap,
        heap_writes: lowered.heap_writes,
        view_marks: lowered.view_marks,
    };
    if let Ok(mut v) = validator::Validator::load(validator::Validator::default_path().as_deref()) {
        let mut copy = shader.dxil.clone();
        v.validate(&mut copy).expect("DXIL validation");
    }

    let src_data = [0x1111_1111u32, 0x2222_2222, 0x3333_3333, 0x4444_4444];
    let views = [
        View::RawUav(2),
        View::Buffer {
            data: &src_data,
            stride: Stride::Raw,
        },
    ];
    let _one = ONE_AT_A_TIME.lock().unwrap_or_else(|e| e.into_inner());
    let mut ran = 0;
    for (label, warp) in [("hardware", false), ("warp", true)] {
        let Ok(host) = Host::new(warp) else {
            eprintln!("no SM 6.6 {label} device; skipped");
            continue;
        };
        let (root, pso) = host.compute_pipeline(&shader).unwrap();
        for (soffset, want0, want1) in [
            (3u32, 0x1111_1111u32, 0x2222_2222u32),
            (7u32, 0x2222_2222u32, 0x3333_3333u32),
        ] {
            let mut ud = vec![0u32; shader.user_data_dwords().max(1) as usize];
            for u in &shader.user_data {
                ud[u.index as usize] = match (u.root, u.slot) {
                    (RootType::Constants, 0) => soffset,
                    (RootType::Table, 1) => 0,
                    other => panic!("unexpected user data {other:?}"),
                };
            }
            let out = host
                .dispatch(
                    &root,
                    &pso,
                    &Dispatch {
                        roots: vec![(0, Root::Constants(ud))],
                        views: &views,
                        samplers: &[],
                        groups: [1, 1, 1],
                    },
                )
                .unwrap();
            assert_eq!(out[0][0], want0, "{label}: soffset={soffset} offset=3");
            assert_eq!(out[0][1], want1, "{label}: soffset={soffset} offset=7");
        }
        ran += 1;
    }
    assert!(ran > 0, "neither hardware nor WARP SM 6.6 device available");
}
