//! End-to-end GPU readback tests for whole-wave prefix sums (`Op::AddF` and
//! `Op::AddU`) across Wave32 and Wave64 on all SM 6.6 D3D12 adapters and WARP.

use windows::Win32::Graphics::Direct3D::*;
use windows::Win32::Graphics::Direct3D12::*;
use windows::Win32::Graphics::Dxgi::*;

use super::container::SigElement;
use super::host::{Dispatch, Draw, Host, Root, Stride, View};
use super::valu_tests::ONE_AT_A_TIME;
use super::{HostShader, lower, validator};
use crate::native::{
    ComputeIo, HwStage, NativeProgram, ResourceType, RootParam, RootType, StageIo, Usage,
    UsedResource, VsInput,
};
use crate::{io, ir, resource, structure};

const S_ENDPGM: u32 = 0xBF81_0000;

const fn sop1(op: u8, sdst: u8, src0: u8) -> u32 {
    0xBE80_0000 | (sdst as u32 & 0x7F) << 16 | (op as u32) << 8 | src0 as u32
}

const fn s_load_dwordx4(dst: u8, base: u8, byte_off: u32) -> [u32; 2] {
    [
        0x3D << 26 | 0x02 << 18 | (dst as u32) << 6 | (base as u32 >> 1),
        125 << 25 | (byte_off & 0x1F_FFFF),
    ]
}

const fn vop1(op: u8, vdst: u8, src0: u16) -> u32 {
    0x7E00_0000 | (vdst as u32) << 17 | (op as u32) << 9 | (src0 as u32 & 0x1FF)
}

const fn vop1_dpp16(
    op: u8,
    vdst: u8,
    vsrc0: u8,
    ctrl: u16,
    row_mask: u8,
    bound_ctrl: bool,
) -> [u32; 2] {
    [
        vop1(op, vdst, 0xFA),
        (row_mask as u32) << 28
            | 0xF << 24
            | (bound_ctrl as u32) << 19
            | (ctrl as u32 & 0x1FF) << 8
            | vsrc0 as u32,
    ]
}

const fn vop2(op: u8, vdst: u8, src0: u16, vsrc1: u8) -> u32 {
    (op as u32 & 0x3F) << 25 | (vdst as u32) << 17 | (vsrc1 as u32) << 9 | (src0 as u32 & 0x1FF)
}

const fn vop2_dpp16(
    op: u8,
    vdst: u8,
    vsrc0: u8,
    vsrc1: u8,
    ctrl: u16,
    row_mask: u8,
    bound_ctrl: bool,
) -> [u32; 2] {
    [
        vop2(op, vdst, 0xFA, vsrc1),
        (row_mask as u32) << 28
            | 0xF << 24
            | (bound_ctrl as u32) << 19
            | (ctrl as u32 & 0x1FF) << 8
            | vsrc0 as u32,
    ]
}

const fn vop3(op: u16, vdst: u8, src0: u16, src1: u16, src2: u16) -> [u32; 2] {
    [
        0x35 << 26 | (op as u32 & 0x3FF) << 16 | vdst as u32,
        (src0 as u32 & 0x1FF) | (src1 as u32 & 0x1FF) << 9 | (src2 as u32 & 0x1FF) << 18,
    ]
}

/// MUBUF `buffer_load_dword vdata, vaddr, s[srsrc:srsrc+3], 0 idxen` (structured).
const fn buffer_load_dword_idxen(vdata: u8, vaddr: u8, srsrc: u8) -> [u32; 2] {
    [
        0x38 << 26 | 0x0C << 18 | 1 << 13,
        0x80 << 24 | ((srsrc as u32 >> 2) & 0x1F) << 16 | (vdata as u32) << 8 | vaddr as u32,
    ]
}

/// MUBUF `buffer_store_dword vdata, vaddr, s[srsrc:srsrc+3], 0 offen` (raw).
const fn buffer_store_dword_offen(vdata: u8, vaddr: u8, srsrc: u8) -> [u32; 2] {
    [
        0x38 << 26 | 0x1C << 18 | 1 << 12,
        0x80 << 24 | ((srsrc as u32 >> 2) & 0x1F) << 16 | (vdata as u32) << 8 | vaddr as u32,
    ]
}

const fn sgpr(n: u8) -> u16 {
    n as u16
}

const fn vgpr(n: u8) -> u16 {
    256 + n as u16
}

struct AdapterHost {
    name: String,
    warp: bool,
    wave_min: u32,
    wave_max: u32,
    host: Host,
}

fn all_hosts() -> Vec<AdapterHost> {
    let mut out = Vec::new();
    let warp_only = std::env::var("VICTORY_WARP").is_ok_and(|v| v == "1");
    // SAFETY: standard DXGI enumeration and D3D12 feature queries.
    unsafe {
        let Ok(factory) = CreateDXGIFactory2::<IDXGIFactory4>(DXGI_CREATE_FACTORY_FLAGS(0)) else {
            return out;
        };
        if !warp_only {
            let mut idx = 0u32;
            while let Ok(adapter) = factory.EnumAdapters1(idx) {
                idx += 1;
                let Ok(desc) = adapter.GetDesc1() else {
                    continue;
                };
                if (desc.Flags & DXGI_ADAPTER_FLAG_SOFTWARE.0 as u32) != 0
                    || desc.DedicatedVideoMemory == 0
                {
                    continue;
                }
                let len = desc
                    .Description
                    .iter()
                    .position(|&c| c == 0)
                    .unwrap_or(desc.Description.len());
                let name = String::from_utf16_lossy(&desc.Description[..len]);
                if let Some(h) = open_adapter(&adapter, name, false) {
                    out.push(h);
                }
            }
        }
        if let Ok(warp) = factory.EnumWarpAdapter::<IDXGIAdapter1>()
            && let Some(h) = open_adapter(&warp, "D3D12 WARP".into(), true)
        {
            out.push(h);
        }
    }
    out
}

unsafe fn open_adapter(adapter: &IDXGIAdapter1, name: String, warp: bool) -> Option<AdapterHost> {
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

#[derive(Clone, Copy)]
enum Mode {
    Plain,
    /// Narrows EXEC to active bitmask `s0` (Wave32) or `s0..s1` (Wave64);
    /// inactive lanes keep sentinel `0xDEAD_0000 | lane` unless overwritten.
    Narrowed {
        branch: bool,
    },
}

/// Builds a synthetic Wave32 or Wave64 compute shader that loads one dword per
/// lane from structured SRV `t0` (heap index 0), runs a whole-wave prefix sum
/// (`float`: `v_add_f32`, else `v_add_nc_u32`; `inclusive` or exclusive), and
/// writes each lane's result to raw UAV `u0` (heap index 1) at `lane * 4`.
fn build_prefix_cs(wave64: bool, float: bool, inclusive: bool, mode: Mode) -> HostShader {
    let wave = if wave64 { 64u32 } else { 32 };
    let add_op: u8 = if float { 0x03 } else { 0x25 }; // v_add_f32 / v_add_nc_u32
    let mut w = Vec::new();

    // User SGPRs: s0..s1 = active lane mask (root constants), s2 = table base (0).
    // System SGPR after s0..s2: s3 = TGID.x (unused); v0 = ThreadIdInGroup.x (0..wave-1).
    w.push(sop1(0x03, 5, 0x80)); // s_mov_b32 s5, 0
    w.push(sop1(0x03, 4, 2)); // s_mov_b32 s4, s2
    w.extend_from_slice(&s_load_dwordx4(8, 4, 0)); // SRV at heap 0 (offset 0) -> s[8:11]
    w.extend_from_slice(&s_load_dwordx4(12, 4, 32)); // UAV at heap 1 (offset 32) -> s[12:15]

    // v1 = In[v0]; v4 = sentinel 0xDEAD_0000 | v0 (default output for inactive lanes).
    w.extend_from_slice(&buffer_load_dword_idxen(1, 0, 8));
    w.push(vop2(0x1C, 4, 0xFF, 0)); // v_or_b32 v4, 0xDEAD0000, v0
    w.push(0xDEAD_0000);

    // Optional EXEC narrowing from per-lane mask: `(s0 >> v0) & 1 != 0` (or 64-bit).
    let mut patch_branch = None;
    if let Mode::Narrowed { branch } = mode {
        // Compute bit = (v0 < 32 ? s0 : s1) >> (v0 & 31) & 1; v_cmpx_ne_u32 exec, 0, bit.
        w.push(vop2(0x1B, 5, 128 + 31, 0)); // v_and_b32 v5, 31, v0
        w.push(vop2(0x16, 6, sgpr(0), 5)); // v_lshrrev_b32 v6, s0, v5 ( Wait: v_lshrrev_b32 is src1 >> src0! )
        // Fix operand order of v_lshrrev_b32: vdst = vsrc1 >> src0!
        w.pop();
        w.push(vop1(0x01, 6, sgpr(0))); // v_mov_b32 v6, s0
        if wave64 {
            w.push(vop1(0x01, 7, sgpr(1))); // v_mov_b32 v7, s1
            // v_cmp_ge_u32_e64 s[16:17], v0, 32 (VOP3 0x0C6)
            w.extend_from_slice(&vop3(0x0C6, 16, vgpr(0), 128 + 32, 0));
            // v_cndmask_b32_e64 v6, v6, v7, s[16:17]
            w.extend_from_slice(&vop3(0x101, 6, vgpr(6), vgpr(7), sgpr(16)));
        }
        w.push(vop2(0x16, 6, vgpr(5), 6)); // v_lshrrev_b32 v6, v5, v6 -> v6 >> v5
        w.push(vop2(0x1B, 6, 128 + 1, 6)); // v_and_b32 v6, 1, v6
        // v_cmpx_ne_u32_e32 exec, 0, v6 (VOPC 0xD5)
        w.push(0x3E << 25 | 0xD5 << 17 | 6 << 9 | 128);
        if branch {
            patch_branch = Some(w.len());
            w.push(0); // placeholder for s_cbranch_execz
        }
    }

    // ---- Whole-wave prefix sum section ----
    if wave64 {
        w.push(sop1(0x25, 6, 193)); // s_or_saveexec_b64 s[6:7], -1
    } else {
        w.push(sop1(0x3D, 6, 193)); // s_or_saveexec_b32 s6, -1
    }
    // v2 = saved_exec ? v1 : 0
    w.extend_from_slice(&vop3(0x101, 2, 128, vgpr(1), sgpr(6)));
    // Intra-row inclusive scan: row_shr:1, 2, 4, 8 with bound_ctrl:1
    for shift in [1u16, 2, 4, 8] {
        w.extend_from_slice(&vop2_dpp16(add_op, 2, 2, 2, 0x110 | shift, 0xF, true));
    }
    // Cross-row propagation using v_readlane_b32 + constant EXEC masks.
    // Row 0 -> Row 1 (lanes 16..31):
    w.extend_from_slice(&vop3(0x360, 18, vgpr(2), 128 + 15, 0)); // v_readlane_b32 s18, v2, 15
    w.push(sop1(0x03, 126, 0xFF)); // s_mov_b32 exec_lo, 0xFFFF_0000
    w.push(0xFFFF_0000);
    if wave64 {
        w.push(sop1(0x03, 127, 128)); // s_mov_b32 exec_hi, 0
    }
    w.push(vop2(add_op, 2, sgpr(18), 2)); // v_add v2, s18, v2

    if wave64 {
        // Row 1 -> Row 2 (lanes 32..47):
        w.extend_from_slice(&vop3(0x360, 19, vgpr(2), 128 + 31, 0)); // v_readlane_b32 s19, v2, 31
        w.push(sop1(0x03, 126, 128)); // s_mov_b32 exec_lo, 0
        w.push(sop1(0x03, 127, 0xFF)); // s_mov_b32 exec_hi, 0x0000_FFFF
        w.push(0x0000_FFFF);
        w.push(vop2(add_op, 2, sgpr(19), 2));

        // Row 2 -> Row 3 (lanes 48..63):
        w.extend_from_slice(&vop3(0x360, 20, vgpr(2), 128 + 47, 0)); // v_readlane_b32 s20, v2, 47
        w.push(sop1(0x03, 127, 0xFF)); // s_mov_b32 exec_hi, 0xFFFF_0000
        w.push(0xFFFF_0000);
        w.push(vop2(add_op, 2, sgpr(20), 2));
    }

    let out_vgpr = if inclusive {
        2
    } else {
        // Restore full EXEC to shift right by 1 across rows into v3:
        if wave64 {
            w.push(sop1(0x04, 126, 193)); // s_mov_b64 exec, -1
        } else {
            w.push(sop1(0x03, 126, 193)); // s_mov_b32 exec_lo, -1
        }
        w.extend_from_slice(&vop1_dpp16(0x01, 3, 2, 0x111, 0xF, true)); // v_mov_b32_dpp v3, v2 row_shr:1 bound_ctrl:1
        // Fix lane 16 from s18 (v2[15]):
        w.push(sop1(0x03, 126, 0xFF));
        w.push(0x0001_0000);
        if wave64 {
            w.push(sop1(0x03, 127, 128));
        }
        w.push(vop1(0x01, 3, sgpr(18)));
        if wave64 {
            // Fix lane 32 from s19 (v2[31]):
            w.push(sop1(0x03, 126, 128));
            w.push(sop1(0x03, 127, 128 + 1));
            w.push(vop1(0x01, 3, sgpr(19)));
            // Fix lane 48 from s20 (v2[47]):
            w.push(sop1(0x03, 127, 0xFF));
            w.push(0x0001_0000);
            w.push(vop1(0x01, 3, sgpr(20)));
        }
        3
    };

    // Restore saved EXEC:
    if wave64 {
        w.push(sop1(0x04, 126, 6)); // s_mov_b64 exec, s[6:7]
    } else {
        w.push(sop1(0x03, 126, 6)); // s_mov_b32 exec_lo, s6
    }
    // Copy prefix result into v4 for active lanes.
    w.push(vop1(0x01, 4, vgpr(out_vgpr)));

    if let Some(br_idx) = patch_branch {
        let rel = (w.len() - (br_idx + 1)) as i16;
        w[br_idx] = 0xBF88_0000 | (rel as u16 as u32); // s_cbranch_execz
    }

    // Restore full EXEC so every thread writes its v4 out to UAV[v0 * 4].
    if matches!(mode, Mode::Narrowed { .. }) {
        if wave64 {
            w.push(sop1(0x04, 126, 193));
        } else {
            w.push(sop1(0x03, 126, 193));
        }
    }
    w.push(vop2(0x1A, 5, 128 + 2, 0)); // v_lshlrev_b32 v5, 2, v0 (v0 << 2)
    w.extend_from_slice(&buffer_store_dword_offen(4, 5, 12));
    w.push(S_ENDPGM);

    let code: Vec<u8> = w.iter().flat_map(|x| x.to_le_bytes()).collect();
    let native = NativeProgram {
        stage: HwStage::Cs,
        wave64,
        vgprs: 10,
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
                root: RootType::Table,
                table_offset_dwords: 0,
                sgprs: 1,
                base_register: 0,
                user_data: 2,
                slot: 1,
            },
        ],
        used: vec![
            UsedResource {
                kind: ResourceType::Srv,
                root: RootType::Table,
                table_offset_dwords: 0,
                lower_bound: 0,
                stride: 4,
                dwords: 4,
                user_data: 2,
                slot: 1,
            },
            UsedResource {
                kind: ResourceType::Uav,
                root: RootType::Table,
                table_offset_dwords: 8,
                lower_bound: 1,
                stride: 0,
                dwords: 4,
                user_data: 2,
                slot: 1,
            },
        ],
        io: StageIo::Cs(ComputeIo {
            rsrc2: 3 << 1, // 3 user SGPRs: s0, s1, s2
            threads: [wave, 1, 1],
        }),
        code,
    };

    let p = ir::build(&native.code, native.wave64, native.vgprs).expect("ir::build");
    // Ensure the whole-wave section was recognised as Shape::Prefix rather than falling through.
    assert!(
        (0..p.blocks.len()).any(|b| {
            super::wave::sections(&p, b).values().any(|s| {
                s.shapes
                    .values()
                    .any(|sh| matches!(sh, super::wave::Shape::Prefix(..)))
            })
        }),
        "expected Shape::Prefix in whole-wave section"
    );
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
        view_marks: false,
    })
    .expect("lower");

    let shader = HostShader {
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
    };

    if let Ok(mut v) = validator::Validator::load(validator::Validator::default_path().as_deref()) {
        let mut copy = shader.dxil.clone();
        let ours = copy[4..20].to_vec();
        v.validate(&mut copy).expect("DXIL validation failed");
        assert_eq!(copy[4..20], ours[..], "container hash mismatch");
    }

    shader
}

fn cpu_f32_prefix(terms: &[u32], active_mask: u64, inclusive: bool) -> Vec<u32> {
    let mut out = Vec::with_capacity(terms.len());
    let mut acc: Option<f32> = None;
    for (lane, &bits) in terms.iter().enumerate() {
        let on = (active_mask >> lane) & 1 != 0;
        if !on {
            out.push(0xDEAD_0000 | lane as u32);
            continue;
        }
        let x = f32::from_bits(bits);
        if inclusive {
            let next = match acc {
                None => x,
                Some(prev) => prev + x,
            };
            acc = Some(next);
            out.push(next.to_bits());
        } else {
            out.push(acc.unwrap_or(0.0).to_bits());
            acc = Some(match acc {
                None => x,
                Some(prev) => prev + x,
            });
        }
    }
    out
}

fn assert_f32_lanes(label: &str, case_name: &str, got: &[u32], want: &[u32]) {
    for (lane, (&g, &w)) in got.iter().zip(want).enumerate() {
        let gf = f32::from_bits(g);
        let wf = f32::from_bits(w);
        if wf.is_nan() && (w & 0xFFFF_0000) != 0xDEAD_0000 {
            assert!(
                gf.is_nan(),
                "{label} [{case_name}] lane {lane}: got {gf} ({g:#010x}), want NaN ({w:#010x})"
            );
        } else {
            assert_eq!(
                g, w,
                "{label} [{case_name}] lane {lane}: got {gf} ({g:#010x}), want {wf} ({w:#010x})"
            );
        }
    }
}

fn base_terms(n: usize) -> Vec<u32> {
    // Exact multiples of 0.25 with alternating signs; lane 0..8 sum = -3.75, 0..10 sum = -17.75.
    let preset: [f32; 16] = [
        1.25, -2.5, 0.75, -1.0, 3.5, -4.25, 0.5, -2.0, 0.0, -6.5, -7.5, 2.25, -1.5, 4.0, -0.75,
        3.25,
    ];
    (0..n)
        .map(|i| {
            let base = preset[i % 16];
            let adj = (i / 16) as f32 * 0.5;
            (base + adj).to_bits()
        })
        .collect()
}

fn run_f32_suite(wave64: bool, inclusive: bool, mode: Mode) {
    let wave = if wave64 { 64usize } else { 32 };
    let shader = build_prefix_cs(wave64, true, inclusive, mode);

    let nan = 0x7FC0_0000u32;
    let pinf = f32::INFINITY.to_bits();
    let ninf = f32::NEG_INFINITY.to_bits();
    let two60 = 0x5D80_0000u32; // 2^60

    let mut cases: Vec<(&str, Vec<u32>, u64)> = Vec::new();
    let full = if wave64 { u64::MAX } else { u32::MAX as u64 };

    cases.push(("exact_quarters", base_terms(wave), full));

    let mut nan_9_20 = base_terms(wave);
    nan_9_20[9] = nan;
    nan_9_20[20] = nan;
    if wave64 {
        nan_9_20[45] = nan;
    }
    cases.push(("nan_lanes_9_20", nan_9_20, full));

    let mut nan_0 = base_terms(wave);
    nan_0[0] = nan;
    cases.push(("nan_lane_0", nan_0, full));

    let mut two60_11 = base_terms(wave);
    two60_11[11] = two60;
    if wave64 {
        two60_11[40] = two60;
    }
    cases.push(("two60_lane_11", two60_11, full));

    let mut pinf_11 = base_terms(wave);
    pinf_11[11] = pinf;
    if wave64 {
        pinf_11[40] = pinf;
    }
    cases.push(("pos_inf_lane_11", pinf_11, full));

    let mut ninf_11 = base_terms(wave);
    ninf_11[11] = ninf;
    cases.push(("neg_inf_lane_11", ninf_11, full));

    if matches!(mode, Mode::Narrowed { .. }) {
        // Inactive lanes hold NaN/+inf/2^60 while active lanes also test NaN/+inf/2^60 across gaps.
        let mask = if wave64 {
            0xA5A5_5A5A_DB6D_B6DBu64
        } else {
            0xDB6D_B6DBu64
        };
        let mut masked_exotic = base_terms(wave);
        for l in 0..wave {
            if (mask >> l) & 1 == 0 {
                masked_exotic[l] = [nan, pinf, two60][l % 3];
            }
        }
        // Put 2^60 and +inf on active lanes too (lane 11 is active in 0xDB6D_B6DB: 0xDB6 = 1101_1011_0110, bit 11 = 1).
        let mut m_two60 = masked_exotic.clone();
        m_two60[11] = two60;
        cases.push(("masked_two60", m_two60, mask));

        let mut m_pinf = masked_exotic.clone();
        m_pinf[11] = pinf;
        cases.push(("masked_pos_inf", m_pinf, mask));

        let mut m_nan = masked_exotic;
        m_nan[9] = nan; // bit 9 of 0xDB6D_B6DB: 0xB6DB >> 9 -> 0xDB6 = 1101_1011_0110, bit 9 = 1.
        cases.push(("masked_nan", m_nan, mask));
    }

    let _one = ONE_AT_A_TIME.lock().unwrap_or_else(|e| e.into_inner());
    let hosts = all_hosts();
    assert!(!hosts.is_empty(), "no SM 6.6 wave-capable adapter found");
    let mut ran = 0usize;
    for h in &hosts {
        let w = wave as u32;
        if w < h.wave_min || w > h.wave_max {
            eprintln!(
                "skip Wave{wave} on {} (wave range {}..={})",
                h.name, h.wave_min, h.wave_max
            );
            continue;
        }
        let (root, pso) = h
            .host
            .compute_pipeline(&shader)
            .unwrap_or_else(|e| panic!("{} (warp={}): PSO failed: {e}", h.name, h.warp));
        for (case_name, terms, mask) in &cases {
            let mut ud = vec![0u32; shader.user_data_dwords().max(1) as usize];
            for u in &shader.user_data {
                ud[u.index as usize] = match (u.root, u.slot, u.dword) {
                    (RootType::Constants, 0, 0) => *mask as u32,
                    (RootType::Constants, 0, 1) => (*mask >> 32) as u32,
                    (RootType::Table, 1, 0) => 0,
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
                        views: &[
                            View::Buffer {
                                data: terms,
                                stride: Stride::Structured(4),
                            },
                            View::RawUav(wave as u32),
                        ],
                        samplers: &[],
                        groups: [1, 1, 1],
                    },
                )
                .unwrap_or_else(|e| panic!("{}: dispatch failed: {e}", h.name));
            let want = cpu_f32_prefix(terms, *mask, inclusive);
            let tag = format!("{} (Wave{wave}, inclusive={inclusive})", h.name);
            assert_f32_lanes(&tag, case_name, &out[0], &want);
        }
        eprintln!(
            "ok: {} [Wave{wave}, inclusive={inclusive}, wave={}..={}]",
            h.name, h.wave_min, h.wave_max
        );
        ran += 1;
    }
    if ran == 0 {
        eprintln!("no Wave{wave} adapter; skipped");
    }
}

#[test]
fn wave_prefix_sum_f32_exclusive_wave32() {
    run_f32_suite(false, false, Mode::Plain);
    run_f32_suite(false, false, Mode::Narrowed { branch: false });
    run_f32_suite(false, false, Mode::Narrowed { branch: true });
}

#[test]
fn wave_prefix_sum_f32_inclusive_wave32() {
    run_f32_suite(false, true, Mode::Plain);
    run_f32_suite(false, true, Mode::Narrowed { branch: false });
    run_f32_suite(false, true, Mode::Narrowed { branch: true });
}

#[test]
fn wave_prefix_sum_f32_wave64() {
    run_f32_suite(true, false, Mode::Plain);
    run_f32_suite(true, true, Mode::Plain);
    run_f32_suite(true, false, Mode::Narrowed { branch: true });
    run_f32_suite(true, true, Mode::Narrowed { branch: true });
}

#[test]
fn wave_prefix_sum_u32_reads_back() {
    let shader = build_prefix_cs(false, false, false, Mode::Plain);
    let terms: Vec<u32> = (0..32u32)
        .map(|i| i.wrapping_mul(0x0101_0003).wrapping_add(7))
        .collect();
    let mut want = Vec::with_capacity(32);
    let mut acc = 0u32;
    for &t in &terms {
        want.push(acc);
        acc = acc.wrapping_add(t);
    }
    let _one = ONE_AT_A_TIME.lock().unwrap_or_else(|e| e.into_inner());
    for h in all_hosts() {
        if 32 < h.wave_min || 32 > h.wave_max {
            continue;
        }
        let (root, pso) = h.host.compute_pipeline(&shader).expect("u32 PSO");
        let ud = vec![0u32; shader.user_data_dwords().max(1) as usize];
        let out = h
            .host
            .dispatch(
                &root,
                &pso,
                &Dispatch {
                    roots: vec![(0, Root::Constants(ud))],
                    views: &[
                        View::Buffer {
                            data: &terms,
                            stride: Stride::Structured(4),
                        },
                        View::RawUav(32),
                    ],
                    samplers: &[],
                    groups: [1, 1, 1],
                },
            )
            .expect("u32 dispatch");
        assert_eq!(out[0], want, "{}: u32 exclusive prefix mismatch", h.name);
    }
}

/// Pixel shader (`wave_size == None`, `MaskData::Host`) drawn over a 2x2 quad
/// so every SM 6.6 backend — including D3D12 WARP (`wave_min..=wave_max = 4..=4`) —
/// executes both exclusive and inclusive `Op::AddF` prefix sums on GPU.
#[test]
fn wave_prefix_sum_f32_pixel_quad_all_backends() {
    // Pass-through VS: POSITION0 (v4..v7) -> SV_Position.
    let vs_code: Vec<u8> = [
        0xF800_08CFu32, // exp pos0 v4, v5, v6, v7 done
        0x0706_0504,
        0xF800_0200, // exp param0 off, off, off, off
        0,
        S_ENDPGM,
    ]
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
    let vp = ir::build(&vs_native.code, false, vs_native.vgprs).expect("vs ir");
    let vst = structure::structure(&vp).expect("vs structure");
    let vres = resource::track(&vp, &vs_native.layout, HwStage::Vs, Some(&vst)).expect("vs res");
    let vio = io::raise(&vp, &vs_native).expect("vs io");
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
        view_marks: false,
    })
    .expect("vs lower");
    let vs = HostShader {
        stage: HwStage::Vs,
        dxil: vlow.dxil,
        user_data: vlow.user_data,
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

    // PS: root constants s0..s3 hold terms cycled by active rank (`rank & 3`).
    // Entering the whole-wave section saves EXEC in s6; `s_bcnt1_i32_b32 s7, s6`
    // gives the wave's active lane count and `v_mbcnt_lo_u32_b32 v0, s6, 0`
    // gives each active pixel's rank `0..count-1` within its host wave.
    let mut pw = Vec::new();
    pw.push(sop1(0x3D, 6, 193)); // s_or_saveexec_b32 s6, -1
    pw.push(sop1(0x03, 126, 6)); // s_mov_b32 exec_lo, s6 (capture entry EXEC in s6)
    pw.push(sop1(0x0F, 7, 6)); // s_bcnt1_i32_b32 s7, s6 (active lane count)
    pw.extend_from_slice(&vop3(0x365, 0, sgpr(6), 128, 0)); // v_mbcnt_lo_u32_b32 v0, s6, 0
    pw.push(vop2(0x1B, 8, 128 + 3, 0)); // v_and_b32 v8, 3, v0 (rank & 3)
    pw.push(vop1(0x01, 1, sgpr(3))); // v1 = s3
    for k in (0..3u8).rev() {
        // v_cmp_eq_u32_e64 s10, v8, k (VOP3 0x0C2)
        pw.extend_from_slice(&vop3(0x0C2, 10, vgpr(8), 128 + k as u16, 0));
        // v_cndmask_b32_e64 v1, v1, s[k], s10
        pw.extend_from_slice(&vop3(0x101, 1, vgpr(1), sgpr(k), sgpr(10)));
    }
    // Whole-wave inclusive + exclusive f32 prefix sum:
    pw.push(sop1(0x3D, 6, 193)); // s_or_saveexec_b32 s6, -1
    pw.extend_from_slice(&vop3(0x101, 2, 128, vgpr(1), sgpr(6)));
    for shift in [1u16, 2, 4, 8] {
        pw.extend_from_slice(&vop2_dpp16(0x03, 2, 2, 2, 0x110 | shift, 0xF, true));
    }
    pw.extend_from_slice(&vop3(0x360, 18, vgpr(2), 128 + 15, 0));
    pw.push(sop1(0x03, 126, 0xFF));
    pw.push(0xFFFF_0000);
    pw.push(vop2(0x03, 2, sgpr(18), 2));
    // Shift into v3 for exclusive prefix:
    pw.push(sop1(0x03, 126, 193));
    pw.extend_from_slice(&vop1_dpp16(0x01, 3, 2, 0x111, 0xF, true));
    pw.push(sop1(0x03, 126, 0xFF));
    pw.push(0x0001_0000);
    pw.push(vop1(0x01, 3, sgpr(18)));
    pw.push(sop1(0x03, 126, 6)); // restore exec
    // Pack (count << 8) | rank into v6; export (excl, incl, count_rank, own).
    pw.extend_from_slice(&vop3(0x36F, 6, sgpr(7), 128 + 8, vgpr(0))); // v_lshl_or_b32 v6, s7, 8, v0
    pw.push(vop1(0x01, 4, vgpr(3)));
    pw.push(vop1(0x01, 5, vgpr(2)));
    pw.push(vop1(0x01, 7, vgpr(1)));
    // exp mrt0 v4, v5, v6, v7 done vm
    pw.push(0xF800_180F);
    pw.push(4 | 5 << 8 | 6 << 16 | 7 << 24);
    pw.push(S_ENDPGM);

    let ps_code: Vec<u8> = pw.iter().flat_map(|w| w.to_le_bytes()).collect();
    let ps_native = NativeProgram {
        stage: HwStage::Ps,
        wave64: false,
        vgprs: 10,
        layout: vec![RootParam {
            root: RootType::Constants,
            table_offset_dwords: 0,
            sgprs: 4,
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
    let pp = ir::build(&ps_native.code, false, ps_native.vgprs).expect("ps ir");
    let pst = structure::structure(&pp).expect("ps structure");
    let pres = resource::track(&pp, &ps_native.layout, HwStage::Ps, Some(&pst)).expect("ps res");
    let pio = io::raise(&pp, &ps_native).expect("ps io");
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
        view_marks: false,
    })
    .expect("ps lower");
    let ps = HostShader {
        stage: HwStage::Ps,
        dxil: plow.dxil,
        user_data: plow.user_data,
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
        for sh in [&vs, &ps] {
            let mut copy = sh.dxil.clone();
            v.validate(&mut copy).expect("PS/VS DXIL validation failed");
        }
    }

    let nan = 0x7FC0_0000u32;
    let pinf = f32::INFINITY.to_bits();
    let ninf = f32::NEG_INFINITY.to_bits();
    let two60 = 0x5D80_0000u32;
    let f = f32::to_bits;
    let quad_cases: &[(&str, [u32; 4])] = &[
        ("exact", [f(1.25), f(-2.5), f(0.75), f(3.5)]),
        ("nan_lane_0", [nan, f(-2.5), f(0.75), f(3.5)]),
        ("nan_lane_2", [f(1.25), f(-2.5), nan, f(3.5)]),
        ("two60_lane_2", [f(1.25), f(-2.5), two60, f(3.5)]),
        ("pos_inf_lane_2", [f(1.25), f(-2.5), pinf, f(3.5)]),
        ("neg_inf_lane_2", [f(1.25), f(-2.5), ninf, f(3.5)]),
    ];
    let tri: [f32; 12] = [
        -1.0, -1.0, 0.0, 1.0, //
        3.0, -1.0, 0.0, 1.0, //
        -1.0, 3.0, 0.0, 1.0,
    ];

    let _one = ONE_AT_A_TIME.lock().unwrap_or_else(|e| e.into_inner());
    let hosts = all_hosts();
    if !hosts.iter().any(|h| h.warp) {
        eprintln!("no SM 6.6 WARP device; skipped");
    }
    let mut ran = 0usize;
    for h in &hosts {
        let root = h.host.root_signature(&vs, &ps).expect("gfx root");
        let pso = h
            .host
            .pipeline_with(&root, &vs.dxil, &ps.dxil)
            .unwrap_or_else(|e| panic!("{}: gfx PSO failed: {e}", h.name));
        for &(name, inp) in quad_cases {
            let mut ud = vec![0u32; ps.user_data_dwords().max(1) as usize];
            for u in &ps.user_data {
                ud[u.index as usize] = inp[u.dword as usize];
            }
            let pixels = h
                .host
                .draw(
                    &root,
                    &pso,
                    &Draw {
                        roots: vec![(0, Root::Constants(vec![0])), (1, Root::Constants(ud))],
                        textures: &[],
                        samplers: &[],
                        vertices: &tri,
                        stride: 16,
                        size: (4, 4),
                        mesh_groups: 0,
                    },
                )
                .unwrap_or_else(|e| panic!("{}: draw failed: {e}", h.name));
            let mut max_wave = 0usize;
            for (p_idx, px) in pixels.iter().enumerate() {
                let info = px[2].to_bits();
                let count = (info >> 8) as usize;
                let rank = (info & 0xFF) as usize;
                assert!(
                    count >= 1 && rank < count,
                    "{} [{name}] px {p_idx}: invalid wave info {info:#x}",
                    h.name
                );
                max_wave = max_wave.max(count);
                let wave_terms: Vec<u32> = (0..count).map(|r| inp[r & 3]).collect();
                let mask = if count == 64 {
                    u64::MAX
                } else {
                    (1u64 << count) - 1
                };
                let excl_want = cpu_f32_prefix(&wave_terms, mask, false);
                let incl_want = cpu_f32_prefix(&wave_terms, mask, true);
                assert_f32_lanes(
                    &format!("{} PS excl (count={count}, rank={rank})", h.name),
                    name,
                    &[px[0].to_bits()],
                    &[excl_want[rank]],
                );
                assert_f32_lanes(
                    &format!("{} PS incl (count={count}, rank={rank})", h.name),
                    name,
                    &[px[1].to_bits()],
                    &[incl_want[rank]],
                );
            }
            assert!(
                max_wave >= 4,
                "{}: expected at least a 4-lane wave on 4x4 target, got max_wave={max_wave}",
                h.name
            );
        }
        ran += 1;
        eprintln!(
            "ok: {} [PS 4x4 target, wave={}..={}]",
            h.name, h.wave_min, h.wave_max
        );
    }
    assert!(ran > 0, "no SM 6.6 wave-capable adapter found");
}
