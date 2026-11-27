//! End-to-end tests for `ds_permute_b32` (forward cross-lane scatter) and
//! `ds_nop`: synthetic GFX10 programs lowered to DXIL, validated by `dxil.dll`
//! when present, and run across all SM 6.6 D3D12 hardware adapters and WARP.

use windows::Win32::Graphics::Direct3D::*;
use windows::Win32::Graphics::Direct3D12::*;
use windows::Win32::Graphics::Dxgi::*;

use super::host::{Dispatch, Host, Root, Stride, View};
use super::valu_tests::ONE_AT_A_TIME;
use super::{HostShader, StaticSampler, lower, validator};
use crate::native::{
    ComputeIo, HwStage, MeshIo, NativeProgram, ResourceType, RootParam, RootType, StageIo,
    UsedResource,
};
use crate::{ShaderError, io, ir, resource, structure};

const S_ENDPGM: u32 = 0xBF81_0000;
const S_WAITCNT_LGKM0: u32 = 0xBF8C_C07F;
const S_WAITCNT_VM0: u32 = 0xBF8C_3F70;

const fn sop1(op: u8, sdst: u8, src0: u8) -> u32 {
    0xBE80_0000 | ((sdst as u32 & 0x7F) << 16) | ((op as u32) << 8) | src0 as u32
}

const fn s_load_dwordx4(dst: u8, base: u8, byte_off: u32) -> [u32; 2] {
    [
        0x3D << 26 | 0x02 << 18 | ((dst as u32) << 6) | (base as u32 >> 1),
        (125 << 25) | (byte_off & 0x1F_FFFF),
    ]
}

const fn vop1(op: u8, vdst: u8, src0: u16) -> u32 {
    0x7E00_0000 | ((vdst as u32) << 17) | ((op as u32) << 9) | (src0 as u32 & 0x1FF)
}

const fn vop2(op: u8, vdst: u8, src0: u16, vsrc1: u8) -> u32 {
    ((op as u32 & 0x3F) << 25)
        | ((vdst as u32) << 17)
        | ((vsrc1 as u32) << 9)
        | (src0 as u32 & 0x1FF)
}

const fn vop3(op: u16, vdst: u8, src0: u16, src1: u16, src2: u16) -> [u32; 2] {
    [
        (0x35 << 26) | ((op as u32 & 0x3FF) << 16) | vdst as u32,
        (src0 as u32 & 0x1FF) | ((src1 as u32 & 0x1FF) << 9) | ((src2 as u32 & 0x1FF) << 18),
    ]
}

/// MUBUF `buffer_load_dwordx2 v[vdata:vdata+1], vaddr, s[srsrc:srsrc+3], 0 offen` (raw).
const fn buffer_load_dwordx2_offen(vdata: u8, vaddr: u8, srsrc: u8) -> [u32; 2] {
    [
        (0x38 << 26) | (0x0D << 18) | (1 << 12),
        (0x80 << 24) | (((srsrc as u32 >> 2) & 0x1F) << 16) | ((vdata as u32) << 8) | vaddr as u32,
    ]
}

/// MUBUF `buffer_store_dword vdata, vaddr, s[srsrc:srsrc+3], 0 offen` (raw).
const fn buffer_store_dword_offen(vdata: u8, vaddr: u8, srsrc: u8) -> [u32; 2] {
    [
        (0x38 << 26) | (0x1C << 18) | (1 << 12),
        (0x80 << 24) | (((srsrc as u32 >> 2) & 0x1F) << 16) | ((vdata as u32) << 8) | vaddr as u32,
    ]
}

const fn ds_nop() -> [u32; 2] {
    [0xD850_0000, 0]
}

const fn ds_permute_b32(vdst: u8, vaddr: u8, vdata: u8, offset: u16) -> [u32; 2] {
    [
        0xDAC8_0000 | offset as u32,
        ((vdst as u32) << 24) | ((vdata as u32) << 8) | vaddr as u32,
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
    wave_min: u32,
    wave_max: u32,
    host: Host,
}

fn all_hosts() -> Vec<AdapterHost> {
    let mut out = Vec::new();
    let warp_only = std::env::var("VICTORY_WARP").is_ok_and(|v| v == "1");
    // SAFETY: standard DXGI adapter enumeration and D3D12 feature queries.
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
                if let Some(h) = open_adapter(&adapter, name) {
                    out.push(h);
                }
            }
        }
        if let Ok(warp) = factory.EnumWarpAdapter::<IDXGIAdapter1>()
            && let Some(h) = open_adapter(&warp, "D3D12 WARP".into())
        {
            out.push(h);
        }
    }
    out
}

unsafe fn open_adapter(adapter: &IDXGIAdapter1, name: String) -> Option<AdapterHost> {
    // SAFETY: valid adapter handle and feature structs.
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
            wave_min: opts1.WaveLaneCountMin,
            wave_max: opts1.WaveLaneCountMax,
            host: Host::from_device(device),
        })
    }
}

fn lower_native(native: &NativeProgram) -> Result<HostShader, ShaderError> {
    let p = ir::build(&native.code, native.wave64, native.vgprs)?;
    let st = structure::structure(&p)?;
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
        view_marks: false,
    })?;
    let shader = HostShader {
        stage: native.stage,
        dxil: lowered.dxil,
        user_data: lowered.user_data,
        static_samplers: lowered
            .samplers
            .iter()
            .map(|&w| StaticSampler::from_words(w))
            .collect(),
        samplers: lowered.samplers,
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
        v.validate(&mut shader.dxil.clone())
            .expect("DXIL validation failed");
    } else {
        eprintln!("no dxil.dll; validation skipped");
    }
    Ok(shader)
}

/// Builds a compute program with 0 allocated LDS (`rsrc2` LDS size = 0) that:
/// 1. Reads per-lane `(vaddr, vdata)` from raw SRV `t0` at byte `8 * lane`;
/// 2. Pre-loads destination `v3 = 0xDEAD_0000 | lane`;
/// 3. Narrows `EXEC` to the 32- or 64-bit active mask in root constants `s0..s1`;
/// 4. Executes `ds_nop` followed by `ds_permute_b32 v3, v1, v2 offset:24`;
/// 5. Restores `EXEC` and writes `v3` to raw UAV `u0` at byte `4 * lane`.
fn build_permute_cs(wave64: bool, offset: u16) -> HostShader {
    let wave = if wave64 { 64 } else { 32 };
    let mut w = Vec::new();

    // User SGPRs: s0..s1 = active mask (root constants), s2 = descriptor table.
    w.push(sop1(0x03, 4, 2)); // s_mov_b32 s4, s2
    w.push(sop1(0x03, 5, 0x80)); // s_mov_b32 s5, 0
    w.extend_from_slice(&s_load_dwordx4(8, 4, 0)); // SRV at table offset 0 -> s[8:11]
    w.extend_from_slice(&s_load_dwordx4(12, 4, 32)); // UAV at table offset 32 -> s[12:15]

    // v4 = lane * 8 (SRV byte offset), v5 = lane * 4 (UAV byte offset).
    w.push(vop2(0x1A, 4, 128 + 3, 0)); // v_lshlrev_b32 v4, 3, v0
    w.push(vop2(0x1A, 5, 128 + 2, 0)); // v_lshlrev_b32 v5, 2, v0
    w.push(S_WAITCNT_LGKM0);

    // v[1:2] = (vaddr, vdata) from raw SRV; v3 = 0xDEAD0000 | lane.
    w.extend_from_slice(&buffer_load_dwordx2_offen(1, 4, 8));
    w.push(vop2(0x1C, 3, 0xFF, 0)); // v_or_b32 v3, 0xDEAD0000, v0
    w.push(0xDEAD_0000);
    w.push(S_WAITCNT_VM0);

    // Compute per-lane active predicate from s0 (lanes 0..31) / s1 (lanes 32..63):
    // v6 = word, v7 = lane & 31, v6 = (v6 >> v7) & 1, v_cmpx_ne_u32 exec, 0, v6.
    w.push(vop2(0x1B, 7, 128 + 31, 0)); // v_and_b32 v7, 31, v0
    w.push(vop1(0x01, 6, sgpr(0))); // v_mov_b32 v6, s0
    if wave64 {
        w.push(vop1(0x01, 8, sgpr(1))); // v_mov_b32 v8, s1
        // v_cmp_gt_u32 vcc, 32, v0 -> true when v0 < 32
        w.extend_from_slice(&vop3(0x0C4, 106, 128 + 32, vgpr(0), 0));
        // v_cndmask_b32 v6, v8, v6, vcc -> v0 < 32 ? s0 : s1
        w.extend_from_slice(&vop3(0x101, 6, vgpr(8), vgpr(6), 106));
    }
    w.push(vop2(0x16, 6, vgpr(7), 6)); // v_lshrrev_b32 v6, v7, v6
    w.push(vop2(0x1B, 6, 128 + 1, 6)); // v_and_b32 v6, 1, v6

    if wave64 {
        w.push(sop1(0x04, 20, 0x7E)); // s_mov_b64 s[20:21], exec
        w.push(0x7DAA_0080 | (6 << 9)); // v_cmpx_ne_u32 exec, 0, v6
    } else {
        w.push(sop1(0x03, 20, 0x7E)); // s_mov_b32 s20, exec_lo
        w.push(0x7DAA_0080 | (6 << 9)); // v_cmpx_ne_u32 exec_lo, 0, v6
    }

    w.extend_from_slice(&ds_nop());
    w.extend_from_slice(&ds_permute_b32(3, 1, 2, offset));
    w.push(S_WAITCNT_LGKM0);

    if wave64 {
        w.push(sop1(0x04, 0x7E, 20)); // s_mov_b64 exec, s[20:21]
    } else {
        w.push(sop1(0x03, 0x7E, 20)); // s_mov_b32 exec_lo, s20
    }

    w.extend_from_slice(&buffer_store_dword_offen(3, 5, 12));
    w.push(S_ENDPGM);

    let range = |kind, table_offset_dwords| UsedResource {
        kind,
        root: RootType::Table,
        table_offset_dwords,
        lower_bound: 0,
        stride: 0,
        dwords: 4,
        user_data: 2,
        slot: 1,
    };
    let native = NativeProgram {
        stage: HwStage::Cs,
        wave64,
        vgprs: 16,
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
        used: vec![range(ResourceType::Srv, 0), range(ResourceType::Uav, 8)],
        io: StageIo::Cs(ComputeIo {
            // USER_SGPR_COUNT = 3 (s0, s1, s2), LDS_SIZE = 0.
            rsrc2: 3 << 1,
            threads: [wave, 1, 1],
        }),
        code: w.iter().flat_map(|x| x.to_le_bytes()).collect(),
    };
    lower_native(&native).expect("lower permute compute shader")
}

/// Reference RDNA2 model for `ds_permute_b32` per 32-lane sub-vector.
/// For multi-writer collisions, checks that the result equals one of the active
/// writers targeting that destination lane.
fn check_permute_result(
    label: &str,
    case: &str,
    wave: usize,
    offset: u16,
    active_mask: u64,
    inputs: &[(u32, u32)],
    got: &[u32],
) {
    assert_eq!(got.len(), wave);
    for (me, &got_val) in got.iter().enumerate() {
        let me_on = (active_mask >> me) & 1 != 0;
        if !me_on {
            let sentinel = 0xDEAD_0000 | me as u32;
            assert_eq!(
                got_val, sentinel,
                "{label} [{case}] inactive dst lane {me} modified"
            );
            continue;
        }
        let half = me & !31;
        let mut writers = Vec::new();
        for k in 0..32 {
            let src = half + k;
            if (active_mask >> src) & 1 == 0 {
                continue;
            }
            let (addr, data) = inputs[src];
            let target = half | (((addr.wrapping_add(u32::from(offset)) >> 2) & 31) as usize);
            if target == me {
                writers.push(data);
            }
        }
        if writers.is_empty() {
            assert_eq!(
                got_val, 0,
                "{label} [{case}] unwritten active dst lane {me} expected 0"
            );
        } else {
            assert!(
                writers.contains(&got_val),
                "{label} [{case}] lane {me}: got {got_val:#010x}, expected one of {writers:#010x?}",
            );
        }
    }
}

type PermuteCase<'a> = (&'a str, u64, Vec<(u32, u32)>);

fn run_permute_cases(wave64: bool, offset: u16, cases: &[PermuteCase<'_>]) {
    let wave = if wave64 { 64u32 } else { 32 };
    let shader = build_permute_cs(wave64, offset);
    let _guard = ONE_AT_A_TIME.lock().unwrap_or_else(|e| e.into_inner());
    let hosts = all_hosts();
    let any_capable = hosts
        .iter()
        .any(|h| h.wave_min <= wave && wave <= h.wave_max);
    let mut ran = 0usize;

    for h in &hosts {
        if !(h.wave_min <= wave && wave <= h.wave_max) {
            eprintln!(
                "{}: waves {}..{} do not include {wave}; skipped",
                h.name, h.wave_min, h.wave_max
            );
            continue;
        }
        let (root, pso) = h
            .host
            .compute_pipeline(&shader)
            .unwrap_or_else(|e| panic!("{} (wave{wave}): pipeline failed: {e}", h.name));

        for (case, mask, inputs) in cases {
            let srv_words: Vec<u32> = inputs.iter().flat_map(|&(a, d)| [a, d]).collect();
            let views = [
                View::Buffer {
                    data: &srv_words,
                    stride: Stride::Raw,
                },
                View::RawUav(wave),
            ];
            let mut ud = vec![0u32; shader.user_data_dwords().max(1) as usize];
            for u in &shader.user_data {
                ud[u.index as usize] = match u.root {
                    RootType::Constants if u.index == 0 => *mask as u32,
                    RootType::Constants => (*mask >> 32) as u32,
                    RootType::Table => 0,
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
                        views: &views,
                        samplers: &[],
                        groups: [1, 1, 1],
                    },
                )
                .unwrap_or_else(|e| panic!("{} [{case}]: dispatch failed: {e}", h.name));
            check_permute_result(&h.name, case, wave as usize, offset, *mask, inputs, &out[0]);
        }
        ran += 1;
    }

    if any_capable {
        assert!(ran > 0, "capable Wave{wave} adapter failed to run");
    }
}

#[test]
fn ds_permute_b32_wave32_bijection_partial_exec_and_collisions() {
    let offset = 24u16; // +6 dwords
    let full = u32::MAX as u64;

    // 1. Full-wave bijection with unaligned low bits (+3) and wrap past 32 dwords:
    //    lane `i` scatters to `(7 * i + 6 + 9) mod 32` (since 7 is odd, gcd(7, 32) = 1).
    let bijective: Vec<(u32, u32)> = (0..32u32)
        .map(|i| (((7 * i + 9 + 64) * 4) + (i & 3), 0x1000_0000 | (i + 1)))
        .collect();

    // 2. Partial EXEC:
    //    - Active lanes: even lanes 0, 2, .., 22 and odd lane 25 (`mask_partial`).
    //    - Inactive lane 1 tries to write active lane 0 (`0xBAD0_0001`) -> ignored, lane 0 stays 0!
    //    - Active lane 2 writes inactive lane 3 (`0xABCD_0002`) -> lane 3 keeps `0xDEAD_0003`!
    //    - Active lanes 4 and 20 BOTH target active lane 8 -> collision resolves to one of them,
    //      while inactive lane 13 ALSO targets lane 8 (`0xBAD0_000D`) -> never picked!
    let mut partial_mask = 0u64;
    for l in (0..24).step_by(2) {
        partial_mask |= 1u64 << l;
    }
    partial_mask |= 1u64 << 25;

    let addr_for_target = |target: u32| ((target + 32 - 6) * 4) + 2;
    let mut partial_inputs = vec![(addr_for_target(22), 0u32); 32];
    for (l, slot) in partial_inputs.iter_mut().enumerate() {
        *slot = (addr_for_target(22), 0x2000_0000 | l as u32);
    }
    // Make lane 0 unwritten by any active lane, targeted only by inactive lane 1:
    partial_inputs[1] = (addr_for_target(0), 0xBAD0_0001);
    // Active lane 2 targets inactive lane 3:
    partial_inputs[2] = (addr_for_target(3), 0xABCD_0002);
    // Active lanes 4 and 20 + inactive lane 13 all target active lane 8:
    partial_inputs[4] = (addr_for_target(8), 0xCAFE_0004);
    partial_inputs[13] = (addr_for_target(8), 0xBAD0_000D);
    partial_inputs[20] = (addr_for_target(8), 0xCAFE_0014);
    // Active lane 25 targets active lane 10:
    partial_inputs[25] = (addr_for_target(10), 0xBEEF_0019);

    run_permute_cases(
        false,
        offset,
        &[
            ("wave32_bijective_wrap_unaligned", full, bijective),
            (
                "wave32_partial_exec_and_collision",
                partial_mask,
                partial_inputs,
            ),
        ],
    );
}

#[test]
fn ds_permute_b32_wave64_stays_within_32_lane_halves() {
    let offset = 12u16; // +3 dwords
    // In Wave64, RDNA2 scopes `ds_permute_b32` to two independent 32-lane halves
    // (0..31 and 32..63). Pass addresses with bit 7 (`32 * 4 = 128`) set in the
    // lower half and cleared in the upper half to prove neither half crosses 32.
    let full64 = u64::MAX;
    let half_isolated: Vec<(u32, u32)> = (0..64u32)
        .map(|i| {
            let in_half = i & 31;
            let dst_in_half = (5 * in_half + 11) & 31;
            // Deliberately point raw dword index into the OTHER half (`^ 32`):
            // modulo-32 masking must keep the scatter inside `i`'s own half.
            let raw_dword = ((dst_in_half + 32 - 3) & 31) ^ ((i < 32) as u32 * 32);
            (raw_dword * 4 + 1, 0x5000_0000 | (i << 8) | dst_in_half)
        })
        .collect();

    // Partial 64-bit EXEC across both halves:
    let partial64: u64 = 0x5555_AAAA_3333_CCCC;
    let partial_inputs: Vec<(u32, u32)> = (0..64u32)
        .map(|i| {
            let dst_in_half = (3 * i + 7) & 31;
            let raw_dword = dst_in_half + 64 - 3;
            (raw_dword * 4 + 3, 0x6000_0000 | i)
        })
        .collect();

    run_permute_cases(
        true,
        offset,
        &[
            ("wave64_half_isolation", full64, half_isolated),
            ("wave64_partial_exec", partial64, partial_inputs),
        ],
    );
}

#[test]
fn ds_nop_executes_without_lds_on_hardware_and_warp() {
    let mut w = Vec::new();
    w.push(sop1(0x03, 4, 0)); // s_mov_b32 s4, s0
    w.push(sop1(0x03, 5, 0x80)); // s_mov_b32 s5, 0
    w.extend_from_slice(&s_load_dwordx4(8, 4, 0)); // UAV at table offset 0 -> s[8:11]
    w.push(vop2(0x1A, 1, 128 + 2, 0)); // v_lshlrev_b32 v1, 2, v0
    w.push(vop2(0x1C, 2, 0xFF, 0)); // v_or_b32 v2, 0x5AFE0000, v0
    w.push(0x5AFE_0000);
    w.extend_from_slice(&ds_nop());
    w.push(S_WAITCNT_LGKM0);
    w.extend_from_slice(&buffer_store_dword_offen(2, 1, 8));
    w.push(S_ENDPGM);

    let native = NativeProgram {
        stage: HwStage::Cs,
        wave64: false,
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
            rsrc2: 1 << 1, // USER_SGPR_COUNT = 1, LDS_SIZE = 0
            threads: [32, 1, 1],
        }),
        code: w.iter().flat_map(|x| x.to_le_bytes()).collect(),
    };
    let shader = lower_native(&native).expect("lower ds_nop compute shader");

    let _guard = ONE_AT_A_TIME.lock().unwrap_or_else(|e| e.into_inner());
    let mut ran = 0usize;
    for h in all_hosts() {
        let (root, pso) = h
            .host
            .compute_pipeline(&shader)
            .unwrap_or_else(|e| panic!("{}: ds_nop pipeline failed: {e}", h.name));
        let ud = vec![0u32; shader.user_data_dwords().max(1) as usize];
        let views = [View::RawUav(32)];
        let out = h
            .host
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
            .unwrap_or_else(|e| panic!("{}: ds_nop dispatch failed: {e}", h.name));
        let want: Vec<u32> = (0..32u32).map(|i| 0x5AFE_0000 | i).collect();
        assert_eq!(out[0], want, "{}", h.name);
        ran += 1;
    }
    assert!(ran > 0, "neither hardware nor WARP SM 6.6 device available");
}

#[test]
fn ds_permute_b32_refuses_non_compute_stages() {
    let mut w = Vec::new();
    w.push(sop1(0x03, 4, 0)); // s_mov_b32 s4, s0
    w.push(sop1(0x03, 5, 0x80)); // s_mov_b32 s5, 0
    w.extend_from_slice(&s_load_dwordx4(8, 4, 0)); // UAV -> s[8:11]
    w.push(S_WAITCNT_LGKM0);
    w.extend_from_slice(&ds_nop());
    w.extend_from_slice(&ds_permute_b32(2, 0, 1, 0));
    w.push(S_WAITCNT_LGKM0);
    w.extend_from_slice(&buffer_store_dword_offen(2, 0, 8));
    w.push(S_ENDPGM);
    let code: Vec<u8> = w.iter().flat_map(|x| x.to_le_bytes()).collect();

    let layout = vec![RootParam {
        root: RootType::Table,
        table_offset_dwords: 0,
        sgprs: 1,
        base_register: 0,
        user_data: 0,
        slot: 0,
    }];
    let used = vec![UsedResource {
        kind: ResourceType::Uav,
        root: RootType::Table,
        table_offset_dwords: 0,
        lower_bound: 0,
        stride: 0,
        dwords: 4,
        user_data: 0,
        slot: 0,
    }];

    let ps = NativeProgram {
        stage: HwStage::Ps,
        wave64: false,
        vgprs: 4,
        layout: layout.clone(),
        used: used.clone(),
        io: StageIo::Ps {
            inputs: Vec::new(),
            input_addr: 0,
            db_shader_control: 0,
            patches: Vec::new(),
        },
        code: code.clone(),
    };
    let Err(ShaderError::Unsupported { what, .. }) = lower_native(&ps) else {
        panic!("expected ds_permute_b32 to be refused in a pixel shader");
    };
    assert_eq!(what, "a cross-lane operation outside compute");

    let ms_words: [u32; 7] = [
        0xD850_0000,
        0x0000_0000, // ds_nop
        0xDAC8_0000,
        0x0200_0100, // ds_permute_b32 v2, v0, v1
        0xD834_0000,
        0x0000_0200, // ds_write_b32 v0, v2
        S_ENDPGM,
    ];
    let ms = NativeProgram {
        stage: HwStage::Ms,
        wave64: false,
        vgprs: 4,
        layout: Vec::new(),
        used: Vec::new(),
        io: StageIo::Ms(MeshIo {
            threads: 32,
            max_vertices: 32,
            max_primitives: 32,
            clip_cntl: 0,
            topology: 2,
            lds_bytes: 256,
            outputs: Vec::new(),
        }),
        code: ms_words.iter().flat_map(|x| x.to_le_bytes()).collect(),
    };
    let Err(ShaderError::Unsupported { what, .. }) = lower_native(&ms) else {
        panic!("expected ds_permute_b32 to be refused in a mesh shader");
    };
    assert_eq!(what, "a cross-lane operation in a mesh shader");
}
