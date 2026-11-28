//! Title memory at addresses the program computes (decision 0005, the
//! unified-memory item), run on the host GPU and WARP with the test host
//! serving the contract (`emit` "Title memory"): a synthetic GFX10 program
//! takes a 64-bit pointer from root constants, reads through it with a
//! global and a scalar load, follows a pointer it loaded with another
//! scalar load, stores through a pointer it computes (64-bit multiply-add)
//! and does float min/max atomics, reads through the address it takes
//! from a buffer view's descriptor and an image descriptor's size fields
//! (the host's view addresses), reads
//! pointers into memory no copy holds (zeros, marked missed) and stores and
//! loads through a root UAV (title memory in such a program). Also: buffer
//! loads of a table in the program's code, zeros out of the records.

use super::host::{Dispatch, Host, Root, View};
use super::{HostShader, lower, validator};
use crate::native::{
    ComputeIo, HwStage, NativeProgram, ResourceType, RootParam, RootType, StageIo, UsedResource,
};
use crate::{io, ir, resource, structure};

const S_ENDPGM: u32 = 0xBF81_0000;
const S_WAITCNT_ALL: u32 = 0xBF8C_0000;
/// Inline constant 0, 64.
const ZERO: u16 = 128;
const SIXTY_FOUR: u16 = 192;
/// The title address of the memory the program reads: above 4 GiB.
const TITLE: u64 = 0x12_3450_0000;

const fn s_mov_b32(sdst: u8, src0: u8) -> u32 {
    0xBE80_0000 | (sdst as u32) << 16 | 0x03 << 8 | src0 as u32
}

/// SMEM `s_load_dword*` (`op` 0, 1, 2: one, two, four dwords) at
/// `s[base:base+1] + offset`.
const fn s_load(op: u32, sdata: u8, sbase: u8, offset: u32) -> [u32; 2] {
    [
        0x3D << 26 | op << 18 | (sdata as u32) << 6 | (sbase as u32 >> 1),
        125 << 25 | offset,
    ]
}

/// VOP1 `v_mov_b32 vdst, src0`.
const fn v_mov_b32(vdst: u8, src0: u16) -> u32 {
    0x7E00_0000 | (vdst as u32) << 17 | 0x01 << 9 | (src0 as u32 & 0x1FF)
}

/// GLOBAL `op` with address `s[saddr:saddr+1] + vaddr + offset`.
const fn global(op: u32, vdst: u8, vaddr: u8, data: u8, saddr: u8, offset: u32) -> [u32; 2] {
    [
        0x37 << 26 | op << 18 | 2 << 14 | (offset & 0xFFF),
        vaddr as u32 | (data as u32) << 8 | (saddr as u32) << 16 | (vdst as u32) << 24,
    ]
}

/// MUBUF `buffer_store_dword vdata, off, s[srsrc:srsrc+3], 0 offset:byte_off`.
const fn buffer_store_dword(vdata: u8, srsrc: u8, byte_off: u16) -> [u32; 2] {
    [
        0x38 << 26 | 0x1C << 18 | (byte_off as u32 & 0xFFF),
        0x80 << 24 | ((srsrc as u32 >> 2) & 0x1F) << 16 | (vdata as u32) << 8,
    ]
}

fn program() -> HostShader {
    let mut w = vec![s_mov_b32(5, 0x80)]; // s[4:5]: the table at s4
    w.extend(s_load(2, 8, 4, 0)); // s[8:11]: the result UAV
    w.push(v_mov_b32(1, ZERO));
    w.extend(global(12, 2, 1, 0, 0, 0)); // global_load_dword v2, v1, s[0:1]
    w.extend(s_load(1, 12, 0, 8)); // s[12:13] = the pointer at +8
    w.push(S_WAITCNT_ALL);
    w.extend(s_load(0, 14, 12, 0)); // s14 = what it points at
    w.push(v_mov_b32(3, SIXTY_FOUR));
    w.push(v_mov_b32(4, 2)); // v4 = s2
    // v_mad_u64_u32 v[8:9], s[20:21], v3, 1, s[0:1]: the pointer plus 0x40,
    // stored through as a VGPR pair.
    w.extend([0xD576_1408, 0x0001_0303]);
    w.extend(global(28, 0, 8, 4, 0x7D, 0)); // global_store_dword v[8:9], v4, off
    w.push(v_mov_b32(6, 3)); // v6 = s3 (a float)
    w.extend(global(63, 0, 3, 6, 0, 0x40)); // global_atomic_fmin at +0x80
    w.extend(global(64, 0, 3, 6, 0, 0x44)); // global_atomic_fmax at +0x84
    // The address of the view at heap index 1 from its descriptor (as the
    // tracers take their BVH's), read through.
    w.extend(s_load(1, 16, 4, 0x20)); // s[16:17] = descriptor 1, dwords 0-1
    w.push(S_WAITCNT_ALL);
    w.extend(s_load(0, 18, 16, 0x100)); // s18 = the dword at +0x100
    w.push(S_WAITCNT_ALL);
    // Pointers into a page no copy holds and into a 256 MiB region with no
    // block, read through: zeros, and both marked missed.
    w.extend(s_load(1, 22, 0, 0x10)); // s[22:23]
    w.extend(s_load(1, 24, 0, 0x18)); // s[24:25]
    w.push(S_WAITCNT_ALL);
    w.extend(s_load(0, 26, 22, 0));
    w.extend(s_load(0, 27, 24, 0));
    w.push(S_WAITCNT_ALL);
    w.push(v_mov_b32(5, 14));
    w.push(v_mov_b32(7, 18));
    w.extend(buffer_store_dword(2, 8, 0));
    w.extend(buffer_store_dword(5, 8, 4));
    w.extend(buffer_store_dword(7, 8, 8));
    w.push(v_mov_b32(5, 26));
    w.push(v_mov_b32(7, 27));
    w.extend(buffer_store_dword(5, 8, 12));
    w.extend(buffer_store_dword(7, 8, 16));
    // A root UAV (s[6:7]), title memory in such a program: a store and a
    // load at its address.
    w.extend(global(28, 0, 1, 4, 6, 0x20)); // global_store_dword v1, v4, s[6:7]
    w.extend(global(12, 9, 1, 0, 6, 0x24)); // global_load_dword v9, v1, s[6:7]
    w.push(S_WAITCNT_ALL);
    w.extend(buffer_store_dword(9, 8, 20));
    // The image descriptor at heap index 2 loaded whole and read only for
    // its size fields (dwords 1-3).
    w.extend(s_load(3, 28, 4, 0x40)); // s[28:35]
    w.push(S_WAITCNT_ALL);
    for (k, at) in [(29, 24), (30, 28), (31, 32)] {
        w.push(v_mov_b32(5, k));
        w.extend(buffer_store_dword(5, 8, at));
    }
    w.push(S_ENDPGM);
    let (l, memory) = lowered(&w);
    assert_eq!(
        memory, 11,
        "every access through the pointer is title memory"
    );
    assert!(l.memory_stores, "it stores and does atomics there");
    assert!(l.memory && l.memory_writes && l.view_addresses);
    assert!(
        l.root_uavs.is_empty(),
        "the root UAV is read as title memory"
    );
    let shader = HostShader {
        wave_size: l.wave_size,
        stage: HwStage::Cs,
        dxil: l.dxil,
        user_data: l.user_data,
        samplers: l.samplers,
        static_samplers: Vec::new(),
        root_cbvs: l.root_cbvs,
        root_srvs: l.root_srvs,
        root_uavs: l.root_uavs,
        memory: l.memory,
        memory_writes: l.memory_writes,
        memory_stores: l.memory_stores,
        view_addresses: l.view_addresses,
        heap: l.heap,
        heap_writes: l.heap_writes,
        view_marks: l.view_marks,
    };
    if let Ok(mut v) = validator::Validator::load(validator::Validator::default_path().as_deref()) {
        v.validate(&mut shader.dxil.clone())
            .expect("DXIL validation failed");
    }
    shader
}

/// Lowers a program of `w` with [`program`]'s root layout (constants, the
/// result UAV's table, a root UAV), and counts its title-memory accesses.
fn lowered(w: &[u32]) -> (lower::Lowered, usize) {
    let native = NativeProgram {
        stage: HwStage::Cs,
        wave64: false,
        vgprs: 10,
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
                root: RootType::Descriptor,
                table_offset_dwords: 0,
                sgprs: 2,
                base_register: 0,
                user_data: 6,
                slot: 2,
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
                kind: ResourceType::Uav,
                root: RootType::Descriptor,
                table_offset_dwords: 0,
                lower_bound: 0,
                stride: 0,
                dwords: 2,
                user_data: 6,
                slot: 2,
            },
        ],
        io: StageIo::Cs(ComputeIo {
            rsrc2: 8 << 1,
            threads: [1, 1, 1],
        }),
        code: w.iter().flat_map(|w| w.to_le_bytes()).collect(),
    };
    let p = ir::build(&native.code, native.wave64, native.vgprs).expect("ir::build");
    let st = structure::structure(&p).expect("structure");
    let res = resource::track(&p, &native.layout, native.stage, Some(&st)).expect("track");
    let memory = res
        .scalar
        .iter()
        .filter(|s| s.1.param == resource::MEMORY)
        .count();
    let io = io::raise(&p, &native).expect("io::raise");
    let l = lower::lower(&lower::Parts {
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
        console_cf: false,
        per_lane_only: false,
        view_marks: false,
    })
    .expect("lower");
    (l, memory)
}

/// A program that only loads through the page table marks misses in the
/// bitmap but stores nothing there: its submissions write no title memory
/// through the table (`memory_stores`, which the host's page-table barriers
/// go by); [`program`], storing and doing atomics, does.
#[test]
fn only_stores_and_atomics_through_the_table_count_as_stores() {
    let mut w = vec![s_mov_b32(5, 0x80)]; // s[4:5]: the table at s4
    w.extend(s_load(2, 8, 4, 0)); // s[8:11]: the result UAV
    w.push(v_mov_b32(1, ZERO));
    w.extend(global(12, 2, 1, 0, 0, 0)); // global_load_dword v2, v1, s[0:1]
    w.extend(s_load(1, 12, 0, 8)); // s[12:13] = the pointer at +8
    w.push(S_WAITCNT_ALL);
    w.extend(buffer_store_dword(2, 8, 0)); // into the result UAV
    w.push(S_ENDPGM);
    let (l, memory) = lowered(&w);
    assert!(memory > 0, "it reads title memory");
    assert!(l.memory && l.memory_writes, "its misses are marked");
    assert!(!l.memory_stores);
    assert!(program().memory_stores);
}

/// The page table for one copy of title memory at [`TITLE`] whose views
/// are at heap index `view` (null pair at `null`).
fn page_table(view: u32, null: u32) -> Vec<u32> {
    let mut t = vec![0u32; 4096];
    let (hi, lo) = ((TITLE >> 32) as u32, TITLE as u32);
    t[(hi << 4 | lo >> 28) as usize] = 1;
    for _ in 0..2 * 4096 {
        t.extend([null, 0]);
    }
    let entry = 4096 + 2 * (4096 + (lo >> 16 & 0xFFF) as usize);
    t[entry..entry + 2].copy_from_slice(&[view, lo]);
    t
}

#[test]
fn title_memory_is_read_and_written_at_computed_addresses() {
    let shader = program();
    let _one = super::valu_tests::ONE_AT_A_TIME
        .lock()
        .unwrap_or_else(|e| e.into_inner());
    let mut memory = vec![0u32; 256];
    memory[0] = 0x1111_1111;
    let pointer = TITLE + 0x100;
    memory[2] = pointer as u32;
    memory[3] = (pointer >> 32) as u32;
    // The next page (no copy) and another 256 MiB region (no block).
    let (next, far) = (TITLE + 0x1_0010, 0x55_0000_0040u64);
    memory[4..8].copy_from_slice(&[
        next as u32,
        (next >> 32) as u32,
        far as u32,
        (far >> 32) as u32,
    ]);
    memory[0x40] = 0x2222_2222;
    // The root UAV's address, and what it loads there.
    let root_uav = TITLE + 0x200;
    memory[(0x200 + 0x24) / 4] = 0x3333_3333;
    memory[0x80 / 4] = 1.0f32.to_bits();
    memory[0x84 / 4] = (-1.0f32).to_bits();
    let value = 0xCAFE_F00D;
    // Heap index 1 (the memory's view) has the title address, index 2 a
    // texture's size fields.
    let mut view_addresses = vec![0u32; 4 * 5];
    view_addresses[4..6].copy_from_slice(&[TITLE as u32, (TITLE >> 32) as u32]);
    let image = [0xC000_0000, 0x0957_C0F9, 0x9004_2FAC];
    view_addresses[9..12].copy_from_slice(&image);
    let mut ran = 0;
    for (label, warp) in Host::test_backends() {
        let Ok(host) = Host::new(warp) else {
            eprintln!("no SM 6.6 {label} device; skipped");
            continue;
        };
        let (root, pso) = host.compute_pipeline(&shader).expect("pipeline");
        // fmin with 1.0 and fmax with -1.0 in memory, against a negative
        // and a positive operand.
        for operand in [-2.0f32, 0.5] {
            let mut ud = vec![0u32; shader.user_data_dwords() as usize];
            for u in &shader.user_data {
                let constants = [TITLE as u32, (TITLE >> 32) as u32, value, operand.to_bits()];
                ud[u.index as usize] = match u.root {
                    RootType::Constants if u.slot != super::DISPATCH_SLOT => {
                        constants[u.dword as usize]
                    }
                    RootType::Descriptor => (root_uav >> (32 * u.dword)) as u32,
                    _ => 0,
                };
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
                                    data: page_table(1, 3),
                                },
                            ),
                            (2, Root::Uav(2 * 4096 / 32)),
                            (
                                3,
                                Root::View {
                                    srv: true,
                                    data: view_addresses.clone(),
                                },
                            ),
                        ],
                        views: &[
                            View::RawUav(9),
                            View::Memory(&memory),
                            View::Pair,
                            View::NullPair,
                            View::Pair,
                        ],
                        samplers: &[],
                        groups: [1, 1, 1],
                    },
                )
                .unwrap_or_else(|e| panic!("{label}: {e}"));
            assert_eq!(
                out[0],
                [
                    0x1111_1111,
                    0x2222_2222,
                    0x2222_2222,
                    0,
                    0,
                    0x3333_3333,
                    image[0],
                    image[1],
                    image[2]
                ],
                "{label}: reads"
            );
            assert_eq!(out[1][0x40 / 4], value, "{label}: the store");
            assert_eq!(
                out[1][(0x200 + 0x20) / 4],
                value,
                "{label}: the root UAV store"
            );
            let floats = [f32::from_bits(out[1][0x20]), f32::from_bits(out[1][0x21])];
            let want = [operand.min(1.0), operand.max(-1.0)];
            assert_eq!(floats, want, "{label}: float min and max");
            // The far region by its level-1 index, the page written, the
            // page missed after it; nothing else.
            let entry = 4096 + (TITLE as u32 >> 16 & 0xFFF) as usize;
            let marked: Vec<usize> = (0..out[2].len() * 32)
                .filter(|&b| out[2][b / 32] >> (b % 32) & 1 != 0)
                .collect();
            assert_eq!(
                marked,
                [(far >> 28) as usize, entry, entry + 1],
                "{label}: pages"
            );
        }
        ran += 1;
    }
    if ran == 0 {
        crate::emit::host::no_device("no SM 6.6 device", &[]);
    }
}

fn global_store_barrier_shader(wave32: bool) -> HostShader {
    let mut w = if wave32 {
        // Use the wave lane index to require wave32. v1 = 1 + wave lane + wave base.
        vec![
            0xD765_0001, // v_mbcnt_lo_u32_b32 v1, -1, 1
            0x0001_02C1,
            0x3606_00A0, // v_and_b32 v3, 32, v0
            0x4A02_0701, // v_add_nc_u32 v1, v1, v3
        ]
    } else {
        vec![0x4A02_0081] // v_add_nc_u32 v1, 1, v0
    };
    w.push(0x3404_0082); // v_lshlrev_b32 v2, 2, v0
    w.extend(global(28, 0, 2, 1, 0, 0)); // global_store_dword v2, v1, s[0:1]
    w.extend([S_WAITCNT_ALL, 0xBF8A_0000]); // s_waitcnt; s_barrier
    w.extend([
        0x3A06_00A0, // v_xor_b32 v3, 32, v0
        0x3406_0682, // v_lshlrev_b32 v3, 2, v3
    ]);
    w.extend(global(12, 3, 3, 0, 0, 0)); // global_load_dword v3, v3, s[0:1]
    w.push(S_WAITCNT_ALL);
    // Keep the results after the source values so the reads cannot race the stores.
    w.extend(global(28, 0, 2, 3, 0, 64 * 4)); // global_store_dword v2, v3, s[0:1] offset:256
    w.push(S_ENDPGM);
    let native = NativeProgram {
        stage: HwStage::Cs,
        wave64: false,
        vgprs: 4,
        layout: vec![RootParam {
            root: RootType::Descriptor,
            table_offset_dwords: 0,
            sgprs: 2,
            base_register: 0,
            user_data: 0,
            slot: 0,
        }],
        used: vec![UsedResource {
            kind: ResourceType::Uav,
            root: RootType::Descriptor,
            table_offset_dwords: 0,
            lower_bound: 0,
            stride: 0,
            dwords: 2,
            user_data: 0,
            slot: 0,
        }],
        io: StageIo::Cs(ComputeIo {
            rsrc2: 2 << 1,
            threads: [64, 1, 1],
        }),
        code: w.iter().flat_map(|w| w.to_le_bytes()).collect(),
    };
    let p = ir::build(&native.code, native.wave64, native.vgprs).expect("ir::build");
    let st = structure::structure(&p).expect("structure");
    let res = resource::track(&p, &native.layout, native.stage, Some(&st)).expect("track");
    let io = io::raise(&p, &native).expect("io::raise");
    let container = crate::Container {
        digest: [0; 16],
        version: (1, 0),
        parts: vec![],
    };
    let mut shader = super::emit(
        &container,
        &native,
        &p,
        &st,
        &io,
        &res,
        crate::emit::Target::ANY,
    )
    .expect("emit");
    if let Ok(mut v) = validator::Validator::load(validator::Validator::default_path().as_deref()) {
        v.validate(&mut shader.dxil)
            .expect("DXIL validation failed");
    }
    shader
}

#[test]
fn global_store_values_are_visible_across_waves_after_a_barrier() {
    use windows::Win32::Graphics::Direct3D12::{
        D3D12_FEATURE_D3D12_OPTIONS1, D3D12_FEATURE_DATA_D3D12_OPTIONS1,
    };

    let _one = super::valu_tests::ONE_AT_A_TIME
        .lock()
        .unwrap_or_else(|e| e.into_inner());
    let values: Vec<u32> = (1..=64).collect();
    let swapped: Vec<u32> = (0..64).map(|lane| (lane ^ 32) + 1).collect();
    let warp_only = std::env::var("VICTORY_WARP").is_ok_and(|v| v == "1");
    let mut ran = 0;
    for (label, warp) in [("hardware", false), ("warp", true)] {
        if warp_only && !warp {
            continue;
        }
        let Ok(host) = Host::new(warp) else {
            eprintln!("no SM 6.6 {label} device; skipped");
            continue;
        };
        let mut options = D3D12_FEATURE_DATA_D3D12_OPTIONS1::default();
        // SAFETY: the feature query writes the structure of the given size.
        unsafe {
            host.device.CheckFeatureSupport(
                D3D12_FEATURE_D3D12_OPTIONS1,
                (&raw mut options).cast(),
                size_of_val(&options) as u32,
            )
        }
        .expect("wave size limits");
        if options.WaveLaneCountMin > 32 {
            eprintln!("no wave size of 32 or less on {label}; skipped");
            continue;
        }
        let shader = global_store_barrier_shader(options.WaveLaneCountMax > 32);
        let (root, pso) = host
            .compute_pipeline(&shader)
            .unwrap_or_else(|e| panic!("{label}: {e}"));
        // User data selects the root UAV's copy at heap index and byte offset 0.
        let ud = vec![0u32; shader.user_data_dwords().max(1) as usize];
        let out = host
            .dispatch(
                &root,
                &pso,
                &Dispatch {
                    roots: vec![(0, Root::Constants(ud))],
                    views: &[View::RawUav(128)],
                    samplers: &[],
                    groups: [1, 1, 1],
                },
            )
            .unwrap_or_else(|e| panic!("{label}: {e}"));
        assert_eq!(out[0][..64], values, "{label}: source values");
        assert_eq!(out[0][64..], swapped, "{label}: swapped values");
        ran += 1;
    }
    assert!(ran > 0, "neither hardware nor WARP SM 6.6 device available");
}

#[test]
fn global_atomic_fmin_and_fmax_ignore_nan_operands() {
    let shader = program();
    let _one = super::valu_tests::ONE_AT_A_TIME
        .lock()
        .unwrap_or_else(|e| e.into_inner());
    let mut memory = vec![0u32; 256];
    let pointer = TITLE + 0x100;
    memory[2] = pointer as u32;
    memory[3] = (pointer >> 32) as u32;
    let (next, far) = (TITLE + 0x1_0010, 0x55_0000_0040u64);
    memory[4..8].copy_from_slice(&[
        next as u32,
        (next >> 32) as u32,
        far as u32,
        (far >> 32) as u32,
    ]);
    let root_uav = TITLE + 0x200;
    let view_addresses = vec![0u32; 4 * 5];
    let mut ran = 0;
    for (label, warp) in [("hardware", false), ("warp", true)] {
        let Ok(host) = Host::new(warp) else {
            eprintln!("no SM 6.6 {label} device; skipped");
            continue;
        };
        let (root, pso) = host.compute_pipeline(&shader).expect("pipeline");
        // Verify all four NaN bit patterns leave finite and infinite memory
        // cells bit-identical, while +/-Inf and -0.0 still participate.
        let cases = [
            (
                1.0f32.to_bits(),
                (-1.0f32).to_bits(),
                0x7FC0_0000u32,
                1.0f32.to_bits(),
                (-1.0f32).to_bits(),
                "+qNaN vs finite",
            ),
            (
                1.0f32.to_bits(),
                (-1.0f32).to_bits(),
                0xFFC0_0000,
                1.0f32.to_bits(),
                (-1.0f32).to_bits(),
                "-qNaN vs finite",
            ),
            (
                1.0f32.to_bits(),
                (-1.0f32).to_bits(),
                0x7F80_0001,
                1.0f32.to_bits(),
                (-1.0f32).to_bits(),
                "+sNaN vs finite",
            ),
            (
                1.0f32.to_bits(),
                (-1.0f32).to_bits(),
                0xFF80_0001,
                1.0f32.to_bits(),
                (-1.0f32).to_bits(),
                "-sNaN vs finite",
            ),
            (
                f32::NEG_INFINITY.to_bits(),
                f32::INFINITY.to_bits(),
                0x7FC0_0000,
                f32::NEG_INFINITY.to_bits(),
                f32::INFINITY.to_bits(),
                "+qNaN vs Inf",
            ),
            (
                f32::NEG_INFINITY.to_bits(),
                f32::INFINITY.to_bits(),
                0xFFC0_0000,
                f32::NEG_INFINITY.to_bits(),
                f32::INFINITY.to_bits(),
                "-qNaN vs Inf",
            ),
            (
                1.0f32.to_bits(),
                (-1.0f32).to_bits(),
                f32::INFINITY.to_bits(),
                1.0f32.to_bits(),
                f32::INFINITY.to_bits(),
                "+Inf updates fmax",
            ),
            (
                1.0f32.to_bits(),
                (-1.0f32).to_bits(),
                f32::NEG_INFINITY.to_bits(),
                f32::NEG_INFINITY.to_bits(),
                (-1.0f32).to_bits(),
                "-Inf updates fmin",
            ),
            (
                1.0f32.to_bits(),
                (-1.0f32).to_bits(),
                (-0.0f32).to_bits(),
                (-0.0f32).to_bits(),
                (-0.0f32).to_bits(),
                "-0.0 updates both",
            ),
        ];
        for (init_min, init_max, op_bits, want_min, want_max, desc) in cases {
            memory[0x80 / 4] = init_min;
            memory[0x84 / 4] = init_max;
            let mut ud = vec![0u32; shader.user_data_dwords() as usize];
            for u in &shader.user_data {
                let constants = [TITLE as u32, (TITLE >> 32) as u32, 0, op_bits];
                ud[u.index as usize] = match u.root {
                    RootType::Constants => constants[u.dword as usize],
                    RootType::Descriptor => (root_uav >> (32 * u.dword)) as u32,
                    _ => 0,
                };
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
                                    data: page_table(1, 3),
                                },
                            ),
                            (2, Root::Uav(2 * 4096 / 32)),
                            (
                                3,
                                Root::View {
                                    srv: true,
                                    data: view_addresses.clone(),
                                },
                            ),
                        ],
                        views: &[
                            View::RawUav(9),
                            View::Memory(&memory),
                            View::Pair,
                            View::NullPair,
                            View::Pair,
                        ],
                        samplers: &[],
                        groups: [1, 1, 1],
                    },
                )
                .unwrap_or_else(|e| panic!("{label} ({desc}): {e}"));
            assert_eq!(
                out[1][0x20], want_min,
                "{label} ({desc}): global_atomic_fmin got {:#010x}, want {want_min:#010x}",
                out[1][0x20]
            );
            assert_eq!(
                out[1][0x21], want_max,
                "{label} ({desc}): global_atomic_fmax got {:#010x}, want {want_max:#010x}",
                out[1][0x21]
            );
        }
        ran += 1;
    }
    assert!(ran > 0, "neither hardware nor WARP SM 6.6 device available");
}

/// A root SRV in a program that does not use title memory is read through
/// the host's view of its host copy (`emit`, "Root SRVs and root UAVs"): the
/// view's index and the root address's offset in it come as user data, and
/// the shader adds the offset. Lanes read 64 bytes apart from the root
/// address 16 bytes into a 1 KiB copy: those inside it read the copy, those
/// past its end read 0 (a host root descriptor read whatever lay there).
#[test]
fn root_views_read_their_copy_and_zeros_past_it() {
    const fn vop2(op: u32, vdst: u8, src0: u16, vsrc1: u8) -> u32 {
        op << 25 | (vdst as u32) << 17 | (vsrc1 as u32) << 9 | src0 as u32
    }
    /// MUBUF `op` (offen) of `vdata` at `vaddr` through `s[srsrc:srsrc+3]`.
    const fn mubuf(op: u32, vdata: u8, vaddr: u8, srsrc: u8) -> [u32; 2] {
        [
            0x38 << 26 | op << 18 | 1 << 12,
            0x80 << 24 | (srsrc as u32 >> 2) << 16 | (vdata as u32) << 8 | vaddr as u32,
        ]
    }
    const V_LSHLREV_B32: u32 = 0x1A;
    let mut w = vec![
        s_mov_b32(5, ZERO as u8), // s[4:5]: the table at s4
        s_mov_b32(0, 6),          // s[0:3]: the root SRV's buffer descriptor
        0x8801_FF07,
        0x0001_0000, // s_or_b32 s1, s7, stride 1
        0xBE82_03C1, // s_mov_b32 s2, -1
        0xBE83_03FF,
        0x1101_4FAC, // s_mov_b32 s3, (the console's buffer word 3)
    ];
    w.extend(s_load(2, 8, 4, 0)); // s[8:11]: the result UAV
    w.push(vop2(V_LSHLREV_B32, 1, 128 + 6, 0)); // v1 = lane << 6
    w.extend(mubuf(0x0C, 2, 1, 0)); // buffer_load_dword v2, v1, s[0:3] offen
    w.push(vop2(V_LSHLREV_B32, 3, 128 + 2, 0)); // v3 = lane << 2
    w.push(S_WAITCNT_ALL);
    w.extend(mubuf(0x1C, 2, 3, 8)); // buffer_store_dword v2, v3, s[8:11] offen
    w.push(S_ENDPGM);
    let native = NativeProgram {
        stage: HwStage::Cs,
        wave64: false,
        vgprs: 4,
        layout: vec![
            RootParam {
                root: RootType::Table,
                table_offset_dwords: 0,
                sgprs: 1,
                base_register: 0,
                user_data: 4,
                slot: 1,
            },
            RootParam {
                root: RootType::Descriptor,
                table_offset_dwords: 0,
                sgprs: 2,
                base_register: 0,
                user_data: 6,
                slot: 2,
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
                root: RootType::Descriptor,
                table_offset_dwords: 0,
                lower_bound: 0,
                stride: 1,
                dwords: 4,
                user_data: 6,
                slot: 2,
            },
        ],
        io: StageIo::Cs(ComputeIo {
            rsrc2: 8 << 1,
            threads: [32, 1, 1],
        }),
        code: w.iter().flat_map(|w| w.to_le_bytes()).collect(),
    };
    let p = ir::build(&native.code, native.wave64, native.vgprs).expect("ir::build");
    let st = structure::structure(&p).expect("structure");
    let res = resource::track(&p, &native.layout, native.stage, Some(&st)).expect("track");
    let io = io::raise(&p, &native).expect("io::raise");
    let l = lower::lower(&lower::Parts {
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
        console_cf: false,
        per_lane_only: false,
        view_marks: false,
    })
    .expect("lower");
    assert!(
        l.root_srvs.is_empty() && !l.memory,
        "no root descriptor, no title memory"
    );
    let marks: Vec<(u32, u32)> = l
        .user_data
        .iter()
        .filter(|u| u.root == RootType::Descriptor)
        .map(|u| (u.index, u.dword))
        .collect();
    assert_eq!(
        marks,
        [(6, super::ROOT_SRV_VIEW), (7, super::ROOT_VIEW_OFFSET)]
    );
    let shader = HostShader {
        wave_size: l.wave_size,
        stage: HwStage::Cs,
        dxil: l.dxil,
        user_data: l.user_data,
        samplers: l.samplers,
        static_samplers: Vec::new(),
        root_cbvs: l.root_cbvs,
        root_srvs: l.root_srvs,
        root_uavs: l.root_uavs,
        memory: l.memory,
        memory_writes: l.memory_writes,
        memory_stores: l.memory_stores,
        view_addresses: l.view_addresses,
        heap: l.heap,
        heap_writes: l.heap_writes,
        view_marks: l.view_marks,
    };
    if let Ok(mut v) = validator::Validator::load(validator::Validator::default_path().as_deref()) {
        v.validate(&mut shader.dxil.clone())
            .expect("DXIL validation failed");
    }
    let _one = super::valu_tests::ONE_AT_A_TIME
        .lock()
        .unwrap_or_else(|e| e.into_inner());
    let copy: Vec<u32> = (0..256).map(|k| 0x1000 + k).collect();
    let mut ran = 0;
    for (label, warp) in Host::test_backends() {
        let Ok(host) = Host::new(warp) else {
            eprintln!("no SM 6.6 {label} device; skipped");
            continue;
        };
        let (root, pso) = host.compute_pipeline(&shader).expect("pipeline");
        let mut ud = vec![0u32; shader.user_data_dwords() as usize];
        // The table at heap index 0, the copy's view at 1, the root address
        // 16 bytes into it.
        (ud[4], ud[6], ud[7]) = (0, 1, 16);
        let out = host
            .dispatch(
                &root,
                &pso,
                &Dispatch {
                    roots: vec![(0, Root::Constants(ud))],
                    views: &[
                        View::RawUav(32),
                        View::Buffer {
                            data: &copy,
                            stride: super::host::Stride::Raw,
                        },
                    ],
                    samplers: &[],
                    groups: [1, 1, 1],
                },
            )
            .unwrap_or_else(|e| panic!("{label}: {e}"));
        let want: Vec<u32> = (0..32)
            .map(|lane| if lane < 16 { 0x1000 + 4 + 16 * lane } else { 0 })
            .collect();
        assert_eq!(out[0], want, "{label}");
        ran += 1;
    }
    if ran == 0 {
        crate::emit::host::no_device("no SM 6.6 device", &[]);
    }
}

/// Writes through unordered buffer views mark their pages (`emit`, "Write
/// marks"): a dword store and a 16-byte store through view 0 (base bit 100,
/// its first byte 16 bytes before a 64 KiB page ends, so the second store
/// crosses into the next page), a store through view 1 (not marked) and an
/// atomic add through view 2 (base bit 200). The stores land; exactly bits
/// 100, 101 and 200 are set.
#[test]
fn writes_through_views_mark_their_pages() {
    /// MUBUF `op` of `vdata` at `byte_off` through `s[srsrc:srsrc+3]`.
    const fn mubuf(op: u32, vdata: u8, srsrc: u8, byte_off: u16) -> [u32; 2] {
        [
            0x38 << 26 | op << 18 | (byte_off as u32 & 0xFFF),
            0x80 << 24 | ((srsrc as u32 >> 2) & 0x1F) << 16 | (vdata as u32) << 8,
        ]
    }
    const BUFFER_STORE_DWORDX4: u32 = 0x1E;
    const BUFFER_ATOMIC_ADD: u32 = 0x32;
    let mut w = vec![s_mov_b32(5, ZERO as u8)];
    for (k, sdata) in [8u8, 12, 16].into_iter().enumerate() {
        w.extend(s_load(2, sdata, 4, 32 * k as u32));
    }
    w.push(S_WAITCNT_ALL);
    w.extend((0..4).map(|v| v_mov_b32(v, 0)));
    w.extend(buffer_store_dword(0, 8, 0));
    w.extend(mubuf(BUFFER_STORE_DWORDX4, 0, 8, 12));
    w.extend(buffer_store_dword(0, 12, 4));
    w.extend(mubuf(BUFFER_ATOMIC_ADD, 0, 16, 8));
    w.push(S_ENDPGM);
    let param = |root, user_data, sgprs, slot| RootParam {
        root,
        table_offset_dwords: 0,
        sgprs,
        base_register: 0,
        user_data,
        slot,
    };
    let uav = |k: u32| UsedResource {
        kind: ResourceType::Uav,
        root: RootType::Table,
        table_offset_dwords: 8 * k,
        lower_bound: k,
        stride: 0,
        dwords: 4,
        user_data: 4,
        slot: 1,
    };
    let native = NativeProgram {
        stage: HwStage::Cs,
        wave64: false,
        vgprs: 8,
        layout: vec![
            param(RootType::Constants, 0, 4, 0),
            param(RootType::Table, 4, 1, 1),
        ],
        used: (0..3).map(uav).collect(),
        io: StageIo::Cs(ComputeIo {
            rsrc2: 5 << 1,
            threads: [1, 1, 1],
        }),
        code: w.iter().flat_map(|w| w.to_le_bytes()).collect(),
    };
    let p = ir::build(&native.code, native.wave64, native.vgprs).expect("ir::build");
    let st = structure::structure(&p).expect("structure");
    let res = resource::track(&p, &native.layout, native.stage, Some(&st)).expect("track");
    let io = io::raise(&p, &native).expect("io::raise");
    let l = lower::lower(&lower::Parts {
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
        console_cf: false,
        per_lane_only: false,
        view_marks: true,
    })
    .expect("lower");
    assert!(l.view_marks && !l.memory, "marks without the page table");
    let shader = HostShader {
        wave_size: l.wave_size,
        stage: HwStage::Cs,
        dxil: l.dxil,
        user_data: l.user_data,
        samplers: l.samplers,
        static_samplers: Vec::new(),
        root_cbvs: l.root_cbvs,
        root_srvs: l.root_srvs,
        root_uavs: l.root_uavs,
        memory: l.memory,
        memory_writes: l.memory_writes,
        memory_stores: l.memory_stores,
        view_addresses: l.view_addresses,
        heap: l.heap,
        heap_writes: l.heap_writes,
        view_marks: l.view_marks,
    };
    if let Ok(mut v) = validator::Validator::load(validator::Validator::default_path().as_deref()) {
        v.validate(&mut shader.dxil.clone())
            .expect("DXIL validation failed");
    }
    let _one = super::valu_tests::ONE_AT_A_TIME
        .lock()
        .unwrap_or_else(|e| e.into_inner());
    let value = 0x1234_5678;
    let marks = vec![100, 0xFFF0, super::VIEW_MARK_NONE, 0, 200, 0];
    let mut ran = 0;
    for (label, warp) in Host::test_backends() {
        let Ok(host) = Host::new(warp) else {
            eprintln!("no SM 6.6 {label} device; skipped");
            continue;
        };
        let (root, pso) = host.compute_pipeline(&shader).expect("pipeline");
        let mut ud = vec![0u32; shader.user_data_dwords() as usize];
        for u in &shader.user_data {
            ud[u.index as usize] =
                if u.root == RootType::Constants && u.slot != super::DISPATCH_SLOT {
                    value
                } else {
                    0
                };
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
                                data: marks.clone(),
                            },
                        ),
                        (2, Root::Uav(8)),
                    ],
                    views: &[View::RawUav(8), View::RawUav(8), View::RawUav(8)],
                    samplers: &[],
                    groups: [1, 1, 1],
                },
            )
            .unwrap_or_else(|e| panic!("{label}: {e}"));
        assert_eq!(
            out[0],
            [value, 0, 0, value, value, value, value, 0],
            "{label}: view 0"
        );
        assert_eq!(out[1][1], value, "{label}: view 1");
        assert_eq!(out[2][2], value, "{label}: the atomic");
        let marked: Vec<usize> = (0..out[3].len() * 32)
            .filter(|&b| out[3][b / 32] >> (b % 32) & 1 != 0)
            .collect();
        assert_eq!(marked, [100, 101, 200], "{label}: pages");
        ran += 1;
    }
    if ran == 0 {
        crate::emit::host::no_device("no SM 6.6 device", &[]);
    }
}

/// Reads of texture views the program takes from the resource heap itself
/// mark their heap index (`emit`, "Read marks"): an `image_load` of the view
/// at heap byte offset 32 (index 1) sets exactly the bit one past
/// [`super::VIEW_READS`]. The same read through a table marks nothing (a
/// table read makes its slot on the host), unless the table entry is
/// computed (a descriptor array: the host makes only the entries the
/// used-resource list names).
#[test]
fn heap_direct_texture_reads_mark_their_heap_index() {
    let lower_with = |root: RootType, used: Vec<UsedResource>, offset: u32, dynamic: bool| {
        let mut w = vec![s_mov_b32(5, ZERO as u8)];
        let mut load = s_load(3, 12, 4, offset);
        if dynamic {
            // v_readfirstlane_b32 s6, v0; s_lshl_b32 s6, s6, 5; s6 the soffset.
            w.extend([0x7E0C_0500, 0x8F06_8506]);
            load[1] = 6 << 25 | offset;
        }
        w.extend(load);
        w.push(S_WAITCNT_ALL);
        w.extend((0..2).map(|v| v_mov_b32(v, ZERO)));
        // image_load v[2:5], v[0:1], s[12:19] dmask:0xf dim:2D unorm
        w.extend([0xF000_1F08, 0x0003_0200]);
        w.push(S_WAITCNT_ALL);
        w.push(S_ENDPGM);
        let native = NativeProgram {
            stage: HwStage::Cs,
            wave64: false,
            vgprs: 8,
            layout: vec![RootParam {
                root,
                table_offset_dwords: 0,
                sgprs: 1,
                base_register: 0,
                user_data: 4,
                slot: 0,
            }],
            used,
            io: StageIo::Cs(ComputeIo {
                rsrc2: 5 << 1,
                threads: [1, 1, 1],
            }),
            code: w.iter().flat_map(|w| w.to_le_bytes()).collect(),
        };
        let p = ir::build(&native.code, native.wave64, native.vgprs).expect("ir::build");
        let st = structure::structure(&p).expect("structure");
        let res = resource::track(&p, &native.layout, native.stage, Some(&st)).expect("track");
        let io = io::raise(&p, &native).expect("io::raise");
        lower::lower(&lower::Parts {
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
            console_cf: false,
            per_lane_only: false,
            view_marks: true,
        })
        .expect("lower")
    };
    let used = vec![UsedResource {
        kind: ResourceType::Srv,
        root: RootType::Table,
        table_offset_dwords: 0,
        lower_bound: 0,
        stride: 0,
        dwords: 8,
        user_data: 4,
        slot: 0,
    }];
    let table = lower_with(RootType::Table, used.clone(), 0, false);
    assert!(!table.view_marks, "a table read marks nothing");
    let computed = lower_with(RootType::Table, used, 0, true);
    assert!(
        computed.view_marks && !computed.heap,
        "a read at a computed table entry is marked"
    );
    let l = lower_with(RootType::ResourceHeap, Vec::new(), 32, false);
    assert!(l.view_marks && l.heap, "a heap-direct read is marked");
    let shader = HostShader {
        stage: HwStage::Cs,
        wave_size: l.wave_size,
        dxil: l.dxil,
        user_data: l.user_data,
        samplers: l.samplers,
        static_samplers: Vec::new(),
        root_cbvs: l.root_cbvs,
        root_srvs: l.root_srvs,
        root_uavs: l.root_uavs,
        memory: l.memory,
        memory_writes: l.memory_writes,
        memory_stores: l.memory_stores,
        view_addresses: l.view_addresses,
        heap: l.heap,
        heap_writes: l.heap_writes,
        view_marks: l.view_marks,
    };
    if let Ok(mut v) = validator::Validator::load(validator::Validator::default_path().as_deref()) {
        v.validate(&mut shader.dxil.clone())
            .expect("DXIL validation failed");
    }
    let _one = super::valu_tests::ONE_AT_A_TIME
        .lock()
        .unwrap_or_else(|e| e.into_inner());
    let words = (super::VIEW_READS + 64) / 32;
    let texel = [[0.5f32; 4]];
    let mut ran = 0;
    for (label, warp) in Host::test_backends() {
        let Ok(host) = Host::new(warp) else {
            eprintln!("no SM 6.6 {label} device; skipped");
            continue;
        };
        let (root, pso) = host.compute_pipeline(&shader).expect("pipeline");
        // The heap's base is 0 in the user data (`emit`, "Binding").
        let ud = vec![0u32; shader.user_data_dwords() as usize];
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
                                data: vec![super::VIEW_MARK_NONE; 4],
                            },
                        ),
                        (2, Root::Uav(words)),
                    ],
                    views: &[View::Texture(1, 1, &texel), View::Texture(1, 1, &texel)],
                    samplers: &[],
                    groups: [1, 1, 1],
                },
            )
            .unwrap_or_else(|e| panic!("{label}: {e}"));
        let bits = &out[0];
        let marked: Vec<usize> = (0..bits.len() * 32)
            .filter(|&b| bits[b / 32] >> (b % 32) & 1 != 0)
            .collect();
        assert_eq!(
            marked,
            [super::VIEW_READS as usize + 1],
            "{label}: the slot read"
        );
        ran += 1;
    }
    assert!(ran > 0, "neither hardware nor WARP SM 6.6 device available");
}

/// Multi-byte writes into title memory that cross a 256 MiB region boundary
/// or a 4 GiB boundary look up their last byte's region and page in the
/// page table rather than offsetting the first page's entry or adding the
/// low dword alone (`Lower::memory`).
#[test]
fn straddling_writes_across_256mib_and_4gib_boundaries_mark_exact_pages() {
    let mut w = vec![v_mov_b32(0, ZERO)];
    for k in 0..4u8 {
        // Inline constants 129..=132 are 1..=4.
        w.push(v_mov_b32(1 + k, 129 + u16::from(k)));
    }
    w.extend(global(30, 0, 0, 1, 0, 0)); // global_store_dwordx4 v0, v[1:4], s[0:1]
    w.push(S_ENDPGM);
    let native = NativeProgram {
        stage: HwStage::Cs,
        wave64: false,
        vgprs: 5,
        layout: vec![RootParam {
            root: RootType::Constants,
            table_offset_dwords: 0,
            sgprs: 2,
            base_register: 0,
            user_data: 0,
            slot: 0,
        }],
        used: Vec::new(),
        io: StageIo::Cs(ComputeIo {
            rsrc2: 2 << 1,
            threads: [1, 1, 1],
        }),
        code: w.iter().flat_map(|w| w.to_le_bytes()).collect(),
    };
    let p = ir::build(&native.code, native.wave64, native.vgprs).expect("ir::build");
    let st = structure::structure(&p).expect("structure");
    let res = resource::track(&p, &native.layout, native.stage, Some(&st)).expect("track");
    let io = io::raise(&p, &native).expect("io::raise");
    let l = lower::lower(&lower::Parts {
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
        console_cf: false,
        per_lane_only: false,
        view_marks: false,
    })
    .expect("lower");
    assert!(l.memory && l.memory_writes);
    let shader = HostShader {
        stage: HwStage::Cs,
        wave_size: l.wave_size,
        dxil: l.dxil,
        user_data: l.user_data,
        samplers: l.samplers,
        static_samplers: Vec::new(),
        root_cbvs: l.root_cbvs,
        root_srvs: l.root_srvs,
        root_uavs: l.root_uavs,
        memory: l.memory,
        memory_writes: l.memory_writes,
        memory_stores: l.memory_stores,
        view_addresses: l.view_addresses,
        heap: l.heap,
        heap_writes: l.heap_writes,
        view_marks: l.view_marks,
    };
    if let Ok(mut v) = validator::Validator::load(validator::Validator::default_path().as_deref()) {
        v.validate(&mut shader.dxil.clone())
            .expect("DXIL validation failed");
    }

    // 513 blocks (block 0 null + 512 capacity blocks), 4096 entries each,
    // View::Memory at heap slot 0 (UAV at 1) and View::NullPair at slot 2.
    const BLOCKS: usize = 513;
    const ENTRIES: usize = 4096;
    let mut table = vec![0u32; ENTRIES];
    for _ in 0..BLOCKS * ENTRIES {
        table.extend([2, 0]);
    }
    let set_entry = |table: &mut Vec<u32>, block: usize, page: usize, start_lo: u32| {
        let at = ENTRIES + 2 * (block * ENTRIES + page);
        table[at..at + 2].copy_from_slice(&[0, start_lo]);
    };
    // 4 GiB boundary: 0x1_FFFF_0000 (l1 0x1F -> block 2) -> 0x2_0000_0000 (l1 0x20 -> block 1).
    table[0x1F] = 2;
    table[0x20] = 1;
    set_entry(&mut table, 2, 0xFFF, 0xFFFF_0000);
    set_entry(&mut table, 1, 0x000, 0xFFFF_0000);
    // 256 MiB / capacity edge: 0x1_0FFF_0000 (l1 0x10 -> block 512) -> 0x1_1000_0000 (l1 0x11 -> block 0, unmapped).
    table[0x10] = 512;
    set_entry(&mut table, 512, 0xFFF, 0x0FFF_0000);

    // 128 KiB copy spanning two 64 KiB pages; the store writes at +0xFFF8..+0x10008.
    let memory = vec![0u32; 32 * 1024];
    let bitmap_dwords = BLOCKS * ENTRIES / 32;
    let _one = super::valu_tests::ONE_AT_A_TIME
        .lock()
        .unwrap_or_else(|e| e.into_inner());
    let mut ran = 0;
    for (label, warp) in Host::test_backends() {
        let Ok(host) = Host::new(warp) else {
            eprintln!("no SM 6.6 {label} device; skipped");
            continue;
        };
        let (root, pso) = host.compute_pipeline(&shader).expect("pipeline");
        let cases: [(&str, u64, Vec<usize>); 3] = [
            (
                "4 GiB boundary",
                0x1_FFFF_FFF8,
                vec![4096, 2 * ENTRIES + 0xFFF],
            ),
            (
                "256 MiB boundary into unmapped region at block capacity",
                0x1_0FFF_FFF8,
                vec![0x11, 512 * ENTRIES + 0xFFF],
            ),
            (
                "same-page write inside 0x1_FFFF_0000",
                0x1_FFFF_FF00,
                vec![2 * ENTRIES + 0xFFF],
            ),
        ];
        for (case, addr, want_bits) in cases {
            let mut ud = vec![0u32; shader.user_data_dwords() as usize];
            for u in &shader.user_data {
                if u.root == RootType::Constants {
                    ud[u.index as usize] = (addr >> (32 * u.dword)) as u32;
                }
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
                                    data: table.clone(),
                                },
                            ),
                            (2, Root::Uav(bitmap_dwords as u32)),
                        ],
                        views: &[
                            View::Memory(&memory),
                            View::Pair,
                            View::NullPair,
                            View::Pair,
                        ],
                        samplers: &[],
                        groups: [1, 1, 1],
                    },
                )
                .unwrap_or_else(|e| panic!("{label} ({case}): {e}"));
            let off = ((addr as u32) & 0xFFFF) as usize / 4;
            assert_eq!(
                &out[0][off..off + 4],
                &[1, 2, 3, 4],
                "{label} ({case}): store data"
            );
            let marked: Vec<usize> = (0..out[1].len() * 32)
                .filter(|&b| out[1][b / 32] >> (b % 32) & 1 != 0)
                .collect();
            assert_eq!(marked, want_bits, "{label} ({case}): dirty bits");
        }
        ran += 1;
    }
    assert!(ran > 0, "neither hardware nor WARP SM 6.6 device available");
}

/// A compute program that stores through four marked views: one word and
/// one bit; one word, two bits, lanes < 40 in a branch; one word per lane,
/// `t % 4 == 0` in a branch; and one word per 32 lanes.
///
/// With `wave64` the group is one console wave (the console flow on a host
/// without 64-lane waves) and the 63 is a whole-wave OR reduction of the
/// lane ids. Otherwise two console waves of 32 branch per lane.
///
/// `per_lane` lowers with the leader election disabled (test-only), which
/// emits the original per-lane mark path. The bitmap starts with bits 5,
/// 11 and 320 set.
fn lower_coalescing_program(
    target: super::Target,
    wave64: bool,
    per_lane: bool,
) -> Result<lower::Lowered, String> {
    const fn store_offen(vdata: u8, vaddr: u8, srsrc: u8) -> [u32; 2] {
        [
            0xE070_1000,
            0x80 << 24 | ((srsrc as u32 >> 2) & 0x1F) << 16 | (vdata as u32) << 8 | vaddr as u32,
        ]
    }
    // s_or_b{32,64} exec, exec, s14 / s[14:15]; s_and_saveexec_b{32,64}
    // s14 / s[14:15], vcc; s_cbranch_execz past one store.
    let (or_exec, and_saveexec) = if wave64 {
        (0x88FE_0E7E, 0xBE8E_246A)
    } else {
        (0x887E_0E7E, 0xBE8E_3C6A)
    };
    const SKIP_STORE: u32 = 0xBF88_0002;
    let mut w = Vec::new();
    if wave64 {
        // The scan of `split_tests::reduction_scan` with v_or_b32: the OR
        // of the console wave's lane ids in s10.
        let v_or = 28 << 25;
        w.extend([
            0xBE80_25C1, // s_or_saveexec_b64 s[0:1], -1
            0xD501_0001,
            0x0002_0080, // v_cndmask_b32 v1, 0, v0, s[0:1]
        ]);
        for row_shr in [1, 2, 4, 8] {
            w.extend([v_or | 0x0002_02FA, 0xFF01_0001 | (0x110 + row_shr) << 8]); // v_or_b32 v1, v1, v1 row_shr
        }
        w.extend([
            0xD778_0002,
            0x0305_8301, // v_permlanex16_b32 v2, v1, -1, -1
            0x7E06_0280, // v_mov_b32 v3, 0
            0x7E06_02FA,
            0xAF00_E402,        // v_mov_b32 v3, v2 row_mask:0xa
            v_or | 0x0002_0701, // v_or_b32 v1, v1, v3
            0xD760_0009,
            0x0001_3F01,        // v_readlane_b32 s9, v1, 31
            0xBEFE_0380,        // s_mov_b32 exec_lo, 0
            0xBEFF_03C1,        // s_mov_b32 exec_hi, -1
            v_or | 0x0002_0209, // v_or_b32 v1, s9, v1
            0xD760_000A,
            0x0001_7F01, // v_readlane_b32 s10, v1, 63
            0xBEFE_0400, // s_mov_b64 exec, s[0:1]
            0x7E08_020A, // v_mov_b32 v4, s10
        ]);
    } else {
        w.push(v_mov_b32(4, 128 + 63));
    }
    w.push(s_mov_b32(5, ZERO as u8));
    for (k, sdata) in [8u8, 16, 20, 24].into_iter().enumerate() {
        w.extend(s_load(2, sdata, 4, 32 * k as u32));
    }
    w.extend([
        S_WAITCNT_ALL,
        0x3406_0082, // v_lshlrev_b32 v3, 2, v0
        0x4A10_0081, // v_add_nc_u32 v8, 1, v0
        0x340A_0095, // v_lshlrev_b32 v5, 21, v0
        0x2C0E_0085, // v_lshrrev_b32 v7, 5, v0
        0x340E_0E95, // v_lshlrev_b32 v7, 21, v7
        0x4A0E_0707, // v_add_nc_u32 v7, v7, v3
        0x360C_0083, // v_and_b32 v6, 3, v0
    ]);
    w.extend(store_offen(4, 3, 8));
    w.extend([0x7D88_00A8, and_saveexec, SKIP_STORE]); // v_cmp_gt_u32 vcc, 40, v0
    w.extend(store_offen(8, 3, 16));
    w.extend([or_exec, 0x7D84_0C80, and_saveexec, SKIP_STORE]); // v_cmp_eq_u32 vcc, 0, v6
    w.extend(store_offen(8, 5, 20));
    w.push(or_exec);
    w.extend(store_offen(8, 7, 24));
    w.push(S_ENDPGM);
    let param = |root, user_data, sgprs, slot| RootParam {
        root,
        table_offset_dwords: 0,
        sgprs,
        base_register: 0,
        user_data,
        slot,
    };
    let uav = |k: u32| UsedResource {
        kind: ResourceType::Uav,
        root: RootType::Table,
        table_offset_dwords: 8 * k,
        lower_bound: k,
        stride: 0,
        dwords: 4,
        user_data: 4,
        slot: 1,
    };
    let native = NativeProgram {
        stage: HwStage::Cs,
        wave64,
        vgprs: 10,
        layout: vec![
            param(RootType::Constants, 0, 4, 0),
            param(RootType::Table, 4, 1, 1),
        ],
        used: (0..4).map(uav).collect(),
        io: StageIo::Cs(ComputeIo {
            rsrc2: 5 << 1,
            threads: [64, 1, 1],
        }),
        code: w.iter().flat_map(|w| w.to_le_bytes()).collect(),
    };
    let p = ir::build(&native.code, native.wave64, native.vgprs).map_err(|e| e.to_string())?;
    let st = structure::structure(&p).map_err(|e| e.to_string())?;
    let res =
        resource::track(&p, &native.layout, native.stage, Some(&st)).map_err(|e| e.to_string())?;
    let io = io::raise(&p, &native).map_err(|e| e.to_string())?;
    let lower_in = |st, console_cf| {
        let parts = lower::Parts {
            native: &native,
            p: &p,
            st,
            io: &io,
            res: &res,
            inputs: Vec::new(),
            outputs: Vec::new(),
            prims: Vec::new(),
            mesh_stride: 0,
            target,
            console_cf,
            per_lane_only: false,
            view_marks: true,
        };
        if per_lane {
            lower::per_lane_marks(|| lower::lower(&parts))
        } else {
            lower::lower(&parts)
        }
    };
    // As `emit_with`: the console flow where the per-lane form asks for it.
    match lower_in(&st, false) {
        Err(crate::ShaderError::Unsupported { what, .. }) if what == lower::CONSOLE_FLOW => {
            let console = structure::structure_console(&p).map_err(|e| e.to_string())?;
            lower_in(&console, true)
        }
        l => l,
    }
    .map_err(|e| e.to_string())
}

/// The leader-based coalescing of PR #2958 sets exactly the bits the
/// per-lane path sets, and leaves the program's own results alone, on the
/// hardware adapter and WARP: [`lower_coalescing_program`] run both ways and
/// compared word for word, and against the bits and values the console
/// program means.
#[test]
fn wave_coalesced_marks_equal_per_lane_marks_on_the_gpu() {
    // Per heap index: the first bit of its pages and the byte offset of
    // its first byte in its first page.
    let marks = vec![3, 0, 10, 0xFFF0, 64, 0, 4096, 0];
    let mut preset = vec![0u32; 256];
    for bit in [5usize, 11, 320] {
        preset[bit / 32] |= 1 << (bit % 32);
    }
    let mut want_bits: Vec<usize> = vec![3, 5, 10, 11, 4096, 4128];
    want_bits.extend((0..64).step_by(4).map(|t| 64 + 32 * t));
    want_bits.sort_unstable();
    let want_out: [Vec<u32>; 4] = [
        vec![63; 64],
        (0..64).map(|t| if t < 40 { t + 1 } else { 0 }).collect(),
        (0..64).map(|t| u32::from(t == 0)).collect(),
        (0..64).map(|t| if t < 32 { t + 1 } else { 0 }).collect(),
    ];
    let _one = super::valu_tests::ONE_AT_A_TIME
        .lock()
        .unwrap_or_else(|e| e.into_inner());
    let mut ran = 0;
    for (label, warp) in [("hardware", false), ("warp", true)] {
        let Ok(host) = Host::new(warp) else {
            eprintln!("no SM 6.6 {label} device; skipped");
            continue;
        };
        if !host.target().runs(32) {
            eprintln!("{label}: no 32-lane waves; skipped");
            continue;
        }
        let mut results = Vec::new();
        for (wave64, per_lane) in [(false, true), (false, false), (true, true), (true, false)] {
            let label = &format!("{label}, wave64={wave64}, coalesced={}", !per_lane);
            let l = lower_coalescing_program(host.target(), wave64, per_lane)
                .unwrap_or_else(|e| panic!("{label}: {e}"));
            let shader = HostShader {
                stage: HwStage::Cs,
                wave_size: l.wave_size,
                dxil: l.dxil,
                user_data: l.user_data,
                samplers: l.samplers,
                static_samplers: Vec::new(),
                root_cbvs: l.root_cbvs,
                root_srvs: l.root_srvs,
                root_uavs: l.root_uavs,
                memory: l.memory,
                memory_writes: l.memory_writes,
                memory_stores: l.memory_stores,
                view_addresses: l.view_addresses,
                heap: l.heap,
                heap_writes: l.heap_writes,
                view_marks: l.view_marks,
            };
            if let Ok(mut v) =
                validator::Validator::load(validator::Validator::default_path().as_deref())
            {
                v.validate(&mut shader.dxil.clone())
                    .unwrap_or_else(|e| panic!("{label}: DXIL validation: {e}"));
            }
            let (root, pso) = host
                .compute_pipeline(&shader)
                .unwrap_or_else(|e| panic!("{label}: pipeline: {e}"));
            let ud = vec![0u32; shader.user_data_dwords() as usize];
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
                                    data: marks.clone(),
                                },
                            ),
                            (2, Root::UavData(preset.clone())),
                        ],
                        views: &[
                            View::RawUav(64),
                            View::RawUav(64),
                            View::RawUav(64),
                            View::RawUav(64),
                        ],
                        samplers: &[],
                        groups: [1, 1, 1],
                    },
                )
                .unwrap_or_else(|e| panic!("{label}: {e}"));
            for (k, want) in want_out.iter().enumerate() {
                assert_eq!(&out[k], want, "{label}: view {k}");
            }
            let bits = &out[4];
            let marked: Vec<usize> = (0..bits.len() * 32)
                .filter(|&b| bits[b / 32] >> (b % 32) & 1 != 0)
                .collect();
            assert_eq!(marked, want_bits, "{label}: marked pages");
            results.push(out);
        }
        assert_eq!(
            results[0], results[1],
            "{label}: per-lane and coalesced runs differ (wave32)"
        );
        assert_eq!(
            results[2], results[3],
            "{label}: per-lane and coalesced runs differ (wave64)"
        );
        eprintln!("{label}: coalesced marks and results equal the per-lane ones");
        ran += 1;
    }
    assert!(ran > 0, "neither hardware nor WARP SM 6.6 device available");
}

const OFFEN: u32 = 1 << 12;
const IDXEN: u32 = 1 << 13;

/// MUBUF `op vdata, vaddr, s[srsrc:srsrc+3], soffset`; `low` holds the
/// immediate offset and [`OFFEN`] or [`IDXEN`].
const fn mubuf(op: u32, vdata: u8, vaddr: u8, srsrc: u8, soffset: u8, low: u32) -> [u32; 2] {
    [
        0x38 << 26 | op << 18 | low,
        (soffset as u32) << 24 | (srsrc as u32 >> 2) << 16 | (vdata as u32) << 8 | vaddr as u32,
    ]
}

/// The table [`code_table_program`] appends to its code.
fn code_table() -> Vec<u32> {
    (1..=64u32).map(|k| k.wrapping_mul(0x9E37_79B9)).collect()
}

/// Record limits of [`code_table_program`]'s two descriptors: bytes of the
/// raw one (lane 29's dword past it, its byte 1 not), elements of the
/// indexed one.
const RAW_RECORDS: u32 = 4 * 29 + 2;
const INDEXED_RECORDS: u32 = 21;

/// A wave32 program of 32 lanes reading a table after its code through two
/// descriptors built around `s_getpc_b64`: a raw one (`OOB_SELECT` 3) read
/// per lane at `lane * 4` as a dword, its byte 1, and two dwords 4 bytes
/// further through `soffset`, and a signed halfword at byte 3 (its high
/// byte in the next dword); a structured one (`OOB_SELECT` 1, stride 8)
/// read by `idxen` at the lane's element plus 4. Lane `l` stores the six
/// dwords at `32 * l` of the UAV at heap index 0.
fn code_table_program() -> Vec<u32> {
    let mut w = vec![s_mov_b32(5, ZERO as u8)];
    w.extend(s_load(2, 8, 4, 0)); // s[8:11]: the result UAV
    // s[12:15] and s[16:19]: the table's descriptors (the literals patched
    // below with the table's distance from each s_getpc_b64's next word).
    let raw = w.len();
    w.extend([0xBE8C_1F00, 0x800C_FF0C, 0, 0x820D_800D]); // s_getpc_b64 s[12:13]; s_add_u32; s_addc_u32
    w.extend([0xBE8E_03FF, RAW_RECORDS, 0xBE8F_03FF, 0x3101_4FAC]); // s_mov_b32 s14, s15
    let indexed = w.len();
    w.extend([0xBE90_1F00, 0x8010_FF10, 0, 0x8211_8011]); // s_getpc_b64 s[16:17]; s_add_u32; s_addc_u32
    w.extend([0x8811_FF11, 8 << 16]); // s_or_b32 s17, s17, stride 8
    w.extend([0xBE92_03FF, INDEXED_RECORDS, 0xBE93_03FF, 0x1101_4FAC]); // s_mov_b32 s18, s19
    w.push(0xBE94_0384); // s_mov_b32 s20, 4
    w.push(0x3402_0082); // v_lshlrev_b32 v1, 2, v0
    w.push(0x3402_0085 | 7 << 17); // v_lshlrev_b32 v7, 5, v0
    w.extend(mubuf(0x0C, 2, 1, 12, 0x80, OFFEN));
    w.extend(mubuf(0x08, 3, 1, 12, 0x80, OFFEN | 1));
    w.extend(mubuf(0x0C, 4, 0, 16, 0x80, IDXEN | 4));
    w.extend(mubuf(0x0D, 5, 1, 12, 20, OFFEN));
    w.extend(mubuf(0x0B, 8, 1, 12, 0x80, OFFEN | 3));
    w.push(S_WAITCNT_ALL);
    for (k, v) in [2u8, 3, 4, 5, 6, 8].into_iter().enumerate() {
        w.extend(mubuf(0x1C, v, 7, 8, 0x80, OFFEN | (4 * k as u32)));
    }
    w.push(S_ENDPGM);
    let table = 4 * w.len() as u32;
    for at in [raw, indexed] {
        w[at + 2] = table - 4 * (at as u32 + 1);
    }
    w.extend(code_table());
    w
}

/// What lane `l` of [`code_table_program`] stores: out of range, a load
/// reads 0 (a raw load by dword, each checked as `offset + soffset +
/// size > records`; an indexed one by `index >= records`).
fn code_table_model(l: u32) -> [u32; 6] {
    let t = code_table();
    let raw = |byte: u32, size: u32, x: u32| if byte + size > RAW_RECORDS { 0 } else { x };
    [
        raw(4 * l, 4, t[l as usize]),
        raw(4 * l + 1, 1, t[l as usize] >> 8 & 0xFF),
        if l >= INDEXED_RECORDS {
            0
        } else {
            t[2 * l as usize + 1]
        },
        raw(4 * l + 4, 4, t[l as usize + 1]),
        raw(4 * l + 8, 4, t[l as usize + 2]),
        raw(
            4 * l + 3,
            2,
            (t[l as usize] >> 24 | (t[l as usize + 1] & 0xFF) << 8) as u16 as i16 as u32,
        ),
    ]
}

fn lower_code_table_program(words: &[u32]) -> Result<HostShader, String> {
    let native = NativeProgram {
        stage: HwStage::Cs,
        wave64: false,
        vgprs: 9,
        layout: vec![RootParam {
            root: RootType::Table,
            table_offset_dwords: 0,
            sgprs: 1,
            base_register: 0,
            user_data: 4,
            slot: 1,
        }],
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
            threads: [32, 1, 1],
        }),
        code: words.iter().flat_map(|w| w.to_le_bytes()).collect(),
    };
    let e = |e: crate::ShaderError| e.to_string();
    let p = ir::build(&native.code, native.wave64, native.vgprs).map_err(e)?;
    let st = structure::structure(&p).map_err(e)?;
    let res = resource::track(&p, &native.layout, native.stage, Some(&st)).map_err(e)?;
    let io = io::raise(&p, &native).map_err(e)?;
    let l = lower::lower(&lower::Parts {
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
        console_cf: false,
        per_lane_only: false,
        view_marks: false,
    })
    .map_err(e)?;
    Ok(HostShader {
        stage: HwStage::Cs,
        wave_size: None,
        dxil: l.dxil,
        user_data: l.user_data,
        samplers: l.samplers,
        static_samplers: Vec::new(),
        root_cbvs: l.root_cbvs,
        root_srvs: l.root_srvs,
        root_uavs: l.root_uavs,
        memory: l.memory,
        memory_writes: l.memory_writes,
        memory_stores: l.memory_stores,
        view_addresses: l.view_addresses,
        heap: l.heap,
        heap_writes: l.heap_writes,
        view_marks: l.view_marks,
    })
}

/// [`code_table_program`] decodes as written and lowers.
#[test]
fn code_table_buffer_loads_lower() {
    let words = code_table_program();
    let code: Vec<u8> = words.iter().flat_map(|w| w.to_le_bytes()).collect();
    let text: Vec<String> = crate::gfx10::decode(&code)
        .expect("the program decodes")
        .iter()
        .map(|i| i.text(false))
        .collect();
    let table = 4 * (words.len() - 64);
    let want = [
        "s_getpc_b64 s[12:13]".to_owned(),
        format!("s_add_u32 s12, s12, lit({:#010x})", table - 16),
        "s_addc_u32 s13, s13, 0".to_owned(),
        format!("s_mov_b32 s14, lit({RAW_RECORDS:#010x})"),
        "s_mov_b32 s15, lit(0x31014fac)".to_owned(),
        "s_getpc_b64 s[16:17]".to_owned(),
        format!("s_add_u32 s16, s16, lit({:#010x})", table - 48),
        "s_addc_u32 s17, s17, 0".to_owned(),
        "s_or_b32 s17, s17, lit(0x00080000)".to_owned(),
        format!("s_mov_b32 s18, lit({INDEXED_RECORDS:#010x})"),
        "s_mov_b32 s19, lit(0x11014fac)".to_owned(),
        "s_mov_b32 s20, 4".to_owned(),
        "v_lshlrev_b32 v1, 2, v0".to_owned(),
        "v_lshlrev_b32 v7, 5, v0".to_owned(),
        "buffer_load_dword v2, v1, s[12:15], 0 offen".to_owned(),
        "buffer_load_ubyte v3, v1, s[12:15], 0 offen offset:1".to_owned(),
        "buffer_load_dword v4, v0, s[16:19], 0 idxen offset:4".to_owned(),
        "buffer_load_dwordx2 v[5:6], v1, s[12:15], s20 offen".to_owned(),
        "buffer_load_sshort v8, v1, s[12:15], 0 offen offset:3".to_owned(),
    ];
    assert_eq!(text[2..2 + want.len()], want[..]);
    let shader = lower_code_table_program(&words).expect("lower");
    if let Ok(mut v) = validator::Validator::load(validator::Validator::default_path().as_deref()) {
        v.validate(&mut shader.dxil.clone())
            .expect("DXIL validation failed");
    }
    // OOB_SELECT 0 (offsets checked against the stride) is refused.
    let mut checked = words.clone();
    let at = words
        .iter()
        .position(|&w| w == 0x3101_4FAC)
        .expect("dword 3");
    checked[at] = 0x0101_4FAC;
    let Err(e) = lower_code_table_program(&checked) else {
        panic!("OOB_SELECT 0 lowered");
    };
    assert!(e.contains("checking offsets against its stride"), "{e}");
}

/// [`code_table_program`] on the host GPU and WARP matches
/// [`code_table_model`] in every lane.
#[test]
fn code_table_buffer_loads_read_zeros_out_of_range() {
    let shader = lower_code_table_program(&code_table_program()).expect("lower");
    let want: Vec<u32> = (0..32)
        .flat_map(|l| {
            let mut row = code_table_model(l).to_vec();
            row.resize(8, 0);
            row
        })
        .collect();
    let _one = super::valu_tests::ONE_AT_A_TIME
        .lock()
        .unwrap_or_else(|e| e.into_inner());
    let mut ran = 0;
    for (label, warp) in [("hardware", false), ("warp", true)] {
        let Ok(host) = Host::new(warp) else {
            eprintln!("no SM 6.6 {label} device; skipped");
            continue;
        };
        let (root, pso) = host.compute_pipeline(&shader).expect("pipeline");
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
                    views: &[View::RawUav(32 * 8)],
                    samplers: &[],
                    groups: [1, 1, 1],
                },
            )
            .unwrap_or_else(|e| panic!("{label}: {e}"));
        assert_eq!(out[0], want, "{label}");
        ran += 1;
    }
    assert!(ran > 0, "neither hardware nor WARP SM 6.6 device available");
}

enum BcConst {
    Int(i64),
    Agg(Vec<u32>),
}
enum BcInst {
    Bin(u64, u32, u32),
    Cast(u64, u32),
    Cmp(u64, u32, u32),
    Extract(u32, u64),
    Call(String, Vec<u32>),
    Other,
}

/// Decoded instructions and conditional branches of `main` from a DXIL container.
struct Bc {
    consts: std::collections::HashMap<u32, BcConst>,
    insts: std::collections::HashMap<u32, (usize, BcInst)>,
    cond_brs: Vec<(usize, usize, u32)>,
}

impl Bc {
    fn parse(dxil: &[u8]) -> Self {
        use std::collections::HashMap;
        let c = crate::dxbc::Container::parse(dxil).expect("dxbc");
        let bc = &c.part(crate::dxbc::FourCC::DXIL).expect("DXIL").data[24..];
        assert_eq!(&bc[..4], b"BC\xC0\xDE");
        let mut pos = 32usize;
        let read = |p: &mut usize, n: u32| -> u64 {
            (0..n).fold(0u64, |v, i| {
                let b = (bc[*p / 8] >> (*p % 8)) & 1;
                *p += 1;
                v | u64::from(b) << i
            })
        };
        let vbr = |p: &mut usize, w: u32| -> u64 {
            let (mut v, mut sh) = (0u64, 0u32);
            loop {
                let x = read(p, w);
                v |= (x & ((1 << (w - 1)) - 1)) << sh;
                if x >> (w - 1) == 0 {
                    return v;
                }
                sh += w - 1;
            }
        };
        let (mut stack, mut void_ty, mut ty_void) =
            (Vec::<(u64, u32)>::new(), None, HashMap::<u32, bool>::new());
        let (mut ty_n, mut globals, mut funcs, mut const_n, mut next_id, mut blk) =
            (0u32, 0u32, 0u32, 0u32, 0u32, 0usize);
        let (mut consts, mut names, mut insts, mut cond_brs) = (
            HashMap::new(),
            HashMap::<u32, String>::new(),
            HashMap::new(),
            Vec::new(),
        );
        while pos < bc.len() * 8 {
            match read(&mut pos, stack.last().map_or(2, |s| s.1)) {
                0 => {
                    pos = pos.next_multiple_of(32);
                    stack.pop();
                    if stack.is_empty() {
                        break;
                    }
                }
                1 => {
                    let (id, nw) = (vbr(&mut pos, 8), vbr(&mut pos, 4) as u32);
                    pos = pos.next_multiple_of(32) + 32;
                    if id == 12 {
                        (next_id, blk) = (globals + funcs + const_n, 0);
                    }
                    stack.push((id, nw));
                }
                3 => {
                    let (code, n) = (vbr(&mut pos, 6), vbr(&mut pos, 6) as usize);
                    let ops: Vec<u64> = (0..n).map(|_| vbr(&mut pos, 6)).collect();
                    match stack.last().map(|s| s.0) {
                        Some(17) if code != 1 && code != 19 => {
                            if code == 2 {
                                void_ty = Some(ty_n);
                            } else if code == 21 {
                                ty_void.insert(ty_n, Some(ops[1] as u32) == void_ty);
                            }
                            ty_n += 1;
                        }
                        Some(8) if code == 7 || code == 8 => {
                            if code == 7 {
                                globals += 1
                            } else {
                                funcs += 1
                            }
                        }
                        Some(11) if code != 1 => {
                            let id = globals + funcs + const_n;
                            if code == 4 {
                                let m = (ops[0] >> 1) as i64;
                                consts.insert(
                                    id,
                                    BcConst::Int(if ops[0] & 1 == 0 {
                                        m
                                    } else {
                                        m.wrapping_neg()
                                    }),
                                );
                            } else if code == 7 {
                                consts.insert(
                                    id,
                                    BcConst::Agg(ops.iter().map(|&o| o as u32).collect()),
                                );
                            }
                            const_n += 1;
                        }
                        Some(14) if code == 1 => {
                            names.insert(
                                ops[0] as u32,
                                ops[1..].iter().map(|&c| c as u8 as char).collect(),
                            );
                        }
                        Some(12) if code != 1 => {
                            let rel = |r: u64| next_id - r as u32;
                            match code {
                                10 | 11 => {
                                    if code == 11 && ops.len() == 3 {
                                        cond_brs.push((blk, ops[0] as usize, rel(ops[2])));
                                    }
                                    blk += 1;
                                }
                                44 => {}
                                34 if ty_void.get(&(ops[2] as u32)).copied().unwrap_or(false) => {}
                                _ => {
                                    let inst = match code {
                                        2 => BcInst::Bin(ops[2], rel(ops[0]), rel(ops[1])),
                                        3 => BcInst::Cast(ops[2], rel(ops[0])),
                                        26 => BcInst::Extract(rel(ops[0]), ops[1]),
                                        28 => BcInst::Cmp(ops[2], rel(ops[0]), rel(ops[1])),
                                        34 => BcInst::Call(
                                            names.get(&rel(ops[3])).cloned().unwrap_or_default(),
                                            ops[4..].iter().map(|&r| rel(r)).collect(),
                                        ),
                                        _ => BcInst::Other,
                                    };
                                    insts.insert(next_id, (blk, inst));
                                    next_id += 1;
                                }
                            }
                        }
                        _ => {}
                    }
                }
                other => panic!("unexpected bitcode abbrev {other}"),
            }
        }
        Self {
            consts,
            insts,
            cond_brs,
        }
    }

    fn int(&self, id: u32) -> Option<i64> {
        match self.consts.get(&id) {
            Some(BcConst::Int(v)) => Some(*v),
            _ => None,
        }
    }

    fn binding(&self, h: u32) -> Option<(i64, i64, i64)> {
        let (_, BcInst::Call(name, args)) = self.insts.get(&h)? else {
            return None;
        };
        let inner = if name == "dx.op.annotateHandle" {
            args[1]
        } else {
            h
        };
        let (_, BcInst::Call(name, args)) = self.insts.get(&inner)? else {
            return None;
        };
        let BcConst::Agg(f) =
            (name == "dx.op.createHandleFromBinding").then(|| self.consts.get(&args[1]))??
        else {
            return None;
        };
        Some((self.int(f[0])?, self.int(f[2])?, self.int(f[3])?))
    }

    fn calls_on(&self, op: &str, bind: (i64, i64, i64)) -> Vec<(usize, &[u32])> {
        self.insts
            .values()
            .filter_map(|(b, i)| match i {
                BcInst::Call(n, a) if n == op && self.binding(a[1]) == Some(bind) => {
                    Some((*b, a.as_slice()))
                }
                _ => None,
            })
            .collect()
    }

    fn assert_guarded_dirty_or(&self, blk: usize, args: &[u32], bitmap: (i64, i64, i64)) -> usize {
        assert_eq!(
            (self.binding(args[1]), self.int(args[2])),
            (Some(bitmap), Some(2))
        );
        let (word, bit) = (args[3], args[6]);
        let &(test_blk, _, cond) = self
            .cond_brs
            .iter()
            .find(|&&(_, then, _)| then == blk)
            .expect("dirty atomic must be branched");
        let (_, BcInst::Cmp(32, a, b)) = self.insts[&cond] else {
            panic!("guard must be icmp eq 0")
        };
        let set = if self.int(b) == Some(0) {
            a
        } else {
            assert_eq!(self.int(a), Some(0));
            b
        };
        let (_, BcInst::Bin(10, x, y)) = self.insts[&set] else {
            panic!("test must mask word with bit")
        };
        let old = if y == bit {
            x
        } else {
            assert_eq!(x, bit);
            y
        };
        let (_, BcInst::Cast(11, ext)) = self.insts[&old] else {
            panic!("expected bitcast")
        };
        let (_, BcInst::Extract(load, 0)) = self.insts[&ext] else {
            panic!("expected extract 0")
        };
        let (load_blk, BcInst::Call(ref name, ref la)) = self.insts[&load] else {
            panic!("expected rawBufferLoad")
        };
        assert_eq!(
            (load_blk, name.as_str(), self.binding(la[1]), la[2]),
            (test_blk, "dx.op.rawBufferLoad.f32", Some(bitmap), word)
        );
        assert_eq!(
            (
                self.int(la[4]),
                self.int(la[5]),
                self.insts[&word].0,
                self.insts[&bit].0
            ),
            (Some(1), Some(4), test_blk, test_blk)
        );
        test_blk
    }

    fn gated_by_ne(&self, blk: usize, word: u32) -> bool {
        let Some((_, BcInst::Bin(7, sh, _))) = self.insts.get(&word) else {
            return false;
        };
        let Some((_, BcInst::Bin(8, n, _))) = self.insts.get(sh) else {
            return false;
        };
        self.cond_brs.iter().any(|&(_, t, c)| t == blk && matches!(self.insts.get(&c), Some((_, BcInst::Cmp(33, a, b))) if a == n || b == n))
    }

    fn verify_dirty_marks(
        &self,
        bitmap: (i64, i64, i64),
        want_atomics: usize,
        want_ne: Option<usize>,
    ) {
        let atomics = self.calls_on("dx.op.atomicBinOp.i32", bitmap);
        assert_eq!(atomics.len(), want_atomics);
        let ne = atomics
            .iter()
            .filter(|&&(b, a)| self.gated_by_ne(self.assert_guarded_dirty_or(b, a, bitmap), a[3]))
            .count();
        if let Some(expected_ne) = want_ne {
            assert_eq!(
                ne, expected_ne,
                "2nd marks must check n != bits[0] before loading bitmap"
            );
        }
    }
}

/// Across 13 title-memory accesses (`program`), `memory_null` is loaded once in
/// block 0, `(view, start)` loads coalesce via `raw_dword2`, every dirty-mark
/// atomic is guarded by load + bit-test, and `WAVE_OPS` is not set.
#[test]
fn memory_entry_caches_null_coalesces_loads_and_guards_dirty_atomics() {
    let shader = program();
    let sfi0 = crate::dxbc::Container::parse(&shader.dxil)
        .expect("dxbc")
        .part(crate::dxbc::FourCC::SFI0)
        .expect("SFI0")
        .data;
    assert_eq!(
        u64::from_le_bytes(sfi0.try_into().unwrap()) & 0x4000,
        0,
        "no WAVE_OPS"
    );
    let bc = Bc::parse(&shader.dxil);
    let pt_loads = bc.calls_on("dx.op.rawBufferLoad.f32", (0, 2, 0));
    assert_eq!(
        pt_loads
            .iter()
            .filter(|(_, a)| bc.int(a[2]) == Some(4096 * 4))
            .map(|(b, _)| *b)
            .collect::<Vec<_>>(),
        [0]
    );
    assert_eq!(
        pt_loads
            .iter()
            .filter(|(_, a)| (bc.int(a[4]), bc.int(a[5])) == (Some(3), Some(8)))
            .count(),
        13
    );
    // 1 hoisted memory_null + 13 primary memory_page L1 loads + 4 straddling-write L1 loads.
    assert_eq!(
        pt_loads
            .iter()
            .filter(|(_, a)| bc.int(a[4]) == Some(1))
            .count(),
        18
    );
    bc.verify_dirty_marks((0, 2, 1), 17, None);
}

/// Single-byte writes omit the 2nd mark statically in both `Lower::memory` and
/// `Lower::mark_view_write`; `mark_view_write` coalesces `(base, phase)` and
/// checks `n != bits[0]` before loading the bitmap; and `memory_null` hoists
/// to entry block 0 even from a non-entry branch.
#[test]
fn dirty_mark_width_and_branch_hoisting() {
    let mut val = validator::Validator::load(validator::Validator::default_path().as_deref()).ok();
    for (view_mark, bytes, in_branch) in [
        (false, 1, false),
        (false, 4, true),
        (true, 1, false),
        (true, 4, false),
    ] {
        let dxil = lower::synthetic_mark_dxil(view_mark, bytes, in_branch);
        if let Some(v) = val.as_mut() {
            v.validate(&mut dxil.clone())
                .expect("DXIL validation failed");
        }
        let bc = Bc::parse(&dxil);
        let (table, bitmap) = if view_mark {
            ((2, 2, 0), (1, 2, 1))
        } else {
            ((0, 2, 0), (0, 2, 1))
        };
        let t_loads = bc.calls_on("dx.op.rawBufferLoad.f32", table);
        assert_eq!(
            t_loads
                .iter()
                .filter(|(_, a)| (bc.int(a[4]), bc.int(a[5])) == (Some(3), Some(8)))
                .count(),
            1
        );
        if view_mark {
            assert_eq!(t_loads.len(), 1);
        } else {
            assert_eq!(
                t_loads
                    .iter()
                    .filter(|(_, a)| bc.int(a[2]) == Some(4096 * 4))
                    .map(|(b, _)| *b)
                    .collect::<Vec<_>>(),
                [0]
            );
        }
        let want = if bytes <= 1 { 1 } else { 2 };
        assert_eq!(bc.calls_on("dx.op.rawBufferLoad.f32", bitmap).len(), want);
        bc.verify_dirty_marks(bitmap, want, view_mark.then_some(usize::from(bytes > 1)));
    }
}
