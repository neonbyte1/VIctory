//! End-to-end tests of descriptors whose emission needs more than one
//! binding: a buffer read by element through the resource heap (no
//! used-resource entry, so no stride; the view-stride table gives it), one
//! of two immediate samplers chosen by uniform control flow, and a buffer
//! sized and read through one descriptor load (a handle each). Synthetic
//! GFX10 programs lowered to DXIL, validated by `dxil.dll` when present,
//! and run on the host GPU and WARP.

use windows::Win32::Graphics::Direct3D12::D3D12_SAMPLER_DESC;

use windows::Win32::Graphics::Dxgi::Common::{
    DXGI_FORMAT, DXGI_FORMAT_R16_FLOAT, DXGI_FORMAT_R16_UINT, DXGI_FORMAT_R16G16B16A16_FLOAT,
    DXGI_FORMAT_R16G16B16A16_UINT,
};

use super::host::{Dispatch, Host, Root, Stride, View};
use super::{HostShader, StaticSampler, VIEW_INTEGER, VIEW_STRIDES, lower, validator};
use crate::native::{
    ComputeIo, HwStage, NativeProgram, ResourceType, RootParam, RootType, StageIo, UsedResource,
};
use crate::{io, ir, resource, structure};

/// User data: s0 the heap index of a structured buffer (a root constant),
/// s1 the resource heap, s2 a table with a raw UAV. Each lane stores dword
/// 1 of element `lane` to dword `lane` of the UAV.
const HEAP_ELEMENT_WAVE32: &[u32] = &[
    0x8F04_8500, // s_lshl_b32 s4, s0, 5
    0xBE86_0301, // s_mov_b32 s6, s1
    0xBE87_0380, // s_mov_b32 s7, 0
    0xF408_0203,
    0x0800_0000, // s_load_dwordx4 s[8:11], s[6:7], s4
    0xBE8C_0302, // s_mov_b32 s12, s2
    0xBE8D_0380, // s_mov_b32 s13, 0
    0xF408_0406,
    0xFA00_0000, // s_load_dwordx4 s[16:19], s[12:13], 0x0
    0x3402_0082, // v_lshlrev_b32 v1, 2, v0
    0xBF8C_C07F, // s_waitcnt lgkmcnt(0)
    0xE030_2004,
    0x8002_0200, // buffer_load_dword v2, v0, s[8:11], 0 idxen offset:4
    0xBF8C_3F70, // s_waitcnt vmcnt(0)
    0xE070_1000,
    0x8004_0201, // buffer_store_dword v2, v1, s[16:19], 0 offen
    0xBF81_0000, // s_endpgm
];

/// User data: s0 a table with a 2x1 texture, s1 a root constant, s2 a table
/// with a raw UAV. The root constant picks the immediate sampler by a
/// branch, clamping (non-zero) or wrapping (zero); the program samples the
/// texture at u = 1.25 and stores red to dword 0 of the UAV.
const CHOSEN_SAMPLER_WAVE32: &[u32] = &[
    0xBE84_0300, // s_mov_b32 s4, s0
    0xBE85_0380, // s_mov_b32 s5, 0
    0xF40C_0202,
    0xFA00_0000, // s_load_dwordx8 s[8:15], s[4:5], 0x0
    0xBE86_0302, // s_mov_b32 s6, s2
    0xBE87_0380, // s_mov_b32 s7, 0
    0xF408_0403,
    0xFA00_0000, // s_load_dwordx4 s[16:19], s[6:7], 0x0
    0xBF06_8001, // s_cmp_eq_u32 s1, 0
    0xBF85_0006, // s_cbranch_scc1 -> 0x40
    0xBE94_03FF,
    0x0000_0092, // s_mov_b32 s20, 0x92 (clamp x, y, z)
    0xBE95_0380, // s_mov_b32 s21, 0
    0xBE96_0380, // s_mov_b32 s22, 0
    0xBE97_0380, // s_mov_b32 s23, 0
    0xBF82_0004, // s_branch -> 0x50
    0xBE94_0380, // 0x40: s_mov_b32 s20, 0 (wrap)
    0xBE95_0380, // s_mov_b32 s21, 0
    0xBE96_0380, // s_mov_b32 s22, 0
    0xBE97_0380, // s_mov_b32 s23, 0
    0x7E00_02FF,
    0x3FA0_0000, // 0x50: v_mov_b32 v0, 1.25
    0x7E02_02FF,
    0x3F00_0000, // v_mov_b32 v1, 0.5
    0xBF8C_C07F, // s_waitcnt lgkmcnt(0)
    0xF09C_0108,
    0x00A2_0200, // image_sample_lz v2, v[0:1], s[8:15], s[20:23] dmask:0x1 dim:2D
    0xBF8C_3F70, // s_waitcnt vmcnt(0)
    0x7E06_0280, // v_mov_b32 v3, 0
    0xE070_1000,
    0x8004_0203, // buffer_store_dword v2, v3, s[16:19], 0 offen
    0xBF81_0000, // s_endpgm
];

/// User data: s0 the heap index of a constant buffer (a root constant), s1
/// the resource heap, s2 a table with a raw UAV. Each lane stores dword 2
/// of the constant buffer (a scalar buffer load) to dword `lane` of the UAV.
const HEAP_CONSTANT_WAVE32: &[u32] = &[
    0x8F04_8500, // s_lshl_b32 s4, s0, 5
    0xBE86_0301, // s_mov_b32 s6, s1
    0xBE87_0380, // s_mov_b32 s7, 0
    0xF408_0203,
    0x0800_0000, // s_load_dwordx4 s[8:11], s[6:7], s4
    0xBE8C_0302, // s_mov_b32 s12, s2
    0xBE8D_0380, // s_mov_b32 s13, 0
    0xF408_0406,
    0xFA00_0000, // s_load_dwordx4 s[16:19], s[12:13], 0x0
    0xBF8C_C07F, // s_waitcnt lgkmcnt(0)
    0xF420_0504,
    0xFA00_0008, // s_buffer_load_dword s20, s[8:11], 0x8
    0xBF8C_C07F, // s_waitcnt lgkmcnt(0)
    0x7E02_0214, // v_mov_b32 v1, s20
    0x3404_0082, // v_lshlrev_b32 v2, 2, v0
    0xE070_1000,
    0x8004_0102, // buffer_store_dword v1, v2, s[16:19], 0 offen
    0xBF81_0000, // s_endpgm
];

/// User data: s0 a table with a raw SRV, s2 a table with a raw UAV. Each
/// lane loads the half at byte `2 × lane` and stores it to dword `lane`.
const HALF_LOAD_WAVE32: &[u32] = &[
    0xBE84_0300, // s_mov_b32 s4, s0
    0xBE85_0380, // s_mov_b32 s5, 0
    0xF408_0202,
    0xFA00_0000, // s_load_dwordx4 s[8:11], s[4:5], 0x0
    0xBE8C_0302, // s_mov_b32 s12, s2
    0xBE8D_0380, // s_mov_b32 s13, 0
    0xF408_0406,
    0xFA00_0000, // s_load_dwordx4 s[16:19], s[12:13], 0x0
    0x3402_0081, // v_lshlrev_b32 v1, 1, v0
    0xBF8C_C07F, // s_waitcnt lgkmcnt(0)
    0xE028_1000,
    0x8002_0201, // buffer_load_ushort v2, v1, s[8:11], 0 offen
    0xBF8C_3F70, // s_waitcnt vmcnt(0)
    0x3406_0082, // v_lshlrev_b32 v3, 2, v0
    0xE070_1000,
    0x8004_0203, // buffer_store_dword v2, v3, s[16:19], 0 offen
    0xBF81_0000, // s_endpgm
];

/// User data: s0 a table with a raw SRV, s2 a table with a raw UAV. Each
/// lane loads dword `lane` and stores it to dword `lane`, and stores the
/// SRV's record count (its descriptor's dword 2, read as a number) to dword
/// `32 + lane`: one descriptor load serves the loads and the size.
const RAW_SIZE_AND_LOAD_WAVE32: &[u32] = &[
    0xBE84_0300, // s_mov_b32 s4, s0
    0xBE85_0380, // s_mov_b32 s5, 0
    0xF408_0202,
    0xFA00_0000, // s_load_dwordx4 s[8:11], s[4:5], 0x0
    0xBE8C_0302, // s_mov_b32 s12, s2
    0xBE8D_0380, // s_mov_b32 s13, 0
    0xF408_0406,
    0xFA00_0000, // s_load_dwordx4 s[16:19], s[12:13], 0x0
    0x3402_0082, // v_lshlrev_b32 v1, 2, v0
    0xBF8C_C07F, // s_waitcnt lgkmcnt(0)
    0xE030_1000,
    0x8002_0201, // buffer_load_dword v2, v1, s[8:11], 0 offen
    0xBF8C_3F70, // s_waitcnt vmcnt(0)
    0x7E06_020A, // v_mov_b32 v3, s10
    0xE070_1000,
    0x8004_0201, // buffer_store_dword v2, v1, s[16:19], 0 offen
    0xE070_1080,
    0x8004_0301, // buffer_store_dword v3, v1, s[16:19], 0 offen offset:128
    0xBF81_0000, // s_endpgm
];

/// User data: s2 a table with a raw UAV. Each lane stores its index as the
/// half at byte `2 × lane` (two lanes share each dword).
const HALF_STORE_WAVE32: &[u32] = &[
    0xBE8C_0302, // s_mov_b32 s12, s2
    0xBE8D_0380, // s_mov_b32 s13, 0
    0xF408_0406,
    0xFA00_0000, // s_load_dwordx4 s[16:19], s[12:13], 0x0
    0x3402_0081, // v_lshlrev_b32 v1, 1, v0
    0xBF8C_C07F, // s_waitcnt lgkmcnt(0)
    0xE068_1000,
    0x8004_0001, // buffer_store_short v0, v1, s[16:19], 0 offen
    0xBF81_0000, // s_endpgm
];

/// User data: s2 a table with a raw UAV. Each lane stores the high half of
/// a VGPR whose low half is junk: its index as the half at byte `2 × lane`
/// (`buffer_store_short_d16_hi`), and `0x40 + lane` as the byte at
/// `64 + lane` (`buffer_store_byte_d16_hi`).
const HIGH_HALF_STORES_WAVE32: &[u32] = &[
    0xBE8C_0302, // s_mov_b32 s12, s2
    0xBE8D_0380, // s_mov_b32 s13, 0
    0xF408_0406,
    0xFA00_0000, // s_load_dwordx4 s[16:19], s[12:13], 0x0
    0x3402_0081, // v_lshlrev_b32 v1, 1, v0
    0x3404_0090, // v_lshlrev_b32 v2, 16, v0
    0x3804_04FF,
    0x0000_5A5A, // v_or_b32 v2, 0x5a5a, v2
    0x4A06_00C0, // v_add_nc_u32 v3, 64, v0
    0x3406_0690, // v_lshlrev_b32 v3, 16, v3
    0x3806_06FF,
    0x0000_A5A5, // v_or_b32 v3, 0xa5a5, v3
    0xBF8C_C07F, // s_waitcnt lgkmcnt(0)
    0xE06C_1000,
    0x8004_0201, // buffer_store_short_d16_hi v2, v1, s[16:19], 0 offen
    0xE064_1040,
    0x8004_0300, // buffer_store_byte_d16_hi v3, v0, s[16:19], 0 offen offset:64
    0xBF81_0000, // s_endpgm
];

fn param(root: RootType, user_data: u8) -> RootParam {
    RootParam {
        root,
        table_offset_dwords: 0,
        sgprs: 1,
        base_register: 0,
        user_data,
        slot: user_data,
    }
}

/// An entry at offset 0 of the table in user data `user_data`.
fn entry(kind: ResourceType, dwords: u8, user_data: u8) -> UsedResource {
    UsedResource {
        kind,
        root: RootType::Table,
        table_offset_dwords: 0,
        lower_bound: 0,
        stride: 0,
        dwords,
        user_data,
        slot: user_data,
    }
}

fn build(
    words: &[u32],
    layout: Vec<RootParam>,
    used: Vec<UsedResource>,
    threads: u32,
) -> HostShader {
    let native = NativeProgram {
        stage: HwStage::Cs,
        wave64: false,
        vgprs: 8,
        layout,
        used,
        io: StageIo::Cs(ComputeIo {
            rsrc2: 3 << 1,
            threads: [threads, 1, 1],
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
    })
    .expect("lower");
    let shader = HostShader {
        wave_size: lowered.wave_size,
        stage: HwStage::Cs,
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
    shader
}

#[test]
fn a_heap_buffer_read_by_element_takes_the_views_stride() {
    let shader = build(
        HEAP_ELEMENT_WAVE32,
        vec![
            param(RootType::Constants, 0),
            param(RootType::ResourceHeap, 1),
            param(RootType::Table, 2),
        ],
        vec![entry(ResourceType::Uav, 4, 2)],
        32,
    );
    assert_eq!(shader.root_srvs, [VIEW_STRIDES]);
    // 32 elements of 12 bytes: (100 + e, 200 + e, 300 + e).
    let data: Vec<u32> = (0..32).flat_map(|e| [100 + e, 200 + e, 300 + e]).collect();
    let views = [
        View::RawUav(32),
        View::Buffer {
            data: &data,
            stride: Stride::Structured(12),
        },
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
        let (root, pso) = host
            .compute_pipeline(&shader)
            .unwrap_or_else(|e| panic!("{label}: compute_pipeline failed: {e}"));
        let mut ud = vec![0u32; shader.user_data_dwords().max(1) as usize];
        for u in &shader.user_data {
            ud[u.index as usize] = match u.root {
                // The buffer is at heap index 1, the UAV's table at 0.
                RootType::Constants => 1,
                RootType::Table => 0,
                other => panic!("unexpected user data {other:?}"),
            };
        }
        let words = host
            .dispatch(
                &root,
                &pso,
                &Dispatch {
                    // User data, then the stride table: heap index 1 holds
                    // 12-byte elements.
                    roots: vec![
                        (0, Root::Constants(ud)),
                        (
                            1,
                            Root::View {
                                srv: true,
                                data: vec![4, 12],
                            },
                        ),
                    ],
                    views: &views,
                    samplers: &[],
                    groups: [1, 1, 1],
                },
            )
            .unwrap_or_else(|e| panic!("{label}: dispatch failed: {e}"));
        let want: Vec<u32> = (0..32).map(|l| 200 + l).collect();
        assert_eq!(words[0], want, "{label}");
        ran += 1;
    }
    if ran == 0 {
        crate::emit::host::no_device("no SM 6.6 device", &[]);
    }
}

/// A buffer read at a heap index from data whose slot holds no buffer view
/// (the view-stride table holds 0 there; a texture here) reads the host's
/// null buffer view instead (`emit`, "Buffer views at computed heap
/// indices"): a texture descriptor read as a buffer one is a wild address.
#[test]
fn a_heap_buffer_read_where_no_buffer_view_is_reads_zeros() {
    let shader = build(
        HEAP_ELEMENT_WAVE32,
        vec![
            param(RootType::Constants, 0),
            param(RootType::ResourceHeap, 1),
            param(RootType::Table, 2),
        ],
        vec![entry(ResourceType::Uav, 4, 2)],
        32,
    );
    let texel = [[1.0f32, 2.0, 3.0, 4.0]];
    let views = [View::RawUav(32), View::Texture(1, 1, &texel), View::NullBuffers];
    let _one = super::valu_tests::ONE_AT_A_TIME
        .lock()
        .unwrap_or_else(|e| e.into_inner());
    let mut ran = 0;
    for (label, warp) in [("hardware", false), ("warp", true)] {
        let Ok(host) = Host::new(warp) else {
            eprintln!("no SM 6.6 {label} device; skipped");
            continue;
        };
        let (root, pso) = host
            .compute_pipeline(&shader)
            .unwrap_or_else(|e| panic!("{label}: compute_pipeline failed: {e}"));
        let mut ud = vec![0u32; shader.user_data_dwords().max(1) as usize];
        for u in &shader.user_data {
            // The texture at heap index 1, the UAV's table at 0.
            ud[u.index as usize] = u32::from(u.root == RootType::Constants);
        }
        let dispatch = Dispatch {
            // Heap index 1 holds no buffer view: stride 0.
            roots: vec![(0, Root::Constants(ud)), (1, Root::View { srv: true, data: vec![4, 0] })],
            views: &views,
            samplers: &[],
            groups: [1, 1, 1],
        };
        let words = host
            .dispatch(&root, &pso, &dispatch)
            .unwrap_or_else(|e| panic!("{label}: dispatch failed: {e}"));
        assert_eq!(words[0], [0; 32], "{label}");
        ran += 1;
    }
    if ran == 0 {
        crate::emit::host::no_device("no SM 6.6 device", &[]);
    }
}

/// Constant data through the resource heap itself is read from a raw view
/// (the host's constant buffer views need an alignment the title's lack).
#[test]
fn a_constant_buffer_through_the_heap_reads_as_a_raw_buffer() {
    let shader = build(
        HEAP_CONSTANT_WAVE32,
        vec![
            param(RootType::Constants, 0),
            param(RootType::ResourceHeap, 1),
            param(RootType::Table, 2),
        ],
        vec![entry(ResourceType::Uav, 4, 2)],
        32,
    );
    assert!(shader.heap && !shader.heap_writes, "reads the heap itself, writes through a table");
    assert_eq!(shader.root_srvs, [VIEW_STRIDES], "the guard's stride table");
    let data: Vec<u32> = vec![7, 8, 9, 10];
    let views = [
        View::RawUav(32),
        View::Buffer {
            data: &data,
            stride: Stride::Raw,
        },
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
        let (root, pso) = host
            .compute_pipeline(&shader)
            .unwrap_or_else(|e| panic!("{label}: compute_pipeline failed: {e}"));
        let mut ud = vec![0u32; shader.user_data_dwords().max(1) as usize];
        for u in &shader.user_data {
            ud[u.index as usize] = match u.root {
                // The constants are at heap index 1, the UAV's table at 0.
                RootType::Constants => 1,
                RootType::Table => 0,
                other => panic!("unexpected user data {other:?}"),
            };
        }
        let words = host
            .dispatch(
                &root,
                &pso,
                &Dispatch {
                    // User data, then the stride table: heap index 1 holds a
                    // buffer view.
                    roots: vec![(0, Root::Constants(ud)), (1, Root::View { srv: true, data: vec![4, 4] })],
                    views: &views,
                    samplers: &[],
                    groups: [1, 1, 1],
                },
            )
            .unwrap_or_else(|e| panic!("{label}: dispatch failed: {e}"));
        assert_eq!(words[0], vec![9; 32], "{label}");
        ran += 1;
    }
    if ran == 0 {
        crate::emit::host::no_device("no SM 6.6 device", &[]);
    }
}

/// `buffer_load_ushort`: the half at its byte address, zero-extended (the
/// BVH builder reads 16-bit fields this way).
#[test]
fn a_half_buffer_load_reads_the_half_at_its_address() {
    let shader = build(
        HALF_LOAD_WAVE32,
        vec![param(RootType::Table, 0), param(RootType::Table, 2)],
        vec![entry(ResourceType::Srv, 4, 0), entry(ResourceType::Uav, 4, 2)],
        32,
    );
    // 32 halves 0x8000 + 3 × lane, two per dword (low half first).
    let half = |l: u32| 0x8000 + 3 * l;
    let data: Vec<u32> = (0..16).map(|k| half(2 * k) | half(2 * k + 1) << 16).collect();
    let views = [
        View::Buffer {
            data: &data,
            stride: Stride::Raw,
        },
        View::RawUav(32),
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
        let (root, pso) = host
            .compute_pipeline(&shader)
            .unwrap_or_else(|e| panic!("{label}: compute_pipeline failed: {e}"));
        let mut ud = vec![0u32; shader.user_data_dwords().max(1) as usize];
        for u in &shader.user_data {
            // The SRV's table at heap index 0, the UAV's at 1.
            ud[u.index as usize] = if u.slot == 0 { 0 } else { 32 };
        }
        let words = host
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
            .unwrap_or_else(|e| panic!("{label}: dispatch failed: {e}"));
        let want: Vec<u32> = (0..32).map(half).collect();
        assert_eq!(words[0], want, "{label}");
        ran += 1;
    }
    if ran == 0 {
        crate::emit::host::no_device("no SM 6.6 device", &[]);
    }
}

/// A raw buffer's size read from the descriptor load its loads go through:
/// the size in bytes, beside what the loads read. AMD's RDNA 4 driver ends
/// the process at the pipeline's creation when both share one handle.
#[test]
fn a_raw_buffers_size_beside_its_loads_reads_its_byte_count() {
    let shader = build(
        RAW_SIZE_AND_LOAD_WAVE32,
        vec![param(RootType::Table, 0), param(RootType::Table, 2)],
        vec![
            entry(ResourceType::Srv, 4, 0),
            entry(ResourceType::Uav, 4, 2),
        ],
        32,
    );
    let data: Vec<u32> = (0..40).map(|k| 0x1000 + 7 * k).collect();
    let views = [
        View::Buffer {
            data: &data,
            stride: Stride::Raw,
        },
        View::RawUav(64),
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
        let (root, pso) = host
            .compute_pipeline(&shader)
            .unwrap_or_else(|e| panic!("{label}: compute_pipeline failed: {e}"));
        let mut ud = vec![0u32; shader.user_data_dwords().max(1) as usize];
        for u in &shader.user_data {
            // The SRV's table at heap index 0, the UAV's at 1.
            ud[u.index as usize] = if u.slot == 0 { 0 } else { 32 };
        }
        let words = host
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
            .unwrap_or_else(|e| panic!("{label}: dispatch failed: {e}"));
        // 40 dwords: 160 bytes.
        let want: Vec<u32> = (0..32).map(|l| 0x1000 + 7 * l).chain([160; 32]).collect();
        assert_eq!(words[0], want, "{label}");
        ran += 1;
    }
    assert!(ran > 0, "neither hardware nor WARP SM 6.6 device available");
}

/// `buffer_store_short`: each lane's half lands at its byte address and
/// leaves the other half of the dword alone.
#[test]
fn half_buffer_stores_leave_the_other_half_alone() {
    let shader = build(
        HALF_STORE_WAVE32,
        vec![param(RootType::Table, 2)],
        vec![entry(ResourceType::Uav, 4, 2)],
        32,
    );
    let views = [View::RawUav(16)];
    let _one = super::valu_tests::ONE_AT_A_TIME
        .lock()
        .unwrap_or_else(|e| e.into_inner());
    let mut ran = 0;
    for (label, warp) in [("hardware", false), ("warp", true)] {
        let Ok(host) = Host::new(warp) else {
            eprintln!("no SM 6.6 {label} device; skipped");
            continue;
        };
        let (root, pso) = host
            .compute_pipeline(&shader)
            .unwrap_or_else(|e| panic!("{label}: compute_pipeline failed: {e}"));
        let ud = vec![0u32; shader.user_data_dwords().max(1) as usize];
        let words = host
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
            .unwrap_or_else(|e| panic!("{label}: dispatch failed: {e}"));
        let want: Vec<u32> = (0..16).map(|k| (2 * k) | ((2 * k + 1) << 16)).collect();
        assert_eq!(words[0], want, "{label}");
        ran += 1;
    }
    if ran == 0 {
        crate::emit::host::no_device("no SM 6.6 device", &[]);
    }
}

/// `buffer_store_short_d16_hi` and `buffer_store_byte_d16_hi` store the
/// high half of their VGPR, each lane's field at its byte address, leaving
/// the rest of the dword alone.
#[test]
fn high_half_buffer_stores_take_the_vgprs_high_half() {
    let shader = build(
        HIGH_HALF_STORES_WAVE32,
        vec![param(RootType::Table, 2)],
        vec![entry(ResourceType::Uav, 4, 2)],
        32,
    );
    let views = [View::RawUav(32)];
    let _one = super::valu_tests::ONE_AT_A_TIME
        .lock()
        .unwrap_or_else(|e| e.into_inner());
    let mut want: Vec<u32> = (0..16).map(|k| (2 * k) | ((2 * k + 1) << 16)).collect();
    want.extend((0..8u8).map(|k| u32::from_le_bytes([0, 1, 2, 3].map(|b| 0x40 + 4 * k + b))));
    want.resize(32, 0);
    let mut ran = 0;
    for (label, warp) in [("hardware", false), ("warp", true)] {
        let Ok(host) = Host::new(warp) else {
            eprintln!("no SM 6.6 {label} device; skipped");
            continue;
        };
        let (root, pso) = host
            .compute_pipeline(&shader)
            .unwrap_or_else(|e| panic!("{label}: compute_pipeline failed: {e}"));
        let ud = vec![0u32; shader.user_data_dwords().max(1) as usize];
        let words = host
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
            .unwrap_or_else(|e| panic!("{label}: dispatch failed: {e}"));
        assert_eq!(words[0], want, "{label}");
        ran += 1;
    }
    assert!(ran > 0, "neither hardware nor WARP SM 6.6 device available");
}

#[test]
fn a_sampler_chosen_by_a_uniform_branch_samples_as_chosen() {
    let shader = build(
        CHOSEN_SAMPLER_WAVE32,
        vec![
            param(RootType::Table, 0),
            param(RootType::Constants, 1),
            param(RootType::Table, 2),
        ],
        vec![
            entry(ResourceType::Srv, 8, 0),
            entry(ResourceType::Uav, 4, 2),
        ],
        1,
    );
    assert_eq!(shader.samplers.len(), 2);
    // Red 1 at u < 0.5, 2 from there: u = 1.25 wraps to 0.25, clamps to 1.
    let texels = [[1.0f32, 0.0, 0.0, 1.0], [2.0, 0.0, 0.0, 1.0]];
    let views = [View::Texture(2, 1, &texels), View::RawUav(1)];
    let samplers: [D3D12_SAMPLER_DESC; 0] = [];
    let _one = super::valu_tests::ONE_AT_A_TIME
        .lock()
        .unwrap_or_else(|e| e.into_inner());
    let mut ran = 0;
    for (label, warp) in [("hardware", false), ("warp", true)] {
        let Ok(host) = Host::new(warp) else {
            eprintln!("no SM 6.6 {label} device; skipped");
            continue;
        };
        let (root, pso) = host
            .compute_pipeline(&shader)
            .unwrap_or_else(|e| panic!("{label}: compute_pipeline failed: {e}"));
        for (clamp, want) in [(1, 2.0f32), (0, 1.0)] {
            let mut ud = vec![0u32; shader.user_data_dwords().max(1) as usize];
            for u in &shader.user_data {
                ud[u.index as usize] = match (u.root, u.slot) {
                    (RootType::Table, 0) => 0,
                    (RootType::Constants, 1) => clamp,
                    // The UAV at heap index 1: 32 bytes in.
                    (RootType::Table, 2) => 32,
                    other => panic!("unexpected user data {other:?}"),
                };
            }
            let words = host
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
            assert_eq!(f32::from_bits(words[0][0]), want, "{label}, clamp {clamp}");
        }
        ran += 1;
    }
    if ran == 0 {
        crate::emit::host::no_device("no SM 6.6 device", &[]);
    }
}
/// User data: s0 a table with a typed buffer UAV, s1 a table with a raw
/// buffer of 2 dwords a lane, s2 a table with a raw UAV. Each lane stores
/// its 4 halves to element `lane` of the typed view (`_d16`), loads them
/// back the same way and stores what it read to dwords `2 lane, 2 lane + 1`
/// of the raw UAV.
const D16_ROUND_TRIP_WAVE32: &[u32] = &[
    0xBE84_0300, // s_mov_b32 s4, s0
    0xBE85_0380, // s_mov_b32 s5, 0
    0xF408_0202,
    0xFA00_0000, // s_load_dwordx4 s[8:11], s[4:5], 0x0
    0xBE86_0301, // s_mov_b32 s6, s1
    0xBE87_0380, // s_mov_b32 s7, 0
    0xF408_0303,
    0xFA00_0000, // s_load_dwordx4 s[12:15], s[6:7], 0x0
    0xBE94_0302, // s_mov_b32 s20, s2
    0xBE95_0380, // s_mov_b32 s21, 0
    0xF408_040A,
    0xFA00_0000, // s_load_dwordx4 s[16:19], s[20:21], 0x0
    0x3406_0083, // v_lshlrev_b32 v3, 3, v0
    0xBF8C_C07F, // s_waitcnt lgkmcnt(0)
    0xE034_1000,
    0x8003_0103, // buffer_load_dwordx2 v[1:2], v3, s[12:15], 0 offen
    0xBF8C_3F70, // s_waitcnt vmcnt(0)
    0xE21C_2000,
    0x8002_0100, // buffer_store_format_d16_xyzw v[1:2], v0, s[8:11], 0 idxen
    0xE20C_2000,
    0x8002_0400, // buffer_load_format_d16_xyzw v[4:5], v0, s[8:11], 0 idxen
    0xBF8C_3F70, // s_waitcnt vmcnt(0)
    0xE074_1000,
    0x8004_0403, // buffer_store_dwordx2 v[4:5], v3, s[16:19], 0 offen
    0xBF81_0000, // s_endpgm
];

/// User data: s0 a table with a typed buffer SRV, s1 a table with a raw UAV.
/// Each lane loads element `lane` (`_d16_x`) and stores the dword it got to
/// dword `lane` of the raw UAV.
const D16_X_LOAD_WAVE32: &[u32] = &[
    0xBE84_0300, // s_mov_b32 s4, s0
    0xBE85_0380, // s_mov_b32 s5, 0
    0xF408_0202,
    0xFA00_0000, // s_load_dwordx4 s[8:11], s[4:5], 0x0
    0xBE86_0301, // s_mov_b32 s6, s1
    0xBE87_0380, // s_mov_b32 s7, 0
    0xF408_0403,
    0xFA00_0000, // s_load_dwordx4 s[16:19], s[6:7], 0x0
    0x3406_0082, // v_lshlrev_b32 v3, 2, v0
    0x7E08_0280, // v_mov_b32 v4, 0
    0xBF8C_C07F, // s_waitcnt lgkmcnt(0)
    0xE200_2000,
    0x8002_0400, // buffer_load_format_d16_x v4, v0, s[8:11], 0 idxen
    0xBF8C_3F70, // s_waitcnt vmcnt(0)
    0xE070_1000,
    0x8004_0403, // buffer_store_dword v4, v3, s[16:19], 0 offen
    0xBF81_0000, // s_endpgm
];

/// Runs `shader` (user data: tables at heap indices 0, 1, 2 in order) on
/// hardware and WARP with `views` and the view-stride table `strides`;
/// each adapter's UAV contents.
fn run_tables(shader: &HostShader, views: &[View], strides: Vec<u32>) -> Vec<(&'static str, Vec<Vec<u32>>)> {
    assert_eq!(shader.root_srvs, [VIEW_STRIDES]);
    let _one = super::valu_tests::ONE_AT_A_TIME
        .lock()
        .unwrap_or_else(|e| e.into_inner());
    let mut out = Vec::new();
    for (label, warp) in [("hardware", false), ("warp", true)] {
        let Ok(host) = Host::new(warp) else {
            eprintln!("no SM 6.6 {label} device; skipped");
            continue;
        };
        let (root, pso) = host
            .compute_pipeline(shader)
            .unwrap_or_else(|e| panic!("{label}: compute_pipeline failed: {e}"));
        let mut ud = vec![0u32; shader.user_data_dwords().max(1) as usize];
        for u in &shader.user_data {
            // Table k is the view at heap index k (32 bytes a view).
            ud[u.index as usize] = 32 * u32::from(u.slot);
        }
        let words = host
            .dispatch(
                &root,
                &pso,
                &Dispatch {
                    roots: vec![(0, Root::Constants(ud)), (1, Root::View { srv: true, data: strides.clone() })],
                    views,
                    samplers: &[],
                    groups: [1, 1, 1],
                },
            )
            .unwrap_or_else(|e| panic!("{label}: dispatch failed: {e}"));
        out.push((label, words));
    }
    assert!(!out.is_empty(), "neither hardware nor WARP SM 6.6 device available");
    out
}

/// 16-bit typed stores and loads (`_d16`) through views of integer formats
/// pass the bits (the console's buffer format decides the conversion; d16
/// only packs 16-bit results), and through float views convert as halves:
/// Grand Theft Auto VI's face network keeps its halves in UINT views.
#[test]
fn d16_typed_buffer_accesses_pass_bits_through_integer_views() {
    let shader = build(
        D16_ROUND_TRIP_WAVE32,
        vec![
            param(RootType::Table, 0),
            param(RootType::Table, 1),
            param(RootType::Table, 2),
        ],
        vec![
            entry(ResourceType::Uav, 4, 0),
            entry(ResourceType::Srv, 4, 1),
            entry(ResourceType::Uav, 4, 2),
        ],
        32,
    );
    // Lane l: negative zero and small, signalling NaN, denormal and
    // all-ones patterns as bits; normal halves for the float view.
    let bits: Vec<u32> = (0..32u32)
        .flat_map(|l| [(0x8000 | l) | (0x7C01 + l) << 16, (1 + l) | (0xFFFF - l) << 16])
        .collect();
    let halves: Vec<u32> = (0..32u32)
        .flat_map(|l| [(0x3C00 + l) | (0xC000 + 3 * l) << 16, (0x1400 + l) | (0x5BFF - l) << 16])
        .collect();
    let cases: [(DXGI_FORMAT, u32, &[u32]); 2] = [
        (DXGI_FORMAT_R16G16B16A16_UINT, 8 | VIEW_INTEGER, &bits),
        (DXGI_FORMAT_R16G16B16A16_FLOAT, 8, &halves),
    ];
    for (format, stride, data) in cases {
        let views = [
            View::TypedUav(format, 32, 8),
            View::Buffer { data, stride: Stride::Raw },
            View::RawUav(64),
        ];
        for (label, words) in run_tables(&shader, &views, vec![stride, 4, 4]) {
            assert_eq!(words[0], data, "{label} {format:?}: the typed view's bytes");
            assert_eq!(words[1], data, "{label} {format:?}: what the loads read");
        }
    }
}

/// A one-channel `_d16_x` load from a typed SRV: an integer view's 16 bits
/// as they are, a float view's half as it is.
#[test]
fn d16_x_loads_read_r16_uint_and_r16_float_views() {
    let shader = build(
        D16_X_LOAD_WAVE32,
        vec![param(RootType::Table, 0), param(RootType::Table, 1)],
        vec![entry(ResourceType::Srv, 4, 0), entry(ResourceType::Uav, 4, 1)],
        32,
    );
    let pack = |h: &dyn Fn(u32) -> u32| -> Vec<u32> { (0..16u32).map(|k| h(2 * k) | h(2 * k + 1) << 16).collect() };
    let uint = |l: u32| 0xF000 | l;
    let half = |l: u32| 0x3C00 + 5 * l;
    let cases: [(DXGI_FORMAT, u32, Vec<u32>, Vec<u32>); 2] = [
        (DXGI_FORMAT_R16_UINT, 2 | VIEW_INTEGER, pack(&uint), (0..32).map(uint).collect()),
        (DXGI_FORMAT_R16_FLOAT, 2, pack(&half), (0..32).map(half).collect()),
    ];
    for (format, stride, data, want) in cases {
        let views = [View::Buffer { data: &data, stride: Stride::Format(format, 2) }, View::RawUav(32)];
        for (label, words) in run_tables(&shader, &views, vec![stride, 4]) {
            assert_eq!(words[0], want, "{label} {format:?}");
        }
    }
}

const S_ENDPGM: u32 = 0xBF81_0000;
const S_WAITCNT_LGKM: u32 = 0xBF8C_C07F;
const S_WAITCNT_VM: u32 = 0xBF8C_3F70;

const fn s_mov_b32(sdst: u8, src0: u8) -> u32 {
    0xBE80_0300 | (sdst as u32) << 16 | src0 as u32
}

const fn s_load_x4(sdata: u8, sbase: u8, offset: u32) -> [u32; 2] {
    [
        0x3D << 26 | 2 << 18 | (sdata as u32) << 6 | (sbase as u32 >> 1),
        125 << 25 | (offset & 0x001F_FFFF),
    ]
}

const fn vop2(op: u32, vdst: u8, src0: u16, vsrc1: u8) -> u32 {
    op << 25 | (vdst as u32) << 17 | (vsrc1 as u32) << 9 | (src0 as u32 & 0x1FF)
}

const fn v_cmpx_eq_u32(src0: u16, vsrc1: u8) -> u32 {
    0x7C00_0000 | 0xD2 << 17 | (vsrc1 as u32) << 9 | (src0 as u32 & 0x1FF)
}

/// MTBUF `tbuffer_load_format_*` (`op` 0..=3 for x..=xyzw).
#[allow(clippy::too_many_arguments)]
const fn mtbuf(
    op: u32,
    format: u32,
    idxen: bool,
    offen: bool,
    offset: u16,
    vaddr: u8,
    vdata: u8,
    srsrc: u8,
    soffset: u8,
) -> [u32; 2] {
    [
        0x3A << 26
            | (format & 0x7F) << 19
            | (op & 7) << 16
            | (idxen as u32) << 13
            | (offen as u32) << 12
            | (offset as u32 & 0xFFF),
        (soffset as u32) << 24
            | ((op >> 3) & 1) << 21
            | ((srsrc as u32 >> 2) & 0x1F) << 16
            | (vdata as u32) << 8
            | vaddr as u32,
    ]
}

/// MUBUF `op` (`0x18` store_byte, `0x1A` store_short, `0x1C` store_dword).
#[allow(clippy::too_many_arguments)]
const fn mubuf(
    op: u32,
    idxen: bool,
    offen: bool,
    offset: u16,
    vaddr: u8,
    vdata: u8,
    srsrc: u8,
    soffset: u8,
) -> [u32; 2] {
    [
        0x38 << 26
            | ((op >> 7) & 1) << 25
            | (op & 0x7F) << 18
            | (idxen as u32) << 13
            | (offen as u32) << 12
            | (offset as u32 & 0xFFF),
        (soffset as u32) << 24
            | ((srsrc as u32 >> 2) & 0x1F) << 16
            | (vdata as u32) << 8
            | vaddr as u32,
    ]
}

fn structured_entry(kind: ResourceType, stride: u32, user_data: u8) -> UsedResource {
    UsedResource {
        kind,
        root: RootType::Table,
        table_offset_dwords: 0,
        lower_bound: 0,
        stride,
        dwords: 4,
        user_data,
        slot: user_data,
    }
}

fn try_lower(
    words: &[u32],
    layout: Vec<RootParam>,
    used: Vec<UsedResource>,
    vgprs: u16,
    threads: u32,
) -> Result<HostShader, crate::ShaderError> {
    let native = NativeProgram {
        stage: HwStage::Cs,
        wave64: false,
        vgprs,
        layout,
        used,
        io: StageIo::Cs(ComputeIo {
            rsrc2: 8 << 1,
            threads: [threads, 1, 1],
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
        view_marks: false,
    })?;
    let shader = HostShader {
        stage: HwStage::Cs,
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

/// Indexed `tbuffer_load_format_*` on a declared `StructuredBuffer` SRV
/// (stride 20): sub-dword `8_UNORM` at unaligned offset 5, `8_8_SINT` at
/// offset 6, `8_8_8_8_UNORM` at offset 0, 64-bit `16_16_16_16_FLOAT` at
/// offset 8, channel-padded `8_UINT` (`xyzw` -> `(x, 0, 0, 1)`), and
/// runtime `idxen + offen + soffset` `16_16_SNORM`.
#[test]
fn indexed_tbuffer_load_format_reads_structured_srv_formats() {
    // s0: table with StructuredBuffer SRV (stride 20), s1: root constant soffset (4),
    // s2: table with raw output UAV (16 dwords per lane).
    const V_LSHLREV_B32: u32 = 0x1A;
    const V_AND_B32: u32 = 0x1B;
    const V_ADD_NC_U32: u32 = 0x25;
    let mut w = vec![
        s_mov_b32(4, 0),
        s_mov_b32(5, 0x80),
        s_mov_b32(6, 2),
        s_mov_b32(7, 0x80),
    ];
    w.extend(s_load_x4(8, 4, 0)); // s[8:11] = StructuredBuffer SRV (stride 20)
    w.extend(s_load_x4(12, 6, 0)); // s[12:15] = raw output UAV
    w.push(S_WAITCNT_LGKM);
    // Out base in bytes: v15 = lane * 64 (16 dwords per lane).
    w.push(vop2(V_LSHLREV_B32, 15, 128 + 6, 0));
    // 1. tbuffer_load_format_xyzw 8_8_8_8_UNORM (fmt 56) idxen offset:0 -> v[2:5]
    w.extend(mtbuf(3, 56, true, false, 0, 0, 2, 8, 0x80));
    // 2. tbuffer_load_format_x 8_UNORM (fmt 1) idxen offset:5 -> v6
    w.extend(mtbuf(0, 1, true, false, 5, 0, 6, 8, 0x80));
    // 3. tbuffer_load_format_xy 8_8_SINT (fmt 19) idxen offset:6 -> v[7:8]
    w.extend(mtbuf(1, 19, true, false, 6, 0, 7, 8, 0x80));
    // 4. tbuffer_load_format_xyzw 16_16_16_16_FLOAT (fmt 71) idxen offset:8 -> v[9:12]
    w.extend(mtbuf(3, 71, true, false, 8, 0, 9, 8, 0x80));
    // 5. tbuffer_load_format_xyzw 8_UINT (fmt 5) idxen offset:4 -> v[16:19] (padding: x,0,0,1)
    w.extend(mtbuf(3, 5, true, false, 4, 0, 16, 8, 0x80));
    // 6. idxen + offen + soffset(s1=4): v[20:21] = (lane, (lane & 1) ? 12 : 0); offset:0
    //    total offset = 4 + (0 or 12) = 4 (dword 1) or 16 (dword 4), fmt 24 = 16_16_SNORM -> v[22:23]
    w.push(vop2(V_AND_B32, 21, 128 + 1, 0)); // v21 = lane & 1
    w.push(vop2(V_LSHLREV_B32, 22, 128 + 2, 21)); // v22 = (lane & 1) * 4
    w.push(vop2(V_LSHLREV_B32, 23, 128 + 3, 21)); // v23 = (lane & 1) * 8
    w.push(vop2(V_ADD_NC_U32, 21, 256 + 22, 23)); // v21 = (lane & 1) * 12
    w.push(0x7E28_0300); // v_mov_b32 v20, v0
    w.extend(mtbuf(1, 24, true, true, 0, 20, 22, 8, 1));
    w.push(S_WAITCNT_VM);
    // Store all 16 result dwords per lane.
    for (slot, reg) in [
        (0, 2),
        (1, 3),
        (2, 4),
        (3, 5),
        (4, 6),
        (5, 7),
        (6, 8),
        (7, 9),
        (8, 10),
        (9, 11),
        (10, 12),
        (11, 16),
        (12, 17),
        (13, 18),
        (14, 19),
        (15, 22),
    ] {
        w.extend(mubuf(0x1C, false, true, slot * 4, 15, reg, 12, 0x80));
    }
    w.push(S_ENDPGM);

    let shader = try_lower(
        &w,
        vec![
            param(RootType::Table, 0),
            param(RootType::Constants, 1),
            param(RootType::Table, 2),
        ],
        vec![
            structured_entry(ResourceType::Srv, 20, 0),
            entry(ResourceType::Uav, 4, 2),
        ],
        24,
        32,
    )
    .expect("lower");

    // Half-float constants: 0.0 (0x0000), 1.0 (0x3C00), -2.0 (0xC000), 0.5 (0x3800), 1.5 (0x3E00).
    const F16_TABLE: [(u16, f32); 5] = [
        (0x0000, 0.0),
        (0x3C00, 1.0),
        (0xC000, -2.0),
        (0x3800, 0.5),
        (0x3E00, 1.5),
    ];
    // 32 elements of 20 bytes (5 dwords each):
    // dword 0: 8_8_8_8_UNORM (r, g, b, a)
    // dword 1: byte0 = 8_UINT, byte1 = 8_UNORM, bytes2..3 = 8_8_SINT (also read as 16_16_SNORM for even lanes)
    // dwords 2..3: 16_16_16_16_FLOAT
    // dword 4: 16_16_SNORM (read by odd lanes via idxen + offen + soffset)
    let mut data = Vec::with_capacity(32 * 5);
    for l in 0..32u32 {
        let r = (l * 7) & 0xFF;
        let g = (l * 13 + 17) & 0xFF;
        let b = (255 - l * 5) & 0xFF;
        let a = (l * 3 + 1) & 0xFF;
        let w0 = r | (g << 8) | (b << 16) | (a << 24);
        let b0 = (0x80 + l) & 0xFF;
        let b1 = (l * 8) & 0xFF;
        let s2 = (l as i32 - 16) as i8 as u8 as u32;
        let s3 = (-(l as i32) * 3) as i8 as u8 as u32;
        let w1 = b0 | (b1 << 8) | (s2 << 16) | (s3 << 24);
        let h0 = F16_TABLE[(l as usize) % 5].0 as u32;
        let h1 = F16_TABLE[(l as usize + 1) % 5].0 as u32;
        let h2 = F16_TABLE[(l as usize + 2) % 5].0 as u32;
        let h3 = F16_TABLE[(l as usize + 3) % 5].0 as u32;
        let w2 = h0 | (h1 << 16);
        let w3 = h2 | (h3 << 16);
        let sn0: i16 = if l % 3 == 0 {
            i16::MIN // tests clamp to -1.0
        } else {
            ((l as i32 - 16) * 1024) as i16
        };
        let sn1: i16 = ((l as i32 * 700) - 10_000) as i16;
        let w4 = (sn0 as u16 as u32) | ((sn1 as u16 as u32) << 16);
        data.extend([w0, w1, w2, w3, w4]);
    }
    let views = [
        View::Buffer {
            data: &data,
            stride: Stride::Structured(20),
        },
        View::RawUav(32 * 16),
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
        let (root, pso) = host
            .compute_pipeline(&shader)
            .unwrap_or_else(|e| panic!("{label}: compute_pipeline failed: {e}"));
        let mut ud = vec![0u32; shader.user_data_dwords().max(1) as usize];
        for u in &shader.user_data {
            ud[u.index as usize] = match (u.root, u.slot) {
                (RootType::Table, 0) => 0,
                (RootType::Constants, 1) => 4,
                (RootType::Table, 2) => 32,
                other => panic!("unexpected user data {other:?}"),
            };
        }
        let words = host
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
            .unwrap_or_else(|e| panic!("{label}: dispatch failed: {e}"));
        for l in 0..32usize {
            let out = &words[0][l * 16..(l + 1) * 16];
            let w0 = data[l * 5];
            let w1 = data[l * 5 + 1];
            let w4 = data[l * 5 + 4];
            for (ch, &bits) in out.iter().enumerate().take(4) {
                let want = ((w0 >> (8 * ch)) & 0xFF) as f32 / 255.0;
                let got = f32::from_bits(bits);
                assert!(
                    (got - want).abs() <= 1e-6,
                    "{label} lane {l} 8_8_8_8_UNORM ch {ch}: got {got}, want {want}"
                );
            }
            let want_b1 = ((w1 >> 8) & 0xFF) as f32 / 255.0;
            let got_b1 = f32::from_bits(out[4]);
            assert!(
                (got_b1 - want_b1).abs() <= 1e-6,
                "{label} lane {l} 8_UNORM@5: got {got_b1}, want {want_b1}"
            );
            assert_eq!(
                out[5] as i32,
                i32::from((w1 >> 16) as i8),
                "{label} lane {l} 8_8_SINT.x"
            );
            assert_eq!(
                out[6] as i32,
                i32::from((w1 >> 24) as i8),
                "{label} lane {l} 8_8_SINT.y"
            );
            for ch in 0..4 {
                let want = F16_TABLE[(l + ch) % 5].1;
                assert_eq!(
                    f32::from_bits(out[7 + ch]),
                    want,
                    "{label} lane {l} 16_16_16_16_FLOAT ch {ch}"
                );
            }
            assert_eq!(out[11], w1 & 0xFF, "{label} lane {l} 8_UINT.x");
            assert_eq!(out[12], 0, "{label} lane {l} 8_UINT.y pad");
            assert_eq!(out[13], 0, "{label} lane {l} 8_UINT.z pad");
            assert_eq!(out[14], 1, "{label} lane {l} 8_UINT.w pad");
            let sn_word = if l % 2 == 1 { w4 } else { w1 };
            let want_sn0 = (f32::from(sn_word as i16) / 32767.0).max(-1.0);
            let got_sn0 = f32::from_bits(out[15]);
            assert!(
                (got_sn0 - want_sn0).abs() <= 1e-6,
                "{label} lane {l} 16_16_SNORM idxen+offen+soffset: got {got_sn0}, want {want_sn0}"
            );
        }
        ran += 1;
    }
    assert!(ran > 0, "neither hardware nor WARP SM 6.6 device available");
}

/// Indexed `tbuffer_load_format_*` on a declared `RWStructuredBuffer` UAV
/// (stride 16) reads `8_8_8_8_UINT` and `32_32_FLOAT` (with `(0.0, 1.0)`
/// missing-channel padding).
#[test]
fn indexed_tbuffer_load_format_reads_structured_uav() {
    const V_LSHLREV_B32: u32 = 0x1A;
    let mut w = vec![
        s_mov_b32(4, 0),
        s_mov_b32(5, 0x80),
        s_mov_b32(6, 2),
        s_mov_b32(7, 0x80),
    ];
    w.extend(s_load_x4(8, 4, 0)); // s[8:11] = RWStructuredBuffer UAV (stride 16)
    w.extend(s_load_x4(12, 6, 0)); // s[12:15] = raw output UAV
    w.push(S_WAITCNT_LGKM);
    w.push(vop2(V_LSHLREV_B32, 1, 128 + 5, 0)); // v1 = lane * 32 (8 dwords/lane)
    // tbuffer_load_format_xyzw 8_8_8_8_UINT (fmt 60) idxen offset:0 -> v[2:5]
    w.extend(mtbuf(3, 60, true, false, 0, 0, 2, 8, 0x80));
    // tbuffer_load_format_xyzw 32_32_FLOAT (fmt 64) idxen offset:4 -> v[6:9]
    w.extend(mtbuf(3, 64, true, false, 4, 0, 6, 8, 0x80));
    w.push(S_WAITCNT_VM);
    for (slot, reg) in [
        (0, 2),
        (1, 3),
        (2, 4),
        (3, 5),
        (4, 6),
        (5, 7),
        (6, 8),
        (7, 9),
    ] {
        w.extend(mubuf(0x1C, false, true, slot * 4, 1, reg, 12, 0x80));
    }
    w.push(S_ENDPGM);

    let shader = try_lower(
        &w,
        vec![param(RootType::Table, 0), param(RootType::Table, 2)],
        vec![
            structured_entry(ResourceType::Uav, 16, 0),
            entry(ResourceType::Uav, 4, 2),
        ],
        10,
        32,
    )
    .expect("lower");

    let mut uav_data = Vec::with_capacity(32 * 4);
    for l in 0..32u32 {
        let packed = (l + 1) | ((l + 33) << 8) | ((l + 65) << 16) | ((l + 97) << 24);
        let fx = (l as f32 * 1.25 - 10.0).to_bits();
        let fy = (l as f32 * -0.5 + 3.5).to_bits();
        uav_data.extend([packed, fx, fy, 0xCAFE_BABE]);
    }
    let views = [
        View::StructuredUav {
            data: &uav_data,
            stride: 16,
        },
        View::RawUav(32 * 8),
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
        let (root, pso) = host
            .compute_pipeline(&shader)
            .unwrap_or_else(|e| panic!("{label}: compute_pipeline failed: {e}"));
        let mut ud = vec![0u32; shader.user_data_dwords().max(1) as usize];
        for u in &shader.user_data {
            ud[u.index as usize] = if u.slot == 0 { 0 } else { 32 };
        }
        let words = host
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
            .unwrap_or_else(|e| panic!("{label}: dispatch failed: {e}"));
        // words[0] is the input StructuredUav (untouched), words[1] is the output RawUav.
        assert_eq!(words[0], uav_data, "{label}: input StructuredUav unchanged");
        for l in 0..32usize {
            let out = &words[1][l * 8..(l + 1) * 8];
            let lu = l as u32;
            assert_eq!(
                out[..4],
                [lu + 1, lu + 33, lu + 65, lu + 97],
                "{label} lane {l} uint4"
            );
            assert_eq!(out[4], uav_data[l * 4 + 1], "{label} lane {l} float2.x");
            assert_eq!(out[5], uav_data[l * 4 + 2], "{label} lane {l} float2.y");
            assert_eq!(f32::from_bits(out[6]), 0.0, "{label} lane {l} float2.z pad");
            assert_eq!(f32::from_bits(out[7]), 1.0, "{label} lane {l} float2.w pad");
        }
        ran += 1;
    }
    assert!(ran > 0, "neither hardware nor WARP SM 6.6 device available");
}

/// Indexed `tbuffer_load_format_*` through both a root SRV descriptor
/// (`Source::Built`, stride 16 in dword 1) and a bindless heap descriptor
/// (`ResourceDescriptorHeap` via `VIEW_STRIDES`, stride 12).
#[test]
fn indexed_tbuffer_load_format_reads_root_view_and_heap_view() {
    const V_LSHLREV_B32: u32 = 0x1A;
    // Part A: root SRV (`Source::Built`, stride 16 in dword 1 bits 29:16, words[0] = -1).
    let mut w_root = vec![
        s_mov_b32(5, 0x80), // s[4:5] = table at s4
        s_mov_b32(0, 6),    // s[0:3] = root SRV descriptor built from s[6:7]
        0x8801_FF07,
        16 << 16,    // s_or_b32 s1, s7, (16 << 16)
        0xBE82_03C1, // s_mov_b32 s2, -1
        0xBE83_03FF,
        0x1101_4FAC, // s_mov_b32 s3, buffer word 3
    ];
    w_root.extend(s_load_x4(8, 4, 0)); // s[8:11] = output raw UAV
    w_root.push(S_WAITCNT_LGKM);
    // tbuffer_load_format_xyzw 8_8_8_8_UNORM (fmt 56) idxen offset:4 -> v[2:5]
    w_root.extend(mtbuf(3, 56, true, false, 4, 0, 2, 0, 0x80));
    w_root.push(vop2(V_LSHLREV_B32, 1, 128 + 4, 0)); // v1 = lane * 16
    w_root.push(S_WAITCNT_VM);
    for (slot, reg) in [(0, 2), (1, 3), (2, 4), (3, 5)] {
        w_root.extend(mubuf(0x1C, false, true, slot * 4, 1, reg, 8, 0x80));
    }
    w_root.push(S_ENDPGM);

    let root_shader = try_lower(
        &w_root,
        vec![
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
        vec![
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
                stride: 16,
                dwords: 4,
                user_data: 6,
                slot: 2,
            },
        ],
        8,
        32,
    )
    .expect("lower root view");

    // Part B: heap-indexed buffer (`used.is_none()`) through `ResourceDescriptorHeap`.
    let mut w_heap = vec![
        0x8F04_8500, // s_lshl_b32 s4, s0, 5
        s_mov_b32(6, 1),
        s_mov_b32(7, 0x80),
        0xF408_0203,
        0x0800_0000, // s_load_dwordx4 s[8:11], s[6:7], s4
        s_mov_b32(12, 2),
        s_mov_b32(13, 0x80),
    ];
    w_heap.extend(s_load_x4(16, 12, 0)); // s[16:19] = output raw UAV
    w_heap.push(S_WAITCNT_LGKM);
    // tbuffer_load_format_xy 16_16_UINT (fmt 27) idxen offset:4 -> v[2:3]
    w_heap.extend(mtbuf(1, 27, true, false, 4, 0, 2, 8, 0x80));
    w_heap.push(vop2(V_LSHLREV_B32, 1, 128 + 3, 0)); // v1 = lane * 8
    w_heap.push(S_WAITCNT_VM);
    w_heap.extend(mubuf(0x1C, false, true, 0, 1, 2, 16, 0x80));
    w_heap.extend(mubuf(0x1C, false, true, 4, 1, 3, 16, 0x80));
    w_heap.push(S_ENDPGM);

    let heap_shader = try_lower(
        &w_heap,
        vec![
            param(RootType::Constants, 0),
            param(RootType::ResourceHeap, 1),
            param(RootType::Table, 2),
        ],
        vec![entry(ResourceType::Uav, 4, 2)],
        6,
        32,
    )
    .expect("lower heap view");
    assert_eq!(heap_shader.root_srvs, [VIEW_STRIDES]);

    // Data for root view: 32 elements * 16 bytes (4 dwords each); offset 4 holds 8_8_8_8_UNORM.
    let root_data: Vec<u32> = (0..32u32)
        .flat_map(|l| {
            let rgba = ((l * 5) & 0xFF)
                | (((l * 9 + 11) & 0xFF) << 8)
                | (((255 - l * 7) & 0xFF) << 16)
                | (((l * 4 + 3) & 0xFF) << 24);
            [0xDEAD_0000 + l, rgba, 0, 0]
        })
        .collect();
    // Data for heap view: 32 elements * 12 bytes (3 dwords each); offset 4 holds 16_16_UINT.
    let heap_data: Vec<u32> = (0..32u32)
        .flat_map(|l| {
            [
                0x1111_0000 + l,
                (0x100 + l) | ((0x2000 + l * 3) << 16),
                0x2222_0000 + l,
            ]
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
        // Dispatch root view test (raw SRV at heap index 1, output UAV at index 0).
        let (r_root, pso_root) = host
            .compute_pipeline(&root_shader)
            .unwrap_or_else(|e| panic!("{label}: root pipeline failed: {e}"));
        let mut ud = vec![0u32; root_shader.user_data_dwords().max(1) as usize];
        for u in &root_shader.user_data {
            ud[u.index as usize] = match (u.root, u.dword) {
                (RootType::Table, _) => 0,
                (RootType::Descriptor, super::ROOT_SRV_VIEW) => 1,
                (RootType::Descriptor, super::ROOT_VIEW_OFFSET) => 0,
                other => panic!("unexpected root user data {other:?}"),
            };
        }
        let views_root = [
            View::RawUav(32 * 4),
            View::Buffer {
                data: &root_data,
                stride: Stride::Raw,
            },
        ];
        let out_root = host
            .dispatch(
                &r_root,
                &pso_root,
                &Dispatch {
                    roots: vec![(0, Root::Constants(ud))],
                    views: &views_root,
                    samplers: &[],
                    groups: [1, 1, 1],
                },
            )
            .unwrap_or_else(|e| panic!("{label}: root dispatch failed: {e}"));
        for l in 0..32usize {
            let rgba = root_data[l * 4 + 1];
            for ch in 0..4 {
                let want = ((rgba >> (8 * ch)) & 0xFF) as f32 / 255.0;
                let got = f32::from_bits(out_root[0][l * 4 + ch]);
                assert!(
                    (got - want).abs() <= 1e-6,
                    "{label} root lane {l} ch {ch}: got {got}, want {want}"
                );
            }
        }

        // Dispatch heap view test (raw SRV at heap index 1 with VIEW_STRIDES[1] = 12).
        let (r_heap, pso_heap) = host
            .compute_pipeline(&heap_shader)
            .unwrap_or_else(|e| panic!("{label}: heap pipeline failed: {e}"));
        let mut ud_heap = vec![0u32; heap_shader.user_data_dwords().max(1) as usize];
        for u in &heap_shader.user_data {
            ud_heap[u.index as usize] = match u.root {
                RootType::Constants => 1,
                RootType::Table => 0,
                other => panic!("unexpected heap user data {other:?}"),
            };
        }
        let views_heap = [
            View::RawUav(32 * 2),
            View::Buffer {
                data: &heap_data,
                stride: Stride::Raw,
            },
        ];
        let out_heap = host
            .dispatch(
                &r_heap,
                &pso_heap,
                &Dispatch {
                    roots: vec![
                        (0, Root::Constants(ud_heap)),
                        (
                            1,
                            Root::View {
                                srv: true,
                                data: vec![4, 12],
                            },
                        ),
                    ],
                    views: &views_heap,
                    samplers: &[],
                    groups: [1, 1, 1],
                },
            )
            .unwrap_or_else(|e| panic!("{label}: heap dispatch failed: {e}"));
        for l in 0..32usize {
            let lu = l as u32;
            assert_eq!(
                &out_heap[0][l * 2..l * 2 + 2],
                &[0x100 + lu, 0x2000 + lu * 3],
                "{label} heap lane {l}"
            );
        }
        ran += 1;
    }
    assert!(ran > 0, "neither hardware nor WARP SM 6.6 device available");
}

/// Indexed `buffer_store_byte` and `buffer_store_short` on a pre-populated
/// `RWStructuredBuffer` UAV (stride 16, 8 elements): concurrent lanes share
/// a dword, zero bytes clear non-zero sentinel bits, a divergent `EXEC`
/// guard leaves inactive lanes' bytes/halves untouched, and untouched
/// dwords retain their initial contents.
#[test]
fn indexed_byte_and_short_buffer_stores_preserve_neighbors_and_guard() {
    const V_LSHRREV_B32: u32 = 0x16;
    const V_LSHLREV_B32: u32 = 0x1A;
    const V_AND_B32: u32 = 0x1B;
    const V_ADD_NC_U32: u32 = 0x25;
    // User data: s0 table with RWStructuredBuffer UAV (stride 16), s1 constant soffset (2).
    let mut w = vec![s_mov_b32(4, 0), s_mov_b32(5, 0x80)];
    w.extend(s_load_x4(8, 4, 0)); // s[8:11] = RWStructuredBuffer UAV (stride 16)
    w.push(S_WAITCNT_LGKM);
    // v1 = elem = lane >> 2 (0..7); v2 = lane & 3 (byte 0..3 in dword 0).
    w.push(vop2(V_LSHRREV_B32, 1, 128 + 2, 0));
    w.push(vop2(V_AND_B32, 2, 128 + 3, 0));
    // Value for byte store: v3 = lane * 5 (lane 0 writes 0x00 over non-zero sentinel!).
    w.push(vop2(V_LSHLREV_B32, 3, 128 + 2, 0)); // v3 = lane * 4
    w.push(vop2(V_ADD_NC_U32, 3, 256 + 3, 0)); // v3 = lane * 5 (lane 0 -> 0x00!)
    // 1. Concurrent byte stores to dword 0 of each element (`idxen + offen`, offset:0):
    //    addr = v[1:2] = (elem, byte_off).
    w.extend(mubuf(0x18, true, true, 0, 1, 3, 8, 0x80));
    // 2. Divergent short stores to dword 1 (`offset:4`): only lanes with (lane & 2) == 0 run.
    //    addr = v[4:5] = (elem, (lane & 1) << 1); val v6 = 0x1000 + lane * 0x11.
    w.push(0x7E08_0301); // v_mov_b32 v4, v1 (elem)
    w.push(vop2(V_AND_B32, 5, 128 + 1, 0)); // v5 = lane & 1
    w.push(vop2(V_LSHLREV_B32, 5, 128 + 1, 5)); // v5 = (lane & 1) * 2 (0 or 2)
    w.push(vop2(V_LSHLREV_B32, 6, 128 + 4, 0)); // v6 = lane << 4
    w.push(vop2(V_ADD_NC_U32, 6, 256 + 6, 0)); // v6 = lane * 17
    w.push(vop2(V_AND_B32, 7, 128 + 2, 0)); // v7 = lane & 2
    w.push(s_mov_b32(28, 0x7E)); // s28 = exec_lo
    w.push(v_cmpx_eq_u32(128, 7)); // exec_lo &= (0 == (lane & 2))
    w.extend(mubuf(0x1A, true, true, 4, 4, 6, 8, 0x80)); // buffer_store_short idxen offen offset:4
    w.push(s_mov_b32(0x7E, 28)); // restore exec_lo
    // 3. Divergent byte store to dword 3 byte 2 (`idxen` only, offset:12 + soffset s1=2 -> 14):
    //    only lanes where (lane & 7) == 4 (i.e. odd elem and lane % 4 == 0) run.
    w.push(vop2(V_AND_B32, 7, 128 + 7, 0)); // v7 = lane & 7
    w.push(s_mov_b32(28, 0x7E));
    w.push(v_cmpx_eq_u32(128 + 4, 7)); // exec_lo &= (4 == (lane & 7))
    w.extend(mubuf(0x18, true, false, 12, 1, 3, 8, 1)); // buffer_store_byte idxen offset:12 soffset:s1(2)
    w.push(s_mov_b32(0x7E, 28));
    w.push(S_ENDPGM);

    let shader = try_lower(
        &w,
        vec![param(RootType::Table, 0), param(RootType::Constants, 1)],
        vec![structured_entry(ResourceType::Uav, 16, 0)],
        8,
        32,
    )
    .expect("lower");

    let initial: Vec<u32> = (0..32u32)
        .map(|k| 0xA55A_C33C ^ k.wrapping_mul(0x0103_0507))
        .collect();
    let mut expected = initial.clone();
    for lane in 0..32u32 {
        let elem = (lane >> 2) as usize;
        let byte_off = lane & 3;
        let byte_val = (lane * 5) & 0xFF;
        // Dword 0: all 4 bytes overwritten by lanes 4*elem..4*elem+3.
        let sh = byte_off * 8;
        expected[elem * 4] = (expected[elem * 4] & !(0xFF << sh)) | (byte_val << sh);
        // Dword 1: only lanes with (lane & 2) == 0 write halves at 0 and 2.
        if (lane & 2) == 0 {
            let half_sh = (lane & 1) * 16;
            let half_val = (lane * 17) & 0xFFFF;
            expected[elem * 4 + 1] =
                (expected[elem * 4 + 1] & !(0xFFFF << half_sh)) | (half_val << half_sh);
        }
        // Dword 2: untouched sentinel!
        // Dword 3: odd elements get byte 2 overwritten by lane 4*elem.
        if (elem & 1) == 1 && byte_off == 0 {
            expected[elem * 4 + 3] = (expected[elem * 4 + 3] & !0x00FF_0000) | (byte_val << 16);
        }
    }

    let views = [View::StructuredUav {
        data: &initial,
        stride: 16,
    }];
    let _one = super::valu_tests::ONE_AT_A_TIME
        .lock()
        .unwrap_or_else(|e| e.into_inner());
    let mut ran = 0;
    for (label, warp) in [("hardware", false), ("warp", true)] {
        let Ok(host) = Host::new(warp) else {
            eprintln!("no SM 6.6 {label} device; skipped");
            continue;
        };
        let (root, pso) = host
            .compute_pipeline(&shader)
            .unwrap_or_else(|e| panic!("{label}: compute_pipeline failed: {e}"));
        let mut ud = vec![0u32; shader.user_data_dwords().max(1) as usize];
        for u in &shader.user_data {
            ud[u.index as usize] = match (u.root, u.slot) {
                (RootType::Table, 0) => 0,
                (RootType::Constants, 1) => 2,
                other => panic!("unexpected user data {other:?}"),
            };
        }
        let words = host
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
            .unwrap_or_else(|e| panic!("{label}: dispatch failed: {e}"));
        assert_eq!(words[0], expected, "{label}");
        ran += 1;
    }
    assert!(ran > 0, "neither hardware nor WARP SM 6.6 device available");
}

/// Refusal checks: unknown or non-dword-aligned stride on declared
/// structured buffers and record-limited root views are still refused.
#[test]
fn indexed_tbuffer_and_subdword_stores_refuse_unknown_stride_and_record_limit() {
    // 1. Indexed tbuffer_load_format_xyzw with stride 0 and stride 6 -> refused
    //    (store v2 to a raw UAV so the load is live and lowered).
    let mut w_tbuf = vec![
        s_mov_b32(4, 0),
        s_mov_b32(5, 0x80),
        s_mov_b32(6, 2),
        s_mov_b32(7, 0x80),
    ];
    w_tbuf.extend(s_load_x4(8, 4, 0));
    w_tbuf.extend(s_load_x4(12, 6, 0));
    w_tbuf.push(S_WAITCNT_LGKM);
    w_tbuf.extend(mtbuf(3, 56, true, false, 0, 0, 2, 8, 0x80));
    w_tbuf.push(S_WAITCNT_VM);
    w_tbuf.extend(mubuf(0x1C, false, false, 0, 0, 2, 12, 0x80));
    w_tbuf.push(S_ENDPGM);
    for bad_stride in [0u32, 6] {
        let err = try_lower(
            &w_tbuf,
            vec![param(RootType::Table, 0), param(RootType::Table, 2)],
            vec![
                structured_entry(ResourceType::Srv, bad_stride, 0),
                entry(ResourceType::Uav, 4, 2),
            ],
            6,
            32,
        )
        .expect_err("must refuse unknown stride");
        let msg = err.to_string();
        assert!(
            msg.contains("unknown stride"),
            "stride {bad_stride}: unexpected refusal {msg}"
        );
    }

    // 2. Indexed buffer_store_byte and buffer_store_short with stride 0 and stride 6 -> refused.
    for op in [0x18u32, 0x1A] {
        let mut w_st = vec![s_mov_b32(4, 0), s_mov_b32(5, 0x80)];
        w_st.extend(s_load_x4(8, 4, 0));
        w_st.push(S_WAITCNT_LGKM);
        w_st.extend(mubuf(op, true, false, 0, 0, 0, 8, 0x80));
        w_st.push(S_ENDPGM);
        for bad_stride in [0u32, 6] {
            let err = try_lower(
                &w_st,
                vec![param(RootType::Table, 0)],
                vec![structured_entry(ResourceType::Uav, bad_stride, 0)],
                4,
                32,
            )
            .expect_err("must refuse unknown stride");
            let msg = err.to_string();
            assert!(
                msg.contains("unknown stride"),
                "op {op:#x} stride {bad_stride}: unexpected refusal {msg}"
            );
        }
    }

    // 3. Indexed tbuffer_load_format_xyzw on a root view with a non-MAX record limit -> refused.
    let mut w_lim = vec![
        s_mov_b32(5, 0x80),
        s_mov_b32(0, 6),
        0x8801_FF07,
        16 << 16,
        s_mov_b32(2, 0xFF),
        64, // record limit 64 != u32::MAX
        0xBE83_03FF,
        0x1101_4FAC,
    ];
    w_lim.extend(s_load_x4(8, 4, 0));
    w_lim.push(S_WAITCNT_LGKM);
    w_lim.extend(mtbuf(3, 56, true, false, 0, 0, 2, 0, 0x80));
    w_lim.push(S_WAITCNT_VM);
    w_lim.extend(mubuf(0x1C, false, false, 0, 0, 2, 8, 0x80));
    w_lim.push(S_ENDPGM);
    let err = try_lower(
        &w_lim,
        vec![
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
        vec![
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
                stride: 16,
                dwords: 4,
                user_data: 6,
                slot: 2,
            },
        ],
        6,
        32,
    )
    .expect_err("must refuse record-limited root view");
    let msg = err.to_string();
    assert!(msg.contains("record limit"), "unexpected refusal {msg}");
}
