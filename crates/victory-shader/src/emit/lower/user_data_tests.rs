use super::*;
use crate::native::{ComputeIo, HwStage, NativeProgram, RootParam, RootType, StageIo, UsedResource};
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

fn lower_with(
    words: &[u32],
    layout: Vec<RootParam>,
    used: Vec<UsedResource>,
) -> Result<Lowered, crate::ShaderError> {
    let code: Vec<u8> = words.iter().flat_map(|w| w.to_le_bytes()).collect();
    let sgprs: u32 = layout.iter().map(|p| p.sgprs).sum();
    let native = NativeProgram {
        stage: HwStage::Cs,
        wave64: false,
        vgprs: 256,
        layout,
        used,
        io: StageIo::Cs(ComputeIo {
            rsrc2: 1 << 15 | sgprs.max(2) << 1,
            threads: [1, 1, 1],
        }),
        code,
    };
    let p = ir::build(&native.code, false, native.vgprs)?;
    let st = structure::structure(&p)?;
    let res = resource::track(&p, &native.layout, native.stage, Some(&st))?;
    let io = io::raise(&p, &native)?;
    lower(&Parts {
        native: &native,
        p: &p,
        st: &st,
        io: &io,
        res: &res,
        inputs: vec![],
        outputs: vec![],
        prims: vec![],
        mesh_stride: 0,
        target: crate::emit::Target::ANY,
        console_cf: false,
        per_lane_only: false,
        view_marks: false,
    })
}


#[test]
fn user_data_cbv_rows_and_dwords_are_cached_in_entry_block() {
    // 1) Direct user SGPRs s0..s7 (8 dwords across CBV rows 0 and 1):
    // row 0 read in entry block, row 1 first read after a branch along
    // with repeated reads of row 0.
    let direct_words = [
        0x8008_0100u32, // s_add_u32 s8, s0, s1
        0x8008_0208,    // s_add_u32 s8, s8, s2
        0x8008_0308,    // s_add_u32 s8, s8, s3
        0xBF06_8000,    // s_cmp_eq_u32 s0, 0
        0xBF85_000B,    // s_cbranch_scc1 -> s_endpgm
        0x8008_0408,    // s_add_u32 s8, s8, s4
        0x8008_0508,    // s_add_u32 s8, s8, s5
        0x8008_0608,    // s_add_u32 s8, s8, s6
        0x8008_0708,    // s_add_u32 s8, s8, s7
        0x8008_0008,    // s_add_u32 s8, s8, s0 (repeat)
        0x8008_0108,    // s_add_u32 s8, s8, s1 (repeat)
        0x8008_0208,    // s_add_u32 s8, s8, s2 (repeat)
        0x8008_0308,    // s_add_u32 s8, s8, s3 (repeat)
        0x7E02_0208,    // v_mov_b32 v1, s8
        0xD834_0000,
        0x0000_0100,    // ds_write_b32 v0, v1
        S_ENDPGM,
    ];
    // 2) Spilled root-signature loads via s[0:1] (8 dwords at indices
    // 16..24, CBV rows 4 and 5): row 4 loaded in entry block, row 5 first
    // loaded after a branch, then repeated s_load_dwordx4 / s_load_dwordx2
    // re-read the same dwords through `user_at` directly.
    let spilled_words = [
        0xF408_0100u32,
        0xFA00_0000,    // s_load_dwordx4 s[4:7], s[0:1], 0x0 (indices 16..20)
        0x8014_0504,    // s_add_u32 s20, s4, s5
        0x8014_0614,    // s_add_u32 s20, s20, s6
        0x8014_0714,    // s_add_u32 s20, s20, s7
        0xBF06_8004,    // s_cmp_eq_u32 s4, 0
        0xBF85_0013,    // s_cbranch_scc1 -> s_endpgm
        0xF408_0200,
        0xFA00_0010,    // s_load_dwordx4 s[8:11], s[0:1], 0x10 (indices 20..24)
        0xF408_0300,
        0xFA00_0000,    // s_load_dwordx4 s[12:15], s[0:1], 0x0 (repeat 16..20)
        0xF404_0400,
        0xFA00_0014,    // s_load_dwordx2 s[16:17], s[0:1], 0x14 (repeat 21..23)
        0x8014_0814,    // s_add_u32 s20, s20, s8
        0x8014_0914,    // s_add_u32 s20, s20, s9
        0x8014_0A14,    // s_add_u32 s20, s20, s10
        0x8014_0B14,    // s_add_u32 s20, s20, s11
        0x8014_0C14,    // s_add_u32 s20, s20, s12
        0x8014_0D14,    // s_add_u32 s20, s20, s13
        0x8014_0E14,    // s_add_u32 s20, s20, s14
        0x8014_0F14,    // s_add_u32 s20, s20, s15
        0x8014_1014,    // s_add_u32 s20, s20, s16
        0x8014_1114,    // s_add_u32 s20, s20, s17
        0x7E02_0214,    // v_mov_b32 v1, s20
        0xD834_0000,
        0x0000_0100,    // ds_write_b32 v0, v1
        S_ENDPGM,
    ];

    for (label, words, user_data, expected_rows) in [
        ("direct", &direct_words[..], 0u8, [0u32, 1]),
        ("spilled", &spilled_words[..], 16u8, [4u32, 5]),
    ] {
        let layout = vec![RootParam {
            sgprs: 8,
            ..root(RootType::Constants, user_data, 0)
        }];
        // Where our stack's compute prologue reads the dispatch pair from:
        // the dword above the layout's last (module notes, "group spread").
        let dispatch_at = layout.iter().map(|p| u32::from(p.user_data) + p.sgprs).max().unwrap();
        let emitted = lower_with(words, layout, vec![]).unwrap();
        // Every compute program we lower also reads the dispatch pair (the
        // row the host spreads groups with, and how many it asked for) at
        // `dispatch_at`, slot [`super::super::DISPATCH_SLOT`]: #2207's
        // counts are the program's own dwords; the pair is checked apart.
        let dispatch: Vec<u32> = emitted
            .user_data
            .iter()
            .filter(|u| u.slot == super::super::DISPATCH_SLOT)
            .map(|u| u.index)
            .collect();
        assert_eq!(
            dispatch,
            vec![dispatch_at, dispatch_at + 1],
            "{label}: the dispatch pair above the layout"
        );
        assert_eq!(
            emitted.user_data.len() - dispatch.len(),
            8,
            "{label}: 8 distinct dwords of the program's own"
        );

        let stats = emitted.cbuf_stats;
        assert!(
            stats.blocks >= 3,
            "{label}: expected multi-block shader, got {} blocks",
            stats.blocks
        );
        // The pair is a row of its own, loaded in the entry block as the
        // program's are.
        let dispatch_row = dispatch_at / 4;
        assert_eq!(
            stats.loads,
            vec![(0, dispatch_row), (0, expected_rows[0]), (0, expected_rows[1])],
            "{label}: expected 3 entry-block cbufferLoadLegacy.i32 calls (the dispatch row, then the program's 2)"
        );
        assert_eq!(
            stats.extracts.len(),
            8 + dispatch.len(),
            "{label}: expected 10 cached extractvalue ops, got {:?}",
            stats.extracts
        );
        assert!(
            stats.extracts.iter().all(|&(b, _)| b == 0),
            "{label}: all user-data extracts must live in entry block 0: {:?}",
            stats.extracts
        );

        #[cfg(windows)]
        {
            let path = super::super::validator::Validator::default_path();
            match super::super::validator::Validator::load(path.as_deref()) {
                Ok(mut v) => v
                    .validate(&mut emitted.dxil.clone())
                    .unwrap_or_else(|e| panic!("{label}: dxil.dll rejected shader: {e}")),
                Err(e) if std::env::var_os("VICTORY_DXIL").is_some() => {
                    panic!("{label}: VICTORY_DXIL set but validator failed to load: {e}");
                }
                Err(_) => {}
            }
        }
    }
}

#[test]
#[ignore = "benchmark"]
fn bench_user_data_cache() {
    // 32 spilled user-data dwords (8 CBV rows) loaded 4x = 128 reads.
    let mut words = Vec::new();
    for pass in 0..4u32 {
        for row in 0..8u32 {
            let dst_base = 4 + ((row & 3) * 4);
            let byte_off = row * 16;
            // s_load_dwordx4 s[dst_base..dst_base+4], s[0:1], byte_off
            words.push(0xF408_0000 | (dst_base << 6));
            words.push(0xFA00_0000 | byte_off);
            for k in 0..4u32 {
                let s = dst_base + k;
                if pass == 0 && row == 0 && k == 0 {
                    words.push(0xBE94_0300 | s); // s_mov_b32 s20, s
                } else {
                    words.push(0x8014_0014 | (s << 8)); // s_add_u32 s20, s20, s
                }
            }
        }
    }
    words.push(0x7E02_0214); // v_mov_b32 v1, s20
    words.push(0xD834_0000);
    words.push(0x0000_0100); // ds_write_b32 v0, v1
    words.push(S_ENDPGM);
    let layout = vec![RootParam {
        sgprs: 32,
        ..root(RootType::Constants, 16, 0)
    }];
    let emitted = lower_with(&words, layout.clone(), vec![]).unwrap();
    let t0 = std::time::Instant::now();
    for _ in 0..20 {
        let e = lower_with(&words, layout.clone(), vec![]).unwrap();
        std::hint::black_box(e.dxil.len());
    }
    let us = t0.elapsed().as_secs_f64() * 1e6 / 20.0;
    eprintln!(
        "user_data 128 reads (32 dwords / 8 rows): dxil={} B, loads={}, extracts={}, {:.1} us/run",
        emitted.dxil.len(),
        emitted.cbuf_stats.loads.len(),
        emitted.cbuf_stats.extracts.len(),
        us
    );
}
