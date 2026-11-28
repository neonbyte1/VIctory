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
fn subdword_buffer_stores_emit_cas_loop_and_validate() {
    for (name, mubuf_op) in [
        ("buffer_store_byte", 0x18u32),
        ("buffer_store_short", 0x1Au32),
    ] {
        for guarded in [false, true] {
            for view_marks in [false, true] {
                // User data: s0 a descriptor table holding a raw buffer UAV.
                let mut words = vec![
                    0xBE84_0300u32, // s_mov_b32 s4, s0
                    0xBE85_0380,    // s_mov_b32 s5, 0
                    0xF408_0202,
                    0xFA00_0000,    // s_load_dwordx4 s[8:11], s[4:5], 0x0
                    0xBF8C_C07F,    // s_waitcnt lgkmcnt(0)
                    0x3402_0081,    // v_lshlrev_b32 v1, 1, v0
                ];
                if guarded {
                    words.push(0x7DA8_0084); // v_cmpx_gt_u32 exec_lo, 4, v0
                }
                // MUBUF buffer_store_{byte,short} v0, v1, s[8:11], 0 offen
                words.push(0x38 << 26 | mubuf_op << 18 | 1 << 12);
                words.push(0x80 << 24 | (8 >> 2) << 16 | 1);
                if guarded {
                    words.push(0xBEFE_03C1); // s_mov_b32 exec_lo, -1
                }
                words.push(S_ENDPGM);

                let code: Vec<u8> = words.iter().flat_map(|w| w.to_le_bytes()).collect();
                let native = NativeProgram {
                    stage: HwStage::Cs,
                    wave64: false,
                    vgprs: 8,
                    layout: vec![root(RootType::Table, 0, 0)],
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
                        rsrc2: 2 << 1,
                        threads: [32, 1, 1],
                    }),
                    code,
                };
                let p = ir::build(&native.code, false, 8).unwrap();
                assert!(
                    p.blocks
                        .iter()
                        .flat_map(|b| &b.insts)
                        .any(|i| i.name() == name),
                    "expected {name} in disassembled IR"
                );
                let st = structure::structure(&p).unwrap();
                let res =
                    resource::track(&p, &native.layout, native.stage, Some(&st)).unwrap();
                let io = io::raise(&p, &native).unwrap();
                let lowered = lower(&Parts {
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
                    view_marks,
                })
                .unwrap_or_else(|e| {
                    panic!("{name} (guarded={guarded}, view_marks={view_marks}): {e}")
                });

                assert_eq!(
                    lowered.cas_loops, 1,
                    "{name} (guarded={guarded}, view_marks={view_marks}): expected 1 CAS loop"
                );
                assert_eq!(
                    lowered.atomic_cas_ops, 1,
                    "{name} (guarded={guarded}, view_marks={view_marks}): expected 1 atomicCompareExchange.i32"
                );
                if !view_marks {
                    assert_eq!(
                        lowered.atomic_bin_ops, 0,
                        "{name} (guarded={guarded}): target buffer store must not emit atomicBinOp.i32"
                    );
                    assert_eq!(
                        lowered.raw_buffer_loads, 1,
                        "{name} (guarded={guarded}): expected initial rawBufferLoad.f32 before CAS loop"
                    );
                }
                let container = crate::Container::parse(&lowered.dxil).unwrap();
                let sfi0 = container.part(crate::FourCC::SFI0).unwrap();
                let features = u64::from_le_bytes(sfi0.data.try_into().unwrap());
                assert_eq!(
                    features & 0x800,
                    0,
                    "{name}: raw UAV load in CAS loop must not require TYPED_UAV_LOADS"
                );

                #[cfg(windows)]
                {
                    let path = crate::emit::validator::Validator::default_path();
                    match crate::emit::validator::Validator::load(path.as_deref()) {
                        Ok(mut v) => {
                            let mut copy = lowered.dxil.clone();
                            let ours = copy[4..20].to_vec();
                            v.validate(&mut copy).unwrap_or_else(|e| {
                                panic!(
                                    "{name} (guarded={guarded}, view_marks={view_marks}): DXIL validation failed: {e}"
                                )
                            });
                            assert_eq!(
                                copy[4..20],
                                ours[..],
                                "{name}: container hash mismatch"
                            );
                        }
                        Err(e) if std::env::var_os("VICTORY_DXIL").is_some() => {
                            panic!("VICTORY_DXIL is set but dxil.dll failed to load: {e}");
                        }
                        Err(_) => {}
                    }
                }
            }
        }
    }
}
