//! Cross-device differential fuzzing of single VALU instructions.
//!
//! Every VOP3- and VOP3P-encoded GFX10 opcode the translator lowers runs as a one-thread
//! compute shader on every hardware adapter and on WARP, on the same inputs:
//! special floats (signed zeros, denormals, infinities, quiet and signalling
//! NaNs), integer edges, f16 edges in either half, and pseudo-random dwords.
//! The results must agree bit for bit (any NaN matching any NaN) unless
//! the operation is approximate by definition (`rcp`, `rsq`, `sqrt`, `log`,
//! `exp`, `sin`, `cos`), where a few ULPs are allowed. A disagreement means the
//! translation leans on behaviour DXIL leaves to the device, so a title can
//! compute different values on different GPUs.
//!
//! Ignored by default: run with
//! `cargo test -p victory-shader --lib fuzz_tests -- --ignored --nocapture`.
//! `VICTORY_FUZZ_CASES` sets the inputs per opcode (default 160),
//! `VICTORY_FUZZ_OPS` limits the run to names containing one of its
//! comma-separated parts, `VICTORY_FUZZ_MODS=1` also tries float VOP3 opcodes
//! with source and output modifiers (abs, neg, clamp, omod), and
//! `VICTORY_FUZZ_OUT` names a Markdown report.
//! Each finding lists its simplest disagreeing inputs first, with the
//! instruction words and inputs a regression test needs. With fewer than two
//! devices (CI's runners have WARP only) there is nothing to compare, and the
//! test returns.

use super::host::{Dispatch, Host, Root, View};
use super::{HostShader, lower, validator};
use crate::native::{
    ComputeIo, HwStage, NativeProgram, ResourceType, RootParam, RootType, StageIo, UsedResource,
};
use crate::{io, ir, range, resource};
use std::fmt::Write as _;

const S_ENDPGM: u32 = 0xBF81_0000;

const fn s_mov_b32(sdst: u8, src0: u8) -> u32 {
    0xBE80_0000 | (sdst as u32) << 16 | 0x03 << 8 | src0 as u32
}

const fn s_load_dwordx4(dst: u8, base: u8) -> [u32; 2] {
    [
        0x3D << 26 | 0x02 << 18 | (dst as u32) << 6 | (base as u32 >> 1),
        125 << 25,
    ]
}

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

const fn buffer_store_dword(vdata: u8, srsrc: u8) -> [u32; 2] {
    [
        0x38 << 26 | 0x1C << 18,
        0x80 << 24 | ((srsrc as u32 >> 2) & 0x1F) << 16 | (vdata as u32) << 8,
    ]
}

const fn vgpr(n: u8) -> u16 {
    256 + n as u16
}

/// Opcodes at and above this are VOP3P (packed), less it.
const PACKED: u16 = 0x1000;

/// VOP3P `op vdst, src0, src1, src2`, each half from its own half
/// (op_sel 0, op_sel_hi 1), no modifiers.
const fn vop3p(op: u16, vdst: u8, src0: u16, src1: u16, src2: u16) -> [u32; 2] {
    [
        0x33 << 26 | (op as u32 & 0x7F) << 16 | 1 << 14 | vdst as u32,
        (src0 as u32 & 0x1FF) | (src1 as u32 & 0x1FF) << 9 | (src2 as u32 & 0x1FF) << 18 | 3 << 27,
    ]
}

/// The instruction words of `op` (VOP3, or VOP3P from [`PACKED`]) reading
/// v1, v2, v3 (VOPC: v1, v2) and leaving its result in v0.
fn body(op: u16) -> Vec<u32> {
    if op >= PACKED {
        vop3p(op - PACKED, 0, vgpr(1), vgpr(2), vgpr(3)).to_vec()
    } else if op < 0x100 {
        // A compare writes s[20:21]; v_cndmask_b32 v0, 0, 1, s[20:21].
        let mut w = vop3(op, 20, vgpr(1), vgpr(2), 0x80).to_vec();
        w.extend(vop3(0x101, 0, 0x80, 0x81, 20));
        w
    } else {
        vop3(op, 0, vgpr(1), vgpr(2), vgpr(3)).to_vec()
    }
}

/// The machine name of `op` (see [`body`]), if GFX10 has one.
fn name_of(op: u16) -> Option<&'static str> {
    let words = body(op);
    let code: Vec<u8> = [words[0], words[1], S_ENDPGM]
        .iter()
        .flat_map(|w| w.to_le_bytes())
        .collect();
    let p = ir::build(&code, false, 16).ok()?;
    let name = p.blocks.first()?.insts.first()?.name();
    (!name.is_empty()).then_some(name)
}

/// How a report names `op`'s encoding.
fn encoding(op: u16) -> String {
    if op >= PACKED {
        format!("VOP3P {:#04x}", op - PACKED)
    } else {
        format!("VOP3 {op:#05x}")
    }
}
/// Whether the fuzzer leaves `name` out: lane, EXEC and interpolation
/// instructions, and ones with no single-dword result to compare.
fn skipped(name: &str) -> bool {
    const PARTS: &[&str] = &[
        "v_cmpx",
        "lane",
        "interp",
        "v_nop",
        "v_clrexcp",
        "mbcnt",
        "v_movrel",
        "v_swap",
        "pipeflush",
        "v_mov_b64",
        "trig_preop",
        "readfirstlane",
    ];
    PARTS.iter().any(|p| name.contains(p))
}

/// Results that are approximate on every device: the bits may differ by a
/// few ULPs between correct implementations.
fn approximate(name: &str) -> bool {
    ["rcp", "rsq", "sqrt", "log", "exp_", "sin", "cos"]
        .iter()
        .any(|p| name.contains(p))
}

/// Builds the one-thread shader for `words`: s0..s3 carry the four input
/// dwords into v1..v4, the body runs, and v0 goes to dword 0 of the raw UAV
/// at heap index 0.
fn build(words: &[u32]) -> Result<HostShader, String> {
    let mut w = vec![s_mov_b32(5, 0x80)];
    w.extend(s_load_dwordx4(8, 4));
    for i in 0..4u8 {
        w.push(vop1(0x001, i + 1, u16::from(i)));
    }
    // v0 = v4: a defined accumulator for the opcodes that read their
    // destination (v_mac_*, v_fmac_*).
    w.push(vop1(0x001, 0, vgpr(4)));
    w.extend_from_slice(words);
    w.extend(buffer_store_dword(0, 8));
    w.push(S_ENDPGM);
    let code: Vec<u8> = w.iter().flat_map(|w| w.to_le_bytes()).collect();
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
    let (p, st) = range::build(&native.code, native.wave64, native.vgprs)
        .map_err(|e| format!("ir: {e:?}"))?;
    let st = st.map_err(|e| format!("structure: {e:?}"))?;
    let res = resource::track(&p, &native.layout, native.stage, Some(&st))
        .map_err(|e| format!("resource: {e:?}"))?;
    let io = io::raise(&p, &native).map_err(|e| format!("io: {e:?}"))?;
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
    .map_err(|e| format!("lower: {e:?}"))?;
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
        v.validate(&mut copy)
            .map_err(|e| format!("DXIL validation: {e}"))?;
    }
    Ok(shader)
}

/// xorshift64*: the same inputs on every run.
struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u32 {
        self.0 ^= self.0 >> 12;
        self.0 ^= self.0 << 25;
        self.0 ^= self.0 >> 27;
        (self.0.wrapping_mul(0x2545_F491_4F6C_DD1D) >> 32) as u32
    }
    fn pick<T: Copy>(&mut self, from: &[T]) -> T {
        from[self.next() as usize % from.len()]
    }
}

const F32_EDGES: &[u32] = &[
    0x0000_0000,
    0x8000_0000,
    0x3F80_0000,
    0xBF80_0000,
    0x3F00_0000,
    0x4000_0000,
    0x4040_0000,
    0x3FC0_0000,
    0xC020_0000,
    0x7F7F_FFFF,
    0xFF7F_FFFF,
    0x0080_0000,
    0x8080_0000,
    0x0000_0001,
    0x007F_FFFF,
    0x8000_0001,
    0x807F_FFFF,
    0x7F80_0000,
    0xFF80_0000,
    0x7FC0_0000,
    0x7F80_0001,
    0xFFC0_0000,
    0x7FFF_FFFF,
    0x5014_F8B6,
    0x2EDB_E6FF,
    0x4049_0FDB,
    0x4B80_0000,
    0x4F00_0000,
    0x4F80_0000,
    0xCF00_0000,
    0x3F7F_FFFF,
    0x3F80_0001,
    0x4B7F_FFFF,
    0x3EFF_FFFF,
    0x4E80_0000,
];

const INT_EDGES: &[u32] = &[
    0,
    1,
    2,
    3,
    15,
    16,
    31,
    32,
    33,
    63,
    64,
    0x7F,
    0x80,
    0xFF,
    0x100,
    0x7FFF,
    0x8000,
    0xFFFF,
    0x1_0000,
    0x7FFF_FFFF,
    0x8000_0000,
    0x8000_0001,
    0xFFFF_FFFF,
    0xFFFF_FFFE,
    0xFFFF_FF80,
    0xFFFF_8000,
    0x0001_0001,
    0xDEAD_BEEF,
    0x0F0F_0F0F,
    0xAAAA_AAAA,
];

const F16_EDGES: &[u16] = &[
    0x0000, 0x8000, 0x3C00, 0xBC00, 0x3800, 0x4000, 0x7BFF, 0xFBFF, 0x0400, 0x8400, 0x0001, 0x03FF,
    0x8001, 0x83FF, 0x7C00, 0xFC00, 0x7E00, 0x7C01, 0xFE00, 0x3BFF, 0x3C01, 0x5640, 0x1400,
];

fn input(rng: &mut Rng) -> u32 {
    match rng.next() % 6 {
        0 | 1 => rng.pick(F32_EDGES),
        2 => rng.pick(INT_EDGES),
        3 => {
            let (lo, hi) = (rng.pick(F16_EDGES), rng.pick(F16_EDGES));
            u32::from(lo) | u32::from(hi) << 16
        }
        4 => u32::from(rng.pick(F16_EDGES)) | (rng.next() & 0xFFFF_0000),
        _ => rng.next(),
    }
}

fn is_f32_nan(x: u32) -> bool {
    x & 0x7FFF_FFFF > 0x7F80_0000
}

fn is_f16_nan(x: u32) -> bool {
    x & 0x7FFF > 0x7C00
}

/// Whether `a` and `b` agree as results of `name`.
fn agree(name: &str, a: u32, b: u32) -> bool {
    if a == b {
        return true;
    }
    if name == "v_fma_mixhi_f16" {
        // The f16 result is the high half; the low half is kept.
        let (x, y) = (a >> 16, b >> 16);
        return a & 0xFFFF == b & 0xFFFF && (x == y || (is_f16_nan(x) && is_f16_nan(y)));
    }
    if (name.starts_with("v_pk_") && name.ends_with("_f16")) || name == "v_cvt_pkrtz_f16_f32" {
        // Two f16 results: each half equal, or NaN in both.
        return [0, 16].iter().all(|s| {
            let (x, y) = (a >> s & 0xFFFF, b >> s & 0xFFFF);
            x == y || (is_f16_nan(x) && is_f16_nan(y))
        });
    }
    // The result's type is the first in the name (v_cvt_f16_f32,
    // v_dot2_f32_f16).
    const TYPES: &[&str] = &[
        "f16", "f32", "f64", "i16", "u16", "i32", "u32", "i64", "u64", "b16", "b32", "b64",
    ];
    let half = name.split('_').find(|t| TYPES.contains(t)) == Some("f16");
    let float = name.contains("_f") || name.contains("frexp") || name.contains("ldexp");
    if float && !half && is_f32_nan(a) && is_f32_nan(b) {
        return true;
    }
    if half && is_f16_nan(a) && is_f16_nan(b) && a >> 16 == b >> 16 {
        return true;
    }
    if approximate(name) {
        let ulps = if half { 1 } else { 4 };
        let (a, b) = if half {
            (a & 0xFFFF, b & 0xFFFF)
        } else {
            (a, b)
        };
        let sign = if half { 0x8000 } else { 0x8000_0000 };
        return a & sign == b & sign && a.abs_diff(b) <= ulps;
    }
    false
}

struct Finding {
    op: u16,
    name: &'static str,
    variant: &'static str,
    words: Vec<u32>,
    cases: usize,
    mismatches: Vec<([u32; 4], Vec<u32>)>,
}

/// VOP3 source and output modifiers: (label, abs bits, neg bits, clamp,
/// omod). Tried on float VOP3 opcodes with `VICTORY_FUZZ_MODS=1`.
const MODS: &[(&str, u32, u32, bool, u32)] = &[
    ("abs src0, neg src1", 0b001, 0b010, false, 0),
    ("neg src0 and src2", 0, 0b101, false, 0),
    ("abs src1 and src2", 0b110, 0, false, 0),
    ("clamp", 0, 0, true, 0),
    ("omod *2", 0, 0, false, 1),
    ("omod *4", 0, 0, false, 2),
    ("omod /2, clamp", 0, 0, true, 3),
];

/// `words` (one VOP3 instruction) with these modifiers.
fn with_mods(words: &[u32], abs: u32, neg: u32, clamp: bool, omod: u32) -> Vec<u32> {
    let mut w = words.to_vec();
    w[0] |= abs << 8 | u32::from(clamp) << 15;
    w[1] |= omod << 27 | neg << 29;
    w
}

#[test]
#[ignore = "cross-device fuzzing: minutes on a GPU; run with --ignored"]
fn valu_hardware_and_warp_agree() {
    let cases = std::env::var("VICTORY_FUZZ_CASES")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(160usize);
    let mods = std::env::var("VICTORY_FUZZ_MODS").is_ok_and(|v| v == "1");
    let only: Vec<String> = std::env::var("VICTORY_FUZZ_OPS")
        .map(|v| v.split(',').map(str::to_string).collect())
        .unwrap_or_default();
    let _one = super::valu_tests::ONE_AT_A_TIME
        .lock()
        .unwrap_or_else(|e| e.into_inner());

    let mut ops: Vec<(u16, &'static str)> = (0u16..0x400)
        .chain(PACKED..PACKED + 0x80)
        .filter_map(|op| name_of(op).map(|n| (op, n)))
        .filter(|(_, n)| n.starts_with("v_") && !skipped(n))
        .filter(|(_, n)| only.is_empty() || only.iter().any(|o| n.contains(o.as_str())))
        .collect();
    ops.dedup_by_key(|o| o.1);

    // Every lowered opcode with its inputs.
    let mut refused = Vec::new();
    let mut work = Vec::new();
    let mut variants: Vec<(u16, &'static str, &'static str, Vec<u32>)> = Vec::new();
    for &(op, name) in &ops {
        variants.push((op, name, "", body(op)));
        let float = name.contains("_f32") || name.contains("_f16");
        if mods && float && (0x100..PACKED).contains(&op) {
            for &(label, abs, neg, clamp, omod) in MODS {
                variants.push((op, name, label, with_mods(&body(op), abs, neg, clamp, omod)));
            }
        }
    }
    for (op, name, variant, words) in variants {
        match build(&words) {
            Ok(shader) => {
                let mut rng = Rng(0x9E37_79B9_7F4A_7C15 ^ u64::from(op));
                let inputs: Vec<[u32; 4]> = (0..cases)
                    .map(|_| {
                        [
                            input(&mut rng),
                            input(&mut rng),
                            input(&mut rng),
                            input(&mut rng),
                        ]
                    })
                    .collect();
                work.push((op, name, variant, words, shader, inputs));
            }
            Err(e) => refused.push((name, format!("{variant}: {e}"))),
        }
    }

    // Every hardware adapter, then WARP; one device at a time (two at once
    // suspend WARP), each over every opcode.
    let mut devices: Vec<(String, Option<u32>)> = hardware_adapters()
        .into_iter()
        .map(|(i, n)| (n, Some(i)))
        .collect();
    devices.push(("WARP".to_string(), None));
    let mut lost = Vec::new();
    let mut results: Vec<Vec<Option<Vec<u32>>>> = Vec::new();
    let mut labels = Vec::new();
    for (label, adapter) in &devices {
        let Ok(h) = device(*adapter) else {
            eprintln!("{label}: no SM 6.6 device; skipped");
            continue;
        };
        let mut host = Some(h);
        let mut column = Vec::new();
        for (_, name, _, _, shader, inputs) in &work {
            let run = (|| -> Result<Vec<u32>, String> {
                let host = host.as_ref().ok_or("no device")?;
                let (root, pso) = host.compute_pipeline(shader)?;
                let mut got = Vec::with_capacity(inputs.len());
                for inp in inputs {
                    let mut ud = vec![0u32; shader.user_data_dwords().max(1) as usize];
                    for u in &shader.user_data {
                        ud[u.index as usize] = match (u.root, u.slot) {
                            (RootType::Constants, 0) => inp[u.dword as usize],
                            _ => 0,
                        };
                    }
                    let out = host.dispatch(
                        &root,
                        &pso,
                        &Dispatch {
                            roots: vec![(0, Root::Constants(ud))],
                            views: &[View::RawUav(1)],
                            samplers: &[],
                            groups: [1, 1, 1],
                        },
                    )?;
                    got.push(out[0][0]);
                }
                Ok(got)
            })();
            match run {
                Ok(v) => column.push(Some(v)),
                Err(e) => {
                    lost.push((*name, label.clone(), e));
                    column.push(None);
                    // Release every reference first: D3D12 hands back the
                    // same removed device while one is alive.
                    drop(host.take());
                    host = device(*adapter).ok();
                }
            }
        }
        results.push(column);
        labels.push(label.clone());
    }
    if labels.len() < 2 {
        // CI's runners have WARP only: nothing to compare it with.
        eprintln!(
            "fewer than two SM 6.6 devices ({}); nothing to compare",
            labels.join(", ")
        );
        return;
    }

    let mut findings = Vec::new();
    let mut tested = 0;
    for (i, (op, name, variant, words, _, inputs)) in work.iter().enumerate() {
        let cols: Vec<&Vec<u32>> = results.iter().filter_map(|c| c[i].as_ref()).collect();
        if cols.len() != results.len() {
            continue;
        }
        tested += 1;
        let mismatches: Vec<([u32; 4], Vec<u32>)> = inputs
            .iter()
            .enumerate()
            .filter(|&(j, _)| cols.iter().any(|c| !agree(name, c[j], cols[0][j])))
            .map(|(j, inp)| (*inp, cols.iter().map(|c| c[j]).collect()))
            .collect();
        if !mismatches.is_empty() {
            let mut mismatches = mismatches;
            // Simplest inputs first: fewest non-zero dwords, then smallest.
            mismatches.sort_by_key(|(i, _)| (i.iter().filter(|&&x| x != 0).count(), *i));
            eprintln!(
                "{name} {variant} ({}): {} of {cases} differ",
                encoding(*op),
                mismatches.len()
            );
            findings.push(Finding {
                op: *op,
                name,
                variant,
                words: words.clone(),
                cases,
                mismatches,
            });
        }
    }

    let mut md = String::new();
    let _ = writeln!(
        md,
        "# VALU cross-device fuzz\n\nDevices: {}.\n\n{tested} opcodes compared, {cases} inputs \
         each; {} differ; {} refused by the translator; {} lost a device.\n",
        labels.join(", "),
        findings.len(),
        refused.len(),
        lost.len()
    );
    for f in &findings {
        let _ = writeln!(
            md,
            "## `{}`{} ({}): {} of {} differ\n\n| v1 | v2 | v3 | v4 | {} |\n|---|---|---|---|{}",
            f.name,
            if f.variant.is_empty() {
                String::new()
            } else {
                format!(", {}", f.variant)
            },
            encoding(f.op),
            f.mismatches.len(),
            f.cases,
            labels.join(" | "),
            "---|".repeat(labels.len())
        );
        for (i, outs) in f.mismatches.iter().take(8) {
            let outs: Vec<String> = outs.iter().map(|x| format!("`{x:#010x}`")).collect();
            let _ = writeln!(
                md,
                "| `{:#010x}` | `{:#010x}` | `{:#010x}` | `{:#010x}` | {} |",
                i[0],
                i[1],
                i[2],
                i[3],
                outs.join(" | ")
            );
        }
        // A regression test needs only these: the instruction, and the
        // first (simplest) input in v1..v4 (see `build`).
        if let Some((i, _)) = f.mismatches.first() {
            let words: Vec<String> = f.words.iter().map(|w| format!("{w:#010x}")).collect();
            let _ = writeln!(
                md,
                "\nRepro: `build(&[{}])` with v1..v4 = `[{:#010x}, {:#010x}, {:#010x}, {:#010x}]`.",
                words.join(", "),
                i[0],
                i[1],
                i[2],
                i[3]
            );
        }
        md.push('\n');
    }
    if !lost.is_empty() {
        md.push_str("## Device lost or dispatch failed\n\n");
        for (n, l, e) in &lost {
            let _ = writeln!(md, "- `{n}` on {l}: {e}");
        }
        md.push('\n');
    }
    md.push_str("## Refused by the translator\n\n");
    for (n, e) in &refused {
        let _ = writeln!(md, "- `{n}`: {}", e.lines().next().unwrap_or(""));
    }
    match std::env::var_os("VICTORY_FUZZ_OUT") {
        Some(p) => std::fs::write(p, &md).expect("write the report"),
        None => println!("{md}"),
    }
    eprintln!(
        "{tested} opcodes compared on {}, {} differ, {} refused, {} lost",
        labels.join(", "),
        findings.len(),
        refused.len(),
        lost.len()
    );
}

/// The hardware adapters (index, description): as `host.rs`'s
/// `hardware_adapter`, not flagged software and with video memory of their
/// own, so CI's basic render driver is left out with WARP.
fn hardware_adapters() -> Vec<(u32, String)> {
    use windows::Win32::Graphics::Dxgi::*;
    let mut out = Vec::new();
    // SAFETY: plain DXGI calls with valid out pointers.
    unsafe {
        let Ok(factory) = CreateDXGIFactory2::<IDXGIFactory4>(DXGI_CREATE_FACTORY_FLAGS(0)) else {
            return out;
        };
        let mut i = 0;
        while let Ok(a) = factory.EnumAdapters1(i) {
            if let Ok(d) = a.GetDesc1()
                && d.Flags & DXGI_ADAPTER_FLAG_SOFTWARE.0 as u32 == 0
                && d.DedicatedVideoMemory > 0
            {
                let n = d.Description.iter().take_while(|&&c| c != 0).count();
                out.push((i, String::from_utf16_lossy(&d.Description[..n])));
            }
            i += 1;
        }
    }
    out
}

/// A device on hardware adapter `adapter`, or on WARP.
fn device(adapter: Option<u32>) -> Result<Host, String> {
    use windows::Win32::Graphics::Direct3D::D3D_FEATURE_LEVEL_12_0;
    use windows::Win32::Graphics::Direct3D12::*;
    use windows::Win32::Graphics::Dxgi::*;
    let Some(i) = adapter else {
        return Host::new(true);
    };
    // SAFETY: plain D3D12/DXGI calls with valid out pointers.
    unsafe {
        let factory: IDXGIFactory4 =
            CreateDXGIFactory2(DXGI_CREATE_FACTORY_FLAGS(0)).map_err(|e| e.to_string())?;
        let a = factory.EnumAdapters1(i).map_err(|e| e.to_string())?;
        let mut device: Option<ID3D12Device> = None;
        D3D12CreateDevice(&a, D3D_FEATURE_LEVEL_12_0, &mut device).map_err(|e| e.to_string())?;
        let device = device.ok_or("no device")?;
        let mut sm = D3D12_FEATURE_DATA_SHADER_MODEL {
            HighestShaderModel: D3D_SHADER_MODEL_6_6,
        };
        device
            .CheckFeatureSupport(
                D3D12_FEATURE_SHADER_MODEL,
                (&raw mut sm).cast(),
                size_of_val(&sm) as u32,
            )
            .map_err(|e| e.to_string())?;
        if sm.HighestShaderModel.0 < D3D_SHADER_MODEL_6_6.0 {
            return Err("below SM 6.6".into());
        }
        Ok(Host { device })
    }
}
