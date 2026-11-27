//! Random GFX10 programs through the whole translation: SSA and
//! structurisation with bounded relative moves (`range`), resource
//! tracking, I/O and DXIL emission. Each program translates or is refused
//! with a reason, and a program that translates translates to the same
//! bytes again. Half the programs are made with write marks (odd seeds),
//! as the title's pipelines are, half without.
//!
//! The title translates its shaders when it creates pipelines, in its own
//! process: a panic in any of these stages ends the title, and translation
//! that differs between runs defeats the shader cache and the pipeline
//! library. Several earlier panics were found this way (see the tests that
//! say "found by random-input testing"); this keeps the search running on
//! every change.
//!
//! The programs are random but plausible, so that most get past decoding:
//! opcodes from our tables with small register numbers, descriptors loaded
//! from the root tables first, branches that land on an instruction a few
//! before or after their own (loops among them), and a store or export at
//! the end that needs what the body computes. Compute programs run as 32
//! or 64 lanes, pixel programs read one interpolated input. About one
//! program in three has a relative move, and in half of those `range`
//! proves a bound on M0 and builds the program again.
//!
//! `VICTORY_RANDOM_PROGRAMS=<n>` sets how many programs run (4,096 by
//! default, a few seconds in a debug build) and `VICTORY_RANDOM_SEED=<s>`
//! the first program's seed; a failure names the seeds of the programs that
//! failed and lists the first. A stack overflow aborts the test before it
//! can: `VICTORY_RANDOM_TRACE=1` prints each seed before its program runs.
//! A long search runs best in a release build with overflow checks:
//!
//! ```text
//! CARGO_PROFILE_RELEASE_OVERFLOW_CHECKS=true VICTORY_RANDOM_PROGRAMS=1000000 \
//!     cargo test --release -p victory-shader random_programs
//! ```

use super::container::{self, SigElement};
use crate::dxbc::{self, Container, FourCC};
use crate::gfx10::{self, Encoding};
use crate::native::{
    ComputeIo, HwStage, NativeProgram, PsInput, ResourceType, RootParam, RootType, StageIo, Usage,
    UsedResource,
};
use crate::{io, range, resource};
use std::io::Write;

const DEFAULT_PROGRAMS: u64 = 4096;
const S_ENDPGM: u32 = 0xBF81_0000;
/// Operand codes: `vcc_lo`, `m0`, `null`, `exec_lo`, `scc`, a literal.
const VCC: u32 = 106;
const M0: u32 = 124;
const NULL: u32 = 125;
const EXEC: u32 = 126;
const SCC: u32 = 253;
const LITERAL: u32 = 255;
/// A branch whose offset is filled in once every instruction's place is known.
const BRANCH_PENDING: u32 = 0xDEAD;

/// xorshift64, seeded through splitmix64 so neighbouring seeds diverge.
struct Rng(u64);

impl Rng {
    fn new(seed: u64) -> Rng {
        let mut z = seed.wrapping_add(0x9E37_79B9_7F4A_7C15);
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        Rng((z ^ (z >> 31)) | 1)
    }

    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }

    fn word(&mut self) -> u32 {
        self.next() as u32
    }

    /// 0 to `n - 1`.
    fn below(&mut self, n: u32) -> u32 {
        (self.next() % u64::from(n.max(1))) as u32
    }

    fn percent(&mut self, p: u32) -> bool {
        self.below(100) < p
    }

    fn opcode(&mut self, encoding: Encoding) -> u32 {
        let t = gfx10::opcodes(encoding);
        u32::from(t[self.below(t.len() as u32) as usize].0)
    }

    fn vgpr(&mut self) -> u32 {
        self.below(16)
    }

    /// An SGPR to write: mostly s24-s39, clear of the root parameters
    /// (s0-s7) and the descriptors (s8-s23), so that most programs keep
    /// their resources.
    fn sdst(&mut self) -> u32 {
        if self.percent(85) {
            24 + self.below(16)
        } else {
            self.sreg()
        }
    }

    fn sreg(&mut self) -> u32 {
        match self.below(100) {
            0..30 => self.below(8),
            30..60 => 24 + self.below(16),
            60..66 => VCC + self.below(2),
            66..68 => M0,
            68..70 => NULL,
            70..76 => EXEC + self.below(2),
            _ => self.below(40),
        }
    }

    /// An 8-bit scalar source; a literal's value goes to `lit`.
    fn ssrc(&mut self, lit: &mut Option<u32>) -> u32 {
        match self.below(100) {
            0..55 => self.sreg(),
            55..80 => 128 + self.below(81), // 0 to 64
            80..88 => 240 + self.below(9),  // ±0.5, ±1, ±2, ±4, 1/(2π)
            88..92 => SCC,
            92..97 => {
                *lit = Some(self.word());
                LITERAL
            }
            _ => 193 + self.below(16), // -1 to -16
        }
    }

    /// A 9-bit vector source.
    fn vsrc(&mut self, lit: &mut Option<u32>) -> u32 {
        if self.percent(60) {
            256 + self.vgpr()
        } else {
            self.ssrc(lit)
        }
    }
}

/// A register-relative move (`v_movrel*`, `v_swaprel_b32`), mostly after
/// an M0 that `range` can bound: a constant, or an SGPR clamped by
/// `s_min_u32` or `s_and_b32`. Branches around it make the bounds pass
/// through phis, and a move that writes the registers its own index comes
/// from tests its induction.
fn relative_move(r: &mut Rng) -> Vec<u32> {
    let moves: Vec<u32> = gfx10::opcodes(Encoding::Vop1)
        .iter()
        .filter(|(_, name)| name.contains("movrel") || name.contains("swaprel"))
        .map(|&(op, _)| u32::from(op))
        .collect();
    let op = moves[r.below(moves.len() as u32) as usize];
    let k = 128 + r.below(17); // 0 to 16
    let mut words = match r.below(4) {
        0 => vec![0xBE80_0300 | M0 << 16 | k], // s_mov_b32 m0, k
        1 => vec![0x8380_0000 | M0 << 16 | k << 8 | r.sreg()], // s_min_u32 m0, s, k
        2 => vec![0x8700_0000 | M0 << 16 | k << 8 | r.sreg()], // s_and_b32 m0, s, k
        _ => vec![],
    };
    words.push(0x7E00_0000 | r.vgpr() << 17 | op << 9 | (256 + r.vgpr()));
    words
}

/// One instruction: its words, a pending branch's offset still unset.
fn instruction(r: &mut Rng, stage: HwStage) -> Vec<u32> {
    if stage == HwStage::Ps && r.percent(8) {
        // v_interp_p1_f32, v_interp_p2_f32 or v_interp_mov_f32 of a channel
        // of attribute 0 (rarely 1, which no input holds).
        let attr = u32::from(r.percent(5));
        return vec![
            0xC800_0000
                | r.vgpr() << 18
                | r.below(3) << 16
                | attr << 10
                | r.below(4) << 8
                | r.below(4),
        ];
    }
    if r.percent(4) {
        return relative_move(r);
    }
    let mut lit = None;
    let mut words = match r.below(100) {
        // VOP2 and VOP1, a tenth of them SDWA, DPP16 or DPP8.
        0..26 => {
            let w = if r.percent(55) {
                let op = r.opcode(Encoding::Vop2);
                op << 25 | r.vgpr() << 17 | r.vgpr() << 9 | r.vsrc(&mut lit)
            } else {
                let op = r.opcode(Encoding::Vop1);
                0x7E00_0000 | r.vgpr() << 17 | op << 9 | r.vsrc(&mut lit)
            };
            if r.percent(10) {
                lit = None;
                let form = [0xF9, 0xFA, 0xE9][r.below(3) as usize];
                vec![w & !0x1FF | form, r.word() & 0xFFFF_FF0F]
            } else {
                vec![w]
            }
        }
        26..34 => {
            let op = r.opcode(Encoding::Vopc);
            vec![0x7C00_0000 | op << 17 | r.vgpr() << 9 | r.vsrc(&mut lit)]
        }
        // VOP3: its own opcodes and the VOP3 forms of VOPC, VOP2 and VOP1,
        // with modifiers now and then.
        34..48 => {
            let op = match r.below(4) {
                0 => r.opcode(Encoding::Vopc),
                1 => 0x100 + r.opcode(Encoding::Vop2),
                2 => 0x180 + r.opcode(Encoding::Vop1),
                _ => r.opcode(Encoding::Vop3),
            };
            let mods = if r.percent(20) { r.below(0x80) } else { 0 };
            let dst = if op < 0x100 && r.percent(50) {
                r.sdst()
            } else {
                r.vgpr()
            };
            let srcs = r.vsrc(&mut lit) | r.vsrc(&mut lit) << 9 | r.vsrc(&mut lit) << 18;
            let high = if r.percent(20) { r.below(32) << 27 } else { 0 };
            vec![0xD400_0000 | op << 16 | mods << 8 | dst, srcs | high]
        }
        48..51 => {
            let op = r.opcode(Encoding::Vop3p);
            let srcs = r.vsrc(&mut lit) | r.vsrc(&mut lit) << 9 | r.vsrc(&mut lit) << 18;
            vec![
                0xCC00_0000 | op << 16 | r.below(0x100) << 8 | r.vgpr(),
                srcs | r.below(32) << 27,
            ]
        }
        51..60 => {
            let op = r.opcode(Encoding::Sop2);
            let (a, b) = (r.ssrc(&mut lit), r.ssrc(&mut lit));
            vec![0x8000_0000 | op << 23 | r.sdst() << 16 | b << 8 | a]
        }
        60..66 => {
            let op = r.opcode(Encoding::Sop1);
            vec![0xBE80_0000 | r.sdst() << 16 | op << 8 | r.ssrc(&mut lit)]
        }
        66..71 => {
            let op = r.opcode(Encoding::Sopc);
            let (a, b) = (r.ssrc(&mut lit), r.ssrc(&mut lit));
            vec![0xBF00_0000 | op << 16 | b << 8 | a]
        }
        71..73 => {
            let op = r.opcode(Encoding::Sopk);
            vec![0xB000_0000 | op << 23 | r.sdst() << 16 | r.below(0x10000)]
        }
        // s_branch, s_cbranch_scc0/scc1/vccz/vccnz/execz/execnz.
        73..80 => {
            let op = [0x02, 0x04, 0x05, 0x06, 0x07, 0x08, 0x09][r.below(7) as usize];
            vec![0xBF80_0000 | op << 16 | BRANCH_PENDING]
        }
        // Scalar loads (and rarely other SMEM) from the root tables' addresses.
        80..85 => {
            let op = loop {
                let op = r.opcode(Encoding::Smem);
                if op < 0x20 || r.percent(5) {
                    break op;
                }
            };
            let base = [2, 3][r.below(2) as usize]; // s[4:5], s[6:7]
            let soffset = if r.percent(70) { NULL } else { r.sreg() };
            vec![
                0xF400_0000 | op << 18 | r.sdst() << 6 | base,
                soffset << 25 | (r.below(64) * 4),
            ]
        }
        // Buffer access through the UAV at s[8:11].
        85..89 => {
            let op = r.opcode(Encoding::Mubuf);
            let soffset = if r.percent(70) { 0x80 } else { r.sreg() };
            vec![
                0xE000_0000 | op << 18 | r.below(4) << 12 | (r.below(64) * 4),
                soffset << 24 | 2 << 16 | r.vgpr() << 8 | r.vgpr(),
            ]
        }
        89..90 => {
            let op = r.opcode(Encoding::Mtbuf);
            vec![
                0xE800_0000 | r.below(128) << 19 | (op & 7) << 16 | r.below(4) << 12,
                0x80 << 24 | (op >> 3 & 1) << 21 | 2 << 16 | r.vgpr() << 8 | r.vgpr(),
            ]
        }
        // Images: the texture at s[12:19] (or the UAV), the sampler at s[20:23].
        90..95 => {
            let op = r.opcode(Encoding::Mimg);
            let r128 = if r.percent(10) { 1 << 15 } else { 0 };
            let w0 = 0xF000_0000
                | (op & 0x7F) << 18
                | op >> 7
                | (1 + r.below(15)) << 8
                | r.below(8) << 3
                | r128;
            let rsrc = [3, 2][r.below(2) as usize];
            vec![
                w0,
                r.below(4) << 30 | 5 << 21 | rsrc << 16 | r.vgpr() << 8 | r.vgpr(),
            ]
        }
        95..98 => {
            let op = r.opcode(Encoding::Ds);
            vec![
                0xD800_0000 | op << 18 | r.below(0x10000),
                r.vgpr() << 24 | r.vgpr() << 16 | r.vgpr() << 8 | r.vgpr(),
            ]
        }
        _ => {
            let op = r.opcode(Encoding::Global);
            let saddr = if r.percent(50) {
                NULL
            } else {
                [4, 6][r.below(2) as usize]
            };
            vec![
                0xDC00_8000 | op << 18 | r.below(0x1000),
                r.vgpr() << 24 | saddr << 16 | r.vgpr() << 8 | r.vgpr(),
            ]
        }
    };
    words.extend(lit);
    words
}

/// The program of `seed`.
fn program(seed: u64) -> NativeProgram {
    let mut r = Rng::new(seed);
    let stage = if r.percent(70) {
        HwStage::Cs
    } else {
        HwStage::Ps
    };
    let wave64 = stage == HwStage::Cs && r.percent(30);
    let mut words = vec![
        0xBE85_0380, // s_mov_b32 s5, 0
        0xBE87_0380, // s_mov_b32 s7, 0
        0xF408_0202,
        0xFA00_0000, // s_load_dwordx4 s[8:11], s[4:5], 0
        0xF40C_0302,
        0xFA00_0020, // s_load_dwordx8 s[12:19], s[4:5], 0x20
        0xF408_0503,
        0xFA00_0000, // s_load_dwordx4 s[20:23], s[6:7], 0
    ];
    let body: Vec<Vec<u32>> = (0..2 + r.below(30))
        .map(|_| instruction(&mut r, stage))
        .collect();
    // Where each instruction starts, and the end of the body.
    let mut starts = Vec::with_capacity(body.len() + 1);
    let mut at = words.len();
    for inst in &body {
        starts.push(at);
        at += inst.len();
    }
    starts.push(at);
    for (k, mut inst) in body.into_iter().enumerate() {
        if inst[0] >> 23 == 0x17F && inst[0] & 0xFFFF == BRANCH_PENDING {
            let (lo, hi) = (k.saturating_sub(6), (k + 6).min(starts.len() - 1));
            let target = starts[lo + r.below((hi - lo + 1) as u32) as usize];
            let offset = target as i64 - starts[k] as i64 - 1;
            inst[0] = inst[0] & 0xFFFF_0000 | offset as u32 & 0xFFFF;
        }
        words.extend(inst);
    }
    words.extend(match stage {
        HwStage::Ps => [0xF800_180F, 0x0302_0100], // exp mrt0 v0, v1, v2, v3 done vm
        _ => [0xE070_0000, 0x8002_0000],           // buffer_store_dword v0, off, s[8:11], 0
    });
    words.push(S_ENDPGM);

    let param = |root, sgprs, user_data, slot| RootParam {
        root,
        table_offset_dwords: 0,
        sgprs,
        base_register: 0,
        user_data,
        slot,
    };
    let used = |kind, table_offset_dwords, dwords, user_data, slot| UsedResource {
        kind,
        root: RootType::Table,
        table_offset_dwords,
        lower_bound: 0,
        stride: 0,
        dwords,
        user_data,
        slot,
    };
    let (col, cols) = input_channels(seed);
    let io = match stage {
        HwStage::Ps => StageIo::Ps {
            inputs: vec![PsInput {
                usage: Usage::Generic,
                row: 0,
                mask: ((1 << cols) - 1) << col,
                attr: 0,
                flat: false,
                fp16: false,
                pack2: false,
                row2: 0,
            }],
            // PERSP_CENTER, PERSP_SAMPLE, LINEAR_CENTER, or several.
            input_addr: [1 << 1, 1, 1 << 4, 0x7F][r.below(4) as usize],
            db_shader_control: 0,
            patches: vec![],
        },
        // 8 user SGPRs, group ids, thread ids and LDS now and then.
        _ => StageIo::Cs(ComputeIo {
            rsrc2: 8 << 1
                | r.below(8) << 7
                | r.below(3) << 11
                | if r.percent(70) {
                    (1 + r.below(16)) << 15
                } else {
                    0
                },
            threads: [if wave64 { 64 } else { 32 }, 1 + r.below(2), 1],
        }),
    };
    NativeProgram {
        stage,
        wave64,
        vgprs: 16,
        layout: vec![
            param(RootType::Constants, 4, 0, 0),
            param(RootType::Table, 1, 4, 1),
            param(RootType::Table, 1, 6, 2),
        ],
        used: vec![
            used(ResourceType::Uav, 0, 4, 4, 1),
            used(ResourceType::Srv, 8, 8, 4, 1),
            used(ResourceType::Sampler, 0, 4, 6, 2),
        ],
        io,
        code: words.iter().flat_map(|w| w.to_le_bytes()).collect(),
    }
}

/// The channels of a pixel program's input element: the first (x to w)
/// and how many, from the program's seed.
fn input_channels(seed: u64) -> (u8, u8) {
    let col = (seed >> 3 & 3) as u8;
    (col, 1 + (seed >> 5) as u8 % (4 - col))
}

/// The container of the program of `seed`: for a pixel program the
/// signatures it uses (one float input element in row 0, a float4
/// target), for compute none.
fn container(seed: u64, stage: HwStage) -> Vec<u8> {
    if stage != HwStage::Ps {
        return dxbc::build(&[]);
    }
    let el = |name: &str, kind, col, cols: u8| SigElement {
        name: name.into(),
        index: 0,
        kind,
        comp: 9,
        interp: 0,
        row: Some(0),
        col,
        cols,
        usage: (1 << cols) - 1,
        half: false,
    };
    let (col, cols) = input_channels(seed);
    let isg1 = container::signature(&[el("TEXCOORD", 0, col, cols)], true);
    let osg1 = container::signature(&[el("SV_Target", 16, 0, 4)], false);
    dxbc::build(&[(FourCC::ISG1, &isg1), (FourCC::OSG1, &osg1)])
}

/// What translating `native` came to.
#[derive(Debug, PartialEq, Eq)]
enum Outcome {
    Translated(super::HostShader),
    Refused(String),
}

/// The stages in the order and with the inputs [`super::translate_with`]
/// gives them.
fn translate(native: &NativeProgram, c: &Container, options: super::Options) -> Outcome {
    let run = || -> Result<super::HostShader, crate::ShaderError> {
        let (p, st) = range::build(&native.patched_code(), native.wave64, native.vgprs)?;
        let res = resource::track(&p, &native.layout, native.stage, st.as_ref().ok())?;
        let io = io::raise(&p, native)?;
        super::emit_with(c, native, &p, &st?, &io, &res, options)
    };
    match run() {
        Ok(host) => Outcome::Translated(host),
        Err(e) => Outcome::Refused(e.to_string()),
    }
}

/// The program as SP3 text, to see what failed.
fn listing(native: &NativeProgram) -> String {
    let mut text = format!(
        "wave{} {:?}\n",
        if native.wave64 { 64 } else { 32 },
        native.io
    );
    for inst in gfx10::decode(&native.patched_code()).unwrap_or_default() {
        let words: Vec<String> = inst.words.iter().map(|w| format!("{w:08x}")).collect();
        text += &format!(
            "  {:4x}: {:28} {}\n",
            inst.offset,
            words.join(" "),
            inst.text(native.wave64)
        );
    }
    text
}

fn env(name: &str, default: u64) -> u64 {
    std::env::var(name)
        .ok()
        .and_then(|s| s.trim().parse().ok())
        .unwrap_or(default)
}

#[test]
fn random_programs_translate_or_are_refused_the_same_way_every_time() {
    let count = env("VICTORY_RANDOM_PROGRAMS", DEFAULT_PROGRAMS);
    let first = env("VICTORY_RANDOM_SEED", 1);
    let trace = env("VICTORY_RANDOM_TRACE", 0) != 0;
    let mut failed = Vec::new();
    let mut translated = 0;
    for seed in first..first.saturating_add(count) {
        if trace {
            // Past the test's output capture, so that the last seed shows
            // even when a stack overflow aborts the process.
            let _ = writeln!(std::io::stderr(), "random program {seed}");
        }
        let native = program(seed);
        let bytes = container(seed, native.stage);
        let c = Container::parse(&bytes).unwrap();
        // Odd seeds with write marks, as the title's pipelines are made
        // (`victory-gpu`'s shader cache), even ones without, as `translate`.
        let options = super::Options {
            view_marks: seed % 2 == 1,
        };
        let once = || std::panic::catch_unwind(|| translate(&native, &c, options));
        let why = match (once(), once()) {
            (Ok(a), Ok(b)) if a == b => {
                translated += u32::from(matches!(a, Outcome::Translated(_)));
                continue;
            }
            (Ok(a), Ok(b)) => format!("translates differently: {a:?} then {b:?}"),
            (Err(panic), _) | (_, Err(panic)) => {
                let message = panic
                    .downcast_ref::<String>()
                    .map(String::as_str)
                    .or_else(|| panic.downcast_ref::<&str>().copied())
                    .unwrap_or("(no message)");
                format!("panics: {message}")
            }
        };
        if failed.is_empty() {
            eprintln!("seed {seed} {why}\n{}", listing(&native));
        }
        failed.push(seed);
    }
    assert!(
        failed.is_empty(),
        "{} of {count} random programs failed (rerun one with VICTORY_RANDOM_SEED=<seed> \
         VICTORY_RANDOM_PROGRAMS=1); the first seeds: {:?}",
        failed.len(),
        &failed[..failed.len().min(20)]
    );
    // The generator still reaches DXIL emission (about one program in ten
    // translates): a change that has every program refused early would
    // leave this test checking nothing.
    if count >= 1000 {
        assert!(
            u64::from(translated) * 50 >= count,
            "only {translated} of {count} random programs translated"
        );
    }
}

/// A seed always makes the same program, and neighbouring seeds different
/// ones.
#[test]
fn random_programs_are_reproducible_from_their_seed() {
    for seed in 1..64 {
        assert_eq!(program(seed).code, program(seed).code);
    }
    assert_ne!(program(1).code, program(2).code);
}
