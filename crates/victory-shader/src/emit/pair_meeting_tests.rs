//! Pair meetings (`lower::PairMeetings`): console wave64 programs whose
//! thread group holds several console waves, on a host of 32-lane waves,
//! against what the console computes.
//!
//! A synthetic program (a loop each console wave runs a different number
//! of times, with a lane read across the halves in it) runs anywhere. The
//! Grand Theft Auto VI BVH builder's five such programs run from containers
//! on the local disk (`VICTORY_BVH_PROGRAMS`: a directory of
//! `<first 16 hex digits of the BLAKE3>.dxbc`, never committed), each
//! against a reference of what it computes on the console: a block scan, a
//! digit histogram, a block sort and scatter.

use super::host::{Dispatch, Host, Root, View};
use super::{HostShader, Options, Target, lower};
use crate::native::{ComputeIo, RootParam, RootType, StageIo};

const NVIDIA: Target = Target {
    lanes_min: 32,
    lanes_max: 32,
};

/// Lane t of a group of `threads` runs a loop t times and reads lane `lane`'s
/// thread id into s9 in every iteration (`lane_read_loop_tests`' program).
/// After the loop s9 is the thread id of lane `lane` of t's console wave,
/// in every lane: lane t stores t + 64 * (t / 64) + lane. Each console wave
/// of the group leaves the loop after a different number of iterations.
fn looping(threads: u32, lane: u32) -> crate::NativeProgram {
    let k = |n: u32| 128 + n;
    let words = [
        0xBE82_0300, // s_mov_b32 s2, s0
        0xBE83_0380, // s_mov_b32 s3, 0
        0xF408_0101,
        0xFA00_0000, // s_load_dwordx4 s[4:7], s[2:3], null
        0xBF8C_C07F, // s_waitcnt lgkmcnt(0)
        0xBE89_0380, // s_mov_b32 s9, 0
        0x7E04_0280, // v_mov_b32 v2, 0
        0x7D8A_0080, // v_cmp_ne_u32 vcc, 0, v0
        0xBE8A_246A, // s_and_saveexec_b64 s[10:11], vcc
        0xBF88_0006, // s_cbranch_execz end
        // loop:
        0xD760_0009,
        k(lane) << 9 | 0x100, // v_readlane_b32 s9, v0, lane
        0x4A04_0481,          // v_add_nc_u32 v2, 1, v2
        0x7D82_0102,          // v_cmp_lt_u32 vcc, v2, v0
        0x87FE_6A7E,          // s_and_b64 exec, exec, vcc
        0xBF89_FFFA,          // s_cbranch_execnz loop
        // end:
        0x88FE_0A7E, // s_or_b64 exec, exec, s[10:11]
        0x4A04_0409, // v_add_nc_u32 v2, s9, v2
        0x3406_0082, // v_lshlrev_b32 v3, 2, v0
        0xE070_1000,
        0x8001_0203, // buffer_store_dword v2, v3, s[4:7], 0 offen
        0xBF81_0000, // s_endpgm
    ];
    crate::NativeProgram {
        stage: crate::HwStage::Cs,
        wave64: true,
        vgprs: 8,
        layout: vec![RootParam {
            root: RootType::Table,
            table_offset_dwords: 0,
            sgprs: 1,
            base_register: 0,
            user_data: 0,
            slot: 0,
        }],
        used: vec![],
        io: StageIo::Cs(ComputeIo {
            rsrc2: 1 << 1,
            threads: [threads, 1, 1],
        }),
        code: words.iter().flat_map(|w| w.to_le_bytes()).collect(),
    }
}

fn emit(native: &crate::NativeProgram, target: Target, pair_meetings: bool) -> Result<HostShader, String> {
    let c = crate::Container {
        digest: [0; 16],
        version: (1, 0),
        parts: vec![],
    };
    let p = crate::ir::build(&native.code, native.wave64, native.vgprs).map_err(|e| e.to_string())?;
    let st = crate::structure::structure(&p).map_err(|e| e.to_string())?;
    let res = crate::resource::track(&p, &native.layout, native.stage, Some(&st)).map_err(|e| e.to_string())?;
    let io = crate::io::raise(&p, native).map_err(|e| e.to_string())?;
    let options = Options {
        view_marks: false,
        target,
        pair_meetings,
    };
    super::emit_with(&c, native, &p, &st, &io, &res, options).map_err(|e| e.to_string())
}

/// Without pair meetings a group of several console waves is refused on a
/// 32-lane host, as before; with them it translates. Hosts with 64-lane
/// waves are unchanged.
#[test]
fn several_console_waves_translate_with_pair_meetings() {
    let native = looping(256, 40);
    let refused = emit(&native, NVIDIA, false).expect_err("refused without pair meetings");
    assert!(refused.contains(lower::SEVERAL_CONSOLE_WAVES), "{refused}");
    emit(&native, NVIDIA, true).expect("pair meetings");
    let rdna = Target {
        lanes_min: 32,
        lanes_max: 64,
    };
    assert_eq!(emit(&native, rdna, false), emit(&native, rdna, true));
}

fn hardware() -> Option<Host> {
    match Host::new(false) {
        Ok(h) if h.target() == NVIDIA => Some(h),
        Ok(_) => {
            eprintln!("the hardware adapter runs 64-lane waves: nothing to test; skipped");
            None
        }
        Err(e) => {
            eprintln!("no SM 6.6 hardware device ({e}); skipped");
            None
        }
    }
}

/// The synthetic loop in groups of four console waves, run on a 32-lane
/// host: every lane as the console computes it.
#[test]
fn a_lane_read_in_a_loop_each_console_wave_leaves_on_its_own() {
    let Some(host) = hardware() else { return };
    let _one = super::valu_tests::ONE_AT_A_TIME.lock().unwrap_or_else(|e| e.into_inner());
    for lane in [5, 40] {
        let threads = 256;
        let cs = emit(&looping(threads, lane), host.target(), true).unwrap();
        let (root, pso) = host.compute_pipeline(&cs).unwrap();
        let views = [View::RawUav(2 * threads)];
        let data = vec![0; (cs.user_data_dwords() as usize).max(1)];
        let out = host
            .dispatch(
                &root,
                &pso,
                &Dispatch {
                    roots: vec![(0, Root::Constants(data))],
                    views: &views,
                    samplers: &[],
                    groups: [2, 1, 1],
                },
            )
            .unwrap()
            .remove(0);
        // (both groups write the same dwords: the same values)
        let want: Vec<u32> = (0..threads).map(|t| t + 64 * (t / 64) + lane).collect();
        let bad: Vec<String> = (0..threads as usize)
            .filter(|&t| out[t] != want[t])
            .map(|t| format!("{t}: {} (want {})", out[t], want[t]))
            .collect();
        assert!(bad.is_empty(), "lane {lane}: {} lanes wrong: {}", bad.len(), bad.join(", "));
    }
}

// ---- the BVH builder's programs ----

/// Where the title-memory copy of a run lies (one 64 KiB page).
const TITLE: u64 = 0x12_3450_0000;
const PAGE_DWORDS: usize = 16384;

/// The page table (`emit` "Title memory") for one copy at [`TITLE`] with
/// its views at heap index `view`, the null pair at `null`.
fn page_table(view: u32, null: u32) -> Vec<u32> {
    let (hi, lo) = ((TITLE >> 32) as u32, TITLE as u32);
    let mut t = vec![0u32; 4096];
    t[(hi << 4 | lo >> 28) as usize] = 1;
    for _ in 0..2 * 4096 {
        t.extend([null, 0]);
    }
    let entry = 4096 + 2 * (4096 + (lo >> 16 & 0xFFF) as usize);
    t[entry..entry + 2].copy_from_slice(&[view, lo]);
    t
}

struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u32 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        (self.0 >> 32) as u32
    }
}

/// The builder's program `hash` from `VICTORY_BVH_PROGRAMS`, translated for
/// `host`; `None` when the directory is not set.
fn builder_program(host: &Host, hash: &str) -> Option<HostShader> {
    let Some(dir) = std::env::var_os("VICTORY_BVH_PROGRAMS") else {
        eprintln!("VICTORY_BVH_PROGRAMS not set; skipped");
        return None;
    };
    let path = std::path::Path::new(&dir).join(format!("{hash}.dxbc"));
    let bytes = std::fs::read(&path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
    let options = Options {
        view_marks: false,
        target: host.target(),
        pair_meetings: true,
    };
    assert!(
        super::translate_with(&bytes, Options { pair_meetings: false, ..options }).is_err(),
        "{hash}: translates without pair meetings"
    );
    Some(super::translate_with(&bytes, options).unwrap_or_else(|e| panic!("{hash}: {e}")))
}

/// Runs a builder program over one 64 KiB page of title memory `memory`:
/// root descriptor `slot` at `TITLE + 4 * buffers[slot]`, root constants
/// `constants`, `groups` groups. Returns the page afterwards.
fn run_builder(
    host: &Host,
    cs: &HostShader,
    buffers: &[usize],
    constants: &[u32],
    memory: &[u32],
    groups: u32,
) -> Vec<u32> {
    assert!(cs.memory, "a builder program reads title memory");
    assert!(cs.root_cbvs.is_empty() && cs.root_srvs.is_empty() && cs.root_uavs.is_empty());
    let mut ud = vec![0u32; cs.user_data_dwords().max(1) as usize];
    // (the constants are the last root parameter)
    let constants_slot = buffers.len() as u8;
    for u in &cs.user_data {
        ud[u.index as usize] = match (u.root, u.slot) {
            (RootType::Constants, super::DISPATCH_SLOT) => 0,
            (RootType::Descriptor, s) if (s as usize) < buffers.len() && u.dword < 2 => {
                ((TITLE + 4 * buffers[s as usize] as u64) >> (32 * u.dword)) as u32
            }
            (RootType::Constants, s) if s == constants_slot => constants[u.dword as usize],
            _ => panic!("user data {u:?}"),
        };
    }
    let mut roots = vec![(0, Root::Constants(ud)), (1, Root::View { srv: true, data: page_table(0, 2) })];
    if cs.memory_writes {
        roots.push((roots.len() as u32, Root::Uav(2 * 4096 / 32)));
    }
    if cs.view_addresses {
        roots.push((roots.len() as u32, Root::View { srv: true, data: vec![0; 16] }));
    }
    let views = [View::Memory(memory), View::Pair, View::NullPair, View::Pair];
    let (root, pso) = host.compute_pipeline(cs).unwrap();
    host.dispatch(
        &root,
        &pso,
        &Dispatch {
            roots,
            views: &views,
            samplers: &[],
            groups: [groups, 1, 1],
        },
    )
    .unwrap()
    .remove(0)
}

/// Words of `got[at..at + want.len()]` that differ from `want`, listed.
fn differ(what: &str, got: &[u32], at: usize, want: &[u32]) -> Option<String> {
    let bad: Vec<String> = want
        .iter()
        .enumerate()
        .filter(|&(i, &w)| got[at + i] != w)
        .take(12)
        .map(|(i, &w)| format!("[{i}] {:#x} (want {w:#x})", got[at + i]))
        .collect();
    let n = want.iter().enumerate().filter(|&(i, &w)| got[at + i] != w).count();
    (n > 0).then(|| format!("{what}: {n} of {} words wrong: {}", want.len(), bad.join(", ")))
}

/// Block scan (`9fc111d3be79364f`: 512 threads of eight dwords,
/// `77360df21e12ea43`: 512 of four, `39321a95223001ed`: 256 of four): each
/// thread's dwords of root UAV 0 are replaced by the exclusive prefix sum
/// of its group's dwords in order; with
/// bit 0 of the constant, the group's total goes to dword `group` of root
/// UAV 1.
fn scan(host: &Host, hash: &str, threads: usize, dwords: usize, seeds: u64) -> Vec<String> {
    let Some(cs) = builder_program(host, hash) else { return Vec::new() };
    let mut wrong = Vec::new();
    let n = dwords * threads;
    let totals = 0x3000;
    for seed in 1..=seeds {
        let mut rng = Rng(seed * 0x9E37_79B9_7F4A_7C15);
        let input: Vec<u32> = (0..n).map(|_| rng.next() & 0xFFFF).collect();
        let mut memory = vec![0u32; PAGE_DWORDS];
        memory[..n].copy_from_slice(&input);
        memory[totals] = 0xDEAD_BEEF;
        let got = run_builder(host, &cs, &[0, totals], &[1], &memory, 1);
        let mut sum = 0u32;
        let want: Vec<u32> = input
            .iter()
            .map(|&x| {
                let before = sum;
                sum = sum.wrapping_add(x);
                before
            })
            .collect();
        wrong.extend(differ(&format!("{hash} seed {seed} prefixes"), &got, 0, &want));
        wrong.extend(differ(&format!("{hash} seed {seed} total"), &got, totals, &[sum]));
    }
    wrong
}

/// Digit histogram (`14394a322f6dd5c2`, 512 threads): the 2048 keys of a
/// group (root UAV 0) counted by `(key >> shift) & 255` into dword
/// `bin * stride + group` of root UAV 1 (constants: stride, shift).
fn histogram(host: &Host, seeds: u64) -> Vec<String> {
    let hash = "14394a322f6dd5c2";
    let Some(cs) = builder_program(host, hash) else { return Vec::new() };
    let mut wrong = Vec::new();
    let out = 0x2000;
    for seed in 1..=seeds {
        let mut rng = Rng(seed * 0xA076_1D64_78BD_642F);
        let shift = 8 * (seed as u32 % 4);
        // Few distinct digits in some seeds: long runs of equal ballots.
        let keys: Vec<u32> = (0..2048)
            .map(|_| if seed % 2 == 0 { rng.next() & 0x0303_0303 } else { rng.next() })
            .collect();
        let mut memory = vec![0u32; PAGE_DWORDS];
        memory[..2048].copy_from_slice(&keys);
        for d in &mut memory[out..out + 256] {
            *d = 0xDEAD_BEEF;
        }
        let got = run_builder(host, &cs, &[0, out, 0, 0, 0, 0], &[1, shift, 0], &memory, 1);
        let mut want = vec![0u32; 256];
        for &k in &keys {
            want[(k >> shift & 255) as usize] += 1;
        }
        wrong.extend(differ(&format!("{hash} seed {seed} shift {shift}"), &got, out, &want));
    }
    wrong
}

/// Block sort and scatter (`e63d22c4def9ba6a`, 512 threads): a group's 2048
/// keys (root UAV 0) and values (1) sorted stably by the digit
/// `(key >> shift) & 255` (eight one-bit passes), then each written to
/// `offset[digit * stride + group] + extra[(digit * stride + group) >> s]`
/// plus its place among its digit's keys, keys to root UAV 4 and values to
/// root UAV 5 (offsets in root UAV 2, extras in 3; constants: stride,
/// shift, s). With the offsets the exclusive prefix of the digit counts and
/// no extras, the outputs are the sorted keys and values.
fn sort(host: &Host, seeds: u64) -> Vec<String> {
    let hash = "e63d22c4def9ba6a";
    let Some(cs) = builder_program(host, hash) else { return Vec::new() };
    let mut wrong = Vec::new();
    let (keys_at, values_at, offsets_at, extra_at, keys_out, values_out) =
        (0, 0x800, 0x1000, 0x1200, 0x2000, 0x2800);
    for seed in 1..=seeds {
        let mut rng = Rng(seed * 0xE703_7ED1_A0B4_28DB);
        let shift = 4 * (seed as u32 % 7);
        let keys: Vec<u32> = (0..2048)
            .map(|_| if seed % 2 == 0 { rng.next() & 0x0F0F_0F0F } else { rng.next() })
            .collect();
        let values: Vec<u32> = (0..2048).map(|i| 0x5000_0000 | i).collect();
        let digit = |k: u32| (k >> shift & 255) as usize;
        let mut count = [0u32; 256];
        for &k in &keys {
            count[digit(k)] += 1;
        }
        let mut offsets = vec![0u32; 256];
        for d in 1..256 {
            offsets[d] = offsets[d - 1] + count[d - 1];
        }
        let mut memory = vec![0u32; PAGE_DWORDS];
        memory[keys_at..keys_at + 2048].copy_from_slice(&keys);
        memory[values_at..values_at + 2048].copy_from_slice(&values);
        memory[offsets_at..offsets_at + 256].copy_from_slice(&offsets);
        for d in &mut memory[keys_out..values_out + 2048] {
            *d = 0xDEAD_BEEF;
        }
        let buffers = [keys_at, values_at, offsets_at, extra_at, keys_out, values_out];
        let got = run_builder(host, &cs, &buffers, &[1, shift, 0], &memory, 1);
        let mut order: Vec<usize> = (0..2048).collect();
        order.sort_by_key(|&i| digit(keys[i]));
        let want_keys: Vec<u32> = order.iter().map(|&i| keys[i]).collect();
        let want_values: Vec<u32> = order.iter().map(|&i| values[i]).collect();
        let label = format!("{hash} seed {seed} shift {shift}");
        wrong.extend(differ(&format!("{label} keys"), &got, keys_out, &want_keys));
        wrong.extend(differ(&format!("{label} values"), &got, values_out, &want_values));
    }
    wrong
}

fn builder_programs(polls: Option<u32>, seeds: u64) {
    let Some(host) = hardware() else { return };
    let _one = super::valu_tests::ONE_AT_A_TIME.lock().unwrap_or_else(|e| e.into_inner());
    lower::TEST_MEETING_POLLS.set(polls);
    let mut wrong = Vec::new();
    wrong.extend(scan(&host, "9fc111d3be79364f", 512, 8, seeds));
    wrong.extend(scan(&host, "77360df21e12ea43", 512, 4, seeds));
    wrong.extend(scan(&host, "39321a95223001ed", 256, 4, seeds));
    wrong.extend(histogram(&host, seeds));
    wrong.extend(sort(&host, seeds));
    lower::TEST_MEETING_POLLS.set(None);
    assert!(wrong.is_empty(), "{}", wrong.join("\n"));
}

/// The builder's five programs with pair meetings, word for word as the
/// console computes them.
#[test]
fn builder_programs_equal_the_console() {
    builder_programs(None, 8);
}

/// The same with meetings that give up after 2^10 polls instead of 2^20:
/// still exact, so no wait comes near the limit the title runs with.
#[test]
fn builder_meetings_end_far_below_the_poll_limit() {
    builder_programs(Some(1 << 10), 4);
}
