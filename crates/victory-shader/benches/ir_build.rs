//! Time the IR's construction (liveness, renaming, phis) on synthetic
//! programs; they need no title files or GPU.

use std::hint::black_box;
use std::time::Instant;

use victory_shader::ir;

/// `regs` VGPRs defined first and read at the end, so they are live
/// through `ifs` divergent branches in between.
fn program(regs: u32, ifs: usize) -> Vec<u8> {
    let mut words = Vec::new();
    for i in 0..regs {
        words.push(0x7E00_0280 | (i << 17)); // v_mov_b32 v[i], 0
    }
    for _ in 0..ifs {
        words.extend([
            0x7D84_0080, // v_cmp_eq_u32 vcc, 0, v0
            0xBE82_246A, // s_and_saveexec_b64 s[2:3], vcc
            0xBF88_0001, // s_cbranch_execz +1
            0x4A00_0081, // v_add_nc_u32 v0, 1, v0
            0x88FE_027E, // s_or_b64 exec, exec, s[2:3]
        ]);
    }
    for i in 0..regs {
        words.push(0x4A00_0000 | (i << 9) | (i << 17)); // v_add_nc_u32 v[i], v[i], v[i]
    }
    words.push(0xBF81_0000); // s_endpgm
    words.into_iter().flat_map(u32::to_le_bytes).collect()
}

fn main() {
    for (regs, ifs) in [(8, 500), (64, 500), (128, 500), (8, 2000), (128, 2000)] {
        let code = program(regs, ifs);
        black_box(ir::build(&code, true, 256).unwrap());
        let mut samples = [std::time::Duration::ZERO; 9];
        for sample in &mut samples {
            let start = Instant::now();
            black_box(ir::build(black_box(&code), true, 256).unwrap());
            *sample = start.elapsed();
        }
        samples.sort_unstable();
        println!(
            "{regs} registers, {ifs} divergent ifs: {:?} (median)",
            samples[samples.len() / 2]
        );
    }
}
