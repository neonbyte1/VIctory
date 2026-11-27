//! Compare backwards dataflow with a path-to-first-use oracle.
use super::*;

const REGISTERS: [Reg; 16] = [
    Reg::S(0),
    Reg::S(1),
    Reg::S(63),
    Reg::S(64),
    Reg::S(105),
    Reg::S(106),
    Reg::S(126),
    Reg::S(127),
    Reg::Scc,
    Reg::V(0),
    Reg::V(1),
    Reg::V(62),
    Reg::V(63),
    Reg::V(126),
    Reg::V(127),
    Reg::V(255),
];

fn mask(bits: u64) -> RegSet {
    let mut set = [0; REGS.div_ceil(64)];
    for (i, &reg) in REGISTERS.iter().enumerate() {
        if bits & (1 << i) != 0 {
            insert(&mut set, reg);
        }
    }
    set
}

// Independently follow paths until the register is read or overwritten.
fn reaches_use(blocks: &[Block], uses: &[RegSet], defs: &[RegSet], start: usize, reg: Reg) -> bool {
    let mut seen = vec![false; blocks.len()];
    let mut work = vec![start];
    while let Some(b) = work.pop() {
        if seen[b] {
            continue;
        }
        seen[b] = true;
        if contains(&uses[b], reg) {
            return true;
        }
        if !contains(&defs[b], reg) {
            work.extend(blocks[b].exit.successors());
        }
    }
    false
}

fn check(exits: Vec<Exit>, uses: Vec<RegSet>, defs: Vec<RegSet>) {
    let mut blocks: Vec<_> = exits
        .into_iter()
        .enumerate()
        .map(|(offset, exit)| Block {
            offset,
            preds: Vec::new(),
            phis: Vec::new(),
            insts: Vec::new(),
            exit,
        })
        .collect();
    link(&mut blocks);
    let order = reverse_postorder(&blocks);
    let (actual_in, actual_out) = solve_liveness(&blocks, &order, &uses, &defs);
    let empty = [0; REGS.div_ceil(64)];
    let mut expected_in = vec![empty; blocks.len()];
    let mut expected_out = vec![empty; blocks.len()];
    for &b in &order {
        for reg in REGISTERS {
            if reaches_use(&blocks, &uses, &defs, b, reg) {
                insert(&mut expected_in[b], reg);
            }
            if blocks[b]
                .exit
                .successors()
                .iter()
                .any(|&s| reaches_use(&blocks, &uses, &defs, s, reg))
            {
                insert(&mut expected_out[b], reg);
            }
        }
    }
    assert_eq!(actual_in, expected_in);
    assert_eq!(actual_out, expected_out);
}

#[test]
fn duplicate_edges_and_unreachable_predecessors() {
    check(
        vec![
            Exit::Branch { taken: 1, not_taken: 1 },
            Exit::Branch { taken: 1, not_taken: 2 },
            Exit::End,
            Exit::Jump(1),
            Exit::Jump(4),
        ],
        vec![mask(0), mask(1), mask(1 << 15), mask(2), mask(4)],
        vec![mask(1), mask(0), mask(0), mask(0), mask(0)],
    );
}

#[test]
fn generated_control_flow_matches_path_oracle() {
    for seed in 1..=192u64 {
        let mut state = seed;
        let mut next = || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            state
        };
        let n = 4 + (seed as usize % 29);
        let mut exits = Vec::with_capacity(n);
        let mut uses = Vec::with_capacity(n);
        let mut defs = Vec::with_capacity(n);
        for b in 0..n {
            let random = (next() % n as u64) as usize;
            let forward = (b + 1).min(n - 1);
            let exit = match seed % 6 {
                0 if b + 1 < n => Exit::Jump(forward),
                1 if b + 1 < n => Exit::Branch { taken: forward, not_taken: random },
                2 if b + 1 < n => Exit::Branch { taken: forward, not_taken: forward },
                3 => Exit::Jump(random),
                _ => match next() % 4 {
                    0 => Exit::End,
                    1 => Exit::Jump(random),
                    _ => Exit::Branch { taken: random, not_taken: (next() % n as u64) as usize },
                },
            };
            exits.push(exit);
            uses.push(mask(next() & next()));
            defs.push(mask(next() & next()));
        }
        check(exits, uses, defs);
    }
}
