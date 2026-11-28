//! The order of a console wave's UAV accesses in the console flow in halves.
//!
//! The console runs a wave64's 64 lanes in one wave, one instruction after
//! the other: a lane's UAV load sees what any lane of its wave stored
//! before it in program order, and a store lands after any lane's earlier
//! load or store. On a host without 64-lane waves the console flow runs the
//! console wave as two host waves, which keep in step only where they meet
//! (exchanges, LDS accesses, the title's barriers), and meetings and the
//! console flow's exchange barriers fence group-shared memory only. So
//! before a UAV access that conflicts with an earlier one (either writes),
//! with no title barrier between them on some path, the halves meet with
//! a UAV fence: a pair meeting, or a group barrier for a group of one
//! console wave. Programs without UAV writes have no such access.

use std::collections::HashSet;

use super::super::bitcode::Attr;
use super::Lower;
use crate::gfx10::Fields;
use crate::ir::{Inst, Program};

/// `(reads, writes)` of a UAV (or title-memory) access: buffer, image and
/// global loads, stores and atomics. Samples, gathers and resource queries
/// read views, not UAVs; scalar loads read through the scalar cache, which
/// the console does not keep in step with vector stores either.
fn access(i: &Inst) -> Option<(bool, bool)> {
    let name = i.name();
    let memory = match i.fields {
        Fields::Buffer { .. } => true,
        Fields::Image { .. } => {
            name.starts_with("image_load")
                || name.starts_with("image_store")
                || name.starts_with("image_atomic")
                || name.starts_with("image_bvh")
        }
        Fields::Flat { .. } => name.starts_with("global_"),
        _ => false,
    };
    let atomic = name.contains("atomic");
    let store = name.contains("store");
    let read = atomic || name.contains("load") || name.starts_with("image_bvh");
    (memory && (read || store)).then_some((read, store || atomic))
}

/// The instructions `(block, index)` the halves must meet before: a UAV
/// access that reads where an earlier access may have written, or writes
/// where one may have read or written, since the last title barrier
/// (`s_barrier`, a group barrier with a UAV fence when the program writes
/// UAVs). Earlier means on some path from the entry, around loops too.
pub(super) fn meetings(p: &Program) -> HashSet<(usize, usize)> {
    const READ: u8 = 1;
    const WRITE: u8 = 2;
    // What happened since the last barrier, at a block's end.
    let walk = |b: usize, mut since: u8, out: &mut HashSet<(usize, usize)>| {
        for (n, i) in p.blocks[b].insts.iter().enumerate() {
            if i.name() == "s_barrier" {
                since = 0;
            } else if let Some((r, w)) = access(i) {
                if r && since & WRITE != 0 || w && since != 0 {
                    out.insert((b, n));
                    since = 0;
                }
                if r {
                    since |= READ;
                }
                if w {
                    since |= WRITE;
                }
            }
        }
        since
    };
    let n = p.blocks.len();
    let mut entry = vec![0u8; n];
    let mut seen = vec![false; n];
    let mut work = vec![0];
    let mut scratch = HashSet::new();
    while let Some(b) = work.pop() {
        seen[b] = true;
        let since = walk(b, entry[b], &mut scratch);
        for s in p.blocks[b].exit.successors() {
            if !seen[s] || entry[s] | since != entry[s] {
                entry[s] |= since;
                work.push(s);
            }
        }
    }
    let mut out = HashSet::new();
    for b in (0..n).filter(|&b| seen[b]) {
        walk(b, entry[b], &mut out);
    }
    out
}

impl Lower<'_> {
    /// The halves of this thread's console wave meet with a UAV fence, at
    /// [`Self::cur`] (a new block after a pair meeting).
    pub(super) fn order_uavs(&mut self) {
        // Group-scope UAV fence (4) and group-shared memory fence (8).
        if self.pairs.is_some() {
            self.meet_fenced(4 | 8);
        } else {
            let (mode, void) = (self.s.m.i32(1 | 4 | 8), self.s.t.void);
            self.s
                .op(self.cur, "barrier", Attr::NoDuplicate, void, 80, &[mode]);
        }
    }
}

#[cfg(test)]
mod tests {
    /// The byte offsets of the instructions the halves meet before.
    fn marked(words: &[u32]) -> Vec<usize> {
        let code: Vec<u8> = words.iter().flat_map(|w| w.to_le_bytes()).collect();
        let p = crate::ir::build(&code, true, 8).unwrap();
        let mut at: Vec<usize> = super::meetings(&p)
            .into_iter()
            .map(|(b, i)| p.blocks[b].insts[i].machine.offset)
            .collect();
        at.sort_unstable();
        at
    }

    const STORE: [u32; 2] = [0xE070_1000, 0x8001_0203]; // buffer_store_dword v2, v3, s[4:7], 0 offen
    const LOAD: [u32; 2] = [0xE030_1000, 0x8001_0203]; // buffer_load_dword v2, v3, s[4:7], 0 offen

    /// A read after a write, then (past a title barrier) a write after a
    /// read; the read right after the barrier needs nothing.
    #[test]
    fn conflicts_since_the_last_barrier() {
        let words = [
            &STORE[..],
            &LOAD,          // 8: reads what the store wrote
            &[0xBF8A_0000], // 16: s_barrier
            &LOAD,          // 20
            &STORE,         // 28: writes what the load read
            &[0xBF81_0000], // s_endpgm
        ]
        .concat();
        assert_eq!(marked(&words), [8, 28]);
    }

    /// In a loop, a load meets the store of the iteration before.
    #[test]
    fn conflicts_around_a_loop() {
        let words = [
            &[0xBE89_0380][..], // s_mov_b32 s9, 0
            &LOAD,              // 4
            &STORE,             // 12
            &[
                0x8009_8109, // s_add_u32 s9, s9, 1
                0xBF0A_8409, // s_cmp_lt_u32 s9, 4
                0xBF85_FFF9, // s_cbranch_scc1 4
                0xBF81_0000, // s_endpgm
            ],
        ]
        .concat();
        assert_eq!(marked(&words), [4, 12]);
    }

    /// Loads alone, or stores after nothing, need no meeting.
    #[test]
    fn no_conflict_no_meeting() {
        let loads = [&LOAD[..], &LOAD, &[0xBF81_0000]].concat();
        assert!(marked(&loads).is_empty());
        let store = [&STORE[..], &[0xBF81_0000]].concat();
        assert!(marked(&store).is_empty());
    }
}
