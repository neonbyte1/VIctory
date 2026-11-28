//! Partial dispatch: the thread-count guard of a `DispatchX` whose flag
//! 0x20 gives thread counts.

use super::{Lower, Val, Why};
use crate::emit::UserData;
use crate::emit::bitcode::{Bin, Term, V, pred};
use crate::emit::shader::USER_DISPATCH_THREADS_X;
use crate::native::RootType;

/// What a guarded lowering stops with when its host form has a group
/// barrier, which a lane that left at the guard would not reach: lowered
/// again without the guard (the host then keeps the dispatch's counts).
pub(super) const THREAD_LIMIT_BARRIER: Why = "a thread-count guard before a group barrier";

impl Lower<'_> {
    pub(super) fn host_word(&mut self, root: RootType, slot: u8, dword: u32) -> V {
        let known = self
            .user_reads
            .values()
            .find(|u| u.root == root && u.slot == slot && u.dword == dword)
            .copied();
        let u = match known {
            Some(u) => u,
            None => {
                let mut index = self
                    .x
                    .native
                    .layout
                    .iter()
                    .map(|e| u32::from(e.user_data) + e.sgprs)
                    .max()
                    .unwrap_or(0);
                while self.user_reads.contains_key(&index) {
                    index += 1;
                }
                let u = UserData {
                    index,
                    slot,
                    root,
                    dword,
                };
                self.user_reads.insert(index, u);
                u
            }
        };
        self.user_at(u)
    }

    /// Lanes whose global X thread index (from `group`, the group's X id)
    /// is at or past the host's thread count return at [`Self::cur`].
    pub(super) fn partial_dispatch_prologue(&mut self, group: V) {
        let original = self.cur;
        let width = self.s.threads.expect("compute thread count")[0];
        let limit = self.host_word(RootType::Other(USER_DISPATCH_THREADS_X), 0, width);
        let Val::I32(thread) = self.thread_id(0) else {
            unreachable!()
        };
        let width = self.s.m.i32(width);
        let first = self.s.m.bin(original, Bin::Mul, group, width);
        let x = self.s.m.bin(original, Bin::Add, first, thread);
        let zero = self.s.m.i32(0);
        let unlimited = self.s.m.cmp(original, pred::EQ, limit, zero);
        let inside = self.s.m.cmp(original, pred::IULT, x, limit);
        let active = self.s.m.bin(original, Bin::Or, unlimited, inside);
        let body = self.s.m.new_block();
        let done = self.s.m.new_block();
        self.s
            .m
            .terminate(original, Term::CondBr(active, body, done));
        self.s.m.terminate(done, Term::Ret);
        self.cur = body;
    }

    /// Whether the program reads the host's thread count.
    pub(super) fn thread_limited(&self) -> bool {
        self.user_reads
            .values()
            .any(|u| u.root == RootType::Other(USER_DISPATCH_THREADS_X))
    }
}

#[cfg(test)]
mod tests {
    use super::super::{Parts, lower_with_thread_limit};
    use crate::emit::shader::USER_DISPATCH_THREADS_X;
    use crate::native::{ComputeIo, HwStage, NativeProgram, RootType, StageIo};
    use crate::{io, ir, resource, structure};

    const S_ENDPGM: u32 = 0xBF81_0000;
    const S_BARRIER: u32 = 0xBF8A_0000;

    /// The thread-count word a guarded lowering of `words` (groups of
    /// `width` x 1 x 1) reads, if any: (slot, dword, index).
    fn thread_word(width: u32, words: &[u32]) -> Option<(u8, u32, u32)> {
        let native = NativeProgram {
            stage: HwStage::Cs,
            wave64: false,
            vgprs: 256,
            layout: vec![],
            used: vec![],
            io: StageIo::Cs(ComputeIo {
                rsrc2: 1 << 15 | 2 << 1,
                threads: [width, 1, 1],
            }),
            code: words.iter().flat_map(|w| w.to_le_bytes()).collect(),
        };
        let p = ir::build(&native.code, false, 256).unwrap();
        let st = structure::structure(&p).unwrap();
        let res = resource::track(&p, &native.layout, native.stage, Some(&st)).unwrap();
        let io = io::raise(&p, &native).unwrap();
        let parts = Parts {
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
        };
        let l = lower_with_thread_limit(&parts, true).unwrap();
        l.user_data
            .iter()
            .find(|u| u.root == RootType::Other(USER_DISPATCH_THREADS_X))
            .map(|u| (u.slot, u.dword, u.index))
    }

    #[test]
    fn partial_dispatch_guards_threads_past_the_tail() {
        // After the two dispatch-spread words; `dword` is the group width.
        assert_eq!(thread_word(64, &[S_ENDPGM]), Some((0, 64, 2)));
    }

    #[test]
    fn a_group_barrier_lowers_without_the_guard() {
        assert_eq!(thread_word(32, &[S_BARRIER, S_ENDPGM]), None);
    }
}
