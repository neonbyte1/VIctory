//! Explicit-stack operations for lane-mask decision diagrams.
//!
//! Shader control flow may contain thousands of distinct conditions.
//! Keep traversal depth on the heap rather than the compiler thread's stack.
//! The low branch is still processed first; canonical nodes, memoization,
//! and the conservative satisfiability budget retain their original meaning.

use super::{Bool, Bools, Map, MAX_MEET, MAX_MEMO, TERMINAL};
use std::collections::HashSet;

type Triple = (Bool, Bool, Bool);
type Conditions = Vec<(Bool, bool)>;

enum IteTask {
    Evaluate(Triple),
    Combine(Triple, u32),
}

#[derive(Clone, Copy)]
enum Rewrite<'a> {
    ForgetFrom(usize),
    Restrict(&'a [(u32, bool)]),
}

enum RewriteTask {
    Evaluate(Bool),
    Combine(Bool, u32),
    Decided(Bool),
}

enum MeetTask {
    Evaluate(Conditions),
    Refute(Conditions),
}

impl Bools {
    fn ite_cached(&self, (f, g, h): Triple) -> Option<Bool> {
        if self.overflow {
            Some(Bool::FALSE)
        } else if f == Bool::TRUE || g == h {
            Some(g)
        } else if f == Bool::FALSE {
            Some(h)
        } else if (g, h) == (Bool::TRUE, Bool::FALSE) {
            Some(f)
        } else {
            self.memo.get(&(f, g, h)).copied()
        }
    }

    pub(super) fn ite(&mut self, f: Bool, g: Bool, h: Bool) -> Bool {
        let root = (f, g, h);
        if let Some(r) = self.ite_cached(root) {
            return r;
        }
        let mut tasks = vec![IteTask::Evaluate(root)];
        let mut results = Vec::new();
        while let Some(task) = tasks.pop() {
            match task {
                IteTask::Evaluate(key @ (f, g, h)) => {
                    if let Some(r) = self.ite_cached(key) {
                        results.push(r);
                        continue;
                    }
                    let top = |b: Bool| self.nodes[b.0 as usize].0;
                    let v = top(f).min(top(g)).min(top(h));
                    let ((f0, f1), (g0, g1), (h0, h1)) =
                        (self.split(f, v), self.split(g, v), self.split(h, v));
                    tasks.push(IteTask::Combine(key, v));
                    tasks.push(IteTask::Evaluate((f1, g1, h1)));
                    tasks.push(IteTask::Evaluate((f0, g0, h0)));
                }
                IteTask::Combine(key, v) => {
                    let hi = results.pop().expect("high result");
                    let lo = results.pop().expect("low result");
                    let r = self.mk(v, lo, hi);
                    if !self.overflow {
                        if self.memo.len() >= MAX_MEMO {
                            self.memo.clear();
                        }
                        self.memo.insert(key, r);
                    }
                    results.push(r);
                }
            }
        }
        debug_assert_eq!(results.len(), 1);
        results.pop().expect("ITE result")
    }

    /// Existentially quantify atoms defined in the rewritten part of a loop.
    pub(super) fn forget(
        &mut self,
        b: Bool,
        from: usize,
        memo: &mut Map<Bool, Bool>,
    ) -> Bool {
        self.rewrite(b, Rewrite::ForgetFrom(from), memo)
    }

    /// Substitute the sorted set of known variables in one traversal.
    pub(super) fn restrict(
        &mut self,
        b: Bool,
        values: &[(u32, bool)],
        memo: &mut Map<Bool, Bool>,
    ) -> Bool {
        self.rewrite(b, Rewrite::Restrict(values), memo)
    }

    fn rewrite(&mut self, root: Bool, op: Rewrite<'_>, memo: &mut Map<Bool, Bool>) -> Bool {
        let mut tasks = vec![RewriteTask::Evaluate(root)];
        let mut results = Vec::new();
        while let Some(task) = tasks.pop() {
            match task {
                RewriteTask::Evaluate(b) => {
                    if matches!(op, Rewrite::ForgetFrom(_)) && self.overflow {
                        results.push(Bool::FALSE);
                        continue;
                    }
                    let (v, lo, hi) = self.nodes[b.0 as usize];
                    if v == TERMINAL {
                        results.push(b);
                        continue;
                    }
                    if let Rewrite::Restrict(values) = op
                        && values.last().is_none_or(|&(last, _)| v > last)
                    {
                        results.push(b);
                        continue;
                    }
                    if let Some(&r) = memo.get(&b) {
                        results.push(r);
                        continue;
                    }
                    if let Rewrite::Restrict(values) = op
                        && let Ok(i) = values.binary_search_by_key(&v, |&(v, _)| v)
                    {
                        tasks.push(RewriteTask::Decided(b));
                        tasks.push(RewriteTask::Evaluate(if values[i].1 { hi } else { lo }));
                        continue;
                    }
                    tasks.push(RewriteTask::Combine(b, v));
                    tasks.push(RewriteTask::Evaluate(hi));
                    tasks.push(RewriteTask::Evaluate(lo));
                }
                RewriteTask::Decided(b) => {
                    let r = *results.last().expect("decided result");
                    if !memo.is_empty() {
                        memo.insert(b, r);
                    }
                }
                RewriteTask::Combine(b, v) => {
                    let hi = results.pop().expect("high result");
                    let lo = results.pop().expect("low result");
                    let r = match op {
                        Rewrite::ForgetFrom(from) if self.born[v as usize] >= from => {
                            self.or(lo, hi)
                        }
                        _ => self.mk(v, lo, hi),
                    };
                    memo.insert(b, r);
                    results.push(r);
                }
            }
        }
        debug_assert_eq!(results.len(), 1);
        results.pop().expect("rewrite result")
    }

    /// Whether any lane satisfies all conditions. A budget exhaustion is
    /// still a conservative yes, never a proof that no lane can satisfy them.
    pub fn meet(&self, all: &[(Bool, bool)]) -> bool {
        self.meet_with_budget(all, MAX_MEET)
    }

    fn meet_with_budget(&self, all: &[(Bool, bool)], mut budget: usize) -> bool {
        let mut refuted = HashSet::new();
        let mut tasks = vec![MeetTask::Evaluate(all.to_vec())];
        while let Some(task) = tasks.pop() {
            let mut all = match task {
                MeetTask::Refute(all) => {
                    // Both children have been exhausted; any successful or
                    // unknown child would have returned from the whole search.
                    refuted.insert(all);
                    continue;
                }
                MeetTask::Evaluate(all) => all,
            };
            if all.iter().any(|&(b, holds)| {
                b == if holds { Bool::FALSE } else { Bool::TRUE }
            }) {
                continue;
            }
            all.retain(|&(b, _)| b != Bool::FALSE && b != Bool::TRUE);
            all.sort_unstable();
            all.dedup();
            if refuted.contains(&all) {
                continue;
            }
            if all.is_empty() || budget == 0 {
                return true;
            }
            budget -= 1;
            let v = all
                .iter()
                .map(|&(b, _)| self.nodes[b.0 as usize].0)
                .min()
                .expect("nonempty conditions");
            let branch = |high: bool| {
                all.iter()
                    .map(|&(b, holds)| {
                        let (lo, hi) = self.split(b, v);
                        (if high { hi } else { lo }, holds)
                    })
                    .collect()
            };
            let lo = branch(false);
            let hi = branch(true);
            tasks.push(MeetTask::Refute(all));
            tasks.push(MeetTask::Evaluate(hi));
            tasks.push(MeetTask::Evaluate(lo));
        }
        false
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::Value;
    use crate::structure::Atom;

    fn eval(b: &Bools, mut node: Bool, assignment: usize) -> bool {
        loop {
            let (v, lo, hi) = b.nodes[node.0 as usize];
            if v == TERMINAL {
                return node == Bool::TRUE;
            }
            node = if assignment & (1 << v) == 0 { lo } else { hi };
        }
    }

    #[test]
    fn iterative_operations_match_truth_tables() {
        let mut b = Bools::new();
        let a = b.atom(Atom::Lane(Value(0)), 0);
        let c = b.atom(Atom::Lane(Value(1)), 1);
        let d = b.atom(Atom::Lane(Value(2)), 2);
        let mut forms = vec![Bool::FALSE, Bool::TRUE, a, c, d];
        forms.push(b.and(a, c));
        forms.push(b.or(c, d));
        forms.push(b.xor(a, d));
        forms.push(b.not(c));
        for &f in &forms {
            for &g in &forms {
                for &h in &forms {
                    let r = b.ite(f, g, h);
                    for x in 0..8 {
                        let expected = if eval(&b, f, x) { g } else { h };
                        assert_eq!(eval(&b, r, x), eval(&b, expected, x));
                    }
                }
                for hold_f in [false, true] {
                    for hold_g in [false, true] {
                        let expected = (0..8).any(|x| {
                            eval(&b, f, x) == hold_f && eval(&b, g, x) == hold_g
                        });
                        assert_eq!(b.meet(&[(f, hold_f), (g, hold_g)]), expected);
                    }
                }
            }
            for v in 0..3 {
                for value in [false, true] {
                    let r = b.restrict(f, &[(v, value)], &mut Map::default());
                    for x in 0..8 {
                        let assigned = (x & !(1 << v)) | (usize::from(value) << v);
                        assert_eq!(eval(&b, r, x), eval(&b, f, assigned));
                    }
                }
                let r = b.forget(f, v as usize, &mut Map::default());
                for x in 0..8 {
                    let fixed = (1 << v) - 1;
                    let expected = (0..8).any(|y| {
                        (x & fixed) == (y & fixed) && eval(&b, f, y)
                    });
                    assert_eq!(eval(&b, r, x), expected);
                }
            }
        }
    }

    #[test]
    fn restriction_preserves_multiple_decisions_in_one_walk() {
        let mut b = Bools::new();
        let a = b.atom(Atom::Lane(Value(0)), 0);
        let c = b.atom(Atom::Lane(Value(1)), 1);
        let d = b.atom(Atom::Lane(Value(2)), 2);
        let ac = b.and(a, c);
        let root = b.or(ac, d);
        let r = b.restrict(root, &[(0, true), (2, false)], &mut Map::default());
        assert_eq!(r, c);
        assert_eq!(b.restrict(root, &[], &mut Map::default()), root);
    }

    #[test]
    fn deep_diagrams_do_not_use_the_call_stack() {
        const DEPTH: u32 = 20_000;
        let mut b = Bools::new();
        for i in 0..DEPTH {
            b.atom(Atom::Lane(Value(i)), i as usize);
        }
        // Construct a long conjunction in linear work, from the bottom up.
        let mut all = Bool::TRUE;
        for i in (0..DEPTH).rev() {
            all = b.mk(i, Bool::FALSE, all);
        }
        let not_all = b.not(all);
        assert_ne!(not_all, all);
        assert_eq!(b.not(not_all), all);
        assert_eq!(
            b.restrict(all, &[(DEPTH - 1, false)], &mut Map::default()),
            Bool::FALSE
        );
        assert_eq!(b.forget(all, 0, &mut Map::default()), Bool::TRUE);
        assert!(b.meet(&[(all, true)]));
        assert!(!b.meet(&[(all, true), (all, false)]));
    }

    #[test]
    fn exhausted_search_is_conservative() {
        let mut b = Bools::new();
        let a = b.atom(Atom::Lane(Value(0)), 0);
        assert!(b.meet_with_budget(&[(a, true), (a, false)], 0));
        assert!(!b.meet_with_budget(&[(a, true), (a, false)], 1));
        assert!(!b.meet_with_budget(&[(Bool::FALSE, true)], 0));
    }
}
