//! The order of a console wave's UAV accesses on a host without 64-lane
//! waves.
//!
//! The console runs a wave64's 64 lanes in one wave, one instruction after
//! the other: a lane's UAV load sees what any lane of its wave stored
//! before it in program order, and a store lands after any lane's earlier
//! load or store. A host without 64-lane waves runs the console wave as two
//! host waves, in every form (the per-lane form, plain split halves and
//! the console flow), and nothing orders one's UAV accesses after the
//! other's. Only the console flow keeps the halves at the same place in
//! the program, so a program with a UAV access that conflicts with an
//! earlier one (either writes, they may touch the same bytes from
//! different lanes, and no title barrier lies between them on some path)
//! takes the console flow where it can ([`super::Lower::new`]), and there
//! the halves meet with a UAV fence before that access: a pair meeting,
//! or a group barrier for a group of one console wave.
//!
//! Two accesses cannot touch the same bytes from different lanes of a
//! console wave when they go through the same descriptor (the same table
//! slot or root descriptor) and their addresses differ only by constants
//! and by what each lane computes from its thread id alone, with every
//! lane's bytes apart from the other lanes': `base + 4 * tid` loaded and
//! stored by each lane for itself, or a texel at coordinates the thread id
//! gives. That is checked lane by lane, with the values each lane computes;
//! terms the lanes share (a root constant, the group id) must match.
//! Anything else may touch the same bytes.

use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet};
use std::rc::Rc;

use super::super::bitcode::Attr;
use super::{Lower, used_resource};
use crate::gfx10::{Fields, NULL, Operand};
use crate::ir::{EXEC, Inst, Program, Reg, Value};
use crate::native::{NativeProgram, ResourceType};
use crate::resource::{Address, Resources, Role, Site, Source};

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

/// What the address analysis knows of the program and its group.
pub(super) struct Group<'a> {
    pub native: &'a NativeProgram,
    pub res: &'a Resources,
    /// The values of the thread id's components (x, y, z) on entry.
    pub tids: [Option<Value>; 3],
    pub threads: [u32; 3],
}

/// The instructions `(block, index)` the halves must meet before: a UAV
/// access that reads where an earlier access may have written, or writes
/// where one may have read or written, from another lane, since the last
/// title barrier (`s_barrier`, a group barrier with a UAV fence when the
/// program writes UAVs). Earlier means on some path from the entry, around
/// loops too. With `raw` (`VICTORY_UAV_ORDER=raw`): only a read where an
/// earlier access may have written.
pub(super) fn meetings(p: &Program, g: &Group, raw: bool) -> HashSet<(usize, usize)> {
    let mut a = Addresses::new(p, g);
    let mut accesses = Vec::new();
    let mut index = HashMap::new();
    for (b, block) in p.blocks.iter().enumerate() {
        for (n, i) in block.insts.iter().enumerate() {
            if let Some((r, w)) = access(i) {
                index.insert((b, n), accesses.len());
                accesses.push((r, w, a.location(b, n)));
            }
        }
    }
    if !accesses.iter().any(|x| x.1) {
        return HashSet::new();
    }
    let words = accesses.len().div_ceil(64);
    let lanes = a.lanes;
    let mut aliases: HashMap<(usize, usize), bool> = HashMap::new();
    // The accesses since the last barrier or meeting, as a set; at a
    // block's end.
    let mut walk = |b: usize, mut since: Vec<u64>, out: &mut HashSet<(usize, usize)>| {
        for (n, i) in p.blocks[b].insts.iter().enumerate() {
            if i.name() == "s_barrier" {
                since.fill(0);
                continue;
            }
            let Some(&x) = index.get(&(b, n)) else {
                continue;
            };
            let (r, w, ref at) = accesses[x];
            let conflicts = (0..accesses.len())
                .filter(|&e| since[e / 64] >> (e % 64) & 1 != 0)
                .any(|e| {
                    let (er, ew, ref earlier) = accesses[e];
                    (r && ew || !raw && w && (er || ew))
                        && *aliases
                            .entry((e, x))
                            .or_insert_with(|| may_alias(earlier, at, lanes))
                });
            if conflicts {
                out.insert((b, n));
                since.fill(0);
            }
            since[x / 64] |= 1 << (x % 64);
        }
        since
    };
    let n = p.blocks.len();
    let mut entry = vec![vec![0u64; words]; n];
    let mut seen = vec![false; n];
    let mut work = vec![0];
    let mut scratch = HashSet::new();
    while let Some(b) = work.pop() {
        seen[b] = true;
        let since = walk(b, entry[b].clone(), &mut scratch);
        for s in p.blocks[b].exit.successors() {
            let grows = entry[s].iter().zip(&since).any(|(&e, &x)| x & !e != 0);
            if !seen[s] || grows {
                for (e, x) in entry[s].iter_mut().zip(&since) {
                    *e |= x;
                }
                work.push(s);
            }
        }
    }
    let mut out = HashSet::new();
    for b in (0..n).filter(|&b| seen[b]) {
        walk(b, entry[b].clone(), &mut out);
    }
    out
}

/// A value modulo 2^32 as `c + lane[t] + Σ coefficient · atom` for the
/// group's thread `t` (flat id, x fastest): `lane` is what each thread
/// computes from its thread id alone (`None`: 0 in every thread); an atom is
/// a value every lane of a console wave holds alike, the same in every pass
/// of the program's loops (defined outside them). Only in the lanes of
/// every EXEC in `guards` (both dwords): vector instructions computed it
/// there, and other lanes kept older values.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
struct Affine {
    c: u32,
    lane: Option<Rc<[u32]>>,
    atoms: BTreeMap<Value, u32>,
    guards: BTreeSet<(Value, Value)>,
}

impl Affine {
    fn constant(c: u32) -> Affine {
        Affine {
            c,
            ..Affine::default()
        }
    }

    fn atom(v: Value) -> Affine {
        Affine {
            atoms: BTreeMap::from([(v, 1)]),
            ..Affine::default()
        }
    }

    /// Each thread's own value.
    fn lanes(values: Vec<u32>) -> Affine {
        match values.first() {
            Some(&c) if values.iter().any(|&v| v != c) => Affine {
                lane: Some(values.into()),
                ..Affine::default()
            },
            first => Affine::constant(first.copied().unwrap_or(0)),
        }
    }

    fn add(&self, o: &Affine) -> Affine {
        let mut atoms = self.atoms.clone();
        for (&v, &k) in &o.atoms {
            let e = atoms.entry(v).or_insert(0);
            *e = e.wrapping_add(k);
            if *e == 0 {
                atoms.remove(&v);
            }
        }
        let mut sum = match (&self.lane, &o.lane) {
            (Some(x), Some(y)) => Affine::lanes(
                x.iter()
                    .zip(y.iter())
                    .map(|(a, b)| a.wrapping_add(*b))
                    .collect(),
            ),
            (l, None) | (None, l) => Affine {
                lane: l.clone(),
                ..Affine::default()
            },
        };
        sum.c = sum.c.wrapping_add(self.c).wrapping_add(o.c);
        sum.atoms = atoms;
        sum.guards = self.guards.union(&o.guards).copied().collect();
        sum
    }

    fn scale(&self, m: u32) -> Affine {
        let mut out = match &self.lane {
            Some(l) => Affine::lanes(l.iter().map(|k| k.wrapping_mul(m)).collect()),
            None => Affine::default(),
        };
        out.c = out.c.wrapping_add(self.c.wrapping_mul(m));
        out.atoms = self
            .atoms
            .iter()
            .map(|(&v, &k)| (v, k.wrapping_mul(m)))
            .filter(|&(_, k)| k != 0)
            .collect();
        out.guards = self.guards.clone();
        out
    }

    /// The same value, guards aside.
    fn same(&self, o: &Affine) -> bool {
        self.c == o.c && self.lane == o.lane && self.atoms == o.atoms
    }

    /// This value, held only in the lanes of `guards` too.
    fn guarded(&self, guards: &BTreeSet<(Value, Value)>) -> Affine {
        let mut out = self.clone();
        out.guards.extend(guards);
        out
    }

    fn sub(&self, o: &Affine) -> Affine {
        self.add(&o.scale(u32::MAX))
    }

    fn known(&self) -> Option<u32> {
        (self.uniform() && self.atoms.is_empty()).then_some(self.c)
    }

    fn uniform(&self) -> bool {
        self.lane.is_none()
    }

    /// Thread `t`'s value without the atoms.
    fn at(&self, t: usize) -> u32 {
        self.c.wrapping_add(self.lane.as_ref().map_or(0, |l| l[t]))
    }

    /// Every thread's value, with no atoms.
    fn values(&self, threads: usize) -> Option<Vec<u32>> {
        self.atoms
            .is_empty()
            .then(|| (0..threads).map(|t| self.at(t)).collect())
    }

    /// The least common power of two dividing every term (32: none).
    fn zeros(&self) -> u32 {
        let lane = self.lane.iter().flat_map(|l| l.iter());
        self.atoms
            .values()
            .chain(lane)
            .fold(self.c.trailing_zeros(), |z, k| z.min(k.trailing_zeros()))
    }
}

/// What resource an access goes through.
#[derive(Debug, Clone, PartialEq, Eq)]
enum Base {
    /// A descriptor loaded from (or built around) a root parameter's
    /// address: parameter, byte offset, the address's computed term,
    /// first dword, and what a built one ORs in.
    View {
        param: usize,
        offset: u32,
        dynamic: Option<Affine>,
        first: u8,
        built: Option<(u32, [u32; 2])>,
    },
    /// Title memory at an address held in SGPRs (both dwords), or (`None`
    /// low dword) the high dword of a VGPR address.
    Global(Option<Affine>, Affine),
}

/// Where an access reads or writes, as far as the analysis knows.
#[derive(Debug, Clone, PartialEq, Eq)]
struct Location {
    base: Base,
    /// Bytes: the byte offset. Elements: the element's address dwords
    /// (texel coordinates, a typed buffer's index...).
    coords: Vec<Affine>,
    /// Bytes from the offset (`None`: one element).
    bytes: Option<u32>,
}

/// Whether the access `e` by one lane and `a` by another lane of the same
/// console wave may touch the same bytes; `lanes` threads in the group.
fn may_alias(e: &Option<Location>, a: &Option<Location>, lanes: usize) -> bool {
    let (Some(e), Some(a)) = (e, a) else {
        return true;
    };
    if e.base != a.base
        || e.coords.len() != a.coords.len()
        || e.bytes.is_some() != a.bytes.is_some()
        || e.coords
            .iter()
            .zip(&a.coords)
            .any(|(x, y)| x.atoms != y.atoms)
    {
        return true;
    }
    // (the atoms are the same on both sides: only the rest differs)
    let waves = (0..lanes).step_by(64).map(|w| w..lanes.min(w + 64));
    match (e.bytes, a.bytes) {
        // Lane L's bytes [x, x + eb) and lane M's [y, y + ab) meet when
        // -ab < y - x < eb.
        (Some(eb), Some(ab)) => {
            let (x, y) = (&e.coords[0], &a.coords[0]);
            let meet = |l: usize, m: usize| {
                let d = i64::from(y.at(m).wrapping_sub(x.at(l)) as i32);
                d > -i64::from(ab) && d < i64::from(eb)
            };
            if x.uniform() && y.uniform() {
                return meet(0, 0);
            }
            waves
                .into_iter()
                .any(|w| w.clone().any(|l| w.clone().any(|m| l != m && meet(l, m))))
        }
        // The same element: every address dword equal.
        _ => waves.into_iter().any(|w| {
            let mut seen: HashMap<Vec<u32>, usize> = HashMap::new();
            for m in w.clone() {
                let key = a.coords.iter().map(|c| c.at(m)).collect();
                // (two lanes are enough to tell)
                seen.entry(key).and_modify(|k| *k = usize::MAX).or_insert(m);
            }
            w.into_iter().any(|l| {
                let key: Vec<u32> = e.coords.iter().map(|c| c.at(l)).collect();
                seen.get(&key).is_some_and(|&m| m != l)
            })
        }),
    }
}

#[derive(Debug, Clone, Copy)]
enum Def {
    Input,
    Phi(usize, usize),
    Inst(usize, usize),
}

/// Addresses as [`Affine`]s of the program's values.
struct Addresses<'a> {
    p: &'a Program,
    g: &'a Group<'a>,
    /// Threads in the group.
    lanes: usize,
    defs: HashMap<Value, Def>,
    scalar: HashMap<Site, Address>,
    /// Blocks in a cycle (that run more than once).
    cyclic: Vec<bool>,
    values: HashMap<Value, Option<Affine>>,
    full: HashMap<Value, bool>,
    within: HashMap<(Value, Value), bool>,
}

/// Deeper chains of arithmetic than this are not followed.
const DEPTH: u32 = 64;

impl<'a> Addresses<'a> {
    fn new(p: &'a Program, g: &'a Group<'a>) -> Self {
        let mut defs = HashMap::new();
        for &(_, v) in &p.inputs {
            defs.insert(v, Def::Input);
        }
        for (b, block) in p.blocks.iter().enumerate() {
            for (n, phi) in block.phis.iter().enumerate() {
                defs.insert(phi.value, Def::Phi(b, n));
            }
            for (n, i) in block.insts.iter().enumerate() {
                for &(_, v) in &i.writes {
                    defs.insert(v, Def::Inst(b, n));
                }
            }
        }
        Addresses {
            p,
            g,
            lanes: g.threads.iter().product::<u32>() as usize,
            defs,
            scalar: g.res.scalar.iter().copied().collect(),
            cyclic: cyclic(p),
            values: HashMap::new(),
            full: HashMap::new(),
            within: HashMap::new(),
        }
    }

    /// The descriptor the access at `site` reads.
    fn descriptor(&self, site: Site) -> Option<&'a Source> {
        let g = self.g;
        g.res
            .descriptors
            .iter()
            .find(|d| d.site == site && d.role == Role::Resource)
            .map(|d| &d.source)
    }

    /// Where the access at `(b, n)` goes, if known: in the lanes that run
    /// it, its address values must hold (their guards' EXEC covers the
    /// access's).
    fn location(&mut self, b: usize, n: usize) -> Option<Location> {
        let i = &self.p.blocks[b].insts[n];
        let at = self.place(b, n)?;
        let exec = i.read(Reg::S(EXEC)).zip(i.read(Reg::S(EXEC + 1)))?;
        let mut guards = BTreeSet::new();
        for c in &at.coords {
            guards.extend(&c.guards);
        }
        match &at.base {
            Base::View { dynamic, .. } => guards.extend(dynamic.iter().flat_map(|d| &d.guards)),
            Base::Global(lo, hi) => {
                guards.extend(lo.iter().flat_map(|d| &d.guards));
                guards.extend(&hi.guards);
            }
        }
        guards
            .into_iter()
            .all(|(lo, hi)| self.within(exec.0, lo, 0) && self.within(exec.1, hi, 0))
            .then_some(at)
    }

    /// Whether the lanes of mask `x` are among those of `t` (one dword of
    /// a lane mask each): `x` is `t`, `t` holds every lane, or `x` is
    /// made from `t` by ANDs and copies (`s_and_saveexec_b64`,
    /// `v_cmpx_*`...), ORs of such, and phis of such.
    fn within(&mut self, x: Value, t: Value, depth: u32) -> bool {
        if x == t || self.full(t, 0) {
            return true;
        }
        if depth > DEPTH {
            return false;
        }
        if let Some(&r) = self.within.get(&(x, t)) {
            return r;
        }
        self.within.insert((x, t), false);
        let p = self.p;
        let r = match self.defs.get(&x).copied() {
            Some(Def::Phi(b, n)) => p.blocks[b].phis[n]
                .args
                .iter()
                .filter(|&&a| a != x)
                .all(|&a| self.within(a, t, depth + 1)),
            Some(Def::Inst(b, n)) => {
                let i = &p.blocks[b].insts[n];
                let name = i.name();
                let r = i.writes.iter().find(|w| w.1 == x).map(|w| w.0);
                let old_exec = |s: &mut Self, k: u8| {
                    i.read(Reg::S(EXEC + k))
                        .is_some_and(|e| s.within(e, t, depth + 1))
                };
                match (&i.fields, r) {
                    (
                        Fields::Sop {
                            dst: Some(Operand::Scalar { code, .. }),
                            src,
                            ..
                        },
                        Some(Reg::S(r)),
                    ) => {
                        let exec = (EXEC..=EXEC + 1).contains(&r);
                        let k = if exec && name.contains("saveexec") {
                            r - EXEC
                        } else {
                            match r.checked_sub(*code) {
                                Some(k) => k,
                                None => return false,
                            }
                        };
                        let src_in = |s: &mut Self, j: usize| match src.get(j) {
                            Some(&Operand::Scalar { code, .. }) if code != NULL => i
                                .read(Reg::S(code + k))
                                .is_some_and(|e| s.within(e, t, depth + 1)),
                            _ => false,
                        };
                        match name {
                            // The saved EXEC.
                            _ if !exec && name.contains("saveexec") => old_exec(self, k),
                            // EXEC & src, ~src & EXEC.
                            "s_and_saveexec_b64" | "s_and_saveexec_b32" => {
                                old_exec(self, k) || src_in(self, 0)
                            }
                            "s_andn1_saveexec_b64" | "s_andn1_saveexec_b32" => old_exec(self, k),
                            "s_andn2_saveexec_b64" | "s_andn2_saveexec_b32" => src_in(self, 0),
                            "s_or_saveexec_b64" | "s_or_saveexec_b32" => {
                                old_exec(self, k) && src_in(self, 0)
                            }
                            "s_mov_b64" | "s_mov_b32" => src_in(self, 0),
                            "s_and_b64" | "s_and_b32" => src_in(self, 0) || src_in(self, 1),
                            "s_andn2_b64" | "s_andn2_b32" => src_in(self, 0),
                            "s_or_b64" | "s_or_b32" => src_in(self, 0) && src_in(self, 1),
                            _ => false,
                        }
                    }
                    // A compare's bits, or `v_cmpx`'s EXEC: lanes of EXEC
                    // only.
                    (Fields::Valu(v), Some(Reg::S(r))) if name.starts_with("v_cmp") => {
                        let code = match v.dst {
                            Some(Operand::Scalar { code, .. }) => code,
                            _ => EXEC,
                        };
                        r.checked_sub(code).is_some_and(|k| old_exec(self, k))
                    }
                    _ => false,
                }
            }
            _ => false,
        };
        self.within.insert((x, t), r);
        r
    }

    /// [`Self::location`] before the guards are checked.
    fn place(&mut self, b: usize, n: usize) -> Option<Location> {
        let i = &self.p.blocks[b].insts[n];
        let bytes = |o: &Option<Operand>| match o {
            Some(Operand::Vector { dwords, .. }) => 4 * u32::from(*dwords),
            _ => 16,
        };
        match &i.fields {
            Fields::Buffer {
                data,
                addr,
                soffset,
                offset,
                offen,
                idxen,
                ..
            } => {
                let src = self.descriptor((b, n))?;
                let base = self.view(src)?;
                let mut off = self.operand(i, soffset, 0, 0)?;
                off = off.add(&Affine::constant(u32::from(*offset)));
                if *offen {
                    off = off.add(&self.operand(i, addr, u8::from(*idxen), 0)?);
                }
                if !*idxen {
                    return Some(Location {
                        base,
                        coords: vec![off],
                        bytes: Some(bytes(data)),
                    });
                }
                // By element: the element times the stride the host
                // addresses it with (a root view's own, the used
                // resource's), or a typed buffer's element.
                let index = self.operand(i, addr, 0, 0)?;
                let used = used_resource(self.g.native, &self.scalar, src);
                let stride = match src {
                    _ if i.name().contains("format") => {
                        return Some(Location {
                            base,
                            coords: vec![index, off],
                            bytes: None,
                        });
                    }
                    Source::Built { bits, .. } => bits >> 16 & 0x3FFF,
                    _ => match used? {
                        u if u.kind == ResourceType::Cbv => 16,
                        u if u.stride >= 4 && u.stride.is_multiple_of(4) => u.stride,
                        _ => return None,
                    },
                };
                Some(Location {
                    base,
                    coords: vec![index.scale(stride).add(&off)],
                    bytes: Some(bytes(data)),
                })
            }
            Fields::Image { addr, dim, a16, .. } if !i.name().starts_with("image_bvh") && !*a16 => {
                let base = self.view(self.descriptor((b, n))?)?;
                let mut coords = Vec::new();
                for o in addr {
                    let Operand::Vector { dwords, .. } = *o else {
                        return None;
                    };
                    for k in 0..dwords {
                        coords.push(self.operand(i, o, k, 0)?);
                    }
                }
                // x, y, z of the dimension: array slices, cube faces,
                // samples and mip levels must be the same in every lane
                // (the hardware may clamp them).
                let spatial = [1, 2, 3, 2, 1, 2, 2, 2][usize::from(*dim & 7)];
                if coords.len() < spatial || coords[spatial..].iter().any(|c| !c.uniform()) {
                    return None;
                }
                Some(Location {
                    base,
                    coords,
                    bytes: None,
                })
            }
            Fields::Flat {
                dst,
                addr: Some(addr),
                data,
                saddr,
                offset,
                ..
            } => {
                let off = Affine::constant(*offset as i32 as u32);
                let (base, at) = match saddr {
                    Some(s @ Operand::Scalar { code, .. }) if *code != NULL => {
                        let lo = self.operand(i, s, 0, 0)?;
                        let hi = self.operand(i, s, 1, 0)?;
                        let lane = self.operand(i, addr, 0, 0)?;
                        (Base::Global(Some(lo), hi), lane.add(&off))
                    }
                    _ => {
                        let hi = self.operand(i, addr, 1, 0)?;
                        (
                            Base::Global(None, hi),
                            self.operand(i, addr, 0, 0)?.add(&off),
                        )
                    }
                };
                if let Base::Global(lo, hi) = &base
                    && (!hi.uniform() || lo.as_ref().is_some_and(|l| !l.uniform()))
                {
                    return None;
                }
                Some(Location {
                    base,
                    coords: vec![at],
                    bytes: Some(bytes(dst).max(bytes(data)).min(16)),
                })
            }
            _ => None,
        }
    }

    /// The descriptor `src` as a [`Base`], if the same in every lane and
    /// every pass.
    fn view(&mut self, src: &Source) -> Option<Base> {
        let (address, first, built) = match src {
            Source::Loaded { load, first } => (*self.scalar.get(load)?, *first, None),
            Source::Built {
                address,
                bits,
                words,
            } => {
                // (no swizzled buffers: ADD_TID_ENABLE)
                if words[1] & 1 << 23 != 0 {
                    return None;
                }
                (*address, 0, Some((*bits, *words)))
            }
            _ => return None,
        };
        let dynamic = match address.dynamic {
            Some(v) => Some(self.value(v, 0).filter(Affine::uniform)?),
            None => None,
        };
        Some(Base::View {
            param: address.param,
            offset: address.offset,
            dynamic,
            first,
            built,
        })
    }

    /// Dword `k` of operand `o` of `i`.
    fn operand(&mut self, i: &Inst, o: &Operand, k: u8, depth: u32) -> Option<Affine> {
        match *o {
            Operand::Scalar { code: NULL, .. } => Some(Affine::constant(0)),
            Operand::Scalar { code, .. } => self.value(i.read(Reg::S(code + k))?, depth),
            Operand::Vector { reg, .. } => self.value(i.read(Reg::V(reg + k))?, depth),
            Operand::Const(c) if k == 0 => crate::resource::inline(c).map(Affine::constant),
            Operand::Literal(x) if k == 0 => Some(Affine::constant(x)),
            _ => None,
        }
    }

    fn value(&mut self, v: Value, depth: u32) -> Option<Affine> {
        if let Some(a) = self.values.get(&v) {
            return a.clone();
        }
        if depth > DEPTH {
            return None;
        }
        // (a cycle through phis is not followed)
        self.values.insert(v, None);
        let a = match self.defs.get(&v).copied() {
            Some(Def::Input) => {
                let reg = self.p.inputs.iter().find(|x| x.1 == v).map(|x| x.0);
                match self.g.tids.iter().position(|&t| t == Some(v)) {
                    Some(c) => {
                        let [x, y, _] = self.g.threads;
                        let id = |t: u32| [t % x, t / x % y, t / (x * y)][c];
                        Some(Affine::lanes((0..self.lanes as u32).map(id).collect()))
                    }
                    None if matches!(reg, Some(Reg::S(_))) => Some(Affine::atom(v)),
                    None => None,
                }
            }
            Some(Def::Phi(b, n)) => {
                let phi = &self.p.blocks[b].phis[n];
                // (the lanes of every arm's EXEC: a join is not checked)
                let mut agreed: Option<Option<Affine>> = None;
                for &x in phi.args.iter().filter(|&&x| x != v) {
                    let a = self.value(x, depth + 1);
                    agreed = Some(match (agreed, a) {
                        (None, a) => a,
                        (Some(Some(s)), Some(a)) if s.same(&a) => Some(s.guarded(&a.guards)),
                        _ => None,
                    });
                    if agreed == Some(None) {
                        break;
                    }
                }
                match agreed.flatten() {
                    Some(a) => Some(a),
                    // An SGPR joined outside loops: one value in the wave.
                    None if matches!(phi.reg, Reg::S(_)) && !self.cyclic[b] => {
                        Some(Affine::atom(v))
                    }
                    None => None,
                }
            }
            Some(Def::Inst(b, n)) => self.inst_value(b, n, v, depth + 1),
            None => None,
        };
        self.values.insert(v, a.clone());
        a
    }

    fn inst_value(&mut self, b: usize, n: usize, v: Value, depth: u32) -> Option<Affine> {
        let p = self.p;
        let i = &p.blocks[b].insts[n];
        let name = i.name();
        let reg = i.writes.iter().find(|w| w.1 == v)?.0;
        let once = !self.cyclic[b];
        match (&i.fields, reg) {
            (Fields::Sop { dst, src, imm }, Reg::S(r)) => {
                let k = match dst {
                    Some(Operand::Scalar { code, .. }) => r.checked_sub(*code)?,
                    _ => return once.then(|| Affine::atom(v)),
                };
                let mut op = |j: usize| -> Option<Affine> {
                    let o = src.get(j)?;
                    self.operand(i, o, k, depth)
                };
                let simm = imm.map(|x| x as i16 as i32 as u32);
                let a = match (name, k) {
                    ("s_mov_b32" | "s_mov_b64", _) => op(0),
                    ("s_movk_i32", 0) => simm.map(Affine::constant),
                    ("s_add_u32" | "s_add_i32", 0) => op(0).zip(op(1)).map(|(x, y)| x.add(&y)),
                    ("s_sub_u32" | "s_sub_i32", 0) => op(0).zip(op(1)).map(|(x, y)| x.sub(&y)),
                    ("s_lshl_b32", 0) => {
                        let s = op(1)?.known()? & 31;
                        op(0).map(|x| x.scale(1 << s))
                    }
                    ("s_mul_i32", 0) => mul(op(0), op(1)),
                    (
                        "s_lshl1_add_u32" | "s_lshl2_add_u32" | "s_lshl3_add_u32"
                        | "s_lshl4_add_u32",
                        0,
                    ) => {
                        let s = u32::from(name.as_bytes()[6] - b'0');
                        op(0).zip(op(1)).map(|(x, y)| x.scale(1 << s).add(&y))
                    }
                    ("s_addk_i32" | "s_mulk_i32", 0) => {
                        let old = self.value(i.read(Reg::S(r))?, depth)?;
                        let k = Affine::constant(simm?);
                        if name == "s_addk_i32" {
                            Some(old.add(&k))
                        } else {
                            mul(Some(old), Some(k))
                        }
                    }
                    _ => None,
                };
                // Any SGPR is one value in the console wave.
                a.or_else(|| once.then(|| Affine::atom(v)))
            }
            (_, Reg::S(_)) => once.then(|| Affine::atom(v)),
            (Fields::Valu(x), Reg::V(_)) => {
                let plain = x.sdwa.is_none()
                    && x.dpp.is_none()
                    && !x.clamp
                    && x.omod == 0
                    && x.neg == 0
                    && x.abs == 0;
                if !plain {
                    return None;
                }
                // Lanes outside EXEC keep the old value: the value holds in
                // the lanes of this EXEC.
                let exec = i.read(Reg::S(EXEC)).zip(i.read(Reg::S(EXEC + 1)))?;
                let mut guards = BTreeSet::new();
                if !self.full(exec.0, 0) || !self.full(exec.1, 0) {
                    guards.insert(exec);
                }
                if name.starts_with("v_mbcnt_") {
                    return Some(self.lane_count(i, &x.src, depth)?.guarded(&guards));
                }
                let mut ops = Vec::new();
                for o in &x.src {
                    ops.push(self.operand(i, o, 0, depth));
                }
                for o in ops.iter().flatten() {
                    guards.extend(&o.guards);
                }
                let lanes = self.lanes;
                let a = valu(name, &ops, lanes).or_else(|| {
                    // A pure function of values the wave holds alike.
                    let uniform = ops.iter().all(|o| o.as_ref().is_some_and(Affine::uniform));
                    (once && uniform && PURE.contains(&name)).then(|| Affine::atom(v))
                })?;
                Some(a.guarded(&guards))
            }
            _ => None,
        }
    }

    /// `v_mbcnt_lo/hi_u32_b32` of every lane: the lanes below in the
    /// console wave, plus the addend.
    fn lane_count(&mut self, i: &Inst, src: &[Operand], depth: u32) -> Option<Affine> {
        let every = match *src.first()? {
            Operand::Const(c) => crate::resource::inline(c) == Some(u32::MAX),
            Operand::Literal(x) => x == u32::MAX,
            Operand::Scalar { code, .. } if code != NULL => {
                i.read(Reg::S(code)).is_some_and(|x| self.full(x, 0))
            }
            _ => false,
        };
        if !every {
            return None;
        }
        let addend = self.operand(i, src.get(1)?, 0, depth)?;
        let hi = i.name().starts_with("v_mbcnt_hi");
        let below = |t: usize| {
            let l = (t % 64) as u32;
            if hi { l.saturating_sub(32) } else { l.min(32) }
        };
        Some(addend.add(&Affine::lanes((0..self.lanes).map(below).collect())))
    }

    /// Whether lane mask dword `v` holds every lane of the console wave
    /// that runs the program: EXEC as on entry, or restored to that.
    fn full(&mut self, v: Value, depth: u32) -> bool {
        if let Some(&f) = self.full.get(&v) {
            return f;
        }
        if depth > DEPTH {
            return false;
        }
        self.full.insert(v, false);
        let p = self.p;
        let f = match self.defs.get(&v).copied() {
            Some(Def::Input) => p
                .inputs
                .iter()
                .any(|&(r, x)| x == v && matches!(r, Reg::S(EXEC) | Reg::S(127))),
            Some(Def::Phi(b, n)) => {
                let phi = &p.blocks[b].phis[n];
                phi.args
                    .iter()
                    .filter(|&&x| x != v)
                    .all(|&x| self.full(x, depth + 1))
            }
            Some(Def::Inst(b, n)) => {
                let i = &p.blocks[b].insts[n];
                let name = i.name();
                let Some(&(Reg::S(r), _)) = i.writes.iter().find(|w| w.1 == v) else {
                    return false;
                };
                let Fields::Sop {
                    dst: Some(Operand::Scalar { code, .. }),
                    src,
                    ..
                } = &i.fields
                else {
                    return false;
                };
                let src_full = |s: &mut Self, o: &Operand, k: u8| match *o {
                    Operand::Const(193) => true,
                    Operand::Literal(u32::MAX) => k == 0,
                    Operand::Scalar { code, .. } if code != NULL => i
                        .read(Reg::S(code + k))
                        .is_some_and(|x| s.full(x, depth + 1)),
                    _ => false,
                };
                let old_exec = |s: &mut Self, k: u8| {
                    i.read(Reg::S(EXEC + k))
                        .is_some_and(|x| s.full(x, depth + 1))
                };
                if (EXEC..=EXEC + 1).contains(&r) && name.contains("saveexec") {
                    // exec = src | exec: s_or_saveexec
                    let k = r - EXEC;
                    name.starts_with("s_or_saveexec")
                        && (old_exec(self, k) || src.first().is_some_and(|o| src_full(self, o, k)))
                } else if name.contains("saveexec") {
                    // The saved EXEC.
                    r.checked_sub(*code).is_some_and(|k| old_exec(self, k))
                } else {
                    let Some(k) = r.checked_sub(*code) else {
                        return false;
                    };
                    match name {
                        "s_mov_b64" | "s_mov_b32" => {
                            src.first().is_some_and(|o| src_full(self, o, k))
                        }
                        "s_or_b64" | "s_or_b32" => src.iter().any(|o| src_full(self, o, k)),
                        _ => false,
                    }
                }
            }
            None => false,
        };
        self.full.insert(v, f);
        f
    }
}

/// VALU operations of values the wave holds alike only: such a result is
/// an atom of its own where the program computes it once.
const PURE: &[&str] = &[
    "v_mul_lo_u32",
    "v_mul_hi_u32",
    "v_lshrrev_b32",
    "v_ashrrev_i32",
    "v_and_b32",
    "v_or_b32",
    "v_xor_b32",
    "v_bfe_u32",
    "v_bfe_i32",
    "v_lshl_or_b32",
    "v_and_or_b32",
    "v_or3_b32",
    "v_min_u32",
    "v_max_u32",
    "v_min_i32",
    "v_max_i32",
];

/// The result of VALU operation `name` on `ops` (a missing operand
/// unknown), for a group of `lanes` threads.
fn valu(name: &str, ops: &[Option<Affine>], lanes: usize) -> Option<Affine> {
    let op = |j: usize| ops.get(j).cloned().flatten();
    // Thread by thread, with no atoms.
    let each = |f: &dyn Fn(&[u32]) -> u32| -> Option<Affine> {
        let values: Option<Vec<Vec<u32>>> = (0..ops.len()).map(|j| op(j)?.values(lanes)).collect();
        let values = values?;
        Some(Affine::lanes(
            (0..lanes)
                .map(|t| f(&values.iter().map(|v| v[t]).collect::<Vec<_>>()))
                .collect(),
        ))
    };
    let u24 = |x: u32| x & 0xFF_FFFF;
    let i24 = |x: u32| ((x << 8) as i32 >> 8) as u32;
    let fits = |a: Option<Affine>, unsigned: bool| {
        let v = a.as_ref()?.values(lanes)?;
        v.iter()
            .all(|&x| {
                if unsigned {
                    x < 1 << 24
                } else {
                    (x as i32) >= -(1 << 23) && (x as i32) < 1 << 23
                }
            })
            .then_some(a)
            .flatten()
    };
    // (x << k) + y where the bits are apart: an OR.
    let apart = |x: Affine, y: Affine| -> Option<Affine> {
        let small = |a: &Affine, z: u32| {
            z < 32
                && a.values(lanes)
                    .is_some_and(|v| v.iter().all(|&k| k < 1 << z))
        };
        (small(&y, x.zeros()) || small(&x, y.zeros())).then(|| x.add(&y))
    };
    let linear = match name {
        "v_mov_b32" => op(0),
        "v_add_nc_u32" | "v_add_co_u32" | "v_add_nc_i32" => {
            op(0).zip(op(1)).map(|(x, y)| x.add(&y))
        }
        "v_sub_nc_u32" | "v_sub_co_u32" | "v_sub_nc_i32" => {
            op(0).zip(op(1)).map(|(x, y)| x.sub(&y))
        }
        "v_subrev_nc_u32" | "v_subrev_co_u32" => op(1).zip(op(0)).map(|(x, y)| x.sub(&y)),
        "v_add3_u32" => op(0)
            .zip(op(1))
            .zip(op(2))
            .map(|((x, y), z)| x.add(&y).add(&z)),
        "v_lshlrev_b32" => op(0)
            .and_then(|s| s.known())
            .and_then(|s| op(1).map(|x| x.scale(1 << (s & 31)))),
        "v_lshl_add_u32" => op(1).and_then(|s| s.known()).and_then(|s| {
            op(0)
                .zip(op(2))
                .map(|(x, y)| x.scale(1 << (s & 31)).add(&y))
        }),
        "v_add_lshl_u32" => op(2).and_then(|s| s.known()).and_then(|s| {
            op(0)
                .zip(op(1))
                .map(|(x, y)| x.add(&y).scale(1 << (s & 31)))
        }),
        "v_mul_lo_u32" => mul(op(0), op(1)),
        "v_mul_u32_u24" => mul(fits(op(0), true), fits(op(1), true)),
        "v_mul_i32_i24" => mul(fits(op(0), false), fits(op(1), false)),
        "v_mad_u32_u24" => mul(fits(op(0), true), fits(op(1), true))
            .zip(op(2))
            .map(|(x, y)| x.add(&y)),
        "v_mad_i32_i24" => mul(fits(op(0), false), fits(op(1), false))
            .zip(op(2))
            .map(|(x, y)| x.add(&y)),
        "v_or_b32" => op(0).zip(op(1)).and_then(|(x, y)| apart(x, y)),
        "v_lshl_or_b32" => op(1).and_then(|s| s.known()).and_then(|s| {
            op(0)
                .zip(op(2))
                .and_then(|(x, y)| apart(x.scale(1 << (s & 31)), y))
        }),
        _ => None,
    };
    linear.or_else(|| match name {
        "v_and_b32" => each(&|v| v[0] & v[1]),
        "v_or_b32" => each(&|v| v[0] | v[1]),
        "v_xor_b32" => each(&|v| v[0] ^ v[1]),
        "v_lshrrev_b32" => each(&|v| v[1] >> (v[0] & 31)),
        "v_ashrrev_i32" => each(&|v| ((v[1] as i32) >> (v[0] & 31)) as u32),
        "v_lshlrev_b32" => each(&|v| v[1] << (v[0] & 31)),
        "v_bfe_u32" => each(&|v| {
            let w = v[2] & 31;
            if w == 0 {
                0
            } else {
                v[0] >> (v[1] & 31) & ((1u64 << w) - 1) as u32
            }
        }),
        "v_bfe_i32" => each(&|v| {
            let w = v[2] & 31;
            if w == 0 {
                0
            } else {
                let x = v[0] >> (v[1] & 31);
                ((x << (32 - w)) as i32 >> (32 - w)) as u32
            }
        }),
        "v_lshl_or_b32" => each(&|v| v[0] << (v[1] & 31) | v[2]),
        "v_lshl_add_u32" => each(&|v| (v[0] << (v[1] & 31)).wrapping_add(v[2])),
        "v_add_lshl_u32" => each(&|v| v[0].wrapping_add(v[1]) << (v[2] & 31)),
        "v_and_or_b32" => each(&|v| v[0] & v[1] | v[2]),
        "v_or3_b32" => each(&|v| v[0] | v[1] | v[2]),
        "v_xad_u32" => each(&|v| (v[0] ^ v[1]).wrapping_add(v[2])),
        "v_mul_lo_u32" => each(&|v| v[0].wrapping_mul(v[1])),
        "v_mul_hi_u32" => each(&|v| ((u64::from(v[0]) * u64::from(v[1])) >> 32) as u32),
        "v_mul_u32_u24" => each(&|v| u24(v[0]).wrapping_mul(u24(v[1]))),
        "v_mul_i32_i24" => each(&|v| i24(v[0]).wrapping_mul(i24(v[1]))),
        "v_mad_u32_u24" => each(&|v| u24(v[0]).wrapping_mul(u24(v[1])).wrapping_add(v[2])),
        "v_mad_i32_i24" => each(&|v| i24(v[0]).wrapping_mul(i24(v[1])).wrapping_add(v[2])),
        "v_min_u32" => each(&|v| v[0].min(v[1])),
        "v_max_u32" => each(&|v| v[0].max(v[1])),
        "v_min_i32" => each(&|v| (v[0] as i32).min(v[1] as i32) as u32),
        "v_max_i32" => each(&|v| (v[0] as i32).max(v[1] as i32) as u32),
        "v_med3_u32" => each(&|v| v[0].max(v[1]).min(v[0].min(v[1]).max(v[2]))),
        "v_med3_i32" => each(&|v| {
            let [a, b, c] = [v[0], v[1], v[2]].map(|x| x as i32);
            a.max(b).min(a.min(b).max(c)) as u32
        }),
        _ => None,
    })
}

/// A product with a known factor.
fn mul(x: Option<Affine>, y: Option<Affine>) -> Option<Affine> {
    let (x, y) = (x?, y?);
    match (x.known(), y.known()) {
        (Some(k), _) => Some(y.scale(k)),
        (_, Some(k)) => Some(x.scale(k)),
        _ => None,
    }
}

/// The blocks in a cycle of the control flow graph.
fn cyclic(p: &Program) -> Vec<bool> {
    let n = p.blocks.len();
    let succs = |b: usize| p.blocks[b].exit.successors();
    // Kosaraju: finish order on the graph, then components on its reverse.
    let mut order = Vec::with_capacity(n);
    let mut seen = vec![false; n];
    for root in 0..n {
        if seen[root] {
            continue;
        }
        seen[root] = true;
        let mut stack = vec![(root, 0)];
        while let Some(&mut (b, ref mut next)) = stack.last_mut() {
            let s = succs(b);
            if let Some(&c) = s.get(*next) {
                *next += 1;
                if !seen[c] {
                    seen[c] = true;
                    stack.push((c, 0));
                }
            } else {
                order.push(b);
                stack.pop();
            }
        }
    }
    let mut comp = vec![usize::MAX; n];
    let mut size = Vec::new();
    for &root in order.iter().rev() {
        if comp[root] != usize::MAX {
            continue;
        }
        let id = size.len();
        size.push(0);
        comp[root] = id;
        let mut work = vec![root];
        while let Some(b) = work.pop() {
            size[id] += 1;
            for &q in &p.blocks[b].preds {
                if comp[q] == usize::MAX {
                    comp[q] = id;
                    work.push(q);
                }
            }
        }
    }
    (0..n)
        .map(|b| size[comp[b]] > 1 || succs(b).contains(&b))
        .collect()
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
    use crate::native::{ComputeIo, RootParam, RootType, StageIo};

    /// The byte offsets of the instructions the halves meet before, for a
    /// group of `threads` with a UAV table in user SGPR 0 (s0).
    fn marked_in(words: &[u32], raw: bool, threads: [u32; 3]) -> Vec<usize> {
        let mut native = crate::emit::tests::compute(words, true);
        native.layout = vec![RootParam {
            root: RootType::Table,
            table_offset_dwords: 0,
            sgprs: 1,
            base_register: 0,
            user_data: 0,
            slot: 0,
        }];
        native.io = StageIo::Cs(ComputeIo {
            rsrc2: 1 << 1 | 2 << 11,
            threads,
        });
        let p = crate::ir::build(&native.code, true, 8).unwrap();
        let res =
            crate::resource::track(&p, &native.layout, native.stage, None).unwrap_or_default();
        let tid = |c: u8| {
            p.inputs
                .iter()
                .find(|x| x.0 == crate::ir::Reg::V(c))
                .map(|x| x.1)
        };
        let g = super::Group {
            native: &native,
            res: &res,
            tids: [tid(0), tid(1), tid(2)],
            threads,
        };
        let mut at: Vec<usize> = super::meetings(&p, &g, raw)
            .into_iter()
            .map(|(b, i)| p.blocks[b].insts[i].machine.offset)
            .collect();
        at.sort_unstable();
        at
    }

    fn marked(words: &[u32], raw: bool) -> Vec<usize> {
        marked_in(words, raw, [64, 1, 1])
    }

    const STORE: [u32; 2] = [0xE070_1000, 0x8001_0203]; // buffer_store_dword v2, v3, s[4:7], 0 offen
    const LOAD: [u32; 2] = [0xE030_1000, 0x8001_0203]; // buffer_load_dword v2, v3, s[4:7], 0 offen

    /// A read after a write, then (past a title barrier) a write after a
    /// read; the read right after the barrier needs nothing. Reads after
    /// writes alone: the first only.
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
        assert_eq!(marked(&words, false), [8, 28]);
        assert_eq!(marked(&words, true), [8]);
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
        assert_eq!(marked(&words, false), [4, 12]);
        assert_eq!(marked(&words, true), [4]);
    }

    /// Loads alone, or stores after nothing, need no meeting.
    #[test]
    fn no_conflict_no_meeting() {
        let loads = [&LOAD[..], &LOAD, &[0xBF81_0000]].concat();
        assert!(marked(&loads, false).is_empty());
        let store = [&STORE[..], &[0xBF81_0000]].concat();
        assert!(marked(&store, false).is_empty());
    }

    /// The UAV table's descriptor in s[4:7] (user SGPR 0 holds the table).
    const TABLE: [u32; 5] = [
        0xBE82_0300, // s_mov_b32 s2, s0
        0xBE83_0380, // s_mov_b32 s3, 0
        0xF408_0101,
        0xFA00_0000, // s_load_dwordx4 s[4:7], s[2:3], null
        0xBF8C_C07F, // s_waitcnt lgkmcnt(0)
    ];

    /// Each lane loads, then stores, the dword at `address` (from v0, the
    /// thread id x, into v3), then loads it again.
    fn own_dword(address: &[u32]) -> Vec<u32> {
        let tail = [
            0xE030_1000,
            0x8001_0203, // buffer_load_dword v2, v3, s[4:7], 0 offen
            0xBF8C_3F70, // s_waitcnt vmcnt(0)
            0x4A04_0481, // v_add_nc_u32 v2, 1, v2
            0xE070_1000,
            0x8001_0203, // buffer_store_dword v2, v3, s[4:7], 0 offen
            0xE030_1000,
            0x8001_0403, // buffer_load_dword v4, v3, s[4:7], 0 offen
            0xBF81_0000, // s_endpgm
        ];
        [&TABLE[..], address, &tail].concat()
    }

    /// A lane's read-modify-write of its own dword (4 bytes per thread
    /// id) conflicts with no other lane's: no meeting. With 2 bytes per
    /// thread, a shared dword, or lanes of other rows (y) at the same x,
    /// accesses of different lanes overlap.
    #[test]
    fn own_dwords_need_no_meeting() {
        let lanes = |shift: u32| own_dword(&[0x3406_0080 | shift]); // v_lshlrev_b32 v3, shift, v0
        let (store, reload) = (40, 48);
        assert!(marked(&lanes(2), false).is_empty());
        assert_eq!(marked_in(&lanes(2), false, [8, 8, 1]), [store, reload]);
        let offset = [0x3406_0082, 0x4A06_06FF, 0x40]; // v_lshlrev_b32 v3, 2, v0; v_add_nc_u32 v3, 0x40, v3
        assert!(marked(&own_dword(&offset), false).is_empty());
        assert_eq!(marked(&lanes(1), false), [store, reload]);
        assert_eq!(marked(&lanes(1), true), [reload]);
        let shared = own_dword(&[0x7E06_0280]); // v_mov_b32 v3, 0
        assert_eq!(marked(&shared, false), [store, reload]);
        // Thread ids x and y as rows of 8 dwords: 8 x 8 apart; 16 x 4
        // overlap.
        let rows = [
            0x3406_0082, // v_lshlrev_b32 v3, 2, v0
            0xD746_0003,
            0x040D_0B01, // v_lshl_add_u32 v3, v1, 5, v3
        ];
        assert!(marked_in(&own_dword(&rows), false, [8, 8, 1]).is_empty());
        assert_eq!(
            marked_in(&own_dword(&rows), false, [16, 4, 1]),
            [store + 8, reload + 8]
        );
    }

    /// Each lane's dword from its thread id by operations other than
    /// sums (an 8 x 8 tile of a flat id): its own still.
    #[test]
    fn lane_functions_of_the_thread_id() {
        let tile = [
            0x3602_0087, // v_and_b32 v1, 7, v0 (x in the tile)
            0x2C0A_0083, // v_lshrrev_b32 v5, 3, v0 (y)
            0xD746_0003,
            0x0405_0B05, // v_lshl_add_u32 v3, v5, 5, v1 ((y << 5) + x)
            0x3406_0682, // v_lshlrev_b32 v3, 2, v3
        ];
        assert!(marked(&own_dword(&tile), false).is_empty());
        // x alone: lanes of every row share it.
        let column = [0x3602_0087, 0x3406_0282]; // v_and_b32 v1, 7, v0; v_lshlrev_b32 v3, 2, v1
        assert_eq!(marked(&own_dword(&column), false), [44, 52]);
    }

    /// Under a lane test, the address register keeps the old value in the
    /// other lanes: not the thread's own dword.
    #[test]
    fn an_address_set_in_some_lanes_is_unknown() {
        let address = [
            0x7E06_0280, // v_mov_b32 v3, 0
            0x7D88_0080, // v_cmp_gt_u32 vcc, 0, v0 (never: every lane keeps 0)
            0xBE8A_246A, // s_and_saveexec_b64 s[10:11], vcc
            0x3406_0082, // v_lshlrev_b32 v3, 2, v0
            0x88FE_0A7E, // s_or_b64 exec, exec, s[10:11]
        ];
        let words = own_dword(&address);
        let at = marked(&words, false);
        assert_eq!(at.len(), 2, "{at:?}");
        // Restored EXEC: the shift runs in every lane again.
        let restored = [&address[..], &[0x3406_0082]].concat();
        assert!(marked(&own_dword(&restored), false).is_empty());
        // Address and accesses in the same branch: its lanes computed it.
        let branch = [
            0x7D88_00B0, // v_cmp_gt_u32 vcc, 48, v0
            0xBE8A_246A, // s_and_saveexec_b64 s[10:11], vcc
            0x3406_0082, // v_lshlrev_b32 v3, 2, v0
        ];
        assert!(marked(&own_dword(&branch), false).is_empty());
    }
}
