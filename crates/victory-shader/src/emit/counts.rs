//! What an emitted shader's `main` holds, counted from the container's
//! bitcode: the instructions the host's compiler is given, whatever made
//! them, so two revisions of the translator compare by reading what each
//! wrote. It reads the unabbreviated records `bitcode` writes and refuses
//! anything else (a module DXC wrote has abbreviations).
//!
//! The counts are static: one per instruction written, however often it
//! runs. They rank what the translation adds and show what a change takes
//! out; they are no measure of time.

use std::collections::BTreeMap;

use crate::dxbc::{Container, FourCC};

type Why = &'static str;

/// The instructions of one shader's `main`, or of several added up.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Counts {
    /// Basic blocks.
    pub blocks: u64,
    /// Instructions, terminators among them.
    pub instructions: u64,
    /// The same by kind: LLVM's name of the instruction, and for a call
    /// the function called.
    pub kinds: BTreeMap<String, u64>,
}

impl Counts {
    pub fn add(&mut self, other: &Counts) {
        self.blocks += other.blocks;
        self.instructions += other.instructions;
        for (kind, n) in &other.kinds {
            *self.kinds.entry(kind.clone()).or_default() += n;
        }
    }
}

/// Counts the `main` of the shader in `container` (a DXIL container as
/// [`super::HostShader::dxil`] holds).
pub fn of(container: &[u8]) -> Result<Counts, Why> {
    let c = Container::parse(container).map_err(|_| "no container")?;
    let part = c.part(FourCC::DXIL).ok_or("no DXIL part")?.data;
    // The program header: version, size, "DXIL", DXIL version, then the
    // bitcode's offset from the magic and its size.
    let dword = |at: usize| -> Result<usize, Why> {
        let b = part.get(at..at + 4).ok_or("a truncated program header")?;
        Ok(u32::from_le_bytes(b.try_into().expect("4")) as usize)
    };
    let start = 8 + dword(16)?;
    let bitcode = part
        .get(start..start + dword(20)?)
        .ok_or("bitcode past the part")?;
    count(bitcode)
}

const MODULE_BLOCK: u64 = 8;
const CONSTANTS_BLOCK: u64 = 11;
const FUNCTION_BLOCK: u64 = 12;
const VALUE_SYMTAB_BLOCK: u64 = 14;
const TYPE_BLOCK: u64 = 17;

/// Counts the one function body of a module `bitcode::Module::write` wrote.
pub(crate) fn count(bitcode: &[u8]) -> Result<Counts, Why> {
    if bitcode.get(..4) != Some(b"BC\xC0\xDE") {
        return Err("no bitcode");
    }
    let mut r = Bits {
        data: bitcode,
        at: 32,
    };
    let mut m = Module::default();
    // The stream holds the module block, entered at the top-level width.
    if r.fixed(2)? != 1 || r.vbr(8)? != MODULE_BLOCK {
        return Err("no module block");
    }
    let width = r.vbr(4)? as u32;
    r.align();
    r.fixed(32)?;
    m.block(&mut r, MODULE_BLOCK, width)?;
    let mut counts = m.counts;
    for (callee, n) in m.calls {
        let name = m.names.remove(&callee).unwrap_or_else(|| "call".into());
        *counts.kinds.entry(name).or_default() += n;
    }
    Ok(counts)
}

enum Type {
    Void,
    Float,
    Vector(u64),
    Array(u64),
    Struct(Vec<u64>),
    Function(u64),
    Other,
}

#[derive(Clone, Copy)]
enum ValueType {
    Void,
    /// A result whose type cannot be a float or contain a float field.
    Other,
    Id(u64),
}

#[derive(Default)]
struct Module {
    types: Vec<Type>,
    /// Value ids run through globals, functions, constants, then the
    /// instructions that have a value.
    values: Vec<ValueType>,
    constant_type: Option<u64>,
    names: BTreeMap<u64, String>,
    /// Calls by the value id called.
    calls: BTreeMap<u64, u64>,
    counts: Counts,
}

impl Module {
    /// Reads block `id` up to its end; `r` is just past its length word.
    fn block(&mut self, r: &mut Bits, id: u64, width: u32) -> Result<(), Why> {
        if !(2..=32).contains(&width) {
            return Err("a block's abbreviation width");
        }
        let mut ops = Vec::new();
        loop {
            match r.fixed(width)? {
                0 => {
                    r.align();
                    return Ok(());
                }
                1 => {
                    let inner = r.vbr(8)?;
                    let inner_width = r.vbr(4)? as u32;
                    r.align();
                    let words = r.fixed(32)? as usize;
                    // The module's own tables; anything else is passed over
                    // (attributes, metadata, a function's inner blocks).
                    let read = id == MODULE_BLOCK
                        && matches!(
                            inner,
                            TYPE_BLOCK | CONSTANTS_BLOCK | VALUE_SYMTAB_BLOCK | FUNCTION_BLOCK
                        );
                    if read {
                        let base = self.values.len();
                        self.block(r, inner, inner_width)?;
                        if inner == FUNCTION_BLOCK {
                            self.values.truncate(base);
                        }
                    } else {
                        r.at += words * 32;
                    }
                }
                3 => {
                    let code = r.vbr(6)?;
                    let n = r.vbr(6)?;
                    ops.clear();
                    for _ in 0..n {
                        ops.push(r.vbr(6)?);
                    }
                    self.record(id, code, &ops)?;
                }
                _ => return Err("an abbreviated record"),
            }
        }
    }

    fn record(&mut self, block: u64, code: u64, ops: &[u64]) -> Result<(), Why> {
        let op = |i: usize| ops.get(i).copied().ok_or("a short record");
        match (block, code) {
            // NUMENTRY and a struct's name make no type.
            (TYPE_BLOCK, 1 | 19) => {}
            (TYPE_BLOCK, 2) => self.types.push(Type::Void),
            (TYPE_BLOCK, 3 | 4 | 10) => self.types.push(Type::Float),
            (TYPE_BLOCK, 11) => self.types.push(Type::Array(op(1)?)),
            (TYPE_BLOCK, 12) => self.types.push(Type::Vector(op(1)?)),
            (TYPE_BLOCK, 18 | 20) => self
                .types
                .push(Type::Struct(ops.get(1..).ok_or("a short record")?.to_vec())),
            (TYPE_BLOCK, 21) => self.types.push(Type::Function(op(1)?)),
            (TYPE_BLOCK, _) => self.types.push(Type::Other),
            (MODULE_BLOCK, 7 | 8) => self.values.push(ValueType::Other),
            (MODULE_BLOCK, _) => {}
            // SETTYPE makes no value.
            (CONSTANTS_BLOCK, 1) => self.constant_type = Some(op(0)?),
            (CONSTANTS_BLOCK, _) => self.values.push(ValueType::Id(
                self.constant_type.ok_or("a constant without its type")?,
            )),
            (VALUE_SYMTAB_BLOCK, 1) => {
                let name = ops[1.min(ops.len())..]
                    .iter()
                    .map(|&c| c as u8 as char)
                    .collect();
                self.names.insert(op(0)?, name);
            }
            (VALUE_SYMTAB_BLOCK, _) => {}
            (FUNCTION_BLOCK, 1) => self.counts.blocks += op(0)?,
            (FUNCTION_BLOCK, _) => self.instruction(code, ops)?,
            _ => {}
        }
        Ok(())
    }

    /// The type of a leading relative operand and the index after it.
    fn operand_type(&self, ops: &[u64]) -> Result<(ValueType, usize), Why> {
        let relative = *ops.first().ok_or("a short record")?;
        let next = self.values.len();
        if let Some(ty) = next
            .checked_sub(relative as usize)
            .and_then(|id| self.values.get(id))
        {
            Ok((*ty, 1))
        } else {
            // A forward reference is followed by an explicit type id.
            Ok((ValueType::Id(*ops.get(1).ok_or("a short record")?), 2))
        }
    }

    fn ty(&self, id: u64) -> Result<&Type, Why> {
        self.types.get(id as usize).ok_or("a type id")
    }

    fn is_float(&self, ty: ValueType) -> Result<bool, Why> {
        let ValueType::Id(id) = ty else {
            return Ok(false);
        };
        Ok(match self.ty(id)? {
            Type::Float => true,
            Type::Vector(element) => matches!(self.ty(*element)?, Type::Float),
            _ => false,
        })
    }

    fn extract_type(&self, mut ty: ValueType, indices: &[u64]) -> Result<ValueType, Why> {
        for &index in indices {
            let ValueType::Id(id) = ty else {
                // The writer also extracts the integer result of cmpxchg.
                return Ok(ValueType::Other);
            };
            ty = ValueType::Id(match self.ty(id)? {
                Type::Array(element) => *element,
                Type::Struct(fields) => *fields.get(index as usize).ok_or("a field index")?,
                _ => return Err("an extract without an aggregate type"),
            });
        }
        Ok(ty)
    }

    fn instruction(&mut self, code: u64, ops: &[u64]) -> Result<(), Why> {
        let op = |i: usize| ops.get(i).copied().ok_or("a short record");
        let (kind, ty) = match code {
            2 => {
                let (ty, rhs) = self.operand_type(ops)?;
                let opcode = op(rhs + 1)?;
                let kind = match (opcode, self.is_float(ty)?) {
                    (0, true) => "fadd",
                    (1, true) => "fsub",
                    (2, true) => "fmul",
                    (4, true) => "fdiv",
                    (6, true) => "frem",
                    _ => *[
                        "add", "sub", "mul", "udiv", "sdiv", "urem", "srem", "shl", "lshr", "ashr",
                        "and", "or", "xor",
                    ]
                    .get(opcode as usize)
                    .ok_or("a binary opcode")?,
                };
                (kind, ty)
            }
            3 => {
                let (_, dest) = self.operand_type(ops)?;
                let kind = *[
                    "trunc", "zext", "sext", "fptoui", "fptosi", "uitofp", "sitofp", "fptrunc",
                    "fpext", "ptrtoint", "inttoptr", "bitcast",
                ]
                .get(op(dest + 1)? as usize)
                .ok_or("a cast opcode")?;
                (kind, ValueType::Id(op(dest)?))
            }
            10 => ("ret", ValueType::Void),
            11 => ("br", ValueType::Void),
            16 => ("phi", ValueType::Id(op(0)?)),
            20 => ("load", ValueType::Id(op(self.operand_type(ops)?.1)?)),
            26 => {
                let (aggregate, first_index) = self.operand_type(ops)?;
                op(first_index)?;
                (
                    "extractvalue",
                    self.extract_type(aggregate, &ops[first_index..])?,
                )
            }
            28 => {
                let (_, rhs) = self.operand_type(ops)?;
                (
                    if op(rhs + 1)? < 32 { "fcmp" } else { "icmp" },
                    ValueType::Other,
                )
            }
            29 => ("select", self.operand_type(ops)?.0),
            38 => ("atomicrmw", ValueType::Other),
            43 => ("getelementptr", ValueType::Other),
            44 => ("store", ValueType::Void),
            46 => ("cmpxchg", ValueType::Other),
            34 => {
                // [attributes, calling convention, function type, callee
                // (relative), arguments...]
                let callee = (self.values.len() as u64)
                    .checked_sub(op(3)?)
                    .ok_or("a call of a later value")?;
                let Type::Function(ret) = self.ty(op(2)?)? else {
                    return Err("a call without its function type");
                };
                if !matches!(self.ty(*ret)?, Type::Void) {
                    self.values.push(ValueType::Id(*ret));
                }
                *self.calls.entry(callee).or_default() += 1;
                self.counts.instructions += 1;
                return Ok(());
            }
            _ => return Err("an instruction record of another kind"),
        };
        self.counts.instructions += 1;
        *self.counts.kinds.entry(kind.into()).or_default() += 1;
        if !matches!(ty, ValueType::Void) {
            self.values.push(ty);
        }
        Ok(())
    }
}

/// The bitstream: bytes read from bit 0 up.
struct Bits<'a> {
    data: &'a [u8],
    at: usize,
}

impl Bits<'_> {
    /// The next `width` bits (at most 32).
    fn fixed(&mut self, width: u32) -> Result<u64, Why> {
        let byte = self.at / 8;
        if self.at + width as usize > self.data.len() * 8 {
            return Err("truncated bitcode");
        }
        let mut buf = [0u8; 8];
        let end = (byte + 8).min(self.data.len());
        buf[..end - byte].copy_from_slice(&self.data[byte..end]);
        let v = u64::from_le_bytes(buf) >> (self.at % 8);
        self.at += width as usize;
        Ok(v & ((1u64 << width) - 1))
    }

    fn vbr(&mut self, width: u32) -> Result<u64, Why> {
        let hi = 1u64 << (width - 1);
        let (mut v, mut shift) = (0u64, 0u32);
        loop {
            let chunk = self.fixed(width)?;
            if shift >= 64 {
                return Err("a number past 64 bits");
            }
            v |= (chunk & (hi - 1)) << shift;
            if chunk & hi == 0 {
                return Ok(v);
            }
            shift += width - 1;
        }
    }

    fn align(&mut self) {
        self.at = self.at.next_multiple_of(32);
    }
}

#[cfg(test)]
mod tests {
    use super::super::bitcode::{Attr, Bin, Cast, Module, Rmw, Term, pred};
    use super::super::shader::Shader;
    use super::*;

    fn kinds(c: &Counts) -> Vec<(&str, u64)> {
        c.kinds.iter().map(|(k, &n)| (k.as_str(), n)).collect()
    }

    /// Every instruction the writer has, in a loop with a phi nothing reads
    /// (which the writer leaves out) and an empty latch (which it folds);
    /// what is computed is stored, as the writer keeps only what effects
    /// read.
    #[test]
    fn counts_what_the_writer_wrote() {
        let mut m = Module::default();
        let (void, i1, i32t, f32t) = (m.void(), m.int(1), m.int(32), m.float());
        let load_ty = m.func_ty(i32t, vec![i32t]);
        let load = m.function("dx.op.load.i32", load_ty, Some(Attr::ReadOnly), false);
        let store_ty = m.func_ty(void, vec![i32t, i32t]);
        let store = m.function("dx.op.store.i32", store_ty, Some(Attr::NoUnwind), false);
        let main_ty = m.func_ty(void, vec![]);
        m.function("main", main_ty, None, true);
        let arr = m.array(4, i32t);
        let lds = m.global("lds", arr, 3);
        let (entry, head, body, latch, exit) = (
            m.new_block(),
            m.new_block(),
            m.new_block(),
            m.new_block(),
            m.new_block(),
        );
        let (zero, one, four) = (m.i32(0), m.i32(1), m.i32(4));
        m.terminate(entry, Term::Br(head));
        let i = m.phi(head, i32t);
        let unread = m.phi(head, i32t);
        m.terminate(head, Term::Br(body));
        let x = m.call(body, load, vec![i]);
        let y = m.call(body, load, vec![x]);
        let sum = m.bin(body, Bin::Add, x, y);
        let shifted = m.bin(body, Bin::Shl, sum, one);
        let f = m.cast(body, Cast::Bitcast, shifted, f32t);
        let nan = m.cmp(body, pred::UNO, f, f);
        let big = m.cmp(body, pred::IUGT, shifted, four);
        let both = m.bin(body, Bin::And, nan, big);
        let picked = m.select(body, both, zero, shifted);
        let _ = m.call(body, store, vec![i, picked]);
        let at = m.element(body, lds, i);
        let old = m.load(body, at);
        m.store(body, at, picked);
        let _ = m.atomic(body, Rmw::Add, at, old);
        let swapped = m.cmpxchg(body, at, old, picked);
        m.store(body, at, swapped);
        let next = m.bin(body, Bin::Add, i, one);
        let more = m.cmp(body, pred::IULT, next, four);
        m.terminate(body, Term::CondBr(more, latch, exit));
        m.terminate(latch, Term::Br(head));
        m.set_phi(i, vec![(zero, entry), (next, latch)]);
        m.set_phi(unread, vec![(zero, entry), (next, latch)]);
        m.terminate(exit, Term::Ret);
        let _ = i1;
        let c = count(&m.write()).expect("counts");
        // Entry, header, body and exit: the latch is folded into the body.
        assert_eq!(c.blocks, 4);
        assert_eq!(
            kinds(&c),
            [
                ("add", 2),
                ("and", 1),
                ("atomicrmw", 1),
                ("bitcast", 1),
                ("br", 3),
                ("cmpxchg", 1),
                ("dx.op.load.i32", 2),
                ("dx.op.store.i32", 1),
                ("extractvalue", 1),
                ("fcmp", 1),
                ("getelementptr", 1),
                ("icmp", 2),
                ("load", 1),
                ("phi", 1),
                ("ret", 1),
                ("select", 1),
                ("shl", 1),
                ("store", 2),
            ]
        );
        assert_eq!(c.instructions, c.kinds.values().sum::<u64>());
        assert_eq!(c.instructions, 24);
    }

    /// Every shared opcode the writer has in both forms, the floats typed
    /// by each record that gives a value its type, and integers read after
    /// them, so a value id off by one names a binary op wrongly. The last
    /// of each are stored, so the writer keeps them and what they read.
    #[test]
    fn counts_integer_and_float_binary_ops_separately() {
        let mut m = Module::default();
        let (void, float, i32t) = (m.void(), m.float(), m.int(32));
        let pair = m.named_struct("pair", vec![float, float]);
        let load_ty = m.func_ty(pair, vec![]);
        let load = m.function("load_pair", load_ty, None, false);
        let main_ty = m.func_ty(void, vec![]);
        m.function("main", main_ty, None, true);
        let arr = m.array(4, float);
        let lds = m.global("lds", arr, 3);
        let ints = m.array(4, i32t);
        let lds_ints = m.global("lds_ints", ints, 3);
        let (entry, exit) = (m.new_block(), m.new_block());

        let i = m.i32(2);
        let mut x = i;
        for op in [
            Bin::Add,
            Bin::Sub,
            Bin::Mul,
            Bin::UDiv,
            Bin::SDiv,
            Bin::URem,
        ] {
            x = m.bin(entry, op, x, i);
        }

        // A constant, a call's result, a field, a cast, a load, a select
        // and a phi.
        let two = m.f32(2.0f32.to_bits());
        let pair = m.call(entry, load, vec![]);
        let f = m.extract(entry, pair, 0);
        let sum = m.bin(entry, Bin::Add, f, two);
        let cast = m.cast(entry, Cast::Bitcast, x, float);
        let difference = m.bin(entry, Bin::Sub, cast, sum);
        let zero = m.i32(0);
        let at = m.element(entry, lds, zero);
        let loaded = m.load(entry, at);
        let product = m.bin(entry, Bin::Mul, loaded, difference);
        let less = m.cmp(entry, pred::IULT, x, i);
        let picked = m.select(entry, less, product, two);
        let quotient = m.bin(entry, Bin::SDiv, picked, two);
        m.terminate(entry, Term::Br(exit));
        let merged = m.phi(exit, float);
        m.set_phi(merged, vec![(quotient, entry)]);
        let squared = m.bin(exit, Bin::Mul, merged, merged);
        m.store(exit, at, squared);
        let total = m.bin(exit, Bin::Add, x, i);
        let at_int = m.element(exit, lds_ints, zero);
        m.store(exit, at_int, total);
        m.terminate(exit, Term::Ret);

        let c = count(&m.write()).expect("counts");
        assert_eq!(
            kinds(&c),
            [
                ("add", 2),
                ("bitcast", 1),
                ("br", 1),
                ("extractvalue", 1),
                ("fadd", 1),
                ("fdiv", 1),
                ("fmul", 2),
                ("fsub", 1),
                ("getelementptr", 2),
                ("icmp", 1),
                ("load", 1),
                ("load_pair", 1),
                ("mul", 1),
                ("phi", 1),
                ("ret", 1),
                ("sdiv", 1),
                ("select", 1),
                ("store", 2),
                ("sub", 1),
                ("udiv", 1),
                ("urem", 1),
            ]
        );
        // Naming by type moves no instruction between kinds' totals.
        assert_eq!(c.instructions, c.kinds.values().sum::<u64>());
        assert_eq!((c.blocks, c.instructions), (2, 25));
    }

    #[test]
    fn counts_float_binary_ops_with_forward_operand_types() {
        // The writer emits neither vectors nor forward binary operands.
        let mut m = super::Module::default();
        m.record(TYPE_BLOCK, 10, &[]).unwrap(); // half
        m.record(TYPE_BLOCK, 4, &[]).unwrap(); // double
        m.record(TYPE_BLOCK, 12, &[2, 0]).unwrap(); // <2 x half>
        let forward = u64::from(u32::MAX);
        m.instruction(2, &[forward, 2, forward, 1]).unwrap(); // fsub of vectors
        m.instruction(2, &[1, 1, 6]).unwrap(); // frem of vectors
        m.instruction(2, &[forward, 1, forward, 0]).unwrap(); // fadd of doubles
        m.record(TYPE_BLOCK, 7, &[32]).unwrap(); // i32
        m.record(TYPE_BLOCK, 12, &[4, 3]).unwrap(); // <4 x i32>
        m.instruction(2, &[forward, 0, forward, 2]).unwrap(); // fmul of halves
        m.instruction(2, &[forward, 4, forward, 0]).unwrap(); // add of vectors
        m.instruction(2, &[1, 1, 4]).unwrap(); // sdiv of vectors
        m.instruction(2, &[forward, 3, forward, 6]).unwrap(); // srem of i32
        assert_eq!(
            kinds(&m.counts),
            [
                ("add", 1),
                ("fadd", 1),
                ("fmul", 1),
                ("frem", 1),
                ("fsub", 1),
                ("sdiv", 1),
                ("srem", 1),
            ]
        );
        assert_eq!(m.counts.instructions, 7);
    }

    #[test]
    fn counts_each_body_from_module_values() {
        let c = count(&super::super::bitcode::two_function_bitcode()).expect("counts");
        assert_eq!((c.blocks, c.instructions), (2, 4));
        assert_eq!(kinds(&c), [("add", 1), ("fadd", 1), ("ret", 2)]);
    }

    /// Through a whole container, as a shader is handed to the host.
    #[test]
    fn reads_a_container() {
        let mut s = Shader::new(true);
        let entry = s.entry;
        s.m.terminate(entry, Term::Ret);
        let c = of(&s.finish()).expect("counts");
        assert_eq!((c.blocks, c.instructions), (1, 1));
        assert_eq!(kinds(&c), [("ret", 1)]);
    }

    #[test]
    fn sums() {
        let one = Counts {
            blocks: 2,
            instructions: 3,
            kinds: [("add".to_owned(), 2), ("ret".to_owned(), 1)].into(),
        };
        let mut total = one.clone();
        total.add(&one);
        assert_eq!((total.blocks, total.instructions), (4, 6));
        assert_eq!(kinds(&total), [("add", 4), ("ret", 2)]);
    }

    /// What is no module of ours is refused, not miscounted.
    #[test]
    fn refuses_other_streams() {
        assert_eq!(count(b"BC"), Err("no bitcode"));
        assert_eq!(count(b"BC\xC0\xDE"), Err("truncated bitcode"));
        // A module block whose first entry defines an abbreviation (id 2).
        let mut bytes = b"BC\xC0\xDE".to_vec();
        // ENTER_SUBBLOCK (1, 2 bits), block 8 (vbr8), width 3 (vbr4).
        let head: u32 = 1 | 8 << 2 | 3 << 10;
        bytes.extend(head.to_le_bytes());
        bytes.extend(1u32.to_le_bytes());
        bytes.extend(2u32.to_le_bytes());
        assert_eq!(count(&bytes), Err("an abbreviated record"));
        let truncated = {
            let mut m = Module::default();
            let void = m.void();
            let main_ty = m.func_ty(void, vec![]);
            m.function("main", main_ty, None, true);
            let b = m.new_block();
            m.terminate(b, Term::Ret);
            let mut w = m.write();
            w.truncate(w.len() - 8);
            w
        };
        assert_eq!(count(&truncated), Err("truncated bitcode"));
    }
}
