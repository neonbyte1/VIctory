//! Survey shader containers: bitcode, console additions and native code;
//! and list shader packs, naming each stage's container where one of the
//! given files holds it.

use std::borrow::Cow;
use std::collections::{BTreeMap, HashMap, HashSet};
use std::path::{Path, PathBuf};

use anyhow::Result;
use victory_game::{sgd5, shaderpack};
use victory_shader::gfx10::{self, Instruction};
use victory_shader::{Container, DxilProgram, FourCC, HwStage, ShaderError, native_programs};

/// Files above this size are skipped (read whole into memory for now).
pub(crate) const MAX_FILE: u64 = 2 << 30;

/// Shader inputs under `root`, with the caller's metadata policy for regular files.
pub(crate) fn shader_files(
    root: &Path,
    allow_missing_metadata: bool,
) -> impl Iterator<Item = walkdir::DirEntry> {
    scan_shader_files(root, allow_missing_metadata, |_| {})
}

fn scan_shader_files(
    root: &Path,
    allow_missing_metadata: bool,
    on_large_file: impl Fn(&Path),
) -> impl Iterator<Item = walkdir::DirEntry> {
    walkdir::WalkDir::new(root)
        .into_iter()
        .filter_map(Result::ok)
        .filter(move |entry| {
            let kind = entry.file_type();
            if kind.is_symlink() {
                // Inspect the target without changing which directories WalkDir traverses.
                return std::fs::metadata(entry.path())
                    .is_ok_and(|target| target.is_file() && target.len() <= MAX_FILE);
            }
            if !kind.is_file() {
                return false;
            }
            match entry.metadata() {
                Ok(metadata) if metadata.len() > MAX_FILE => {
                    on_large_file(entry.path());
                    false
                }
                Ok(_) => true,
                Err(_) => allow_missing_metadata,
            }
        })
}

/// A shader container in a file: as is, or packed in an SGD5 record.
pub(crate) enum Source<'a> {
    Raw { offset: usize, data: &'a [u8] },
    Packed(sgd5::Record<'a>),
}

impl Source<'_> {
    /// The container bytes; `None` if a packed record is corrupt.
    pub(crate) fn bytes(&self) -> Option<Cow<'_, [u8]>> {
        match self {
            Source::Raw { data, .. } => Some(Cow::Borrowed(data)),
            Source::Packed(r) => r.unpack().map(Cow::Owned),
        }
    }

    pub(crate) fn label(&self) -> String {
        match self {
            Source::Raw { offset, .. } => format!("@{offset:#x}"),
            Source::Packed(r) => format!("@{:#x} {}", r.offset, r.name.unwrap_or("?")),
        }
    }
}

/// The packed records of SGD5 data, and every container stored as is (the
/// executable keeps some records uncompressed), found by its magic.
pub(crate) fn sources(data: &[u8]) -> Vec<Source<'_>> {
    let mut out: Vec<Source<'_>> = Vec::new();
    if sgd5::is_sgd5(data) {
        out.extend(sgd5::records(data).into_iter().map(Source::Packed));
    }
    let raw = memchr::memmem::find_iter(data, b"DXBC")
        .filter(|&offset| Container::parse(&data[offset..]).is_ok())
        .map(|offset| Source::Raw {
            offset,
            data: &data[offset..],
        });
    out.extend(raw);
    out
}

#[derive(Default)]
struct Totals {
    containers: usize,
    with_dxil_part: usize,
    with_bitcode: usize,
    standard_layout: usize,
    xdat: usize,
    xbop: usize,
    kinds: BTreeMap<String, usize>,
    with_native: usize,
    native_stages: BTreeMap<HwStage, usize>,
    native_errors: BTreeMap<String, usize>,
    decoded: usize,
    instructions: usize,
    unnamed: BTreeMap<String, usize>,
    decode_errors: BTreeMap<&'static str, usize>,
    files_with_shaders: usize,
    /// A native program's hash instructions -> its container's BLAKE3
    /// (first 8 bytes, hex, as the GPU side logs it) and name.
    programs: HashMap<[u8; 24], String>,
    /// Shader hashes stored in parsed databases.
    engine: HashSet<u64>,
}

impl Totals {
    fn record_database_hashes(&mut self, data: &[u8]) {
        self.engine.extend(
            sgd5::databases(data)
                .into_iter()
                .flat_map(|db| db.shaders.into_iter().flatten())
                .map(|s| s.hash),
        );
    }

    fn describe_dxil(&mut self, part: Option<&[u8]>) -> String {
        let Some(part) = part else {
            return "no parsable DXIL program".to_owned();
        };
        self.with_dxil_part += 1;

        // Titles can strip the bitcode while keeping the Xbox additions.
        let xdat = part.get(0x18..0x1c) == Some(b"XDAT");
        let xbop =
            dxil_trailer(part).is_some_and(|trailer| trailer.windows(4).any(|w| w == b"xbop"));
        self.xdat += usize::from(xdat);
        self.xbop += usize::from(xbop);

        let description = match DxilProgram::parse(part) {
            Ok(p) => {
                self.with_bitcode += 1;
                self.standard_layout += usize::from(p.is_standard_layout());
                *self.kinds.entry(format!("{:?}", p.kind)).or_default() += 1;
                format!(
                    "{:?} sm{}.{} bitcode {} B",
                    p.kind,
                    p.shader_model.0,
                    p.shader_model.1,
                    p.bitcode.len()
                )
            }
            Err(_) => "no parsable DXIL program".to_owned(),
        };
        format!("{description}, xdat {xdat}, xbop {xbop}")
    }
}

/// Locate the trailer from the DXIL header without reading the bitcode.
fn dxil_trailer(part: &[u8]) -> Option<&[u8]> {
    if part.get(8..12) != Some(b"DXIL") {
        return None;
    }
    let word = |at: usize| {
        let bytes = part.get(at..at + 4)?.try_into().ok()?;
        Some(u32::from_le_bytes(bytes) as usize)
    };
    let size = word(4)?.checked_mul(4)?;
    let bitcode = 8usize.checked_add(word(0x10)?)?;
    if bitcode < 0x18 {
        return None;
    }
    let end = bitcode.checked_add(word(0x14)?)?;
    part.get(end..size)
}

pub fn run(paths: Vec<PathBuf>, verbose: bool, disasm: bool, effects: bool) -> Result<()> {
    let paths = crate::default_paths(paths)?;
    let mut t = Totals::default();
    let mut packs = Vec::new();
    let mut effect_databases = 0;
    for root in &paths {
        for entry in scan_shader_files(root, true, |path| {
            eprintln!("skipping large file {}", path.display());
        }) {
            let path = entry.path();
            let Ok(data) = std::fs::read(path) else {
                continue;
            };
            if effects {
                effect_databases += list_effects(path, &data);
                continue;
            }
            if data.starts_with(shaderpack::MAGIC) {
                packs.push((path.to_owned(), data));
                continue;
            }
            let mut found_here = 0;
            t.record_database_hashes(&data);
            for source in sources(&data) {
                let Some(bytes) = source.bytes() else {
                    eprintln!("corrupt record {}{}", path.display(), source.label());
                    continue;
                };
                let Ok(container) = Container::parse(&bytes) else {
                    continue;
                };
                found_here += 1;
                t.containers += 1;
                let dxil = container.part(FourCC::DXIL);
                let mut listing = Vec::new();
                // Independent of the bitcode: titles may strip it.
                let code = dxil.or(container.part(FourCC::ACCC));
                let native = match code.map(|p| native_programs(p.data)) {
                    Some(Ok(programs)) if !programs.is_empty() => {
                        t.with_native += 1;
                        let mut list = Vec::new();
                        // `parse` checked the size field against the data.
                        let size = u32::from_le_bytes(bytes[0x18..0x1c].try_into()?) as usize;
                        let name = match &source {
                            Source::Raw { offset, .. } => {
                                let file = path.file_name().unwrap_or_default();
                                format!("{}@{offset:#x}", file.to_string_lossy())
                            }
                            Source::Packed(r) => r.name.unwrap_or("?").to_owned(),
                        };
                        let id = format!("{}:{name}", &blake3::hash(&bytes[..size]).to_hex()[..16]);
                        for n in &programs {
                            if let Some(key) = shaderpack::program_id(&n.code) {
                                t.programs.entry(key).or_insert_with(|| id.clone());
                            }
                            *t.native_stages.entry(n.stage).or_default() += 1;
                            list.push(format!("{:?} {} B", n.stage, n.code.len()));
                            if disasm {
                                listing.push(format!("    {:?} stage I/O: {:?}", n.stage, n.io));
                                listing.extend(n.layout.iter().map(|p| format!("    {p:?}")));
                                listing.extend(n.used.iter().map(|u| format!("    {u:?}")));
                            }
                            match gfx10::decode(&n.code) {
                                Ok(insts) => {
                                    t.decoded += 1;
                                    t.instructions += insts.len();
                                    for i in insts.iter().filter(|i| i.name().is_none()) {
                                        let key = format!("{:?} {:#x}", i.encoding, i.opcode);
                                        *t.unnamed.entry(key).or_default() += 1;
                                    }
                                    if disasm {
                                        listing.extend(
                                            insts.iter().map(|i| format_instruction(i, n.wave64)),
                                        );
                                    }
                                }
                                Err(e) => {
                                    let reason = match e {
                                        ShaderError::Isa { reason, .. } => reason,
                                        _ => "other",
                                    };
                                    *t.decode_errors.entry(reason).or_default() += 1;
                                    list.push(format!("({e})"));
                                }
                            }
                        }
                        format!(", native {}", list.join(" "))
                    }
                    Some(Err(e)) => {
                        *t.native_errors.entry(e.to_string()).or_default() += 1;
                        format!(", native: {e}")
                    }
                    _ => String::new(),
                };
                let line = t.describe_dxil(dxil.map(|p| p.data)) + &native;
                if verbose {
                    println!("{}{}: {line}", path.display(), source.label());
                    for l in &listing {
                        println!("{l}");
                    }
                }
            }
            t.files_with_shaders += usize::from(found_here > 0);
        }
    }
    if effects {
        println!("shader effect databases found: {effect_databases}");
        return Ok(());
    }
    println!(
        "containers found       {} (in {} files)",
        t.containers, t.files_with_shaders
    );
    println!("  with DXIL part       {}", t.with_dxil_part);
    println!("  with LLVM bitcode    {}", t.with_bitcode);
    println!("  PC-standard layout   {}", t.standard_layout);
    println!("  XDAT prefix          {}", t.xdat);
    println!("  xbop trailer         {}", t.xbop);
    for (k, n) in &t.kinds {
        println!("  {k:<20} {n}");
    }
    println!("  with native code     {}", t.with_native);
    for (stage, n) in &t.native_stages {
        println!("    {:<18} {n}", format!("{stage:?}"));
    }
    for (e, n) in &t.native_errors {
        println!("  native unreadable    {n}: {e}");
    }
    if t.with_native > 0 {
        println!(
            "  decoded programs     {} ({} instructions)",
            t.decoded, t.instructions
        );
        for (e, n) in &t.decode_errors {
            println!("  decode failed        {n}: {e}");
        }
        let mut unnamed: Vec<_> = t.unnamed.iter().collect();
        unnamed.sort_by_key(|&(_, n)| std::cmp::Reverse(*n));
        for (op, n) in unnamed.iter().take(10) {
            println!("  opcode not in table  {op} x{n}");
        }
    }
    if t.containers == 0 && packs.is_empty() {
        println!("(none found: shaders may sit in compressed or encrypted archives)");
    }
    for (path, data) in &packs {
        list_pack(path, data, &t, verbose);
    }
    Ok(())
}

/// A shader pack's pipelines; with `verbose` one line each, naming every
/// stage's container (BLAKE3 prefix and name) found in the other files.
fn list_pack(path: &Path, data: &[u8], t: &Totals, verbose: bool) {
    let pack = match shaderpack::Pack::parse(data) {
        Ok(pack) => pack,
        Err(e) => return println!("{}: {e}", path.display()),
    };
    let name = |code: &[u8]| {
        let id = shaderpack::program_id(code).and_then(|id| t.programs.get(&id));
        id.map_or("?", String::as_str)
    };
    let hex = |h: u64| format!("{h:016x}");
    let (mut graphics, mut compute, mut stages, mut named, mut keyed) = (0, 0, 0, 0, 0);
    for g in pack.graphics() {
        graphics += 1;
        keyed += usize::from(t.engine.contains(&g.shader_hash));
        let mut line = format!(
            "graphics key={} shader={} pipeline={} depth-stencil={} rasterizer={} blend={}",
            hex(g.key),
            hex(g.shader_hash),
            hex(g.pipeline_hash),
            hex(g.depth_stencil_hash),
            hex(g.rasterizer_hash),
            hex(g.blend_hash)
        );
        if let (Some(f), Some((depth, count, topology)), Some((mask, samples, quality))) =
            (g.target_formats(), g.depth_targets_topology(), g.samples())
        {
            let f: Vec<_> = f.iter().map(u8::to_string).collect();
            line += &format!(
                " targets={count}:{} depth={depth} topology={topology} samples={samples}.{quality} mask={mask:#x}",
                f.join(",")
            );
        }
        if let Some(inputs) = g.input_layout() {
            let e: Vec<_> = (inputs.iter())
                .map(|e| {
                    let (s, i, f, slot, o) = (e.semantic, e.index, e.format, e.slot, e.offset);
                    format!("{s}{i}:{f}:{slot}:{o}:{}:{}", e.class, e.step_rate)
                })
                .collect();
            line += &format!(" inputs={}", e.join(","));
        }
        for (stage, s) in shaderpack::STAGES.iter().zip(&g.stages) {
            let Some(s) = s else { continue };
            stages += 1;
            let n = name(s.program);
            named += usize::from(n != "?");
            line += &format!(" {stage}={n}");
        }
        if verbose {
            println!("{line}");
        }
    }
    let mut compute_keyed = 0;
    for c in pack.compute() {
        compute += 1;
        compute_keyed += usize::from(t.engine.contains(&c.shader_hash));
        let n = name(c.stage.program);
        named += usize::from(n != "?");
        stages += 1;
        if verbose {
            println!("compute shader={} CS={n}", hex(c.shader_hash));
        }
    }
    println!(
        "{}: shader pack ({}, compiler {}, database {:016x}): {graphics} graphics and {compute} compute pipelines",
        path.display(),
        pack.driver(),
        pack.compiler(),
        pack.fingerprint()
    );
    println!("  stages with their container found   {named} of {stages}");
    println!("  graphics shader hash in a database  {keyed} of {graphics}");
    println!("  compute shader hash in a database   {compute_keyed} of {compute}");
}

/// Effect -> technique -> pass -> shaders (name, container hash, the
/// engine's hash) and the state each pass sets, for every database in `data`.
fn list_effects(path: &std::path::Path, data: &[u8]) -> usize {
    let databases = sgd5::databases(data);
    let count = databases.len();
    for db in databases {
        let passes: usize = (db.effects.iter())
            .flat_map(|e| &e.techniques)
            .map(|t| t.passes.len())
            .sum();
        println!(
            "{}@{:#x}: {} effects, {passes} passes",
            path.display(),
            db.start,
            db.effects.len()
        );
        let mut containers = HashMap::new();
        for e in &db.effects {
            println!("effect {}", e.name);
            for t in &e.techniques {
                println!("  technique {}", t.name);
                for (i, p) in t.passes.iter().enumerate() {
                    let mut parts = Vec::new();
                    for (slot, index) in p.shaders.iter().enumerate() {
                        let Some(index) = *index else { continue };
                        let s = &db.shaders[slot][index];
                        let container = containers.entry((slot, index)).or_insert_with(|| {
                            s.record.unpack().map_or("corrupt".to_owned(), |c| {
                                blake3::hash(&c).to_hex()[..16].to_owned()
                            })
                        });
                        let name = s.record.name.unwrap_or("?");
                        let stage = sgd5::STAGES[slot];
                        parts.push(format!("{stage} {name} {container} {:016x}", s.hash));
                    }
                    let states = [
                        ("rasterizer", p.rasterizer),
                        ("depth-stencil", p.depth_stencil),
                        ("blend", p.blend),
                    ];
                    for (what, bytes) in states {
                        if let Some(b) = bytes {
                            let hex: String = b.iter().map(|x| format!("{x:02x}")).collect();
                            parts.push(format!("{what} {hex}"));
                        }
                    }
                    if let Some(r) = p.stencil_ref {
                        parts.push(format!("stencil ref {r:#x}"));
                    }
                    println!("    pass {i}: {}", parts.join(", "));
                }
            }
        }
    }
    count
}

fn format_instruction(i: &Instruction, wave64: bool) -> String {
    let words: Vec<_> = i.words().iter().map(|w| format!("{w:08X}")).collect();
    let text = match i.name() {
        Some(_) => i.text(wave64),
        None => format!("<{:?} {:#x}>", i.encoding, i.opcode),
    };
    format!("    {:06x}: {:<44} {text}", i.offset, words.join(" "))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn words(out: &mut Vec<u8>, values: &[u32]) {
        for value in values {
            out.extend_from_slice(&value.to_le_bytes());
        }
    }

    fn dxbc(prefix: &[u8], bitcode: &[u8], trailer: &[u8]) -> Vec<u8> {
        let size = (24 + prefix.len() + bitcode.len() + trailer.len()) as u32;
        let mut out = b"DXBC".to_vec();
        out.extend_from_slice(&[0; 16]);
        words(&mut out, &[1, 44 + size, 1, 36]);
        out.extend_from_slice(b"DXIL");
        words(&mut out, &[size, 0x60, size / 4]);
        out.extend_from_slice(b"DXIL");
        words(
            &mut out,
            &[0x100, 16 + prefix.len() as u32, bitcode.len() as u32],
        );
        out.extend_from_slice(prefix);
        out.extend_from_slice(bitcode);
        out.extend_from_slice(trailer);
        out
    }

    #[test]
    fn counts_standard_dxil_database_hash() {
        let payload = dxbc(&[], b"BC\xc0\xde", &[]);
        let shader_hash = 0x0123_4567_89ab_cdef_u64;
        let mut db = sgd5::MAGIC.to_vec();
        db.extend_from_slice(&[0; 8]);
        // One pixel shader with zero shared blocks, strings and vertex shaders.
        words(&mut db, &[0, 0, 0, 1]);
        db.extend_from_slice(b"\x03PS\0\x32");

        // A single final, uncompressed DEFLATE block (LEN, then its complement).
        let len = payload.len() as u16;
        let size = 12 + 5 + u32::from(len);
        words(&mut db, &[size, 0x000D_DCE7, size, u32::from(len)]);
        db.push(1);
        db.extend_from_slice(&len.to_le_bytes());
        db.extend_from_slice(&(!len).to_le_bytes());
        db.extend_from_slice(&payload);
        db.extend_from_slice(&shader_hash.to_le_bytes());
        db.extend_from_slice(&[0; 0x90]);
        words(&mut db, &[0; 6]); // No shader metadata, other stages or effects.

        let bytes = sgd5::records(&db)[0].unpack().unwrap();
        let container = Container::parse(&bytes).unwrap();
        let part = container.part(FourCC::DXIL).unwrap().data;
        assert!(DxilProgram::parse(part).unwrap().is_standard_layout());
        assert!(native_programs(part).unwrap().is_empty());

        let mut totals = Totals::default();
        totals.record_database_hashes(&db);
        assert_eq!(totals.engine, HashSet::from([shader_hash]));
    }

    #[test]
    fn counts_xbox_extras_without_llvm_bitcode() {
        let prefix = b"XDAT\0\0\0\0\x24\0\0\0";
        for (bitcode, trailer, xbop) in [
            (&b""[..], &b"xbop"[..], true),
            (&b"bad!"[..], &b"xbop"[..], true),
            (&b"xbop"[..], &b""[..], false),
        ] {
            let bytes = dxbc(prefix, bitcode, trailer);
            let container = Container::parse(&bytes).unwrap();
            let mut totals = Totals::default();
            let description = totals.describe_dxil(container.part(FourCC::DXIL).map(|p| p.data));

            assert_eq!(totals.with_dxil_part, 1);
            assert_eq!(totals.with_bitcode, 0);
            assert_eq!((totals.xdat, totals.xbop), (1, usize::from(xbop)));
            assert_eq!(
                description,
                format!("no parsable DXIL program, xdat true, xbop {xbop}")
            );
        }
    }
}
