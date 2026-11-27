//! Which imports do titles need, and which do we cover?

use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet};
use std::fmt::Write as _;
use std::io::{self, Read};
use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use victory_hle::{HleResolver, HostPolicy};
use victory_loader::Provenance;
use victory_pe::{ExportInfo, ImageInfo, ImportSymbol};

#[derive(Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
enum FunctionKey {
    Name(String),
    Ordinal(u16),
}

impl From<&ImportSymbol> for FunctionKey {
    fn from(symbol: &ImportSymbol) -> Self {
        match symbol {
            ImportSymbol::Name { name, .. } => Self::Name(name.clone()),
            ImportSymbol::Ordinal(ordinal) => Self::Ordinal(*ordinal),
        }
    }
}

#[derive(Default)]
struct DllStats {
    functions: BTreeMap<FunctionKey, Option<Provenance>>,
    hle: usize,
    host: usize,
    title: usize,
    missing: usize,
    images: BTreeSet<PathBuf>,
}

#[derive(Default)]
struct TitleExports {
    names: HashSet<String>,
    ordinals: HashSet<u16>,
}

impl TitleExports {
    fn insert_all(&mut self, exports: &[ExportInfo]) {
        for e in exports {
            if e.rva.is_none() {
                continue;
            }
            self.ordinals.insert(e.ordinal);
            if let Some(name) = &e.name {
                self.names.insert(name.clone());
            }
        }
    }

    fn contains(&self, symbol: &ImportSymbol) -> bool {
        match symbol {
            ImportSymbol::Name { name, .. } => self.names.contains(name),
            ImportSymbol::Ordinal(ordinal) => self.ordinals.contains(ordinal),
        }
    }
}

fn resolves_title_export(
    dir: &Option<PathBuf>,
    dll_key: &str,
    symbol: &ImportSymbol,
    resolver: &HleResolver,
    title_exports: &HashMap<(Option<PathBuf>, String), TitleExports>,
) -> bool {
    title_dll(&resolver.host, dll_key)
        && title_exports
            .get(&(dir.clone(), dll_key.to_owned()))
            .is_some_and(|exp| exp.contains(symbol))
}

struct Census {
    images: usize,
    failed: Vec<String>,
    dlls: BTreeMap<String, DllStats>,
    missing: BTreeMap<(String, String), BTreeSet<PathBuf>>,
}

pub fn run(paths: Vec<PathBuf>, strict: bool, list_missing: bool) -> Result<()> {
    let paths = crate::default_paths(paths)?;
    let report = collect(&paths, strict)?;
    print!("{}", render(&report, list_missing));
    Ok(())
}

fn collect(paths: &[PathBuf], strict: bool) -> Result<Census> {
    let resolver = HleResolver::new(if strict {
        HostPolicy::none()
    } else {
        HostPolicy::default()
    });

    let mut seen = BTreeSet::new();
    let mut parsed = Vec::new();
    let mut failed = Vec::new();

    for root in paths {
        for entry in walkdir::WalkDir::new(root) {
            let entry = match entry {
                Ok(entry) => entry,
                Err(error) if error.depth() == 0 => {
                    return Err(error).context("cannot scan census input");
                }
                Err(error) => {
                    failed.push(error.to_string());
                    continue;
                }
            };
            let path = entry.path();
            if !entry.file_type().is_file() {
                continue;
            }
            match looks_like_pe(path) {
                Ok(true) => {}
                Ok(false) => continue,
                Err(e) => {
                    failed.push(format!("{}: {e}", path.display()));
                    continue;
                }
            }
            // Use the resolved path for deduplication and keep the original for output.
            let identity = path.canonicalize().unwrap_or_else(|_| path.to_path_buf());
            if !seen.insert(identity.clone()) {
                continue;
            }
            match std::fs::read(path)
                .map_err(anyhow::Error::from)
                .and_then(|d| ImageInfo::parse(&d).map_err(anyhow::Error::from))
            {
                Ok(info) => {
                    let dir = identity.parent().map(Path::to_path_buf);
                    parsed.push((path.to_path_buf(), dir, info));
                }
                Err(e) => failed.push(format!("{}: {e}", path.display())),
            }
        }
    }

    let mut title_exports: HashMap<(Option<PathBuf>, String), TitleExports> = HashMap::new();
    for (path, dir, info) in &parsed {
        if !info.is_dll() {
            continue;
        }
        let Some(name) = path
            .file_name()
            .map(|n| n.to_string_lossy().to_ascii_lowercase())
        else {
            continue;
        };
        title_exports
            .entry((dir.clone(), name))
            .or_default()
            .insert_all(&info.exports);
    }

    let mut dlls: BTreeMap<String, DllStats> = BTreeMap::new();
    let mut missing: BTreeMap<(String, String), BTreeSet<PathBuf>> = BTreeMap::new();
    for (path, dir, info) in &parsed {
        tally_image(
            path,
            dir.as_deref(),
            info,
            &resolver,
            &title_exports,
            &mut dlls,
            &mut missing,
        );
    }

    Ok(Census {
        images: parsed.len(),
        failed,
        dlls,
        missing,
    })
}

fn render(report: &Census, list_missing: bool) -> String {
    let mut out = String::new();
    let _ = writeln!(
        out,
        "{} images scanned, {} unreadable",
        report.images,
        report.failed.len()
    );
    for error in &report.failed {
        let _ = writeln!(out, "  skipped {error}");
    }
    let _ = writeln!(
        out,
        "\n{:<56} {:>6} {:>6} {:>6} {:>6} {:>8} {:>7}",
        "dll", "funcs", "hle", "host", "title", "missing", "images"
    );
    let (mut f, mut h, mut o, mut t, mut m) = (0, 0, 0, 0, 0);
    for (dll, s) in &report.dlls {
        let _ = writeln!(
            out,
            "{dll:<56} {:>6} {:>6} {:>6} {:>6} {:>8} {:>7}",
            s.functions.len(),
            s.hle,
            s.host,
            s.title,
            s.missing,
            s.images.len()
        );
        f += s.functions.len();
        h += s.hle;
        o += s.host;
        t += s.title;
        m += s.missing;
    }
    let _ = writeln!(out, "{:<56} {f:>6} {h:>6} {o:>6} {t:>6} {m:>8}", "TOTAL");

    if list_missing {
        let mut rows: Vec<_> = report.missing.iter().collect();
        rows.sort_by(|a, b| b.1.len().cmp(&a.1.len()).then_with(|| a.0.cmp(b.0)));
        let _ = writeln!(out, "\nmissing (by number of images using it)");
        for ((dll, sym), users) in rows {
            let _ = writeln!(out, "  {:>4}  {dll}!{sym}", users.len());
        }
    }
    out
}

/// DLLs the host or our runtime provide are never taken from the title.
fn title_dll(host: &HostPolicy, dll: &str) -> bool {
    !host.allows(dll) && !victory_hle::registry().iter().any(|m| m.serves(dll))
}

/// Cheap pre-filter: an `MZ` header, regardless of extension (console
/// packages do not always use .exe/.dll).
fn looks_like_pe(path: &Path) -> io::Result<bool> {
    let mut file = std::fs::File::open(path)?;
    let mut magic = [0u8; 2];
    match file.read_exact(&mut magic) {
        Ok(()) => Ok(&magic == b"MZ"),
        Err(e) if e.kind() == io::ErrorKind::UnexpectedEof => Ok(false),
        Err(e) => Err(e),
    }
}

fn tally_image(
    path: &Path,
    dir: Option<&Path>,
    info: &ImageInfo,
    resolver: &HleResolver,
    title_exports: &HashMap<(Option<PathBuf>, String), TitleExports>,
    dlls: &mut BTreeMap<String, DllStats>,
    missing: &mut BTreeMap<(String, String), BTreeSet<PathBuf>>,
) {
    let dir_key = dir.map(Path::to_path_buf);
    for module in info.imports.iter().chain(&info.delay_imports) {
        let key = module.dll.to_ascii_lowercase();
        let stats = dlls.entry(key.clone()).or_default();
        stats.images.insert(path.to_path_buf());
        for e in &module.entries {
            let class = resolver.classify(&module.dll, &e.symbol).or_else(|| {
                resolves_title_export(&dir_key, &key, &e.symbol, resolver, title_exports)
                    .then_some(Provenance::Title)
            });
            if class.is_none() {
                missing
                    .entry((key.clone(), e.symbol.to_string()))
                    .or_default()
                    .insert(path.to_path_buf());
            }
            // Count each dll!function once across all images; missing in any image wins over title.
            let func_key = FunctionKey::from(&e.symbol);
            match stats.functions.get_mut(&func_key) {
                None => {
                    match class {
                        Some(Provenance::Hle) => stats.hle += 1,
                        Some(Provenance::Host) => stats.host += 1,
                        Some(Provenance::Title) => stats.title += 1,
                        None => stats.missing += 1,
                    }
                    stats.functions.insert(func_key, class);
                }
                Some(prev) => {
                    if prev.is_some() && class.is_none() {
                        if *prev == Some(Provenance::Title) {
                            stats.title -= 1;
                            stats.missing += 1;
                        }
                        *prev = None;
                    }
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use std::sync::atomic::{AtomicUsize, Ordering};

    use super::*;
    use victory_pe::{ImportEntry, ImportModule};

    fn entry(symbol: ImportSymbol) -> ImportEntry {
        ImportEntry {
            symbol,
            iat_slot_rva: 0,
        }
    }

    fn named(hint: u16, name: &str) -> ImportEntry {
        entry(ImportSymbol::Name {
            hint,
            name: name.into(),
        })
    }

    fn ordinal(ord: u16) -> ImportEntry {
        entry(ImportSymbol::Ordinal(ord))
    }

    fn synthetic_image(imports: Vec<ImportModule>, delay_imports: Vec<ImportModule>) -> ImageInfo {
        ImageInfo {
            machine: 0,
            characteristics: 0,
            timestamp: 0,
            subsystem: 0,
            dll_characteristics: 0,
            image_base: 0,
            size_of_image: 0,
            entry_rva: 0,
            stack_reserve: 0,
            stack_commit: 0,
            sections: Vec::new(),
            imports,
            delay_imports,
            exports: Vec::new(),
            tls: None,
            has_relocations: false,
            has_exception_table: false,
            has_load_config: false,
            codeview: None,
        }
    }

    #[test]
    fn deduplicates_imports_ignoring_hint_across_images_and_delay_imports() {
        let resolver = HleResolver::new(HostPolicy::none());
        let exports = HashMap::new();
        let mut dlls: BTreeMap<String, DllStats> = BTreeMap::new();
        let mut missing: BTreeMap<(String, String), BTreeSet<PathBuf>> = BTreeMap::new();

        let image_a = synthetic_image(
            vec![ImportModule {
                dll: "XGameRuntime.dll".into(),
                entries: vec![
                    named(1, "XGameRuntimeInitialize"),
                    named(10, "SyntheticMissingApi"),
                    ordinal(42),
                ],
            }],
            vec![ImportModule {
                dll: "XGameRuntime.dll".into(),
                entries: vec![
                    named(2, "XGameRuntimeInitialize"),
                    named(20, "SyntheticMissingApi"),
                ],
            }],
        );
        let image_b = synthetic_image(
            vec![ImportModule {
                dll: "xgameruntime.dll".into(),
                entries: vec![
                    named(99, "XGameRuntimeInitialize"),
                    named(200, "SyntheticMissingApi"),
                    ordinal(42),
                ],
            }],
            vec![ImportModule {
                dll: "XGameRuntime.dll".into(),
                entries: vec![named(300, "SyntheticMissingApi"), ordinal(42)],
            }],
        );

        tally_image(
            Path::new("image_a.exe"),
            None,
            &image_a,
            &resolver,
            &exports,
            &mut dlls,
            &mut missing,
        );
        tally_image(
            Path::new("image_b.dll"),
            None,
            &image_b,
            &resolver,
            &exports,
            &mut dlls,
            &mut missing,
        );

        assert_eq!(dlls.len(), 1);
        let stats = &dlls["xgameruntime.dll"];
        assert_eq!(stats.functions.len(), 3);
        assert_eq!(stats.hle, 1);
        assert_eq!(stats.host, 0);
        assert_eq!(stats.title, 0);
        assert_eq!(stats.missing, 2);
        assert_eq!(stats.images.len(), 2);
        assert_eq!(
            dlls.values().map(|s| s.missing).sum::<usize>(),
            missing.len()
        );
        assert_eq!(
            missing[&("xgameruntime.dll".into(), "SyntheticMissingApi".into())].len(),
            2
        );
        assert_eq!(missing[&("xgameruntime.dll".into(), "#42".into())].len(), 2);
    }

    #[test]
    fn counts_distinct_symbols_and_dlls_separately() {
        let resolver = HleResolver::new(HostPolicy::none());
        let exports = HashMap::new();
        let mut dlls: BTreeMap<String, DllStats> = BTreeMap::new();
        let mut missing: BTreeMap<(String, String), BTreeSet<PathBuf>> = BTreeMap::new();

        let image = synthetic_image(
            vec![
                ImportModule {
                    dll: "XGameRuntime.dll".into(),
                    entries: vec![
                        named(0, "XGameRuntimeInitialize"),
                        named(0, "SyntheticMissingA"),
                        named(0, "SyntheticMissingB"),
                        ordinal(7),
                        ordinal(8),
                    ],
                },
                ImportModule {
                    dll: "SyntheticExtra.dll".into(),
                    entries: vec![named(0, "SyntheticMissingA"), ordinal(7)],
                },
            ],
            Vec::new(),
        );

        tally_image(
            Path::new("single.exe"),
            None,
            &image,
            &resolver,
            &exports,
            &mut dlls,
            &mut missing,
        );

        assert_eq!(dlls.len(), 2);
        let xgr = &dlls["xgameruntime.dll"];
        assert_eq!(xgr.functions.len(), 5);
        assert_eq!(xgr.hle, 1);
        assert_eq!(xgr.title, 0);
        assert_eq!(xgr.missing, 4);
        assert_eq!(xgr.images.len(), 1);

        let extra = &dlls["syntheticextra.dll"];
        assert_eq!(extra.functions.len(), 2);
        assert_eq!(extra.hle, 0);
        assert_eq!(extra.title, 0);
        assert_eq!(extra.missing, 2);
        assert_eq!(extra.images.len(), 1);

        assert_eq!(
            dlls.values().map(|s| s.missing).sum::<usize>(),
            missing.len()
        );
    }

    struct TempDir(PathBuf);

    impl TempDir {
        fn new() -> Self {
            static NEXT: AtomicUsize = AtomicUsize::new(0);
            let dir = std::env::temp_dir().join(format!(
                "victory-census-{}-{}",
                std::process::id(),
                NEXT.fetch_add(1, Ordering::Relaxed)
            ));
            let _ = std::fs::remove_dir_all(&dir);
            std::fs::create_dir_all(&dir).unwrap();
            Self(dir)
        }

        fn path(&self) -> &Path {
            &self.0
        }
    }

    impl Drop for TempDir {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }

    enum Sym<'a> {
        Name(&'a str),
        Ordinal(u16),
    }

    struct ExportSpec<'a> {
        name: Option<&'a str>,
        ordinal: u16,
    }

    fn build_pe(dll: bool, exports: &[ExportSpec<'_>], imports: &[(&str, &[Sym<'_>])]) -> Vec<u8> {
        const SECTION_RVA: u32 = 0x1000;
        const HEADER_SIZE: usize = 0x200;

        let mut body = vec![0u8; 0x10];
        let mut export_dir = (0u32, 0u32);
        let mut import_dir = (0u32, 0u32);

        let put32 =
            |b: &mut [u8], at: usize, v: u32| b[at..at + 4].copy_from_slice(&v.to_le_bytes());
        let put16 =
            |b: &mut [u8], at: usize, v: u16| b[at..at + 2].copy_from_slice(&v.to_le_bytes());
        let align = |b: &mut Vec<u8>, to: usize| {
            let rem = b.len() % to;
            if rem != 0 {
                b.resize(b.len() + to - rem, 0);
            }
        };
        let push_cstr = |b: &mut Vec<u8>, s: &str| -> u32 {
            let rva = SECTION_RVA + b.len() as u32;
            b.extend_from_slice(s.as_bytes());
            b.push(0);
            rva
        };

        if !exports.is_empty() {
            let base = exports.iter().map(|e| e.ordinal).min().unwrap_or(1);
            let max_ord = exports.iter().map(|e| e.ordinal).max().unwrap_or(base);
            let num_funcs = usize::from(max_ord - base) + 1;

            let mut named: Vec<(&str, u16)> = exports
                .iter()
                .filter_map(|e| e.name.map(|n| (n, e.ordinal - base)))
                .collect();
            named.sort_by_key(|(n, _)| *n);

            let exp_off = body.len();
            let exp_rva = SECTION_RVA + exp_off as u32;
            body.resize(exp_off + 40, 0);

            let eat_off = body.len();
            body.resize(eat_off + num_funcs * 4, 0);

            let names_off = body.len();
            body.resize(names_off + named.len() * 4, 0);

            let ords_off = body.len();
            body.resize(ords_off + named.len() * 2, 0);

            let dll_name_rva = push_cstr(&mut body, "t.dll");
            for (i, (name, eat_idx)) in named.iter().enumerate() {
                let nrva = push_cstr(&mut body, name);
                put32(&mut body, names_off + i * 4, nrva);
                put16(&mut body, ords_off + i * 2, *eat_idx);
            }
            for e in exports {
                let idx = usize::from(e.ordinal - base);
                put32(&mut body, eat_off + idx * 4, SECTION_RVA);
            }

            put32(&mut body, exp_off + 12, dll_name_rva);
            put32(&mut body, exp_off + 16, u32::from(base));
            put32(&mut body, exp_off + 20, num_funcs as u32);
            put32(&mut body, exp_off + 24, named.len() as u32);
            put32(&mut body, exp_off + 28, SECTION_RVA + eat_off as u32);
            put32(&mut body, exp_off + 32, SECTION_RVA + names_off as u32);
            put32(&mut body, exp_off + 36, SECTION_RVA + ords_off as u32);
            export_dir = (exp_rva, (body.len() - exp_off) as u32);
        }

        if !imports.is_empty() {
            align(&mut body, 4);
            let desc_off = body.len();
            let desc_rva = SECTION_RVA + desc_off as u32;
            let desc_size = (imports.len() + 1) * 20;
            body.resize(desc_off + desc_size, 0);

            for (idx, (dll_name, syms)) in imports.iter().enumerate() {
                let dll_rva = push_cstr(&mut body, dll_name);
                align(&mut body, 8);
                let int_off = body.len();
                body.resize(int_off + (syms.len() + 1) * 8, 0);
                let iat_off = body.len();
                body.resize(iat_off + (syms.len() + 1) * 8, 0);

                for (sidx, sym) in syms.iter().enumerate() {
                    let thunk: u64 = match sym {
                        Sym::Name(name) => {
                            align(&mut body, 2);
                            let ibn_rva = SECTION_RVA + body.len() as u32;
                            body.extend_from_slice(&0u16.to_le_bytes());
                            body.extend_from_slice(name.as_bytes());
                            body.push(0);
                            u64::from(ibn_rva)
                        }
                        Sym::Ordinal(ord) => (1u64 << 63) | u64::from(*ord),
                    };
                    body[int_off + sidx * 8..int_off + sidx * 8 + 8]
                        .copy_from_slice(&thunk.to_le_bytes());
                    body[iat_off + sidx * 8..iat_off + sidx * 8 + 8]
                        .copy_from_slice(&thunk.to_le_bytes());
                }

                let d_at = desc_off + idx * 20;
                put32(&mut body, d_at, SECTION_RVA + int_off as u32);
                put32(&mut body, d_at + 12, dll_rva);
                put32(&mut body, d_at + 16, SECTION_RVA + iat_off as u32);
            }
            import_dir = (desc_rva, desc_size as u32);
        }

        align(&mut body, 0x200);
        let raw_size = body.len() as u32;
        let size_of_image = SECTION_RVA + raw_size.div_ceil(0x1000) * 0x1000;

        let mut img = vec![0u8; HEADER_SIZE];
        img[0..2].copy_from_slice(b"MZ");
        put32(&mut img, 0x3C, 0x40);
        img[0x40..0x44].copy_from_slice(b"PE\0\0");
        put16(&mut img, 0x44, 0x8664); // AMD64
        put16(&mut img, 0x46, 1); // 1 section
        put16(&mut img, 0x54, 240); // optional header size
        put16(&mut img, 0x56, if dll { 0x2022 } else { 0x0022 });

        let opt = 0x58;
        put16(&mut img, opt, 0x20B); // PE32+
        img[opt + 24..opt + 32].copy_from_slice(&0x1_8000_0000u64.to_le_bytes());
        put32(&mut img, opt + 32, 0x1000);
        put32(&mut img, opt + 36, 0x200);
        put32(&mut img, opt + 56, size_of_image);
        put32(&mut img, opt + 60, HEADER_SIZE as u32);
        put32(&mut img, opt + 108, 16);
        put32(&mut img, opt + 112, export_dir.0);
        put32(&mut img, opt + 116, export_dir.1);
        put32(&mut img, opt + 120, import_dir.0);
        put32(&mut img, opt + 124, import_dir.1);

        let sec = opt + 240;
        img[sec..sec + 6].copy_from_slice(b".rdata");
        put32(&mut img, sec + 8, raw_size);
        put32(&mut img, sec + 12, SECTION_RVA);
        put32(&mut img, sec + 16, raw_size);
        put32(&mut img, sec + 20, HEADER_SIZE as u32);

        img.extend_from_slice(&body);
        img
    }

    #[test]
    fn resolves_same_dir_title_dll_exports_by_name_and_ordinal() {
        let tmp = TempDir::new();
        let root = tmp.path();

        let title_dll = build_pe(
            true,
            &[
                ExportSpec {
                    name: Some("FooInit"),
                    ordinal: 1,
                },
                ExportSpec {
                    name: None,
                    ordinal: 3,
                },
            ],
            &[],
        );
        std::fs::write(root.join("TitleLib.dll"), title_dll).unwrap();

        let exe = build_pe(
            false,
            &[],
            &[(
                "titlelib.DLL",
                &[Sym::Name("FooInit"), Sym::Ordinal(3)],
            )],
        );
        let exe_path = root.join("game.exe");
        std::fs::write(&exe_path, exe).unwrap();

        // Duplicate input paths are scanned once while preserving title DLL resolution.
        let report = collect(&[root.to_path_buf(), exe_path], false).unwrap();
        assert_eq!(report.images, 2);
        assert!(report.failed.is_empty());
        assert!(report.missing.is_empty());

        let stats = &report.dlls["titlelib.dll"];
        assert_eq!(stats.functions.len(), 2);
        assert_eq!(stats.hle, 0);
        assert_eq!(stats.host, 0);
        assert_eq!(stats.title, 2);
        assert_eq!(stats.missing, 0);

        let table = render(&report, false);
        let rows: Vec<Vec<&str>> = table
            .lines()
            .map(|l| l.split_whitespace().collect())
            .filter(|cols: &Vec<&str>| !cols.is_empty())
            .collect();
        assert_eq!(
            rows[1],
            ["dll", "funcs", "hle", "host", "title", "missing", "images"]
        );
        assert_eq!(rows[2], ["titlelib.dll", "2", "0", "0", "2", "0", "1"]);
        assert_eq!(rows[3], ["TOTAL", "2", "0", "0", "2", "0"]);
    }

    #[test]
    fn unexported_symbols_and_other_dir_dlls_stay_missing() {
        let tmp = TempDir::new();
        let root = tmp.path();
        let other = root.join("sub");
        std::fs::create_dir_all(&other).unwrap();

        // Ordinal 2 is an unused hole (not exported); only ordinal 1 ("Present") exists.
        let local_dll = build_pe(
            true,
            &[
                ExportSpec {
                    name: Some("Present"),
                    ordinal: 1,
                },
                ExportSpec {
                    name: Some("AfterHole"),
                    ordinal: 3,
                },
            ],
            &[],
        );
        std::fs::write(root.join("local.dll"), local_dll).unwrap();

        let far_dll = build_pe(
            true,
            &[ExportSpec {
                name: Some("FarFn"),
                ordinal: 1,
            }],
            &[],
        );
        std::fs::write(other.join("remote.dll"), far_dll).unwrap();

        let exe = build_pe(
            false,
            &[],
            &[
                (
                    "local.dll",
                    &[Sym::Name("Present"), Sym::Name("Absent"), Sym::Ordinal(2)],
                ),
                ("remote.dll", &[Sym::Name("FarFn")]),
            ],
        );
        std::fs::write(root.join("game.exe"), exe).unwrap();
        std::fs::write(root.join("broken.dll"), b"MZ\0\0").unwrap();

        let report = collect(&[root.to_path_buf()], false).unwrap();
        assert_eq!(report.images, 3);
        assert_eq!(report.failed.len(), 1);
        let local = &report.dlls["local.dll"];
        assert_eq!(local.title, 1);
        assert_eq!(local.missing, 2);

        let remote = &report.dlls["remote.dll"];
        assert_eq!(remote.title, 0);
        assert_eq!(remote.missing, 1);

        assert!(
            report
                .missing
                .contains_key(&("local.dll".into(), "Absent".into()))
        );
        assert!(
            report
                .missing
                .contains_key(&("local.dll".into(), "#2".into()))
        );
        assert!(
            report
                .missing
                .contains_key(&("remote.dll".into(), "FarFn".into()))
        );

        assert!(collect(&[root.join("does-not-exist")], false).is_err());
    }

    #[test]
    fn host_and_hle_dlls_are_not_taken_from_title_except_host_under_strict() {
        let tmp = TempDir::new();
        let root = tmp.path();

        // Host-allowed DLL shipped in the title dir with a synthetic export.
        let host_dll = build_pe(
            true,
            &[ExportSpec {
                name: Some("SyntheticCrtHook"),
                ordinal: 1,
            }],
            &[],
        );
        std::fs::write(root.join("vcruntime140.dll"), host_dll).unwrap();

        // HLE-served DLL shipped in the title dir with an unimplemented export.
        let hle_dll = build_pe(
            true,
            &[ExportSpec {
                name: Some("SyntheticXMemHook"),
                ordinal: 1,
            }],
            &[],
        );
        std::fs::write(root.join("xmem.dll"), hle_dll).unwrap();

        let exe = build_pe(
            false,
            &[],
            &[
                ("vcruntime140.dll", &[Sym::Name("SyntheticCrtHook")]),
                ("xmem.dll", &[Sym::Name("SyntheticXMemHook")]),
            ],
        );
        std::fs::write(root.join("game.exe"), exe).unwrap();

        let default_report = collect(&[root.to_path_buf()], false).unwrap();
        assert_eq!(default_report.dlls["vcruntime140.dll"].title, 0);
        assert_eq!(default_report.dlls["vcruntime140.dll"].missing, 1);
        assert_eq!(default_report.dlls["xmem.dll"].title, 0);
        assert_eq!(default_report.dlls["xmem.dll"].missing, 1);

        let strict_report = collect(&[root.to_path_buf()], true).unwrap();
        assert_eq!(strict_report.dlls["vcruntime140.dll"].title, 1);
        assert_eq!(strict_report.dlls["vcruntime140.dll"].missing, 0);
        assert_eq!(strict_report.dlls["xmem.dll"].title, 0);
        assert_eq!(strict_report.dlls["xmem.dll"].missing, 1);
    }

    #[test]
    fn cross_dir_same_dll_missing_is_order_independent() {
        let tmp = TempDir::new();
        let root = tmp.path();

        let shared_dll = build_pe(
            true,
            &[ExportSpec {
                name: Some("SharedFn"),
                ordinal: 1,
            }],
            &[],
        );
        let app = build_pe(false, &[], &[("shared.dll", &[Sym::Name("SharedFn")])]);

        let order1 = root.join("order1");
        std::fs::create_dir_all(order1.join("t1_has_dll")).unwrap();
        std::fs::create_dir_all(order1.join("t2_no_dll")).unwrap();
        std::fs::write(order1.join("t1_has_dll").join("shared.dll"), &shared_dll).unwrap();
        std::fs::write(order1.join("t1_has_dll").join("app1.exe"), &app).unwrap();
        std::fs::write(order1.join("t2_no_dll").join("app2.exe"), &app).unwrap();

        let order2 = root.join("order2");
        std::fs::create_dir_all(order2.join("t1_no_dll")).unwrap();
        std::fs::create_dir_all(order2.join("t2_has_dll")).unwrap();
        std::fs::write(order2.join("t1_no_dll").join("app1.exe"), &app).unwrap();
        std::fs::write(order2.join("t2_has_dll").join("shared.dll"), &shared_dll).unwrap();
        std::fs::write(order2.join("t2_has_dll").join("app2.exe"), &app).unwrap();

        let r1 = collect(&[order1], false).unwrap();
        let r2 = collect(&[order2], false).unwrap();

        for report in [&r1, &r2] {
            assert_eq!(report.images, 3);
            assert!(report.failed.is_empty());
            let stats = &report.dlls["shared.dll"];
            assert_eq!(stats.functions.len(), 1);
            assert_eq!(stats.hle, 0);
            assert_eq!(stats.host, 0);
            assert_eq!(stats.title, 0);
            assert_eq!(stats.missing, 1);
            assert_eq!(stats.images.len(), 2);
            assert_eq!(report.missing.len(), 1);
            assert_eq!(
                report.missing[&("shared.dll".into(), "SharedFn".into())].len(),
                1
            );
        }
        assert_eq!(render(&r1, true), render(&r2, true));
    }
}
