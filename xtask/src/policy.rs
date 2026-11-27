//! Rule #1, statically: the repository only ever contains small text files
//! that we wrote. See docs/asset-policy.md.
//!
//! This is an allowlist, not a blocklist: a file passes only if its name is
//! on the list, it is not too big, it is text, and it does not smuggle data
//! inside text (long hex/base64 runs).

use std::path::Path;
use std::process::Command;

use anyhow::{Context, Result, bail};

const MAX_FILE_SIZE: usize = 512 * 1024;
const MAX_ENCODED_RUN: usize = 512;

const ALLOWED_EXTENSIONS: &[&str] = &[
    "rs", "toml", "md", "txt", "lock", "yml", "yaml", "json", "hlsl", "hlsli", "ps1", "cmd", "sh",
];
const ALLOWED_NAMES: &[&str] = &[
    ".gitignore",
    ".gitattributes",
    ".editorconfig",
    ".victory-root",
    "pre-commit",
    "LICENSE",
    "LICENSE-MIT",
    "LICENSE-APACHE",
];

/// Paths explicitly exempted, with a reason. Keep this empty if at all possible.
const EXCEPTIONS: &[(&str, &str)] = &[(
    "crates/victory-formats/src/bcpack-tables.bin",
    "the BCPack format's constant tables: nothing decodes without them and they cannot be derived (owner's decision, 2026-11-23; docs/asset-policy.md)",
)];

/// Signatures of formats that must never be committed.
const FORBIDDEN_MAGIC: &[(&[u8], &str)] = &[
    (b"MZ", "PE executable"),
    (b"DXBC", "shader container"),
    (b"DDS ", "DDS texture"),
    (b"RIFF", "RIFF (audio/video)"),
    (b"BC\xC0\xDE", "LLVM bitcode"),
    (b"Microsoft C/C++ MSF", "PDB"),
    (b"!<arch>", "library archive"),
    (b"PK\x03\x04", "zip"),
    (b"7z\xBC\xAF", "7-zip"),
    (b"MSCF", "cabinet"),
    (b"\xD0\xCF\x11\xE0", "MSI/OLE"),
];

pub fn run(root: &Path, staged: bool) -> Result<()> {
    let files = list_files(root, staged)?;
    let mut violations = Vec::new();
    for path in &files {
        if EXCEPTIONS.iter().any(|(p, _)| p == path) {
            continue;
        }
        let content = if staged {
            let out = Command::new("git")
                .args(["show", &format!(":{path}")])
                .current_dir(root)
                .output()?;
            if !out.status.success() {
                continue; // deleted in the index
            }
            out.stdout
        } else {
            match std::fs::read(root.join(path)) {
                Ok(c) => c,
                Err(_) => continue,
            }
        };
        for problem in check(path, &content) {
            violations.push(format!("{path}: {problem}"));
        }
    }
    if violations.is_empty() {
        println!("asset policy: {} files ok", files.len());
        return Ok(());
    }
    for v in &violations {
        eprintln!("POLICY: {v}");
    }
    bail!(
        "{} asset-policy violation(s). Nothing from the game may be committed; \
         see docs/asset-policy.md",
        violations.len()
    )
}

fn list_files(root: &Path, staged: bool) -> Result<Vec<String>> {
    let git = |args: &[&str]| -> Result<Option<Vec<String>>> {
        let out = Command::new("git")
            .args(args)
            .current_dir(root)
            .output()
            .context("running git")?;
        if !out.status.success() {
            return Ok(None);
        }
        Ok(Some(
            out.stdout
                .split(|&b| b == 0)
                .filter(|s| !s.is_empty())
                .map(|s| String::from_utf8_lossy(s).replace('\\', "/"))
                .collect(),
        ))
    };
    if staged {
        return git(&[
            "diff",
            "--cached",
            "--name-only",
            "--diff-filter=ACMR",
            "-z",
        ])?
        .context("--staged needs a git repository");
    }
    if let Some(mut tracked) = git(&[
        "ls-files",
        "-z",
        "--cached",
        "--others",
        "--exclude-standard",
    ])? {
        tracked.sort();
        tracked.dedup();
        return Ok(tracked);
    }
    // Not a git repository: walk everything except build output.
    Ok(walkdir::WalkDir::new(root)
        .into_iter()
        .filter_entry(|e| !matches!(e.file_name().to_str(), Some("target" | ".git")))
        .filter_map(Result::ok)
        .filter(|e| e.file_type().is_file())
        .filter_map(|e| {
            e.path()
                .strip_prefix(root)
                .ok()
                .map(|p| p.to_string_lossy().replace('\\', "/"))
        })
        .collect())
}

/// All problems with one file. Pure, for testing.
pub fn check(path: &str, content: &[u8]) -> Vec<String> {
    let mut problems = Vec::new();
    let name = path.rsplit('/').next().unwrap_or(path);
    let ext = name.rsplit_once('.').map(|(_, e)| e.to_ascii_lowercase());
    let allowed = ALLOWED_NAMES.contains(&name)
        || ext
            .as_deref()
            .is_some_and(|e| ALLOWED_EXTENSIONS.contains(&e));
    if !allowed {
        problems.push("file type not on the allowlist".to_owned());
    }
    if content.len() > MAX_FILE_SIZE {
        problems.push(format!(
            "{} bytes exceeds the {MAX_FILE_SIZE}-byte limit",
            content.len()
        ));
    }
    for (magic, what) in FORBIDDEN_MAGIC {
        if content.starts_with(magic) {
            problems.push(format!("looks like a {what}"));
        }
    }
    if content.contains(&0) {
        problems.push("binary content (NUL bytes)".to_owned());
    } else if let Some(run) = longest_encoded_run(content)
        && run > MAX_ENCODED_RUN
    {
        problems.push(format!("{run}-character hex/base64 run (embedded data?)"));
    } else {
        let digits = longest_hex_literal_run(content);
        if digits > MAX_ENCODED_RUN {
            problems.push(format!(
                "{digits} hex digits in one run of \\x/0x literals (embedded data?)"
            ));
        }
        let bytes = longest_decimal_run(content);
        if bytes > MAX_ENCODED_RUN {
            problems.push(format!(
                "{bytes} byte-range decimal literals in one run (embedded data?)"
            ));
        }
    }
    problems
}

/// Longest run of characters that could be hex or base64 payload.
fn longest_encoded_run(content: &[u8]) -> Option<usize> {
    let is_payload = |b: &u8| b.is_ascii_alphanumeric() || matches!(b, b'+' | b'/' | b'=');
    content.split(|b| !is_payload(b)).map(<[u8]>::len).max()
}

/// Most hex digits in `\xNN` escapes or `0x` literals with only commas,
/// whitespace or backslashes between them: a byte string or `xxd -i` array,
/// which `longest_encoded_run` sees as many 3- or 4-character runs.
fn longest_hex_literal_run(content: &[u8]) -> usize {
    let (mut i, mut run, mut longest) = (0, 0, 0);
    while i < content.len() {
        match &content[i..] {
            [b'\\' | b'0', b'x' | b'X', rest @ ..] => {
                let len = rest
                    .iter()
                    .take_while(|b| b.is_ascii_hexdigit() || **b == b'_')
                    .count();
                run += rest[..len].iter().filter(|b| b.is_ascii_hexdigit()).count();
                longest = longest.max(run);
                // Skip a type suffix such as `u8` or `u32`.
                let suffix = rest[len..]
                    .iter()
                    .take_while(|b| b.is_ascii_alphanumeric())
                    .count();
                i += 2 + len + suffix;
            }
            [b, ..] if matches!(b, b',' | b'\\') || b.is_ascii_whitespace() => i += 1,
            _ => {
                run = 0;
                i += 1;
            }
        }
    }
    longest
}

/// Most byte-range decimal literals in one run of `[77, 90, ...]` with only
/// commas and whitespace between them: a `[u8]` array of copied data, whose
/// 1- to 3-character literals `longest_encoded_run` never sees as one run.
///
/// Counted in literals, not digits, so a table of small numbers that a
/// decoder really needs (`COEF0_LENS`, 272 entries) stays legal; the largest
/// such table in the tree has 272, well under the limit.
fn longest_decimal_run(content: &[u8]) -> usize {
    let (mut i, mut run, mut longest) = (0, 0, 0);
    while i < content.len() {
        let digits = content[i..]
            .iter()
            .take_while(|b| b.is_ascii_digit())
            .count();
        if digits == 0 {
            if !matches!(content[i], b',' | b' ' | b'\t' | b'\r' | b'\n') {
                run = 0;
            }
            i += 1;
            continue;
        }
        // A bare literal: at most three digits, no leading zero, no type
        // suffix or fraction after it, nothing but a separator before the
        // next one. Past three digits it cannot be a byte, and folding a
        // long run of digits (an address, a hash) would overflow.
        let followed = content.get(i + digits).copied();
        let bare = digits <= 3
            && !(digits > 1 && content[i] == b'0')
            && matches!(followed, None | Some(b',' | b' ' | b'\t' | b'\r' | b'\n'));
        let value = if bare {
            Some(
                content[i..i + digits]
                    .iter()
                    .fold(0u32, |v, b| v * 10 + u32::from(b - b'0')),
            )
        } else {
            None
        };
        if matches!(value, Some(v) if v <= 0xFF) {
            run += 1;
            longest = longest.max(run);
        } else {
            run = 0;
        }
        i += digits;
    }
    longest
}

#[cfg(test)]
mod tests {
    use super::check;

    /// Enforces the policy wherever tests run, even in clones without the hook.
    #[test]
    fn repository_passes_policy() {
        super::run(&crate::workspace_root(), false).unwrap();
    }

    #[test]
    fn accepts_source() {
        assert!(check("crates/x/src/lib.rs", b"fn main() {}\n").is_empty());
        assert!(check(".gitignore", b"/target/\n").is_empty());
        assert!(check(".githooks/pre-commit", b"#!/bin/sh\n").is_empty());
    }

    #[test]
    fn rejects_game_like_files() {
        assert!(!check("assets/tex.dds", b"DDS \x7c\0\0\0").is_empty());
        assert!(
            !check("notes.txt", b"MZ\x90\0").is_empty(),
            "magic wins over extension"
        );
        assert!(
            !check("data.bin", b"hello").is_empty(),
            "extension not allowed"
        );
        assert!(
            !check("a.rs", &vec![b'a'; 600 * 1024]).is_empty(),
            "too big"
        );
    }

    #[test]
    fn rejects_data_smuggled_into_text() {
        let blob = format!("const DATA: &str = \"{}\";", "A1".repeat(400));
        assert!(!check("src/blob.rs", blob.as_bytes()).is_empty());
    }

    /// Synthetic bytes, never game data.
    fn synthetic(n: u32) -> impl Iterator<Item = u32> {
        (0..n).map(|i| (i * 37 + 11) % 256)
    }

    #[test]
    fn rejects_data_smuggled_as_an_escaped_byte_string() {
        // 4096 bytes as one unbroken `\xNN` hex run, the shape a byte-string
        // literal of copied data takes.
        let esc: String = synthetic(4096).map(|b| format!("\\x{b:02x}")).collect();
        let src = format!("pub const DATA: &[u8] = b\"{esc}\";\n");
        assert!(
            !check("src/data.rs", src.as_bytes()).is_empty(),
            "{} bytes of \\x-escaped hex passed",
            src.len()
        );
    }

    #[test]
    fn rejects_data_smuggled_as_a_hex_byte_array() {
        // `xxd -i` style: 4096 bytes as hex literals.
        let lits: Vec<String> = synthetic(4096).map(|b| format!("0x{b:02x}")).collect();
        let src = format!("pub const DATA: [u8; 4096] = [{}];\n", lits.join(", "));
        assert!(
            !check("src/data.rs", src.as_bytes()).is_empty(),
            "{} bytes of hex array passed",
            src.len()
        );
        // A short constant table, like the 16-entry ones in the tree, stays legal.
        let table: Vec<String> = synthetic(16).map(|w| format!("0x{w:08x}_u32")).collect();
        let src = format!(
            "const T: [u32; 16] = [\n    {},\n];\n",
            table.join(",\n    ")
        );
        assert!(check("src/table.rs", src.as_bytes()).is_empty());
    }

    #[test]
    fn rejects_data_smuggled_as_a_decimal_byte_array() {
        // 4096 bytes as plain decimal literals, the shape a `[u8]` dump of
        // copied data takes once a formatter has re-spaced it.
        let lits: Vec<String> = synthetic(4096).map(|b| b.to_string()).collect();
        let src = format!("pub const DATA: [u8; 4096] = [{}];\n", lits.join(", "));
        let problem = check("src/data.rs", src.as_bytes());
        assert!(
            problem.iter().any(|p| p.contains("decimal literals")),
            "{} bytes of decimal literals passed: {problem:?}",
            src.len()
        );

        // The same bytes wrapped one per line, as `rustfmt` would leave it.
        let src = format!(
            "pub const DATA: [u8; 4096] = [\n    {},\n];\n",
            lits.join(",\n    ")
        );
        assert!(
            !check("src/data.rs", src.as_bytes()).is_empty(),
            "4096 decimal literals one per line passed"
        );
    }

    #[test]
    fn accepts_tables_of_small_decimals() {
        // The shape of the trees' own decoder tables: many entries, all
        // byte-sized, that nothing can be decoded without.
        let lens: Vec<String> = (0..272u32).map(|i| ((i * 7) % 22).to_string()).collect();
        let src = format!(
            "pub(super) const COEF0_LENS: [u8; 272] = [\n    {},\n];\n",
            lens.join(", ")
        );
        assert!(
            check("crates/victory-formats/src/xma/tables.rs", src.as_bytes()).is_empty(),
            "a 272-entry byte table was rejected"
        );

        // Values past a byte are indices, not payload.
        let syms: Vec<String> = (0..272u32).map(|i| i.to_string()).collect();
        let src = format!(
            "pub(super) const COEF0_SYMS: [u16; 272] = [\n    {},\n];\n",
            syms.join(", ")
        );
        assert!(
            check("crates/victory-formats/src/xma/tables.rs", src.as_bytes()).is_empty(),
            "a table of indices was rejected"
        );

        // Type suffixes, fractions and leading zeros are not byte literals.
        assert!(
            check(
                "src/x.rs",
                format!("let a = [{}];\n", vec!["1u8"; 400].join(", ")).as_bytes()
            )
            .is_empty(),
            "1u8 literals counted as payload"
        );
        assert!(
            check(
                "src/x.rs",
                format!("let a = [{}];\n", vec!["0.5"; 400].join(", ")).as_bytes()
            )
            .is_empty(),
            "fractions counted as payload"
        );
    }
}
