#![cfg(windows)]

use victory_loader::{load_title_dll, load_title_dlls, prepare};
use victory_pe::{ImageInfo, ImportModule, layout};

/// Builds a valid synthetic DLL image from the test executable, optionally
/// rewriting its first import descriptor's DLL name to `first_import`.
fn synthetic_dll(first_import: Option<&[u8]>) -> Vec<u8> {
    let mut dll = std::fs::read(std::env::current_exe().unwrap()).unwrap();
    let info = ImageInfo::parse(&dll).unwrap();
    let fh = layout::file_header_offset(&dll).unwrap();
    let flags = info.characteristics | layout::file_flags::DLL;
    dll[fh + layout::FILE_HEADER_CHARACTERISTICS..][..2].copy_from_slice(&flags.to_le_bytes());

    if let Some(replacement) = first_import {
        assert!(replacement.ends_with(b"\0"));
        let imp_dd = layout::data_directory_offset(&dll, layout::dir::IMPORT).unwrap();
        let imp_rva = layout::read_u32(&dll, imp_dd).unwrap();
        let desc_off = info.rva_to_file_offset(imp_rva).unwrap() as usize;
        let name_rva = layout::read_u32(&dll, desc_off + 12).unwrap();
        let name_off = info.rva_to_file_offset(name_rva).unwrap() as usize;
        let existing_len = dll[name_off..].iter().position(|&b| b == 0).unwrap() + 1;
        assert!(
            replacement.len() <= existing_len,
            "replacement import name must fit inside existing name slot"
        );
        dll[name_off..name_off + replacement.len()].copy_from_slice(replacement);
    }

    dll
}

/// If a transitive dependency fails to prepare or load inside `map_dll`, the
/// already-mapped parent DLL must be unloaded so its `HMODULE` and cached file
/// are not leaked for the life of the process.
#[test]
fn load_title_dll_unloads_parent_when_dependency_fails() {
    let title_dir = tempfile::tempdir().unwrap();
    let cache_dir = tempfile::tempdir().unwrap();

    let parent = synthetic_dll(Some(b"bad.dll\0"));
    std::fs::write(title_dir.path().join("parent.dll"), &parent).unwrap();
    std::fs::write(title_dir.path().join("bad.dll"), b"not a valid PE image").unwrap();

    let wanted = |name: &str| matches!(name, "parent.dll" | "bad.dll");
    let err = load_title_dll("parent.dll", title_dir.path(), cache_dir.path(), &wanted);
    assert!(
        err.is_err(),
        "expected corrupt transitive dependency to fail"
    );

    let cached = cache_dir.path().join(prepare(&parent).unwrap().file_name());
    assert!(
        std::fs::remove_file(&cached).is_ok(),
        "parent.dll remained mapped (leaked HMODULE) after bad.dll failed"
    );
}

/// If a later sibling DLL fails during `load_title_dlls`, earlier already-mapped
/// sibling DLLs must be unloaded instead of leaking their `HMODULE`s.
#[test]
fn load_title_dlls_unloads_earlier_siblings_when_later_dll_fails() {
    let title_dir = tempfile::tempdir().unwrap();
    let cache_dir = tempfile::tempdir().unwrap();

    let good = synthetic_dll(None);
    std::fs::write(title_dir.path().join("good.dll"), &good).unwrap();
    std::fs::write(title_dir.path().join("bad.dll"), b"not a valid PE image").unwrap();

    let mut main = ImageInfo::parse(&good).unwrap();
    main.imports = vec![
        ImportModule {
            dll: "good.dll".to_string(),
            entries: Vec::new(),
        },
        ImportModule {
            dll: "bad.dll".to_string(),
            entries: Vec::new(),
        },
    ];
    main.delay_imports.clear();

    let wanted = |name: &str| matches!(name, "good.dll" | "bad.dll");
    let err = load_title_dlls(&main, title_dir.path(), cache_dir.path(), &wanted);
    assert!(err.is_err(), "expected corrupt sibling DLL to fail");

    let cached = cache_dir.path().join(prepare(&good).unwrap().file_name());
    assert!(
        std::fs::remove_file(&cached).is_ok(),
        "good.dll remained mapped (leaked HMODULE) after bad.dll failed"
    );
}
