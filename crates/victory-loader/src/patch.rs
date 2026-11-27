//! `VICTORY_PATCH=<module>:<rva>:<hex bytes>,...` writes bytes into a loaded
//! module. The write happens in [`crate::load`], after the Windows loader
//! has mapped and relocated the image and before TLS redirection and import
//! binding, and each application is logged with the bytes it replaced, so a
//! run is reproducible from its own log.

use std::env;

/// One `<module>:<rva>:<hex bytes>` entry of `VICTORY_PATCH`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct Patch {
    /// File name of the module the entry applies to, matched
    /// case-insensitively.
    module: String,
    /// Offset from the image base.
    rva: usize,
    /// Bytes to write at `rva`.
    bytes: Vec<u8>,
}

/// A hex number, with an optional `0x` prefix, like the other knobs.
fn hex(s: &str) -> Option<usize> {
    let s = s.trim().trim_start_matches("0x");
    // `from_str_radix` would accept a leading `+`; hex digits only.
    if s.is_empty() || !s.bytes().all(|b| b.is_ascii_hexdigit()) {
        return None;
    }
    usize::from_str_radix(s, 16).ok()
}

/// An even-length run of hex digits, with an optional `0x` prefix.
fn hex_bytes(s: &str) -> Option<Vec<u8>> {
    let s = s.trim().trim_start_matches("0x");
    if s.is_empty() || !s.len().is_multiple_of(2) || !s.bytes().all(|b| b.is_ascii_hexdigit()) {
        return None;
    }
    (0..s.len())
        .step_by(2)
        .map(|i| u8::from_str_radix(&s[i..i + 2], 16).ok())
        .collect()
}

/// The parsed `VICTORY_PATCH` list; malformed entries are skipped, like the
/// other debugging knobs.
pub(crate) fn from_env() -> Vec<Patch> {
    let Ok(spec) = env::var("VICTORY_PATCH") else {
        return Vec::new();
    };
    parse(&spec)
}

fn parse(spec: &str) -> Vec<Patch> {
    spec.split(',')
        .filter_map(|entry| {
            let mut parts = entry.splitn(3, ':');
            let module = parts.next()?.trim();
            let patch = Patch {
                module: module.into(),
                rva: hex(parts.next()?)?,
                bytes: hex_bytes(parts.next()?)?,
            };
            if module.is_empty() {
                return None;
            }
            Some(patch)
        })
        .collect()
}

/// The entries of `patches` that name `module` (its file name,
/// case-insensitively) and fit an image of `size` bytes.
fn planned<'a>(patches: &'a [Patch], module: &str, size: usize) -> Vec<&'a Patch> {
    patches
        .iter()
        .filter(|p| p.module.eq_ignore_ascii_case(module))
        .filter(|p| p.rva.checked_add(p.bytes.len()).is_some_and(|end| end <= size))
        .collect()
}

/// Writes `patch` at the start of `target` — the image bytes at
/// `patch.rva` — returning the replaced bytes. The caller has bounds-checked
/// the range through [`planned`].
fn write(target: &mut [u8], patch: &Patch) -> Vec<u8> {
    let old = target[..patch.bytes.len()].to_vec();
    target[..patch.bytes.len()].copy_from_slice(&patch.bytes);
    old
}

/// Applies this module's `VICTORY_PATCH` entries at `base` (`size` bytes).
#[cfg(windows)]
pub(crate) fn apply(base: usize, size: usize, module: &str) {
    use std::ffi::c_void;

    use windows_sys::Win32::System::Diagnostics::Debug::FlushInstructionCache;
    use windows_sys::Win32::System::Memory::{PAGE_EXECUTE_READWRITE, VirtualProtect};
    use windows_sys::Win32::System::Threading::GetCurrentProcess;

    /// Restores a range's page protection on drop, including on unwind.
    struct RestoreProtection(*const c_void, usize, u32);
    impl Drop for RestoreProtection {
        fn drop(&mut self) {
            let mut ignored = 0;
            // SAFETY: restoring the range's own protection.
            unsafe { VirtualProtect(self.0, self.1, self.2, &mut ignored) };
        }
    }

    let patches = from_env();
    let planned = planned(&patches, module, size);
    let named = patches
        .iter()
        .filter(|p| p.module.eq_ignore_ascii_case(module))
        .count();
    if named > planned.len() {
        tracing::warn!(
            module,
            skipped = named - planned.len(),
            "VICTORY_PATCH entries outside the image, skipped"
        );
    }
    for patch in planned {
        let at = base + patch.rva;
        let len = patch.bytes.len();
        let mut protection = 0;
        // SAFETY: `at..at + len` lies inside the mapped image, checked by
        // `planned`; the pages may be read-only (`.text`), so lift the
        // protection for the write, then restore it. The target is code, so
        // keep the pages executable while writable, like the TLS
        // redirection.
        if unsafe {
            VirtualProtect(at as *const c_void, len, PAGE_EXECUTE_READWRITE, &mut protection)
        } == 0
        {
            tracing::warn!(
                module,
                rva = format_args!("{:#x}", patch.rva),
                error = %std::io::Error::last_os_error(),
                "VirtualProtect failed; VICTORY_PATCH entry skipped"
            );
            continue;
        }
        let restore = RestoreProtection(at as *const c_void, len, protection);
        // SAFETY: in-bounds of the mapped image, made writable above; the
        // slice starts at the patch's rva, which is where `write` puts the
        // bytes.
        let old = write(unsafe { std::slice::from_raw_parts_mut(at as *mut u8, len) }, patch);
        drop(restore);
        // SAFETY: flushing the code this patch wrote, like the TLS
        // redirection.
        unsafe { FlushInstructionCache(GetCurrentProcess(), at as *const c_void, len) };
        tracing::info!(
            module,
            rva = format_args!("{:#x}", patch.rva),
            old = format_args!("{old:02x?}"),
            new = format_args!("{:02x?}", patch.bytes),
            "VICTORY_PATCH applied"
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_entries_like_the_other_knobs() {
        let patches = parse("game.exe:1234:90, Game.DLL : 0x234 : 90cc ");
        assert_eq!(
            patches,
            vec![
                Patch {
                    module: "game.exe".into(),
                    rva: 0x1234,
                    bytes: vec![0x90],
                },
                Patch {
                    module: "Game.DLL".into(),
                    rva: 0x234,
                    bytes: vec![0x90, 0xcc],
                },
            ]
        );
    }

    #[test]
    fn skips_malformed_entries() {
        // Missing fields, a bad rva (including the `+` `from_str_radix`
        // would accept), odd-length and empty byte runs, a non-hex byte and
        // an empty module name are all dropped.
        let patches = parse("game.exe:10, game.exe:zz:90, game.exe:+10:90, game.exe:0x+10:90, game.exe:10:9, game.exe:10:0x, game.exe:10:9g, :10:90");
        assert!(patches.is_empty(), "{patches:?}");
        // One good entry survives among them.
        let patches = parse("game.exe:zz:90,game.exe:10:9091");
        assert_eq!(patches.len(), 1);
        assert_eq!(patches[0].bytes, vec![0x90, 0x91]);
    }

    #[test]
    fn planned_matches_the_module_and_the_image_bounds() {
        let patches = parse("game.exe:0:90, GAME.EXE:e:9091, game.exe:f:9090, other.exe:0:90");
        // An 0x10-byte image: the entry at 0xf with two bytes is out.
        let picked = planned(&patches, "game.exe", 0x10);
        assert_eq!(picked.len(), 2);
        assert_eq!(picked[0].rva, 0);
        assert_eq!(picked[1].rva, 0xe);
        // Case-insensitive on the module name.
        let picked = planned(&patches, "Game.Exe", 0x10);
        assert_eq!(picked.len(), 2);
        // A module nothing names plans nothing.
        assert!(planned(&patches, "unrelated.exe", 0x10).is_empty());
    }

    #[test]
    fn planned_cannot_overflow_the_image_end() {
        let patches = parse(&format!("game.exe:{:x}:9090", usize::MAX - 1));
        assert!(planned(&patches, "game.exe", 0x10).is_empty());
    }

    #[test]
    fn write_replaces_and_reports_the_old_bytes() {
        let patches = parse("game.exe:4:010203");
        let planned = planned(&patches, "game.exe", 16);
        let mut image = [0xccu8; 16];
        // `apply` passes the image bytes at the patch's rva.
        let old = write(&mut image[4..7], planned[0]);
        assert_eq!(old, [0xcc, 0xcc, 0xcc]);
        assert_eq!(&image[4..7], &[1, 2, 3]);
        assert_eq!(image[7], 0xcc);
    }
}
