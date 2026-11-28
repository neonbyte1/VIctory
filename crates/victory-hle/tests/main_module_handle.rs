//! A module lookup without a name finds the title's exe, not ours. In a
//! process of its own: the main image is process-wide.
#![cfg(windows)]

use std::ffi::c_void;

use victory_hle::{registry, runtime};

type GetModuleHandleEx = extern "system" fn(u32, *const c_void, *mut *mut c_void) -> i32;

#[test]
fn no_name_is_the_titles_exe() {
    let base = 0x7300_0000_0000usize;
    runtime::set_main_image(base);
    for name in ["GetModuleHandleExA", "GetModuleHandleExW"] {
        let export = registry()
            .iter()
            .filter(|m| m.serves("kernel32.dll"))
            .flat_map(|m| &m.exports)
            .find(|e| e.name == name)
            .unwrap();
        // SAFETY: the export's signature; the name's type is not read.
        let f = unsafe { std::mem::transmute::<usize, GetModuleHandleEx>(export.address) };
        // 0, `PIN` and `UNCHANGED_REFCOUNT`.
        for flags in 0..3 {
            let mut out = std::ptr::null_mut();
            assert_eq!(f(flags, std::ptr::null(), &mut out), 1, "{name} {flags}");
            assert_eq!(out as usize, base, "{name} {flags}");
        }
    }
}
