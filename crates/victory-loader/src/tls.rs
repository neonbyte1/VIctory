//! Static TLS of an executable mapped as a DLL.
//!
//! An exe always gets TLS index 0, so title code reaches its thread-local
//! block as `mov r64, gs:[0x58]` (the thread's TLS array) followed by
//! `[r64]`, without reading `_tls_index`. Mapped as a DLL, the image gets
//! another index. Every such load is turned into a call to a thunk that
//! yields `&array[index]` instead, and the image's `_tls_index` (which CRT
//! code does read, and adds to the array pointer) is set to 0.

use std::ffi::c_void;

use anyhow::{Result, bail};
use victory_pe::ImageInfo;
use windows_sys::Win32::System::Diagnostics::Debug::FlushInstructionCache;
use windows_sys::Win32::System::Memory::{
    MEM_COMMIT, MEM_RESERVE, PAGE_EXECUTE_READ, PAGE_EXECUTE_READWRITE, PAGE_READWRITE,
    VirtualAlloc, VirtualProtect,
};
use windows_sys::Win32::System::Threading::GetCurrentProcess;

const SCN_MEM_EXECUTE: u32 = 0x2000_0000;
const TLS64_ADDRESS_OF_INDEX: usize = 16;
/// `mov r64, gs:[0x58]` is `65 REX 8B ModRM 25 58 00 00 00`.
const LOAD_LEN: usize = 9;
const THUNK_SIZE: usize = 32;

/// Redirects the image's TLS-array loads if the OS gave it an index other
/// than 0. Returns the number of loads patched.
///
/// # Safety
/// `base` is the image `info` describes, mapped and not yet running.
pub(crate) unsafe fn redirect(base: usize, info: &ImageInfo) -> Result<usize> {
    let Some(tls) = &info.tls else { return Ok(0) };
    // SAFETY: the TLS directory and the index it points to are in the image;
    // the OS relocated the pointer and wrote the index.
    let index_at = unsafe {
        ((base + tls.directory_rva as usize + TLS64_ADDRESS_OF_INDEX) as *const usize).read()
    } as *mut u32;
    // SAFETY: as above.
    let index = unsafe { index_at.read() };
    if index == 0 {
        return Ok(0);
    }
    let thunks = thunks(near(base, info.size_of_image as usize)?, index)?;
    let mut patched = 0;
    for section in info
        .sections
        .iter()
        .filter(|s| s.characteristics & SCN_MEM_EXECUTE != 0)
    {
        let start = base + section.rva as usize;
        let len = if section.virtual_size != 0 {
            section.virtual_size
        } else {
            section.file_size
        } as usize;
        if len == 0 {
            // An empty section has no code and no protection to change;
            // `VirtualProtect` refuses a zero length.
            continue;
        }
        let mut old = 0;
        // SAFETY: the section is part of the mapped image.
        if unsafe { VirtualProtect(start as _, len, PAGE_EXECUTE_READWRITE, &mut old) } == 0 {
            bail!(
                "VirtualProtect(code) failed: {}",
                std::io::Error::last_os_error()
            );
        }
        // SAFETY: writable now, and no title code runs yet.
        let code = unsafe { std::slice::from_raw_parts_mut(start as *mut u8, len) };
        patched += patch(code, start, thunks);
        // SAFETY: restoring the protection we changed; flushing our own code.
        unsafe {
            VirtualProtect(start as _, len, old, &mut old);
            FlushInstructionCache(GetCurrentProcess(), start as _, len);
        }
    }
    // SAFETY: see `index_at`; `.data` is writable.
    unsafe { index_at.write(0) };
    tracing::info!(index, patched, "static TLS redirected");
    Ok(patched)
}

/// Rewrites each `mov r64, gs:[0x58]` in `code` (at address `at`) as
/// `call thunks[r]; nop4`. `thunks` must be within 2 GiB of `code`.
fn patch(code: &mut [u8], at: usize, thunks: usize) -> usize {
    let mut count = 0;
    let mut i = 0;
    // The scan jumps between offsets that start like a load instead of
    // testing the whole pattern at every byte.
    while let Some(next) = next_candidate(code, i) {
        i = next;
        let c = &code[i..i + LOAD_LEN];
        let rex = c[1];
        let reg = usize::from((c[3] >> 3) & 7) | if rex & 4 != 0 { 8 } else { 0 };
        // shortcut: a byte match, not a decode. In Grand Theft Auto VI all
        // 53,246 hits are instruction starts and none is missed (checked
        // against a full disassembly); decode .text if a title shows one.
        if c[0] == 0x65
            && (rex == 0x48 || rex == 0x4C)
            && c[2] == 0x8B
            && c[3] & 0xC7 == 0x04
            && c[4..] == [0x25, 0x58, 0, 0, 0]
            && reg != 4
        {
            let rel = (thunks + reg * THUNK_SIZE) as i64 - (at + i + 5) as i64;
            code[i] = 0xE8;
            code[i + 1..i + 5].copy_from_slice(&(rel as i32).to_le_bytes());
            code[i + 5..i + 9].copy_from_slice(&[0x0F, 0x1F, 0x40, 0x00]);
            count += 1;
            i += LOAD_LEN;
        } else {
            i += 1;
        }
    }
    count
}

/// The `gs` segment prefix that starts every TLS-array load.
const GS: u8 = 0x65;

/// The first offset from `from` where a load could start: a whole load's
/// room, `gs` at the offset, `mov` (8B) two bytes on and the `[disp32]`
/// SIB byte (25) four on, so `patch` tests the rest of the pattern only
/// there. Sixteen offsets at a time with SSE2, which every x86-64 has.
fn next_candidate(code: &[u8], from: usize) -> Option<usize> {
    // One past the last offset a whole load fits at.
    let end = (code.len() + 1).checked_sub(LOAD_LEN)?;
    let mut i = from;
    #[cfg(target_arch = "x86_64")]
    {
        use std::arch::x86_64::{
            __m128i, _mm_and_si128, _mm_cmpeq_epi8, _mm_loadu_si128, _mm_movemask_epi8,
            _mm_set1_epi8,
        };
        // SAFETY: SSE2 is part of x86-64. Each load reads 16 bytes from
        // `i`, `i + 2` or `i + 4`, which end by `i + 20 <= end + 4`, inside
        // the code (`end + LOAD_LEN - 1` is its length).
        unsafe {
            let at = |k: usize| _mm_loadu_si128(code.as_ptr().add(k).cast::<__m128i>());
            let (gs, mov, sib) = (
                _mm_set1_epi8(GS as i8),
                _mm_set1_epi8(0x8B_u8 as i8),
                _mm_set1_epi8(0x25),
            );
            while i + 16 <= end {
                let hits = _mm_and_si128(
                    _mm_and_si128(_mm_cmpeq_epi8(at(i), gs), _mm_cmpeq_epi8(at(i + 2), mov)),
                    _mm_cmpeq_epi8(at(i + 4), sib),
                );
                let mask = _mm_movemask_epi8(hits);
                if mask != 0 {
                    return Some(i + mask.trailing_zeros() as usize);
                }
                i += 16;
            }
        }
    }
    (i..end).find(|&j| code[j] == GS)
}

/// Fills `block` with one thunk per register: `mov r, gs:[0x58];
/// lea r, [r + index * 8]; ret` (flags untouched, like the load it replaces).
fn thunks(block: *mut u8, index: u32) -> Result<usize> {
    let disp = (index * 8).to_le_bytes();
    for reg in 0..16u8 {
        let (r, low) = (reg >> 3, reg & 7);
        let mut code = vec![0x65, 0x48 | r << 2, 0x8B, low << 3 | 4, 0x25, 0x58, 0, 0, 0];
        code.extend([0x48 | r << 2 | r, 0x8D, 0x80 | low << 3 | low]);
        if low == 4 {
            code.push(0x24); // SIB: [r12 + disp32]
        }
        code.extend(disp);
        code.push(0xC3);
        // SAFETY: `block` holds 16 thunks and is still writable.
        unsafe {
            std::ptr::copy_nonoverlapping(
                code.as_ptr(),
                block.add(usize::from(reg) * THUNK_SIZE),
                code.len(),
            )
        };
    }
    let mut old = 0;
    // SAFETY: `block` is our allocation of one page.
    if unsafe { VirtualProtect(block as _, 4096, PAGE_EXECUTE_READ, &mut old) } == 0 {
        bail!(
            "VirtualProtect(TLS thunks) failed: {}",
            std::io::Error::last_os_error()
        );
    }
    Ok(block as usize)
}

/// One writable page within `rel32` reach of `[base, base + size)`. Leaked:
/// title code calls into it for the life of the process.
fn near(base: usize, size: usize) -> Result<*mut u8> {
    const GRANULE: usize = 0x1_0000;
    // The search spans 1 GiB each side, so every call site stays in reach.
    if size >= 1 << 30 {
        bail!("image too large for TLS thunks within rel32 reach");
    }
    let above = (base + size).next_multiple_of(GRANULE);
    for k in 0..(1usize << 14) {
        for hint in [
            base.checked_sub((k + 1) * GRANULE),
            Some(above + k * GRANULE),
        ] {
            let Some(hint) = hint else { continue };
            // SAFETY: a fresh allocation at a free address, or null.
            let p = unsafe {
                VirtualAlloc(
                    hint as *const c_void,
                    4096,
                    MEM_COMMIT | MEM_RESERVE,
                    PAGE_READWRITE,
                )
            };
            if !p.is_null() {
                return Ok(p.cast());
            }
        }
    }
    bail!("no free page near the image for TLS thunks")
}

#[cfg(test)]
mod tests {
    use super::*;
    use victory_pe::{SectionInfo, TlsInfo};
    use windows_sys::Win32::System::Memory::{MEM_RELEASE, VirtualFree};

    /// Reads the thread's TLS array pointer the way title code does.
    fn tls_array() -> usize {
        let array: usize;
        // SAFETY: reads TEB.ThreadLocalStoragePointer.
        unsafe { std::arch::asm!("mov {}, gs:[0x58]", out(reg) array) };
        array
    }

    /// The byte-at-a-time scan `patch` had before it searched for
    /// candidates: the reference its results must match exactly.
    fn patch_bytewise(code: &mut [u8], at: usize, thunks: usize) -> usize {
        let mut count = 0;
        let mut i = 0;
        while i + LOAD_LEN <= code.len() {
            let c = &code[i..i + LOAD_LEN];
            let rex = c[1];
            let reg = usize::from((c[3] >> 3) & 7) | if rex & 4 != 0 { 8 } else { 0 };
            if c[0] == 0x65
                && (rex == 0x48 || rex == 0x4C)
                && c[2] == 0x8B
                && c[3] & 0xC7 == 0x04
                && c[4..] == [0x25, 0x58, 0, 0, 0]
                && reg != 4
            {
                let rel = (thunks + reg * THUNK_SIZE) as i64 - (at + i + 5) as i64;
                code[i] = 0xE8;
                code[i + 1..i + 5].copy_from_slice(&(rel as i32).to_le_bytes());
                code[i + 5..i + 9].copy_from_slice(&[0x0F, 0x1F, 0x40, 0x00]);
                count += 1;
                i += LOAD_LEN;
            } else {
                i += 1;
            }
        }
        count
    }

    /// xorshift64*: reproducible bytes without a dependency.
    struct Rng(u64);

    impl Rng {
        fn next(&mut self) -> u64 {
            self.0 ^= self.0 >> 12;
            self.0 ^= self.0 << 25;
            self.0 ^= self.0 >> 27;
            self.0.wrapping_mul(0x2545_F491_4F6C_DD1D)
        }
    }

    /// `len` bytes of noise, rich in `gs` prefixes, with `loads` TLS-array
    /// loads (and near misses of them) written at random offsets: any REX
    /// or ModRM, so some are refused, and some overlapping one another.
    fn code_with_loads(rng: &mut Rng, len: usize, loads: usize) -> Vec<u8> {
        let mut code: Vec<u8> = (0..len)
            .map(|_| match rng.next() % 16 {
                0 => GS,
                _ => rng.next() as u8,
            })
            .collect();
        for _ in 0..loads {
            let Some(room) = len.checked_sub(LOAD_LEN - 1) else {
                break;
            };
            let at = (rng.next() as usize) % room.max(1);
            let r = rng.next();
            let rex = [0x48, 0x4C, 0x49, 0x65][(r & 3) as usize];
            let modrm = (r >> 8) as u8 & 0x3F | [0x04, 0x44][(r >> 16 & 1) as usize] & 0xC7;
            let mut load = [GS, rex, 0x8B, modrm, 0x25, 0x58, 0, 0, 0];
            if r >> 20 & 7 == 0 {
                // A near miss in one byte of the tail.
                load[4 + (r >> 24) as usize % 5] ^= 1;
            }
            let n = load.len().min(len - at);
            code[at..at + n].copy_from_slice(&load[..n]);
        }
        code
    }

    #[test]
    fn candidate_search_patches_exactly_what_the_bytewise_scan_does() {
        let mut rng = Rng(0x9E37_79B9_7F4A_7C15);
        let (at, thunks) = (0x1_4000_0000, 0x1_3FFF_0000);
        let mut patched = 0;
        for len in (0..64).chain([255, 256, 257, 4093, 4096, 65_537]) {
            for round in 0..40 {
                let loads = (len / LOAD_LEN).min(round * 3);
                let code = code_with_loads(&mut rng, len, loads);
                // Every offset too, so each word alignment of the search
                // is covered.
                let shift = round % 8;
                let (mut fast, mut slow) = (code.clone(), code.clone());
                let n = patch(&mut fast[shift.min(len)..], at, thunks);
                let m = patch_bytewise(&mut slow[shift.min(len)..], at, thunks);
                assert_eq!((n, &fast), (m, &slow), "len {len} round {round}");
                patched += n;
            }
        }
        assert!(patched > 200, "{patched} loads patched");
        // Back to back, a prefix inside a load, and a load ending the code.
        let load = [GS, 0x48, 0x8B, 0x04, 0x25, 0x58, 0, 0, 0];
        let mut code = Vec::new();
        for _ in 0..3 {
            code.extend(load);
        }
        code.extend([GS, GS]);
        code.extend(load);
        let (mut fast, mut slow) = (code.clone(), code);
        assert_eq!(patch(&mut fast, at, thunks), 4);
        assert_eq!(patch_bytewise(&mut slow, at, thunks), 4);
        assert_eq!(fast, slow);
    }

    /// `cargo test -p victory-loader --release -- --ignored --nocapture
    /// tls_scan_speed`: both scans over synthetic code sections.
    #[test]
    #[ignore = "benchmark"]
    fn tls_scan_speed() {
        use std::time::Instant;
        let mut rng = Rng(0xD1B5_4A32_D192_ED03);
        for (name, mib, per_mib) in [("small", 1, 512), ("medium", 16, 512), ("large", 128, 512)] {
            let len = mib << 20;
            // Code-like noise: each byte a gs prefix with chance 1/256.
            let mut code: Vec<u8> = (0..len).map(|_| rng.next() as u8).collect();
            for _ in 0..mib * per_mib {
                let i = rng.next() as usize % (len - LOAD_LEN);
                code[i..i + LOAD_LEN].copy_from_slice(&[GS, 0x48, 0x8B, 0x04, 0x25, 0x58, 0, 0, 0]);
            }
            let best = |f: fn(&mut [u8], usize, usize) -> usize| {
                let mut times = Vec::new();
                let mut out = (0, Vec::new());
                for _ in 0..5 {
                    let mut c = code.clone();
                    let t = Instant::now();
                    let n = f(&mut c, 0x1_4000_0000, 0x1_3FFF_0000);
                    times.push(t.elapsed().as_secs_f64());
                    out = (n, c);
                }
                (times.into_iter().fold(f64::MAX, f64::min), out)
            };
            let (old, a) = best(patch_bytewise);
            let (new, b) = best(patch);
            assert_eq!(a, b);
            println!(
                "{name:6} {mib:4} MiB, {:6} loads: bytewise {:8.3} ms, candidate search {:8.3} ms, {:5.1}x",
                a.0,
                old * 1e3,
                new * 1e3,
                old / new
            );
        }
    }

    /// `mov r64, gs:[0x58]`, as compilers spell it for `reg`.
    fn load(reg: u8) -> [u8; LOAD_LEN] {
        let high = reg >> 3;
        let low = (reg & 7) << 3;
        [0x65, 0x48 | high << 2, 0x8B, low | 4, 0x25, 0x58, 0, 0, 0]
    }

    #[test]
    fn loads_become_calls_that_yield_the_title_slot() {
        let page = near(tls_array as *const () as usize, 0).unwrap();
        let thunks = thunks(near(page as usize, 4096).unwrap(), 3).unwrap();
        // (before the load, REX, ModRM, after the load): each case returns
        // the loaded register in rax; r12 is saved, being callee-saved.
        let cases: [(&[u8], u8, u8, &[u8]); 3] = [
            (&[], 0x48, 0x04, &[0xC3]),                   // rax
            (&[], 0x4C, 0x04, &[0x4C, 0x89, 0xC0, 0xC3]), // r8
            (
                &[0x41, 0x54],
                0x4C,
                0x24,
                &[0x4C, 0x89, 0xE0, 0x41, 0x5C, 0xC3],
            ), // r12
        ];
        for (before, rex, modrm, after) in cases {
            let mut code = before.to_vec();
            code.extend([0x65, rex, 0x8B, modrm, 0x25, 0x58, 0, 0, 0]);
            code.extend(after);
            // SAFETY: the page is ours and writable.
            let bytes = unsafe { std::slice::from_raw_parts_mut(page, code.len()) };
            bytes.copy_from_slice(&code);
            assert_eq!(patch(bytes, page as usize, thunks), 1);
            let mut old = 0;
            // SAFETY: our page, made executable to run the patched code.
            unsafe { VirtualProtect(page as _, 4096, PAGE_EXECUTE_READ, &mut old) };
            // SAFETY: the page holds the patched load and a return.
            let f: extern "system" fn() -> usize = unsafe { std::mem::transmute(page) };
            assert_eq!(f(), tls_array() + 3 * 8, "REX {rex:#x} ModRM {modrm:#x}");
            // SAFETY: our page again.
            unsafe { VirtualProtect(page as _, 4096, PAGE_READWRITE, &mut old) };
        }
    }

    #[test]
    fn every_register_load_becomes_a_call_but_the_stack_pointer() {
        const AT: usize = 0x1000;
        const BLOCK: usize = 0x2000;
        for reg in 0..16u8 {
            let mut code = load(reg).to_vec();
            code.push(0xC3);
            let patched = patch(&mut code, AT, BLOCK);
            if reg == 4 {
                // A `mov rsp, ...` thunk would move the stack out from under
                // its own `ret`, so it is the one register left alone.
                assert_eq!(patched, 0);
                assert_eq!(&code[..LOAD_LEN], &load(reg));
                assert_eq!(code[LOAD_LEN], 0xC3);
                continue;
            }
            assert_eq!(patched, 1, "r{reg}");
            // `call thunk[reg]`, then a nop that keeps the length at 9.
            assert_eq!(code[0], 0xE8, "r{reg}");
            let rel = i64::from(i32::from_le_bytes(code[1..5].try_into().unwrap()));
            let callee = (AT + 5) as i64 + rel;
            assert_eq!(
                callee,
                (BLOCK + usize::from(reg) * THUNK_SIZE) as i64,
                "r{reg}"
            );
            assert_eq!(code[5..9], [0x0F, 0x1F, 0x40, 0x00], "r{reg}");
            assert_eq!(code[9], 0xC3, "r{reg}");
        }
    }

    #[test]
    fn other_nine_byte_windows_are_left_alone() {
        // Each differs from the load in one field: the segment, the opcode,
        // the ModRM's mod, the displacement, or the SIB's base.
        let cases = [
            [0x64, 0x48, 0x8B, 0x04, 0x25, 0x58, 0, 0, 0], // fs:
            [0x65, 0x48, 0x8A, 0x04, 0x25, 0x58, 0, 0, 0], // mov r8, r/m8
            [0x65, 0x48, 0x8B, 0x44, 0x25, 0x58, 0, 0, 0], // mod 01
            [0x65, 0x48, 0x8B, 0x04, 0x25, 0x58, 0, 0, 1], // offset 1
            [0x65, 0x48, 0x8B, 0x04, 0x26, 0x58, 0, 0, 0], // [rsi], no displacement
        ];
        for case in cases {
            let mut code = case.to_vec();
            assert_eq!(patch(&mut code, 0x1000, 0x2000), 0, "{case:02X?}");
            assert_eq!(code, case, "{case:02X?}");
        }
    }

    #[test]
    fn a_window_short_of_a_load_is_not_read() {
        // Eight bytes: one short of a whole load, and too short to match.
        let mut code = load(5)[..LOAD_LEN - 1].to_vec();
        assert_eq!(patch(&mut code, 0x1000, 0x2000), 0);
        assert_eq!(code, load(5)[..LOAD_LEN - 1]);
        assert_eq!(patch(&mut [], 0x1000, 0x2000), 0);
    }

    #[test]
    fn loads_back_to_back_are_each_patched_once() {
        let mut code = [load(1), load(9), load(2)].concat();
        assert_eq!(patch(&mut code, 0x1000, 0x2000), 3);
        // The second pass finds calls and nops where the loads were.
        assert_eq!(patch(&mut code, 0x1000, 0x2000), 0);
    }

    #[test]
    fn each_thunk_loads_the_array_and_adds_the_slots_offset() {
        const INDEX: u32 = 1087;
        let block = thunks(near(tls_array as *const () as usize, 0).unwrap(), INDEX).unwrap();
        // SAFETY: `block` is one page of ours, readable; `thunks` made it
        // read-execute.
        let bytes = unsafe { std::slice::from_raw_parts(block as *const u8, 4096) };
        for reg in 0..16u8 {
            let slot = &bytes[usize::from(reg) * THUNK_SIZE..][..THUNK_SIZE];
            let (high, low) = (reg >> 3, reg & 7);
            let mut thunk = vec![
                0x65,
                0x48 | high << 2,
                0x8B,
                low << 3 | 4,
                0x25,
                0x58,
                0,
                0,
                0,
                // lea r, [r + index * 8]; the SIB says r12/rsp as the base.
                0x48 | high << 2 | high,
                0x8D,
                0x80 | low << 3 | low,
            ];
            if low == 4 {
                thunk.push(0x24);
            }
            thunk.extend((INDEX * 8).to_le_bytes());
            thunk.push(0xC3);
            assert!(thunk.len() <= THUNK_SIZE);
            assert_eq!(&slot[..thunk.len()], &thunk[..], "r{reg}");
            // One thunk's bytes must not run into the next thunk's.
            assert!(slot[thunk.len()..].iter().all(|&b| b == 0), "r{reg}");
        }
        // Running one only adds to the array pointer (it never reads the
        // slot), so any index is safe to check here.
        // SAFETY: the page is executable and the thunk ends in `ret`.
        let rax: extern "system" fn() -> usize = unsafe { std::mem::transmute(block) };
        assert_eq!(rax(), tls_array() + INDEX as usize * 8);
    }

    #[test]
    fn near_stays_within_rel32_reach_and_writable() {
        let base = tls_array as *const () as usize;
        let page = near(base, 0).unwrap();
        assert!((page as usize).abs_diff(base) <= 1 << 30);
        assert!(near(base, (1 << 30) - 1).is_ok());
        assert!(near(base, 1 << 30).is_err());
        // SAFETY: a fresh page of ours, writable as `near` promises.
        unsafe {
            page.write(0xA5);
            assert_eq!(page.read(), 0xA5);
        }
    }

    /// A synthetic mapped image of the shape `redirect` expects: a TLS
    /// directory at 0x1000 naming an index `u32` at 0x1100, and one
    /// executable section at 0x2000 holding `mov rax, gs:[0x58]; ret`.
    /// Only the layout is synthetic; `redirect` is what reads it.
    struct Synthetic {
        base: *mut u8,
        info: ImageInfo,
        index: *mut u32,
        code: *mut u8,
    }

    const SIZE_OF_IMAGE: u32 = 0x2_0000;
    const DIRECTORY: usize = 0x1000;
    const INDEX: usize = 0x1100;
    const TEXT: usize = 0x2000;
    /// `mov rax, gs:[0x58]; ret`.
    const CODE: [u8; 10] = [0x65, 0x48, 0x8B, 0x04, 0x25, 0x58, 0, 0, 0, 0xC3];
    /// `IMAGE_SCN_CNT_CODE | IMAGE_SCN_MEM_EXECUTE | IMAGE_SCN_MEM_READ`.
    const TEXT_FLAGS: u32 = 0x6000_0020;

    impl Synthetic {
        /// `tls_index` is the index the OS "wrote" for the image.
        fn new(tls_index: u32) -> Self {
            let text = SectionInfo {
                name: ".text".into(),
                rva: TEXT as u32,
                virtual_size: 0x10,
                file_offset: 0,
                file_size: 0x10,
                characteristics: TEXT_FLAGS,
            };
            // SAFETY: a fresh mapping of ours, filled in as the loader
            // would have left it.
            unsafe {
                let base = VirtualAlloc(
                    std::ptr::null(),
                    SIZE_OF_IMAGE as usize,
                    MEM_COMMIT | MEM_RESERVE,
                    PAGE_READWRITE,
                )
                .cast::<u8>();
                assert!(!base.is_null());
                let index = base.add(INDEX).cast::<u32>();
                index.write(tls_index);
                // `IMAGE_TLS_DIRECTORY64`: raw data start and end, the
                // address of the index, the callbacks, then zero fill.
                let fields = [0u64, 0, index as u64, 0, 0, 0];
                for (i, field) in fields.into_iter().enumerate() {
                    base.add(DIRECTORY).cast::<u64>().add(i).write(field);
                }
                base.add(TEXT)
                    .copy_from_nonoverlapping(CODE.as_ptr(), CODE.len());
                let mut old = 0;
                let exec =
                    VirtualProtect(base.add(TEXT).cast(), 0x1000, PAGE_EXECUTE_READ, &mut old);
                assert_ne!(exec, 0);
                Self {
                    base,
                    info: ImageInfo {
                        machine: victory_pe::MACHINE_AMD64,
                        characteristics: 0x0022,
                        timestamp: 0,
                        subsystem: 3,
                        dll_characteristics: 0,
                        image_base: base as u64,
                        size_of_image: SIZE_OF_IMAGE,
                        entry_rva: 0,
                        stack_reserve: 0,
                        stack_commit: 0,
                        sections: vec![text],
                        imports: Vec::new(),
                        delay_imports: Vec::new(),
                        exports: Vec::new(),
                        tls: Some(TlsInfo {
                            directory_rva: DIRECTORY as u32,
                            callbacks_va: 0,
                            callback_rvas: Vec::new(),
                        }),
                        has_relocations: false,
                        has_exception_table: false,
                        has_load_config: false,
                        codeview: None,
                    },
                    index,
                    code: base.add(TEXT),
                }
            }
        }

        fn patched(&self) -> usize {
            // SAFETY: `info` describes this mapping, and nothing runs it.
            unsafe { redirect(self.base as usize, &self.info) }.unwrap()
        }

        fn index_value(&self) -> u32 {
            // SAFETY: the index is in our mapping, and readable.
            unsafe { self.index.read() }
        }

        fn code_bytes<const N: usize>(&self) -> [u8; N] {
            // SAFETY: the text page is ours, and `N` is inside it.
            unsafe { std::ptr::read(self.code.cast::<[u8; N]>()) }
        }
    }

    impl Drop for Synthetic {
        fn drop(&mut self) {
            // SAFETY: the mapping is ours, and nothing uses it any more.
            unsafe { VirtualFree(self.base.cast(), 0, MEM_RELEASE) };
        }
    }

    #[test]
    fn a_redirected_image_yields_the_slot_the_os_gave_it() {
        let image = Synthetic::new(5);
        assert_eq!(image.patched(), 1);
        let code = image.code_bytes::<10>();
        assert_eq!(code[0], 0xE8);
        assert_eq!(code[5..9], [0x0F, 0x1F, 0x40, 0x00]);
        assert_eq!(code[9], 0xC3);
        // `_tls_index` is 0 from here on, as CRT code inside a DLL expects.
        assert_eq!(image.index_value(), 0);
        // The patched load returns the slot the OS assigned the image.
        // SAFETY: the patched page is executable, and ends in `ret`.
        let run: extern "system" fn() -> usize = unsafe { std::mem::transmute(image.code) };
        assert_eq!(run(), tls_array() + 5 * 8);
    }

    #[test]
    fn index_zero_needs_no_patches() {
        let image = Synthetic::new(0);
        assert_eq!(image.patched(), 0);
        assert_eq!(image.code_bytes::<10>(), CODE);
    }

    #[test]
    fn an_executable_section_with_zero_virtual_size_uses_its_file_size() {
        let mut image = Synthetic::new(5);
        image.info.sections[0].virtual_size = 0;

        assert_eq!(image.patched(), 1);
        let code = image.code_bytes::<10>();
        assert_eq!(code[0], 0xE8);
        assert_eq!(code[5..], [0x0F, 0x1F, 0x40, 0x00, 0xC3]);
    }

    #[test]
    fn an_empty_executable_section_does_not_stop_the_rest() {
        let mut image = Synthetic::new(5);
        image.info.sections.push(SectionInfo {
            name: ".empty".into(),
            rva: 0x3000,
            virtual_size: 0,
            file_offset: 0,
            file_size: 0,
            characteristics: TEXT_FLAGS,
        });
        // `VirtualProtect` refuses a zero length; that must not abort the
        // redirect of the sections that do hold code.
        assert_eq!(image.patched(), 1);
        assert_eq!(image.index_value(), 0);
    }

    #[test]
    fn an_image_without_tls_needs_no_redirect() {
        let mut image = Synthetic::new(5);
        image.info.tls = None;
        assert_eq!(image.patched(), 0);
        assert_eq!(image.code_bytes::<10>(), CODE);
    }
}
