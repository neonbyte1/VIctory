//! Copies of title memory in user mode (decision 0007, "Copies of title
//! memory"). A range checked when it was recorded may be unmapped by the
//! time its list is submitted, so reading title memory can fault. Having
//! the kernel copy it (`ReadProcessMemory`) avoids that but costs a system
//! call, many times the copy itself for constant buffers, which a frame
//! fills thousands of.
//!
//! [`copy`] reads the memory directly, in one small routine of ours; a
//! vectored handler turns a fault on one of the routine's own instructions
//! into a `false` return. Faults anywhere else are not touched. The routine
//! is a leaf that never moves the stack pointer and changes volatile
//! registers only, so going on at its failing exit is a plain return.

use windows_sys::Win32::Foundation::{
    EXCEPTION_ACCESS_VIOLATION, EXCEPTION_GUARD_PAGE, EXCEPTION_IN_PAGE_ERROR, NTSTATUS,
};

// `victory_title_copy(dst: rcx, src: rdx, len: r8) -> eax`: 64 bytes at a
// time, then 16, then the last 16 (overlapping what was copied). Ascending,
// with whole-register stores: the destinations are mostly write-combined
// upload memory, where `rep movsb` is slow on some CPUs.
std::arch::global_asm!(
    ".globl victory_title_copy",
    ".globl victory_title_copy_begin",
    ".globl victory_title_copy_end",
    ".globl victory_title_copy_failed",
    ".balign 64",
    "victory_title_copy:",
    "lea r9, [rcx + r8 - 16]",
    "lea r10, [rdx + r8 - 16]",
    "victory_title_copy_begin:",
    "cmp r8, 16",
    "jb 5f",
    "sub r8, 64",
    "jb 3f",
    ".balign 16",
    "2:",
    "movups xmm0, [rdx]",
    "movups xmm1, [rdx + 16]",
    "movups xmm2, [rdx + 32]",
    "movups xmm3, [rdx + 48]",
    "movups [rcx], xmm0",
    "movups [rcx + 16], xmm1",
    "movups [rcx + 32], xmm2",
    "movups [rcx + 48], xmm3",
    "add rdx, 64",
    "add rcx, 64",
    "sub r8, 64",
    "jae 2b",
    "3:",
    "add r8, 64",
    "jz 7f",
    "4:",
    "cmp r8, 16",
    "jb 6f",
    "movups xmm0, [rdx]",
    "movups [rcx], xmm0",
    "add rdx, 16",
    "add rcx, 16",
    "sub r8, 16",
    "jmp 4b",
    "5:",
    "test r8, r8",
    "jz 7f",
    "mov al, [rdx]",
    "mov [rcx], al",
    "inc rdx",
    "inc rcx",
    "dec r8",
    "jmp 5b",
    "6:",
    "test r8, r8",
    "jz 7f",
    "movups xmm0, [r10]",
    "movups [r9], xmm0",
    "victory_title_copy_end:",
    "7:",
    "mov eax, 1",
    "ret",
    "victory_title_copy_failed:",
    "xor eax, eax",
    "ret",
);

unsafe extern "win64" {
    fn victory_title_copy(dst: *mut u8, src: *const u8, len: usize) -> u32;
    static victory_title_copy_begin: u8;
    static victory_title_copy_end: u8;
    static victory_title_copy_failed: u8;
}

/// Copies `len` bytes from `src` to `dst`, which must not overlap; false if
/// a page of either is not accessible (the bytes before it may have been
/// copied).
///
/// # Safety
///
/// `dst` is `len` bytes the caller may write. `src` may be any address.
pub(crate) unsafe fn copy(dst: *mut u8, src: *const u8, len: usize) -> bool {
    install();
    // SAFETY: the caller's; a fault in the routine makes it return 0.
    unsafe { victory_title_copy(dst, src, len) != 0 }
}

/// Where to go on after exception `code` at `rip`: the routine's failing
/// exit for a memory fault of the routine, `None` for anything else.
fn recover(code: NTSTATUS, rip: usize) -> Option<usize> {
    if !matches!(
        code,
        EXCEPTION_ACCESS_VIOLATION | EXCEPTION_IN_PAGE_ERROR | EXCEPTION_GUARD_PAGE
    ) {
        return None;
    }
    let begin = &raw const victory_title_copy_begin as usize;
    let end = &raw const victory_title_copy_end as usize;
    (begin..end)
        .contains(&rip)
        .then_some(&raw const victory_title_copy_failed as usize)
}

/// Makes the page of `at`, whose guard our read just spent, a guard page
/// again: the kernel's copy leaves guards alone, and they are the title's.
///
/// Between the fault and this, a title thread that touches the page is not
/// told (the next touch is), and a copy on another thread reads the page's
/// bytes instead of zeros. A title thread that takes the guard off itself
/// in those microseconds finds it on again, unless it left another
/// protection, which is put back.
fn guard_again(at: usize) {
    use windows_sys::Win32::System::Memory::{
        MEM_COMMIT, MEMORY_BASIC_INFORMATION, PAGE_GUARD, VirtualProtect, VirtualQuery,
    };
    // SAFETY: plain data filled in by the query.
    let mut info: MEMORY_BASIC_INFORMATION = unsafe { std::mem::zeroed() };
    // SAFETY: a valid out pointer of the right size.
    let n = unsafe { VirtualQuery(at as *const _, &mut info, size_of_val(&info)) };
    if n == 0 || info.State != MEM_COMMIT {
        return;
    }
    let protect = |protection: u32| {
        let mut old = 0;
        // SAFETY: only the page's protection changes; fails if it is gone.
        (unsafe { VirtualProtect(at as *const _, 1, protection, &mut old) } != 0).then_some(old)
    };
    if let Some(old) = protect(info.Protect | PAGE_GUARD)
        && old & !PAGE_GUARD != info.Protect & !PAGE_GUARD
    {
        protect(old);
    }
}

/// Registers the handler, once, ahead of the crash reporter's.
fn install() {
    use windows_sys::Win32::System::Diagnostics::Debug::{
        AddVectoredExceptionHandler, EXCEPTION_CONTINUE_EXECUTION, EXCEPTION_CONTINUE_SEARCH,
        EXCEPTION_POINTERS,
    };
    unsafe extern "system" fn on_fault(info: *mut EXCEPTION_POINTERS) -> i32 {
        // SAFETY: the system passes the exception being dispatched.
        let (record, context) = unsafe { (&*(*info).ExceptionRecord, &mut *(*info).ContextRecord) };
        match recover(record.ExceptionCode, context.Rip as usize) {
            Some(exit) => {
                if record.ExceptionCode == EXCEPTION_GUARD_PAGE {
                    guard_again(record.ExceptionInformation[1]);
                }
                context.Rip = exit as u64;
                EXCEPTION_CONTINUE_EXECUTION
            }
            None => EXCEPTION_CONTINUE_SEARCH,
        }
    }
    static ONCE: std::sync::Once = std::sync::Once::new();
    // SAFETY: the handler only changes where our own routine goes on.
    ONCE.call_once(|| unsafe {
        AddVectoredExceptionHandler(1, Some(on_fault));
    });
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::hint::black_box;
    use windows_sys::Win32::System::Diagnostics::Debug::ReadProcessMemory;
    use windows_sys::Win32::System::Memory::{
        MEM_COMMIT, MEM_DECOMMIT, MEM_RELEASE, MEM_RESERVE, MEMORY_BASIC_INFORMATION,
        PAGE_EXECUTE_READ, PAGE_GUARD, PAGE_NOACCESS, PAGE_READONLY, PAGE_READWRITE, VirtualAlloc,
        VirtualFree, VirtualProtect, VirtualQuery,
    };
    use windows_sys::Win32::System::Threading::GetCurrentProcess;

    const PAGE: usize = 4096;

    fn kernel(dst: *mut u8, src: *const u8, len: usize) -> bool {
        let mut done = 0;
        // SAFETY: `dst` is ours; the kernel checks the source.
        unsafe {
            ReadProcessMemory(GetCurrentProcess(), src.cast(), dst.cast(), len, &mut done) != 0
        }
    }

    /// `n` committed pages filled with a pattern.
    fn pages(n: usize) -> *mut u8 {
        // SAFETY: a fresh allocation, filled within its size.
        unsafe {
            let at = VirtualAlloc(
                std::ptr::null(),
                n * PAGE,
                MEM_RESERVE | MEM_COMMIT,
                PAGE_READWRITE,
            )
            .cast::<u8>();
            assert!(!at.is_null());
            for i in 0..n * PAGE {
                at.add(i).write((i * 31 + i / 256 + 1) as u8);
            }
            at
        }
    }

    fn free(at: *mut u8) {
        // SAFETY: an allocation of `pages`.
        assert_ne!(unsafe { VirtualFree(at.cast(), 0, MEM_RELEASE) }, 0);
    }

    fn protect(at: *mut u8, page: usize, protection: u32) {
        let mut old = 0;
        // SAFETY: a page of ours.
        let ok = unsafe { VirtualProtect(at.add(page * PAGE).cast(), PAGE, protection, &mut old) };
        assert_ne!(ok, 0);
    }

    /// Every length and alignment: the bytes of the range and no others.
    #[test]
    fn copies_exactly_the_range() {
        let from: Vec<u8> = (0..1024u32).map(|i| (i * 7 + i / 256) as u8).collect();
        for len in (0..=330).chain([511, 512, 513, 640, 900]) {
            for (a, b) in [(0, 0), (1, 0), (0, 3), (5, 9), (15, 1), (16, 8)] {
                let mut to = [0xeeu8; 1024];
                // SAFETY: both ranges lie in their arrays.
                assert!(unsafe { copy(to.as_mut_ptr().add(b), from.as_ptr().add(a), len) });
                assert_eq!(to[b..b + len], from[a..a + len], "{len} bytes");
                assert!(to[..b].iter().chain(&to[b + len..]).all(|&x| x == 0xee));
            }
        }
    }

    /// A range up to an inaccessible page's edge reads no further, and one
    /// that reaches into the page fails as the kernel's copy does.
    #[test]
    fn agrees_with_the_kernel_at_inaccessible_pages() {
        for decommitted in [false, true] {
            let at = pages(3);
            if decommitted {
                // SAFETY: the middle page of ours.
                let ok = unsafe { VirtualFree(at.add(PAGE).cast(), PAGE, MEM_DECOMMIT) };
                assert_ne!(ok, 0);
            } else {
                protect(at, 1, PAGE_NOACCESS);
            }
            for len in [1usize, 4, 15, 16, 17, 63, 64, 65, 256, 1000] {
                // Ranges ending near the hole, crossing into it, and
                // starting near its end.
                let starts = [PAGE, 2 * PAGE]
                    .into_iter()
                    .flat_map(|edge| edge - len - 1..edge + 2);
                for start in starts {
                    let (mut ours, mut theirs) = (vec![0u8; len], vec![0u8; len]);
                    // SAFETY: an address in our reservation.
                    let src = unsafe { at.add(start) };
                    // SAFETY: `ours` has `len` bytes.
                    let whole = unsafe { copy(ours.as_mut_ptr(), src, len) };
                    let hole = PAGE..2 * PAGE;
                    assert_eq!(whole, start + len <= hole.start || start >= hole.end);
                    assert_eq!(whole, kernel(theirs.as_mut_ptr(), src, len));
                    if whole {
                        assert_eq!(ours, theirs);
                    }
                }
            }
            free(at);
        }
        // No address at all.
        let mut to = [0u8; 300];
        for len in [1, 16, 300] {
            // SAFETY: `to` has room; the source only has to fail.
            assert!(!unsafe { copy(to.as_mut_ptr(), std::ptr::without_provenance(8), len) });
        }
    }

    /// Read-only memory (the copies' own guards) is read, and a guard page
    /// and an unwritable destination fail without ending the process.
    #[test]
    fn protections_are_honoured() {
        let (from, to) = (pages(1), pages(1));
        protect(from, 0, PAGE_READONLY);
        // SAFETY: a page each.
        assert!(unsafe { copy(to, from, PAGE) });
        protect(to, 0, PAGE_READONLY);
        // SAFETY: the routine's faults return.
        assert!(!unsafe { copy(to, from, PAGE) });
        protect(to, 0, PAGE_READWRITE);
        protect(from, 0, PAGE_READWRITE | PAGE_GUARD);
        // SAFETY: as above.
        assert!(!unsafe { copy(to, from, PAGE) });
        // The guard stays, as the kernel's copy leaves it.
        // SAFETY: as above.
        assert!(!unsafe { copy(to, from, PAGE) });
        free(from);
        free(to);
    }

    /// The page's protection, `PAGE_GUARD` included.
    fn protection(at: *mut u8, page: usize) -> u32 {
        // SAFETY: plain data filled in by the query.
        let mut info: MEMORY_BASIC_INFORMATION = unsafe { std::mem::zeroed() };
        // SAFETY: a page of ours and a valid out pointer of the right size.
        let n = unsafe { VirtualQuery(at.add(page * PAGE).cast(), &mut info, size_of_val(&info)) };
        assert_ne!(n, 0);
        info.Protect
    }

    /// A guard page the copy reaches stays one, with its other protection,
    /// and the range gets the bytes a copy through the kernel gives it
    /// (zeros for the page), however it lies over the page.
    #[test]
    fn guard_pages_stay_guarded() {
        for other in [PAGE_READWRITE, PAGE_READONLY, PAGE_EXECUTE_READ] {
            for (start, len) in [
                (PAGE, 1),
                (PAGE + 2000, 1),
                (PAGE + 2000, 100),
                (2 * PAGE - 1, 1),
                (PAGE - 1, 2),
                (PAGE - 7, 16),
                (PAGE - 500, 1000),
                (2 * PAGE - 8, 64),
                (0, 3 * PAGE),
            ] {
                let at = pages(3);
                protect(at, 1, other | PAGE_GUARD);
                let mut got = vec![0xeeu8; len];
                // SAFETY: an address in our allocation.
                let src = unsafe { at.add(start) };
                // SAFETY: `got` has `len` bytes.
                assert!(!unsafe { copy(got.as_mut_ptr(), src, len) });
                assert_eq!(protection(at, 1), other | PAGE_GUARD, "{start} {len}");
                // The whole way, as a submission copies it.
                crate::host::copy_mapped(got.as_mut_ptr(), src, len, false);
                assert_eq!(protection(at, 1), other | PAGE_GUARD, "{start} {len}");
                for (i, &b) in got.iter().enumerate() {
                    let i = start + i;
                    let guarded = (PAGE..2 * PAGE).contains(&i);
                    let want = if guarded {
                        0
                    } else {
                        (i * 31 + i / 256 + 1) as u8
                    };
                    assert_eq!(b, want, "byte {i} of {start} {len}");
                }
                free(at);
            }
        }
    }

    /// Two threads reading the same guard page: neither ends the process,
    /// and the page is a guard page when both are done.
    #[test]
    fn two_copies_of_a_guard_page() {
        let at = pages(1) as usize;
        protect(at as *mut u8, 0, PAGE_READWRITE | PAGE_GUARD);
        std::thread::scope(|s| {
            for _ in 0..2 {
                s.spawn(|| {
                    let mut to = [0u8; 256];
                    for _ in 0..2000 {
                        // SAFETY: `to` has room.
                        black_box(unsafe { copy(to.as_mut_ptr(), (at + 64) as *const u8, 256) });
                    }
                });
            }
        });
        assert_eq!(protection(at as *mut u8, 0), PAGE_READWRITE | PAGE_GUARD);
        free(at as *mut u8);
    }

    /// The title uses a guard page as one (it arms it, and a touch is
    /// reported to it once) while copies reach it: its touches are reported
    /// but for those that meet a copy, and the page ends as a guard page.
    #[test]
    fn a_guard_page_the_title_touches_meanwhile() {
        use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
        use windows_sys::Win32::System::Diagnostics::Debug::{
            AddVectoredExceptionHandler, EXCEPTION_POINTERS, RemoveVectoredExceptionHandler,
        };
        static PAGE_AT: AtomicUsize = AtomicUsize::new(0);
        static REPORTED: AtomicUsize = AtomicUsize::new(0);
        unsafe extern "system" fn title(info: *mut EXCEPTION_POINTERS) -> i32 {
            // SAFETY: the system passes the exception being dispatched.
            let (r, c) = unsafe { (&*(*info).ExceptionRecord, &*(*info).ContextRecord) };
            let ours = r.ExceptionCode == EXCEPTION_GUARD_PAGE
                && r.ExceptionInformation[1] / PAGE * PAGE == PAGE_AT.load(Ordering::Relaxed)
                && recover(r.ExceptionCode, c.Rip as usize).is_none();
            if ours {
                REPORTED.fetch_add(1, Ordering::Relaxed);
            }
            if ours { -1 } else { 0 }
        }
        let at = pages(1) as usize;
        PAGE_AT.store(at, Ordering::Relaxed);
        // SAFETY: removed below.
        let handler = unsafe { AddVectoredExceptionHandler(1, Some(title)) };
        let stop = AtomicBool::new(false);
        let mut touches = 0;
        std::thread::scope(|s| {
            s.spawn(|| {
                let mut to = [0u8; 64];
                while !stop.load(Ordering::Relaxed) {
                    // SAFETY: `to` has room.
                    black_box(unsafe { copy(to.as_mut_ptr(), (at + 64) as *const u8, 64) });
                    for _ in 0..2000 {
                        std::hint::spin_loop();
                    }
                }
            });
            let until = std::time::Instant::now() + std::time::Duration::from_millis(300);
            while std::time::Instant::now() < until {
                protect(at as *mut u8, 0, PAGE_READWRITE | PAGE_GUARD);
                // SAFETY: a committed page; its guard goes to the handler.
                black_box(unsafe { (at as *const u8).read_volatile() });
                touches += 1;
            }
            stop.store(true, Ordering::Relaxed);
        });
        // SAFETY: the handler added above.
        unsafe { RemoveVectoredExceptionHandler(handler) };
        let reported = REPORTED.load(Ordering::Relaxed);
        println!("{reported} of {touches} touches reported");
        assert!(reported > 0);
        protect(at as *mut u8, 0, PAGE_READWRITE | PAGE_GUARD);
        let mut to = [0u8; 64];
        // SAFETY: `to` has room.
        assert!(!unsafe { copy(to.as_mut_ptr(), at as *const u8, 64) });
        assert_eq!(protection(at as *mut u8, 0), PAGE_READWRITE | PAGE_GUARD);
        free(at as *mut u8);
    }

    /// Memory another thread decommits and commits during the copies: each
    /// succeeds or fails, none ends the process.
    #[test]
    fn memory_unmapped_meanwhile_does_not_fault() {
        use std::sync::atomic::{AtomicBool, Ordering};
        let at = pages(4) as usize;
        let stop = AtomicBool::new(false);
        let mut copies = 0u32;
        std::thread::scope(|s| {
            s.spawn(|| {
                while !stop.load(Ordering::Relaxed) {
                    // SAFETY: the last two pages of the reservation.
                    unsafe {
                        let hole = (at + 2 * PAGE) as *mut _;
                        VirtualFree(hole, 2 * PAGE, MEM_DECOMMIT);
                        std::thread::yield_now();
                        VirtualAlloc(hole, 2 * PAGE, MEM_COMMIT, PAGE_READWRITE);
                        std::thread::yield_now();
                    }
                }
            });
            let mut to = vec![0u8; 4 * PAGE];
            let until = std::time::Instant::now() + std::time::Duration::from_millis(300);
            while std::time::Instant::now() < until {
                // SAFETY: `to` has room.
                black_box(unsafe { copy(to.as_mut_ptr(), at as *const u8, 4 * PAGE) });
                copies += 1;
            }
            stop.store(true, Ordering::Relaxed);
        });
        assert!(copies > 0);
        free(at as *mut u8);
    }

    /// Only memory faults of the routine itself are taken.
    #[test]
    fn other_faults_are_left_alone() {
        let begin = &raw const victory_title_copy_begin as usize;
        let end = &raw const victory_title_copy_end as usize;
        let exit = &raw const victory_title_copy_failed as usize;
        assert!(begin < end && end < exit);
        assert_eq!(recover(EXCEPTION_ACCESS_VIOLATION, begin), Some(exit));
        assert_eq!(recover(EXCEPTION_IN_PAGE_ERROR, end - 1), Some(exit));
        assert_eq!(recover(EXCEPTION_GUARD_PAGE, begin), Some(exit));
        assert_eq!(recover(EXCEPTION_ACCESS_VIOLATION, begin - 1), None);
        assert_eq!(recover(EXCEPTION_ACCESS_VIOLATION, end), None);
        const ILLEGAL_INSTRUCTION: NTSTATUS = 0xc000_001d_u32 as NTSTATUS;
        assert_eq!(recover(ILLEGAL_INSTRUCTION, begin), None);
    }

    /// Per copy, through the kernel and in user mode. Run with
    /// `cargo test --release -p victory-gpu -- --ignored --nocapture title_copy_bench`.
    #[test]
    #[ignore = "benchmark"]
    fn title_copy_bench() {
        const SOURCE: usize = 64 << 20;
        let src = pages(SOURCE / PAGE);
        let run = |dst: *mut u8, room: usize, what: &str| {
            for shift in [6, 8, 10, 12, 14, 16, 20, 22] {
                let len = 1usize << shift;
                let n = ((256 << 20) / len).min(100_000);
                // Ranges spread over the source, filling the destination
                // front to back, as a list's scratch memory is.
                let measure = |f: &dyn Fn(*mut u8, *const u8) -> bool| {
                    let mut best = f64::MAX;
                    for _ in 0..5 {
                        let started = std::time::Instant::now();
                        for i in 0..n {
                            let from = ((i * len.max(256) * 7) % (SOURCE - len + 1)) & !15;
                            let to = ((i * len.max(256)) % (room - len + 1)) & !255;
                            assert!(f(dst.wrapping_add(to), src.wrapping_add(from)));
                        }
                        best = best.min(started.elapsed().as_nanos() as f64 / n as f64);
                    }
                    best
                };
                let before = measure(&|d, s| kernel(d, black_box(s), len));
                // SAFETY: both ranges are inside their buffers.
                let after = measure(&|d, s| unsafe { copy(d, black_box(s), len) });
                println!(
                    "{what}, {len:>8} B: kernel {before:>10.1} ns, user {after:>10.1} ns ({:.1}x)",
                    before / after
                );
            }
        };
        let room = 8 << 20;
        let heap = pages(room / PAGE);
        run(heap, room, "to heap memory");
        if let Some((_buffer, data)) =
            crate::host::host().and_then(|h| h.upload_buffer(room as u64))
        {
            run(data, room, "to an upload buffer");
        }
        // Copies that fail: what an unmapped page costs.
        // SAFETY: a page of ours.
        unsafe { VirtualFree(src.cast(), PAGE, MEM_DECOMMIT) };
        for (what, user) in [("kernel", false), ("user", true)] {
            let started = std::time::Instant::now();
            for _ in 0..10_000 {
                let ok = match user {
                    // SAFETY: `heap` has room.
                    true => unsafe { copy(heap, src, 256) },
                    false => kernel(heap, src, 256),
                };
                assert!(!ok);
            }
            let each = started.elapsed().as_nanos() / 10_000;
            println!("a failing copy, {what}: {each} ns");
        }
        free(heap);
        free(src);
    }
}
