//! Walks a thread's stack into return addresses by the unwind data of each
//! image, for the crash report and the stack dump. It does not name the
//! frames (`crash::locate` does) or suspend other threads (`stacks` does).
//! It stops where `rsp` lies outside the thread's committed stack, and where
//! `RtlVirtualUnwind` faults reading what a smashed frame points at: that
//! call is a guarded call, which the vectored handler can end on such a fault
//! ([`recover_unwind_fault`] says when), so the walk ends there instead of
//! the process.

use std::cell::Cell;
use std::ffi::c_void;
use std::ops::Range;
use std::ptr::null_mut;
use std::sync::OnceLock;
use windows_sys::Win32::Foundation::{
    EXCEPTION_ACCESS_VIOLATION, STATUS_GUARD_PAGE_VIOLATION, STATUS_IN_PAGE_ERROR,
};
use windows_sys::Win32::System::Diagnostics::Debug::{
    CONTEXT, EXCEPTION_POINTERS, IMAGE_RUNTIME_FUNCTION_ENTRY, RtlLookupFunctionEntry,
    RtlVirtualUnwind, UNW_FLAG_NHANDLER,
};
use windows_sys::Win32::System::Kernel::NT_TIB;
use windows_sys::Win32::System::LibraryLoader::GetModuleHandleW;
use windows_sys::Win32::System::Memory::{
    MEMORY_BASIC_INFORMATION, PAGE_GUARD, VirtualProtect, VirtualQuery,
};
use windows_sys::core::w;

/// The most frames a walk records, the starting instruction's included.
pub(crate) const MAX_FRAMES: usize = 40;

/// The committed part of the stack the NT_TIB at `tib` describes, from its
/// stack limit up to its stack base. Below it lie the thread's guard pages
/// and pages reserved but not committed. A reserved page faults when any
/// thread reads it; a guard page faults when another thread reads it.
///
/// # Safety
/// `tib` points at a mapped NT_TIB.
pub(crate) unsafe fn tib_stack(tib: *const NT_TIB) -> Range<usize> {
    // SAFETY: the caller's; the fields are read through the raw pointer.
    let (base, limit) = unsafe { ((*tib).StackBase, (*tib).StackLimit) };
    limit.addr()..base.addr()
}

/// This thread's committed stack, read from its own TEB: no system call, so
/// it cannot fail.
fn current_thread_stack() -> Range<usize> {
    let tib: *const NT_TIB;
    // SAFETY: on x64 Windows `gs` holds this thread's TEB, whose NT_TIB keeps
    // its own address in `Self` (how winnt.h's `NtCurrentTeb` reads it); the
    // read touches no stack and no flags.
    unsafe {
        std::arch::asm!(
            "mov {}, gs:[{}]",
            out(reg) tib,
            const std::mem::offset_of!(NT_TIB, Self_),
            options(nostack, readonly, preserves_flags),
        );
    }
    // SAFETY: this thread's own TEB, mapped while it runs.
    unsafe { tib_stack(tib) }
}

/// Return addresses of the faulting thread's frames, from the faulting
/// instruction up, by the unwind data of each image (the Windows loader
/// registered it for title images too), and why the walk ended early if it
/// did. Code without unwind data (our thunks, leaf functions) is taken to
/// have its return address at `rsp`. The stack is its committed part, so
/// `rsp` in a reserved page ends the walk.
pub(crate) fn unwind(context: CONTEXT) -> (Vec<u64>, Result<(), UnwindFault>) {
    let stack = current_thread_stack();
    let mut frames = Vec::with_capacity(MAX_FRAMES);
    let walked = unwind_into(context, stack.start, stack.end, &mut frames);
    (frames, walked)
}

/// A `CONTEXT` at the 16-byte alignment the unwind and capture functions need.
#[repr(C, align(16))]
pub(crate) struct Aligned16(pub(crate) CONTEXT);

/// [`unwind`] for any thread of the process, its stack being `low..high`;
/// pushes no more than `MAX_FRAMES`, so it does not allocate when `frames`
/// has that capacity (another thread may be suspended holding the heap).
/// Returns the fault that ended the walk, if `RtlVirtualUnwind` faulted;
/// `frames` then holds the frames before it.
///
/// Any thread may call it, inside the vectored handler or the unhandled-
/// exception filter too; it takes no lock of ours.
pub(crate) fn unwind_into(
    context: CONTEXT,
    low: usize,
    high: usize,
    frames: &mut Vec<u64>,
) -> Result<(), UnwindFault> {
    // `RtlVirtualUnwind` writes the context with 16-byte aligned stores, and
    // windows-sys declares `CONTEXT` without that alignment: a by-value copy on
    // the stack can land 8 bytes off and fault (seen in a debug build).
    let mut aligned = Aligned16(context);
    let context = &mut aligned.0;
    frames.push(context.Rip);
    while frames.len() < MAX_FRAMES {
        // Stop where reading the stack would fault inside this handler.
        let rsp = context.Rsp as usize;
        if rsp < low || rsp + 8 > high {
            break;
        }
        let mut image = 0;
        // SAFETY: looks up the function table entry covering `Rip`, if any.
        let entry = unsafe { RtlLookupFunctionEntry(context.Rip, &mut image, null_mut()) };
        if entry.is_null() {
            // SAFETY: `rsp` is inside this thread's stack.
            context.Rip = unsafe { (rsp as *const u64).read() };
            context.Rsp += 8;
        } else {
            // SAFETY: `entry` came from `RtlLookupFunctionEntry` for `Rip`
            // with `image`; the context is our copy.
            unsafe { virtual_unwind(image, entry, context) }?;
        }
        if context.Rip == 0 {
            break;
        }
        frames.push(context.Rip);
    }
    Ok(())
}

/// The faults a guarded call recovers: what a read of unmapped, no-access,
/// guard or unreadable paged memory raises. A stack overflow is not one: it
/// is the guard page of this thread's own stack, which the kernel grows
/// without raising anything, until the stack is exhausted. The statuses
/// `RtlVirtualUnwind` raises itself are not memory faults either, so
/// [`recover_unwind_fault`] declines them.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum MemoryFault {
    /// `EXCEPTION_ACCESS_VIOLATION`.
    AccessViolation,
    /// `STATUS_GUARD_PAGE_VIOLATION`.
    GuardPage,
    /// `STATUS_IN_PAGE_ERROR`.
    InPage,
}

impl TryFrom<i32> for MemoryFault {
    type Error = i32;

    /// The memory fault an exception code names, or the code back.
    fn try_from(code: i32) -> Result<Self, i32> {
        match code {
            EXCEPTION_ACCESS_VIOLATION => Ok(Self::AccessViolation),
            STATUS_GUARD_PAGE_VIOLATION => Ok(Self::GuardPage),
            STATUS_IN_PAGE_ERROR => Ok(Self::InPage),
            other => Err(other),
        }
    }
}

/// A memory fault inside `RtlVirtualUnwind`, which ended a walk.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct UnwindFault {
    kind: MemoryFault,
    /// The address the faulting access named.
    address: usize,
}

impl std::fmt::Display for UnwindFault {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let what = match self.kind {
            MemoryFault::AccessViolation => "access violation",
            MemoryFault::GuardPage => "guard-page violation",
            MemoryFault::InPage => "in-page error",
        };
        write!(f, "{what} at {:#x}", self.address)
    }
}

/// A call made through `victory_guarded_call`: the function and its eight
/// arguments, then what the trampoline and the handler write. Shared with
/// the assembly, which uses the offsets the `const` assertions below pin.
#[repr(C)]
struct GuardedCall {
    /// An `extern "system"` function of up to eight integer arguments.
    function: usize,
    arguments: [usize; 8],
    /// `rsp` after the trampoline's prolog, where a recovery resumes.
    resume_rsp: usize,
    /// The fault that ended the call, written by the handler on a recovery.
    fault: Option<UnwindFault>,
}

const _: () = {
    assert!(std::mem::offset_of!(GuardedCall, function) == 0);
    assert!(std::mem::offset_of!(GuardedCall, arguments) == 8);
    assert!(std::mem::offset_of!(GuardedCall, resume_rsp) == 72);
};

/// The address range of ntdll, whose code `RtlVirtualUnwind` runs, set by
/// [`locate_unwinder`]; [`recover_unwind_fault`] recovers only faults raised
/// inside it.
static UNWINDER_IMAGE: OnceLock<Range<usize>> = OnceLock::new();

/// Finds the image `RtlVirtualUnwind` runs in, for [`recover_unwind_fault`].
/// Called once, outside any handler, before the handler is installed (it
/// takes the loader's lock); until it has run, no fault is recovered.
pub(crate) fn locate_unwinder() {
    // SAFETY: looks up a module every process has loaded; no reference taken.
    let base = unsafe { GetModuleHandleW(w!("ntdll.dll")) }.cast::<u8>();
    if base.is_null() {
        return;
    }
    // SAFETY: a loaded image's headers are mapped readable at its base; the
    // optional header lies inside the first page.
    let headers = unsafe { std::slice::from_raw_parts(base, 0x1000) };
    let size = victory_pe::layout::optional_header_offset(headers)
        .and_then(|opt| opt.checked_add(victory_pe::layout::OPT_SIZE_OF_IMAGE))
        .and_then(|at| victory_pe::layout::read_u32(headers, at));
    if let Some(size) = size.and_then(|size| usize::try_from(size).ok()) {
        UNWINDER_IMAGE.get_or_init(|| base.addr()..base.addr() + size);
    }
}

// `victory_guarded_call(call: *mut GuardedCall /* rcx */) -> u32`: saves every
// nonvolatile register (rbx, rbp, rdi, rsi, r12-r15, xmm6-xmm15), MXCSR and
// the x87 control word, stores `rsp` in `call.resume_rsp`, calls
// `call.function` with `call.arguments` (four in registers, four on the stack
// above the 32-byte home area), and returns 1. A recovery resumes at
// `victory_guarded_call_recovered` with `rsp` = `call.resume_rsp`: it clears
// the direction flag, reloads MXCSR and the x87 control word the abandoned
// call may have left changed, and returns 0 through the same epilog, which
// restores the rest. Frame, from `rsp`: home area 0..32, arguments 5-8
// 32..64, xmm6-xmm15 64..224, MXCSR 224, x87 control word 228; 232 bytes
// keep `rsp` 16-byte aligned after the eight pushes.
std::arch::global_asm!(
    ".text",
    ".p2align 4",
    ".globl victory_guarded_call",
    "victory_guarded_call:",
    ".seh_proc victory_guarded_call",
    "    push rbx",
    ".seh_pushreg rbx",
    "    push rbp",
    ".seh_pushreg rbp",
    "    push rdi",
    ".seh_pushreg rdi",
    "    push rsi",
    ".seh_pushreg rsi",
    "    push r12",
    ".seh_pushreg r12",
    "    push r13",
    ".seh_pushreg r13",
    "    push r14",
    ".seh_pushreg r14",
    "    push r15",
    ".seh_pushreg r15",
    "    sub rsp, 232",
    ".seh_stackalloc 232",
    "    movaps xmmword ptr [rsp + 64], xmm6",
    ".seh_savexmm xmm6, 64",
    "    movaps xmmword ptr [rsp + 80], xmm7",
    ".seh_savexmm xmm7, 80",
    "    movaps xmmword ptr [rsp + 96], xmm8",
    ".seh_savexmm xmm8, 96",
    "    movaps xmmword ptr [rsp + 112], xmm9",
    ".seh_savexmm xmm9, 112",
    "    movaps xmmword ptr [rsp + 128], xmm10",
    ".seh_savexmm xmm10, 128",
    "    movaps xmmword ptr [rsp + 144], xmm11",
    ".seh_savexmm xmm11, 144",
    "    movaps xmmword ptr [rsp + 160], xmm12",
    ".seh_savexmm xmm12, 160",
    "    movaps xmmword ptr [rsp + 176], xmm13",
    ".seh_savexmm xmm13, 176",
    "    movaps xmmword ptr [rsp + 192], xmm14",
    ".seh_savexmm xmm14, 192",
    "    movaps xmmword ptr [rsp + 208], xmm15",
    ".seh_savexmm xmm15, 208",
    ".seh_endprologue",
    "    stmxcsr dword ptr [rsp + 224]",
    "    fnstcw word ptr [rsp + 228]",
    "    mov rbx, rcx",
    "    mov qword ptr [rbx + 72], rsp",
    "    mov rax, qword ptr [rbx + 40]",
    "    mov qword ptr [rsp + 32], rax",
    "    mov rax, qword ptr [rbx + 48]",
    "    mov qword ptr [rsp + 40], rax",
    "    mov rax, qword ptr [rbx + 56]",
    "    mov qword ptr [rsp + 48], rax",
    "    mov rax, qword ptr [rbx + 64]",
    "    mov qword ptr [rsp + 56], rax",
    "    mov rcx, qword ptr [rbx + 8]",
    "    mov rdx, qword ptr [rbx + 16]",
    "    mov r8, qword ptr [rbx + 24]",
    "    mov r9, qword ptr [rbx + 32]",
    "    call qword ptr [rbx]",
    "    mov eax, 1",
    "    jmp victory_guarded_call_restore",
    ".globl victory_guarded_call_recovered",
    "victory_guarded_call_recovered:",
    "    cld",
    "    ldmxcsr dword ptr [rsp + 224]",
    "    fldcw word ptr [rsp + 228]",
    "    xor eax, eax",
    "victory_guarded_call_restore:",
    "    movaps xmm6, xmmword ptr [rsp + 64]",
    "    movaps xmm7, xmmword ptr [rsp + 80]",
    "    movaps xmm8, xmmword ptr [rsp + 96]",
    "    movaps xmm9, xmmword ptr [rsp + 112]",
    "    movaps xmm10, xmmword ptr [rsp + 128]",
    "    movaps xmm11, xmmword ptr [rsp + 144]",
    "    movaps xmm12, xmmword ptr [rsp + 160]",
    "    movaps xmm13, xmmword ptr [rsp + 176]",
    "    movaps xmm14, xmmword ptr [rsp + 192]",
    "    movaps xmm15, xmmword ptr [rsp + 208]",
    "    add rsp, 232",
    "    pop r15",
    "    pop r14",
    "    pop r13",
    "    pop r12",
    "    pop rsi",
    "    pop rdi",
    "    pop rbp",
    "    pop rbx",
    "    ret",
    ".seh_endproc",
);

unsafe extern "C" {
    /// `call` is a `GuardedCall`; the assembly reads only the fields whose
    /// offsets the `const` assertions pin, so the pointer is untyped here.
    fn victory_guarded_call(call: *mut c_void) -> u32;
    /// Where a recovered call resumes, to return 0.
    #[link_name = "victory_guarded_call_recovered"]
    static GUARDED_CALL_RECOVERED: u8;
}

thread_local! {
    /// The guarded call this thread is in, null outside one. A `const`
    /// initialiser and a type without `Drop`: no allocation, no destructor,
    /// readable inside an exception handler.
    static GUARDED: Cell<*mut GuardedCall> = const { Cell::new(null_mut()) };
}

/// `RtlVirtualUnwind` on `context` for `entry` of `image`, through
/// [`guarded_call`]: a fault [`recover_unwind_fault`] recovers ends the call
/// and comes back as the error.
///
/// # Safety
/// `entry` came from `RtlLookupFunctionEntry` for `context.Rip`, with
/// `image`.
unsafe fn virtual_unwind(
    image: u64,
    entry: *mut IMAGE_RUNTIME_FUNCTION_ENTRY,
    context: &mut CONTEXT,
) -> Result<(), UnwindFault> {
    let (mut data, mut frame) = (null_mut::<c_void>(), 0_u64);
    let mut call = GuardedCall::new(
        (RtlVirtualUnwind as *const ()).expose_provenance(),
        [
            UNW_FLAG_NHANDLER as usize,
            image as usize,
            context.Rip as usize,
            entry.expose_provenance(),
            (&raw mut *context).expose_provenance(),
            (&raw mut data).expose_provenance(),
            (&raw mut frame).expose_provenance(),
            0,
        ],
    );
    // SAFETY: `RtlVirtualUnwind` takes these eight arguments: the caller
    // vouches for the entry and image, and the context and outputs are ours,
    // alive across the call.
    unsafe { guarded_call(&mut call) }
}

impl GuardedCall {
    fn new(function: usize, arguments: [usize; 8]) -> Self {
        Self {
            function,
            arguments,
            resume_rsp: 0,
            fault: None,
        }
    }
}

/// Sets `GUARDED` to `call`, calls `call`'s function through the trampoline,
/// then clears `GUARDED`. A fault [`recover_unwind_fault`] recovers ends the
/// call and comes back as the error; a guard-page violation's page then goes
/// to [`rearm_guard_page`].
///
/// # Safety
/// `call.function` is an `extern "system"` function that may be called with
/// `call.arguments`.
unsafe fn guarded_call(call: &mut GuardedCall) -> Result<(), UnwindFault> {
    GUARDED.set(&raw mut *call);
    // The trampoline's return value tells assembly callers the same as
    // `call.fault`, which a recovery writes before resuming it.
    // SAFETY: the caller vouches for the function and its arguments. A fault
    // `recover_unwind_fault` recovers abandons the call mid-way; when it
    // recovers one, what that was measured to leave behind, and what it was
    // not, is in its rustdoc. The trampoline restores every
    // nonvolatile register, MXCSR, the x87 control word and the direction
    // flag, so the caller's state is as the ABI promises.
    unsafe { victory_guarded_call((&raw mut *call).cast()) };
    GUARDED.set(null_mut());
    let Some(fault) = call.fault else {
        return Ok(());
    };
    if fault.kind == MemoryFault::GuardPage {
        rearm_guard_page(fault.address);
    }
    Err(fault)
}

/// Sets `PAGE_GUARD` again on the page holding `address`, which a read
/// cleared when it raised the guard-page violation, so the page keeps its
/// job (another thread's stack grows through it). Two windows stay open,
/// which no code can close: between the fault that cleared the guard and
/// this call, the owning thread can grow its stack through the page without
/// a guard fault; and between `VirtualQuery` and `VirtualProtect` the page's
/// protection can change (Windows has no compare-and-set for it).
fn rearm_guard_page(address: usize) {
    // SAFETY: a zeroed MEMORY_BASIC_INFORMATION is valid.
    let mut info: MEMORY_BASIC_INFORMATION = unsafe { std::mem::zeroed() };
    // SAFETY: queries the page holding `address`; writes only `info`.
    let queried = unsafe { VirtualQuery(address as *const c_void, &mut info, size_of_val(&info)) };
    if queried == 0 || info.Protect & PAGE_GUARD != 0 {
        return;
    }
    let mut old = 0;
    // SAFETY: changes only the guard bit of one committed page, which had
    // it until the read.
    unsafe {
        VirtualProtect(
            address as *const c_void,
            1,
            info.Protect | PAGE_GUARD,
            &mut old,
        )
    };
}

/// Recovers a guarded call from a memory fault raised in the unwinder's
/// image. It does so when all of these hold: this thread is in a guarded
/// call (`GUARDED` names it and the trampoline has stored its resume point),
/// the exception is an access violation, a guard-page violation or an
/// in-page error ([`MemoryFault`]), and it was raised inside
/// [`UNWINDER_IMAGE`] (so not before [`locate_unwinder`] has run). It then
/// writes the fault into the call's record and resumes the trampoline at its
/// recovery label with the stack it had before the call, which abandons the
/// call. Every exception that reaches it takes the call out of `GUARDED`,
/// recovered or not, so only the first exception of a guarded call that
/// reaches it can be recovered: after a declined one (a status
/// `RtlVirtualUnwind` raises itself, or an exception a function table
/// callback raises outside ntdll and handles itself), every later exception
/// in that call is left to the rest of the vectored handler and the
/// handlers after it, as outside a guarded call. An exception that a
/// vectored handler registered ahead of ours resumes never reaches it and
/// does not count: the GPU write-watch handler resumes only writes, and
/// `RtlVirtualUnwind` writes only our context and outputs. A fault a
/// function table callback raises inside an ntdll routine it calls cannot
/// be told apart from the unwinder's own and is recovered, abandoning the
/// callback.
///
/// The vectored handler calls it first, on whichever thread faulted. It runs
/// nested inside whatever raised the fault (the vectored handler, the
/// unhandled-exception filter or `stacks::dump_stacks`), so it neither logs,
/// allocates nor locks.
///
/// Measured on Windows 11 build 26200.9974 (Ryzen 7 9800X3D), by the tests
/// here: a fault nested in the vectored handler or in an unhandled-exception
/// filter is recovered and the report comes out whole; faults reading frame
/// memory, recovered 10,000 times (half of them after `RtlVirtualUnwind`
/// looked a jump target up itself), leave adding and deleting function
/// tables, walking and panic unwinding usable on another thread; faults
/// reading unwind info and a guard page are recovered too; reading another
/// thread's stack guard page raises a guard-page violation that clears the
/// guard, which [`guarded_call`] sets again. Not covered: a fault inside the
/// search of a registered function table whose entries are gone, which
/// could leave that table's lock held. Resuming by rewriting `Rip` and
/// `Rsp` relies on the executable not being shadow-stack (CET) compatible,
/// which neither `victory.exe` nor the test binaries are.
///
/// # Safety
/// `info` comes from the system, for an exception being dispatched on this
/// thread.
pub(crate) unsafe fn recover_unwind_fault(info: *mut EXCEPTION_POINTERS) -> bool {
    let call = GUARDED.replace(null_mut());
    // SAFETY: the system passes valid exception pointers; the records belong
    // to this thread's dispatch for the handler's duration.
    let (record, context) = unsafe { (&*(*info).ExceptionRecord, &mut *(*info).ContextRecord) };
    let Ok(kind) = MemoryFault::try_from(record.ExceptionCode) else {
        return false;
    };
    let in_unwinder = UNWINDER_IMAGE
        .get()
        .is_some_and(|image| image.contains(&record.ExceptionAddress.addr()));
    if call.is_null() || !in_unwinder {
        return false;
    }
    // SAFETY: `GUARDED` named a `GuardedCall` on this thread's stack, alive
    // until `guarded_call` returns, which it has not: the fault is inside
    // the call. Nothing else writes it meanwhile.
    let call = unsafe { &mut *call };
    if call.resume_rsp == 0 {
        return false;
    }
    call.fault = Some(UnwindFault {
        kind,
        address: record.ExceptionInformation[1],
    });
    context.Rip = (&raw const GUARDED_CALL_RECOVERED).addr() as u64;
    context.Rsp = call.resume_rsp as u64;
    true
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;
    use windows_sys::Win32::Foundation::{
        EXCEPTION_BREAKPOINT, EXCEPTION_ILLEGAL_INSTRUCTION, EXCEPTION_STACK_OVERFLOW,
    };
    use windows_sys::Win32::System::Diagnostics::Debug::{
        EXCEPTION_RECORD, IMAGE_RUNTIME_FUNCTION_ENTRY_0, RtlAddFunctionTable, RtlCaptureContext,
        RtlCaptureStackBackTrace, RtlDeleteFunctionTable,
    };
    use windows_sys::Win32::System::Memory::{
        MEM_COMMIT, MEM_RELEASE, MEM_RESERVE, PAGE_EXECUTE_READWRITE, PAGE_NOACCESS,
        PAGE_READWRITE, VirtualAlloc, VirtualFree,
    };
    use windows_sys::Win32::System::Threading::GetCurrentThreadStackLimits;

    use crate::crash::{memory_at, report_crashes};

    /// Installs the vectored handler, which recovers guarded calls, once per
    /// test process.
    fn install_handler() {
        static ONCE: std::sync::Once = std::sync::Once::new();
        ONCE.call_once(report_crashes);
    }

    /// Two fresh pages, the first `first`, the second no-access; released when
    /// dropped.
    struct Pages(*mut u8);

    impl Pages {
        fn new(first: u32) -> Self {
            // SAFETY: reserves and commits two fresh pages of ours, then
            // changes the protection of the second.
            unsafe {
                let base = VirtualAlloc(std::ptr::null(), 0x2000, MEM_RESERVE | MEM_COMMIT, first)
                    .cast::<u8>();
                assert!(!base.is_null());
                let mut old = 0;
                let second = base.add(0x1000).cast();
                assert_ne!(VirtualProtect(second, 0x1000, PAGE_NOACCESS, &mut old), 0);
                Self(base)
            }
        }

        fn address(&self, offset: usize) -> u64 {
            self.0.addr() as u64 + offset as u64
        }
    }

    impl Drop for Pages {
        fn drop(&mut self) {
            // SAFETY: releases the reservation `new` made.
            unsafe { VirtualFree(self.0.cast(), 0, MEM_RELEASE) };
        }
    }

    /// One of this test binary's tests, run alone in a child process with
    /// one environment variable set.
    struct Child<'a> {
        test: &'a str,
        variable: &'a str,
        value: &'a str,
    }

    impl Child<'_> {
        /// Runs it; returns its exit code and its stderr.
        fn run(&self) -> (Option<i32>, String) {
            let out = std::process::Command::new(std::env::current_exe().unwrap())
                .args(["--exact", self.test, "--nocapture"])
                .env(self.variable, self.value)
                .output()
                .unwrap();
            let text = String::from_utf8_lossy(&out.stderr).into_owned();
            (out.status.code(), text)
        }
    }

    /// An address in the reserved, never committed part of this thread's
    /// stack.
    fn reserved_stack_address() -> usize {
        let (mut low, mut high) = (0, 0);
        // SAFETY: writes the current thread's stack bounds, its whole
        // reservation.
        unsafe { GetCurrentThreadStackLimits(&mut low, &mut high) };
        let reserved = low + 0x2000;
        assert_eq!(memory_at(reserved), "reserved, never committed");
        reserved
    }

    /// In a child process of the next test: walks from `rsp` in the reserved,
    /// uncommitted part of its stack, with `rip` where no function table
    /// reaches, so the walk would read the return address at `rsp`; that read
    /// ends the process if the walk's bounds allow it.
    #[test]
    fn stack_pointer_child() {
        if std::env::var_os("VICTORY_STACK_POINTER_CHILD").is_none() {
            return;
        }
        // SAFETY: a zeroed CONTEXT is valid.
        let mut context: CONTEXT = unsafe { std::mem::zeroed() };
        context.Rip = 0x1234;
        context.Rsp = reserved_stack_address() as u64;
        // On stderr: libtest writes test names to stdout, without a newline
        // before the test runs when it runs one test at a time.
        eprintln!("{} frames", unwind(context).0.len());
        std::process::exit(0);
    }

    #[test]
    fn a_stack_pointer_in_the_uncommitted_stack_ends_the_walk() {
        let (code, text) = Child {
            test: "unwind::tests::stack_pointer_child",
            variable: "VICTORY_STACK_POINTER_CHILD",
            value: "1",
        }
        .run();
        assert_eq!(code, Some(0), "{text}");
        assert!(text.lines().any(|l| l == "1 frames"), "{text}");
    }

    /// What [`capture_frames`] saw: our [`unwind`], then the system's walk.
    type Captured = (Vec<u64>, Vec<usize>);

    /// A frame we expect [`unwind`] to report: the helper it is in, and the
    /// helper's name for the message.
    type Expected = (&'static str, fn() -> Captured);

    /// The runtime function entry start covering `at`: the image base the
    /// lookup writes, plus the entry's `BeginAddress`. Asking the system for
    /// both sides of the comparison means a thunk or an incremental-link jump
    /// is not mistaken for a different function.
    fn entry_start_at(at: u64) -> Option<u64> {
        let mut image = 0;
        // SAFETY: looks up the function table entry covering `at`; nothing
        // keeps a reference to the entry past this call.
        let entry = unsafe { RtlLookupFunctionEntry(at, &mut image, null_mut()) };
        if entry.is_null() {
            return None;
        }
        // SAFETY: a non-null entry belongs to `image`, which the lookup wrote,
        // and `BeginAddress` is an offset into that image.
        Some(image + u64::from(unsafe { (*entry).BeginAddress }))
    }

    #[test]
    fn unwinds_like_the_system_walk() {
        // Asking the system's own walk whether the stack looks right is not
        // reliable: on Windows 10 x64 debug builds `RtlCaptureStackBackTrace`
        // returns no frames about half the time. The frames above us are
        // known instead (see `level_a`), so check those first.
        let (frames, walked) = level_a();
        // Frame 0 is in `capture_frames`, which took the context; then its
        // callers in call order, each named by its runtime function entry.
        let helpers: [Expected; 4] = [
            ("capture_frames", capture_frames),
            ("level_c", level_c),
            ("level_b", level_b),
            ("level_a", level_a),
        ];
        assert!(frames.len() >= helpers.len(), "{frames:?}");
        for (i, (name, helper)) in helpers.iter().enumerate() {
            let want = entry_start_at(*helper as usize as u64)
                .unwrap_or_else(|| panic!("no unwind data for {name}"));
            let got = entry_start_at(frames[i])
                .unwrap_or_else(|| panic!("no unwind data for frame {i} at {:#x}", frames[i]));
            assert_eq!(got, want, "frame {i} should be {name}: {frames:?}");
        }
        // A walk that repeats an address is not a walk.
        let mut distinct = frames[..helpers.len()].to_vec();
        distinct.sort_unstable();
        distinct.dedup();
        assert_eq!(distinct.len(), helpers.len(), "repeated: {frames:?}");
        // Where the system's walk does return frames, it must still agree with
        // ours: frame 0 differs (both are in `capture_frames`, at different
        // call sites), and the callers above are the same.
        if walked.len() >= helpers.len() {
            assert!(frames.len() >= walked.len(), "{frames:?} vs {walked:?}");
            let ours: Vec<usize> = frames[1..walked.len()]
                .iter()
                .map(|&f| f as usize)
                .collect();
            assert_eq!(ours.as_slice(), &walked[1..]);
        }
    }

    /// The frames of this call, both ways: our [`unwind`], and the system's
    /// walk, which Windows 10 can come up empty on.
    #[inline(never)]
    fn capture_frames() -> Captured {
        // SAFETY: a zeroed CONTEXT is valid and both calls write buffers of
        // ours; `RtlCaptureContext` wants the context 16-byte aligned too.
        unsafe {
            let mut context = Aligned16(std::mem::zeroed());
            RtlCaptureContext(&mut context.0);
            let mut walked = [null_mut(); 8];
            let n = RtlCaptureStackBackTrace(0, 8, walked.as_mut_ptr(), null_mut());
            let frames = unwind(context.0).0;
            let walked = walked[..usize::from(n)]
                .iter()
                .map(|&f| f as usize)
                .collect();
            (frames, walked)
        }
    }

    /// `level_a` → `level_b` → `level_c` → [`capture_frames`]: three known
    /// callers above the capture. `black_box` after each call keeps it from
    /// becoming a tail call, which would erase the caller's frame.
    #[inline(never)]
    fn level_a() -> Captured {
        let captured = level_b();
        std::hint::black_box(&captured);
        captured
    }

    /// See [`level_a`].
    #[inline(never)]
    fn level_b() -> Captured {
        let captured = level_c();
        std::hint::black_box(&captured);
        captured
    }

    /// See [`level_a`].
    #[inline(never)]
    fn level_c() -> Captured {
        let captured = capture_frames();
        std::hint::black_box(&captured);
        captured
    }

    /// The frame register its unwind info names: the low nibble of the
    /// header's byte 3 (Microsoft's "x64 exception handling").
    fn frame_register(image: u64, entry: &IMAGE_RUNTIME_FUNCTION_ENTRY) -> u8 {
        // SAFETY: both union fields are the same `u32`; the unwind info of
        // this test binary's own code is mapped with it.
        unsafe {
            let info = (image + u64::from(entry.Anonymous.UnwindData)) as *const [u8; 4];
            info.read()[3] & 15
        }
    }

    /// Captures its own context, puts `bogus` in rbp, its frame register, and
    /// walks from there; returns the frames walked and how the walk ended.
    /// The 64-byte aligned local makes Win64 code realign the stack through a
    /// frame pointer, checked here from the unwind info.
    #[inline(never)]
    fn walk_with_frame_register(bogus: u64) -> (usize, Result<(), UnwindFault>) {
        #[repr(align(64))]
        struct Line([u8; 64]);
        std::hint::black_box(&Line([0; 64]).0);
        // SAFETY: a zeroed CONTEXT is valid; the calls write our buffer, and
        // the entry found is this test binary's own, mapped with its image.
        let mut context = unsafe {
            let mut context = Aligned16(std::mem::zeroed());
            RtlCaptureContext(&mut context.0);
            let mut image = 0;
            let entry = RtlLookupFunctionEntry(context.0.Rip, &mut image, null_mut());
            assert!(!entry.is_null());
            assert_eq!(
                frame_register(image, &*entry),
                5,
                "rbp is not the frame register"
            );
            context.0.Rbp = bogus;
            context
        };
        let (frames, walked) = unwind(context.0);
        std::hint::black_box(&mut context);
        (frames.len(), walked)
    }

    /// In a child process of the next tests: walks with a frame register
    /// outside the stack (`outside`) or in its reserved, uncommitted part
    /// (`uncommitted`), which ends the process if the walk's fault is not
    /// recovered.
    #[test]
    fn frame_register_child() {
        let Some(case) = std::env::var_os("VICTORY_FRAME_REGISTER_CHILD") else {
            return;
        };
        report_crashes();
        let bogus = if case == "uncommitted" {
            reserved_stack_address() as u64
        } else {
            0x1000
        };
        let (frames, walked) = walk_with_frame_register(bogus);
        // On stderr: libtest writes test names to stdout, without a newline
        // before the test runs when it runs one test at a time.
        eprintln!("{frames} frames");
        if let Err(fault) = walked {
            eprintln!("walk ended: {fault}");
        }
        std::process::exit(0);
    }

    /// Runs [`frame_register_child`] with `case`; asserts it walked one frame
    /// and named the fault that ended the walk.
    fn assert_one_frame_walked(case: &str) {
        let (code, text) = Child {
            test: "unwind::tests::frame_register_child",
            variable: "VICTORY_FRAME_REGISTER_CHILD",
            value: case,
        }
        .run();
        assert_eq!(code, Some(0), "{text}");
        assert!(text.lines().any(|l| l == "1 frames"), "{text}");
        let ended = text
            .lines()
            .any(|l| l.starts_with("walk ended: access violation at "));
        assert!(ended, "{text}");
    }

    #[test]
    fn a_frame_register_outside_the_stack_ends_the_walk() {
        assert_one_frame_walked("outside");
    }

    #[test]
    fn a_frame_register_in_the_uncommitted_stack_ends_the_walk() {
        assert_one_frame_walked("uncommitted");
    }

    /// push rbp; mov rbp, rsp
    const PROLOG: [u8; 4] = [0x55, 0x48, 0x89, 0xe5];
    /// Version 1, a 4-byte prolog, 2 codes, frame register rbp at offset 0:
    /// UWOP_SET_FPREG after byte 4, UWOP_PUSH_NONVOL rbp after byte 1.
    const FRAMED_INFO: [u8; 8] = [0x01, 4, 2, 0x05, 4, 0x03, 1, 0x50];

    /// The crash child's code page: `smashed` at 0 (`PROLOG`; mov rbp, 0x1000;
    /// int3; pop rbp; ret) and `framed` at 0x40 (`PROLOG`; int3; pop rbp;
    /// ret), both with `FRAMED_INFO` at 0x20; the table at 0x80.
    const SMASHED_BREAK: usize = 11;
    const FRAMED_AT: usize = 0x40;

    /// For the crash child, after ours: resumes the breakpoint in `smashed`
    /// with rbp repaired, so Windows' own dispatch never walks the smashed
    /// frame (it would fault in ntdll, outside any guarded call).
    unsafe extern "system" fn resume_smashed(info: *mut EXCEPTION_POINTERS) -> i32 {
        // SAFETY: the system passes valid exception pointers, for this
        // thread's dispatch.
        let (record, context) = unsafe { (&*(*info).ExceptionRecord, &mut *(*info).ContextRecord) };
        let at = CODE.load(std::sync::atomic::Ordering::Relaxed) + SMASHED_BREAK as u64;
        if record.ExceptionCode != EXCEPTION_BREAKPOINT || context.Rip != at {
            return 0; // EXCEPTION_CONTINUE_SEARCH
        }
        // `mov rbp, rsp` had set it to `rsp`, which nothing has moved since.
        context.Rbp = context.Rsp;
        context.Rip += 1;
        eprintln!("resumed");
        -1 // EXCEPTION_CONTINUE_EXECUTION
    }

    /// For the crash child, in place of ours, as the title's filter that
    /// calls `report_unhandled` first: reports a copy of the exception whose
    /// rbp is smashed, after Windows' dispatch walked the real frame. The
    /// real context stays whole: Windows unwinds it next, to the thread's
    /// top-level handler.
    unsafe extern "system" fn smash_then_report(info: *const EXCEPTION_POINTERS) -> i32 {
        // SAFETY: the system passes the exception being dispatched; the copy
        // and its pointers are ours, alive across the report.
        unsafe {
            let mut context = Aligned16(*(*info).ContextRecord);
            context.0.Rbp = 0x1000;
            let copy = EXCEPTION_POINTERS {
                ExceptionRecord: (*info).ExceptionRecord,
                ContextRecord: &raw mut context.0,
            };
            crate::crash::report_unhandled(&copy);
        }
        0 // EXCEPTION_CONTINUE_SEARCH
    }

    /// The crash child's code page.
    static CODE: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

    /// In a child process of the next test. First a breakpoint in `smashed`,
    /// whose report from our vectored handler walks through rbp = 0x1000 and
    /// faults there, inside the handler; `resume_smashed` resumes it. Then a
    /// breakpoint in `framed` nobody handles, whose report from the filter
    /// walks through rbp = 0x1000 and faults there, inside the filter.
    #[test]
    fn crash_child() {
        use windows_sys::Win32::System::Diagnostics::Debug::{
            AddVectoredExceptionHandler, SetUnhandledExceptionFilter,
        };
        if std::env::var_os("VICTORY_CRASH_CHILD").is_none() {
            return;
        }
        report_crashes();
        // SAFETY: registers a handler after ours and a filter in place of
        // ours; both touch only the exception they are given.
        unsafe {
            AddVectoredExceptionHandler(0, Some(resume_smashed));
            SetUnhandledExceptionFilter(Some(smash_then_report));
        }
        // mov rbp, 0x1000; int3; pop rbp; ret
        let smashed_body = [0x48, 0xc7, 0xc5, 0x00, 0x10, 0x00, 0x00, 0xcc, 0x5d, 0xc3];
        // int3; pop rbp; ret
        let framed_body = [0xcc, 0x5d, 0xc3];
        // SAFETY: a fresh page of ours holding both functions and their
        // unwind data, registered, then run.
        unsafe {
            let base = VirtualAlloc(
                std::ptr::null(),
                0x1000,
                MEM_RESERVE | MEM_COMMIT,
                PAGE_EXECUTE_READWRITE,
            )
            .cast::<u8>();
            assert!(!base.is_null());
            CODE.store(base as u64, std::sync::atomic::Ordering::Relaxed);
            for (at, body) in [(0, &smashed_body[..]), (FRAMED_AT, &framed_body[..])] {
                base.add(at)
                    .copy_from_nonoverlapping(PROLOG.as_ptr(), PROLOG.len());
                base.add(at + 4)
                    .copy_from_nonoverlapping(body.as_ptr(), body.len());
            }
            base.add(0x20)
                .copy_from_nonoverlapping(FRAMED_INFO.as_ptr(), FRAMED_INFO.len());
            let table = base.add(0x80).cast::<IMAGE_RUNTIME_FUNCTION_ENTRY>();
            let entry = |begin: usize, len: usize| IMAGE_RUNTIME_FUNCTION_ENTRY {
                BeginAddress: begin as u32,
                EndAddress: (begin + len) as u32,
                Anonymous: IMAGE_RUNTIME_FUNCTION_ENTRY_0 { UnwindData: 0x20 },
            };
            table.write(entry(0, 4 + smashed_body.len()));
            table.add(1).write(entry(FRAMED_AT, 4 + framed_body.len()));
            assert!(RtlAddFunctionTable(table, 2, base as u64));
            let smashed: extern "C" fn() = std::mem::transmute(base);
            smashed();
            let framed: extern "C" fn() = std::mem::transmute(base.add(FRAMED_AT));
            framed();
        }
    }

    #[test]
    fn a_crash_whose_walk_faults_is_reported_whole() {
        let (code, text) = Child {
            test: "unwind::tests::crash_child",
            variable: "VICTORY_CRASH_CHILD",
            value: "1",
        }
        .run();
        // The second breakpoint's own code: the process ends as it would have.
        assert_eq!(code, Some(EXCEPTION_BREAKPOINT), "{text}");
        let lines: Vec<_> = text.lines().collect();
        let report = |what: &str| {
            lines
                .iter()
                .position(|l| l.starts_with(&format!("VIctory: {what} 0x80000003")))
                .unwrap_or_else(|| panic!("no {what}: {text}"))
        };
        // From the vectored handler: registers, frame 0, then the walk's end,
        // and the child went on.
        let first = report("first-chance exception");
        assert!(lines[first + 3].starts_with("  #0 "), "{text}");
        assert_eq!(
            lines[first + 4],
            "  walk ended: access violation at 0x1000",
            "{text}"
        );
        assert_eq!(lines[first + 5], "resumed", "{text}");
        // From the filter, the same.
        let last = report("unhandled exception");
        assert!(lines[last + 3].starts_with("  #0 "), "{text}");
        assert_eq!(
            lines[last + 4],
            "  walk ended: access violation at 0x1000",
            "{text}"
        );
    }

    /// In a child process of the next test: recurses until the stack runs out.
    #[test]
    fn stack_overflow_child() {
        #[inline(never)]
        fn recurse(depth: u64) -> u64 {
            if depth == u64::MAX {
                return 0;
            }
            let mut page = [0_u8; 4096];
            std::hint::black_box(&mut page);
            recurse(depth + 1) + u64::from(page[0])
        }
        if std::env::var_os("VICTORY_STACK_OVERFLOW_CHILD").is_none() {
            return;
        }
        report_crashes();
        std::hint::black_box(recurse(0));
    }

    #[test]
    fn a_stack_overflow_is_still_reported() {
        let (code, text) = Child {
            test: "unwind::tests::stack_overflow_child",
            variable: "VICTORY_STACK_OVERFLOW_CHILD",
            value: "1",
        }
        .run();
        assert_eq!(code, Some(EXCEPTION_STACK_OVERFLOW), "{text}");
        let lines: Vec<_> = text.lines().collect();
        let header = lines
            .iter()
            .position(|l| l.starts_with("VIctory: first-chance exception 0xc00000fd"))
            .unwrap_or_else(|| panic!("no report: {text}"));
        // Registers, then the walk on the exhausted stack: the faulting frame
        // and its caller.
        assert!(lines[header + 3].starts_with("  #0 "), "{text}");
        assert!(lines[header + 4].starts_with("  #1 "), "{text}");
    }

    /// Code, unwind info and the function table that covers them, laid out
    /// by [`walk_synthetic`] on a page followed by a no-access page.
    struct Synthetic<'a> {
        /// At offset 0, at most 16 bytes.
        code: &'a [u8],
        /// At offset 16, at most 32 bytes.
        unwind_info: &'a [u8],
        /// Begin, end and unwind info RVAs of each entry, sorted; at offset 48.
        functions: &'a [[u32; 3]],
    }

    /// `PROLOG`, then a `nop` of body.
    const FRAMED_BODY: [u8; 5] = [0x55, 0x48, 0x89, 0xe5, 0x90];

    /// Registers `function` and walks from `rip_offset` bytes into it, with
    /// `rbp` and `stack` as the whole stack. Returns the frames, how the walk
    /// ended, and the walk's start address.
    fn walk_synthetic(
        function: &Synthetic,
        rip_offset: u64,
        stack: &[u64],
        rbp: u64,
    ) -> (Vec<u64>, Result<(), UnwindFault>, u64) {
        install_handler();
        assert!(function.code.len() <= 16 && function.unwind_info.len() <= 32);
        let pages = Pages::new(PAGE_READWRITE);
        // SAFETY: the first of `pages`, ours, read but never run, registered
        // as these functions' code and unwind data, then unregistered.
        unsafe {
            let base = pages.0;
            base.copy_from_nonoverlapping(function.code.as_ptr(), function.code.len());
            base.add(16).copy_from_nonoverlapping(
                function.unwind_info.as_ptr(),
                function.unwind_info.len(),
            );
            let table = base.add(48).cast::<IMAGE_RUNTIME_FUNCTION_ENTRY>();
            for (i, &[begin, end, info]) in function.functions.iter().enumerate() {
                table.add(i).write(IMAGE_RUNTIME_FUNCTION_ENTRY {
                    BeginAddress: begin,
                    EndAddress: end,
                    Anonymous: IMAGE_RUNTIME_FUNCTION_ENTRY_0 { UnwindData: info },
                });
            }
            let count = function.functions.len() as u32;
            assert!(RtlAddFunctionTable(table, count, base as u64));
            let mut context: CONTEXT = std::mem::zeroed();
            context.Rip = pages.address(0) + rip_offset;
            context.Rsp = stack.as_ptr() as u64;
            context.Rbp = rbp;
            let low = stack.as_ptr() as usize;
            let mut frames = Vec::with_capacity(MAX_FRAMES);
            let walked = unwind_into(context, low, low + size_of_val(stack), &mut frames);
            RtlDeleteFunctionTable(table);
            (frames, walked, pages.address(0) + rip_offset)
        }
    }

    #[test]
    fn a_return_address_in_a_no_access_page_ends_the_walk() {
        // The frame's last 8 bytes, the saved rbp, end a page; its return
        // address would be the first of the next, no-access page.
        let frame = Pages::new(PAGE_READWRITE);
        let saved_rbp = frame.address(0xff8);
        // SAFETY: the last 8 bytes of the frame's first page, ours.
        let stack = unsafe { std::slice::from_raw_parts(saved_rbp as *const u64, 1) };
        let function = Synthetic {
            code: &FRAMED_BODY,
            unwind_info: &FRAMED_INFO,
            functions: &[[0, 5, 16]],
        };
        // At the `nop`, with rbp at the saved rbp.
        let (frames, walked, rip) = walk_synthetic(&function, 4, stack, saved_rbp);
        assert_eq!(frames, [rip]);
        let fault = UnwindFault {
            kind: MemoryFault::AccessViolation,
            address: frame.address(0x1000) as usize,
        };
        assert_eq!(walked, Err(fault));
    }

    #[test]
    fn unwind_info_in_a_no_access_page_ends_the_walk() {
        let function = Synthetic {
            code: &FRAMED_BODY,
            unwind_info: &FRAMED_INFO,
            // The unwind info named is the first byte of the no-access page.
            functions: &[[0, 5, 0x1000]],
        };
        let stack = [0x5678, 0x1234, 0];
        let (frames, walked, rip) = walk_synthetic(&function, 4, &stack, stack.as_ptr() as u64);
        assert_eq!(frames, [rip]);
        let fault = walked.unwrap_err();
        assert_eq!(fault.kind, MemoryFault::AccessViolation);
        assert_eq!(fault.address as u64, rip - 4 + 0x1000);
    }

    /// Walks from the body of a framed function whose frame register points at
    /// `guard`, a guard page; asserts the walk ends at its first frame with
    /// the guard-page fault there, and that the page is a guard page again.
    fn assert_guard_page_walk(guard: usize) {
        let function = Synthetic {
            code: &FRAMED_BODY,
            unwind_info: &FRAMED_INFO,
            functions: &[[0, 5, 16]],
        };
        let (frames, walked, rip) = walk_synthetic(&function, 4, &[0x1234, 0], guard as u64);
        assert_eq!(frames, [rip]);
        let fault = UnwindFault {
            kind: MemoryFault::GuardPage,
            address: guard,
        };
        assert_eq!(walked, Err(fault));
        assert_eq!(memory_at(guard), "a guard page");
    }

    #[test]
    fn a_frame_in_a_guard_page_ends_the_walk_and_keeps_the_guard() {
        // SAFETY: reserves and commits a fresh guard page of ours.
        let guard = unsafe {
            VirtualAlloc(
                std::ptr::null(),
                0x1000,
                MEM_RESERVE | MEM_COMMIT,
                PAGE_READWRITE | PAGE_GUARD,
            )
        };
        assert!(!guard.is_null());
        assert_guard_page_walk(guard.addr());
        // SAFETY: releases the page reserved above.
        unsafe { VirtualFree(guard, 0, MEM_RELEASE) };
    }

    /// Uses about `BYTES` of stack below its caller's.
    #[inline(never)]
    fn use_stack<const BYTES: usize>() -> u8 {
        let mut block = [0_u8; BYTES];
        std::hint::black_box(&mut block)[BYTES / 2]
    }

    #[test]
    fn a_frame_in_another_threads_stack_guard_page_ends_the_walk_and_keeps_the_guard() {
        let (sent, got) = std::sync::mpsc::channel();
        let (go, wait) = std::sync::mpsc::channel::<()>();
        let parked = std::thread::spawn(move || {
            // Deeper first than anything it runs until the wait, so its
            // stack does not grow past the limit it sends meanwhile.
            use_stack::<0x10000>();
            sent.send(current_thread_stack().start).unwrap();
            wait.recv().unwrap();
            // Its stack still grows through the page the walk read, and the
            // thread finishes unharmed.
            use_stack::<0x20000>()
        });
        let committed_start = got.recv().unwrap();
        let guard = committed_start - 0x1000;
        assert_eq!(memory_at(guard), "a guard page");
        assert_guard_page_walk(guard);
        go.send(()).unwrap();
        assert_eq!(parked.join().ok(), Some(0));
    }

    // `victory_test_clobber_and_fault(junk: *const u32 /* rcx */, fault: u64
    // /* rdx */, compare: usize /* r8 */)`: sets every nonvolatile register to
    // junk, sets the direction flag, loads MXCSR from `junk[0]` and the x87
    // control word from `junk[1]`, then jumps to `compare`, ntdll's
    // `RtlCompareMemoryUlong`, as `compare(fault, 8, 0)`, which reads
    // `fault`: the fault is raised in ntdll, as `RtlVirtualUnwind`'s are.
    // Uses no stack itself; it never returns when `fault` is unreadable.
    //
    // `victory_test_guarded_registers(call: *mut GuardedCall /* rcx */) ->
    // u64`: loads a pattern into every nonvolatile register, calls
    // `victory_guarded_call(call)`, and returns which of them, the direction
    // flag, MXCSR and the x87 control word differ afterwards (bits ORed
    // together, 0 when all match), ORed with the trampoline's return value
    // (1 when the call was not recovered).
    std::arch::global_asm!(
        ".text",
        ".globl victory_test_clobber_and_fault",
        "victory_test_clobber_and_fault:",
        "    mov rbx, -1",
        "    mov rbp, -1",
        "    mov rsi, -1",
        "    mov rdi, -1",
        "    mov r12, -1",
        "    mov r13, -1",
        "    mov r14, -1",
        "    mov r15, -1",
        "    pcmpeqd xmm6, xmm6",
        "    pcmpeqd xmm7, xmm7",
        "    pcmpeqd xmm8, xmm8",
        "    pcmpeqd xmm9, xmm9",
        "    pcmpeqd xmm10, xmm10",
        "    pcmpeqd xmm11, xmm11",
        "    pcmpeqd xmm12, xmm12",
        "    pcmpeqd xmm13, xmm13",
        "    pcmpeqd xmm14, xmm14",
        "    pcmpeqd xmm15, xmm15",
        "    std",
        "    ldmxcsr dword ptr [rcx]",
        "    fldcw word ptr [rcx + 4]",
        "    mov rax, r8",
        "    mov rcx, rdx",
        "    mov edx, 8",
        "    xor r8d, r8d",
        "    jmp rax",
        ".p2align 4",
        ".globl victory_test_guarded_registers",
        "victory_test_guarded_registers:",
        ".seh_proc victory_test_guarded_registers",
        "    push rbx",
        ".seh_pushreg rbx",
        "    push rbp",
        ".seh_pushreg rbp",
        "    push rdi",
        ".seh_pushreg rdi",
        "    push rsi",
        ".seh_pushreg rsi",
        "    push r12",
        ".seh_pushreg r12",
        "    push r13",
        ".seh_pushreg r13",
        "    push r14",
        ".seh_pushreg r14",
        "    push r15",
        ".seh_pushreg r15",
        "    sub rsp, 216",
        ".seh_stackalloc 216",
        "    movaps xmmword ptr [rsp + 32], xmm6",
        ".seh_savexmm xmm6, 32",
        "    movaps xmmword ptr [rsp + 48], xmm7",
        ".seh_savexmm xmm7, 48",
        "    movaps xmmword ptr [rsp + 64], xmm8",
        ".seh_savexmm xmm8, 64",
        "    movaps xmmword ptr [rsp + 80], xmm9",
        ".seh_savexmm xmm9, 80",
        "    movaps xmmword ptr [rsp + 96], xmm10",
        ".seh_savexmm xmm10, 96",
        "    movaps xmmword ptr [rsp + 112], xmm11",
        ".seh_savexmm xmm11, 112",
        "    movaps xmmword ptr [rsp + 128], xmm12",
        ".seh_savexmm xmm12, 128",
        "    movaps xmmword ptr [rsp + 144], xmm13",
        ".seh_savexmm xmm13, 144",
        "    movaps xmmword ptr [rsp + 160], xmm14",
        ".seh_savexmm xmm14, 160",
        "    movaps xmmword ptr [rsp + 176], xmm15",
        ".seh_savexmm xmm15, 176",
        ".seh_endprologue",
        "    stmxcsr dword ptr [rsp + 192]",
        "    fnstcw word ptr [rsp + 196]",
        "    mov rbx, 0x1111111111111111",
        "    mov rbp, 0x2222222222222222",
        "    mov rsi, 0x3333333333333333",
        "    mov rdi, 0x4444444444444444",
        "    mov r12, 0x5555555555555555",
        "    mov r13, 0x6666666666666666",
        "    mov r14, 0x7777777777777777",
        "    mov r15, 0x0123456789abcdef",
        "    movq xmm6, rbx",
        "    punpcklqdq xmm6, xmm6",
        "    movq xmm7, rbp",
        "    punpcklqdq xmm7, xmm7",
        "    movq xmm8, rsi",
        "    punpcklqdq xmm8, xmm8",
        "    movq xmm9, rdi",
        "    punpcklqdq xmm9, xmm9",
        "    movq xmm10, r12",
        "    punpcklqdq xmm10, xmm10",
        "    movq xmm11, r13",
        "    punpcklqdq xmm11, xmm11",
        "    movq xmm12, r14",
        "    punpcklqdq xmm12, xmm12",
        "    movq xmm13, r15",
        "    punpcklqdq xmm13, xmm13",
        "    movq xmm14, rbx",
        "    punpcklqdq xmm14, xmm14",
        "    movq xmm15, rbp",
        "    punpcklqdq xmm15, xmm15",
        "    call victory_guarded_call",
        "    mov r10, rax",
        "    mov r11, 0x1111111111111111",
        "    xor r11, rbx",
        "    or r10, r11",
        "    mov r11, 0x2222222222222222",
        "    xor r11, rbp",
        "    or r10, r11",
        "    mov r11, 0x3333333333333333",
        "    xor r11, rsi",
        "    or r10, r11",
        "    mov r11, 0x4444444444444444",
        "    xor r11, rdi",
        "    or r10, r11",
        "    mov r11, 0x5555555555555555",
        "    xor r11, r12",
        "    or r10, r11",
        "    mov r11, 0x6666666666666666",
        "    xor r11, r13",
        "    or r10, r11",
        "    mov r11, 0x7777777777777777",
        "    xor r11, r14",
        "    or r10, r11",
        "    mov r11, 0x0123456789abcdef",
        "    xor r11, r15",
        "    or r10, r11",
        // Each xmm register: both halves against the GPR it was loaded from.
        "    movq r11, xmm6",
        "    xor r11, rbx",
        "    or r10, r11",
        "    pextrq r11, xmm6, 1",
        "    xor r11, rbx",
        "    or r10, r11",
        "    movq r11, xmm7",
        "    xor r11, rbp",
        "    or r10, r11",
        "    pextrq r11, xmm7, 1",
        "    xor r11, rbp",
        "    or r10, r11",
        "    movq r11, xmm8",
        "    xor r11, rsi",
        "    or r10, r11",
        "    pextrq r11, xmm8, 1",
        "    xor r11, rsi",
        "    or r10, r11",
        "    movq r11, xmm9",
        "    xor r11, rdi",
        "    or r10, r11",
        "    pextrq r11, xmm9, 1",
        "    xor r11, rdi",
        "    or r10, r11",
        "    movq r11, xmm10",
        "    xor r11, r12",
        "    or r10, r11",
        "    pextrq r11, xmm10, 1",
        "    xor r11, r12",
        "    or r10, r11",
        "    movq r11, xmm11",
        "    xor r11, r13",
        "    or r10, r11",
        "    pextrq r11, xmm11, 1",
        "    xor r11, r13",
        "    or r10, r11",
        "    movq r11, xmm12",
        "    xor r11, r14",
        "    or r10, r11",
        "    pextrq r11, xmm12, 1",
        "    xor r11, r14",
        "    or r10, r11",
        "    movq r11, xmm13",
        "    xor r11, r15",
        "    or r10, r11",
        "    pextrq r11, xmm13, 1",
        "    xor r11, r15",
        "    or r10, r11",
        "    movq r11, xmm14",
        "    xor r11, rbx",
        "    or r10, r11",
        "    pextrq r11, xmm14, 1",
        "    xor r11, rbx",
        "    or r10, r11",
        "    movq r11, xmm15",
        "    xor r11, rbp",
        "    or r10, r11",
        "    pextrq r11, xmm15, 1",
        "    xor r11, rbp",
        "    or r10, r11",
        // The direction flag (bit 10 of RFLAGS), MXCSR, x87 control word.
        "    pushfq",
        "    pop r11",
        "    and r11, 0x400",
        "    or r10, r11",
        "    stmxcsr dword ptr [rsp + 200]",
        "    fnstcw word ptr [rsp + 204]",
        "    mov r11d, dword ptr [rsp + 200]",
        "    xor r11d, dword ptr [rsp + 192]",
        "    or r10, r11",
        "    movzx r11d, word ptr [rsp + 204]",
        "    movzx r8d, word ptr [rsp + 196]",
        "    xor r11, r8",
        "    or r10, r11",
        "    mov rax, r10",
        "    movaps xmm6, xmmword ptr [rsp + 32]",
        "    movaps xmm7, xmmword ptr [rsp + 48]",
        "    movaps xmm8, xmmword ptr [rsp + 64]",
        "    movaps xmm9, xmmword ptr [rsp + 80]",
        "    movaps xmm10, xmmword ptr [rsp + 96]",
        "    movaps xmm11, xmmword ptr [rsp + 112]",
        "    movaps xmm12, xmmword ptr [rsp + 128]",
        "    movaps xmm13, xmmword ptr [rsp + 144]",
        "    movaps xmm14, xmmword ptr [rsp + 160]",
        "    movaps xmm15, xmmword ptr [rsp + 176]",
        "    add rsp, 216",
        "    pop r15",
        "    pop r14",
        "    pop r13",
        "    pop r12",
        "    pop rsi",
        "    pop rdi",
        "    pop rbp",
        "    pop rbx",
        "    ret",
        ".seh_endproc",
    );

    unsafe extern "C" {
        fn victory_test_clobber_and_fault(junk: *const u32, fault: u64, compare: usize);
        fn victory_test_guarded_registers(call: *mut c_void) -> u64;
    }

    /// The address of ntdll's export `name`, which lies inside the unwinder's
    /// image.
    fn ntdll_export(name: &std::ffi::CStr) -> usize {
        use windows_sys::Win32::System::LibraryLoader::GetProcAddress;
        install_handler();
        // SAFETY: looks up a loaded module and one of its exports by name.
        let export =
            unsafe { GetProcAddress(GetModuleHandleW(w!("ntdll.dll")), name.as_ptr().cast()) };
        let address = (export.unwrap() as *const ()).addr();
        assert!(UNWINDER_IMAGE.get().unwrap().contains(&address));
        address
    }

    #[test]
    fn a_recovered_call_keeps_every_nonvolatile_register() {
        // Round toward zero, in MXCSR and in the x87 control word, unlike the
        // defaults (0x1f80, 0x27f).
        let junk: [u32; 2] = [0x7f80, 0x0c7f];
        let compare = ntdll_export(c"RtlCompareMemoryUlong");
        let mut call = GuardedCall::new(
            (victory_test_clobber_and_fault as *const ()).expose_provenance(),
            [junk.as_ptr().expose_provenance(), 0, compare, 0, 0, 0, 0, 0],
        );
        GUARDED.set(&raw mut call);
        // SAFETY: the routine keeps every nonvolatile register as the ABI
        // wants; the call it makes reads `junk`, then ntdll reads 0 and
        // faults, which the handler recovers while `GUARDED` names the call.
        let differences = unsafe { victory_test_guarded_registers((&raw mut call).cast()) };
        GUARDED.set(null_mut());
        assert_eq!(differences, 0, "{differences:#x}");
        let fault = UnwindFault {
            kind: MemoryFault::AccessViolation,
            address: 0,
        };
        assert_eq!(call.fault, Some(fault));
    }

    /// Gives `recover_unwind_fault` a fault of each of `codes` in turn, raised
    /// at `at` on 0x1000, with `call` as this thread's guarded call when the
    /// first is given and none after the last. Returns, for each, whether it
    /// recovered and `Rip` and `Rsp` after.
    fn recover_as(
        codes: &[i32],
        at: usize,
        call: Option<&mut GuardedCall>,
    ) -> Vec<(bool, u64, u64)> {
        install_handler();
        GUARDED.set(call.map_or(null_mut(), std::ptr::from_mut));
        let dispatched = codes
            .iter()
            .map(|&code| {
                // SAFETY: zeroed records are valid; `recover_unwind_fault`
                // reads them and writes only the context and `call`, which
                // the caller lends for this function's span.
                unsafe {
                    let mut record: EXCEPTION_RECORD = std::mem::zeroed();
                    record.ExceptionCode = code;
                    record.ExceptionAddress = std::ptr::without_provenance_mut(at);
                    record.NumberParameters = 2;
                    record.ExceptionInformation[1] = 0x1000;
                    let mut context: CONTEXT = std::mem::zeroed();
                    context.Rip = 0x1234;
                    context.Rsp = 0x5678;
                    let mut pointers = EXCEPTION_POINTERS {
                        ExceptionRecord: &mut record,
                        ContextRecord: &mut context,
                    };
                    let recovered = recover_unwind_fault(&mut pointers);
                    (recovered, context.Rip, context.Rsp)
                }
            })
            .collect();
        GUARDED.set(null_mut());
        dispatched
    }

    /// An address in the unwinder's image, where its faults are raised.
    fn in_unwinder() -> usize {
        ntdll_export(c"RtlCompareMemoryUlong")
    }

    #[test]
    fn a_fault_outside_a_guarded_call_is_not_recovered() {
        let recovered = recover_as(&[EXCEPTION_ACCESS_VIOLATION], in_unwinder(), None);
        assert_eq!(recovered, [(false, 0x1234, 0x5678)]);
    }

    #[test]
    fn a_fault_that_is_not_a_memory_fault_is_not_recovered() {
        let mut call = GuardedCall::new(0, [0; 8]);
        call.resume_rsp = 0x9000;
        let codes = [
            EXCEPTION_BREAKPOINT,
            EXCEPTION_ILLEGAL_INSTRUCTION,
            EXCEPTION_STACK_OVERFLOW,
        ];
        for code in codes {
            let recovered = recover_as(&[code], in_unwinder(), Some(&mut call));
            assert_eq!(recovered, [(false, 0x1234, 0x5678)]);
        }
        assert_eq!(call.fault, None);
    }

    #[test]
    fn a_memory_fault_raised_outside_the_unwinder_is_not_recovered() {
        let mut call = GuardedCall::new(0, [0; 8]);
        call.resume_rsp = 0x9000;
        // This test's own code, as a function table callback's would be.
        let ours =
            (a_memory_fault_raised_outside_the_unwinder_is_not_recovered as *const ()).addr();
        let recovered = recover_as(&[EXCEPTION_ACCESS_VIOLATION], ours, Some(&mut call));
        assert_eq!(recovered, [(false, 0x1234, 0x5678)]);
        assert_eq!(call.fault, None);
    }

    #[test]
    fn a_memory_fault_in_a_guarded_call_resumes_the_trampoline() {
        let mut call = GuardedCall::new(0, [0; 8]);
        call.resume_rsp = 0x9000;
        let recovered = recover_as(&[STATUS_IN_PAGE_ERROR], in_unwinder(), Some(&mut call));
        let label = (&raw const GUARDED_CALL_RECOVERED).addr() as u64;
        assert_eq!(recovered, [(true, label, 0x9000)]);
        let fault = UnwindFault {
            kind: MemoryFault::InPage,
            address: 0x1000,
        };
        assert_eq!(call.fault, Some(fault));
    }

    #[test]
    fn an_exception_declined_inside_a_guarded_call_ends_its_recovery() {
        let mut call = GuardedCall::new(0, [0; 8]);
        call.resume_rsp = 0x9000;
        // An exception the handler declines, then a memory fault in the unwinder.
        let codes = [EXCEPTION_BREAKPOINT, EXCEPTION_ACCESS_VIOLATION];
        let dispatched = recover_as(&codes, in_unwinder(), Some(&mut call));
        assert_eq!(dispatched, [(false, 0x1234, 0x5678); 2]);
        assert_eq!(call.fault, None);
    }

    #[test]
    fn a_code_that_is_not_a_memory_fault_does_not_convert() {
        let converted = MemoryFault::try_from(EXCEPTION_BREAKPOINT);
        assert_eq!(converted, Err(EXCEPTION_BREAKPOINT));
    }

    #[test]
    fn the_trampoline_has_unwind_data_listing_its_saves() {
        let trampoline = (victory_guarded_call as *const ()).addr() as u64;
        let mut image = 0;
        // SAFETY: looks up the entry covering the trampoline; its unwind info
        // is mapped with this test binary.
        let codes: Vec<(u8, u8, Option<u16>)> = unsafe {
            let entry = RtlLookupFunctionEntry(trampoline, &mut image, null_mut());
            assert!(!entry.is_null());
            assert_eq!(image + u64::from((*entry).BeginAddress), trampoline);
            let info = (image + u64::from((*entry).Anonymous.UnwindData)) as *const u8;
            let [version, _prolog, count, frame] = info.cast::<[u8; 4]>().read();
            assert_eq!((version, frame), (1, 0));
            let slots = std::slice::from_raw_parts(info.add(4).cast::<[u8; 2]>(), count.into());
            // Each code's operation, register and the slot after it, which
            // UWOP_ALLOC_LARGE (size / 8) and UWOP_SAVE_XMM128 (offset / 16)
            // take; UWOP_PUSH_NONVOL takes none.
            let mut codes = Vec::new();
            let mut i = 0;
            while let Some(&[_, op_info]) = slots.get(i) {
                let (op, info) = (op_info & 15, op_info >> 4);
                let operand = match op {
                    0 => None,
                    1 | 8 => slots.get(i + 1).map(|&slot| u16::from_le_bytes(slot)),
                    _ => panic!("unexpected unwind code {op}"),
                };
                codes.push((op, info, operand));
                i += 1 + usize::from(operand.is_some());
            }
            codes
        };
        // Last saved first: xmm15 to xmm6 at 208 to 64 (scaled by 16), the
        // 232-byte allocation (scaled by 8), then r15 to rbx.
        let expected = [
            (8, 15, Some(13)),
            (8, 14, Some(12)),
            (8, 13, Some(11)),
            (8, 12, Some(10)),
            (8, 11, Some(9)),
            (8, 10, Some(8)),
            (8, 9, Some(7)),
            (8, 8, Some(6)),
            (8, 7, Some(5)),
            (8, 6, Some(4)),
            (1, 0, Some(29)),
            (0, 15, None),
            (0, 14, None),
            (0, 13, None),
            (0, 12, None),
            (0, 6, None),
            (0, 7, None),
            (0, 5, None),
            (0, 3, None),
        ];
        assert_eq!(codes, expected);
    }

    /// One walk whose `RtlVirtualUnwind` faults reading through rbp, from the
    /// body of a framed function.
    fn faulting_walk_in_a_body() -> Result<(), UnwindFault> {
        let function = Synthetic {
            code: &FRAMED_BODY,
            unwind_info: &FRAMED_INFO,
            functions: &[[0, 5, 16]],
        };
        walk_synthetic(&function, 4, &[0x1234, 0], 0x1000).1
    }

    /// One walk from a direct `jmp` into a chained fragment of the same
    /// function, where `RtlVirtualUnwind` looks the target up itself and then
    /// faults reading through rbp.
    fn faulting_walk_at_a_jump() -> Result<(), UnwindFault> {
        let function = Synthetic {
            // push rbp; mov rbp, rsp; jmp to the fragment | pop rbp; ret
            code: &[0x55, 0x48, 0x89, 0xe5, 0xe9, 0, 0, 0, 0, 0x5d, 0xc3],
            // The primary's `FRAMED_INFO`, then the fragment's: version 1
            // with `UNW_FLAG_CHAININFO`, no prolog, no codes, frame register
            // rbp, and the primary's entry `{0, 9, 16}`.
            unwind_info: &[
                0x01, 4, 2, 0x05, 4, 0x03, 1, 0x50, // primary
                0x21, 0, 0, 0x05, 0, 0, 0, 0, 9, 0, 0, 0, 16, 0, 0, 0, // fragment
            ],
            functions: &[[0, 9, 16], [9, 11, 24]],
        };
        walk_synthetic(&function, 4, &[0x1234, 0], 0x1000).1
    }

    #[test]
    fn recovered_faults_leave_function_tables_and_unwinding_usable() {
        let fault = UnwindFault {
            kind: MemoryFault::AccessViolation,
            address: 0x1000,
        };
        for _ in 0..5000 {
            assert_eq!(faulting_walk_in_a_body(), Err(fault));
            assert_eq!(faulting_walk_at_a_jump(), Err(fault));
        }
        // Function tables and unwinding a panic take the locks a mid-call
        // abandon could have left held; on another thread, so a lock this
        // thread still held would block them.
        let (done, finished) = std::sync::mpsc::channel();
        std::thread::spawn(move || {
            let function = Synthetic {
                code: &FRAMED_BODY,
                unwind_info: &FRAMED_INFO,
                functions: &[[0, 5, 16]],
            };
            // The saved rbp, then the return address.
            let stack = [0x5678, 0x1234, 0];
            let (frames, walked, rip) = walk_synthetic(&function, 4, &stack, stack.as_ptr() as u64);
            let caught = std::panic::catch_unwind(|| std::panic::resume_unwind(Box::new(())));
            done.send((frames == [rip, 0x1234], walked, caught.is_err()))
                .unwrap();
        });
        let outcome = finished.recv_timeout(Duration::from_secs(5));
        assert_eq!(outcome, Ok((true, Ok(()), true)));
    }
}
