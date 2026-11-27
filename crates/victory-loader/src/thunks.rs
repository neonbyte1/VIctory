//! Stubs for imports nobody implements yet.
//!
//! Each unresolved import gets 32 bytes of generated code:
//!
//! ```text
//! mov rcx, <&'static ImportSite>   ; 48 B9 imm64
//! mov rax, <unimplemented_import>  ; 48 B8 imm64
//! jmp rax                          ; FF E0
//! ```
//!
//! The jump keeps the title's return address on the stack, so the handler
//! returns straight to the caller. Bring-up then shows exactly which API to
//! implement next, rather than failing at load time.

use std::sync::atomic::{AtomicU64, Ordering};

use anyhow::{Result, bail};
use windows_sys::Win32::System::Diagnostics::Debug::{
    DebugBreak, FlushInstructionCache, IsDebuggerPresent,
};
use windows_sys::Win32::System::Memory::{
    MEM_COMMIT, MEM_RESERVE, PAGE_EXECUTE_READ, PAGE_READWRITE, VirtualAlloc, VirtualProtect,
};
use windows_sys::Win32::System::Threading::GetCurrentProcess;

/// What an unimplemented import does when called.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum UnimplementedPolicy {
    /// Log, break into an attached debugger, then abort. The default:
    /// silently continuing usually crashes later and further from the cause.
    #[default]
    Trap,
    /// Log once per import and return 0. On Windows x86-64 this clears both
    /// integer and floating-point return registers. Useful during bring-up.
    ReturnZero,
}

/// Identifies one unresolved import. Leaked: lives as long as the process.
#[derive(Debug)]
pub struct ImportSite {
    pub dll: String,
    pub symbol: String,
    pub policy: UnimplementedPolicy,
    pub calls: AtomicU64,
}

static UNIMPLEMENTED_CALLS: AtomicU64 = AtomicU64::new(0);

/// Total number of calls that reached an unimplemented-import stub.
pub fn unimplemented_call_count() -> u64 {
    UNIMPLEMENTED_CALLS.load(Ordering::Relaxed)
}

extern "system" fn handle_unimplemented_import(site: &'static ImportSite) -> u64 {
    UNIMPLEMENTED_CALLS.fetch_add(1, Ordering::Relaxed);
    let previous = site.calls.fetch_add(1, Ordering::Relaxed);
    match site.policy {
        UnimplementedPolicy::ReturnZero => {
            if previous == 0 {
                tracing::warn!(dll = %site.dll, symbol = %site.symbol, "unimplemented import called, returning 0");
            }
            0
        }
        UnimplementedPolicy::Trap => {
            tracing::error!(dll = %site.dll, symbol = %site.symbol, "unimplemented import called");
            // Into the crash log too: `abort` ends the process without
            // running the handlers that write it.
            let (dll, symbol) = (&site.dll, &site.symbol);
            let text = format!("VIctory: unimplemented import {dll}!{symbol} called");
            crate::log_crash(&text);
            // SAFETY: plain Win32 calls without preconditions.
            unsafe {
                if IsDebuggerPresent() != 0 {
                    DebugBreak();
                }
            }
            std::process::abort();
        }
    }
}

// Reserve 32 bytes of Win64 shadow space and 8 bytes to align RSP before the
// call. The handler returns zero in RAX, and XORPS clears XMM0 without touching it.
#[cfg(all(windows, target_arch = "x86_64"))]
std::arch::global_asm!(
    ".text",
    ".globl victory_loader_return_zero",
    ".seh_proc victory_loader_return_zero",
    "victory_loader_return_zero:",
    "sub rsp, 40",
    ".seh_stackalloc 40",
    ".seh_endprologue",
    "call {handler}",
    "xorps xmm0, xmm0",
    "add rsp, 40",
    "ret",
    ".seh_endproc",
    handler = sym handle_unimplemented_import,
);

#[cfg(all(windows, target_arch = "x86_64"))]
unsafe extern "system" {
    #[link_name = "victory_loader_return_zero"]
    fn unimplemented_import(site: *const std::ffi::c_void) -> u64;
}

const THUNK_SIZE: usize = 32;

/// Executable memory holding one stub per unresolved import.
pub(crate) struct ThunkBlock {
    base: *mut u8,
    len: usize,
    used: usize,
}

impl ThunkBlock {
    /// Reserves room for `count` stubs (writable until [`Self::seal`]).
    pub(crate) fn new(count: usize) -> Result<Self> {
        let len = (count.max(1) * THUNK_SIZE).next_multiple_of(4096);
        // SAFETY: fresh anonymous allocation; no existing memory is touched.
        let base = unsafe {
            VirtualAlloc(
                std::ptr::null(),
                len,
                MEM_COMMIT | MEM_RESERVE,
                PAGE_READWRITE,
            )
        } as *mut u8;
        if base.is_null() {
            bail!(
                "VirtualAlloc for import stubs failed: {}",
                std::io::Error::last_os_error()
            );
        }
        Ok(ThunkBlock { base, len, used: 0 })
    }

    /// Emits a stub for `site` and returns its address.
    pub(crate) fn emit(&mut self, site: &'static ImportSite) -> Result<usize> {
        if self.used + THUNK_SIZE > self.len {
            bail!("import stub block is full");
        }
        let mut code = [0xCC_u8; THUNK_SIZE]; // int3 padding
        code[0..2].copy_from_slice(&[0x48, 0xB9]);
        code[2..10].copy_from_slice(&(site as *const ImportSite as u64).to_le_bytes());
        code[10..12].copy_from_slice(&[0x48, 0xB8]);
        let handler = handle_unimplemented_import as *const ();
        #[cfg(all(windows, target_arch = "x86_64"))]
        let handler = if site.policy == UnimplementedPolicy::ReturnZero {
            unimplemented_import as *const ()
        } else {
            handler
        };
        code[12..20].copy_from_slice(&(handler as u64).to_le_bytes());
        code[20..22].copy_from_slice(&[0xFF, 0xE0]);
        // SAFETY: `used + THUNK_SIZE <= len`, and the block is still writable.
        let at = unsafe {
            let at = self.base.add(self.used);
            std::ptr::copy_nonoverlapping(code.as_ptr(), at, THUNK_SIZE);
            at
        };
        self.used += THUNK_SIZE;
        Ok(at as usize)
    }

    /// Makes the block read+execute. The block is intentionally leaked:
    /// title code may hold these addresses for the life of the process.
    pub(crate) fn seal(self) -> Result<()> {
        let mut old = 0;
        // SAFETY: `base..base+len` is the allocation made in `new`.
        let ok = unsafe { VirtualProtect(self.base as _, self.len, PAGE_EXECUTE_READ, &mut old) };
        if ok == 0 {
            bail!(
                "VirtualProtect for import stubs failed: {}",
                std::io::Error::last_os_error()
            );
        }
        // SAFETY: flushing our own process's instruction cache for our own block.
        unsafe { FlushInstructionCache(GetCurrentProcess(), self.base as _, self.len) };
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// In a child process of the next test: calls an unimplemented import
    /// that traps, with the crash log where `VICTORY_TRAP_LOG` says.
    #[test]
    fn trap_child() {
        let Some(log) = std::env::var_os("VICTORY_TRAP_LOG") else {
            return;
        };
        crate::log_crashes_to(log.into());
        let site = Box::leak(Box::new(ImportSite {
            dll: "probe.dll".into(),
            symbol: "Missing".into(),
            policy: UnimplementedPolicy::Trap,
            calls: AtomicU64::new(0),
        }));
        handle_unimplemented_import(site);
    }

    #[test]
    fn a_trapped_import_is_in_the_crash_log() {
        let dir = tempfile::tempdir().unwrap();
        let log = dir.path().join("crashes.log");
        let out = std::process::Command::new(std::env::current_exe().unwrap())
            .args(["--exact", "thunks::tests::trap_child", "--nocapture"])
            .env("VICTORY_TRAP_LOG", &log)
            .output()
            .unwrap();
        assert!(!out.status.success(), "the trap ends the process");
        let text = std::fs::read_to_string(&log).unwrap_or_default();
        let named = text.contains("VIctory: unimplemented import probe.dll!Missing called");
        assert!(named, "crash log: {text:?}");
    }

    #[cfg(all(windows, target_arch = "x86_64"))]
    #[test]
    fn return_zero_clears_integer_and_float_returns() {
        // Isolate the process-wide counter from other tests that call thunks.
        const CHILD: &str = "VICTORY_THUNK_RETURN_ZERO_CHILD";
        if std::env::var_os(CHILD).is_none() {
            let output = std::process::Command::new(std::env::current_exe().unwrap())
                .args([
                    "--exact",
                    "thunks::tests::return_zero_clears_integer_and_float_returns",
                ])
                .env(CHILD, "1")
                .output()
                .unwrap();
            assert!(
                output.status.success(),
                "child exited with {}\n{}\n{}",
                output.status,
                String::from_utf8_lossy(&output.stdout),
                String::from_utf8_lossy(&output.stderr),
            );
            return;
        }

        let site = Box::leak(Box::new(ImportSite {
            dll: "probe.dll".into(),
            symbol: "ReturnZero".into(),
            policy: UnimplementedPolicy::ReturnZero,
            calls: AtomicU64::new(0),
        }));
        let mut block = ThunkBlock::new(1).unwrap();
        let address = block.emit(site).unwrap();
        block.seal().unwrap();

        // SAFETY: `address` points to the sealed Win64 thunk. It ignores its
        // caller's arguments and provides zero in both scalar return registers.
        let (call_u64, call_f64, call_f32) = unsafe {
            (
                std::mem::transmute::<usize, extern "system" fn(u64) -> u64>(address),
                std::mem::transmute::<usize, extern "system" fn(f64) -> f64>(address),
                std::mem::transmute::<usize, extern "system" fn(f32) -> f32>(address),
            )
        };

        // The first call takes the logging path. Both float calls then enter
        // the handler with a nonzero value in XMM0.
        assert_eq!(call_u64(u64::MAX), 0);
        assert_eq!(site.calls.load(Ordering::Relaxed), 1);
        assert_eq!(unimplemented_call_count(), 1);

        let f64_bits = call_f64(1.25).to_bits();
        assert_eq!(site.calls.load(Ordering::Relaxed), 2);
        assert_eq!(unimplemented_call_count(), 2);

        let f32_bits = call_f32(-2.5).to_bits();
        assert_eq!(site.calls.load(Ordering::Relaxed), 3);
        assert_eq!(unimplemented_call_count(), 3);
        assert_eq!((f64_bits, f32_bits), (0, 0));
    }
}
