//! Loading a prepared image as a real Windows module and binding it.

use std::collections::HashSet;
use std::ffi::{CString, c_void};
use std::os::windows::ffi::OsStrExt;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::AtomicU64;

use anyhow::{Context, Result, bail};
use victory_pe::{ImageInfo, ImportSymbol};
use windows_sys::Win32::Foundation::{FreeLibrary, HMODULE};
use windows_sys::Win32::System::LibraryLoader::{GetProcAddress, LoadLibraryExW};
use windows_sys::Win32::System::Memory::{PAGE_READWRITE, VirtualProtect};

use crate::thunks::{ImportSite, ThunkBlock, UnimplementedPolicy};
use crate::{
    DLL_PROCESS_ATTACH, DLL_PROCESS_DETACH, ImportResolver, PreparedImage, Provenance, Resolution,
    prepare,
};

/// A title image mapped into this process. A clone names the same mapping
/// (no reference of its own): unload one copy only.
#[derive(Debug, Clone)]
pub struct LoadedImage {
    module: HMODULE,
    path: PathBuf,
    info: ImageInfo,
}

// SAFETY: HMODULE is a process-wide handle (the base address); using it from
// any thread is fine.
unsafe impl Send for LoadedImage {}
// SAFETY: as above; `LoadedImage` has no interior mutability.
unsafe impl Sync for LoadedImage {}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MissingImport {
    pub dll: String,
    pub symbol: String,
}

/// Outcome of [`LoadedImage::bind_imports`].
#[derive(Debug, Default, Clone)]
pub struct BindReport {
    pub hle: usize,
    pub host: usize,
    pub title: usize,
    pub missing: Vec<MissingImport>,
}

/// Writes the prepared copy into `dir` and maps it with the Windows loader.
/// `original_name` is the title's own exe/DLL file name, used by diagnostics
/// rather than the content-hashed name of the prepared copy.
///
/// `dir` receives derived game data, so it must be outside the repository.
pub fn load(prepared: &PreparedImage, dir: &Path, original_name: &str) -> Result<LoadedImage> {
    #[cfg(not(feature = "diagnostics"))]
    let _ = original_name;
    victory_core::ensure_outside_repo(dir)?;
    std::fs::create_dir_all(dir).with_context(|| format!("creating {}", dir.display()))?;
    let path = dir.join(prepared.file_name());
    write_if_changed(&path, &prepared.bytes)?;

    let wide: Vec<u16> = path.as_os_str().encode_wide().chain(Some(0)).collect();
    // SAFETY: `wide` is a NUL-terminated path. The image has no entry point
    // and no imports, so loading it runs no title code.
    let module = unsafe { LoadLibraryExW(wide.as_ptr(), std::ptr::null_mut(), 0) };
    if module.is_null() {
        bail!(
            "LoadLibraryExW({}) failed: {}{}",
            path.display(),
            std::io::Error::last_os_error(),
            if prepared.info.has_relocations {
                ""
            } else {
                " (image has no relocations; its preferred base may be taken)"
            }
        );
    }
    let loaded = LoadedImage {
        module,
        path,
        info: prepared.info.clone(),
    };
    // Debugging aid (README): VICTORY_PATCH entries naming this module write
    // their bytes now — mapped and relocated, before TLS redirection and
    // import binding — and are logged with the bytes they replaced.
    #[cfg(all(windows, feature = "diagnostics"))]
    crate::patch::apply(
        loaded.base(),
        loaded.info.size_of_image as usize,
        original_name,
    );
    // A DLL indexes its TLS properly; only an exe assumes index 0.
    if !loaded.info.is_dll() {
        // SAFETY: just mapped, and no title code has run.
        if let Err(error) = unsafe { crate::tls::redirect(loaded.base(), &loaded.info) } {
            // Nothing has seen this module yet (`threads::mapped` is below),
            // so a failed load must not leave it mapped for the life of the
            // process.
            // SAFETY: no title code has run and the module is not registered.
            unsafe { FreeLibrary(module) };
            return Err(error);
        }
    }
    tracing::info!(
        base = format_args!("{:#x}", loaded.base()),
        preferred = format_args!("{:#x}", loaded.info.image_base),
        path = %loaded.path.display(),
        "image mapped"
    );
    crate::threads::mapped(&loaded);
    Ok(loaded)
}

/// Maps the DLLs the title ships in `title_dir` that `main` imports,
/// directly or through each other, in initialisation order (dependencies
/// first). Only DLL names `wanted` accepts are considered, so DLLs the host
/// or our runtime provide stay with them. Imports are not bound.
pub fn load_title_dlls(
    main: &ImageInfo,
    title_dir: &Path,
    cache_dir: &Path,
    wanted: &dyn Fn(&str) -> bool,
) -> Result<Vec<(String, LoadedImage)>> {
    let mut loaded = Vec::new();
    let dirs = (title_dir, cache_dir);
    map_imports(main, dirs, wanted, &mut HashSet::new(), &mut loaded)?;
    Ok(loaded)
}

/// Maps the DLL `name` from `title_dir` like [`load_title_dlls`] does, with
/// the title DLLs it imports (dependencies first, `name` last); empty if
/// `wanted` refuses it or the folder has no such file. For DLLs a title
/// loads itself; `wanted` must refuse those already mapped.
pub fn load_title_dll(
    name: &str,
    title_dir: &Path,
    cache_dir: &Path,
    wanted: &dyn Fn(&str) -> bool,
) -> Result<Vec<(String, LoadedImage)>> {
    let mut loaded = Vec::new();
    let dirs = (title_dir, cache_dir);
    map_dll(name, dirs, wanted, &mut HashSet::new(), &mut loaded)?;
    Ok(loaded)
}

fn map_imports(
    info: &ImageInfo,
    dirs: (&Path, &Path),
    wanted: &dyn Fn(&str) -> bool,
    seen: &mut HashSet<String>,
    loaded: &mut Vec<(String, LoadedImage)>,
) -> Result<()> {
    for module in info.imports.iter().chain(&info.delay_imports) {
        map_dll(&module.dll, dirs, wanted, seen, loaded)?;
    }
    Ok(())
}

fn map_dll(
    name: &str,
    dirs: (&Path, &Path),
    wanted: &dyn Fn(&str) -> bool,
    seen: &mut HashSet<String>,
    loaded: &mut Vec<(String, LoadedImage)>,
) -> Result<()> {
    let name = name.to_ascii_lowercase();
    let path = dirs.0.join(&name);
    if !seen.insert(name.clone()) || !wanted(&name) || !path.is_file() {
        return Ok(());
    }
    let original = std::fs::read(&path).with_context(|| format!("reading {}", path.display()))?;
    let prepared = prepare(&original).with_context(|| format!("preparing {}", path.display()))?;
    let image =
        load(&prepared, dirs.1, &name).with_context(|| format!("loading {}", path.display()))?;
    map_imports(&image.info, dirs, wanted, seen, loaded)?;
    loaded.push((name, image));
    Ok(())
}

/// Resolves through `inner`, then through the exports of the title's DLLs.
pub struct WithTitleDlls<'a> {
    pub inner: &'a dyn ImportResolver,
    pub dlls: &'a [(String, LoadedImage)],
}

impl ImportResolver for WithTitleDlls<'_> {
    fn resolve(&self, dll: &str, symbol: &ImportSymbol) -> Resolution {
        let found = self.inner.resolve(dll, symbol);
        if found != Resolution::Missing {
            return found;
        }
        self.dlls
            .iter()
            .find(|(name, _)| name.eq_ignore_ascii_case(dll))
            .and_then(|(_, image)| image.export_symbol(symbol))
            .map_or(Resolution::Missing, |address| Resolution::Found {
                address,
                provenance: Provenance::Title,
            })
    }
}

fn write_if_changed(path: &Path, bytes: &[u8]) -> Result<()> {
    crate::publish::write_if_changed(path, bytes)
        .with_context(|| format!("publishing {}", path.display()))
}

impl LoadedImage {
    pub fn base(&self) -> usize {
        self.module as usize
    }

    /// Parsed from the original image (imports, entry point, TLS callbacks).
    pub fn info(&self) -> &ImageInfo {
        &self.info
    }

    /// Address of an export of the image, if it has one by that name.
    pub fn export(&self, name: &str) -> Option<usize> {
        let name = CString::new(name).ok()?;
        // SAFETY: valid module handle and NUL-terminated name.
        unsafe { GetProcAddress(self.module, name.as_ptr().cast()) }
            .map(|f| f as *const () as usize)
    }

    /// Address of an export, by name or ordinal.
    pub fn export_symbol(&self, symbol: &ImportSymbol) -> Option<usize> {
        match symbol {
            ImportSymbol::Name { name, .. } => self.export(name),
            // SAFETY: MAKEINTRESOURCE-style ordinal lookup on a valid module.
            ImportSymbol::Ordinal(o) => unsafe { GetProcAddress(self.module, *o as usize as _) }
                .map(|f| f as *const () as usize),
        }
    }

    /// Fills every IAT slot of the image.
    ///
    /// Each import goes to `resolver`; whatever it cannot resolve gets an
    /// unimplemented-import stub that behaves according to `policy`.
    pub fn bind_imports(
        &self,
        resolver: &dyn ImportResolver,
        policy: UnimplementedPolicy,
    ) -> Result<BindReport> {
        let mut report = BindReport::default();
        let mut slots: Vec<(u32, usize)> = Vec::with_capacity(self.info.import_count());
        let mut unresolved: Vec<(u32, &str, &ImportSymbol)> = Vec::new();

        // Delay-load slots are bound eagerly too, so the title's own
        // delay-load helper (which would ask the host) never runs.
        for module in self.info.imports.iter().chain(&self.info.delay_imports) {
            for entry in &module.entries {
                match resolver.resolve(&module.dll, &entry.symbol) {
                    Resolution::Found {
                        address,
                        provenance,
                    } => {
                        match provenance {
                            Provenance::Hle => report.hle += 1,
                            Provenance::Host => report.host += 1,
                            Provenance::Title => report.title += 1,
                        }
                        slots.push((entry.iat_slot_rva, address));
                    }
                    Resolution::Missing => {
                        unresolved.push((entry.iat_slot_rva, &module.dll, &entry.symbol));
                    }
                }
            }
        }

        if !unresolved.is_empty() {
            let mut block = ThunkBlock::new(unresolved.len())?;
            for (slot, dll, symbol) in unresolved {
                let site: &'static ImportSite = Box::leak(Box::new(ImportSite {
                    dll: dll.to_owned(),
                    symbol: symbol.to_string(),
                    policy,
                    calls: AtomicU64::new(0),
                }));
                slots.push((slot, block.emit(site)?));
                report.missing.push(MissingImport {
                    dll: site.dll.clone(),
                    symbol: site.symbol.clone(),
                });
            }
            block.seal()?;
        }

        self.write_slots(&slots)?;
        tracing::info!(
            hle = report.hle,
            host = report.host,
            missing = report.missing.len(),
            "imports bound"
        );
        Ok(report)
    }

    /// Protects and restores each slot on its own: import tables can sit in
    /// sections with different protections (delay-load tables in `.didat`).
    fn write_slots(&self, slots: &[(u32, usize)]) -> Result<()> {
        for &(rva, address) in slots {
            let slot = (self.base() + rva as usize) as *mut u64;
            let mut old = 0;
            // SAFETY: the parser only yields slots inside the image.
            if unsafe { VirtualProtect(slot as *const c_void, 8, PAGE_READWRITE, &mut old) } == 0 {
                bail!(
                    "VirtualProtect(IAT slot {rva:#x}) failed: {}",
                    std::io::Error::last_os_error()
                );
            }
            // SAFETY: inside the image, now writable, 8-byte aligned by the PE format.
            unsafe { slot.write_volatile(address as u64) };
            let mut ignored = 0;
            // SAFETY: restoring the protection we just changed.
            unsafe { VirtualProtect(slot as *const c_void, 8, old, &mut ignored) };
        }
        Ok(())
    }

    /// Calls the image's TLS callbacks with `reason` (e.g. `DLL_PROCESS_ATTACH`).
    ///
    /// # Safety
    /// Runs title code. Imports must be bound first.
    pub unsafe fn run_tls_callbacks(&self, reason: u32) {
        // SAFETY: forwarded from the caller.
        unsafe { self.tls_callbacks(reason, std::ptr::null_mut()) }
    }

    /// # Safety
    /// See [`Self::run_tls_callbacks`].
    unsafe fn tls_callbacks(&self, reason: u32, reserved: *mut c_void) {
        let Some(tls) = &self.info.tls else { return };
        for &rva in &tls.callback_rvas {
            type TlsCallback = unsafe extern "system" fn(*mut c_void, u32, *mut c_void);
            // SAFETY: the RVA came from the image's own TLS directory.
            let callback: TlsCallback = unsafe { std::mem::transmute(self.base() + rva as usize) };
            // SAFETY: the caller guarantees imports are bound.
            unsafe { callback(self.module, reason, reserved) };
        }
    }

    /// Initialises a DLL as the Windows loader would: its TLS callbacks,
    /// then `DllMain(DLL_PROCESS_ATTACH)`. From then on it hears of every
    /// thread start and end (see `threads`).
    ///
    /// # Safety
    /// Runs title code. Imports must be bound first, and the DLLs this one
    /// depends on attached.
    pub unsafe fn attach(&self) -> Result<()> {
        // SAFETY: forwarded from the caller.
        if unsafe { self.notify(DLL_PROCESS_ATTACH, true, std::ptr::null_mut()) } == Some(0) {
            bail!("DllMain returned FALSE");
        }
        crate::threads::attached(self.base());
        Ok(())
    }

    /// `DLL_PROCESS_DETACH` as `FreeLibrary` sends it (TLS callbacks, then
    /// `DllMain`); the image hears of no more threads.
    ///
    /// # Safety
    /// Runs title code. The image must have been attached (or its attach
    /// failed), and no other thread may be using it.
    pub unsafe fn detach(&self) {
        crate::threads::detached(self.base());
        // SAFETY: forwarded from the caller.
        unsafe { self.notify(DLL_PROCESS_DETACH, true, std::ptr::null_mut()) };
    }

    /// Runs the image's TLS callbacks with `reason`, then, if `dll_main` and
    /// the image is a DLL with an entry point, its `DllMain`, whose result
    /// is returned. `reserved` is the entry points' third argument.
    ///
    /// # Safety
    /// Runs title code. Imports must be bound first.
    pub(crate) unsafe fn notify(
        &self,
        reason: u32,
        dll_main: bool,
        reserved: *mut c_void,
    ) -> Option<i32> {
        // SAFETY: forwarded from the caller.
        unsafe { self.tls_callbacks(reason, reserved) };
        if !dll_main || !self.info.is_dll() || self.info.entry_rva == 0 {
            return None;
        }
        type DllMain = unsafe extern "system" fn(*mut c_void, u32, *mut c_void) -> i32;
        // SAFETY: the entry RVA came from the original image header.
        let main: DllMain =
            unsafe { std::mem::transmute(self.base() + self.info.entry_rva as usize) };
        // SAFETY: forwarded from the caller.
        Some(unsafe { main(self.module, reason, reserved) })
    }

    /// Starts the title: on a new thread with the stack size from the image
    /// header, attaches `dlls` in order (see [`Self::attach`]), runs the
    /// image's TLS callbacks (`DLL_PROCESS_ATTACH`), then its original entry
    /// point. Returns the thread so the caller can join it.
    ///
    /// # Safety
    /// Runs title code. Imports of the image and of `dlls` must be bound.
    pub unsafe fn spawn_main(
        self: Arc<Self>,
        dlls: Vec<(String, LoadedImage)>,
    ) -> Result<std::thread::JoinHandle<Result<u32>>> {
        if self.info.entry_rva == 0 {
            bail!("image has no entry point");
        }
        let stack = usize::try_from(self.info.stack_reserve)
            .unwrap_or(usize::MAX)
            .max(1 << 20);
        std::thread::Builder::new()
            .name("title-main".into())
            .stack_size(stack)
            .spawn(move || {
                for (name, dll) in &dlls {
                    // SAFETY: forwarded from the caller of `spawn_main`.
                    unsafe { dll.attach() }.with_context(|| format!("attaching {name}"))?;
                }
                // SAFETY: forwarded from the caller of `spawn_main`.
                unsafe { self.run_tls_callbacks(DLL_PROCESS_ATTACH) };
                crate::threads::attached(self.base());
                type Entry = unsafe extern "system" fn(*mut c_void) -> u32;
                // SAFETY: the entry RVA came from the original image header.
                let entry: Entry =
                    unsafe { std::mem::transmute(self.base() + self.info.entry_rva as usize) };
                // SAFETY: forwarded from the caller of `spawn_main`.
                Ok(unsafe { entry(std::ptr::null_mut()) })
            })
            .context("spawning title main thread")
    }

    /// Unmaps the image.
    ///
    /// # Safety
    /// No title code may be running or run later, and no pointer into the
    /// image may be used afterwards.
    pub unsafe fn unload(self) {
        crate::threads::unmapped(self.base());
        // SAFETY: guaranteed by the caller.
        unsafe { FreeLibrary(self.module) };
    }
}

#[cfg(test)]
mod tests {
    use victory_pe::{ImportEntry, ImportModule, ImportSymbol};
    use windows_sys::Win32::System::LibraryLoader::GetModuleHandleW;
    use windows_sys::Win32::System::Memory::{
        MEM_COMMIT, MEM_RELEASE, MEM_RESERVE, MEMORY_BASIC_INFORMATION, PAGE_READONLY,
        PAGE_READWRITE, VirtualAlloc, VirtualFree, VirtualQuery,
    };

    use super::*;
    use crate::{disable_thread_calls, unimplemented_call_count};

    const PAGE: usize = 0x1000;

    /// A committed scratch block standing in for a mapped image.
    fn alloc(pages: usize, protect: u32) -> usize {
        // SAFETY: a fresh anonymous allocation, freed once by `free`.
        let at = unsafe {
            VirtualAlloc(
                std::ptr::null(),
                pages * PAGE,
                MEM_COMMIT | MEM_RESERVE,
                protect,
            )
        } as usize;
        assert_ne!(at, 0, "VirtualAlloc failed");
        at
    }

    fn free(at: usize) {
        // SAFETY: `at` is a live block from `alloc`, freed once.
        unsafe { VirtualFree(at as *mut c_void, 0, MEM_RELEASE) };
    }

    fn protection(at: usize) -> u32 {
        // SAFETY: a zeroed output buffer of the type the call fills.
        let mut info: MEMORY_BASIC_INFORMATION = unsafe { std::mem::zeroed() };
        // SAFETY: `at` is inside a live allocation; `info` is sized for the call.
        let n = unsafe { VirtualQuery(at as *const c_void, &mut info, size_of_val(&info)) };
        assert_ne!(n, 0, "VirtualQuery failed");
        info.Protect
    }

    fn import(dll: &str, rva: u32) -> ImportModule {
        ImportModule {
            dll: dll.to_owned(),
            entries: vec![ImportEntry {
                symbol: ImportSymbol::Name {
                    hint: 0,
                    name: format!("f{rva:x}"),
                },
                iat_slot_rva: rva,
            }],
        }
    }

    fn info(size_of_image: u32) -> ImageInfo {
        ImageInfo {
            machine: victory_pe::MACHINE_AMD64,
            characteristics: victory_pe::layout::file_flags::DLL,
            timestamp: 0,
            subsystem: victory_pe::layout::subsystem::WINDOWS_CUI,
            dll_characteristics: 0,
            image_base: 0,
            size_of_image,
            entry_rva: 0,
            stack_reserve: 0,
            stack_commit: 0,
            sections: Vec::new(),
            imports: Vec::new(),
            delay_imports: Vec::new(),
            exports: Vec::new(),
            tls: None,
            has_relocations: false,
            has_exception_table: false,
            has_load_config: false,
            codeview: None,
        }
    }

    /// A `LoadedImage` whose `module` is a scratch allocation, not a real PE.
    fn image_at(base: usize, info: ImageInfo) -> LoadedImage {
        LoadedImage {
            module: base as *mut c_void,
            path: PathBuf::new(),
            info,
        }
    }

    fn slot(image: &LoadedImage, rva: u32) -> u64 {
        // SAFETY: every slot the tests read lies inside a live allocation.
        unsafe { ((image.base() + rva as usize) as *const u64).read_volatile() }
    }

    fn found(address: usize) -> Resolution {
        Resolution::Found {
            address,
            provenance: Provenance::Hle,
        }
    }

    fn prepared_own() -> PreparedImage {
        let original = std::fs::read(std::env::current_exe().unwrap()).unwrap();
        prepare(&original).unwrap()
    }

    fn module_by_name(name: &str) -> usize {
        let wide: Vec<u16> = name.encode_utf16().chain(Some(0)).collect();
        // SAFETY: a NUL-terminated UTF-16 name; the query takes no reference.
        unsafe { GetModuleHandleW(wide.as_ptr()) as usize }
    }

    struct Fixed(Resolution);

    impl ImportResolver for Fixed {
        fn resolve(&self, _: &str, _: &ImportSymbol) -> Resolution {
            self.0
        }
    }

    /// Provenance per DLL name, as a title's imports arrive.
    struct ByDll;

    impl ImportResolver for ByDll {
        fn resolve(&self, dll: &str, _: &ImportSymbol) -> Resolution {
            let found = match dll.to_ascii_lowercase().as_str() {
                "hle.dll" => Some((0x1111, Provenance::Hle)),
                "host.dll" => Some((0x2222, Provenance::Host)),
                "title.dll" => Some((0x3333, Provenance::Title)),
                _ => None,
            };
            found.map_or(Resolution::Missing, |(address, provenance)| {
                Resolution::Found {
                    address,
                    provenance,
                }
            })
        }
    }

    #[test]
    fn maps_and_unloads_a_prepared_copy() {
        let prepared = prepared_own();
        let dir = tempfile::tempdir().unwrap();
        let image = load(&prepared, dir.path()).unwrap();
        assert_ne!(image.base(), 0);
        assert_eq!(image.base() % PAGE, 0, "the loader page-aligns the mapping");
        assert_eq!(image.info().import_count(), prepared.info.import_count());
        assert_eq!(
            std::fs::read(dir.path().join(prepared.file_name())).unwrap(),
            prepared.bytes
        );
        // The prepared copy is registered as mapped, and keeps its static TLS.
        assert_eq!(disable_thread_calls(image.base()), Some(false));
        // SAFETY: never attached, and no pointer into it outlives this test.
        unsafe { image.unload() };
    }

    #[test]
    fn write_if_changed_rewrites_only_when_the_bytes_differ() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("victory-image-test.dll");
        write_if_changed(&path, b"one").unwrap();
        let read_only = |on: bool| {
            let mut permissions = std::fs::metadata(&path).unwrap().permissions();
            permissions.set_readonly(on);
            std::fs::set_permissions(&path, permissions).unwrap();
        };
        // Identical bytes: no write happens, so even a read-only file is fine
        // (replacing a read-only file on Windows fails).
        read_only(true);
        write_if_changed(&path, b"one").unwrap();
        read_only(false);
        write_if_changed(&path, b"two").unwrap();
        assert_eq!(std::fs::read(&path).unwrap(), b"two");
        assert!(
            !path.with_extension("tmp").exists(),
            "the temporary file is renamed, not left behind"
        );
    }

    #[test]
    fn binds_imports_and_delay_imports_by_provenance() {
        let base = alloc(1, PAGE_READONLY);
        let mut info = info(PAGE as u32);
        info.imports = vec![
            import("hle.dll", 0x100),
            import("host.dll", 0x108),
            import("title.dll", 0x110),
        ];
        info.delay_imports = vec![import("missing.dll", 0x118)];
        let image = image_at(base, info);
        let report = image
            .bind_imports(&ByDll, UnimplementedPolicy::ReturnZero)
            .unwrap();
        assert_eq!((report.hle, report.host, report.title), (1, 1, 1));
        assert_eq!(report.missing.len(), 1);
        assert_eq!(report.missing[0].dll, "missing.dll");
        assert_eq!(report.missing[0].symbol, "f118");
        assert_eq!(slot(&image, 0x100), 0x1111);
        assert_eq!(slot(&image, 0x108), 0x2222);
        assert_eq!(slot(&image, 0x110), 0x3333);
        assert_ne!(slot(&image, 0x118), 0, "the delay-import slot got a stub");
        assert_eq!(
            protection(base + 0x100),
            PAGE_READONLY,
            "the slot's protection is restored"
        );
        free(base);
    }

    #[test]
    fn binding_restores_the_protection_of_each_section() {
        let base = alloc(2, PAGE_READWRITE);
        let mut old = 0;
        // SAFETY: the second page is inside the allocation.
        let changed = unsafe {
            VirtualProtect(
                (base + PAGE) as *const c_void,
                PAGE,
                PAGE_READONLY,
                &mut old,
            )
        };
        assert_ne!(changed, 0);
        let before = (protection(base + 0x100), protection(base + PAGE + 0x100));
        assert_eq!(before, (PAGE_READWRITE, PAGE_READONLY));
        let mut info = info((2 * PAGE) as u32);
        info.imports = vec![import("a.dll", 0x100), import("b.dll", PAGE as u32 + 0x100)];
        let image = image_at(base, info);
        let report = image
            .bind_imports(&Fixed(found(0x1234)), UnimplementedPolicy::ReturnZero)
            .unwrap();
        assert_eq!(report.hle, 2);
        assert_eq!(slot(&image, 0x100), 0x1234);
        assert_eq!(slot(&image, PAGE as u32 + 0x100), 0x1234);
        assert_eq!(
            (protection(base + 0x100), protection(base + PAGE + 0x100)),
            before
        );
        free(base);
    }

    #[test]
    fn binding_no_imports_does_nothing() {
        let base = alloc(1, PAGE_READWRITE);
        let image = image_at(base, info(PAGE as u32));
        let report = image
            .bind_imports(&Fixed(Resolution::Missing), UnimplementedPolicy::Trap)
            .unwrap();
        assert_eq!((report.hle, report.host, report.title), (0, 0, 0));
        assert!(report.missing.is_empty());
        free(base);
    }

    #[test]
    fn unresolved_imports_get_a_callable_stub() {
        let base = alloc(1, PAGE_READWRITE);
        let mut info = info(PAGE as u32);
        info.imports = vec![ImportModule {
            dll: "missing.dll".into(),
            entries: vec![
                ImportEntry {
                    symbol: ImportSymbol::Name {
                        hint: 0,
                        name: "one".into(),
                    },
                    iat_slot_rva: 0x100,
                },
                ImportEntry {
                    symbol: ImportSymbol::Ordinal(7),
                    iat_slot_rva: 0x108,
                },
            ],
        }];
        let image = image_at(base, info);
        let report = image
            .bind_imports(&Fixed(Resolution::Missing), UnimplementedPolicy::ReturnZero)
            .unwrap();
        assert_eq!(report.missing.len(), 2);
        assert_eq!(report.missing[0].dll, "missing.dll");
        assert_eq!(report.missing[0].symbol, "one");
        assert_eq!(report.missing[1].symbol, "#7");
        let before = unimplemented_call_count();
        for rva in [0x100, 0x108] {
            // SAFETY: the loader wrote a stub to this slot; the stub ignores
            // its arguments and returns 0 under `ReturnZero`.
            let stub: unsafe extern "system" fn() -> u64 =
                unsafe { std::mem::transmute::<u64, _>(slot(&image, rva)) };
            // SAFETY: as above.
            assert_eq!(unsafe { stub() }, 0);
        }
        assert!(unimplemented_call_count() >= before + 2);
        free(base);
    }

    #[test]
    fn a_failed_tls_redirect_unmaps_the_image() {
        let mut original = std::fs::read(std::env::current_exe().unwrap()).unwrap();
        // A unique build hash, so the module name belongs to this test alone.
        original[16] ^= 0xFF;
        let prepared = prepare(&original).unwrap();
        assert!(!prepared.info.is_dll());
        assert!(
            prepared.info.tls.is_some(),
            "the test binary uses static TLS"
        );
        let mut info = prepared.info.clone();
        // The TLS thunk search refuses images this large, so `redirect` fails
        // after `load` has already mapped the copy.
        info.size_of_image = 1 << 30;
        let prepared = PreparedImage {
            info,
            bytes: prepared.bytes.clone(),
            original_hash: prepared.original_hash,
        };
        let name = prepared.file_name();
        assert_eq!(module_by_name(&name), 0, "no module with this name yet");
        let dir = tempfile::tempdir().unwrap();
        let error = load(&prepared, dir.path()).unwrap_err();
        let text = format!("{error:#}").to_ascii_lowercase();
        assert!(
            text.contains("tls"),
            "expected the TLS redirect to fail: {text}"
        );
        assert_eq!(
            module_by_name(&name),
            0,
            "a load that failed after mapping must unmap"
        );
    }

    #[test]
    fn exports_are_looked_up_by_name() {
        let title = tempfile::tempdir().unwrap();
        std::fs::copy(
            r"C:\Windows\System32\version.dll",
            title.path().join("v.dll"),
        )
        .unwrap();
        let loaded = load_title_dll("v.dll", title.path(), title.path(), &|_| true).unwrap();
        assert_eq!(loaded.len(), 1);
        let image = &loaded[0].1;
        assert!(image.export("GetFileVersionInfoW").is_some());
        assert!(image.export("NoSuchExport").is_none());
        assert!(
            image.export("Get\0FileVersionInfoW").is_none(),
            "an interior NUL is refused"
        );
        assert!(
            image.export("getfileversioninfow").is_none(),
            "export names are case-sensitive"
        );
        assert!(
            image
                .export_symbol(&ImportSymbol::Name {
                    hint: 0,
                    name: "GetFileVersionInfoW".into(),
                })
                .is_some()
        );
        assert!(image.export_symbol(&ImportSymbol::Ordinal(0)).is_none());
        for (_, dll) in loaded {
            // SAFETY: never attached, and no pointer into it outlives the test.
            unsafe { dll.unload() };
        }
    }

    #[test]
    fn title_dlls_are_mapped_once_and_named_in_lowercase() {
        let title = tempfile::tempdir().unwrap();
        let cache = tempfile::tempdir().unwrap();
        std::fs::copy(
            r"C:\Windows\System32\version.dll",
            title.path().join("Late.dll"),
        )
        .unwrap();
        let mut main = info(PAGE as u32);
        main.imports = vec![import("Late.dll", 0), import("LATE.DLL", 0)];
        main.delay_imports = vec![import("late.dll", 0)];
        let dlls =
            load_title_dlls(&main, title.path(), cache.path(), &|dll| dll == "late.dll").unwrap();
        assert_eq!(dlls.len(), 1, "one mapping for three spellings");
        assert_eq!(dlls[0].0, "late.dll");
        assert!(dlls[0].1.export("GetFileVersionInfoW").is_some());
        // `wanted` refuses it, or the folder has no such file: nothing maps.
        assert!(
            load_title_dlls(&main, title.path(), cache.path(), &|_| false)
                .unwrap()
                .is_empty()
        );
        let mut absent = info(PAGE as u32);
        absent.imports = vec![import("absent.dll", 0)];
        assert!(
            load_title_dlls(&absent, title.path(), cache.path(), &|_| true)
                .unwrap()
                .is_empty()
        );
        for (_, dll) in dlls {
            // SAFETY: never attached, and no pointer into it outlives the test.
            unsafe { dll.unload() };
        }
    }
}
