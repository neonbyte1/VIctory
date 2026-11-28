//! Compiled host pipelines kept on disk, and the title pipelines worth
//! starting early (decision 0007, "Pipeline cache").
//!
//! 1. Every host pipeline goes into the host's pipeline library, named by
//!    a hash of all that its host description holds (the translated
//!    shaders' code, which changes when a translation does, our root
//!    signature, the graphics state). The library is saved under the cache
//!    folder; the host refuses a library of another adapter or driver, and
//!    an entry whose description differs, so neither is ever used.
//! 2. The title pipelines a run drew or dispatched with (made in time or
//!    not) are listed by a key known when the title creates them (their
//!    shader containers and graphics state); the next run hands those to
//!    the workers at creation, during loading. A full list keeps the
//!    latest run's first.
//!
//! 3. The pre-build (`victory translate --cache --pipelines`) lists the
//!    title pipelines it has put in the library, by the same key, so a
//!    second pre-build skips them without translating or loading anything.
//!    The list holds only for the library bytes it was saved with and for
//!    what names the pipelines (the translator, this module and `host`).
//!
//! The library is free-threaded but for loads of one pipeline on several
//! threads (`ID3D12Device1::CreatePipelineLibrary`, "Thread Safety"), so
//! pipelines of different names load at the same time. A load is not always
//! quick: the library of an AMD driver holds about 140 bytes per pipeline,
//! and what its own cache lacks is compiled inside the load, for seconds.
//! Loaded one at a time, 650 pipelines kept the eight workers to one core
//! for minutes, while each title frame could wait out its 500 ms for
//! them (`make`). Stores and serialising keep the lock: the size asked for
//! must still be right when the bytes are written. Loads take no lock at
//! all, since a load adds nothing to the library: a save holds the lock
//! while it serialises the whole library (hundreds of megabytes in the
//! world), and loads that waited on it stalled every worker for as long.

use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Condvar, Mutex, OnceLock};

use windows::Win32::Graphics::Direct3D12::*;
use windows::core::{HSTRING, Interface, PCWSTR};

static DIR: OnceLock<PathBuf> = OnceLock::new();

/// Keeps the pipeline library and the early list in `dir` from now on.
///
/// The library's pipelines are named after the shaders they were made from,
/// so a change to the translator (`victory_shader::SOURCE_HASH`) makes every
/// name in it useless. The first run after one used to write the few
/// pipelines it made over the old library: measured on the title, a 926 MB
/// library (27 160 pipelines, a 7.5 minute warm-up) became 612 KB, and
/// bisecting between two builds paid that warm-up for each of them, in
/// turn, every time. So each translation keeps its library beside the
/// others, in `dir/<SOURCE_HASH>`, and a run only ever writes its own;
/// [`KEPT`] of the newest are kept.
pub fn use_pipeline_cache(dir: PathBuf) {
    DIR.get_or_init(|| {
        let dir = keyed(&dir);
        sweep_staging(&dir, std::process::id(), STAGING_KEPT);
        dir
    });
}

/// How long another process's staging file is left alone: the folder is
/// shared, and a running instance may be writing it.
const STAGING_KEPT: std::time::Duration = std::time::Duration::from_secs(10 * 60);

/// Removes the staging files of cut writes (`write_atomically`) from `dir`:
/// those of process `pid` now (no write of this run has started yet), other
/// processes' once untouched for `kept`. A cut write leaves its file, and no
/// later write reuses the name.
fn sweep_staging(dir: &std::path::Path, pid: u32, kept: std::time::Duration) {
    let Ok(entries) = std::fs::read_dir(dir) else {
        return;
    };
    for entry in entries.flatten() {
        let name = entry.file_name();
        let Some(writer) = name.to_str().and_then(staging_pid) else {
            continue;
        };
        let Ok(meta) = entry.metadata() else {
            continue;
        };
        if !meta.is_file() {
            continue;
        }
        let age = meta.modified().ok().and_then(|t| t.elapsed().ok());
        if writer == pid || age.is_some_and(|age| age >= kept) {
            let _ = std::fs::remove_file(entry.path());
        }
    }
}

/// The process whose staging file `name` is (`<file>.tmp-<pid>-<n>` of a
/// file this module writes), if it is one.
fn staging_pid(name: &str) -> Option<u32> {
    let rest = [LIBRARY_FILE, EARLY_FILE, PREBUILT_FILE]
        .into_iter()
        .find_map(|file| name.strip_prefix(file)?.strip_prefix(".tmp-"))?;
    let (pid, n) = rest.split_once('-')?;
    let digits = |s: &str| !s.is_empty() && s.bytes().all(|b| b.is_ascii_digit());
    if !digits(pid) || !digits(n) {
        return None;
    }
    pid.parse().ok()
}

/// Whether `dir` is a library directory this wrote: a name that is 16
/// lowercase hex digits (a [`victory_shader::SOURCE_HASH`]) holding a
/// library.
fn is_translation(dir: &Path) -> bool {
    let ours = dir
        .file_name()
        .and_then(|name| name.to_str())
        .is_some_and(|name| {
            name.len() == 16
                && name
                    .bytes()
                    .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
        });
    ours && dir.is_dir() && library_in(dir)
}

/// Whether `dir` holds a library: [`LIBRARY_FILE`], or any of its shards
/// where the library is split (`#1653` puts one file per shard,
/// `host-pipelines.00.bin` to `host-pipelines.15.bin`, in the same
/// directory, and writes only the shards that changed, so a small cache can
/// hold shard 15 and no shard 00). Both are checked so that the two changes
/// read the same directories whichever of them lands first - the names are
/// written out because `shard_file` does not exist without the split.
fn library_in(dir: &Path) -> bool {
    dir.join(LIBRARY_FILE).is_file()
        || (0..16).any(|shard| dir.join(format!("host-pipelines.{shard:02}.bin")).is_file())
}

/// How many translations keep a library: the one in use and the previous
/// build's, which is what a bisection between two builds needs.
const KEPT: usize = 2;

/// `dir` under this build's translation, with older libraries pruned to
/// [`KEPT`] (by modification time).
fn keyed(dir: &Path) -> PathBuf {
    let path = dir.join(victory_shader::SOURCE_HASH);
    let Ok(entries) = std::fs::read_dir(keyed(&dir)) else {
        return path;
    };
    let mut others: Vec<(std::time::SystemTime, PathBuf)> = entries
        .flatten()
        .map(|entry| entry.path())
        // Only a directory this wrote: named like a translation and holding
        // a library. Anything else in the cache directory (`backup/`, a
        // copy of one) is not ours to delete.
        .filter(|other| *other != path && is_translation(other))
        .filter_map(|other| {
            let at = other.metadata().ok()?.modified().ok()?;
            Some((at, other))
        })
        .collect();
    others.sort_by_key(|(at, _)| std::cmp::Reverse(*at));
    for (_, older) in others.into_iter().skip(KEPT - 1) {
        // A library and a list or two: gone with the translation that made
        // them. Nothing this run reads is ever one of them.
        let _ = std::fs::remove_dir_all(&older);
    }
    path
}

/// Host pipelines created, loaded from the library, and title pipelines
/// started at creation, since the start.
pub(crate) static MADE: AtomicU64 = AtomicU64::new(0);
pub(crate) static LOADED: AtomicU64 = AtomicU64::new(0);
pub(crate) static EARLY: AtomicU64 = AtomicU64::new(0);

struct Library {
    library: ID3D12PipelineLibrary1,
    /// The blob the library was made from: it must outlive the library.
    _blob: Vec<u8>,
    /// Pipelines stored since the last save.
    added: u64,
    /// Title pipelines the pre-build put in the library or found there, by
    /// key, and whether that list changed since the last save.
    prebuilt: HashSet<String>,
    prebuilt_changed: bool,
}

// SAFETY: D3D12 objects are free-threaded (a pipeline library with the
// exception `NAMES` takes care of).
unsafe impl Send for Library {}

/// The library for loads, beside the one stores and saves lock.
struct Shared {
    /// The same library as `state`'s, loaded from without its lock.
    loads: ID3D12PipelineLibrary1,
    state: Mutex<Library>,
}

// SAFETY: as for `Library`.
unsafe impl Send for Shared {}
// SAFETY: the library is free-threaded but for loads of one name, which
// `NAMES` keeps apart; the rest is behind `state`'s lock.
unsafe impl Sync for Shared {}

/// Names one thread at a time holds.
struct Names {
    held: Mutex<Option<HashSet<String>>>,
    released: Condvar,
}

/// A name held until this is dropped.
struct Held<'a> {
    names: &'a Names,
    name: &'a str,
}

impl Names {
    const fn new() -> Names {
        Names {
            held: Mutex::new(None),
            released: Condvar::new(),
        }
    }

    /// Holds `name`, once no other thread does.
    fn hold<'a>(&'a self, name: &'a str) -> Held<'a> {
        let mut held = self.held.lock().unwrap_or_else(|e| e.into_inner());
        while held.as_ref().is_some_and(|h| h.contains(name)) {
            held = self.released.wait(held).unwrap_or_else(|e| e.into_inner());
        }
        held.get_or_insert_default().insert(name.to_owned());
        Held { names: self, name }
    }
}

impl Drop for Held<'_> {
    fn drop(&mut self) {
        let mut held = self.names.held.lock().unwrap_or_else(|e| e.into_inner());
        if let Some(h) = held.as_mut() {
            h.remove(self.name);
        }
        drop(held);
        self.names.released.notify_all();
    }
}

/// Pipelines being loaded, or made and stored.
static NAMES: Names = Names::new();

const LIBRARY_FILE: &str = "host-pipelines.bin";
/// The library of a WARP host: the hardware's refuses its pipelines and the
/// other way round, so one file for both was emptied by every switch (#2637).
const SOFTWARE_LIBRARY_FILE: &str = "host-pipelines-warp.bin";

/// The library file of a host on the software adapter or on hardware.
fn library_file(software: bool) -> &'static str {
    if software { SOFTWARE_LIBRARY_FILE } else { LIBRARY_FILE }
}

/// The library file of this run's host.
fn host_library_file() -> &'static str {
    library_file(host_software())
}

/// The pre-build's list of a host on the software adapter or on hardware: it
/// is tagged with its library's bytes, so one list for both was replaced by
/// whichever pre-built last and refused by the other.
fn prebuilt_file(software: bool) -> &'static str {
    if software { SOFTWARE_PREBUILT_FILE } else { PREBUILT_FILE }
}

/// Whether this run's host is on the software adapter (WARP).
fn host_software() -> bool {
    crate::host::host().is_some_and(|h| h.software)
}
const EARLY_FILE: &str = "early-pipelines.txt";
const PREBUILT_FILE: &str = "prebuilt-pipelines.txt";
const SOFTWARE_PREBUILT_FILE: &str = "prebuilt-pipelines-warp.txt";
/// Title pipelines listed for an early start, at most.
const EARLY_MAX: usize = 8192;

static LIBRARY: OnceLock<Option<Shared>> = OnceLock::new();

fn library() -> Option<&'static Shared> {
    LIBRARY
        .get_or_init(|| {
            let device = crate::host::host()?.device.cast::<ID3D12Device1>().ok()?;
            let read = |file| DIR.get().and_then(|d| std::fs::read(d.join(file)).ok());
            let blob = read(host_library_file()).unwrap_or_default();
            let tag = tag(&blob);
            let mut library = open(&device, blob);
            // The pre-build's list, if it was saved with these bytes.
            if let (Some(l), Some(list)) = (library.as_mut(), read(prebuilt_file(host_software())))
                && !l._blob.is_empty()
            {
                l.prebuilt = listed(&list, &tag);
            }
            if library.is_none() {
                tracing::warn!("no host pipeline library: pipelines are not kept");
            }
            library.map(|l| Shared { loads: l.library.clone(), state: Mutex::new(l) })
        })
        .as_ref()
}

/// The library in `blob`, or an empty one if the host rejects the blob (an
/// adapter or driver change, damage).
fn open(device: &ID3D12Device1, blob: Vec<u8>) -> Option<Library> {
    let make = |bytes: &[u8]| -> windows::core::Result<ID3D12PipelineLibrary1> {
        // SAFETY: the bytes outlive the library (kept beside it).
        unsafe {
            device.CreatePipelineLibrary::<ID3D12PipelineLibrary>(bytes)?.cast()
        }
    };
    if !blob.is_empty() {
        let started = std::time::Instant::now();
        match make(&blob) {
            Ok(library) => {
                let ms = started.elapsed().as_millis() as u64;
                tracing::info!(bytes = blob.len(), ms, "host pipeline library loaded");
                return Some(Library {
                    library,
                    _blob: blob,
                    added: 0,
                    prebuilt: HashSet::new(),
                    prebuilt_changed: false,
                });
            }
            Err(e) => tracing::info!(error = %e, "host pipeline library rejected; starting an empty one"),
        }
    } else {
        tracing::info!("no host pipeline library yet; starting an empty one");
    }
    // An empty library is asked for with a null blob. An empty Rust slice has
    // a dangling, non-null pointer; Windows only looks at the length, but
    // vkd3d-proton (Wine) refuses a non-null blob of length 0 with
    // E_INVALIDARG, and no pipeline was ever kept there.
    let empty = || -> windows::core::Result<ID3D12PipelineLibrary1> {
        use windows::core::Interface;
        let mut out = std::ptr::null_mut();
        // SAFETY: the method's own arguments; a null blob of length 0 makes
        // an empty library.
        unsafe {
            (Interface::vtable(device).CreatePipelineLibrary)(
                Interface::as_raw(device),
                std::ptr::null(),
                0,
                &ID3D12PipelineLibrary::IID,
                &mut out,
            )
            .ok()?;
            ID3D12PipelineLibrary::from_raw(out).cast()
        }
    };
    let library = match empty() {
        Ok(library) => library,
        Err(e) => {
            tracing::info!(error = %e, "empty host pipeline library refused");
            return None;
        }
    };
    Some(Library {
        library,
        _blob: Vec::new(),
        added: 0,
        prebuilt: HashSet::new(),
        prebuilt_changed: false,
    })
}

/// What the pre-build's list holds for: the library's bytes, and what
/// names its pipelines (the translation, as the shader cache keys it; the
/// host descriptions).
fn tag(blob: &[u8]) -> String {
    let limit = victory_shader::emit::loop_limit().to_le_bytes();
    let translator = victory_shader::SOURCE_HASH.as_bytes();
    let mut h = blake3::Hasher::new();
    for part in [
        blob,
        translator,
        include_bytes!("../shader_cache.rs"),
        &limit,
        include_bytes!("host.rs"),
        include_bytes!("library.rs"),
    ] {
        h.update(&(part.len() as u64).to_le_bytes());
        h.update(part);
    }
    h.finalize().to_hex().to_string()
}

/// The pre-build's list as saved: the tag it holds for, then its keys.
fn list(tag: &str, keys: &HashSet<String>) -> String {
    let keys = keys.iter().map(String::as_str);
    std::iter::once(tag).chain(keys).collect::<Vec<_>>().join("\n")
}

/// The keys of a saved list, if it holds for `tag` (else none).
fn listed(list: &[u8], tag: &str) -> HashSet<String> {
    let list = String::from_utf8_lossy(list);
    let mut lines = list.lines();
    match lines.next() == Some(tag) {
        true => lines.map(str::to_owned).collect(),
        false => HashSet::new(),
    }
}

/// Whether the pre-build has the title pipeline `key` in the library.
pub(crate) fn prebuilt(key: &str) -> bool {
    library().is_some_and(|l| l.state.lock().unwrap_or_else(|e| e.into_inner()).prebuilt.contains(key))
}

/// The pre-build has the title pipeline `key` in the library now.
pub(crate) fn note_prebuilt(key: String) {
    if let Some(l) = library() {
        let mut l = l.state.lock().unwrap_or_else(|e| e.into_inner());
        l.prebuilt_changed |= l.prebuilt.insert(key);
    }
}

/// A pipeline's name in the library: the hash of what its host
/// description holds.
pub(crate) fn name(kind: &str, root: &[u8], code: &[&[u8]], state: &str) -> String {
    let mut h = blake3::Hasher::new();
    for part in [kind.as_bytes(), root].into_iter().chain(code.iter().copied()).chain([state.as_bytes()]) {
        h.update(&(part.len() as u64).to_le_bytes());
        h.update(part);
    }
    h.finalize().to_hex()[..32].to_owned()
}

pub(crate) struct PipelineState {
    pub(crate) pso: ID3D12PipelineState,
    /// The library contains this PSO under its name.
    pub(crate) in_library: bool,
}

/// The pipeline named `name`: loaded from the library by `load` if it is
/// there with the same description, else made by `create` and stored. A
/// name asked for by several threads is loaded or made by one at a time
/// (the others then find it in the library); different names overlap.
pub(crate) fn pipeline(
    name: &str,
    load: impl FnOnce(&ID3D12PipelineLibrary1, PCWSTR) -> windows::core::Result<ID3D12PipelineState>,
    create: impl FnOnce() -> windows::core::Result<ID3D12PipelineState>,
) -> windows::core::Result<PipelineState> {
    let wide = HSTRING::from(name);
    #[cfg(feature = "diagnostics")]
    LAST.set(Some((name.to_owned(), true)));
    let Some(library) = library() else {
        MADE.fetch_add(1, Ordering::Relaxed);
        return create().map(|pso| PipelineState { pso, in_library: false });
    };
    let _held = NAMES.hold(name);
    // Outside the lock: loads take up to seconds, and a save holds the lock
    // for as long as it serialises. The library's blob lives as long as it
    // does, which is never dropped.
    if let Ok(pso) = load(&library.loads, PCWSTR(wide.as_ptr())) {
        LOADED.fetch_add(1, Ordering::Relaxed);
        return Ok(PipelineState { pso, in_library: true });
    }
    // Compiled outside the lock: compiles take up to seconds.
    #[cfg(feature = "diagnostics")]
    LAST.set(Some((name.to_owned(), false)));
    let pso = create()?;
    MADE.fetch_add(1, Ordering::Relaxed);
    let mut l = library.state.lock().unwrap_or_else(|e| e.into_inner());
    // SAFETY: a live pipeline; a name already stored (a description that
    // differs) is refused, which leaves the library as it was.
    let in_library = unsafe { l.library.StorePipeline(PCWSTR(wide.as_ptr()), &pso) }.is_ok();
    if in_library {
        l.added += 1;
    }
    Ok(PipelineState { pso, in_library })
}

#[cfg(feature = "diagnostics")]
thread_local! {
    /// This thread's last pipeline: its name, and whether it was loaded
    /// (else made), for the pipeline log (diagnostics).
    pub(crate) static LAST: std::cell::Cell<Option<(String, bool)>> =
        const { std::cell::Cell::new(None) };
}

/// Held by a save from start to end.
static SAVING: Mutex<()> = Mutex::new(());

/// Writes the library and the early list if pipelines were added since the
/// last save and `SAVE_EVERY` has passed (or `now`).
pub(crate) fn save(now: bool) {
    static LAST: Mutex<Option<std::time::Instant>> = Mutex::new(None);
    const SAVE_EVERY: std::time::Duration = std::time::Duration::from_secs(20);
    // One save at a time, from serialising to the last file: an older blob
    // never lands over a newer, nor beside another's prebuilt list. The
    // periodic saves (after each pipeline a worker makes) are best effort:
    // they leave it to the save running; shutdown's waits.
    let _saving = if now {
        SAVING.lock().unwrap_or_else(|e| e.into_inner())
    } else {
        match SAVING.try_lock() {
            Ok(saving) => saving,
            Err(std::sync::TryLockError::Poisoned(e)) => e.into_inner(),
            Err(std::sync::TryLockError::WouldBlock) => return,
        }
    };
    let Some(dir) = DIR.get() else { return };
    if LIBRARY.get().is_none() && !USED_CHANGED.load(Ordering::Relaxed) {
        return;
    }
    let Some(library) = library() else { return };
    {
        let mut last = LAST.lock().unwrap_or_else(|e| e.into_inner());
        if !now && last.is_some_and(|t| t.elapsed() < SAVE_EVERY) {
            return;
        }
        *last = Some(std::time::Instant::now());
    }
    let (blob, prebuilt, dirty) = {
        let mut l = library.state.lock().unwrap_or_else(|e| e.into_inner());
        let l = &mut *l;
        let Some(dirty) = Dirty::take(&mut l.added, &mut l.prebuilt_changed, &USED_CHANGED) else {
            return;
        };
        // SAFETY: a plain size query on a live library.
        let size = unsafe { l.library.GetSerializedSize() };
        let Some(mut bytes) = zeroed(size) else {
            // Left for a later save: the pending changes stay marked.
            tracing::warn!(bytes = size, "no memory to serialise the host pipeline library; saved later");
            dirty.restore(&mut l.added, &mut l.prebuilt_changed, &USED_CHANGED);
            return;
        };
        // SAFETY: a buffer of the size the library asks for.
        let bytes = unsafe { l.library.Serialize(&mut bytes) }.map(|()| bytes);
        match bytes {
            Ok(bytes) => {
                // Kept beside the bytes it holds for (also when a run adds
                // pipelines).
                let prebuilt = (!l.prebuilt.is_empty()).then(|| list(&tag(&bytes), &l.prebuilt));
                (bytes, prebuilt, dirty)
            }
            Err(e) => {
                tracing::warn!(error = %e, "host pipeline library not serialised");
                dirty.restore(&mut l.added, &mut l.prebuilt_changed, &USED_CHANGED);
                return;
            }
        }
    };
    let used = used().as_ref().map(Used::list).unwrap_or_default();
    let written = std::fs::create_dir_all(dir)
        .and_then(|()| write_atomically(&dir.join(host_library_file()), &blob))
        .and_then(|()| write_atomically(&dir.join(EARLY_FILE), used.join("\n").as_bytes()))
        .and_then(|()| match &prebuilt {
            Some(list) => write_atomically(&dir.join(prebuilt_file(host_software())), list.as_bytes()),
            None => Ok(()),
        });
    let (created, distinct) = created();
    let (shaders_cached, shaders_translated) = crate::shader_cache::counts();
    match written {
        Ok(()) => tracing::info!(
            bytes = blob.len(),
            early = used.len(),
            made = MADE.load(Ordering::Relaxed),
            loaded = LOADED.load(Ordering::Relaxed),
            started_early = EARLY.load(Ordering::Relaxed),
            created,
            distinct,
            shaders_cached,
            shaders_translated,
            "host pipeline library saved"
        ),
        Err(e) => {
            tracing::warn!(error = %e, "host pipeline library not saved");
            let mut l = library.state.lock().unwrap_or_else(|e| e.into_inner());
            let l = &mut *l;
            dirty.restore(&mut l.added, &mut l.prebuilt_changed, &USED_CHANGED);
        }
    }
}

/// What changed since the last save, taken by the save that writes it and
/// given back if that save fails, so that the next one tries again.
struct Dirty {
    added: u64,
    prebuilt_changed: bool,
    used_changed: bool,
}

impl Dirty {
    /// The marks, cleared; none if nothing changed.
    fn take(added: &mut u64, prebuilt_changed: &mut bool, used_changed: &AtomicBool) -> Option<Dirty> {
        let dirty = Dirty {
            added: std::mem::take(added),
            prebuilt_changed: std::mem::take(prebuilt_changed),
            used_changed: used_changed.swap(false, Ordering::Relaxed),
        };
        (dirty.added != 0 || dirty.prebuilt_changed || dirty.used_changed).then_some(dirty)
    }

    /// The marks, back (with what changed meanwhile).
    fn restore(self, added: &mut u64, prebuilt_changed: &mut bool, used_changed: &AtomicBool) {
        *added += self.added;
        *prebuilt_changed |= self.prebuilt_changed;
        if self.used_changed {
            used_changed.store(true, Ordering::Relaxed);
        }
    }
}

/// `n` zero bytes, or `None` when the host has no memory for them. The
/// library is about 1.7 GB on NVIDIA (each pipeline's machine code), and an
/// infallible allocation of that size aborted the whole process mid-run
/// ("memory allocation of 2650101862 bytes failed", RTX 4050 Laptop).
fn zeroed(n: usize) -> Option<Vec<u8>> {
    let mut v = Vec::new();
    v.try_reserve_exact(n).ok()?;
    v.resize(n, 0);
    Some(v)
}

/// Writes `bytes` to a temporary file beside `path`, then renames it over
/// `path`: a reader never sees a part. Serialised so concurrent worker and
/// shutdown saves do not race on the same staging file or target rename.
fn write_atomically(path: &std::path::Path, bytes: &[u8]) -> std::io::Result<()> {
    use std::sync::atomic::AtomicU64;
    static WRITING: Mutex<()> = Mutex::new(());
    static NEXT: AtomicU64 = AtomicU64::new(0);
    let _guard = WRITING.lock().unwrap_or_else(|e| e.into_inner());
    let n = NEXT.fetch_add(1, Ordering::Relaxed);
    let mut name = path.file_name().unwrap_or_default().to_os_string();
    name.push(format!(".tmp-{}-{n}", std::process::id()));
    let tmp = path.with_file_name(name);
    let res = std::fs::write(&tmp, bytes).and_then(|()| std::fs::rename(&tmp, path));
    if res.is_err() {
        let _ = std::fs::remove_file(&tmp);
    }
    res
}

/// Title pipelines created, and their distinct keys.
static CREATED: Mutex<(u64, Option<HashSet<String>>)> = Mutex::new((0, None));

/// The title created a pipeline with `key`.
pub(crate) fn note_created(key: &str) {
    let mut created = CREATED.lock().unwrap_or_else(|e| e.into_inner());
    created.0 += 1;
    created.1.get_or_insert_with(HashSet::new).insert(key.to_owned());
}

/// Title pipelines created so far, and how many distinct ones.
fn created() -> (u64, usize) {
    let created = CREATED.lock().unwrap_or_else(|e| e.into_inner());
    (created.0, created.1.as_ref().map_or(0, HashSet::len))
}

/// Title pipeline keys drawn or dispatched with: in earlier runs (the
/// list as loaded), and in this one.
#[derive(Default)]
struct Used {
    /// Each earlier key with its place in the list as loaded: the list is
    /// kept most recent run first, so a lower place is a more recent run.
    before: HashMap<String, usize>,
    now: HashSet<String>,
}

impl Used {
    /// The list as saved: one key a line, most recent run first (a list
    /// in no order, as earlier builds saved it, reads as well).
    fn read(listed: &str) -> Used {
        let mut before = HashMap::new();
        for (place, key) in listed.lines().filter(|l| !l.is_empty()).enumerate() {
            before.entry(key.to_owned()).or_insert(place);
        }
        Used { before, now: HashSet::new() }
    }

    /// The list to keep: this run's keys, then earlier runs' others, most
    /// recent first, at most `EARLY_MAX` (the oldest go once it is full).
    fn list(&self) -> Vec<String> {
        let mut earlier: Vec<_> = self.before.iter().filter(|(k, _)| !self.now.contains(*k)).collect();
        earlier.sort_unstable_by_key(|&(_, &place)| place);
        let earlier = earlier.into_iter().map(|(k, _)| k);
        self.now.iter().chain(earlier).take(EARLY_MAX).cloned().collect()
    }
}

static USED: Mutex<Option<Used>> = Mutex::new(None);
static USED_CHANGED: AtomicBool = AtomicBool::new(false);

fn used() -> std::sync::MutexGuard<'static, Option<Used>> {
    let mut used = USED.lock().unwrap_or_else(|e| e.into_inner());
    if used.is_none() {
        let listed = DIR
            .get()
            .and_then(|d| std::fs::read_to_string(d.join(EARLY_FILE)).ok())
            .unwrap_or_default();
        let read = Used::read(&listed);
        tracing::info!(pipelines = read.before.len(), "title pipelines to start early");
        *used = Some(read);
    }
    used
}

/// Reads the early list now if not yet (a pre-build asks for no title
/// pipeline).
pub(crate) fn keep_early_list() {
    drop(used());
}

/// Whether the title pipeline with `key` was drawn with in an earlier run.
pub(crate) fn wanted_early(key: &str) -> bool {
    used().as_ref().is_some_and(|u| u.before.contains_key(key))
}

/// The title pipeline with `key` was drawn or dispatched with.
pub(crate) fn note_used(key: &str) {
    if used().get_or_insert_with(Used::default).now.insert(key.to_owned()) {
        USED_CHANGED.store(true, Ordering::Relaxed);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A size the host cannot give is refused, not fatal; others are zeros.
    #[test]
    fn a_save_buffer_the_host_cannot_give_is_refused() {
        assert!(zeroed(usize::MAX / 2).is_none());
        assert_eq!(zeroed(100).as_deref(), Some(&[0u8; 100][..]));
    }

    /// A hardware pre-build and a WARP one in one cache folder each read back
    /// their own list: neither replaces the other's (#2646 review).
    #[test]
    fn each_host_reads_back_its_own_prebuilt_list() {
        let dir = tempfile::tempdir().unwrap();
        let keys = |k: &[&str]| k.iter().map(|s| s.to_string()).collect::<HashSet<_>>();
        let (hw, warp) = (keys(&["a", "b"]), keys(&["c"]));
        std::fs::write(dir.path().join(prebuilt_file(false)), list("hw-tag", &hw)).unwrap();
        std::fs::write(dir.path().join(prebuilt_file(true)), list("warp-tag", &warp)).unwrap();
        let read = |software| std::fs::read(dir.path().join(prebuilt_file(software))).unwrap();
        assert_eq!(listed(&read(false), "hw-tag"), hw);
        assert_eq!(listed(&read(true), "warp-tag"), warp);
        // a list read for another library's bytes is refused, as before
        assert!(listed(&read(true), "hw-tag").is_empty());
    }

    /// A WARP host and a hardware one keep their libraries apart, so a run on
    /// one does not empty the other's; the hardware file keeps its name (#2637).
    #[test]
    fn warp_and_hardware_hosts_keep_their_own_library() {
        assert_eq!(library_file(false), "host-pipelines.bin");
        assert_ne!(library_file(true), library_file(false));
        assert!(library_file(true).ends_with(".bin"));
        // the pre-build's list too: it names its library's bytes
        assert_eq!(prebuilt_file(false), "prebuilt-pipelines.txt");
        assert_ne!(prebuilt_file(true), prebuilt_file(false));
        assert!(prebuilt_file(true).ends_with(".txt"));
    }

    /// Cache state is process-global, so use fresh WARP processes to save
    /// and reload while the hardware library and its index hold marker bytes.
    #[cfg(feature = "diagnostics")]
    #[test]
    fn warp_cache_save_and_reload_preserve_hardware_files() {
        const STEP: &str = "VICTORY_TEST_WARP_CACHE_STEP";
        const CACHE: &str = "VICTORY_TEST_WARP_CACHE_DIR";
        const NAME: &str = "pipeline::library::tests::warp_cache_save_and_reload_preserve_hardware_files";
        const KEY: &str = "synthetic-warp-cache-key";
        const NO_LIBRARY: &str = "WARP pipeline libraries unavailable";
        const HARDWARE_BLOB: &[u8] = b"synthetic bytes from a different adapter";
        const HARDWARE_INDEX: &[u8] = b"hardware-tag\nsynthetic-hardware-key";

        if let Ok(step) = std::env::var(STEP) {
            use_pipeline_cache(PathBuf::from(std::env::var_os(CACHE).unwrap()));
            if step == "save" {
                if library().is_none() {
                    println!("{NO_LIBRARY}");
                    return;
                }
                note_prebuilt(KEY.to_owned());
                save(true);
            } else {
                assert_eq!(step, "reload");
                assert!(prebuilt(KEY), "WARP did not reload its tagged entry");
            }
            return;
        }

        let dir = tempfile::tempdir().unwrap();
        let hardware = dir.path().join("host-pipelines.bin");
        let hardware_index = dir.path().join("prebuilt-pipelines.txt");
        let software = dir.path().join("host-pipelines-warp.bin");
        let software_index = dir.path().join("prebuilt-pipelines-warp.txt");
        std::fs::write(&hardware, HARDWARE_BLOB).unwrap();
        std::fs::write(&hardware_index, HARDWARE_INDEX).unwrap();

        let run = |step| {
            std::process::Command::new(std::env::current_exe().unwrap())
                .args(["--exact", NAME, "--nocapture"])
                .env(STEP, step)
                .env(CACHE, dir.path())
                .env("VICTORY_D3D12_WARP", "1")
                .env("VICTORY_D3D12_DEBUG", "0")
                .env("VICTORY_D3D12_GBV", "0")
                .output()
                .unwrap()
        };
        let saved = run("save");
        let message = String::from_utf8_lossy(&saved.stdout);
        if saved.status.success() && message.contains(NO_LIBRARY) {
            eprintln!("{NO_LIBRARY}; skipped");
            return;
        }
        assert!(
            saved.status.success(),
            "{message}\n{}",
            String::from_utf8_lossy(&saved.stderr),
        );
        let hardware_after = std::fs::read(&hardware).unwrap();
        let hardware_index_after = std::fs::read(&hardware_index).unwrap();
        println!(
            "WARP save: hardware blob {} -> {} bytes, hardware index {} -> {} bytes",
            HARDWARE_BLOB.len(),
            hardware_after.len(),
            HARDWARE_INDEX.len(),
            hardware_index_after.len(),
        );
        assert!(hardware_after == HARDWARE_BLOB, "WARP save replaced the hardware file");
        assert!(
            hardware_index_after == HARDWARE_INDEX,
            "WARP save replaced the hardware prebuilt list",
        );
        assert!(software.is_file() && software_index.is_file());
        let reloaded = run("reload");
        assert!(
            reloaded.status.success(),
            "{}\n{}",
            String::from_utf8_lossy(&reloaded.stdout),
            String::from_utf8_lossy(&reloaded.stderr),
        );
    }

    #[test]
    fn names_change_with_what_goes_into_the_pipeline() {
        let a = name("graphics", b"root", &[b"vs", b"ps"], "state");
        assert_eq!(a, name("graphics", b"root", &[b"vs", b"ps"], "state"));
        assert_ne!(a, name("graphics", b"root", &[b"vs", b"ps2"], "state"));
        assert_ne!(a, name("graphics", b"root", &[b"vs", b"ps"], "state2"));
        assert_ne!(a, name("graphics", b"root2", &[b"vs", b"ps"], "state"));
        assert_ne!(a, name("mesh", b"root", &[b"vs", b"ps"], "state"));
        // Parts do not run into each other.
        assert_ne!(a, name("graphics", b"root", &[b"vsp", b"s"], "state"));
    }

    /// This run's keys stay in the list; earlier runs' fill it up to the cap.
    #[test]
    fn the_early_list_keeps_this_run_first() {
        let keys = |r: std::ops::Range<usize>| r.map(|i| format!("{i:032x}")).collect::<HashSet<_>>();
        let before = keys(0..EARLY_MAX).into_iter().enumerate().map(|(i, k)| (k, i)).collect();
        let used = Used { before, now: keys(EARLY_MAX - 10..EARLY_MAX + 10) };
        let list = used.list();
        assert_eq!(list.len(), EARLY_MAX);
        assert!(used.now.iter().all(|k| list.contains(k)));
        assert_eq!(list.iter().collect::<HashSet<_>>().len(), EARLY_MAX, "no key twice");
    }

    /// A full list makes room by the oldest runs' keys, and keeps that
    /// order across saves.
    #[test]
    fn a_full_early_list_drops_the_oldest_runs_keys() {
        let key = |i: usize| format!("{i:032x}");
        // As saved: the last run's keys, then the run before's.
        let saved: Vec<String> = (0..EARLY_MAX).map(key).collect();
        let mut used = Used::read(&saved.join("\n"));
        used.now = (EARLY_MAX..EARLY_MAX + 10).map(key).collect();
        used.now.insert(key(EARLY_MAX - 1)); // the oldest key, used again
        let list = used.list();
        assert_eq!(list.len(), EARLY_MAX);
        assert!(used.now.iter().all(|k| list.contains(k)));
        // The ten that go are the oldest ones not used again.
        let dropped: HashSet<String> = (EARLY_MAX - 11..EARLY_MAX - 1).map(key).collect();
        assert!(list.iter().all(|k| !dropped.contains(k)));
        // Read back, this run's keys come first, then the earlier ones in order.
        let again = Used::read(&list.join("\n"));
        assert!(used.now.iter().all(|k| again.before[k] < used.now.len()));
        assert!(again.before[&key(0)] < again.before[&key(1)]);
    }

    /// A save that fails gives its changes back for the next to write.
    #[test]
    fn a_failed_save_leaves_its_changes_to_the_next() {
        let (mut added, mut prebuilt, used) = (3u64, true, AtomicBool::new(true));
        let dirty = Dirty::take(&mut added, &mut prebuilt, &used).expect("changes");
        assert_eq!((added, prebuilt, used.load(Ordering::Relaxed)), (0, false, false), "taken");
        assert!(Dirty::take(&mut added, &mut prebuilt, &used).is_none(), "nothing more to save");
        added = 2; // pipelines stored while the save wrote
        dirty.restore(&mut added, &mut prebuilt, &used);
        assert_eq!((added, prebuilt, used.load(Ordering::Relaxed)), (5, true, true));
        assert!(Dirty::take(&mut added, &mut prebuilt, &used).is_some(), "the next save writes them");
    }

    /// A save waits for the one running (the worker's and shutdown's do not
    /// interleave their files).
    #[test]
    fn a_save_waits_for_the_save_running() {
        let running = SAVING.lock().unwrap();
        let done = std::sync::Arc::new(AtomicBool::new(false));
        let second = {
            let done = done.clone();
            std::thread::spawn(move || {
                save(true);
                done.store(true, Ordering::SeqCst);
            })
        };
        std::thread::sleep(std::time::Duration::from_millis(300));
        assert!(!done.load(Ordering::SeqCst), "a second save ran under the first");
        drop(running);
        second.join().unwrap();
    }

    /// A periodic save finding one running returns at once: compile workers
    /// call it after every pipeline and must not wait out a save's files.
    #[test]
    fn a_periodic_save_does_not_wait_for_the_save_running() {
        let running = SAVING.lock().unwrap();
        let done = std::sync::Arc::new(AtomicBool::new(false));
        let periodic = {
            let done = done.clone();
            std::thread::spawn(move || {
                save(false);
                done.store(true, Ordering::SeqCst);
            })
        };
        for _ in 0..100 {
            if done.load(Ordering::SeqCst) {
                break;
            }
            std::thread::sleep(std::time::Duration::from_millis(10));
        }
        let returned = done.load(Ordering::SeqCst);
        drop(running);
        periodic.join().unwrap();
        assert!(returned, "a periodic save waited for the one running");
    }

    /// The pre-build's list holds only for the library bytes it was
    /// saved with.
    #[test]
    fn the_prebuilt_list_holds_for_its_library_only() {
        let keys: HashSet<String> = ["a".repeat(32), "b".repeat(32)].into();
        let saved = list(&tag(b"library"), &keys);
        assert_eq!(listed(saved.as_bytes(), &tag(b"library")), keys);
        assert!(listed(saved.as_bytes(), &tag(b"library2")).is_empty());
        assert!(listed(b"", &tag(b"library")).is_empty());
    }

    #[test]
    fn different_names_are_held_at_the_same_time() {
        static NAMES: Names = Names::new();
        // Each thread holds its name until every thread holds its own.
        let all = std::sync::Barrier::new(8);
        std::thread::scope(|s| {
            for i in 0..8 {
                let all = &all;
                s.spawn(move || {
                    let name = format!("pipeline {i}");
                    let _held = NAMES.hold(&name);
                    all.wait();
                });
            }
        });
        assert!(NAMES.held.lock().unwrap().as_ref().unwrap().is_empty());
    }

    #[test]
    fn one_name_is_held_by_one_thread_at_a_time() {
        static NAMES: Names = Names::new();
        static INSIDE: AtomicU64 = AtomicU64::new(0);
        std::thread::scope(|s| {
            for _ in 0..8 {
                s.spawn(|| {
                    for _ in 0..500 {
                        let _held = NAMES.hold("pipeline");
                        assert_eq!(INSIDE.fetch_add(1, Ordering::SeqCst), 0);
                        std::hint::spin_loop();
                        INSIDE.fetch_sub(1, Ordering::SeqCst);
                    }
                });
            }
        });
        // A holder that panics lets go of it.
        let panicked = std::thread::spawn(|| {
            let _held = NAMES.hold("pipeline");
            panic!("on purpose");
        });
        assert!(panicked.join().is_err());
        drop(NAMES.hold("pipeline"));
    }

    /// A library's pipelines load on many threads at once, each with its
    /// own description (a load with another's is refused), while others are
    /// stored and the library is serialised.
    #[test]
    fn pipelines_load_on_many_threads_at_once() {
        use windows::Win32::Graphics::Direct3D::Fxc::D3DCompile;
        use windows::core::s;
        const N: usize = 48;
        let Some(device) = crate::host::host().and_then(|h| h.device.cast::<ID3D12Device1>().ok())
        else {
            return; // no host GPU here
        };
        let shader = |i: usize| {
            // Nothing runs: the group size alone makes each program another.
            // No resources and no root signature in the source, which Wine's
            // compiler would refuse (raw and structured buffer access) or
            // drop.
            let source = format!("[numthreads({}, 1, 1)] void main() {{}}", i + 1);
            let mut blob = None;
            // SAFETY: the source outlives the call; the blob is read within it.
            unsafe {
                D3DCompile(
                    source.as_ptr().cast(),
                    source.len(),
                    None,
                    None,
                    None,
                    s!("main"),
                    s!("cs_5_1"),
                    0,
                    0,
                    &mut blob,
                    None,
                )
                .ok()?;
                let blob = blob?;
                Some(
                    std::slice::from_raw_parts(
                        blob.GetBufferPointer().cast::<u8>(),
                        blob.GetBufferSize(),
                    )
                    .to_vec(),
                )
            }
        };
        let Some(codes) = (0..2 * N).map(shader).collect::<Option<Vec<_>>>() else {
            return; // no shader compiler here
        };
        // SAFETY: an empty description; the blob is read while it lives.
        let root: ID3D12RootSignature = unsafe {
            let mut blob = None;
            D3D12SerializeRootSignature(
                &D3D12_ROOT_SIGNATURE_DESC::default(),
                D3D_ROOT_SIGNATURE_VERSION_1,
                &mut blob,
                None,
            )
            .unwrap();
            let blob = blob.unwrap();
            let bytes = std::slice::from_raw_parts(
                blob.GetBufferPointer().cast::<u8>(),
                blob.GetBufferSize(),
            );
            device.CreateRootSignature(0, bytes).unwrap()
        };
        struct Shared(ID3D12Device1, ID3D12RootSignature, ID3D12PipelineLibrary1);
        // SAFETY: D3D12 objects are free-threaded.
        unsafe impl Sync for Shared {}
        impl Shared {
            fn desc(&self, code: &[u8]) -> D3D12_COMPUTE_PIPELINE_STATE_DESC {
                D3D12_COMPUTE_PIPELINE_STATE_DESC {
                    // SAFETY: a borrowed pointer the description never releases.
                    pRootSignature: unsafe { std::mem::transmute_copy(&self.1) },
                    CS: D3D12_SHADER_BYTECODE {
                        pShaderBytecode: code.as_ptr().cast(),
                        BytecodeLength: code.len(),
                    },
                    ..Default::default()
                }
            }
            fn store(&self, i: usize, code: &[u8]) {
                let name = HSTRING::from(format!("p{i}"));
                // SAFETY: the description points at live data.
                unsafe {
                    let pso: ID3D12PipelineState =
                        self.0.CreateComputePipelineState(&self.desc(code)).unwrap();
                    self.2.StorePipeline(&name, &pso).unwrap();
                }
            }
            fn load(&self, i: usize, code: &[u8]) -> windows::core::Result<ID3D12PipelineState> {
                let name = HSTRING::from(format!("p{i}"));
                // SAFETY: as above.
                unsafe { self.2.LoadComputePipeline(&name, &self.desc(code)) }
            }
            fn serialized(&self) -> Vec<u8> {
                // SAFETY: a buffer of the size asked for; nothing is stored
                // meanwhile.
                unsafe {
                    let mut blob = vec![0; self.2.GetSerializedSize()];
                    self.2.Serialize(&mut blob).unwrap();
                    blob
                }
            }
        }
        let first = open(&device, Vec::new()).expect("an empty library");
        let made = Shared(device.clone(), root.clone(), first.library.clone());
        (0..N).for_each(|i| made.store(i, &codes[i]));
        let again = open(&device, made.serialized()).expect("the library again");
        let shared = Shared(device, root, again.library.clone());
        let stores = Mutex::new(());
        std::thread::scope(|s| {
            for t in 0..8 {
                let (shared, codes, stores) = (&shared, &codes, &stores);
                s.spawn(move || {
                    for i in (t..N).step_by(8) {
                        shared.load(i, &codes[i]).expect("a stored pipeline");
                        assert!(
                            shared.load(i, &codes[(i + 1) % N]).is_err(),
                            "another description"
                        );
                        assert!(shared.load(N + i, &codes[N + i]).is_err(), "not stored yet");
                        let _one = stores.lock().unwrap();
                        shared.store(N + i, &codes[N + i]);
                        shared.serialized();
                    }
                });
            }
        });
        for (i, code) in codes.iter().enumerate() {
            shared.load(i, code).expect("every pipeline");
        }
    }

    /// A stored pipeline loads while a save holds the library's lock (one
    /// serialises the whole library, for seconds in the world).
    #[test]
    fn a_pipeline_loads_while_a_save_holds_the_library() {
        use windows::Win32::Graphics::Direct3D::Fxc::D3DCompile;
        use windows::core::s;
        let (Some(device), Some(library)) = (
            crate::host::host().and_then(|h| h.device.cast::<ID3D12Device1>().ok()),
            library(),
        ) else {
            return; // no host GPU here
        };
        // No resources, and the root signature made apart: a compiler need
        // not keep one the source names (Wine's does not).
        let source = "[numthreads(1, 1, 1)] void main() {}";
        let mut blob = None;
        // SAFETY: the source outlives the call; the blob is read within it.
        let code = unsafe {
            if D3DCompile(
                source.as_ptr().cast(),
                source.len(),
                None,
                None,
                None,
                s!("main"),
                s!("cs_5_1"),
                0,
                0,
                &mut blob,
                None,
            )
            .is_err()
            {
                return; // no shader compiler here
            }
            let blob = blob.unwrap();
            std::slice::from_raw_parts(blob.GetBufferPointer().cast::<u8>(), blob.GetBufferSize())
                .to_vec()
        };
        // SAFETY: an empty description; the blob is read while it lives.
        let root: ID3D12RootSignature = unsafe {
            let mut blob = None;
            D3D12SerializeRootSignature(
                &D3D12_ROOT_SIGNATURE_DESC::default(),
                D3D_ROOT_SIGNATURE_VERSION_1,
                &mut blob,
                None,
            )
            .unwrap();
            let blob = blob.unwrap();
            let bytes = std::slice::from_raw_parts(
                blob.GetBufferPointer().cast::<u8>(),
                blob.GetBufferSize(),
            );
            device.CreateRootSignature(0, bytes).unwrap()
        };
        struct Made(ID3D12Device1, ID3D12RootSignature, Vec<u8>);
        // SAFETY: D3D12 objects are free-threaded.
        unsafe impl Sync for Made {}
        impl Made {
            fn get(&self, made: bool) -> windows::core::Result<PipelineState> {
                let desc = D3D12_COMPUTE_PIPELINE_STATE_DESC {
                    // SAFETY: a borrowed pointer the description never releases.
                    pRootSignature: unsafe { std::mem::transmute_copy(&self.1) },
                    CS: D3D12_SHADER_BYTECODE {
                        pShaderBytecode: self.2.as_ptr().cast(),
                        BytecodeLength: self.2.len(),
                    },
                    ..Default::default()
                };
                pipeline(
                    "test: a pipeline loads while a save holds the library",
                    // SAFETY: the description points at live data.
                    |library, name| unsafe { library.LoadComputePipeline(name, &desc) },
                    || match made {
                        // SAFETY: as above.
                        true => unsafe { self.0.CreateComputePipelineState(&desc) },
                        false => panic!("made again, not loaded"),
                    },
                )
            }
        }
        let made = Made(device, root, code);
        made.get(true).expect("made and stored");
        let saving = library.state.lock().unwrap_or_else(|e| e.into_inner());
        let (done, loaded) = std::sync::mpsc::channel();
        std::thread::scope(|s| {
            let made = &made;
            s.spawn(move || done.send(made.get(false).is_ok()));
            let loaded = loaded.recv_timeout(std::time::Duration::from_secs(30));
            drop(saving);
            assert_eq!(loaded, Ok(true), "the load waited for the save");
        });
    }

    #[test]
    fn a_rejected_blob_gives_an_empty_library() {
        let Some(device) = crate::host::host().and_then(|h| h.device.cast::<ID3D12Device1>().ok()) else {
            return; // no host GPU here
        };
        let library = open(&device, b"not a pipeline library".repeat(16)).expect("an empty library");
        // SAFETY: a plain query on a live library.
        assert!(unsafe { library.library.GetSerializedSize() } > 0);
        assert!(library._blob.is_empty());
    }

    /// Staging files of cut writes go: this process's at once, others'
    /// once old; the files themselves and anything else stay.
    #[test]
    fn stale_staging_files_are_swept() {
        let dir = tempfile::tempdir().unwrap();
        let old = std::time::SystemTime::now() - 2 * STAGING_KEPT;
        let file = |name: &str, modified: Option<std::time::SystemTime>| {
            let f = std::fs::File::create(dir.path().join(name)).unwrap();
            if let Some(t) = modified {
                f.set_modified(t).unwrap();
            }
        };
        let gone = [
            "host-pipelines.bin.tmp-7-0",
            "early-pipelines.txt.tmp-7-3",
            "host-pipelines.bin.tmp-8-1",
            "prebuilt-pipelines.txt.tmp-9-12",
        ];
        let kept = [
            "host-pipelines.bin",
            "early-pipelines.txt",
            "host-pipelines.bin.tmp-8-2", // another process's, recent
            "host-pipelines.bin.tmp-7-x", // not a staging name
            "host-pipelines.bin.tmp-7",   // not a staging name
            "other.bin.tmp-7-0",          // not a file of ours
        ];
        file(gone[0], None);
        file(gone[1], None);
        file(gone[2], Some(old));
        file(gone[3], Some(old));
        for name in kept {
            file(name, (!name.contains("tmp-8-2")).then_some(old));
        }
        std::fs::create_dir(dir.path().join("host-pipelines.bin.tmp-9-0")).unwrap();
        sweep_staging(dir.path(), 7, STAGING_KEPT);
        for name in gone {
            assert!(!dir.path().join(name).exists(), "{name} swept");
        }
        for name in kept.into_iter().chain(["host-pipelines.bin.tmp-9-0"]) {
            assert!(dir.path().join(name).exists(), "{name} kept");
        }
    }

    #[test]
    fn concurrent_write_atomically_never_fails_or_truncates() {
        let dir = tempfile::tempdir().unwrap();
        let target = dir.path().join("host-pipelines.bin");
        let threads: Vec<_> = (0..4)
            .map(|t| {
                let target = target.clone();
                std::thread::spawn(move || {
                    let payload = vec![b'a' + t as u8; 4096];
                    for _ in 0..30 {
                        write_atomically(&target, &payload).unwrap();
                    }
                })
            })
            .collect();
        for t in threads {
            t.join().unwrap();
        }
        let final_bytes = std::fs::read(&target).unwrap();
        assert_eq!(final_bytes.len(), 4096);
        assert!(final_bytes.iter().all(|&b| b == final_bytes[0]));
    }
}

#[cfg(test)]
mod keyed_tests {
    use super::*;

    /// Pruning keeps only directories this wrote: a `backup/` (or any
    /// directory without a hash name) in the cache directory survives,
    /// with what is in it, even when it is the oldest thing there.
    #[test]
    fn pruning_leaves_foreign_directories_alone() {
        let dir = std::env::temp_dir().join(format!("victory-keyed-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        // An old directory of ours, an old one of someone else's, and the
        // one this build uses.
        for (name, own) in [("00112233aabbccdd", true), ("backup", false)] {
            let at = dir.join(name);
            std::fs::create_dir_all(&at).unwrap();
            std::fs::write(at.join("sentinel"), b"keep me").unwrap();
            if own {
                std::fs::write(at.join("host-pipelines.bin"), b"library").unwrap();
            }
        }
        let current = dir.join(victory_shader::SOURCE_HASH);
        std::fs::create_dir_all(&current).unwrap();
        std::fs::write(current.join("host-pipelines.bin"), b"library").unwrap();

        assert_eq!(keyed(&dir), current);
        assert!(dir.join("backup").join("sentinel").is_file(), "a foreign directory was deleted");
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// A translation's directory is a library with the one-file library or
    /// with any single shard of a split one, not only shard 00 (review on
    /// #2706: the split writes only the shards that changed). A shard number
    /// past the split's sixteen, or no library file at all, is not one.
    #[test]
    fn any_shard_makes_a_library_directory() {
        let dir = std::env::temp_dir().join(format!("victory-shards-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        for (name, file, library) in [
            ("legacy", Some("host-pipelines.bin"), true),
            ("first", Some("host-pipelines.00.bin"), true),
            ("last", Some("host-pipelines.15.bin"), true),
            ("past", Some("host-pipelines.16.bin"), false),
            ("other", Some("sentinel"), false),
            ("empty", None, false),
        ] {
            let at = dir.join(name);
            std::fs::create_dir_all(&at).unwrap();
            if let Some(file) = file {
                std::fs::write(at.join(file), b"library").unwrap();
            }
            assert_eq!(library_in(&at), library, "{name}");
        }
        let _ = std::fs::remove_dir_all(&dir);
    }
}

