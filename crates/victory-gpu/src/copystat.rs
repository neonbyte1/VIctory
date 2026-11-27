//! Diagnostics: where the submission copies of title memory go (which
//! ranges, how often, of what memory) and what limits their rate, logged
//! with the frame-rate line. `VICTORY_COPY_MODE=plain` copies without the
//! mapped-page guard, `none` skips the copies (both for timing only).

use std::collections::HashMap;
use std::sync::Mutex;
use std::sync::atomic::{AtomicU64, Ordering::Relaxed};

/// Counters since the last line.
pub(crate) enum C {
    Submissions,
    Copies,
    CopyBytes,
    Refreshes,
    RefreshBytes,
    Skipped,
    SkippedBytes,
    Runs,
    RunUs,
    Queries,
    QueryUs,
    MoveUs,
    Unmapped,
    /// Write-watch askings and their time.
    Watches,
    WatchUs,
    /// Bytes of watched ranges checked, and of those found unchanged.
    Checked,
    Unchanged,
    /// Copies that met an unmapped page.
    Fallbacks,
    /// Writes taken by the guard on view memory.
    GuardFaults,
    /// Texture bytes detiled after a watched write; hashed; detiled after a
    /// hash change.
    TexWatched,
    TexHashed,
    TexChanged,
    /// Time in the kernel's write-watch answers, in the view check of the
    /// guard, in protecting; bytes and time of copies of memory neither
    /// watched nor guarded (copied whole).
    KernelUs,
    ViewQueryUs,
    ProtectUs,
    WholeBytes,
    WholeUs,
    /// Snapshots of ranges shaders write (bytes, time); page faults while
    /// preparing submissions and while copying back.
    SnapshotBytes,
    SnapshotUs,
    FaultsPrep,
    FaultsBack,
    /// Pages marked changed because a write landed on another mapping of
    /// them.
    AliasMarks,
    /// Time in the guard handler; protection changes making pages read-only
    /// (calls, pages) and giving write access back (pages).
    HandlerUs,
    ProtectCalls,
    ProtectPages,
    OpenPages,
    /// Tracked pages written (distinct, per frame), and of those the ones
    /// written the frame before too.
    WrittenPages,
    WrittenAgain,
    /// In change checks: time asking the runtime for other mappings, and
    /// waiting for the guard lock.
    AliasUs,
    GuardWaitUs,
    /// Preparing submissions: refresh batches done (copies and change
    /// checks); time in each kind of step (a step that flushes the batch
    /// counts that too); texture uploads for heaps; state preludes.
    FlushUs,
    StepRefreshUs,
    StepCopyUs,
    StepWorkUs,
    StepGpuWriteUs,
    StepTextureUs,
    StepHeapUs,
    StepMemoryUs,
    UploadsUs,
    PreludeUs,
    /// Copying back: bytes of copies compared with their snapshots, and
    /// the time.
    CompareBytes,
    CompareUs,
    /// Copying back pages shaders marked as written through views: pages
    /// compared, bytes written back, the time; pages that kept the title's
    /// bytes (it wrote them, their bytes from before unknown); pages whose
    /// bytes the title's first write saved.
    MarkedPages,
    MarkedBytes,
    MarkedUs,
    MarkedKept,
    SavedPages,
    /// Preparing submissions: descriptors the title copied into its heaps
    /// as bytes, made on the host first.
    SyncHeapsUs,
    /// Of those: heap pages and slots looked at (every slot of each page
    /// written), of the slots those whose record differed from the slot's
    /// (made again), those whose record names no live view any more
    /// (nulled), and the pages whose unchanged records had their view
    /// checked because a view died since.
    SyncPages,
    SyncSlots,
    SyncChanged,
    SyncStale,
    SyncRechecks,
    /// View entries freed by a resource's release (they go stale).
    ViewDeaths,
    /// Time in copies redone piece by piece after meeting an unmapped page.
    FallbackUs,
    /// Title threads writing a view into a shader-visible heap: time
    /// waiting for the heaps' lock (held by heap syncs and other writers).
    MirrorWaitUs,
    /// Copies in video memory (#3312), on completion: bytes the list after
    /// a submission read back put into the mirrors, and the time; pages the
    /// views' marks and the lists' page bitmaps named, of them those read
    /// back in that list already (predicted from the last completion that
    /// took them) and the bytes; bytes read back waited for, and the time.
    VramStagedBytes,
    VramStagedUs,
    VramNamedPages,
    VramPredictedPages,
    VramPredictedBytes,
    VramWaitedBytes,
    VramWaitedUs,
    /// Bytes read back speculatively (predicted) that no mark named.
    VramWastedBytes,
    /// Preparing: bytes of copies in video memory staged for upload, and
    /// the time.
    VramUpBytes,
    VramUpUs,
    /// Completions: all, barriers (the next submission waits for them),
    /// those that took the views' write marks.
    Completions,
    HeavyCompletions,
    MarksTaken,
    /// Submissions by why they are heavy (`HostList::heavy_reasons`, any
    /// of their lists): writes through the heap's views, work after them,
    /// bottom-of-pipe writes, missed-memory checks, declared GPU writes,
    /// title memory through page tables; and those marking through the
    /// heap's views that only read (not heavy).
    HeavyHeapWrites,
    HeavyAfter,
    HeavyWrites,
    HeavyMissed,
    HeavyGpuWrites,
    HeavyMemory,
    HeapReadsOnly,
    /// Submissions heavy for bottom-of-pipe writes alone.
    HeavyWritesOnly,
    /// Pages the marks or page bitmaps named whose bytes the GPU left as
    /// the mirror had them: copied back without a look at title memory.
    VramUnchangedPages,
    /// Argument clamps (`crate::indirect`): passes of one execution;
    /// windows opened (`VICTORY_CLAMP_WINDOW`), indirect draws they took;
    /// indirect draws clamped alone because their shaders may write
    /// arguments; indirect draws clamped in a prelude (hoisted); of the
    /// hoisted and windowed, those with read marks (the read-mark rule,
    /// `crate::draw`'s `may_write_arguments`, let them in).
    ClampSingle,
    ClampWindows,
    ClampWindowDraws,
    ClampStoring,
    ClampHoisted,
    ClampReadMarked,
    /// Copying back marked pages (`crate::shadow::copy_back_marks`): rounds
    /// of the guard lock and protection changes the copy-back made; runs
    /// gone back in one lease and their pages; byte runs and bytes written;
    /// pages written that a title write reached while open (left changed);
    /// pages the title wrote, gone back by their saved bytes (left changed).
    MarkLeases,
    MarkProtects,
    MarkRuns,
    MarkRunPages,
    MarkDiffRuns,
    MarkDiffBytes,
    MarkMisses,
    MarkMerged,
    /// Of the runs: pages opened only to bridge them; microseconds reading
    /// and comparing before the lease, waiting for it, changing
    /// protections, writing, checking after.
    MarkBridged,
    MarkPreUs,
    MarkWaitUs,
    MarkProtectUs,
    MarkWriteUs,
    MarkVerifyUs,
    /// Marked pages in the runtime's pools (private, write-watched) and in
    /// views (sections' pages), and the protection changes each cost.
    MarkPoolPages,
    MarkViewPages,
    MarkPoolProtects,
    MarkViewProtects,
    /// Marked pages written through the runtime's own mapping of them.
    MarkThrough,
    /// Submissions prepared while an earlier one was still in flight.
    PreparedOverlapping,
    /// Submissions whose preparing first completed those in flight (any of
    /// them, `Host::submit`): for a barrier among them, being one, the
    /// upload list, a list run again, the eviction pass, overlapping
    /// declared ranges.
    WaitBarrier,
    WaitAlone,
    WaitUploads,
    WaitListAgain,
    WaitEviction,
    WaitDeclared,
    /// Completions during a submission's preparing (`Host::settle`), by
    /// what it was about to read: pages the title changed under ranges of
    /// the same copy, another copy of written memory, bottom-of-pipe
    /// writes, title memory read directly, textures, title copies at
    /// submission; and the time.
    SettleSame,
    SettleAlias,
    SettleLabel,
    SettleFill,
    SettleTexture,
    SettleCopy,
    SettleUs,
    /// Early release (`crate::early`, `VICTORY_EARLY_RELEASE`): held
    /// segments released on values of writes in flight that title memory
    /// alone held; value writes registered in flight; refusals (counted
    /// once per word and writes in flight) by why: the word changed since,
    /// a later writer through a copy, a barrier, a timestamp, another size.
    EarlyReleased,
    EarlyWrites,
    EarlyRefusedCpu,
    EarlyRefusedWriter,
    EarlyRefusedBarrier,
    EarlyRefusedStamp,
    EarlyRefusedSize,
    /// Barriers (`crate::host`), each once by its first kind: title memory
    /// through page tables, missed-memory checks, writes through the heap's
    /// views, unnamed work after, named work after (without
    /// `VICTORY_EARLY_BARRIERS`), marks not captured apart. Barriers
    /// prepared with earlier submissions left in flight, and submissions
    /// with named work after left light (`VICTORY_EARLY_BARRIERS`); heavy
    /// submissions with named work after.
    BarrierMemory,
    BarrierMissed,
    BarrierHeapWrites,
    BarrierAfter,
    BarrierAfterNamed,
    BarrierMarks,
    AloneWaived,
    AfterNamedLight,
    HeavyAfterNamed,
    /// Clean copies of heaps and page tables refreshed because a write in
    /// flight crosses them (`crate::shadow::refresh_where_needed`), and the
    /// settles only such clean copies (and unchanged textures) asked for.
    CleanCrossed,
    SettleAliasClean,
    SettleLabelClean,
    SettleTextureClean,
    /// Lazy view pages (`VICTORY_LAZY_VIEWS`, `crate::watch::Lease::lazify`):
    /// offered and made lazy instead of written; lazy pages shaders wrote
    /// again before any touch (nothing read or copied); first touches,
    /// each copying the page in (title reads, title writes, our host);
    /// dropped unwritten (withheld); time of those copies; protection
    /// changes for lazy pages; offered but not made so: changed, not
    /// guarded, shown elsewhere where it cannot be, held by another
    /// tracker, other.
    LazyOffered,
    LazyMade,
    LazyAgain,
    LazyTitleReads,
    LazyTitleWrites,
    LazyHostTouches,
    LazyDropped,
    LazyResolveUs,
    LazyProtects,
    LazyRefusedChanged,
    LazyRefusedUnguarded,
    LazyRefusedAlias,
    LazyRefusedHeld,
    LazyRefusedOther,
    /// `VICTORY_OWNER`: lazy pages our host read with the copy's bytes
    /// (`crate::watch::serve`), and the bytes; lazy pages another copy's
    /// copy-back took over (`crate::watch::Lease::take_over`).
    OwnerServed,
    OwnerServedBytes,
    OwnerTakenOver,
    /// `VICTORY_OWNER`: page addresses barred from going lazy for a while,
    /// given back twice within a few frames (`crate::watch::churn`).
    OwnerBarred,
}
const COUNTERS: usize = 165;
static COUNT: [AtomicU64; COUNTERS] = [const { AtomicU64::new(0) }; COUNTERS];

pub(crate) fn add(c: C, n: u64) {
    COUNT[c as usize].fetch_add(n, Relaxed);
}

/// How copies run: 0 guarded (as always), 1 plain, 2 not at all.
pub(crate) fn mode() -> u8 {
    static MODE: std::sync::OnceLock<u8> = std::sync::OnceLock::new();
    *MODE.get_or_init(|| match std::env::var("VICTORY_COPY_MODE").as_deref() {
        Ok("plain") => 1,
        Ok("none") => 2,
        _ => 0,
    })
}

/// Per source range (start, length): copies, bytes, microseconds, memory type.
type Ranges = HashMap<(usize, usize), (u64, u64, u64, u32)>;
static RANGES: Mutex<Option<Ranges>> = Mutex::new(None);

/// A copy of `len` bytes from `src` (memory type `kind`) took `us`.
pub(crate) fn range(src: usize, len: usize, us: u64, kind: u32) {
    let mut map = RANGES.lock().unwrap_or_else(|e| e.into_inner());
    let e = map
        .get_or_insert_with(HashMap::new)
        .entry((src, len))
        .or_insert((0, 0, 0, kind));
    e.0 += 1;
    e.1 += len as u64;
    e.2 += us;
}

/// Snapshot bytes per host copy (its title start and length).
static SNAPSHOTS: Mutex<Option<HashMap<(u64, u64), u64>>> = Mutex::new(None);

/// A snapshot of `bytes` of the copy of `[start, start + len)`.
pub(crate) fn snapshot(start: u64, len: u64, bytes: u64) {
    *SNAPSHOTS
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .get_or_insert_with(HashMap::new)
        .entry((start, len))
        .or_default() += bytes;
}

/// Our process's working set, bytes.
fn working_set() -> u64 {
    use windows_sys::Win32::System::ProcessStatus::{
        K32GetProcessMemoryInfo, PROCESS_MEMORY_COUNTERS,
    };
    use windows_sys::Win32::System::Threading::GetCurrentProcess;
    let mut c = PROCESS_MEMORY_COUNTERS {
        cb: size_of::<PROCESS_MEMORY_COUNTERS>() as u32,
        ..Default::default()
    };
    // SAFETY: our own process and a sized structure.
    unsafe { K32GetProcessMemoryInfo(GetCurrentProcess(), &mut c, c.cb) };
    c.WorkingSetSize as u64
}

/// Our process's hard page faults so far (read from disk or the page
/// file), from the system's process list: `HardFaultCount` at +0x10 of
/// each entry, the process id at +0x50.
fn hard_faults() -> u64 {
    #[link(name = "ntdll")]
    unsafe extern "system" {
        fn NtQuerySystemInformation(class: u32, info: *mut u8, len: u32, ret: *mut u32) -> i32;
    }
    const PROCESSES: u32 = 5;
    let id = std::process::id() as usize;
    let mut buf = vec![0u8; 4 << 20];
    let mut ret = 0;
    // SAFETY: a buffer of that length.
    if unsafe { NtQuerySystemInformation(PROCESSES, buf.as_mut_ptr(), buf.len() as u32, &mut ret) } < 0 {
        return 0;
    }
    let mut at = 0usize;
    loop {
        let entry = &buf[at..];
        let next = u32::from_le_bytes(entry[0..4].try_into().unwrap_or_default()) as usize;
        let pid = usize::from_le_bytes(entry[0x50..0x58].try_into().unwrap_or_default());
        if pid == id {
            return u64::from(u32::from_le_bytes(entry[0x10..0x14].try_into().unwrap_or_default()));
        }
        if next == 0 || at + next + 0x58 > buf.len() {
            return 0;
        }
        at += next;
    }
}

pub(crate) fn page_faults() -> u64 {
    use windows_sys::Win32::System::ProcessStatus::{
        K32GetProcessMemoryInfo, PROCESS_MEMORY_COUNTERS,
    };
    use windows_sys::Win32::System::Threading::GetCurrentProcess;
    let mut c = PROCESS_MEMORY_COUNTERS {
        cb: size_of::<PROCESS_MEMORY_COUNTERS>() as u32,
        ..Default::default()
    };
    // SAFETY: our own process and a sized structure.
    unsafe { K32GetProcessMemoryInfo(GetCurrentProcess(), &mut c, c.cb) };
    u64::from(c.PageFaultCount)
}

/// Logs the counters per frame over `frames` frames, and the ranges that
/// took most bytes.
pub(crate) fn log(frames: f64) {
    static FAULTS: AtomicU64 = AtomicU64::new(0);
    let faults = page_faults();
    let faults = faults - FAULTS.swap(faults, Relaxed).min(faults);
    let c: Vec<f64> = COUNT
        .iter()
        .map(|a| a.swap(0, Relaxed) as f64 / frames)
        .collect();
    let mb = |v: f64| v / 1e6;
    let ms = |v: f64| v / 1e3;
    tracing::info!(
        "copies per frame: {:.1} submissions; {:.0} title copies {:.1} MB; {:.0} refreshes {:.1} MB, \
         {:.0} skipped {:.1} MB; {:.0} runs {:.1} ms; {:.0} queries {:.2} ms, moves {:.2} ms, \
         {:.2} MB unmapped; {:.0} page faults; {:.0} watches {:.2} ms, {:.1} MB checked, {:.1} MB \
         unchanged; {:.1} fallbacks ({:.1} ms); {:.0} guard faults; textures {:.1} MB watched and \
         changed, {:.1} MB hashed, {:.1} MB of them changed; watch time: kernel {:.1} ms, view queries {:.1} ms, protecting {:.1} ms; copied whole (not tracked) {:.1} MB in {:.1} ms; snapshots {:.1} MB in {:.1} ms; page faults {:.0} preparing, {:.0} copying back; {:.0} tracked pages marked through another mapping; \
         guards: {:.2} ms in the handler, {:.0} protect calls for {:.0} pages, {:.0} pages opened; \
         {:.0} tracked pages written ({:.0} of them the frame before too); change checks: {:.1} ms \
         asking for other mappings, {:.1} ms waiting for the guard lock; preparing: {:.1} ms in \
         refresh batches, steps (each with the batch it flushes): refresh {:.1} ms, copy {:.1} ms, \
         work {:.1} ms, GPU writes {:.1} ms, textures {:.1} ms, heaps {:.1} ms, title memory {:.1} ms; \
         heap texture uploads {:.1} ms, state preludes {:.1} ms, heap slots the title copied {:.1} ms ({:.0} pages, {:.0} slots: {:.0} changed, {:.0} of a dead view, {:.0} pages re-checked after {:.0} view deaths), title threads waiting to write views {:.1} ms; copying back compared {:.1} MB          in {:.1} ms; marked pages: {:.0} compared, {:.1} MB written back in {:.1} ms, {:.0} kept the title's, {:.0} saved; working set {:.0} MB, \
         {:.0} hard faults",
        c[C::Submissions as usize],
        c[C::Copies as usize],
        mb(c[C::CopyBytes as usize]),
        c[C::Refreshes as usize],
        mb(c[C::RefreshBytes as usize]),
        c[C::Skipped as usize],
        mb(c[C::SkippedBytes as usize]),
        c[C::Runs as usize],
        ms(c[C::RunUs as usize]),
        c[C::Queries as usize],
        ms(c[C::QueryUs as usize]),
        ms(c[C::MoveUs as usize]),
        mb(c[C::Unmapped as usize]),
        faults as f64 / frames,
        c[C::Watches as usize],
        ms(c[C::WatchUs as usize]),
        mb(c[C::Checked as usize]),
        mb(c[C::Unchanged as usize]),
        c[C::Fallbacks as usize],
        ms(c[C::FallbackUs as usize]),
        c[C::GuardFaults as usize],
        mb(c[C::TexWatched as usize]),
        mb(c[C::TexHashed as usize]),
        mb(c[C::TexChanged as usize]),
        ms(c[C::KernelUs as usize]),
        ms(c[C::ViewQueryUs as usize]),
        ms(c[C::ProtectUs as usize]),
        mb(c[C::WholeBytes as usize]),
        ms(c[C::WholeUs as usize]),
        mb(c[C::SnapshotBytes as usize]),
        ms(c[C::SnapshotUs as usize]),
        c[C::FaultsPrep as usize],
        c[C::FaultsBack as usize],
        c[C::AliasMarks as usize],
        ms(c[C::HandlerUs as usize]),
        c[C::ProtectCalls as usize],
        c[C::ProtectPages as usize],
        c[C::OpenPages as usize],
        c[C::WrittenPages as usize],
        c[C::WrittenAgain as usize],
        ms(c[C::AliasUs as usize]),
        ms(c[C::GuardWaitUs as usize]),
        ms(c[C::FlushUs as usize]),
        ms(c[C::StepRefreshUs as usize]),
        ms(c[C::StepCopyUs as usize]),
        ms(c[C::StepWorkUs as usize]),
        ms(c[C::StepGpuWriteUs as usize]),
        ms(c[C::StepTextureUs as usize]),
        ms(c[C::StepHeapUs as usize]),
        ms(c[C::StepMemoryUs as usize]),
        ms(c[C::UploadsUs as usize]),
        ms(c[C::PreludeUs as usize]),
        ms(c[C::SyncHeapsUs as usize]),
        c[C::SyncPages as usize],
        c[C::SyncSlots as usize],
        c[C::SyncChanged as usize],
        c[C::SyncStale as usize],
        c[C::SyncRechecks as usize],
        c[C::ViewDeaths as usize],
        ms(c[C::MirrorWaitUs as usize]),
        mb(c[C::CompareBytes as usize]),
        ms(c[C::CompareUs as usize]),
        c[C::MarkedPages as usize],
        mb(c[C::MarkedBytes as usize]),
        ms(c[C::MarkedUs as usize]),
        c[C::MarkedKept as usize],
        c[C::SavedPages as usize],
        working_set() as f64 / 1e6,
        {
            static HARD: AtomicU64 = AtomicU64::new(0);
            let hard = hard_faults();
            (hard - HARD.swap(hard, Relaxed).min(hard)) as f64 / frames
        },
    );
    let (single, windows, draws, hoisted) = (
        c[C::ClampSingle as usize],
        c[C::ClampWindows as usize],
        c[C::ClampWindowDraws as usize],
        c[C::ClampHoisted as usize],
    );
    tracing::info!(
        "clamps per frame: {:.0} passes ({single:.0} single, {windows:.0} windows of {draws:.0} indirect draws, {:.1} a window), {hoisted:.0} indirect draws hoisted to preludes, {:.0} hoisted or windowed by the read-mark rule, {:.0} indirect draws alone as they may write arguments",
        single + windows,
        if windows > 0.0 { draws / windows } else { 0.0 },
        c[C::ClampReadMarked as usize],
        c[C::ClampStoring as usize],
    );
    let held: Vec<String> = crate::watch::held_times()
        .iter()
        .map(|(line, total, longest)| format!("watch.rs:{line} {:.1} ms (longest {longest:.1})", total / frames))
        .collect();
    tracing::info!("  guard lock held per frame: {}", held.join(", "));
    tracing::info!(
        "  marked copy-back per frame: {:.0} guard lock rounds, {:.0} protection changes; {:.0} runs of \
         {:.0} pages (the rest page by page); {:.0} byte runs, {:.1} MB written; {:.0} pages a title \
         write reached while open, {:.0} pages the title wrote gone back by saved bytes, {:.0} pages \
         opened to bridge runs; in runs: {:.1} ms reading and comparing, {:.1} ms waiting for the \
         lease, {:.1} ms changing protections, {:.1} ms writing, {:.1} ms checking",
        c[C::MarkLeases as usize],
        c[C::MarkProtects as usize],
        c[C::MarkRuns as usize],
        c[C::MarkRunPages as usize],
        c[C::MarkDiffRuns as usize],
        mb(c[C::MarkDiffBytes as usize]),
        c[C::MarkMisses as usize],
        c[C::MarkMerged as usize],
        c[C::MarkBridged as usize],
        ms(c[C::MarkPreUs as usize]),
        ms(c[C::MarkWaitUs as usize]),
        ms(c[C::MarkProtectUs as usize]),
        ms(c[C::MarkWriteUs as usize]),
        ms(c[C::MarkVerifyUs as usize]),
    );
    tracing::info!(
        "  marked pages per frame by memory: pools {:.0} pages, {:.0} protection changes; views {:.0} \
         pages, {:.0} protection changes; {:.0} pages written through our own mapping",
        c[C::MarkPoolPages as usize],
        c[C::MarkPoolProtects as usize],
        c[C::MarkViewPages as usize],
        c[C::MarkViewProtects as usize],
        c[C::MarkThrough as usize],
    );
    let (checks, missed, last) = crate::watch::take_lazy_checks();
    tracing::info!(
        "  lazy view pages per frame (on: {}, owner: {}): {:.0} offered, {:.0} made lazy (not copied back), {:.0} marked \
         again before any touch (never copied); copied on demand: {:.0} at title reads, {:.0} at title writes, \
         {:.0} for our host, {:.1} ms; {:.0} dropped unwritten; {} lazy now; {:.0} protection changes; not made \
         lazy: {:.0} changed, {:.0} not guarded, {:.0} shown elsewhere, {:.0} held by another tracker, {:.0} other; checked {:.0}, {:.0} \
         written behind our back; our host given the copy's bytes {:.0} pages ({:.1} MB), taken over by another copy {:.0}, \
         barred for churning {:.1}",
        crate::watch::lazy_views(),
        crate::watch::owner(),
        c[C::LazyOffered as usize],
        c[C::LazyMade as usize],
        c[C::LazyAgain as usize],
        c[C::LazyTitleReads as usize],
        c[C::LazyTitleWrites as usize],
        c[C::LazyHostTouches as usize],
        ms(c[C::LazyResolveUs as usize]),
        c[C::LazyDropped as usize],
        crate::watch::lazy_live(),
        c[C::LazyProtects as usize],
        c[C::LazyRefusedChanged as usize],
        c[C::LazyRefusedUnguarded as usize],
        c[C::LazyRefusedAlias as usize],
        c[C::LazyRefusedHeld as usize],
        c[C::LazyRefusedOther as usize],
        checks as f64 / frames,
        missed as f64 / frames,
        c[C::OwnerServed as usize],
        c[C::OwnerServedBytes as usize] / 1e6,
        c[C::OwnerTakenOver as usize],
        c[C::OwnerBarred as usize],
    );
    if missed > 0 {
        tracing::warn!(pages = missed, last = ?last, "lazy view pages written behind our back (their bytes lost at the first touch)");
    }
    let sites: Vec<String> = crate::watch::site_counts()
        .iter()
        .zip(crate::watch::SITE_NAMES)
        .filter(|(n, _)| n[0] > 0)
        .map(|(n, name)| {
            format!(
                "{name} {:.0} calls {:.0} pages ({:.1} failed ranges)",
                n[0] as f64 / frames,
                n[1] as f64 / frames,
                n[2] as f64 / frames
            )
        })
        .collect();
    tracing::info!("  protection changes per frame by caller: {}", sites.join(", "));
    tracing::info!(
        "  copies in video memory per frame: {:.1} completions, {:.1} barriers (heavy), {:.1} taking write marks; \
         uploads staged {:.1} MB in {:.1} ms; read back in the submission and put in the mirrors \
         {:.1} MB in {:.1} ms; pages named by marks and page bitmaps {:.0}, {:.0} of them read back in \
         the submission ({:.1} MB, {:.1} MB read for none); read back waited for {:.1} MB in {:.1} ms; \
         {:.0} named pages left as they were by the GPU",
        c[C::Completions as usize],
        c[C::HeavyCompletions as usize],
        c[C::MarksTaken as usize],
        mb(c[C::VramUpBytes as usize]),
        ms(c[C::VramUpUs as usize]),
        mb(c[C::VramStagedBytes as usize]),
        ms(c[C::VramStagedUs as usize]),
        c[C::VramNamedPages as usize],
        c[C::VramPredictedPages as usize],
        mb(c[C::VramPredictedBytes as usize]),
        mb(c[C::VramWastedBytes as usize]),
        mb(c[C::VramWaitedBytes as usize]),
        ms(c[C::VramWaitedUs as usize]),
        c[C::VramUnchangedPages as usize],
    );
    tracing::info!(
        "  heavy submissions per frame by why (any of their lists): {:.1} write through the heap's views, \
         {:.1} work after, {:.1} bottom-of-pipe writes ({:.1} for those alone), {:.1} missed-memory checks, \
         {:.1} declared GPU writes, {:.1} title memory through page tables; {:.1} light that marking heap reads alone made heavy before",
        c[C::HeavyHeapWrites as usize],
        c[C::HeavyAfter as usize] + c[C::HeavyAfterNamed as usize],
        c[C::HeavyWrites as usize],
        c[C::HeavyWritesOnly as usize],
        c[C::HeavyMissed as usize],
        c[C::HeavyGpuWrites as usize],
        c[C::HeavyMemory as usize],
        c[C::HeapReadsOnly as usize],
    );
    tracing::info!(
        "  light submissions per frame: {:.1} prepared with an earlier one in flight; waited for those in flight \
         before preparing: {:.1} behind a barrier, {:.1} being one, {:.1} \
         uploads, {:.1} a list again, {:.1} eviction, {:.1} overlapping declared ranges; completed while \
         preparing: {:.1} changed pages under their ranges, {:.1} other copies, {:.1} bottom-of-pipe writes, \
         {:.1} title reads, {:.1} textures, {:.1} title copies ({:.1} ms)",
        c[C::PreparedOverlapping as usize],
        c[C::WaitBarrier as usize],
        c[C::WaitAlone as usize],
        c[C::WaitUploads as usize],
        c[C::WaitListAgain as usize],
        c[C::WaitEviction as usize],
        c[C::WaitDeclared as usize],
        c[C::SettleSame as usize],
        c[C::SettleAlias as usize],
        c[C::SettleLabel as usize],
        c[C::SettleFill as usize],
        c[C::SettleTexture as usize],
        c[C::SettleCopy as usize],
        ms(c[C::SettleUs as usize]),
    );
    tracing::info!(
        "  early release per frame: {:.1} held segments released on values in flight, of {:.1} bottom-of-pipe \
         value writes in flight; refused (title memory read): {:.1} word changed since, {:.1} a later writer \
         through a copy, {:.1} a barrier, {:.1} a timestamp, {:.1} another size",
        c[C::EarlyReleased as usize],
        c[C::EarlyWrites as usize],
        c[C::EarlyRefusedCpu as usize],
        c[C::EarlyRefusedWriter as usize],
        c[C::EarlyRefusedBarrier as usize],
        c[C::EarlyRefusedStamp as usize],
        c[C::EarlyRefusedSize as usize],
    );
    tracing::info!(
        "  barriers per frame by kind: {:.1} title memory through page tables, {:.1} missed-memory checks, \
         {:.1} writes through the heap's views, {:.1} unnamed work after, {:.1} named work after, {:.1} marks \
         not captured; {:.1} prepared with earlier ones left in flight, {:.1} with named work after left light \
         (VICTORY_EARLY_BARRIERS); clean copies refreshed for writes in flight across them {:.1}, settles they \
         alone asked for: {:.1} other copies, {:.1} bottom-of-pipe writes, {:.1} textures",
        c[C::BarrierMemory as usize],
        c[C::BarrierMissed as usize],
        c[C::BarrierHeapWrites as usize],
        c[C::BarrierAfter as usize],
        c[C::BarrierAfterNamed as usize],
        c[C::BarrierMarks as usize],
        c[C::AloneWaived as usize],
        c[C::AfterNamedLight as usize],
        c[C::CleanCrossed as usize],
        c[C::SettleAliasClean as usize],
        c[C::SettleLabelClean as usize],
        c[C::SettleTextureClean as usize],
    );
    for line in crate::census::lines(frames) {
        tracing::info!("{line}");
    }
    let mut snapshots: Vec<((u64, u64), u64)> =
        SNAPSHOTS.lock().unwrap_or_else(|e| e.into_inner()).take().into_iter().flatten().collect();
    snapshots.sort_by_key(|&(_, bytes)| std::cmp::Reverse(bytes));
    let snapshots: Vec<String> = snapshots
        .iter()
        .take(6)
        .map(|((start, len), bytes)| format!("{start:#x}+{len:#x} {:.1} MB", *bytes as f64 / 1e6 / frames))
        .collect();
    tracing::info!("  snapshots per frame by copy: {}", snapshots.join(", "));
    let ranges = RANGES.lock().unwrap_or_else(|e| e.into_inner()).take();
    let mut ranges: Vec<_> = ranges.into_iter().flatten().collect();
    ranges.sort_by_key(|&(_, (_, bytes, _, _))| std::cmp::Reverse(bytes));
    for ((src, len), (n, bytes, us, _)) in ranges.into_iter().take(10) {
        use windows_sys::Win32::System::Memory::{MEMORY_BASIC_INFORMATION, VirtualQuery};
        // SAFETY: plain data filled in by the query.
        let mut info: MEMORY_BASIC_INFORMATION = unsafe { std::mem::zeroed() };
        // SAFETY: a valid out pointer of the right size.
        unsafe { VirtualQuery(src as *const _, &mut info, size_of_val(&info)) };
        let kind = info.Type;
        tracing::info!(
            "  copied from {src:#x}+{len:#x} (memory {kind:#x}): {:.1}x/frame, {:.2} MB/frame, {:.2} ms/frame",
            n as f64 / frames,
            bytes as f64 / 1e6 / frames,
            us as f64 / 1e3 / frames
        );
    }
}

/// A global allocator that counts the bytes Rust code holds (the
/// statistics' "our heap"; the binary installs it in diagnostics builds).
pub struct CountingAlloc;

/// Bytes [`CountingAlloc`] has handed out and not taken back.
pub(crate) static HEAP_BYTES: AtomicU64 = AtomicU64::new(0);

/// Allocations of at least this many bytes name their caller, once per
/// power-of-two size class (who holds the big blocks of our heap).
const LARGE: usize = 32 << 20;

/// Logs the stack of a large allocation of `size` bytes, the first of its
/// size class.
fn note_large(size: usize) {
    static SEEN: std::sync::Mutex<u64> = std::sync::Mutex::new(0);
    std::thread_local!(static INSIDE: std::cell::Cell<bool> = const { std::cell::Cell::new(false) });
    let class = 1u64 << size.ilog2().min(63);
    if INSIDE.with(|i| i.replace(true)) {
        return;
    }
    let first = {
        let mut seen = SEEN.lock().unwrap_or_else(|e| e.into_inner());
        let first = *seen & class == 0;
        *seen |= class;
        first
    };
    if first {
        let trace = std::backtrace::Backtrace::force_capture().to_string();
        let frames: Vec<&str> = trace.lines().filter(|l| l.contains("victory")).take(24).collect();
        tracing::info!(mb = size >> 20, "a large heap allocation (first of its size class): {}", frames.join(" | "));
    }
    INSIDE.with(|i| i.set(false));
}

// SAFETY: every call goes to the system allocator unchanged; only counted.
unsafe impl std::alloc::GlobalAlloc for CountingAlloc {
    unsafe fn alloc(&self, layout: std::alloc::Layout) -> *mut u8 {
        // SAFETY: the caller's layout, passed on.
        let p = unsafe { std::alloc::System.alloc(layout) };
        if !p.is_null() {
            HEAP_BYTES.fetch_add(layout.size() as u64, Relaxed);
            if layout.size() >= LARGE {
                note_large(layout.size());
            }
        }
        p
    }

    unsafe fn alloc_zeroed(&self, layout: std::alloc::Layout) -> *mut u8 {
        // SAFETY: as above.
        let p = unsafe { std::alloc::System.alloc_zeroed(layout) };
        if !p.is_null() {
            HEAP_BYTES.fetch_add(layout.size() as u64, Relaxed);
            if layout.size() >= LARGE {
                note_large(layout.size());
            }
        }
        p
    }

    unsafe fn dealloc(&self, p: *mut u8, layout: std::alloc::Layout) {
        HEAP_BYTES.fetch_sub(layout.size() as u64, Relaxed);
        // SAFETY: a block this allocator handed out with this layout.
        unsafe { std::alloc::System.dealloc(p, layout) }
    }

    unsafe fn realloc(&self, p: *mut u8, layout: std::alloc::Layout, size: usize) -> *mut u8 {
        // SAFETY: as above.
        let q = unsafe { std::alloc::System.realloc(p, layout, size) };
        if !q.is_null() {
            HEAP_BYTES.fetch_sub(layout.size() as u64, Relaxed);
            HEAP_BYTES.fetch_add(size as u64, Relaxed);
            if size >= LARGE {
                note_large(size);
            }
        }
        q
    }
}
