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
    /// waiting for the guard lock; rounds of the guard lock the takes made
    /// (a refresh batch takes its copies' pages in one).
    AliasUs,
    GuardWaitUs,
    TakeRounds,
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
    /// `VICTORY_GPU_GAPS` (`crate::indirect::gaps_on`): indirect dispatches
    /// and indirect draws whose shaders may store clamped in a window;
    /// executions the open window would have taken but for a barrier or a
    /// pipeline sync after one that may store. Clamped alone: dispatches
    /// (switch off), dispatches with spread groups, executions a window
    /// refused (alignment, size, sources) or a full prelude sent there.
    /// Resource-barrier calls the clamps recorded (passes and their
    /// transitions back); the title's barrier and barrier-group calls; its
    /// pipeline syncs (they record nothing). Host lists run (preludes and
    /// state lists included) and `ExecuteCommandLists` calls.
    ClampGapsDispatches,
    ClampGapsStoring,
    ClampGapsStale,
    ClampAloneDispatch,
    ClampAloneSpread,
    ClampAloneOther,
    ClampBarrierCalls,
    TitleBarrierCalls,
    PipelineSyncs,
    ListsRun,
    EclCalls,
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
    /// A settle before a page table's rewrite (`crate::memory::prepare`,
    /// `VICTORY_PT_BARRIER`), counted only when its last user was in flight.
    SettleTable,
    /// A settle of a completion whose middle runs off the submission lock
    /// (`VICTORY_PIPELINE`) by a step that touches its copies without a
    /// crossing (`crate::host::Settled::Busy`).
    SettleBusy,
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
    /// `VICTORY_OWNER`: lazy pages' bytes armed copies kept: at arming
    /// (copies the GPU writes in place), when another copy's copy-back
    /// marked them (copies in video memory).
    OwnerKeptArming,
    OwnerKeptMarking,
    /// `VICTORY_OWNER`: microseconds in `crate::watch::serve` (lazy pages'
    /// part) and in `crate::watch::Tracked::arm_keeping`.
    OwnerServeUs,
    OwnerKeepUs,
    /// Dispatches of the BVH builder's pipelines run and held back
    /// (`crate::pipeline::builder`, `VICTORY_BVH_HALVES`).
    BvhRun,
    BvhHeld,
    /// The frame's serial chain, per frame: time a submitting thread was
    /// blocked on the host GPU, by what it waited for (`Host::complete_while`
    /// and `Host::finish`, [`crate::host::Waited`]): a barrier's page tables,
    /// another barrier (a submission waited for only because it must be alone
    /// is one too), the upload list, a list run again, the eviction pass,
    /// overlapping declared ranges, a settle, and other completions (the
    /// completion thread's, `complete_all`/`complete_held`, `execute`). The
    /// rest of the frame line's "on the GPU" is `ExecuteCommandLists`.
    BlockBarrierMemoryUs,
    BlockBarrierOtherUs,
    BlockUploadsUs,
    BlockListAgainUs,
    BlockEvictionUs,
    BlockDeclaredUs,
    BlockSettleUs,
    BlockOtherUs,
    /// Copying back, by phase (`Host::finish`, under the submission lock):
    /// the copies in video memory read back into the mirrors (`vram_completed`,
    /// including the wait for unpredicted pages, `VramWaitedUs`), the lazy-page
    /// check, the marked pages' copy-back (`MarkedUs`), the heap slots to read
    /// (`take_reads` + `make_read`), the declared ranges' copy-back
    /// (`CompareUs`), and the lists' work after them. The rest of the frame
    /// line's "copying back" is the steps around those.
    BackVramUs,
    BackVerifyUs,
    BackMarksUs,
    BackReadUs,
    BackRangesUs,
    BackAfterUs,
    /// Submissions prepared with nothing in flight: the pipe was empty, so
    /// nothing of theirs overlapped this submission's GPU work
    /// (`PreparedOverlapping` counts the others).
    PreparedEmpty,
    /// Submissions using title memory through page tables
    /// (`crate::host`'s `light_reasons`, `VICTORY_PT_BARRIER`): barriers with
    /// the switch off; with it, left in flight, or barriers still for a
    /// program storing through a table or for another reason. Ranges a miss
    /// may withhold registered in flight for those left there
    /// (`crate::shadow::KEPT`). How those left in flight completed: by a
    /// settle of a later preparing, waited for before one, or by the
    /// completion thread (or a wait for all); the fence time of all three.
    TablesOff,
    TablesLight,
    TablesKeptStores,
    TablesKeptOther,
    TablesKeptRanges,
    TablesBySettle,
    TablesByWait,
    TablesByOther,
    TablesWaitedUs,
    /// Copying back (the frame line's "copying back"), by who did it: the
    /// completion thread, a settle of a preparing, the waits before a title
    /// submission is prepared, anything else (`crate::host`'s
    /// `pipeline::copied_back`).
    BackThreadUs,
    BackSettleUs,
    BackWaitUs,
    BackOtherUs,
    /// `VICTORY_PIPELINE`: completions whose middle ran off the submission
    /// lock, and completions the completion thread ran whole for want of a
    /// footprint (a page table's probes named pages); the middles' time,
    /// of it the time a preparing ran beside it (the preparing's own waits
    /// for it left out), and the waits of a lock holder for a middle still
    /// running, and their time.
    PipeSplit,
    PipeWhole,
    PipeMiddleUs,
    PipeOverlapUs,
    PipeWaits,
    PipeWaitUs,
    /// Not a counter: how many there are. Keep it last.
    Count,
}
const COUNTERS: usize = C::Count as usize;

struct Counters([AtomicU64; COUNTERS]);

impl Counters {
    const fn new() -> Self {
        Counters([const { AtomicU64::new(0) }; COUNTERS])
    }

    fn add(&self, c: C, n: u64) {
        self.0[c as usize].fetch_add(n, Relaxed);
    }

    /// Each counter per frame over `frames` frames, zeroing it.
    fn take(&self, frames: f64) -> Vec<f64> {
        self.0
            .iter()
            .map(|a| a.swap(0, Relaxed) as f64 / frames)
            .collect()
    }
}

static COUNT: Counters = Counters::new();

pub(crate) fn add(c: C, n: u64) {
    COUNT.add(c, n);
}

/// The counters since they were last taken or logged, by [`C`] (tests).
#[cfg(test)]
pub(crate) fn take() -> Vec<u64> {
    COUNT.0.iter().map(|a| a.swap(0, Relaxed)).collect()
}

/// `us` microseconds of a submitting thread blocked on the host GPU,
/// waiting for `wait` (`crate::host::Waited`).
pub(crate) fn waited(wait: crate::host::Waited, us: u64) {
    use crate::host::Waited::{
        BarrierMemory, BarrierOther, Completion, Declared, Eviction, ListAgain, Settle, Uploads,
    };
    add(
        match wait {
            BarrierMemory => C::BlockBarrierMemoryUs,
            BarrierOther => C::BlockBarrierOtherUs,
            Uploads => C::BlockUploadsUs,
            ListAgain => C::BlockListAgainUs,
            Eviction => C::BlockEvictionUs,
            Declared => C::BlockDeclaredUs,
            Settle => C::BlockSettleUs,
            Completion => C::BlockOtherUs,
        },
        us,
    );
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

/// Copies, bytes, microseconds, memory type.
type Copies = (u64, u64, u64, u32);

/// Per source range (start, length): its copies.
struct Ranges(Mutex<Option<HashMap<(usize, usize), Copies>>>);

impl Ranges {
    const fn new() -> Self {
        Ranges(Mutex::new(None))
    }

    fn add(&self, src: usize, len: usize, us: u64, kind: u32) {
        let mut map = self.0.lock().unwrap_or_else(|e| e.into_inner());
        let e = map
            .get_or_insert_with(HashMap::new)
            .entry((src, len))
            .or_insert((0, 0, 0, kind));
        e.0 += 1;
        e.1 += len as u64;
        e.2 += us;
    }

    /// The ranges so far, most bytes first, emptying the table.
    fn take(&self) -> Vec<((usize, usize), Copies)> {
        let ranges = self.0.lock().unwrap_or_else(|e| e.into_inner()).take();
        let mut ranges: Vec<_> = ranges.into_iter().flatten().collect();
        ranges.sort_by_key(|&(_, (_, bytes, _, _))| std::cmp::Reverse(bytes));
        ranges
    }
}

static RANGES: Ranges = Ranges::new();

/// A copy of `len` bytes from `src` (memory type `kind`) took `us`.
pub(crate) fn range(src: usize, len: usize, us: u64, kind: u32) {
    RANGES.add(src, len, us, kind);
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
    if unsafe { NtQuerySystemInformation(PROCESSES, buf.as_mut_ptr(), buf.len() as u32, &mut ret) }
        < 0
    {
        return 0;
    }
    let mut at = 0usize;
    loop {
        let entry = &buf[at..];
        let next = u32::from_le_bytes(entry[0..4].try_into().unwrap_or_default()) as usize;
        let pid = usize::from_le_bytes(entry[0x50..0x58].try_into().unwrap_or_default());
        if pid == id {
            return u64::from(u32::from_le_bytes(
                entry[0x10..0x14].try_into().unwrap_or_default(),
            ));
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
/// took most bytes. `gpu_ms` and `back_ms`: the frame line's "on the GPU"
/// (`SUBMISSION_US[1]`) and "copying back" (`COPY_BACK[0]`) per frame, which
/// the first line splits.
pub(crate) fn log(frames: f64, gpu_ms: f64, back_ms: f64) {
    static FAULTS: AtomicU64 = AtomicU64::new(0);
    let faults = page_faults();
    let faults = faults - FAULTS.swap(faults, Relaxed).min(faults);
    let c = COUNT.take(frames);
    let mb = |v: f64| v / 1e6;
    let ms = |v: f64| v / 1e3;
    // The frame's serial chain in one line, in the five-second block: what the
    // time blocked on the GPU was waiting for, what copy-back spent its time
    // on (the phases of `Host::finish` under the submission lock), and how
    // many submissions were prepared with an earlier one still in flight.
    let blocks = [
        C::BlockBarrierMemoryUs,
        C::BlockBarrierOtherUs,
        C::BlockUploadsUs,
        C::BlockListAgainUs,
        C::BlockEvictionUs,
        C::BlockDeclaredUs,
        C::BlockSettleUs,
        C::BlockOtherUs,
    ]
    .map(|k| ms(c[k as usize]));
    tracing::info!(
        "chain per frame: {gpu_ms:.1} ms on the GPU or in ExecuteCommandLists: {:.1} waiting for a \
         barrier's page tables, {:.1} another barrier, {:.1} the upload list, {:.1} a list run \
         again, {:.1} the eviction pass, {:.1} overlapping declared ranges, {:.1} settles, {:.1} \
         other completions; copying back {back_ms:.1} ms: \
         {:.1} read-backs ({:.1} waited for), {:.1} marked pages ({:.1} reading and comparing, {:.1} \
         waiting for the lease, {:.1} protecting, {:.1} writing, {:.1} verifying), {:.1} declared \
         ranges ({:.1} comparing), {:.1} slots to read, {:.1} work after, {:.1} lazy-page check; \
         {:.0} submissions: {:.0} prepared with work in flight, {:.0} with the pipe empty; \
         copied back by the completion thread {:.1} ms, in settles {:.1}, before preparing {:.1}, \
         elsewhere {:.1}; pipeline (VICTORY_PIPELINE): {:.1} completions with the copy-back off \
         the lock ({:.1} whole under it), {:.1} ms copied back off the lock, {:.1} of it beside a \
         preparing, {:.1} waits for one in progress ({:.1} ms), {:.1} settles of one",
        blocks[0],
        blocks[1],
        blocks[2],
        blocks[3],
        blocks[4],
        blocks[5],
        blocks[6],
        blocks[7],
        ms(c[C::BackVramUs as usize]),
        ms(c[C::VramWaitedUs as usize]),
        ms(c[C::BackMarksUs as usize]),
        ms(c[C::MarkPreUs as usize]),
        ms(c[C::MarkWaitUs as usize]),
        ms(c[C::MarkProtectUs as usize]),
        ms(c[C::MarkWriteUs as usize]),
        ms(c[C::MarkVerifyUs as usize]),
        ms(c[C::BackRangesUs as usize]),
        ms(c[C::CompareUs as usize]),
        ms(c[C::BackReadUs as usize]),
        ms(c[C::BackAfterUs as usize]),
        ms(c[C::BackVerifyUs as usize]),
        c[C::Submissions as usize],
        c[C::PreparedOverlapping as usize],
        c[C::PreparedEmpty as usize],
        ms(c[C::BackThreadUs as usize]),
        ms(c[C::BackSettleUs as usize]),
        ms(c[C::BackWaitUs as usize]),
        ms(c[C::BackOtherUs as usize]),
        c[C::PipeSplit as usize],
        c[C::PipeWhole as usize],
        ms(c[C::PipeMiddleUs as usize]),
        ms(c[C::PipeOverlapUs as usize]),
        c[C::PipeWaits as usize],
        ms(c[C::PipeWaitUs as usize]),
        c[C::SettleBusy as usize],
    );
    tracing::info!(
        "copies per frame: {:.1} submissions; {:.0} title copies {:.1} MB; {:.0} refreshes {:.1} MB, \
         {:.0} skipped {:.1} MB; {:.0} runs {:.1} ms; {:.0} queries {:.2} ms, moves {:.2} ms, \
         {:.2} MB unmapped; {:.0} page faults; {:.0} watches {:.2} ms, {:.1} MB checked, {:.1} MB \
         unchanged; {:.1} fallbacks ({:.1} ms); {:.0} guard faults; textures {:.1} MB watched and \
         changed, {:.1} MB hashed, {:.1} MB of them changed; watch time: kernel {:.1} ms, view queries {:.1} ms, protecting {:.1} ms; copied whole (not tracked) {:.1} MB in {:.1} ms; snapshots {:.1} MB in {:.1} ms; page faults {:.0} preparing, {:.0} copying back; {:.0} tracked pages marked through another mapping; \
         guards: {:.2} ms in the handler, {:.0} protect calls for {:.0} pages, {:.0} pages opened; \
         {:.0} tracked pages written ({:.0} of them the frame before too); change checks: {:.1} ms \
         asking for other mappings, {:.1} ms waiting for the guard lock, {:.0} guard-lock rounds; preparing: {:.1} ms in \
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
        c[C::TakeRounds as usize],
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
    tracing::info!(
        "  clamps per frame by reason (VICTORY_GPU_GAPS): in windows {:.0} indirect dispatches and {:.0} indirect draws that may store, {:.0} left a window for a barrier or pipeline sync after one that may store; alone {:.0} indirect dispatches, {:.0} with spread groups, {:.0} a window or prelude refused; {:.0} barrier calls by clamps, {:.0} title barrier calls, {:.0} pipeline syncs (no barrier); {:.0} host lists in {:.0} ExecuteCommandLists calls",
        c[C::ClampGapsDispatches as usize],
        c[C::ClampGapsStoring as usize],
        c[C::ClampGapsStale as usize],
        c[C::ClampAloneDispatch as usize],
        c[C::ClampAloneSpread as usize],
        c[C::ClampAloneOther as usize],
        c[C::ClampBarrierCalls as usize],
        c[C::TitleBarrierCalls as usize],
        c[C::PipelineSyncs as usize],
        c[C::ListsRun as usize],
        c[C::EclCalls as usize],
    );
    let snapshots = crate::shadow::idle_snapshot_bytes();
    let textures = crate::texture::cached_staging_bytes();
    if let (Some(idle_snapshot_bytes), Some((texture_upload_bytes, texture_read_capacity_bytes))) =
        (snapshots, textures)
    {
        tracing::info!(
            idle_snapshot_bytes,
            texture_upload_bytes,
            texture_read_capacity_bytes,
            "retained copy memory"
        );
    }

    let held: Vec<String> = crate::watch::held_times()
        .iter()
        .map(|(line, total, longest)| {
            format!(
                "watch.rs:{line} {:.1} ms (longest {longest:.1})",
                total / frames
            )
        })
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
         barred for churning {:.1}; armed copies kept their bytes of lazy pages {:.0} at arming, {:.1} when marked; \
         serving {:.1} ms, keeping at arming {:.1} ms",
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
        c[C::OwnerKeptArming as usize],
        c[C::OwnerKeptMarking as usize],
        ms(c[C::OwnerServeUs as usize]),
        ms(c[C::OwnerKeepUs as usize]),
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
    tracing::info!(
        "  protection changes per frame by caller: {}",
        sites.join(", ")
    );
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
    tracing::info!(
        "  page tables per frame (VICTORY_PT_BARRIER): {:.1} barriers with the switch off; with it {:.1} \
         left in flight, {:.1} kept for a program storing through a table, {:.1} kept for another \
         reason; {:.1} ranges a miss may withhold in flight; those left in flight completed: {:.1} \
         settled by a later preparing ({:.1} before a table's rewrite), {:.1} waited for before one, \
         {:.1} by the completion thread or a wait for all ({:.1} ms on their fences)",
        c[C::TablesOff as usize],
        c[C::TablesLight as usize],
        c[C::TablesKeptStores as usize],
        c[C::TablesKeptOther as usize],
        c[C::TablesKeptRanges as usize],
        c[C::TablesBySettle as usize],
        c[C::SettleTable as usize],
        c[C::TablesByWait as usize],
        c[C::TablesByOther as usize],
        ms(c[C::TablesWaitedUs as usize]),
    );
    for line in crate::census::lines(frames) {
        tracing::info!("{line}");
    }
    let mut snapshots: Vec<((u64, u64), u64)> = SNAPSHOTS
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .take()
        .into_iter()
        .flatten()
        .collect();
    snapshots.sort_by_key(|&(_, bytes)| std::cmp::Reverse(bytes));
    let snapshots: Vec<String> = snapshots
        .iter()
        .take(6)
        .map(|((start, len), bytes)| {
            format!("{start:#x}+{len:#x} {:.1} MB", *bytes as f64 / 1e6 / frames)
        })
        .collect();
    tracing::info!("  snapshots per frame by copy: {}", snapshots.join(", "));
    tracing::info!(
        "  BVH builder per frame: {:.1} dispatches run, {:.1} held back (VICTORY_BVH_HALVES)",
        c[C::BvhRun as usize],
        c[C::BvhHeld as usize],
    );
    for ((src, len), (n, bytes, us, _)) in RANGES.take().into_iter().take(10) {
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
        let frames: Vec<&str> = trace
            .lines()
            .filter(|l| l.contains("victory"))
            .take(24)
            .collect();
        tracing::info!(
            mb = size >> 20,
            "a large heap allocation (first of its size class): {}",
            frames.join(" | ")
        );
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

#[cfg(test)]
mod tests {
    use super::*;

    // Each test has its own counters and ranges: the GPU tests count into
    // the global ones while these run.

    #[test]
    fn range_accumulates_same_key_and_separates_distinct_ranges() {
        let ranges = Ranges::new();
        ranges.add(0x1000, 64, 10, 0x20000);
        ranges.add(0x1000, 64, 25, 0x20000);
        ranges.add(0x2000, 256, 7, 0x40000);

        assert_eq!(
            ranges.take(),
            [
                ((0x2000, 256), (1, 256, 7, 0x40000)),
                ((0x1000, 64), (2, 128, 35, 0x20000))
            ]
        );
        assert!(ranges.take().is_empty());
    }

    #[test]
    fn take_drains_counters_per_frame() {
        let counters = Counters::new();
        counters.add(C::Submissions, 4);
        counters.add(C::Copies, 12);
        counters.add(C::Copies, 8);
        counters.add(C::SavedPages, 6);

        let c = counters.take(4.0);
        assert_eq!(c.len(), COUNTERS);
        assert_eq!(c[C::Submissions as usize], 1.0);
        assert_eq!(c[C::Copies as usize], 5.0);
        assert_eq!(c[C::SavedPages as usize], 1.5);
        assert_eq!(c.iter().sum::<f64>(), 7.5);

        // The next line counts from zero.
        counters.add(C::Copies, 2);
        let c = counters.take(1.0);
        assert_eq!(c[C::Copies as usize], 2.0);
        assert_eq!(c.iter().sum::<f64>(), 2.0);
    }
}
