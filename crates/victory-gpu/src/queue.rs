//! Command queues. Submitted lists run on the host GPU and complete later
//! (`crate::host::Host::submit`): a queue's fence Signal behind them moves
//! the fence once they have.
#![allow(clippy::not_unsafe_ptr_arg_deref)]

use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};

use crate::object::{self, E_INVALIDARG, HRESULT, Iid, Object, QUEUE, S_OK, Table, This, guid};
use crate::resource::PAGEABLE_IIDS;

const QUEUE_IIDS: [Iid; 4] = [
    guid(
        0x0ec870a6,
        0x5d7e,
        0x4c22,
        [0x8c, 0xfc, 0x5b, 0xaa, 0xe0, 0x76, 0x16, 0xed],
    ),
    PAGEABLE_IIDS[0],
    PAGEABLE_IIDS[1],
    PAGEABLE_IIDS[2],
];

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub(crate) struct QueueDesc {
    kind: u32,
    priority: i32,
    flags: u32,
    nodes: u32,
}

/// The title reads the list type right after the header.
#[repr(C)]
struct Queue {
    object: Object,
    kind: u32,
    desc: QueueDesc,
}
const _: () = assert!(std::mem::offset_of!(Queue, kind) == 0x10);

fn queue<'a>(this: This) -> &'a Queue {
    // SAFETY: called from queue methods only.
    unsafe { &*this.cast::<Queue>() }
}

pub(crate) extern "system" fn create(
    _: This,
    desc: *const QueueDesc,
    iid: *const Iid,
    out: *mut This,
) -> HRESULT {
    // SAFETY: the title passes a description (null is tolerated).
    let Some(&desc) = (unsafe { desc.as_ref() }) else {
        return E_INVALIDARG;
    };
    tracing::debug!(?desc, "command queue");
    let queue = Box::new(Queue {
        object: if crate::device::newer() {
            Object::new(&NEWER_FUNCTIONS)
        } else {
            Object::new(&FUNCTIONS)
        },
        kind: desc.kind,
        desc,
    });
    object::hand_out(QUEUE, queue, iid, &QUEUE_IIDS, out)
}

/// The console's queue description: an engine or pipe and a queue on it
/// in place of PC's priority and node mask.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub(crate) struct QueueDescX {
    kind: u32,
    flags: u32,
    engine_or_pipe: u32,
    queue: u32,
}

/// Device method `CreateCommandQueueX`. The host picks engines itself, so
/// the queue is created as on PC.
pub(crate) extern "system" fn create_x(
    this: This,
    desc: *const QueueDescX,
    iid: *const Iid,
    out: *mut This,
) -> HRESULT {
    // SAFETY: the title passes a description (null is tolerated).
    let Some(&x) = (unsafe { desc.as_ref() }) else {
        return E_INVALIDARG;
    };
    tracing::debug!(desc = ?x, "console command queue");
    let desc = QueueDesc {
        kind: x.kind,
        priority: 0,
        flags: x.flags,
        nodes: 1,
    };
    create(this, &desc, iid, out)
}

extern "system" fn query(this: This, iid: *const Iid, out: *mut This) -> HRESULT {
    object::answer(
        QUEUE,
        iid,
        object::is(iid, &QUEUE_IIDS).then_some(this),
        out,
    )
}
extern "system" fn release(this: This) -> u32 {
    object::release_boxed::<Queue>(this)
}

/// Queues the lists' host lists behind what the queue holds, and runs what
/// may run (`crate::hold`; decision 0007).
extern "system" fn execute(this: This, count: u32, lists: *const This) {
    if lists.is_null() {
        return;
    }
    #[cfg(windows)]
    if crate::host::host().is_some() {
        // SAFETY: the title passes `count` lists.
        let lists = unsafe { std::slice::from_raw_parts(lists, count as usize) };
        let mut held = held();
        for &l in lists {
            for segment in crate::list::segments(l) {
                let waits = segment.waits();
                held.push(this as usize, Item::Segment(segment, waits));
            }
        }
        account().submissions += 1;
        pump(&mut held, Why::Submission);
        return;
    }
    static ONCE: AtomicBool = AtomicBool::new(false);
    if !ONCE.swap(true, Ordering::Relaxed) {
        tracing::warn!(count, "command lists submitted; no host GPU runs them");
    }
}

/// What ran held work: the title's submission, a queue's Signal or Wait, a
/// CPU fence call, PresentX, our timer, or a submission completing.
#[derive(Clone, Copy, Debug)]
pub(crate) enum Why {
    Submission,
    Signal,
    Wait,
    Fence,
    Present,
    Timer,
    Completed,
}

impl Why {
    /// Every reason, by its index: the frame line counts releases by them.
    const ALL: [Why; 7] = [
        Why::Submission,
        Why::Signal,
        Why::Wait,
        Why::Fence,
        Why::Present,
        Why::Timer,
        Why::Completed,
    ];

    /// The reason's name in the frame line.
    fn name(self) -> &'static str {
        match self {
            Why::Submission => "submission",
            Why::Signal => "signal",
            Why::Wait => "wait",
            Why::Fence => "fence",
            Why::Present => "present",
            Why::Timer => "timer",
            Why::Completed => "completed",
        }
    }
}

// Every reason is in `Why::ALL`, at its own index: one added later and left
// out would count past the frame line's counts.
const _: () = {
    let mut i = 0;
    while i < Why::ALL.len() {
        assert!(Why::ALL[i] as usize == i);
        i += 1;
    }
    assert!(Why::Completed as usize + 1 == Why::ALL.len());
};

/// How many released segments each reason released, by name, in
/// [`Why::ALL`]'s order (those that released none left out).
fn released_by(released: &[(u32, std::time::Duration, Why)]) -> Vec<(&'static str, u32)> {
    let mut by = [0u32; Why::ALL.len()];
    for &(_, _, why) in released {
        by[why as usize] += 1;
    }
    Why::ALL
        .iter()
        .zip(by)
        .filter(|&(_, n)| n > 0)
        .map(|(why, n)| (why.name(), n))
        .collect()
}

#[cfg(windows)]
use crate::hold::Item;

/// Every queue's held work.
#[cfg(windows)]
type Holding = crate::hold::Held<std::sync::Arc<crate::host::HostList>>;

#[cfg(windows)]
static HELD: std::sync::Mutex<Holding> = std::sync::Mutex::new(crate::hold::Held::new());

#[cfg(windows)]
fn held() -> std::sync::MutexGuard<'static, Holding> {
    HELD.lock().unwrap_or_else(|e| e.into_inner())
}

/// Runs held work that may run now (`why` for the frame line).
pub(crate) fn pump_held(why: Why) {
    #[cfg(windows)]
    pump(&mut held(), why);
    #[cfg(not(windows))]
    let _ = why;
}

/// The same unless another thread is at it (calls that poll: a fence's
/// value, our timer).
pub(crate) fn pump_held_if_free(why: Why) {
    #[cfg(windows)]
    if let Ok(mut held) = HELD.try_lock() {
        pump(&mut held, why);
    }
    #[cfg(not(windows))]
    let _ = why;
}

/// How long a queue's head may wait before it runs anyway.
// shortcut: a safety valve against a wait nothing satisfies (a write we do
// not make), which would hold the title's fences until its 30 s watchdog
// ends the run; counted and logged, remove once runs show it never fires.
#[cfg(windows)]
const VALVE: std::time::Duration = std::time::Duration::from_secs(10);

#[cfg(windows)]
fn pump(held: &mut Holding, why: Why) {
    if held.is_empty() {
        return;
    }
    let Some(host) = crate::host::host() else { return };
    // With early release, waits read the values of bottom-of-pipe writes in
    // flight over title memory (`crate::early`).
    let early = crate::early::on();
    let overlaid = |address: u64, bytes: u8| crate::early::read(address, bytes, &read_value);
    let read: &dyn Fn(u64, u8) -> Option<u64> = if early { &overlaid } else { &read_value };
    for (queue, waits) in held.force_older_than(VALVE) {
        account().forced += 1;
        tracing::warn!(queue = kind(queue), ?waits, "held work runs without its waits after {VALVE:?} (safety valve)");
    }
    let fence = |f: usize| crate::fence::value(f as This);
    let started = std::time::Instant::now();
    let mut ran = std::time::Duration::ZERO;
    let mut run = |queue: usize, segments: Vec<std::sync::Arc<crate::host::HostList>>| {
        let at = std::time::Instant::now();
        host.submit(queue, segments);
        ran += at.elapsed();
    };
    // Behind the queue's work submitted before it, which may still run.
    let mut signal = |queue: usize, f: usize, value: u64| host.signal_after(queue, f, value);
    let pumped = held.pump(&mut crate::hold::World {
        read,
        memory: if early { Some(&read_value) } else { None },
        fence: &fence,
        run: &mut run,
        signal: &mut signal,
    });
    #[cfg(feature = "diagnostics")]
    crate::copystat::add(crate::copystat::C::EarlyReleased, u64::from(pumped.early));
    if pumped.segments > 0 || pumped.unknown > 0 {
        let mut a = account();
        a.pump += started.elapsed().saturating_sub(ran);
        a.pieces += pumped.pieces;
        a.segments += pumped.segments;
        a.unknown += pumped.unknown;
        a.early += pumped.early;
        for (queue, waited) in pumped.waited {
            a.released.push((kind(queue), waited, why));
        }
    }
    // Once work stays held, a timer tests it every 2 ms: memory the title
    // writes from the CPU, or the valve, may let it go with no call of ours.
    if held.blocked(&read_value).is_empty() {
        return;
    }
    static TIMER: std::sync::Once = std::sync::Once::new();
    TIMER.call_once(|| {
        let _ = std::thread::Builder::new().name("victory held work".into()).spawn(|| loop {
            std::thread::sleep(std::time::Duration::from_millis(2));
            pump_held_if_free(Why::Timer);
        });
    });
}

/// The value a wait tests (4 or 8 bytes), if the memory can be read.
#[cfg(windows)]
fn read_value(address: u64, bytes: u8) -> Option<u64> {
    let mut value = [0u8; 8];
    crate::host::read_into(address as *const u8, &mut value[..usize::from(bytes.min(8))]).then(|| u64::from_le_bytes(value))
}

/// A queue's type, for logs (0 graphics, 2 compute, 3 copy). Titles do not
/// release queues that hold work.
#[cfg(windows)]
fn kind(queue: usize) -> u32 {
    self::queue(queue as This).kind
}

/// What queues did with held work since the last presented frame.
#[cfg(windows)]
#[derive(Default)]
struct Account {
    submissions: u32,
    pieces: u32,
    segments: u32,
    /// Segments that waited: queue type, how long, what ran them.
    released: Vec<(u32, std::time::Duration, Why)>,
    unknown: u32,
    forced: u32,
    mappings: u32,
    /// PresentX calls queued behind work their queue held.
    presents_held: u32,
    /// Time in the pump outside host submissions.
    pump: std::time::Duration,
    /// Segments released on values of writes in flight (`crate::early`)
    /// that title memory alone held.
    early: u32,
}

#[cfg(windows)]
fn account() -> std::sync::MutexGuard<'static, Account> {
    static ACCOUNT: std::sync::OnceLock<std::sync::Mutex<Account>> = std::sync::OnceLock::new();
    ACCOUNT.get_or_init(Default::default).lock().unwrap_or_else(|e| e.into_inner())
}

/// The frame line of held work: what waited, how long and what released it,
/// and what is still held at the present, with the waits that do not hold.
#[cfg(windows)]
fn account_frame(frame: u64) {
    let (cuts, early) = crate::list::take_cuts();
    let (fence_waits, fence_waited) = crate::fence::take_waited();
    let a = std::mem::take(&mut *account());
    let blocked = held().blocked(&read_value);
    if a.released.is_empty()
        && blocked.is_empty()
        && a.forced == 0
        && a.unknown == 0
        && a.mappings == 0
        && a.presents_held == 0
    {
        return;
    }
    let longest = a.released.iter().map(|r| r.1).max().unwrap_or_default();
    let by = released_by(&a.released);
    tracing::info!(
        frame,
        submissions = a.submissions,
        pieces = a.pieces,
        segments = a.segments,
        cuts,
        early_resets = early,
        waited = a.released.len(),
        longest_ms = longest.as_secs_f64() * 1e3,
        pump_ms = a.pump.as_secs_f64() * 1e3,
        fence_waits,
        fence_wait_ms = fence_waited.as_secs_f64() * 1e3,
        presents_held = a.presents_held,
        released_early = a.early,
        "held work this frame (submissions; host submissions, segments; lists cut at waits; lists reset or released \
         while held, each keeping its own host list; segments that waited, and how many each reason released \
         {by:?}; time in the pump outside host submissions; events the          title waited on for fence values and their summed waits; \
         PresentX calls queued behind held work; segments released on values in flight, VICTORY_EARLY_RELEASE); \
         held at the present: {}",
        blocked.len()
    );
    tracing::debug!(
        frame,
        "held work released this frame (queue type, ms, by): {:?}",
        a.released.iter().map(|(k, d, w)| (k, d.as_secs_f64() * 1e3, w)).collect::<Vec<_>>()
    );
    for (queue, items, waited, unmet) in &blocked {
        let level_warn = blocked.len() > 1;
        let unmet: Vec<String> = unmet
            .iter()
            .map(|w| format!("{:#x} cmp {} {:#x} (is {:?})", w.address, w.comparison, w.reference, read_value(w.address, w.bytes)))
            .collect();
        if level_warn {
            tracing::warn!(frame, queue = kind(*queue), items, waited_ms = waited.as_millis(), ?unmet, "two or more queues hold work at the present");
        } else {
            tracing::info!(frame, queue = kind(*queue), items, waited_ms = waited.as_millis(), ?unmet, "a queue holds work at the present");
        }
    }
    if a.forced > 0 || a.unknown > 0 || a.mappings > 0 {
        tracing::warn!(
            frame,
            forced = a.forced,
            unknown = a.unknown,
            mappings = a.mappings,
            "held work: run without its waits (valve), waits with an unknown comparison (taken as holding), \
             page mappings done ahead of held work"
        );
    }
}

/// One plane of a `PresentX` call (0x48-byte stride): frame token,
/// resource count, and pointer to the plane's resources.
#[repr(C)]
pub(crate) struct Plane {
    token: u64,
    count: u32,
    resources: *const This,
    _rest: [u64; 6],
}
const _: () = assert!(std::mem::size_of::<Plane>() == 0x48);
/// Submission tagged with the frame it belongs to `(count, lists, frame
/// token, flags)`; frames are paced elsewhere, so it runs as a plain one.
extern "system" fn execute_x(this: This, count: u32, lists: *const This, _token: u64, _flags: u32) {
    execute(this, count, lists);
}
/// Signal with sync settings for the top and bottom of the pipe; the fence
/// moves once the work before it has completed, as with Signal.
extern "system" fn signal2_x(
    this: This,
    fence: This,
    value: u64,
    _bottom: *const u8,
    _top: *const u8,
) -> HRESULT {
    signal(this, fence, value)
}
/// Markers and events only matter to profilers.
extern "system" fn marker(_: This) {}
/// Behind the work the queue holds, or the work submitted so far (its
/// fence, and the memory the title reads it at, move only once that work
/// has completed).
extern "system" fn signal(this: This, fence: This, value: u64) -> HRESULT {
    if fence.is_null() {
        return E_INVALIDARG;
    }
    #[cfg(windows)]
    {
        let mut held = held();
        if held.holds_work(this as usize) {
            crate::fence::keep(fence);
            held.push(this as usize, Item::Signal(fence as usize, value));
        } else if let Some(host) = crate::host::host() {
            crate::fence::keep(fence);
            host.signal_after(this as usize, fence as usize, value);
        } else {
            crate::fence::set(fence, value);
        }
        // Another queue may wait for this fence.
        pump(&mut held, Why::Signal);
    }
    #[cfg(not(windows))]
    {
        let _ = this;
        crate::fence::set(fence, value);
    }
    S_OK
}
/// The queue's later work waits until the fence reaches `value`.
///
/// shortcut: the fence is not kept alive while the queue waits for it;
/// Grand Theft Auto VI makes no queue wait for a fence.
extern "system" fn wait(this: This, fence: This, value: u64) -> HRESULT {
    if fence.is_null() {
        return E_INVALIDARG;
    }
    #[cfg(windows)]
    if crate::host::host().is_some() && crate::fence::value(fence) < value {
        let mut held = held();
        held.push(this as usize, Item::FenceWait(fence as usize, value));
        pump(&mut held, Why::Wait);
    }
    #[cfg(not(windows))]
    let _ = (this, value);
    S_OK
}
/// Ticks per second of GPU timestamps (see [`Written::GpuTimestamp`]).
extern "system" fn timestamp_frequency(_: This, out: *mut u64) -> HRESULT {
    if out.is_null() {
        return E_INVALIDARG;
    }
    // SAFETY: checked non-null.
    unsafe { out.write(TIMESTAMP_FREQUENCY) };
    S_OK
}
const TIMESTAMP_FREQUENCY: u64 = 100_000_000;

/// What a list's bottom-of-pipe write stores when the list has run.
#[derive(Debug, Clone, Copy)]
pub(crate) enum Written {
    Value(u64),
    /// The GPU clock: [`TIMESTAMP_FREQUENCY`] ticks per second since the
    /// process started (our GPU work is done on the CPU's timeline).
    GpuTimestamp,
    /// The CPU's performance counter.
    SystemTimestamp,
}

impl Written {
    /// Stores the low `bytes` (4 or 8) of the value at `at`.
    pub(crate) fn write(self, at: *mut u8, bytes: usize) {
        let value = match self {
            Written::Value(v) => v,
            Written::GpuTimestamp => {
                static START: std::sync::OnceLock<std::time::Instant> = std::sync::OnceLock::new();
                let elapsed = START.get_or_init(std::time::Instant::now).elapsed();
                (elapsed.as_nanos() / (1_000_000_000 / u128::from(TIMESTAMP_FREQUENCY))) as u64
            }
            Written::SystemTimestamp => system_timestamp(),
        };
        // SAFETY: the title names writable GPU memory (unified with ours).
        unsafe {
            if bytes == 4 {
                at.cast::<u32>().write_unaligned(value as u32);
            } else {
                at.cast::<u64>().write_unaligned(value);
            }
        }
    }
}

fn system_timestamp() -> u64 {
    #[cfg(windows)]
    {
        let mut count = 0;
        // SAFETY: a valid out pointer.
        unsafe { windows_sys::Win32::System::Performance::QueryPerformanceCounter(&mut count) };
        count as u64
    }
    #[cfg(not(windows))]
    0
}

/// Frames presented so far.
static FRAMES: AtomicU64 = AtomicU64::new(0);

pub(crate) fn presented() -> u64 {
    FRAMES.load(Ordering::Relaxed)
}

/// Shows the first resource of each presented plane in our window. With
/// overlap ([`crate::host::overlap`]), behind the work the queue holds: the
/// frame is shown once that has been given to the host GPU.
extern "system" fn present(
    this: This,
    planes: u32,
    plane_parameters: *const Plane,
    _parameters: *const u8,
) -> HRESULT {
    // What may run before the frame is shown does.
    // shortcut: without overlap, PresentX does not wait behind work the
    // queue still holds (it shows the frame as far as it ran), as before;
    // the frame line names such work.
    pump_held(Why::Present);
    let frame = FRAMES.fetch_add(1, Ordering::Relaxed) + 1;
    #[cfg(windows)]
    account_frame(frame);
    #[cfg(feature = "diagnostics")]
    crate::frame::presented();
    if frame.is_power_of_two() {
        let (draws, dispatches) = crate::list::take_counts();
        #[cfg(windows)]
        let (drawn, dispatched) = {
            static DRAWN: AtomicU64 = AtomicU64::new(0);
            static DISPATCHED: AtomicU64 = AtomicU64::new(0);
            let since = |last: &AtomicU64, now: u64| now - last.swap(now, Ordering::Relaxed);
            (
                since(&DRAWN, crate::draw::drawn()),
                since(&DISPATCHED, crate::draw::dispatched()),
            )
        };
        #[cfg(not(windows))]
        let (drawn, dispatched) = (0, 0);
        tracing::info!(
            frame,
            draws,
            drawn,
            dispatches,
            dispatched,
            "frame presented (since the last line: draws and dispatches, and those run on the host)"
        );
        #[cfg(windows)]
        if frame >= 256 {
            let (hits, misses) = crate::shader_cache::counts();
            tracing::info!(hits, misses, "shaders from the cache and translated so far");
        }
    }
    #[cfg(all(windows, feature = "diagnostics"))]
    {
        crate::capture::presented(frame);
        if crate::capture::listed(frame) || crate::capture::has_targets() {
            crate::capture::dump_targets(frame);
        }
    }
    #[cfg(windows)]
    if planes > 0
        && !plane_parameters.is_null()
        && let Some(host) = crate::host::host()
        // SAFETY: the title passes the plane descriptions.
        && !holds(frame, unsafe { std::slice::from_raw_parts(plane_parameters, planes.min(8) as usize) })
    {
        use windows::Win32::Graphics::Direct3D12::ID3D12Resource;
        use windows::core::Interface;

        const COPY_SOURCE: u32 = 0x800;
        // SAFETY: the title passes the plane descriptions.
        let slice = unsafe { std::slice::from_raw_parts(plane_parameters, planes.min(8) as usize) };
        let mut resources: Vec<(This, ID3D12Resource, u32, bool)> =
            Vec::with_capacity(slice.len());
        for plane in slice {
            if plane.count == 0 || plane.resources.is_null() {
                continue;
            }
            // SAFETY: the plane lists `count` resources.
            let resource = unsafe { *plane.resources };
            if let Some(texture) = crate::resource::host_resource(resource) {
                // Distinct title resources may refer to one host texture.
                // Keep its original state for every plane and restore it once.
                let prior = resources.iter().find_map(|(_, earlier, state, _)| {
                    (earlier.as_raw() == texture.as_raw()).then_some(*state)
                });
                let state = prior.unwrap_or_else(|| {
                    crate::resource::host_transition(resource, COPY_SOURCE)
                        .map_or(COPY_SOURCE, |(_, before)| before)
                });
                resources.push((resource, texture, state, prior.is_none()));
            }
        }
        if !resources.is_empty() {
            // The states the frame leaves its planes in: the copy list moves
            // them back to these (later work the queue holds runs after it).
            for &(resource, _, state, first) in &resources {
                if first {
                    crate::resource::host_transition(resource, state);
                }
            }
            #[cfg(feature = "diagnostics")]
            let (first, first_state) = (resources[0].0 as usize, resources[0].2);
            let planes: Vec<(ID3D12Resource, u32)> = resources
                .into_iter()
                .map(|(_, tex, state, _)| (tex, state))
                .collect();
            let show = move || {
                let sources: Vec<(&_, u32)> =
                    planes.iter().map(|(tex, state)| (tex, *state)).collect();
                let shown = host.present(&sources, frame);
                #[cfg(feature = "diagnostics")]
                blinks(frame, first as This, first_state, shown);
                #[cfg(not(feature = "diagnostics"))]
                let _ = shown;
            };
            let mut held = held();
            if crate::host::overlap() && held.holds_work(this as usize) {
                account().presents_held += 1;
                held.push(this as usize, Item::Then(Box::new(show)));
            } else {
                drop(held);
                show();
            }
        }
    }
    #[cfg(not(windows))]
    let _ = (this, planes, plane_parameters);
    S_OK
}

/// Whether frame `frame` is not shown: a draw dropped for a pipeline being
/// made left one of its `planes` unwritten, so it holds an older frame and
/// the window keeps the last one shown, for a while ([`crate::shown`]).
/// Logs where a run of held frames starts and ends, and a hold given up.
/// The pipelines of the frame's dropped passes the planes read go first
/// ([`crate::feeds`]).
#[cfg(windows)]
fn holds(frame: u64, planes: &[Plane]) -> bool {
    use crate::shown::Frame;
    let images: Vec<usize> = planes
        .iter()
        .filter(|plane| plane.count != 0 && !plane.resources.is_null())
        // SAFETY: the plane lists `count` resources.
        .map(|plane| unsafe { *plane.resources } as usize)
        .collect();
    crate::pipeline::hurry_feeders(&crate::feeds::presented(&images));
    match crate::shown::frame(images) {
        Frame::Show => false,
        Frame::ShowAgain { held } => {
            tracing::info!(frame, held, "frames shown again");
            false
        }
        Frame::Hold { first } => {
            if first {
                tracing::info!(
                    frame,
                    "frames held: a draw into the presented image waits for its pipeline"
                );
            }
            true
        }
        Frame::GiveUp { held, waited_for } => {
            tracing::warn!(
                frame,
                held,
                waited_for,
                "frames shown again unwritten: their pipeline was not made in time to hold for it"
            );
            false
        }
    }
}

/// Diagnostics: logs a presented frame that is black after one that was
/// not, with what was presented and how many draws ran since.
#[cfg(all(windows, feature = "diagnostics"))]
fn blinks(frame: u64, resource: This, state: u32, shown: Option<bool>) {
    static LAST: std::sync::Mutex<(bool, u64, u64)> = std::sync::Mutex::new((false, 0, 0));
    let mut last = LAST.lock().unwrap_or_else(|e| e.into_inner());
    let drawn = crate::draw::drawn();
    let black = shown.unwrap_or(false);
    let draws = crate::draw::take_frame();
    if black && !last.0 {
        last.2 += 1;
        tracing::warn!(
            frame,
            resource = ?resource,
            state = format_args!("{state:#x}"),
            draws = drawn - last.1,
            copied = shown.is_some(),
            black_frames = last.2,
            levels = ?crate::capture::recent_levels(),
            "a black frame after a frame that was not"
        );
        for d in draws {
            tracing::warn!("black frame {frame}: {d}");
        }
        AFTER.store(frame + 6, Ordering::Relaxed);
    }
    // Six frames on: the levels around it (a fade or a single drop).
    static AFTER: AtomicU64 = AtomicU64::new(0);
    if AFTER.load(Ordering::Relaxed) == frame {
        tracing::warn!(frame, levels = ?crate::capture::recent_levels(), "levels after the black frame");
    }
    *last = (black, drawn, last.2);
}

/// A run of destination pages: how they are mapped (0 pool pages from
/// `start` on, 1 pool page `start` each, 2 unmapped, 3 left as they are)
/// and how many.
#[repr(C)]
pub(crate) struct PageRange {
    kind: u32,
    start: u32,
    count: u32,
}

/// The ranges mapped from a destination address on.
#[repr(C)]
pub(crate) struct PageBatch {
    destination: u64,
    count: u32,
    ranges: *const PageRange,
}

/// Bytes per mapped page.
const PAGE: u64 = 64 << 10;

/// `CopyPageMappingsBatchX(count, batches, pool, pool pages, flags)`: maps
/// pages of a pool (memory the title mapped physical pages into) at the
/// batches' addresses. Grand Theft Auto VI streams textures into pool pages
/// and maps them into its tiled heap this way (title+0xf2c470, ranges of
/// one constant page or of unmapped pages). Done at once.
///
/// shortcut: done ahead of work the queue still holds (counted in the frame
/// line); queue it as an item like Signal if that count is not 0.
extern "system" fn copy_page_mappings(
    this: This,
    count: u32,
    batches: *const PageBatch,
    pool: u64,
    pool_pages: u32,
    _flags: u32,
) -> HRESULT {
    let Some(map) = crate::resource::page_mapper() else {
        return E_INVALIDARG;
    };
    if batches.is_null() && count > 0 {
        return E_INVALIDARG;
    }
    #[cfg(windows)]
    if held().holds_work(this as usize) {
        account().mappings += 1;
    }
    #[cfg(not(windows))]
    let _ = this;
    let batches = (count > 0).then(|| {
        // SAFETY: the title passes `count` batches.
        unsafe { std::slice::from_raw_parts(batches, count as usize) }
    });
    let mut ok = true;
    for b in batches.into_iter().flatten() {
        if b.ranges.is_null() {
            continue;
        }
        // SAFETY: each batch holds `count` ranges.
        let ranges = unsafe { std::slice::from_raw_parts(b.ranges, b.count as usize) };
        let mut at = b.destination;
        for r in ranges {
            let len = u64::from(r.count) * PAGE;
            let inside = |pages: u32| r.start.checked_add(pages).is_some_and(|e| e <= pool_pages);
            let source = pool + u64::from(r.start) * PAGE;
            ok &= match r.kind {
                0 if inside(r.count) => map(at, source, len),
                1 if inside(1) => (0..u64::from(r.count)).all(|k| map(at + k * PAGE, source, PAGE)),
                2 => map(at, 0, len),
                3 => true,
                _ => false,
            };
            at += len;
        }
    }
    if !ok {
        static ONCE: std::sync::Once = std::sync::Once::new();
        ONCE.call_once(|| tracing::warn!(pool = format_args!("{pool:#x}"), "a page mapping failed"));
    }
    S_OK
}

extern "system" fn desc(this: This, out: *mut QueueDesc) {
    if !out.is_null() {
        // SAFETY: checked non-null; the title passes room for a description.
        unsafe { out.write(queue(this).desc) };
    }
}

/// The queue methods by their slot in older titles' queues (the 2020-era
/// layout): IUnknown 3, object 4, device child 1, queue 26 (12 as on PC,
/// then console ones).
const METHODS: [(usize, *const ()); 15] = [
    (0, query as *const ()),
    (2, release as *const ()),
    (7, object::parent_device as *const ()), // 0x38
    (10, execute as *const ()),              // 0x50
    (11, marker as *const ()),               // 0x58 set marker
    (12, marker as *const ()),               // 0x60 begin event
    (13, marker as *const ()),               // 0x68 end event
    (14, signal as *const ()),               // 0x70
    (15, wait as *const ()),                 // 0x78
    (16, timestamp_frequency as *const ()),  // 0x80
    (26, desc as *const ()),                 // 0xd0
    (29, copy_page_mappings as *const ()),   // 0xe8 page mapping batch
    (30, present as *const ()),              // 0xf0
    (33, execute_x as *const ()),            // 0x108
    (32, signal2_x as *const ()),            // 0x100
];

/// Newer titles' queues have two more methods before Signal, which moved
/// from 0x70 to 0x80: every slot from 0x70 on is two later (taken here as
/// starting exactly at 0x70, the methods before it kept in place).
/// Inferred from Grand Theft Auto VI's calls: Signal 0x80 `(fence,
/// value)`, timestamp frequency 0x90 (title+0x24841a6), suspend and resume
/// 0xb8/0xc0, PresentX 0x100, the frame-token submission 0x118 `(count,
/// lists, token, flags)` (title+0x24821dd).
const fn newer_slot(slot: usize) -> usize {
    if slot >= 14 { slot + 2 } else { slot }
}

const fn queue_table(newer: bool) -> [*const (); 36] {
    let mut f = object::d3d12_object(object::missing_table::<QUEUE, 36>());
    let mut i = 0;
    while i < METHODS.len() {
        let (slot, method) = METHODS[i];
        f[if newer { newer_slot(slot) } else { slot }] = method;
        i += 1;
    }
    f
}

static FUNCTIONS: Table<36> = Table(queue_table(false));
static NEWER_FUNCTIONS: Table<36> = Table(queue_table(true));

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::object::slot;

    /// Held by tests that submit through title queues: their work stays in
    /// flight after they submit, and a test holding the host GPU
    /// ([`crate::host::Host::hold_gpu`]) must find none of another's there.
    pub(crate) static TITLE_QUEUES: std::sync::Mutex<()> = std::sync::Mutex::new(());

    /// Waits as a title does until the work given to `queue` so far has
    /// completed: a fence the queue signals behind it, waited for (a null
    /// event blocks until the value is reached).
    pub(crate) fn finish(queue: This) {
        let null = std::ptr::null();
        let mut fence = null;
        assert_eq!(
            crate::fence::create(null, 0, 0, &crate::fence::FENCE_IIDS[0], &mut fence),
            S_OK
        );
        let signal =
            slot::<extern "system" fn(This, This, u64) -> HRESULT>(queue, signal_slot(queue));
        assert_eq!(signal(queue, fence, 1), S_OK);
        let wait =
            slot::<extern "system" fn(This, u64, *mut std::ffi::c_void) -> HRESULT>(fence, 0x48);
        assert_eq!(wait(fence, 1, std::ptr::null_mut()), S_OK);
        slot::<extern "system" fn(This) -> u32>(fence, 0x10)(fence);
    }

    /// Distinct title resources can name the same host texture. Presenting
    /// both planes must preserve that texture's original rendering state.
    #[cfg(windows)]
    #[test]
    fn aliased_present_planes_preserve_the_host_texture_state() {
        use windows::Win32::Graphics::Direct3D12::D3D12_RESOURCE_STATE_RENDER_TARGET;

        let Some(host) = crate::host::host() else {
            return;
        };
        const TARGET: u32 = D3D12_RESOURCE_STATE_RENDER_TARGET.0 as u32;
        let address = crate::watch::tests::pool(1);
        let null = std::ptr::null();
        let iid = crate::object::guid(
            0x696442be,
            0xa72e,
            0x4059,
            [0xbc, 0x79, 0x5b, 0x5c, 0x98, 0x04, 0x0f, 0xad],
        );
        #[repr(align(8))]
        struct TextureWords([u32; 14]);
        let desc = TextureWords([3, 0, 0, 0, 32, 0, 32, 1 | 1 << 16, 87, 1, 0, 1, 1, 0]);
        let resources = std::array::from_fn::<_, 2, _>(|_| {
            let mut resource = null;
            assert_eq!(
                crate::resource::create_at_address(
                    null,
                    address as u64,
                    desc.0.as_ptr().cast(),
                    0,
                    null.cast(),
                    &iid,
                    &mut resource,
                ),
                S_OK
            );
            resource
        });
        assert_ne!(resources[0], resources[1]);
        let primary = crate::resource::host_copy(resources[0]).unwrap();
        let alias = crate::resource::host_copy(resources[1]).unwrap();
        assert!(std::sync::Arc::ptr_eq(&primary, &alias));
        let list = host.new_list().unwrap();
        crate::list::ensure_state(&list, resources[0], TARGET);
        // SAFETY: the texture is a live render target, on an open list.
        unsafe {
            list.list.ClearRenderTargetView(
                host.rtv(&primary.texture, None),
                &[0.25, 0.5, 0.75, 1.0],
                None,
            );
        }
        list.close();
        host.execute(&[&list]);
        assert_eq!(primary.state.load(Ordering::Relaxed), TARGET);

        let planes = resources.each_ref().map(|resource| Plane {
            token: 0,
            count: 1,
            resources: std::ptr::from_ref(resource),
            _rest: [0; 6],
        });
        assert_eq!(present(null, 2, planes.as_ptr(), null.cast()), S_OK);
        let restored = primary.state.load(Ordering::Relaxed);
        let release = |resource| slot::<extern "system" fn(This) -> u32>(resource, 0x10)(resource);
        for resource in resources {
            assert_eq!(release(resource), 0);
        }
        drop((list, primary, alias));
        crate::watch::tests::free(address);
        // SAFETY: a plain device query.
        assert!(unsafe { host.device.GetDeviceRemovedReason() }.is_ok());
        assert_eq!(restored, TARGET, "the two planes share one rendering state");
    }

    /// With overlap, a PresentX on a queue that holds work is queued behind
    /// it, and shown once that work has been released.
    #[cfg(windows)]
    #[test]
    fn a_present_waits_behind_the_work_its_queue_holds() {
        let Some(host) = crate::host::host() else {
            return;
        };
        if !crate::host::overlap() {
            return;
        }
        let _queues = TITLE_QUEUES.lock().unwrap_or_else(|e| e.into_inner());
        let null = std::ptr::null();
        let desc = QueueDesc {
            kind: 0,
            priority: 0,
            flags: 0,
            nodes: 0,
        };
        let mut q = null;
        assert_eq!(create(null, &desc, &QUEUE_IIDS[0], &mut q), S_OK);
        let mut fence = null;
        assert_eq!(
            crate::fence::create(null, 0, 0, &crate::fence::FENCE_IIDS[0], &mut fence),
            S_OK
        );
        let address = crate::watch::tests::pool(1);
        let iid = crate::object::guid(
            0x696442be,
            0xa72e,
            0x4059,
            [0xbc, 0x79, 0x5b, 0x5c, 0x98, 0x04, 0x0f, 0xad],
        );
        #[repr(align(8))]
        struct TextureWords([u32; 14]);
        let words = TextureWords([3, 0, 0, 0, 32, 0, 32, 1 | 1 << 16, 87, 1, 0, 1, 1, 0]);
        let mut resource = null;
        assert_eq!(
            crate::resource::create_at_address(
                null,
                address as u64,
                words.0.as_ptr().cast(),
                0,
                null.cast(),
                &iid,
                &mut resource,
            ),
            S_OK
        );
        let texture = crate::resource::host_copy(resource).unwrap();
        let state = texture.state.load(Ordering::Relaxed);
        let holding = || {
            held()
                .blocked(&read_value)
                .iter()
                .find(|b| b.0 == q as usize)
                .map(|b| b.1)
        };

        assert_eq!(wait(q, fence, 1), S_OK);
        assert_eq!(holding(), Some(1), "the queue waits for the fence");
        let plane = Plane {
            token: 0,
            count: 1,
            resources: &raw const resource,
            _rest: [0; 6],
        };
        assert_eq!(present(q, 1, &plane, null.cast()), S_OK);
        assert_eq!(holding(), Some(2), "the present is queued behind the wait");
        assert_eq!(
            texture.state.load(Ordering::Relaxed),
            state,
            "the frame's state is kept meanwhile"
        );
        // The title releases what it presented: the queued present keeps it.
        let release = |o: This| slot::<extern "system" fn(This) -> u32>(o, 0x10)(o);
        assert_eq!(release(resource), 0);

        crate::fence::set(fence, 1);
        pump_held(Why::Fence);
        assert_eq!(holding(), None, "the present ran once the fence moved");
        host.complete_all();
        drop(texture);
        crate::watch::tests::free(address);
        // SAFETY: a plain device query.
        assert!(unsafe { host.device.GetDeviceRemovedReason() }.is_ok());
        release(fence);
        assert_eq!(release(q), 0);
    }

    /// Where a queue's signal is: newer layouts have it two slots later.
    /// Another test may switch layouts while this one runs, so the queue's
    /// own table decides.
    pub(crate) fn signal_slot(q: This) -> usize {
        // SAFETY: one of our queues; its first field is the table.
        let table = unsafe { *q.cast::<*const *const ()>() };
        if table == NEWER_FUNCTIONS.0.as_ptr() {
            0x80
        } else {
            0x70
        }
    }

    /// Held work released by every reason, a completion's among them, is
    /// counted by reason in the frame line, and the frame's line is made
    /// (the count of a reason past the counts once panicked at the present).
    #[test]
    fn every_reason_held_work_is_released_by_is_counted_in_the_frame_line() {
        let ms = std::time::Duration::from_millis;
        let released: Vec<(u32, std::time::Duration, Why)> = Why::ALL
            .iter()
            .map(|&why| (0, ms(1), why))
            .chain([(2, ms(3), Why::Completed)])
            .collect();
        let by = released_by(&released);
        assert_eq!(by.len(), Why::ALL.len(), "{by:?}");
        assert_eq!(by.last(), Some(&("completed", 2)));
        assert!(
            by.iter().take(Why::ALL.len() - 1).all(|&(_, n)| n == 1),
            "{by:?}"
        );
        assert_eq!(
            released_by(&[(0, ms(1), Why::Completed)]),
            [("completed", 1)]
        );
        #[cfg(windows)]
        {
            account().released.extend(released);
            account_frame(1);
            assert!(account().released.is_empty(), "the frame took them");
        }
    }

    #[test]
    fn queue_signals_complete_at_once() {
        let null = std::ptr::null();
        let desc = QueueDesc {
            kind: 0,
            priority: 0,
            flags: 0,
            nodes: 0,
        };
        let mut q = null;
        assert_eq!(create(null, &desc, &QUEUE_IIDS[0], &mut q), S_OK);
        // SAFETY: the title reads the type at +0x10.
        assert_eq!(unsafe { q.cast::<u8>().add(0x10).cast::<u32>().read() }, 0);
        let mut fence = null;
        let fence_iid = &crate::fence::FENCE_IIDS[0];
        assert_eq!(
            crate::fence::create(null, 0, 0, fence_iid, &mut fence),
            S_OK
        );
        let signal = slot::<extern "system" fn(This, This, u64) -> HRESULT>(q, signal_slot(q));
        assert_eq!(signal(q, fence, 7), S_OK);
        assert_eq!(
            slot::<extern "system" fn(This) -> u64>(fence, 0x40)(fence),
            7
        );
        assert_eq!(slot::<extern "system" fn(This) -> u32>(q, 0x10)(q), 0);
    }

    /// Grand Theft Auto VI's compute queue: `{COMPUTE, 0, pipe 1, queue 0}`.
    #[test]
    fn console_queues_are_created_as_on_pc() {
        let null = std::ptr::null();
        let desc = QueueDescX {
            kind: 2,
            flags: 0,
            engine_or_pipe: 1,
            queue: 0,
        };
        let mut q = null;
        assert_eq!(create_x(null, &desc, &QUEUE_IIDS[0], &mut q), S_OK);
        assert_eq!(queue(q).desc.kind, 2);
        let none = std::ptr::null();
        assert_eq!(create_x(null, none, &QUEUE_IIDS[0], &mut q), E_INVALIDARG);
    }

    #[test]
    fn newer_queues_have_their_methods_two_slots_later() {
        let (old, new) = (queue_table(false), queue_table(true));
        assert_eq!(old[10], new[10]); // submission stays at 0x50
        assert_eq!(old[14], new[16]); // signal 0x70 → 0x80
        assert_eq!(old[30], new[32]); // present 0xf0 → 0x100
        assert_eq!(new[35], execute_x as *const ()); // 0x118
    }

    #[test]
    fn timestamps_and_values_are_written() {
        let mut at = [0u64; 2];
        Written::Value(0x1_0000_0002).write(at.as_mut_ptr().cast(), 4);
        assert_eq!(at[0], 2);
        Written::GpuTimestamp.write(at[1..].as_mut_ptr().cast(), 8);
        let first = at[1];
        std::thread::sleep(std::time::Duration::from_millis(2));
        Written::GpuTimestamp.write(at[1..].as_mut_ptr().cast(), 8);
        assert!(at[1] > first);
    }
}
