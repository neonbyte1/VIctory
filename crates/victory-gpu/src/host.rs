//! The host GPU: one D3D12 device and direct queue (decision 0007), and a
//! window to present on.
//!
//! The title's submissions are waited for, so CPU and GPU never overlap.
//! With `VICTORY_GPU_OVERLAP=1` they return once the host GPU has them and
//! complete later, as on the console, so that the title records and
//! simulates while the GPU runs: what shaders wrote goes back to title
//! memory, and the fences signalled behind them move, when the completion
//! thread sees them done ([`Host::submit`], [`Host::finish`]).
//!
//! Overlapping, a submission completes before the next one is prepared
//! only if its completion writes title memory nobody declared (work after
//! it, writes through the heap's views, page tables) or takes marks shared
//! with later ones: a barrier. The others stay in flight with what they
//! write in title memory on completion named (declared ranges, bottom-of-
//! pipe writes, `crate::shadow::fly`), and each has its own snapshots
//! (`crate::shadow::Shadow::writing`) and write marks
//! (`crate::shadow::capture_write_marks`). A later submission completes
//! them first only where its preparing reads what they write
//! ([`Settled`]), or where it declares ranges they declare
//! ([`declared_overlap`]).

use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::{Mutex, OnceLock};
use std::time::{Duration, Instant};

use windows::Win32::Foundation::{HANDLE, HWND, LPARAM, LRESULT, RECT, WPARAM};
use windows::Win32::Graphics::Direct3D::Fxc::D3DCompile;
use windows::Win32::Graphics::Direct3D::{
    D3D_FEATURE_LEVEL_12_0, D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST, ID3DBlob,
};
use windows::Win32::Graphics::Direct3D12::*;
use windows::Win32::Graphics::Dxgi::Common::*;
use windows::Win32::Graphics::Dxgi::*;
use windows::Win32::System::LibraryLoader::GetModuleHandleW;
use windows::Win32::System::Threading::{CreateEventW, INFINITE, WaitForSingleObject};
use windows::Win32::UI::WindowsAndMessaging::*;
use windows::core::{Interface, PCSTR, PCWSTR, s, w};

/// Resource states without the console's own bits (compression hints from
/// bit 25 up), which the host rejects.
pub(crate) fn pc_state(state: u32) -> u32 {
    state & 0x01ff_ffff
}

/// A console state as a host texture state: PC bits only, and none of the
/// buffer-only ones the console allows on textures (vertex, constant and
/// index buffer, stream out, indirect arguments, as in "generic read").
pub(crate) fn texture_state(state: u32) -> u32 {
    const BUFFER_ONLY: u32 = 0x1 | 0x2 | 0x100 | 0x200;
    pc_state(state) & !BUFFER_ONLY
}

/// Render target views live only until their command is recorded, so a
/// ring of host descriptors is enough.
const RTV_RING: u32 = 1024;

pub(crate) struct Host {
    pub(crate) device: ID3D12Device,
    pub(crate) depth_bounds_supported: bool,
    queue: ID3D12CommandQueue,
    fence: ID3D12Fence,
    /// The last value signalled on `fence` and the submissions in flight:
    /// the submission lock.
    submitted: Mutex<Submitted>,
    /// A submission went into flight (the completion thread waits for one).
    flown: std::sync::Condvar,
    event: HANDLE,
    rtvs: ID3D12DescriptorHeap,
    rtv_size: u32,
    /// A ring of depth views, used as `rtvs`.
    dsvs: ID3D12DescriptorHeap,
    dsv_size: u32,
    next_dsv: AtomicU32,
    /// A ring of CPU-only views, for the CPU handles of unordered access
    /// clears, used as `rtvs`.
    cpu_views: ID3D12DescriptorHeap,
    cpu_view_size: u32,
    next_cpu_view: AtomicU32,
    /// With `VICTORY_D3D12_DEBUG` set (diagnostics builds): the debug
    /// layer's messages.
    messages: Option<ID3D12InfoQueue>,
    next_rtv: AtomicU32,
    presenter: Mutex<Presentation>,
}

// SAFETY: D3D12 devices, queues, fences and descriptor heaps are
// free-threaded; the event handle is only waited on under `submitted`.
unsafe impl Send for Host {}
// SAFETY: as above.
unsafe impl Sync for Host {}

/// Time spent in submissions since the last frame-rate line, in
/// microseconds: preparing them on the CPU (title memory copies, texture
/// detiling), then running them and waiting for the GPU (where the
/// submitting thread waits: for its own submission out of overlap, for
/// those in flight before work that needs them done).
static SUBMISSION_US: [std::sync::atomic::AtomicU64; 2] =
    [std::sync::atomic::AtomicU64::new(0), std::sync::atomic::AtomicU64::new(0)];

/// After submissions: shader-written ranges compared and copied back into
/// title memory (microseconds, bytes that differed), since the last
/// frame-rate line.
static COPY_BACK: [std::sync::atomic::AtomicU64; 2] =
    [std::sync::atomic::AtomicU64::new(0), std::sync::atomic::AtomicU64::new(0)];

/// Of the preparing: title memory copies (microseconds, bytes) and texture
/// refreshes (microseconds, bytes hashed), since the last frame-rate line.
pub(crate) static PREPARING: [std::sync::atomic::AtomicU64; 4] = [
    std::sync::atomic::AtomicU64::new(0),
    std::sync::atomic::AtomicU64::new(0),
    std::sync::atomic::AtomicU64::new(0),
    std::sync::atomic::AtomicU64::new(0),
];

/// What the runtime does when the host GPU fails for good: report it like a
/// crash and end the run. The console has no such state, and carrying on
/// draws nothing at seconds per frame.
static ON_FAILURE: OnceLock<fn(&str) -> !> = OnceLock::new();

/// Sets what a host GPU failure does (the runtime's crash report).
pub fn on_failure(report: fn(&str) -> !) {
    let _ = ON_FAILURE.set(report);
}

/// Ends the run on a host GPU failure: what failed, the removal reason and
/// the debug layer's first error, if any (`VICTORY_D3D12_DEBUG=1` in a
/// diagnostics build names the command).
pub(crate) fn fail(what: std::fmt::Arguments) -> ! {
    let mut text = what.to_string();
    if let Some(host) = host() {
        // SAFETY: a plain query.
        if let Err(e) = unsafe { host.device.GetDeviceRemovedReason() } {
            text.push_str(&format!("; device removed: {e}"));
        }
        match host.log_messages() {
            Some(first) => text.push_str(&format!("; debug layer: {first}")),
            None if host.messages.is_none() => text.push_str(
                "; a diagnostics build with VICTORY_D3D12_DEBUG=1 names the command",
            ),
            None => {}
        }
    }
    tracing::error!("{text}");
    match ON_FAILURE.get() {
        Some(report) => report(&text),
        None => std::process::abort(),
    }
}

/// The host GPU, created on first use; `None` without one (a run then
/// behaves as before: nothing executes).
pub(crate) fn host() -> Option<&'static Host> {
    static HOST: OnceLock<Option<Host>> = OnceLock::new();
    HOST.get_or_init(|| match create() {
        Ok(host) => Some(host),
        Err(e) => {
            tracing::warn!(error = %e, "no host D3D12 device; command lists will not run");
            None
        }
    })
    .as_ref()
}

fn create() -> windows::core::Result<Host> {
    // SAFETY: plain D3D12 object creation with valid descriptions.
    unsafe {
        // The debug layer only in builds with diagnostics.
        let debug = cfg!(feature = "diagnostics") && crate::switch::on("VICTORY_D3D12_DEBUG");
        if debug {
            let mut layer: Option<ID3D12Debug> = None;
            match D3D12GetDebugInterface(&mut layer) {
                Ok(()) => layer.iter().for_each(|l| l.EnableDebugLayer()),
                Err(e) => tracing::warn!(error = %e, "no D3D12 debug layer"),
            }
            // GPU-based validation (`VICTORY_D3D12_GBV`): out-of-range
            // descriptor heap indices and the like become messages naming
            // the command instead of page faults. Without state tracking
            // (directly indexed textures are not transitioned).
            if crate::switch::on("VICTORY_D3D12_GBV")
                && let Some(l) = &layer
            {
                if let Ok(d1) = l.cast::<ID3D12Debug1>() {
                    d1.SetEnableGPUBasedValidation(true);
                }
                if let Ok(d2) = l.cast::<ID3D12Debug2>() {
                    d2.SetGPUBasedValidationFlags(
                        D3D12_GPU_BASED_VALIDATION_FLAGS_DISABLE_STATE_TRACKING,
                    );
                }
                tracing::info!("GPU-based validation on");
            }
        }
        // Diagnostics: device-removed data (the command that faulted, the
        // faulting address), read when the device is removed.
        if dred_on() {
            let mut dred: Option<ID3D12DeviceRemovedExtendedDataSettings> = None;
            if D3D12GetDebugInterface(&mut dred).is_ok()
                && let Some(dred) = dred
            {
                dred.SetAutoBreadcrumbsEnablement(D3D12_DRED_ENABLEMENT_FORCED_ON);
                dred.SetPageFaultEnablement(D3D12_DRED_ENABLEMENT_FORCED_ON);
            }
        }
        let mut device: Option<ID3D12Device> = None;
        // Diagnostics: the software adapter, on which a shader's wild access
        // is an access violation in this process (a crash report with the
        // address and instruction) instead of a GPU fault and driver reset.
        if cfg!(feature = "diagnostics") && crate::switch::on("VICTORY_D3D12_WARP") {
            let adapter = warp_adapter()?;
            D3D12CreateDevice(&adapter, D3D_FEATURE_LEVEL_12_0, &mut device)?;
            tracing::info!("host device on the software adapter (WARP)");
        } else {
            D3D12CreateDevice(None, D3D_FEATURE_LEVEL_12_0, &mut device)?;
            // Whether a GPU hang ends the run or freezes the PC (#1054).
            crate::tdr::log();
        }
        let device = device.ok_or_else(windows::core::Error::empty)?;
        let mut options = D3D12_FEATURE_DATA_D3D12_OPTIONS2::default();
        let depth_bounds_supported = device
            .CheckFeatureSupport(
                D3D12_FEATURE_D3D12_OPTIONS2,
                (&raw mut options).cast(),
                size_of_val(&options) as u32,
            )
            .is_ok()
            && options.DepthBoundsTestSupported.as_bool();
        let messages = debug
            .then(|| device.cast::<ID3D12InfoQueue>().ok())
            .flatten();
        // A pipeline library the host rejects (made on another adapter or
        // driver, or damaged) is expected: `pipeline::library` opens an
        // empty one instead and says so. Not stored, it is no error a later
        // check of the messages finds.
        if let Some(queue) = &messages {
            let mut expected = [
                D3D12_MESSAGE_ID_CREATEPIPELINELIBRARY_INVALIDLIBRARYBLOB,
                D3D12_MESSAGE_ID_CREATEPIPELINELIBRARY_DRIVERVERSIONMISMATCH,
                D3D12_MESSAGE_ID_CREATEPIPELINELIBRARY_ADAPTERVERSIONMISMATCH,
                D3D12_MESSAGE_ID_CREATEPIPELINELIBRARY_UNSUPPORTED,
            ];
            let filter = D3D12_INFO_QUEUE_FILTER {
                DenyList: D3D12_INFO_QUEUE_FILTER_DESC {
                    NumIDs: expected.len() as u32,
                    pIDList: expected.as_mut_ptr(),
                    ..Default::default()
                },
                ..Default::default()
            };
            if let Err(e) = queue.PushStorageFilter(&filter) {
                tracing::debug!(error = %e, "expected pipeline library messages not filtered");
            }
        }
        let queue: ID3D12CommandQueue = device.CreateCommandQueue(&D3D12_COMMAND_QUEUE_DESC {
            Type: D3D12_COMMAND_LIST_TYPE_DIRECT,
            ..Default::default()
        })?;
        let fence: ID3D12Fence = device.CreateFence(0, D3D12_FENCE_FLAG_NONE)?;
        let event = CreateEventW(None, false, false, None)?;
        let rtvs: ID3D12DescriptorHeap =
            device.CreateDescriptorHeap(&D3D12_DESCRIPTOR_HEAP_DESC {
                Type: D3D12_DESCRIPTOR_HEAP_TYPE_RTV,
                NumDescriptors: RTV_RING,
                ..Default::default()
            })?;
        let rtv_size = device.GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_RTV);
        let dsvs: ID3D12DescriptorHeap =
            device.CreateDescriptorHeap(&D3D12_DESCRIPTOR_HEAP_DESC {
                Type: D3D12_DESCRIPTOR_HEAP_TYPE_DSV,
                NumDescriptors: RTV_RING,
                ..Default::default()
            })?;
        let dsv_size = device.GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_DSV);
        let cpu_views: ID3D12DescriptorHeap =
            device.CreateDescriptorHeap(&D3D12_DESCRIPTOR_HEAP_DESC {
                Type: D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV,
                NumDescriptors: RTV_RING,
                ..Default::default()
            })?;
        let cpu_view_size =
            device.GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
        tracing::info!("host D3D12 device ready");
        // The console fuses its multiply-adds (`v_fma_f32`, `v_fmac_f32`,
        // `v_fma_mix*`); the lowering's DXIL `FMad` is fused only where the
        // host chooses (the RX 6700 XT does, WARP does not).
        match victory_shader::emit::host::Host::from_device(device.clone()).fuses_fmad()
        {
            Ok(true) => {}
            Ok(false) => tracing::warn!(
                "the host GPU does not fuse multiply-adds: results of the console's fused v_fma and v_fmac can differ in kind here"
            ),
            Err(e) => tracing::debug!(error = %e, "multiply-add fusing not checked"),
        }
        Ok(Host {
            device,
            depth_bounds_supported,
            queue,
            fence,
            submitted: Mutex::new(Submitted::default()),
            flown: std::sync::Condvar::new(),
            event,
            rtvs,
            rtv_size,
            dsvs,
            dsv_size,
            next_dsv: AtomicU32::new(0),
            cpu_views,
            cpu_view_size,
            next_cpu_view: AtomicU32::new(0),
            messages,
            next_rtv: AtomicU32::new(0),
            presenter: Mutex::new(Presentation::default()),
        })
    }
}

/// Diagnostics: our process's private commit and working set, bytes.
#[cfg(feature = "diagnostics")]
fn process_memory() -> (u64, u64) {
    use windows_sys::Win32::System::ProcessStatus::{K32GetProcessMemoryInfo, PROCESS_MEMORY_COUNTERS};
    use windows_sys::Win32::System::Threading::GetCurrentProcess;
    let mut c = PROCESS_MEMORY_COUNTERS {
        cb: size_of::<PROCESS_MEMORY_COUNTERS>() as u32,
        ..Default::default()
    };
    // SAFETY: our own process and a sized structure.
    unsafe { K32GetProcessMemoryInfo(GetCurrentProcess(), &mut c, c.cb) };
    (c.PagefileUsage as u64, c.WorkingSetSize as u64)
}

/// DXGI may return hardware for `EnumWarpAdapter` (notably under Proton).
/// Reject it before creating a device in diagnostics' software-only mode.
fn warp_adapter() -> windows::core::Result<IDXGIAdapter> {
    // SAFETY: plain DXGI adapter queries.
    unsafe {
        let factory: IDXGIFactory4 = CreateDXGIFactory2(DXGI_CREATE_FACTORY_FLAGS(0))?;
        let adapter: IDXGIAdapter = factory.EnumWarpAdapter()?;
        let desc = adapter.cast::<IDXGIAdapter1>()?.GetDesc1()?;
        if desc.Flags & DXGI_ADAPTER_FLAG_SOFTWARE.0 as u32 == 0 {
            tracing::warn!(
                vendor = format_args!("{:#x}", desc.VendorId),
                flags = format_args!("{:#x}", desc.Flags),
                "DXGI did not identify the WARP adapter as software; refusing device"
            );
            return Err(windows::core::Error::from_hresult(
                windows::Win32::Foundation::E_FAIL,
            ));
        }
        Ok(adapter)
    }
}

/// Diagnostics, `VICTORY_GPU_TIMING`: GPU time per kind of command and
/// pipeline (ms, commands) and lists' wall time against the sum of their
/// commands' (ms), since the last frame-rate line.
#[cfg(feature = "diagnostics")]
type GpuTimes = (std::collections::HashMap<String, (f64, u64)>, f64, f64);
#[cfg(feature = "diagnostics")]
static GPU_TIMES: Mutex<Option<GpuTimes>> = Mutex::new(None);

/// Adds a timed list's commands (from [`HostList::slowest`]) to [`GPU_TIMES`],
/// by the command's kind and pipeline: its note before the first ` | `,
/// without counts and addresses.
#[cfg(feature = "diagnostics")]
fn gpu_times(wall: Duration, commands: &[(f64, String)]) {
    let mut times = GPU_TIMES.lock().unwrap_or_else(|e| e.into_inner());
    let (by, wall_ms, sum) = times.get_or_insert_with(Default::default);
    *wall_ms += wall.as_secs_f64() * 1e3;
    for (ms, note) in commands {
        // "draw 3x1 [mesh 1f.., pixel 2e..] | ...": the kind and the pipeline.
        let head = note.split(" | ").next().unwrap_or_default();
        let kind = head.split(' ').next().unwrap_or_default();
        let key = match head.find('[') {
            Some(at) => format!("{kind} {}", &head[at..]),
            None => kind.to_owned(),
        };
        let e = by.entry(key).or_default();
        (e.0, e.1) = (e.0 + ms, e.1 + 1);
        *sum += ms;
    }
}

/// Logs [`GPU_TIMES`] per frame over `frames` frames: the 25 costliest.
#[cfg(feature = "diagnostics")]
fn log_gpu_times(frames: f64) {
    let Some((by, wall, sum)) = GPU_TIMES.lock().unwrap_or_else(|e| e.into_inner()).take() else {
        return;
    };
    tracing::info!(
        "GPU timing: lists {:.1} ms/frame of wall time, their commands {:.1} ms/frame",
        wall / frames,
        sum / frames
    );
    let mut by: Vec<_> = by.into_iter().collect();
    by.sort_by(|a, b| b.1.0.total_cmp(&a.1.0));
    for (key, (ms, n)) in by.iter().take(25) {
        tracing::info!("  GPU {:.2} ms/frame in {:.1} commands/frame: {key}", ms / frames, *n as f64 / frames);
    }
}

#[cfg(all(test, feature = "diagnostics"))]
#[path = "host/timing_warp.rs"]
mod timing_warp;

#[cfg(all(test, feature = "diagnostics"))]
fn finish_timing_test_window(_: Instant) -> u64 {
    let count = GPU_TIMES
        .lock()
        .unwrap()
        .as_ref()
        .map_or(0, |(by, _, _)| by.values().map(|(_, count)| *count).sum());
    log_gpu_times(1.0);
    count
}

/// Diagnostics: the host device's video memory use and budget, in MB, local
/// (on the card) and non-local (system memory the GPU reads), as the
/// system's memory manager sees them: use over budget makes it page
/// allocations at submissions.
pub(crate) fn video_memory() -> Option<[(f64, f64); 2]> {
    static ADAPTER: OnceLock<Option<IDXGIAdapter3>> = OnceLock::new();
    let adapter = ADAPTER
        .get_or_init(|| {
            let device = &host()?.device;
            // SAFETY: plain device and DXGI queries.
            unsafe {
                let luid = device.GetAdapterLuid();
                let factory: IDXGIFactory4 = CreateDXGIFactory2(DXGI_CREATE_FACTORY_FLAGS(0)).ok()?;
                factory.EnumAdapterByLuid(luid).ok()
            }
        })
        .as_ref()?;
    let info = |group| {
        let mut info = DXGI_QUERY_VIDEO_MEMORY_INFO::default();
        // SAFETY: a valid out pointer.
        unsafe { adapter.QueryVideoMemoryInfo(0, group, &mut info) }.ok()?;
        Some((info.CurrentUsage as f64 / 1e6, info.Budget as f64 / 1e6))
    };
    Some([info(DXGI_MEMORY_SEGMENT_GROUP_LOCAL)?, info(DXGI_MEMORY_SEGMENT_GROUP_NON_LOCAL)?])
}

#[cfg(test)]
pub(crate) mod tests {
    thread_local! {
        /// Every transition barrier built on this thread (resource, before,
        /// after), in order.
        pub(crate) static ISSUED: std::cell::RefCell<Vec<(usize, u32, u32)>> = const { std::cell::RefCell::new(Vec::new()) };
    }

    /// The barriers built on this thread since the last call.
    pub(crate) fn issued() -> Vec<(usize, u32, u32)> {
        ISSUED.with(|l| std::mem::take(&mut *l.borrow_mut()))
    }

    /// Replays transitions in the order the GPU runs them from the states
    /// in `states` (resources not in it are not checked), by the debug
    /// layer's rule: each barrier's before-state is the resource's current
    /// one. The first that is not, else the states after.
    pub(crate) fn replay(issued: &[(usize, u32, u32)], states: &mut std::collections::HashMap<usize, u32>) -> Result<(), String> {
        for (k, &(resource, before, after)) in issued.iter().enumerate() {
            if let Some(state) = states.get_mut(&resource) {
                if *state != before {
                    return Err(format!("barrier {k} on {resource:#x}: before {before:#x}, but the resource is in {state:#x}"));
                }
                *state = after;
            }
        }
        Ok(())
    }

    #[test]
    fn swap_chain_failures_retry_once_per_second_and_success_clears_the_wait() {
        use super::{Duration, Instant, SwapChainRetry};

        let start = Instant::now();
        let mut retry = SwapChainRetry::default();
        let mut attempts = 0;
        // Two seconds at 60 Hz, with a failure on every eligible attempt.
        for frame in 0..120 {
            retry.attempt(start + Duration::from_millis(frame * 1000 / 60), || {
                attempts += 1;
                Err::<(), _>("desktop unavailable")
            });
        }
        assert_eq!(attempts, 2);
        let recovered = start + Duration::from_secs(2);
        assert_eq!(retry.attempt(recovered, || Ok::<_, ()>(42)), Some(Ok(42)));
        assert_eq!(retry.attempt(recovered, || Ok::<_, ()>(43)), Some(Ok(43)));
    }

    use super::*;

    /// Shaders that use title memory mark their reads in their own bitmap:
    /// read marks of both bitmaps are taken, each slot once until it is
    /// forgotten, and a forgotten slot's next read is taken again.
    #[test]
    fn read_marks_of_both_bitmaps_are_taken() {
        let Some(host) = host() else { return };
        // No submission takes the marks meanwhile.
        let _one = host.submitted.lock().unwrap_or_else(|e| e.into_inner());
        let (plain, memory) = (victory_shader::emit::HEAP_VIEWS - 2, victory_shader::emit::HEAP_VIEWS - 1);
        crate::shadow::tests::mark_read(false, plain);
        crate::shadow::tests::mark_read(true, memory);
        let taken = crate::shadow::take_reads(true);
        assert!(taken.contains(&plain), "a plain shader's read");
        assert!(taken.contains(&memory), "a read by a shader that uses title memory");
        let again = crate::shadow::take_reads(true);
        assert!(!again.contains(&plain) && !again.contains(&memory), "each once");
        crate::shadow::forget_read(memory);
        assert!(
            !crate::shadow::take_reads(true).contains(&memory),
            "forgotten: its mark cleared"
        );
        crate::shadow::tests::mark_read(true, memory);
        assert!(
            crate::shadow::take_reads(true).contains(&memory),
            "read again after it was forgotten"
        );
    }

    /// The textures shaders read through the heap since the last eviction
    /// pass are named again at each pass (a slot's read is reported as new
    /// only once), and the marks are cleared for the next.
    #[test]
    fn heap_reads_since_the_last_pass_name_their_textures() {
        let Some(host) = host() else { return };
        let null = std::ptr::null();
        let heap_desc: [u32; 8] = [0, 4, 1, 0, 0, 0, 0, 0];
        let mut heap = null;
        let iid = crate::object::guid(0x8efb471d, 0x616c, 0x4f49, [0x90, 0xf7, 0x12, 0x7b, 0xb7, 0x63, 0xfa, 0x51]);
        assert_eq!(crate::descriptor::create_heap(null, heap_desc.as_ptr().cast(), &iid, &mut heap), 0);
        let title_heap = crate::descriptor::heap(heap);
        // A 4x1 RGBA8 texture, linear, in memory of its own, and a 2D view.
        let memory = vec![0u8; 1 << 16];
        let address = memory.as_ptr() as u64;
        let words: [u32; 14] = [3, 0, 0, 0, 4, 0, 1, 1 | 1 << 16, 28, 1, 0, 0x100, 0, 0];
        let mut view = [0u32; 16];
        view[..6].copy_from_slice(&[28, 4, 0x1688, 0, 0, 1]);
        (view[10], view[11]) = (address as u32, (address >> 32) as u32);
        crate::descriptor::create_placed_srv(null, &words, &view, title_heap.start);
        let heap_host = title_heap.host().expect("a host view heap");
        assert!(heap_host.mirror(0, title_heap.start, true).is_some());
        let texture = crate::resource::texture_at(address, &words).expect("its resource");
        let raw = windows::core::Interface::as_raw(&crate::resource::made_copy(texture).unwrap().texture) as usize;
        // No submission takes the marks meanwhile.
        let _one = host.submitted.lock().unwrap_or_else(|e| e.into_inner());
        for _ in 0..2 {
            crate::shadow::tests::mark_read(false, 0);
            let slots = crate::shadow::reads_since();
            assert!(slots.contains(&0), "the slot read");
            assert!(crate::descriptor::read_textures(&slots).contains(&raw), "its texture named");
            assert!(!crate::shadow::reads_since().contains(&0), "the marks cleared");
        }
    }

    /// The idle rule against the heap reads that thrashed (run evict1): a
    /// texture shaders read through the heap in every frame stays over more
    /// frames and time than the idle limits; one read once is given up. A
    /// list of the title's naming a texture (a table read, a copy) stamps it
    /// as used; a list of ours (uploads) does not.
    #[test]
    fn textures_read_every_frame_stay_and_title_lists_stamp_theirs() {
        let Some(host) = host() else { return };
        let null = std::ptr::null();
        let heap_desc: [u32; 8] = [0, 8, 1, 0, 0, 0, 0, 0];
        let mut heap = null;
        let iid = crate::object::guid(0x8efb471d, 0x616c, 0x4f49, [0x90, 0xf7, 0x12, 0x7b, 0xb7, 0x63, 0xfa, 0x51]);
        assert_eq!(crate::descriptor::create_heap(null, heap_desc.as_ptr().cast(), &iid, &mut heap), 0);
        let title_heap = crate::descriptor::heap(heap);
        let step = 32;
        // Two 4x1 RGBA8 textures, linear, in memory of their own, viewed at
        // slots 5 and 6.
        let memory = vec![0u8; 1 << 17];
        let words: [u32; 14] = [3, 0, 0, 0, 4, 0, 1, 1 | 1 << 16, 28, 1, 0, 0x100, 0, 0];
        let heap_host = title_heap.host().expect("a host view heap");
        let textures: Vec<crate::object::This> = [5u32, 6]
            .iter()
            .map(|&slot| {
                let address = memory.as_ptr() as u64 + u64::from(slot) * (1 << 13);
                let mut view = [0u32; 16];
                view[..6].copy_from_slice(&[28, 4, 0x1688, 0, 0, 1]);
                (view[10], view[11]) = (address as u32, (address >> 32) as u32);
                let at = title_heap.start + u64::from(slot) * step;
                crate::descriptor::create_placed_srv(null, &words, &view, at);
                assert!(heap_host.mirror(slot, at, true).is_some());
                crate::resource::texture_at(address, &words).expect("its resource")
            })
            .collect();
        let owned: Vec<_> = textures.iter().map(|&t| crate::resource::owned(t).expect("may be given up")).collect();
        let (idle_frames, idle_ms) = (crate::resource::IDLE_FRAMES, crate::resource::IDLE_MS);
        {
            // No submission takes the marks meanwhile.
            let _one = host.submitted.lock().unwrap_or_else(|e| e.into_inner());
            // Passes a third of the idle limits apart, over three times them.
            for k in 0..10 {
                crate::shadow::tests::mark_read(false, 5);
                if k == 0 {
                    crate::shadow::tests::mark_read(false, 6);
                }
                let read = crate::descriptor::read_textures(&crate::shadow::reads_since());
                crate::resource::evict_among(&owned, &read, 1000 + k * idle_frames / 3, 1_000_000 + k * idle_ms / 3, 0);
                assert!(crate::resource::made_copy(textures[0]).is_some(), "read every frame: stays (frame {k})");
            }
        }
        assert!(crate::resource::made_copy(textures[1]).is_none(), "read once: given up");
        drop(owned);
        // A title list naming the texture stamps it, an upload list does not.
        let copy = crate::resource::host_copy(textures[0]).unwrap();
        for (ours, stamped) in [(true, false), (false, true)] {
            copy.unstamp();
            let list = host.new_list().unwrap();
            if ours {
                list.set_ours();
            }
            let _ = list.transition(textures[0], 0x40);
            assert_eq!(copy.last_used().0 != u64::MAX, stamped, "ours: {ours}");
        }
    }

    /// Textures the heaps view are brought up to date only if the title used
    /// them lately: one whose pages changed but that no shader read is not
    /// uploaded; read in the last pass, it is.
    #[test]
    fn heap_textures_unread_lately_are_not_brought_up() {
        let Some(host) = host() else { return };
        let address = crate::watch::tests::pool(1) as u64;
        let words: [u32; 14] = [3, 0, 0, 0, 64, 0, 64, 1 | 1 << 16, 28, 1, 0, 0x100, 0, 0];
        let iid = crate::object::guid(0x696442be, 0xa72e, 0x4059, [0xbc, 0x79, 0x5b, 0x5c, 0x98, 0x04, 0x0f, 0xad]);
        let mut object = std::ptr::null();
        let null = std::ptr::null();
        assert_eq!(crate::resource::create_at_address(null, address, words.as_ptr().cast(), 0, std::ptr::null(), &iid, &mut object), 0);
        let copy = crate::resource::host_copy(object).expect("a host texture");
        let mut desc = [0u32; 10];
        desc[..6].copy_from_slice(&[28, 4, 0x1688, 0, 0, 1]);
        let viewed = std::sync::Arc::new(crate::bindless::Viewed::default());
        viewed.set(0, Some(&crate::descriptor::View::Shader(object as usize, Some(desc))), true);
        // No submission uploads meanwhile.
        let _one = host.submitted.lock().unwrap_or_else(|e| e.into_inner());
        // SAFETY: the texture's memory, 256-byte rows of 64 texels.
        unsafe { std::ptr::write_bytes(address as *mut u8, 0x40, 64 * 64 * 4) };
        copy.unstamp();
        let _ = crate::bindless::uploads(std::slice::from_ref(&viewed), &[], &mut crate::host::nothing_in_flight);
        assert_eq!(crate::texture::uploads(object), 0, "not read lately: not brought up");
        copy.stamp();
        let _ = crate::bindless::uploads(std::slice::from_ref(&viewed), &[], &mut crate::host::nothing_in_flight);
        assert_eq!(crate::texture::uploads(object), 1, "read in the last pass: brought up");
    }

    /// Heap textures brought up in one submission are refreshed side by
    /// side, each into its own staging and its own upload.
    #[test]
    fn heap_textures_brought_up_together_each_get_their_upload() {
        let Some(host) = host() else { return };
        let iid = crate::object::guid(0x696442be, 0xa72e, 0x4059, [0xbc, 0x79, 0x5b, 0x5c, 0x98, 0x04, 0x0f, 0xad]);
        let pool = crate::watch::tests::pool(6);
        let viewed = std::sync::Arc::new(crate::bindless::Viewed::default());
        let objects: Vec<crate::object::This> = (0..6u32)
            .map(|k| {
                // RGBA8 128x128 (one page each), linear, each its own bytes.
                let address = (pool + k as usize * crate::watch::PAGE) as u64;
                // SAFETY: the texture's own page of the pool.
                unsafe { std::ptr::write_bytes(address as *mut u8, 0x10 * k as u8, crate::watch::PAGE) };
                let words: [u32; 14] = [3, 0, 0, 0, 128, 0, 128, 1 | 1 << 16, 28, 1, 0, 0x100, 0, 0];
                let mut object = std::ptr::null();
                assert_eq!(crate::resource::create_at_address(std::ptr::null(), address, words.as_ptr().cast(), 0, std::ptr::null(), &iid, &mut object), 0);
                crate::resource::host_copy(object).expect("a host texture").stamp();
                let mut desc = [0u32; 10];
                desc[..6].copy_from_slice(&[28, 4, 0x1688, 0, 0, 1]);
                viewed.set(k, Some(&crate::descriptor::View::Shader(object as usize, Some(desc))), true);
                object
            })
            .collect();
        // No submission uploads meanwhile.
        let _one = host.submitted.lock().unwrap_or_else(|e| e.into_inner());
        let _ = crate::bindless::uploads(std::slice::from_ref(&viewed), &[], &mut crate::host::nothing_in_flight);
        for &object in &objects {
            assert_eq!(crate::texture::uploads(object), 1, "each brought up once");
        }
        let _ = crate::bindless::uploads(std::slice::from_ref(&viewed), &[], &mut crate::host::nothing_in_flight);
        for &object in &objects {
            assert_eq!(crate::texture::uploads(object), 1, "unchanged: not again");
        }
    }

    #[test]
    fn list_scratch_grows_to_fit_and_reuses_fenced_chunks() {
        let Some(host) = host() else { return };
        let list = host.new_list().unwrap();
        let first = list.scratch(1).unwrap();
        let second = list.scratch(1).unwrap();
        assert_eq!(second.0 - first.0, 256);
        assert_eq!(second.1 as usize - first.1 as usize, 256);
        assert_eq!(list.scratch.lock().unwrap().chunks[0].capacity, FIRST_SCRATCH_CHUNK);
        let large = list.scratch(96 << 10).unwrap();
        assert_eq!(list.scratch.lock().unwrap().chunks[1].capacity, 128 << 10);
        assert!(list.scratch(u64::MAX).is_none());
        assert!(list.scratch(SCRATCH_CHUNK + 1).is_some());
        list.close();
        host.execute(&[&list]);
        list.reset();
        assert_eq!(list.scratch(96 << 10).unwrap(), large);
        assert_eq!(list.scratch.lock().unwrap().chunks.len(), 2);
        list.close();
        host.execute(&[&list]);
        list.reset();
        assert_eq!(list.scratch(512).unwrap(), first);
        list.close();
    }

    #[test]
    fn present_compositor_builds_both_pipeline_states() {
        let Some(host) = host() else {
            eprintln!("skipping compositor initialization: no D3D12 device");
            return;
        };
        Compositor::new(&host.device, DXGI_FORMAT_R8G8B8A8_UNORM)
            .expect("create compositor root signature and both pipeline states");
        eprintln!("PresentX compositor initialized");
    }

    #[test]
    fn warp_adapter_accepts_only_software() {
        // This probes DXGI without creating a D3D12 device or running a title.
        // SAFETY: plain DXGI adapter queries.
        let raw = unsafe {
            let Ok(factory) = CreateDXGIFactory2::<IDXGIFactory4>(DXGI_CREATE_FACTORY_FLAGS(0))
            else {
                return;
            };
            factory.EnumWarpAdapter::<IDXGIAdapter>()
        };
        let Ok(raw) = raw else { return };
        // SAFETY: EnumWarpAdapter returns a DXGI adapter, which implements
        // IDXGIAdapter1; `raw` owns that interface.
        let desc = unsafe { raw.cast::<IDXGIAdapter1>().unwrap().GetDesc1().unwrap() };
        eprintln!(
            "EnumWarpAdapter: vendor={:#x}, flags={:#x}",
            desc.VendorId, desc.Flags
        );
        assert_eq!(
            warp_adapter().is_ok(),
            desc.Flags & DXGI_ADAPTER_FLAG_SOFTWARE.0 as u32 != 0,
        );
    }

    /// A node of the device-removed data is a list by pointer or by name.
    /// The name is what still matches when the runtime records another
    /// object than the one we hold (the debug layer wraps ours), and a
    /// different list or an unnamed node never matches.
    #[test]
    fn a_breadcrumb_node_is_found_by_pointer_or_by_name() {
        let ours = [0x1000, 0x1008];
        let name = list_name(7);
        let is = |pointer, text| node_is_list(pointer, text, &ours, &name);
        assert_eq!(name, "victory host list 7");
        assert!(is(Some(0x1000), None));
        assert!(is(Some(0x1008), Some("other")));
        assert!(is(Some(0x9999), Some("victory host list 7")));
        assert!(is(None, Some("victory host list 7")));
        assert!(!is(Some(0x9999), Some("victory host list 70")));
        assert!(!is(Some(0x9999), None));
        assert!(!is(None, None));
        // The debug layer appends its own note to the name we gave.
        assert!(is(None, Some("victory host list 7 (debug layer indirect)")));
        assert!(is(None, Some("victory host list 7 (x)")));
        assert!(!is(
            None,
            Some("victory host list 70 (debug layer indirect)")
        ));
        assert!(!is(None, Some("victory host list 7 debug layer")));
        assert!(!is(None, Some("victory host list 7(")));
        assert!(is(None, Some("victory host list 7")));
        assert_eq!(list_name(70), "victory host list 70");
    }

    /// The pointers of one list: the interface pointers it answers with
    /// are all there, and another list's are not.
    #[test]
    fn a_list_knows_every_pointer_it_answers_with() {
        // SAFETY: plain device and list creation on the software adapter.
        unsafe {
            let Ok(adapter) = warp_adapter() else { return };
            let mut device: Option<ID3D12Device> = None;
            if D3D12CreateDevice(&adapter, D3D_FEATURE_LEVEL_12_0, &mut device).is_err() {
                return;
            }
            let device = device.unwrap();
            // An open list holds its allocator: one each.
            let make = || -> ID3D12GraphicsCommandList {
                let allocator: ID3D12CommandAllocator = device
                    .CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT)
                    .unwrap();
                device
                    .CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, &allocator, None)
                    .unwrap()
            };
            let (a, b) = (make(), make());
            let pointers = identities(&a);
            assert!(pointers.contains(&(a.as_raw() as usize)));
            let unknown = a.cast::<windows::core::IUnknown>().unwrap();
            assert!(pointers.contains(&(unknown.as_raw() as usize)));
            let command_list = a.cast::<ID3D12CommandList>().unwrap();
            assert!(pointers.contains(&(command_list.as_raw() as usize)));
            let other = identities(&b);
            assert!(pointers.iter().all(|p| !other.contains(p)));
            // As a node: the same object by any of its pointers; a node
            // naming the other list by pointer and our name still matches.
            let name = list_name(3);
            let is = |pointer, text| node_is_list(pointer, text, &pointers, &name);
            assert!(is(Some(command_list.as_raw() as usize), None));
            assert!(!is(Some(b.as_raw() as usize), None));
            assert!(is(Some(b.as_raw() as usize), Some(&name)));
        }
    }

    /// A node as the runtime gives it, built by hand: `HostList::is_node`
    /// reads the pointer and the copied name, and the name alone finds a
    /// list the runtime records under another object.
    #[test]
    fn is_node_reads_the_name_the_runtime_copied() {
        let Some(host) = host() else { return };
        let Ok(list) = host.new_list() else { return };
        let mut wide: Vec<u16> = list_name(list.id).encode_utf16().collect();
        wide.push(0);
        let mut node = D3D12_AUTO_BREADCRUMB_NODE::default();
        assert!(!list.is_node(&node), "an empty node is no list");
        node.pCommandListDebugNameW = windows::core::PCWSTR(wide.as_ptr());
        assert!(list.is_node(&node), "found by its name alone");
        let mut other: Vec<u16> = list_name(list.id + 1).encode_utf16().collect();
        other.push(0);
        node.pCommandListDebugNameW = windows::core::PCWSTR(other.as_ptr());
        assert!(!list.is_node(&node), "another list's name is no match");
        // With the debug layer: our name and the layer's note, in either
        // name field; and a list whose number only starts with ours is not it.
        let note = std::ffi::CString::new(format!("{} (debug layer indirect)", list_name(list.id)))
            .unwrap();
        node.pCommandListDebugNameW = windows::core::PCWSTR::null();
        node.pCommandListDebugNameA = note.as_ptr().cast();
        assert!(
            list.is_node(&node),
            "found by its ANSI name with the layer's note"
        );
        let longer =
            std::ffi::CString::new(format!("{}0 (debug layer indirect)", list_name(list.id)))
                .unwrap();
        node.pCommandListDebugNameA = longer.as_ptr().cast();
        assert!(!list.is_node(&node), "a number that only starts with ours is no match");
    }
}

/// A host command list and its allocator, always used together.
pub(crate) struct HostList {
    allocator: ID3D12CommandAllocator,
    pub(crate) list: ID3D12GraphicsCommandList,
    pub(crate) depth_bounds: Option<ID3D12GraphicsCommandList1>,
    /// Work for submission time, in recorded order.
    deferred: Mutex<Deferred>,
    /// Upload memory for data the list's draws make (expanded vertices),
    /// reused after a reset.
    scratch: Mutex<Scratch>,
    /// Host texture states the list assumes, by host copy: at its start and
    /// at its end. Lists are recorded on many threads and run in submission
    /// order, so the transitions from a texture's real state are made at
    /// submission, in `prelude`. Holding the copies keeps every texture the
    /// list names alive until it is reset, even if the title releases the
    /// resource first.
    states: Mutex<States>,
    /// Depth textures with a writable twin the list uses (`crate::twin`),
    /// by host copy: the side its first use needs and the side it leaves
    /// newer.
    twins: Mutex<std::collections::HashMap<usize, crate::twin::Use>>,
    /// A copy of ours replaced the list's compute root signature and
    /// pipeline: draws and dispatches set theirs again.
    rebind: std::sync::atomic::AtomicBool,
    /// A list of ours (uploads), whose uses are no use of the title's
    /// (`crate::resource::evict`).
    ours: std::sync::atomic::AtomicBool,
    /// The draw or dispatch being recorded runs a program using title
    /// memory ([`HostList::writer_uses_memory`]).
    memory_writer: std::sync::atomic::AtomicBool,
    /// Shaders of the list mark in the views' written-pages and read-marks
    /// bitmaps ([`HostList::marking`]).
    marks: std::sync::atomic::AtomicBool,
    /// A program of the list may write through the bound heap's views
    /// (`heap_writes` of its translation): its write marks name pages no
    /// declared range does ([`HostList::heavy_reasons`]).
    heap_writes: std::sync::atomic::AtomicBool,
    /// The list that runs those transitions just before this one.
    prelude: (ID3D12CommandAllocator, ID3D12GraphicsCommandList),
    /// The pipelines the list binds, for reports of slow submissions.
    pipelines: Mutex<std::collections::BTreeSet<String>>,
    /// Which list this is, for the name that tells it apart in device-removed
    /// data (see [`HostList::is_node`]).
    id: u64,
    /// The work its indirect executions may still do (`indirect`): a
    /// shader-writable word, zeroed when the list is submitted.
    budget: OnceLock<Option<(ID3D12Resource, usize)>>,
    /// Clamped indirect arguments (`crate::indirect`), reused by each
    /// execution in turn; replaced by a bigger one when too small.
    arguments: Mutex<Option<ID3D12Resource>>,
    /// Indirect argument clamps hoisted from this list's initial non-writing
    /// prefix into its submission prelude (`crate::indirect::HoistedClamps`).
    hoisted_indirect: Mutex<crate::indirect::HoistedClamps>,
    /// The window of indirect draws clamped together (`crate::indirect`,
    /// `VICTORY_CLAMP_WINDOW`).
    clamp_window: Mutex<crate::indirect::Window>,
    /// The texts of the markers `note_op` set, in order (for device-removed
    /// data without marker texts), with the host GPU range a copy reads.
    #[cfg(feature = "diagnostics")]
    notes: Mutex<Vec<Note>>,
    /// In a frame `crate::capture` lists: the textures the list renders to
    /// or writes, and the pipeline that first wrote each (diagnostics).
    #[cfg(feature = "diagnostics")]
    targets: Mutex<Vec<(usize, String)>>,
    /// With `VICTORY_GPU_TIMING`: a GPU timestamp before each noted command
    /// and at the end, and where they are read back.
    #[cfg(feature = "diagnostics")]
    stamps: OnceLock<Option<(ID3D12QueryHeap, ID3D12Resource)>>,
}

// SAFETY: D3D12 objects are free-threaded; a host list is recorded by the
// title list's one recording thread, then run by one submission at a time
// (the queues hold it between, `crate::hold`).
unsafe impl Send for HostList {}
// SAFETY: as above.
unsafe impl Sync for HostList {}

pub(crate) fn warn_depth_bounds() {
    static ONCE: std::sync::Once = std::sync::Once::new();
    ONCE.call_once(|| tracing::warn!("the host GPU has no depth bounds test; depth bounds are ignored"));
}

/// Timestamps per list (diagnostics); commands past them are not timed.
#[cfg(feature = "diagnostics")]
const STAMPS: u32 = 4096;

/// A marker's text and the host range its command reads (diagnostics).
#[cfg(feature = "diagnostics")]
type Note = (String, Option<(u64, u64)>);

type States = std::collections::HashMap<
    usize,
    (u32, u32, std::sync::Arc<crate::resource::HostTexture>),
>;

/// Whether device-removed data is on (`VICTORY_D3D12_DRED`, diagnostics
/// builds).
pub(crate) fn dred_on() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| cfg!(feature = "diagnostics") && crate::switch::on("VICTORY_D3D12_DRED"))
}

/// Whether each list of a submission runs and is timed on its own, slow
/// ones logged with their commands (`VICTORY_GPU_TIMING`, diagnostics
/// builds; the commands are noted with `VICTORY_D3D12_DRED` on).
#[cfg(feature = "diagnostics")]
fn timing() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| crate::switch::on("VICTORY_GPU_TIMING"))
}

/// Names a host object for the device-removed report, which names the
/// allocations a page fault hit (only with device-removed data on).
pub(crate) fn name(object: &impl Interface, text: impl FnOnce() -> String) {
    if dred_on()
        && let Ok(object) = object.cast::<ID3D12Object>()
    {
        // SAFETY: a plain call on a live object.
        let _ = unsafe { object.SetName(&windows::core::HSTRING::from(text())) };
    }
}

/// The id of the next host list (see [`HostList::is_node`]).
static NEXT_LIST: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(1);

/// The name a host list gets with device-removed data on.
fn list_name(id: u64) -> String {
    format!("victory host list {id}")
}

/// The pointers one of our list objects answers to: its own and the one each
/// interface a runtime may record it by gives (COM identity is the
/// `IUnknown` one). Only our live object is queried: a node of the
/// device-removed data may name a list that is long released, so its pointer
/// is compared and never dereferenced.
fn identities(list: &impl Interface) -> Vec<usize> {
    let mut found = vec![list.as_raw() as usize];
    let mut add = |pointer: Option<*mut core::ffi::c_void>| {
        if let Some(pointer) = pointer {
            found.push(pointer as usize);
        }
    };
    add(list
        .cast::<windows::core::IUnknown>()
        .ok()
        .map(|i| i.as_raw()));
    add(list.cast::<ID3D12Object>().ok().map(|i| i.as_raw()));
    add(list.cast::<ID3D12DeviceChild>().ok().map(|i| i.as_raw()));
    add(list.cast::<ID3D12CommandList>().ok().map(|i| i.as_raw()));
    found.sort_unstable();
    found.dedup();
    found
}

/// Whether a node of the device-removed data is a list: by pointer, or by the
/// name the runtime copied when the list was submitted. With the debug layer
/// on, the object we hold is its wrapper and the runtime records another
/// one, so no pointer of ours matches, and it appends its own note to our
/// name ("victory host list 158 (debug layer indirect)"). The name matches
/// exactly or followed by " (": never as the start of another number.
fn node_is_list(
    node_pointer: Option<usize>,
    node_name: Option<&str>,
    pointers: &[usize],
    name: &str,
) -> bool {
    node_pointer.is_some_and(|p| pointers.contains(&p))
        || node_name.is_some_and(|n| {
            n == name
                || n.strip_prefix(name)
                    .is_some_and(|rest| rest.starts_with(" ("))
        })
}

/// Upload chunks handed out front to back: the chunks, the one in use and
/// the bytes used in it.
#[derive(Default)]
struct Scratch {
    chunks: Vec<ScratchChunk>,
    current: usize,
    used: u64,
}

// SAFETY: the chunk pointers are persistent mappings of upload buffers.
unsafe impl Send for Scratch {}

const SCRATCH_CHUNK: u64 = 4 << 20;
const FIRST_SCRATCH_CHUNK: u64 = 64 << 10;

struct ScratchChunk {
    _buffer: ID3D12Resource,
    mapped: usize,
    gpu: u64,
    capacity: u64,
}

/// One piece of submission-time work, in recorded order.
enum Step {
    /// A copy into title memory (destination, source, bytes): the title's
    /// buffer copies, which later refreshes must see.
    Copy(usize, usize, usize),
    /// A copy of title memory into the list's own upload memory.
    Fill(usize, usize, usize),
    /// Brings a range (address, bytes) of a buffer's host copy up to date:
    /// once per submission, whichever lists ask, until a `Copy` or
    /// `CopyMemory` intervenes.
    Refresh(std::sync::Arc<crate::shadow::Shadow>, usize, usize),
    /// A copy between title memory addresses (destination, source, bytes),
    /// made only where both are committed (`CopyMemoryToMemoryX`).
    CopyMemory(usize, usize, usize),
    /// Work that writes host memory only.
    Host(Box<dyn Fn() + Send>),
    /// Shaders write a range (address, bytes) of a buffer's host copy;
    /// whether they mark the pages they write (no snapshot then), and
    /// whether their program uses title memory (then the range is withheld
    /// when such a list misses, `crate::memory::withholding`).
    GpuWrite(std::sync::Arc<crate::shadow::Shadow>, usize, usize, bool, bool),
    /// Brings a texture's staging buffer up to date: once per submission,
    /// whichever lists sample it, until a `Copy` or `CopyMemory` intervenes.
    Texture(std::sync::Arc<crate::texture::Staging>, std::sync::Arc<crate::resource::HostTexture>),
    /// Brings every buffer range a view heap's views reach up to date
    /// (`crate::bindless`).
    Heap(std::sync::Arc<crate::bindless::Viewed>),
    /// Shaders read title memory through a view heap's page table (the
    /// heap's key, `crate::memory`): the table names every copy, each
    /// brought up to date.
    Memory(usize),
}

type VramRead = (ID3D12Resource, u64, usize, u64);
type VramProbe = Box<dyn Fn(&mut Vec<VramRead>) + Send>;

/// A direct list of ours copying between copies in video memory and system
/// memory around a submission's own lists (`Host::run`), with its
/// allocator: one per submission in flight, back in [`COPY_LISTS`] once it
/// has completed.
struct CopyList {
    alloc: ID3D12CommandAllocator,
    list: ID3D12GraphicsCommandList,
}

/// Copy lists whose submission has completed.
static COPY_LISTS: Mutex<Vec<CopyList>> = Mutex::new(Vec::new());

/// A mapped buffer that a submission's copies in video memory take their
/// bytes from (upload) or put them in (read-back): one per submission and
/// direction, back in [`STAGING`] once it has completed.
struct Staging {
    buffer: ID3D12Resource,
    data: usize,
    size: u64,
}

/// Staging buffers whose submission has completed: uploads, read-backs.
static STAGING: Mutex<[Vec<Staging>; 2]> = Mutex::new([Vec::new(), Vec::new()]);

/// Bytes of one submission's uploads or read-backs that go through its own
/// staging: past them, a burst (copies made in video memory are filled
/// whole) goes through #3312's ring, waited for.
const STAGED_MAX: u64 = 256 << 20;

/// Bytes of staging kept in [`STAGING`] per direction; more is let go.
const STAGING_KEPT: u64 = 512 << 20;

/// What a submission's copies in video memory need once it has run: the
/// copy lists run before and after its own lists and their staging, held
/// until then, and what the list after read back into staging, which goes
/// into the copies' system-memory mirrors first thing on completion
/// (`Host::finish`).
#[derive(Default)]
struct VramSync {
    lists: Vec<CopyList>,
    /// `(readback, buffer)`.
    staging: Vec<(bool, Staging)>,
    /// Read back: from (in read-back staging), to (a mirror), bytes.
    reads: Vec<(usize, usize, usize)>,
    /// Ranges past [`STAGED_MAX`]: read back on completion, waited for.
    later: Vec<VramRead>,
    /// Pages read back by the list after the submission because the last
    /// completion of a submission like it needed them ([`PREDICTED`]): where
    /// they are in read-back staging, by (buffer, offset, bytes).
    predicted: std::collections::HashMap<(usize, u64, u64), usize>,
    /// The submission's kind ([`signature`]), if completing it takes marks
    /// or page bitmaps: its named pages become the next prediction for it.
    signature: Option<u64>,
    /// Where the list after it put its shaders' write marks in read-back
    /// staging (`crate::shadow::capture_write_marks`), and the bytes of
    /// each: its own marks, whatever ran before or after it.
    marks: Option<([Option<u64>; 2], u64, usize)>,
}

/// A range `(copy, start, end, marked)` (title addresses) a submission's
/// shaders declare as written through a buffer copy (`Step::GpuWrite`).
type Declared = (usize, usize, usize, bool);

/// Whether a submission declaring `next` must wait for one in flight that
/// declared `flying` before it is prepared: they write overlapping ranges
/// of one copy, and not both by marks. An unmarked range's snapshot is taken
/// from the copy when the submission is prepared, so it would miss what the
/// one in flight writes there (and copy it back again over later title
/// writes); marked ranges keep no snapshot (`crate::shadow::Shadow::writing`).
fn declared_overlap(next: &[Declared], flying: &[Declared]) -> bool {
    next.iter().any(|&(copy, a, e, marked)| {
        flying
            .iter()
            .any(|&(c, x, y, m)| c == copy && a < y && x < e && !(marked && m))
    })
}

/// [`HostList::heavy_reasons`] bits that keep a submission's completion
/// before the next one is prepared whatever it touches: writes through the
/// heap's views (declared nowhere), work after it that names nothing it
/// writes, missed-memory checks, title memory through page tables
/// (anywhere). The others (bottom-of-pipe writes, declared GPU writes) are
/// bounded and known at submission: later submissions wait for it only
/// where they meet them (`Host::submit`, [`Settled`]).
const BARRIER_REASONS: u8 = 1 | 1 << 1 | 1 << 3 | 1 << 5;

/// [`HostList::heavy_reasons`] bit of work after it that names the title
/// memory it writes ([`HostList::defer_after_writing`]): registered in
/// flight like declared writes, a barrier only without
/// `VICTORY_EARLY_BARRIERS` (`crate::early::barriers`).
const AFTER_NAMED: u8 = 1 << 7;

/// Whether a submission with heavy reasons `reasons` that marks
/// (`marks`) completes before the next is prepared: for a barrier reason,
/// or marks its shaders set where each submission's are not captured apart
/// (`crate::shadow::captures_write_marks`: they are taken from the one
/// bitmap with nothing in flight after it). `relaxed`: with
/// `VICTORY_EARLY_BARRIERS`.
fn barrier(reasons: u8, marks: bool, captured: bool, relaxed: bool) -> bool {
    let named = if relaxed { 0 } else { AFTER_NAMED };
    reasons & (BARRIER_REASONS | named) != 0 || marks && !captured
}

/// Whether a submission completes every one in flight before it is
/// prepared: a barrier, but with `VICTORY_EARLY_BARRIERS` (`relaxed`) only
/// one whose marks are not its own (taken from the one bitmap, with nothing
/// else in flight). Its preparing reads what those in flight write back
/// only where it settles them, as any submission's ([`Settled`]); and they
/// complete before it, in order (`Host::finish`). What it writes unnamed
/// keeps the next submission waiting for it ([`barrier`]).
fn alone(reasons: u8, marks: bool, captured: bool, relaxed: bool) -> bool {
    if relaxed { marks && !captured } else { barrier(reasons, marks, captured, false) }
}

/// Why a submission being prepared completes one in flight first
/// (`Host::run`): what it is about to read is title memory that one writes
/// when it completes.
#[derive(Clone, Copy, Debug)]
pub(crate) enum Settled {
    /// A refresh would copy pages the title changed under that one's
    /// declared ranges of the same copy (or memory copied whole there).
    Same,
    /// A refresh of another copy of memory it writes.
    Alias,
    /// A refresh of memory its bottom-of-pipe writes go to.
    Label,
    /// A title memory read at submission (`Step::Fill`).
    Fill,
    /// A texture over memory it writes (`crate::bindless::uploads`).
    Texture,
    /// A title memory copy at submission (`Step::Copy`, `Step::CopyMemory`)
    /// whose source or destination it writes: on the GPU's timeline the
    /// copy comes after it, so it reads and overwrites its results.
    Copy,
}

/// Tests: the `settle` of a preparing with nothing in flight, never asked.
#[cfg(test)]
pub(crate) fn nothing_in_flight(owner: u64, why: Settled) {
    panic!("nothing is in flight to complete (submission {owner}, {why:?})");
}

/// The pages of copies in video memory that the last completion of each
/// kind of submission taking the views' write marks or the lists' page
/// bitmaps had to read back (`Host::vram_completed`), by [`signature`]: the
/// next one of that kind reads them back in its own list after it, before
/// its fence value, so that completing it waits only for pages it did not
/// name. A pass marks mostly the same pages each frame; the passes of a
/// frame mark different ones (async2-u1: the last completion's pages were
/// 39 of the ~2,800 a frame's 19 completions named).
static PREDICTED: Mutex<std::collections::BTreeMap<u64, Vec<VramRead>>> =
    Mutex::new(std::collections::BTreeMap::new());

/// Kinds of submission kept in [`PREDICTED`] at most.
const PREDICTED_KINDS: usize = 256;

/// A submission's kind, the same from frame to frame: the pipelines each
/// of its lists bound, in order.
fn signature(runs: &[&HostList]) -> u64 {
    use std::hash::{Hash, Hasher};
    let mut h = std::collections::hash_map::DefaultHasher::new();
    for list in runs {
        list.pipelines.lock().unwrap_or_else(|e| e.into_inner()).hash(&mut h);
    }
    h.finish()
}

/// Puts `len` read-back bytes at `from` into a copy's mirror at `to`, unless
/// it holds them already: then a whole read (`whole`, not a piece of one)
/// is noted in `unchanged` by `to`, and its page's copy-back needs no look
/// at title memory ([`crate::shadow::copy_back_marks`]).
///
/// # Safety
/// `from` and `to` are valid for `len` bytes, and do not overlap.
unsafe fn into_mirror(
    from: *const u8,
    to: usize,
    len: usize,
    whole: bool,
    unchanged: &mut std::collections::HashSet<usize>,
) {
    // SAFETY: the caller's.
    let (new, old) = unsafe { (std::slice::from_raw_parts(from, len), std::slice::from_raw_parts_mut(to as *mut u8, len)) };
    if new == &*old {
        if whole {
            unchanged.insert(to);
        }
    } else {
        old.copy_from_slice(new);
    }
}

/// How many of `lens` (in order) are staged at 16-byte offsets within
/// `max` bytes, and those bytes.
fn staged(lens: impl IntoIterator<Item = u64>, max: u64) -> (usize, u64) {
    let (mut n, mut at) = (0, 0u64);
    for len in lens {
        let end = at + len;
        if end > max {
            break;
        }
        at = end.next_multiple_of(16);
        n += 1;
    }
    (n, at)
}

/// Of staging buffers of `sizes`, the smallest that holds `bytes`.
fn smallest_fitting(sizes: impl IntoIterator<Item = u64>, bytes: u64) -> Option<usize> {
    sizes
        .into_iter()
        .enumerate()
        .filter(|&(_, size)| size >= bytes)
        .min_by_key(|&(_, size)| size)
        .map(|(i, _)| i)
}

/// What a list does on the CPU when it is submitted: copies between host
/// memory regions (title buffers, upload buffers and buffer copies), and
/// the upload buffers its GPU commands read, kept until the list is reset.
#[derive(Default)]
struct Deferred {
    /// Destination, source, bytes.
    copies: Vec<Step>,
    uploads: Vec<ID3D12Resource>,
    /// Host objects the list's commands name (pipelines, root signatures,
    /// descriptor heaps): the title may release its objects before it
    /// submits the list.
    kept: Vec<windows::core::IUnknown>,
    /// Buffer ranges refreshed since the last title copy (host copy,
    /// address, bytes): refreshing them again would find the same bytes.
    refreshed: std::collections::HashSet<(usize, usize, usize)>,
    /// Textures whose contents the list brings up to date.
    textures: std::collections::HashSet<usize>,
    /// Bottom-of-pipe writes, done once the list has run: address, what,
    /// bytes.
    writes: Vec<(usize, crate::queue::Written, usize)>,
    /// The memory values the list waits for before it starts (a list the
    /// title records with a wait after commands is cut there,
    /// `crate::list`).
    waits: Vec<crate::hold::ValueWait>,
    /// Whether a step zeroes the indirect-work budget.
    budget_zeroed: bool,
    /// Work done once the list has run, before the bottom-of-pipe writes
    /// (title memory shaders wrote at addresses they computed goes back).
    after: Vec<Box<dyn Fn() + Send>>,
    /// Some of `after` writes title memory it does not name
    /// ([`HostList::defer_after`]).
    after_unnamed: bool,
    /// The title memory `[start, end)` the rest of `after` writes
    /// ([`HostList::defer_after_writing`]).
    after_writes: Vec<(usize, usize)>,
    /// Asked once the list has run, before anything goes back: whether its
    /// shaders read title memory with no copy (`crate::memory`); then what
    /// they wrote through views does not go back either.
    missed: Vec<Box<dyn Fn() -> bool + Send>>,
    /// Kept until the list is reset ([`HostList::keep_until_reset`]).
    held: Vec<Box<dyn std::any::Any + Send>>,
    /// Per-list VRAM/RAM dirty bitmaps (`crate::memory::bind`) to zero before
    /// execution and read back after.
    vram_bitmaps: Vec<(ID3D12Resource, ID3D12Resource)>,
    /// Probes run after bitmap readbacks to collect any VRAM-backed `memory`
    /// shadow pages that need CPU readback before `after` callbacks run.
    vram_probes: Vec<VramProbe>,
}

/// [`HostList::heavy_reasons`] of a list's deferred work, its shaders marking
/// through views (`marks`) and possibly writing through the heap's
/// (`heap_writes`).
fn reasons_of(d: &Deferred, marks: bool, heap_writes: bool) -> u8 {
    let heap = marks && d.copies.iter().any(|s| matches!(s, Step::Heap(_)));
    u8::from(heap && heap_writes)
        | u8::from(d.after_unnamed) << 1
        | u8::from(!d.writes.is_empty()) << 2
        | u8::from(!d.missed.is_empty()) << 3
        | u8::from(d.copies.iter().any(|s| matches!(s, Step::GpuWrite(..)))) << 4
        | u8::from(d.copies.iter().any(|s| matches!(s, Step::Memory(_)))) << 5
        | u8::from(heap && !heap_writes) << 6
        | if d.after_writes.is_empty() { 0 } else { AFTER_NAMED }
}

/// [`HostList::flying`] of a list's deferred work.
fn flying_of(d: &Deferred) -> Vec<(usize, usize, usize)> {
    let declared = d.copies.iter().filter_map(|s| match s {
        Step::GpuWrite(shadow, address, len, marked, _) => {
            let (a, n) = if *marked {
                let (a, n) = shadow.page_range(*address as u64, *len as u64);
                (a as usize, n as usize)
            } else {
                (*address, *len)
            };
            Some((a, a + n, std::sync::Arc::as_ptr(shadow) as usize))
        }
        _ => None,
    });
    let labels = d.writes.iter().map(|&(at, _, bytes)| (at, at + bytes, crate::shadow::LABEL));
    let after = d.after_writes.iter().map(|&(a, e)| (a, e, crate::shadow::AFTER));
    declared.chain(labels).chain(after).collect()
}

/// The submission lock's state.
#[derive(Default)]
struct Submitted {
    /// The last value signalled on the host fence.
    value: u64,
    /// Submissions the host GPU may still be running, oldest first: each
    /// completes ([`Host::finish`]) once it has run, in order.
    flight: std::collections::VecDeque<InFlight>,
    /// Submissions prepared so far (the last one's [`InFlight::owner`]).
    owners: u64,
}

/// A submission on the host GPU, not completed yet: what completing it
/// needs.
struct InFlight {
    /// The host fence value signalled after it.
    value: u64,
    /// Its number among submissions, from 1 in the order they are prepared
    /// (`crate::shadow`'s per-submission state names it).
    owner: u64,
    /// The title's queue that submitted it (0: ours).
    queue: usize,
    /// The title's host lists it runs, held until it has completed (a title
    /// list reset meanwhile records into others, `crate::list::restart`),
    /// then recycled.
    lists: Vec<std::sync::Arc<HostList>>,
    /// It completes before the next submission is prepared ([`barrier`]).
    barrier: bool,
    /// The ranges its shaders declare as written ([`declared_overlap`]).
    declared: Vec<Declared>,
    /// It runs the upload list (`crate::bindless::uploads`), which the next
    /// submission with uploads records again.
    uploads: bool,
    /// Its shaders mark in the views' bitmaps: its completion takes the
    /// read marks, and the write marks if they are its own (captured, or it
    /// is a barrier or waited for).
    marks: bool,
    /// It is waited for at once (`Host::execute`): nothing runs after it.
    sync: bool,
    /// When it was handed to the host GPU.
    prepared: Instant,
    /// What its preparing used, released after it.
    batch: Batch,
    heaps: Vec<std::sync::Arc<crate::bindless::Viewed>>,
    /// Title fences signalled behind it: set once it has completed.
    signals: Vec<(usize, u64)>,
    /// Host objects given up before it was submitted ([`retire`]).
    retired: Vec<Box<dyn std::any::Any + Send>>,
    /// What its copies in video memory need once it has run: the copy lists
    /// run around its own, and what they read back ([`VramSync`]).
    vram: VramSync,
    /// A frame `crate::capture` lists.
    #[cfg(feature = "diagnostics")]
    traced: bool,
}

/// Tests: the gate the next submission that may stay in flight waits at
/// ([`Host::hold_gpu`]).
#[cfg(test)]
static GATE: Mutex<Option<ID3D12Fence>> = Mutex::new(None);

/// Whether the title's submissions may complete after they return
/// ([`Host::submit`]): with `VICTORY_GPU_OVERLAP=1` (off by default: each
/// is waited for). Tests run with it on unless it is set to 0: theirs are
/// the paths it adds, and our own lists (`Host::execute`) take the other.
fn overlap_switch() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| {
        let on = if cfg!(test) {
            std::env::var_os("VICTORY_GPU_OVERLAP").is_none_or(|v| v != "0")
        } else {
            crate::switch::on("VICTORY_GPU_OVERLAP")
        };
        if on {
            tracing::info!(
                "the title's submissions complete after they return (VICTORY_GPU_OVERLAP)"
            );
        }
        on
    })
}

/// Whether the title's submissions complete after they return now
/// ([`overlap_switch`]): not while diagnostics time each list or list a
/// frame, which read their results at once.
pub(crate) fn overlap() -> bool {
    let on = overlap_switch();
    #[cfg(feature = "diagnostics")]
    return on && !timing() && !crate::capture::listed(crate::queue::presented() + 1);
    #[cfg(not(feature = "diagnostics"))]
    on
}

/// Waits until the host GPU has run what was signalled on `fence` up to
/// `value`, with an event of the calling thread's (no lock held).
fn wait_unlocked(fence: &ID3D12Fence, value: u64) {
    struct Event(HANDLE);
    impl Drop for Event {
        fn drop(&mut self) {
            // SAFETY: our event.
            let _ = unsafe { windows::Win32::Foundation::CloseHandle(self.0) };
        }
    }
    thread_local! {
        // SAFETY: a plain auto-reset event, the thread's until it ends.
        static EVENT: Option<Event> = unsafe { CreateEventW(None, false, false, None) }.ok().map(Event);
    }
    // SAFETY: our fence; the thread's event.
    unsafe {
        if fence.GetCompletedValue() >= value {
            return;
        }
        EVENT.with(|event| match event {
            Some(Event(event)) if fence.SetEventOnCompletion(value, *event).is_ok() => {
                WaitForSingleObject(*event, INFINITE);
            }
            _ => tracing::warn!(value, "host submissions not waited for"),
        });
    }
}

/// A submission completed since held work was last pumped for it
/// ([`pump_after_completions`]).
static COMPLETED: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

/// After submissions completed on a thread other than the completion
/// thread (which pumps after its own), once the submission lock is let go:
/// held work may wait for a value they wrote or a fence they signalled
/// (`crate::queue`). Inside a pump the lock is taken and nothing runs here:
/// that pump tests its queues again itself.
fn pump_after_completions() {
    if COMPLETED.swap(false, Ordering::AcqRel) {
        crate::queue::pump_held_if_free(crate::queue::Why::Completed);
    }
}

/// Sets a title fence a queue signalled behind its work, which kept it
/// (`crate::fence::keep`), and lets it go.
fn set_title_fence(fence: usize, value: u64) {
    crate::fence::set(fence as crate::object::This, value);
    crate::fence::let_go(fence as crate::object::This);
}

impl Host {
    /// Whether the host runs mesh shaders at shader model 6.6 (translated
    /// mesh shaders need both); logged once.
    pub(crate) fn mesh_shaders(&self) -> bool {
        static MESH: OnceLock<bool> = OnceLock::new();
        *MESH.get_or_init(|| {
            let mut options = D3D12_FEATURE_DATA_D3D12_OPTIONS7::default();
            let mut model = D3D12_FEATURE_DATA_SHADER_MODEL {
                HighestShaderModel: D3D_SHADER_MODEL_6_6,
            };
            // SAFETY: each query writes its own structure of the given size.
            let ok = unsafe {
                self.device
                    .CheckFeatureSupport(
                        D3D12_FEATURE_D3D12_OPTIONS7,
                        (&raw mut options).cast(),
                        size_of_val(&options) as u32,
                    )
                    .is_ok()
                    && self
                        .device
                        .CheckFeatureSupport(
                            D3D12_FEATURE_SHADER_MODEL,
                            (&raw mut model).cast(),
                            size_of_val(&model) as u32,
                        )
                        .is_ok()
            };
            let mesh = ok
                && options.MeshShaderTier != D3D12_MESH_SHADER_TIER_NOT_SUPPORTED
                && model.HighestShaderModel.0 >= D3D_SHADER_MODEL_6_6.0;
            tracing::info!(mesh, "host mesh shader support");
            mesh
        })
    }

    /// The wave sizes the host runs (D3D12 options 1), which translated
    /// compute shaders that declare the console's wave size must fit;
    /// logged once. Unknown (the query fails): any.
    pub(crate) fn target(&self) -> victory_shader::emit::Target {
        static TARGET: OnceLock<victory_shader::emit::Target> = OnceLock::new();
        *TARGET.get_or_init(|| {
            let mut options = D3D12_FEATURE_DATA_D3D12_OPTIONS1::default();
            // SAFETY: the query writes the structure of the given size.
            let ok = unsafe {
                self.device
                    .CheckFeatureSupport(
                        D3D12_FEATURE_D3D12_OPTIONS1,
                        (&raw mut options).cast(),
                        size_of_val(&options) as u32,
                    )
                    .is_ok()
            };
            let target = if ok && options.WaveOps.as_bool() && options.WaveLaneCountMin > 0 {
                victory_shader::emit::Target {
                    lanes_min: options.WaveLaneCountMin,
                    lanes_max: options.WaveLaneCountMax,
                }
            } else {
                victory_shader::emit::Target::ANY
            };
            tracing::info!(
                lanes_min = target.lanes_min,
                lanes_max = target.lanes_max,
                "host wave sizes"
            );
            if !target.runs(64) {
                tracing::warn!(
                    "the host has no 64-lane waves: console wave64 compute programs that need \
                     their whole wave run as two 32-lane waves where there is a form for \
                     them; the rest do not translate, and their dispatches are dropped"
                );
            }
            target
        })
    }

    /// Whether the host runs 64-bit atomics on typed resources in the
    /// descriptor heap (translated 64-bit image atomics); logged once.
    pub(crate) fn heap_atomics64(&self) -> bool {
        static ATOMICS: OnceLock<bool> = OnceLock::new();
        *ATOMICS.get_or_init(|| {
            let mut typed = D3D12_FEATURE_DATA_D3D12_OPTIONS9::default();
            let mut heap = D3D12_FEATURE_DATA_D3D12_OPTIONS11::default();
            // SAFETY: each query writes its own structure of the given size.
            let ok = unsafe {
                self.device
                    .CheckFeatureSupport(
                        D3D12_FEATURE_D3D12_OPTIONS9,
                        (&raw mut typed).cast(),
                        size_of_val(&typed) as u32,
                    )
                    .is_ok()
                    && self
                        .device
                        .CheckFeatureSupport(
                            D3D12_FEATURE_D3D12_OPTIONS11,
                            (&raw mut heap).cast(),
                            size_of_val(&heap) as u32,
                        )
                        .is_ok()
            };
            let atomics = ok
                && typed.AtomicInt64OnTypedResourceSupported.as_bool()
                && heap.AtomicInt64OnDescriptorHeapResourceSupported.as_bool();
            tracing::info!(atomics, "host 64-bit atomics on typed heap resources");
            atomics
        })
    }

    /// A new list, open for recording.
    pub(crate) fn new_list(&self) -> windows::core::Result<HostList> {
        // SAFETY: plain object creation.
        unsafe {
            let allocator: ID3D12CommandAllocator = self
                .device
                .CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT)?;
            let list: ID3D12GraphicsCommandList = self.device.CreateCommandList(
                0,
                D3D12_COMMAND_LIST_TYPE_DIRECT,
                &allocator,
                None,
            )?;
            let prelude_allocator: ID3D12CommandAllocator = self
                .device
                .CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT)?;
            let prelude: ID3D12GraphicsCommandList = self.device.CreateCommandList(
                0,
                D3D12_COMMAND_LIST_TYPE_DIRECT,
                &prelude_allocator,
                None,
            )?;
            prelude.Close()?;
            let id = NEXT_LIST.fetch_add(1, Ordering::Relaxed);
            name(&list, || list_name(id));
            Ok(HostList {
                allocator,
                depth_bounds: self.depth_bounds_supported.then(|| list.cast()).transpose()?,
                list,
                deferred: Mutex::default(),
                scratch: Mutex::default(),
                states: Mutex::default(),
                twins: Mutex::default(),
                rebind: false.into(),
                ours: false.into(),
                memory_writer: false.into(),
                marks: false.into(),
                heap_writes: false.into(),
                prelude: (prelude_allocator, prelude),
                pipelines: Mutex::default(),
                id,
                budget: OnceLock::new(),
                arguments: Mutex::new(None),
                hoisted_indirect: Mutex::default(),
                clamp_window: Mutex::default(),
                #[cfg(feature = "diagnostics")]
                notes: Mutex::default(),
                #[cfg(feature = "diagnostics")]
                targets: Mutex::default(),
                #[cfg(feature = "diagnostics")]
                stamps: OnceLock::new(),
            })
        }
    }

    /// The submission lock: one submission at a time is prepared, run and
    /// completed, in order.
    fn lock_submitted(&self) -> std::sync::MutexGuard<'_, Submitted> {
        self.submitted.lock().unwrap_or_else(|e| e.into_inner())
    }

    /// Runs closed lists and waits for them, and for everything submitted
    /// before them (our own lists, and tests).
    pub(crate) fn execute(&self, runs: &[&HostList]) {
        let mut submitted = self.complete_while(self.lock_submitted(), |_| true);
        let record = self.run(&mut submitted, runs, false);
        #[cfg(feature = "diagnostics")]
        let (traced, sub) = (record.traced, record.value);
        self.finish(record, runs, true);
        drop(submitted);
        pump_after_completions();
        // A listed frame: what this submission's lists drew, as it is now
        // (the dumps submit lists of their own).
        #[cfg(feature = "diagnostics")]
        if traced {
            let mut targets: Vec<(usize, String)> = Vec::new();
            for r in runs {
                for t in r.targets.lock().unwrap_or_else(|e| e.into_inner()).iter() {
                    if !targets.iter().any(|x| x.0 == t.0) {
                        targets.push(t.clone());
                    }
                }
            }
            crate::capture::dump_submission(sub, &targets);
        }
    }

    /// Runs the title's host lists (a queue's piece, `crate::queue`) and
    /// returns once the host GPU has them, as the console's
    /// ExecuteCommandLists does: they complete later ([`Host::finish`]),
    /// on the completion thread or before work that needs them done, and
    /// are recycled then (`crate::list::recycle`). Waited for at once with
    /// `VICTORY_GPU_OVERLAP=0` (and while diagnostics time lists or list a
    /// frame).
    pub(crate) fn submit(&self, queue: usize, lists: Vec<std::sync::Arc<HostList>>) {
        if !overlap() {
            let runs: Vec<&HostList> = lists.iter().map(|l| &**l).collect();
            self.execute(&runs);
            drop(runs);
            lists.into_iter().for_each(crate::list::recycle);
            return;
        }
        let runs: Vec<&HostList> = lists.iter().map(|l| &**l).collect();
        // What must complete before this submission is prepared:
        // - barriers ([`barrier`]): their completion writes title memory
        //   nobody declared, or takes marks shared with later submissions;
        // - all of them before a barrier (its checks and page tables see
        //   every earlier result);
        // - those declaring a range of a copy this one declares too, unless
        //   both mark ([`declared_overlap`]);
        // - the last upload list before another; a host list's last run
        //   before it runs again (its prelude and steps run again); and, once
        //   a frame, before the title's first submission, those whose shaders
        //   mark reads, for the eviction pass to clear the marks (not before a
        //   present's: it would wait for the frame it shows).
        // The rest stay in flight: what this one's preparing reads of what
        // they write, or overwrites (title copies at submission), is waited
        // for where it is touched ([`Settled`]): no GPU work reads title
        // memory itself, only copies taken when it was prepared.
        let uploading = runs.iter().any(|l| l.uploads_at_submission());
        let reasons = runs.iter().fold(0u8, |r, l| r | l.heavy_reasons());
        let marking = runs.iter().any(|l| l.marks.load(Ordering::Relaxed));
        let captured = crate::shadow::captures_write_marks();
        let relaxed = crate::early::barriers();
        let alone = alone(reasons, marking, captured, relaxed);
        // A barrier prepared with those in flight left there (counted).
        let waived = barrier(reasons, marking, captured, relaxed) && !alone;
        let declared: Vec<Declared> = runs.iter().flat_map(|l| l.declared()).collect();
        let why = std::cell::Cell::new(0u8);
        let mut submitted = self.complete_while(self.lock_submitted(), |r| {
            let bits = u8::from(r.barrier)
                | u8::from(alone) << 1
                | u8::from(uploading && r.uploads) << 2
                | u8::from(r.lists.iter().any(|held| lists.iter().any(|l| std::sync::Arc::ptr_eq(held, l)))) << 3
                | u8::from(queue != 0 && r.marks && crate::resource::eviction_due()) << 4
                | u8::from(declared_overlap(&declared, &r.declared)) << 5;
            why.set(why.get() | bits | u8::from(waived) << 6);
            bits != 0
        });
        #[cfg(feature = "diagnostics")]
        {
            use crate::copystat::{C, add};
            for (bit, c) in [
                C::WaitBarrier,
                C::WaitAlone,
                C::WaitUploads,
                C::WaitListAgain,
                C::WaitEviction,
                C::WaitDeclared,
                C::AloneWaived,
            ]
            .into_iter()
            .enumerate()
            {
                add(c, u64::from(why.get() >> bit & 1));
            }
        }
        let _ = why;
        let mut record = self.run(&mut submitted, &runs, true);
        drop(runs);
        (record.queue, record.lists) = (queue, lists);
        submitted.flight.push_back(record);
        drop(submitted);
        self.flown.notify_one();
        pump_after_completions();
        static COMPLETER: std::sync::Once = std::sync::Once::new();
        COMPLETER.call_once(|| {
            let Some(host) = host() else { return };
            let spawned = std::thread::Builder::new()
                .name("victory submissions".into())
                .spawn(move || host.complete_in_background());
            if let Err(e) = spawned {
                fail(format_args!("no thread to complete host submissions ({e})"));
            }
        });
    }

    /// A title fence's Signal behind the work `queue` submitted so far
    /// (`crate::queue`): set once the queue's last submission in flight has
    /// completed, or now if none is; the fence is let go then (it was kept,
    /// `crate::fence::keep`).
    pub(crate) fn signal_after(&self, queue: usize, fence: usize, value: u64) {
        let mut submitted = self.lock_submitted();
        if let Some(last) = submitted.flight.iter_mut().rfind(|r| r.queue == queue) {
            last.signals.push((fence, value));
            return;
        }
        drop(submitted);
        set_title_fence(fence, value);
    }

    /// Tests: the host GPU waits before the next submission that may stay
    /// in flight until the guard goes (a fence the queue waits for, queued
    /// with it under the lock, signalled from the CPU), or for 10 s at
    /// most, so that a test failing meanwhile holds no other.
    #[cfg(test)]
    pub(crate) fn hold_gpu(&self) -> impl Drop {
        struct Gate(ID3D12Fence);
        impl Drop for Gate {
            fn drop(&mut self) {
                GATE.lock().unwrap_or_else(|e| e.into_inner()).take();
                // SAFETY: our fence.
                let _ = unsafe { self.0.Signal(1) };
            }
        }
        // SAFETY: plain object creation.
        let gate: ID3D12Fence =
            unsafe { self.device.CreateFence(0, D3D12_FENCE_FLAG_NONE) }.expect("a gate fence");
        *GATE.lock().unwrap_or_else(|e| e.into_inner()) = Some(gate.clone());
        let late = gate.clone().into_raw() as usize;
        std::thread::spawn(move || {
            std::thread::sleep(std::time::Duration::from_secs(10));
            // SAFETY: the reference taken above, ours to release.
            let late = unsafe { ID3D12Fence::from_raw(late as *mut std::ffi::c_void) };
            // SAFETY: our fence.
            let _ = unsafe { late.Signal(1) };
        });
        Gate(gate)
    }

    /// Completes every submission in flight ([`Host::finish`]).
    pub(crate) fn complete_all(&self) {
        drop(self.complete_while(self.lock_submitted(), |_| true));
        pump_after_completions();
    }

    /// Completes the submissions in flight up to the last one that runs
    /// `list`: it may be reset then.
    fn complete_held(&self, list: &std::sync::Arc<HostList>) {
        drop(self.complete_while(self.lock_submitted(), |r| {
            r.lists.iter().any(|l| std::sync::Arc::ptr_eq(l, list))
        }));
        pump_after_completions();
    }

    /// Completes submissions in flight, oldest first, while one left
    /// matches `matches`: waits for the host GPU without the lock (the
    /// completion thread and other submitting threads go on), and returns
    /// it held, with none left matching.
    fn complete_while<'a>(
        &'a self,
        mut submitted: std::sync::MutexGuard<'a, Submitted>,
        matches: impl Fn(&InFlight) -> bool,
    ) -> std::sync::MutexGuard<'a, Submitted> {
        while let Some(last) = submitted
            .flight
            .iter()
            .rev()
            .find(|r| matches(r))
            .map(|r| r.value)
        {
            drop(submitted);
            let waiting = std::time::Instant::now();
            wait_unlocked(&self.fence, last);
            SUBMISSION_US[1].fetch_add(waiting.elapsed().as_micros() as u64, Ordering::Relaxed);
            submitted = self.lock_submitted();
            // SAFETY: a plain query.
            let reached = unsafe { self.fence.GetCompletedValue() };
            while submitted.flight.front().is_some_and(|r| r.value <= reached) {
                let record = submitted.flight.pop_front().expect("a first");
                let quiet = !submitted.flight.iter().any(|r| r.marks);
                self.complete(record, quiet);
            }
        }
        submitted
    }

    /// Waits for a submission in flight and completes it: its lists are
    /// recycled once it has.
    fn complete(&self, mut record: InFlight, quiet: bool) {
        let lists = std::mem::take(&mut record.lists);
        let runs: Vec<&HostList> = lists.iter().map(|l| &**l).collect();
        self.finish(record, &runs, quiet);
        drop(runs);
        lists.into_iter().for_each(crate::list::recycle);
    }

    /// The completion thread: completes each submission once the host GPU
    /// has run it, so that title fences signalled behind it reach their
    /// values (and the title's waits for them end) with no call of the
    /// title's, as on the console.
    fn complete_in_background(&self) {
        // SAFETY: a plain auto-reset event, ours for good.
        let event = match unsafe { CreateEventW(None, false, false, None) } {
            Ok(event) => event,
            Err(e) => fail(format_args!(
                "no event to complete host submissions with ({e})"
            )),
        };
        loop {
            let value = {
                let mut submitted = self.lock_submitted();
                loop {
                    if let Some(first) = submitted.flight.front() {
                        break first.value;
                    }
                    submitted = self
                        .flown
                        .wait(submitted)
                        .unwrap_or_else(|e| e.into_inner());
                }
            };
            // Without the lock: the title submits meanwhile.
            // SAFETY: our fence and event.
            unsafe {
                if self.fence.GetCompletedValue() < value
                    && self.fence.SetEventOnCompletion(value, event).is_ok()
                {
                    WaitForSingleObject(event, INFINITE);
                }
            }
            let mut submitted = self.lock_submitted();
            // SAFETY: a plain query.
            let reached = unsafe { self.fence.GetCompletedValue() };
            let mut completed = false;
            while submitted.flight.front().is_some_and(|r| r.value <= reached) {
                let record = submitted.flight.pop_front().expect("a first");
                let quiet = !submitted.flight.iter().any(|r| r.marks);
                self.complete(record, quiet);
                completed = true;
            }
            drop(submitted);
            // Held work may wait for what completed (fences, memory values).
            if completed {
                crate::queue::pump_held_if_free(crate::queue::Why::Completed);
            }
        }
    }

    /// Waits until the host GPU has run what was signalled up to `value`.
    /// Under the submission lock, which the event is used under.
    fn wait_for(&self, value: u64) {
        // SAFETY: our fence and event.
        unsafe {
            if self.fence.GetCompletedValue() >= value {
                return;
            }
            match self.fence.SetEventOnCompletion(value, self.event) {
                Ok(()) => {
                    WaitForSingleObject(self.event, INFINITE);
                }
                Err(e) => tracing::warn!(error = %e, "host submission not waited for"),
            }
        }
    }

    /// While a submission is prepared ([`Host::run`]): completes those in
    /// flight up to submission `owner`, whose completion writes title
    /// memory the preparing is about to read (`why`, counted).
    fn settle(&self, submitted: &mut Submitted, owner: u64, why: Settled) {
        #[cfg(feature = "diagnostics")]
        {
            use crate::copystat::{C, add};
            add(
                match why {
                    Settled::Same => C::SettleSame,
                    Settled::Alias => C::SettleAlias,
                    Settled::Label => C::SettleLabel,
                    Settled::Fill => C::SettleFill,
                    Settled::Texture => C::SettleTexture,
                    Settled::Copy => C::SettleCopy,
                },
                1,
            );
        }
        let _ = why;
        let waiting = std::time::Instant::now();
        while submitted.flight.front().is_some_and(|r| r.owner <= owner) {
            let record = submitted.flight.pop_front().expect("a first");
            let quiet = !submitted.flight.iter().any(|r| r.marks);
            self.complete(record, quiet);
        }
        #[cfg(feature = "diagnostics")]
        crate::copystat::add(crate::copystat::C::SettleUs, waiting.elapsed().as_micros() as u64);
        let _ = waiting;
    }

    /// Prepares closed lists (title memory brought into host copies,
    /// uploads, preludes) and runs them on the host GPU, without waiting
    /// (but with `VICTORY_GPU_TIMING`, which waits for each list): what
    /// completing them needs. `overlap`: submissions may stay in flight
    /// after this one. Those in flight that it needs done have completed
    /// (`Host::submit`); others may still be when it reads title memory
    /// they write, and complete then ([`Host::settle`]).
    fn run(&self, submitted: &mut Submitted, runs: &[&HostList], overlap: bool) -> InFlight {
        let started = std::time::Instant::now();
        #[cfg(feature = "diagnostics")]
        let faults = crate::copystat::page_faults();
        #[cfg(feature = "diagnostics")]
        crate::copystat::add(crate::copystat::C::Submissions, 1);
        // Host textures the title has not used for a while (or past the
        // card's budget) are given up first (kept alive until this
        // submission has run: one in flight may still read them).
        // Not while a submission that marks reads is in flight: the pass
        // clears the read marks, which the GPU's ORs may race (it waits for
        // the next submission; `Host::submit` completes those first once a
        // frame).
        if !submitted.flight.iter().any(|r| r.marks) {
            crate::resource::evict_unused();
        }
        // Descriptors the title copied into its heaps itself, before the
        // heaps' views are brought up to date below.
        #[cfg(feature = "diagnostics")]
        let syncing = crate::watch::Timed(std::time::Instant::now(), crate::copystat::C::SyncHeapsUs);
        crate::descriptor::sync_heaps();
        #[cfg(feature = "diagnostics")]
        drop(syncing);
        // Textures heap-indexing shaders may read, uploaded first by a list
        // of ours where their title memory changed.
        let heaps: Vec<std::sync::Arc<crate::bindless::Viewed>> = runs
            .iter()
            .flat_map(|l| {
                l.deferred()
                    .copies
                    .iter()
                    .filter_map(|s| match s {
                        Step::Heap(v) => Some(v.clone()),
                        _ => None,
                    })
                    .collect::<Vec<_>>()
            })
            .collect();
        // And textures read through tables, uploaded by the same list.
        let textures: Vec<_> = runs
            .iter()
            .flat_map(|l| {
                l.deferred()
                    .copies
                    .iter()
                    .filter_map(|s| match s {
                        Step::Texture(staging, copy) => Some((staging.clone(), copy.clone())),
                        _ => None,
                    })
                    .collect::<Vec<_>>()
            })
            .collect();
        // Its number, and what reading title memory that submissions in
        // flight write waits for (`Host::settle`), kept out of the time
        // preparing.
        submitted.owners += 1;
        let owner = submitted.owners;
        #[cfg(feature = "diagnostics")]
        let numbered = submitted.value + 1;
        let mut settled = std::time::Duration::ZERO;
        let mut settle = |o: u64, why: Settled| {
            let waiting = std::time::Instant::now();
            self.settle(submitted, o, why);
            settled += waiting.elapsed();
        };
        // The upload list and its staging memory are used again: the
        // submission that last ran them has completed (`Host::submit`).
        let uploading = !heaps.is_empty() || !textures.is_empty();
        #[cfg(feature = "diagnostics")]
        let timed = crate::watch::Timed(std::time::Instant::now(), crate::copystat::C::UploadsUs);
        let uploads = uploading
            .then(|| crate::bindless::uploads(&heaps, &textures, &mut settle))
            .flatten();
        #[cfg(feature = "diagnostics")]
        drop(timed);
        let all: Vec<&HostList> = uploads
            .as_ref()
            .map(|u| u.list())
            .into_iter()
            .chain(runs.iter().copied())
            .collect();
        let all = &all[..];
        // A frame `crate::capture` lists: every command of every list, in
        // the order the GPU runs them (with `VICTORY_D3D12_DRED`, which
        // notes them).
        #[cfg(feature = "diagnostics")]
        let traced = crate::capture::listed(crate::queue::presented() + 1);
        #[cfg(feature = "diagnostics")]
        if traced {
            for (i, r) in all.iter().enumerate() {
                let notes = r.notes.lock().unwrap_or_else(|e| e.into_inner());
                for (k, (note, _)) in notes.iter().enumerate() {
                    tracing::info!(
                        target: "victory_gpu::trace",
                        "s{numbered} l{i}/{} #{k}: {note}",
                        all.len()
                    );
                }
            }
        }
        let mut lists: Vec<Option<ID3D12CommandList>> = Vec::with_capacity(2 * all.len());
        // Where each run's lists start in `lists`.
        let mut firsts = Vec::with_capacity(all.len());
        let mut batch = Batch::default();
        for list in all {
            list.run_deferred(&mut batch, owner, &mut settle);
            firsts.push(lists.len());
            #[cfg(feature = "diagnostics")]
            let _timed = crate::watch::Timed(std::time::Instant::now(), crate::copystat::C::PreludeUs);
            if let Some(prelude) = list.prelude() {
                lists.push(Some(prelude));
            }
            lists.push(list.list.cast().ok());
        }
        batch.flush(owner, &mut settle);
        let vram_uploads = crate::shadow::take_vram_uploads();
        let mut vram_bitmaps: Vec<(ID3D12Resource, ID3D12Resource)> = Vec::new();
        let mut seen_bitmaps = std::collections::HashSet::new();
        for list in runs {
            for (vram, ram) in &list.deferred().vram_bitmaps {
                use windows::core::Interface;
                if seen_bitmaps.insert(vram.as_raw() as usize) {
                    vram_bitmaps.push((vram.clone(), ram.clone()));
                }
            }
        }
        // Out of overlap nothing else is in flight: the bitmaps are this
        // submission's whatever marked them (tests bind them directly).
        let marks = !overlap || all.iter().any(|l| l.marks.load(Ordering::Relaxed));
        // Its shaders' write marks are captured apart by the list after it
        // (`crate::shadow::capture_write_marks`), from the bytes in use now.
        let captured = marks && crate::shadow::captures_write_marks();
        let mark_bytes = crate::shadow::write_marks_bytes();
        // Copies in video memory: brought up to date by a list run before the
        // submission's own, read back by one run after them, before its fence
        // value; the read-back is taken on completion (`Host::finish`), so the
        // submission stays in flight like any other. The views' bitmaps go
        // with a submission that marks only: one that does not would copy the
        // read marks back over what the CPU cleared meanwhile (`crate::shadow`).
        let mut vram = VramSync::default();
        let before = self.vram_before(vram_uploads, &vram_bitmaps, marks, mark_bytes, &mut vram);
        let reasons = runs.iter().fold(0u8, |r, l| r | l.heavy_reasons());
        let relaxed = crate::early::barriers();
        let barrier = barrier(reasons, marks, captured, relaxed);
        #[cfg(feature = "diagnostics")]
        {
            use crate::copystat::{C, add};
            for (bit, c) in [
                (0, C::HeavyHeapWrites),
                (1, C::HeavyAfter),
                (2, C::HeavyWrites),
                (3, C::HeavyMissed),
                (4, C::HeavyGpuWrites),
                (5, C::HeavyMemory),
                (7, C::HeavyAfterNamed),
            ] {
                add(c, u64::from(reasons >> bit & 1));
            }
            add(C::HeavyWritesOnly, u64::from(reasons & 0x3f == 1 << 2));
            // Heavy before the heap's read-only marks stopped counting.
            add(C::HeapReadsOnly, u64::from(reasons == 1 << 6));
            // Each barrier once, by its first kind: page tables, missed
            // checks, writes through the heap's views, unnamed work after,
            // named work after (without the switch), marks not captured.
            if barrier {
                let kind = [5, 3, 0, 1, 7].into_iter().find(|&bit| reasons >> bit & 1 != 0);
                add(
                    match kind {
                        Some(5) => C::BarrierMemory,
                        Some(3) => C::BarrierMissed,
                        Some(0) => C::BarrierHeapWrites,
                        Some(1) => C::BarrierAfter,
                        Some(_) => C::BarrierAfterNamed,
                        None => C::BarrierMarks,
                    },
                    1,
                );
            }
            // Named work after left light by the switch.
            add(C::AfterNamedLight, u64::from(reasons & AFTER_NAMED != 0 && !barrier));
        }
        // Completing it takes the write marks (`Host::finish`) or reads the
        // lists' page bitmaps (`crate::memory::bind`'s probes).
        let takes = (marks && (captured || barrier || !overlap)
            || runs.iter().any(|l| !l.deferred().vram_probes.is_empty()))
        .then(|| signature(runs));
        let after = self.vram_after(runs, &vram_bitmaps, marks, captured.then_some(mark_bytes), takes, &mut vram);
        // Later submissions are prepared while it runs: what it writes in
        // title memory on completion, for them to wait for where they read it.
        let declared: Vec<Declared> = runs.iter().flat_map(|l| l.declared()).collect();
        if overlap {
            crate::shadow::fly(owner, runs.iter().flat_map(|l| l.flying()).collect());
            // Held waits for its bottom-of-pipe values may read them now
            // (`crate::early`): later submissions run after it on this queue.
            // In the order `Host::finish` writes them.
            if crate::early::on() {
                let writes: Vec<_> = runs.iter().flat_map(|l| l.deferred().writes.clone()).collect();
                crate::early::fly(owner, barrier, &writes);
            }
        }
        #[cfg(feature = "diagnostics")]
        crate::copystat::add(crate::copystat::C::PreparedOverlapping, u64::from(!submitted.flight.is_empty()));
        let prepared = std::time::Instant::now();
        #[cfg(feature = "diagnostics")]
        crate::copystat::add(crate::copystat::C::FaultsPrep, crate::copystat::page_faults().saturating_sub(faults));
        // Diagnostics: each list on its own, timed (`VICTORY_GPU_TIMING`).
        #[cfg(feature = "diagnostics")]
        let groups: Vec<&[Option<ID3D12CommandList>]> = if timing() {
            firsts
                .iter()
                .zip(firsts.iter().skip(1).chain([&lists.len()]))
                .map(|(&a, &b)| &lists[a..b])
                .collect()
        } else {
            vec![&lists[..]]
        };
        #[cfg(not(feature = "diagnostics"))]
        let groups = [&lists[..]];
        let mut removed_in = None;
        for (i, group) in groups.iter().enumerate() {
            submitted.value += 1;
            let started = std::time::Instant::now();
            #[cfg(feature = "diagnostics")]
            crate::resource::life::note(0, crate::resource::life::SUBMITTED, submitted.value, 0);
            // Tests: a gate the GPU waits at first ([`Host::hold_gpu`]).
            #[cfg(test)]
            if overlap && let Some(gate) = GATE.lock().unwrap_or_else(|e| e.into_inner()).take() {
                // SAFETY: our queue and the gate's fence.
                unsafe { self.queue.Wait(&gate, 1) }.expect("the queue waits for the gate");
            }
            // SAFETY: closed lists; the fence is ours.
            unsafe {
                if i == 0 && before.is_some() {
                    self.queue.ExecuteCommandLists(std::slice::from_ref(&before));
                }
                self.queue.ExecuteCommandLists(group);
                if i + 1 == groups.len() && after.is_some() {
                    self.queue.ExecuteCommandLists(std::slice::from_ref(&after));
                }
                if let Err(e) = self.queue.Signal(&self.fence, submitted.value) {
                    tracing::warn!(error = %e, "host submission not signalled");
                }
            }
            if groups.len() == 1 {
                break;
            }
            self.wait_for(submitted.value);
            #[cfg(feature = "diagnostics")]
            crate::resource::life::note(0, crate::resource::life::RAN, submitted.value, 0);
            // Time every submission when timing is on, including one with a single
            // list: its wall was in `on the GPU` but missing from `GPU timing`.
            #[cfg(feature = "diagnostics")]
            if timing() {
                // SAFETY: a plain query.
                let frequency = unsafe { self.queue.GetTimestampFrequency() }.unwrap_or(1);
                gpu_times(started.elapsed(), &all[i].slowest(frequency, usize::MAX));
            }
            #[cfg(feature = "diagnostics")]
            if started.elapsed() > std::time::Duration::from_millis(50) {
                // SAFETY: a plain query.
                let frequency = unsafe { self.queue.GetTimestampFrequency() }.unwrap_or(1);
                let commands = all[i].notes.lock().unwrap_or_else(|e| e.into_inner()).len();
                tracing::warn!(ms = started.elapsed().as_millis(), commands, "slow host list");
                for (ms, op) in all[i].slowest(frequency, 6) {
                    tracing::warn!("slow list command, {ms:.1} ms: {op}");
                }
            }
            // SAFETY: a plain query.
            if removed_in.is_none() && unsafe { self.device.GetDeviceRemovedReason() }.is_err() {
                removed_in = Some(i);
            }
            let _ = (i, started);
        }
        let _ = &firsts;
        // Lists timed on their own ran here: a removal is reported with the
        // list it happened in.
        // SAFETY: a plain query.
        if groups.len() > 1 && unsafe { self.device.GetDeviceRemovedReason() }.is_err() {
            fail(format_args!(
                "host GPU device removed after a submission of {} lists{}{}{}",
                all.len(),
                self.removal_data(all),
                last_notes(all),
                removed_during(all, removed_in)
            ));
        }
        SUBMISSION_US[0].fetch_add((prepared - started).saturating_sub(settled).as_micros() as u64, Ordering::Relaxed);
        SUBMISSION_US[1].fetch_add(prepared.elapsed().as_micros() as u64, Ordering::Relaxed);
        InFlight {
            value: submitted.value,
            owner,
            queue: 0,
            lists: Vec::new(),
            barrier,
            declared,
            uploads: uploads.is_some(),
            marks,
            sync: !overlap,
            prepared,
            batch,
            heaps,
            signals: Vec::new(),
            retired: std::mem::take(&mut *RETIRED.lock().unwrap_or_else(|e| e.into_inner())),
            vram,
            #[cfg(feature = "diagnostics")]
            traced,
        }
    }

    /// Completes a submission once the host GPU has run it (waiting for that
    /// under the submission lock): what shaders wrote goes back to title
    /// memory, then the lists' work after them and bottom-of-pipe writes,
    /// then the title fences signalled behind it. `runs`: its lists, but
    /// our upload list; `quiet`: no submission whose shaders mark reads is
    /// in flight after it (`crate::shadow::take_reads`).
    fn finish(&self, mut record: InFlight, runs: &[&HostList], quiet: bool) {
        let waiting = std::time::Instant::now();
        self.wait_for(record.value);
        SUBMISSION_US[1].fetch_add(waiting.elapsed().as_micros() as u64, Ordering::Relaxed);
        #[cfg(feature = "diagnostics")]
        crate::resource::life::note(0, crate::resource::life::RAN, record.value, 0);
        // The host resets a device whose submission runs for about 2 s (from
        // its submission: those before it in flight count too).
        let ran = record.prepared.elapsed();
        if ran > std::time::Duration::from_millis(250) {
            let pipelines: std::collections::BTreeSet<String> = runs
                .iter()
                .flat_map(|l| l.pipelines.lock().unwrap_or_else(|e| e.into_inner()).clone())
                .collect();
            tracing::warn!(
                ms = ran.as_millis(),
                lists = runs.len(),
                ?pipelines,
                "slow host submission"
            );
        }
        // SAFETY: a plain query.
        if unsafe { self.device.GetDeviceRemovedReason() }.is_err() {
            fail(format_args!(
                "host GPU device removed after a submission of {} lists{}{}",
                runs.len(),
                self.removal_data(runs),
                last_notes(runs)
            ));
        }
        // What shaders wrote goes back before the bottom-of-pipe writes,
        // which tell the title the GPU is done; each range once per
        // submission, however many dispatches declared it (the world's
        // passes declare the same big ranges dozens of times).
        let copying = std::time::Instant::now();
        #[cfg(feature = "diagnostics")]
        let faults = crate::copystat::page_faults();
        // Copies in video memory first: their mirrors are what goes back. The
        // views' written pages are read back when their marks are taken below:
        // its own (captured after it), or taken from the bitmaps with nothing
        // after it in flight (a barrier, or waited for).
        let marks_taken = record.marks && (record.vram.marks.is_some() || record.barrier || record.sync);
        #[cfg(feature = "diagnostics")]
        {
            use crate::copystat::{C, add};
            add(C::Completions, 1);
            add(C::HeavyCompletions, u64::from(record.barrier));
            add(C::MarksTaken, u64::from(marks_taken));
        }
        let (unchanged, marks) = self.vram_completed(std::mem::take(&mut record.vram), runs, marks_taken);
        // One list's lock at a time: a submission may name a list twice.
        let mut back: Vec<(std::sync::Arc<crate::shadow::Shadow>, usize, usize, bool)> = Vec::new();
        let mut index: std::collections::HashMap<(usize, usize, usize), usize> = std::collections::HashMap::new();
        // Shaders that read memory with no copy computed something else than
        // the console's: it stays out of title memory, where it broke the
        // title's own data (runs c1, c4), and so does what every program
        // using title memory wrote while any misses
        // (`crate::memory::withholding`). Only lists with such programs have
        // checks; all are asked before anything goes back. Other programs in
        // those lists write through bounded views from their own inputs:
        // their ranges go back (the imposter transforms beside the BVH
        // builder in one list, run light13).
        let missed: Vec<bool> = runs
            .iter()
            .map(|list| list.deferred().missed.iter().any(|check| check()))
            .collect();
        crate::memory::note_submission(missed.contains(&true));
        let withholding = crate::memory::withholding();
        for (list, &missed) in runs.iter().zip(&missed) {
            let deferred = list.deferred();
            let missed = missed || withholding && !deferred.missed.is_empty();
            for step in &deferred.copies {
                if let Step::GpuWrite(shadow, address, len, _, memory) = step {
                    let key = (std::sync::Arc::as_ptr(shadow) as usize, *address, *len);
                    // A range both kinds declare is withheld whole.
                    let goes_back = !(missed && *memory);
                    match index.get(&key) {
                        Some(&i) => back[i].3 &= goes_back,
                        None => {
                            index.insert(key, back.len());
                            back.push((shadow.clone(), *address, *len, goes_back));
                        }
                    }
                }
            }
        }
        // A range overlapping a withheld one of the same copy is withheld
        // too: its copy-back would take the other's bytes along (pages a
        // program using title memory and another both wrote).
        let held: Vec<(usize, usize, usize)> = back
            .iter()
            .filter(|b| !b.3)
            .map(|(s, a, n, _)| (std::sync::Arc::as_ptr(s) as usize, *a, *n))
            .collect();
        for b in back.iter_mut().filter(|b| b.3) {
            let copy = std::sync::Arc::as_ptr(&b.0) as usize;
            b.3 = !held.iter().any(|&(c, a, n)| c == copy && a < b.1 + b.2 && b.1 < a + n);
        }
        let withheld: Vec<(std::sync::Arc<crate::shadow::Shadow>, usize, usize)> = back
            .iter()
            .filter(|b| !b.3)
            .map(|(s, a, n, _)| (s.clone(), *a, *n))
            .collect();
        // Lazy view pages still hold the title bytes we left (diagnostics:
        // a writer that went past us loses its bytes at the first touch).
        #[cfg(feature = "diagnostics")]
        crate::watch::verify_lazy(32);
        if marks_taken {
            // Pages shaders marked first: the copy-backs below end what the
            // copies keep for them.
            crate::shadow::copy_back_marks(record.owner, &marks, &withheld, withholding, &unchanged);
        }
        // Texture views left null that shaders read: made for what follows
        // (a later submission's marks taken with them only make its views
        // early).
        if record.marks {
            let read = crate::shadow::take_reads(quiet);
            if !read.is_empty() {
                crate::descriptor::make_read(&read);
            }
        }
        for (shadow, address, len, goes_back) in &back {
            let (address, len) = (*address, *len);
            if *goes_back {
                let written = shadow.copy_back(record.owner, address, len);
                COPY_BACK[1].fetch_add(written as u64, Ordering::Relaxed);
            } else {
                shadow.keep_writes(record.owner, address, len);
                crate::memory::note_withheld(len as u64);
                crate::memory::drop_withheld(address, len);
            }
        }
        for list in runs {
            for work in &list.deferred().after {
                work();
            }
        }
        COPY_BACK[0].fetch_add(copying.elapsed().as_micros() as u64, Ordering::Relaxed);
        #[cfg(feature = "diagnostics")]
        crate::copystat::add(crate::copystat::C::FaultsBack, crate::copystat::page_faults().saturating_sub(faults));
        for list in runs {
            for &(at, what, bytes) in &list.deferred().writes {
                what.write(at as *mut u8, bytes);
            }
        }
        // What it wrote is in title memory now: later preparings read it
        // there.
        crate::shadow::land(record.owner);
        crate::early::land(record.owner);
        // Host objects given up before the submission had their last
        // holder in it.
        let InFlight {
            batch,
            heaps,
            signals,
            retired,
            ..
        } = record;
        release_later(Box::new((batch, back, withheld, heaps, retired)));
        self.log_messages();
        // The title's fences signalled behind it, in order, under the
        // submission lock: a later Signal of the same fence cannot pass them.
        for (fence, value) in signals {
            set_title_fence(fence, value);
        }
        // Last, once all it writes and signals has landed: a pump that takes
        // the flag sees all of it (set first, another thread's pump could
        // take it before this one's fences moved, and this one's own find
        // it taken).
        COMPLETED.store(true, Ordering::Release);
    }

    /// With `VICTORY_D3D12_DRED`: the faulting address and the allocations
    /// it lies in (by the names we give them) or the title memory copy just
    /// below it; for each list that did not finish, the command it stopped
    /// at and what the draws and dispatches in flight there run (or, in
    /// lists that never started, the pipelines they bind).
    fn removal_data(&self, runs: &[&HostList]) -> String {
        let Ok(dred) = self.device.cast::<ID3D12DeviceRemovedExtendedData1>() else {
            return String::new();
        };
        let mut text = String::new();
        let mut fault_va = None;
        // SAFETY: the outputs are the runtime's, valid while the device
        // lives; nodes are walked as linked, histories have their counts.
        unsafe {
            if let Ok(fault) = dred.GetPageFaultAllocationOutput1()
                && fault.PageFaultVA != 0
            {
                let va = fault.PageFaultVA;
                fault_va = Some(va);
                text.push_str(&format!("; page fault at {va:#x}"));
                for (what, head) in [
                    ("inside", fault.pHeadExistingAllocationNode),
                    ("inside freed", fault.pHeadRecentFreedAllocationNode),
                ] {
                    let mut names = Vec::new();
                    let mut node = head;
                    while let Some(n) = node.as_ref()
                        && names.len() < 8
                    {
                        let name = if n.ObjectNameW.is_null() {
                            "unnamed".to_owned()
                        } else {
                            n.ObjectNameW.to_string().unwrap_or_default()
                        };
                        names.push(format!("{name} (type {})", n.AllocationType.0));
                        node = n.pNext;
                    }
                    if !names.is_empty() {
                        text.push_str(&format!("; {what} {names:?}"));
                    }
                    #[cfg(feature = "diagnostics")]
                    for name in names.iter().filter(|_| what == "inside freed") {
                        text.push_str(&crate::resource::life::of(name));
                    }
                }
                text.push_str(&crate::shadow::near(va));
                text.push_str(&crate::descriptor::host::near_heaps(va));
            }
            // The plain output: the extended one (with marker texts) came
            // back without stopped lists on this host.
            let crumbs = match dred.GetAutoBreadcrumbsOutput() {
                Ok(crumbs) => crumbs,
                Err(e) => {
                    text.push_str(&format!("; no breadcrumbs ({e})"));
                    return text;
                }
            };
            // The nodes, oldest list first, and which run each is.
            let mut nodes = Vec::new();
            let mut node = crumbs.pHeadAutoBreadcrumbNode;
            while let Some(p) = node.as_ref() {
                nodes.push(p);
                node = p.pNext;
            }
            let index: Vec<Option<usize>> = nodes
                .iter()
                .map(|p| runs.iter().position(|r| r.is_node(p)))
                .collect();
            if let Some(last) = nodes.last()
                && index.iter().all(Option::is_none)
            {
                let name = |w: windows::core::PCWSTR, a: *const u8| {
                    let w = (!w.is_null()).then(|| w.to_string().unwrap_or_default());
                    let a = (!a.is_null()).then(|| {
                        std::ffi::CStr::from_ptr(a.cast())
                            .to_string_lossy()
                            .into_owned()
                    });
                    format!("{w:?}/{a:?}")
                };
                tracing::warn!(
                    nodes = nodes.len(),
                    last_pointer = ?last.pCommandList.as_ref().map(|l| l.as_raw()),
                    last_names = %name(last.pCommandListDebugNameW, last.pCommandListDebugNameA),
                    "no breadcrumb node named one of our lists"
                );
            }
            // Every list's progress (done/count), in the runtime's order.
            let progress: Vec<String> = nodes
                .iter()
                .zip(&index)
                .map(|(p, index)| {
                    let done = p.pLastBreadcrumbValue.as_ref().copied().unwrap_or(0);
                    format!("{done}/{} {index:?}", p.BreadcrumbCount)
                })
                .collect();
            text.push_str(&format!("; breadcrumbs (done/count, list) {progress:?}"));
            for (p, index) in nodes.iter().zip(&index) {
                let done = p.pLastBreadcrumbValue.as_ref().copied().unwrap_or(0);
                if done < p.BreadcrumbCount && !p.pCommandHistory.is_null() {
                    let history =
                        std::slice::from_raw_parts(p.pCommandHistory, p.BreadcrumbCount as usize);
                    text.push_str(&format!(
                        "; a list stopped at command {done} of {} (op {})",
                        p.BreadcrumbCount, history[done as usize].0
                    ));
                    let run = index.map(|i| runs[i]);
                    // The k-th marker command is the k-th note.
                    #[cfg(feature = "diagnostics")]
                    let markers: Vec<(u32, String)> = run.map_or_else(Vec::new, |r| {
                        let notes = r.notes.lock().unwrap_or_else(|e| e.into_inner());
                        history
                            .iter()
                            .enumerate()
                            .filter(|(_, op)| **op == D3D12_AUTO_BREADCRUMB_OP_SETMARKER)
                            .zip(notes.iter())
                            .map(|((at, _), (note, range))| {
                                let place = fault_va.zip(*range).map(|(va, r)| fault_in(va, r));
                                (at as u32, format!("{note}{}", place.unwrap_or_default()))
                            })
                            .collect()
                    });
                    #[cfg(not(feature = "diagnostics"))]
                    let markers: Vec<(u32, String)> = Vec::new();
                    #[cfg(not(feature = "diagnostics"))]
                    let _ = fault_va;
                    if done > 0 {
                        text.push_str(&in_flight(history, &markers, done as usize));
                    } else if let Some(r) = run {
                        let pipelines = r.pipelines.lock().unwrap_or_else(|e| e.into_inner());
                        text.push_str(&format!(", its pipelines {pipelines:?}"));
                    }
                }
            }
        }
        text
    }

    /// Moves the debug layer's messages into our log; returns the first
    /// error among them.
    pub(crate) fn log_messages(&self) -> Option<String> {
        let Some(queue) = &self.messages else {
            return None;
        };
        let mut first_error = None;
        // SAFETY: each message is read into a buffer of the size the queue
        // reports, then the queue is emptied.
        unsafe {
            for i in 0..queue.GetNumStoredMessages() {
                let mut size = 0;
                if queue.GetMessage(i, None, &mut size).is_err() || size == 0 {
                    continue;
                }
                let mut buffer = vec![0u64; size.div_ceil(8)];
                let message = buffer.as_mut_ptr().cast::<D3D12_MESSAGE>();
                if queue.GetMessage(i, Some(message), &mut size).is_ok() {
                    let m = &*message;
                    let text = std::slice::from_raw_parts(m.pDescription, m.DescriptionByteLength);
                    let text = String::from_utf8_lossy(text);
                    let text = text.trim_end_matches('\0');
                    tracing::warn!(id = m.ID.0, severity = m.Severity.0, "D3D12: {text}");
                    if m.Severity.0 <= D3D12_MESSAGE_SEVERITY_ERROR.0 && first_error.is_none() {
                        first_error = Some(text.to_owned());
                    }
                }
            }
            queue.ClearStoredMessages();
        }
        first_error
    }

    /// A host render target view of `resource`, valid until it is recorded.
    pub(crate) fn rtv(
        &self,
        resource: &ID3D12Resource,
        desc: Option<&D3D12_RENDER_TARGET_VIEW_DESC>,
    ) -> D3D12_CPU_DESCRIPTOR_HANDLE {
        let slot = self.next_rtv.fetch_add(1, Ordering::Relaxed) % RTV_RING;
        // SAFETY: a slot of our heap and a live resource.
        unsafe {
            let mut handle = self.rtvs.GetCPUDescriptorHandleForHeapStart();
            handle.ptr += (slot * self.rtv_size) as usize;
            self.device
                .CreateRenderTargetView(resource, desc.map(|d| d as *const _), handle);
            handle
        }
    }

    /// A slot of CPU-only views, valid until it is recorded.
    pub(crate) fn cpu_view(&self) -> D3D12_CPU_DESCRIPTOR_HANDLE {
        let slot = self.next_cpu_view.fetch_add(1, Ordering::Relaxed) % RTV_RING;
        // SAFETY: a live heap.
        let mut handle = unsafe { self.cpu_views.GetCPUDescriptorHandleForHeapStart() };
        handle.ptr += (slot * self.cpu_view_size) as usize;
        handle
    }

    /// A host depth view of `resource`, valid until it is recorded.
    pub(crate) fn dsv(
        &self,
        resource: &ID3D12Resource,
        desc: Option<&D3D12_DEPTH_STENCIL_VIEW_DESC>,
    ) -> D3D12_CPU_DESCRIPTOR_HANDLE {
        let slot = self.next_dsv.fetch_add(1, Ordering::Relaxed) % RTV_RING;
        // SAFETY: a slot of our heap and a live resource.
        unsafe {
            let mut handle = self.dsvs.GetCPUDescriptorHandleForHeapStart();
            handle.ptr += (slot * self.dsv_size) as usize;
            self.device
                .CreateDepthStencilView(resource, desc.map(|d| d as *const _), handle);
            handle
        }
    }
}

impl HostList {
    /// The timestamp heap and its read-back buffer (diagnostics, timing on).
    #[cfg(feature = "diagnostics")]
    fn stamps(&self) -> Option<&(ID3D12QueryHeap, ID3D12Resource)> {
        if !timing() {
            return None;
        }
        self.stamps
            .get_or_init(|| {
                let device = &host()?.device;
                let mut heap: Option<ID3D12QueryHeap> = None;
                let mut back: Option<ID3D12Resource> = None;
                let readback = D3D12_HEAP_PROPERTIES { Type: D3D12_HEAP_TYPE_READBACK, ..Default::default() };
                let desc = D3D12_RESOURCE_DESC {
                    Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
                    Width: u64::from(STAMPS) * 8,
                    Height: 1,
                    DepthOrArraySize: 1,
                    MipLevels: 1,
                    SampleDesc: DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
                    Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
                    ..Default::default()
                };
                // SAFETY: plain object creation.
                unsafe {
                    device
                        .CreateQueryHeap(
                            &D3D12_QUERY_HEAP_DESC { Type: D3D12_QUERY_HEAP_TYPE_TIMESTAMP, Count: STAMPS, NodeMask: 0 },
                            &mut heap,
                        )
                        .ok()?;
                    device
                        .CreateCommittedResource(
                            &readback,
                            D3D12_HEAP_FLAG_NONE,
                            &desc,
                            D3D12_RESOURCE_STATE_COPY_DEST,
                            None,
                            &mut back,
                        )
                        .ok()?;
                }
                Some((heap?, back?))
            })
            .as_ref()
    }

    /// The noted commands that took longest on the GPU, longest first, as
    /// (milliseconds, text) (diagnostics, timing on, after the list ran).
    #[cfg(feature = "diagnostics")]
    fn slowest(&self, frequency: u64, n: usize) -> Vec<(f64, String)> {
        let notes = self.notes.lock().unwrap_or_else(|e| e.into_inner());
        let (Some((_, back)), count) = (self.stamps(), notes.len().min(STAMPS as usize - 1)) else {
            return Vec::new();
        };
        let mut data = std::ptr::null_mut();
        let range = D3D12_RANGE { Begin: 0, End: (count + 1) * 8 };
        // SAFETY: a read-back buffer the finished list resolved into.
        let stamps: Vec<u64> = unsafe {
            if back.Map(0, Some(&range), Some(&mut data)).is_err() {
                return Vec::new();
            }
            let s = std::slice::from_raw_parts(data.cast::<u64>(), count + 1).to_vec();
            back.Unmap(0, None);
            s
        };
        let mut times: Vec<(f64, String)> = (0..count)
            .map(|i| {
                let ticks = stamps[i + 1].saturating_sub(stamps[i]);
                (ticks as f64 * 1e3 / frequency.max(1) as f64, notes[i].0.chars().take(300).collect())
            })
            .collect();
        times.sort_by(|a, b| b.0.total_cmp(&a.0));
        times.truncate(n);
        times
    }

    /// Whether `node` of the device-removed data (its breadcrumbs) is this
    /// list.
    fn is_node(&self, node: &D3D12_AUTO_BREADCRUMB_NODE) -> bool {
        // SAFETY: only the pointer and the name text are read: the pointer is
        // compared, never used, and the name is the runtime's copy.
        let (pointer, wide, ansi) = unsafe {
            (
                node.pCommandList.as_ref().map(|l| l.as_raw() as usize),
                (!node.pCommandListDebugNameW.is_null())
                    .then(|| node.pCommandListDebugNameW.to_string().unwrap_or_default()),
                (!node.pCommandListDebugNameA.is_null()).then(|| {
                    std::ffi::CStr::from_ptr(node.pCommandListDebugNameA.cast())
                        .to_string_lossy()
                        .into_owned()
                }),
            )
        };
        let (pointers, name) = (identities(&self.list), list_name(self.id));
        node_is_list(pointer, wide.as_deref(), &pointers, &name)
            || node_is_list(None, ansi.as_deref(), &pointers, &name)
    }

    pub(crate) fn close(&self) {
        self.end_hoisted_indirect();
        crate::indirect::end_window(self);
        #[cfg(feature = "diagnostics")]
        if let Some((heap, back)) = self.stamps() {
            let count = (self.notes.lock().unwrap_or_else(|e| e.into_inner()).len() as u32).min(STAMPS - 1);
            // SAFETY: an open list, our heap and read-back buffer.
            unsafe {
                self.list.EndQuery(heap, D3D12_QUERY_TYPE_TIMESTAMP, count);
                self.list.ResolveQueryData(heap, D3D12_QUERY_TYPE_TIMESTAMP, 0, count + 1, back, 0);
            }
        }
        // SAFETY: our list.
        if let Err(e) = unsafe { self.list.Close() } {
            fail(format_args!(
                "a host command list did not close ({e}): a command recorded into it was invalid"
            ));
        }
    }

    /// Drops resources and state tracked by this list without resetting the
    /// D3D12 command allocator or command list (only after the list's last
    /// submission finished).
    pub(crate) fn clear_held(&self) {
        *self.deferred() = Deferred::default();
        self.marks.store(false, Ordering::Relaxed);
        self.heap_writes.store(false, Ordering::Relaxed);
        crate::indirect::forget_windows(self);
        self.states.lock().unwrap_or_else(|e| e.into_inner()).clear();
        self.twins.lock().unwrap_or_else(|e| e.into_inner()).clear();
        self.pipelines.lock().unwrap_or_else(|e| e.into_inner()).clear();
        self.hoisted_indirect.lock().unwrap_or_else(|e| e.into_inner()).reset();
        #[cfg(feature = "diagnostics")]
        self.notes.lock().unwrap_or_else(|e| e.into_inner()).clear();
        #[cfg(feature = "diagnostics")]
        self.targets.lock().unwrap_or_else(|e| e.into_inner()).clear();
    }

    /// Only after the list's last submission finished (every one is waited
    /// for).
    pub(crate) fn reset(&self) {
        self.clear_held();
        {
            let mut scratch = self.scratch.lock().unwrap_or_else(|e| e.into_inner());
            (scratch.current, scratch.used) = (0, 0);
        }
        // SAFETY: the list is not executing.
        let reset = unsafe {
            self.allocator
                .Reset()
                .and_then(|()| self.list.Reset(&self.allocator, None))
        };
        if let Err(e) = reset {
            fail(format_args!("a host command list did not reset ({e})"));
        }
    }

    /// Moves a title texture's host copy to `wanted` (a host state) in this
    /// list: the host texture and the state it leaves, if that is a change.
    /// The list's first use of a texture records nothing: its prelude brings
    /// the texture there at submission.
    /// A depth texture with a writable twin (`crate::twin`) has the side
    /// the use needs brought up to date first; unordered access moves the
    /// twin, not the depth texture (nothing returned).
    pub(crate) fn transition(
        &self,
        resource: crate::object::This,
        wanted: u32,
    ) -> Option<(ID3D12Resource, u32)> {
        // One host texture for the whole move (`crate::resource::evict` may
        // give it up meanwhile; this list then holds it).
        let copy = crate::resource::host_copy(resource)?;
        let texture = copy.texture.clone();
        if let Some(Some(twin)) = copy.twin.get()
            && crate::twin::route(self, &copy, twin, wanted)
        {
            return None;
        }
        let wanted = legal_state(&texture, wanted);
        self.track(&copy, wanted).map(|before| (texture, before))
    }

    /// The list's state for host texture `copy` becomes `wanted` (a legal
    /// one): the state it leaves, if that is a change. Nothing for the
    /// list's first use (its prelude moves the texture there).
    fn track(&self, copy: &std::sync::Arc<crate::resource::HostTexture>, wanted: u32) -> Option<u32> {
        let mut states = self.states.lock().unwrap_or_else(|e| e.into_inner());
        match states.entry(std::sync::Arc::as_ptr(copy) as usize) {
            std::collections::hash_map::Entry::Vacant(v) => {
                copy.note_used();
                if !self.ours.load(Ordering::Relaxed) {
                    copy.stamp();
                }
                v.insert((wanted, wanted, copy.clone()));
                None
            }
            std::collections::hash_map::Entry::Occupied(mut o) => {
                let before = std::mem::replace(&mut o.get_mut().1, wanted);
                (before != wanted).then_some(before)
            }
        }
    }

    /// [`HostList::transition`] for a host texture of ours (a twin's
    /// plane), without twins: the barrier to record, if one is needed.
    pub(crate) fn move_texture(
        &self,
        copy: &std::sync::Arc<crate::resource::HostTexture>,
        wanted: u32,
    ) -> Option<D3D12_RESOURCE_BARRIER> {
        let wanted = legal_state(&copy.texture, wanted);
        let before = self.track(copy, wanted)?;
        Some(transition(
            &copy.texture,
            D3D12_RESOURCE_STATES(before as i32),
            D3D12_RESOURCE_STATES(wanted as i32),
        ))
    }

    /// A use of the depth texture `depth` (with a twin) needing `side`
    /// (`crate::twin::DEPTH` or `TWIN`): whether a copy must be recorded
    /// now (true: into the depth texture). The list's first use is the
    /// prelude's to prepare.
    pub(crate) fn twin_use(
        &self,
        depth: &std::sync::Arc<crate::resource::HostTexture>,
        side: u8,
        writes: bool,
    ) -> Option<bool> {
        let mut twins = self.twins.lock().unwrap_or_else(|e| e.into_inner());
        match twins.entry(std::sync::Arc::as_ptr(depth) as usize) {
            std::collections::hash_map::Entry::Vacant(v) => {
                v.insert(crate::twin::Use::new(depth, side, writes));
                None
            }
            std::collections::hash_map::Entry::Occupied(mut o) => o.get_mut().again(side, writes),
        }
    }

    /// The list is one of ours (uploads): what it uses is no use of the
    /// title's.
    pub(crate) fn set_ours(&self) {
        self.ours.store(true, Ordering::Relaxed);
    }

    /// What the list's shaders write from now until the guard goes is
    /// written by a program using title memory (`uses`) or not.
    pub(crate) fn writer_uses_memory(&self, uses: bool) -> impl Drop + '_ {
        struct Reset<'a>(&'a std::sync::atomic::AtomicBool);
        impl Drop for Reset<'_> {
            fn drop(&mut self) {
                self.0.store(false, Ordering::Relaxed);
            }
        }
        self.memory_writer.store(uses, Ordering::Relaxed);
        Reset(&self.memory_writer)
    }

    /// A copy of ours replaced the compute root signature and pipeline.
    /// Shaders of the list mark their writes and reads in the views'
    /// bitmaps (`crate::shadow::view_writes`).
    pub(crate) fn marking(&self) {
        self.marks.store(true, Ordering::Relaxed);
    }

    /// A program of the list indexing the bound heap may write through its
    /// views.
    pub(crate) fn writes_through_heap(&self) {
        self.heap_writes.store(true, Ordering::Relaxed);
    }

    /// Why completing a submission of the list writes title memory or takes
    /// what its shaders marked as written, as bits (any of 0–5 or 7: the
    /// list is heavy): 0 writes marked through the heap's views, 1 work after
    /// it naming nothing it writes, 2 bottom-of-pipe writes, 3 missed-memory
    /// checks, 4 copy-backs of declared ranges its shaders write, 5 title
    /// memory through page tables, 7 work after it naming what it writes
    /// ([`AFTER_NAMED`]). The next submission's refreshes and copies read
    /// title memory: it is prepared only once a submission with bits 0, 1,
    /// 3 or 5 (and 7 without `VICTORY_EARLY_BARRIERS`) has completed
    /// ([`barrier`]); bits 2, 4 and 7 name where it writes
    /// ([`HostList::declared`], [`HostList::flying`]). Marks of views read
    /// alone (tables, root views, and the heap's when no program of the list
    /// writes through them) do not count: taken early, they make views
    /// early. Bit 6, not heavy: the list marks through the heap's views and
    /// only reads them.
    fn heavy_reasons(&self) -> u8 {
        reasons_of(&self.deferred(), self.marks.load(Ordering::Relaxed), self.heap_writes.load(Ordering::Relaxed))
    }

    /// The ranges its shaders declare as written ([`Declared`]).
    fn declared(&self) -> Vec<Declared> {
        self.deferred()
            .copies
            .iter()
            .filter_map(|s| match s {
                Step::GpuWrite(shadow, address, len, marked, _) => {
                    Some((std::sync::Arc::as_ptr(shadow) as usize, *address, *address + *len, *marked))
                }
                _ => None,
            })
            .collect()
    }

    /// What a submission of the list writes in title memory when it
    /// completes, for `crate::shadow::fly`: its declared ranges through
    /// their copies (marked ones as the whole pages their copy-back writes),
    /// its bottom-of-pipe writes and what its work after names.
    fn flying(&self) -> Vec<(usize, usize, usize)> {
        flying_of(&self.deferred())
    }

    /// Whether a submission of the list brings textures up to date with
    /// the upload list (`crate::bindless::uploads`).
    fn uploads_at_submission(&self) -> bool {
        self.deferred()
            .copies
            .iter()
            .any(|s| matches!(s, Step::Heap(_) | Step::Texture(..)))
    }

    /// Keeps `what` until the list is reset (after its last submission has
    /// completed): what a title list owned that this host list still runs
    /// with (`crate::list::leave`).
    pub(crate) fn keep_until_reset(&self, what: Box<dyn std::any::Any + Send>) {
        self.deferred().held.push(what);
    }

    pub(crate) fn need_rebind(&self) {
        self.rebind.store(true, Ordering::Relaxed);
    }

    /// Whether the list's compute root signature and pipeline must be set
    /// again (once).
    pub(crate) fn take_rebind(&self) -> bool {
        self.rebind.swap(false, Ordering::Relaxed)
    }

    /// At submission: the list that moves each texture this list uses from
    /// its real state to the one the list starts with; the textures' real
    /// states become those the list ends with.
    fn prelude(&self) -> Option<ID3D12CommandList> {
        // Depth textures and their twins first, from their real states and
        // back: the side each first use needs, made newer.
        let twins = self.twins.lock().unwrap_or_else(|e| e.into_inner());
        let copies = crate::twin::at_submission(twins.values());
        drop(twins);
        let states = self.states.lock().unwrap_or_else(|e| e.into_inner());
        // The copies start from the real states, then the list's moves from
        // them; only after both do they become the states the list leaves.
        let moves = || -> Vec<D3D12_RESOURCE_BARRIER> {
            states
                .values()
                .filter_map(|(first, _, copy)| {
                    let real = copy.state.load(Ordering::Relaxed);
                    (real != *first).then(|| {
                        transition(&copy.texture, D3D12_RESOURCE_STATES(real as i32), D3D12_RESOURCE_STATES(*first as i32))
                    })
                })
                .collect()
        };
        let left = || {
            for (_, last, copy) in states.values() {
                copy.state.store(*last, Ordering::Relaxed);
            }
        };
        let has_hoisted = !self.hoisted().is_empty();
        if copies.is_empty()
            && !has_hoisted
            && states.values().all(|(first, _, copy)| copy.state.load(Ordering::Relaxed) == *first)
        {
            left();
            return None;
        }
        let (allocator, list) = &self.prelude;
        // SAFETY: the prelude is not executing (every submission is waited
        // for); the textures are live.
        let recorded = unsafe {
            allocator.Reset().and_then(|()| {
                list.Reset(allocator, None)?;
                for (depth, to_depth) in &copies {
                    let Some(Some(twin)) = depth.twin.get() else { continue };
                    // From each texture's real state to the copy's and back.
                    let mut back = Vec::new();
                    twin.copy(list, depth, *to_depth, &mut |t, wanted| {
                        let real = t.state.load(Ordering::Relaxed);
                        let wanted = legal_state(&t.texture, wanted);
                        (real != wanted).then(|| {
                            let there = transition(&t.texture, D3D12_RESOURCE_STATES(real as i32), D3D12_RESOURCE_STATES(wanted as i32));
                            back.push(transition(
                                &t.texture,
                                D3D12_RESOURCE_STATES(wanted as i32),
                                D3D12_RESOURCE_STATES(real as i32),
                            ));
                            there
                        })
                    });
                    if !back.is_empty() {
                        list.ResourceBarrier(&back);
                    }
                }
                let barriers = moves();
                if !barriers.is_empty() {
                    list.ResourceBarrier(&barriers);
                }
                if has_hoisted
                    && let Some((budget_res, budget_gpu)) = self.budget_resource()
                {
                    self.hoisted().record_prelude(list, budget_res, budget_gpu);
                }
                list.Close()
            })
        };
        left();
        if let Err(e) = recorded {
            fail(format_args!("a host state prelude did not record ({e})"));
        }
        list.cast().ok()
    }

    /// Notes what the command about to be recorded does, as a marker
    /// command whose text the list keeps (diagnostics, with device-removed
    /// data on): the report pairs the breadcrumbs' marker commands with the
    /// texts in order and names the commands a fault stopped among.
    #[cfg(feature = "diagnostics")]
    pub(crate) fn note_op(&self, what: impl FnOnce() -> String) {
        self.note(what, None);
    }

    /// As [`HostList::note_op`], for a copy that reads the host GPU range
    /// `(address, bytes)`: the report places the faulting address in it.
    #[cfg(feature = "diagnostics")]
    pub(crate) fn note_read(&self, what: impl FnOnce() -> String, range: (u64, u64)) {
        self.note(what, Some(range));
    }

    #[cfg(feature = "diagnostics")]
    fn note(&self, what: impl FnOnce() -> String, range: Option<(u64, u64)>) {
        let dred = dred_on();
        if !dred && !timing() {
            return;
        }
        let at = self.notes.lock().unwrap_or_else(|e| e.into_inner()).len() as u32;
        if let Some((heap, _)) = self.stamps() && at < STAMPS - 1 {
            // SAFETY: an open list and our query heap.
            unsafe { self.list.EndQuery(heap, D3D12_QUERY_TYPE_TIMESTAMP, at) };
        }
        let what = what();
        // Timing uses the labels and queries; DRED also gets GPU markers.
        if dred {
            traced(&what);
            let text: Vec<u16> = what.encode_utf16().chain([0]).collect();
            const UNICODE: u32 = 0;
            // SAFETY: an open list; the marker is copied when recorded.
            unsafe {
                self.list.SetMarker(UNICODE, Some(text.as_ptr().cast()), 2 * text.len() as u32)
            };
        }
        self.notes
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .push((what, range));
    }

    /// Notes a texture the list renders to or writes, in a listed frame.
    #[cfg(feature = "diagnostics")]
    pub(crate) fn note_target(&self, resource: usize, by: &str) {
        let mut targets = self.targets.lock().unwrap_or_else(|e| e.into_inner());
        if !targets.iter().any(|t| t.0 == resource) {
            targets.push((resource, by.to_owned()));
        }
    }

    /// Notes a pipeline the list binds.
    pub(crate) fn used_pipeline(&self, name: &str) {
        let mut pipelines = self.pipelines.lock().unwrap_or_else(|e| e.into_inner());
        if !pipelines.contains(name) {
            pipelines.insert(name.to_owned());
        }
    }

    fn deferred(&self) -> std::sync::MutexGuard<'_, Deferred> {
        self.deferred.lock().unwrap_or_else(|e| e.into_inner())
    }

    /// Copies `len` bytes from `src` to `dst` when the list is submitted.
    pub(crate) fn defer_copy(&self, dst: *mut u8, src: *const u8, len: usize) {
        let mut deferred = self.deferred();
        deferred
            .copies
            .push(Step::Copy(dst as usize, src as usize, len));
        deferred.refreshed.clear();
    }

    /// Copies `len` bytes of title memory at `src` into the list's upload
    /// memory at `dst` when the list is submitted.
    pub(crate) fn defer_fill(&self, dst: *mut u8, src: *const u8, len: usize) {
        self.deferred()
            .copies
            .push(Step::Fill(dst as usize, src as usize, len));
    }

    /// Brings `[address, address + len)` of a buffer's host copy up to date
    /// when the list is submitted (once per range until a title copy
    /// intervenes), keeping the copy alive until the list is reset.
    pub(crate) fn defer_refresh(
        &self,
        shadow: &std::sync::Arc<crate::shadow::Shadow>,
        address: usize,
        len: usize,
    ) {
        let mut deferred = self.deferred();
        let key = (std::sync::Arc::as_ptr(shadow) as usize, address, len);
        if deferred.refreshed.insert(key) {
            deferred
                .copies
                .push(Step::Refresh(shadow.clone(), address, len));
        }
    }

    /// Shaders of the list write `[address, address + len)` of a buffer's
    /// host copy: copied back into title memory once the list has run
    /// (`marked`: the pages they mark).
    pub(crate) fn defer_gpu_write(
        &self,
        shadow: &std::sync::Arc<crate::shadow::Shadow>,
        address: usize,
        len: usize,
        marked: bool,
    ) {
        let memory = self.memory_writer.load(Ordering::Relaxed);
        self.deferred()
            .copies
            .push(Step::GpuWrite(shadow.clone(), address, len, marked, memory));
    }

    /// Shaders of the list take views from the heap itself
    /// (`crate::bindless`): what the heap's views reach is brought up to
    /// date at submission (once until a title copy intervenes).
    pub(crate) fn defer_heap(&self, viewed: &std::sync::Arc<crate::bindless::Viewed>) {
        let mut deferred = self.deferred();
        let key = (std::sync::Arc::as_ptr(viewed) as usize, usize::MAX, 0);
        if deferred.refreshed.insert(key) {
            deferred.copies.push(Step::Heap(viewed.clone()));
        }
    }

    /// Shaders of the list read title memory through the page table of
    /// view heap `heap` (`crate::memory::prepare` at submission).
    pub(crate) fn defer_memory(&self, heap: usize) {
        self.deferred().copies.push(Step::Memory(heap));
    }

    /// After the list has run, before its writes go back: `check` says
    /// whether its shaders read title memory with no copy.
    pub(crate) fn defer_missed(&self, check: impl Fn() -> bool + Send + 'static) {
        self.deferred().missed.push(Box::new(check));
    }

    /// Runs `work` once the list has run (each submission), before its
    /// bottom-of-pipe writes. What it writes in title memory is not named:
    /// the submission completes before the next is prepared ([`barrier`]).
    pub(crate) fn defer_after(&self, work: impl Fn() + Send + 'static) {
        let mut d = self.deferred();
        d.after.push(Box::new(work));
        d.after_unnamed = true;
    }

    /// [`HostList::defer_after`] of `work` that writes title memory only in
    /// `writes` (`[start, end)` each): registered in flight like declared
    /// writes ([`HostList::flying`]), so with `VICTORY_EARLY_BARRIERS` later
    /// submissions wait for it only where they read there ([`AFTER_NAMED`]).
    pub(crate) fn defer_after_writing(&self, writes: Vec<(usize, usize)>, work: impl Fn() + Send + 'static) {
        let mut d = self.deferred();
        d.after.push(Box::new(work));
        d.after_writes.extend(writes);
    }

    /// Registers a per-list VRAM/RAM dirty bitmap pair to zero before
    /// execution and read back afterward.
    pub(crate) fn defer_vram_bitmap(&self, vram: ID3D12Resource, ram: ID3D12Resource) {
        self.deferred().vram_bitmaps.push((vram, ram));
    }

    /// Registers a probe run after bitmap readbacks to collect VRAM-backed
    /// shadow pages written through the title-memory page table.
    pub(crate) fn defer_vram_probe(
        &self,
        probe: impl Fn(&mut Vec<(ID3D12Resource, u64, usize, u64)>) + Send + 'static,
    ) {
        self.deferred().vram_probes.push(Box::new(probe));
    }

    /// Stores `what` (`bytes` of it) at `dst` once the list has run.
    pub(crate) fn defer_write(&self, dst: *mut u8, what: crate::queue::Written, bytes: usize) {
        self.deferred().writes.push((dst as usize, what, bytes));
    }

    /// A memory value the list waits for before it starts.
    pub(crate) fn defer_wait(&self, wait: crate::hold::ValueWait) {
        self.deferred().waits.push(wait);
    }

    /// The memory values the list waits for before it starts.
    pub(crate) fn waits(&self) -> Vec<crate::hold::ValueWait> {
        self.deferred().waits.clone()
    }

    /// Whether nothing has been recorded since the list was made or reset
    /// that a wait recorded now would have to follow: no work, texture
    /// use, bottom-of-pipe write or pipeline (state settings alone carry
    /// over).
    pub(crate) fn is_empty(&self) -> bool {
        let d = self.deferred();
        let deferred_empty = d.copies.is_empty() && d.uploads.is_empty() && d.writes.is_empty() && d.after.is_empty() && d.missed.is_empty();
        drop(d);
        deferred_empty
            && self.states.lock().unwrap_or_else(|e| e.into_inner()).is_empty()
            && self.twins.lock().unwrap_or_else(|e| e.into_inner()).is_empty()
            && self.pipelines.lock().unwrap_or_else(|e| e.into_inner()).is_empty()
    }

    /// A mapped upload buffer of `size` bytes, alive until the list is
    /// reset.
    ///
    /// shortcut: one buffer per copy; pool them once uploads are frequent.
    pub(crate) fn upload(&self, host: &Host, size: u64) -> Option<(ID3D12Resource, *mut u8)> {
        let (buffer, data) = host.upload_buffer(size)?;
        name(&buffer, || format!("list upload {size:#x}"));
        self.deferred().uploads.push(buffer.clone());
        Some((buffer, data))
    }

    /// A mapped readback buffer of `size` bytes, alive until the list is
    /// reset.
    pub(crate) fn readback(&self, host: &Host, size: u64) -> Option<(ID3D12Resource, *mut u8)> {
        let (buffer, data) = host.mapped_buffer(size, true)?;
        name(&buffer, || format!("list readback {size:#x}"));
        self.deferred().uploads.push(buffer.clone());
        Some((buffer, data))
    }

    /// `len` bytes of upload memory (256-byte aligned) until the list is
    /// reset: the host GPU address and the CPU pointer.
    pub(crate) fn scratch(&self, len: u64) -> Option<(u64, *mut u8)> {
        let host = host()?;
        let mut s = self.scratch.lock().unwrap_or_else(|e| e.into_inner());
        let len = len.max(1).checked_next_multiple_of(256)?;
        if len > SCRATCH_CHUNK {
            let (buffer, data) = host.upload_buffer(len)?;
            name(&buffer, || format!("list scratch {len:#x}"));
            // SAFETY: a live buffer.
            let gpu = unsafe { buffer.GetGPUVirtualAddress() };
            self.deferred().uploads.push(buffer);
            return Some((gpu, data));
        }
        while let Some(chunk) = s.chunks.get(s.current) {
            if len <= chunk.capacity - s.used {
                break;
            }
            (s.current, s.used) = (s.current + 1, 0);
        }
        if s.current == s.chunks.len() {
            let next = s.chunks.last().map_or(FIRST_SCRATCH_CHUNK, |c| 2 * c.capacity);
            let capacity = len.next_multiple_of(FIRST_SCRATCH_CHUNK).max(next).min(SCRATCH_CHUNK);
            let (buffer, data) = host.upload_buffer(capacity)?;
            name(&buffer, || format!("list scratch {capacity:#x}"));
            // SAFETY: a live buffer.
            let gpu = unsafe { buffer.GetGPUVirtualAddress() };
            #[cfg(feature = "diagnostics")]
            tracing::debug!(target: "victory_gpu::scratch", list = ?std::ptr::from_ref(self),
                chunk = s.current, requested = len, capacity, "allocated list scratch");
            s.chunks.push(ScratchChunk { _buffer: buffer, mapped: data as usize, gpu, capacity });
            s.used = 0;
        }
        let chunk = &s.chunks[s.current];
        let at = s.used;
        let result = (chunk.gpu + at, (chunk.mapped + at as usize) as *mut u8);
        s.used += len;
        Some(result)
    }

    /// The host address of the list's indirect-work budget, zeroed at each
    /// submission of the list from the first use after a reset on.
    pub(crate) fn budget(&self) -> Option<u64> {
        let (buffer, data) = self
            .budget
            .get_or_init(|| {
                let (b, d) = host()?.shared_buffer(256)?;
                Some((b, d as usize))
            })
            .as_ref()?;
        let mut deferred = self.deferred();
        if !deferred.budget_zeroed {
            deferred.budget_zeroed = true;
            let at = *data;
            // SAFETY: the list's own mapped word, written before it runs.
            deferred
                .copies
                .push(Step::Host(Box::new(move || unsafe { (at as *mut u32).write_volatile(0) })));
        }
        // SAFETY: a live buffer.
        Some(unsafe { buffer.GetGPUVirtualAddress() })
    }

    /// The list's indirect work so far, after it has run.
    #[cfg(test)]
    pub(crate) fn budget_spent(&self) -> Option<u32> {
        let &(_, data) = self.budget.get()?.as_ref()?;
        // SAFETY: the list's own mapped word; the list has run.
        Some(unsafe { (data as *const u32).read_volatile() })
    }

    /// The already-created indirect-work budget buffer and its GPU address.
    pub(crate) fn budget_resource(&self) -> Option<(&ID3D12Resource, u64)> {
        let (buffer, _) = self.budget.get()?.as_ref()?;
        // SAFETY: a live buffer.
        Some((buffer, unsafe { buffer.GetGPUVirtualAddress() }))
    }

    pub(crate) fn hoisted(&self) -> std::sync::MutexGuard<'_, crate::indirect::HoistedClamps> {
        self.hoisted_indirect.lock().unwrap_or_else(|e| e.into_inner())
    }

    /// Ends the hoistable prefix of indirect clamps on this list, returning
    /// its hoisted argument chunks from `INDIRECT_ARGUMENT` back to `COMMON`
    /// once so later GPU writes and subsequent submissions see `COMMON`.
    pub(crate) fn end_hoisted_indirect(&self) {
        self.hoisted().end(&self.list);
    }

    /// The list's argument buffer, at least `bytes` long, common between
    /// its uses (`crate::indirect`).
    pub(crate) fn arguments(&self, bytes: u64) -> Option<ID3D12Resource> {
        let mut arguments = self.arguments.lock().unwrap_or_else(|e| e.into_inner());
        // SAFETY: a plain query on a live buffer.
        if let Some(b) = arguments.as_ref().filter(|b| unsafe { b.GetDesc() }.Width >= bytes) {
            return Some(b.clone());
        }
        let (made, _) = host()?.shared_buffer(bytes.next_power_of_two().max(1 << 16))?;
        name(&made, || "indirect arguments".into());
        // Commands recorded earlier may still name the smaller one.
        if let Some(old) = arguments.replace(made.clone()) {
            self.keep(&old);
        }
        Some(made)
    }

    /// The list's window of indirect draws clamped together.
    pub(crate) fn clamp_window(&self) -> std::sync::MutexGuard<'_, crate::indirect::Window> {
        self.clamp_window.lock().unwrap_or_else(|e| e.into_inner())
    }

    /// Keeps `object` alive until the list is reset.
    pub(crate) fn keep(&self, object: &impl Interface) {
        if let Ok(object) = object.cast::<windows::core::IUnknown>() {
            self.deferred().kept.push(object);
        }
    }

    /// Whether this is the first time since the reset that the list asks
    /// about `key` (a texture to bring up to date).
    pub(crate) fn first_use(&self, key: usize) -> bool {
        self.deferred().textures.insert(key)
    }

    /// Copies `len` bytes of title memory from `src` to `dst` when the list
    /// is submitted, in order with its other copies, where both are
    /// committed (refreshes after it copy again).
    pub(crate) fn defer_copy_memory(&self, dst: *mut u8, src: *const u8, len: usize) {
        let mut deferred = self.deferred();
        deferred.copies.push(Step::CopyMemory(dst as usize, src as usize, len));
        deferred.refreshed.clear();
    }

    /// As [`HostList::defer`], for work that writes only host memory.
    pub(crate) fn defer_host(&self, work: impl Fn() + Send + 'static) {
        self.deferred().copies.push(Step::Host(Box::new(work)));
    }

    /// Brings a texture's staging buffer and host texture `copy` up to date
    /// when the list is submitted (`crate::bindless::uploads`), keeping
    /// both alive until the list is reset.
    pub(crate) fn defer_texture(
        &self,
        staging: std::sync::Arc<crate::texture::Staging>,
        copy: std::sync::Arc<crate::resource::HostTexture>,
    ) {
        self.deferred().copies.push(Step::Texture(staging, copy));
    }

    /// Runs the list's submission-time steps for submission `owner`;
    /// refreshes go into `batch` (the title's big buffers, a 174 MB one
    /// among them, are read by most lists of a submission, through many
    /// ranges). Title memory read that a submission in flight writes on
    /// completion is read once `settle` has completed it ([`Settled`]).
    fn run_deferred(&self, batch: &mut Batch, owner: u64, settle: &mut dyn FnMut(u64, Settled)) {
        #[cfg(feature = "diagnostics")]
        use crate::copystat::{C, add};
        for step in &self.deferred().copies {
            let started = std::time::Instant::now();
            #[cfg(feature = "diagnostics")]
            let _timed = crate::watch::Timed(
                started,
                match step {
                    Step::Refresh(..) => C::StepRefreshUs,
                    Step::Copy(..) | Step::CopyMemory(..) | Step::Fill(..) => C::StepCopyUs,
                    Step::Host(_) => C::StepWorkUs,
                    Step::GpuWrite(..) => C::StepGpuWriteUs,
                    Step::Texture(..) => C::StepTextureUs,
                    Step::Heap(_) => C::StepHeapUs,
                    Step::Memory(_) => C::StepMemoryUs,
                },
            );
            let copied = match step {
                &Step::Copy(dst, src, len) | &Step::CopyMemory(dst, src, len) => {
                    #[cfg(feature = "diagnostics")]
                    {
                        add(C::Copies, 1);
                        add(C::CopyBytes, len as u64);
                    }
                    batch.title_written(owner, settle);
                    let crossed = [(dst, len), (src, len)]
                        .into_iter()
                        .filter_map(|(at, n)| crate::shadow::crossing(at, at + n, 0).map(|c| c.0))
                        .max();
                    if let Some(o) = crossed {
                        settle(o, Settled::Copy);
                    }
                    #[cfg(feature = "diagnostics")]
                    let _census = {
                        let kind = crate::census::as_kind(crate::census::Kind::TitleCopy, 0);
                        crate::census::touch(src, len);
                        crate::census::written(dst, len, crate::census::Writer::Kernel);
                        kind
                    };
                    if matches!(step, Step::Copy(..)) {
                        copy_mapped(dst as *mut u8, src as *const u8, len, true);
                    } else if crate::texture::committed(src as *const u8, len)
                        && crate::texture::committed(dst as *const u8, len)
                    {
                        // SAFETY: both ranges are committed title memory.
                        unsafe { std::ptr::copy(src as *const u8, dst as *mut u8, len) };
                    }
                    len
                }
                &Step::Fill(dst, src, len) => {
                    #[cfg(feature = "diagnostics")]
                    {
                        add(C::Copies, 1);
                        add(C::CopyBytes, len as u64);
                    }
                    if let Some((o, _)) = crate::shadow::crossing(src, src + len, 0) {
                        settle(o, Settled::Fill);
                    }
                    #[cfg(feature = "diagnostics")]
                    let _census = crate::census::as_kind(crate::census::Kind::TitleCopy, 0);
                    copy_mapped(dst as *mut u8, src as *const u8, len, false);
                    len
                }
                Step::Refresh(shadow, address, len) => {
                    #[cfg(feature = "diagnostics")]
                    {
                        add(C::Refreshes, 1);
                        add(C::RefreshBytes, *len as u64);
                    }
                    batch.add(shadow, *address, *len);
                    continue;
                }
                Step::Host(work) => {
                    work();
                    #[cfg(feature = "diagnostics")]
                    {
                        add(C::Runs, 1);
                        add(C::RunUs, started.elapsed().as_micros() as u64);
                    }
                    continue;
                }
                Step::GpuWrite(shadow, address, len, marked, _) => {
                    // The shaders start from the title's data, which a
                    // snapshot must hold (marked ranges keep none).
                    if !marked {
                        batch.flush(owner, settle);
                    }
                    shadow.writing(owner, *address, *len, *marked);
                    continue;
                }
                // Brought up to date before the submission's lists were made.
                Step::Texture(..) => continue,
                Step::Heap(viewed) => {
                    crate::bindless::refresh(viewed, |s, a, l, clean| batch.add_reached(s, a, l, clean));
                    continue;
                }
                &Step::Memory(heap) => {
                    crate::memory::prepare(heap, |s, a, l, clean| batch.add_reached(s, a, l, clean));
                    continue;
                }
            };
            PREPARING[0].fetch_add(started.elapsed().as_micros() as u64, Ordering::Relaxed);
            PREPARING[1].fetch_add(copied as u64, Ordering::Relaxed);
        }
    }
}

/// A submission's refreshes not done yet, merged per buffer copy into the
/// range that covers them: done before a step that writes title memory or
/// gives a range to shaders, and before the lists run.
///
/// Also the textures brought up to date in the submission so far: most
/// lists of a world submission sample the same textures, and a texture in
/// physical-page memory is hashed whole each time it is checked. Checked
/// once, it is checked again only after a step that writes title memory.
///
/// A copy's range brought up to date is not copied again in the submission
/// until a step writes title memory: every list of a world submission using
/// title memory asks for every copy whole. What the title's threads write
/// meanwhile is for later submissions (on the console it would race the
/// GPU).
///
/// Submissions in flight that write title memory a refresh reads complete
/// before the refreshes are done ([`Settled`]): through another copy of it
/// or by bottom-of-pipe writes (asked when the refresh is added, clean or
/// not: their writes make it change), or where the title changed pages
/// under their ranges of the same copy (`crate::shadow::Shadow::update`).
#[derive(Default)]
struct Batch {
    refreshes: Vec<(std::sync::Arc<crate::shadow::Shadow>, usize, usize)>,
    /// Where each copy's refresh is in `refreshes`, by copy.
    pending: std::collections::HashMap<usize, usize>,
    /// The range of each copy brought up to date since the last title
    /// write, by copy (held, so that its address names no other).
    current: std::collections::HashMap<usize, (std::sync::Arc<crate::shadow::Shadow>, usize, usize)>,
    /// The latest submission in flight writing title memory that a refresh
    /// in `refreshes` reads other than through its copy, and whether by
    /// bottom-of-pipe writes ([`crate::shadow::crossing`]).
    crossed: Option<(u64, bool)>,
    /// The same over refreshes other than of clean copies of heaps and page
    /// tables crossing writes in flight ([`Batch::add_reached`]): a later
    /// `crossed` is a settle those alone make (counted).
    crossed_dirty: Option<(u64, bool)>,
}

impl Batch {
    fn add(&mut self, shadow: &std::sync::Arc<crate::shadow::Shadow>, address: usize, len: usize) {
        self.add_as(shadow, address, len, false);
    }

    /// [`Batch::add`] of a copy of a heap or page table (`clean`: the title
    /// has not written it since it took it, and a submission in flight
    /// writes its memory through another copy or after it ran,
    /// `crate::shadow::refresh_where_needed`).
    fn add_reached(&mut self, shadow: &std::sync::Arc<crate::shadow::Shadow>, address: usize, len: usize, clean: bool) {
        #[cfg(feature = "diagnostics")]
        crate::copystat::add(crate::copystat::C::CleanCrossed, u64::from(clean));
        self.add_as(shadow, address, len, clean);
    }

    fn add_as(&mut self, shadow: &std::sync::Arc<crate::shadow::Shadow>, address: usize, len: usize, clean: bool) {
        let (key, end) = (std::sync::Arc::as_ptr(shadow) as usize, address + len);
        let crossed = crate::shadow::crossing(address, end, key);
        if let Some(c) = crossed {
            if self.crossed.is_none_or(|k| k.0 < c.0) {
                self.crossed = Some(c);
            }
            if !clean && self.crossed_dirty.is_none_or(|k| k.0 < c.0) {
                self.crossed_dirty = Some(c);
            }
        }
        // Nothing to copy: a world submission asks for every copy its heaps'
        // views and page tables reach (~9,000), a few of them written.
        if crossed.is_none() && shadow.clean() {
            return;
        }
        if crossed.is_none() && self.current.get(&key).is_some_and(|&(_, a, b)| a <= address && end <= b) {
            return;
        }
        match self.pending.get(&key) {
            Some(&i) => {
                let (_, from, to) = &mut self.refreshes[i];
                (*from, *to) = ((*from).min(address), (*to).max(end));
            }
            None => {
                self.pending.insert(key, self.refreshes.len());
                self.refreshes.push((shadow.clone(), address, end));
            }
        }
    }

    /// Before a step that writes title memory: pending refreshes are done
    /// first, and textures are checked again after it.
    fn title_written(&mut self, owner: u64, settle: &mut dyn FnMut(u64, Settled)) {
        self.flush(owner, settle);
        self.current.clear();
    }

    /// Does the refreshes for submission `owner`, once those in flight
    /// whose writes they read have completed (`settle`); returns the bytes
    /// copied.
    fn flush(&mut self, owner: u64, settle: &mut dyn FnMut(u64, Settled)) -> usize {
        #[cfg(feature = "diagnostics")]
        let _timed = crate::watch::Timed(std::time::Instant::now(), crate::copystat::C::FlushUs);
        let dirty = self.crossed_dirty.take();
        if let Some((o, label)) = self.crossed.take() {
            // Only clean copies crossing writes in flight asked for it (or
            // for a later submission): what correctness adds.
            #[cfg(feature = "diagnostics")]
            if dirty.is_none_or(|d| d.0 < o) {
                crate::copystat::add(
                    if label { crate::copystat::C::SettleLabelClean } else { crate::copystat::C::SettleAliasClean },
                    1,
                );
            }
            settle(o, if label { Settled::Label } else { Settled::Alias });
        }
        let _ = dirty;
        self.pending.clear();
        let mut total = 0;
        for (shadow, from, to) in self.refreshes.drain(..) {
            let started = std::time::Instant::now();
            let copied = loop {
                match shadow.update(owner, from, to - from) {
                    Ok(copied) => break copied,
                    Err(o) => settle(o, Settled::Same),
                }
            };
            let us = started.elapsed().as_micros() as u64;
            #[cfg(feature = "diagnostics")]
            crate::copystat::range(from, to - from, us, 0);
            PREPARING[0].fetch_add(us, Ordering::Relaxed);
            PREPARING[1].fetch_add(copied as u64, Ordering::Relaxed);
            total += copied;
            // Kept: the union with what was current if they touch, else
            // the larger.
            let key = std::sync::Arc::as_ptr(&shadow) as usize;
            let (_, a, b) = self.current.entry(key).or_insert((shadow, from, to));
            if from <= *b && *a <= to {
                (*a, *b) = ((*a).min(from), (*b).max(to));
            } else if to - from > *b - *a {
                (*a, *b) = (from, to);
            }
        }
        total
    }
}

/// Copies title memory at submission. The ranges were checked against their
/// resources when recorded, but the title may since have unmapped pages of
/// them (tiled memory, page mappings), which read as zeros here; a title
/// destination (`title_dst`) that is not all mapped takes nothing from its
/// first unmapped page on.
///
/// The copy runs in the kernel (`ReadProcessMemory`), which reports an
/// unmapped page of the source, or of the destination it writes, instead of
/// faulting. Asking `VirtualQuery` first cost about 2 ms per copy in the
/// title's big pools (it walks the page tables to the region's end), which
/// made copies run at 100-200 MB/s; a title destination is asked only once
/// its copy has failed (asked before each, the world's 80 copies per frame
/// cost 60 ms).
pub(crate) fn copy_mapped(dst: *mut u8, src: *const u8, len: usize, title_dst: bool) {
    #[cfg(feature = "diagnostics")]
    match crate::copystat::mode() {
        1 => {
            // SAFETY: timing only: the caller vouches for the source.
            unsafe { std::ptr::copy(src, dst, len) };
            return;
        }
        2 => return,
        _ => {}
    }
    if len == 0 {
        return;
    }
    #[cfg(feature = "diagnostics")]
    crate::census::touch(src as usize, len);
    // The kernel writes the destination: it must not be guarded meanwhile.
    let _writing = title_dst.then(|| crate::watch::writing(dst as usize, len));
    // Nor may the source be a page shaders wrote that has not gone back: its
    // copy's bytes are read instead (`VICTORY_OWNER`), or go back first.
    #[cfg(feature = "diagnostics")]
    let moved = std::time::Instant::now();
    let whole = crate::watch::serve(src as usize, dst, len, &mut |s, d, n| read(s as *const u8, d, n));
    #[cfg(feature = "diagnostics")]
    crate::copystat::add(crate::copystat::C::MoveUs, moved.elapsed().as_micros() as u64);
    if whole {
        return;
    }
    // The copy stopped at an unmapped page. One of a title destination ends
    // it: the retries below write the destination themselves (zeros for
    // what cannot be read), and would fault there.
    if title_dst && !crate::texture::committed(dst, len) {
        return;
    }
    #[cfg(feature = "diagnostics")]
    crate::copystat::add(crate::copystat::C::Fallbacks, 1);
    #[cfg(feature = "diagnostics")]
    let _timed = crate::watch::Timed(std::time::Instant::now(), crate::copystat::C::FallbackUs);
    // Something is unmapped: page by page (64 KiB, then 4 KiB in a piece
    // that fails), zeros for what cannot be read; lazy pages served as above.
    let piece = |at: usize, to: *mut u8, n: usize| crate::watch::serve(at, to, n, &mut |s, d, n| read(s as *const u8, d, n));
    for_pieces(src as usize, len, 64 << 10, |at, n| {
        let to = dst.wrapping_add(at - src as usize);
        if !piece(at, to, n) {
            for_pieces(at, n, 4 << 10, |at, n| {
                let to = dst.wrapping_add(at - src as usize);
                if !piece(at, to, n) {
                    #[cfg(feature = "diagnostics")]
                    crate::copystat::add(crate::copystat::C::Unmapped, n as u64);
                    // SAFETY: inside the destination, which is mapped.
                    unsafe { to.write_bytes(0, n) };
                }
            });
        }
    });
}

/// Reads title memory at `src` into `dst`; false if a page of it is not
/// readable. Pages shaders wrote that have not gone back are read from the
/// copy that wrote them, or go back first (`crate::watch::serve`).
pub(crate) fn read_into(src: *const u8, dst: &mut [u8]) -> bool {
    #[cfg(feature = "diagnostics")]
    crate::census::touch(src as usize, dst.len());
    crate::watch::serve(src as usize, dst.as_mut_ptr(), dst.len(), &mut |s, d, n| read(s as *const u8, d, n))
}

/// [`read_into`] with no look at pages that have not gone back (a page with
/// no access fails the read): for holders of the guard lock, which that look
/// takes.
pub(crate) fn read_raw(src: *const u8, dst: &mut [u8]) -> bool {
    read(src, dst.as_mut_ptr(), dst.len())
}

/// Copies `len` bytes from `src` to `dst` in the kernel: false if a source
/// page is not readable or a destination page not writable (the bytes
/// before that page may have been copied).
fn read(src: *const u8, dst: *mut u8, len: usize) -> bool {
    use windows_sys::Win32::System::Diagnostics::Debug::ReadProcessMemory;
    use windows_sys::Win32::System::Threading::GetCurrentProcess;
    let mut done = 0;
    // SAFETY: the destination is `len` bytes the caller may write, host
    // memory or title memory; the kernel checks both ranges, and a page it
    // cannot read or write fails the call instead of faulting.
    unsafe { ReadProcessMemory(GetCurrentProcess(), src.cast(), dst.cast(), len, &mut done) != 0 }
}

/// Copies host memory into title memory after a submission (what shaders
/// wrote), leaving pages that are not mapped alone.
///
/// Through [`read`]: the kernel refuses an unmapped destination page as it
/// does a source page. `WriteProcessMemory` makes the same copy, but asks
/// for the destination's region first, at `VirtualQuery`'s cost (see
/// [`copy_mapped`]): 0.35 ms for each piece of the world's copy-back, 39 ms
/// per frame.
pub(crate) fn write_mapped(dst: *mut u8, src: *const u8, len: usize) {
    let write = |at: usize, n: usize| read(src.wrapping_add(at - dst as usize), at as *mut u8, n);
    if len == 0 || write(dst as usize, len) {
        return;
    }
    for_pieces(dst as usize, len, 64 << 10, |at, n| {
        if !write(at, n) {
            for_pieces(at, n, 4 << 10, |at, n| {
                write(at, n);
            });
        }
    });
}

/// Calls `f` with the pieces of `[start, start + len)` cut at multiples of
/// `step`.
fn for_pieces(start: usize, len: usize, step: usize, mut f: impl FnMut(usize, usize)) {
    let end = start + len;
    let mut at = start;
    while at < end {
        let next = ((at / step + 1) * step).min(end);
        f(at, next - at);
        at = next;
    }
}

#[cfg(test)]
mod title_copies {
    use super::{copy_mapped, write_mapped};
    use windows_sys::Win32::System::Memory::{
        MEM_COMMIT, MEM_RELEASE, MEM_RESERVE, PAGE_NOACCESS, PAGE_READONLY, PAGE_READWRITE,
        VirtualAlloc, VirtualFree, VirtualProtect,
    };

    const BLOCK: usize = 64 << 10;

    /// Three reserved blocks, the first and the last committed and filled
    /// with `fill`: title memory with an unmapped block in it.
    fn holed(fill: u8) -> *mut u8 {
        // SAFETY: a fresh reservation; the committed blocks are ours to fill.
        unsafe {
            let base =
                VirtualAlloc(std::ptr::null(), 3 * BLOCK, MEM_RESERVE, PAGE_NOACCESS).cast::<u8>();
            assert!(!base.is_null());
            for block in [0, 2] {
                let at = base.add(block * BLOCK);
                assert!(!VirtualAlloc(at.cast(), BLOCK, MEM_COMMIT, PAGE_READWRITE).is_null());
                at.write_bytes(fill, BLOCK);
            }
            base
        }
    }

    fn block<'a>(base: *const u8, n: usize) -> &'a [u8] {
        // SAFETY: tests ask for committed blocks only.
        unsafe { std::slice::from_raw_parts(base.add(n * BLOCK), BLOCK) }
    }

    fn release(base: *mut u8) {
        // SAFETY: a reservation of `holed`.
        assert_ne!(unsafe { VirtualFree(base.cast(), 0, MEM_RELEASE) }, 0);
    }

    fn pattern() -> Vec<u8> {
        (0..3 * BLOCK).map(|i| (i % 251) as u8 + 1).collect()
    }

    #[test]
    fn title_copies_reach_mapped_destinations_and_stop_at_unmapped_pages() {
        let from = pattern();
        // All mapped: copied, through a destination that was never asked.
        let to = holed(0xee);
        copy_mapped(to, from.as_ptr(), BLOCK, true);
        assert_eq!(block(to, 0), &from[..BLOCK]);
        release(to);
        // An unmapped block in the destination: nothing faults, and nothing
        // arrives from that block on.
        let to = holed(0xee);
        copy_mapped(to, from.as_ptr(), 3 * BLOCK, true);
        assert!(block(to, 2).iter().all(|&b| b == 0xee));
        release(to);
        // An unmapped block in the source reads as zeros.
        let holes = holed(7);
        let mut to = vec![0xeeu8; 3 * BLOCK];
        copy_mapped(to.as_mut_ptr(), holes, 3 * BLOCK, true);
        assert!(to[..BLOCK].iter().all(|&b| b == 7));
        assert!(to[BLOCK..2 * BLOCK].iter().all(|&b| b == 0));
        assert!(to[2 * BLOCK..].iter().all(|&b| b == 7));
        release(holes);
    }

    #[test]
    fn copy_back_writes_mapped_pages_only() {
        let from = pattern();
        let to = holed(0xee);
        write_mapped(to, from.as_ptr(), 3 * BLOCK);
        assert_eq!(block(to, 0), &from[..BLOCK]);
        assert_eq!(block(to, 2), &from[2 * BLOCK..]);
        release(to);
        // Neither aligned nor whole: the bytes around the range stay.
        let to = holed(0xee);
        // SAFETY: inside the first block.
        write_mapped(unsafe { to.add(100) }, from.as_ptr(), BLOCK - 300);
        let first = block(to, 0);
        assert_eq!(first[100..BLOCK - 200], from[..BLOCK - 300]);
        let around = first[..100].iter().chain(&first[BLOCK - 200..]);
        assert!(around.into_iter().all(|&b| b == 0xee));
        // A page that is mapped but not writable is not written either
        // (guarded title blocks are opened before they are copied back).
        let mut old = 0;
        // SAFETY: a committed block of ours.
        let protected =
            unsafe { VirtualProtect(to.add(2 * BLOCK).cast(), BLOCK, PAGE_READONLY, &mut old) };
        assert_ne!(protected, 0);
        // SAFETY: the third block.
        write_mapped(unsafe { to.add(2 * BLOCK) }, from.as_ptr(), BLOCK);
        assert!(block(to, 2).iter().all(|&b| b == 0xee));
        release(to);
    }
}

impl Host {
    /// Whether the host adapter uses unified memory architecture (APUs, WARP).
    pub(crate) fn is_uma(&self) -> bool {
        static UMA: OnceLock<bool> = OnceLock::new();
        *UMA.get_or_init(|| {
            let mut arch = D3D12_FEATURE_DATA_ARCHITECTURE::default();
            // SAFETY: valid D3D12 feature query.
            unsafe {
                self.device.CheckFeatureSupport(
                    D3D12_FEATURE_ARCHITECTURE,
                    (&raw mut arch).cast(),
                    size_of_val(&arch) as u32,
                )
            }
            .is_ok()
                && arch.UMA.as_bool()
        })
    }

    /// Safe VRAM budget (bytes) for hot shadow buffers on discrete GPUs.
    pub(crate) fn vram_shadow_budget(&self) -> u64 {
        if self.is_uma() {
            return 0;
        }
        static BUDGET: OnceLock<u64> = OnceLock::new();
        *BUDGET.get_or_init(|| {
            let Some([local, _]) = video_memory() else {
                return 0;
            };
            // Cap at 25% of local DXGI budget (~2.5-3.0 GB on a 12 GB card) so
            // streaming textures never get evicted.
            ((local.1 * 1_000_000.0) as u64) / 4
        })
    }

    /// A buffer in the upload heap, mapped for writing.
    /// A GPU-only buffer of `size` bytes in the copy-destination state.
    pub(crate) fn default_buffer(&self, size: u64) -> Option<ID3D12Resource> {
        self.default_buffer_with(size, D3D12_RESOURCE_FLAG_NONE, D3D12_RESOURCE_STATE_COPY_DEST)
    }

    /// A GPU-only buffer of `size` bytes with unordered-access support in the
    /// common state (promotes implicitly to SRV/CBV/UAV/copy/indirect).
    pub(crate) fn default_uav_buffer(&self, size: u64) -> Option<ID3D12Resource> {
        self.default_buffer_with(
            size,
            D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
            D3D12_RESOURCE_STATE_COMMON,
        )
    }

    fn default_buffer_with(
        &self,
        size: u64,
        flags: D3D12_RESOURCE_FLAGS,
        state: D3D12_RESOURCE_STATES,
    ) -> Option<ID3D12Resource> {
        let heap = D3D12_HEAP_PROPERTIES {
            Type: D3D12_HEAP_TYPE_DEFAULT,
            ..Default::default()
        };
        let desc = D3D12_RESOURCE_DESC {
            Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
            Width: size.max(1),
            Height: 1,
            DepthOrArraySize: 1,
            MipLevels: 1,
            SampleDesc: DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
            Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
            Flags: flags,
            ..Default::default()
        };
        let mut buffer: Option<ID3D12Resource> = None;
        // SAFETY: plain creation.
        let made = unsafe {
            self.device.CreateCommittedResource(
                &heap,
                D3D12_HEAP_FLAG_NONE,
                &desc,
                state,
                None,
                &mut buffer,
            )
        };
        made.ok().and(buffer)
    }

    /// Streams CPU shadow ranges into VRAM via a reusable 64 MiB upload ring,
    /// waited for: a submission's uploads past [`STAGED_MAX`] (they go
    /// before its own, [`Host::vram_before`]), and tests.
    fn upload_vram_copies(&self, uploads: &[crate::shadow::VramCopy]) {
        if uploads.is_empty() {
            return;
        }
        const RING_BYTES: u64 = 64 << 20;
        static RING: Mutex<Option<(ID3D12Resource, usize)>> = Mutex::new(None);
        let (ring_buf, ring_ptr) = {
            let mut guard = RING.lock().unwrap_or_else(|e| e.into_inner());
            let (buf, ptr) = guard.get_or_insert_with(|| {
                let (b, p) = self.upload_buffer(RING_BYTES).expect("vram upload ring");
                (b, p as usize)
            });
            (buf.clone(), *ptr as *mut u8)
        };

        let mut offset = 0u64;
        let mut batch: Vec<(ID3D12Resource, u64, u64, u64)> = Vec::new();
        let flush = |batch: &mut Vec<(ID3D12Resource, u64, u64, u64)>| {
            if batch.is_empty() {
                return;
            }
            self.run_copy_list(
                |list| {
                    for (dst, dst_off, src_off, len) in batch.iter() {
                        // SAFETY: valid upload ring and destination VRAM buffer regions.
                        unsafe { list.CopyBufferRegion(dst, *dst_off, &ring_buf, *src_off, *len) };
                    }
                },
                true,
            );
            batch.clear();
        };

        for u in uploads {
            let mut done = 0u64;
            while done < u.len {
                let mut avail = RING_BYTES - offset;
                if avail == 0 {
                    flush(&mut batch);
                    offset = 0;
                    avail = RING_BYTES;
                }
                let chunk = (u.len - done).min(avail);
                // SAFETY: `u.src_ptr + done` is kept live by `u._cpu` and `ring_ptr + offset` is mapped.
                unsafe {
                    std::ptr::copy_nonoverlapping(
                        (u.src_ptr as *const u8).add(done as usize),
                        ring_ptr.add(offset as usize),
                        chunk as usize,
                    );
                }
                batch.push((u.dst.clone(), u.dst_off + done, offset, chunk));
                offset = (offset + chunk).next_multiple_of(16).min(RING_BYTES);
                done += chunk;
            }
        }
        flush(&mut batch);
    }

    /// Copies GPU-written VRAM ranges back to their CPU mirrors via a reusable
    /// 32 MiB readback ring, waited for: on a submission's completion
    /// (`Host::finish`), the ranges known only then and those past
    /// [`STAGED_MAX`]. Reads whose bytes the mirror holds already are noted
    /// in `unchanged` by mirror address ([`into_mirror`]).
    fn readback_vram_copies(&self, reads: &[VramRead], unchanged: &mut std::collections::HashSet<usize>) {
        if reads.is_empty() {
            return;
        }
        const RING_BYTES: u64 = 32 << 20;
        static RING: Mutex<Option<(ID3D12Resource, usize)>> = Mutex::new(None);
        let (ring_buf, ring_ptr) = {
            let mut guard = RING.lock().unwrap_or_else(|e| e.into_inner());
            let (buf, ptr) = guard.get_or_insert_with(|| {
                let (b, p) = self.mapped_buffer(RING_BYTES, true).expect("vram readback ring");
                (b, p as usize)
            });
            (buf.clone(), *ptr as *mut u8)
        };

        let mut offset = 0u64;
        // (source, its offset, ring offset, mirror, bytes, the whole read).
        type Chunk = (ID3D12Resource, u64, u64, usize, u64, bool);
        let mut batch: Vec<Chunk> = Vec::new();
        let mut flush = |batch: &mut Vec<Chunk>| {
            if batch.is_empty() {
                return;
            }
            self.run_copy_list(
                |list| {
                    for (src, src_off, dst_off, _, len, _) in batch.iter() {
                        // SAFETY: valid VRAM buffer and readback ring regions.
                        unsafe { list.CopyBufferRegion(&ring_buf, *dst_off, src, *src_off, *len) };
                    }
                },
                true,
            );
            for (_, _, ring_off, cpu_ptr, len, whole) in batch.drain(..) {
                // SAFETY: GPU finished writing `ring_buf`; `cpu_ptr` is valid for `len` bytes.
                unsafe { into_mirror(ring_ptr.add(ring_off as usize), cpu_ptr, len as usize, whole, unchanged) };
            }
        };

        for (src, src_off, cpu_ptr, total_len) in reads {
            let mut done = 0u64;
            while done < *total_len {
                let mut avail = RING_BYTES - offset;
                if avail == 0 {
                    flush(&mut batch);
                    offset = 0;
                    avail = RING_BYTES;
                }
                let chunk = (*total_len - done).min(avail);
                let whole = chunk == *total_len;
                batch.push((src.clone(), *src_off + done, offset, *cpu_ptr + done as usize, chunk, whole));
                offset = (offset + chunk).next_multiple_of(16).min(RING_BYTES);
                done += chunk;
            }
        }
        flush(&mut batch);
    }

    /// A copy list of ours, open (`COPY_LISTS`, or a new one).
    fn copy_list(&self) -> CopyList {
        let spare = COPY_LISTS.lock().unwrap_or_else(|e| e.into_inner()).pop();
        // SAFETY: a list back in the pool has completed: its allocator and
        // the list are free to reset; or plain creation.
        unsafe {
            if let Some(c) = spare {
                c.alloc.Reset().expect("reset copy allocator");
                c.list.Reset(&c.alloc, None).expect("reset copy list");
                return c;
            }
            let alloc: ID3D12CommandAllocator = self
                .device
                .CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT)
                .expect("a copy allocator");
            let list: ID3D12GraphicsCommandList = self
                .device
                .CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, &alloc, None)
                .expect("a copy list");
            name(&list, || "copies in video memory".into());
            CopyList { alloc, list }
        }
    }

    /// A staging buffer of at least `bytes` (`readback`: read-back memory):
    /// the smallest of `STAGING` that holds them, or a new one.
    fn staging(&self, readback: bool, bytes: u64) -> Option<Staging> {
        {
            let mut pool = STAGING.lock().unwrap_or_else(|e| e.into_inner());
            let pool = &mut pool[usize::from(readback)];
            if let Some(i) = smallest_fitting(pool.iter().map(|s| s.size), bytes) {
                return Some(pool.swap_remove(i));
            }
        }
        let size = bytes.next_power_of_two().max(1 << 20);
        let (buffer, data) = self.mapped_buffer(size, readback)?;
        name(&buffer, || {
            format!("copies in video memory, {}", if readback { "read back" } else { "uploads" })
        });
        Some(Staging {
            buffer,
            data: data as usize,
            size,
        })
    }

    /// Records, closed, the list that runs before a submission's own: the
    /// copies in video memory brought up to date from their mirrors (through
    /// staging held in `sync`), page tables, the lists' page bitmaps zeroed,
    /// and (`marks`: its shaders mark) the views' write marks in video
    /// memory cleared (`mark_bytes` of them) and their read marks as the CPU
    /// left them. Uploads past [`STAGED_MAX`] go first, waited for.
    fn vram_before(
        &self,
        uploads: Vec<crate::shadow::VramCopy>,
        bitmaps: &[(ID3D12Resource, ID3D12Resource)],
        marks: bool,
        mark_bytes: u64,
        sync: &mut VramSync,
    ) -> Option<ID3D12CommandList> {
        let views = marks && crate::shadow::has_vram_bitmaps();
        if uploads.is_empty() && !crate::memory::needs_vram_pre(bitmaps) && !views {
            return None;
        }
        let (n, bytes) = staged(uploads.iter().map(|u| u.len), STAGED_MAX);
        let staging = (bytes > 0).then(|| self.staging(false, bytes)).flatten();
        let n = if staging.is_some() { n } else { 0 };
        self.upload_vram_copies(&uploads[n..]);
        let c = self.copy_list();
        if let Some(s) = &staging {
            #[cfg(feature = "diagnostics")]
            let _timed = crate::watch::Timed(std::time::Instant::now(), crate::copystat::C::VramUpUs);
            #[cfg(feature = "diagnostics")]
            crate::copystat::add(crate::copystat::C::VramUpBytes, bytes);
            let mut at = 0u64;
            for u in &uploads[..n] {
                // SAFETY: `u.src_ptr` is live for `u.len` bytes (held by
                // `u._cpu`); the staging is mapped for `bytes`; the GPU copy
                // reads it after this, and it is held until it has run.
                unsafe {
                    std::ptr::copy_nonoverlapping(
                        u.src_ptr as *const u8,
                        (s.data as *mut u8).add(at as usize),
                        u.len as usize,
                    );
                    c.list.CopyBufferRegion(&u.dst, u.dst_off, &s.buffer, at, u.len);
                }
                at = (at + u.len).next_multiple_of(16);
            }
        }
        crate::memory::sync_vram_pre(&c.list, bitmaps);
        if views {
            crate::shadow::sync_bitmaps_pre(&c.list, mark_bytes);
        }
        // SAFETY: our open list.
        unsafe { c.list.Close() }.expect("close copy list");
        let run = c.list.cast().ok();
        sync.lists.push(c);
        sync.staging.extend(staging.map(|s| (false, s)));
        run
    }

    /// Records, closed, the list that runs after a submission's own and
    /// before its fence value: what its shaders wrote of copies in video
    /// memory (declared ranges not marked) into read-back staging, the
    /// lists' page bitmaps and (`marks`) the views' read marks into system
    /// memory, and (`capture`: bytes of each bitmap) the views' write marks
    /// its shaders set into staging (`crate::shadow::capture_write_marks`).
    /// On completion the staged bytes go into the mirrors
    /// ([`Host::vram_completed`]); ranges past [`STAGED_MAX`] are read back
    /// then. `takes`: completing it takes the views' write marks or reads the
    /// lists' page bitmaps, and the submission's kind: the pages its kind
    /// named last time ([`PREDICTED`]) are read back too, as far as staging
    /// goes.
    fn vram_after(
        &self,
        runs: &[&HostList],
        bitmaps: &[(ID3D12Resource, ID3D12Resource)],
        marks: bool,
        capture: Option<u64>,
        takes: Option<u64>,
        sync: &mut VramSync,
    ) -> Option<ID3D12CommandList> {
        let mut seen = std::collections::HashSet::new();
        let mut reads: Vec<VramRead> = Vec::new();
        for list in runs {
            for step in &list.deferred().copies {
                // Marked ranges of tracked copies keep no snapshot
                // (`Shadow::writing`): they go back by their marks, whose
                // pages are read back on completion.
                if let Step::GpuWrite(shadow, address, len, false, _) = step
                    && seen.insert((std::sync::Arc::as_ptr(shadow) as usize, *address, *len))
                    && let Some((cpu_ptr, base, start, buf_len)) = shadow.vram_readback()
                {
                    let s = (*address as u64).max(start);
                    let e = ((*address + *len) as u64).min(start + buf_len);
                    if e > s {
                        let off_in_shadow = s - start;
                        reads.push((shadow.buffer.clone(), base + off_in_shadow, cpu_ptr + off_in_shadow as usize, e - s));
                    }
                }
            }
        }
        let views = marks && crate::shadow::has_vram_bitmaps();
        let mut predicted = takes
            .and_then(|kind| PREDICTED.lock().unwrap_or_else(|e| e.into_inner()).get(&kind).cloned())
            .unwrap_or_default();
        sync.signature = takes;
        if reads.is_empty() && predicted.is_empty() && bitmaps.is_empty() && !views && capture.is_none() {
            return None;
        }
        let (n, _) = staged(reads.iter().map(|r| r.3), STAGED_MAX);
        sync.later.extend(reads.drain(n..));
        let (m, bytes) = staged(reads.iter().chain(&predicted).map(|r| r.3), STAGED_MAX);
        predicted.truncate(m - reads.len());
        // The write marks after the rest, whatever the cap.
        let marks_at = bytes.next_multiple_of(16);
        let room = marks_at + capture.map_or(0, crate::shadow::capture_bytes);
        let staging = (room > 0).then(|| self.staging(true, room)).flatten();
        if staging.is_none() {
            if room > marks_at {
                fail(format_args!("no read-back staging for a submission's write marks ({room} bytes)"));
            }
            sync.later.append(&mut reads);
            predicted.clear();
        }
        let c = self.copy_list();
        if let Some(s) = &staging {
            let mut at = 0u64;
            for (src, src_off, cpu_ptr, len) in &reads {
                // SAFETY: our open list; live buffers, `at + len` inside the staging.
                unsafe { c.list.CopyBufferRegion(&s.buffer, at, src, *src_off, *len) };
                sync.reads.push((s.data + at as usize, *cpu_ptr, *len as usize));
                at = (at + len).next_multiple_of(16);
            }
            for (src, src_off, _, len) in &predicted {
                // SAFETY: as above (the prediction holds the buffers); its
                // mirror pointers may be stale and are not used.
                unsafe { c.list.CopyBufferRegion(&s.buffer, at, src, *src_off, *len) };
                sync.predicted.insert((src.as_raw() as usize, *src_off, *len), s.data + at as usize);
                at = (at + len).next_multiple_of(16);
            }
        }
        if let Some(bytes) = capture {
            let placed = staging
                .as_ref()
                .map_or([None; 2], |s| crate::shadow::capture_write_marks(&c.list, &s.buffer, marks_at, bytes));
            sync.marks = Some((placed, bytes, staging.as_ref().map_or(0, |s| s.data)));
        }
        crate::memory::sync_vram_post(&c.list, bitmaps);
        if views {
            crate::shadow::sync_bitmaps_post(&c.list);
        }
        // SAFETY: our open list.
        unsafe { c.list.Close() }.expect("close copy list");
        let run = c.list.cast().ok();
        sync.lists.push(c);
        sync.staging.extend(staging.map(|s| (true, s)));
        run
    }

    /// First thing once a submission has run (`Host::finish`): what the list
    /// after it read back goes into the copies' mirrors; then the pages only
    /// the bitmaps name: the views' written pages (`marked`: their marks are
    /// taken now, its own as the list after it captured them, else from the
    /// bitmaps in system memory; `crate::shadow::copy_back_marks`) and those
    /// the lists' page bitmaps name (`crate::memory::bind`). Those the list
    /// after already read back (predicted) are taken from staging, as the
    /// submission left them; the rest are read back, waited for (behind
    /// what was submitted since: only submissions writing other ranges of
    /// those copies, or marking them too, run meanwhile, `Host::submit`).
    /// The pages named become the next prediction. The lists and staging go
    /// back to their pools. Returns the named pages whose bytes the GPU left
    /// as the mirror had them, by mirror address ([`into_mirror`]), and the
    /// write marks taken.
    fn vram_completed(
        &self,
        sync: VramSync,
        runs: &[&HostList],
        marked: bool,
    ) -> (std::collections::HashSet<usize>, crate::shadow::WriteMarks) {
        let VramSync {
            lists,
            staging,
            reads,
            later,
            predicted,
            signature,
            marks,
        } = sync;
        let marks = match (marked, marks) {
            (false, _) => crate::shadow::WriteMarks::default(),
            // SAFETY: the submission has run; its read-back staging is held
            // until the end of this call.
            (true, Some((placed, bytes, data))) => unsafe { crate::shadow::WriteMarks::staged(data, placed, bytes) },
            (true, None) => crate::shadow::WriteMarks::take_ram(),
        };
        let mut named: Vec<VramRead> = Vec::new();
        if marked {
            named.extend(crate::shadow::vram_marked_reads(&marks));
        }
        let mut probed = false;
        for list in runs {
            for probe in &list.deferred().vram_probes {
                probe(&mut named);
                probed = true;
            }
        }
        // Each page once: a second look would find the bytes just put in.
        let mut seen = std::collections::HashSet::new();
        named.retain(|r| seen.insert((r.0.as_raw() as usize, r.1, r.3)));
        let mut unchanged = std::collections::HashSet::new();
        let mut used = std::collections::HashSet::new();
        let mut missed = Vec::new();
        for read in &named {
            let key = (read.0.as_raw() as usize, read.1, read.3);
            if let Some(&from) = predicted.get(&key) {
                // SAFETY: as above; `read`'s mirror is a live copy's, just
                // named.
                unsafe { into_mirror(from as *const u8, read.2, read.3 as usize, true, &mut unchanged) };
                used.insert(key);
            } else {
                missed.push(read.clone());
            }
        }
        #[cfg(feature = "diagnostics")]
        {
            use crate::copystat::{C, add};
            add(C::VramNamedPages, named.len() as u64);
            add(C::VramPredictedPages, used.len() as u64);
            add(C::VramPredictedBytes, used.iter().map(|k| k.2).sum());
            add(
                C::VramWastedBytes,
                predicted.keys().filter(|k| !used.contains(*k)).map(|k| k.2).sum(),
            );
            add(C::VramWaitedBytes, later.iter().chain(&missed).map(|r| r.3).sum());
        }
        {
            #[cfg(feature = "diagnostics")]
            let _timed = crate::watch::Timed(std::time::Instant::now(), crate::copystat::C::VramWaitedUs);
            self.readback_vram_copies(&missed, &mut unchanged);
            // Declared ranges past the staging are not pages: what they hold
            // says nothing of a page's copy-back.
            self.readback_vram_copies(&later, &mut std::collections::HashSet::new());
        }
        // Declared ranges last: the named pages were compared with the
        // mirrors as the submission found them.
        {
            #[cfg(feature = "diagnostics")]
            let _timed = crate::watch::Timed(std::time::Instant::now(), crate::copystat::C::VramStagedUs);
            for &(from, to, len) in &reads {
                // SAFETY: the submission has run: the staging holds the bytes;
                // the mirror lives while the lists that declared the range do.
                unsafe { std::ptr::copy_nonoverlapping(from as *const u8, to as *mut u8, len) };
            }
            #[cfg(feature = "diagnostics")]
            crate::copystat::add(crate::copystat::C::VramStagedBytes, reads.iter().map(|r| r.2 as u64).sum());
        }
        #[cfg(feature = "diagnostics")]
        crate::copystat::add(crate::copystat::C::VramUnchangedPages, unchanged.len() as u64);
        if let Some(kind) = signature
            && (marked || probed)
        {
            let mut kinds = PREDICTED.lock().unwrap_or_else(|e| e.into_inner());
            if kinds.len() >= PREDICTED_KINDS && !kinds.contains_key(&kind) {
                kinds.clear();
            }
            kinds.insert(kind, named);
        }
        COPY_LISTS.lock().unwrap_or_else(|e| e.into_inner()).extend(lists);
        let mut pool = STAGING.lock().unwrap_or_else(|e| e.into_inner());
        for (readback, s) in staging {
            let pool = &mut pool[usize::from(readback)];
            if pool.iter().map(|s| s.size).sum::<u64>() + s.size <= STAGING_KEPT {
                pool.push(s);
            }
        }
        (unchanged, marks)
    }

    #[cfg(test)]
    fn run_test_copy_sync<F: FnOnce(&ID3D12GraphicsCommandList)>(&self, record: F) {
        // SAFETY: valid D3D12 device creating a test fence.
        let Ok(fence): windows::core::Result<ID3D12Fence> =
            (unsafe { self.device.CreateFence(0, D3D12_FENCE_FLAG_NONE) })
        else {
            return;
        };
        self.run_copy_list(record, false);
        // SAFETY: valid test fence and host event/queue.
        unsafe {
            if self.queue.Signal(&fence, 1).is_ok()
                && fence.SetEventOnCompletion(1, self.event).is_ok()
            {
                WaitForSingleObject(self.event, INFINITE);
            }
        }
    }

    #[cfg(test)]
    pub(crate) fn test_upload_buffer(&self, dst: &ID3D12Resource, dst_off: u64, src: *const u8, len: u64) {
        let Some((up, ptr)) = self.upload_buffer(len.max(1)) else { return };
        // SAFETY: `src` is valid for `len` bytes and `ptr` is mapped.
        unsafe { std::ptr::copy_nonoverlapping(src, ptr, len as usize) };
        self.run_test_copy_sync(|list| {
            // SAFETY: valid test buffer regions.
            unsafe { list.CopyBufferRegion(dst, dst_off, &up, 0, len) };
        });
    }

    #[cfg(test)]
    pub(crate) fn flush_test_vram_uploads(&self) {
        let uploads = crate::shadow::take_vram_uploads();
        if !uploads.is_empty() {
            self.upload_vram_copies(&uploads);
        }
    }

    #[cfg(test)]
    pub(crate) fn test_readback_buffer(&self, src: &ID3D12Resource, src_off: u64, dst: *mut u8, len: u64) {
        let Some((rb, ptr)) = self.mapped_buffer(len.max(1), true) else { return };
        self.run_test_copy_sync(|list| {
            // SAFETY: valid test buffer regions.
            unsafe { list.CopyBufferRegion(&rb, 0, src, src_off, len) };
        });
        // SAFETY: GPU copy finished before fence wait returned.
        unsafe { std::ptr::copy_nonoverlapping(ptr, dst, len as usize) };
    }

    fn run_copy_list<F: FnOnce(&ID3D12GraphicsCommandList)>(&self, record: F, wait: bool) {
        static CMD: Mutex<Option<(ID3D12CommandAllocator, ID3D12GraphicsCommandList, ID3D12Fence, u64)>> =
            Mutex::new(None);
        let mut guard = CMD.lock().unwrap_or_else(|e| e.into_inner());
        let (alloc, list, fence, seq) = guard.get_or_insert_with(|| {
            // SAFETY: valid D3D12 device creating a direct allocator, fence, and closed command list.
            unsafe {
                let alloc: ID3D12CommandAllocator = self
                    .device
                    .CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT)
                    .unwrap();
                let list: ID3D12GraphicsCommandList = self
                    .device
                    .CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, &alloc, None)
                    .unwrap();
                let fence: ID3D12Fence = self.device.CreateFence(0, D3D12_FENCE_FLAG_NONE).unwrap();
                let _ = list.Close();
                (alloc, list, fence, 0)
            }
        });
        // SAFETY: `alloc` and `list` are exclusively held under `CMD`; `fence` or caller's
        // test fence serializes reuse between submissions.
        unsafe {
            alloc.Reset().expect("reset copy allocator");
            list.Reset(&*alloc, None).expect("reset copy list");
            record(list);
            list.Close().expect("close copy list");
            let cmd: Option<ID3D12CommandList> = list.cast().ok();
            self.queue.ExecuteCommandLists(&[cmd]);
            if wait {
                *seq += 1;
                if self.queue.Signal(&*fence, *seq).is_ok()
                    && fence.SetEventOnCompletion(*seq, self.event).is_ok()
                {
                    WaitForSingleObject(self.event, INFINITE);
                }
            }
        }
    }

    pub(crate) fn upload_buffer(&self, size: u64) -> Option<(ID3D12Resource, *mut u8)> {
        self.mapped_buffer(size, false)
    }

    /// A mapped buffer of `size` bytes: upload memory, or (`readback`)
    /// readback memory in the copy-destination state.
    fn mapped_buffer(&self, size: u64, readback: bool) -> Option<(ID3D12Resource, *mut u8)> {
        let heap = D3D12_HEAP_PROPERTIES {
            Type: if readback { D3D12_HEAP_TYPE_READBACK } else { D3D12_HEAP_TYPE_UPLOAD },
            ..Default::default()
        };
        let desc = D3D12_RESOURCE_DESC {
            Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
            Width: size.max(1),
            Height: 1,
            DepthOrArraySize: 1,
            MipLevels: 1,
            SampleDesc: DXGI_SAMPLE_DESC {
                Count: 1,
                Quality: 0,
            },
            Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
            ..Default::default()
        };
        let mut buffer: Option<ID3D12Resource> = None;
        let mut data = std::ptr::null_mut();
        // SAFETY: plain creation and mapping of an upload buffer.
        let made = unsafe {
            self.device
                .CreateCommittedResource(
                    &heap,
                    D3D12_HEAP_FLAG_NONE,
                    &desc,
                    if readback { D3D12_RESOURCE_STATE_COPY_DEST } else { D3D12_RESOURCE_STATE_GENERIC_READ },
                    None,
                    &mut buffer,
                )
                .and_then(|()| {
                    let b = buffer.as_ref().ok_or_else(windows::core::Error::empty)?;
                    b.Map(0, None, Some(&mut data))
                })
        };
        match made {
            Ok(()) => Some((buffer?, data.cast())),
            Err(e) => {
                tracing::warn!(error = %e, size, readback, "no mapped buffer");
                None
            }
        }
    }
}

/// The swap chain formats a presented texture can be copied into: its own
/// family's.
fn swap_format(format: DXGI_FORMAT) -> Option<DXGI_FORMAT> {
    Some(match format.0 {
        27..=32 => DXGI_FORMAT_R8G8B8A8_UNORM,
        87 | 90 | 91 => DXGI_FORMAT_B8G8R8A8_UNORM,
        23..=25 => DXGI_FORMAT_R10G10B10A2_UNORM,
        9..=14 => DXGI_FORMAT_R16G16B16A16_FLOAT,
        _ => return None,
    })
}

/// Swap chain buffers: with overlap, one shown, one queued to be, and one
/// the next frame is drawn into while the GPU runs on (a present does not
/// wait for its frame; with two, the queue stalled until the shown one was
/// let go).
fn swap_buffers() -> u32 {
    if overlap_switch() { 3 } else { 2 }
}
/// Present lists in turn ([`Presenter::copies`]).
const PRESENT_LISTS: usize = 3;
const MAX_PLANES: u32 = 8;

/// Built-in fullscreen-triangle compositor for multi-plane `PresentX` calls
/// (e.g. Series S presenting a 1080p scene plane with a 4K UI overlay plane).
struct Compositor {
    format: DXGI_FORMAT,
    root: ID3D12RootSignature,
    opaque: ID3D12PipelineState,
    overlay: ID3D12PipelineState,
    rtv_heap: ID3D12DescriptorHeap,
    srv_heap: ID3D12DescriptorHeap,
    srv_stride: u32,
}

impl Compositor {
    fn new(device: &ID3D12Device, format: DXGI_FORMAT) -> windows::core::Result<Self> {
        const SOURCE: &str = r#"
            Texture2D tex : register(t0);
            SamplerState smp : register(s0);
            void vs(uint id : SV_VertexID, out float4 pos : SV_Position, out float2 uv : TEXCOORD) {
                uv = float2((id << 1) & 2, id & 2);
                pos = float4(uv * float2(2, -2) + float2(-1, 1), 0, 1);
            }
            float4 ps(float4 pos : SV_Position, float2 uv : TEXCOORD) : SV_Target {
                return tex.Sample(smp, uv);
            }
        "#;
        let compile = |entry: PCSTR, target: PCSTR| -> windows::core::Result<ID3DBlob> {
            let mut code: Option<ID3DBlob> = None;
            let mut errors: Option<ID3DBlob> = None;
            // SAFETY: static source and entry/target strings; out pointers ours.
            let res = unsafe {
                D3DCompile(
                    SOURCE.as_ptr().cast(),
                    SOURCE.len(),
                    s!("present_composite"),
                    None,
                    None,
                    entry,
                    target,
                    0,
                    0,
                    &mut code,
                    Some(&mut errors),
                )
            };
            if let Err(e) = res {
                if let Some(errors) = errors {
                    // SAFETY: the blob's own bytes.
                    let text = unsafe {
                        std::slice::from_raw_parts(
                            errors.GetBufferPointer().cast::<u8>(),
                            errors.GetBufferSize(),
                        )
                    };
                    tracing::warn!("present compositor: {}", String::from_utf8_lossy(text));
                }
                return Err(e);
            }
            code.ok_or_else(windows::core::Error::empty)
        };
        let vs = compile(s!("vs"), s!("vs_5_1"))?;
        let ps = compile(s!("ps"), s!("ps_5_1"))?;
        // SAFETY: compiled shader blobs live through PSO creation.
        unsafe {
            let vs_bytes =
                std::slice::from_raw_parts(vs.GetBufferPointer().cast::<u8>(), vs.GetBufferSize());
            let ps_bytes =
                std::slice::from_raw_parts(ps.GetBufferPointer().cast::<u8>(), ps.GetBufferSize());
            // Wine's D3DCompile does not embed a root signature in the shader blobs.
            let range = D3D12_DESCRIPTOR_RANGE {
                RangeType: D3D12_DESCRIPTOR_RANGE_TYPE_SRV,
                NumDescriptors: 1,
                OffsetInDescriptorsFromTableStart: D3D12_DESCRIPTOR_RANGE_OFFSET_APPEND,
                ..Default::default()
            };
            let parameter = D3D12_ROOT_PARAMETER {
                ParameterType: D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE,
                Anonymous: D3D12_ROOT_PARAMETER_0 {
                    DescriptorTable: D3D12_ROOT_DESCRIPTOR_TABLE {
                        NumDescriptorRanges: 1,
                        pDescriptorRanges: &range,
                    },
                },
                ShaderVisibility: D3D12_SHADER_VISIBILITY_PIXEL,
            };
            let sampler = D3D12_STATIC_SAMPLER_DESC {
                Filter: D3D12_FILTER_MIN_MAG_MIP_LINEAR,
                AddressU: D3D12_TEXTURE_ADDRESS_MODE_CLAMP,
                AddressV: D3D12_TEXTURE_ADDRESS_MODE_CLAMP,
                AddressW: D3D12_TEXTURE_ADDRESS_MODE_WRAP,
                MaxAnisotropy: 16,
                ComparisonFunc: D3D12_COMPARISON_FUNC_LESS_EQUAL,
                BorderColor: D3D12_STATIC_BORDER_COLOR_OPAQUE_WHITE,
                MaxLOD: f32::MAX,
                ShaderVisibility: D3D12_SHADER_VISIBILITY_PIXEL,
                ..Default::default()
            };
            let desc = D3D12_ROOT_SIGNATURE_DESC {
                NumParameters: 1,
                pParameters: &parameter,
                NumStaticSamplers: 1,
                pStaticSamplers: &sampler,
                ..Default::default()
            };
            let mut signature = None;
            D3D12SerializeRootSignature(&desc, D3D_ROOT_SIGNATURE_VERSION_1, &mut signature, None)?;
            let signature = signature.ok_or_else(windows::core::Error::empty)?;
            let signature = std::slice::from_raw_parts(
                signature.GetBufferPointer().cast::<u8>(),
                signature.GetBufferSize(),
            );
            let root: ID3D12RootSignature = device.CreateRootSignature(0, signature)?;
            let make_pso = |blend: bool| -> windows::core::Result<ID3D12PipelineState> {
                let mut rt_blend = D3D12_RENDER_TARGET_BLEND_DESC {
                    RenderTargetWriteMask: D3D12_COLOR_WRITE_ENABLE_ALL.0 as u8,
                    ..Default::default()
                };
                if blend {
                    rt_blend.BlendEnable = true.into();
                    rt_blend.SrcBlend = D3D12_BLEND_ONE;
                    rt_blend.DestBlend = D3D12_BLEND_INV_SRC_ALPHA;
                    rt_blend.BlendOp = D3D12_BLEND_OP_ADD;
                    rt_blend.SrcBlendAlpha = D3D12_BLEND_ONE;
                    rt_blend.DestBlendAlpha = D3D12_BLEND_INV_SRC_ALPHA;
                    rt_blend.BlendOpAlpha = D3D12_BLEND_OP_ADD;
                }
                let mut blend_desc = D3D12_BLEND_DESC::default();
                blend_desc.RenderTarget[0] = rt_blend;
                let mut rtv_formats = [DXGI_FORMAT_UNKNOWN; 8];
                rtv_formats[0] = format;
                let desc = D3D12_GRAPHICS_PIPELINE_STATE_DESC {
                    pRootSignature: std::mem::ManuallyDrop::new(Some(root.clone())),
                    VS: D3D12_SHADER_BYTECODE {
                        pShaderBytecode: vs_bytes.as_ptr().cast(),
                        BytecodeLength: vs_bytes.len(),
                    },
                    PS: D3D12_SHADER_BYTECODE {
                        pShaderBytecode: ps_bytes.as_ptr().cast(),
                        BytecodeLength: ps_bytes.len(),
                    },
                    BlendState: blend_desc,
                    SampleMask: u32::MAX,
                    RasterizerState: D3D12_RASTERIZER_DESC {
                        FillMode: D3D12_FILL_MODE_SOLID,
                        CullMode: D3D12_CULL_MODE_NONE,
                        DepthClipEnable: false.into(),
                        ..Default::default()
                    },
                    PrimitiveTopologyType: D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE,
                    NumRenderTargets: 1,
                    RTVFormats: rtv_formats,
                    SampleDesc: DXGI_SAMPLE_DESC {
                        Count: 1,
                        Quality: 0,
                    },
                    ..Default::default()
                };
                let pso = device.CreateGraphicsPipelineState(&desc);
                let _ = std::mem::ManuallyDrop::into_inner(desc.pRootSignature);
                pso
            };
            let opaque = make_pso(false)?;
            let overlay = make_pso(true)?;
            let rtv_heap: ID3D12DescriptorHeap =
                device.CreateDescriptorHeap(&D3D12_DESCRIPTOR_HEAP_DESC {
                    Type: D3D12_DESCRIPTOR_HEAP_TYPE_RTV,
                    NumDescriptors: 1,
                    Flags: D3D12_DESCRIPTOR_HEAP_FLAG_NONE,
                    NodeMask: 0,
                })?;
            let srv_heap: ID3D12DescriptorHeap =
                device.CreateDescriptorHeap(&D3D12_DESCRIPTOR_HEAP_DESC {
                    Type: D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV,
                    NumDescriptors: MAX_PLANES,
                    Flags: D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE,
                    NodeMask: 0,
                })?;
            let srv_stride =
                device.GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
            Ok(Compositor {
                format,
                root,
                opaque,
                overlay,
                rtv_heap,
                srv_heap,
                srv_stride,
            })
        }
    }
}

#[derive(Default)]
struct Presentation {
    presenter: Option<Presenter>,
    retry: SwapChainRetry,
}

#[derive(Default)]
struct SwapChainRetry {
    next_attempt: Option<Instant>,
}

impl SwapChainRetry {
    fn ready(&self, now: Instant) -> bool {
        self.next_attempt.is_none_or(|next| now >= next)
    }

    /// A desktop can become presentable again. Keep retrying while limiting
    /// the DXGI calls and warnings during the outage.
    fn attempt<T, E>(
        &mut self,
        now: Instant,
        operation: impl FnOnce() -> Result<T, E>,
    ) -> Option<Result<T, E>> {
        if !self.ready(now) {
            return None;
        }
        let result = operation();
        self.next_attempt = result.is_err().then(|| now + Duration::from_secs(1));
        Some(result)
    }
}

struct Presenter {
    swap_chain: IDXGISwapChain3,
    width: u32,
    height: u32,
    format: DXGI_FORMAT,
    /// The lists that copy or compose frames into the swap chain, in turn:
    /// a present does not wait for its frame, so the ones before may still
    /// run (each completes before it is used again).
    copies: [std::sync::Arc<HostList>; PRESENT_LISTS],
    next_copy: usize,
    hwnd: HWND,
    meter: Meter,
    compositor: Option<Compositor>,
}

/// The frame rate, shown in the window's caption every second and logged
/// every few seconds.
struct Meter {
    since: std::time::Instant,
    frames: u32,
    report_frames: u64,
    drawn: u64,
    logged: std::time::Instant,
    /// The caption the window had, once read.
    caption: Option<Vec<u16>>,
}

impl Meter {
    fn new() -> Self {
        let now = std::time::Instant::now();
        Meter {
            since: now,
            frames: 0,
            report_frames: 0,
            drawn: crate::draw::drawn(),
            logged: now,
            caption: None,
        }
    }

    /// Counts a presented frame.
    fn frame(&mut self, hwnd: HWND) {
        #[cfg(feature = "diagnostics")]
        tracing::debug!(target: "victory_gpu::frame", token = crate::frame::current(), "present");
        self.frames += 1;
        crate::memory::log_frame();
        crate::watch::frame_end();
        #[cfg(feature = "diagnostics")]
        crate::watch::frame_done();
        self.report_frames += 1;
        let elapsed = self.since.elapsed();
        if elapsed < std::time::Duration::from_secs(1) {
            return;
        }
        let drawn = crate::draw::drawn();
        let frames = f64::from(self.frames);
        let fps = frames / elapsed.as_secs_f64();
        let ms = elapsed.as_secs_f64() * 1000.0 / frames;
        let draws = (drawn - self.drawn) as f64 / frames;
        (self.since, self.frames, self.drawn) = (std::time::Instant::now(), 0, drawn);
        // The counters are read in the five-second block, where `GPU timing`
        // is logged, so both lines cover the same frames.
        if self.logged.elapsed() >= std::time::Duration::from_secs(5) {
            self.logged = std::time::Instant::now();
            let report_frames = std::mem::take(&mut self.report_frames) as f64;
            // Of a frame, ms preparing submissions and ms waiting for the GPU.
            let [cpu, gpu] = [0, 1].map(|i| {
                SUBMISSION_US[i].swap(0, Ordering::Relaxed) as f64 / 1000.0 / report_frames
            });
            let [copy_ms, copy_mb, texture_ms, texture_mb] = [0, 1, 2, 3].map(|i| {
                let scale = if i % 2 == 0 { 1e3 } else { 1e6 };
                PREPARING[i].swap(0, Ordering::Relaxed) as f64 / scale / report_frames
            });
            let [back_ms, back_mb] = [0, 1].map(|i| {
                COPY_BACK[i].swap(0, Ordering::Relaxed) as f64 / [1e3, 1e6][i] / report_frames
            });
            let dropped = crate::draw::take_dropped() as f64 / report_frames;
            let indirect = crate::list::take_indirect() as f64 / report_frames;
            let root_views = crate::draw::take_root_view_slots() as f64 / report_frames;
            tracing::info!(
                "{fps:.1} fps, {ms:.1} ms/frame ({cpu:.1} preparing: copies {copy_ms:.1} ms \
                 {copy_mb:.1} MB, textures {texture_ms:.1} ms {texture_mb:.1} MB; {gpu:.1} on the \
                 GPU; {back_ms:.1} copying back {back_mb:.1} MB), {draws:.0} draws/frame, \
                 dropped {dropped:.0} without a host pipeline ({making} being made) and \
                 {indirect:.0} indirect; {root_views:.0} root-view slots/frame",
                making = crate::pipeline::making()
            );
            let (textures, bytes, used, shared, saved, replaced) = crate::resource::texture_counts();
            let (heaps, heap_used, heap_free) = crate::placement::stats();
            tracing::info!(
                "host textures: {textures} ({:.0} MB, {:.0} MB of them used by lists), title textures sharing one {shared} (saving {:.0} MB), replacing one {replaced}; \
                 heaps of textures {heaps} ({:.0} MB used, {:.0} MB free); \
                 copies between textures of one memory {:.1}/frame, depth twin copies {:.1}/frame",
                bytes as f64 / 1e6,
                used as f64 / 1e6,
                saved as f64 / 1e6,
                heap_used as f64 / 1e6,
                heap_free as f64 / 1e6,
                crate::texture::ALIAS_COPIES.swap(0, Ordering::Relaxed) as f64 / report_frames,
                crate::twin::COPIES.swap(0, Ordering::Relaxed) as f64 / report_frames,
            );
            #[cfg(feature = "diagnostics")]
            {
                if let Some([(local, budget), (system, system_budget)]) = video_memory() {
                    let (commit, working_set) = process_memory();
                    tracing::info!(
                        "video memory: {local:.0} MB of a {budget:.0} MB budget on the card, \
                         {system:.0} MB of {system_budget:.0} MB in system memory; process commit {:.0} MB, \
                         working set {:.0} MB; Rust heap {:.0} MB; placed textures made {:.0} MB; list upload memory {} blocks {:.0} MB",
                        commit as f64 / 1e6,
                        working_set as f64 / 1e6,
                        crate::copystat::HEAP_BYTES.load(Ordering::Relaxed) as f64 / 1e6,
                        crate::resource::PLACED_BYTES.load(Ordering::Relaxed) as f64 / 1e6,
                        crate::list::UPLOADS[0].load(Ordering::Relaxed),
                        crate::list::UPLOADS[1].load(Ordering::Relaxed) as f64 / 1e6,
                    );
                }
                let [gpu, title] = crate::resource::texture_census().map(|r| r.map(|b| b as f64 / 1e6));
                tracing::info!(
                    "host texture MB (used by lists, viewed by heap slots only, neither): GPU-written {:.0} {:.0} {:.0}, \
                     from title memory {:.0} {:.0} {:.0}",
                    gpu[0], gpu[1], gpu[2], title[0], title[1], title[2]
                );
                // Window caption updates leave the report counters intact.
                let drops = crate::pipeline::take_drops();
                crate::frame::log();
                log_gpu_times(report_frames);
                crate::copystat::log(report_frames);
                crate::bindless::log(report_frames);
                crate::shadow::log_views(report_frames);
                let [n, bytes] = [0, 1].map(|i| crate::list::READ_BACKS[i].swap(0, Ordering::Relaxed));
                if n > 0 {
                    tracing::info!(
                        "texture read backs: {:.1}/frame, {:.1} KB/frame",
                        n as f64 / report_frames,
                        bytes as f64 / report_frames / 1e3
                    );
                }
                for (why, n) in drops.iter().take(40) {
                    tracing::info!(
                        "dropped {:.1}/frame without a host pipeline: {why}",
                        *n as f64 / report_frames
                    );
                }
            }
        }
        if self.caption.is_none() {
            // The window's own text, read without a message.
            let mut buffer = vec![0u16; 512];
            // SAFETY: a window handle and a buffer of that length.
            let n = unsafe { InternalGetWindowText(hwnd, &mut buffer) };
            buffer.truncate(n.max(0) as usize);
            self.caption = Some(buffer);
        }
        let Some(caption) = &self.caption else { return };
        let text = format!(" | VIctory {fps:.0} fps {ms:.1} ms/frame {draws:.0} draws/frame");
        let mut text: Vec<u16> = caption.iter().copied().chain(text.encode_utf16()).collect();
        text.push(0);
        // Setting the text waits for the window's thread (a title thread,
        // which may not pump for a while), so a thread of ours does it and
        // presenting never waits; it takes the newest text when free.
        static PENDING: Mutex<Option<(usize, Vec<u16>)>> = Mutex::new(None);
        static SETTER: std::sync::Once = std::sync::Once::new();
        *PENDING.lock().unwrap_or_else(|e| e.into_inner()) = Some((hwnd.0 as usize, text));
        SETTER.call_once(|| {
            let _ = std::thread::Builder::new()
                .name("victory-caption".into())
                .spawn(|| {
                    loop {
                        let next = PENDING.lock().unwrap_or_else(|e| e.into_inner()).take();
                        match next {
                            // SAFETY: a window handle and a NUL-terminated text.
                            Some((hwnd, text)) => unsafe {
                                let _ = SetWindowTextW(HWND(hwnd as *mut _), PCWSTR(text.as_ptr()));
                            },
                            None => std::thread::sleep(std::time::Duration::from_millis(100)),
                        }
                    }
                });
        });
    }
}

impl Host {
    /// Shows one or more `planes` (each a live host texture and its current
    /// host state, kept in that state), compositing planes 1.. on top of
    /// plane 0 when multiple planes are presented, as frame `frame` (saved if
    /// captures are on). `None` if nothing was shown; else, in diagnostics
    /// builds, whether the frame was black.
    pub(crate) fn present(&self, planes: &[(&ID3D12Resource, u32)], frame: u64) -> Option<bool> {
        self.present_with_compositor(
            planes,
            frame,
            Compositor::new,
            #[cfg(all(test, feature = "diagnostics"))]
            |_, _| {},
        )
    }

    fn present_with_compositor(
        &self,
        planes: &[(&ID3D12Resource, u32)],
        frame: u64,
        create_compositor: impl FnOnce(&ID3D12Device, DXGI_FORMAT) -> windows::core::Result<Compositor>,
        #[cfg(all(test, feature = "diagnostics"))] inspect: impl FnOnce(
            &ID3D12Resource,
            &D3D12_PLACED_SUBRESOURCE_FOOTPRINT,
        ),
    ) -> Option<bool> {
        let &(primary, _) = planes.first()?;
        // SAFETY: live resources.
        let primary_desc = unsafe { primary.GetDesc() };
        let Some(format) = swap_format(primary_desc.Format) else {
            tracing::warn!(format = primary_desc.Format.0, "cannot present this format");
            return None;
        };
        let mut presentation = self.presenter.lock().unwrap_or_else(|e| e.into_inner());
        let Presentation { presenter, retry } = &mut *presentation;
        if presenter.is_none() && !retry.ready(Instant::now()) {
            return None;
        }
        let cached_format = presenter
            .as_ref()
            .and_then(|p| p.compositor.as_ref())
            .map(|c| c.format);
        let mut new_compositor = None;
        let planes = if planes.len() > 1 && cached_format != Some(format) {
            match create_compositor(&self.device, format) {
                Ok(c) => {
                    new_compositor = Some(c);
                    planes
                }
                Err(e) => {
                    tracing::warn!(error = %e, "presenting the primary plane after compositor initialization failed");
                    &planes[..1]
                }
            }
        } else {
            planes
        };
        let (mut width, mut height) = (primary_desc.Width as u32, primary_desc.Height);
        for &(tex, _) in &planes[1..] {
            // SAFETY: live resource.
            let d = unsafe { tex.GetDesc() };
            width = width.max(d.Width as u32);
            height = height.max(d.Height);
        }
        if presenter.is_none() {
            *presenter = match retry
                .attempt(Instant::now(), || self.presenter(width, height, format))?
            {
                Ok(p) => Some(p),
                Err((step, e)) => {
                    tracing::warn!(error = %e, step, width, height, format = format.0, "no swap chain");
                    return None;
                }
            };
        }
        let p = presenter.as_mut()?;
        if let Some(c) = new_compositor {
            p.compositor = Some(c);
        }
        if (p.width, p.height, p.format) != (width, height, format) {
            let resized = retry.attempt(Instant::now(), || {
                // No list may still use the back buffers.
                self.complete_all();
                // SAFETY: no back buffer is held; every submission has completed.
                unsafe {
                    p.swap_chain.ResizeBuffers(
                        swap_buffers(),
                        width,
                        height,
                        format,
                        DXGI_SWAP_CHAIN_FLAG(0),
                    )
                }
            })?;
            match resized {
                Ok(()) => (p.width, p.height, p.format) = (width, height, format),
                Err(e) => {
                    tracing::warn!(error = %e, width, height, format = format.0, "swap chain not resized");
                    return None;
                }
            }
        }
        // SAFETY: our swap chain, list and live source textures.
        unsafe {
            let index = p.swap_chain.GetCurrentBackBufferIndex();
            let Ok(back) = p.swap_chain.GetBuffer::<ID3D12Resource>(index) else {
                return None;
            };
            let copy = p.copies[p.next_copy].clone();
            p.next_copy = (p.next_copy + 1) % PRESENT_LISTS;
            // Its last frame may still run.
            self.complete_held(&copy);
            copy.reset();
            let list = &copy.list;
            let (capture, probe) = if planes.len() == 1 {
                let (texture, state) = planes[0];
                let state = D3D12_RESOURCE_STATES(state as i32);
                let mut before = Vec::with_capacity(2);
                if state != D3D12_RESOURCE_STATE_COPY_SOURCE {
                    before.push(transition(texture, state, D3D12_RESOURCE_STATE_COPY_SOURCE));
                }
                before.push(transition(
                    &back,
                    D3D12_RESOURCE_STATE_PRESENT,
                    D3D12_RESOURCE_STATE_COPY_DEST,
                ));
                list.ResourceBarrier(&before);
                list.CopyResource(&back, texture);
                #[cfg(all(test, feature = "diagnostics"))]
                let capture = {
                    // Inspect what reached the swap chain, before Present.
                    list.ResourceBarrier(&[transition(
                        &back,
                        D3D12_RESOURCE_STATE_COPY_DEST,
                        D3D12_RESOURCE_STATE_COPY_SOURCE,
                    )]);
                    let readback = crate::capture::read_back(&self.device, list, &back);
                    list.ResourceBarrier(&[transition(
                        &back,
                        D3D12_RESOURCE_STATE_COPY_SOURCE,
                        D3D12_RESOURCE_STATE_COPY_DEST,
                    )]);
                    readback
                };
                #[cfg(all(not(test), feature = "diagnostics"))]
                let capture = crate::capture::wanted(frame)
                    .then(|| crate::capture::read_back(&self.device, list, texture))
                    .flatten();
                #[cfg(feature = "diagnostics")]
                let probe = crate::capture::probe(&self.device, list, texture);
                #[cfg(not(feature = "diagnostics"))]
                let (capture, probe) = {
                    let _ = frame;
                    (None::<()>, None::<()>)
                };
                let mut after = Vec::with_capacity(2);
                if state != D3D12_RESOURCE_STATE_COPY_SOURCE {
                    after.push(transition(texture, D3D12_RESOURCE_STATE_COPY_SOURCE, state));
                }
                after.push(transition(
                    &back,
                    D3D12_RESOURCE_STATE_COPY_DEST,
                    D3D12_RESOURCE_STATE_PRESENT,
                ));
                list.ResourceBarrier(&after);
                (capture, probe)
            } else {
                let comp = p.compositor.as_ref().expect("compositor was prepared for multiple planes");
                // Its views are written again: the frames before have run.
                for c in &p.copies {
                    self.complete_held(c);
                }
                let rtv = comp.rtv_heap.GetCPUDescriptorHandleForHeapStart();
                self.device.CreateRenderTargetView(
                    &back,
                    Some(&D3D12_RENDER_TARGET_VIEW_DESC {
                        Format: format,
                        ViewDimension: D3D12_RTV_DIMENSION_TEXTURE2D,
                        Anonymous: D3D12_RENDER_TARGET_VIEW_DESC_0 {
                            Texture2D: D3D12_TEX2D_RTV {
                                MipSlice: 0,
                                PlaneSlice: 0,
                            },
                        },
                    }),
                    rtv,
                );
                let srv_cpu_base = comp.srv_heap.GetCPUDescriptorHandleForHeapStart();
                let srv_gpu_base = comp.srv_heap.GetGPUDescriptorHandleForHeapStart();
                let count = planes.len().min(MAX_PLANES as usize);
                let mut before = Vec::with_capacity(count + 1);
                let mut after = Vec::with_capacity(count);
                for (i, &(tex, state)) in planes[..count].iter().enumerate() {
                    let d = tex.GetDesc();
                    let srv_fmt = swap_format(d.Format).unwrap_or(d.Format);
                    let handle = D3D12_CPU_DESCRIPTOR_HANDLE {
                        ptr: srv_cpu_base.ptr + i * comp.srv_stride as usize,
                    };
                    self.device.CreateShaderResourceView(
                        tex,
                        Some(&D3D12_SHADER_RESOURCE_VIEW_DESC {
                            Format: srv_fmt,
                            ViewDimension: D3D12_SRV_DIMENSION_TEXTURE2D,
                            Shader4ComponentMapping: D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING,
                            Anonymous: D3D12_SHADER_RESOURCE_VIEW_DESC_0 {
                                Texture2D: D3D12_TEX2D_SRV {
                                    MostDetailedMip: 0,
                                    MipLevels: 1,
                                    PlaneSlice: 0,
                                    ResourceMinLODClamp: 0.0,
                                },
                            },
                        }),
                        handle,
                    );
                    let state = D3D12_RESOURCE_STATES(state as i32);
                    if state != D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE
                        && planes[..i]
                            .iter()
                            .all(|(earlier, _)| earlier.as_raw() != tex.as_raw())
                    {
                        before.push(transition(
                            tex,
                            state,
                            D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE,
                        ));
                        after.push(transition(
                            tex,
                            D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE,
                            state,
                        ));
                    }
                }
                before.push(transition(
                    &back,
                    D3D12_RESOURCE_STATE_PRESENT,
                    D3D12_RESOURCE_STATE_RENDER_TARGET,
                ));
                list.ResourceBarrier(&before);
                list.SetDescriptorHeaps(&[Some(comp.srv_heap.clone())]);
                list.SetGraphicsRootSignature(&comp.root);
                list.RSSetViewports(&[D3D12_VIEWPORT {
                    TopLeftX: 0.0,
                    TopLeftY: 0.0,
                    Width: width as f32,
                    Height: height as f32,
                    MinDepth: 0.0,
                    MaxDepth: 1.0,
                }]);
                list.RSSetScissorRects(&[RECT {
                    left: 0,
                    top: 0,
                    right: width as i32,
                    bottom: height as i32,
                }]);
                list.OMSetRenderTargets(1, Some(&rtv), false, None);
                list.IASetPrimitiveTopology(D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
                for i in 0..count {
                    list.SetPipelineState(if i == 0 { &comp.opaque } else { &comp.overlay });
                    list.SetGraphicsRootDescriptorTable(
                        0,
                        D3D12_GPU_DESCRIPTOR_HANDLE {
                            ptr: srv_gpu_base.ptr + (i as u64) * u64::from(comp.srv_stride),
                        },
                    );
                    list.DrawInstanced(3, 1, 0, 0);
                }
                #[cfg(feature = "diagnostics")]
                let (capture, probe) = {
                    list.ResourceBarrier(&[transition(
                        &back,
                        D3D12_RESOURCE_STATE_RENDER_TARGET,
                        D3D12_RESOURCE_STATE_COPY_SOURCE,
                    )]);
                    let capture = (cfg!(test) || crate::capture::wanted(frame))
                        .then(|| crate::capture::read_back(&self.device, list, &back))
                        .flatten();
                    let probe = crate::capture::probe(&self.device, list, &back);
                    list.ResourceBarrier(&[transition(
                        &back,
                        D3D12_RESOURCE_STATE_COPY_SOURCE,
                        D3D12_RESOURCE_STATE_PRESENT,
                    )]);
                    (capture, probe)
                };
                #[cfg(not(feature = "diagnostics"))]
                let (capture, probe) = {
                    let _ = frame;
                    list.ResourceBarrier(&[transition(
                        &back,
                        D3D12_RESOURCE_STATE_RENDER_TARGET,
                        D3D12_RESOURCE_STATE_PRESENT,
                    )]);
                    (None::<()>, None::<()>)
                };
                if !after.is_empty() {
                    list.ResourceBarrier(&after);
                }
                (capture, probe)
            };
            // The frame's textures stay alive until the list has run: the
            // title may release or give them up meanwhile. Not the back
            // buffer: the swap chain keeps it, and holding it past the frame
            // would make a resize fail (every list completes before one).
            for &(texture, _) in planes {
                copy.keep(texture);
            }
            copy.close();
            self.submit(0, vec![copy.clone()]);
            // What diagnostics read back is read once the frame has run.
            #[cfg(feature = "diagnostics")]
            if capture.is_some() || probe.is_some() {
                self.complete_held(&copy);
            }
            #[cfg(feature = "diagnostics")]
            if let Some((buffer, footprint)) = capture {
                // Tests inspect the completed copy before flip-discard Present
                // can invalidate the back buffer, with the presenter still locked.
                #[cfg(test)]
                inspect(&buffer, &footprint);
                crate::capture::save_read_back(frame, &buffer, &footprint);
            }
            let _ = p.swap_chain.Present(0, DXGI_PRESENT(0));
            p.meter.frame(p.hwnd);
            #[cfg(feature = "diagnostics")]
            return Some(probe.is_some_and(|b| crate::capture::black(&b)));
            #[cfg(not(feature = "diagnostics"))]
            {
                let _ = (capture, probe);
                Some(false)
            }
        }
    }

    fn presenter(
        &self,
        width: u32,
        height: u32,
        format: DXGI_FORMAT,
    ) -> Result<Presenter, (&'static str, windows::core::Error)> {
        let hwnd = match title_window() {
            Some(hwnd) => hwnd,
            None => window(width, height).map_err(|e| ("window", e))?,
        };
        // SAFETY: plain object creation for our window.
        unsafe {
            let factory: IDXGIFactory2 = CreateDXGIFactory2(DXGI_CREATE_FACTORY_FLAGS(0))
                .map_err(|e| ("DXGI factory", e))?;
            let desc = DXGI_SWAP_CHAIN_DESC1 {
                Width: width,
                Height: height,
                Format: format,
                SampleDesc: DXGI_SAMPLE_DESC {
                    Count: 1,
                    Quality: 0,
                },
                BufferUsage: DXGI_USAGE_RENDER_TARGET_OUTPUT,
                BufferCount: swap_buffers(),
                SwapEffect: DXGI_SWAP_EFFECT_FLIP_DISCARD,
                ..Default::default()
            };
            let swap_chain = factory
                .CreateSwapChainForHwnd(&self.queue, hwnd, &desc, None, None)
                .and_then(|chain| chain.cast::<IDXGISwapChain3>())
                .map_err(|e| ("swap chain", e))?;
            let copy = || {
                let copy = self.new_list().map_err(|e| ("copy list", e))?;
                copy.close();
                Ok(std::sync::Arc::new(copy))
            };
            let copies = [copy()?, copy()?, copy()?];
            tracing::info!(width, height, format = format.0, "presenting to a window");
            Ok(Presenter {
                swap_chain,
                width,
                height,
                format,
                copies,
                next_copy: 0,
                hwnd,
                meter: Meter::new(),
                compositor: None,
            })
        }
    }
}

/// A visible top-level window of this process: the one the title made, if
/// it made one.
fn title_window() -> Option<HWND> {
    extern "system" fn visit(hwnd: HWND, found: LPARAM) -> windows::core::BOOL {
        let mut process = 0;
        // SAFETY: a window handle from the enumeration; `found` is ours.
        unsafe {
            GetWindowThreadProcessId(hwnd, Some(&mut process));
            if process == windows::Win32::System::Threading::GetCurrentProcessId()
                && IsWindowVisible(hwnd).as_bool()
            {
                *(found.0 as *mut Option<HWND>) = Some(hwnd);
                return false.into();
            }
        }
        true.into()
    }
    let mut found: Option<HWND> = None;
    // SAFETY: the callback only writes `found`, which outlives the call.
    let _ = unsafe { EnumWindows(Some(visit), LPARAM(&raw mut found as isize)) };
    found
}

/// `state` without the bits the host texture's flags forbid: the console
/// lets the title name states its own resources allow (unordered access on
/// a depth target, whose host copy cannot have both).
pub(crate) fn legal_state(texture: &ID3D12Resource, state: u32) -> u32 {
    // SAFETY: a plain query.
    legal_for(unsafe { texture.GetDesc() }.Flags.0 as u32, state)
}

/// [`legal_state`] for a texture with resource flags `flags`.
pub(crate) fn legal_for(flags: u32, state: u32) -> u32 {
    // (flag, set or clear forbids, state bits): render target, unordered
    // access, depth, denied shader resource.
    const FORBIDDEN: [(u32, bool, u32); 4] = [
        (0x1, false, 0x4),
        (0x4, false, 0x8),
        (0x2, false, 0x10 | 0x20),
        (0x8, true, 0x40 | 0x80),
    ];
    FORBIDDEN
        .iter()
        .filter(|&&(flag, deny, _)| (flags & flag != 0) == deny)
        .fold(state, |s, &(_, _, bits)| s & !bits)
}

impl Host {
    /// Diagnostics, `VICTORY_COPIES_IN_VIDEO_MEMORY`: [`Host::shared_buffer`]s
    /// (host copies of title memory among them) in video memory the CPU maps
    /// (a GPU upload heap, where the host has one), to measure how much of
    /// the GPU's time is shaders reading the copies over the bus. CPU reads
    /// of them (snapshots, copy-backs) become slow: for that measurement only.
    #[cfg(feature = "diagnostics")]
    fn copies_in_video_memory(&self) -> bool {
        static ON: OnceLock<bool> = OnceLock::new();
        *ON.get_or_init(|| {
            if !crate::switch::on("VICTORY_COPIES_IN_VIDEO_MEMORY") {
                return false;
            }
            let mut options = D3D12_FEATURE_DATA_D3D12_OPTIONS16::default();
            // SAFETY: the query writes its own structure of the given size.
            let ok = unsafe {
                self.device.CheckFeatureSupport(
                    D3D12_FEATURE_D3D12_OPTIONS16,
                    (&raw mut options).cast(),
                    size_of_val(&options) as u32,
                )
            }
            .is_ok()
                && options.GPUUploadHeapSupported.as_bool();
            tracing::warn!(on = ok, "host copies of title memory in video memory (diagnostics)");
            ok
        })
    }

    /// A buffer in system memory that the CPU maps (write-back, so reading
    /// it is fast too) and shaders read and write: for host copies of title
    /// memory that shaders write, which upload memory cannot be. Starts in
    /// the common state (buffers move out of it implicitly).
    pub(crate) fn shared_buffer(&self, size: u64) -> Option<(ID3D12Resource, *mut u8)> {
        let heap = D3D12_HEAP_PROPERTIES {
            Type: D3D12_HEAP_TYPE_CUSTOM,
            CPUPageProperty: D3D12_CPU_PAGE_PROPERTY_WRITE_BACK,
            MemoryPoolPreference: D3D12_MEMORY_POOL_L0,
            ..Default::default()
        };
        #[cfg(feature = "diagnostics")]
        let heap = if self.copies_in_video_memory() {
            D3D12_HEAP_PROPERTIES { Type: D3D12_HEAP_TYPE_GPU_UPLOAD, ..Default::default() }
        } else {
            heap
        };
        let desc = D3D12_RESOURCE_DESC {
            Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
            Width: size.max(1),
            Height: 1,
            DepthOrArraySize: 1,
            MipLevels: 1,
            SampleDesc: DXGI_SAMPLE_DESC {
                Count: 1,
                Quality: 0,
            },
            Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
            Flags: D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
            ..Default::default()
        };
        let mut buffer: Option<ID3D12Resource> = None;
        let mut data = std::ptr::null_mut();
        // SAFETY: plain creation and mapping of a CPU-visible buffer.
        let made = unsafe {
            self.device
                .CreateCommittedResource(
                    &heap,
                    D3D12_HEAP_FLAG_NONE,
                    &desc,
                    D3D12_RESOURCE_STATE_COMMON,
                    None,
                    &mut buffer,
                )
                .and_then(|()| {
                    let b = buffer.as_ref().ok_or_else(windows::core::Error::empty)?;
                    b.Map(0, None, Some(&mut data))
                })
        };
        match made {
            Ok(()) => Some((buffer?, data.cast())),
            Err(e) => {
                tracing::warn!(error = %e, size, "no shared buffer");
                None
            }
        }
    }
}

/// Host objects GPU work in flight may still use ([`retire`],
/// [`retire_any`]): the next submission takes them, and they go once it
/// has completed (those before it complete first).
static RETIRED: Mutex<Vec<Box<dyn std::any::Any + Send>>> = Mutex::new(Vec::new());

/// Keeps `resource` alive until the GPU work running now (or the next
/// submission's) has run: title threads rewrite view heap slots while the
/// GPU runs, and a shader that took the old descriptor still reads what it
/// views.
pub(crate) fn retire(resource: ID3D12Resource) {
    retire_any(Box::new(resource));
}

/// As [`retire`], for any holder of host objects (host textures given up).
pub(crate) fn retire_any(what: Box<dyn std::any::Any + Send>) {
    RETIRED.lock().unwrap_or_else(|e| e.into_inner()).push(what);
}

/// With no submission in flight (tests): the retired objects go.
#[cfg(test)]
pub(crate) fn release_retired() {
    let retired = std::mem::take(&mut *RETIRED.lock().unwrap_or_else(|e| e.into_inner()));
    if !retired.is_empty() {
        release_later(Box::new(retired));
    }
}

/// Drops `what` on a thread of its own: releasing a D3D12 resource is a
/// kernel call (~5 % of a world frame on the title's submitting thread,
/// run perf1). Only for objects no GPU work in flight uses.
fn release_later(what: Box<dyn std::any::Any + Send>) {
    type Release = std::sync::mpsc::Sender<Box<dyn std::any::Any + Send>>;
    static RELEASE: OnceLock<Release> = OnceLock::new();
    let release = RELEASE.get_or_init(|| {
        let (tx, rx) = std::sync::mpsc::channel::<Box<dyn std::any::Any + Send>>();
        // Without the thread the receiver is gone: sends fail and drop here.
        let _ = std::thread::Builder::new()
            .name("victory-release".into())
            .spawn(move || rx.into_iter().for_each(drop));
        tx
    });
    let _ = release.send(what);
}

/// Diagnostics: a noted command running a shader `VICTORY_TRACE_SHADERS`
/// names (hashes, comma-separated) is logged with what it binds: all of
/// them in listed capture frames, the first 500 elsewhere.
#[cfg(feature = "diagnostics")]
fn traced(what: &str) {
    static LOGGED: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
    if is_traced(what)
        && (crate::capture::listed(crate::queue::presented() + 1)
            || LOGGED.fetch_add(1, Ordering::Relaxed) < 500)
    {
        tracing::info!("traced command: {what}");
    }
}

/// The shader hashes `VICTORY_TRACE_SHADERS` lists (comma-separated;
/// diagnostics).
#[cfg(feature = "diagnostics")]
pub(crate) fn traced_shaders() -> &'static [String] {
    static HASHES: OnceLock<Vec<String>> = OnceLock::new();
    HASHES.get_or_init(|| {
        let v = std::env::var("VICTORY_TRACE_SHADERS").unwrap_or_default();
        v.split(',').map(str::trim).filter(|h| !h.is_empty()).map(str::to_owned).collect()
    })
}

/// Whether `what` names a shader [`traced_shaders`] lists.
#[cfg(feature = "diagnostics")]
pub(crate) fn is_traced(what: &str) -> bool {
    traced_shaders().iter().any(|h| what.contains(h.as_str()))
}

/// Diagnostics, `VICTORY_GPU_TIMING`: every command of the list during
/// whose run (each list alone) the device was found removed.
fn removed_during(runs: &[&HostList], list: Option<usize>) -> String {
    #[cfg(feature = "diagnostics")]
    if let Some(i) = list {
        let notes = runs[i].notes.lock().unwrap_or_else(|e| e.into_inner());
        let mut text = format!("; removed while list {i} ran alone ({} commands):", notes.len());
        for (k, (n, _)) in notes.iter().enumerate() {
            text.push_str(&format!(" [{k}] {n}"));
        }
        return text;
    }
    let _ = (runs, list);
    String::new()
}

/// For adapters without device-removed data (the software one): the last
/// commands each list of the failed submission noted (diagnostics, with
/// `VICTORY_D3D12_DRED`).
fn last_notes(runs: &[&HostList]) -> String {
    #[cfg(feature = "diagnostics")]
    {
        let mut text = String::new();
        for (i, r) in runs.iter().enumerate() {
            let notes = r.notes.lock().unwrap_or_else(|e| e.into_inner());
            for (n, _) in notes.iter().rev().take(6).rev() {
                text.push_str(&format!("; list {i} of {}, of its {}: {n}", runs.len(), notes.len()));
            }
        }
        text
    }
    #[cfg(not(feature = "diagnostics"))]
    {
        let _ = runs;
        String::new()
    }
}

/// Where the faulting address `va` lies against a range `(start, bytes)` a
/// command reads (diagnostics).
#[cfg(feature = "diagnostics")]
fn fault_in(va: u64, (start, bytes): (u64, u64)) -> String {
    let end = start + bytes;
    if va < start {
        format!(" (fault {:#x} below its source)", start - va)
    } else if va < end {
        format!(" (fault at source +{:#x})", va - start)
    } else {
        format!(" (fault {:#x} past its source's end)", va - end)
    }
}

/// The commands that may have been running where a list stopped at
/// command `done`: those between the barriers around it (work between
/// barriers overlaps on the GPU), by kind, and what the markers just
/// before them say (`HostList::note_op`: diagnostics builds with
/// device-removed data on).
fn in_flight(history: &[D3D12_AUTO_BREADCRUMB_OP], markers: &[(u32, String)], done: usize) -> String {
    let barrier = |op: &D3D12_AUTO_BREADCRUMB_OP| *op == D3D12_AUTO_BREADCRUMB_OP_RESOURCEBARRIER;
    let from = history[..done].iter().rposition(barrier).map_or(0, |b| b + 1);
    let to = history[done..]
        .iter()
        .position(barrier)
        .map_or(history.len(), |b| (done + b).max(done + 1));
    let mut text = format!(", commands {from}..{to} between its barriers:");
    let mut shown = 0;
    for i in from..to {
        let op = history[i];
        if op == D3D12_AUTO_BREADCRUMB_OP_SETMARKER || shown == 12 {
            continue;
        }
        shown += 1;
        let marker = i
            .checked_sub(1)
            .filter(|&m| history[m] == D3D12_AUTO_BREADCRUMB_OP_SETMARKER)
            .and_then(|m| markers.iter().find(|(at, _)| *at as usize == m))
            .map_or("no marker", |(_, s)| s.as_str());
        text.push_str(&format!("; command {i} (op {}): {marker}", op.0));
    }
    text
}

pub(crate) fn transition(
    resource: &ID3D12Resource,
    before: D3D12_RESOURCE_STATES,
    after: D3D12_RESOURCE_STATES,
) -> D3D12_RESOURCE_BARRIER {
    #[cfg(test)]
    tests::ISSUED.with(|l| l.borrow_mut().push((resource.as_raw() as usize, before.0 as u32, after.0 as u32)));
    D3D12_RESOURCE_BARRIER {
        Type: D3D12_RESOURCE_BARRIER_TYPE_TRANSITION,
        Flags: D3D12_RESOURCE_BARRIER_FLAG_NONE,
        Anonymous: D3D12_RESOURCE_BARRIER_0 {
            Transition: std::mem::ManuallyDrop::new(D3D12_RESOURCE_TRANSITION_BARRIER {
                // SAFETY: a borrowed pointer; the barrier never releases it.
                pResource: unsafe { std::mem::transmute_copy(resource) },
                Subresource: D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES,
                StateBefore: before,
                StateAfter: after,
            }),
        },
    }
}

/// A window of the given client size, owned by its own message thread.
fn window(width: u32, height: u32) -> windows::core::Result<HWND> {
    let (sender, receiver) = std::sync::mpsc::channel();
    std::thread::Builder::new()
        .name("victory-window".into())
        .spawn(move || {
            let made = make_window(width, height);
            let ok = made.is_ok();
            let _ = sender.send(made.map(|h| h.0 as usize));
            if ok {
                pump();
            }
        })
        .map_err(|_| windows::core::Error::empty())?;
    let hwnd = receiver
        .recv()
        .map_err(|_| windows::core::Error::empty())??;
    Ok(HWND(hwnd as *mut _))
}

extern "system" fn window_proc(
    hwnd: HWND,
    message: u32,
    wparam: WPARAM,
    lparam: LPARAM,
) -> LRESULT {
    // SAFETY: the default handling for our window.
    unsafe { DefWindowProcW(hwnd, message, wparam, lparam) }
}

fn make_window(width: u32, height: u32) -> windows::core::Result<HWND> {
    const CLASS: PCWSTR = w!("VIctory");
    // SAFETY: registers our class once and creates a window of it.
    unsafe {
        let instance = GetModuleHandleW(None)?;
        let class = WNDCLASSEXW {
            cbSize: size_of::<WNDCLASSEXW>() as u32,
            lpfnWndProc: Some(window_proc),
            hInstance: instance.into(),
            lpszClassName: CLASS,
            hCursor: LoadCursorW(None, IDC_ARROW)?,
            ..Default::default()
        };
        RegisterClassExW(&class);
        let mut rect = RECT {
            left: 0,
            top: 0,
            right: width as i32,
            bottom: height as i32,
        };
        AdjustWindowRect(&mut rect, WS_OVERLAPPEDWINDOW, false)?;
        CreateWindowExW(
            WINDOW_EX_STYLE(0),
            CLASS,
            w!("VIctory"),
            WS_OVERLAPPEDWINDOW | WS_VISIBLE,
            CW_USEDEFAULT,
            CW_USEDEFAULT,
            rect.right - rect.left,
            rect.bottom - rect.top,
            None,
            None,
            Some(instance.into()),
            None,
        )
    }
}

fn pump() {
    let mut message = MSG::default();
    // SAFETY: the standard message loop of this thread's window.
    unsafe {
        while GetMessageW(&mut message, None, 0, 0).as_bool() {
            let _ = TranslateMessage(&message);
            DispatchMessageW(&message);
        }
    }
}

#[cfg(test)]
mod batch {   // `mod tests` is already taken at the top of this file
    use super::{Batch, nothing_in_flight};

    /// A host copy of tracked title memory: a submission whose lists all ask
    /// for it whole copies the one page the title wrote, once, and again
    /// only after a step that writes title memory.
    #[test]
    fn a_copy_is_refreshed_once_per_submission_by_the_pages_written() {
        use crate::watch::PAGE;
        let p = crate::watch::tests::pool(4);
        let len = 4 * PAGE;
        crate::shadow::register(p, p as u64, len as u64);
        let Some(copy) = crate::shadow::find(p as u64, len as u64) else {
            crate::shadow::unregister(p, p as u64);
            return; // no host GPU here
        };
        let mut batch = Batch::default();
        batch.add(&copy, p, len);
        assert_eq!(batch.flush(0, &mut nothing_in_flight), len, "all of it at first");
        // SAFETY: our pool, guarded since the copy took it.
        unsafe { ((p + PAGE + 8) as *mut u8).write_volatile(7) };
        // The next submission: two lists ask for the copy.
        let mut batch = Batch::default();
        batch.add(&copy, p, len);
        batch.add(&copy, p + 100, 200);
        assert_eq!(batch.flush(0, &mut nothing_in_flight), PAGE, "the page written");
        assert_eq!(copy.read_host((p + PAGE + 8) as u64, 1)[0], 7);
        // SAFETY: as above.
        unsafe { (p as *mut u8).write_volatile(1) };
        batch.add(&copy, p, len);
        assert_eq!(batch.flush(0, &mut nothing_in_flight), 0, "nothing again in the submission");
        batch.title_written(0, &mut nothing_in_flight);
        batch.add(&copy, p, len);
        assert_eq!(batch.flush(0, &mut nothing_in_flight), PAGE, "after a title write, what changed");
        crate::shadow::unregister(p, p as u64);
        drop(copy);
        crate::watch::tests::free(p);
    }
}

#[cfg(test)]
mod vram_async {
    use super::*;

    /// A submission's uploads and read-backs are staged in order, each at
    /// a 16-byte offset, while they fit; the rest goes through the ring.
    #[test]
    fn staging_takes_what_fits_in_order_at_16_byte_offsets() {
        assert_eq!(staged([], 64), (0, 0));
        assert_eq!(staged([5, 16, 3], 64), (3, 48));
        assert_eq!(staged([40, 30], 64), (1, 48), "past the cap: left");
        assert_eq!(staged([40, 30, 8], 64), (1, 48), "in order: none after one left");
        assert_eq!(staged([64], 64), (1, 64));
        assert_eq!(staged([65], 64), (0, 0));
    }

    /// A staging buffer is the smallest in the pool that holds the bytes.
    #[test]
    fn the_smallest_staging_that_fits_is_taken() {
        assert_eq!(smallest_fitting([4, 1, 2, 8], 2), Some(2));
        assert_eq!(smallest_fitting([4, 1, 2, 8], 3), Some(0));
        assert_eq!(smallest_fitting([4, 1, 2, 8], 9), None);
        assert_eq!(smallest_fitting([], 1), None);
    }

    /// A list marking through the heap's views is heavy only if one of its
    /// programs may write through them; reading alone, it is light (its read
    /// marks may be taken early). A reset list is neither.
    #[test]
    fn heap_marks_make_a_list_heavy_only_when_it_writes_through_the_heap() {
        let Some(host) = host() else {
            return; // no host GPU here
        };
        let list = host.new_list().unwrap();
        let viewed = std::sync::Arc::<crate::bindless::Viewed>::default();
        list.marking();
        list.defer_heap(&viewed);
        assert_eq!(list.heavy_reasons(), 1 << 6, "reads only: light");
        list.writes_through_heap();
        assert_eq!(list.heavy_reasons(), 1, "writes through the heap: heavy");
        list.close();
        list.reset();
        assert_eq!(list.heavy_reasons(), 0);
    }

    /// Overlapping, a submission whose shaders write a copy in video memory
    /// returns with the host GPU held: a list of its own reads the write
    /// back before its fence value, and it goes back to title memory once
    /// the submission has completed. (#3312 alone waited for the read-back
    /// inside the submission.)
    #[test]
    fn a_vram_copy_is_read_back_after_its_submission_returns() {
        use crate::watch::PAGE;
        let Some(host) = host() else {
            return; // no host GPU here
        };
        if !overlap() {
            return; // VICTORY_GPU_OVERLAP=0
        }
        let _queues = crate::queue::tests::TITLE_QUEUES.lock().unwrap_or_else(|e| e.into_inner());
        let p = crate::watch::tests::pool(1);
        let at = p + 16;
        // SAFETY: our pool, not guarded yet.
        unsafe { (at as *mut u32).write_volatile(0x11) };
        crate::shadow::register(p, p as u64, PAGE as u64);
        let found = crate::shadow::find(p as u64, PAGE as u64).and_then(|c| c.vram_readback().map(|v| (c, v)));
        let Some((copy, (_, base, start, _))) = found else {
            crate::shadow::unregister(p, p as u64);
            crate::watch::tests::free(p);
            return; // no copies in video memory here
        };
        let list = host.new_list().unwrap();
        copy.refresh(&list, p as u64, PAGE as u64);
        copy.gpu_writes(&list, at as u64, 4, false);
        // The shader's write: a copy into the copy in video memory.
        let (up, data) = host.upload_buffer(4).expect("an upload buffer");
        // SAFETY: a mapped upload buffer of 4 bytes; an open list, live buffers.
        unsafe {
            data.cast::<u32>().write_volatile(0xcafe_f00d);
            list.list.CopyBufferRegion(&copy.buffer, base + (at as u64 - start), &up, 0, 4);
        }
        list.keep(&up);
        list.close();
        // SAFETY: a word of our pool.
        let title = || unsafe { (at as *const u32).read_volatile() };
        let gate = host.hold_gpu();
        let submitted = Instant::now();
        host.submit(0, vec![std::sync::Arc::new(list)]);
        let returned = submitted.elapsed();
        assert!(returned < Duration::from_secs(5), "back with the host GPU held ({returned:?})");
        assert_eq!(title(), 0x11, "nothing went back before the GPU ran");
        drop(gate);
        host.complete_all();
        assert_eq!(title(), 0xcafe_f00d, "read back and gone back once completed");
        crate::shadow::unregister(p, p as u64);
        drop(copy);
        crate::watch::tests::free(p);
    }
}

#[cfg(test)]
mod light {
    use super::*;
    use crate::shadow::Shadow;
    use crate::watch::PAGE;
    use std::sync::Arc;

    /// A submission declaring a range of a copy waits for one in flight
    /// declaring an overlapping range of the same copy, unless both mark.
    #[test]
    fn declared_ranges_wait_unless_both_marked() {
        let a = [(1, 0x1000, 0x2000, false)];
        assert!(declared_overlap(&a, &[(1, 0x1800, 0x1900, false)]), "unmarked over unmarked");
        assert!(declared_overlap(&a, &[(1, 0x1800, 0x1900, true)]), "unmarked over marked");
        assert!(declared_overlap(&[(1, 0x1000, 0x2000, true)], &[(1, 0x1fff, 0x3000, false)]), "marked over unmarked");
        assert!(!declared_overlap(&[(1, 0x1000, 0x2000, true)], &[(1, 0x1800, 0x1900, true)]), "both marked");
        assert!(!declared_overlap(&a, &[(2, 0x1800, 0x1900, false)]), "another copy");
        assert!(!declared_overlap(&a, &[(1, 0x2000, 0x3000, false)]), "touching");
        assert!(!declared_overlap(&[], &a));
    }

    /// Barriers: writes through the heap's views, unnamed work after,
    /// missed checks, page tables, and marks shaders set where they are not
    /// captured per submission; bottom-of-pipe writes and declared ranges
    /// are not, nor named work after with `VICTORY_EARLY_BARRIERS`.
    #[test]
    fn only_undeclared_writes_and_shared_marks_are_barriers() {
        for bit in 0..8 {
            assert_eq!(barrier(1 << bit, false, true, false), [0, 1, 3, 5, 7].contains(&bit), "bit {bit}");
            assert_eq!(barrier(1 << bit, false, true, true), [0, 1, 3, 5].contains(&bit), "bit {bit}, relaxed");
        }
        for relaxed in [false, true] {
            assert!(!barrier(1 << 2 | 1 << 4 | 1 << 6, true, true, relaxed), "declared, bottom-of-pipe, captured marks");
            assert!(barrier(0, true, false, relaxed), "marks not captured apart");
            assert!(!barrier(0, false, false, relaxed));
        }
    }

    /// With `VICTORY_EARLY_BARRIERS` a barrier no longer completes every
    /// submission in flight before it is prepared, unless its marks are
    /// taken from the one bitmap; without, every barrier does.
    #[test]
    fn relaxed_barriers_are_prepared_with_others_in_flight() {
        for bit in [0, 1, 3, 5, 7] {
            assert!(alone(1 << bit, true, true, false), "bit {bit}");
            assert!(!alone(1 << bit, true, true, true), "bit {bit}, relaxed");
            assert!(alone(1 << bit, true, false, true), "bit {bit}, marks not captured");
        }
        assert!(!alone(1 << 2 | 1 << 4, true, true, false), "not a barrier");
    }

    /// Work after a list that names what it writes (a texture read back)
    /// is its own heavy reason, registered in flight as title writes;
    /// unnamed work after stays a barrier reason.
    #[test]
    fn named_work_after_is_registered_in_flight() {
        let d = Deferred {
            after_writes: vec![(0x1000, 0x1100), (0x2000, 0x2040)],
            ..Deferred::default()
        };
        assert_eq!(reasons_of(&d, false, false), AFTER_NAMED);
        assert!(!barrier(reasons_of(&d, false, false), false, true, true));
        assert_eq!(flying_of(&d), [(0x1000, 0x1100, crate::shadow::AFTER), (0x2000, 0x2040, crate::shadow::AFTER)]);
        let d = Deferred {
            after_unnamed: true,
            ..d
        };
        assert_eq!(reasons_of(&d, false, false), AFTER_NAMED | 1 << 1);
        assert!(barrier(reasons_of(&d, false, false), false, true, true));
    }

    /// A copy in video memory of a fresh pool of `pages` pages, and the
    /// pool; none without one.
    fn vram_copy(pages: usize) -> Option<(usize, Arc<Shadow>)> {
        let p = crate::watch::tests::pool(pages);
        let len = (pages * PAGE) as u64;
        crate::shadow::register(p, p as u64, len);
        match crate::shadow::find(p as u64, len).filter(|c| c.vram_readback().is_some()) {
            Some(copy) => Some((p, copy)),
            None => {
                release(p, None);
                None
            }
        }
    }

    fn release(p: usize, copy: Option<Arc<Shadow>>) {
        crate::shadow::unregister(p, p as u64);
        drop(copy);
        crate::watch::tests::free(p);
    }

    /// A list that reads `copy` whole and whose "shader" writes `value` at
    /// title address `at` of it: a declared range (`marked`: its page's
    /// mark set too), written by a copy on the GPU.
    fn writer(host: &Host, copy: &Arc<Shadow>, at: usize, value: u32, marked: bool) -> HostList {
        let list = host.new_list().unwrap();
        let (start, len) = copy.range();
        copy.refresh(&list, start, len);
        if marked {
            list.marking();
            copy.gpu_writes(&list, start, len, true);
            crate::shadow::tests::mark_on_gpu(&list, copy, ((at as u64 - start) / PAGE as u64) as u32).expect("marked");
        } else {
            copy.gpu_writes(&list, at as u64, 4, false);
        }
        let (up, data) = host.upload_buffer(4).expect("an upload buffer");
        // SAFETY: a mapped upload buffer of 4 bytes; an open list, live buffers.
        unsafe {
            data.cast::<u32>().write_volatile(value);
            list.list.CopyBufferRegion(&copy.buffer, copy.offset(at as u64), &up, 0, 4);
        }
        list.keep(&up);
        list.close();
        list
    }

    /// A list that only reads `copy` whole.
    fn reader(host: &Host, copy: &Arc<Shadow>) -> HostList {
        let list = host.new_list().unwrap();
        let (start, len) = copy.range();
        copy.refresh(&list, start, len);
        list.close();
        list
    }

    /// A title word.
    fn word(at: usize) -> u32 {
        // SAFETY: a word of a test's pool.
        unsafe { (at as *const u32).read_volatile() }
    }

    /// Lets the host GPU go `ms` after now, from another thread (the test's
    /// own may be waiting for it).
    fn release_gpu_after<G: 'static>(gate: G, ms: u64) -> std::thread::JoinHandle<()> {
        struct Sent<T>(T);
        // SAFETY: the gate holds a fence, free-threaded.
        unsafe impl<T> Send for Sent<T> {}
        let gate = Sent(gate);
        std::thread::spawn(move || {
            std::thread::sleep(Duration::from_millis(ms));
            drop(gate);
        })
    }

    fn in_flight(host: &Host) -> usize {
        host.lock_submitted().flight.len()
    }

    /// Overlapping, two submissions writing different copies stay in
    /// flight together with the host GPU held (each would have been
    /// completed before the next was prepared), and both go back.
    #[test]
    fn submissions_writing_apart_stay_in_flight_together() {
        let Some(host) = host() else { return };
        if !overlap() {
            return; // VICTORY_GPU_OVERLAP=0
        }
        let _queues = crate::queue::tests::TITLE_QUEUES.lock().unwrap_or_else(|e| e.into_inner());
        let Some((p, a)) = vram_copy(1) else { return };
        let Some((q, b)) = vram_copy(1) else { return release(p, Some(a)) };
        let gate = host.hold_gpu();
        let started = Instant::now();
        host.submit(0, vec![Arc::new(writer(host, &a, p + 16, 0xaaaa_0001, false))]);
        host.submit(0, vec![Arc::new(writer(host, &b, q + 16, 0xbbbb_0002, false))]);
        let returned = started.elapsed();
        assert!(returned < Duration::from_secs(5), "back with the host GPU held ({returned:?})");
        assert_eq!(in_flight(host), 2, "both in flight");
        assert_eq!((word(p + 16), word(q + 16)), (0, 0), "nothing went back yet");
        drop(gate);
        host.complete_all();
        assert_eq!((word(p + 16), word(q + 16)), (0xaaaa_0001, 0xbbbb_0002));
        release(p, Some(a));
        release(q, Some(b));
    }

    /// The title changes a page a submission in flight writes: a later
    /// submission reading it completes that one first while it is prepared
    /// (its refresh would have put the title's page over the GPU's write in
    /// the copy), so the copy ends with both writes.
    #[test]
    fn a_refresh_of_a_page_changed_under_a_range_in_flight_waits_for_it() {
        let Some(host) = host() else { return };
        if !overlap() {
            return; // VICTORY_GPU_OVERLAP=0
        }
        let _queues = crate::queue::tests::TITLE_QUEUES.lock().unwrap_or_else(|e| e.into_inner());
        let Some((p, a)) = vram_copy(1) else { return };
        let gate = host.hold_gpu();
        host.submit(0, vec![Arc::new(writer(host, &a, p + 16, 0xcafe_0001, false))]);
        // SAFETY: a word of our pool (guarded: the write marks the page).
        unsafe { ((p + 64) as *mut u32).write_volatile(0x3333) };
        let released = release_gpu_after(gate, 500);
        let started = Instant::now();
        host.submit(0, vec![Arc::new(reader(host, &a))]);
        let took = started.elapsed();
        assert!(took >= Duration::from_millis(400), "waited for the first ({took:?})");
        assert_eq!(word(p + 16), 0xcafe_0001, "the first completed while the second was prepared");
        released.join().unwrap();
        host.complete_all();
        assert_eq!(a.read_host((p + 16) as u64, 1)[0], 0xcafe_0001, "the GPU's write stays in the copy");
        assert_eq!(a.read_host((p + 64) as u64, 1)[0], 0x3333, "the title's write reaches it");
        release(p, Some(a));
    }

    /// A submission declaring an unmarked range a submission in flight
    /// declares too waits for it before it is prepared; the later write
    /// wins.
    #[test]
    fn a_range_declared_again_waits_for_the_one_in_flight() {
        let Some(host) = host() else { return };
        if !overlap() {
            return; // VICTORY_GPU_OVERLAP=0
        }
        let _queues = crate::queue::tests::TITLE_QUEUES.lock().unwrap_or_else(|e| e.into_inner());
        let Some((p, a)) = vram_copy(1) else { return };
        let gate = host.hold_gpu();
        host.submit(0, vec![Arc::new(writer(host, &a, p + 16, 0x0001_0001, false))]);
        let second = writer(host, &a, p + 16, 0x0002_0002, false);
        let released = release_gpu_after(gate, 500);
        let started = Instant::now();
        host.submit(0, vec![Arc::new(second)]);
        let took = started.elapsed();
        assert!(took >= Duration::from_millis(400), "waited for the first ({took:?})");
        assert_eq!(word(p + 16), 0x0001_0001, "the first completed before the second was prepared");
        released.join().unwrap();
        host.complete_all();
        assert_eq!(word(p + 16), 0x0002_0002);
        release(p, Some(a));
    }

    /// Two submissions marking pages of one copy stay in flight together;
    /// each completion takes its own marks (the second's shaders set the
    /// same dword of the bitmap anew), and both pages go back.
    #[test]
    fn marking_submissions_take_their_own_marks() {
        let Some(host) = host() else { return };
        if !overlap() {
            return; // VICTORY_GPU_OVERLAP=0
        }
        let _queues = crate::queue::tests::TITLE_QUEUES.lock().unwrap_or_else(|e| e.into_inner());
        let Some((p, a)) = vram_copy(2) else { return };
        if !a.marked() {
            return release(p, Some(a)); // no mark bits left
        }
        let first = writer(host, &a, p + 16, 0x1111_0001, true);
        let second = writer(host, &a, p + PAGE + 16, 0x2222_0002, true);
        if !crate::shadow::captures_write_marks() {
            return release(p, Some(a)); // marks in system memory: barriers
        }
        let gate = host.hold_gpu();
        host.submit(0, vec![Arc::new(first)]);
        host.submit(0, vec![Arc::new(second)]);
        assert_eq!(in_flight(host), 2, "both in flight");
        drop(gate);
        host.complete_all();
        assert_eq!((word(p + 16), word(p + PAGE + 16)), (0x1111_0001, 0x2222_0002));
        release(p, Some(a));
    }

    /// A preparing's refreshes complete the submissions in flight that
    /// write their memory first: through another copy, or by bottom-of-pipe
    /// writes (even of a copy that is clean), or under pages the title
    /// changed in the same copy's range; not for a range of the same copy
    /// whose pages are unchanged.
    #[test]
    fn refreshes_wait_for_writers_of_their_memory_in_flight() {
        if host().is_none() {
            return;
        }
        let Some((p, a)) = vram_copy(2) else { return };
        const ELSEWHERE: u64 = u64::MAX / 2;
        let copy = Arc::as_ptr(&a) as usize;
        let settled = std::cell::RefCell::new(Vec::new());
        let mut settle = |owner: u64, why: Settled| {
            settled.borrow_mut().push((owner, format!("{why:?}")));
            crate::shadow::land(owner);
            a.copy_back(owner, p + PAGE, 64);
        };
        let mut batch = Batch::default();
        batch.add(&a, p, 2 * PAGE);
        batch.flush(1, &mut settle);
        // Another copy of page 1 written in flight.
        crate::shadow::fly(ELSEWHERE, vec![(p + PAGE, p + PAGE + 64, 1)]);
        batch.add(&a, p, 2 * PAGE);
        batch.flush(2, &mut settle);
        // The same copy: none.
        crate::shadow::fly(ELSEWHERE, vec![(p + PAGE, p + PAGE + 64, copy)]);
        batch.add(&a, p, 2 * PAGE);
        batch.flush(2, &mut settle);
        crate::shadow::land(ELSEWHERE);
        // A bottom-of-pipe write into page 0, the copy clean.
        crate::shadow::fly(ELSEWHERE, vec![(p + 8, p + 16, crate::shadow::LABEL)]);
        assert!(a.clean());
        let mut batch = Batch::default();
        batch.add(&a, p, 2 * PAGE);
        batch.flush(3, &mut settle);
        // Shaders of a submission in flight write page 1; the title changes it.
        a.writing(ELSEWHERE, p + PAGE, 64, false);
        // SAFETY: a byte of our pool.
        unsafe { ((p + PAGE + 200) as *mut u8).write_volatile(5) };
        let mut batch = Batch::default();
        batch.add(&a, p, 2 * PAGE);
        assert_eq!(batch.flush(4, &mut settle), PAGE, "page 1 once that one went back");
        assert_eq!(
            settled.into_inner(),
            [(ELSEWHERE, "Alias".to_owned()), (ELSEWHERE, "Label".to_owned()), (ELSEWHERE, "Same".to_owned())]
        );
        release(p, Some(a));
    }

    /// Title copies at submission complete a submission in flight first only
    /// where their source or destination meets what it writes (its declared
    /// ranges, through any copy, or its bottom-of-pipe writes); others run
    /// with it in flight. Both kinds of copy.
    #[test]
    fn title_copies_wait_only_for_writers_of_their_ranges() {
        let Some(host) = host() else { return };
        const ELSEWHERE: u64 = u64::MAX / 2 + 1;
        let p = crate::watch::tests::pool(4);
        let settled = std::cell::RefCell::new(Vec::new());
        let mut settle = |owner: u64, why: Settled| {
            settled.borrow_mut().push((owner, format!("{why:?}")));
            crate::shadow::land(owner);
        };
        let list = host.new_list().unwrap();
        // Page 0 → page 1 meets nothing in flight; page 2 → page 3 reads what
        // the first writes; page 1 → page 3 + 1 KiB overwrites the second's
        // bottom-of-pipe write.
        list.defer_copy((p + PAGE) as *mut u8, p as *const u8, 64);
        list.defer_copy_memory((p + 3 * PAGE) as *mut u8, (p + 2 * PAGE) as *const u8, 64);
        list.defer_copy((p + 3 * PAGE + 1024) as *mut u8, (p + PAGE) as *const u8, 64);
        crate::shadow::fly(ELSEWHERE, vec![(p + 2 * PAGE + 32, p + 2 * PAGE + 40, 1)]);
        crate::shadow::fly(ELSEWHERE + 1, vec![(p + 3 * PAGE + 1032, p + 3 * PAGE + 1040, crate::shadow::LABEL)]);
        list.run_deferred(&mut Batch::default(), 1, &mut settle);
        assert_eq!(
            settled.into_inner(),
            [(ELSEWHERE, "Copy".to_owned()), (ELSEWHERE + 1, "Copy".to_owned())]
        );
        list.close();
        crate::watch::tests::free(p);
    }
}


#[cfg(test)]
mod withholding {
    /// A list in which a program using title memory missed memory: what
    /// that program wrote is withheld from title memory and leaves the
    /// copies at once; what an ordinary program wrote in the same list goes
    /// back and stays in its copy (run light13: the imposter transforms in
    /// the BVH builder's list); an ordinary range overlapping a withheld one
    /// is withheld with it, since its copy-back would take the other's bytes
    /// along.
    #[test]
    fn only_programs_using_title_memory_are_withheld() {
        use crate::watch::PAGE;
        let Some(host) = super::host() else {
            return; // no host GPU here
        };
        let pools = [crate::watch::tests::pool(1), crate::watch::tests::pool(1)];
        for &p in &pools {
            // SAFETY: our pool, not guarded yet.
            unsafe { ((p + 16) as *mut u8).write_volatile(0x11) };
            crate::shadow::register(p, p as u64, PAGE as u64);
        }
        let copies: Vec<_> = pools.iter().map(|&p| crate::shadow::find(p as u64, PAGE as u64).unwrap()).collect();
        let list = host.new_list().unwrap();
        // (copy, start, bytes, the writer uses title memory, byte written at +0 and +16)
        let writes = [(0, 0, 32, false, 0x22), (1, 16, 16, true, 0x33), (1, 0, 32, false, 0x44)];
        for &(k, at, len, memory, value) in &writes {
            let _writer = list.writer_uses_memory(memory);
            let (copy, address) = (copies[k].clone(), (pools[k] + at) as u64);
            copy.refresh(&list, address, len);
            copy.gpu_writes(&list, address, len, false);
            // The shader's write, after the step that keeps the range's bytes.
            list.defer_host(move || copy.write_host(address, &[value]));
        }
        list.defer_missed(|| true);
        list.close();
        host.execute(&[&list]);
        // SAFETY: bytes of our pools.
        let title = |k: usize, at: usize| unsafe { ((pools[k] + at) as *const u8).read_volatile() };
        let copy = |k: usize, at: usize| {
            copies[k].update(0, pools[k], PAGE).unwrap();
            copies[k].read_host((pools[k] + at) as u64, 1)[0] & 0xff
        };
        assert_eq!((title(0, 0), copy(0, 0)), (0x22, 0x22), "an ordinary program's write goes back and stays");
        assert_eq!((title(1, 16), copy(1, 16)), (0x11, 0x11), "withheld, then dropped from the copy");
        assert_eq!((title(1, 0), copy(1, 0)), (0, 0), "overlapping a withheld range: withheld with it");
        for (&p, copy) in pools.iter().zip(copies) {
            crate::shadow::unregister(p, p as u64);
            drop(copy);
            crate::watch::tests::free(p);
        }
    }
}

#[cfg(all(test, feature = "diagnostics"))]
mod present_plane_tests {
    use super::*;

    #[test]
    fn compositor_failure_presents_primary_in_the_same_frame() {
        let Some(host) = host() else {
            eprintln!("skipping PresentX recovery: no D3D12 device");
            return;
        };
        let list = host.new_list().unwrap();
        let make_plane = |width, height, color: &[f32; 4]| {
            let mut texture = None;
            // SAFETY: live device, valid render target description, and an open list.
            unsafe {
                host.device
                    .CreateCommittedResource(
                        &D3D12_HEAP_PROPERTIES {
                            Type: D3D12_HEAP_TYPE_DEFAULT,
                            ..Default::default()
                        },
                        D3D12_HEAP_FLAG_NONE,
                        &D3D12_RESOURCE_DESC {
                            Dimension: D3D12_RESOURCE_DIMENSION_TEXTURE2D,
                            Width: width,
                            Height: height,
                            DepthOrArraySize: 1,
                            MipLevels: 1,
                            Format: DXGI_FORMAT_R8G8B8A8_UNORM,
                            SampleDesc: DXGI_SAMPLE_DESC {
                                Count: 1,
                                Quality: 0,
                            },
                            Flags: D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET,
                            ..Default::default()
                        },
                        D3D12_RESOURCE_STATE_RENDER_TARGET,
                        None,
                        &mut texture,
                    )
                    .unwrap();
                let texture = texture.unwrap();
                list.list
                    .ClearRenderTargetView(host.rtv(&texture, None), color, None);
                texture
            }
        };
        let scene = make_plane(32, 32, &[1.0, 0.0, 0.0, 1.0]);
        let overlay = make_plane(64, 64, &[0.0, 0.0, 128.0 / 255.0, 128.0 / 255.0]);
        list.close();
        host.execute(&[&list]);
        let state = D3D12_RESOURCE_STATE_RENDER_TARGET.0 as u32;
        assert!(
            host.present(&[(&scene, state)], 1).is_some(),
            "single-plane control"
        );
        let first_pixel =
            |buffer: &ID3D12Resource, footprint: &D3D12_PLACED_SUBRESOURCE_FOOTPRINT| {
                let mut mapped = std::ptr::null_mut();
                // SAFETY: the present copy has completed. The footprint names
                // a live RGBA pixel inside the readback buffer.
                let rgba = unsafe {
                    buffer.Map(0, None, Some(&mut mapped)).unwrap();
                    let rgba = mapped
                        .cast::<u8>()
                        .add(footprint.Offset as usize)
                        .cast::<[u8; 4]>()
                        .read();
                    buffer.Unmap(0, None);
                    rgba
                };
                (footprint.Footprint.Width, footprint.Footprint.Height, rgba)
            };

        let mut attempted = false;
        let mut fallback = None;
        let shown = host.present_with_compositor(
            &[(&scene, state), (&overlay, state)],
            2,
            |_, _| {
                attempted = true;
                Err(windows::core::Error::from_hresult(
                    windows::Win32::Foundation::E_FAIL,
                ))
            },
            |buffer, footprint| fallback = Some(first_pixel(buffer, footprint)),
        );
        assert!(attempted);
        assert!(
            shown.is_some(),
            "the primary plane must still be shown in this frame"
        );
        assert_eq!(fallback, Some((32, 32, [255, 0, 0, 255])));
        let presentation = host.presenter.lock().unwrap_or_else(|e| e.into_inner());
        let swap_chain = presentation.presenter.as_ref().unwrap();
        assert_eq!((swap_chain.width, swap_chain.height), (32, 32));
        drop(presentation);

        let mut composite = None;
        assert!(
            host.present_with_compositor(
                &[(&scene, state), (&overlay, state)],
                3,
                Compositor::new,
                |buffer, footprint| composite = Some(first_pixel(buffer, footprint)),
            )
            .is_some(),
            "compositor initialization can be retried after failure"
        );
        // Premultiplied half-blue over opaque red.
        assert_eq!(composite, Some((64, 64, [127, 0, 128, 255])));
    }
}
