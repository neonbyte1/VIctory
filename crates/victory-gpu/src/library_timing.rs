//! What host libraries cost (diagnostics, `VICTORY_LIBRARY_TIMING=1`): the
//! work a host library records into a title's list (`external::record`,
//! the host FidelityFX upscaler) gets a GPU timestamp before and after it,
//! read when the list is reset (its submission has finished), and the CPU
//! time of the recording call. Every 60 calls a line reports their GPU
//! time (average, median, slowest) and CPU time, to judge an upscaler's
//! overhead against the frame it is part of; each call is logged at debug
//! level (`VICTORY_LOG=victory_gpu::library_timing=debug`).
//!
//! shortcut: 64 calls can be in flight; a call past that is not timed.

use std::collections::HashMap;
use std::sync::{Mutex, OnceLock};
use std::time::Duration;

use windows::Win32::Graphics::Direct3D12::*;
use windows::Win32::Graphics::Dxgi::Common::DXGI_SAMPLE_DESC;

use crate::host::{HostList, host};

/// Calls in flight, each a pair of timestamps.
const PAIRS: u32 = 64;
/// Calls per report.
const REPORT: usize = 60;

struct Timer {
    heap: ID3D12QueryHeap,
    /// Where the pairs are resolved, 16 bytes each, read once the list ran.
    back: ID3D12Resource,
    frequency: u64,
    state: Mutex<State>,
}

// SAFETY: D3D12 objects are free-threaded; the rest is behind the mutex.
unsafe impl Send for Timer {}
// SAFETY: as above.
unsafe impl Sync for Timer {}

#[derive(Default)]
struct State {
    next: u32,
    busy: Vec<bool>,
    /// Per host list: its timed calls (pair, CPU time), read at its reset.
    pending: HashMap<usize, Vec<(u32, Duration)>>,
    gpu_ms: Vec<f64>,
    cpu: Vec<Duration>,
}

fn on() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_LIBRARY_TIMING").is_some_and(|v| v == "1"))
}

fn timer() -> Option<&'static Timer> {
    static TIMER: OnceLock<Option<Timer>> = OnceLock::new();
    TIMER
        .get_or_init(|| {
            let host = host()?;
            let mut heap: Option<ID3D12QueryHeap> = None;
            let mut back: Option<ID3D12Resource> = None;
            let readback = D3D12_HEAP_PROPERTIES { Type: D3D12_HEAP_TYPE_READBACK, ..Default::default() };
            let desc = D3D12_RESOURCE_DESC {
                Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
                Width: u64::from(PAIRS) * 16,
                Height: 1,
                DepthOrArraySize: 1,
                MipLevels: 1,
                SampleDesc: DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
                Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
                ..Default::default()
            };
            // SAFETY: plain object creation; our out pointers.
            unsafe {
                let query = D3D12_QUERY_HEAP_DESC { Type: D3D12_QUERY_HEAP_TYPE_TIMESTAMP, Count: PAIRS * 2, NodeMask: 0 };
                host.device.CreateQueryHeap(&query, &mut heap).ok()?;
                host.device
                    .CreateCommittedResource(&readback, D3D12_HEAP_FLAG_NONE, &desc, D3D12_RESOURCE_STATE_COPY_DEST, None, &mut back)
                    .ok()?;
            }
            tracing::info!("host library work timed on the GPU (VICTORY_LIBRARY_TIMING)");
            Some(Timer {
                heap: heap?,
                back: back?,
                frequency: host.timestamp_frequency(),
                state: Mutex::new(State { busy: vec![false; PAIRS as usize], ..State::default() }),
            })
        })
        .as_ref()
}

/// Before a host library records into `list`: its first timestamp, and the
/// pair to end with [`end`]. `None` when timing is off or 64 are in flight.
pub(crate) fn begin(list: &HostList) -> Option<u32> {
    if !on() {
        return None;
    }
    let t = timer()?;
    let pair = {
        let mut s = t.state.lock().unwrap_or_else(|e| e.into_inner());
        let start = s.next;
        let pair = (0..PAIRS).map(|i| (start + i) % PAIRS).find(|&p| !s.busy[p as usize])?;
        s.busy[pair as usize] = true;
        s.next = (pair + 1) % PAIRS;
        pair
    };
    // SAFETY: an open list and our heap.
    unsafe { list.list.EndQuery(&t.heap, D3D12_QUERY_TYPE_TIMESTAMP, pair * 2) };
    Some(pair)
}

/// After the library recorded into `list` (its call took `cpu`): the second
/// timestamp, resolved for reading at the list's reset.
pub(crate) fn end(list: &HostList, pair: u32, cpu: Duration) {
    let Some(t) = timer() else { return };
    // SAFETY: an open list, our heap and read-back buffer.
    unsafe {
        list.list.EndQuery(&t.heap, D3D12_QUERY_TYPE_TIMESTAMP, pair * 2 + 1);
        list.list.ResolveQueryData(&t.heap, D3D12_QUERY_TYPE_TIMESTAMP, pair * 2, 2, &t.back, u64::from(pair) * 16);
    }
    let mut s = t.state.lock().unwrap_or_else(|e| e.into_inner());
    s.pending.entry(std::ptr::from_ref(list) as usize).or_default().push((pair, cpu));
}

/// At `list`'s reset (its submission finished): reads its timed calls.
pub(crate) fn list_reset(list: &HostList) {
    if !on() {
        return;
    }
    let Some(t) = timer() else { return };
    let mut s = t.state.lock().unwrap_or_else(|e| e.into_inner());
    let Some(calls) = s.pending.remove(&(std::ptr::from_ref(list) as usize)) else { return };
    let mut data = std::ptr::null_mut();
    // SAFETY: a read-back buffer the finished list resolved into.
    let mapped = unsafe { t.back.Map(0, Some(&D3D12_RANGE { Begin: 0, End: PAIRS as usize * 16 }), Some(&mut data)) };
    for (pair, cpu) in calls {
        s.busy[pair as usize] = false;
        if mapped.is_ok() {
            // SAFETY: the pair's 16 bytes inside the mapped buffer.
            let [a, b] = unsafe { data.cast::<[u64; 2]>().add(pair as usize).read_unaligned() };
            let gpu_ms = ticks_ms(a, b, t.frequency);
            tracing::debug!(gpu_ms = format_args!("{gpu_ms:.3}"), cpu_us = cpu.as_micros(), "host library call");
            s.gpu_ms.push(gpu_ms);
            s.cpu.push(cpu);
        }
    }
    if mapped.is_ok() {
        // SAFETY: mapped above; nothing written.
        unsafe { t.back.Unmap(0, Some(&D3D12_RANGE { Begin: 0, End: 0 })) };
    }
    if s.gpu_ms.len() >= REPORT {
        let (gpu, cpu) = (std::mem::take(&mut s.gpu_ms), std::mem::take(&mut s.cpu));
        let r = report(&gpu, &cpu);
        tracing::info!(
            calls = gpu.len(),
            gpu_avg_ms = format_args!("{:.2}", r.avg),
            gpu_median_ms = format_args!("{:.2}", r.median),
            gpu_max_ms = format_args!("{:.2}", r.max),
            cpu_avg_us = r.cpu_avg_us,
            "host library work (FidelityFX upscale) on the GPU"
        );
    }
}

/// Milliseconds between two timestamps; 0 if they are out of order (a call
/// whose list did not run).
fn ticks_ms(a: u64, b: u64, frequency: u64) -> f64 {
    b.saturating_sub(a) as f64 * 1e3 / frequency.max(1) as f64
}

struct Report {
    avg: f64,
    median: f64,
    max: f64,
    cpu_avg_us: u128,
}

fn report(gpu: &[f64], cpu: &[Duration]) -> Report {
    let mut sorted = gpu.to_vec();
    sorted.sort_by(f64::total_cmp);
    let n = sorted.len().max(1);
    Report {
        avg: sorted.iter().sum::<f64>() / n as f64,
        median: sorted.get(sorted.len() / 2).copied().unwrap_or(0.0),
        max: sorted.last().copied().unwrap_or(0.0),
        cpu_avg_us: cpu.iter().map(Duration::as_micros).sum::<u128>() / cpu.len().max(1) as u128,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ticks_become_milliseconds() {
        assert_eq!(ticks_ms(1_000, 26_000, 25_000_000), 1.0);
        assert_eq!(ticks_ms(5, 3, 1_000), 0.0);
    }

    #[test]
    fn reports_average_median_slowest_and_cpu() {
        let gpu = [1.0, 3.0, 2.0, 10.0];
        let cpu = [Duration::from_micros(10), Duration::from_micros(30)];
        let r = report(&gpu, &cpu);
        assert_eq!((r.avg, r.median, r.max, r.cpu_avg_us), (4.0, 3.0, 10.0, 20));
    }

    /// A timed host library call on a real list: the GPU time is read at
    /// the list's reset.
    #[test]
    fn a_timed_call_is_read_at_the_lists_reset() {
        let Some(host) = host() else { return };
        let Some(t) = timer() else { return };
        let list = host.new_list().unwrap();
        let pair = {
            let mut s = t.state.lock().unwrap();
            s.busy[0] = true;
            0
        };
        // SAFETY: an open list and our heap.
        unsafe { list.list.EndQuery(&t.heap, D3D12_QUERY_TYPE_TIMESTAMP, pair * 2) };
        end(&list, pair, Duration::from_micros(5));
        list.close();
        host.execute(&[&list]);
        // `on()` is off in tests: read through the same path directly.
        let calls = t.state.lock().unwrap().pending.get(&(std::ptr::from_ref(&list) as usize)).cloned();
        assert_eq!(calls.map(|c| c.len()), Some(1));
        let mut data = std::ptr::null_mut();
        // SAFETY: the finished list resolved into the buffer.
        let [a, b] = unsafe {
            t.back.Map(0, None, Some(&mut data)).unwrap();
            let pair = data.cast::<[u64; 2]>().read_unaligned();
            t.back.Unmap(0, None);
            pair
        };
        assert!(b >= a && a != 0, "timestamps {a} {b}");
    }
}
