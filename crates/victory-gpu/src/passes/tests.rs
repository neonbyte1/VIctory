use super::*;
use crate::host::{Host, HostList};
use windows::Win32::Graphics::Direct3D12::*;
use windows::Win32::Graphics::Dxgi::Common::DXGI_SAMPLE_DESC;

#[test]
fn the_switch_says_how_often() {
    assert_eq!(parse_every(None), 0);
    assert_eq!(parse_every(Some("0")), 0);
    assert_eq!(parse_every(Some("1")), 1);
    assert_eq!(parse_every(Some("4")), 4);
    assert_eq!(parse_every(Some("yes")), 1, "on: every frame");
}

/// Each command is charged from its stamp to the next; a list of ours has
/// its start as a row of its own; a list whose stamps were not resolved is
/// left out.
#[test]
fn each_command_is_charged_to_its_stamp_and_the_next() {
    let keys = [
        Key::class(Class::ListStart),
        Key::class(Class::Dispatch),
        Key::class(Class::Copy),
        Key::class(Class::Dispatch),
    ];
    let mut w = Window::default();
    let mut span = (u64::MAX, 0);
    w.add(
        &Timed {
            stamps: &[100, 130, 430, 440, 640],
            keys: &keys,
            ours: false,
            untimed: 2,
        },
        &mut span,
    );
    w.add(
        &Timed {
            stamps: &[700, 750],
            keys: &keys[..1],
            ours: true,
            untimed: 0,
        },
        &mut span,
    );
    w.add(
        &Timed {
            stamps: &[0, 0],
            keys: &keys[..1],
            ours: false,
            untimed: 0,
        },
        &mut span,
    );
    let row = |c| w.rows.get(&(c, false, 0)).map(|r| (r.ticks, r.count));
    assert_eq!(row(Class::ListStart), Some((30, 1)));
    assert_eq!(row(Class::Dispatch), Some((500, 2)));
    assert_eq!(row(Class::Copy), Some((10, 1)));
    assert_eq!(row(Class::OurList), Some((50, 1)));
    assert_eq!(
        (w.lists, w.list_ticks, w.commands, w.untimed),
        (2, 590, 5, 2)
    );
    assert_eq!(span, (100, 750));
}

/// A loop of `n` steps per thread, kept by a write it almost never makes.
const SOURCE: &str = r"
cbuffer c : register(b0) { uint n; };
RWStructuredBuffer<uint> o : register(u0);
[numthreads(64, 1, 1)]
void main(uint3 id : SV_DispatchThreadID) {
    uint x = id.x;
    for (uint i = 0; i < n; i++) { x = x * 1664525u + 1013904223u; }
    if (x == 0x12345678u) { o[id.x & 63] = x; }
}
";

/// The host, if it is a real GPU (the loops take WARP too long).
fn real_gpu() -> Option<&'static Host> {
    let host = crate::host::host().filter(|_| !crate::switch::on("VICTORY_D3D12_WARP"));
    if host.is_none() {
        eprintln!("skipped: no host GPU");
    }
    host
}

struct Looper {
    root: ID3D12RootSignature,
    pso: ID3D12PipelineState,
    out: ID3D12Resource,
}

impl Looper {
    fn new(host: &Host) -> Looper {
        let (root, pso) = crate::compute::compute(
            &host.device,
            windows::core::s!("pass timing test"),
            SOURCE,
            0,
            &[
                crate::compute::constants(0, 1),
                crate::compute::root_view(D3D12_ROOT_PARAMETER_TYPE_UAV, 0, 0),
            ],
        )
        .expect("the test shader compiles");
        let out = host.default_uav_buffer(256).expect("a buffer");
        Looper { root, pso, out }
    }

    /// A list (its passes timed or not) of `dispatches` (steps, groups),
    /// each noted as a dispatch, closed.
    fn list(&self, host: &Host, timed: bool, dispatches: &[(u32, u32)]) -> HostList {
        FORCE.store(timed, Ordering::Relaxed);
        let list = host.new_list().expect("a list");
        FORCE.store(false, Ordering::Relaxed);
        // SAFETY: an open list; our live root signature, pipeline, buffer.
        unsafe {
            list.list.SetComputeRootSignature(&self.root);
            list.list.SetPipelineState(&self.pso);
            list.list
                .SetComputeRootUnorderedAccessView(1, self.out.GetGPUVirtualAddress());
        }
        for &(steps, groups) in dispatches {
            // SAFETY: as above.
            unsafe { list.list.SetComputeRoot32BitConstant(0, steps, 0) };
            list.note_pass(|| Key::class(Class::Dispatch), String::new);
            // SAFETY: as above.
            unsafe { list.list.Dispatch(groups, 1, 1) };
        }
        list
    }
}

/// The ticks of each dispatch of a timed list that ran.
fn dispatch_ticks(list: &HostList) -> Vec<u64> {
    let mut ticks = Vec::new();
    list.pass_times(&mut |t| {
        ticks = t
            .keys
            .iter()
            .zip(t.stamps.windows(2))
            .filter(|(k, _)| k.class == Class::Dispatch)
            .map(|(_, s)| s[1].saturating_sub(s[0]))
            .collect();
    });
    ticks
}

/// A timed dispatch reports a plausible time: one with 4 times the loop
/// steps takes 2 to 8 times as long, and neither takes under 5 us or over
/// a second.
#[test]
fn a_timed_dispatch_takes_a_plausible_time() {
    let Some(host) = real_gpu() else {
        return;
    };
    let looper = Looper::new(host);
    let list = looper.list(host, true, &[(4096, 65535), (4 * 4096, 65535)]);
    list.close();
    host.execute(&[&list]);
    let frequency = host.timestamp_frequency() as f64;
    let ms: Vec<f64> = dispatch_ticks(&list)
        .iter()
        .map(|&t| t as f64 * 1e3 / frequency)
        .collect();
    eprintln!("timed dispatches: {ms:?} ms");
    assert_eq!(ms.len(), 2, "{ms:?}");
    assert!(ms[0] > 0.005 && ms[1] < 1000.0, "{ms:?}");
    let ratio = ms[1] / ms[0];
    assert!((2.0..8.0).contains(&ratio), "{ms:?}: {ratio}");
}

/// What a stamp costs the GPU: 2000 empty dispatches timed one by one
/// against the same untimed between two stamps of their own.
#[test]
fn a_stamp_costs_the_gpu_little() {
    let Some(host) = real_gpu() else {
        return;
    };
    const COMMANDS: u32 = 2000;
    let looper = Looper::new(host);
    let empty = vec![(0, 1); COMMANDS as usize];
    let timed = looper.list(host, true, &empty);
    timed.close();
    host.execute(&[&timed]);
    let stamped: u64 = dispatch_ticks(&timed).iter().sum();

    let plain = looper.list(host, false, &[]);
    let (heap, back) = stamps(host);
    // SAFETY: an open list; our heap and read-back buffer.
    unsafe { plain.list.EndQuery(&heap, D3D12_QUERY_TYPE_TIMESTAMP, 0) };
    for _ in 0..COMMANDS {
        // SAFETY: as above, the looper's state bound.
        unsafe {
            plain.list.SetComputeRoot32BitConstant(0, 0, 0);
            plain.list.Dispatch(1, 1, 1);
        }
    }
    // SAFETY: as above.
    unsafe {
        plain.list.EndQuery(&heap, D3D12_QUERY_TYPE_TIMESTAMP, 1);
        plain
            .list
            .ResolveQueryData(&heap, D3D12_QUERY_TYPE_TIMESTAMP, 0, 2, &back, 0);
    }
    plain.close();
    host.execute(&[&plain]);
    let mut data = std::ptr::null_mut();
    // SAFETY: the read-back buffer the finished list resolved two stamps into.
    let untimed = unsafe {
        back.Map(0, None, Some(&mut data)).expect("mapped");
        let s = std::slice::from_raw_parts(data.cast::<u64>(), 2);
        let ticks = s[1].saturating_sub(s[0]);
        back.Unmap(0, None);
        ticks
    };
    let frequency = host.timestamp_frequency() as f64;
    let us = |ticks: u64| ticks as f64 * 1e6 / frequency;
    let per = (us(stamped) - us(untimed)) / f64::from(COMMANDS);
    eprintln!(
        "{COMMANDS} empty dispatches: {:.1} us timed one by one, {:.1} us untimed: {per:.2} us per stamp",
        us(stamped),
        us(untimed)
    );
    assert!(per < 20.0, "{per} us per stamp");
}

/// A heap of two timestamps and a read-back buffer for them.
fn stamps(host: &Host) -> (ID3D12QueryHeap, ID3D12Resource) {
    let mut heap: Option<ID3D12QueryHeap> = None;
    let mut back: Option<ID3D12Resource> = None;
    let readback = D3D12_HEAP_PROPERTIES {
        Type: D3D12_HEAP_TYPE_READBACK,
        ..Default::default()
    };
    let desc = D3D12_RESOURCE_DESC {
        Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
        Width: 16,
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
    // SAFETY: plain object creation; our out pointers.
    unsafe {
        host.device
            .CreateQueryHeap(
                &D3D12_QUERY_HEAP_DESC {
                    Type: D3D12_QUERY_HEAP_TYPE_TIMESTAMP,
                    Count: 2,
                    NodeMask: 0,
                },
                &mut heap,
            )
            .expect("a query heap");
        host.device
            .CreateCommittedResource(
                &readback,
                D3D12_HEAP_FLAG_NONE,
                &desc,
                D3D12_RESOURCE_STATE_COPY_DEST,
                None,
                &mut back,
            )
            .expect("a read-back buffer");
    }
    (heap.expect("made"), back.expect("made"))
}
