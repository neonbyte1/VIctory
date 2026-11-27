use super::*;
use std::fmt::Debug;
use std::sync::Arc;
use tracing::{Event, Metadata, Subscriber, field, span};

struct Reports(Arc<Mutex<Vec<String>>>);

impl Subscriber for Reports {
    fn enabled(&self, metadata: &Metadata<'_>) -> bool {
        metadata.is_event() && metadata.target().starts_with("victory_gpu::host")
    }

    fn new_span(&self, _: &span::Attributes<'_>) -> span::Id {
        unreachable!()
    }

    fn record(&self, _: &span::Id, _: &span::Record<'_>) {}
    fn record_follows_from(&self, _: &span::Id, _: &span::Id) {}
    fn enter(&self, _: &span::Id) {}
    fn exit(&self, _: &span::Id) {}

    fn event(&self, event: &Event<'_>) {
        struct Message(String);
        impl field::Visit for Message {
            fn record_debug(&mut self, field: &field::Field, value: &dyn Debug) {
                if field.name() == "message" {
                    self.0 = format!("{value:?}");
                }
            }
        }
        let mut message = Message(String::new());
        event.record(&mut message);
        if message.0.starts_with("GPU timing:") || message.0.starts_with("  GPU ") {
            self.0.lock().unwrap().push(message.0);
        }
    }
}

fn copy_list(host: &Host, label: &str) -> (HostList, *mut u32) {
    let list = host.new_list().unwrap();
    let (upload, input) = list.upload(host, 8).unwrap();
    let (readback, output) = list.readback(host, 8).unwrap();
    // SAFETY: the upload mapping has space for both u32 values.
    unsafe {
        input.cast::<u32>().write(0x1234);
        input.cast::<u32>().add(1).write(0x5678);
    }
    for (i, offset) in [0, 4].into_iter().enumerate() {
        list.note_op(|| format!("copy [fixture {label}-{i}]"));
        // SAFETY: the resources are live and stay in upload and readback states.
        unsafe { list.list.CopyBufferRegion(&readback, offset, &upload, offset, 4) };
    }
    list.close();
    (list, output.cast())
}

#[test]
#[ignore = "requires an isolated WARP process with VICTORY_GPU_TIMING=1"]
fn timing_report_uses_each_submitted_list() {
    assert!(timing());
    assert!(crate::switch::on("VICTORY_D3D12_WARP"));
    println!("DRED requested: {}", dred_on());
    let host = host().expect("WARP device");
    let (a, a_out) = copy_list(host, "a");
    let (b, b_out) = copy_list(host, "b");
    let lines = Arc::new(Mutex::new(Vec::new()));
    let start = Instant::now();

    let counts = tracing::subscriber::with_default(Reports(lines.clone()), || {
        [
            ("single", &[&a][..]),
            ("two lists", &[&a, &b][..]),
            ("same list twice", &[&a, &a][..]),
        ]
        .into_iter()
        .enumerate()
        .map(|(i, (name, lists))| {
            host.execute(lists);
            // SAFETY: execute waits for the submission and both mappings are live.
            unsafe {
                assert_eq!(std::slice::from_raw_parts(a_out, 2), [0x1234, 0x5678]);
                if name == "two lists" {
                    assert_eq!(std::slice::from_raw_parts(b_out, 2), [0x1234, 0x5678]);
                }
            }
            let count = finish_timing_test_window(start + Duration::from_secs(6 * (i as u64 + 1)));
            println!("{name}: {count} commands");
            let report = std::mem::take(&mut *lines.lock().unwrap());
            if report.is_empty() {
                println!("  GPU timing: no report");
            }
            for line in report {
                println!("{line}");
            }
            count
        })
        .collect::<Vec<_>>()
    });
    // Each list: its `list start` row and its two copies.
    assert_eq!(counts, [3, 6, 6]);
}
