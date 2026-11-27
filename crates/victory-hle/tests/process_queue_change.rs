//! A call begun with no queue of its own runs on the process task queue of
//! when it began: setting another process queue while it is in flight does
//! not move its work or completion (as Microsoft's libHttpClient keeps the
//! queue it resolved, `Source/Task/AsyncLib.cpp`). A binary of its own: the
//! process queue is process-wide.

use std::sync::atomic::{AtomicU32, Ordering};

use victory_hle::modules::xasync::{
    AsyncBlock, Queue, XAsyncGetStatus, XAsyncRun, XTaskQueueCloseHandle, XTaskQueueCreate,
    XTaskQueueDispatch, XTaskQueueGetCurrentProcessTaskQueue, XTaskQueueSetCurrentProcessTaskQueue,
};
use victory_hle::modules::xgameruntime::{HRESULT, S_OK};

const MANUAL: u32 = 0;
const WORK: u32 = 0;
const COMPLETION: u32 = 1;

static COMPLETED: AtomicU32 = AtomicU32::new(0);

extern "system" fn work(_: *mut AsyncBlock) -> HRESULT {
    S_OK
}
extern "system" fn completed(_: *mut AsyncBlock) {
    COMPLETED.fetch_add(1, Ordering::SeqCst);
}

fn manual() -> *const Queue {
    let mut q = std::ptr::null();
    assert_eq!(XTaskQueueCreate(MANUAL, MANUAL, &mut q), S_OK);
    q
}

#[test]
fn a_call_stays_on_the_process_queue_it_began_on() {
    let mut saved = std::ptr::null();
    assert!(XTaskQueueGetCurrentProcessTaskQueue(&mut saved));
    let (a, b) = (manual(), manual());
    XTaskQueueSetCurrentProcessTaskQueue(a);
    let mut block = AsyncBlock {
        queue: std::ptr::null(),
        context: std::ptr::null_mut(),
        callback: Some(completed),
        internal: [0; 4],
    };
    assert_eq!(XAsyncRun(&mut block, Some(work)), S_OK);
    // The title sets another process queue while the call is in flight.
    XTaskQueueSetCurrentProcessTaskQueue(b);
    assert!(!XTaskQueueDispatch(b, WORK, 0), "no work on the new queue");
    assert!(
        XTaskQueueDispatch(a, WORK, 0),
        "the work on the queue it began on"
    );
    assert!(
        !XTaskQueueDispatch(b, COMPLETION, 0),
        "no completion on the new queue"
    );
    assert!(
        XTaskQueueDispatch(a, COMPLETION, 0),
        "the completion on the queue it began on"
    );
    assert_eq!(COMPLETED.load(Ordering::SeqCst), 1);
    assert_eq!(XAsyncGetStatus(&mut block, false), S_OK);
    XTaskQueueSetCurrentProcessTaskQueue(saved);
    for q in [saved, a, b] {
        XTaskQueueCloseHandle(q);
    }
}
