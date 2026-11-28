//! With the process task queue set to null, a call given no queue fails
//! with `E_NO_TASK_QUEUE`, as documented for
//! `XTaskQueueSetCurrentProcessTaskQueue`. A binary of its own: the process
//! queue is process-wide, and the crate's unit tests begin calls with null
//! queues on other threads.

use std::ffi::c_void;
use std::sync::atomic::{AtomicU32, Ordering};

use victory_hle::modules::xasync::{
    AsyncBlock, E_NO_TASK_QUEUE, ProviderData, Queue, XAsyncBegin, XAsyncGetStatus, XAsyncRun,
    XTaskQueueCloseHandle, XTaskQueueCreate, XTaskQueueDispatch,
    XTaskQueueGetCurrentProcessTaskQueue, XTaskQueueSetCurrentProcessTaskQueue,
    XTaskQueueSubmitDelayedCallback, finished,
};
use victory_hle::modules::xgameruntime::{HRESULT, S_OK};

static CALLED: AtomicU32 = AtomicU32::new(0);

extern "system" fn provider(_: u32, _: *const ProviderData) -> HRESULT {
    CALLED.fetch_add(1, Ordering::SeqCst);
    S_OK
}
extern "system" fn work(_: *mut AsyncBlock) -> HRESULT {
    CALLED.fetch_add(1, Ordering::SeqCst);
    S_OK
}
extern "system" fn completed(_: *mut AsyncBlock) {
    CALLED.fetch_add(1, Ordering::SeqCst);
}
extern "system" fn callback(_: *mut c_void, _: bool) {
    CALLED.fetch_add(1, Ordering::SeqCst);
}

fn block(queue: *const Queue) -> AsyncBlock {
    AsyncBlock {
        queue,
        context: std::ptr::null_mut(),
        callback: Some(completed),
        internal: [0; 4],
    }
}

#[test]
fn a_null_process_queue_refuses_calls_given_no_queue() {
    let mut saved = std::ptr::null();
    assert!(XTaskQueueGetCurrentProcessTaskQueue(&mut saved));
    assert!(!saved.is_null());

    XTaskQueueSetCurrentProcessTaskQueue(std::ptr::null());
    let mut current = saved;
    assert!(!XTaskQueueGetCurrentProcessTaskQueue(&mut current));
    assert!(current.is_null());

    let null = std::ptr::null();
    let mut b = block(null);
    assert_eq!(XAsyncRun(&mut b, Some(work)), E_NO_TASK_QUEUE);
    let mut b = block(null);
    let begun = XAsyncBegin(
        &mut b,
        std::ptr::null_mut(),
        std::ptr::null(),
        std::ptr::null(),
        Some(provider),
    );
    assert_eq!(begun, E_NO_TASK_QUEUE);
    // A call the runtime starts for the title fails the same way.
    let mut b = block(null);
    assert_eq!(finished(&mut b, std::ptr::null(), S_OK), E_NO_TASK_QUEUE);
    assert_eq!(
        XTaskQueueSubmitDelayedCallback(null, 0, 0, std::ptr::null_mut(), Some(callback)),
        E_NO_TASK_QUEUE
    );
    assert_eq!(CALLED.load(Ordering::SeqCst), 0, "nothing ran");

    // A queue of the title's own still works.
    let mut own = std::ptr::null();
    assert_eq!(XTaskQueueCreate(0, 0, &mut own), S_OK);
    let mut b = block(own);
    assert_eq!(XAsyncRun(&mut b, Some(work)), S_OK);
    assert!(XTaskQueueDispatch(own, 0, 0));
    assert!(XTaskQueueDispatch(own, 1, 0));
    assert_eq!(XAsyncGetStatus(&mut b, false), S_OK);
    assert_eq!(CALLED.load(Ordering::SeqCst), 2, "work and completion");
    XTaskQueueCloseHandle(own);

    // Restored, a null queue means the process queue again.
    XTaskQueueSetCurrentProcessTaskQueue(saved);
    XTaskQueueCloseHandle(saved);
    let mut b = block(null);
    assert_eq!(XAsyncRun(&mut b, Some(work)), S_OK);
    assert_eq!(XAsyncGetStatus(&mut b, true), S_OK);
    // The wait lasts until the completion routine has run: after it, nothing
    // of the runtime's touches the block, which goes as the test returns.
    assert_eq!(CALLED.load(Ordering::SeqCst), 4, "work and completion");
}
