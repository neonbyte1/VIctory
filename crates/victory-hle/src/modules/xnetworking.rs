//! XNetworking as an offline console answers it: the network stack is up
//! (titles wait for that before anything else) but reaches nothing, the
//! connectivity hint and preferred port are reported once on registration,
//! and the title's security allow list cannot be fetched from the service.
#![allow(clippy::not_unsafe_ptr_arg_deref)]

use std::collections::BTreeMap;
use std::ffi::{c_char, c_void};
use std::num::NonZeroUsize;
use std::sync::{Condvar, Mutex, MutexGuard, PoisonError};
use std::thread::{self, ThreadId};

use super::xasync::{self, AsyncBlock, E_ABORT, E_INVALIDARG};
use super::xgameruntime::{
    Guid, HRESULT, Interface, Methods, Object, S_OK, add_ref, missing, put, text, token,
};

pub const PROVIDER: Guid = Guid::new(
    0x37e5_6907,
    0x2f10,
    0x41e8,
    [0xb7, 0x2f, 0x36, 0xed, 0xb1, 0x85, 0x33, 0x1a],
);

/// As Grand Theft Auto VI (2026) asks for it; the slots it calls (0x18-0x88)
/// take the same arguments as under the provider's own id (older titles).
pub const IID_2026: Guid = Guid::new(
    0xbf23_46b2,
    0x39af,
    0x4658,
    [0xb5, 0xea, 0x44, 0x71, 0x3c, 0x7e, 0x83, 0xb3],
);

pub static OBJECT: Object = Object::new("networking", &VTABLE);
pub const INTERFACES: &[Interface] =
    &[(PROVIDER, PROVIDER, &OBJECT), (PROVIDER, IID_2026, &OBJECT)];

/// `HRESULT_FROM_WIN32(ERROR_NETWORK_UNREACHABLE)`: what needs the service.
pub const UNREACHABLE: HRESULT = 0x8007_04CF_u32 as i32;

/// The documented default for the preferred local UDP multiplayer port.
const UDP_PORT: u16 = 3074;

/// The connectivity hint: level, cost, interface type, then four flags.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ConnectivityHint {
    pub level: u32,
    pub cost: u32,
    pub interface_type: u32,
    pub initialized: bool,
    pub approaching_data_limit: bool,
    pub over_data_limit: bool,
    pub roaming: bool,
}

/// Initialized, no connectivity (level 1), cost unknown (0).
const OFFLINE: ConnectivityHint = ConnectivityHint {
    level: 1,
    cost: 0,
    interface_type: 0,
    initialized: true,
    approaching_data_limit: false,
    over_data_limit: false,
    roaming: false,
};

type This = *const Object;

extern "system" fn udp_port(_: This, out: *mut u16) -> HRESULT {
    put(out, UDP_PORT)
}

/// Tells our port queries apart from other calls on the same block.
static PORT: u8 = 0;

extern "system" fn udp_port_async(_: This, block: *mut AsyncBlock) -> HRESULT {
    xasync::finished_with(block, (&raw const PORT).cast(), Ok(xasync::bytes(UDP_PORT)))
}
extern "system" fn udp_port_result(_: This, block: *mut AsyncBlock, out: *mut u16) -> HRESULT {
    xasync::result(block, (&raw const PORT).cast(), out)
}

static SECURITY: u8 = 0;

fn wide(text: *const u16) -> Option<String> {
    // SAFETY: the title passes a NUL-terminated UTF-16 string or null.
    (!text.is_null()).then(|| unsafe {
        let len = (0..).take_while(|&i| *text.add(i) != 0).count();
        String::from_utf16_lossy(std::slice::from_raw_parts(text, len))
    })
}

/// The allow list comes from the service: unreachable offline.
extern "system" fn security_info_async(
    _: This,
    url: *const c_char,
    block: *mut AsyncBlock,
) -> HRESULT {
    tracing::debug!(
        url = text(url).unwrap_or_default(),
        "security information: offline"
    );
    xasync::finished(block, (&raw const SECURITY).cast(), UNREACHABLE)
}
extern "system" fn security_info_utf16_async(
    _: This,
    url: *const u16,
    block: *mut AsyncBlock,
) -> HRESULT {
    tracing::debug!(
        url = wide(url).unwrap_or_default(),
        "security information (utf16): offline"
    );
    xasync::finished(block, (&raw const SECURITY).cast(), UNREACHABLE)
}
extern "system" fn security_info_size(_: This, block: *mut AsyncBlock, out: *mut usize) -> HRESULT {
    xasync::XAsyncGetResultSize(block, out)
}
extern "system" fn security_info_result(
    _: This,
    block: *mut AsyncBlock,
    size: usize,
    used: *mut usize,
    buffer: *mut u8,
    _info: *mut *const c_void,
) -> HRESULT {
    xasync::XAsyncGetResult(
        block,
        (&raw const SECURITY).cast(),
        size,
        buffer.cast(),
        used,
    )
}

/// No security information can exist offline, so no certificate passes.
extern "system" fn verify_certificate(
    _: This,
    _request: *mut c_void,
    _info: *const c_void,
) -> HRESULT {
    UNREACHABLE
}

extern "system" fn connectivity_hint(_: This, out: *mut ConnectivityHint) -> HRESULT {
    put(out, OFFLINE)
}

type PortCallback = extern "system" fn(*mut c_void, u16);
type HintCallback = extern "system" fn(*mut c_void, *const ConnectivityHint);

#[derive(Clone, Copy, PartialEq, Eq)]
enum Event {
    Port,
    Hint,
}

#[derive(Clone, Copy)]
enum Callback {
    Port(PortCallback),
    Hint(HintCallback),
}

enum Delivery {
    Pending(Callback, usize),
    Running(ThreadId),
    Finished,
}

struct Registration {
    event: Event,
    delivery: Delivery,
}

static REGISTRATIONS: Mutex<BTreeMap<u64, Registration>> = Mutex::new(BTreeMap::new());
static RETURNED: Condvar = Condvar::new();

fn registrations() -> MutexGuard<'static, BTreeMap<u64, Registration>> {
    REGISTRATIONS.lock().unwrap_or_else(PoisonError::into_inner)
}

extern "system" fn deliver_initial(task_context: *mut c_void, canceled: bool) {
    // The task carries only a token. Unregister can discard a pending
    // callback without waiting for its queue to dispatch the task.
    let id = task_context.addr() as u64;
    let (callback, context) = {
        let mut registrations = registrations();
        let Some(registration) = registrations.get_mut(&id) else {
            return;
        };
        let Delivery::Pending(callback, context) = registration.delivery else {
            return;
        };
        if canceled {
            registration.delivery = Delivery::Finished;
            return;
        }
        registration.delivery = Delivery::Running(thread::current().id());
        (callback, context as *mut c_void)
    };

    match callback {
        Callback::Port(callback) => callback(context, UDP_PORT),
        Callback::Hint(callback) => callback(context, &OFFLINE),
    }

    // A running registration stays in the map until the callback returns.
    registrations().get_mut(&id).unwrap().delivery = Delivery::Finished;
    RETURNED.notify_all();
}

fn register(
    queue: *mut c_void,
    event: Event,
    context: *mut c_void,
    callback: Option<Callback>,
    out: *mut u64,
) -> HRESULT {
    let Some(callback) = callback.filter(|_| !out.is_null()) else {
        return E_INVALIDARG;
    };
    let id = token();
    let Some(task_context) = usize::try_from(id).ok().and_then(NonZeroUsize::new) else {
        return E_ABORT;
    };
    registrations().insert(
        id,
        Registration {
            event,
            delivery: Delivery::Pending(callback, context as usize),
        },
    );
    // Immediate dispatch can call back into unregister before submit returns.
    put(out, id);
    let hr = xasync::XTaskQueueSubmitDelayedCallback(
        queue.cast(),
        xasync::COMPLETION,
        0,
        std::ptr::without_provenance_mut(task_context.get()),
        Some(deliver_initial),
    );
    if hr != S_OK {
        registrations().remove(&id);
        put(out, 0);
    }
    hr
}

fn unregister(event: Event, id: u64, wait: bool) -> bool {
    let mut registrations = registrations();
    loop {
        let Some(registration) = registrations.get(&id).filter(|r| r.event == event) else {
            return false;
        };
        match registration.delivery {
            Delivery::Running(runner) => {
                // Keep the token usable for a later wait. A callback cannot
                // wait for itself, even on an immediate queue.
                if !wait || runner == thread::current().id() {
                    return false;
                }
                registrations = RETURNED
                    .wait(registrations)
                    .unwrap_or_else(PoisonError::into_inner);
            }
            Delivery::Pending(..) | Delivery::Finished => {
                registrations.remove(&id);
                return true;
            }
        }
    }
}

extern "system" fn register_port(
    _: This,
    queue: *mut c_void,
    context: *mut c_void,
    callback: Option<PortCallback>,
    out: *mut u64,
) -> HRESULT {
    register(queue, Event::Port, context, callback.map(Callback::Port), out)
}

extern "system" fn unregister_port(_: This, token: u64, wait: bool) -> bool {
    unregister(Event::Port, token, wait)
}

extern "system" fn register_hint(
    _: This,
    queue: *mut c_void,
    context: *mut c_void,
    callback: Option<HintCallback>,
    out: *mut u64,
) -> HRESULT {
    register(queue, Event::Hint, context, callback.map(Callback::Hint), out)
}

extern "system" fn unregister_hint(_: This, token: u64, wait: bool) -> bool {
    unregister(Event::Hint, token, wait)
}

static VTABLE: Methods<18> = Methods([
    missing::<0> as *const (),
    add_ref as *const (),
    add_ref as *const (),
    udp_port as *const (),                  // 0x18
    udp_port_async as *const (),            // 0x20
    udp_port_result as *const (),           // 0x28
    register_port as *const (),             // 0x30
    unregister_port as *const (),           // 0x38
    security_info_async as *const (),       // 0x40
    security_info_size as *const (),        // 0x48
    security_info_result as *const (),      // 0x50
    security_info_utf16_async as *const (), // 0x58
    security_info_size as *const (),        // 0x60
    security_info_result as *const (),      // 0x68
    verify_certificate as *const (),        // 0x70
    connectivity_hint as *const (),         // 0x78
    register_hint as *const (),             // 0x80
    unregister_hint as *const (),           // 0x88
]);

#[cfg(test)]
mod tests {
    use std::ptr::{null, null_mut};

    use super::*;
    use crate::modules::xgameruntime::method;

    type Register<T> = extern "system" fn(
        This,
        *mut c_void,
        *mut c_void,
        Option<extern "system" fn(*mut c_void, T)>,
        *mut u64,
    ) -> HRESULT;
    type Unregister = extern "system" fn(This, u64, bool) -> bool;

    fn block() -> AsyncBlock {
        AsyncBlock {
            queue: null(),
            context: null_mut(),
            callback: None,
            internal: [0; 4],
        }
    }

    #[test]
    fn offline_but_initialized() {
        let this: This = &OBJECT;
        type Hint = extern "system" fn(This, *mut ConnectivityHint) -> HRESULT;
        let mut hint = OFFLINE;
        hint.initialized = false;
        assert_eq!(method::<Hint>(&OBJECT, 0x78)(this, &mut hint), S_OK);
        assert!(hint.initialized && hint.level == 1);
        assert_eq!(size_of::<ConnectivityHint>(), 16);

        type Begin = extern "system" fn(This, *mut AsyncBlock) -> HRESULT;
        type Port = extern "system" fn(This, *mut AsyncBlock, *mut u16) -> HRESULT;
        let mut b = block();
        assert_eq!(method::<Begin>(&OBJECT, 0x20)(this, &mut b), S_OK);
        assert_eq!(xasync::XAsyncGetStatus(&mut b, true), S_OK);
        let mut port = 0;
        assert_eq!(method::<Port>(&OBJECT, 0x28)(this, &mut b, &mut port), S_OK);
        assert_eq!(port, 3074);

        type Url = extern "system" fn(This, *const c_char, *mut AsyncBlock) -> HRESULT;
        type Size = extern "system" fn(This, *mut AsyncBlock, *mut usize) -> HRESULT;
        let mut b = block();
        let url = c"https://example.invalid/";
        assert_eq!(
            method::<Url>(&OBJECT, 0x40)(this, url.as_ptr(), &mut b),
            S_OK
        );
        assert_eq!(xasync::XAsyncGetStatus(&mut b, true), UNREACHABLE);
        let mut size = 0;
        assert_eq!(
            method::<Size>(&OBJECT, 0x48)(this, &mut b, &mut size),
            UNREACHABLE
        );

        type UrlUtf16 = extern "system" fn(This, *const u16, *mut AsyncBlock) -> HRESULT;
        let mut b = block();
        let url_utf16: Vec<u16> = "https://example.invalid/\0".encode_utf16().collect();
        assert_eq!(
            method::<UrlUtf16>(&OBJECT, 0x58)(this, url_utf16.as_ptr(), &mut b),
            S_OK
        );
        assert_eq!(xasync::XAsyncGetStatus(&mut b, true), UNREACHABLE);
        let mut size = 0;
        assert_eq!(
            method::<Size>(&OBJECT, 0x60)(this, &mut b, &mut size),
            UNREACHABLE
        );
    }

    fn queue(completion: u32) -> *const xasync::Queue {
        let mut queue = null();
        assert_eq!(xasync::XTaskQueueCreate(0, completion, &mut queue), S_OK);
        queue
    }

    #[derive(Default)]
    struct Observed {
        ports: Vec<u16>,
        hints: Vec<ConnectivityHint>,
    }

    extern "system" fn observe_port(context: *mut c_void, port: u16) {
        // SAFETY: the test keeps its observation buffer alive during dispatch.
        unsafe { &mut *context.cast::<Observed>() }.ports.push(port);
    }

    extern "system" fn observe_hint(context: *mut c_void, hint: *const ConnectivityHint) {
        // SAFETY: the buffer and the delivered hint are live during dispatch.
        unsafe { (*context.cast::<Observed>()).hints.push(*hint) };
    }

    fn register_observers(queue: *const xasync::Queue, observed: *mut Observed) -> (u64, u64) {
        let handle = queue.cast_mut().cast();
        let context = observed.cast();
        let mut port_token = 0;
        let mut hint_token = 0;
        assert_eq!(
            method::<Register<u16>>(&OBJECT, 0x30)(
                &OBJECT,
                handle,
                context,
                Some(observe_port),
                &mut port_token,
            ),
            S_OK
        );
        assert_eq!(
            method::<Register<*const ConnectivityHint>>(&OBJECT, 0x80)(
                &OBJECT,
                handle,
                context,
                Some(observe_hint),
                &mut hint_token,
            ),
            S_OK
        );
        (port_token, hint_token)
    }

    #[test]
    fn initial_callbacks_use_the_completion_port() {
        let queue = queue(0);
        let mut observed = Observed::default();
        let (port_token, hint_token) = register_observers(queue, &raw mut observed);
        assert_ne!(port_token, 0);
        assert_ne!(hint_token, 0);
        assert_ne!(port_token, hint_token);
        assert!(!xasync::XTaskQueueDispatch(queue, 0, 0));
        assert!(observed.ports.is_empty() && observed.hints.is_empty());
        assert!(xasync::XTaskQueueDispatch(queue, xasync::COMPLETION, 0));
        assert!(xasync::XTaskQueueDispatch(queue, xasync::COMPLETION, 0));
        assert!(!xasync::XTaskQueueDispatch(queue, xasync::COMPLETION, 0));
        assert_eq!(observed.ports, [UDP_PORT]);
        assert_eq!(observed.hints, [OFFLINE]);
        assert!(method::<Unregister>(&OBJECT, 0x38)(&OBJECT, port_token, false));
        assert!(method::<Unregister>(&OBJECT, 0x88)(&OBJECT, hint_token, true));
        xasync::XTaskQueueCloseHandle(queue);
    }

    #[test]
    fn unregister_cancels_only_the_matching_token() {
        let queue = queue(0);
        let mut observed = Observed::default();
        let (port_token, hint_token) = register_observers(queue, &raw mut observed);
        let unregister_port = method::<Unregister>(&OBJECT, 0x38);
        let unregister_hint = method::<Unregister>(&OBJECT, 0x88);
        assert!(!unregister_hint(&OBJECT, port_token, true));
        assert!(!unregister_port(&OBJECT, hint_token, true));
        assert!(unregister_port(&OBJECT, port_token, false));
        assert!(unregister_hint(&OBJECT, hint_token, true));
        assert!(!unregister_port(&OBJECT, port_token, true));
        assert!(!unregister_hint(&OBJECT, hint_token, true));
        while xasync::XTaskQueueDispatch(queue, xasync::COMPLETION, 0) {}
        assert!(observed.ports.is_empty() && observed.hints.is_empty());
        xasync::XTaskQueueCloseHandle(queue);
    }

    #[test]
    fn registration_rejects_missing_callback_or_token() {
        fn check<T>(slot: usize, callback: extern "system" fn(*mut c_void, T)) {
            let queue = queue(0);
            let handle = queue.cast_mut().cast();
            let register = method::<Register<T>>(&OBJECT, slot);
            let mut observed = Observed::default();
            let context = (&raw mut observed).cast();
            let mut token = 0;
            assert_eq!(
                register(&OBJECT, handle, context, None, &mut token),
                E_INVALIDARG
            );
            assert_eq!(token, 0);
            assert_eq!(
                register(&OBJECT, handle, context, Some(callback), null_mut()),
                E_INVALIDARG
            );
            assert!(!xasync::XTaskQueueDispatch(queue, xasync::COMPLETION, 0));
            xasync::XTaskQueueCloseHandle(queue);
        }

        check(0x30, observe_port);
        check(0x80, observe_hint);
    }

    #[test]
    fn immediate_callback_can_try_to_unregister_itself() {
        #[derive(Default)]
        struct Probe {
            token: u64,
            result: Option<(u64, bool)>,
        }

        extern "system" fn check_self_wait(context: *mut c_void, _: u16) {
            // SAFETY: the probe is alive on the registering thread.
            let probe = unsafe { &mut *context.cast::<Probe>() };
            let unregister = method::<Unregister>(&OBJECT, 0x38);
            probe.result = Some((probe.token, unregister(&OBJECT, probe.token, true)));
        }

        let queue = queue(3);
        let mut probe = Probe::default();
        assert_eq!(
            method::<Register<u16>>(&OBJECT, 0x30)(
                &OBJECT,
                queue.cast_mut().cast(),
                (&raw mut probe).cast(),
                Some(check_self_wait),
                &raw mut probe.token,
            ),
            S_OK
        );
        assert_ne!(probe.token, 0);
        assert_eq!(probe.result, Some((probe.token, false)));
        assert!(method::<Unregister>(&OBJECT, 0x38)(&OBJECT, probe.token, true));
        xasync::XTaskQueueCloseHandle(queue);
    }
}
