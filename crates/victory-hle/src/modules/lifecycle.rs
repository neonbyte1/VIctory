//! The title's life cycle. A console title is never told it is closing:
//! quitting it or launching a new game first constrains it (its
//! constrained-change routines run), then suspends it (its suspend routines
//! run, and it has a second to save) and then ends the process. Closing the
//! title's window does the same here; a new launch then starts `victory`
//! again, which waits for this process to be gone.
#![allow(non_snake_case)]
#![allow(clippy::not_unsafe_ptr_arg_deref)]
#![allow(clippy::too_many_arguments)]

use std::ffi::c_void;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Mutex, MutexGuard};
use std::time::Duration;

use windows_sys::Win32::Foundation::{HINSTANCE, HWND, LPARAM, LRESULT, RECT, WPARAM};
use windows_sys::Win32::System::Threading::{GetCurrentProcess, TerminateProcess};
use windows_sys::Win32::UI::WindowsAndMessaging::{self as wm, HMENU};
use windows_sys::core::{PCSTR, PCWSTR, w};

use crate::{HleModule, hle_export};

static OUTPUT_RESOLUTION: std::sync::OnceLock<victory_core::Resolution> =
    std::sync::OnceLock::new();

/// Set before creating any title window.
pub fn set_output_resolution(
    resolution: victory_core::Resolution,
) -> Result<(), victory_core::Resolution> {
    OUTPUT_RESOLUTION.set(resolution)
}

/// Checks the current display area and configured window limit before a
/// title is launched with a reduced output profile.
pub fn preflight_output_resolution(
    resolution: victory_core::Resolution,
) -> Result<(), &'static str> {
    let area = work_area().ok_or("could not determine the display work area or screen size")?;
    let target = (resolution.width as i32, resolution.height as i32);
    let outer = outer_size(0, wm::WS_OVERLAPPEDWINDOW, target.0, target.1);
    let chrome = (outer.0 - target.0, outer.1 - target.1);
    resolution_window_preflight(&area, target, chrome, window_limit())
        .then_some(())
        .ok_or("the resolution window cannot fit the display work area and VICTORY_WINDOW_MAX")
}

fn work_area_with(
    query: impl FnOnce(&mut RECT) -> bool,
    screen_size: impl FnOnce() -> Option<(i32, i32)>,
) -> Option<RECT> {
    let mut area = RECT::default();
    if query(&mut area) && area.right > area.left && area.bottom > area.top {
        return Some(area);
    }
    let (width, height) = screen_size()?;
    (width > 0 && height > 0).then_some(RECT {
        left: 0,
        top: 0,
        right: width,
        bottom: height,
    })
}

fn work_area() -> Option<RECT> {
    work_area_with(
        |area| {
            // SAFETY: `area` receives the primary monitor's work area.
            unsafe {
                wm::SystemParametersInfoW(wm::SPI_GETWORKAREA, 0, (&raw mut *area).cast(), 0) != 0
            }
        },
        || {
            // SAFETY: plain system metrics queries with no pointer arguments.
            let width = unsafe { wm::GetSystemMetrics(wm::SM_CXSCREEN) };
            // SAFETY: plain system metrics queries with no pointer arguments.
            let height = unsafe { wm::GetSystemMetrics(wm::SM_CYSCREEN) };
            Some((width, height))
        },
    )
}

/// A suspend routine (`quiesced`) or a constrained-change routine
/// (`constrained`): both take a flag and the title's context.
type Routine = extern "system" fn(flag: u8, context: *mut c_void);

/// Registration id, routine and context.
type Registration = (usize, Routine, usize);

/// Each suspend routine.
static ROUTINES: Mutex<Vec<Registration>> = Mutex::new(Vec::new());

/// Each constrained-change routine.
static CONSTRAINED: Mutex<Vec<Registration>> = Mutex::new(Vec::new());

fn routines() -> MutexGuard<'static, Vec<Registration>> {
    ROUTINES.lock().unwrap_or_else(|e| e.into_inner())
}

fn constrained() -> MutexGuard<'static, Vec<Registration>> {
    CONSTRAINED.lock().unwrap_or_else(|e| e.into_inner())
}

/// How long the console lets suspend routines run before it ends the title.
const SUSPEND_DEADLINE: Duration = Duration::from_secs(1);

const ERROR_INVALID_PARAMETER: u32 = 87;

/// Adds `routine` to `list` and writes its registration id.
fn register(
    mut list: MutexGuard<'static, Vec<Registration>>,
    routine: Option<Routine>,
    context: *mut c_void,
    registration: *mut usize,
) -> u32 {
    let Some(routine) = routine else {
        return ERROR_INVALID_PARAMETER;
    };
    if registration.is_null() {
        return ERROR_INVALID_PARAMETER;
    }
    static NEXT: AtomicUsize = AtomicUsize::new(1);
    let id = NEXT.fetch_add(1, Ordering::Relaxed);
    list.push((id, routine, context as usize));
    // SAFETY: checked non-null; the title passes room for its registration.
    unsafe { registration.write(id) };
    0
}

pub extern "system" fn RegisterAppStateChangeNotification(
    routine: Option<Routine>,
    context: *mut c_void,
    registration: *mut usize,
) -> u32 {
    register(routines(), routine, context, registration)
}

pub extern "system" fn UnregisterAppStateChangeNotification(registration: usize) {
    routines().retain(|&(id, ..)| id != registration);
}

/// The host's own version never calls the routine (a desktop app is never
/// constrained), so a title waiting for it would never suspend.
pub extern "system" fn RegisterAppConstrainedChangeNotification(
    routine: Option<Routine>,
    context: *mut c_void,
    registration: *mut usize,
) -> u32 {
    register(constrained(), routine, context, registration)
}

pub extern "system" fn UnregisterAppConstrainedChangeNotification(registration: usize) {
    constrained().retain(|&(id, ..)| id != registration);
}

/// The window procedure a title window had before ours.
const TITLE_PROC: PCWSTR = w!("victory.wndproc");

/// The stand-in window shown for a title window (a property of the title's).
const STAND_IN: PCWSTR = w!("victory.standin");

// The window the player sees is never the title's. A title makes its window
// on one of its own threads and reads its messages only when that thread
// gets round to it, which in a slow frame or a load is seconds apart; the
// system then shows the busy cursor over the window, and a click turns it
// into a "Not responding" ghost. So a top-level title window is made
// message-only (never shown), and a stand-in with the title's text, style
// and (fitted) size is shown in its place, owned by a thread of ours that
// does nothing but read its messages. The title loses nothing: its input
// comes from GameInput (polled, not window messages), and frames are
// presented to the visible window of the process, the stand-in.

pub extern "system" fn CreateWindowExW(
    ex_style: u32,
    class: PCWSTR,
    name: PCWSTR,
    style: u32,
    x: i32,
    y: i32,
    width: i32,
    height: i32,
    parent: HWND,
    menu: HMENU,
    instance: HINSTANCE,
    param: *const c_void,
) -> HWND {
    let top = parent.is_null();
    // SAFETY: the title's own arguments, forwarded (a top-level window
    // made message-only).
    let hwnd = unsafe {
        wm::CreateWindowExW(
            ex_style,
            class,
            name,
            style,
            x,
            y,
            width,
            height,
            if top { wm::HWND_MESSAGE } else { parent },
            menu,
            instance,
            param,
        )
    };
    hook(hwnd);
    if top {
        stand_in(hwnd, ex_style, style, x, y, width, height);
    }
    hwnd
}

/// See [`CreateWindowExW`].
pub extern "system" fn CreateWindowExA(
    ex_style: u32,
    class: PCSTR,
    name: PCSTR,
    style: u32,
    x: i32,
    y: i32,
    width: i32,
    height: i32,
    parent: HWND,
    menu: HMENU,
    instance: HINSTANCE,
    param: *const c_void,
) -> HWND {
    let top = parent.is_null();
    // SAFETY: the title's own arguments, forwarded (a top-level window
    // made message-only).
    let hwnd = unsafe {
        wm::CreateWindowExA(
            ex_style,
            class,
            name,
            style,
            x,
            y,
            width,
            height,
            if top { wm::HWND_MESSAGE } else { parent },
            menu,
            instance,
            param,
        )
    };
    hook(hwnd);
    if top {
        stand_in(hwnd, ex_style, style, x, y, width, height);
    }
    hwnd
}

/// A console title sizes its window to its render target (Grand Theft
/// Auto VI: 3840x2160 on a Series X), which can exceed the host's screen.
/// A top-level window larger than the work area is shrunk to fit, keeping
/// its proportions, and centred; what it shows is scaled to it (the swap
/// chain stretches). `VICTORY_WINDOW_MAX=WIDTHxHEIGHT` applies a smaller
/// limit without changing the title's render targets.
fn fit(
    parent: HWND,
    ex_style: u32,
    style: u32,
    x: i32,
    y: i32,
    width: i32,
    height: i32,
) -> Option<(i32, i32, i32, i32)> {
    let area = work_area();
    let known = area.is_some();
    let area = area.unwrap_or_default();
    if parent.is_null()
        && let Some(resolution) = OUTPUT_RESOLUTION.get().copied()
        && width >= resolution.width as i32
        && height >= resolution.height as i32
    {
        let target = (resolution.width as i32, resolution.height as i32);
        let requested_outer = outer_size(ex_style, style, target.0, target.1);
        let chrome = (requested_outer.0 - target.0, requested_outer.1 - target.1);
        let limit = window_limit();
        let Some((client, outer)) = fit_resolution_into(&area, target, chrome, limit) else {
            tracing::error!(
                target_client = ?target,
                ?chrome,
                ?limit,
                work_area = ?(area.right - area.left, area.bottom - area.top),
                "resolution window cannot fit its title bar and configured bounds"
            );
            return None;
        };
        let (left, top) = if known {
            (
                area.left + (area.right - area.left - outer.0) / 2,
                area.top + (area.bottom - area.top - outer.1) / 2,
            )
        } else {
            (x, y)
        };
        tracing::info!(
            asked = ?(width, height),
            client = ?client,
            outer = ?outer,
            ?chrome,
            "resolution window fitted to the work area and configured limit"
        );
        return Some((left, top, outer.0, outer.1));
    }
    let limit = window_limit();
    let fitted = fit_with_limit(&area, width, height, limit);
    if !parent.is_null() || !known || fitted == (width, height) {
        return Some((x, y, width, height));
    }
    tracing::info!(width, height, to = ?fitted, ?limit, "title window fitted to the screen");
    let (w, h) = fitted;
    let left = area.left + (area.right - area.left - w) / 2;
    let top = area.top + (area.bottom - area.top - h) / 2;
    Some((left, top, w, h))
}

fn outer_size(ex_style: u32, style: u32, width: i32, height: i32) -> (i32, i32) {
    let mut rect = RECT {
        left: 0,
        top: 0,
        right: width,
        bottom: height,
    };
    // SAFETY: `rect` is valid writable storage; the style flags describe the
    // title window that will be created.
    if unsafe { wm::AdjustWindowRectEx(&mut rect, style, 0, ex_style) } == 0 {
        return (width, height);
    }
    (rect.right - rect.left, rect.bottom - rect.top)
}

fn parse_window_limit(value: &str) -> Option<(i32, i32)> {
    let (width, height) = value.trim().split_once(['x', 'X'])?;
    let (width, height) = (width.trim().parse().ok()?, height.trim().parse().ok()?);
    (width > 0 && height > 0).then_some((width, height))
}

fn window_limit() -> Option<(i32, i32)> {
    static LIMIT: std::sync::OnceLock<Option<(i32, i32)>> = std::sync::OnceLock::new();
    *LIMIT.get_or_init(|| {
        let value = std::env::var_os("VICTORY_WINDOW_MAX")?;
        let parsed = value.to_str().and_then(parse_window_limit);
        if parsed.is_none() {
            tracing::warn!(
                ?value,
                "ignoring invalid VICTORY_WINDOW_MAX; expected positive WIDTHxHEIGHT"
            );
        }
        parsed
    })
}

fn fit_with_limit(area: &RECT, width: i32, height: i32, limit: Option<(i32, i32)>) -> (i32, i32) {
    let Some((max_width, max_height)) = limit else {
        return fit_into(area, width, height);
    };
    let bounded = RECT {
        left: 0,
        top: 0,
        right: (area.right - area.left).min(max_width),
        bottom: (area.bottom - area.top).min(max_height),
    };
    fit_into(&bounded, width, height)
}

fn fit_resolution_into(
    area: &RECT,
    client: (i32, i32),
    chrome: (i32, i32),
    limit: Option<(i32, i32)>,
) -> Option<((i32, i32), (i32, i32))> {
    let area_width = (area.right - area.left).max(0);
    let area_height = (area.bottom - area.top).max(0);
    let max_outer = limit.map_or((area_width, area_height), |(w, h)| {
        (area_width.min(w), area_height.min(h))
    });
    if chrome.0 < 0
        || chrome.1 < 0
        || max_outer.0 <= chrome.0
        || max_outer.1 <= chrome.1
        || client.0 <= 0
        || client.1 <= 0
    {
        return None;
    }
    let content_area = RECT {
        left: 0,
        top: 0,
        right: (max_outer.0 - chrome.0).max(0),
        bottom: (max_outer.1 - chrome.1).max(0),
    };
    let fitted_client = fit_into(&content_area, client.0, client.1);
    if fitted_client.0 <= 0 || fitted_client.1 <= 0 {
        return None;
    }
    let fitted_outer = (fitted_client.0 + chrome.0, fitted_client.1 + chrome.1);
    (fitted_outer.0 <= max_outer.0 && fitted_outer.1 <= max_outer.1)
        .then_some((fitted_client, fitted_outer))
}

fn resolution_window_preflight(
    area: &RECT,
    client: (i32, i32),
    chrome: (i32, i32),
    limit: Option<(i32, i32)>,
) -> bool {
    fit_resolution_into(area, client, chrome, limit).is_some()
}

/// `width` x `height` scaled down to fit `area`, or unchanged if it fits
/// (or is `CW_USEDEFAULT`).
fn fit_into(area: &RECT, width: i32, height: i32) -> (i32, i32) {
    let (aw, ah) = (area.right - area.left, area.bottom - area.top);
    if width <= 0 || height <= 0 || aw <= 0 || ah <= 0 || (width <= aw && height <= ah) {
        return (width, height);
    }
    let scale = f64::min(
        f64::from(aw) / f64::from(width),
        f64::from(ah) / f64::from(height),
    );
    (
        (f64::from(width) * scale) as i32,
        (f64::from(height) * scale) as i32,
    )
}

/// Shows a stand-in for the top-level title window `title` (see
/// [`STAND_IN`]); `ShowWindow` on `title` then shows or hides the stand-in.
fn stand_in(title: HWND, ex_style: u32, style: u32, x: i32, y: i32, width: i32, height: i32) {
    if title.is_null() {
        return;
    }
    let Some((x, y, width, height)) =
        fit(std::ptr::null_mut(), ex_style, style, x, y, width, height)
    else {
        tracing::error!("resolution window could not be created within its configured bounds");
        // No visible stand-in would leave the title running without output.
        // SAFETY: `GetCurrentProcess` returns a valid pseudo-handle.
        unsafe { TerminateProcess(GetCurrentProcess(), 1) };
        std::process::abort();
    };
    let mut text = vec![0u16; 512];
    // SAFETY: a window this thread just created; the buffer holds 512
    // characters. Its icon, if its class has one.
    let (icon, text) = unsafe {
        let n = wm::GetWindowTextW(title, text.as_mut_ptr(), text.len() as i32);
        text.truncate(n.max(0) as usize);
        text.push(0);
        (wm::GetClassLongPtrW(title, wm::GCLP_HICON), text)
    };
    let (sender, receiver) = std::sync::mpsc::channel();
    let spawned = std::thread::Builder::new()
        .name("victory-window".into())
        .spawn(move || {
            let window = make_stand_in(&text, ex_style, style, (x, y, width, height), icon);
            let _ = sender.send(window as usize);
            if !window.is_null() {
                pump();
            }
        });
    let window = match spawned {
        Ok(_) => receiver.recv().unwrap_or(0) as HWND,
        Err(_) => std::ptr::null_mut(),
    };
    if window.is_null() {
        tracing::warn!("no stand-in window; the title's own window is not shown");
        return;
    }
    // SAFETY: the title's window, on the thread that made it.
    unsafe { wm::SetPropW(title, STAND_IN, window) };
}

/// The stand-in window for `stand_in`, on the calling thread.
fn make_stand_in(
    text: &[u16],
    ex_style: u32,
    style: u32,
    (x, y, width, height): (i32, i32, i32, i32),
    icon: usize,
) -> HWND {
    use windows_sys::Win32::System::LibraryLoader::GetModuleHandleW;
    const CLASS: PCWSTR = w!("VIctory.title");
    // SAFETY: registers our class (again is harmless) and creates a window
    // of it; `text` is NUL-terminated.
    unsafe {
        let instance = GetModuleHandleW(std::ptr::null());
        let class = wm::WNDCLASSEXW {
            cbSize: size_of::<wm::WNDCLASSEXW>() as u32,
            lpfnWndProc: Some(stand_in_proc),
            hInstance: instance,
            hCursor: wm::LoadCursorW(std::ptr::null_mut(), wm::IDC_ARROW),
            lpszClassName: CLASS,
            ..std::mem::zeroed()
        };
        wm::RegisterClassExW(&class);
        let window = wm::CreateWindowExW(
            ex_style,
            CLASS,
            text.as_ptr(),
            style,
            x,
            y,
            width,
            height,
            std::ptr::null_mut(),
            std::ptr::null_mut(),
            instance,
            std::ptr::null(),
        );
        if !window.is_null() && icon != 0 {
            for kind in [wm::ICON_BIG, wm::ICON_SMALL] {
                wm::SendMessageW(window, wm::WM_SETICON, kind as usize, icon as isize);
            }
        }
        window
    }
}

/// Destroys a stand-in (posted when the title's window goes).
const DESTROY_STAND_IN: u32 = wm::WM_APP + 0x7649;

extern "system" fn stand_in_proc(
    hwnd: HWND,
    message: u32,
    wparam: WPARAM,
    lparam: LPARAM,
) -> LRESULT {
    match message {
        // A user close (close button, Alt+F4, window or taskbar menu).
        wm::WM_CLOSE => {
            end_title("window closed", Box::new(|| {}));
            return 0;
        }
        // No window menu from the keyboard (see `window_proc`): here it
        // would stop only this thread, but it takes keys the title wants.
        wm::WM_SYSCOMMAND if wparam & 0xFFF0 == wm::SC_KEYMENU as usize => return 0,
        DESTROY_STAND_IN => {
            // SAFETY: our window, on its thread.
            unsafe { wm::DestroyWindow(hwnd) };
            return 0;
        }
        // SAFETY: ends `pump`, the loop of this window's thread.
        wm::WM_DESTROY => unsafe { wm::PostQuitMessage(0) },
        _ => {}
    }
    // SAFETY: the default handling for our window.
    unsafe { wm::DefWindowProcW(hwnd, message, wparam, lparam) }
}

/// The message loop of a stand-in's thread, until its window is destroyed.
fn pump() {
    // SAFETY: the standard message loop of this thread's window.
    unsafe {
        let mut message: wm::MSG = std::mem::zeroed();
        while wm::GetMessageW(&mut message, std::ptr::null_mut(), 0, 0) > 0 {
            wm::TranslateMessage(&message);
            wm::DispatchMessageW(&message);
        }
    }
}

/// The stand-in shown for `hwnd`, or null.
fn stand_in_of(hwnd: HWND) -> HWND {
    // SAFETY: a plain query; another window gives null.
    unsafe { wm::GetPropW(hwnd, STAND_IN) }
}

/// A title shows its window: shows the stand-in instead. Returns whether it
/// was visible before, as `ShowWindow` does.
pub extern "system" fn ShowWindow(hwnd: HWND, command: i32) -> i32 {
    let window = stand_in_of(hwnd);
    let target = if window.is_null() { hwnd } else { window };
    // SAFETY: the title's window, or our stand-in for it, whose thread
    // always reads its messages, so this never waits long.
    unsafe { wm::ShowWindow(target, command) }
}

/// Puts our procedure in front of the title's.
fn hook(hwnd: HWND) {
    if hwnd.is_null() {
        return;
    }
    // SAFETY: a window this thread just created. The title's procedure is
    // stored before ours can run.
    unsafe {
        let title = wm::GetWindowLongPtrW(hwnd, wm::GWLP_WNDPROC);
        wm::SetPropW(hwnd, TITLE_PROC, title as *mut c_void);
        wm::SetWindowLongPtrW(hwnd, wm::GWLP_WNDPROC, window_proc as *const () as isize);
    }
}

extern "system" fn window_proc(
    hwnd: HWND,
    message: u32,
    wparam: WPARAM,
    lparam: LPARAM,
) -> LRESULT {
    // A user close (close button, Alt+F4, window or taskbar menu); a
    // console title never sees one, nor a WM_CLOSE from it.
    if message == wm::WM_SYSCOMMAND && wparam & 0xFFF0 == wm::SC_CLOSE as usize {
        end_title("window closed", Box::new(|| {}));
        return 0;
    }
    // Alt or F10 puts a window into menu mode: the system runs a modal menu
    // loop on the thread that owns the window, which is the title's. The
    // title then stops (no frames, no input) until a key ends the loop, and
    // nothing shows it, since the window has no menu bar; Grand Theft Auto
    // VI aborts itself about half a minute later. A console has no window
    // menu. The mouse still opens it from the title bar.
    if message == wm::WM_SYSCOMMAND && wparam & 0xFFF0 == wm::SC_KEYMENU as usize {
        return 0;
    }
    // SAFETY: the procedure `hook` stored, called the way the system would.
    unsafe {
        let title = wm::GetPropW(hwnd, TITLE_PROC);
        let window = stand_in_of(hwnd);
        if message == wm::WM_NCDESTROY {
            wm::RemovePropW(hwnd, TITLE_PROC);
            if !window.is_null() {
                wm::RemovePropW(hwnd, STAND_IN);
                wm::PostMessageW(window, DESTROY_STAND_IN, 0, 0);
            }
        }
        let title = std::mem::transmute::<*mut c_void, wm::WNDPROC>(title);
        let result = wm::CallWindowProcW(title, hwnd, message, wparam, lparam);
        // The stand-in shows the title's text. Its thread never waits for
        // the title's, so this cannot deadlock.
        if message == wm::WM_SETTEXT && !window.is_null() {
            wm::SendMessageTimeoutW(
                window,
                wm::WM_SETTEXT,
                0,
                lparam,
                wm::SMTO_ABORTIFHUNG,
                1000,
                std::ptr::null_mut(),
            );
        }
        result
    }
}

/// Suspends the title as the console does when it ends it, runs `then`
/// (once), then ends the process. Returns at once; later calls do nothing.
pub(crate) fn end_title(why: &'static str, then: Box<dyn FnOnce() + Send>) {
    static ENDING: AtomicBool = AtomicBool::new(false);
    if ENDING.swap(true, Ordering::SeqCst) {
        return;
    }
    let constrain = constrained().clone();
    let suspend = routines().clone();
    tracing::info!(
        constrained = constrain.len(),
        routines = suspend.len(),
        "{why}: suspending the title"
    );
    // Whoever comes first (the routines or the deadline) runs `then`; the
    // other waits for it, so the process never ends halfway through.
    let then = std::sync::Arc::new(Mutex::new(Some(then)));
    let finish = move |how: &str| {
        let mut then = then.lock().unwrap_or_else(|e| e.into_inner());
        tracing::info!("{how}; ending the title");
        if let Some(then) = then.take() {
            then();
        }
        end();
    };
    let late = finish.clone();
    // Not on the calling thread: titles hand the suspend to their window
    // thread and wait for it inside the routine.
    std::thread::spawn(move || {
        constrain_then_suspend(&constrain, &suspend);
        finish("title suspended");
    });
    std::thread::spawn(move || {
        std::thread::sleep(SUSPEND_DEADLINE);
        late("title did not finish suspending in time");
    });
}

/// Runs the constrained-change routines (constrained), then the suspend
/// routines (quiesced), as the console does when it suspends a title.
/// Grand Theft Auto VI's suspend routine waits for a state that only its
/// constrained routine brings about; called alone it never returns.
fn constrain_then_suspend(constrain: &[Registration], suspend: &[Registration]) {
    for &(_, routine, context) in constrain.iter().chain(suspend) {
        routine(1, context as *mut c_void);
    }
}

/// Set in the environment of a relaunched `victory`: an inherited handle
/// of the process it replaces.
const RELAUNCHED_FROM: &str = "VICTORY_RELAUNCHED_FROM";

/// Starts `victory` again with `args` (after the title has suspended); the
/// new process waits until this one has ended.
pub(crate) fn relaunch(args: Vec<std::ffi::OsString>) {
    use windows_sys::Win32::Foundation::{DUPLICATE_SAME_ACCESS, DuplicateHandle, HANDLE};
    let mut me: HANDLE = std::ptr::null_mut();
    // SAFETY: an inheritable duplicate of our own process handle.
    let duplicated = unsafe {
        DuplicateHandle(
            GetCurrentProcess(),
            GetCurrentProcess(),
            GetCurrentProcess(),
            &mut me,
            0,
            1,
            DUPLICATE_SAME_ACCESS,
        )
    } != 0;
    let spawned = std::env::current_exe().and_then(|exe| {
        let mut command = std::process::Command::new(exe);
        command.args(&args);
        if duplicated {
            command.env(RELAUNCHED_FROM, (me as usize).to_string());
        }
        command.spawn()
    });
    match spawned {
        Ok(child) => tracing::info!(pid = child.id(), ?args, "relaunched"),
        Err(e) => tracing::warn!(%e, "relaunch failed; the title just ends"),
    }
}

/// In a relaunched `victory`: waits (30 s at most) for the process it
/// replaces to be gone, so the two never share the title's files.
pub fn wait_for_previous() {
    use windows_sys::Win32::Foundation::{CloseHandle, HANDLE};
    use windows_sys::Win32::System::Threading::WaitForSingleObject;
    let Some(handle) = std::env::var(RELAUNCHED_FROM)
        .ok()
        .and_then(|h| h.parse::<usize>().ok())
    else {
        return;
    };
    // SAFETY: removing our own variable before any other thread reads it.
    unsafe { std::env::remove_var(RELAUNCHED_FROM) };
    let handle = handle as HANDLE;
    // SAFETY: the inherited process handle the previous process gave us.
    unsafe {
        WaitForSingleObject(handle, 30_000);
        CloseHandle(handle);
    }
}

fn end() {
    // Compiled host pipelines kept for the next run.
    #[cfg(windows)]
    victory_gpu::pipeline::save_pipeline_cache();
    // SAFETY: ends this process, as the console ends a suspended title.
    unsafe { TerminateProcess(GetCurrentProcess(), 0) };
}

pub(crate) fn module() -> HleModule {
    HleModule {
        name: "lifecycle",
        dlls: &[
            "api-ms-win-core-psm-appnotify-*",
            "ext-ms-win-rtcore-ntuser-window-*",
        ],
        exports: vec![
            hle_export!(RegisterAppStateChangeNotification),
            hle_export!(UnregisterAppStateChangeNotification),
            hle_export!(RegisterAppConstrainedChangeNotification),
            hle_export!(UnregisterAppConstrainedChangeNotification),
            hle_export!(CreateWindowExW),
            hle_export!(CreateWindowExA),
            hle_export!(ShowWindow),
        ],
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    extern "system" fn noop(_: u8, _: *mut c_void) {}

    #[test]
    fn registrations_come_and_go() {
        let (mut a, mut b) = (0, 0);
        assert_eq!(
            RegisterAppStateChangeNotification(Some(noop), std::ptr::null_mut(), &mut a),
            0
        );
        assert_eq!(
            RegisterAppStateChangeNotification(Some(noop), std::ptr::null_mut(), &mut b),
            0
        );
        UnregisterAppStateChangeNotification(a);
        let left: Vec<_> = routines().iter().map(|r| r.0).collect();
        assert_eq!(left, [b]);
    }

    #[test]
    fn constrained_registrations_come_and_go() {
        let (mut a, mut b) = (0, 0);
        assert_eq!(
            RegisterAppConstrainedChangeNotification(Some(noop), std::ptr::null_mut(), &mut a),
            0
        );
        assert_eq!(
            RegisterAppConstrainedChangeNotification(Some(noop), std::ptr::null_mut(), &mut b),
            0
        );
        assert_eq!(
            RegisterAppConstrainedChangeNotification(None, std::ptr::null_mut(), &mut a),
            ERROR_INVALID_PARAMETER
        );
        UnregisterAppConstrainedChangeNotification(a);
        let left: Vec<_> = constrained().iter().map(|r| r.0).collect();
        assert_eq!(left, [b]);
    }

    /// What `recording_routine` saw: (context, flag) per call, in order.
    static CALLS: Mutex<Vec<(usize, u8)>> = Mutex::new(Vec::new());

    extern "system" fn recording_routine(flag: u8, context: *mut c_void) {
        CALLS.lock().unwrap().push((context as usize, flag));
    }

    #[test]
    fn constrained_routines_run_before_suspend_routines() {
        let r: Routine = recording_routine;
        constrain_then_suspend(&[(1, r, 10), (2, r, 11)], &[(3, r, 20), (4, r, 21)]);
        assert_eq!(*CALLS.lock().unwrap(), [(10, 1), (11, 1), (20, 1), (21, 1)]);
    }

    #[test]
    fn oversized_windows_shrink_to_the_screen() {
        let area = RECT {
            left: 0,
            top: 0,
            right: 1920,
            bottom: 1040,
        };
        assert_eq!(fit_into(&area, 3840, 2160), (1848, 1040));
        assert_eq!(fit_into(&area, 1280, 720), (1280, 720));
        assert_eq!(
            fit_into(&area, wm::CW_USEDEFAULT, 0),
            (wm::CW_USEDEFAULT, 0)
        );
    }

    #[test]
    fn configured_window_limit_fits_4k_into_1080p() {
        let area = RECT {
            left: 100,
            top: 50,
            right: 2660,
            bottom: 1490,
        };
        assert_eq!(
            fit_with_limit(&area, 3840, 2160, Some((1920, 1080))),
            (1920, 1080)
        );
        assert_eq!(
            fit_with_limit(&area, 640, 480, Some((1920, 1080))),
            (640, 480)
        );
        assert_eq!(fit_with_limit(&area, 3840, 2160, None), (2560, 1440));
    }

    #[test]
    fn configured_window_limit_still_respects_the_work_area() {
        let area = RECT {
            left: 0,
            top: 0,
            right: 1920,
            bottom: 1040,
        };
        assert_eq!(
            fit_with_limit(&area, 3840, 2160, Some((2560, 1440))),
            (1848, 1040)
        );
        assert_eq!(
            fit_with_limit(&area, wm::CW_USEDEFAULT, 0, Some((1920, 1080))),
            (wm::CW_USEDEFAULT, 0)
        );
    }

    #[test]
    fn configured_window_limit_preserves_non_widescreen_proportions() {
        let area = RECT {
            left: 0,
            top: 0,
            right: 2560,
            bottom: 1440,
        };
        assert_eq!(
            fit_with_limit(&area, 1600, 1200, Some((1920, 1080))),
            (1440, 1080)
        );
    }

    #[test]
    fn resolution_window_fits_1366_by_768_with_title_bar_and_aspect_ratio() {
        let area = RECT {
            left: 0,
            top: 0,
            right: 1366,
            bottom: 768,
        };
        let (client, outer) = fit_resolution_into(&area, (1920, 1080), (16, 39), None).unwrap();
        assert_eq!(client, (1296, 729));
        assert_eq!(outer, (1312, 768));
        assert!(outer.0 <= area.right - area.left);
        assert!(outer.1 <= area.bottom - area.top);
        assert!((client.0 * 9 - client.1 * 16).abs() <= 16);
    }

    #[test]
    fn failed_work_area_query_uses_the_same_screen_fallback_for_preflight_and_fit() {
        let area = work_area_with(|_| false, || Some((1366, 768))).unwrap();
        assert_eq!(
            (area.left, area.top, area.right, area.bottom),
            (0, 0, 1366, 768)
        );

        let target = (1920, 1080);
        let chrome = (16, 39);
        let fitted = fit_resolution_into(&area, target, chrome, None).unwrap();
        assert!(resolution_window_preflight(&area, target, chrome, None));
        assert_eq!(fitted, ((1296, 729), (1312, 768)));
    }

    #[test]
    fn resolution_window_max_includes_title_bar_and_keeps_client_aspect_ratio() {
        let area = RECT {
            left: 0,
            top: 0,
            right: 2560,
            bottom: 1440,
        };
        let limit = (1280, 720);
        let chrome = (16, 39);
        let (client, outer) =
            fit_resolution_into(&area, (1920, 1080), chrome, Some(limit)).unwrap();
        assert_eq!(client, (1210, 681));
        assert_eq!(outer, (1226, 720));
        assert!(outer.0 <= limit.0);
        assert!(outer.1 <= limit.1);
        assert!((client.0 * 9 - client.1 * 16).abs() <= 16);
        assert_eq!((outer.0 - client.0, outer.1 - client.1), chrome);
    }

    #[test]
    fn resolution_window_rejects_a_limit_smaller_than_its_title_bar() {
        let area = RECT {
            left: 0,
            top: 0,
            right: 2560,
            bottom: 1440,
        };
        assert_eq!(
            fit_resolution_into(&area, (1920, 1080), (16, 39), Some((100, 20))),
            None
        );
        assert!(!resolution_window_preflight(
            &area,
            (1920, 1080),
            (16, 39),
            Some((100, 20))
        ));
    }

    #[test]
    fn window_limit_requires_two_positive_dimensions() {
        assert_eq!(parse_window_limit("1920x1080"), Some((1920, 1080)));
        assert_eq!(parse_window_limit(" 1920 X 1080 "), Some((1920, 1080)));
        for bad in [
            "",
            "1920",
            "0x1080",
            "1920x0",
            "-1x1080",
            "1920x-1",
            "1920x1080x2",
            "2147483648x1080",
        ] {
            assert_eq!(parse_window_limit(bad), None, "{bad}");
        }
    }

    #[test]
    fn hooked_windows_still_reach_the_title_procedure() {
        let null = std::ptr::null_mut();
        let hwnd = CreateWindowExW(
            0,
            w!("STATIC"),
            w!("before"),
            0,
            0,
            0,
            10,
            10,
            null,
            null,
            null,
            std::ptr::null(),
        );
        assert!(!hwnd.is_null());
        let mut text = [0u16; 16];
        // SAFETY: our window; the buffer holds 16 characters.
        unsafe {
            wm::SetWindowTextW(hwnd, w!("after"));
            let n = wm::GetWindowTextW(hwnd, text.as_mut_ptr(), 16);
            assert_eq!(String::from_utf16_lossy(&text[..n as usize]), "after");
            assert!(!wm::GetPropW(hwnd, TITLE_PROC).is_null());
            wm::DestroyWindow(hwnd);
        }
    }

    /// Whether `hwnd`'s thread answers a message within `ms`.
    fn answers(hwnd: HWND, ms: u32) -> bool {
        let mut result = 0;
        // SAFETY: a plain message with a timeout.
        unsafe {
            wm::SendMessageTimeoutW(hwnd, wm::WM_NULL, 0, 0, wm::SMTO_NORMAL, ms, &mut result) != 0
        }
    }

    fn thread_of(hwnd: HWND) -> u32 {
        // SAFETY: a plain query.
        unsafe { wm::GetWindowThreadProcessId(hwnd, std::ptr::null_mut()) }
    }

    #[test]
    fn a_title_window_is_shown_through_a_stand_in_that_always_answers() {
        // A "title thread" makes its window and then stops reading messages
        // for a while, as a title does in a slow frame or a load.
        let (made, window) = std::sync::mpsc::channel();
        let (done, finish) = std::sync::mpsc::channel::<()>();
        let title = std::thread::spawn(move || {
            let null = std::ptr::null_mut();
            let hwnd = CreateWindowExW(
                0,
                w!("STATIC"),
                w!("stand-in"),
                0,
                0,
                0,
                10,
                10,
                null,
                null,
                null,
                std::ptr::null(),
            );
            made.send(hwnd as usize).unwrap();
            let _ = finish.recv();
            // SAFETY: this thread's window.
            unsafe { wm::DestroyWindow(hwnd) };
        });
        let hwnd = window.recv().unwrap() as HWND;
        assert!(!hwnd.is_null());
        let stand_in = stand_in_of(hwnd);
        assert!(!stand_in.is_null());
        // SAFETY: plain queries of the two windows.
        unsafe {
            // The title's window is message-only: it can never be shown.
            assert_ne!(wm::GetAncestor(hwnd, wm::GA_PARENT), wm::GetDesktopWindow());
            assert_eq!(
                wm::GetAncestor(stand_in, wm::GA_PARENT),
                wm::GetDesktopWindow()
            );
            assert_eq!(wm::IsWindowVisible(hwnd), 0);
            let mut text = [0u16; 16];
            let n = wm::GetWindowTextW(stand_in, text.as_mut_ptr(), 16);
            assert_eq!(String::from_utf16_lossy(&text[..n as usize]), "stand-in");
        }
        assert_ne!(thread_of(stand_in), thread_of(hwnd));
        assert!(!answers(hwnd, 200), "the title's thread is not reading");
        assert!(answers(stand_in, 1000), "the stand-in's thread always is");
        done.send(()).unwrap();
        title.join().unwrap();
        // The stand-in goes with the title's window.
        let gone = (0..100).any(|_| {
            std::thread::sleep(Duration::from_millis(10));
            // SAFETY: a plain query; a destroyed handle gives false.
            unsafe { wm::IsWindow(stand_in) == 0 }
        });
        assert!(gone);
    }

    #[test]
    fn showing_a_title_window_shows_its_stand_in() {
        let null = std::ptr::null_mut();
        let hwnd = CreateWindowExW(
            0,
            w!("STATIC"),
            w!("show"),
            wm::WS_OVERLAPPEDWINDOW,
            0,
            0,
            10,
            10,
            null,
            null,
            null,
            std::ptr::null(),
        );
        let stand_in = stand_in_of(hwnd);
        assert!(!stand_in.is_null());
        // SAFETY: our windows. Minimised and not activated, so the test
        // takes no focus from anything.
        unsafe {
            assert_eq!(ShowWindow(hwnd, wm::SW_SHOWMINNOACTIVE), 0);
            assert_ne!(wm::IsWindowVisible(stand_in), 0);
            assert_ne!(wm::IsIconic(stand_in), 0);
            assert_eq!(wm::IsWindowVisible(hwnd), 0);
            // The title's text follows it to the stand-in.
            wm::SetWindowTextW(hwnd, w!("renamed"));
            let mut text = [0u16; 16];
            let n = wm::GetWindowTextW(stand_in, text.as_mut_ptr(), 16);
            assert_eq!(String::from_utf16_lossy(&text[..n as usize]), "renamed");
            wm::DestroyWindow(hwnd);
        }
    }

    thread_local! {
        /// The system commands `recording` received, low four bits cleared.
        static COMMANDS: std::cell::RefCell<Vec<usize>> = const { std::cell::RefCell::new(Vec::new()) };
    }

    /// Stands in for a title's procedure: notes system commands and acts
    /// on none of them (no menu loop, no minimising).
    extern "system" fn recording(
        hwnd: HWND,
        message: u32,
        wparam: WPARAM,
        lparam: LPARAM,
    ) -> LRESULT {
        if message == wm::WM_SYSCOMMAND {
            COMMANDS.with(|c| c.borrow_mut().push(wparam & 0xFFF0));
            return 0;
        }
        // SAFETY: the system's default procedure, called as the system would.
        unsafe { wm::DefWindowProcW(hwnd, message, wparam, lparam) }
    }

    #[test]
    fn the_keyboard_menu_never_reaches_the_title() {
        let null = std::ptr::null_mut();
        let hwnd = CreateWindowExW(
            0,
            w!("STATIC"),
            w!("menu"),
            0,
            0,
            0,
            10,
            10,
            null,
            null,
            null,
            std::ptr::null(),
        );
        assert!(!hwnd.is_null());
        // SAFETY: our window, owned by this thread, so each message is
        // handled before SendMessageW returns. `recording` stands in for the
        // title's procedure and the original is put back before the window
        // is destroyed.
        unsafe {
            let title = wm::GetPropW(hwnd, TITLE_PROC);
            wm::SetPropW(hwnd, TITLE_PROC, recording as *const () as *mut c_void);
            let command = |c: u32, key: u8| {
                wm::SendMessageW(hwnd, wm::WM_SYSCOMMAND, c as usize, LPARAM::from(key))
            };
            command(wm::SC_KEYMENU, 0); // Alt or F10
            command(wm::SC_KEYMENU, b' '); // Alt+Space
            command(wm::SC_KEYMENU | 1, 0); // the low bits are the system's
            command(wm::SC_MINIMIZE, 0);
            wm::SetPropW(hwnd, TITLE_PROC, title);
            wm::DestroyWindow(hwnd);
        }
        let seen = COMMANDS.with(|c| c.borrow().clone());
        assert_eq!(seen, [wm::SC_MINIMIZE as usize]);
    }
}
