//! Accessibility: closed captions, high contrast, speech-to-text overlay
//! stubs, and speech synthesis for the title's narration. No voice is
//! installed here, and every stream is valid WAVE audio of no length, so
//! narration is silent.
//!
//! Slots `0x18..=0x38` and `0x88..=0xa0` map to the public API reference
//! (`XClosedCaptionGetProperties` at `0x18`, `XClosedCaptionSetEnabled` at
//! `0x20`, `XHighContrastGetMode` at `0x28`, `XSpeechToTextSetPositionHint` at
//! `0x30`, `XSpeechToTextSendString` at `0x38`, and `0x88..=0xa0` for
//! `XSpeechToText{Begin,Update,Finalize,Cancel}HypothesisString`), matching the
//! existing vtable groups and public C signatures.
// shortcut: silent speech and no speech-to-text overlay. The host's engine
// (SAPI through COM, or WinRT speech synthesis) needs COM bindings this crate
// has only in diagnostics builds, plus voice enumeration and SSML the title
// sends (slot 0xa8); the title then plays the empty stream and goes on.
#![allow(clippy::not_unsafe_ptr_arg_deref)]

use std::ffi::{c_char, c_void};
use std::sync::atomic::{AtomicBool, AtomicU32, Ordering};

use super::xasync::{E_INVALIDARG, E_NOT_SUFFICIENT_BUFFER};
use super::xgameruntime::{
    Guid, HRESULT, Interface, Methods, Object, S_OK, add_ref, missing, put, text,
};

/// Closed captions, high contrast, speech to text and synthesis.
pub const PROVIDER: Guid = Guid::new(
    0x3e24_1977,
    0x6237,
    0x41e9,
    [0x85, 0x59, 0x10, 0x2c, 0x2d, 0x99, 0x83, 0xf1],
);
/// Unchanged since 2020.
pub const IID: Guid = Guid::new(
    0xd722_b373,
    0x8c4d,
    0x4692,
    [0x9e, 0x51, 0xd6, 0xad, 0x9b, 0x37, 0xaa, 0x7d],
);

pub static OBJECT: Object = Object::new("accessibility", &VTABLE);
pub const INTERFACES: &[Interface] = &[(PROVIDER, IID, &OBJECT)];

/// A WAVE file of 16 kHz mono 16-bit PCM with no samples: RIFF size 36,
/// a 16-byte format chunk (PCM, 1 channel, 16,000 Hz, 32,000 bytes/s,
/// block 2, 16 bits), an empty data chunk.
const SILENCE: [u8; 44] =
    *b"RIFF\x24\0\0\0WAVEfmt \x10\0\0\0\x01\0\x01\0\x80\x3e\0\0\0\x7d\0\0\x02\0\x10\0data\0\0\0\0";

/// Synthesizers and streams hold nothing, so their handles are fixed
/// addresses and closing them is free.
static SYNTHESIZER: u8 = 0;
static STREAM: u8 = 0;

type This = *const Object;

/// `XClosedCaptionProperties` from the public API reference: `BackgroundColor`,
/// `FontColor`, and `WindowColor` (32-bit ARGB), `FontEdgeAttribute`
/// (`XClosedCaptionFontEdgeAttribute`), `FontStyle` (`XClosedCaptionFontStyle`),
/// `FontScale` (`f32`), and `Enabled` (`bool`).
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ClosedCaptionProperties {
    pub background_color: u32,
    pub font_color: u32,
    pub window_color: u32,
    pub font_edge_attribute: u32,
    pub font_style: u32,
    pub font_scale: f32,
    pub enabled: bool,
}

const _: () = assert!(size_of::<ClosedCaptionProperties>() == 28);
const _: () = assert!(align_of::<ClosedCaptionProperties>() == 4);

static CAPTIONS_ENABLED: AtomicBool = AtomicBool::new(false);
static NEXT_HYPOTHESIS: AtomicU32 = AtomicU32::new(1);

/// Default caption appearance: white text (`0xffff_ffff`), default edge and
/// style (`0`), `1.0` scale, and `enabled: false`.
const DEFAULT_CAPTIONS: ClosedCaptionProperties = ClosedCaptionProperties {
    background_color: 0x0000_0000,
    font_color: 0xffff_ffff,
    window_color: 0x0000_0000,
    font_edge_attribute: 0,
    font_style: 0,
    font_scale: 1.0,
    enabled: false,
};

/// Slot `0x18` (`XClosedCaptionGetProperties`): writes `ClosedCaptionProperties`
/// with the current `enabled` state and default appearance, returning `S_OK`
/// (or `E_INVALIDARG` if `out` is null).
extern "system" fn caption_properties(_: This, out: *mut ClosedCaptionProperties) -> HRESULT {
    put(
        out,
        ClosedCaptionProperties {
            enabled: CAPTIONS_ENABLED.load(Ordering::Relaxed),
            ..DEFAULT_CAPTIONS
        },
    )
}

/// Slot `0x20` (`XClosedCaptionSetEnabled`): tracks whether closed captions are
/// enabled so `XClosedCaptionGetProperties` follows the title's state.
extern "system" fn set_captions_enabled(_: This, enabled: u8) -> HRESULT {
    let enabled = enabled != 0;
    CAPTIONS_ENABLED.store(enabled, Ordering::Relaxed);
    tracing::debug!(enabled, "closed captions");
    S_OK
}

/// Slot `0x28` (`XHighContrastGetMode`): reports `XHighContrastMode::Off`
/// (`0`), matching standard SDR presentation with no high-contrast theme.
extern "system" fn high_contrast_mode(_: This, out: *mut u32) -> HRESULT {
    put(out, 0u32)
}

/// Slot `0x30` (`XSpeechToTextSetPositionHint`): accepts an
/// `XSpeechToTextPositionHint` (`u32`) and logs it without drawing an overlay.
extern "system" fn set_position_hint(_: This, position: u32) -> HRESULT {
    tracing::debug!(position, "speech-to-text position hint");
    S_OK
}

/// Slot `0x38` (`XSpeechToTextSendString`): logs `speakerName`, `content`, and
/// `XSpeechToTextType` (`kind`) without drawing an overlay.
extern "system" fn send_string(
    _: This,
    speaker: *const c_char,
    content: *const c_char,
    kind: u32,
) -> HRESULT {
    tracing::debug!(
        speaker = text(speaker).unwrap_or_default(),
        content = text(content).unwrap_or_default(),
        kind,
        "speech-to-text string"
    );
    S_OK
}

/// Slot `0x88` (`XSpeechToTextBeginHypothesisString`): assigns a monotonic
/// non-zero `u32` `hypothesisId` (or `E_INVALIDARG` if `out` is null).
extern "system" fn begin_hypothesis(
    _: This,
    speaker: *const c_char,
    content: *const c_char,
    kind: u32,
    out: *mut u32,
) -> HRESULT {
    if out.is_null() {
        return E_INVALIDARG;
    }
    let id = NEXT_HYPOTHESIS
        .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |n| {
            Some(n.wrapping_add(1).max(1))
        })
        .unwrap_or(1);
    tracing::debug!(
        id,
        speaker = text(speaker).unwrap_or_default(),
        content = text(content).unwrap_or_default(),
        kind,
        "speech-to-text begin hypothesis"
    );
    put(out, id)
}

/// Slot `0x90` (`XSpeechToTextUpdateHypothesisString`): logs the updated
/// hypothesis text without drawing an overlay.
extern "system" fn update_hypothesis(_: This, id: u32, content: *const c_char) -> HRESULT {
    tracing::debug!(
        id,
        content = text(content).unwrap_or_default(),
        "speech-to-text update hypothesis"
    );
    S_OK
}

/// Slot `0x98` (`XSpeechToTextFinalizeHypothesisString`): logs the finalized
/// hypothesis text without drawing an overlay.
extern "system" fn finalize_hypothesis(_: This, id: u32, content: *const c_char) -> HRESULT {
    tracing::debug!(
        id,
        content = text(content).unwrap_or_default(),
        "speech-to-text finalize hypothesis"
    );
    S_OK
}

/// Slot `0xa0` (`XSpeechToTextCancelHypothesisString`): logs cancellation of an
/// active hypothesis ID.
extern "system" fn cancel_hypothesis(_: This, id: u32) -> HRESULT {
    tracing::debug!(id, "speech-to-text cancel hypothesis");
    S_OK
}

/// No voice is installed, so the callback never runs.
extern "system" fn enumerate_voices(
    _: This,
    _context: *mut c_void,
    _callback: *const c_void,
) -> HRESULT {
    S_OK
}
extern "system" fn create(_: This, out: *mut *const u8) -> HRESULT {
    put(out, &raw const SYNTHESIZER)
}
extern "system" fn close(_: This, _: *const u8) -> HRESULT {
    S_OK
}
extern "system" fn set_default_voice(_: This, _: *const u8) -> HRESULT {
    S_OK
}
extern "system" fn set_custom_voice(_: This, _: *const u8, _voice: *const c_char) -> HRESULT {
    S_OK
}
extern "system" fn create_stream(
    _: This,
    _: *const u8,
    words: *const c_char,
    out: *mut *const u8,
) -> HRESULT {
    tracing::debug!(text = text(words).unwrap_or_default(), "speech (silent)");
    put(out, &raw const STREAM)
}
extern "system" fn stream_size(_: This, _: *const u8, out: *mut usize) -> HRESULT {
    put(out, SILENCE.len())
}
extern "system" fn stream_data(
    _: This,
    _: *const u8,
    size: usize,
    buffer: *mut u8,
    used: *mut usize,
) -> HRESULT {
    if buffer.is_null() || size < SILENCE.len() {
        return E_NOT_SUFFICIENT_BUFFER;
    }
    // SAFETY: the buffer holds `size` bytes, checked above.
    unsafe { buffer.copy_from_nonoverlapping(SILENCE.as_ptr(), SILENCE.len()) };
    if !used.is_null() {
        // SAFETY: checked non-null.
        unsafe { used.write(SILENCE.len()) };
    }
    S_OK
}

static VTABLE: Methods<22> = Methods([
    missing::<0> as *const (),
    add_ref as *const (),
    add_ref as *const (),
    caption_properties as *const (), // 0x18 XClosedCaptionGetProperties
    set_captions_enabled as *const (), // 0x20 XClosedCaptionSetEnabled
    high_contrast_mode as *const (), // 0x28 XHighContrastGetMode
    set_position_hint as *const (),  // 0x30 XSpeechToTextSetPositionHint
    send_string as *const (),        // 0x38 XSpeechToTextSendString
    enumerate_voices as *const (),   // 0x40
    create as *const (),             // 0x48
    close as *const (),              // 0x50
    set_default_voice as *const (),  // 0x58
    set_custom_voice as *const (),   // 0x60
    create_stream as *const (),      // 0x68 from text
    close as *const (),              // 0x70 a stream
    stream_size as *const (),        // 0x78
    stream_data as *const (),        // 0x80
    begin_hypothesis as *const (),   // 0x88 XSpeechToTextBeginHypothesisString
    update_hypothesis as *const (),  // 0x90 XSpeechToTextUpdateHypothesisString
    finalize_hypothesis as *const (), // 0x98 XSpeechToTextFinalizeHypothesisString
    cancel_hypothesis as *const (),  // 0xa0 XSpeechToTextCancelHypothesisString
    create_stream as *const (),      // 0xa8 from SSML
]);

#[cfg(test)]
mod tests {
    use super::*;
    use crate::modules::xgameruntime::method;

    #[test]
    fn streams_are_empty_wave_files() {
        let this: This = &OBJECT;
        type Create = extern "system" fn(This, *mut *const u8) -> HRESULT;
        type Stream = extern "system" fn(This, *const u8, *const c_char, *mut *const u8) -> HRESULT;
        type Size = extern "system" fn(This, *const u8, *mut usize) -> HRESULT;
        type Data = extern "system" fn(This, *const u8, usize, *mut u8, *mut usize) -> HRESULT;
        let (mut synth, mut stream, mut size) = (std::ptr::null(), std::ptr::null(), 0);
        assert_eq!(method::<Create>(&OBJECT, 0x48)(this, &mut synth), S_OK);
        let say = method::<Stream>(&OBJECT, 0x68);
        assert_eq!(say(this, synth, c"Hello".as_ptr(), &mut stream), S_OK);
        assert_eq!(method::<Size>(&OBJECT, 0x78)(this, stream, &mut size), S_OK);
        let mut wav = vec![0u8; size];
        let mut used = 0;
        let data = method::<Data>(&OBJECT, 0x80);
        assert_eq!(data(this, stream, size, wav.as_mut_ptr(), &mut used), S_OK);
        assert_eq!(
            (&wav[..4], &wav[8..16], &wav[36..40]),
            (&b"RIFF"[..], &b"WAVEfmt "[..], &b"data"[..])
        );
        assert_eq!(u32::from_le_bytes(wav[24..28].try_into().unwrap()), 16_000);
        assert_eq!(used, 44);
    }

    #[test]
    fn closed_captions_follow_enabled_state_and_layout() {
        let this: This = &OBJECT;
        type Get = extern "system" fn(This, *mut ClosedCaptionProperties) -> HRESULT;
        type Set = extern "system" fn(This, u8) -> HRESULT;
        let get = method::<Get>(&OBJECT, 0x18);
        let set = method::<Set>(&OBJECT, 0x20);

        assert_eq!(
            std::mem::offset_of!(ClosedCaptionProperties, font_scale),
            20
        );
        assert_eq!(std::mem::offset_of!(ClosedCaptionProperties, enabled), 24);
        assert_eq!(get(this, std::ptr::null_mut()), E_INVALIDARG);

        let mut props = ClosedCaptionProperties {
            background_color: 0xdead_beef,
            font_color: 0,
            window_color: 0xdead_beef,
            font_edge_attribute: 9,
            font_style: 9,
            font_scale: 0.0,
            enabled: true,
        };
        assert_eq!(get(this, &mut props), S_OK);
        assert_eq!(props, DEFAULT_CAPTIONS);

        // Any non-zero byte enables captions.
        assert_eq!(set(this, 1), S_OK);
        assert_eq!(get(this, &mut props), S_OK);
        assert_eq!(
            props,
            ClosedCaptionProperties {
                enabled: true,
                ..DEFAULT_CAPTIONS
            }
        );

        assert_eq!(set(this, 0), S_OK);
        assert_eq!(get(this, &mut props), S_OK);
        assert_eq!(props, DEFAULT_CAPTIONS);
    }

    #[test]
    fn high_contrast_reports_off() {
        let this: This = &OBJECT;
        type Mode = extern "system" fn(This, *mut u32) -> HRESULT;
        let mode = method::<Mode>(&OBJECT, 0x28);

        assert_eq!(mode(this, std::ptr::null_mut()), E_INVALIDARG);
        let mut value = u32::MAX;
        assert_eq!(mode(this, &mut value), S_OK);
        assert_eq!(value, 0);
    }

    #[test]
    fn speech_to_text_slots_accept_strings_and_assign_hypothesis_ids() {
        let this: This = &OBJECT;
        type Position = extern "system" fn(This, u32) -> HRESULT;
        type Send = extern "system" fn(This, *const c_char, *const c_char, u32) -> HRESULT;
        type Begin =
            extern "system" fn(This, *const c_char, *const c_char, u32, *mut u32) -> HRESULT;
        type Update = extern "system" fn(This, u32, *const c_char) -> HRESULT;
        type Cancel = extern "system" fn(This, u32) -> HRESULT;

        let speaker = c"Dispatcher".as_ptr();
        let line = c"Proceed to waypoint".as_ptr();
        assert_eq!(method::<Position>(&OBJECT, 0x30)(this, 2), S_OK);
        assert_eq!(method::<Send>(&OBJECT, 0x38)(this, speaker, line, 0), S_OK);
        assert_eq!(
            method::<Send>(&OBJECT, 0x38)(this, std::ptr::null(), std::ptr::null(), 1),
            S_OK
        );

        let begin = method::<Begin>(&OBJECT, 0x88);
        assert_eq!(
            begin(this, speaker, line, 0, std::ptr::null_mut()),
            E_INVALIDARG
        );
        let (mut first, mut second) = (0u32, 0u32);
        assert_eq!(begin(this, speaker, line, 0, &mut first), S_OK);
        assert_ne!(first, 0);
        assert_eq!(
            begin(this, std::ptr::null(), std::ptr::null(), 1, &mut second),
            S_OK
        );
        assert_eq!(second, first + 1);

        assert_eq!(method::<Update>(&OBJECT, 0x90)(this, first, line), S_OK);
        assert_eq!(
            method::<Update>(&OBJECT, 0x90)(this, first, std::ptr::null()),
            S_OK
        );
        assert_eq!(method::<Update>(&OBJECT, 0x98)(this, first, line), S_OK);
        assert_eq!(method::<Cancel>(&OBJECT, 0xa0)(this, second), S_OK);
    }
}
