//! The console's audio decoder (`xapu.dll`): Opus packets in, float PCM out.
//!
//! A title connects once for one kind of processing, gets a block of memory
//! for its packets and output, and then queues per-stream commands
//! (activate, process, reset, deactivate) and dequeues one result per
//! command. We decode each command when it is queued, so its result is ready
//! at once; the order of results is the order of commands.
#![allow(non_snake_case)]
#![allow(clippy::not_unsafe_ptr_arg_deref)]

use std::alloc::Layout;
use std::collections::VecDeque;
use std::ffi::c_void;
use std::sync::Mutex;

use opus_pure::OpusDecoder;
use windows_sys::Win32::Foundation::{CloseHandle, HANDLE};
use windows_sys::Win32::System::Threading::{CreateEventW, SetEvent};

use super::xasync::E_INVALIDARG;
use super::xgameruntime::{HRESULT, S_OK};
use crate::{HleModule, hle_export};

const E_NOTIMPL: HRESULT = 0x8000_4001_u32 as i32;
const E_OUTOFMEMORY: HRESULT = 0x8007_000E_u32 as i32;
const QUEUE_FULL: HRESULT = 0x8AC8_1000_u32 as i32;
/// No result yet, which titles poll for.
pub(crate) const PENDING_RESULTS: HRESULT = 0x8AC8_1001_u32 as i32;
const INVALID_COMMAND: HRESULT = 0x8AC8_1002_u32 as i32;
const INVALID_STREAM: HRESULT = 0x8AC8_1003_u32 as i32;
const ALREADY_ACTIVE: HRESULT = 0x8AC8_1005_u32 as i32;
const BAD_PACKET: HRESULT = 0x8AC8_1008_u32 as i32;

/// Processing kinds; only plain decoding is implemented.
const DECODE_OPUS: u32 = 1;

const ACTIVATE: u32 = 1;
const DEACTIVATE: u32 = 2;
const PROCESS: u32 = 3;
const NOTIFY: u32 = 4;
const RESET: u32 = 5;

/// Command option: signal the connection's event when the result is ready.
const SIGNAL: u32 = 1;
/// Connect options.
const DEINTERLEAVED_OUTPUT: u32 = 4;
const RESULT_EVENT: u32 = 8;

/// The longest Opus packet duration at 48 kHz (120 ms).
const MAX_FRAMES: usize = 5760;

#[repr(C)]
pub struct ConnectInput {
    process: u32,
    streams: u32,
    commands_per_stream: u32,
    memory: u32,
    frames_per_output: u32,
    options: u32,
}

#[repr(C)]
pub struct ConnectOutput {
    memory: *mut c_void,
    result_event: HANDLE,
}

/// What every command starts with, and what its result echoes.
#[repr(C, packed)]
#[derive(Clone, Copy)]
pub struct CommandId {
    kind: u32,
    stream: u32,
    sequence: u32,
    options: u32,
    context: u64,
}

#[repr(C, packed)]
struct Activate {
    id: CommandId,
    channels: u32,
    init_length: u32,
    init: *const c_void,
    start_pitch: f32,
    processing_length: u32,
    processing: *mut c_void,
}

#[repr(C, packed)]
#[derive(Clone, Copy)]
struct Process {
    id: CommandId,
    pitch_ramp: f32,
    target_pitch: f32,
    first_frame: u32,
    frame_count: u32,
    input_length: u32,
    input: *const u8,
    output_length: u32,
    output: *mut u8,
}

#[repr(C, packed)]
#[derive(Clone, Copy)]
pub struct ApuResult {
    id: CommandId,
    state: u32,
    state_data: u64,
    usec: u64,
    error: HRESULT,
    frames: u32,
    output_length: u32,
    output: *mut u8,
    pitch: f32,
    input_length: u32,
    input: *const u8,
}

// SAFETY: the raw pointers are title addresses carried through, never
// dereferenced after the result is made.
unsafe impl Send for ApuResult {}

pub struct Connection {
    memory: *mut u8,
    layout: Layout,
    event: HANDLE,
    options: u32,
    per_stream: u32,
    state: Mutex<State>,
}

struct State {
    streams: Vec<Option<OpusDecoder>>,
    /// Results not yet dequeued, per stream.
    queued: Vec<u32>,
    results: VecDeque<ApuResult>,
    commands: u32,
    dequeued: u32,
}

fn lock(c: &Connection) -> std::sync::MutexGuard<'_, State> {
    c.state.lock().unwrap_or_else(|e| e.into_inner())
}

pub extern "system" fn XApuConnect(
    input: *const ConnectInput,
    output: *mut ConnectOutput,
    handle: *mut *mut Connection,
) -> HRESULT {
    if input.is_null() || output.is_null() || handle.is_null() {
        return E_INVALIDARG;
    }
    // SAFETY: checked non-null.
    let input = unsafe { &*input };
    if !(1..=500).contains(&input.streams)
        || !(1..=16).contains(&input.commands_per_stream)
        || input.memory == 0
    {
        return E_INVALIDARG;
    }
    if input.process != DECODE_OPUS {
        // shortcut: rate conversion (decode-convert, PCM convert) is not
        // implemented; add a resampler if a title connects for it.
        tracing::warn!(
            process = input.process,
            "audio decoder: processing kind not implemented"
        );
        return E_NOTIMPL;
    }
    if input.frames_per_output != 0 {
        return E_INVALIDARG;
    }
    let Ok(layout) = Layout::from_size_align(input.memory as usize, 16) else {
        return E_INVALIDARG;
    };
    // SAFETY: non-zero size.
    let memory = unsafe { std::alloc::alloc_zeroed(layout) };
    if memory.is_null() {
        return E_OUTOFMEMORY;
    }
    let event = if input.options & RESULT_EVENT != 0 {
        // SAFETY: plain auto-reset event, no name.
        let event = unsafe { CreateEventW(std::ptr::null(), 0, 0, std::ptr::null()) };
        if event.is_null() {
            // SAFETY: allocated above with `layout`.
            unsafe { std::alloc::dealloc(memory, layout) };
            return E_OUTOFMEMORY;
        }
        event
    } else {
        std::ptr::null_mut()
    };
    let streams = input.streams as usize;
    tracing::info!(
        streams,
        memory = input.memory,
        options = input.options,
        "audio decoder connected"
    );
    let connection = Box::new(Connection {
        memory,
        layout,
        event,
        options: input.options,
        per_stream: input.commands_per_stream,
        state: Mutex::new(State {
            streams: (0..streams).map(|_| None).collect(),
            queued: vec![0; streams],
            results: VecDeque::new(),
            commands: 0,
            dequeued: 0,
        }),
    });
    // SAFETY: checked non-null.
    unsafe {
        output.write(ConnectOutput {
            memory: memory.cast(),
            result_event: event,
        });
        handle.write(Box::into_raw(connection));
    }
    S_OK
}

pub extern "system" fn XApuDisconnect(handle: *mut Connection) -> HRESULT {
    if handle.is_null() {
        return E_INVALIDARG;
    }
    // SAFETY: made by `XApuConnect`; disconnected once.
    let connection = unsafe { Box::from_raw(handle) };
    if !connection.event.is_null() {
        // SAFETY: our event.
        unsafe { CloseHandle(connection.event) };
    }
    // SAFETY: allocated in `XApuConnect` with this layout.
    unsafe { std::alloc::dealloc(connection.memory, connection.layout) };
    S_OK
}

pub extern "system" fn XApuEnqueueCommand(
    handle: *mut Connection,
    command: *const CommandId,
    total: *mut u32,
) -> HRESULT {
    if handle.is_null() || command.is_null() {
        return E_INVALIDARG;
    }
    // SAFETY: a connection from `XApuConnect`; every command starts with an id.
    let (connection, id) = unsafe { (&*handle, command.read_unaligned()) };
    let mut state = lock(connection);
    let stream = id.stream as usize;
    if stream >= state.streams.len() {
        return INVALID_STREAM;
    }
    if state.queued[stream] >= connection.per_stream {
        return QUEUE_FULL;
    }
    let mut result = ApuResult {
        id,
        state: 0,
        state_data: 0,
        usec: 0,
        error: S_OK,
        frames: 0,
        output_length: 0,
        output: std::ptr::null_mut(),
        pitch: 1.0,
        input_length: 0,
        input: std::ptr::null(),
    };
    let decoder = &mut state.streams[stream];
    match id.kind {
        ACTIVATE => {
            // SAFETY: an activate command is this struct.
            let channels = unsafe { command.cast::<Activate>().read_unaligned() }.channels;
            tracing::debug!(stream, channels, "Opus stream activated");
            result.error = if decoder.is_some() {
                ALREADY_ACTIVE
            } else if !(1..=2).contains(&channels) {
                E_INVALIDARG
            } else {
                match OpusDecoder::new(48_000, channels as usize) {
                    Ok(d) => {
                        *decoder = Some(d);
                        S_OK
                    }
                    Err(_) => E_INVALIDARG,
                }
            };
        }
        DEACTIVATE => *decoder = None,
        RESET => {
            if let Some(d) = decoder {
                let _ = d.reset_state();
            }
        }
        NOTIFY => {}
        PROCESS => {
            // SAFETY: a process command is this struct.
            let command = unsafe { command.cast::<Process>().read_unaligned() };
            match decoder {
                Some(d) => process(d, connection.options, &command, &mut result),
                None => result.error = INVALID_STREAM,
            }
        }
        _ => return INVALID_COMMAND,
    }
    state.queued[stream] += 1;
    state.commands = state.commands.wrapping_add(1);
    state.results.push_back(result);
    if !total.is_null() {
        // SAFETY: optional out parameter.
        unsafe { total.write(state.commands) };
    }
    if id.options & SIGNAL != 0 && !connection.event.is_null() {
        // SAFETY: our event.
        unsafe { SetEvent(connection.event) };
    }
    S_OK
}

/// Decodes one packet into the command's output buffer, keeping the frames
/// from `first_frame` on, at most `frame_count` of them.
fn process(decoder: &mut OpusDecoder, options: u32, command: &Process, result: &mut ApuResult) {
    let channels = decoder.channels();
    let (input, input_length) = (command.input, command.input_length);
    let (output, output_length) = (command.output, command.output_length);
    result.input = input;
    result.input_length = input_length;
    result.output = output;
    if input.is_null() || output.is_null() || input_length == 0 {
        result.error = E_INVALIDARG;
        return;
    }
    // SAFETY: the title's packet in its connection memory.
    let packet = unsafe { std::slice::from_raw_parts(input, input_length as usize) };
    let mut pcm = vec![0f32; MAX_FRAMES * channels];
    let decoded = match decoder.decode(packet, MAX_FRAMES, &mut pcm) {
        Ok(n) => n,
        Err(_) => {
            result.error = BAD_PACKET;
            return;
        }
    };
    let first = (command.first_frame as usize).min(decoded);
    let frames = (decoded - first)
        .min(command.frame_count as usize)
        .min(output_length as usize / (4 * channels));
    let kept = &pcm[first * channels..(first + frames) * channels];
    let bytes = frames * channels * 4;
    if bytes > 0 {
        // SAFETY: `output` is non-null and holds at least `bytes <= output_length` bytes;
        // writing raw bytes avoids requiring 4-byte alignment on the title's `*mut u8`.
        let out = unsafe { std::slice::from_raw_parts_mut(output, bytes) };
        if options & DEINTERLEAVED_OUTPUT != 0 && channels == 2 {
            let (left, right) = out.split_at_mut(frames * 4);
            let (left, _) = left.as_chunks_mut::<4>();
            let (right, _) = right.as_chunks_mut::<4>();
            for (i, frame) in kept.as_chunks::<2>().0.iter().enumerate() {
                left[i] = frame[0].to_le_bytes();
                right[i] = frame[1].to_le_bytes();
            }
        } else {
            for (dst, &sample) in out.as_chunks_mut::<4>().0.iter_mut().zip(kept) {
                *dst = sample.to_le_bytes();
            }
        }
    }
    result.frames = frames as u32;
    result.output_length = bytes as u32;
}

pub extern "system" fn XApuDequeueResult(handle: *mut Connection, out: *mut ApuResult) -> HRESULT {
    if handle.is_null() || out.is_null() {
        return E_INVALIDARG;
    }
    // SAFETY: a connection from `XApuConnect`.
    let mut state = lock(unsafe { &*handle });
    let Some(result) = state.results.pop_front() else {
        return PENDING_RESULTS;
    };
    let stream = result.id.stream as usize;
    state.queued[stream] -= 1;
    state.dequeued = state.dequeued.wrapping_add(1);
    // SAFETY: the title passes room for one result.
    unsafe { out.write_unaligned(result) };
    S_OK
}

pub extern "system" fn XApuGetResultCount(handle: *mut Connection, total: *mut u32) -> u32 {
    if handle.is_null() {
        return 0;
    }
    // SAFETY: a connection from `XApuConnect`.
    let state = lock(unsafe { &*handle });
    if !total.is_null() {
        // SAFETY: optional out parameter.
        unsafe { total.write(state.dequeued) };
    }
    state.results.len() as u32
}

pub(crate) fn module() -> HleModule {
    HleModule {
        name: "apu",
        dlls: &["xapu.dll"],
        exports: vec![
            hle_export!(XApuConnect),
            hle_export!(XApuDisconnect),
            hle_export!(XApuEnqueueCommand),
            hle_export!(XApuDequeueResult),
            hle_export!(XApuGetResultCount),
        ],
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use opus_pure::{Application, OpusEncoder};
    use windows_sys::Win32::Foundation::{WAIT_OBJECT_0, WAIT_TIMEOUT};
    use windows_sys::Win32::System::Threading::WaitForSingleObject;

    /// Connection memory for one test: every packet and output buffer fits.
    const MEMORY: u32 = 0x4_0000;
    /// Room the tests give a decode for its output.
    const OUTPUT: u32 = 0x2000;

    fn id(kind: u32, stream: u32) -> CommandId {
        CommandId {
            kind,
            stream,
            sequence: 7,
            options: 0,
            context: 0x1234,
        }
    }

    fn activate_command(stream: u32, channels: u32) -> Activate {
        Activate {
            id: id(ACTIVATE, stream),
            channels,
            init_length: 0,
            init: std::ptr::null(),
            start_pitch: 1.0,
            processing_length: 0,
            processing: std::ptr::null_mut(),
        }
    }

    fn process_command(stream: u32, packet: &[u8], output: *mut u8) -> Process {
        Process {
            id: id(PROCESS, stream),
            pitch_ramp: 0.0,
            target_pitch: 1.0,
            first_frame: 0,
            frame_count: 960,
            input_length: packet.len() as u32,
            input: packet.as_ptr(),
            output_length: OUTPUT,
            output,
        }
    }

    // `ApuResult` is packed, so every field has to be copied out by value
    // before it can be compared; these do that and keep the tests readable.
    fn kind(result: &ApuResult) -> u32 {
        result.id.kind
    }

    fn stream(result: &ApuResult) -> u32 {
        result.id.stream
    }

    fn sequence(result: &ApuResult) -> u32 {
        result.id.sequence
    }

    fn outcome(result: &ApuResult) -> (HRESULT, u32, u32) {
        (result.error, result.frames, result.output_length)
    }

    /// An Opus packet we make ourselves: `frames` samples per channel of a sine
    /// at `hz`, the right channel a fifth up and quieter so the two differ.
    fn tone_packet(channels: u32, frames: usize, hz: f32) -> Vec<u8> {
        let mut tone = Vec::with_capacity(frames * channels as usize);
        for i in 0..frames {
            let t = i as f32 / 48_000.0;
            tone.push((t * hz * std::f32::consts::TAU).sin() * 0.5);
            if channels == 2 {
                tone.push((t * hz * 1.5 * std::f32::consts::TAU).sin() * 0.25);
            }
        }
        let mut encoder = OpusEncoder::new(48_000, channels as usize, Application::Audio).unwrap();
        let mut data = vec![0u8; 4000];
        let written = encoder.encode(&tone, frames, &mut data).unwrap();
        data.truncate(written);
        data
    }

    /// What a decoder that has seen nothing else produces for `packet`.
    fn fresh_pcm(packet: &[u8], channels: usize, frames: usize) -> Vec<f32> {
        let mut decoder = OpusDecoder::new(48_000, channels).unwrap();
        let mut pcm = vec![0f32; MAX_FRAMES * channels];
        let produced = decoder.decode(packet, MAX_FRAMES, &mut pcm).unwrap();
        assert_eq!(produced, frames);
        pcm.truncate(produced * channels);
        pcm
    }

    struct Fixture {
        handle: *mut Connection,
        base: *mut u8,
        event: HANDLE,
    }

    impl Fixture {
        fn new(streams: u32, per_stream: u32, options: u32) -> Self {
            let input = ConnectInput {
                process: DECODE_OPUS,
                streams,
                commands_per_stream: per_stream,
                memory: MEMORY,
                frames_per_output: 0,
                options,
            };
            let mut output = ConnectOutput {
                memory: std::ptr::null_mut(),
                result_event: std::ptr::null_mut(),
            };
            let mut handle = std::ptr::null_mut();
            assert_eq!(XApuConnect(&input, &mut output, &mut handle), S_OK);
            assert!(!handle.is_null(), "a connection hands out a handle");
            assert!(!output.memory.is_null(), "a connection hands out memory");
            Self {
                handle,
                base: output.memory.cast(),
                event: output.result_event,
            }
        }

        fn enqueue(&self, command: *const CommandId) -> HRESULT {
            XApuEnqueueCommand(self.handle, command, std::ptr::null_mut())
        }

        fn dequeue(&self) -> ApuResult {
            let mut result = std::mem::MaybeUninit::<ApuResult>::zeroed();
            assert_eq!(XApuDequeueResult(self.handle, result.as_mut_ptr()), S_OK);
            // SAFETY: the call above wrote one whole result.
            unsafe { result.assume_init() }
        }

        fn count(&self) -> u32 {
            XApuGetResultCount(self.handle, std::ptr::null_mut())
        }

        /// The floats the last decode wrote at the output buffer.
        fn output(&self, floats: usize) -> &[f32] {
            // SAFETY: the decoder wrote that many floats at the buffer's start.
            unsafe { std::slice::from_raw_parts(self.base.cast::<f32>(), floats) }
        }
    }

    impl Drop for Fixture {
        fn drop(&mut self) {
            assert_eq!(XApuDisconnect(self.handle), S_OK);
        }
    }

    /// Activate `stream` and check that the result is a success.
    fn activate_stream(fixture: &Fixture, stream: u32, channels: u32) {
        let command = activate_command(stream, channels);
        assert_eq!(fixture.enqueue(std::ptr::from_ref(&command).cast()), S_OK);
        let result = fixture.dequeue();
        assert_eq!(outcome(&result).0, S_OK);
    }

    /// Decode `packet` on a stream that is already active.
    fn process_packet(
        fixture: &Fixture,
        stream: u32,
        packet: &[u8],
        tweak: impl FnOnce(&mut Process),
    ) -> ApuResult {
        let mut command = process_command(stream, packet, fixture.base);
        tweak(&mut command);
        assert_eq!(fixture.enqueue(std::ptr::from_ref(&command).cast()), S_OK);
        fixture.dequeue()
    }

    /// Activate a stream that has never decoded, then decode `packet` on it.
    fn decode(
        fixture: &Fixture,
        stream: u32,
        channels: u32,
        packet: &[u8],
        tweak: impl FnOnce(&mut Process),
    ) -> ApuResult {
        activate_stream(fixture, stream, channels);
        process_packet(fixture, stream, packet, tweak)
    }

    #[test]
    fn decodes_opus_as_the_title_drives_it() {
        assert_eq!(size_of::<ApuResult>(), 0x50);
        // 256 mono/stereo streams, two commands each, as the title connects.
        let input = ConnectInput {
            process: DECODE_OPUS,
            streams: 256,
            commands_per_stream: 2,
            memory: 0x10_4000,
            frames_per_output: 0,
            options: 1,
        };
        let mut output = ConnectOutput {
            memory: std::ptr::null_mut(),
            result_event: std::ptr::null_mut(),
        };
        let mut handle = std::ptr::null_mut();
        assert_eq!(XApuConnect(&input, &mut output, &mut handle), S_OK);
        let base = output.memory.cast::<u8>();

        let activate = Activate {
            id: id(ACTIVATE, 3),
            channels: 2,
            init_length: 0,
            init: std::ptr::null(),
            start_pitch: 1.0,
            processing_length: 0,
            processing: std::ptr::null_mut(),
        };
        assert_eq!(
            XApuEnqueueCommand(handle, (&raw const activate).cast(), std::ptr::null_mut()),
            S_OK
        );

        // A 20 ms stereo packet of a 440 Hz tone.
        let tone: Vec<f32> = (0..960 * 2)
            .map(|i| ((i / 2) as f32 * 440.0 * std::f32::consts::TAU / 48_000.0).sin() * 0.5)
            .collect();
        let mut encoder = OpusEncoder::new(48_000, 2, Application::Audio).unwrap();
        let mut packet = [0u8; 1500];
        let n = encoder.encode(&tone, 960, &mut packet).unwrap();
        // SAFETY: inside the connection memory.
        unsafe { std::ptr::copy_nonoverlapping(packet.as_ptr(), base, n) };
        let process = Process {
            id: id(PROCESS, 3),
            pitch_ramp: 0.0,
            target_pitch: 1.0,
            first_frame: 0,
            frame_count: 960,
            input_length: n as u32,
            input: base,
            output_length: 0xf00 * 2,
            output: base.wrapping_add(0x14000),
        };
        let mut total = 0;
        assert_eq!(
            XApuEnqueueCommand(handle, (&raw const process).cast(), &mut total),
            S_OK
        );
        assert_eq!(total, 2);
        // Two results queued for stream 3: its queue is full.
        assert_eq!(
            XApuEnqueueCommand(handle, (&raw const process).cast(), &mut total),
            QUEUE_FULL
        );
        assert_eq!(XApuGetResultCount(handle, std::ptr::null_mut()), 2);

        let mut result = std::mem::MaybeUninit::<ApuResult>::zeroed();
        assert_eq!(XApuDequeueResult(handle, result.as_mut_ptr()), S_OK);
        // SAFETY: written above.
        let first = unsafe { result.assume_init() };
        let (kind, error, context) = (first.id.kind, first.error, first.id.context);
        assert_eq!((kind, error, context), (ACTIVATE, S_OK, 0x1234));
        assert_eq!(XApuDequeueResult(handle, result.as_mut_ptr()), S_OK);
        // SAFETY: written above.
        let decoded = unsafe { result.assume_init() };
        let (kind, error, frames, length) = (
            decoded.id.kind,
            decoded.error,
            decoded.frames,
            decoded.output_length,
        );
        assert_eq!((kind, error, frames, length), (PROCESS, S_OK, 960, 960 * 8));
        // SAFETY: the decoder wrote 960 stereo frames there.
        let pcm =
            unsafe { std::slice::from_raw_parts(base.wrapping_add(0x14000).cast::<f32>(), 1920) };
        let energy: f32 = pcm.iter().map(|s| s * s).sum::<f32>() / 1920.0;
        assert!(energy > 0.01, "decoded a tone, energy {energy}");
        assert_eq!(
            XApuDequeueResult(handle, result.as_mut_ptr()),
            PENDING_RESULTS
        );
        let mut dequeued = 0;
        assert_eq!(XApuGetResultCount(handle, &mut dequeued), 0);
        assert_eq!(dequeued, 2);

        // A process on a stream that was never activated fails in its result.
        let mut stray = process;
        stray.id.stream = 4;
        assert_eq!(
            XApuEnqueueCommand(handle, (&raw const stray).cast(), &mut total),
            S_OK
        );
        assert_eq!(XApuDequeueResult(handle, result.as_mut_ptr()), S_OK);
        // SAFETY: written above.
        let error = unsafe { result.assume_init() }.error;
        assert_eq!(error, INVALID_STREAM);
        assert_eq!(XApuDisconnect(handle), S_OK);
    }

    #[test]
    fn decodes_into_unaligned_output_interleaved_and_deinterleaved() {
        let tone: Vec<f32> = (0..960 * 2)
            .map(|i| {
                let phase = (i / 2) as f32 * std::f32::consts::TAU / 48_000.0;
                if i % 2 == 0 {
                    (phase * 440.0).sin() * 0.5
                } else {
                    (phase * 880.0).sin() * 0.25
                }
            })
            .collect();
        let mut encoder = OpusEncoder::new(48_000, 2, Application::Audio).unwrap();
        let mut packet = [0u8; 1500];
        let n = encoder.encode(&tone, 960, &mut packet).unwrap();

        for (options, deinterleaved) in [(0u32, false), (DEINTERLEAVED_OUTPUT, true)] {
            let input = ConnectInput {
                process: DECODE_OPUS,
                streams: 4,
                commands_per_stream: 4,
                memory: 0x4_0000,
                frames_per_output: 0,
                options,
            };
            let mut output = ConnectOutput {
                memory: std::ptr::null_mut(),
                result_event: std::ptr::null_mut(),
            };
            let mut handle = std::ptr::null_mut();
            assert_eq!(XApuConnect(&input, &mut output, &mut handle), S_OK);
            let base = output.memory.cast::<u8>();

            // Invalid channel counts fail cleanly without activating the stream.
            for bad_channels in [0u32, 3, u32::MAX] {
                let bad = Activate {
                    id: id(ACTIVATE, 0),
                    channels: bad_channels,
                    init_length: 0,
                    init: std::ptr::null(),
                    start_pitch: 1.0,
                    processing_length: 0,
                    processing: std::ptr::null_mut(),
                };
                assert_eq!(
                    XApuEnqueueCommand(handle, (&raw const bad).cast(), std::ptr::null_mut()),
                    S_OK
                );
                let mut res = std::mem::MaybeUninit::<ApuResult>::zeroed();
                assert_eq!(XApuDequeueResult(handle, res.as_mut_ptr()), S_OK);
                // SAFETY: written by `XApuDequeueResult`.
                let err = unsafe { res.assume_init() }.error;
                assert_eq!(err, E_INVALIDARG, "channels={bad_channels}");
            }

            // Activate two identical stereo decoders so aligned and unaligned outputs match bit-for-bit.
            for stream in [0u32, 1] {
                let activate = Activate {
                    id: id(ACTIVATE, stream),
                    channels: 2,
                    init_length: 0,
                    init: std::ptr::null(),
                    start_pitch: 1.0,
                    processing_length: 0,
                    processing: std::ptr::null_mut(),
                };
                assert_eq!(
                    XApuEnqueueCommand(handle, (&raw const activate).cast(), std::ptr::null_mut()),
                    S_OK
                );
                let mut res = std::mem::MaybeUninit::<ApuResult>::zeroed();
                assert_eq!(XApuDequeueResult(handle, res.as_mut_ptr()), S_OK);
                // SAFETY: written by `XApuDequeueResult`.
                let err = unsafe { res.assume_init() }.error;
                assert_eq!(err, S_OK);
            }

            // SAFETY: inside the connection memory.
            unsafe { std::ptr::copy_nonoverlapping(packet.as_ptr(), base, n) };
            let aligned_out = base.wrapping_add(0x10000);
            let unaligned_out = base.wrapping_add(0x20001); // 1 mod 4: misaligned for f32

            for (stream, out_ptr, frame_count) in [
                (0u32, aligned_out, 960u32),
                (1u32, unaligned_out, 960u32),
                (1u32, base.wrapping_add(0x30003), 0u32), // zero frames at misaligned ptr
            ] {
                let proc_cmd = Process {
                    id: id(PROCESS, stream),
                    pitch_ramp: 0.0,
                    target_pitch: 1.0,
                    first_frame: 0,
                    frame_count,
                    input_length: n as u32,
                    input: base,
                    output_length: 960 * 8,
                    output: out_ptr,
                };
                assert_eq!(
                    XApuEnqueueCommand(handle, (&raw const proc_cmd).cast(), std::ptr::null_mut()),
                    S_OK
                );
                let mut res = std::mem::MaybeUninit::<ApuResult>::zeroed();
                assert_eq!(XApuDequeueResult(handle, res.as_mut_ptr()), S_OK);
                // SAFETY: written by `XApuDequeueResult`.
                let decoded = unsafe { res.assume_init() };
                let (err, frames, len) = (decoded.error, decoded.frames, decoded.output_length);
                assert_eq!((err, frames, len), (S_OK, frame_count, frame_count * 8));
            }

            // SAFETY: both output regions are inside connection memory and hold 960 * 8 bytes.
            let (aligned_bytes, unaligned_bytes) = unsafe {
                (
                    std::slice::from_raw_parts(aligned_out, 960 * 8),
                    std::slice::from_raw_parts(unaligned_out, 960 * 8),
                )
            };
            assert_eq!(
                unaligned_bytes, aligned_bytes,
                "unaligned output must match aligned output (deinterleaved={deinterleaved})"
            );
            let samples: Vec<f32> = unaligned_bytes
                .as_chunks::<4>()
                .0
                .iter()
                .map(|c| f32::from_le_bytes(*c))
                .collect();
            let energy: f32 = samples.iter().map(|s| s * s).sum::<f32>() / samples.len() as f32;
            assert!(
                energy > 0.01,
                "expected decoded tone (deinterleaved={deinterleaved}), energy={energy}"
            );
            assert_eq!(XApuDisconnect(handle), S_OK);
        }
    }

    #[test]
    fn abi_layouts_are_the_console_ones() {
        assert_eq!(size_of::<CommandId>(), 24);
        assert_eq!(size_of::<ConnectInput>(), 24);
        assert_eq!(size_of::<ConnectOutput>(), 16);
        assert_eq!(size_of::<Activate>(), 56);
        assert_eq!(size_of::<Process>(), 64);
        assert_eq!(size_of::<ApuResult>(), 0x50);
    }

    fn valid_connect() -> ConnectInput {
        ConnectInput {
            process: DECODE_OPUS,
            streams: 4,
            commands_per_stream: 2,
            memory: MEMORY,
            frames_per_output: 0,
            options: 0,
        }
    }

    #[test]
    fn connect_rejects_malformed_arguments() {
        fn rejected(input: ConnectInput) -> HRESULT {
            let mut output = ConnectOutput {
                memory: std::ptr::null_mut(),
                result_event: std::ptr::null_mut(),
            };
            let mut handle = std::ptr::null_mut();
            let result = XApuConnect(&input, &mut output, &mut handle);
            assert_ne!(result, S_OK);
            assert!(handle.is_null(), "a refused connect hands out no handle");
            assert!(
                output.memory.is_null(),
                "a refused connect hands out no memory"
            );
            result
        }

        let input = valid_connect();
        let mut output = ConnectOutput {
            memory: std::ptr::null_mut(),
            result_event: std::ptr::null_mut(),
        };
        let mut handle = std::ptr::null_mut();
        assert_eq!(
            XApuConnect(std::ptr::null(), &mut output, &mut handle),
            E_INVALIDARG
        );
        assert_eq!(
            XApuConnect(&input, std::ptr::null_mut(), &mut handle),
            E_INVALIDARG
        );
        assert_eq!(
            XApuConnect(&input, &mut output, std::ptr::null_mut()),
            E_INVALIDARG
        );

        assert_eq!(
            rejected(ConnectInput {
                streams: 0,
                ..valid_connect()
            }),
            E_INVALIDARG
        );
        assert_eq!(
            rejected(ConnectInput {
                streams: 501,
                ..valid_connect()
            }),
            E_INVALIDARG
        );
        assert_eq!(
            rejected(ConnectInput {
                commands_per_stream: 0,
                ..valid_connect()
            }),
            E_INVALIDARG
        );
        assert_eq!(
            rejected(ConnectInput {
                commands_per_stream: 17,
                ..valid_connect()
            }),
            E_INVALIDARG
        );
        assert_eq!(
            rejected(ConnectInput {
                memory: 0,
                ..valid_connect()
            }),
            E_INVALIDARG
        );
        assert_eq!(
            rejected(ConnectInput {
                frames_per_output: 1,
                ..valid_connect()
            }),
            E_INVALIDARG
        );
        // Only plain decoding exists; anything else is refused, not guessed at.
        assert_eq!(
            rejected(ConnectInput {
                process: 2,
                ..valid_connect()
            }),
            E_NOTIMPL
        );

        // The ends of both accepted ranges connect.
        for (streams, per_stream) in [(1, 1), (500, 16)] {
            let fixture = Fixture::new(streams, per_stream, 0);
            assert_eq!(fixture.count(), 0);
            let notify = id(NOTIFY, 0);
            assert_eq!(fixture.enqueue(std::ptr::from_ref(&notify).cast()), S_OK);
        }
    }

    #[test]
    fn connect_zeroes_its_memory_and_only_creates_an_event_when_asked() {
        let fixture = Fixture::new(2, 1, 0);
        assert_eq!(
            fixture.base as usize % 16,
            0,
            "the memory is 16-byte aligned"
        );
        assert!(fixture.event.is_null(), "no event was requested");
        // SAFETY: the allocator handed out MEMORY zeroed bytes at this address.
        let memory = unsafe { std::slice::from_raw_parts(fixture.base, MEMORY as usize) };
        assert!(
            memory.iter().all(|&byte| byte == 0),
            "the title gets zeroed memory"
        );

        let with_event = Fixture::new(1, 1, RESULT_EVENT);
        assert!(!with_event.event.is_null(), "the requested event exists");
    }

    #[test]
    fn activation_checks_the_channel_count_and_refuses_a_second_activation() {
        let fixture = Fixture::new(4, 4, 0);
        // A channel count no decoder can be built for fails in the result, not
        // in the call: the command was queued, so it owes a result.
        for channels in [0, 3, u32::MAX] {
            let command = activate_command(0, channels);
            assert_eq!(fixture.enqueue(std::ptr::from_ref(&command).cast()), S_OK);
            let result = fixture.dequeue();
            assert_eq!(outcome(&result), (E_INVALIDARG, 0, 0));
        }
        // A refused activation installs no decoder, so the stream stays idle.
        let packet = tone_packet(1, 960, 440.0);
        let command = process_command(0, &packet, fixture.base);
        assert_eq!(fixture.enqueue(std::ptr::from_ref(&command).cast()), S_OK);
        assert_eq!(outcome(&fixture.dequeue()).0, INVALID_STREAM);

        for (stream, channels) in [(1, 1), (2, 2)] {
            activate_stream(&fixture, stream, channels);
            let again = activate_command(stream, channels);
            assert_eq!(fixture.enqueue(std::ptr::from_ref(&again).cast()), S_OK);
            assert_eq!(outcome(&fixture.dequeue()).0, ALREADY_ACTIVE);
        }
    }

    #[test]
    fn decode_writes_the_frames_a_fresh_decoder_would_produce() {
        let fixture = Fixture::new(2, 2, 0);
        let packet = tone_packet(2, 960, 440.0);
        let reference = fresh_pcm(&packet, 2, 960);
        // Two streams that only ever saw this packet agree, so decoding is
        // deterministic and no stream borrows another's history.
        for stream in 0..2u32 {
            let result = decode(&fixture, stream, 2, &packet, |_| {});
            assert_eq!(outcome(&result), (S_OK, 960, 960 * 8));
            assert_eq!(fixture.output(1920), reference.as_slice());
        }
    }

    #[test]
    fn process_clamps_to_the_requested_frames_and_the_output_room() {
        let fixture = Fixture::new(5, 4, 0);
        let packet = tone_packet(2, 960, 440.0);
        let reference = fresh_pcm(&packet, 2, 960);

        let result = decode(&fixture, 0, 2, &packet, |_| {});
        assert_eq!(outcome(&result), (S_OK, 960, 960 * 8));
        assert_eq!(fixture.output(1920), reference.as_slice());

        // `frame_count` stops the copy short.
        let result = decode(&fixture, 1, 2, &packet, |command| command.frame_count = 100);
        assert_eq!(outcome(&result), (S_OK, 100, 100 * 8));
        assert_eq!(fixture.output(200), &reference[..200]);

        // So does the room the title left in its output buffer.
        let result = decode(&fixture, 2, 2, &packet, |command| {
            command.output_length = 100 * 8;
        });
        assert_eq!(outcome(&result), (S_OK, 100, 100 * 8));

        // `first_frame` drops the head of the packet instead.
        let result = decode(&fixture, 3, 2, &packet, |command| command.first_frame = 480);
        assert_eq!(outcome(&result), (S_OK, 480, 480 * 8));
        assert_eq!(fixture.output(960), &reference[960..]);

        // Past the end of the packet there is nothing left to keep.
        let result = decode(&fixture, 4, 2, &packet, |command| {
            command.first_frame = 5000
        });
        assert_eq!(outcome(&result), (S_OK, 0, 0));
    }

    #[test]
    fn deinterleaved_output_is_left_plane_then_right_plane() {
        let fixture = Fixture::new(1, 2, DEINTERLEAVED_OUTPUT);
        let packet = tone_packet(2, 960, 440.0);
        let reference = fresh_pcm(&packet, 2, 960);
        let left: Vec<f32> = reference.iter().step_by(2).copied().collect();
        let right: Vec<f32> = reference.iter().skip(1).step_by(2).copied().collect();
        assert_ne!(left, right, "the packet carries two different channels");

        let result = decode(&fixture, 0, 2, &packet, |_| {});
        assert_eq!(outcome(&result), (S_OK, 960, 960 * 8));
        let planar = fixture.output(1920);
        assert_eq!(&planar[..960], left.as_slice());
        assert_eq!(&planar[960..], right.as_slice());
    }

    #[test]
    fn a_mono_stream_produces_one_channel() {
        let fixture = Fixture::new(2, 2, DEINTERLEAVED_OUTPUT);
        let packet = tone_packet(1, 960, 440.0);
        let reference = fresh_pcm(&packet, 1, 960);
        let result = decode(&fixture, 0, 1, &packet, |_| {});
        assert_eq!(outcome(&result), (S_OK, 960, 960 * 4));
        // Planar output of a single channel is just the interleaved stream.
        assert_eq!(fixture.output(960), reference.as_slice());

        // A stereo packet through a mono stream is mixed down, not refused.
        let stereo = tone_packet(2, 960, 440.0);
        let downmixed = fresh_pcm(&stereo, 1, 960);
        let result = decode(&fixture, 1, 1, &stereo, |_| {});
        assert_eq!(outcome(&result), (S_OK, 960, 960 * 4));
        assert_eq!(fixture.output(960), downmixed.as_slice());
    }

    #[test]
    fn process_refuses_missing_buffers() {
        let fixture = Fixture::new(1, 4, 0);
        let packet = tone_packet(2, 960, 440.0);
        activate_stream(&fixture, 0, 2);

        let mut command = process_command(0, &packet, fixture.base);
        command.input = std::ptr::null();
        assert_eq!(fixture.enqueue(std::ptr::from_ref(&command).cast()), S_OK);
        let result = fixture.dequeue();
        assert_eq!(outcome(&result).0, E_INVALIDARG);
        let input = result.input;
        assert!(input.is_null(), "the result echoes the command's input");

        let mut command = process_command(0, &packet, fixture.base);
        command.output = std::ptr::null_mut();
        assert_eq!(fixture.enqueue(std::ptr::from_ref(&command).cast()), S_OK);
        assert_eq!(outcome(&fixture.dequeue()).0, E_INVALIDARG);

        let mut command = process_command(0, &packet, fixture.base);
        command.input_length = 0;
        assert_eq!(fixture.enqueue(std::ptr::from_ref(&command).cast()), S_OK);
        assert_eq!(outcome(&fixture.dequeue()).0, E_INVALIDARG);
    }

    #[test]
    fn process_reports_a_broken_packet_without_writing_frames() {
        let fixture = Fixture::new(1, 4, 0);
        activate_stream(&fixture, 0, 2);
        // Code 3 counting more frames than a packet may hold (48 is the cap),
        // and a SILK payload that does not decode.
        for broken in [vec![0xFB], vec![0xFF; 48], vec![0x02, 0x02, 0x02]] {
            let command = process_command(0, &broken, fixture.base);
            assert_eq!(fixture.enqueue(std::ptr::from_ref(&command).cast()), S_OK);
            let result = fixture.dequeue();
            assert_eq!(outcome(&result), (BAD_PACKET, 0, 0));
            let length = result.input_length;
            assert_eq!(length, broken.len() as u32);
        }
        // A packet that did not decode leaves the title's buffer alone.
        assert!(fixture.output(1920).iter().all(|&sample| sample == 0.0));
    }

    #[test]
    fn reset_puts_the_decoder_back_to_its_initial_state() {
        let fixture = Fixture::new(1, 6, 0);
        let packet = tone_packet(2, 960, 440.0);
        let reference = fresh_pcm(&packet, 2, 960);
        activate_stream(&fixture, 0, 2);

        let result = process_packet(&fixture, 0, &packet, |_| {});
        assert_eq!(outcome(&result), (S_OK, 960, 960 * 8));
        assert_eq!(fixture.output(1920), reference.as_slice());

        // A decoder that already played the packet carries its history into the
        // next one, so the same bytes do not come out again.
        let warm = process_packet(&fixture, 0, &packet, |_| {});
        assert_eq!(outcome(&warm), (S_OK, 960, 960 * 8));
        assert_ne!(
            fixture.output(1920),
            reference.as_slice(),
            "history changes what the same packet decodes to"
        );

        let reset = id(RESET, 0);
        assert_eq!(fixture.enqueue(std::ptr::from_ref(&reset).cast()), S_OK);
        assert_eq!(outcome(&fixture.dequeue()).0, S_OK);

        let result = process_packet(&fixture, 0, &packet, |_| {});
        assert_eq!(outcome(&result), (S_OK, 960, 960 * 8));
        assert_eq!(
            fixture.output(1920),
            reference.as_slice(),
            "reset cleared the history"
        );
    }

    #[test]
    fn every_command_answers_with_exactly_one_result_in_order() {
        let fixture = Fixture::new(2, 8, 0);
        let packet = tone_packet(2, 960, 440.0);
        let activation = activate_command(0, 2);
        let process = process_command(0, &packet, fixture.base);
        let reset = id(RESET, 0);
        let deactivate = id(DEACTIVATE, 0);
        let notify = id(NOTIFY, 0);
        let idle = process_command(1, &packet, fixture.base);
        // Every command starts with its id, so each goes in by its own pointer.
        assert_eq!(
            fixture.enqueue(std::ptr::from_ref(&activation).cast()),
            S_OK
        );
        assert_eq!(fixture.enqueue(std::ptr::from_ref(&process).cast()), S_OK);
        assert_eq!(fixture.enqueue(std::ptr::from_ref(&reset).cast()), S_OK);
        assert_eq!(fixture.enqueue(std::ptr::from_ref(&process).cast()), S_OK);
        assert_eq!(
            fixture.enqueue(std::ptr::from_ref(&deactivate).cast()),
            S_OK
        );
        assert_eq!(fixture.enqueue(std::ptr::from_ref(&process).cast()), S_OK);
        assert_eq!(fixture.enqueue(std::ptr::from_ref(&notify).cast()), S_OK);
        assert_eq!(fixture.enqueue(std::ptr::from_ref(&idle).cast()), S_OK);
        assert_eq!(fixture.count(), 8);

        let expect = |kind_wanted: u32, error: HRESULT| {
            let result = fixture.dequeue();
            assert_eq!(kind(&result), kind_wanted);
            assert_eq!(outcome(&result).0, error);
        };
        expect(ACTIVATE, S_OK);
        expect(PROCESS, S_OK);
        expect(RESET, S_OK);
        expect(PROCESS, S_OK);
        expect(DEACTIVATE, S_OK);
        expect(PROCESS, INVALID_STREAM);
        expect(NOTIFY, S_OK);
        expect(PROCESS, INVALID_STREAM);
        assert_eq!(fixture.count(), 0);
    }

    #[test]
    fn queues_are_per_stream_and_hold_the_connection_allowance() {
        let fixture = Fixture::new(3, 2, 0);
        let mut first = id(NOTIFY, 0);
        let mut second = id(NOTIFY, 0);
        second.sequence = 8;
        let other = id(NOTIFY, 1);
        assert_eq!(fixture.enqueue(std::ptr::from_ref(&first).cast()), S_OK);
        assert_eq!(fixture.enqueue(std::ptr::from_ref(&second).cast()), S_OK);
        // Two commands per stream is this connection's whole allowance.
        assert_eq!(
            fixture.enqueue(std::ptr::from_ref(&first).cast()),
            QUEUE_FULL
        );
        // Every stream has an allowance of its own.
        assert_eq!(fixture.enqueue(std::ptr::from_ref(&other).cast()), S_OK);
        assert_eq!(fixture.count(), 3);

        // Results come back in the order the commands went in.
        assert_eq!(sequence(&fixture.dequeue()), 7);
        assert_eq!(sequence(&fixture.dequeue()), 8);
        // Dequeueing frees the slot its stream was holding.
        first.sequence = 9;
        assert_eq!(fixture.enqueue(std::ptr::from_ref(&first).cast()), S_OK);
        assert_eq!(stream(&fixture.dequeue()), 1);
        assert_eq!(sequence(&fixture.dequeue()), 9);

        let mut total = 0;
        assert_eq!(XApuGetResultCount(fixture.handle, &mut total), 0);
        assert_eq!(total, 4, "the total counts every result taken");
    }

    #[test]
    fn rejected_commands_leave_no_result_behind() {
        let fixture = Fixture::new(2, 1, 0);
        let mut total = 0;
        let unknown = id(99, 0);
        assert_eq!(
            XApuEnqueueCommand(fixture.handle, std::ptr::from_ref(&unknown), &mut total),
            INVALID_COMMAND
        );
        let missing = id(NOTIFY, 2);
        assert_eq!(
            XApuEnqueueCommand(fixture.handle, std::ptr::from_ref(&missing), &mut total),
            INVALID_STREAM
        );
        assert_eq!(fixture.count(), 0);
        assert_eq!(total, 0, "a rejected command is not counted");

        // Neither refusal used up the stream's one slot.
        let notify = id(NOTIFY, 0);
        assert_eq!(fixture.enqueue(std::ptr::from_ref(&notify).cast()), S_OK);
        assert_eq!(kind(&fixture.dequeue()), NOTIFY);
    }

    #[test]
    fn results_echo_the_command_they_answer() {
        let fixture = Fixture::new(1, 1, 0);
        let command = CommandId {
            kind: NOTIFY,
            stream: 0,
            sequence: 0xABCD,
            options: 0,
            context: 0x1122_3344_5566_7788,
        };
        assert_eq!(fixture.enqueue(std::ptr::from_ref(&command).cast()), S_OK);
        let result = fixture.dequeue();
        assert_eq!(kind(&result), NOTIFY);
        assert_eq!(stream(&result), 0);
        assert_eq!(sequence(&result), 0xABCD);
        let context = result.id.context;
        assert_eq!(context, 0x1122_3344_5566_7788);
        assert_eq!(outcome(&result), (S_OK, 0, 0));
        let (pitch, input, output) = (result.pitch, result.input, result.output);
        assert_eq!(pitch, 1.0);
        assert!(input.is_null() && output.is_null());
    }

    #[test]
    fn dequeue_reports_pending_until_a_result_exists() {
        let fixture = Fixture::new(1, 1, 0);
        let mut result = std::mem::MaybeUninit::<ApuResult>::zeroed();
        assert_eq!(
            XApuDequeueResult(fixture.handle, result.as_mut_ptr()),
            PENDING_RESULTS
        );
        let mut total = 0;
        assert_eq!(XApuGetResultCount(fixture.handle, &mut total), 0);
        assert_eq!(total, 0);
    }

    #[test]
    fn null_arguments_are_refused() {
        let fixture = Fixture::new(1, 1, 0);
        let command = id(NOTIFY, 0);
        assert_eq!(XApuDisconnect(std::ptr::null_mut()), E_INVALIDARG);
        assert_eq!(
            XApuEnqueueCommand(
                std::ptr::null_mut(),
                std::ptr::from_ref(&command),
                std::ptr::null_mut()
            ),
            E_INVALIDARG
        );
        assert_eq!(
            XApuEnqueueCommand(fixture.handle, std::ptr::null(), std::ptr::null_mut()),
            E_INVALIDARG
        );
        let mut result = std::mem::MaybeUninit::<ApuResult>::zeroed();
        assert_eq!(
            XApuDequeueResult(std::ptr::null_mut(), result.as_mut_ptr()),
            E_INVALIDARG
        );
        assert_eq!(
            XApuDequeueResult(fixture.handle, std::ptr::null_mut()),
            E_INVALIDARG
        );
        assert_eq!(
            XApuGetResultCount(std::ptr::null_mut(), std::ptr::null_mut()),
            0
        );
        // The refusals left the connection untouched.
        assert_eq!(fixture.count(), 0);
    }

    #[test]
    fn the_result_event_fires_only_for_commands_that_ask() {
        let fixture = Fixture::new(1, 2, RESULT_EVENT);
        // SAFETY: our own event handle, and a zero timeout does not wait.
        let state = || unsafe { WaitForSingleObject(fixture.event, 0) };
        assert_eq!(state(), WAIT_TIMEOUT, "a new event starts unsignalled");

        let quiet = id(NOTIFY, 0);
        assert_eq!(fixture.enqueue(std::ptr::from_ref(&quiet).cast()), S_OK);
        assert_eq!(state(), WAIT_TIMEOUT, "no signal was asked for");

        let mut loud = id(NOTIFY, 0);
        loud.options = SIGNAL;
        assert_eq!(fixture.enqueue(std::ptr::from_ref(&loud).cast()), S_OK);
        assert_eq!(state(), WAIT_OBJECT_0, "the command asked to signal");

        assert_eq!(state(), WAIT_TIMEOUT, "the event resets once observed");
    }
}
