//! Presented frames written to PNG files, to see what reaches the screen:
//! with `VICTORY_CAPTURE=<folder>`, every frame whose number is a power of
//! two (the frames the log reports) is saved as `frame-<n>.png`, at half
//! size. Only in builds with diagnostics.

use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering::Relaxed};

use windows::Win32::Graphics::Direct3D12::*;
use windows::Win32::Graphics::Dxgi::Common::*;

/// The folder frames go to, if capturing.
pub(crate) fn folder() -> Option<&'static PathBuf> {
    static FOLDER: std::sync::OnceLock<Option<PathBuf>> = std::sync::OnceLock::new();
    FOLDER
        .get_or_init(|| std::env::var_os("VICTORY_CAPTURE").map(PathBuf::from))
        .as_ref()
}

/// Frames listed in `VICTORY_CAPTURE_FRAMES` (comma-separated): saved
/// too, and their draws traced at debug level. `VICTORY_CAPTURE_WORLD`
/// lists frames counted from the first one that ran 500 draws (the world's
/// first, whose number depends on how fast the title loaded).
pub(crate) fn listed(n: u64) -> bool {
    static LIST: std::sync::OnceLock<Vec<u64>> = std::sync::OnceLock::new();
    LIST.get_or_init(|| numbers("VICTORY_CAPTURE_FRAMES")).contains(&n)
        || WORLD
            .get()
            .is_some_and(|&w| world_offsets().iter().any(|&k| w + k == n))
}

fn numbers(var: &str) -> Vec<u64> {
    let list = std::env::var(var).unwrap_or_default();
    list.split(',')
        .filter_map(|f| f.trim().parse().ok())
        .collect()
}

fn world_offsets() -> &'static [u64] {
    static OFFSETS: std::sync::OnceLock<Vec<u64>> = std::sync::OnceLock::new();
    OFFSETS.get_or_init(|| numbers("VICTORY_CAPTURE_WORLD"))
}

/// The first frame that ran 500 draws on the host.
static WORLD: std::sync::OnceLock<u64> = std::sync::OnceLock::new();

/// Frame `n` was presented: notes the world's first frame.
pub(crate) fn presented(n: u64) {
    static LAST: AtomicU64 = AtomicU64::new(0);
    let draws = draws_since_last_present(&LAST, crate::draw::drawn());
    if draws >= 500 && WORLD.get().is_none() && !world_offsets().is_empty() {
        let _ = WORLD.set(n);
        tracing::info!(frame = n, draws, "the world's first frame (VICTORY_CAPTURE_WORLD counts from it)");
    }
}

fn draws_since_last_present(last: &AtomicU64, drawn: u64) -> u64 {
    drawn.saturating_sub(last.fetch_max(drawn, Relaxed))
}

#[test]
fn stale_draw_samples_do_not_trigger_world_capture_early() {
    let last = AtomicU64::new(0);
    assert_eq!(draws_since_last_present(&last, 480), 480);
    assert_eq!(draws_since_last_present(&last, 600), 120);
    assert_eq!(draws_since_last_present(&last, 480), 0);
    assert_eq!(draws_since_last_present(&last, 995), 395);
    assert_eq!(draws_since_last_present(&last, 1495), 500);
}

/// Whether frame `n` is saved.
pub(crate) fn wanted(n: u64) -> bool {
    folder().is_some() && (n.is_power_of_two() || listed(n))
}

/// Saves 8-bit RGBA or BGRA rows (`pitch` bytes apart) as frame `n`.
pub(crate) fn save(n: u64, width: u32, height: u32, pitch: usize, bgra: bool, rows: &[u8]) {
    save_as(&format!("frame-{n}"), width, height, pitch, bgra, rows);
}

/// Saves 8-bit RGBA or BGRA rows as `<name>.png`, at half size.
fn save_as(name: &str, width: u32, height: u32, pitch: usize, bgra: bool, rows: &[u8]) {
    let Some(folder) = folder() else { return };
    // Keep the last sample when the input width or height is odd.
    let (w, h) = (width.div_ceil(2).max(1), height.div_ceil(2).max(1));
    // Filter byte 0 per row, then RGB of every other pixel of every other row.
    let mut raw = Vec::with_capacity((w as usize * 3 + 1) * h as usize);
    for y in 0..h as usize {
        raw.push(0);
        let row = &rows[(2 * y).min(height as usize - 1) * pitch..];
        for x in 0..w as usize {
            let p = &row[(2 * x).min(width as usize - 1) * 4..][..3];
            if bgra {
                raw.extend([p[2], p[1], p[0]]);
            } else {
                raw.extend_from_slice(p);
            }
        }
    }
    let mut png = b"\x89PNG\r\n\x1a\n".to_vec();
    let mut header = Vec::new();
    header.extend(w.to_be_bytes());
    header.extend(h.to_be_bytes());
    header.extend([8, 2, 0, 0, 0]); // 8-bit RGB
    chunk(&mut png, b"IHDR", &header);
    chunk(
        &mut png,
        b"IDAT",
        &miniz_oxide::deflate::compress_to_vec_zlib(&raw, 3),
    );
    chunk(&mut png, b"IEND", &[]);
    let path = folder.join(format!("{name}.png"));
    match std::fs::create_dir_all(folder).and_then(|()| std::fs::write(&path, png)) {
        Ok(()) => tracing::info!(path = %path.display(), "frame saved"),
        Err(e) => tracing::warn!(error = %e, path = %path.display(), "frame not saved"),
    }
}

#[cfg(test)]
#[test]
fn odd_capture_extents_keep_the_final_sampled_row_and_column() {
    const TEST: &str = "capture::odd_capture_extents_keep_the_final_sampled_row_and_column";
    const CHILD: &str = "VICTORY_TEST_ODD_CAPTURE";
    if std::env::var_os(CHILD).is_some() {
        let mut rows = [255u8; 5 * 16];
        for y in 0..5u8 {
            for x in 0..3u8 {
                let at = usize::from(y) * 16 + usize::from(x) * 4;
                rows[at..at + 4].copy_from_slice(&[100 + y, 50 + x, y * 10 + x, 200]);
            }
        }
        save_as("odd", 3, 5, 16, true, &rows);
        return;
    }

    let unique = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    let dir = std::env::temp_dir()
        .join(format!("victory-odd-capture-{}-{unique}", std::process::id()));
    let output = std::process::Command::new(std::env::current_exe().unwrap())
        .args(["--exact", TEST])
        .env(CHILD, "1")
        .env("VICTORY_CAPTURE", &dir)
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "{}\n{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr),
    );
    let png = std::fs::read(dir.join("odd.png")).unwrap();
    std::fs::remove_dir_all(&dir).unwrap();

    let word = |at| u32::from_be_bytes(png[at..at + 4].try_into().unwrap());
    assert_eq!(&png[12..16], b"IHDR");
    assert_eq!([word(16), word(20)], [2, 3]);
    assert_eq!(&png[37..41], b"IDAT");
    let pixels = miniz_oxide::inflate::decompress_to_vec_zlib(&png[41..41 + word(33) as usize]).unwrap();
    let expected = [
        0, 0, 50, 100, 2, 52, 100,
        0, 20, 50, 102, 22, 52, 102,
        0, 40, 50, 104, 42, 52, 104,
    ];
    assert_eq!(pixels, expected);
}

fn chunk(png: &mut Vec<u8>, kind: &[u8; 4], data: &[u8]) {
    png.extend((data.len() as u32).to_be_bytes());
    let start = png.len();
    png.extend(kind);
    png.extend(data);
    let crc = crc32(&png[start..]);
    png.extend(crc.to_be_bytes());
}

fn crc32(bytes: &[u8]) -> u32 {
    let mut crc = !0u32;
    for &b in bytes {
        crc ^= u32::from(b);
        for _ in 0..8 {
            crc = (crc >> 1) ^ (0xedb8_8320 & (crc & 1).wrapping_neg());
        }
    }
    !crc
}

/// Records a copy of `texture` (in the copy-source state) into a
/// readback buffer.
pub(crate) fn read_back(
    device: &ID3D12Device,
    list: &ID3D12GraphicsCommandList,
    texture: &ID3D12Resource,
) -> Option<(ID3D12Resource, D3D12_PLACED_SUBRESOURCE_FOOTPRINT)> {
    let mut footprint = D3D12_PLACED_SUBRESOURCE_FOOTPRINT::default();
    let mut total = 0;
    // SAFETY: a live texture; outputs for one subresource.
    unsafe {
        let desc = texture.GetDesc();
        device.GetCopyableFootprints(
            &desc,
            0,
            1,
            0,
            Some(&mut footprint),
            None,
            None,
            Some(&mut total),
        );
    }
    let heap = D3D12_HEAP_PROPERTIES {
        Type: D3D12_HEAP_TYPE_READBACK,
        ..Default::default()
    };
    let desc = D3D12_RESOURCE_DESC {
        Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
        Width: total,
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
    // SAFETY: plain creation; an open list; locations naming live resources.
    unsafe {
        device
            .CreateCommittedResource(
                &heap,
                D3D12_HEAP_FLAG_NONE,
                &desc,
                D3D12_RESOURCE_STATE_COPY_DEST,
                None,
                &mut buffer,
            )
            .ok()?;
        let buffer = buffer?;
        let to = D3D12_TEXTURE_COPY_LOCATION {
            pResource: std::mem::transmute_copy(&buffer),
            Type: D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT,
            Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 {
                PlacedFootprint: footprint,
            },
        };
        let from = D3D12_TEXTURE_COPY_LOCATION {
            pResource: std::mem::transmute_copy(texture),
            Type: D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX,
            Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 {
                SubresourceIndex: 0,
            },
        };
        list.CopyTextureRegion(&to, 0, 0, 0, &from, None);
        Some((buffer, footprint))
    }
}

/// Saves a frame read back by [`read_back`] (8-bit color only).
pub(crate) fn save_read_back(
    frame: u64,
    buffer: &ID3D12Resource,
    footprint: &D3D12_PLACED_SUBRESOURCE_FOOTPRINT,
) {
    let f = footprint.Footprint;
    let bgra = match f.Format {
        DXGI_FORMAT_R8G8B8A8_UNORM | DXGI_FORMAT_R8G8B8A8_UNORM_SRGB => false,
        DXGI_FORMAT_B8G8R8A8_UNORM | DXGI_FORMAT_B8G8R8A8_UNORM_SRGB => true,
        other => {
            tracing::warn!(format = other.0, "frame not saved: not 8-bit color");
            return;
        }
    };
    let mut data = std::ptr::null_mut();
    // SAFETY: the copy finished (every submission is waited for); the
    // mapping spans the footprint's rows.
    unsafe {
        if buffer.Map(0, None, Some(&mut data)).is_err() {
            return;
        }
        let len = buffer.GetDesc().Width as usize;
        let rows = std::slice::from_raw_parts(data.cast::<u8>(), len);
        save(frame, f.Width, f.Height, f.RowPitch as usize, bgra, rows);
        buffer.Unmap(0, None);
    }
}

/// With `VICTORY_DUMP_TARGETS` set: title textures the frames listed in
/// `VICTORY_CAPTURE_FRAMES` render to or write, in first-use order, saved
/// at that frame's present as `frame-<n>-<i>-<resource>-<w>x<h>-f<format>.png`
/// (what each holds at the end of the frame; formats normalized to show
/// structure: floats scaled by their largest value, depth by its range).
/// The host copies are held: the title may release a target before the
/// frame is presented.
/// Each with the first pipeline that wrote it as an unordered-access view
/// (`-by-<pipeline>` in the name; `-black` when its colour is all zeros).
/// `VICTORY_DUMP_WRITERS` (shader hashes) adds, in any frame, the textures
/// a pipeline with one of them writes, the first frame it runs.
#[allow(clippy::type_complexity)]
static TARGETS: std::sync::Mutex<
    Vec<(usize, std::sync::Arc<crate::resource::HostTexture>, String)>,
> = std::sync::Mutex::new(Vec::new());

thread_local! {
    /// The pipeline whose draw or dispatch this thread is binding.
    static WRITER: std::cell::RefCell<String> = const { std::cell::RefCell::new(String::new()) };
}

/// Names the pipeline whose views this thread binds next (empty: none).
pub(crate) fn binding(pipeline: &str) {
    if dump_on() {
        WRITER.with(|w| pipeline.clone_into(&mut w.borrow_mut()));
    }
}

pub(crate) fn dump_on() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| folder().is_some() && crate::switch::on("VICTORY_DUMP_TARGETS"))
}

/// `VICTORY_DUMP_WRITERS`' hashes not dumped yet.
static WRITERS: std::sync::Mutex<Option<Vec<String>>> = std::sync::Mutex::new(None);

/// Buffers such a pipeline writes: title memory, its bytes when first
/// bound in the frame, the pipeline. Logged at the present: dwords changed
/// and dwords not zero.
#[allow(clippy::type_complexity)]
static BUFFERS: std::sync::Mutex<Vec<(usize, Vec<u8>, String)>> = std::sync::Mutex::new(Vec::new());

/// Notes a title texture the frame being built renders to or writes (by a
/// command of `list`, whose submission dumps it again when the frame is
/// listed).
pub(crate) fn note_target(list: &crate::host::HostList, resource: crate::object::This) {
    if resource.is_null() || !dump_on() {
        return;
    }
    let by = WRITER.with(|w| w.borrow().clone());
    // Lists are recorded ahead of the frame that submits them.
    list.note_target(resource as usize, &by);
    if !listed(crate::queue::presented() + 1) {
        let mut writers = WRITERS.lock().unwrap_or_else(|e| e.into_inner());
        let writers = writers.get_or_insert_with(|| {
            let list = std::env::var("VICTORY_DUMP_WRITERS").unwrap_or_default();
            list.split(',').map(|h| h.trim().to_owned()).filter(|h| !h.is_empty()).collect()
        });
        if by.is_empty() || !writers.iter().any(|h| by.contains(h.as_str())) {
            return;
        }
        if let Some((at, len)) = crate::resource::buffer_bytes(resource) {
            let mut buffers = BUFFERS.lock().unwrap_or_else(|e| e.into_inner());
            if !buffers.iter().any(|b| b.0 == at as usize) {
                let mut before = vec![0u8; (len as usize).min(64 << 20) & !3];
                if crate::host::read_into(at, &mut before) {
                    buffers.push((at as usize, before, by));
                }
            }
            return;
        }
    }
    let mut targets = TARGETS.lock().unwrap_or_else(|e| e.into_inner());
    if !targets.iter().any(|(r, _, _)| *r == resource as usize)
        && let Some(copy) = crate::resource::host_copy(resource)
    {
        targets.push((resource as usize, copy, by));
    }
}

/// Whether textures wait to be dumped at this present.
pub(crate) fn has_targets() -> bool {
    !TARGETS.lock().unwrap_or_else(|e| e.into_inner()).is_empty()
        || !BUFFERS.lock().unwrap_or_else(|e| e.into_inner()).is_empty()
}

/// At the present of frame `frame`: saves the noted targets.
pub(crate) fn dump_targets(frame: u64) {
    let targets = std::mem::take(&mut *TARGETS.lock().unwrap_or_else(|e| e.into_inner()));
    let buffers = std::mem::take(&mut *BUFFERS.lock().unwrap_or_else(|e| e.into_inner()));
    // Writers dumped once.
    if let Some(writers) = WRITERS.lock().unwrap_or_else(|e| e.into_inner()).as_mut() {
        writers.retain(|h| {
            !targets.iter().any(|t| t.2.contains(h.as_str()))
                && !buffers.iter().any(|b| b.2.contains(h.as_str()))
        });
    }
    for (at, before, by) in &buffers {
        let mut after = vec![0u8; before.len()];
        if !crate::host::read_into(*at as *const u8, &mut after) {
            continue;
        }
        let dwords = |b: &[u8]| -> Vec<u32> {
            b.chunks(4).map(|c| u32::from_le_bytes(c.try_into().unwrap_or_default())).collect()
        };
        let (old, new) = (dwords(before), dwords(&after));
        let changed = old.iter().zip(&new).filter(|(a, b)| a != b).count();
        let set = new.iter().filter(|&&x| x != 0).count();
        tracing::info!(
            frame,
            buffer = format_args!("{at:#x}"),
            dwords = new.len(),
            changed,
            set,
            by = by.as_str(),
            "a buffer the named pipeline writes, at the present"
        );
    }
    let Some(host) = crate::host::host() else { return };
    for (i, (resource, copy, by)) in targets.iter().enumerate() {
        let texture = &copy.texture;
        // SAFETY: a live resource.
        let d = unsafe { texture.GetDesc() };
        if d.SampleDesc.Count > 1 || d.Dimension == D3D12_RESOURCE_DIMENSION_BUFFER {
            continue;
        }
        let target = (i, *resource as crate::object::This, by.as_str());
        dump_one(host, copy, &format!("frame-{frame}"), target, &d);
    }
}

/// After submission `sub` of a listed frame has run: the textures its lists
/// rendered to or wrote, as `frame-<n>-s<sub>-<i>-...` (what each holds
/// then, to follow a frame pass by pass).
pub(crate) fn dump_submission(sub: u64, targets: &[(usize, String)]) {
    let Some(host) = crate::host::host() else { return };
    let frame = crate::queue::presented() + 1;
    peek(&format!("frame-{frame}-s{sub:04}"));
    for (i, (resource, by)) in targets.iter().enumerate() {
        let Some(copy) = crate::resource::host_copy(*resource as crate::object::This) else {
            continue;
        };
        let texture = &copy.texture;
        // SAFETY: a live resource.
        let d = unsafe { texture.GetDesc() };
        if d.SampleDesc.Count > 1 || d.Dimension == D3D12_RESOURCE_DIMENSION_BUFFER {
            continue;
        }
        let target = (i, *resource as crate::object::This, by.as_str());
        dump_one(host, &copy, &format!("frame-{frame}-s{sub:04}"), target, &d);
        // A depth texture's writable twin (`crate::twin`): what unordered
        // views wrote, which reaches the depth texture at its next use.
        for (p, plane) in copy.twin.get().and_then(Option::as_ref).iter().flat_map(|t| t.planes.iter().enumerate()) {
            // SAFETY: a live resource.
            let pd = unsafe { plane.texture.GetDesc() };
            let by = format!("{by}-twin{p}");
            let target = (i, *resource as crate::object::This, by.as_str());
            dump_one(host, plane, &format!("frame-{frame}-s{sub:04}"), target, &pd);
        }
    }
}

/// `VICTORY_PEEK=<address>[:<bytes>],...` (hex addresses, 16 bytes by
/// default): those bytes of title memory, as dwords and floats, after each
/// submission of a listed frame.
fn peek(when: &str) {
    static LIST: std::sync::OnceLock<Vec<(u64, usize)>> = std::sync::OnceLock::new();
    let list = LIST.get_or_init(|| {
        std::env::var("VICTORY_PEEK")
            .unwrap_or_default()
            .split(',')
            .filter_map(|e| {
                let (a, n) = e.split_once(':').unwrap_or((e, "16"));
                let a = u64::from_str_radix(a.trim().trim_start_matches("0x"), 16).ok()?;
                Some((a, n.trim().parse().ok()?))
            })
            .collect()
    });
    for &(at, len) in list {
        let mut bytes = vec![0u8; len.min(256)];
        if !crate::host::read_into(at as *const u8, &mut bytes) {
            continue;
        }
        let words: Vec<String> = bytes
            .chunks(4)
            .map(|c| {
                let w = u32::from_le_bytes(c.try_into().unwrap_or_default());
                format!("{w:#x}={}", f32::from_bits(w))
            })
            .collect();
        tracing::info!("peek {when} {at:#x}: {}", words.join(" "));
    }
}

/// In a listed frame: copies an indirect execution's records (`args`: the
/// list's argument buffer and offset, in the indirect-argument state) and
/// its count (a title buffer's copy) to readback memory,
/// logged once the list has run: how many records ask for work, and the
/// first ones.
pub(crate) fn read_indirect(
    list: &crate::host::HostList,
    args: (&ID3D12Resource, u64),
    count: Option<(&ID3D12Resource, u64)>,
    (kind, stride, max): (u32, u32, u32),
    name: String,
    always: bool,
) {
    struct Readback(ID3D12Resource);
    // SAFETY: D3D12 objects are free-threaded.
    unsafe impl Send for Readback {}
    // SAFETY: as above.
    unsafe impl Sync for Readback {}
    let Some(host) = crate::host::host() else { return };
    let len = u64::from(stride) * u64::from(max.max(1));
    let desc = D3D12_RESOURCE_DESC {
        Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
        Width: len + 4,
        Height: 1,
        DepthOrArraySize: 1,
        MipLevels: 1,
        SampleDesc: DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
        Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
        ..Default::default()
    };
    let heap = D3D12_HEAP_PROPERTIES { Type: D3D12_HEAP_TYPE_READBACK, ..Default::default() };
    let mut buffer: Option<ID3D12Resource> = None;
    let l = &list.list;
    use crate::host::transition;
    let (arg_state, copy) = (D3D12_RESOURCE_STATE_INDIRECT_ARGUMENT, D3D12_RESOURCE_STATE_COPY_SOURCE);
    // SAFETY: plain creation; an open list and live buffers in the states
    // named, put back after the copies.
    unsafe {
        if host
            .device
            .CreateCommittedResource(&heap, D3D12_HEAP_FLAG_NONE, &desc, D3D12_RESOURCE_STATE_COPY_DEST, None, &mut buffer)
            .is_err()
        {
            return;
        }
        let Some(buffer) = buffer else { return };
        l.ResourceBarrier(&[transition(args.0, arg_state, copy)]);
        l.CopyBufferRegion(&buffer, 0, args.0, args.1, len);
        if let Some((c, at)) = count {
            // A title buffer's copy: never transitioned, the read promotes it.
            l.CopyBufferRegion(&buffer, len, c, at, 4);
        }
        l.ResourceBarrier(&[transition(args.0, copy, arg_state)]);
        list.keep(&buffer);
        let buffer = Readback(buffer);
        let counted = count.is_some();
        list.defer_after(move || {
            // Recorded ahead: logged if the frame running it is listed.
            let frame = crate::queue::presented() + 1;
            if !always && !listed(frame) {
                return;
            }
            let mut data = std::ptr::null_mut();
            if buffer.0.Map(0, None, Some(&mut data)).is_err() {
                return;
            }
            let words = std::slice::from_raw_parts(data.cast::<u32>(), (len as usize + 4) / 4);
            let n = counted.then(|| words[len as usize / 4]);
            let records = n.unwrap_or(max).min(max) as usize;
            let step = stride as usize / 4;
            let first = |r: usize| &words[r * step..r * step + step.min(4)];
            // Draws: vertices (or indices) × instances; dispatches: groups.
            let work = |r: usize| -> u64 {
                let w = first(r);
                match kind {
                    0 | 1 => u64::from(w[0]) * u64::from(w.get(1).copied().unwrap_or(1)),
                    _ => w.iter().take(3).map(|&x| u64::from(x)).product(),
                }
            };
            let busy = (0..records).filter(|&r| work(r) > 0).count();
            let total: u64 = (0..records).map(work).sum();
            let shown: Vec<&[u32]> = (0..records.min(4)).map(first).collect();
            tracing::info!(
                target: "victory_gpu::trace",
                "indirect kind {kind} max {max} count {n:?}: {busy} of {records} records with work, {total} in all; first {shown:?} {name} frame {frame}"
            );
            buffer.0.Unmap(0, None);
        });
    }
}

fn dump_one(
    host: &crate::host::Host,
    copy: &crate::resource::HostTexture,
    prefix: &str,
    (i, resource, by): (usize, crate::object::This, &str),
    d: &D3D12_RESOURCE_DESC,
) {
    let Ok(list) = host.new_list() else { return };
    let read = host.execute_texture_read(&list, copy, 0, |list| {
        read_back(&host.device, list, &copy.texture)
    });
    let Some((buffer, footprint)) = read else { return };
    let f = footprint.Footprint;
    let mut data = std::ptr::null_mut();
    // SAFETY: the copy finished; the mapping spans the footprint's rows.
    unsafe {
        if buffer.Map(0, None, Some(&mut data)).is_err() {
            return;
        }
        let pitch = f.RowPitch as usize;
        let rows = std::slice::from_raw_parts(data.cast::<u8>(), buffer.GetDesc().Width as usize);
        // Tiny targets (exposure, averages): their values as they are.
        if f.Width * f.Height <= 16 {
            let texel = crate::resource::element(f.Format.0 as u32).map_or(4, |(b, _)| b as usize);
            let words: Vec<String> = (0..f.Height as usize)
                .flat_map(|y| rows[y * pitch..][..f.Width as usize * texel].chunks(4))
                .map(|c| {
                    let w = u32::from_le_bytes(c.try_into().unwrap_or_default());
                    format!("{w:#x}={}", f32::from_bits(w))
                })
                .collect();
            tracing::info!(?resource, format = f.Format.0, "tiny target {prefix}-{i:02}: {}", words.join(" "));
        }
        // A 24-bit depth plane copies out as 32-bit words (`R32_TYPELESS`):
        // decode it as depth, not as floats (all denormals, shown black).
        let format = if matches!(d.Format.0, 44..=47) { d.Format.0 } else { f.Format.0 };
        if let Some(rgba) = to_rgba(format as u32, f.Width as usize, f.Height as usize, pitch, rows) {
            let mut name = format!(
                "{prefix}-{i:02}-{resource:?}-{}x{}-f{}",
                d.Width, d.Height, d.Format.0
            );
            // The pipeline's hashes (its name is "[stage hash, ...]").
            let by: String = by.chars().filter(char::is_ascii_alphanumeric).collect();
            if !by.is_empty() {
                name.push_str("-by-");
                name.push_str(&by);
            }
            // Whether anything but zeros is there (a dump is normalized).
            if rgba.chunks(4).all(|p| p[..3] == [0, 0, 0]) {
                name.push_str("-black");
            }
            save_as(&name, f.Width, f.Height, f.Width as usize * 4, false, &rgba);
        } else {
            tracing::info!(format = f.Format.0, ?resource, "target not dumped: format");
        }
        buffer.Unmap(0, None);
    }
}

fn half(h: u16) -> f32 {
    let (sign, exp, man) = ((h >> 15) as u32, ((h >> 10) & 0x1f) as i32, (h & 0x3ff) as f32);
    let v = match exp {
        0 => man * 2f32.powi(-24),
        31 if man != 0.0 => f32::NAN,
        31 => f32::INFINITY,
        e => (1.0 + man / 1024.0) * 2f32.powi(e - 15),
    };
    if sign == 1 { -v } else { v }
}

/// A float packed in `bits` with a 5-bit exponent and `man` mantissa bits
/// (the 11- and 10-bit floats of R11G11B10).
fn small_float(bits: u32, man: u32) -> f32 {
    let (exp, m) = ((bits >> man) & 0x1f, (bits & ((1 << man) - 1)) as f32);
    match exp {
        0 => m / (1 << man) as f32 * 2f32.powi(-14),
        31 if m != 0.0 => f32::NAN,
        31 => f32::INFINITY,
        e => (1.0 + m / (1 << man) as f32) * 2f32.powi(e as i32 - 15),
    }
}

/// Rows of a host format as 8-bit RGBA: unsigned 8-bit color as is, everything else
/// as channel values scaled so the largest finite one is white (single
/// channels: lowest black, highest white, as grey). `None` for formats not
/// handled.
fn to_rgba(format: u32, w: usize, h: usize, pitch: usize, rows: &[u8]) -> Option<Vec<u8>> {
    let px = |x: usize, y: usize, n: usize| &rows[y * pitch + x * n..][..n];
    let u16_at = |b: &[u8], i: usize| u16::from_le_bytes([b[2 * i], b[2 * i + 1]]);
    let i16_at = |b: &[u8], i: usize| i16::from_le_bytes([b[2 * i], b[2 * i + 1]]);
    let u32_at = |b: &[u8], i: usize| u32::from_le_bytes(b[4 * i..4 * i + 4].try_into().unwrap_or([0; 4]));
    if matches!(format, 42 | 43) {
        let words = (0..h)
            .flat_map(|y| (0..w).map(move |x| (x, y)))
            .map(|(x, y)| u32_at(px(x, y, 4), 0));
        return Some(r32_integer_to_rgba(format, words));
    }
    let mut out = Vec::with_capacity(w * h * 4);
    // Channel values as floats, then scaled.
    type Decode = Box<dyn Fn(&[u8]) -> [f32; 3]>;
    let (n, decode): (usize, Decode) = match format {
        27..=30 => {
            // The PNG drops alpha: its range and how many texels have each
            // channel zero are logged.
            let (mut low, mut high, mut zero) = ([255u8; 4], [0u8; 4], [0usize; 4]);
            for y in 0..h {
                for x in 0..w {
                    let p = px(x, y, 4);
                    for c in 0..4 {
                        (low[c], high[c]) = (low[c].min(p[c]), high[c].max(p[c]));
                        zero[c] += usize::from(p[c] == 0);
                    }
                    out.extend([p[0], p[1], p[2], 255]);
                }
            }
            tracing::info!(format, ?low, ?high, ?zero, texels = w * h, "target values (8-bit, RGBA)");
            return Some(out);
        }
        31..=32 => (4, Box::new(|b| [0, 1, 2].map(|i| f32::from(b[i] as i8)))),
        // B8G8R8A8 and B8G8R8X8 (89 is R10G10B10_XR_BIAS_A2, below).
        87 | 88 | 90..=93 => {
            for y in 0..h {
                for x in 0..w {
                    let p = px(x, y, 4);
                    out.extend([p[2], p[1], p[0], 255]);
                }
            }
            return Some(out);
        }
        23..=25 | 89 => (4, Box::new(move |b| {
            let v = u32_at(b, 0);
            let rgb = [0, 10, 20].map(|shift| (v >> shift & 0x3ff) as f32);
            if format == DXGI_FORMAT_R10G10B10_XR_BIAS_A2_UNORM.0 as u32 {
                // XR_BIAS encodes 0.0 at 384 and 1.0 at 894.
                rgb.map(|channel| (channel - 384.0) / 510.0)
            } else {
                rgb
            }
        })),
        26 => (4, Box::new(move |b| {
            let v = u32_at(b, 0);
            [small_float(v & 0x7ff, 6), small_float(v >> 11 & 0x7ff, 6), small_float(v >> 22, 5)]
        })),
        9..=10 => (8, Box::new(move |b| [half(u16_at(b, 0)), half(u16_at(b, 1)), half(u16_at(b, 2))])),
        11..=12 => (8, Box::new(move |b| [0, 1, 2].map(|i| f32::from(u16_at(b, i))))),
        13..=14 => (8, Box::new(move |b| [0, 1, 2].map(|i| f32::from(i16_at(b, i))))),
        33..=34 => (4, Box::new(move |b| [half(u16_at(b, 0)), half(u16_at(b, 1)), 0.0])),
        // Signed 32-bit RGBA and RG color channels.
        4 => (16, Box::new(move |b| [0, 1, 2].map(|i| u32_at(b, i) as i32 as f32))),
        18 => (8, Box::new(move |b| {
            [u32_at(b, 0) as i32 as f32, u32_at(b, 1) as i32 as f32, 0.0]
        })),
        1..=2 => (16, Box::new(move |b| {
            [0, 1, 2].map(|i| f32::from_bits(u32_at(b, i)))
        })),
        67 => (4, Box::new(move |b| {
            let v = u32_at(b, 0);
            // One exponent with bias 15 scales the three 9-bit fractions.
            let scale = 2f32.powi((v >> 27) as i32 - 15 - 9);
            [0, 9, 18].map(|shift| ((v >> shift) & 0x1ff) as f32 * scale)
        })),
        15..=16 => (8, Box::new(move |b| [f32::from_bits(u32_at(b, 0)), f32::from_bits(u32_at(b, 1)), 0.0])),
        // R32G32_UINT: both color channels are unsigned integers.
        17 => (8, Box::new(move |b| [u32_at(b, 0) as f32, u32_at(b, 1) as f32, 0.0])),
        35..=36 => (4, Box::new(move |b| [f32::from(u16_at(b, 0)), f32::from(u16_at(b, 1)), 0.0])),
        37..=38 => (4, Box::new(move |b| [f32::from(i16_at(b, 0)), f32::from(i16_at(b, 1)), 0.0])),
        48..=50 => (2, Box::new(move |b| [f32::from(b[0]), f32::from(b[1]), 0.0])),
        51..=52 => (2, Box::new(|b| [f32::from(b[0] as i8), f32::from(b[1] as i8), 0.0])),
        // Single channels (and depth planes): shown as grey.
        39..=41 => (4, Box::new(move |b| [f32::from_bits(u32_at(b, 0)); 3])),
        19..=22 => (8, Box::new(move |b| [f32::from_bits(u32_at(b, 0)); 3])),
        44..=47 => (4, Box::new(move |b| [(u32_at(b, 0) & 0xff_ffff) as f32; 3])),
        53..=54 => (2, Box::new(move |b| [half(u16_at(b, 0)); 3])),
        55..=57 => (2, Box::new(move |b| [f32::from(u16_at(b, 0)); 3])),
        58..=59 => (2, Box::new(move |b| [f32::from(i16_at(b, 0)); 3])),
        60..=62 | 65 => (1, Box::new(move |b| [f32::from(b[0]); 3])),
        63..=64 => (1, Box::new(|b| [f32::from(b[0] as i8); 3])),
        _ => return None,
    };
    let values: Vec<[f32; 3]> = (0..h)
        .flat_map(|y| (0..w).map(move |x| (x, y)))
        .map(|(x, y)| decode(px(x, y, n)))
        .collect();
    let grey = matches!(format, 19..=22 | 39..=41 | 44..=47 | 53..=65);
    let finite = values.iter().flatten().copied().filter(|v| v.is_finite());
    let (low, max) = if grey {
        finite.fold((f32::MAX, f32::MIN), |(l, h), v| (l.min(v), h.max(v)))
    } else {
        (0.0, finite.fold(0.0f32, f32::max))
    };
    // Colour scaled by the 99th percentile of the non-zero values, so a few
    // hot pixels do not hide the rest; how many are not zero, and the median.
    let mut set: Vec<f32> = values
        .iter()
        .flatten()
        .copied()
        .filter(|v| v.is_finite() && *v != 0.0)
        .map(f32::abs)
        .collect();
    let nonzero = set.len() as f64 / (values.len() * 3).max(1) as f64;
    let mut at = |q: f64| -> f32 {
        if set.is_empty() {
            return 0.0;
        }
        let k = ((set.len() - 1) as f64 * q).round() as usize;
        *set.select_nth_unstable_by(k, f32::total_cmp).1
    };
    let (p50, p99) = (at(0.5), at(0.99));
    let not_finite = values.iter().flatten().filter(|v| !v.is_finite()).count();
    let high = if grey || p99 == 0.0 { max } else { p99 };
    let range = (high - low).max(1e-20);
    tracing::info!(format, low, high = max, p50, p99, nonzero, not_finite, "target values");
    // A constant nonzero grey target has no contrast to stretch. Keep it
    // distinguishable from an all-zero target in the dump.
    let flat_nonzero = grey && low == high && low != 0.0;
    for v in values {
        let c = v.map(|c| {
            if flat_nonzero && c.is_finite() {
                255
            } else {
                (((c - low) / range).clamp(0.0, 1.0) * 255.0) as u8
            }
        });
        out.extend([c[0], c[1], c[2], 255]);
    }
    Some(out)
}

fn r32_integer_to_rgba(format: u32, words: impl Iterator<Item = u32>) -> Vec<u8> {
    let signed = format == 43;
    let mut values: Vec<i64> = words
        .map(|word| if signed { i64::from(word as i32) } else { i64::from(word) })
        .collect();
    let first = values.first().copied().unwrap_or(0);
    let (low, high) = values.iter().copied().fold((first, first), |(low, high), value| {
        (low.min(value), high.max(value))
    });
    // The full 32-bit span scaled by 255 fits in i64, without float rounding.
    let range = (high - low).max(1);
    let flat_nonzero = high == low && low != 0;
    let mut out = Vec::with_capacity(values.len() * 4);
    for &value in &values {
        let grey = if flat_nonzero {
            255
        } else {
            ((value - low) * 255 / range) as u8
        };
        out.extend([grey, grey, grey, 255]);
    }

    let total = values.len();
    values.retain(|&value| value != 0);
    let nonzero = values.len() as f64 / total.max(1) as f64;
    for value in &mut values {
        *value = value.abs();
    }
    let mut at = |q: f64| -> i64 {
        if values.is_empty() {
            return 0;
        }
        let k = ((values.len() - 1) as f64 * q).round() as usize;
        *values.select_nth_unstable(k).1
    };
    let (p50, p99) = (at(0.5), at(0.99));
    tracing::info!(format, low, high, p50, p99, nonzero, not_finite = 0, "target values");
    out
}

#[test]
fn signed_32_bit_color_captures_decode_negative_components() {
    let rg = [-(1 << 30), 1 << 29, i32::MAX, 1 << 28, -(1 << 29), i32::MAX];
    let rgba = [
        -(1 << 30), 1 << 29, -(1 << 28), i32::MAX, i32::MAX,
        1 << 28, -(1 << 29), 1 << 29, i32::MAX, i32::MAX,
    ];
    let rg: Vec<_> = rg.into_iter().flat_map(i32::to_le_bytes).collect();
    let rgba: Vec<_> = rgba.into_iter().flat_map(i32::to_le_bytes).collect();
    assert_eq!(
        [
            to_rgba(DXGI_FORMAT_R32G32_SINT.0 as u32, 1, 2, 12, &rg),
            to_rgba(DXGI_FORMAT_R32G32B32A32_SINT.0 as u32, 1, 2, 20, &rgba),
        ],
        [
            Some(vec![0, 127, 0, 255, 63, 0, 0, 255]),
            Some(vec![0, 127, 0, 255, 63, 0, 127, 255]),
        ]
    );
}

/// Pixels per side of the grid [`probe`] reads.
const GRID: u32 = 4;
/// Bytes between the grid's pixels in the readback buffer (placement
/// alignment).
const SLOT: u64 = 512;

pub(crate) struct BlackProbe {
    buffer: ID3D12Resource,
    packed_rgb10: bool,
}

/// Copies a grid of pixels from a color texture in copy-source state for
/// black-frame diagnostics.
pub(crate) fn probe(
    device: &ID3D12Device,
    list: &ID3D12GraphicsCommandList,
    texture: &ID3D12Resource,
) -> Option<BlackProbe> {
    // SAFETY: a live resource.
    let d = unsafe { texture.GetDesc() };
    let packed_rgb10 = match d.Format {
        DXGI_FORMAT_R10G10B10A2_TYPELESS | DXGI_FORMAT_R10G10B10A2_UNORM | DXGI_FORMAT_R10G10B10A2_UINT => true,
        format if matches!(format.0, 27..=29 | 87 | 90 | 91) => false,
        _ => return None,
    };
    let format = if d.Format == DXGI_FORMAT_R10G10B10A2_TYPELESS {
        DXGI_FORMAT_R10G10B10A2_UNORM
    } else {
        d.Format
    };
    let desc = D3D12_RESOURCE_DESC {
        Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
        Width: SLOT * u64::from(GRID * GRID),
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
    let heap = D3D12_HEAP_PROPERTIES {
        Type: D3D12_HEAP_TYPE_READBACK,
        ..Default::default()
    };
    // One buffer for every present: each is waited for before the next
    // (made and released per present, it cost a kernel call per frame).
    static BUFFER: std::sync::OnceLock<Option<ID3D12Resource>> = std::sync::OnceLock::new();
    let buffer = BUFFER
        .get_or_init(|| {
            let mut buffer: Option<ID3D12Resource> = None;
            // SAFETY: plain creation.
            unsafe {
                device.CreateCommittedResource(
                    &heap,
                    D3D12_HEAP_FLAG_NONE,
                    &desc,
                    D3D12_RESOURCE_STATE_COPY_DEST,
                    None,
                    &mut buffer,
                )
            }
            .ok()?;
            buffer
        })
        .clone()?;
    // SAFETY: an open list; locations naming live resources.
    unsafe {
        for i in 0..GRID * GRID {
            let (x, y) = (
                (2 * (i % GRID) + 1) * d.Width as u32 / (2 * GRID),
                (2 * (i / GRID) + 1) * d.Height / (2 * GRID),
            );
            let to = D3D12_TEXTURE_COPY_LOCATION {
                pResource: std::mem::transmute_copy(&buffer),
                Type: D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT,
                Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 {
                    PlacedFootprint: D3D12_PLACED_SUBRESOURCE_FOOTPRINT {
                        Offset: SLOT * u64::from(i),
                        Footprint: D3D12_SUBRESOURCE_FOOTPRINT {
                            Format: format,
                            Width: 1,
                            Height: 1,
                            Depth: 1,
                            RowPitch: 256,
                        },
                    },
                },
            };
            let from = D3D12_TEXTURE_COPY_LOCATION {
                pResource: std::mem::transmute_copy(texture),
                Type: D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX,
                Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 {
                    SubresourceIndex: 0,
                },
            };
            let pixel = D3D12_BOX {
                left: x,
                top: y,
                front: 0,
                right: x + 1,
                bottom: y + 1,
                back: 1,
            };
            list.CopyTextureRegion(&to, 0, 0, 0, &from, Some(&pixel));
        }
        Some(BlackProbe { buffer, packed_rgb10 })
    }
}

/// Whether every sampled pixel is (near) black.
pub(crate) fn black(probe: &BlackProbe) -> bool {
    let buffer = &probe.buffer;
    let mut data = std::ptr::null_mut();
    // SAFETY: the copies finished (every submission is waited for).
    unsafe {
        if buffer.Map(0, None, Some(&mut data)).is_err() {
            return false;
        }
        let bytes =
            std::slice::from_raw_parts(data.cast::<u8>(), (SLOT * u64::from(GRID * GRID)) as usize);
        let level = bytes
            .chunks(SLOT as usize)
            .map(|pixel| {
                if probe.packed_rgb10 {
                    let packed = u32::from_le_bytes(pixel[..4].try_into().unwrap());
                    let max = (packed & 1023).max((packed >> 10) & 1023).max((packed >> 20) & 1023);
                    // Keep the existing 8-bit diagnostic scale and ignore alpha.
                    ((max * 255 + 511) / 1023) as u8
                } else {
                    pixel[0].max(pixel[1]).max(pixel[2])
                }
            })
            .max()
            .unwrap_or(0);
        buffer.Unmap(0, None);
        let mut levels = LEVELS.lock().unwrap_or_else(|e| e.into_inner());
        if levels.len() == 12 {
            levels.pop_front();
        }
        levels.push_back(level);
        level < 8
    }
}

/// The brightest probed channel of the last frames, oldest first: a fade
/// darkens over several, a glitch drops at once.
static LEVELS: std::sync::Mutex<std::collections::VecDeque<u8>> =
    std::sync::Mutex::new(std::collections::VecDeque::new());

pub(crate) fn recent_levels() -> Vec<u8> {
    let levels = LEVELS.lock().unwrap_or_else(|e| e.into_inner());
    levels.iter().copied().collect()
}

#[cfg(test)]
mod rgb9e5_tests {
    #[test]
    fn rgb9e5_target_decodes_pitched_rows_with_shared_exponents() {
        use super::{DXGI_FORMAT_R32_UINT, DXGI_FORMAT_R9G9B9E5_SHAREDEXP, to_rgba};

        let pitch = 256;
        let mut rows = [0xff; 256 + 4];
        // The same RGB mantissas yield (1, 0.5, 0.25) and (2, 1, 0.5).
        rows[..4].copy_from_slice(&0x8101_0100u32.to_le_bytes());
        rows[pitch..].copy_from_slice(&0x8901_0100u32.to_le_bytes());
        assert_eq!(
            to_rgba(DXGI_FORMAT_R9G9B9E5_SHAREDEXP.0 as u32, 1, 2, pitch, &rows),
            Some(vec![127, 63, 31, 255, 255, 127, 63, 255]),
        );
        let rgba = to_rgba(DXGI_FORMAT_R32_UINT.0 as u32, 1, 2, pitch, &rows).unwrap();
        assert_eq!(rgba.len(), 8);
        for &pixel in rgba.as_chunks::<4>().0 {
            assert_eq!(pixel, [pixel[0], pixel[0], pixel[0], 255]);
        }
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn half_classifies_special_and_finite_values() {
        use super::half;

        assert!(half(0x7c01).is_nan());
        assert!(half(0xfe00).is_nan());
        assert_eq!(half(0x7c00), f32::INFINITY);
        assert_eq!(half(0xfc00), f32::NEG_INFINITY);
        assert_eq!(half(0x0000).to_bits(), 0.0f32.to_bits());
        assert_eq!(half(0x8000).to_bits(), (-0.0f32).to_bits());
        assert_eq!(half(0x0001), 2f32.powi(-24));
        assert_eq!(half(0xbe00), -1.5);
    }

    #[test]
    fn packed_float_classifies_special_and_finite_values() {
        use super::small_float;

        for mantissa_bits in [5, 6] {
            let special = 31 << mantissa_bits;
            assert!(small_float(special | 1, mantissa_bits).is_nan());
            assert!(small_float(special | (1 << (mantissa_bits - 1)), mantissa_bits).is_nan());
            assert_eq!(small_float(special, mantissa_bits), f32::INFINITY);
            assert_eq!(small_float(0, mantissa_bits).to_bits(), 0.0f32.to_bits());
            assert_eq!(small_float(1, mantissa_bits), 2f32.powi(-14 - mantissa_bits as i32));
            assert_eq!(small_float(15 << mantissa_bits, mantissa_bits), 1.0);
        }
    }

    #[test]
    fn hdr_targets_render_nan_and_infinity_as_distinct_channel_values() {
        // R16G16B16A16_FLOAT: [NaN, +Inf, NaN], [1, 0.5, 0].
        let half = [0x7c01u16, 0x7c00, 0xfe00, 0, 0x3c00, 0x3800, 0, 0];
        let half: Vec<_> = half.into_iter().flat_map(u16::to_le_bytes).collect();
        // R11G11B10_FLOAT: the same values with 6-bit and 5-bit mantissas.
        let packed = [
            0x7c1u32 | (0x7c0 << 11) | (0x3e1 << 22),
            0x3c0 | (0x380 << 11),
        ];
        let packed: Vec<_> = packed.into_iter().flat_map(u32::to_le_bytes).collect();
        let expected = [0, 255, 0, 255, 255, 127, 0, 255];
        for (format, row) in [(10, half), (26, packed)] {
            assert_eq!(
                super::to_rgba(format, 2, 1, row.len(), &row).as_deref(),
                Some(&expected[..]),
                "format {format}"
            );
        }
    }

    #[test]
    fn constant_nonzero_grayscale_targets_do_not_look_black() {
        let capture = |format, values: [f32; 2]| {
            let row: Vec<_> = values.into_iter().flat_map(f32::to_le_bytes).collect();
            super::to_rgba(format, 2, 1, row.len(), &row).unwrap()
        };
        let black = [0, 0, 0, 255];
        let white = [255; 4];
        let depth = super::DXGI_FORMAT_D32_FLOAT.0 as u32;
        let scalar = super::DXGI_FORMAT_R32_FLOAT.0 as u32;
        assert_eq!(capture(depth, [0.25; 2]), white.repeat(2));
        assert_eq!(capture(scalar, [-0.25; 2]), white.repeat(2));
        assert_eq!(capture(depth, [0.0; 2]), black.repeat(2));
        assert_eq!(capture(depth, [0.25, 0.75]), [black, white].concat());
    }

    // These fixtures can share the RGBA8 callsite. Keep its registration
    // from racing the test that installs a scoped tracing subscriber.
    static RGBA8_TRACE: std::sync::Mutex<()> = std::sync::Mutex::new(());

    #[test]
    fn crc_matches_the_png_check_value() {
        assert_eq!(super::crc32(b"IEND"), 0xae42_6082);
    }

    #[test]
    fn signed_r16_colours_do_not_brighten_negative_components() {
        let rgba: Vec<_> = [-1i16, 1024, 512, 0]
            .into_iter()
            .flat_map(i16::to_le_bytes)
            .collect();
        for format in [13, 14] {
            assert_eq!(
                super::to_rgba(format, 1, 1, 8, &rgba),
                Some(vec![0, 255, 127, 255]),
            );
        }
        for format in [37, 38] {
            assert_eq!(
                super::to_rgba(format, 1, 1, 4, &rgba[..4]),
                Some(vec![0, 255, 0, 255]),
            );
        }
        let unsigned = super::to_rgba(12, 1, 1, 8, &rgba).unwrap();
        assert_eq!(unsigned, [255, 3, 1, 255]);
    }

    #[test]
    fn signed_r16_luma_keeps_negative_zero_and_positive_in_order() {
        let mut rows: Vec<_> = [i16::MIN, 0, i16::MAX]
            .into_iter()
            .flat_map(i16::to_le_bytes)
            .collect();
        rows.extend([0xff; 2]);
        for format in [58, 59] {
            assert_eq!(
                super::to_rgba(format, 3, 1, 8, &rows),
                Some(vec![0, 0, 0, 255, 127, 127, 127, 255, 255, 255, 255, 255]),
            );
        }
    }

    #[test]
    fn rg32_uint_channels_are_unsigned_and_skip_padding() {
        let format = super::DXGI_FORMAT_R32G32_UINT.0 as u32;
        let rows: Vec<_> = [0x8000_0000, 0x4000_0000, u32::MAX, 0, 0x8000_0000, u32::MAX]
            .into_iter()
            .flat_map(u32::to_le_bytes)
            .collect();
        assert_eq!(
            super::to_rgba(format, 1, 2, 12, &rows),
            Some(vec![255, 127, 0, 255, 0, 255, 0, 255])
        );
    }

    /// R16G16B16A16_UNORM is integers, not half floats; 89 is 10:10:10:2,
    /// not B8G8R8A8.
    #[test]
    fn unorm16_and_xr_bias_targets_decode_as_their_formats() {
        // Red 0x8000 (a half-float -0), green 0x4000, blue 0.
        let unorm16 = [0x00, 0x80, 0x00, 0x40, 0, 0, 0xff, 0xff];
        assert_eq!(super::to_rgba(11, 1, 1, 8, &unorm16), Some(vec![255, 127, 0, 255]));
        // Red 0x3ff, green 0x1ff, blue 0, alpha 3.
        let xr = (0x3ffu32 | 0x1ff << 10 | 3 << 30).to_le_bytes();
        assert_eq!(super::to_rgba(89, 1, 1, 4, &xr), Some(vec![255, 50, 0, 255]));
        // B8G8R8A8 still swaps to RGBA.
        assert_eq!(super::to_rgba(87, 1, 1, 4, &[1, 2, 3, 4]), Some(vec![3, 2, 1, 255]));
    }

    #[test]
    fn xr_bias_targets_decode_black_and_extended_range_channels() {
        let xr = super::DXGI_FORMAT_R10G10B10_XR_BIAS_A2_UNORM.0 as u32;
        let pixel = |r: u32, g: u32, b: u32| (r | g << 10 | b << 20 | 3 << 30).to_le_bytes();
        assert_eq!(
            super::to_rgba(xr, 1, 1, 4, &pixel(384, 384, 384)),
            Some(vec![0, 0, 0, 255])
        );
        assert_eq!(
            super::to_rgba(xr, 1, 1, 4, &pixel(894, 639, 384)),
            Some(vec![255, 127, 0, 255])
        );
    }
}

#[cfg(test)]
mod r32_tests {
    #[test]
    fn r32_integer_captures_scale_numeric_values() {
        let unsigned = [1u32, 2, 3].map(u32::to_le_bytes).concat();
        let signed = [-2i32, -1, 0].map(i32::to_le_bytes).concat();
        let expected = Some(vec![0, 0, 0, 255, 127, 127, 127, 255, 255, 255, 255, 255]);

        assert_eq!(super::to_rgba(42, 3, 1, 12, &unsigned), expected);
        assert_eq!(super::to_rgba(43, 3, 1, 12, &signed), expected);
    }

    #[test]
    fn r32_integer_captures_keep_adjacent_contrast() {
        let unsigned = [16_777_216u32, 16_777_217].map(u32::to_le_bytes).concat();
        let signed = [-16_777_217i32, -16_777_216].map(i32::to_le_bytes).concat();
        let expected = Some(vec![0, 0, 0, 255, 255, 255, 255, 255]);

        assert_eq!(super::to_rgba(42, 2, 1, 8, &unsigned), expected);
        assert_eq!(super::to_rgba(43, 2, 1, 8, &signed), expected);
    }

    #[test]
    fn r32_integer_captures_handle_full_range_and_row_pitch() {
        let signed = [i32::MIN, -1, i32::MIN, 0, i32::MAX].map(i32::to_le_bytes).concat();
        let expected = Some(vec![0, 0, 0, 255, 127, 127, 127, 255, 127, 127, 127, 255, 255, 255, 255, 255]);

        assert_eq!(super::to_rgba(43, 2, 2, 12, &signed), expected);
    }

    #[test]
    fn r32_integer_captures_constant_uint_is_white() {
        let rows = [7u32; 2].map(u32::to_le_bytes).concat();
        assert_eq!(super::to_rgba(42, 2, 1, 8, &rows), Some(vec![255; 8]));
    }

    #[test]
    fn r32_integer_captures_constant_sint_is_white() {
        let rows = [-7i32; 2].map(i32::to_le_bytes).concat();
        assert_eq!(super::to_rgba(43, 2, 1, 8, &rows), Some(vec![255; 8]));
    }

    #[test]
    fn r32_integer_captures_constant_zero_is_black() {
        let expected = Some([0, 0, 0, 255].repeat(2));
        assert_eq!(super::to_rgba(42, 2, 1, 8, &[0; 8]), expected);
        assert_eq!(super::to_rgba(43, 2, 1, 8, &[0; 8]), expected);
    }

    #[test]
    fn signed_rgba8_clamps_negative_colour_and_keeps_alpha_opaque() {
        let _trace = RGBA8_TRACE.lock().unwrap_or_else(|e| e.into_inner());
        let rows = [0xff, 64, 32, 0];
        let actual = [31, 32].map(|format| super::to_rgba(format, 1, 1, 4, &rows).unwrap());
        assert_eq!(actual, [[0, 255, 127, 255]; 2]);
    }

    #[test]
    fn signed_rg8_clamps_negative_colour() {
        let rows = [0xff, 64];
        let actual = [51, 52].map(|format| super::to_rgba(format, 1, 1, 2, &rows).unwrap());
        assert_eq!(actual, [[0, 255, 0, 255]; 2]);
    }

    #[test]
    fn signed_r8_grayscale_uses_signed_extrema() {
        let rows = [0x80, 0xff, 0, 0x7f];
        let actual = [63, 64].map(|format| super::to_rgba(format, 4, 1, 4, &rows).unwrap());
        let expected = [0, 127, 128, 255].map(|grey| [grey, grey, grey, 255]).concat();
        assert_eq!(actual, [expected.clone(), expected]);
    }

    #[test]
    fn unsigned_srgb_and_alpha_only_keep_their_byte_interpretation() {
        let _trace = RGBA8_TRACE.lock().unwrap_or_else(|e| e.into_inner());
        let rows = [0xff, 64, 32, 0];
        for format in [27, 28, 29, 30] {
            assert_eq!(super::to_rgba(format, 1, 1, 4, &rows).unwrap(), [255, 64, 32, 255]);
        }
        for format in [48, 49, 50] {
            assert_eq!(super::to_rgba(format, 1, 1, 2, &rows).unwrap(), [255, 64, 0, 255]);
        }
        for format in [60, 61, 62, 65] {
            assert_eq!(
                super::to_rgba(format, 2, 1, 2, &rows).unwrap(),
                [255, 255, 255, 255, 0, 0, 0, 255],
            );
        }
    }

    #[test]
    fn unsigned_rgba8_logs_raw_channel_stats_excluding_row_padding() {
        use std::collections::BTreeMap;
        use std::sync::{
            Arc,
            atomic::{AtomicUsize, Ordering},
        };
        use tracing::{Event, Metadata, Subscriber, field, span};

        #[derive(Default)]
        struct Fields(BTreeMap<&'static str, String>);

        impl field::Visit for Fields {
            fn record_debug(&mut self, field: &field::Field, value: &dyn std::fmt::Debug) {
                self.0.insert(field.name(), format!("{value:?}"));
            }
        }

        struct RgbaStats(Arc<AtomicUsize>);

        impl Subscriber for RgbaStats {
            fn enabled(&self, metadata: &Metadata<'_>) -> bool {
                metadata.is_event()
                    && metadata.target() == "victory_gpu::capture"
                    && metadata.fields().field("zero").is_some()
            }

            fn new_span(&self, _: &span::Attributes<'_>) -> span::Id {
                unreachable!()
            }

            fn record(&self, _: &span::Id, _: &span::Record<'_>) {}
            fn record_follows_from(&self, _: &span::Id, _: &span::Id) {}
            fn enter(&self, _: &span::Id) {}
            fn exit(&self, _: &span::Id) {}

            fn event(&self, event: &Event<'_>) {
                let mut fields = Fields::default();
                event.record(&mut fields);
                assert_eq!(fields.0["low"], "[0, 0, 3, 0]");
                assert_eq!(fields.0["high"], "[250, 7, 11, 255]");
                assert_eq!(fields.0["zero"], "[1, 2, 0, 3]");
                assert_eq!(fields.0["texels"], "4");
                self.0.fetch_add(1, Ordering::Relaxed);
            }
        }

        let _trace = RGBA8_TRACE.lock().unwrap_or_else(|e| e.into_inner());
        let rows = [0, 2, 5, 0, 250, 0, 3, 0, 99, 99, 99, 99, 100, 0, 8, 255, 10, 7, 11, 0];
        let expected = [0, 2, 5, 255, 250, 0, 3, 255, 100, 0, 8, 255, 10, 7, 11, 255];
        let events = Arc::new(AtomicUsize::new(0));
        tracing::subscriber::with_default(RgbaStats(events.clone()), || {
            for format in 27..=30 {
                assert_eq!(super::to_rgba(format, 2, 2, 12, &rows).unwrap(), expected);
                assert_eq!(events.load(Ordering::Relaxed), (format - 26) as usize);
            }
        });
    }
}
