//! Texture contents the title writes itself (decision 0007, "Texture
//! memory"): streamed pages, layout-library copies, movie frames, glyph
//! atlases. Such a texture lives in console memory in its tiled layout;
//! before a submission samples its host copy, the memory is detiled into a
//! staging buffer when its content changed, and the staging buffer is
//! copied into the host texture.
//!
//! "Changed" means a page of the memory changed since the last detiling
//! ([`crate::watch`], which follows writes through every mapping of the
//! same physical pages) in the runtime's memory; elsewhere, a content hash
//! of the whole memory at every submission that samples it.
//!
//! shortcut: a change detiles the whole texture, not the pages written.
//! Textures read through descriptor tables are brought up to date, and for
//! shaders that index the heap themselves every texture with a resource
//! object the heap views (`crate::bindless`); textures the title placed in
//! memory itself are not made for the heap (see `descriptor::host`). Plane
//! 0 is uploaded (every mip and array slice), depth and stencil planes are
//! not.

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use windows::Win32::Graphics::Direct3D12::*;

use crate::host::{HostList, host};
use crate::object::This;
use windows::core::Interface;

macro_rules! once {
    ($($arg:tt)*) => {{
        static ONCE: std::sync::Once = std::sync::Once::new();
        ONCE.call_once(|| tracing::warn!($($arg)*));
    }};
}

/// A texture's upload state: how a staging buffer lays out the host
/// texture's subresources, and what was last detiled. The buffer itself is
/// taken for each upload from a small pool shared by all textures
/// ([`staging_buffer`]; kept for each texture's life, the buffers took its
/// size again in system memory: 2.6-2.8 GB with placed textures made on
/// read, run integ2).
pub(crate) struct Staging {
    /// A staging buffer's bytes.
    size: u64,
    footprints: Vec<D3D12_PLACED_SUBRESOURCE_FOOTPRINT>,
    rows: Vec<u32>,
    layout: victory_formats::texture::Layout,
    /// Mips per array slice.
    mips: u32,
    memory: *const u8,
    /// The memory's pages changed since detiled (none: not the runtime's
    /// memory, hashed instead).
    tracked: Option<crate::watch::Tracked>,
    /// Per subresource, the bytes of the texture's memory it lies in (from
    /// its start, whole layout blocks); empty where the layout gives none.
    spans: Vec<(usize, usize)>,
    last: Mutex<Last>,
    /// Uploads [`record_upload`] recorded, and the subresources they
    /// copied.
    uploads: std::sync::atomic::AtomicU64,
    copied: std::sync::atomic::AtomicU64,
}

/// What was last detiled: the hash of the memory (not tracked), or of each
/// subresource's bytes (tracked).
#[derive(Default)]
struct Last {
    hash: Option<blake3::Hash>,
    hashes: Vec<Option<blake3::Hash>>,
}

/// Per subresource of `layout` (`mips` per array slice; a 3D texture's
/// are its mips), the bytes of the texture's memory it lies in, rounded out
/// to the layout's blocks; empty if the layout cannot say.
fn spans(layout: &victory_formats::texture::Layout, count: u32, mips: u32) -> Vec<(usize, usize)> {
    use victory_formats::texture::Dimension;
    let d = &layout.desc;
    let block = layout.alignment.max(256);
    (0..count)
        .map(|i| {
            let (mip, slice) = (i % mips, i / mips);
            let m = layout.mips.get(mip as usize)?;
            let (z0, z1) = if d.dimension == Dimension::Tex3d { (0, m.depth.max(1) - 1) } else { (slice, slice) };
            let first = layout.offset(mip, 0, 0, z0).ok()?;
            let last = layout.offset(mip, m.width.max(1) - 1, m.height.max(1) - 1, z1).ok()? + u64::from(d.element_bytes);
            let (from, to) = (first / block * block, last.div_ceil(block) * block);
            Some((from as usize, (to.min(layout.size)) as usize))
        })
        .collect::<Option<Vec<_>>>()
        .unwrap_or_default()
}

/// Staging buffers for detiles to reuse, by size: the menu movie rewrites
/// the same textures every frame, and a fresh 12.7 MB of upload buffers a
/// frame cost ~5 ms (run textures8). Uploads run only inside
/// `Host::execute`, which waits for each submission, so the buffers of one
/// submission are free from the next on.
struct Buffers {
    /// (bytes, buffer, mapping), oldest first.
    free: std::collections::VecDeque<(u64, ID3D12Resource, usize)>,
    free_bytes: u64,
    /// Taken by the submission being prepared.
    used: Vec<(u64, ID3D12Resource, usize)>,
}

// SAFETY: the mappings are persistent mappings of the upload buffers.
unsafe impl Send for Buffers {}

static BUFFERS: Mutex<Buffers> = Mutex::new(Buffers {
    free: std::collections::VecDeque::new(),
    free_bytes: 0,
    used: Vec::new(),
});

/// Free staging buffers kept at most.
const KEPT_BYTES: u64 = 64 << 20;

/// Called before a submission's uploads: the buffers of the last one are
/// free again (the oldest dropped past `KEPT_BYTES`).
pub(crate) fn recycle() {
    {
        let mut c = CHUNKS.lock().unwrap_or_else(|e| e.into_inner());
        (c.current, c.used) = (0, 0);
        c.all.truncate(CHUNKS_KEPT);
    }
    let mut b = BUFFERS.lock().unwrap_or_else(|e| e.into_inner());
    let b = &mut *b;
    for e in b.used.drain(..) {
        b.free_bytes += e.0;
        b.free.push_back(e);
    }
    while b.free_bytes > KEPT_BYTES
        && let Some((n, ..)) = b.free.pop_front()
    {
        b.free_bytes -= n;
    }
}

/// A free staging buffer of `size` bytes, or a new one.
/// Sizes are taken in powers of two (at least 64 KiB), so a buffer serves
/// textures of other sizes too: a new shot's textures came in new sizes and
/// made buffers 10 % of the submitting thread (run evict5).
fn staging_buffer(size: u64) -> Option<(ID3D12Resource, *mut u8)> {
    let class = size.max(1 << 16).next_power_of_two();
    let taken = {
        let mut b = BUFFERS.lock().unwrap_or_else(|e| e.into_inner());
        let i = b.free.iter().position(|e| e.0 == class);
        i.and_then(|i| b.free.remove(i)).inspect(|_| b.free_bytes -= class)
    };
    // Made without the pool's lock (refreshes run on several threads).
    let e = match taken {
        Some(e) => e,
        None => {
            let (buffer, data) = host()?.upload_buffer(class)?;
            crate::host::name(&buffer, || "texture staging".into());
            (class, buffer, data as usize)
        }
    };
    let made = (e.1.clone(), e.2 as *mut u8);
    BUFFERS.lock().unwrap_or_else(|e| e.into_inner()).used.push(e);
    Some(made)
}

/// Staging space carved from large upload buffers, front to back: most
/// textures take a part of one instead of a buffer of their own (making
/// upload buffers was 10 % of the submitting thread in a new shot, run
/// evict5). Rewound at each submission ([`recycle`]): the last one has run.
struct Chunks {
    /// (buffer, mapping), [`CHUNK`] bytes each.
    all: Vec<(ID3D12Resource, usize)>,
    /// The chunk being carved and the bytes taken from it.
    current: usize,
    used: u64,
}

// SAFETY: the mappings are persistent mappings of the upload buffers.
unsafe impl Send for Chunks {}

static CHUNKS: Mutex<Chunks> = Mutex::new(Chunks { all: Vec::new(), current: 0, used: 0 });

const CHUNK: u64 = 32 << 20;
/// Chunks kept between submissions (more are made while one needs them).
const CHUNKS_KEPT: usize = 16;

/// `size` bytes of staging space: a buffer, the offset of the space in it
/// (a multiple of 512, as texture copies need) and its mapping there.
/// Space larger than a chunk is a buffer of its own ([`staging_buffer`]).
fn staging_space(size: u64) -> Option<(ID3D12Resource, u64, *mut u8)> {
    if size > CHUNK {
        return staging_buffer(size).map(|(b, data)| (b, 0, data));
    }
    let mut c = CHUNKS.lock().unwrap_or_else(|e| e.into_inner());
    loop {
        if let Some((buffer, data)) = c.all.get(c.current) {
            let at = c.used.next_multiple_of(512);
            if at + size <= CHUNK {
                let made = (buffer.clone(), at, (*data + at as usize) as *mut u8);
                c.used = at + size;
                return Some(made);
            }
            (c.current, c.used) = (c.current + 1, 0);
            continue;
        }
        let (buffer, data) = host()?.upload_buffer(CHUNK)?;
        crate::host::name(&buffer, || "texture staging chunk".into());
        c.all.push((buffer, data as usize));
    }
}

// SAFETY: the title memory is only read, under `last`'s lock.
unsafe impl Send for Staging {}
// SAFETY: as above.
unsafe impl Sync for Staging {}

/// Bytes retained by the texture upload pools and the idle per-thread read buffers.
#[cfg(feature = "diagnostics")]
pub(crate) fn cached_staging_bytes() -> Option<(u64, usize)> {
    let chunk_bytes = CHUNKS.try_lock().ok()?.all.len() as u64 * CHUNK;
    let pool = BUFFERS.try_lock().ok()?;
    let large_bytes = pool.free_bytes + pool.used.iter().map(|entry| entry.0).sum::<u64>();
    let read_bytes = RETAINED_SCRATCH.load(std::sync::atomic::Ordering::Relaxed);
    Some((chunk_bytes + large_bytes, read_bytes))
}

/// A host copy or resolve that `list` records writes `resource`: from the
/// submission that runs `list` on, its host texture's contents are the
/// host's, not title memory's, which stays as it was. Marked when that
/// submission is made, not when `list` is recorded: a list recorded before
/// it but run after it must not upload title memory over the host's texels
/// (submissions run in the order they are made, which need not be the
/// order their lists were recorded in), and one run before it must still
/// get its upload.
pub(crate) fn host_writes(list: &HostList, resource: This) {
    if let Some(copy) = crate::resource::host_copy(resource) {
        list.defer_host(move || copy.host_written.store(true, std::sync::atomic::Ordering::Relaxed));
    }
}

fn is_host_written(resource: This) -> bool {
    crate::resource::host_copy(resource)
        .is_some_and(|c| c.host_written.load(std::sync::atomic::Ordering::Relaxed))
}

/// Whether `resource`'s host copy takes its contents from title memory:
/// a texture the host GPU does not render to or write.
pub(crate) fn title_written(texture: &ID3D12Resource) -> bool {
    const TARGET_DEPTH_UNORDERED: i32 = 0x7;
    // SAFETY: a live resource.
    unsafe { texture.GetDesc() }.Flags.0 & TARGET_DEPTH_UNORDERED == 0
}

/// Copies [`alias_copy`] recorded since the last frame-rate line.
pub(crate) static ALIAS_COPIES: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// Maximum number of `(desc_a, desc_b)` footprint plans cached in [`ALIAS_PLANS`].
const ALIAS_PLAN_CAP: usize = 256;

/// Hashable subset of [`D3D12_RESOURCE_DESC`] fields that determine copyable footprints.
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
struct DescKey {
    dimension: i32,
    width: u64,
    height: u32,
    depth_or_array_size: u16,
    mip_levels: u16,
    format: i32,
    sample_count: u32,
    sample_quality: u32,
    layout: i32,
    flags: i32,
}

fn desc_key(d: &D3D12_RESOURCE_DESC) -> DescKey {
    DescKey {
        dimension: d.Dimension.0,
        width: d.Width,
        height: d.Height,
        depth_or_array_size: d.DepthOrArraySize,
        mip_levels: d.MipLevels,
        format: d.Format.0,
        sample_count: d.SampleDesc.Count,
        sample_quality: d.SampleDesc.Quality,
        layout: d.Layout.0,
        flags: d.Flags.0,
    }
}

/// Precomputed subresource footprints and scratch size for an alias copy between two descriptions.
struct AliasPlan {
    fa: Box<[D3D12_PLACED_SUBRESOURCE_FOOTPRINT]>,
    fb: Box<[D3D12_PLACED_SUBRESOURCE_FOOTPRINT]>,
    n: u32,
    total: u64,
}

/// Cache map keyed by `(desc_a, desc_b)` (`None` caches incompatible pairs).
type PlanMap = HashMap<(DescKey, DescKey), Option<Arc<AliasPlan>>>;

/// Bounded mutex-guarded cache of alias-copy footprint plans (`None` caches incompatible pairs).
struct AliasPlanCache {
    entries: Mutex<Option<PlanMap>>,
    #[cfg(test)]
    computations: std::sync::atomic::AtomicU64,
}

impl AliasPlanCache {
    const fn new() -> Self {
        Self {
            entries: Mutex::new(None),
            #[cfg(test)]
            computations: std::sync::atomic::AtomicU64::new(0),
        }
    }

    /// Returns the cached alias-copy plan for `(da, db)`, computing and caching it on a miss.
    fn get_or_compute(
        &self,
        device: &ID3D12Device,
        da: &D3D12_RESOURCE_DESC,
        db: &D3D12_RESOURCE_DESC,
    ) -> Option<Arc<AliasPlan>> {
        let key = (desc_key(da), desc_key(db));
        {
            let cache = self.entries.lock().unwrap_or_else(|e| e.into_inner());
            if let Some(cached) = cache.as_ref().and_then(|m| m.get(&key)) {
                return cached.clone();
            }
        }
        #[cfg(test)]
        self.computations
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let plan = compute_alias_plan(device, da, db).map(Arc::new);
        let mut cache = self.entries.lock().unwrap_or_else(|e| e.into_inner());
        let map = cache.get_or_insert_with(HashMap::new);
        if map.len() >= ALIAS_PLAN_CAP && !map.contains_key(&key) {
            map.clear();
        }
        map.insert(key, plan.clone());
        plan
    }

    #[cfg(test)]
    fn len(&self) -> usize {
        self.entries
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .as_ref()
            .map_or(0, HashMap::len)
    }

    #[cfg(test)]
    fn clear(&self) {
        if let Some(map) = self.entries.lock().unwrap_or_else(|e| e.into_inner()).as_mut() {
            map.clear();
        }
        self.computations
            .store(0, std::sync::atomic::Ordering::Relaxed);
    }
}

/// Global bounded cache of alias-copy footprint plans.
static ALIAS_PLANS: AliasPlanCache = AliasPlanCache::new();

fn subresource_count(d: &D3D12_RESOURCE_DESC) -> u32 {
    if d.Dimension == D3D12_RESOURCE_DIMENSION_TEXTURE3D {
        u32::from(d.MipLevels)
    } else {
        u32::from(d.MipLevels) * u32::from(d.DepthOrArraySize).max(1)
    }
}

fn footprints_for(
    device: &ID3D12Device,
    d: &D3D12_RESOURCE_DESC,
    count: u32,
) -> (
    Vec<D3D12_PLACED_SUBRESOURCE_FOOTPRINT>,
    Vec<u32>,
    Vec<u64>,
    u64,
) {
    let mut f = vec![D3D12_PLACED_SUBRESOURCE_FOOTPRINT::default(); count as usize];
    let mut rows = vec![0u32; count as usize];
    let mut row_sizes = vec![0u64; count as usize];
    let mut total = 0;
    // SAFETY: output slices are sized for `count` subresources starting at subresource 0.
    unsafe {
        device.GetCopyableFootprints(
            d,
            0,
            count,
            0,
            Some(f.as_mut_ptr()),
            Some(rows.as_mut_ptr()),
            Some(row_sizes.as_mut_ptr()),
            Some(&mut total),
        );
    }
    (f, rows, row_sizes, total)
}

fn footprint_span(f: &D3D12_PLACED_SUBRESOURCE_FOOTPRINT, rows: u32, row_size: u64) -> u64 {
    let lines = u64::from(rows) * u64::from(f.Footprint.Depth);
    if lines == 0 {
        f.Offset
    } else {
        f.Offset
            .saturating_add(u64::from(f.Footprint.RowPitch).saturating_mul(lines - 1))
            .saturating_add(row_size)
    }
}

fn compute_alias_plan(
    device: &ID3D12Device,
    da: &D3D12_RESOURCE_DESC,
    db: &D3D12_RESOURCE_DESC,
) -> Option<AliasPlan> {
    let (na, nb) = (subresource_count(da), subresource_count(db));
    if na == 0 || nb == 0 {
        return None;
    }
    let (mut fa, ra, sa, total_a) = footprints_for(device, da, na);
    let (mut fb, rb, sb, total_b) = footprints_for(device, db, nb);
    if total_a == u64::MAX || total_b == u64::MAX {
        return None;
    }
    // Subresources in order while both have the same bytes there (mips of
    // a block-compressed texture past its last whole block do not).
    let n = (0..fa.len().min(fb.len()))
        .take_while(|&i| {
            let (x, y) = (fa[i], fb[i]);
            x.Offset != u64::MAX
                && ra[i] != u32::MAX
                && sa[i] != u64::MAX
                && sb[i] != u64::MAX
                && x.Offset == y.Offset
                && x.Footprint.RowPitch == y.Footprint.RowPitch
                && ra[i] == rb[i]
                && x.Footprint.Depth == y.Footprint.Depth
        })
        .count() as u32;
    if n == 0 {
        return None;
    }
    let total = (0..n as usize)
        .map(|i| footprint_span(&fa[i], ra[i], sa[i]).max(footprint_span(&fb[i], rb[i], sb[i])))
        .max()
        .unwrap_or(0);
    if total == 0 {
        return None;
    }
    fa.truncate(n as usize);
    fb.truncate(n as usize);
    Some(AliasPlan {
        fa: fa.into_boxed_slice(),
        fb: fb.into_boxed_slice(),
        n,
        total,
    })
}

/// Records, once per list, the copy of `from`'s host texture into `to`'s:
/// two title textures at one memory that cannot share a host texture
/// (`resource::alias_source`: block-compressed against plain formats of
/// the same bytes, or a cast the host refused), `to` (a placed view's)
/// reading what the GPU wrote through `from`. Through a buffer, each
/// subresource with its own format, as many as have the same bytes per
/// row and rows in both; `to` no longer takes its contents from title
/// memory.
pub(crate) fn alias_copy(list: &HostList, from: This, to: This) {
    if alias_copy_in(list, from, to, &ALIAS_PLANS) {
        ALIAS_COPIES.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    }
}

fn alias_copy_in(list: &HostList, from: This, to: This, plans: &AliasPlanCache) -> bool {
    const COPY_DEST: u32 = 0x400;
    const COPY_SOURCE: u32 = 0x800;
    let (Some(a), Some(b), Some(host)) = (
        crate::resource::host_resource(from),
        crate::resource::host_resource(to),
        host(),
    ) else {
        return false;
    };
    // An odd key: textures' own first uses are keyed by the host texture.
    if !list.first_use(to as usize | 1) {
        return false;
    }
    // SAFETY: live resources.
    let (da, db) = unsafe { (a.GetDesc(), b.GetDesc()) };
    let Some(plan) = plans.get_or_compute(&host.device, &da, &db) else {
        once!(?da, ?db, "an aliased texture not copied: its layouts differ on the host");
        return false;
    };
    let Some(buffer) = host.default_buffer(plan.total) else {
        return false;
    };
    list.keep(&buffer);
    crate::texture::host_writes(list, to);
    crate::list::ensure_state(list, from, COPY_SOURCE);
    crate::list::ensure_state(list, to, COPY_DEST);
    let place = |r: &ID3D12Resource, f: D3D12_PLACED_SUBRESOURCE_FOOTPRINT| D3D12_TEXTURE_COPY_LOCATION {
        // SAFETY: a borrowed pointer the copy does not release.
        pResource: unsafe { std::mem::transmute_copy(r) },
        Type: D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT,
        Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 { PlacedFootprint: f },
    };
    let sub = |r: &ID3D12Resource, i: u32| D3D12_TEXTURE_COPY_LOCATION {
        // SAFETY: as above.
        pResource: unsafe { std::mem::transmute_copy(r) },
        Type: D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX,
        Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 { SubresourceIndex: i },
    };
    let l = &list.list;
    // SAFETY: an open list; live resources in the states named.
    unsafe {
        for i in 0..plan.n {
            l.CopyTextureRegion(&place(&buffer, plan.fa[i as usize]), 0, 0, 0, &sub(&a, i), None);
        }
        l.ResourceBarrier(&[crate::host::transition(
            &buffer,
            D3D12_RESOURCE_STATE_COPY_DEST,
            D3D12_RESOURCE_STATE_COPY_SOURCE,
        )]);
        for i in 0..plan.n {
            l.CopyTextureRegion(&sub(&b, i), 0, 0, 0, &place(&buffer, plan.fb[i as usize]), None);
        }
    }
    static LOGGED: Mutex<Vec<(usize, usize)>> = Mutex::new(Vec::new());
    let mut logged = LOGGED.lock().unwrap_or_else(|e| e.into_inner());
    if logged.len() < 64 && !logged.contains(&(from as usize, to as usize)) {
        logged.push((from as usize, to as usize));
        tracing::info!(
            from = ?from,
            to = ?to,
            a = da.Format.0,
            b = db.Format.0,
            w = da.Width,
            h = da.Height,
            subresources = plan.n,
            "aliased texture copied"
        );
    }
    true
}

/// Has `resource`'s host copy brought up to date at the submission that
/// runs `list`: its staging buffer detiled again if the memory changed, and
/// then copied into the host texture by the submission's own upload list,
/// once for all its lists (`crate::bindless::uploads`).
///
/// shortcut: a title copy into texture memory recorded in the same
/// submission shows from the next one on.
pub(crate) fn upload(list: &HostList, resource: This, texture: &ID3D12Resource) {
    // shortcut: a texture the host copied into keeps the host's contents
    // even if the title later writes its memory too.
    // Once per list for each host texture, however many title textures of
    // its memory the list samples.
    if is_host_written(resource) {
        #[cfg(feature = "diagnostics")]
        {
            static SEEN: Mutex<Option<std::collections::HashSet<usize>>> = Mutex::new(None);
            if SEEN.lock().unwrap_or_else(|e| e.into_inner()).get_or_insert_default().insert(resource as usize) {
                let memory = crate::resource::texture_memory(resource).map(|(m, _)| m);
                // SAFETY: a live resource.
                let desc = unsafe { texture.GetDesc() };
                tracing::debug!(?resource, ?memory, ?desc, "a sampled texture keeps what the host copied into it: not uploaded from title memory");
            }
        }
        return;
    }
    if !list.first_use(texture.as_raw() as usize) {
        return;
    }
    let (Some(staging), Some(copy)) = (staging(resource, texture), crate::resource::host_copy(resource)) else {
        return;
    };
    // Each list recorded the copy before (run perf7: 149 textures uploaded
    // 3,195 times in a frame, 17 ms of GPU time), changed or not.
    list.defer_texture(staging, copy);
}

/// Records into `list` the copy of `buffer` (laid out as `staging` says)
/// into host texture `copy`. Returns the barrier that leaves it readable by
/// shaders, for the caller to record with the others after every copy: one
/// barrier per texture between the copies kept the GPU from overlapping
/// them (new shots copy hundreds of textures, run evict6).
#[must_use]
pub(crate) fn record_upload(
    list: &HostList,
    copy: &Arc<crate::resource::HostTexture>,
    staging: &Staging,
    (buffer, at, subresources): &Upload,
) -> Option<D3D12_RESOURCE_BARRIER> {
    const COPY_DEST: u32 = 0x400;
    const SHADER_RESOURCE: u32 = 0x40 | 0x80;
    if let Some(b) = list.move_texture(copy, COPY_DEST) {
        // SAFETY: an open list; the barrier names a live texture.
        unsafe { list.list.ResourceBarrier(&[b]) };
    }
    copies(list, &copy.texture, staging, (buffer, *at), subresources, String::new);
    list.keep(buffer);
    staging.uploads.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    staging.copied.fetch_add(subresources.len() as u64, std::sync::atomic::Ordering::Relaxed);
    list.move_texture(copy, SHADER_RESOURCE)
}

/// The uploads [`record_upload`] recorded into `resource`'s host texture
/// (tests: counts of their own texture, whatever other tests upload
/// meanwhile).
#[cfg(test)]
pub(crate) fn uploads(resource: This) -> u64 {
    let staging = crate::resource::host_copy(resource).and_then(|c| c.staging.lock().unwrap_or_else(|e| e.into_inner()).clone());
    staging.map_or(0, |s| s.uploads.load(std::sync::atomic::Ordering::Relaxed))
}

/// The subresources those uploads copied (tests, as [`uploads`]).
#[cfg(test)]
pub(crate) fn copied(resource: This) -> u64 {
    let staging = crate::resource::host_copy(resource).and_then(|c| c.staging.lock().unwrap_or_else(|e| e.into_inner()).clone());
    staging.map_or(0, |s| s.copied.load(std::sync::atomic::Ordering::Relaxed))
}


/// Records the copy of `subresources` of `buffer` (laid out as `staging`
/// says) into `texture` (in the copy destination state); `name` names it
/// in diagnostics notes.
#[cfg_attr(not(feature = "diagnostics"), allow(unused_variables))]
fn copies(
    list: &HostList,
    texture: &ID3D12Resource,
    staging: &Staging,
    (buffer, at): (&ID3D12Resource, u64),
    subresources: &[u32],
    name: impl Fn() -> String,
) {
    for &i in subresources {
        let i = i as usize;
        let mut footprint = staging.footprints[i];
        footprint.Offset += at;
        let footprint = &footprint;
        #[cfg(feature = "diagnostics")]
        {
            let f = footprint.Footprint;
            // SAFETY: live resources.
            let (gpu, d) = unsafe { (buffer.GetGPUVirtualAddress(), texture.GetDesc()) };
            let len = subresource_len(
                staging.footprints[i].Offset,
                f.RowPitch as usize,
                staging.rows[i],
                f.Depth,
                staging.size,
            ) as u64;
            list.note_read(
                || {
                    format!(
                        "texture upload {} ({}x{}x{} f{} mips {}) sub {i} of {} from staging {gpu:#x}+{:#x}: offset {:#x} {}x{}x{} pitch {}",
                        name(),
                        d.Width,
                        d.Height,
                        d.DepthOrArraySize,
                        d.Format.0,
                        d.MipLevels,
                        staging.footprints.len(),
                        staging.size,
                        footprint.Offset,
                        f.Width,
                        f.Height,
                        f.Depth,
                        f.RowPitch
                    )
                },
                (gpu + footprint.Offset, len),
            );
        }
        let to = D3D12_TEXTURE_COPY_LOCATION {
            // SAFETY: a borrowed pointer the location never releases.
            pResource: unsafe { std::mem::transmute_copy(texture) },
            Type: D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX,
            Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 {
                SubresourceIndex: i as u32,
            },
        };
        let from = D3D12_TEXTURE_COPY_LOCATION {
            // SAFETY: as above.
            pResource: unsafe { std::mem::transmute_copy(buffer) },
            Type: D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT,
            Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 {
                PlacedFootprint: *footprint,
            },
        };
        // SAFETY: an open list; both locations name live resources.
        unsafe { list.list.CopyTextureRegion(&to, 0, 0, 0, &from, None) };
    }
}

/// What a refresh gives: a staging buffer, where its space starts in it,
/// and the subresources detiled into it.
pub(crate) type Upload = (ID3D12Resource, u64, Vec<u32>);

/// Runs `f(i)` for each `i` below `n` on up to a quarter of the CPUs (four
/// at most: the title's own threads need the rest), here when that is
/// fewer than two.
pub(crate) fn in_parallel(n: usize, f: impl Fn(usize) + Sync) {
    use std::sync::atomic::{AtomicUsize, Ordering::Relaxed};
    let threads = std::thread::available_parallelism().map_or(4, |n| n.get() / 4).clamp(1, 4).min(n);
    if threads < 2 {
        (0..n).for_each(f);
        return;
    }
    let next = AtomicUsize::new(0);
    std::thread::scope(|scope| {
        for _ in 0..threads {
            scope.spawn(|| {
                loop {
                    let i = next.fetch_add(1, Relaxed);
                    if i >= n {
                        break;
                    }
                    f(i);
                }
            });
        }
    });
}

/// [`Staging::refresh`] of each of `work`, those whose memory may have
/// changed on several threads at once (reading title memory and detiling
/// cost ~6 ms of frame a MB on one thread in a new shot, run evict5).
pub(crate) fn refresh_all(work: &[(Arc<Staging>, Arc<crate::resource::HostTexture>)]) -> Vec<Option<Upload>> {
    // Most find nothing changed (their trackers' clean bits say so): those
    // are not handed to threads.
    let changed: Vec<usize> = (0..work.len()).filter(|&i| work[i].0.maybe_changed()).collect();
    let results: Vec<Mutex<Option<Upload>>> = changed.iter().map(|_| Mutex::new(None)).collect();
    in_parallel(changed.len(), |k| {
        *results[k].lock().unwrap_or_else(|e| e.into_inner()) = work[changed[k]].0.refresh();
    });
    let mut out: Vec<Option<Upload>> = vec![None; work.len()];
    for (k, r) in results.into_iter().enumerate() {
        out[changed[k]] = r.into_inner().unwrap_or_else(|e| e.into_inner());
    }
    out
}

impl Staging {
    /// Whether its memory may have changed since it was last detiled (an
    /// untracked texture always may).
    pub(crate) fn maybe_changed(&self) -> bool {
        self.tracked.as_ref().is_none_or(|t| !t.clean())
    }

    /// A staging buffer with the texture memory detiled into it if it
    /// changed since the last time (and is mapped at all), and the
    /// subresources that changed (only those are detiled and to be copied).
    pub(crate) fn refresh(&self) -> Option<Upload> {
        use std::sync::atomic::Ordering::Relaxed;
        let started = std::time::Instant::now();
        let (read, buffer) = self.refresh_now();
        let preparing = &crate::host::PREPARING;
        preparing[2].fetch_add(started.elapsed().as_micros() as u64, Relaxed);
        preparing[3].fetch_add(read as u64, Relaxed);
        buffer
    }

    /// Returns the bytes of title memory read, and the buffer it detiled
    /// into with the subresources detiled.
    fn refresh_now(&self) -> (usize, Option<Upload>) {
        let all = || (0..self.footprints.len() as u32).collect::<Vec<_>>();
        let size = self.layout.size as usize;
        let start = self.memory as usize;
        let mut last = self.last.lock().unwrap_or_else(|e| e.into_inner());
        let last = &mut *last;
        if let Some(t) = &self.tracked {
            // Most refreshes find nothing: the tracker's dense bit says so
            // without its lock (123,000 refreshes a frame, run evict3).
            if t.clean() {
                return (0, None);
            }
            let ranges = t.take(start, start + size);
            if ranges.is_empty() {
                return (0, None);
            }
            // Only the subresources in the pages that changed (streamed
            // textures change page by page; a whole re-read and detile of
            // each cost up to 430 MB a frame, run integ2).
            let n = self.footprints.len();
            let span = |i: usize| self.spans.get(i).copied().unwrap_or((0, size));
            let touched: Vec<usize> = (0..n)
                .filter(|&i| {
                    let (a, b) = span(i);
                    ranges.iter().any(|&(f, e)| f - start < b && a < e - start)
                })
                .collect();
            // Read through the copy, which stands unmapped pages: memory of
            // big pools is too slow to ask page by page. Not kept: it would
            // double each texture's system memory. Only the touched spans
            // are written, and only they are hashed and detiled.
            let mut scratch = Scratch::take(size);
            let read = &mut scratch.0[..size];
            let mut bytes = 0;
            let mut done: Vec<(usize, usize)> = Vec::new();
            for &i in &touched {
                let (a, b) = span(i);
                if !done.contains(&(a, b)) {
                    // SAFETY: inside the buffer of the texture's size.
                    crate::host::copy_mapped(unsafe { read.as_mut_ptr().add(a) }, self.memory.wrapping_add(a), b - a, false);
                    (bytes, _) = (bytes + b - a, done.push((a, b)));
                }
            }
            // A changed page that holds only the subresource's bytes says it
            // changed. Pages it shares with other data the title writes
            // (small textures): its own bytes decide. (Hashing a movie frame
            // cost 8 ms a frame.)
            let page = crate::watch::PAGE;
            last.hashes.resize(n, None);
            let fresh: Vec<u32> = touched
                .into_iter()
                .filter(|&i| {
                    let (a, b) = span(i);
                    let (a0, b0) = (start + a, start + b);
                    if ranges.iter().any(|&(f, e)| f.max(a0).next_multiple_of(page) + page <= e.min(b0)) {
                        last.hashes[i] = None;
                        return true;
                    }
                    let hash = blake3::hash(&read[a..b]);
                    last.hashes[i].replace(hash) != Some(hash)
                })
                .map(|i| i as u32)
                .collect();
            if fresh.is_empty() {
                return (bytes, None);
            }
            let buffer = self.detile(read, &fresh);
            if buffer.is_none() {
                for &i in &fresh {
                    last.hashes[i as usize] = None;
                }
            }
            #[cfg(feature = "diagnostics")]
            crate::copystat::add(crate::copystat::C::TexWatched, bytes as u64);
            return (bytes, buffer.map(|(b, at)| (b, at, fresh)));
        }
        if !committed(self.memory, size) {
            #[cfg(feature = "diagnostics")]
            tracing::debug!(
                memory = ?self.memory,
                size,
                desc = ?self.layout.desc,
                "a sampled texture's memory is not committed"
            );
            // Never uploaded: zeros, as a committed texture starts (one in
            // a heap of textures starts with whatever the range held).
            if last.hash.is_some() {
                return (0, None);
            }
            let zeros = vec![0u8; size];
            let every = all();
            let buffer = self.detile(&zeros, &every);
            last.hash = buffer.is_some().then(|| blake3::hash(&zeros));
            return (0, buffer.map(|(b, at)| (b, at, every)));
        }
        // SAFETY: committed memory of the texture's layout size.
        let tiled = unsafe { std::slice::from_raw_parts(self.memory, size) };
        let hash = blake3::hash(tiled);
        #[cfg(feature = "diagnostics")]
        crate::copystat::add(crate::copystat::C::TexHashed, size as u64);
        if last.hash == Some(hash) {
            return (size, None);
        }
        let every = all();
        let buffer = self.detile(tiled, &every);
        last.hash = buffer.is_some().then_some(hash);
        #[cfg(feature = "diagnostics")]
        crate::copystat::add(crate::copystat::C::TexChanged, size as u64);
        (size, buffer.map(|(b, at)| (b, at, every)))
    }

    /// Detiles `subresources` of `tiled` (the texture's memory) into a
    /// staging buffer (`None` if none could be had or a subresource could
    /// not be read).
    fn detile(&self, tiled: &[u8], subresources: &[u32]) -> Option<(ID3D12Resource, u64)> {
        let (buffer, at, data) = staging_space(self.size)?;
        #[cfg(feature = "diagnostics")]
        // Scanned only when its message is wanted: a pass over every
        // texture read, in timing runs too.
        if tracing::enabled!(tracing::Level::DEBUG) {
            if tiled.iter().all(|&b| b == 0) {
                tracing::debug!(
                    memory = ?self.memory,
                    desc = ?self.layout.desc,
                    "a sampled texture's memory is all zero (not streamed yet, or a failed BCPack read)"
                );
            } else {
                tracing::debug!(memory = ?self.memory, desc = ?self.layout.desc, "a sampled texture detiled");
            }
        }
        for &i in subresources {
            let (footprint, rows) = (&self.footprints[i as usize], self.rows[i as usize]);
            let (mip, slice) = (i % self.mips, i / self.mips);
            let f = footprint.Footprint;
            let pitch = f.RowPitch as usize;
            let len = subresource_len(footprint.Offset, pitch, rows, f.Depth, self.size);
            // SAFETY: `subresource_len` clamps `len` to `self.size - footprint.Offset`,
            // keeping the slice strictly inside the mapped staging buffer even when
            // GetCopyableFootprints does not pad the final row of the last subresource.
            let linear = unsafe { std::slice::from_raw_parts_mut(data.add(footprint.Offset as usize), len) };
            if let Err(e) =
                self.layout
                    .detile(tiled, mip, slice, linear, pitch, pitch * rows as usize)
            {
                tracing::warn!(?e, mip, slice, "texture not detiled");
                return None;
            }
        }
        Some((buffer, at))
    }
}

/// A buffer for the title memory a refresh reads, kept by the thread for
/// its next refresh: a fresh one for each texture cost a zero-fill and a
/// page fault per 4 KiB (129,000 faults in a frame, run placed1).
struct Scratch(Vec<u8>);

struct ScratchCache(std::cell::Cell<Vec<u8>>);

#[cfg(feature = "diagnostics")]
static RETAINED_SCRATCH: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);

#[cfg(feature = "diagnostics")]
impl Drop for ScratchCache {
    fn drop(&mut self) {
        RETAINED_SCRATCH.fetch_sub(self.0.get_mut().capacity(), std::sync::atomic::Ordering::Relaxed);
    }
}

thread_local! {
    static SCRATCH: ScratchCache = const { ScratchCache(std::cell::Cell::new(Vec::new())) };
}

impl Scratch {
    /// The thread's buffer, at least `size` bytes.
    fn take(size: usize) -> Self {
        let mut v = SCRATCH.with(|cache| cache.0.take());
        #[cfg(feature = "diagnostics")]
        RETAINED_SCRATCH.fetch_sub(v.capacity(), std::sync::atomic::Ordering::Relaxed);
        if v.len() < size {
            v.resize(size, 0);
        }
        Scratch(v)
    }
}

impl Drop for Scratch {
    fn drop(&mut self) {
        // The largest textures' buffers are not kept.
        const KEEP: usize = 32 << 20;
        if self.0.len() <= KEEP {
            SCRATCH.with(|cache| {
                let retained = std::mem::take(&mut self.0);
                #[cfg(feature = "diagnostics")]
                RETAINED_SCRATCH.fetch_add(retained.capacity(), std::sync::atomic::Ordering::Relaxed);
                let previous = cache.0.replace(retained);
                #[cfg(feature = "diagnostics")]
                RETAINED_SCRATCH.fetch_sub(previous.capacity(), std::sync::atomic::Ordering::Relaxed);
                drop(previous);
            });
        }
    }
}

/// Bytes available in the staging buffer for a subresource at `offset`.
/// D3D12's `GetCopyableFootprints` omits trailing row-pitch padding on the
/// final row of the final subresource, so `pitch * rows * depth` can exceed
/// `total - offset`.
fn subresource_len(offset: u64, pitch: usize, rows: u32, depth: u32, total: u64) -> usize {
    let padded = pitch
        .saturating_mul(rows as usize)
        .saturating_mul(depth as usize);
    let remaining = usize::try_from(total.saturating_sub(offset)).unwrap_or(usize::MAX);
    padded.min(remaining)
}

/// Whether `[at, at + len)` is committed, readable memory (streamed textures
/// may be unmapped until their pages arrive).
pub(crate) fn committed(at: *const u8, len: usize) -> bool {
    use windows_sys::Win32::System::Memory::{
        MEM_COMMIT, MEMORY_BASIC_INFORMATION, PAGE_GUARD, PAGE_NOACCESS, VirtualQuery,
    };
    let mut p = at as usize;
    let Some(end) = p.checked_add(len) else {
        return false;
    };
    while p < end {
        // SAFETY: plain data filled in by the query.
        let mut info: MEMORY_BASIC_INFORMATION = unsafe { std::mem::zeroed() };
        // SAFETY: a valid out pointer of the right size.
        let n = unsafe { VirtualQuery(p as *const _, &mut info, size_of_val(&info)) };
        let unreadable = info.Protect == 0
            || info.Protect & (PAGE_NOACCESS | PAGE_GUARD) != 0;
        if n == 0 || info.State != MEM_COMMIT || unreadable {
            return false;
        }
        let next = (info.BaseAddress as usize).saturating_add(info.RegionSize);
        if next <= p {
            return false;
        }
        p = next;
    }
    true
}

/// The staging state of a title texture's host texture, made on first use
/// (one for all title textures of its memory).
pub(crate) fn staging(resource: This, texture: &ID3D12Resource) -> Option<Arc<Staging>> {
    let copy = crate::resource::host_copy(resource)?;
    let mut held = copy.staging.lock().unwrap_or_else(|e| e.into_inner());
    if let Some(s) = held.as_ref() {
        return Some(s.clone());
    }
    let (memory, desc) = crate::resource::texture_memory(resource)?;
    let Some(layout) = crate::resource::texture_layout(&desc).map(|t| t.planes[0].layout.clone())
    else {
        once!(
            ?desc,
            "texture contents without a console layout not uploaded"
        );
        return None;
    };
    if layout.desc.swizzle.xor() {
        once!(?desc, "texture contents in an XOR mode not uploaded");
        return None;
    }
    let host = host()?;
    // SAFETY: a live resource.
    let d = unsafe { texture.GetDesc() };
    let count = if d.Dimension == D3D12_RESOURCE_DIMENSION_TEXTURE3D {
        u32::from(d.MipLevels)
    } else {
        u32::from(d.MipLevels) * u32::from(d.DepthOrArraySize)
    };
    let mut footprints = vec![D3D12_PLACED_SUBRESOURCE_FOOTPRINT::default(); count as usize];
    let mut rows = vec![0u32; count as usize];
    let mut total = 0;
    // SAFETY: outputs for `count` subresources.
    unsafe {
        host.device.GetCopyableFootprints(
            &d,
            0,
            count,
            0,
            Some(footprints.as_mut_ptr()),
            Some(rows.as_mut_ptr()),
            None,
            Some(&mut total),
        );
    }
    // Both D3D12 copies and mapped CPU pointers use these offsets.
    if footprints.iter().any(|f| f.Offset >= total) {
        once!(?desc, total, "texture upload footprint outside its staging buffer");
        return None;
    }
    tracing::debug!(?desc, total, "texture contents uploaded from title memory");
    let layout_size = layout.size as usize;
    let spans = spans(&layout, count, u32::from(d.MipLevels));
    let s = Arc::new(Staging {
        size: total,
        footprints,
        rows,
        layout,
        mips: u32::from(d.MipLevels),
        memory,
        tracked: crate::watch::track(memory as usize, layout_size, true),
        spans,
        last: Mutex::default(),
        uploads: Default::default(),
        copied: Default::default(),
    });
    *held = Some(s.clone());
    Some(s)
}

#[cfg(test)]
mod tests {
    use std::ffi::c_void;

    use windows::Win32::Graphics::Dxgi::Common::{DXGI_FORMAT, DXGI_SAMPLE_DESC};

    use super::*;
    use crate::object::{Iid, S_OK, slot};

    #[test]
    fn subresource_len_stays_inside_staging_total() {
        // A 4x4 RGBA8 texture has RowPitch=256, rows=4, last_row=16 -> total=784.
        // Padded size (1024) must be clamped to 784 so from_raw_parts_mut stays in bounds.
        assert_eq!(subresource_len(0, 256, 4, 1, 784), 784);
        assert_eq!(subresource_len(512, 256, 4, 1, 784), 272);
        assert_eq!(subresource_len(0, 256, 4, 1, 2048), 1024);
        // A 10x6 RGBA8 texture with two mips (the last row contains 20 bytes).
        let total = 1536 + 2 * 256 + 20;
        assert_eq!(subresource_len(0, 256, 6, 1, total), 1536);
        assert_eq!(subresource_len(1536, 256, 3, 1, total), 532);
        assert_eq!(subresource_len(total, 256, 3, 1, total), 0);
        assert_eq!(subresource_len(total + 1, 256, 3, 1, total), 0);
    }

    #[test]
    fn last_bc4_mip_ends_after_the_active_block() {
        use windows::Win32::Graphics::Dxgi::Common::{DXGI_FORMAT_BC4_UNORM, DXGI_SAMPLE_DESC};

        let Some(host) = host() else {
            return;
        };
        let desc = D3D12_RESOURCE_DESC {
            Dimension: D3D12_RESOURCE_DIMENSION_TEXTURE2D,
            Width: 8,
            Height: 4,
            DepthOrArraySize: 1,
            MipLevels: 2,
            Format: DXGI_FORMAT_BC4_UNORM,
            SampleDesc: DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
            ..Default::default()
        };
        let mut footprints = [D3D12_PLACED_SUBRESOURCE_FOOTPRINT::default(); 2];
        let mut rows = [0; 2];
        let mut row_bytes = [0; 2];
        let mut total = 0;
        // SAFETY: each output array has room for both subresources.
        unsafe {
            host.device.GetCopyableFootprints(
                &desc,
                0,
                2,
                0,
                Some(footprints.as_mut_ptr()),
                Some(rows.as_mut_ptr()),
                Some(row_bytes.as_mut_ptr()),
                Some(&mut total),
            );
        }
        let last = footprints[1];
        assert_eq!(row_bytes[1], 8);
        assert!(u64::from(last.Footprint.RowPitch) > row_bytes[1]);
        assert_eq!(
            subresource_len(
                last.Offset,
                last.Footprint.RowPitch as usize,
                rows[1],
                last.Footprint.Depth,
                total
            ),
            row_bytes[1] as usize
        );
    }

    #[test]
    fn committed_rejects_overflow_and_noaccess_pages() {
        use windows_sys::Win32::System::Memory::{
            MEM_COMMIT, MEM_RELEASE, MEM_RESERVE, PAGE_NOACCESS, PAGE_READWRITE, VirtualAlloc,
            VirtualFree,
        };
        assert!(!committed(usize::MAX as *const u8, 16));
        // SAFETY: reserving and committing one private page for the test, freed below.
        let noaccess =
            unsafe { VirtualAlloc(std::ptr::null(), 4096, MEM_RESERVE | MEM_COMMIT, PAGE_NOACCESS) };
        assert!(!noaccess.is_null());
        assert!(!committed(noaccess.cast::<u8>(), 4096));
        // SAFETY: freeing the allocation above.
        unsafe { VirtualFree(noaccess, 0, MEM_RELEASE) };

        // Negative control: a readable committed page still passes.
        // SAFETY: reserving and committing one private page for the test, freed below.
        let rw =
            unsafe { VirtualAlloc(std::ptr::null(), 4096, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE) };
        assert!(!rw.is_null());
        assert!(committed(rw.cast::<u8>(), 4096));
        // SAFETY: freeing the allocation above.
        unsafe { VirtualFree(rw, 0, MEM_RELEASE) };
    }

    struct ClearOnDrop(&'static AliasPlanCache);

    impl Drop for ClearOnDrop {
        fn drop(&mut self) {
            self.0.clear();
        }
    }

    const RESOURCE_IID: Iid = crate::object::guid(
        0x696442be,
        0xa72e,
        0x4059,
        [0xbc, 0x79, 0x5b, 0x5c, 0x98, 0x04, 0x0f, 0xad],
    );

    struct OwnedTitleResource(This);

    impl Drop for OwnedTitleResource {
        fn drop(&mut self) {
            if !self.0.is_null() {
                type Release = extern "system" fn(This) -> u32;
                slot::<Release>(self.0, 0x10)(self.0);
            }
        }
    }

    fn tex2d_desc(
        width: u64,
        height: u32,
        format: i32,
        mip_levels: u16,
        flags: i32,
    ) -> D3D12_RESOURCE_DESC {
        D3D12_RESOURCE_DESC {
            Dimension: D3D12_RESOURCE_DIMENSION_TEXTURE2D,
            Alignment: 0,
            Width: width,
            Height: height,
            DepthOrArraySize: 1,
            MipLevels: mip_levels,
            Format: DXGI_FORMAT(format),
            SampleDesc: DXGI_SAMPLE_DESC {
                Count: 1,
                Quality: 0,
            },
            Layout: D3D12_TEXTURE_LAYOUT_UNKNOWN,
            Flags: D3D12_RESOURCE_FLAGS(flags),
        }
    }

    fn committed_texture(
        width: u64,
        height: u32,
        format: i32,
        mip_levels: u16,
        flags: i32,
    ) -> OwnedTitleResource {
        assert_eq!(
            size_of::<D3D12_RESOURCE_DESC>(),
            size_of::<crate::resource::ResourceDesc>()
        );
        assert_eq!(
            align_of::<D3D12_RESOURCE_DESC>(),
            align_of::<crate::resource::ResourceDesc>()
        );
        assert_eq!(
            size_of::<D3D12_HEAP_PROPERTIES>(),
            size_of::<crate::resource::HeapProperties>()
        );
        assert_eq!(
            align_of::<D3D12_HEAP_PROPERTIES>(),
            align_of::<crate::resource::HeapProperties>()
        );
        let heap = D3D12_HEAP_PROPERTIES {
            Type: D3D12_HEAP_TYPE_DEFAULT,
            ..Default::default()
        };
        let desc = tex2d_desc(width, height, format, mip_levels, flags);
        let mut out = std::ptr::null();
        let hr = crate::resource::create_committed(
            std::ptr::null(),
            std::ptr::from_ref(&heap).cast(),
            0,
            std::ptr::from_ref(&desc).cast(),
            0,
            std::ptr::null::<c_void>(),
            &RESOURCE_IID,
            &mut out,
        );
        assert_eq!(hr, S_OK);
        assert!(!out.is_null());
        OwnedTitleResource(out)
    }

    fn readback_buffer(device: &ID3D12Device, bytes: u64) -> ID3D12Resource {
        let heap = D3D12_HEAP_PROPERTIES {
            Type: D3D12_HEAP_TYPE_READBACK,
            ..Default::default()
        };
        let desc = D3D12_RESOURCE_DESC {
            Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
            Width: bytes.max(1),
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
        let mut out: Option<ID3D12Resource> = None;
        // SAFETY: plain creation of a readback buffer of `bytes` bytes.
        unsafe {
            device
                .CreateCommittedResource(
                    &heap,
                    D3D12_HEAP_FLAG_NONE,
                    &desc,
                    D3D12_RESOURCE_STATE_COPY_DEST,
                    None,
                    &mut out,
                )
                .expect("readback buffer");
        }
        out.expect("readback resource")
    }

    fn placed_loc(
        resource: &ID3D12Resource,
        footprint: D3D12_PLACED_SUBRESOURCE_FOOTPRINT,
    ) -> D3D12_TEXTURE_COPY_LOCATION {
        D3D12_TEXTURE_COPY_LOCATION {
            // SAFETY: borrowed pointer the copy does not release.
            pResource: unsafe { std::mem::transmute_copy(resource) },
            Type: D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT,
            Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 {
                PlacedFootprint: footprint,
            },
        }
    }

    fn subresource_loc(resource: &ID3D12Resource, index: u32) -> D3D12_TEXTURE_COPY_LOCATION {
        D3D12_TEXTURE_COPY_LOCATION {
            // SAFETY: borrowed pointer the copy does not release.
            pResource: unsafe { std::mem::transmute_copy(resource) },
            Type: D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX,
            Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 {
                SubresourceIndex: index,
            },
        }
    }

    #[test]
    fn alias_plan_caches_compatible_and_incompatible_pairs() {
        let Some(host) = host() else {
            return; // no host GPU here
        };
        let plans = AliasPlanCache::new();

        // Compatible block-compressed <-> R32G32 pair across all 3 mips.
        let bc1 = tex2d_desc(16, 16, 71, 3, 0);
        let r32g32 = tex2d_desc(4, 4, 17, 3, 4);
        let first = plans
            .get_or_compute(&host.device, &r32g32, &bc1)
            .expect("compatible alias plan");
        assert_eq!(first.n, 3);
        assert_eq!(first.fa.len(), 3);
        assert_eq!(first.fb.len(), 3);
        assert!(first.total > 0);
        assert_eq!(plans.len(), 1);
        assert_eq!(
            plans.computations.load(std::sync::atomic::Ordering::Relaxed),
            1
        );

        let second = plans
            .get_or_compute(&host.device, &r32g32, &bc1)
            .expect("cached alias plan");
        assert!(Arc::ptr_eq(&first, &second));
        assert_eq!(plans.len(), 1);
        assert_eq!(
            plans.computations.load(std::sync::atomic::Ordering::Relaxed),
            1
        );

        // Incompatible dimensions/bytes-per-row: `None` is also cached cleanly.
        let small = tex2d_desc(4, 4, 42, 1, 4);
        let mismatch = tex2d_desc(8, 8, 28, 1, 0);
        assert!(plans.get_or_compute(&host.device, &small, &mismatch).is_none());
        assert_eq!(plans.len(), 2);
        assert!(plans.get_or_compute(&host.device, &small, &mismatch).is_none());
        assert_eq!(plans.len(), 2);
        assert_eq!(
            plans.computations.load(std::sync::atomic::Ordering::Relaxed),
            2
        );

        // Exceeding `ALIAS_PLAN_CAP` clears cleanly and never grows past the cap.
        for i in 1..=(ALIAS_PLAN_CAP as u64 + 4) {
            let a = tex2d_desc(i * 4, 4, 42, 1, 4);
            let b = tex2d_desc(i * 4, 4, 28, 1, 0);
            let _ = plans.get_or_compute(&host.device, &a, &b);
            assert!(plans.len() <= ALIAS_PLAN_CAP);
        }
        assert_eq!(plans.len(), 6);
    }

    #[test]
    fn footprint_span_computes_exact_subresource_byte_extent() {
        let base = D3D12_PLACED_SUBRESOURCE_FOOTPRINT::default();
        let footprint = |offset, row_pitch, depth| D3D12_PLACED_SUBRESOURCE_FOOTPRINT {
            Offset: offset,
            Footprint: D3D12_SUBRESOURCE_FOOTPRINT {
                RowPitch: row_pitch,
                Depth: depth,
                ..base.Footprint
            },
        };
        assert_eq!(
            footprint_span(&footprint(512, 256, 1), 4, 32),
            512 + 256 * 3 + 32
        );
        assert_eq!(
            footprint_span(&footprint(512, 256, 2), 4, 16),
            512 + 256 * 7 + 16
        );
        assert_eq!(footprint_span(&footprint(512, 256, 1), 0, 32), 512);
        assert_eq!(footprint_span(&footprint(512, 256, 0), 4, 32), 512);
    }

    #[test]
    fn alias_copy_caches_plans_and_copies_bytes() {
        static PLANS: AliasPlanCache = AliasPlanCache::new();
        let Some(host) = host() else {
            return; // no host GPU here
        };
        let _ = host.log_messages();
        PLANS.clear();
        let _clear = ClearOnDrop(&PLANS);

        // Title resources declared before `HostList`s so lists drop first.
        let from = committed_texture(8, 8, 42, 1, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS.0);
        let to = committed_texture(8, 8, 28, 1, D3D12_RESOURCE_FLAG_NONE.0);
        let from_bc = committed_texture(4, 4, 17, 1, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS.0);
        let to_bc = committed_texture(16, 16, 71, 1, D3D12_RESOURCE_FLAG_NONE.0);

        let a = crate::resource::host_resource(from.0).expect("from host texture");
        let b = crate::resource::host_resource(to.0).expect("to host texture");
        let a_bc = crate::resource::host_resource(from_bc.0).expect("from_bc host texture");
        let b_bc = crate::resource::host_resource(to_bc.0).expect("to_bc host texture");
        // SAFETY: live host textures.
        let (da, db, da_bc, db_bc) =
            unsafe { (a.GetDesc(), b.GetDesc(), a_bc.GetDesc(), b_bc.GetDesc()) };

        let (fa, ra, sa, total_a) = footprints_for(&host.device, &da, 1);
        let (fb, rb, sb, total_b) = footprints_for(&host.device, &db, 1);
        let (fa_bc, ra_bc, sa_bc, total_a_bc) = footprints_for(&host.device, &da_bc, 1);
        let (fb_bc, rb_bc, sb_bc, total_b_bc) = footprints_for(&host.device, &db_bc, 1);

        let readback = readback_buffer(&host.device, total_b);
        let readback_bc = readback_buffer(&host.device, total_b_bc);
        let list = host.new_list().expect("host list 1");
        let list2 = host.new_list().expect("host list 2");

        const COPY_DEST: u32 = 0x400;
        const COPY_SOURCE: u32 = 0x800;

        #[allow(clippy::too_many_arguments)]
        let upload_and_alias = |target_list: &HostList,
                                src_res: This,
                                dst_res: This,
                                src_tex: &ID3D12Resource,
                                dst_tex: &ID3D12Resource,
                                src_fp: D3D12_PLACED_SUBRESOURCE_FOOTPRINT,
                                dst_fp: D3D12_PLACED_SUBRESOURCE_FOOTPRINT,
                                rows: u32,
                                row_bytes: usize,
                                total_up: u64,
                                rb: &ID3D12Resource,
                                seed: u8| {
            let (up, up_ptr) = target_list.upload(host, total_up).expect("upload buffer");
            // SAFETY: `up_ptr` is mapped for `total_up` bytes.
            let up_slice = unsafe { std::slice::from_raw_parts_mut(up_ptr, total_up as usize) };
            up_slice.fill(0);
            let mut expected = Vec::with_capacity(rows as usize * row_bytes);
            for r in 0..rows as usize {
                let row_off = src_fp.Offset as usize + r * src_fp.Footprint.RowPitch as usize;
                for c in 0..row_bytes {
                    let v = seed.wrapping_add((r * 37 + c * 13) as u8);
                    up_slice[row_off + c] = v;
                    expected.push(v);
                }
            }
            crate::list::ensure_state(target_list, src_res, COPY_DEST);
            // SAFETY: open list; valid upload buffer and destination texture in COPY_DEST.
            unsafe {
                target_list.list.CopyTextureRegion(
                    &subresource_loc(src_tex, 0),
                    0,
                    0,
                    0,
                    &placed_loc(&up, src_fp),
                    None,
                );
            }
            assert!(alias_copy_in(target_list, src_res, dst_res, &PLANS));
            // Calling a second time on the same open list is suppressed by `first_use`.
            assert!(!alias_copy_in(target_list, src_res, dst_res, &PLANS));
            crate::list::ensure_state(target_list, dst_res, COPY_SOURCE);
            // SAFETY: open list; valid source texture in COPY_SOURCE and readback buffer in COPY_DEST.
            unsafe {
                target_list.list.CopyTextureRegion(
                    &placed_loc(rb, dst_fp),
                    0,
                    0,
                    0,
                    &subresource_loc(dst_tex, 0),
                    None,
                );
            }
            expected
        };

        let verify_readback = |rb: &ID3D12Resource,
                               fp: D3D12_PLACED_SUBRESOURCE_FOOTPRINT,
                               rows: u32,
                               row_bytes: usize,
                               total_rb: u64,
                               expected: &[u8]| {
            let mut mapped = std::ptr::null_mut();
            let range = D3D12_RANGE {
                Begin: 0,
                End: total_rb as usize,
            };
            // SAFETY: `rb` is a `total_rb`-byte readback buffer and GPU work has completed.
            unsafe {
                rb.Map(0, Some(&range), Some(&mut mapped))
                    .expect("map readback");
                let bytes = std::slice::from_raw_parts(mapped.cast::<u8>(), total_rb as usize);
                for r in 0..rows as usize {
                    let off = fp.Offset as usize + r * fp.Footprint.RowPitch as usize;
                    assert_eq!(
                        &bytes[off..off + row_bytes],
                        &expected[r * row_bytes..(r + 1) * row_bytes],
                        "row {r} mismatch"
                    );
                }
                rb.Unmap(0, None);
            }
        };

        // 1. Both R32_UINT -> R8G8B8A8_UNORM and R32G32_UINT -> BC1_UNORM copy byte-exact
        //    through per-copy scratch buffers and mark destinations as host-written.
        let expected_1 = upload_and_alias(
            &list,
            from.0,
            to.0,
            &a,
            &b,
            fa[0],
            fb[0],
            ra[0],
            sa[0] as usize,
            total_a,
            &readback,
            0x11,
        );
        let expected_bc = upload_and_alias(
            &list,
            from_bc.0,
            to_bc.0,
            &a_bc,
            &b_bc,
            fa_bc[0],
            fb_bc[0],
            ra_bc[0],
            sa_bc[0] as usize,
            total_a_bc,
            &readback_bc,
            0xA5,
        );
        assert_eq!(PLANS.len(), 2);
        assert_eq!(
            PLANS.computations.load(std::sync::atomic::Ordering::Relaxed),
            2
        );
        assert!(!is_host_written(to.0));
        assert!(!is_host_written(to_bc.0));
        list.close();
        host.execute(&[&list]);
        assert!(is_host_written(to.0));
        assert!(is_host_written(to_bc.0));
        verify_readback(&readback, fb[0], rb[0], sb[0] as usize, total_b, &expected_1);
        verify_readback(
            &readback_bc,
            fb_bc[0],
            rb_bc[0],
            sb_bc[0] as usize,
            total_b_bc,
            &expected_bc,
        );
        list.reset();

        // 2. Subsequent alias copies across two open lists reuse the cached plans without recomputing.
        let expected_2 = upload_and_alias(
            &list,
            from.0,
            to.0,
            &a,
            &b,
            fa[0],
            fb[0],
            ra[0],
            sa[0] as usize,
            total_a,
            &readback,
            0x79,
        );
        let expected_bc_2 = upload_and_alias(
            &list2,
            from_bc.0,
            to_bc.0,
            &a_bc,
            &b_bc,
            fa_bc[0],
            fb_bc[0],
            ra_bc[0],
            sa_bc[0] as usize,
            total_a_bc,
            &readback_bc,
            0x3C,
        );
        assert_eq!(
            PLANS.computations.load(std::sync::atomic::Ordering::Relaxed),
            2
        );
        list.close();
        list2.close();
        host.execute(&[&list, &list2]);
        verify_readback(&readback, fb[0], rb[0], sb[0] as usize, total_b, &expected_2);
        verify_readback(
            &readback_bc,
            fb_bc[0],
            rb_bc[0],
            sb_bc[0] as usize,
            total_b_bc,
            &expected_bc_2,
        );
        list.reset();
        list2.reset();
        assert!(
            host.log_messages().is_none(),
            "D3D12 debug layer reported an error"
        );
    }
}
