//! Views shaders reach through the resource heap itself (decision 0007,
//! "Heap-indexed views"). Grand Theft Auto VI is largely bindless: its
//! shaders load descriptors at indices they take from root constants or
//! memory (`HostShader::heap`), so no draw names the views it reads. Every
//! view the title writes into a shader-visible heap gets its host view when
//! it is written (`descriptor::host`), and each heap keeps here what its
//! slots view that title memory feeds: ranges of host buffer copies, and
//! textures whose contents the title writes. A submission whose lists run
//! such shaders first brings all of them up to date (what changed since,
//! `crate::watch`).

use std::collections::{BTreeMap, HashMap};
use std::sync::atomic::{AtomicU64, Ordering::Relaxed};
use std::sync::{Arc, Mutex};

use crate::descriptor::View;
use crate::host::HostList;
use crate::object::This;
use crate::shadow::Shadow;

/// What a title heap slot views that title memory feeds.
#[derive(Clone)]
enum Seen {
    /// `len` bytes at `address` of a buffer's host copy; unordered access.
    Buffer(Arc<Shadow>, u64, u64, bool),
    /// A title texture (resource object) whose contents the title writes.
    Texture(usize),
    /// A view left without a host view (textures the title placed itself),
    /// by kind (diagnostics: `KINDS`), with a placed texture's address and
    /// console bytes.
    Left(#[cfg_attr(not(feature = "diagnostics"), allow(dead_code))] (usize, u64, u64)),
}

/// Kinds of views left without a host view, for the census.
#[cfg(feature = "diagnostics")]
const KINDS: [&str; 4] = ["buffer", "unordered buffer", "placed texture", "other"];

/// A heap's slots that view title memory, and the distinct buffer ranges
/// and textures among them (with how many slots view each).
#[derive(Default)]
pub(crate) struct Viewed {
    inner: Mutex<Inner>,
}

type Range = (usize, u64, u64, bool);

/// A merged range: copy, start, bytes, unordered access, the copy's bit in
/// the change tracker (`crate::watch::clean_bit`).
type Hull = (Arc<Shadow>, u64, u64, bool, Option<u32>);

#[derive(Default)]
struct Inner {
    slots: HashMap<u32, Seen>,
    ranges: BTreeMap<Range, (u32, Arc<Shadow>)>,
    /// Distinct title-written textures viewed in this heap: `(slot count,
    /// cached watch::DIRTY bit, idle stamp epoch)` so clean and idle textures
    /// skip `Resource::host` and `HostTexture::staging` locks during [`uploads`].
    textures: BTreeMap<usize, (u32, Option<u32>, u64)>,
    /// `ranges` merged per copy and kind ([`Viewed::ranges`]), kept current
    /// as ranges come and go (each list running heap shaders asks for all
    /// of them; rebuilt whole after every change, they cost 13-27 ms a frame,
    /// run perf4).
    hulls: HashMap<(usize, bool), Hull>,
    /// Per copy and kind (unordered access): the copy, and how many ranges
    /// start and end at each offset, so that the hulls are made from the
    /// copies, not from every range (the title rewrites slots between most
    /// lists, and remerging the world's 130,000 ranges each time cost 0.15
    /// s per frame).
    ends: HashMap<(usize, bool), Ends>,
}

/// A copy, and the counts of its ranges' starts and ends.
type Ends = (Arc<Shadow>, BTreeMap<u64, u32>, BTreeMap<u64, u32>);

/// One more (`add`) or one fewer of `at` in `counts`.
fn tally(counts: &mut BTreeMap<u64, u32>, at: u64, add: bool) {
    let n = counts.entry(at).or_insert(0);
    *n = if add { *n + 1 } else { n.saturating_sub(1) };
    if *n == 0 {
        counts.remove(&at);
    }
}

impl Inner {
    fn count(&mut self, seen: &Seen, add: bool) {
        let step = |n: &mut u32| if add { *n += 1 } else { *n = n.saturating_sub(1) };
        match seen {
            Seen::Buffer(shadow, address, len, uav) => {
                let key = (Arc::as_ptr(shadow) as usize, *address, address + len, *uav);
                let e = self.ranges.entry(key).or_insert_with(|| (0, shadow.clone()));
                let was = e.0;
                step(&mut e.0);
                let now = e.0;
                if now == 0 {
                    self.ranges.remove(&key);
                }
                // A range came or went: its copy's hull may change.
                if (was == 0) != (now == 0) {
                    let ends = self
                        .ends
                        .entry((key.0, key.3))
                        .or_insert_with(|| (shadow.clone(), BTreeMap::new(), BTreeMap::new()));
                    tally(&mut ends.1, key.1, now > 0);
                    tally(&mut ends.2, key.2, now > 0);
                    let hull = match (ends.1.keys().next(), ends.2.keys().next_back()) {
                        (Some(&from), Some(&to)) => Some((shadow.clone(), from, to - from, key.3, shadow.dirty_bit())),
                        _ => None,
                    };
                    match hull {
                        Some(hull) => {
                            self.hulls.insert((key.0, key.3), hull);
                        }
                        None => {
                            self.ends.remove(&(key.0, key.3));
                            self.hulls.remove(&(key.0, key.3));
                        }
                    }
                }
            }
            Seen::Texture(r) => {
                let entry = self.textures.entry(*r).or_insert((0, None, 0));
                step(&mut entry.0);
                if entry.0 == 0 {
                    self.textures.remove(r);
                }
            }
            Seen::Left(_) => {}
        }
    }
}

impl Viewed {
    /// Title slot `index` now holds `view` (none: no view), made on the
    /// host (`made`) or left without a host view.
    pub(crate) fn set(&self, index: u32, view: Option<&View>, made: bool) {
        let seen = view.and_then(|v| if made { seen(v) } else { Some(Seen::Left(left_kind(v))) });
        let mut inner = self.inner.lock().unwrap_or_else(|e| e.into_inner());
        if let Some(old) = inner.slots.remove(&index) {
            inner.count(&old, false);
        }
        if let Some(seen) = seen {
            inner.count(&seen, true);
            inner.slots.insert(index, seen);
        }
    }

    /// The buffer ranges views reach, merged per host copy and kind
    /// (shader resource, unordered access): copy, start, bytes.
    #[cfg(any(test, feature = "diagnostics"))]
    fn ranges(&self) -> Vec<Hull> {
        self.inner.lock().unwrap_or_else(|e| e.into_inner()).hulls.values().cloned().collect()
    }
}

/// What a made view reads that title memory feeds.
fn seen(view: &View) -> Option<Seen> {
    let placed = crate::descriptor::host::placed_texture(view);
    let view = placed.as_ref().unwrap_or(view);
    if let View::Shader(resource, _) = view
        && let Some(texture) = crate::resource::made_copy(*resource as This)
    {
        return crate::texture::title_written(&texture.texture).then_some(Seen::Texture(*resource));
    }
    // The copy the host view reads.
    let (shadow, address, len) = match *view {
        View::Constant(address, size) => {
            let len = u64::from(size);
            ((len > 0).then(|| crate::shadow::find(address, len)).flatten()?, address, len)
        }
        _ => crate::descriptor::host::view_copy(view)?,
    };
    let uav = matches!(view, View::Unordered(..) | View::Placed(true, ..));
    if len == 0 {
        return None;
    }
    let len = len.min(shadow.left(address));
    Some(Seen::Buffer(shadow, address, len, uav))
}

fn left_kind(view: &View) -> (usize, u64, u64) {
    match view {
        View::Placed(_, resource, desc) if resource[0] != 1 => {
            let address = u64::from(desc[10]) | u64::from(desc[11]) << 32;
            (2, address, crate::resource::placed_size(resource).unwrap_or(0))
        }
        View::Unordered(..) | View::Placed(true, ..) => (1, 0, 0),
        View::Shader(..) | View::Placed(false, ..) | View::Constant(..) => (0, 0, 0),
        _ => (3, 0, 0),
    }
}

/// At submission, before anything else of it: the textures the heaps of its
/// heap-indexing lists view, and `textures` (staging and host texture of
/// those its lists read through tables, `crate::texture::upload`), are
/// brought up to date, those whose title memory changed copied into their
/// host textures by a list of ours, which the submission runs first (and
/// leaves them readable by shaders): each at most once.
///
/// shortcut: a title copy into texture memory recorded in the same
/// submission shows from the next one on.
///
/// A submission in flight whose shaders write title memory a texture here
/// lies in (through a buffer copy) completes first (`settle`): its writes
/// reach the texture's memory only then.
pub(crate) fn uploads(
    heaps: &[Arc<Viewed>],
    textures: &[(Arc<crate::texture::Staging>, Arc<crate::resource::HostTexture>)],
    settle: &mut dyn FnMut(u64, crate::host::Settled),
) -> Option<Uploads> {
    static LIST: Mutex<Holder> = Mutex::new(Holder {
        list: None,
        needs_recycle: false,
    });
    let mut guard = LIST.lock().unwrap_or_else(|e| e.into_inner());
    if guard.needs_recycle {
        crate::texture::recycle();
        guard.needs_recycle = false;
    }
    // What to bring up: the textures lists read through tables, and those
    // the heaps view that shaders read lately. Only textures that may have
    // changed are collected: clean tracked textures are skipped via their
    // cached watch::DIRTY bit without taking Resource::host or Staging locks,
    // and staging creation happens outside each heap's lock so title threads
    // writing descriptors are not stalled.
    let mut work: Vec<(Arc<crate::texture::Staging>, Arc<crate::resource::HostTexture>)> = Vec::new();
    let mut staged_set: Option<std::collections::HashSet<*const crate::texture::Staging>> = None;
    let mut push_unique = |staging: Arc<crate::texture::Staging>, copy: Arc<crate::resource::HostTexture>| {
        let ptr = Arc::as_ptr(&staging);
        if let Some(set) = staged_set.as_mut() {
            if set.insert(ptr) {
                work.push((staging, copy));
            }
        } else if work.len() < 8 {
            if !work.iter().any(|(s, _)| Arc::as_ptr(s) == ptr) {
                work.push((staging, copy));
            }
        } else {
            let mut set: std::collections::HashSet<_> = work.iter().map(|(s, _)| Arc::as_ptr(s)).collect();
            if set.insert(ptr) {
                work.push((staging, copy));
            }
            staged_set = Some(set);
        }
    };
    for (staging, copy) in textures {
        // A host copy wrote it in a submission made before this one (it
        // was not when the list was recorded), or its memory has not changed.
        if copy.host_written.load(Relaxed) || !staging.maybe_changed() {
            continue;
        }
        push_unique(staging.clone(), copy.clone());
    }
    let mut pending: Vec<(usize, Arc<crate::resource::HostTexture>)> = Vec::new();
    if !heaps.is_empty() {
        let presented = crate::queue::presented();
        let epoch = crate::resource::stamp_epoch();
        let mut done = (heaps.len() > 1).then(std::collections::HashSet::new);
        for viewed in heaps {
            let mut inner = viewed.inner.lock().unwrap_or_else(|e| e.into_inner());
            for (&r, entry) in inner.textures.iter_mut() {
                if let Some(bit) = entry.1
                    && crate::watch::clean_bit(Some(bit))
                {
                    continue;
                }
                if entry.2 == epoch {
                    continue;
                }
                if let Some(done) = done.as_mut()
                    && !done.insert(r)
                {
                    continue;
                }
                // Only what shaders read lately (the eviction pass's stamps):
                // the heaps view thousands of textures a frame never reads, and
                // bringing each up whenever its pages change re-uploaded about
                // 4 GB in 30-46 world frames (runs evict2, noevict1).
                // shortcut: a texture read again after a gap is brought up after
                // the submission that read it, which can see bytes one refresh
                // old (a slot the title rewrites is made fresh anyway); upload on
                // the mark itself if that shows.
                let (bit, copy) = crate::resource::bindless_upload_check(r as This, presented);
                if entry.1.is_none() && bit.is_some() {
                    entry.1 = bit;
                }
                if let Some(texture) = copy {
                    entry.2 = 0;
                    pending.push((r, texture));
                } else if bit.is_none() || !crate::watch::clean_bit(bit) {
                    // Skipped because it has no host copy, is not recently used,
                    // or is host-written: stays skipped until STAMP_EPOCH bumps.
                    entry.2 = epoch;
                }
            }
        }
    }
    for (r, texture) in pending {
        if let Some(staging) = crate::texture::staging_for(r as This, &texture)
            && staging.maybe_changed()
        {
            push_unique(staging, texture);
        }
    }
    if work.is_empty() {
        return None;
    }
    guard.needs_recycle = true;
    if let Some(owner) = work
        .iter()
        .filter_map(|(staging, _)| {
            let (start, end) = staging.title_range();
            crate::shadow::crossing(start, end, 0).map(|c| c.0)
        })
        .max()
    {
        settle(owner, crate::host::Settled::Texture);
    }
    let uploads = crate::texture::refresh_all(&work);
    let ready: Vec<(&Arc<crate::texture::Staging>, &Arc<crate::resource::HostTexture>, &crate::texture::Upload)> =
        work.iter()
            .zip(uploads.iter())
            .filter_map(|((staging, copy), upload)| upload.as_ref().map(|u| (staging, copy, u)))
            .collect();
    if ready.is_empty() {
        return None;
    }
    match &guard.list {
        // Closed by the last submission that recorded uploads, which has run.
        Some(list) => list.reset(),
        // Made open.
        None => {
            let list = crate::host::host()?.new_list().ok()?;
            list.set_ours();
            guard.list = Some(list);
        }
    }
    let list = guard.list.as_ref()?;
    crate::texture::record_uploads(list, &ready);
    #[cfg(feature = "diagnostics")]
    UPLOADED.fetch_add(ready.len() as u64, Relaxed);
    list.close();
    Some(Uploads(guard))
}

/// The upload list, used under its lock only.
pub(crate) struct Holder {
    list: Option<HostList>,
    /// Whether the last submission may have allocated texture staging buffers
    /// that must be recycled before the next upload.
    needs_recycle: bool,
}
// SAFETY: D3D12 objects are free-threaded; the list is used by one
// submission at a time, under the lock.
unsafe impl Send for Holder {}

/// The upload list of a submission, ours until it has run.
pub(crate) struct Uploads(std::sync::MutexGuard<'static, Holder>);

impl Uploads {
    pub(crate) fn list(&self) -> &HostList {
        self.0.list.as_ref().expect("made before it is handed out")
    }
}

impl Drop for Uploads {
    fn drop(&mut self) {
        // The submission has run (or was never executed): release held
        // HostTextures and staging buffers now so idle submissions do not
        // pin evicted VRAM while skipping D3D12 list reset/close.
        if let Some(list) = &self.0.list {
            list.clear_held();
        }
    }
}

/// Per submission: brings the buffer ranges a heap's views reach up to date
/// (in the submission's batch of refreshes): the pages the title changed
/// since their copies took them.
///
/// shortcut: what shaders write through the heap's unordered access views
/// stays in the host copies (later draws read it there) and does not go
/// back to title memory, so the CPU does not see it, and a page the title
/// also writes is copied over from title memory, losing the GPU's bytes
/// there. Copying back the whole unordered ranges after every submission
/// (about 1 GB of them in the world) cost seconds per frame; the upgrade
/// is shaders marking the pages they write (as `crate::memory` does).
///
/// Copies the title has not written since they took it are passed by: a
/// world submission's heaps reach ~9,000 copies, a few of them written.
pub(crate) fn refresh(viewed: &Viewed, mut add: impl FnMut(&Arc<Shadow>, usize, usize)) {
    let inner = viewed.inner.lock().unwrap_or_else(|e| e.into_inner());
    for (shadow, address, len, uav, bit) in inner.hulls.values() {
        if !crate::watch::clean_bit(*bit) {
            let (address, len) = if *uav && shadow.marked() {
                shadow.page_range(*address, *len)
            } else {
                (*address, *len)
            };
            add(shadow, address as usize, len as usize);
        }
    }
}

/// Draws and dispatches run with heap-indexing shaders, those of them
/// writing through the heap, textures uploaded for heaps (diagnostics).
static HEAP_DRAWS: [AtomicU64; 2] = [AtomicU64::new(0), AtomicU64::new(0)];
#[cfg(feature = "diagnostics")]
static UPLOADED: AtomicU64 = AtomicU64::new(0);

/// A draw or dispatch runs a shader that indexes the heap (and writes
/// through it).
pub(crate) fn note_draw(writes: bool) {
    HEAP_DRAWS[0].fetch_add(1, Relaxed);
    if writes {
        HEAP_DRAWS[1].fetch_add(1, Relaxed);
    }
}

/// Every view heap's registry.
static HEAPS: Mutex<Vec<std::sync::Weak<Viewed>>> = Mutex::new(Vec::new());

pub(crate) fn register(viewed: &Arc<Viewed>) {
    let mut heaps = HEAPS.lock().unwrap_or_else(|e| e.into_inner());
    heaps.retain(|h| h.strong_count() > 0);
    heaps.push(Arc::downgrade(viewed));
}

/// The title released `resource`: slots viewing it as a texture view
/// nothing to upload any more (its memory may be gone).
pub(crate) fn forget(resource: usize) {
    let heaps = HEAPS.lock().unwrap_or_else(|e| e.into_inner()).clone();
    for viewed in heaps.iter().filter_map(std::sync::Weak::upgrade) {
        let mut inner = viewed.inner.lock().unwrap_or_else(|e| e.into_inner());
        if inner.textures.remove(&resource).is_some() {
            inner.slots.retain(|_, seen| !matches!(seen, Seen::Texture(r) if *r == resource));
        }
    }
}

/// Clears cached `watch::DIRTY` bits for textures whose host copies were
/// evicted (`crate::resource::give_up`), before those host copies drop and
/// return their dirty bits to the pool.
pub(crate) fn clear_texture_bits(resources: &[usize]) {
    let heaps = HEAPS.lock().unwrap_or_else(|e| e.into_inner()).clone();
    for viewed in heaps.iter().filter_map(std::sync::Weak::upgrade) {
        let mut inner = viewed.inner.lock().unwrap_or_else(|e| e.into_inner());
        for &r in resources {
            if let Some(entry) = inner.textures.get_mut(&r) {
                entry.1 = None;
                entry.2 = 0;
            }
        }
    }
}

/// Logs, per frame over `frames`, the heap-indexed draws, and what the
/// view heaps' slots view (diagnostics).
#[cfg(feature = "diagnostics")]
pub(crate) fn log(frames: f64) {
    let [draws, writes] = [0, 1].map(|i| HEAP_DRAWS[i].swap(0, Relaxed) as f64 / frames);
    let uploaded = UPLOADED.swap(0, Relaxed) as f64 / frames;
    let heaps = HEAPS.lock().unwrap_or_else(|e| e.into_inner()).clone();
    for viewed in heaps.iter().filter_map(std::sync::Weak::upgrade) {
        let inner = viewed.inner.lock().unwrap_or_else(|e| e.into_inner());
        let mut left = [0u32; KINDS.len()];
        let (mut buffers, mut textures) = (0u32, 0u32);
        // Placed textures left null: the largest at each address.
        let mut placed = HashMap::new();
        for seen in inner.slots.values() {
            match seen {
                Seen::Buffer(..) => buffers += 1,
                Seen::Texture(_) => textures += 1,
                Seen::Left((k, address, bytes)) => {
                    left[*k] += 1;
                    if *k == 2 {
                        let b = placed.entry(*address).or_insert(0u64);
                        *b = (*b).max(*bytes);
                    }
                }
            }
        }
        let placed_mb = placed.values().sum::<u64>() as f64 / 1e6;
        let bytes: u64 = inner.ranges.keys().map(|&(_, a, b, _)| b - a).sum();
        drop(inner);
        let hulls = viewed.ranges();
        let uav_hulls: u64 = hulls.iter().filter(|r| r.3).map(|r| r.2).sum();
        let all_hulls: u64 = hulls.iter().map(|r| r.2).sum();
        let inner = viewed.inner.lock().unwrap_or_else(|e| e.into_inner());
        let copies = {
            let mut c: Vec<usize> = inner.ranges.keys().map(|k| k.0).collect();
            c.dedup();
            c.len()
        };
        let left: Vec<String> = KINDS.iter().zip(left).map(|(k, n)| format!("{k} {n}")).collect();
        let all = crate::shadow::all();
        let all_bytes: u64 = all.iter().map(|s| s.range().1).sum();
        tracing::info!(
            "heap views: {draws:.0} draws/frame index the heap ({writes:.0} writing through it), \
             {uploaded:.0} textures uploaded/frame; slots: {buffers} buffer views ({} ranges, \
             {:.1} MB, {copies} copies), {textures} title-written textures ({} distinct); \
             left null: {} (placed textures at {} addresses, {placed_mb:.0} MB); \
             host copies of title memory: {} ({:.1} MB); hulls {:.1} MB, unordered {:.1} MB",
            inner.ranges.len(),
            bytes as f64 / 1e6,
            inner.textures.len(),
            left.join(", "),
            placed.len(),
            all.len(),
            all_bytes as f64 / 1e6,
            all_hulls as f64 / 1e6,
            uav_hulls as f64 / 1e6
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::object::{HRESULT, S_OK};
    use windows::Win32::Graphics::Direct3D12::*;
    use windows::core::s;

    /// [`uploads`] with no submission in flight to settle.
    fn uploads_now(
        heaps: &[Arc<Viewed>],
        textures: &[(Arc<crate::texture::Staging>, Arc<crate::resource::HostTexture>)],
    ) -> Option<Uploads> {
        uploads(heaps, textures, &mut crate::host::nothing_in_flight)
    }

    /// Reads slots 0 (raw buffer) and 2 (texture) of the heap and writes
    /// slot 1 (raw unordered buffer): through a table over the heap's
    /// start, the same slots a shader indexing the heap reads.
    const SOURCE: &str = r#"
ByteAddressBuffer src : register(t0);
RWByteAddressBuffer dst : register(u0);
Texture2D<float4> tex : register(t1);
[numthreads(16, 1, 1)]
void main(uint3 id : SV_DispatchThreadID) {
    uint texel = (uint)round(tex.Load(int3(id.x % 4, 0, 0)).x * 255.0);
    dst.Store(id.x * 4, src.Load(id.x * 4) + texel);
}
"#;

    fn compile() -> (ID3D12RootSignature, ID3D12PipelineState) {
        use crate::compute::range;
        let ranges = [
            range(D3D12_DESCRIPTOR_RANGE_TYPE_SRV, 0, 0, 1),
            range(D3D12_DESCRIPTOR_RANGE_TYPE_UAV, 0, 0, 1),
            range(D3D12_DESCRIPTOR_RANGE_TYPE_SRV, 1, 0, 1),
        ];
        let device = &crate::host::host().unwrap().device;
        crate::compute::compute(device, s!("heap"), SOURCE, 0, &[crate::compute::table(&ranges)]).unwrap()
    }

    /// A title resource from a description in the console's layout (56
    /// bytes: dimension, alignment, width, height, depth and mips, format,
    /// samples, layout, flags).
    fn resource(dimension: u32, width: u64, height: u32, format: u32, layout: u32) -> This {
        let desc: [u32; 14] = [
            dimension, 0, 0, 0, width as u32, (width >> 32) as u32, height, 1 | 1 << 16, format, 1, 0,
            layout, 0, 0,
        ];
        let properties = [1u32, 0, 0, 0, 0];
        let mut out = std::ptr::null();
        let iid = crate::object::guid(
            0x696442be,
            0xa72e,
            0x4059,
            [0xbc, 0x79, 0x5b, 0x5c, 0x98, 0x04, 0x0f, 0xad],
        );
        let made: HRESULT = crate::resource::create_committed(
            std::ptr::null(),
            properties.as_ptr().cast(),
            0,
            desc.as_ptr().cast(),
            0,
            std::ptr::null(),
            &iid,
            &mut out,
        );
        assert_eq!(made, S_OK);
        out
    }

    /// A title resource at `address` (title memory of its own), as the
    /// title places buffers and textures in its heaps.
    fn resource_at(address: usize, dimension: u32, width: u64, height: u32, format: u32, layout: u32) -> This {
        let desc: [u32; 14] = [
            dimension, 0, 0, 0, width as u32, (width >> 32) as u32, height, 1 | 1 << 16, format, 1, 0,
            layout, 0, 0,
        ];
        let mut out = std::ptr::null();
        let iid = crate::object::guid(
            0x696442be,
            0xa72e,
            0x4059,
            [0xbc, 0x79, 0x5b, 0x5c, 0x98, 0x04, 0x0f, 0xad],
        );
        let made = crate::resource::create_at_address(
            std::ptr::null(),
            address as u64,
            desc.as_ptr().cast(),
            0,
            std::ptr::null(),
            &iid,
            &mut out,
        );
        assert_eq!(made, S_OK);
        out
    }

    const LINEAR: u32 = 0x100;

    /// A copy's hull is its live ranges' first start to last end, kept as
    /// slots come and go; a copy no slot views any more has none.
    #[test]
    fn hulls_follow_the_ranges() {
        let memory = vec![0u8; 0x1000];
        let at = memory.as_ptr() as u64;
        crate::shadow::register(at as usize, at, 0x1000);
        let Some(copy) = crate::shadow::find(at, 0x1000) else {
            crate::shadow::unregister(at as usize, at);
            return; // no host GPU here
        };
        let viewed = Viewed::default();
        let set = |slot: u32, range: Option<(u64, u64)>| {
            let mut inner = viewed.inner.lock().unwrap();
            if let Some(old) = inner.slots.remove(&slot) {
                inner.count(&old, false);
            }
            if let Some((from, len)) = range {
                let seen = Seen::Buffer(copy.clone(), at + from, len, false);
                inner.count(&seen, true);
                inner.slots.insert(slot, seen);
            }
        };
        let hull = || viewed.ranges().iter().map(|h| (h.1 - at, h.2)).collect::<Vec<_>>();
        set(0, Some((0x100, 0x100)));
        set(1, Some((0x800, 0x200)));
        set(2, Some((0x100, 0x100)));
        assert_eq!(hull(), [(0x100, 0x900)]);
        set(1, None);
        assert_eq!(hull(), [(0x100, 0x100)], "the far range gone");
        set(0, None);
        assert_eq!(hull(), [(0x100, 0x100)], "one slot still views it");
        set(2, None);
        assert!(hull().is_empty());
        crate::shadow::unregister(at as usize, at);
    }

    /// The title writes a buffer and a texture, and views of them and of an
    /// output buffer into its heap; a shader reaching them only through the
    /// heap (no table entry names them) reads what the title wrote at
    /// submission, every round (the title rewrites both each time, as Bink
    /// rewrites its planes every frame). The output buffer is not read back
    /// into title memory (`refresh`'s shortcut): it is read from its host
    /// copy.
    #[test]
    fn heap_views_are_current() {
        let (src, dst) = (resource(1, 256, 1, 0, 1), resource(1, 256, 1, 0, 1));
        let tex = resource(3, 4, 1, 28, LINEAR);
        let write = |base: u32, texel: u8| {
            let (src_bytes, _) = crate::resource::buffer_bytes(src).unwrap();
            let (tex_bytes, tex_desc) = crate::resource::texture_memory(tex).unwrap();
            let size = crate::resource::texture_layout(&tex_desc).unwrap().planes[0].layout.size as usize;
            // SAFETY: the resources' own memory, 256 bytes and the texture's.
            unsafe {
                let words = std::slice::from_raw_parts_mut(src_bytes.cast::<u32>(), 64);
                for (i, w) in words.iter_mut().enumerate() {
                    *w = base + i as u32;
                }
                std::ptr::write_bytes(tex_bytes.cast_mut(), texel, size);
            }
        };
        rounds(src, dst, tex, write);
    }

    /// The same with the buffer and the texture in one view of physical
    /// pages and the title writing them through another (the title fills its
    /// heaps through a pool view): every round must reach the shader.
    #[test]
    fn heap_views_follow_writes_through_another_mapping() {
        let (a, b) = crate::watch::tests::twin_views(2);
        let src = resource_at(a, 1, 256, 1, 0, 1);
        let tex = resource_at(a + PAGE_BYTES, 3, 4, 1, 28, LINEAR);
        let dst = resource(1, 256, 1, 0, 1);
        let (_, tex_desc) = crate::resource::texture_memory(tex).unwrap();
        let size = crate::resource::texture_layout(&tex_desc).unwrap().planes[0].layout.size as usize;
        let write = |base: u32, texel: u8| {
            // SAFETY: the second view of the buffer's and the texture's pages.
            unsafe {
                let words = std::slice::from_raw_parts_mut(b as *mut u32, 64);
                for (i, w) in words.iter_mut().enumerate() {
                    *w = base + i as u32;
                }
                std::ptr::write_bytes((b + PAGE_BYTES) as *mut u8, texel, size);
            }
        };
        rounds(src, dst, tex, write);
    }

    const PAGE_BYTES: usize = crate::watch::PAGE;

    /// Three rounds of: record a dispatch reading slots 0 and 2 and writing
    /// slot 1, let the title `write` (after recording, before submitting),
    /// submit, check the output's host copy.
    fn rounds(src: This, dst: This, tex: This, write: impl Fn(u32, u8)) {
        let Some(host) = crate::host::host() else {
            return; // no host GPU here
        };
        let null = std::ptr::null();
        // A shader-visible view heap of 8 slots, in memory of its own.
        let heap_desc: [u32; 8] = [0, 8, 1, 0, 0, 0, 0, 0];
        let mut heap = null;
        let iid = crate::object::guid(
            0x8efb471d,
            0x616c,
            0x4f49,
            [0x90, 0xf7, 0x12, 0x7b, 0xb7, 0x63, 0xfa, 0x51],
        );
        assert_eq!(crate::descriptor::create_heap(null, heap_desc.as_ptr().cast(), &iid, &mut heap), S_OK);
        let title_heap = crate::descriptor::heap(heap);
        let slot = |i: u64| title_heap.start + 32 * i;
        // Raw views of 64 dwords: format R32_TYPELESS, buffer dimension,
        // the default mapping, then first element, count, stride, flags.
        let srv: [u32; 10] = [39, 1, 0x1688, 0, 0, 0, 64, 0, 1, 0];
        let uav: [u32; 10] = [39, 1, 0, 0, 64, 0, 0, 0, 1, 0];
        crate::descriptor::create_srv(null, src, &srv, slot(0));
        crate::descriptor::create_uav(null, dst, null, &uav, slot(1));
        crate::descriptor::create_srv(null, tex, std::ptr::null(), slot(2));
        let heap_host = title_heap.host().expect("a host view heap");
        let (root, pso) = compile();
        let list = host.new_list().unwrap();
        let (dst_bytes, _) = crate::resource::buffer_bytes(dst).unwrap();
        let out = crate::shadow::find(dst_bytes as u64, 256).expect("the output's host copy");
        for (round, (base, texel)) in [(1000u32, 0x40u8), (5000, 0x10), (9000, 0x22)].into_iter().enumerate() {
            if round > 0 {
                list.reset();
            }
            // SAFETY: an open list, live objects of ours.
            unsafe {
                list.list.SetDescriptorHeaps(&[Some(heap_host.heap.clone())]);
                list.list.SetComputeRootSignature(&root);
                list.list.SetComputeRootDescriptorTable(0, heap_host.gpu(0));
                list.list.SetPipelineState(&pso);
                list.list.Dispatch(4, 1, 1);
            }
            list.defer_heap(&heap_host.viewed);
            list.close();
            // The title writes after recording, before submitting.
            write(base, texel);
            host.execute(&[&list]);
            let words = out.read_host(dst_bytes as u64, 64);
            for (i, &v) in words.iter().enumerate() {
                assert_eq!(v, base + i as u32 + u32::from(texel), "round {round}, dword {i}");
            }
        }
    }

    /// Cached watch dirty bits skip clean bindless textures without missing
    /// subsequent writes or stale-bit reuse after eviction and recreation.
    #[test]
    fn cached_dirty_bit_reuploads_after_write_and_after_evict_recreate() {
        if crate::host::host().is_none() {
            return; // no host GPU here
        }
        let pool = crate::watch::tests::pool(1);
        let tex = resource_at(pool, 3, 64, 64, 28, LINEAR);
        let copy = crate::resource::host_copy(tex).expect("host texture");
        let viewed = Arc::new(Viewed::default());
        register(&viewed);
        let mut desc = [0u32; 10];
        desc[..6].copy_from_slice(&[28, 4, 0x1688, 0, 0, 1]);
        viewed.set(0, Some(&View::Shader(tex as usize, Some(desc))), true);

        // SAFETY: 64x64 RGBA8 linear = 16 KiB inside our 64 KiB pool page.
        unsafe { std::ptr::write_bytes(pool as *mut u8, 0x11, 64 * 64 * 4) };
        copy.stamp();
        assert!(uploads_now(std::slice::from_ref(&viewed), &[]).is_some(), "initial upload");
        assert_eq!(crate::texture::uploads(tex), 1);

        // Second submission while clean populates the cached dirty bit in `viewed`
        // and skips recording an upload list.
        assert!(uploads_now(std::slice::from_ref(&viewed), &[]).is_none(), "clean skips list");
        assert_eq!(crate::texture::uploads(tex), 1);
        assert!(
            viewed.inner.lock().unwrap().textures.get(&(tex as usize)).and_then(|e| e.1).is_some(),
            "dirty bit cached in Viewed"
        );
        // Third submission hits the lock-free cached-bit fast path.
        assert!(uploads_now(std::slice::from_ref(&viewed), &[]).is_none(), "cached clean skips");
        assert_eq!(crate::texture::uploads(tex), 1);

        // Writing the guarded page flips the watch bit back to dirty -> re-upload.
        // SAFETY: as above; write-fault handler opens the page and marks dirty.
        unsafe { std::ptr::write_bytes(pool as *mut u8, 0x22, 64 * 64 * 4) };
        assert!(uploads_now(std::slice::from_ref(&viewed), &[]).is_some(), "dirty re-uploads");
        assert_eq!(crate::texture::uploads(tex), 2);

        // Evicting the host texture clears the cached bit so a recreated host
        // texture with a fresh staging/bit is never skipped.
        drop(copy);
        assert!(crate::resource::give_up_now(tex), "evicted");
        assert!(
            viewed.inner.lock().unwrap().textures.get(&(tex as usize)).and_then(|e| e.1).is_none(),
            "cached dirty bit cleared on eviction"
        );
        let copy2 = crate::resource::host_copy(tex).expect("recreated host texture");
        copy2.stamp();
        // SAFETY: as above.
        unsafe { std::ptr::write_bytes(pool as *mut u8, 0x33, 64 * 64 * 4) };
        assert!(uploads_now(std::slice::from_ref(&viewed), &[]).is_some(), "recreated uploads");
        assert_eq!(crate::texture::uploads(tex), 1);

        drop(copy2);
        drop(viewed);
        crate::object::slot::<extern "system" fn(This) -> u32>(tex, 0x10)(tex);
        crate::watch::tests::free(pool);
    }

    /// Untracked (committed, non-pool) bindless textures stay uncached so
    /// content-hash changes are never skipped.
    #[test]
    fn untracked_bindless_texture_reuploads_on_memory_change() {
        if crate::host::host().is_none() {
            return; // no host GPU here
        }
        let tex = resource(3, 16, 16, 28, LINEAR);
        let copy = crate::resource::host_copy(tex).expect("host texture");
        let (mem, desc_meta) = crate::resource::texture_memory(tex).unwrap();
        let size = crate::resource::texture_layout(&desc_meta).unwrap().planes[0].layout.size as usize;
        let viewed = Arc::new(Viewed::default());
        register(&viewed);
        let mut desc = [0u32; 10];
        desc[..6].copy_from_slice(&[28, 4, 0x1688, 0, 0, 1]);
        viewed.set(0, Some(&View::Shader(tex as usize, Some(desc))), true);

        // SAFETY: committed texture memory of `size` bytes.
        unsafe { std::ptr::write_bytes(mem.cast_mut(), 0x41, size) };
        copy.stamp();
        assert!(uploads_now(std::slice::from_ref(&viewed), &[]).is_some(), "initial untracked upload");
        assert_eq!(crate::texture::uploads(tex), 1);
        assert!(uploads_now(std::slice::from_ref(&viewed), &[]).is_none(), "unchanged hash skips");
        assert!(
            viewed.inner.lock().unwrap().textures.get(&(tex as usize)).and_then(|e| e.1).is_none(),
            "untracked texture has no cached dirty bit"
        );
        // SAFETY: as above.
        unsafe { std::ptr::write_bytes(mem.cast_mut(), 0x52, size) };
        assert!(uploads_now(std::slice::from_ref(&viewed), &[]).is_some(), "changed untracked re-uploads");
        assert_eq!(crate::texture::uploads(tex), 2);

        drop(copy);
        drop(viewed);
        crate::object::slot::<extern "system" fn(This) -> u32>(tex, 0x10)(tex);
    }
}
