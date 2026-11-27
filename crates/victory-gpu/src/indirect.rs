//! Indirect execution's arguments, bounded before the GPU reads them. The
//! title's own compute passes write most of them; a pass we drop or
//! translate wrongly leaves whatever the memory held, and a draw of four
//! billion vertices or a dispatch past the host's limits runs until the
//! host resets the display driver. So every record is clamped on the GPU,
//! just before its execution, by a compute shader of ours ([`crate::compute`],
//! which Wine's compiler builds too: without the clamp every indirect
//! execution is dropped).
//!
//! Each list also has a budget (2^27 vertex instances or threads, groups
//! counted as 64 threads, mesh groups as 128): records past it run empty,
//! so a list of thousands of garbage records cannot run for seconds
//! either.
//!
//! shortcut: the bounds (2^24 vertices, 2^26 vertex instances, 65,535
//! groups per dimension and 2^20 per dispatch, one-dimensional mesh
//! dispatches, the budget) cut legitimate work beyond them; raise them if
//! the title needs more.

use windows::Win32::Graphics::Direct3D12::*;
use windows::core::{Interface, PCSTR, s};

use crate::compute::{compute, constants, root_view};
use crate::host::{HostList, host, transition};

/// The groups of a mesh dispatch record, shared by the single clamp and the
/// window's pass so the two cannot drift apart.
macro_rules! mesh_groups {
    () => {
        r#"
// Mesh dispatches: the console's are one-dimensional.
uint3 mesh_groups(uint3 g) {
    return uint3(min(g.x, 65535u), min(g.y, 1u), min(g.z, 1u));
}
"#
    };
}

const SOURCE: &str = concat!(mesh_groups!(), r#"
cbuffer Record : register(b0) { uint kind; uint stride; uint records; uint spread; };
// The title's records, read only; the clamped ones go to the list's own.
// In words: the caller refuses addresses and strides not multiples of 4.
StructuredBuffer<uint> args : register(t0);
RWStructuredBuffer<uint> dest : register(u0);
// The list's work so far (vertex instances, threads): past the budget,
// records run empty.
RWStructuredBuffer<uint> budget : register(u1);
static const uint BUDGET = 1u << 27;
// Groups of one dispatch; more run empty.
static const uint GROUPS = 1u << 20;
uint3 load3(uint at) {
    return uint3(args[at], args[at + 1], args[at + 2]);
}
void store3(uint at, uint3 v) {
    dest[at] = v.x;
    dest[at + 1] = v.y;
    dest[at + 2] = v.z;
}
// Clamps the record at word `at` into `dest`; its cost.
uint clamped(uint at, uint record) {
    uint cost;
    if (kind <= 1) {
        // Draws: vertex or index count, then instances.
        uint count = min(args[at], 1u << 24);
        uint instances = min(args[at + 1], (1u << 26) / max(count, 1u));
        cost = count * instances;
        dest[at] = count;
        dest[at + 1] = instances;
        // The rest of the record as it is: first vertex and instance (4 words
        // in all), or first index, base vertex and first instance (5).
        dest[at + 2] = args[at + 2];
        dest[at + 3] = args[at + 3];
        if (kind == 1) {
            dest[at + 4] = args[at + 4];
        }
    } else if (kind == 2) {
        uint3 g = load3(at);
        uint row = 0;
        uint asked = g.x;
        if (spread != 0 && g.x > 65535 && g.y >= 1 && g.z == 1) {
            uint z = (g.x + 65534) / 65535;
            row = (g.x + z - 1) / z;
            g = uint3(row, g.y, z);
        } else {
            g = min(g, uint3(65535, 65535, 65535));
        }
        // x * y fits 32 bits; the whole count may not.
        if (g.z != 0 && g.x * g.y > GROUPS / g.z) {
            g = uint3(0, 0, 0);
        }
        cost = g.x * g.y * g.z * 64;
        if (spread != 0) {
            dest[record * 5] = row;
            dest[record * 5 + 1] = asked;
            store3(record * 5 + 2, g);
        } else {
            store3(at, g);
        }
    } else {
        uint3 g = mesh_groups(load3(at));
        cost = g.x * g.y * g.z * 128;
        store3(at, g);
    }
    return cost;
}
// A record past the budget (no cost reaches it).
static const uint PAST = 0xffffffffu;
groupshared uint costs[64];
// One group walks the records in tiles of 64: every lane clamps its own
// record, then the first lane charges the tile's records in order. No
// other group shares the budget: a shared add wraps once enough records
// are rejected, and a compare-exchange retries more the more records fit.
[numthreads(64, 1, 1)]
void main(uint lane : SV_GroupIndex) {
    uint left = 0;
    if (lane == 0) {
        left = BUDGET - min(budget[0], BUDGET);
    }
    for (uint tile = 0; tile < records / 64 + min(records % 64, 1u); tile++) {
        uint record = tile * 64 + lane;
        uint at = record * (stride / 4);
        uint cost = 0;
        if (record < records) {
            cost = clamped(at, record);
        }
        costs[lane] = cost;
        GroupMemoryBarrierWithGroupSync();
        if (lane == 0) {
            for (uint i = 0; i < 64; i++) {
                if (costs[i] > left) {
                    // The first record that does not fit takes what is
                    // left, so those after it run empty too.
                    costs[i] = PAST;
                    left = 0;
                } else {
                    left -= costs[i];
                }
            }
        }
        GroupMemoryBarrierWithGroupSync();
        if (costs[lane] == PAST) {
            // Draws keep their layout with no vertices; groups become zero.
            if (spread != 0 && record < records) {
                store3(record * 5 + 2, uint3(0, 0, 0));
            } else {
                dest[at] = 0;
            }
        }
    }
    if (lane == 0) {
        budget[0] = BUDGET - left;
    }
}
"#);

// As `crate::compute` takes it (Wine's compiler as well as Microsoft's): no
// `RootSignature` attribute ([`batch_clamp`] makes it: the batch (b0), the
// jobs (t0), the title's records (t1), the clamped ones (u0), the budget
// (u1)), and the budget, read and written, a structured buffer of `uint`.
const BATCH_SOURCE: &str = r#"
cbuffer Batch : register(b0) { uint first_job; uint job_count; uint unused0; uint unused1; };
// Packed 32-byte job descriptors:
//   j0 = uint4(kind, stride, records, src_offset)
//   j1 = uint4(dst_offset, 0, 0, 0)
ByteAddressBuffer jobs : register(t0);
ByteAddressBuffer src_args : register(t1);
RWByteAddressBuffer dst_args : register(u0);
RWStructuredBuffer<uint> budget : register(u1);
static const uint BUDGET = 1u << 27;
static const uint GROUPS = 1u << 20;
static const uint PAST = 0xffffffffu;
groupshared uint costs[64];
groupshared uint single_run;

uint compute_clamped(uint kind, uint src_at, out uint2 draw_counts, out uint3 groups) {
    draw_counts = uint2(0, 0);
    groups = uint3(0, 0, 0);
    [branch]
    if (kind <= 1) {
        uint count = min(src_args.Load(src_at), 1u << 24);
        uint instances = min(src_args.Load(src_at + 4), (1u << 26) / max(count, 1u));
        draw_counts = uint2(count, instances);
        return count * instances;
    } else if (kind == 2) {
        uint3 g = min(src_args.Load3(src_at), uint3(65535u, 65535u, 65535u));
        if (g.z != 0) {
            if (g.x * g.y > GROUPS / max(g.z, 1u)) {
                g = uint3(0, 0, 0);
            }
        }
        groups = g;
        return g.x * g.y * g.z * 64u;
    } else {
        uint3 g = src_args.Load3(src_at);
        g = uint3(min(g.x, 65535u), min(g.y, 1u), min(g.z, 1u));
        groups = g;
        return g.x * g.y * g.z * 128u;
    }
}

void write_record(uint kind, uint stride, uint src_at, uint dst_at, bool over, uint2 draw_counts, uint3 groups) {
    if (kind <= 1) {
        if (over) {
            draw_counts.x = 0;
        }
        dst_args.Store2(dst_at, draw_counts);
        for (uint off = 8; off < stride; off += 4) {
            dst_args.Store(dst_at + off, src_args.Load(src_at + off));
        }
    } else {
        if (over) {
            groups.x = 0;
        }
        dst_args.Store3(dst_at, groups);
        for (uint off = 12; off < stride; off += 4) {
            dst_args.Store(dst_at + off, src_args.Load(src_at + off));
        }
    }
}

[numthreads(64, 1, 1)]
void main(uint lane : SV_GroupIndex) {
    uint left = 0;
    if (lane == 0) {
        left = BUDGET - min(budget[0], BUDGET);
    }
    uint j = first_job;
    uint end = first_job + job_count;
    while (j < end) {
        // Fast path: clamp up to 64 consecutive single-record jobs in one wave.
        // Guard root-SRV loads with `[branch]` so inactive tail lanes never
        // load past the end of `jobs` (root descriptors have no HW bounds check).
        uint4 j0 = uint4(0, 0, 0, 0);
        uint4 j1 = uint4(0, 0, 0, 0);
        [branch]
        if (j + lane < end) {
            j0 = jobs.Load4((j + lane) * 32);
            j1 = jobs.Load4((j + lane) * 32 + 16);
        }
        costs[lane] = j0.z;
        GroupMemoryBarrierWithGroupSync();
        if (lane == 0) {
            uint run = 0;
            // Avoid `&&` here: FXC evaluates both sides, which would read
            // `costs[64]` past the 64-element groupshared array.
            [loop]
            while (run < 64) {
                if (costs[run] != 1) {
                    break;
                }
                run++;
            }
            single_run = run;
        }
        GroupMemoryBarrierWithGroupSync();
        uint run = single_run;
        if (run > 0) {
            uint cost = 0;
            uint2 draw_counts = uint2(0, 0);
            uint3 groups = uint3(0, 0, 0);
            [branch]
            if (lane < run) {
                cost = compute_clamped(j0.x, j0.w, draw_counts, groups);
            }
            costs[lane] = cost;
            GroupMemoryBarrierWithGroupSync();
            if (lane == 0) {
                for (uint i = 0; i < run; i++) {
                    if (costs[i] > left) {
                        costs[i] = PAST;
                        left = 0;
                    } else {
                        left -= costs[i];
                    }
                }
            }
            GroupMemoryBarrierWithGroupSync();
            [branch]
            if (lane < run) {
                write_record(j0.x, j0.y, j0.w, j1.x, costs[lane] == PAST, draw_counts, groups);
            }
            j += run;
        } else {
            uint4 m0 = jobs.Load4(j * 32);
            uint4 m1 = jobs.Load4(j * 32 + 16);
            uint tiles = m0.z / 64 + min(m0.z % 64, 1u);
            for (uint tile = 0; tile < tiles; tile++) {
                uint record = tile * 64 + lane;
                uint src_at = m0.w + record * m0.y;
                uint dst_at = m1.x + record * m0.y;
                uint cost = 0;
                uint2 draw_counts = uint2(0, 0);
                uint3 groups = uint3(0, 0, 0);
                [branch]
                if (record < m0.z) {
                    cost = compute_clamped(m0.x, src_at, draw_counts, groups);
                }
                costs[lane] = cost;
                GroupMemoryBarrierWithGroupSync();
                if (lane == 0) {
                    for (uint i = 0; i < 64; i++) {
                        if (costs[i] > left) {
                            costs[i] = PAST;
                            left = 0;
                        } else {
                            left -= costs[i];
                        }
                    }
                }
                GroupMemoryBarrierWithGroupSync();
                [branch]
                if (record < m0.z) {
                    write_record(m0.x, m0.y, src_at, dst_at, costs[lane] == PAST, draw_counts, groups);
                }
            }
            j += 1;
        }
    }
    if (lane == 0) {
        budget[0] = BUDGET - left;
    }
}
"#;

struct Clamp {
    root: ID3D12RootSignature,
    pso: ID3D12PipelineState,
}

// SAFETY: D3D12 objects are free-threaded.
unsafe impl Send for Clamp {}
// SAFETY: as above.
unsafe impl Sync for Clamp {}

fn clamp() -> Option<&'static Clamp> {
    static CLAMP: std::sync::OnceLock<Option<Clamp>> = std::sync::OnceLock::new();
    CLAMP
        .get_or_init(|| match make() {
            Ok(c) => Some(c),
            Err(e) => {
                tracing::warn!(error = %e, "no argument clamp: indirect execution dropped");
                None
            }
        })
        .as_ref()
}

/// The clamp, its root signature the record (b0), the title's records
/// (t0), the budget (u1) and the clamped records (u0), as [`bound`] binds.
/// The root signature is built here rather than taken from the shader:
/// this one also takes the spread flag in `Record.w`, and its buffers are
/// buffers of records, not byte addresses, so it binds root descriptors.
fn make() -> windows::core::Result<Clamp> {
    use D3D12_ROOT_PARAMETER_TYPE_SRV as SRV;
    use D3D12_ROOT_PARAMETER_TYPE_UAV as UAV;
    let params = [constants(0, 4), root_view(SRV, 0, 0), root_view(UAV, 1, 0), root_view(UAV, 0, 0)];
    make_with(SOURCE, s!("argument clamp"), &params)
}

/// A clamp pipeline of `source` (named `name` in compiler messages) and
/// its root signature of `params` (`crate::compute`).
fn make_with(source: &str, name: PCSTR, params: &[D3D12_ROOT_PARAMETER]) -> windows::core::Result<Clamp> {
    let device = &host().ok_or_else(windows::core::Error::empty)?.device;
    let (root, pso) = compute(device, name, source, 0, params)?;
    Ok(Clamp { root, pso })
}

fn batch_clamp() -> Option<&'static Clamp> {
    use D3D12_ROOT_PARAMETER_TYPE_SRV as SRV;
    use D3D12_ROOT_PARAMETER_TYPE_UAV as UAV;
    static BATCH: std::sync::OnceLock<Option<Clamp>> = std::sync::OnceLock::new();
    BATCH
        .get_or_init(|| {
            // As [`HoistedClamps::record_prelude`] binds them.
            let params = [constants(0, 4), root_view(SRV, 0, 0), root_view(SRV, 1, 0), root_view(UAV, 0, 0), root_view(UAV, 1, 0)];
            match make_with(BATCH_SOURCE, s!("batch clamp"), &params) {
                Ok(c) => Some(c),
                Err(e) => {
                    tracing::warn!(error = %e, "no batched argument clamp: falling back to inline");
                    None
                }
            }
        })
        .as_ref()
}

const HOIST_CHUNK: u64 = 256 << 10;

struct Job {
    src_gpu: u64,
    dst_gpu: u64,
}

struct JobUpload {
    _buffer: ID3D12Resource,
    mapped: usize,
    gpu: u64,
    capacity: usize,
}

// SAFETY: `mapped` is a persistent upload mapping written only under the
// list's `HoistedClamps` mutex while the list is open for recording.
unsafe impl Send for JobUpload {}

/// Indirect argument clamps hoisted from a [`HostList`]'s initial non-writing
/// prefix into its submission prelude. Each hoisted draw reads the title's
/// argument buffer through an SRV (never transitioning or mutating it) and
/// writes clamped records into the list's own scratch chunks, which move
/// `COMMON -> UNORDERED_ACCESS -> INDIRECT_ARGUMENT` in the prelude and back
/// to `COMMON` once at [`Self::end`] (or [`HostList::close`]).
pub(crate) struct HoistedClamps {
    hoistable: bool,
    restored: bool,
    jobs: Vec<Job>,
    payload: Vec<[u32; 8]>,
    upload: Option<JobUpload>,
    chunks: Vec<ID3D12Resource>,
    used: usize,
    at: u64,
    kept: Vec<ID3D12Resource>,
}

// SAFETY: D3D12 resources are free-threaded; mutated under the list's mutex.
unsafe impl Send for HoistedClamps {}

impl Default for HoistedClamps {
    fn default() -> Self {
        Self {
            hoistable: true,
            restored: false,
            jobs: Vec::new(),
            payload: Vec::new(),
            upload: None,
            chunks: Vec::new(),
            used: 0,
            at: 0,
            kept: Vec::new(),
        }
    }
}

impl HoistedClamps {
    pub(crate) fn reset(&mut self) {
        self.hoistable = true;
        self.restored = false;
        self.jobs.clear();
        self.payload.clear();
        self.used = 0;
        self.at = 0;
        self.kept.clear();
    }

    pub(crate) fn is_empty(&self) -> bool {
        self.jobs.is_empty()
    }

    fn min_stride(kind: u32) -> Option<u32> {
        match kind {
            0 => Some(16),
            1 => Some(20),
            2 | 10 => Some(12),
            _ => None,
        }
    }

    fn byte_len(gpu: u64, (kind, stride, records): (u32, u32, u32)) -> Option<u64> {
        let min = Self::min_stride(kind)?;
        if records == 0 || stride < min || !gpu.is_multiple_of(4) || !stride.is_multiple_of(4) {
            return None;
        }
        let len = u64::from(stride).checked_mul(u64::from(records))?;
        let aligned = len.checked_next_multiple_of(16)?;
        (aligned <= HOIST_CHUNK).then_some(len)
    }

    pub(crate) fn can_accept(&self, gpu: u64, spec: (u32, u32, u32)) -> bool {
        self.hoistable
            && u32::try_from(self.jobs.len()).is_ok()
            && batch_clamp().is_some()
            && Self::byte_len(gpu, spec).is_some()
    }

    fn keep_once(&mut self, resource: &ID3D12Resource) {
        let raw = resource.as_raw();
        if !self.kept.iter().any(|r| r.as_raw() == raw) {
            self.kept.push(resource.clone());
        }
    }

    fn alloc_room(&mut self, bytes: u64) -> Option<(ID3D12Resource, u64)> {
        let aligned = bytes.next_multiple_of(16);
        if aligned > HOIST_CHUNK {
            return None;
        }
        if self.used > 0 && self.at + aligned <= HOIST_CHUNK {
            let at = self.at;
            self.at += aligned;
            return Some((self.chunks[self.used - 1].clone(), at));
        }
        if self.used == self.chunks.len() {
            let (buffer, _) = host()?.shared_buffer(HOIST_CHUNK)?;
            crate::host::name(&buffer, || "hoisted indirect arguments".into());
            self.chunks.push(buffer);
        }
        let i = self.used;
        self.used += 1;
        self.at = aligned;
        Some((self.chunks[i].clone(), 0))
    }

    fn ensure_upload(&mut self, needed_jobs: usize) -> bool {
        if self.upload.as_ref().is_some_and(|u| u.capacity >= needed_jobs) {
            return true;
        }
        let Some(host) = host() else {
            return false;
        };
        let capacity = needed_jobs.next_power_of_two().max(256);
        let Some(bytes) = (capacity as u64).checked_mul(32) else {
            return false;
        };
        let Some((buffer, data)) = host.upload_buffer(bytes) else {
            return false;
        };
        crate::host::name(&buffer, || format!("hoisted clamp jobs ({capacity})"));
        // SAFETY: a live upload buffer.
        let gpu = unsafe { buffer.GetGPUVirtualAddress() };
        if self.upload.is_some() && !self.payload.is_empty() {
            // SAFETY: `data` has room for `capacity >= self.payload.len()` 32-byte jobs.
            unsafe {
                std::ptr::copy_nonoverlapping(
                    self.payload.as_ptr().cast::<u8>(),
                    data,
                    self.payload.len() * 32,
                );
            }
        }
        self.upload = Some(JobUpload {
            _buffer: buffer,
            mapped: data as usize,
            gpu,
            capacity,
        });
        true
    }

    pub(crate) fn push(
        &mut self,
        src_buf: &ID3D12Resource,
        gpu: u64,
        spec: (u32, u32, u32),
    ) -> Option<(ID3D12Resource, u64)> {
        if !self.can_accept(gpu, spec) {
            return None;
        }
        let len = Self::byte_len(gpu, spec)?;
        let slot = self.jobs.len();
        if !self.ensure_upload(slot + 1) {
            return None;
        }
        let (dst, dst_at) = self.alloc_room(len)?;
        // SAFETY: live D3D12 buffers.
        let (buf_gpu, dst_gpu) = unsafe { (src_buf.GetGPUVirtualAddress(), dst.GetGPUVirtualAddress()) };
        let (src_gpu, src_offset) = match gpu.checked_sub(buf_gpu) {
            Some(rel) if rel.checked_add(len).is_some_and(|e| e <= u64::from(u32::MAX)) => (buf_gpu, rel as u32),
            _ => (gpu, 0),
        };
        self.keep_once(src_buf);
        let (kind, stride, records) = spec;
        let words = [kind, stride, records, src_offset, dst_at as u32, 0, 0, 0];
        if let Some(u) = &self.upload {
            // SAFETY: `ensure_upload(slot + 1)` guarantees `[slot * 32 .. (slot + 1) * 32)`
            // lies within the persistent mapped upload buffer.
            unsafe {
                (u.mapped as *mut [u32; 8]).add(slot).write(words);
            }
        }
        self.payload.push(words);
        self.jobs.push(Job { src_gpu, dst_gpu });
        Some((dst, dst_at))
    }

    /// Ends the hoistable prefix on `list`: transitions every used scratch
    /// chunk back from `INDIRECT_ARGUMENT` to `COMMON` once, and blocks
    /// further hoisting on this list until `reset()`.
    pub(crate) fn end(&mut self, list: &ID3D12GraphicsCommandList) {
        self.hoistable = false;
        if self.restored || self.used == 0 {
            return;
        }
        self.restored = true;
        let barriers: Vec<D3D12_RESOURCE_BARRIER> = self.chunks[..self.used]
            .iter()
            .map(|b| {
                transition(
                    b,
                    D3D12_RESOURCE_STATE_INDIRECT_ARGUMENT,
                    D3D12_RESOURCE_STATE_COMMON,
                )
            })
            .collect();
        // SAFETY: an open list and live scratch chunks in `INDIRECT_ARGUMENT`.
        unsafe { list.ResourceBarrier(&barriers) };
    }

    /// Records the batched hoisted clamp pass(es) into `prelude`, moving each
    /// used scratch chunk `COMMON -> UNORDERED_ACCESS -> INDIRECT_ARGUMENT`.
    pub(crate) fn record_prelude(
        &self,
        prelude: &ID3D12GraphicsCommandList,
        budget_resource: &ID3D12Resource,
        budget_gpu: u64,
    ) -> bool {
        if self.jobs.is_empty() || self.used == 0 {
            return false;
        }
        let (Some(c), Some(upload)) = (batch_clamp(), &self.upload) else {
            return false;
        };
        let mut pre = Vec::with_capacity(self.used + 1);
        // Earlier lists' shader writes to the title's argument buffers first.
        pre.push(uav_barrier_of(None));
        for chunk in &self.chunks[..self.used] {
            pre.push(transition(
                chunk,
                D3D12_RESOURCE_STATE_COMMON,
                D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
            ));
        }
        // SAFETY: an open prelude list; root signature, PSO, job buffer,
        // budget buffer, title source buffers (`kept`) and scratch chunks all
        // live through the submission.
        unsafe {
            prelude.ResourceBarrier(&pre);
            prelude.SetComputeRootSignature(&c.root);
            prelude.SetPipelineState(&c.pso);
            prelude.SetComputeRootShaderResourceView(1, upload.gpu);
            prelude.SetComputeRootUnorderedAccessView(4, budget_gpu);
            let mut run_start = 0usize;
            while run_start < self.jobs.len() {
                let key = (self.jobs[run_start].src_gpu, self.jobs[run_start].dst_gpu);
                let mut run_end = run_start + 1;
                while run_end < self.jobs.len()
                    && (self.jobs[run_end].src_gpu, self.jobs[run_end].dst_gpu) == key
                {
                    run_end += 1;
                }
                if run_start > 0 {
                    // Full UAV barrier between runs: keeps both `budget` and
                    // adjacent 16-byte-aligned slices of the same scratch
                    // chunk coherent on wide-cache-line GPUs.
                    prelude.ResourceBarrier(&[uav_barrier_of(None)]);
                }
                let constants = [run_start as u32, (run_end - run_start) as u32, 0, 0];
                prelude.SetComputeRoot32BitConstants(0, 4, constants.as_ptr().cast(), 0);
                prelude.SetComputeRootShaderResourceView(2, key.0);
                prelude.SetComputeRootUnorderedAccessView(3, key.1);
                prelude.Dispatch(1, 1, 1);
                run_start = run_end;
            }
            let mut post = Vec::with_capacity(self.used + 1);
            post.push(uav_barrier_of(Some(budget_resource)));
            for chunk in &self.chunks[..self.used] {
                post.push(transition(
                    chunk,
                    D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                    D3D12_RESOURCE_STATE_INDIRECT_ARGUMENT,
                ));
            }
            prelude.ResourceBarrier(&post);
        }
        true
    }
}

/// Whether `spec` at `gpu` can be hoisted into `list`'s prelude.
pub(crate) fn can_hoist(list: &HostList, gpu: u64, spec: (u32, u32, u32)) -> bool {
    list.hoisted().can_accept(gpu, spec)
}

/// Queues `spec` at `gpu` in `src_buf` to be clamped in `list`'s prelude into
/// one of `list`'s own scratch chunks. Returns `(chunk, offset)` ready to be
/// read in `INDIRECT_ARGUMENT`, or `None` if hoisting is unavailable.
pub(crate) fn hoist(
    list: &HostList,
    src_buf: &ID3D12Resource,
    gpu: u64,
    spec: (u32, u32, u32),
) -> Option<(ID3D12Resource, u64)> {
    list.budget()?;
    list.hoisted().push(src_buf, gpu, spec)
}

/// Records, in `list`, the clamp of `records` records of `kind` (PC
/// argument numbering) `stride` bytes apart at host address `gpu` (a title
/// buffer's copy, only read) into the list's own argument buffer, left
/// ready to be read as arguments: that buffer, for [`done`] after the
/// execution. Replaces the list's compute root signature and pipeline
/// (callers bind theirs after). `None` if the clamp is not available.
///
/// Title buffers' copies are never transitioned: they are common between
/// submissions and every use promotes them (shader and root views, the
/// input assembler, these reads), with unordered-access barriers for order.
/// Only the list's argument buffer moves, common, unordered access,
/// indirect argument, common: its states are known whatever order lists
/// run in.
// shortcut: a buffer a shader writes through an unordered view and the
// input assembler then reads in one submission is, by the letter of D3D12,
// promoted to a write state the input assembler may not use (the debug
// layer cannot see view accesses; GPU-based validation might); upgrade to
// per-list tracked buffer states if a driver or that validation minds.
pub(crate) fn bound(
    list: &HostList,
    gpu: u64,
    (kind, stride, records): (u32, u32, u32),
    spread: bool,
) -> Option<ID3D12Resource> {
    #[cfg(feature = "diagnostics")]
    note_clamp(list, "clamp");
    // An open window first: its draws come before this, in order.
    end_window(list);
    let (Some(c), Some(budget)) = (clamp(), list.budget()) else {
        return None;
    };
    if !gpu.is_multiple_of(4) || !stride.is_multiple_of(4) {
        return None;
    }
    let out_stride = if spread { 20 } else { stride };
    let out = list.arguments(u64::from(out_stride) * u64::from(records.max(1)))?;
    #[cfg(feature = "diagnostics")]
    crate::copystat::add(crate::copystat::C::ClampSingle, 1);
    let l = &list.list;
    let constants = [kind, stride, records, u32::from(spread)];
    // SAFETY: an open list; our root signature, pipeline and buffers, and
    // a title buffer's copy the caller's list holds.
    unsafe {
        // Earlier shader writes (the title's argument passes) first.
        l.ResourceBarrier(&[
            uav_barrier(),
            transition(&out, D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_UNORDERED_ACCESS),
        ]);
        #[cfg(feature = "diagnostics")]
        crate::copystat::add(crate::copystat::C::ClampBarrierCalls, 2);
        l.SetComputeRootSignature(&c.root);
        l.SetPipelineState(&c.pso);
        l.SetComputeRoot32BitConstants(0, 4, constants.as_ptr().cast(), 0);
        l.SetComputeRootShaderResourceView(1, gpu);
        l.SetComputeRootUnorderedAccessView(2, budget);
        l.SetComputeRootUnorderedAccessView(3, out.GetGPUVirtualAddress());
        // One group walks every record.
        l.Dispatch(records.min(1), 1, 1);
        l.ResourceBarrier(&[
            uav_barrier(),
            transition(&out, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_INDIRECT_ARGUMENT),
        ]);
    }
    Some(out)
}

/// After the execution: the list's argument buffer back to common.
pub(crate) fn done(list: &HostList, arguments: &ID3D12Resource) {
    #[cfg(feature = "diagnostics")]
    crate::copystat::add(crate::copystat::C::ClampBarrierCalls, 1);
    // SAFETY: an open list and a live buffer.
    unsafe {
        list.list.ResourceBarrier(&[transition(
            arguments,
            D3D12_RESOURCE_STATE_INDIRECT_ARGUMENT,
            D3D12_RESOURCE_STATE_COMMON,
        )])
    };
}

/// Whether consecutive indirect draws are clamped a window at a time
/// (`VICTORY_CLAMP_WINDOW=1`; off by default).
///
/// Without it, each indirect execution has its own clamp: a wait for all
/// earlier work, a compute dispatch, another wait, then the draw, so the
/// card goes from drawing to compute and back for every record and no
/// draw overlaps the next.
///
/// With it, an indirect draw opens a window: one clamp pass is recorded
/// where the single clamp would be, and the draws after it that may join
/// add their records to the pass's table (upload memory the CPU fills
/// while recording; the GPU reads it when the list runs). They read their
/// clamped records from the window's part of the list's argument buffer,
/// with no pass and no barrier of their own. The pass clamps every record
/// of the window in the order the draws come, charging the list's budget
/// in that order, exactly what the single clamps did one after another.
///
/// Moving a record's clamp before the window's earlier draws changes
/// nothing as long as none of them writes buffers. So a window ends (the
/// next indirect draw opens a new one, after a wait for all before) at:
/// any draw, dispatch or unordered-access clear other than a joining
/// indirect draw ([`end_window`] in `crate::draw`), an indirect dispatch,
/// an indirect draw whose shaders may store to memory (unordered views,
/// heap views, title memory), work a host library records, the list's
/// end; and a draw does not join when the table, the window's argument
/// bytes or its records are full, or its arguments lie in a host copy the
/// pass has no view of. Title copies between buffers run on the CPU at
/// submission, before the list, and stream output is not translated: no
/// other command of a list writes buffer copies.
///
/// With [`gaps_on`] as well, executions whose shaders may store join too,
/// indirect dispatches included: see [`Run`].
pub(crate) fn windows_on() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| crate::switch::on("VICTORY_CLAMP_WINDOW"))
}

/// Whether indirect executions whose shaders may store (indirect
/// dispatches, and draws [`crate::draw`] finds may write arguments) are
/// clamped in windows too (`VICTORY_GPU_GAPS=1`, with
/// `VICTORY_CLAMP_WINDOW=1`; off by default), as long as nothing but
/// indirect executions of the window comes between them ([`Run`]).
///
/// Without it, each such execution has a clamp of its own: a wait for all
/// earlier work, a one-group pass, another wait, the execution, then a
/// transition back that waits for the execution's compute work. A frame
/// has ~900 indirect dispatches, ~89 % of them right after another
/// indirect dispatch with no barrier between (two traced world frames),
/// so the card ran them one at a time, each behind a pass of ours.
pub(crate) fn gaps_on() -> bool {
    static ON: std::sync::LazyLock<bool> = std::sync::LazyLock::new(|| crate::switch::on("VICTORY_GPU_GAPS"));
    *ON
}

/// What may join a window that took an execution whose shaders may store
/// ([`gaps_on`]).
///
/// The window's pass clamps every record before the window's first
/// execution runs. An execution that stores could write the records of
/// the next one; on the console, the next one reads its records when the
/// command processor reaches it, which may be while the one before still
/// runs, unless the title put a barrier or a pipeline sync between them:
/// records written by one execution and read by the next with neither
/// between race on the console too, so the title does not do that. So
/// after an execution that may store, the window takes the next indirect
/// execution only if no barrier or pipeline sync came between ([`stale`],
/// `crate::list`); any other draw, dispatch or clear ends the window
/// anyway ([`end_window`]), as do host-library work and the list's end.
/// Barriers between executions that do not store leave the window as it
/// was: nothing in it wrote.
///
/// The executions then also run without our waits between them, as the
/// title recorded them: in base each indirect dispatch was ordered against
/// its neighbours only by our clamps' barriers, which the title never
/// asked for (direct dispatches have none either).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct Run {
    /// An execution in the window may store.
    stored: bool,
    /// Nothing but executions of the window was recorded since its last.
    fresh: bool,
}

impl Run {
    /// Whether the window may take another execution (it fits otherwise).
    fn takes(self) -> bool {
        !self.stored || self.fresh
    }

    /// The window took an execution that may store (`stores`) or not.
    fn took(&mut self, stores: bool) {
        self.stored |= stores;
        self.fresh = true;
    }

    /// A barrier or pipeline sync was recorded.
    fn stale(&mut self) {
        self.fresh = false;
    }
}

/// Why an indirect execution is clamped alone ([`alone`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Alone {
    /// Windows are off.
    NoWindows,
    /// An indirect dispatch (`VICTORY_GPU_GAPS` off).
    Dispatch,
    /// Its groups are spread.
    Spread,
    /// It may store (`VICTORY_GPU_GAPS` off).
    Stores,
}

/// Why an indirect execution not hoisted is clamped alone, or `None` if a
/// window may take it (`windows`: `VICTORY_CLAMP_WINDOW`, `gaps`:
/// [`gaps_on`]): `dispatch` an indirect dispatch, `stores` its shaders may
/// store, `spread` its groups are spread (only the single clamp writes
/// that layout).
pub(crate) fn alone(windows: bool, gaps: bool, dispatch: bool, stores: bool, spread: bool) -> Option<Alone> {
    if !windows {
        Some(Alone::NoWindows)
    } else if spread {
        Some(Alone::Spread)
    } else if gaps {
        None
    } else if dispatch {
        Some(Alone::Dispatch)
    } else if stores {
        Some(Alone::Stores)
    } else {
        None
    }
}

/// Counts an execution clamped alone by why ([`alone`]; `None`: a window
/// or the prelude refused it). Storing draws are `ClampStoring` already,
/// and with windows off every clamp is alone.
#[cfg(feature = "diagnostics")]
pub(crate) fn count_alone(alone: Option<Alone>) {
    use crate::copystat::{C, add};
    match alone {
        Some(Alone::Dispatch) => add(C::ClampAloneDispatch, 1),
        Some(Alone::Spread) => add(C::ClampAloneSpread, 1),
        None => add(C::ClampAloneOther, 1),
        Some(Alone::Stores | Alone::NoWindows) => {}
    }
}

/// Notes the clamp pass about to be recorded in `list`
/// (`VICTORY_TIMED_CLAMP=1`, diagnostics): the GPU timing table charges
/// it, its barriers and the bindings its execution then needs again to a
/// row of its own (`clamp`, `clamp window`), not to the command before.
#[cfg(feature = "diagnostics")]
fn note_clamp(list: &HostList, what: &'static str) {
    static ON: std::sync::LazyLock<bool> = std::sync::LazyLock::new(|| crate::switch::on("VICTORY_TIMED_CLAMP"));
    if *ON {
        list.note_op(|| what.to_owned());
    }
}

/// Draws a window's table holds at most.
const WINDOW_ENTRIES: u32 = 64;
/// Bytes of clamped records a window holds at most (the list's argument
/// buffer is at least this long).
const WINDOW_BYTES: u64 = 1 << 16;
/// Records a window clamps at most.
const WINDOW_RECORDS: u32 = 4096;
/// Host copies a window's pass can read records from.
const WINDOW_SOURCES: usize = 4;
/// A table: a header of 4 words, then 8 words per draw.
const TABLE_BYTES: u64 = 16 + 32 * WINDOW_ENTRIES as u64;

// As `crate::compute` takes it: no `RootSignature` attribute
// ([`window_clamp`] makes it: t0 to t4, the budget u1, `dest` u0), and the
// budget a structured buffer of `uint`.
const WINDOW_SOURCE: &str = concat!(mesh_groups!(), r#"
// The host copies records are read from, only read.
ByteAddressBuffer a0 : register(t0);
ByteAddressBuffer a1 : register(t1);
ByteAddressBuffer a2 : register(t2);
ByteAddressBuffer a3 : register(t3);
// Records, draws; then per draw (at 16 + 32 * draw) its first record in the
// window, records, kind, stride, source, the records' offset in it, their
// offset in `dest`. Draws in order; none without records.
ByteAddressBuffer table : register(t4);
RWByteAddressBuffer dest : register(u0);
RWStructuredBuffer<uint> budget : register(u1);
static const uint BUDGET = 1u << 27;
static const uint GROUPS = 1u << 20;
uint word(uint source, uint at) {
    [branch] if (source == 0) {
        return a0.Load(at);
    }
    [branch] if (source == 1) {
        return a1.Load(at);
    }
    [branch] if (source == 2) {
        return a2.Load(at);
    }
    return a3.Load(at);
}
// The single clamp's bounds, word for word: the record of `kind` at `from`
// in `source` clamped into `dest` at `to`; its cost.
uint clamped(uint kind, uint source, uint from, uint to) {
    uint cost;
    if (kind <= 1) {
        uint count = min(word(source, from), 1u << 24);
        uint instances = min(word(source, from + 4), (1u << 26) / max(count, 1u));
        cost = count * instances;
        dest.Store2(to, uint2(count, instances));
        dest.Store2(to + 8, uint2(word(source, from + 8), word(source, from + 12)));
        if (kind == 1) {
            dest.Store(to + 16, word(source, from + 16));
        }
    } else if (kind == 2) {
        uint3 g = min(uint3(word(source, from), word(source, from + 4), word(source, from + 8)), uint3(65535, 65535, 65535));
        if (g.z != 0 && g.x * g.y > GROUPS / g.z) {
            g = uint3(0, 0, 0);
        }
        cost = g.x * g.y * g.z * 64;
        dest.Store3(to, g);
    } else {
        uint3 g = mesh_groups(uint3(word(source, from), word(source, from + 4), word(source, from + 8)));
        cost = g.x * g.y * g.z * 128;
        dest.Store3(to, g);
    }
    return cost;
}
static const uint PAST = 0xffffffffu;
groupshared uint costs[64];
// As the single clamp: one group, tiles of 64 records, the first lane
// charges each tile's records in order.
[numthreads(64, 1, 1)]
void main(uint lane : SV_GroupIndex) {
    uint records = table.Load(0);
    uint draws = min(table.Load(4), 64u);
    uint left = 0;
    if (lane == 0) {
        left = BUDGET - min(budget[0], BUDGET);
    }
    for (uint tile = 0; tile < records / 64 + min(records % 64, 1u); tile++) {
        uint record = tile * 64 + lane;
        uint cost = 0;
        uint to = 0;
        if (record < records) {
            uint draw = 0;
            for (uint k = 1; k < draws; k++) {
                if (table.Load(16 + 32 * k) <= record) {
                    draw = k;
                }
            }
            uint4 head = table.Load4(16 + 32 * draw);
            uint3 place = table.Load3(32 + 32 * draw);
            uint index = record - head.x;
            to = place.z + index * head.w;
            cost = clamped(head.z, place.x, place.y + index * head.w, to);
        }
        costs[lane] = cost;
        GroupMemoryBarrierWithGroupSync();
        if (lane == 0) {
            for (uint i = 0; i < 64; i++) {
                if (costs[i] > left) {
                    costs[i] = PAST;
                    left = 0;
                } else {
                    left -= costs[i];
                }
            }
        }
        GroupMemoryBarrierWithGroupSync();
        if (costs[lane] == PAST) {
            dest.Store(to, 0);
        }
    }
    if (lane == 0) {
        budget[0] = BUDGET - left;
    }
}
"#);

fn window_clamp() -> Option<&'static Clamp> {
    use D3D12_ROOT_PARAMETER_TYPE_SRV as SRV;
    use D3D12_ROOT_PARAMETER_TYPE_UAV as UAV;
    static CLAMP: std::sync::OnceLock<Option<Clamp>> = std::sync::OnceLock::new();
    CLAMP
        .get_or_init(|| {
            // As [`open`] binds them: the sources and the table, the budget,
            // the window's records.
            let mut params: Vec<_> = (0..=WINDOW_SOURCES as u32).map(|t| root_view(SRV, t, 0)).collect();
            params.extend([root_view(UAV, 1, 0), root_view(UAV, 0, 0)]);
            match make_with(WINDOW_SOURCE, s!("clamp window"), &params) {
                Ok(c) => Some(c),
                Err(e) => {
                    tracing::warn!(error = %e, "no windowed argument clamp: each indirect draw clamped on its own");
                    None
                }
            }
        })
        .as_ref()
}

/// Whether a window was ever opened: until then (always, with the switch
/// off) ending and forgetting windows takes no lock.
static OPENED: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

/// The window a list has open, and the host copies its windows read.
#[derive(Default)]
pub(crate) struct Window {
    open: Option<Open>,
    /// Host GPU addresses of copies earlier windows of the list read
    /// (newest first): the next window's pass gets views of them too. The
    /// list's refreshes hold those copies until it is reset.
    recent: Vec<u64>,
}

struct Open {
    /// The list's argument buffer, indirect arguments until the window ends.
    out: ID3D12Resource,
    /// The table's CPU address (upload memory of the list).
    table: usize,
    /// The copies the pass reads: host GPU addresses, t0 to t3.
    sources: Vec<u64>,
    draws: u32,
    records: u32,
    bytes: u64,
    run: Run,
}

/// Ends `list`'s window, if one is open: its argument buffer back to
/// common. Before any command that may write buffers.
pub(crate) fn end_window(list: &HostList) {
    if !OPENED.load(std::sync::atomic::Ordering::Relaxed) {
        return;
    }
    let open = list.clamp_window().open.take();
    if let Some(open) = open {
        done(list, &open.out);
    }
}

/// A barrier or pipeline sync in `list`: its open window takes no more
/// executions if one it took may store ([`Run`]).
pub(crate) fn stale_window(list: &HostList) {
    if !OPENED.load(std::sync::atomic::Ordering::Relaxed) {
        return;
    }
    if let Some(open) = list.clamp_window().open.as_mut() {
        open.run.stale();
    }
}

/// Forgets `list`'s windows (it was reset).
pub(crate) fn forget_windows(list: &HostList) {
    if OPENED.load(std::sync::atomic::Ordering::Relaxed) {
        *list.clamp_window() = Window::default();
    }
}

/// The clamped records of an indirect execution of `records` records of
/// `kind` `stride` bytes apart, `offset` bytes into the host copy at host
/// GPU address `source`, whose shaders may store (`stores`, only with
/// [`gaps_on`]), in `list`'s open window, or in one opened for it: the
/// argument buffer, the records' offset in it, and whether a window was
/// opened (its pass replaced the list's compute root signature and
/// pipeline). `None` if it cannot be clamped in a window; the caller
/// clamps it on its own ([`bound`]).
pub(crate) fn windowed(
    list: &HostList,
    source: u64,
    offset: u64,
    (kind, stride, records): (u32, u32, u32),
    stores: bool,
) -> Option<(ID3D12Resource, u64, bool)> {
    let bytes = u64::from(stride) * u64::from(records.max(1));
    if !source.is_multiple_of(4)
        || !offset.is_multiple_of(4)
        || !stride.is_multiple_of(4)
        || records > WINDOW_RECORDS
        || bytes > WINDOW_BYTES
        || offset.checked_add(bytes)? > u64::from(u32::MAX)
    {
        return None;
    }
    let mut window = list.clamp_window();
    let mut opened = false;
    let fits = window.open.as_ref().is_some_and(|o| o.fits(source, bytes, records));
    let takes = fits && window.open.as_ref().is_some_and(|o| o.run.takes());
    #[cfg(feature = "diagnostics")]
    if fits && !takes {
        crate::copystat::add(crate::copystat::C::ClampGapsStale, 1);
    }
    if !takes {
        #[cfg(feature = "diagnostics")]
        note_clamp(list, "clamp window");
        if let Some(old) = window.open.take() {
            done(list, &old.out);
        }
        let mut sources = vec![source];
        sources.extend(window.recent.iter().copied().filter(|&s| s != source).take(WINDOW_SOURCES - 1));
        window.recent.clone_from(&sources);
        OPENED.store(true, std::sync::atomic::Ordering::Relaxed);
        window.open = Some(open(list, sources)?);
        opened = true;
    }
    let o = window.open.as_mut()?;
    o.run.took(stores);
    let at = o.bytes;
    if records > 0 {
        let index = o.sources.iter().position(|&s| s == source)? as u32;
        let entry = [o.records, records, kind, stride, index, offset as u32, at as u32, 0];
        // SAFETY: the table's words lie in the list's upload memory, alive
        // until it is reset; the GPU reads them only once the list runs.
        unsafe {
            let table = o.table as *mut u32;
            for (k, &w) in entry.iter().enumerate() {
                table.add(4 + 8 * o.draws as usize + k).write_volatile(w);
            }
            o.draws += 1;
            o.records += records;
            table.write_volatile(o.records);
            table.add(1).write_volatile(o.draws);
        }
    }
    o.bytes += bytes;
    #[cfg(feature = "diagnostics")]
    {
        crate::copystat::add(crate::copystat::C::ClampWindowDraws, 1);
        if opened {
            crate::copystat::add(crate::copystat::C::ClampWindows, 1);
        }
    }
    Some((o.out.clone(), at, opened))
}

impl Open {
    /// Whether a draw of `records` records (`bytes` bytes) in the copy at
    /// `source` joins the window.
    fn fits(&self, source: u64, bytes: u64, records: u32) -> bool {
        self.draws < WINDOW_ENTRIES
            && self.bytes + bytes <= WINDOW_BYTES
            && self.records + records <= WINDOW_RECORDS
            && self.sources.contains(&source)
    }
}

/// Whether an indirect execution (as [`windowed`] takes it) joins `list`'s
/// open window (tests: to read a window's records before the next one).
#[cfg(test)]
fn joins(list: &HostList, source: u64, (_, stride, records): (u32, u32, u32)) -> bool {
    let bytes = u64::from(stride) * u64::from(records.max(1));
    list.clamp_window().open.as_ref().is_some_and(|o| o.fits(source, bytes, records) && o.run.takes())
}

/// Records a window's pass in `list`, reading records from `sources`.
fn open(list: &HostList, sources: Vec<u64>) -> Option<Open> {
    let (Some(c), Some(budget)) = (window_clamp(), list.budget()) else {
        return None;
    };
    let out = list.arguments(WINDOW_BYTES)?;
    let (table_gpu, table) = list.scratch(TABLE_BYTES)?;
    // SAFETY: the header of the list's fresh upload memory.
    unsafe { std::ptr::write_bytes(table, 0, 16) };
    let l = &list.list;
    // SAFETY: an open list; our root signature, pipeline and buffers, and
    // host copies the list's refreshes hold.
    unsafe {
        // Earlier shader writes (the title's argument passes) first.
        l.ResourceBarrier(&[
            uav_barrier(),
            transition(&out, D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_UNORDERED_ACCESS),
        ]);
        #[cfg(feature = "diagnostics")]
        crate::copystat::add(crate::copystat::C::ClampBarrierCalls, 2);
        l.SetComputeRootSignature(&c.root);
        l.SetPipelineState(&c.pso);
        for k in 0..WINDOW_SOURCES {
            l.SetComputeRootShaderResourceView(k as u32, *sources.get(k).unwrap_or(&sources[0]));
        }
        l.SetComputeRootShaderResourceView(4, table_gpu);
        l.SetComputeRootUnorderedAccessView(5, budget);
        l.SetComputeRootUnorderedAccessView(6, out.GetGPUVirtualAddress());
        l.Dispatch(1, 1, 1);
        l.ResourceBarrier(&[
            uav_barrier(),
            transition(&out, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_INDIRECT_ARGUMENT),
        ]);
    }
    Some(Open { out, table: table as usize, sources, draws: 0, records: 0, bytes: 0, run: Run::default() })
}

fn uav_barrier() -> D3D12_RESOURCE_BARRIER {
    uav_barrier_of(None)
}

fn uav_barrier_of(resource: Option<&ID3D12Resource>) -> D3D12_RESOURCE_BARRIER {
    D3D12_RESOURCE_BARRIER {
        Type: D3D12_RESOURCE_BARRIER_TYPE_UAV,
        Flags: D3D12_RESOURCE_BARRIER_FLAG_NONE,
        Anonymous: D3D12_RESOURCE_BARRIER_0 {
            UAV: std::mem::ManuallyDrop::new(D3D12_RESOURCE_UAV_BARRIER {
                pResource: std::mem::ManuallyDrop::new(resource.map(|r| {
                    // SAFETY: a borrowed pointer; the barrier never releases it.
                    unsafe { std::mem::transmute_copy(r) }
                })),
            }),
        },
    }
}

#[cfg(test)]
mod tests {
    /// The clamp compiles with the system's compiler and the host takes it.
    #[test]
    fn argument_clamp_builds() {
        if super::host().is_none() {
            return; // no host GPU here
        }
        assert!(super::clamp().is_some());
    }

    /// The single, batched and windowed clamps compile with the system's
    /// compiler, Wine's included (no device: their root signatures are
    /// made apart). Wine's took neither of the latter two while they read
    /// their budgets through `RWByteAddressBuffer::Load` (E5005) and
    /// declared their root signatures in an attribute Wine's ignores, so
    /// every hoisted or windowed clamp fell back to a single one there.
    #[test]
    fn every_clamp_compiles_without_a_root_signature_attribute() {
        use windows::Win32::Graphics::Direct3D::{Fxc::D3DCompile, ID3DBlob};
        use windows::core::s;
        for (source, name) in [
            (super::SOURCE, s!("argument clamp")),
            (super::BATCH_SOURCE, s!("batch clamp")),
            (super::WINDOW_SOURCE, s!("clamp window")),
        ] {
            assert!(!source.contains("RootSignature"));
            let (mut code, mut errors): (Option<ID3DBlob>, Option<ID3DBlob>) = (None, None);
            // SAFETY: the source and names live through the call; out pointers ours.
            let compiled = unsafe {
                D3DCompile(source.as_ptr().cast(), source.len(), name, None, None, s!("main"), s!("cs_5_1"), 0, 0, &mut code, Some(&mut errors))
            };
            let text = errors.as_ref().map(|e| String::from_utf8_lossy(crate::compute::bytes(e)).into_owned());
            // SAFETY: a static name.
            assert!(compiled.is_ok() && code.is_some(), "{}: {text:?}", unsafe { name.display() });
        }
    }

    /// Clamps the records of `kind` in `title`, `per` words apart, in a
    /// list of their own: every record word of the list's argument buffer
    /// after, and the list's budget spent. `None` without a host GPU.
    fn every_word(kind: u32, per: usize, title: &[u32]) -> Option<(Vec<u32>, u32)> {
        let host = super::host()?;
        let records = (title.len() / per) as u32;
        let (buffer, data) = host.shared_buffer(std::mem::size_of_val(title) as u64).unwrap();
        let list = host.new_list().unwrap();
        // SAFETY: our mapped buffer, written before the list runs; the
        // list's argument buffer read after it has.
        unsafe {
            data.cast::<u32>().copy_from_nonoverlapping(title.as_ptr(), title.len());
            let out = super::bound(&list, buffer.GetGPUVirtualAddress(), (kind, per as u32 * 4, records), false).unwrap();
            super::done(&list, &out);
            list.close();
            host.execute(&[&list]);
            let mut p = std::ptr::null_mut();
            out.Map(0, None, Some(&mut p)).unwrap();
            let words = std::slice::from_raw_parts(p.cast::<u32>(), title.len()).to_vec();
            Some((words, list.budget_spent().unwrap()))
        }
    }

    /// Every word the clamp writes, of each kind: counts and groups
    /// bounded, the rest of a record copied as it is (5-word records
    /// 20 bytes apart: no words of one land in another), the work charged
    /// to the list's budget; records past it run empty with their other
    /// words kept, and one of no work still runs.
    #[test]
    fn clamped_records_hold_every_word() {
        const MANY: u32 = 1 << 24;
        // Draws: vertex count, instances, first vertex, first instance.
        let Some(draws) = every_word(0, 4, &[MANY + 7, 5, 11, 13, 3, 2, 1, 9]) else {
            return; // no host GPU here
        };
        assert_eq!(draws, (vec![MANY, 4, 11, 13, 3, 2, 1, 9], (1 << 26) + 6));
        // Indexed: index count, instances, first index, base vertex, first
        // instance.
        let indexed = every_word(1, 5, &[10, 3, 100, -16i32 as u32, 7, u32::MAX, u32::MAX, 1, 2, 3]).unwrap();
        assert_eq!(indexed, (vec![10, 3, 100, -16i32 as u32, 7, MANY, 4, 1, 2, 3], (1 << 26) + 30));
        // Dispatches: each dimension bounded, and a dispatch of more than
        // 2^20 groups none.
        let dispatches = every_word(2, 3, &[3, 2, 1, 65_536, 17, 2, 70_000, 7, 2]).unwrap();
        assert_eq!(dispatches, (vec![3, 2, 1, 0, 0, 0, 65_535, 7, 2], (6 + 65_535 * 14) * 64));
        // Mesh dispatches: one-dimensional.
        let meshes = every_word(10, 3, &[70_000, 7, 2, 5, 0, 3]).unwrap();
        assert_eq!(meshes, (vec![65_535, 1, 1, 5, 0, 1], 65_535 * 128));
        // Two draws of 2^26 vertex instances spend the budget; a third, a
        // small one after it, run empty; one of no work runs.
        let mut title = [MANY, 4, 0, 0].repeat(3);
        title.extend([1, 1, 5, 6, 0, 9, 1, 2]);
        let past = every_word(0, 4, &title).unwrap();
        let mut want = [MANY, 4, 0, 0].repeat(2);
        want.extend([0, 4, 0, 0, 0, 1, 5, 6, 0, 9, 1, 2]);
        assert_eq!(past, (want, 1 << 27));
    }

    /// Clamps `records` records of `kind`, filled by repeating `words`,
    /// `calls` times in one list: each record's first word in the list's
    /// argument buffer after (zero when it runs empty). The title's copy,
    /// one record more included, stays as written. `None` without a host
    /// GPU.
    fn clamped(kind: u32, words: &[u32], records: u32, calls: u32) -> Option<Vec<u32>> {
        let host = super::host()?;
        // Words in a record of `kind`.
        let per = match kind {
            0 => 4,
            1 => 5,
            _ => 3,
        };
        let spec = (kind, per as u32 * 4, records);
        let all = per * (records as usize + 1);
        let (buffer, data) = host.shared_buffer(all as u64 * 4).unwrap();
        let data = data.cast::<u32>();
        let list = host.new_list().unwrap();
        // SAFETY: our mapped buffer, written before the list runs and read
        // after it has; a live buffer.
        unsafe {
            for at in 0..all {
                data.add(at).write_volatile(words[at % words.len()]);
            }
            let gpu = buffer.GetGPUVirtualAddress();
            let mut out = None;
            for _ in 0..calls {
                let clamped = super::bound(&list, gpu, spec, false).unwrap();
                super::done(&list, &clamped);
                out = Some(clamped);
            }
            list.close();
            host.execute(&[&list]);
            for at in 0..all {
                assert_eq!(data.add(at).read_volatile(), words[at % words.len()], "the title's copy as written");
            }
            let mut out_data = std::ptr::null_mut();
            out?.Map(0, None, Some(&mut out_data)).unwrap();
            let first = |i: usize| out_data.cast::<u32>().add(i * per).read_volatile();
            Some((0..records as usize).map(first).collect())
        }
    }

    /// Rejected records do not wrap the budget: of 128 garbage draws (2^26
    /// vertex instances each once clamped) two run, where a wrapping add
    /// let four; of 600 garbage mesh dispatches 16, not 31.
    #[test]
    fn rejected_records_do_not_wrap_the_budget() {
        let Some(draws) = clamped(0, &[u32::MAX], 128, 1) else {
            return; // no host GPU here
        };
        assert_eq!(draws.iter().filter(|&&count| count != 0).count(), 2);
        let meshes = clamped(10, &[u32::MAX], 600, 1).unwrap();
        assert_eq!(meshes.iter().filter(|&&groups| groups != 0).count(), 16);
    }

    /// The budget goes to the first records, in order: of 200 draws of
    /// 3 * 2^18 vertices the first 170 run (it ends inside the third tile);
    /// small records that all fit all run.
    #[test]
    fn the_budget_goes_to_the_first_records() {
        let Some(draws) = clamped(0, &[3 << 18, 1, 0, 0], 200, 1) else {
            return; // no host GPU here
        };
        assert!(draws[..170].iter().all(|&count| count != 0));
        assert!(draws[170..].iter().all(|&count| count == 0));
        let small = clamped(0, &[1], 1 << 16, 1).unwrap();
        assert!(small.iter().all(|&count| count == 1));
    }

    /// A record that does not fit ends the budget: after two draws of
    /// 3 * 2^24 vertex instances a third takes what is left, and the small
    /// ones after it run empty.
    #[test]
    fn a_record_that_does_not_fit_ends_the_budget() {
        let mut words = [1 << 24, 3, 0, 0].repeat(3);
        words.extend([3, 1, 0, 0].repeat(6));
        let Some(draws) = clamped(0, &words, 8, 1) else {
            return; // no host GPU here
        };
        let ran: Vec<bool> = draws.iter().map(|&count| count != 0).collect();
        assert_eq!(ran, [true, true, false, false, false, false, false, false]);
    }

    /// Two lists clamping into their own argument buffers, run in either
    /// order: every barrier starts from the buffer's current state (the
    /// debug layer's rule; runs light19 and light21 had the title's GPU
    /// heap moved to unordered access under the input assembler), the
    /// title's copy is never transitioned, and all end common.
    #[test]
    fn argument_buffers_move_from_their_current_states() {
        use crate::host::tests::{issued, replay};
        use windows::core::Interface;
        let Some(host) = super::host() else {
            return; // no host GPU here
        };
        let (title, data) = host.shared_buffer(1 << 16).unwrap();
        // SAFETY: our mapped buffer; a live buffer.
        let gpu = unsafe {
            data.cast::<[u32; 3]>().write_volatile([70_000, 1, 1]);
            title.GetGPUVirtualAddress()
        };
        let key = |r: &super::ID3D12Resource| r.as_raw() as usize;
        for a_first in [true, false] {
            let lists = [host.new_list().unwrap(), host.new_list().unwrap()];
            issued();
            let recorded: Vec<_> = lists
                .iter()
                .map(|l| {
                    let out = super::bound(l, gpu, (2, 12, 1), false).unwrap();
                    super::done(l, &out);
                    l.close();
                    (out, issued())
                })
                .collect();
            let order = if a_first { [0, 1] } else { [1, 0] };
            host.execute(&[&lists[order[0]], &lists[order[1]]]);
            let mut run = issued();
            run.extend(order.iter().flat_map(|&i| recorded[i].1.clone()));
            let mut states: std::collections::HashMap<usize, u32> =
                [key(&title), key(&recorded[0].0), key(&recorded[1].0)].into_iter().map(|k| (k, 0)).collect();
            replay(&run, &mut states).unwrap();
            assert!(run.iter().all(|t| t.0 != key(&title)), "the title's copy is never transitioned");
            assert!(states.values().all(|&s| s == 0), "all common again");
            for (out, _) in &recorded {
                let mut p = std::ptr::null_mut();
                // SAFETY: a mapped buffer of ours, the lists have run.
                let clamped = unsafe {
                    out.Map(0, None, Some(&mut p)).unwrap();
                    p.cast::<[u32; 3]>().read_volatile()
                };
                assert_eq!(clamped, [65_535, 1, 1]);
            }
            // SAFETY: as above.
            assert_eq!(unsafe { data.cast::<[u32; 3]>().read_volatile() }, [70_000, 1, 1], "the title's copy as written");
        }
    }

    /// What a test list records, in order.
    #[derive(Clone, Copy)]
    enum Step {
        /// An indirect draw: title buffer, byte offset, kind, records.
        Draw(usize, u64, u32, u32),
        /// As `Draw`, an execution whose shaders may store (an indirect
        /// dispatch, or a draw `crate::draw` finds may write arguments):
        /// with `VICTORY_GPU_GAPS` it joins windows ([`super::Run`]).
        Stores(usize, u64, u32, u32),
        /// A shader of the title writes `value` at a byte offset of a title
        /// buffer (a draw or dispatch that may store: it ends the window).
        Write(usize, u64, u32),
        /// As `Write`, with the window left open: what a window that did not
        /// end there would clamp.
        WriteUnended(usize, u64, u32),
        /// The title's barrier (or pipeline sync): every earlier shader
        /// write first ([`super::stale_window`]).
        Barrier,
    }

    fn stride(kind: u32) -> u32 {
        match kind {
            0 => 16,
            1 => 20,
            _ => 12,
        }
    }

    const WRITE: &str = r#"
cbuffer Word : register(b0) { uint at; uint value; };
RWByteAddressBuffer title : register(u0);
[numthreads(1, 1, 1)]
void main() { title.Store(at, value); }
"#;

    /// The open window's draws (offset in its buffer, bytes, offset in the
    /// results) and its buffer.
    type Pending = (Vec<(u64, u64, u64)>, Option<super::ID3D12Resource>);

    /// Records `steps` into one list over title buffers holding `init`,
    /// each draw clamped on its own or (`window`) in windows, and runs it:
    /// the words of each draw's clamped records. `None` without a host GPU.
    fn run(init: &[Vec<u32>], steps: &[Step], window: bool) -> Option<Vec<Vec<u32>>> {
        use super::{D3D12_RESOURCE_STATE_COMMON as COMMON, D3D12_RESOURCE_STATE_COPY_SOURCE as COPY_SOURCE};
        use super::{D3D12_RESOURCE_STATE_INDIRECT_ARGUMENT as ARGUMENTS, transition};
        let host = super::host()?;
        let titles: Vec<_> = init
            .iter()
            .map(|words| {
                let (buffer, data) = host.shared_buffer(words.len() as u64 * 4).unwrap();
                // SAFETY: our mapped buffer, written before the list runs; a
                // live buffer.
                let gpu = unsafe {
                    std::ptr::copy_nonoverlapping(words.as_ptr(), data.cast::<u32>(), words.len());
                    buffer.GetGPUVirtualAddress()
                };
                (buffer, gpu)
            })
            .collect();
        let (results, data) = host.shared_buffer(1 << 20).unwrap();
        let writer = super::make_with(
            WRITE,
            super::s!("write"),
            &[super::constants(0, 2), super::root_view(super::D3D12_ROOT_PARAMETER_TYPE_UAV, 0, 0)],
        )
        .unwrap();
        let list = host.new_list().unwrap();
        // Each draw's records: offset in `results`, words.
        let mut placed = Vec::new();
        let mut at_results = 0u64;
        let mut pending: Pending = (Vec::new(), None);
        // Ends the window and copies its records into `results`.
        let flush = |(draws, out): &mut Pending| {
            super::end_window(&list);
            if let (Some(out), false) = (out.as_ref(), draws.is_empty()) {
                // SAFETY: an open list, live buffers.
                unsafe {
                    list.list.ResourceBarrier(&[transition(out, COMMON, COPY_SOURCE)]);
                    for &(at, len, to) in draws.iter() {
                        list.list.CopyBufferRegion(&results, to, out, at, len);
                    }
                    list.list.ResourceBarrier(&[transition(out, COPY_SOURCE, COMMON)]);
                }
            }
            draws.clear();
        };
        for &step in steps {
            match step {
                Step::Draw(buffer, offset, kind, records) | Step::Stores(buffer, offset, kind, records) => {
                    let spec = (kind, stride(kind), records);
                    let len = u64::from(stride(kind)) * u64::from(records.max(1));
                    let source = titles[buffer].1;
                    if window {
                        if !super::joins(&list, source, spec) {
                            flush(&mut pending);
                        }
                        let stores = matches!(step, Step::Stores(..));
                        let (out, at, _) = super::windowed(&list, source, offset, spec, stores).expect("a window");
                        pending.0.push((at, len, at_results));
                        pending.1 = Some(out);
                    } else {
                        let out = super::bound(&list, source + offset, spec, false).unwrap();
                        // SAFETY: an open list, live buffers.
                        unsafe {
                            list.list.ResourceBarrier(&[transition(&out, ARGUMENTS, COPY_SOURCE)]);
                            list.list.CopyBufferRegion(&results, at_results, &out, 0, len);
                            list.list.ResourceBarrier(&[transition(&out, COPY_SOURCE, ARGUMENTS)]);
                        }
                        super::done(&list, &out);
                    }
                    // An execution of no records reads none: nothing to
                    // compare (the single clamp leaves what was there).
                    placed.push((at_results, if records == 0 { 0 } else { len as usize / 4 }));
                    at_results += len.next_multiple_of(256);
                }
                Step::Write(buffer, at, value) | Step::WriteUnended(buffer, at, value) => {
                    if window && matches!(step, Step::Write(..)) {
                        flush(&mut pending);
                    }
                    let constants = [at as u32, value];
                    // SAFETY: an open list; our pipeline and a title buffer.
                    unsafe {
                        list.list.SetComputeRootSignature(&writer.root);
                        list.list.SetPipelineState(&writer.pso);
                        list.list.SetComputeRoot32BitConstants(0, 2, constants.as_ptr().cast(), 0);
                        list.list.SetComputeRootUnorderedAccessView(1, titles[buffer].1);
                        list.list.Dispatch(1, 1, 1);
                    }
                }
                Step::Barrier => {
                    if window {
                        super::stale_window(&list);
                    }
                    // SAFETY: an open list.
                    unsafe { list.list.ResourceBarrier(&[super::uav_barrier()]) };
                }
            }
        }
        if window {
            flush(&mut pending);
        }
        list.close();
        host.execute(&[&list]);
        let words = |&(at, n): &(u64, usize)| {
            // SAFETY: inside our mapped buffer; the list has run.
            unsafe { std::slice::from_raw_parts(data.add(at as usize).cast::<u32>(), n).to_vec() }
        };
        Some(placed.iter().map(words).collect())
    }

    /// The windows' records, after checking they are the single clamps'
    /// word for word.
    fn same(init: &[Vec<u32>], steps: &[Step]) -> Option<Vec<Vec<u32>>> {
        let single = run(init, steps, false)?;
        let windowed = run(init, steps, true)?;
        assert_eq!(single.len(), windowed.len());
        for (k, (a, b)) in single.iter().zip(&windowed).enumerate() {
            assert_eq!(a, b, "draw {k}: single {a:?}, windowed {b:?}");
        }
        Some(windowed)
    }

    /// Mesh dispatches, draws and indexed draws from two buffers, every
    /// seventh mesh record garbage (all ones) so that the budget runs out
    /// in the middle; more draws than a table holds, one of 100 records
    /// across two tiles and one of none: the windows clamp what the single
    /// clamps do.
    #[test]
    fn windows_clamp_as_single_clamps_do() {
        let mut a = Vec::new();
        for i in 0..400u32 {
            a.extend(if i % 7 == 3 { [u32::MAX; 3] } else { [100 + i, 1 + i % 3, i % 2] });
        }
        let draws = a.len() as u64 * 4;
        for i in 0..40u32 {
            a.extend(if i % 9 == 4 { [u32::MAX; 4] } else { [3 * i + 1, 2, i, 0] });
        }
        let indexed = a.len() as u64 * 4;
        for i in 0..40u32 {
            a.extend([6 * i, 1 + i, 7, 1, i]);
        }
        let b: Vec<u32> = (0..64u32).flat_map(|i| [70_000 + i, 2, 3]).collect();
        let mut steps = Vec::new();
        for i in 0..200u64 {
            steps.push(Step::Draw(0, 12 * i, 10, 1));
            if i % 10 == 5 {
                steps.push(Step::Draw(1, 12 * (i % 64), 10, 1));
                steps.push(Step::Draw(0, draws + 16 * (i % 40), 0, 1));
                steps.push(Step::Draw(0, indexed + 20 * (i % 40), 1, 1));
            }
            if i == 120 {
                steps.push(Step::Draw(0, 12 * 200, 10, 100));
                steps.push(Step::Draw(1, 0, 10, 0));
            }
        }
        let Some(out) = same(&[a, b], &steps) else {
            return; // no host GPU here
        };
        let ran = out.iter().filter(|r| r.first().is_some_and(|&w| w != 0)).count();
        assert!(ran > 50 && ran + 50 < out.len(), "the budget ran out in the middle: {ran} of {} ran", out.len());
        assert_eq!(out[0], [100, 1, 0], "a record as it is");
    }

    /// Mesh grids with more than one group along y or z (the case reported
    /// on #3197 with #2996: `(3, 2, 4)` read back as 24 groups clamped on
    /// its own, 3 in a window): a window keeps the single clamp's mesh
    /// rule, whatever that rule is, word for word.
    #[test]
    fn windowed_mesh_grids_are_the_single_clamps() {
        let a: Vec<u32> = [[3, 2, 4], [2, 3, 5], [65_535, 1, 1], [257, 256, 1], [3, 0, 4], [1, 1, 1]].concat();
        let steps: Vec<Step> = (0..6).map(|i| Step::Draw(0, 12 * i, 10, 1)).collect();
        let Some(out) = same(&[a], &steps) else {
            return; // no host GPU here
        };
        assert_eq!(out[5], [1, 1, 1]);
    }

    /// A budget spent before the window opens stays spent: its draws all
    /// run empty, as they do clamped one by one.
    #[test]
    fn a_window_after_a_spent_budget_runs_empty() {
        let mut a: Vec<u32> = [u32::MAX; 3].repeat(20);
        a.extend([5, 1, 1].repeat(20));
        let mut steps: Vec<Step> = (0..20).map(|i| Step::Draw(0, 12 * i, 10, 1)).collect();
        // A store in between: the rest in a window of its own.
        steps.push(Step::Write(0, 0, u32::MAX));
        steps.extend((20..40).map(|i| Step::Draw(0, 12 * i, 10, 1)));
        let Some(out) = same(&[a], &steps) else {
            return; // no host GPU here
        };
        assert_eq!(out.iter().take(20).filter(|r| r[0] != 0).count(), 16, "16 garbage records fit");
        assert!(out[20..].iter().all(|r| r[0] == 0), "the spent budget empties the window");
    }

    /// A shader writing a record between two draws ends the window: the
    /// draw after it reads the record written, as clamped on its own. A
    /// window left open would have clamped the record from before.
    #[test]
    fn a_store_ends_the_window() {
        let a = [5, 1, 1].repeat(4);
        let steps = |write: fn(usize, u64, u32) -> Step| {
            vec![Step::Draw(0, 0, 10, 1), Step::Draw(0, 12, 10, 1), write(0, 24, 9), Step::Draw(0, 24, 10, 1), Step::Draw(0, 36, 10, 1)]
        };
        let Some(out) = same(std::slice::from_ref(&a), &steps(Step::Write)) else {
            return; // no host GPU here
        };
        assert_eq!(out[2], [9, 1, 1], "the record as written");
        assert_eq!(out[3], [5, 1, 1]);
        let unended = run(&[a], &steps(Step::WriteUnended), true).unwrap();
        assert_eq!(unended[2], [5, 1, 1], "a window over the store clamps the record from before it");
    }

    /// Indirect dispatches in windows (`VICTORY_GPU_GAPS`): their records
    /// clamped as the single clamp does, word for word: groups past 65,535
    /// per dimension, too many groups in all, garbage, and a budget that
    /// runs out in the middle; draws between them in the same windows.
    #[test]
    fn dispatch_windows_clamp_as_single_clamps_do() {
        let mut a: Vec<u32> = Vec::new();
        for i in 0..60u32 {
            a.extend(match i % 6 {
                0 => [70_000, 2, 1],
                1 => [65_535, 65_535, 2],
                2 => [3 + i, 4, 5],
                3 => [u32::MAX; 3],
                4 => [1024, 1024, 1],
                _ => [0, 7, 7],
            });
        }
        let draws = a.len() as u64 * 4;
        a.extend([3, 2, 0, 0, 5, 1, 0, 0]);
        let mut steps = Vec::new();
        for i in 0..60u64 {
            steps.push(Step::Stores(0, 12 * i, 2, 1));
            if i % 20 == 9 {
                steps.push(Step::Draw(0, draws + 16 * (i % 2), 0, 1));
            }
        }
        steps.push(Step::Stores(0, 0, 2, 6));
        let Some(out) = same(&[a], &steps) else {
            return; // no host GPU here
        };
        assert_eq!(out[0], [65_535, 2, 1], "x past 65,535");
        assert_eq!(out[1], [0, 0, 0], "too many groups");
        assert_eq!(out[2], [5, 4, 5], "a record as it is");
        let ran = out.iter().filter(|r| r.first().is_some_and(|&w| w != 0)).count();
        assert!(ran + 10 < out.len(), "the budget ran out: {ran} of {} ran", out.len());
    }

    /// After an execution that may store, a barrier ends what the window
    /// takes (`VICTORY_GPU_GAPS`): the execution after it reads the record
    /// as the one before wrote it, as clamped on its own. With no barrier
    /// between (what the title records only for executions that do not
    /// depend on each other) the window takes it, clamped before the store.
    #[test]
    fn a_barrier_after_a_store_ends_what_the_window_takes() {
        let a = [5, 1, 1].repeat(4);
        let steps = |barrier: bool| {
            let mut s = vec![Step::Stores(0, 0, 2, 1), Step::WriteUnended(0, 12, 9)];
            s.extend(barrier.then_some(Step::Barrier));
            s.extend([Step::Stores(0, 12, 2, 1), Step::Draw(0, 24, 10, 1)]);
            s
        };
        let Some(out) = same(std::slice::from_ref(&a), &steps(true)) else {
            return; // no host GPU here
        };
        assert_eq!(out[1], [9, 1, 1], "the record as written before the barrier");
        let joined = run(&[a], &steps(false), true).unwrap();
        assert_eq!(joined[1], [5, 1, 1], "no barrier: one window, clamped before the store");
    }

    /// What a window may take after executions that may store (CPU only).
    #[test]
    fn a_window_takes_after_stores_only_until_a_barrier() {
        let mut run = super::Run::default();
        assert!(run.takes(), "an empty window");
        run.took(false);
        run.stale();
        assert!(run.takes(), "a barrier after executions that do not store");
        run.took(true);
        assert!(run.takes(), "right after one that may store");
        run.stale();
        assert!(!run.takes(), "a barrier after one that may store");
    }

    /// Which executions a window may take (CPU only).
    #[test]
    fn only_gaps_send_dispatches_and_stores_to_windows() {
        use super::{Alone, alone};
        // (windows, gaps, dispatch, stores, spread)
        assert_eq!(alone(false, true, true, true, false), Some(Alone::NoWindows));
        assert_eq!(alone(true, false, false, false, false), None, "a draw that does not store");
        assert_eq!(alone(true, false, true, true, false), Some(Alone::Dispatch));
        assert_eq!(alone(true, false, false, true, false), Some(Alone::Stores));
        assert_eq!(alone(true, true, true, true, false), None, "a dispatch with gaps");
        assert_eq!(alone(true, true, false, true, false), None, "a storing draw with gaps");
        assert_eq!(alone(true, true, true, true, true), Some(Alone::Spread));
        assert_eq!(alone(true, false, true, true, true), Some(Alone::Spread));
    }

    /// An execution of `max` records with a count buffer asking for more
    /// (the GPU runs min(count, max)): exactly `max` records are clamped,
    /// across two tiles, the garbage after them in the title's buffer is
    /// left out, and the next draw's records follow.
    #[test]
    fn a_window_clamps_max_records_whatever_the_count() {
        let mut a: Vec<u32> = (0..70u32).flat_map(|i| [10 + i, 1, 0, 0]).collect();
        a.extend([u32::MAX; 4].repeat(30));
        a.extend([4, 4, 0, 0]);
        let next = (a.len() as u64 - 4) * 4;
        let steps = [Step::Draw(0, 0, 0, 70), Step::Draw(0, next, 0, 1)];
        let Some(out) = same(&[a], &steps) else {
            return; // no host GPU here
        };
        assert_eq!(out[0].len(), 70 * 4);
        assert!((0..70).all(|i| out[0][4 * i] == 10 + i as u32));
        assert_eq!(out[1], [4, 4, 0, 0]);
    }

    /// Windows keep the argument buffer's states right: indirect arguments
    /// while open, common when the list closes, whatever order lists run in.
    #[test]
    fn windows_leave_the_argument_buffer_common() {
        use crate::host::tests::{issued, replay};
        use windows::core::Interface;
        let Some(host) = super::host() else {
            return; // no host GPU here
        };
        let (title, data) = host.shared_buffer(1 << 16).unwrap();
        // SAFETY: our mapped buffer; a live buffer.
        let gpu = unsafe {
            data.cast::<[u32; 6]>().write_volatile([3, 1, 1, 4, 1, 1]);
            title.GetGPUVirtualAddress()
        };
        let lists = [host.new_list().unwrap(), host.new_list().unwrap()];
        issued();
        let recorded: Vec<_> = lists
            .iter()
            .map(|l| {
                let (out, _, opened) = super::windowed(l, gpu, 0, (10, 12, 1), false).unwrap();
                assert!(opened);
                let (again, at, opened) = super::windowed(l, gpu, 12, (10, 12, 1), false).unwrap();
                assert!(!opened && at == 12 && again.as_raw() == out.as_raw(), "the second draw joins");
                l.close();
                (out, issued())
            })
            .collect();
        host.execute(&[&lists[1], &lists[0]]);
        let mut run = issued();
        run.extend(recorded[1].1.clone());
        run.extend(recorded[0].1.clone());
        let key = |r: &super::ID3D12Resource| r.as_raw() as usize;
        let mut states: std::collections::HashMap<usize, u32> =
            [key(&recorded[0].0), key(&recorded[1].0)].into_iter().map(|k| (k, 0)).collect();
        replay(&run, &mut states).unwrap();
        assert!(states.values().all(|&s| s == 0), "all common again");
        for (out, _) in &recorded {
            let mut p = std::ptr::null_mut();
            // SAFETY: a mapped buffer of ours, the lists have run.
            let clamped = unsafe {
                out.Map(0, None, Some(&mut p)).unwrap();
                p.cast::<[u32; 6]>().read_volatile()
            };
            assert_eq!(clamped, [3, 1, 1, 4, 1, 1]);
        }
    }

    /// The budget is the list's: a second execution of the same records
    /// finds it spent, and they all run empty.
    #[test]
    fn a_spent_budget_stays_spent() {
        let Some(twice) = clamped(1, &[u32::MAX], 64, 2) else {
            return; // no host GPU here
        };
        assert!(twice.iter().all(|&count| count == 0));
    }

    #[test]
    fn spread_records_split_large_x_and_keep_small_unspread() {
        let Some(host) = super::host() else {
            return; // no host GPU here
        };
        let records = 7u32;
        let (title, in_ptr) = host.shared_buffer(u64::from(records) * 12).unwrap();
        let list = host.new_list().unwrap();
        // Record 0: 115,200 x 1 x 1 -> [57600, 115200, 57600, 1, 2] (spread)
        // Record 1: 65,537 x 1 x 1  -> [32769, 65537, 32769, 1, 2]  (spread)
        // Record 2: 70,000 x 2 x 1  -> [35000, 70000, 35000, 2, 2]  (spread with y >= 1)
        // Record 3: 100 x 2 x 1     -> [0, 100, 100, 2, 1]          (unspread, small)
        // Record 4: 80,000 x 1 x 2  -> [0, 80000, 65535, 1, 2]      (unspread, z > 1, clamped)
        // Record 5: 65,536 x 0 x 1  -> [0, 65536, 65535, 0, 1]      (unspread, y < 1, clamped)
        // Record 6: 10 x 70,000 x 1 -> [0, 10, 10, 65535, 1]        (unspread, y > 65535 clamped)
        let input = [
            115_200u32, 1, 1,
            65_537, 1, 1,
            70_000, 2, 1,
            100, 2, 1,
            80_000, 1, 2,
            65_536, 0, 1,
            10, 70_000, 1,
        ];
        // SAFETY: mapped buffers kept alive until execute finishes.
        unsafe {
            let in_u32 = in_ptr.cast::<u32>();
            for (i, &w) in input.iter().enumerate() {
                in_u32.add(i).write_volatile(w);
            }
            let out = super::bound(
                &list,
                title.GetGPUVirtualAddress(),
                (2, 12, records),
                true,
            ).unwrap();
            super::done(&list, &out);
            list.close();
            host.execute(&[&list]);
            let mut p = std::ptr::null_mut();
            out.Map(0, None, Some(&mut p)).unwrap();
            let out_u32 = p.cast::<u32>();
            let got: Vec<u32> = (0..35).map(|i| out_u32.add(i).read_volatile()).collect();
            assert_eq!(
                got,
                [
                    57_600, 115_200, 57_600, 1, 2, // record 0 (spread)
                    32_769, 65_537, 32_769, 1, 2,  // record 1 (spread)
                    35_000, 70_000, 35_000, 2, 2,  // record 2 (spread with y >= 1)
                    0, 100, 100, 2, 1,             // record 3 (small unspread)
                    0, 80_000, 65_535, 1, 2,       // record 4 (z > 1 clamped)
                    0, 65_536, 65_535, 0, 1,       // record 5 (y < 1 clamped)
                    0, 10, 10, 65_535, 1,          // record 6 (y > 65535 clamped)
                ]
            );
        }
    }

    #[test]
    fn batch_argument_clamp_builds() {
        if super::host().is_none() {
            return;
        }
        assert!(super::batch_clamp().is_some());
    }

    /// Reads `words` 32-bit words from `buf` at byte offset `at` after execution.
    fn read_words(buf: &super::ID3D12Resource, at: u64, words: usize) -> Vec<u32> {
        let mut p = std::ptr::null_mut();
        // SAFETY: mapped host-visible shared scratch buffer after GPU completion.
        unsafe {
            buf.Map(0, None, Some(&mut p)).unwrap();
            let base = p.cast::<u8>().add(at as usize).cast::<u32>();
            (0..words).map(|i| base.add(i).read_volatile()).collect()
        }
    }

    /// Hoisted batched clamping matches inline per-draw clamping word-for-word
    /// across >64 single-record draws, mixed kinds (`0, 1, 2, 10`), a multi-
    /// record (>64) draw, and mid-wave `BUDGET` exhaustion — while leaving the
    /// title's source buffer completely unmodified.
    #[test]
    fn hoisted_clamps_match_inline_bounds_and_budget_order() {
        let Some(host) = super::host() else {
            return;
        };
        const SRC_BYTES: usize = 64 << 10;
        let (src, src_ptr) = host.shared_buffer(SRC_BYTES as u64).unwrap();
        let mut initial = vec![0xA5u8; SRC_BYTES];
        let put = |buf: &mut [u8], off: usize, v: u32| {
            buf[off..off + 4].copy_from_slice(&v.to_le_bytes());
        };

        struct Spec {
            off: u64,
            kind: u32,
            stride: u32,
            records: u32,
        }
        let mut specs = Vec::new();
        let mut cur = 64usize;
        // 75 consecutive single-record jobs (spanning >1 full 64-lane wave).
        for i in 0..75u32 {
            let (kind, stride) = match i % 4 {
                0 => (0u32, 16u32),
                1 => (1u32, 20u32),
                2 => (2u32, 12u32),
                _ => (10u32, 12u32),
            };
            match kind {
                0 => {
                    put(&mut initial, cur, 5000 + i * 11);
                    put(&mut initial, cur + 4, 3);
                    put(&mut initial, cur + 8, 10 + i);
                    put(&mut initial, cur + 12, 20 + i);
                }
                1 => {
                    put(&mut initial, cur, 7000 + i * 13);
                    put(&mut initial, cur + 4, 2);
                    put(&mut initial, cur + 8, 100 + i);
                    put(&mut initial, cur + 12, (-30i32 + i as i32) as u32);
                    put(&mut initial, cur + 16, 200 + i);
                }
                2 => {
                    put(&mut initial, cur, 10 + (i % 7));
                    put(&mut initial, cur + 4, 2);
                    put(&mut initial, cur + 8, 1);
                }
                _ => {
                    // First mesh tests 65,535 clamp; remaining stay small so BUDGET crosses in the tail.
                    put(&mut initial, cur, if i == 3 { 70_000 } else { 8 + (i % 5) });
                    put(&mut initial, cur + 4, 4); // clamps to 1
                    put(&mut initial, cur + 8, 3); // clamps to 1
                }
            }
            specs.push(Spec {
                off: cur as u64,
                kind,
                stride,
                records: 1,
            });
            cur += stride as usize;
        }
        // Multi-record (>64) indexed draw.
        let multi_off = cur;
        for r in 0..70u32 {
            let b = multi_off + (r * 20) as usize;
            put(&mut initial, b, 3000 + r);
            put(&mut initial, b + 4, 2);
            put(&mut initial, b + 8, r);
            put(&mut initial, b + 12, r * 2);
            put(&mut initial, b + 16, r * 3);
        }
        specs.push(Spec {
            off: multi_off as u64,
            kind: 1,
            stride: 20,
            records: 70,
        });
        cur += 70 * 20;
        // 80 heavy indexed draws that exhaust BUDGET mid-wave (4,000,000 each).
        for i in 0..80u32 {
            put(&mut initial, cur, 1_000_000);
            put(&mut initial, cur + 4, 4);
            put(&mut initial, cur + 8, 700 + i);
            put(&mut initial, cur + 12, 800 + i);
            put(&mut initial, cur + 16, 900 + i);
            specs.push(Spec {
                off: cur as u64,
                kind: 1,
                stride: 20,
                records: 1,
            });
            cur += 20;
        }

        // SAFETY: mapped `SRC_BYTES` shared buffer; live buffer.
        let base_gpu = unsafe {
            std::ptr::copy_nonoverlapping(initial.as_ptr(), src_ptr, SRC_BYTES);
            src.GetGPUVirtualAddress()
        };

        // Hoisted path.
        let list_hoisted = host.new_list().unwrap();
        let mut hoisted_locs = Vec::with_capacity(specs.len());
        for s in &specs {
            let loc = super::hoist(
                &list_hoisted,
                &src,
                base_gpu + s.off,
                (s.kind, s.stride, s.records),
            )
            .unwrap();
            hoisted_locs.push(loc);
        }
        list_hoisted.close();
        host.execute(&[&list_hoisted]);

        // Inline reference path (one submission per readback, sharing one list budget).
        let list_inline = host.new_list().unwrap();
        let (rb_inline, rb_ptr) = host.shared_buffer(SRC_BYTES as u64).unwrap();
        let mut rb_offsets = Vec::with_capacity(specs.len());
        let mut rb_cur = 0u64;
        for s in &specs {
            let out = super::bound(&list_inline, base_gpu + s.off, (s.kind, s.stride, s.records), false).unwrap();
            let bytes = u64::from(s.stride * s.records);
            // SAFETY: open list; `out` is in `INDIRECT_ARGUMENT` after `bound`.
            unsafe {
                list_inline.list.ResourceBarrier(&[super::transition(
                    &out,
                    super::D3D12_RESOURCE_STATE_INDIRECT_ARGUMENT,
                    super::D3D12_RESOURCE_STATE_COPY_SOURCE,
                )]);
                list_inline.list.CopyBufferRegion(&rb_inline, rb_cur, &out, 0, bytes);
                list_inline.list.ResourceBarrier(&[super::transition(
                    &out,
                    super::D3D12_RESOURCE_STATE_COPY_SOURCE,
                    super::D3D12_RESOURCE_STATE_INDIRECT_ARGUMENT,
                )]);
            }
            super::done(&list_inline, &out);
            rb_offsets.push((rb_cur, bytes as usize));
            rb_cur += bytes;
        }
        list_inline.close();
        host.execute(&[&list_inline]);

        // SAFETY: GPU completed; verify title copy untouched and outputs identical.
        let after_src = unsafe { std::slice::from_raw_parts(src_ptr.cast_const(), SRC_BYTES).to_vec() };
        assert_eq!(after_src, initial, "title argument buffer must never be modified");

        let mut tail_kept = 0usize;
        let mut tail_zeroed = 0usize;
        for (idx, ((dst, at), (rb_off, bytes))) in hoisted_locs.iter().zip(rb_offsets.iter()).enumerate() {
            let got = read_words(dst, *at, *bytes / 4);
            // SAFETY: within mapped `rb_inline`.
            let expected = unsafe {
                let p = rb_ptr.add(*rb_off as usize).cast::<u32>();
                (0..*bytes / 4).map(|i| p.add(i).read_volatile()).collect::<Vec<_>>()
            };
            assert_eq!(got, expected, "job {idx} hoisted output must match inline bound()");
            if idx >= 76 {
                if got[0] == 1_000_000 {
                    tail_kept += 1;
                } else {
                    assert_eq!(got[0], 0);
                    tail_zeroed += 1;
                }
            }
        }
        assert!(tail_kept > 0 && tail_zeroed > 0, "tail must cross BUDGET mid-wave");
    }

    /// Multiple source buffers and >256 KiB chunk rollover preserve strict
    /// draw-order budget saturation across dispatch runs and reuse both chunks
    /// and upload buffer across `HostList::reset()`.
    #[test]
    fn hoisted_clamps_across_multiple_buffers_and_chunks_preserve_budget_order() {
        use windows::core::Interface;
        let Some(host) = super::host() else {
            return;
        };
        let list = host.new_list().unwrap();
        let (src_a, ptr_a) = host.shared_buffer(4096).unwrap();
        let (src_b, ptr_b) = host.shared_buffer(4096).unwrap();
        // SAFETY: mapped 4096-byte shared buffers; live buffers.
        let (gpu_a, gpu_b) = unsafe {
            let write = |p: *mut u8, off: usize, count: u32, inst: u32, tag: u32| {
                let w = p.add(off).cast::<u32>();
                w.write(count);
                w.add(1).write(inst);
                w.add(2).write(tag);
                w.add(3).write(0);
                w.add(4).write(0);
            };
            write(ptr_a, 0, 0, 1, 0xAAAA);
            write(ptr_a, 32, 1 << 22, 4, 0x1111); // BUDGET / 8
            write(ptr_b, 32, 1 << 22, 4, 0x2222); // BUDGET / 8
            (src_a.GetGPUVirtualAddress(), src_b.GetGPUVirtualAddress())
        };

        for cycle in 0..2u32 {
            let mut probes = Vec::new();
            for i in 0..4u32 {
                let (s, g) = if i % 2 == 0 { (&src_a, gpu_a) } else { (&src_b, gpu_b) };
                probes.push(super::hoist(&list, s, g + 32, (1, 20, 1)).unwrap());
            }
            // Spill across 256 KiB chunk boundary (8,192 * 32B = 256 KiB).
            let mut spilled = false;
            let first_chunk = probes[0].0.as_raw();
            for _ in 0..8_250 {
                let (c, _) = super::hoist(&list, &src_a, gpu_a, (1, 20, 1)).unwrap();
                if c.as_raw() != first_chunk {
                    spilled = true;
                }
            }
            assert!(spilled, "must span at least two 256 KiB hoisted scratch chunks");
            for i in 0..6u32 {
                let (s, g) = if i % 2 == 0 { (&src_a, gpu_a) } else { (&src_b, gpu_b) };
                probes.push(super::hoist(&list, s, g + 32, (1, 20, 1)).unwrap());
            }
            list.close();
            host.execute(&[&list]);

            for (idx, (dst, at)) in probes.iter().enumerate() {
                let w = read_words(dst, *at, 5);
                if idx < 8 {
                    assert_eq!((w[0], w[1]), (1 << 22, 4), "cycle {cycle} probe {idx} must fit");
                } else {
                    assert_eq!(w[0], 0, "cycle {cycle} probe {idx} must zero past BUDGET");
                }
            }
            list.reset();
        }
    }

    /// Replays all transition barriers emitted by two hoisted lists run in
    /// either order (matching `argument_buffers_move_from_their_current_states`):
    /// the title's buffer is never transitioned and every hoisted scratch chunk
    /// moves `COMMON -> UNORDERED_ACCESS -> INDIRECT_ARGUMENT -> COMMON`.
    #[test]
    fn hoisted_transitions_start_from_current_states_and_never_touch_title_buffer() {
        use crate::host::tests::{issued, replay};
        use windows::core::Interface;
        let Some(host) = super::host() else {
            return;
        };
        let (title, data) = host.shared_buffer(1 << 16).unwrap();
        // SAFETY: mapped buffer; live buffer.
        let gpu = unsafe {
            data.cast::<[u32; 3]>().write_volatile([70_000, 1, 1]);
            title.GetGPUVirtualAddress()
        };
        let key = |r: &super::ID3D12Resource| r.as_raw() as usize;
        for a_first in [true, false] {
            let lists = [host.new_list().unwrap(), host.new_list().unwrap()];
            issued();
            let recorded: Vec<_> = lists
                .iter()
                .map(|l| {
                    let (out, at) = super::hoist(l, &title, gpu, (2, 12, 1)).unwrap();
                    l.close();
                    (out, at, issued())
                })
                .collect();
            let order = if a_first { [0, 1] } else { [1, 0] };
            let mut states: std::collections::HashMap<usize, u32> =
                [key(&title), key(&recorded[0].0), key(&recorded[1].0)]
                    .into_iter()
                    .map(|k| (k, 0))
                    .collect();
            // Run each list one at a time so prelude + list barriers replay in exact GPU order.
            for &i in &order {
                issued();
                host.execute(&[&lists[i]]);
                let mut run = issued();
                run.extend_from_slice(&recorded[i].2);
                replay(&run, &mut states).unwrap();
                assert!(run.iter().all(|t| t.0 != key(&title)), "title copy must never be transitioned");
            }
            assert!(states.values().all(|&s| s == 0), "all buffers must return to COMMON");
            for (out, at, _) in &recorded {
                assert_eq!(read_words(out, *at, 3), vec![65_535, 1, 1]);
            }
            // SAFETY: mapped title buffer untouched.
            assert_eq!(unsafe { data.cast::<[u32; 3]>().read_volatile() }, [70_000, 1, 1]);
        }
    }

    /// Re-submitting the same closed hoisted list twice re-clamps cleanly into
    /// its own scratch chunks without mutating the title's source buffer.
    #[test]
    fn hoisted_list_resubmitted_twice_reclamps_cleanly() {
        let Some(host) = super::host() else {
            return;
        };
        let list = host.new_list().unwrap();
        let (src, ptr) = host.shared_buffer(256).unwrap();
        // SAFETY: mapped buffer; live buffer.
        let gpu = unsafe {
            let p = ptr.cast::<u32>();
            p.write(u32::MAX);
            p.add(1).write(u32::MAX);
            p.add(2).write(42);
            p.add(3).write(7);
            p.add(4).write(99);
            src.GetGPUVirtualAddress()
        };
        let (dst, at) = super::hoist(&list, &src, gpu, (1, 20, 1)).unwrap();
        list.close();
        for _ in 0..2 {
            host.execute(&[&list]);
            assert_eq!(read_words(&dst, at, 5), vec![1 << 24, 4, 42, 7, 99]);
            // SAFETY: title buffer untouched.
            assert_eq!(unsafe { ptr.cast::<u32>().read_volatile() }, u32::MAX);
        }
    }

    /// An earlier list's compute write to a title argument buffer is visible
    /// to the hoisted prelude of the following list.
    #[test]
    fn hoisted_prelude_orders_after_earlier_list_compute_write() {
        let Some(host) = super::host() else {
            return;
        };
        const WRITER: &str = r#"
RWByteAddressBuffer dst : register(u0);
[numthreads(1, 1, 1)]
void main() {
    dst.Store3(0, uint3(90000, 1, 1));
}
"#;
        let writer =
            super::make_with(WRITER, super::s!("writer"), &[super::root_view(super::D3D12_ROOT_PARAMETER_TYPE_UAV, 0, 0)])
                .unwrap();
        let (src, ptr) = host.shared_buffer(256).unwrap();
        // SAFETY: mapped buffer; live buffer.
        let gpu = unsafe {
            std::ptr::write_bytes(ptr, 0, 256);
            src.GetGPUVirtualAddress()
        };
        let write_list = host.new_list().unwrap();
        // SAFETY: open list, live compute pipeline and UAV buffer.
        unsafe {
            write_list.list.SetComputeRootSignature(&writer.root);
            write_list.list.SetPipelineState(&writer.pso);
            write_list.list.SetComputeRootUnorderedAccessView(0, gpu);
            write_list.list.Dispatch(1, 1, 1);
        }
        write_list.close();

        let hoisted_list = host.new_list().unwrap();
        let (dst, at) = super::hoist(&hoisted_list, &src, gpu, (2, 12, 1)).unwrap();
        hoisted_list.close();

        host.execute(&[&write_list, &hoisted_list]);
        assert_eq!(read_words(&dst, at, 3), vec![65_535, 1, 1]);
    }

    /// Release benchmark comparing inline `bound()` + `done()` vs prelude-
    /// hoisted batched clamping across 9,000 indirect indexed draws.
    ///
    /// Run with:
    /// `cargo test --release -p victory-gpu --locked -- --ignored --nocapture benchmark_hoisted_vs_inline_indirect_clamps`
    #[test]
    #[ignore = "benchmark"]
    fn benchmark_hoisted_vs_inline_indirect_clamps() {
        let Some(host) = super::host() else {
            return;
        };
        let list_inline = host.new_list().unwrap();
        let list_hoisted = host.new_list().unwrap();
        const DRAWS: usize = 9_000;
        const STRIDE: u32 = 20;
        let (src, ptr) = host.shared_buffer((DRAWS * STRIDE as usize) as u64).unwrap();
        // SAFETY: mapped buffer; live buffer.
        let base_gpu = unsafe {
            for i in 0..DRAWS {
                let p = ptr.add(i * STRIDE as usize).cast::<u32>();
                p.write(300);
                p.add(1).write(1);
                p.add(2).write(0);
                p.add(3).write(0);
                p.add(4).write(0);
            }
            src.GetGPUVirtualAddress()
        };

        let run_inline = || {
            for i in 0..DRAWS {
                let out = super::bound(&list_inline, base_gpu + (i * STRIDE as usize) as u64, (1, STRIDE, 1), false).unwrap();
                super::done(&list_inline, &out);
            }
            list_inline.close();
            host.execute(&[&list_inline]);
            list_inline.reset();
        };
        let run_hoisted = || {
            for i in 0..DRAWS {
                let _ = super::hoist(&list_hoisted, &src, base_gpu + (i * STRIDE as usize) as u64, (1, STRIDE, 1)).unwrap();
            }
            list_hoisted.close();
            host.execute(&[&list_hoisted]);
            list_hoisted.reset();
        };

        for _ in 0..3 {
            run_inline();
            run_hoisted();
        }
        const ITERS: u32 = 20;
        let t0 = std::time::Instant::now();
        for _ in 0..ITERS {
            run_inline();
        }
        let inline_ms = t0.elapsed().as_secs_f64() * 1e3 / f64::from(ITERS);

        let t1 = std::time::Instant::now();
        for _ in 0..ITERS {
            run_hoisted();
        }
        let hoisted_ms = t1.elapsed().as_secs_f64() * 1e3 / f64::from(ITERS);

        eprintln!(
            "{DRAWS} indirect draw clamps: inline {inline_ms:.2} ms/frame vs hoisted {hoisted_ms:.2} ms/frame ({:.1}x)",
            inline_ms / hoisted_ms.max(1e-6)
        );
    }
}
