# SharedMem: title memory the GPU reads and writes in place (unified memory)

Worktree `scratch/SharedMem/wt`, branch `probe/shared-mem` on fork main `4bf60796e`. Probe:
`crates/victory-gpu/src/shared_mem_probe.rs` (ignored tests `probe_semantics`, `probe_bandwidth`), gate
`/tmp/qsharedmem.sh`, logs `scratch/SharedMem/logs/gpu-*.log`. Builds on ZeroCopy (local://ZeroCopy.md,
branch `perf/zero-copy`, not in fork main), whose results are re-checked or extended here, not repeated.

## 1. Support in our stack (source facts)

vkd3d-proton source `tools/vkd3d-src` is `v3.1-83-gcae8353e`, the exact commit GE-Proton11-14's
`d3d12core.dll` reports (`v3.1-83-gcae8353ed-GE-Proton`). Paths are under `libs/vkd3d/`.

| fact | where |
|---|---|
| `OpenExistingHeapFromAddress1(address, size)` is implemented. It makes a CUSTOM heap (write-back, L0), `ALLOW_ONLY_BUFFERS`, alignment 64 KiB. It adds `SHARED \| SHARED_CROSS_ADAPTER` only when `VK_EXT_external_memory_host` is present; without it, it logs a FIXME and falls back to ordinary memory. | `device.c:7547-7584` |
| `OpenExistingHeapFromAddress(address)` (no size) is a `VirtualQuery` wrapper: the address must be the allocation base, committed, and one protection region to the end of the allocation. Otherwise `E_INVALIDARG`. | `device.c:7586-7627` |
| `OpenExistingHeapFromFileMapping` maps the whole section with `MapViewOfFile`, opens that, then **unmaps it right away**. The heap is left over an unmapped view, so it is unusable for us. | `device.c:7629-7650` |
| `CheckFeatureSupport(EXISTING_HEAPS)` reports `EXT_external_memory_host`. | `device.c:4980-4997` |
| vkd3d checks no host-pointer alignment (`heap.c:273-278` checks only the heap description's `Alignment` field) and never calls `vkGetMemoryHostPointerPropertiesEXT`. The driver's `minImportedHostPointerAlignment` (4 KiB on NVIDIA, ZeroCopy) and the 64 KiB placement alignment are the caller's job. | `memory.c:1226-1251`, `heap.c:273-278` |
| Imported memory is never cleared: zero-init is skipped (`vkd3d_is_imported_allocation`), and so is the command clear. Title bytes survive the import. | `memory.c:1371-1376`, `:1537`, `:2286` |
| Imported memory is never sub-allocated: one `VkDeviceMemory` per heap. | `memory.c:2214` |
| `cpu_address` is the caller's pointer: Map returns title memory. | `memory.c:1624-1630` |
| **Placed resources only.** The heap is `ALLOW_ONLY_BUFFERS`. A resource on a `SHARED_CROSS_ADAPTER` heap must carry `D3D12_RESOURCE_FLAG_ALLOW_CROSS_ADAPTER`, else `E_INVALIDARG`. Buffers get `VkExternalMemoryBufferCreateInfo(HOST_ALLOCATION)`. No textures, no tiled/reserved resources (`UpdateTileMappings` into such a heap is not possible). | `device.c:7564`, `resource.c:4614-4637`, `resource.c:188-196` |
| Wine (`wine-src` @ `51bbcd2`) passes the struct through unchanged (`dlls/winevulkan/vulkan_thunks.c:11314-11324`). `win32u` keeps an app-supplied host pointer as is; it substitutes its own only for host-visible allocations *without* one (the wow64 path). | `dlls/win32u/vulkan.c:924-926`, `:940-943` |

What follows for VIctory:
- Unit: one heap per imported range, any number of placed buffers inside. A buffer never spans two heaps.
- Constraints: 64 KiB-aligned start and size, everything committed and read-write at import, and the range stays mapped while the heap lives.
- A failed pin is silent. Every import needs its own proof that it is in place: a GPU round trip on one word, or a probe of the kernel log.
- Title textures cannot live in such heaps. Only title *buffers*, and raw views into them (the page-table path), can.

## 2. Probe results

### 2a. Semantics (`probe_semantics`, log `logs/gpu-sem1.log`, 14:09, exit 0)

All with 4-byte shader loads and stores, no Map/Unmap, no flush, no copies.

| # | case | result |
|---|---|---|
| 1 | 16 MiB write-watched private memory (as xmem's pools), 5 rounds: CPU writes, then the next dispatch reads; the GPU writes, then the CPU reads after the fence | **0 / 0 words wrong**: coherent both ways |
| 2 | **Guards set after the import** (page 0 `PAGE_NOACCESS`, page 1 `PAGE_READONLY`, as `watch.rs` sets them), then a GPU write | the CPU sees every GPU word through the read-only page (0 wrong). The pin does not care about later CPU protections. |
| 2 | Whole range re-guarded read-only; a vectored handler opens each page on the CPU's write (as the guard-fault handler does) | 256 writes, **256 faults, 2.4 µs each** (handler included, debug build). The next dispatch saw 256/256; the other 4.2 M words were intact. Faults, protections and the pin coexist. |
| 3 | `VirtualProtect` of one 64 KiB page (2,000 toggles) | imported (write-watched) 1.59 µs, plain 1.49 µs: pinning adds nothing |
| 4 | Section: the title's view over a placeholder (`MapViewOfFile3`) has a read-only page; **our own** view of the same section is opened | 0 words wrong both ways. CPU writes through the title's view reach the GPU, and GPU writes are readable through it. |
| 4b | The guarded title view itself opened | `S_OK`, yet **16,384 of 16,384 words wrong**: the silent fallback. The kernel logs `Cannot map memory with base addr … size of 0x1000 pages` |
| 4c | Two overlapping imports of one view (whole + 1 MiB inside) | 0 / 0 wrong: pages may be pinned twice |
| **4d** | **Slices 3 then 1 of a section, two views side by side in a split placeholder, opened as one heap** | **0 / 0 wrong both ways**: a VA range that is *not one run* of its section imports fine, as long as it is all mapped and read-write |
| 5 | Import + place, resident private memory | 64 KiB 0.027 ms, 1 MiB 0.034 ms, 16 MiB 0.24 ms, 256 MiB 2.7 ms, 1 GiB 14 ms (**~11-14 ms/GiB**). Release takes 0.03-8 ms. |
| 5b | 2,048 imports of 64 KiB live at once | all succeeded, 66.5 ms in total (32 µs each) |

Additions to ZeroCopy's §1:
- **(2)** Guards do not interfere once the pin exists. Write tracking by guard pages can stay on a range opened in place, or be dropped for it. Only the *import* needs the range unguarded, and it can be done before any guard (at allocation) or through an unguarded alias.
- **(4d)** "Not one run of a section" is not a blocker. We can map our own alias of the title's layout (one view per run, side by side in a placeholder of ours) and import that.
- **(5)** Import cost is ~12 ms/GiB for resident pages. ZeroCopy's 789 ms for the 6.27 GiB section was for non-resident pages, which the pin has to fault in.

### 2b. Bandwidth (`probe_bandwidth`, log `logs/gpu-bw1.log`, exit 0)

Setup:
- RTX 5090, PCIe Gen5 x8 (ZeroCopy), GE-Proton11-14 vkd3d, DXBC from Wine's compiler.
- 1,024 groups × 256 threads, GPU timestamps, best of 3. The wall clock agrees within 2-10 %.
- In place and system-memory copies: 256 MiB (16 MiB for the slow stores). Video memory: 256 MiB × 8 dispatches and 1 GiB × 4.

| kernel | in place (private / section view) | system-memory copy (CUSTOM L0, today's) | video memory (256 MiB / 1 GiB) | VRAM ÷ in place |
|---|---|---|---|---|
| read 16 B seq (`Load4`) | 26.1 / 25.9 GB/s | 26.1 | 1,532 / 1,585 GB/s | ~60× |
| read 4 B seq | 26.4 / 26.4 GB/s | 26.4 | 1,188 / 1,164 GB/s | ~45× |
| gather 4 B at random | 371 / 371 M loads/s (1.5 GB/s of data) | 372 | 23,100 / 20,500 M/s | ~55-62× |
| write 4 B seq, lane *i* writes dword *i* | 26.0 / 26.0 GB/s | 26.1 | 1,761 / 1,858 GB/s | ~70× |
| **write 8 B seq (`Store2`)** | **0.092 GB/s** (2.9 s for 256 MiB) | 0.094 | 1,385 / 1,540 GB/s | ~15,000× |
| **write 16 B seq (`Store4`)** | **0.094 GB/s** | 0.095 | 1,504 / 1,485 GB/s | ~16,000× |
| **write 4 B, four scalar stores per thread into its own 16 B** | **0.094 GB/s** | 0.092 | 1,392 / 1,604 GB/s | ~15,000× |
| scatter 4 B at random | 489 / 485 M stores/s (2.0 GB/s) | 485 | 5,540 / 4,770 M/s | ~11× |
| atomic add at random | 332 / 324 M/s | 330 | 6,030 / 5,610 M/s | ~18× |
| CPU memcpy 64 MiB into / out of a GPU_UPLOAD (ReBAR) heap | writes 12.7 GB/s, **reads 0.02 GB/s** | | | |
| CPU memcpy, private RAM | writes 16.5, reads 17.9 GB/s | | | |

(The pointer-chase row in the log is valid only for the private buffer: the other buffers held zeros, so their chains stayed in one line. It matches the random gather at 368 M/s.)

What it says:
1. **The GPU sees memory in place exactly as it sees today's system-memory copies.** Every row matches L0 within 2 %, private and section memory alike. Opening in place adds no penalty over today's system-memory copies.
2. **Against video memory it is 45-70× slower per byte streamed and ~60× slower per random 4-byte load.** One sequential pass over 1 MB costs **~38 µs** of GPU time over PCIe, against ~0.6 µs in video memory. A 4-byte gather costs ~2.7 ns per load at full occupancy.
3. **The slow store path is a write pattern, not a store width.** Into system memory:
   - `Store2`, `Store4`, and four scalar dword stores per thread with a 16-byte lane stride all run at **~0.09 GB/s, ~11 ms per MB**: ~280× below the coalesced 4-byte stream.
   - Random 4-byte scatter is 20× faster than those (2 GB/s).
   - [INFERENCE] Consistent with: a store instruction whose lanes cover only part of each 32 B sector (lane stride 8 or 16 B, which is what `Store2`/`Store4` lower to as per-component dword stores in SPIR-V) makes partial writes. Consecutive partial writes to the same line serialise over PCIe (~170 ns each); scattered ones go to different lines and do not.
   - Video memory has no such cliff.
   - Translated stores are `rawBufferStore` with masks up to 0xF (ZeroCopy §9). **Any title shader that writes vectors into memory in place, or into today's L0 copies, pays ~11 ms per MB.**
4. **The inverse (title memory in video memory the CPU maps) is out.** CPU reads of ReBAR run at 20 MB/s, and the title reads its own memory constantly.

## 3. Today's costs mapped onto memory in place

Volumes are per frame from `1002a-next4-q` (steady blocks, 203-207 ms/frame: 86 preparing, 62 on the GPU, 54 copying back). The splits of "on the GPU" and copy-back come from `1003a-chaintimers` (old main + timers, 291 ms/frame), the only run with the chain line.

Unit costs today (1002a):
- refresh move, title → host copy: 14.2 ms for 296 MB = **48 µs/MB** (CPU, preparing);
- staging memcpy for video-memory copies: 14.6 ms for 274 MB = **53 µs/MB** (CPU, preparing);
- upload DMA of the same 274 MB: **~38 µs/MB of GPU** at 26 GB/s, ~10.5 ms [INFERENCE: probe rate, not timed in game];
- marked-page copy-back: 21.4 ms for 1,895 pages = 11.3 µs/page (10.9 comparing, 3.3 protecting, 1.6 writing, 1.9 verifying);
- read-backs into mirrors: 200.8 MB in 7.6 ms, plus 16 MB waited for in 8.7 ms; their DMA is ~165 MB, ~6.3 ms of GPU [INFERENCE];
- write-watch takes 8.6 ms (7.3 GB checked), guard re-arms 8.4 ms, other-mapping queries 3.9 ms, guard-lock waits 2.3 ms (preparing);
- guard faults: 985 per frame, 97.6 ms in the handler on title threads (off the chain). That is ~99 µs per fault, against 2.4 µs for a bare fault + `VirtualProtect` in the probe: the handler's work and lock waits dominate, not the exception.

In place, per MB the GPU reads: **38 µs per sequential pass**, ~670 µs if read as random 4-byte gathers. Per MB the GPU writes: 38 µs if lane-coalesced 4-byte, **~11,000 µs** if vector or strided.

| class of title ranges | volume / frame | chain cost today | in place: what disappears | in place: what the GPU pays | verdict |
|---|---|---|---|---|---|
| **All title memory** (5,367 copies, 8.5 GB; full unification) | GPU reads of copies: several GB. #3312 moving the hot copies into VRAM was worth −216 ms of GPU (CpuAnatomy §4, cited in GpuAnatomy) | ~90 ms of the chain (refresh, staging, watch, copy-back) + ~50 ms of page-table barriers | all of it, plus 97.6 ms of title-thread faults | **+200 ms or more** (45-70× per byte; GatherCopy and TraceRays were 41 and 28 ms with system-memory copies); every vector store at ~11 ms/MB | **No-go on a discrete GPU.** It would make sense only on UMA hosts (Deck, APUs), where in place costs no bandwidth. |
| **CPU-written, GPU-read-only, rewritten about whole each frame** (title constants, dynamic vertex/index data, the title's upload rings) | ≤ 296 MB moved, ≤ 274 MB staged | 101 µs/MB CPU (move + staging) + 38 µs/MB DMA + their share of 23 ms of watch/re-arm/queries | moves, staging, upload DMA, their write-watch takes and re-arms; title guard faults on pages no other copy holds | 38 µs/MB **per read pass**; no stores, so no store cliff | **Gain while the GPU reads each byte ≲ 3-4×** (break-even: 101 + 38 = 38·k). Ceiling: all 296 MB at k = 1 gives ~−30 ms preparing and −10 ms DMA. **Recommended first step (§4).** |
| **GPU-written, CPU-read** (marked pages: pools 1,024 + views 871; declared ranges; VRAM read-backs) | 71 MB changed in 1,895 pages; 200 MB read back | ~21.4 marked + ~16.3 read-backs + ~3.7 declared ≈ 41 ms of copy-back; ~6 ms readback DMA | compares, protections, read-backs, mirrors | 38 µs/MB only for lane-coalesced 4-byte stores; ~11 ms/MB for `Store2`/`Store4`/strided; every later GPU read of the data from system memory | Only per shader with proven scalar, coalesced stores, and only for data no later GPU pass reads much. Needs a per-pipeline store-pattern census first. **Second step.** |
| **Table programs** (BVH builder, ~340-370 dispatches, 4 table submissions) | 341-362 MB of declared view writes **withheld** every frame; misses: 5-11 pages and 5-6 256 MiB regions | ~50 ms of page-table barrier waits; each table submission offers all ~5,300 copies to the refresh | If *every* page the table names were in place: no misses, so no withholding, no bitmap copy-back and no probes. Barrier reasons 1, 2 and 4 (local://PageTables.md) go. | the builder's reads and its 340+ MB of writes over PCIe: ≥ 13 ms if coalesced 4-byte, seconds if vector | **Not first.** In place and withholding are incompatible: a GPU write lands in title memory at once, and withholding exists because the title followed garbage BVH pointers (run drops1). Misses must go first. Serving missed pages in place is cheap (5-11 pages a frame), but it ends withholding, and then the 340 MB of view writes must go back somewhere. PageTables' weaker order (`VICTORY_PT_BARRIER`) attacks the 50 ms without any of this. |
| **The page table itself, bitmaps** | 2 rewrites/frame, 256 KiB bitmap per list | small | — | — | VIctory's own buffers, not title memory: out of scope. |
| **Today's system-memory copies** (heap `0xb10000`, 174 MB, L0: "not one run of a section", ZeroCopy zc1b) | refreshed 19.3×/frame (3.5 GB offered, 1.0 ms moved) | ~1-3 ms | its refresh moves and checks | nothing new: it is read over PCIe already | Now possible: an alias of its runs (2a, 4d). Small gain. Its GPU writes already pay the store cliff today (09034a2i/j: 4.2 MB/frame of marked pages; moving the heap into VRAM changed GPU time by −1.5 ms). |
| **Title textures** (6 GB of host textures) | — | heap texture uploads 3.7 ms | — | — | Impossible: imported heaps take buffers only (§1). |
| **VIctory's upload staging** | 274 MB | 14.6 ms + DMA | goes with each copy taken in place | — | Not a class of its own. |

Per stage, for the first step [INFERENCE; X = MB/frame of class-2 refresh taken in place, k = GPU read passes per byte]:
- preparing: −101·X µs (moves + staging), minus their share of the 23 ms of watch, re-arm and queries. That is −15 ms at X = 150, −30 ms at X = 296.
- on the GPU: −38·X µs of upload DMA, +38·k·X µs of reads over PCIe. That is ±0 at k = 1, +11 ms at X = 150 and k = 3.
- copying back: 0 (no GPU writes in this class).
- title threads: fewer guard faults (unknown share of the 985).
- Example: X = 150, k = 1.5 gives ≈ −24 ms preparing and +3 ms GPU, **net ≈ −20 ms/frame**. At k ≥ 4 it is a loss.

## 4. Recommended first step: GPU-read-only rewrite-every-frame copies in place (`VICTORY_IN_PLACE=1`)

**Class.** A host copy of title memory:
- that no shader has written: no marks, no declared writes, never bound as a UAV;
- that is ≥ 64 KiB and 64 KiB-aligned (or rounded to its pages);
- whose refresh moved ≥ half its size in each of the last 3 frames.

Its bytes already cross the CPU, the staging and PCIe every frame. In place they cross PCIe once per GPU read, and nothing else.

**Mechanism, reusing ZeroCopy's `perf/zero-copy` (94c71ffe7, 93ef7a4b1) on fork main:**
- `zero_copy::open` and the `in_place` shadow, which needs no tracker, refresh or copy-back.
- `find`/`loose` replacing stale copies.
- The startup self-test.
- The per-import one-word GPU round trip; 4b shows the fallback is silent.

New pieces:
1. **Memory source**:
   - Section memory: our own view (`Section::whole`, `xmem.rs:478`).
   - Ranges that are several runs: an alias of their runs side by side, mapped by xmem in a placeholder of ours (2a, 4d).
   - Pools: imported on the title's own VA. This works only if done before any guard (2a, case 2 shows later guards are harmless). That means at `XMemVirtualAlloc`, or at promotion after the copy's guards are lifted. The heap must be dropped before any `MEM_DECOMMIT`/`MEM_RELEASE` of the pool.
2. **Promotion rule**:
   - Per copy: R (bytes refreshed per frame), W (GPU wrote it: a flag) and A (bytes asked by draws per frame).
   - Promote after 3 frames with R ≥ S/2, W = 0, A ≤ 4·S.
   - Demote to a copy when a UAV binds it or a mark lands in it, or when A > 4·S for 3 frames.
   - Swapping follows ZeroCopy's `retire`: lists holding the old copy keep it.
3. **Guards**: a page held only by copies in place needs no guard. Pages another copy also holds keep their guards, which (2a) shows is harmless.
4. **Counters** (diagnostics, in the 5 s block): promoted copies and MB live; R saved (MB, ms of moves and staging); A of the promoted (the GPU's PCIe reads); demotions by reason; imports refused or self-test failed; guard faults on promoted pages.

**One exe, switch off = census.** With the switch off, the same counters report what *would* be promoted, so the off arm sizes X. If the census shows X < ~50 MB/frame, stop there.

**A/B** (Main queues):
- `VICTORY_IN_PLACE=0` vs `=1`, 420 s, common env.
- Read: preparing (moves, staging, watch, protecting), on the GPU, copying back, the new line, guard faults/frame.
- Check the picture against captures.
- Expected [INFERENCE]: −10 to −25 ms preparing, 0 to +8 ms GPU.
- A GPU rise much larger than 38·X µs means the promoted copies are gathered at random. Tighten A ≤ 2·S.

**Not in this step, in order after it:**
- (2) GPU-written CPU-read ranges, per pipeline with a store-pattern census. It needs the translator to report, per shader, whether every store is a lane-coalesced dword.
- (3) Misses served in place, then table-program outputs, only together with an answer to the 340 MB of withheld view writes.
- Full unification: UMA hosts only.

## 5. Rules for any memory opened in place (all observed here or in ZeroCopy)

1. Import only ranges that are wholly mapped, committed and read-write at that moment. Otherwise the import silently gets vkd3d's own memory (4b). Prove each import with a one-word GPU round trip, or watch the kernel log for `Cannot map memory`.
2. After the import, protections and guard faults are free to change (2a case 2), and overlapping imports are fine (4c).
3. Never unmap, decommit or remap a range while its heap lives. Hold sections through our own views.
4. Shaders may write it only with lane-coalesced dword stores (lane *i* → dword *i*). Vector stores (`Store2`/`Store4`) and several dwords per thread at a lane stride above 4 bytes run at ~11 ms/MB.
5. Buffers only. One placed buffer never spans two heaps. Mind a raw view's 4 GiB range when the section (6.1 GiB) is opened whole: use per-window views.
6. Import cost ~12 ms/GiB resident, ~125 ms/GiB if the pin must fault pages in (ZeroCopy f). Do it at load or allocation, never per submission.

## 6. Follow-up (1): the store cliff today

- **No counter on fork main** separates GPU writes into copies in system memory from those into VRAM copies. 1002a/c carry no GPU timing.
- **Bounds from existing runs:**
  - **1000a-main-gputiming:** 59.5 ms/frame of GPU commands. The largest single title command takes 1.3 ms; prelude (9.1 ms) and clamp (8.0 ms) are sums of ~100 and ~1,000 commands. At ~11 ms/MB, no single dispatch writes more than ~0.1 MB in the slow pattern.
  - **09034a2i/j (HeapVram on/off, older build):** moving the only large copy in system memory into VRAM saved ~1.5 ms of GPU. That is the 174 MB heap `0xb10000`, with 4.2 MB/frame of marked whole pages. The frame got ~6 ms *worse* from staging and read-backs.
  - **Open risk on fork main:** in 1002a, title copies total 8.5 GB against a ~7.8 GB VRAM-copy budget, so ≥0.7 GB of copies sit in system memory (non-local use 1,399 MB, against 740 MB in 1000a). Nothing in the logs says whether shaders write them.
- **Counter added** (diagnostics only, `perf/in-place` @ `90d56f785`). One line after `marked pages per frame by memory`:
  `GPU writes into copies in system memory per frame: N copies there (MB); P marked pages (MB of them went back), D declared ranges (MB), T page-table pages (MB went back); the bytes that went back cost X ms written coalesced, Y ms at the store cliff; declared bytes … / … ms; most marked pages: start+len pages, …`
  - Code: `copystat.rs` `sys_written`/`log_sys_writes`; `shadow.rs` `SYSTEM_COPIES`, `sys_write` in `copy_back_page`, `writing_for`, `copy_back_marks`.
- **Estimate [INFERENCE]:** ≤ ~2 ms/frame today.
- **Fix, if the counter shows more than ~0.3 MB/frame going back into copies in system memory:** give GPU-written copies priority for the VRAM budget.
  - A copy in system memory that takes marks or declared writes is replaced by a VRAM copy at its next lookup. The same replace-at-lookup mechanism as below.
  - A read-only copy gives up its budget instead.
  - Gain ≈ (MB on the cliff × 11 ms) − ~0.1 ms/MB of read-back and staging.
  - Not recommended: lifting the 128 MiB cap for the heap alone. HeapVram measured that as a net loss.

## 7. Follow-up (2): `VICTORY_IN_PLACE=1` (branch `perf/in-place`)

Commits on `4bf60796e`:
- `90d56f785`: the store-cliff counter (§6).
- `3adfdf491`: in place. Built on ZeroCopy's three commits (`94c71ffe7`, `0b5616314`, `93ef7a4b1`), with conflicts resolved against fork main.

**Rule** (`zero_copy.rs`, `Traffic`). A copy (title buffer or loose window) qualifies when both hold:
- its refreshes (`Shadow::copy`) moved ≥ half of it in each of 3 frames in a row (frames from `memory::frame()`);
- no shader has written it since it was made: declared (`writing_for`), marked (`copy_back_marks`, held ones too) or through page tables (`copy_back_page`, `table_written`).

**Lookup** (`find`/`loose`, under the buffer or window lock, as ZeroCopy's open):
- `Traffic::looked_up` classifies the copy: one run of a section, several runs, pools, other.
- With `=1` and one run, it promotes the copy. The copy goes stale and is replaced at once.
- `alloc_backing` opens the replacement in place through the runtime's section view, with the buffer placed at its offset in its first 64 KiB page (`InPlace::base`).
- If the open fails (import refused, self-test failed), the range is barred and stays a copy for good.

**In place:**
- No tracker, refresh, staging, upload, snapshot or copy-back.
- `plan` serves lazy pages (`watch::reading`) before the GPU reads.
- A shader write demotes the copy for good. The write itself is handled by ZeroCopy's code: it lands in title memory, other copies take it via `watch::gpu_wrote`, and declared ranges of lists using title memory keep a snapshot and are put back when withheld.
- A remap makes the copy stale.

**Removed from ZeroCopy:**
- modes `=1`/`=2`;
- the refusal while withholding: withholding is permanent in steady state (1002a), and copies shaders only read have nothing to withhold;
- `missed()` (all stale on a miss);
- the probes, which live on `probe/shared-mem`.

**Residual risk:** a table program storing into a promoted copy while results are withheld lands unwithheld. It is counted as `while withheld (not undone)`, and the copy is demoted. In 1003a, table programs wrote no owned page in steady state.

**Lines** (every 5 s, both arms), after the fps line:
- `in place per frame (VICTORY_IN_PLACE=N): X MB refreshed into copies that qualify; … made in place (MB), live (MB); refreshes not needed (MB asked for); GPU writes landed in place …; replaced, demoted`
- `in place since the start: copies that qualified A one run of a section (MB), B several runs (MB), C in pools (MB), D other (MB); made in place, demoted; promoted yet copied: refused by the host, self-test failed`
- Off arm: X is the refresh traffic `=1` would remove, and A-D say how much of it is reachable through one-run sections today. Pools and several runs would need the alias and pool imports (§4), not built.
- On arm: X should fall toward 0, unless heap slots keep holding the old copies, which go on being refreshed.

**Checks:**
- `cargo fmt --all`: clean.
- `clippy --workspace --all-targets -D warnings`, default and `victory-gpu/diagnostics`: clean.
- `cargo xtask policy`: fails on `crates/victory-shader/src/emit/lower.rs` exceeding 512 KiB. That file is untouched here; the failure is on fork main as is.
- victory-hle under Wine (CPU): 301 passed, 2 failed (`xgamesave a_kept_blob_past_the_link_limit_is_copied`, `xmem physical_writes_guard_views_before_writing_and_catch_late_mappings`). Both fail identically on fork main's build. The new `a_run_of_one_section_is_found_in_our_view` passes.
- victory-gpu CPU units: `zero_copy` 2/2.
- **Device tests** (gate `/tmp/qsharedmem.sh`; logs `scratch/SharedMem/logs/gpu-{ipf,basef,ipgpu1,ipgpu2,base1}.log`):
  - Full-suite runs die mid-way on this box for fork main too. `base1` and `ipgpu1` both stop at `host::fills::upload_buffers_are_pooled_across_list_resets` (gamescope segfault in `libnvidia-gpucomp`), and `ipgpu2` (that test skipped) later at a `queue::` present test.
  - So I compared filtered runs (`memory:: shadow:: zero_copy:: watch:: host::tests:: list::tests::`), this branch against fork main + probe:
    - `ipf`: 172 passed, 16 failed. `basef`: 169 passed, the same 16 failed.
    - The 16 failures are identical (descriptor heap caps, list barriers, `memory::tests` ×5, pipeline host, host read marks), so they fail on fork main too.
    - The 3 extra passes are the new `zero_copy` tests, including the device test `a_buffer_rewritten_whole_every_frame_goes_in_place_until_shaders_write_it`. On the 5090 it covers: promotion after 3 frames; in place at an offset of 256 bytes into its page; the GPU reads the title's write with nothing refreshed; a shader write lands in title memory; demotion for good.

**Diag exe:** `/tmp/vexe-in-place/victory.exe`, sha256 `b1524d13dd61f5c20f622e18aeb7a93b4f87fdef3ea5f4b9f8d23c103f11124b` (`3adfdf491`, `--profile diag --features diagnostics`).

**Run spec:**
- Same exe, 420 s, common env.
- Arms: (A) switch unset, which is census plus the store-cliff line; (B) `VICTORY_IN_PLACE=1`.
- Read:
  - the fps line (preparing, on the GPU, copying back);
  - copystat moves, `uploads staged`, watches, protecting, guard faults;
  - both `in place` lines and the store-cliff line;
  - captures for the picture.
