# Frame budget: fork main (0991b/d) vs main-next (0991a/c)

Sources: the four 420 s runs, median of the last 30 five-second blocks per run. ANSI-stripped logs and the parser are in `~/projects/gta6-recomp/scratch/Budget/`. `parse.py <tag>…` prints the medians of every field below, and `rows.json` holds the per-block rows. Code references are to `wt-main320/crates/victory-gpu/src`. HEAD 258f412b8 is one GPU commit past 2a2b052df, and every counter used here is the same at both.
Switches seen on in all four logs: `VICTORY_GPU_OVERLAP`, `VICTORY_EARLY_RELEASE`, `VICTORY_EARLY_BARRIERS`, clamp windows. `VICTORY_OWNER` and `VICTORY_GPU_GAPS` are on by default.

## 1. What the frame line measures

| field | measured where | what it is |
|---|---|---|
| `ms/title frame` | host.rs:5726-5733 (`Meter::report`), frames counted at host.rs:5711 | Wall time / title frames over the last ~1 s. The three fields below are averaged over 5 s. |
| `preparing` | `SUBMISSION_US[0]`, host.rs:2685: `run()` start (2355) → `prepared` (2563) **minus** settle time (2414-2418) | CPU work on the submitting thread before `ExecuteCommandLists`: heap sync, refresh batches, title copies, uploads, VRAM staging, preludes. |
| `copies X ms Y MB` | host.rs:4247-4248 (copy/fill steps), 4418 (refresh bytes) | Part of preparing. The MB figure also counts refresh copies; the ms figure counts only the steps (flushes they trigger included). |
| `on the GPU` | `SUBMISSION_US[1]`: host.rs:2686 (`prepared` → end of `run`, i.e. ECL + Signal), 2224 (`complete_while` waits before preparing, `Host::submit` 2098), 2717 (`finish`'s `wait_for`, including completions done inside settles) | Time a submitting thread is **blocked on the GPU**, plus the ECL calls. This is not GPU busy time. The completion thread's own fence wait (2274-2279) and waits to take the submission lock are not counted. |
| `copying back` | `COPY_BACK[0]`, host.rs:2748-2851 in `Host::finish` | Taking VRAM read-backs into the mirrors, marked-page copy-back, declared-range compare and copy-back, `make_read`, and the lists' after-work. Always runs **under the submission lock** (completion thread, `complete_while`, or `settle`). |

**Serial structure.** The submission lock (host.rs:2020) serialises preparing, completing (copy-back) and submitting. `run()` holds it from preparing through `ExecuteCommandLists`; `finish()` runs under it. A title `ExecuteCommandLists` call goes to `queue.rs:116 execute`, then `pump` (`HELD` lock), then `Host::submit`, all synchronously on the calling thread.
- **main:** held work is released "by submission" (8.4 per frame; held at the present 0-1; longest hold 80-84 ms). The title's own thread runs the chain.
- **main-next:** held work is released "by completed" (12.2-12.5 per frame, pumped from the `victory submissions` completion thread, host.rs:2294/1835). Work is held at the present (5 items) for 211-215 ms, i.e. one whole frame, and PresentX is queued behind it (`presents_held=1`). The title runs a frame ahead and is not the limiter.
- In both builds `frame wait`s end at once (no vsync wait) and `fence_waits=0`.

The GPU overlaps CPU work only while something is in flight. Only 13.2-14.0 of 28-30 submissions per frame are prepared with an earlier one still in flight (`prepared with an earlier one in flight`). Each frame the pipe drains at 5 barrier waits, 1.2 upload-list waits, 0.5-1 eviction waits and 8-10 settles.

## 2. Budget per build (ms/frame, medians)

| | main q1 (b) | main q2 (d) | next q1 (a) | next q2 (c) |
|---|---|---|---|---|
| **frame** | **244.3** | **237.4** | **207.4** | **214.1** |
| preparing (CPU, serial) | 114.7 | 108.5 | 83.3 | 87.8 |
| on the GPU (blocked on GPU + ECL) | 76.7 | 77.1 | 65.3 | 67.1 |
| copying back (CPU, serial, under lock) | 48.4 | 48.9 | 53.2 | 52.9 |
| **unaccounted rest** (median of per-block frame − sum) | 1.4 | 3.0 | 5.6 | 5.1 |
| of the rest: held-work pump outside submissions (`pump_ms`) | 0.1 | 0.1 | 5.2 | 5.2 |

- **The rest** is frame minus the three sums. On main it is title render-thread CPU outside `ExecuteCommandLists`, present, and waits to take the submission lock. On next it is almost entirely the held-work pump, which runs on the completion thread and so on the chain.
- **Title CPU is not on the critical path in either build**, so overlapping title CPU with the GPU is worth ~0 today. Title-thread costs run on threads with slack: guard fault handler 86-99 ms/frame summed over threads, and 68-87 ms/frame of title threads waiting to write views (`descriptor.rs:156`). They become critical only once the chain falls below the title's CPU frame.
- **The frame ≈ preparing + GPU-blocked + copying back, a single serial chain.** Steady-state blocks agree within ±5 ms.

### Inside preparing

Some timers nest, so the rows are not strictly additive. Settles are taken out of preparing; their GPU wait and completions land in "on the GPU" and "copying back".

| consumer | counter (where) | main b / d | next a / c |
|---|---|---|---|
| refresh batches, gross | `FlushUs` host.rs:4363 | 70.7 / 64.8 | 39.7 / 42.2 |
| — of which settles (completing in-flight work mid-prepare) | `SettleUs` host.rs:2336-2343 | 25.7 / 24.8 (10.1 settles: 8.1 other copies, 1.5 same, 0.5 copy) | 12.6 / 12.7 (7.8 other copies) |
| — refresh batches net of settles | | ~45 / ~40 | ~27 / ~29.5 |
| — refresh copies title → host copy (`copy_mapped`, "moves") | host.rs:4469-4482 | 26.2 / 24.7 (495 / 468 MB) | 13.5 / 14.0 (279 / 289 MB) |
| — write-watch + take of written pages | shadow.rs:1136/1183 | 10.4 / 9.2 (4,300 / 3,550 asks; 7.0 / 6.2 GB checked) | 8.3 / 9.2 |
| — protecting (guard re-arm) | watch.rs:1548 | 8.5 / 7.5 | 8.1 / 8.6 |
| — other-mapping (alias) queries | watch.rs:864 | 5.2 / 4.9 | 3.6 / 3.7 |
| heap slots the title copied (`sync_heaps`) | host.rs:2373, descriptor.rs:231 | 19.4 / 19.0 (126 pages, 258k slots scanned, 6.7k changed) | 17.7 / 18.6 (138 pages, 282k slots, 6.4-6.5k changed) |
| VRAM-copy upload staging memcpy (`vram_before`) | host.rs:4963 | 22.9 / 22.1 (453 / 449 MB) | 13.7 / 14.4 (259 / 268 MB) |
| heap texture uploads | host.rs:2423 | 3.4 / 3.5 | 3.7 / 3.7 |
| snapshots of written ranges | shadow.rs:601 | 2.9 / 3.1 | 3.0 / 3.2 |
| state preludes | host.rs:2469 | 0.9 | 0.9-1.0 |
| unattributed in preparing (eviction, `bindless::refresh`/`memory::prepare` enumeration, `vram_after`, list walk) | | ~20 | ~17 |

The main → next drop in preparing (−21 to −31 ms) lines up with these counters:
- clean copies refreshed for writes in flight across them: 2,228 / 2,390 → 24 / 22 per frame;
- settle time: 25.7 → 12.6 ms;
- VRAM uploads: 453 → 259 MB (22.9 → 13.7 ms);
- "moves": 26 → 13.5 ms.

The likely cause is b0e51ff36 + a71de1313 (buffer view slots follow the innermost title buffer's copy, so in-flight writes go through the same copy and no longer "cross") **[INFERENCE: not A/B-tested]**.

### Inside copying back

| consumer | counter (where) | main b / d | next a / c |
|---|---|---|---|
| marked pages copy-back (`copy_back_marks`) | shadow.rs:1705 | 21.5 / 21.1 (1,925 pages, 74 MB written) | 21.0 / 20.8 (1,890 pages, 69 MB) |
| — reading + comparing / changing protections / writing / verify | shadow.rs:1794 | 10.6 / 3.5 / 1.6 / 1.9 | 10.4 / 3.4 / 1.6 / 2.0 |
| — guard-lock rounds / protection changes | | 506 / 388 | 507 / 386 |
| VRAM read-back put into the mirrors (staged) | host.rs:5185 | 6.1 / 7.1 (177-200 MB) | 7.4 / 7.2 (198 MB) |
| VRAM read-back **waited for** (unpredicted pages + past `STAGED_MAX`) | host.rs:5175 | 5.5 / 5.3 (16.7 MB) | 9.5 / 9.3 (15.5 MB) |
| declared-range compare | shadow.rs:662 | 1.2 | 1.1 |
| unattributed (`take_reads`/`make_read` host.rs:2829-2834, after-work 2846-2850, `verify_lazy`) | | ~14 | ~14 |

Lazy view pages are offered **0 per frame** in all four runs.
- Why: every frame a title-memory list misses memory (`missed 3-11 pages and 3-10 256 MiB regions with no copy` on every frame line), so `memory::withholding()` (memory.rs:390) is permanently true. `shadow.rs:2202` (main-next: shadow.rs:2315) then skips `lazify`.
- History: older world runs (0777d86/88) made 418-663 pages per frame lazy.

### Inside "on the GPU"

The logs do not split it by reason. What they do give, per frame:
- waits before preparing: 5.0 behind a barrier (4.0 page-table barriers, 1.0 heap-view writes), 1.2-1.3 for the upload list, 0.5-1.0 for eviction;
- settles: 10.1 on main, 7.8 on next;
- 72-75 ECL calls covering 352-353 host lists.

True GPU busy time is not in these runs. The proxy is cw2-t (Nov 27, an older build with clamp windows on, run with `VICTORY_GPU_TIMING`): 65 ms/frame of command time, of which the top rows are:
- clamp: 9.9 ms in 1,136 commands;
- prelude: 5.9 ms;
- indirect compute: 4.4 ms;
- indirect mesh: 2.8 ms.

rmc-t had 4,901 clamp commands; there clamp was 35.8 ms and indirect mesh 44.5 ms.

The GPU work is about the size of the blocked time (65-77 ms), and half the submissions are prepared with the GPU idle. So most GPU work appears to be **exposed, not overlapped** **[INFERENCE]**.

## 3. Protection changes and faults (per frame)

| caller | main b | main d | next a | next c |
|---|---|---|---|---|
| other (refresh re-arm etc.) | 2,229 calls / 3,381 pages | 1,539 / 2,160 | 2,407 / 3,710 | 3,602 / 6,060 |
| fault handler | 1,440 | 1,228 | 1,364 | 1,384 |
| take (textures, heaps) | 162 | 169 | 188 | 192 |
| marked page by page | 284 | 281 | 284 | 285 |
| run open + run guard | 2 × 52 | 2 × 59 | 2 × 51 | 2 × 52 |
| kernel writes | 32 | 30 | 29 | 29 |
| **total calls** | **~4,260** | **~3,360** | **~4,370** | **~5,600** |

- These runs show ~3.4-5.6k calls per frame, not 15k.
- On the chain they cost about 12 ms: 8.1-8.6 ms protecting in preparing plus 3.0-3.5 ms in copy-back, i.e. 3-9 µs per call under Wine.
- Guard faults run 825-976 per frame (handler protects 1.2-1.4k), costing 86-99 ms/frame of handler time on **title** threads. Guard-lock hold inside the handler is only 11-13 ms (watch.rs:2611 at 2a2b052df / 2737 at 30cb8e3b), so the rest is lock waits plus Wine exception dispatch. This is off the chain today.

## 4. Ranked levers

### 1. Pipeline drains: GPU work exposed — realistic −25 to −40 ms, ceiling ≈ −65 to −77 ms

**Evidence**
- "on the GPU" is 65-77 ms of blocked time in a fully serial chain.
- Only 13-14 of 28-30 submissions are prepared with something in flight.
- Each frame has 5 barrier waits, 1.2 upload-list waits, 0.5-1 eviction waits and 8-10 settles.

**A/B experiments**
- (a) `VICTORY_GPU_OVERLAP=0` vs on. The frame increase is the GPU time hidden today; a small increase confirms the headroom.
- (b) Small diagnostic patch: split `SUBMISSION_US[1]` by cause.
  - In the `complete_while` closure (host.rs:2098-2107), add the wait time per `why` bit.
  - Time settles separately (host.rs:2336).
  - Time ECL separately (host.rs:2614-2625).
- (c) The real change: the 4 page-table barriers per frame (`BARRIER_REASONS` bit 5, host.rs:1488) become bounded writes registered in flight. The lists' page bitmaps already name the pages, via `memory::bind` probes and `vram_probes`. Later submissions would then settle only where they meet them, as declared writes do.
- (d) A 2-3 deep upload-list ring, removing the `uploading && r.uploads` wait at host.rs:2101 (1.2 drains/frame).

**Expected effect**
- Each removed drain hides that submission's GPU tail and its copy-back behind the next preparing.
- (c) alone should be −10 to −25 ms **[INFERENCE]**; run (b) first to size it.

### 2. Heap descriptor sync on the submit path — −10 to −15 ms

**Evidence**
- 17.7-19.4 ms/frame of preparing.
- 258-282k slots are scanned per frame for 6.4-6.7k that changed (2.5 %), about 70 ns per slot.

**Experiment**
- Code change in `descriptor/host.rs:864 sync_slots`: keep the last-synced bytes per 64 KiB heap page.
- memcmp in 64-byte chunks and pass only the differing chunks to `stale_slots`.

**Expected effect**
- About 126-138 pages × a few µs, plus making the 6.5k changed slots: 3-6 ms instead of 19.
- It also shortens the `SHADER_HEAPS` holds that title threads wait on (68-87 ms/frame summed).

### 3. Copy-back — −10 to −15 ms

**Evidence**
- 48-53 ms/frame, all serial under the submission lock.
- Lazy pages are disabled by permanent withholding (0 offered vs 418-663 per frame historically).
- 5.5-9.5 ms of waited VRAM read-back.
- About 14 ms of copy-back is unattributed.

**Experiments**
- (a) Find and give copies to the regions missed every frame, so `withholding()` turns false and lazy pages come back:
  - main: 0x1320000000, 0x1380000000, 0x1390000000, 0x13a0000000…
  - next: 0x7840000000, 0x78a0000000, 0x78c0000000, 0x9d9750000.
  - Timing-only A/B: drop `&& !crate::memory::withholding()` at shadow.rs:2202 (main-next: 2315). It is not correct to ship.
- (b) Predict read-backs from the union of the last 2 completions per kind (`PREDICTED`, host.rs:5196-5204).
- (c) Add `Timed` around `take_reads`/`make_read` (host.rs:2829-2834) and around the after-work loop (2846-2850) to attribute the missing ~14 ms.

**Expected effect**
- (a) −5 to −8 ms: about a third of the marked pages go lazy, saving their compare and protection work.
- (b) −3 to −5 ms.

### 4. Upload volume

**Main only: clean copies crossing in-flight writes — −25 to −30 ms (observed main → next)**

Evidence: clean_crossed 2.2k → 24 per frame; settles 25.7 → 12.6 ms; VRAM staging 453 → 259 MB (22.9 → 13.7 ms); moves 26 → 13.5 ms; preparing −21 to −31 ms.

- **Experiment:** main + cherry-pick b0e51ff36 + a71de1313 (the suspected cause).
- **Expected:** `clean copies refreshed … across them` ≈ 0 and preparing ≈ 85 ms.
- If main-next is the base, this gain is already in it.

**Both builds: the double copy title → host copy → upload staging — −5 to −10 ms**

- main-next still stages 259 MB/frame single-threaded, at about 19 GB/s.
- **Experiment:** split the staging memcpy at host.rs:4966-4980 across 3-4 threads, or use non-temporal stores.
- A later option: the refresh writes the staging directly when a copy is in VRAM (shadow.rs:1043-1044 `copy` + `queue_vram_upload`).

### 5. GPU pass class: argument clamps — −5 to −8 ms of GPU on the 5090 (exposed GPU time, so lever 1 multiplies it)

**Evidence**
- 1,019-1,036 clamp passes per frame. 942 of them run alone only because their groups are spread (`indirect.rs:909 Alone::Spread`; "alone … 942 with spread groups").
- 3,058-3,107 barrier calls per frame come from clamps.
- In the GPU-timing proxy, clamp is the top row: 9.9 ms with windows, 35.8 ms without.

**Experiments**
- Cheap A/Bs first: `VICTORY_CLAMP_WINDOW=0` vs on, which gives the slope per clamp; `VICTORY_GPU_TIMING=1 VICTORY_TIMED_CLAMP=1` on main for a per-class GPU table.
- The real change: let the window clamp write the spread layout. The single clamp shader already handles it via `Record.w spread`, indirect.rs:46-102. Then spread dispatches can join windows.

**Expected effect:** about 900 fewer compute passes and ~2.7k fewer barriers per frame.

### Smaller or later

- **Held-work pump on next's chain:** 5.2 ms/frame (`queue.rs:263-283`, 149 early resets per frame). −3 to −4 ms.
- **Protection-change volume on the chain:** about 12 ms. Batching single-page re-arms (1.5 pages per call in "other") is worth −4 to −6 ms.
- **Title-thread costs:** guard faults (86-99 ms) and view-write waits (68-87 ms) per frame, summed over threads. 0 ms today, but they become the limit once the chain drops below the title's CPU frame.

## 5. Weaker GPUs and VRAM (RTX 4080 16 GB, RTX 2060 6 GB)

Today's chain is a sum, so slower GPU work adds 1:1 to the frame.

- **Lever 1 (overlap)** gains the most on weaker cards. A 4080 at ~1.6× and a 2060 at ~4-5× the 5090's GPU time would turn ~65-77 ms of exposed GPU into ~110 ms and ~300+ ms **[INFERENCE]**. Overlap turns the sum into a max.
- **Lever 5 (clamps)** is GPU work and fixed per-pass bubbles, so it gains more on weaker cards.
- **Lever 4 (upload volume)** cuts PCIe copy traffic and staging, which helps every card. On smaller cards the copies in VRAM are capped at 25 % of the budget (host.rs:4701-4713): about 3.6 GB on 16 GB and 1.4 GB on 6 GB, so more copies stay in system memory and are read over PCIe by shaders.
- **Levers 2 and 3** are CPU-only. They give the same absolute ms on every card and a smaller relative gain unless lever 1 lands.

**VRAM footprint (not frame time on the 5090, decisive below 20 GB)**
- The runs use 19.7-20.2 GB of a 30.9-31.5 GB budget, which already exceeds a 16 GB card.
- Placed-texture heaps hold 1.3 GB used and **3.0-3.6 GB free**: slack that a 4080 or 2060 cannot afford.
- Host textures total 6.0 GB, of which 4.4 GB are used by lists, and 4.4-4.9 GB of placed textures were made.
- Compacting the texture heaps and sizing them to need is the first VRAM lever for the 16 GB and 6 GB cards.
