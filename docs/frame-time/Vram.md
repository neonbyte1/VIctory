# Vram: where video memory goes, and sizing it from the adapter's budget (fork issue #15)

Fork main `6f0aeaefb`. Line numbers are at that commit (`crates/victory-gpu/src/` unless named). The 5090 numbers are steady-state lines of `1011a-inplace1` and `1002a-next4-q`. The 4080 numbers are DieQuelle's in neonbyte1/VIctory#2 (fork main `4bf60796`, 2 runs).

## 1. Where video memory goes (5090, DXGI local usage ≈ 19.4 GB of a 31.1 GB budget)

| consumer | 1011a | 1002a | where it is decided | sized by |
|---|---|---|---|---|
| **Title-buffer copies in VRAM** | ~7.66 GB (8,388 MB of copies − 725 MB of copies in system memory, cliff line) | ~7.7 GB (8,522 MB of copies) | `shadow.rs:129` `try_vram_shadow`. A copy goes to VRAM if it is ≤ 128 MiB (`shadow.rs:2950`, `:3024`) and the running total stays ≤ `vram_shadow_budget()` (`host.rs:5937-5949`). | **25 % of the DXGI local budget, read once**: 7.79 GB on the 5090, ~3.7-3.9 GB on the 4080. First come, first served: whatever is made after the budget fills lands in system memory. |
| **Host textures not in our heaps** (render targets, depth, UAV textures, textures > 16 MiB) | 5,980 − 1,321 = **4.66 GB** | 6,010 − 1,339 = **4.67 GB** | `resource.rs:1711-1761` (committed) | Title's resolution and formats. GPU-written textures total 2.95 + 1.35 GB (`host texture MB` line). Evicted only if fed from title memory, LRU past 90 % of budget (`resource.rs:1927-1929`, `:1985`). |
| **Texture heaps** (non-RT textures ≤ ¼ heap, placed) | 63 × 64 MiB = **4.03 GB**: 1.32 GB used, **2.91 GB free** | 64 heaps: 1.34 used, **2.96 free** | `placement.rs:20` (`HEAP` = 64 MiB), `:22` (`SPARE` = 2 empty heaps kept), `:195` (¼-heap rule) | Fixed. The free space is holes in heaps that one long-lived texture keeps alive. Only 2 empty heaps (128 MB) are spare, so ~2.8 GB is fragmentation. |
| **Rest** (vkd3d descriptor heaps and allocator chunks, page tables 16.8 MB per view heap `memory.rs:55`, VRAM write-mark bitmaps, depth twins, swapchain, driver) | ~3.0 GB | ~3.0 GB | — | Not counted by us. 19.4 − 7.66 − 4.66 − 4.03. |
| *Not in DXGI's per-process usage:* other processes (gamescope, Xwayland) | nvidia-smi 21.1 GB vs DXGI 19.4 | | | |

**System memory the GPU reads** (DXGI non-local, 1.28-1.40 GB on the 5090):
- copies past the budget or over the 128 MiB cap: 153 copies, 725 MB, of which the 174 MB heap `0xb10000`;
- upload and read-back staging: `host.rs:2129-2132` 256/512 MiB per direction, rings `host.rs:6018` (64 MiB) and `:6086` (32 MiB), texture uploads ≤ 512 MiB (`retained copy memory`).

**No acceleration structures or pipeline caches in VRAM.** The title's BVH is title memory built by compute (copies). Pipelines are CPU-side and vkd3d's cache files.

## 2. The 4080 (16 GB, DXGI budget 14.9-15.7 GB)

DieQuelle's runs:
- used 13.5-14.0 GB (86-94 % of budget);
- "spilled" 5.5-5.8 GB;
- GPU 151-154 ms, frame 402-407 ms.

**The spill is by design, not WDDM paging.** Usage stays under the budget. The non-local 5.5 GB ≈ the copies the fixed 25 % pushes out (8.4 GB − ~3.8 GB ≈ 4.6 GB) + ~1 GB of staging. The cost is the GPU reading those copies over PCIe (~25 GB/s on Gen4 x16), where the 5090 reads them from VRAM. #3312 measured −216 ms of GPU from moving hot copies into VRAM; the 4080's GPU ms is ~2.5× the 5090's against ~1.6-1.8× from raw shader throughput.

What fills the 4080's 14 GB [INFERENCE: same scene and resolution as ours, the title renders 2560×1440 on both]:
- ~3.8 GB of copies;
- ~4.7 GB of committed textures;
- ~1.3 GB used in texture heaps;
- the heaps' free space plus the rest: 14.0 − 9.8 ≈ 4.2 GB, against 2.9 + 3.0 on the 5090.

**What a 16 GB card should be set to:**
- **Heap free space ≤ ~0.5 GB, not ~3 GB.** If the 4080 holds a similar ~2.9 GB of heap holes, that frees ~2.4 GB: use ~14.0 → ~11.6 GB, 77 % of a 15 GB budget [INFERENCE].
- **Copies grow into what is then free, up to 85 % of the budget**, instead of stopping at 25 %:
  - 0.85 × 15.0 − 11.6 ≈ 1.15 GB more, so copies in VRAM ~3.8 → ~5 GB of 8.4;
  - non-local ~5.5 → ~4.3 GB [INFERENCE].
  - **A 16 GB card cannot hold all 8.4 GB of copies** next to ~9 GB of everything else. ~3.4 GB stays in system memory however it is sized.
- **Keep 5 % below eviction's 90 % mark** so the title's textures for a new area are not evicted to make room for copies. Copies are never given back once made.
- **The second lever:** what stays in system memory should be the copies the GPU reads least. Today it is the copies made last, whatever their use, and that, not the spill as such, decides the frame on a 16 GB card. The new line counts what lists ask of copies in system memory per frame, which sizes this lever. It needs copies to move between memories at a lookup, as the in-place work did. This change does not do it.

**The 2060 (6 GB, budget ~5.3 GB):** committed textures alone (~4.7 GB) nearly fill it. There it is WDDM paging, not our design. Only smaller heaps and fewer textures (resolution) help.

## 3. Knobs today

| knob | value | where |
|---|---|---|
| copies' VRAM budget | 25 % of the local budget, read once | `host.rs:5937-5949` |
| largest copy in VRAM | 128 MiB | `shadow.rs:2950`, `:3024` |
| view-copy chunks | 16 MiB | `shadow.rs:3285`, `:3320` |
| texture heap size, spare heaps | 64 MiB, 2 | `placement.rs:20`, `:22` |
| texture eviction | above 90 % of budget down to 85 %, title-memory textures only, idle 120 s / 300 frames | `resource.rs:1912-1929`, `:1985` |
| staging kept, staged per submission | 512 / 256 MiB (system memory) | `host.rs:2129-2132` |

No knob reads the card's budget per frame, and none counts the rest.

## 4. Change: `VICTORY_VRAM_FIT=1` (branch `perf/vram-budget` on fork main `6f0aeaefb`)

Commits:
- `2c0eb1d37`: the store-cliff counter, cherry-picked; its `system_copies` gauge is used below.
- `78781c0ed`: the change. It adds `vram_fit.rs` and touches `shadow.rs` `try_vram_shadow`, `placement.rs`, `resource.rs` `evict_unused`, and `host.rs` for the line.

1. **Copies sized from the budget.**
   - Once per presented frame (`evict_unused`, which already queries DXGI): room = budget × 0.85 − local usage.
   - A copy refused by the old 25 % cap goes to VRAM if it fits the room, and takes its bytes off. The 128 MiB single cap stays.
   - On the 5090 the room is ~7 GB, so copies under 128 MiB that are now in system memory go to VRAM as they are made.
   - The target of 85 % is 5 % under texture eviction's 90 % mark.
2. **Texture heaps that drain.**
   - 16 MiB heaps with the switch (64 MiB without), on every card, so the 5090 A/B shows the free space too.
   - The fullest heap that fits is taken first, not the oldest, so sparse heaps empty out and are released past the 2 spare.
   - Textures larger than a quarter of a heap (> 4 MiB with the switch) are committed instead. Committed resources hold no holes.
3. **Counter** (diagnostics, every 5 s, after `video memory:`): `video memory fit (VICTORY_VRAM_FIT=…)`
   - used and budget, the copies' target and room;
   - copies in VRAM (MB) and in system memory (MB);
   - what lists ask of each per frame (MB), the traffic the second lever would move;
   - copies past the 25 % that took the room, and those that did not fit it (per frame, since the start);
   - texture heaps (count, size, MB, free MB);
   - committed textures (MB);
   - the rest (MB).

   With the switch off it prints the same, and "took the room" counts what the switch would have put in VRAM.

**Checks:**
- `cargo fmt --all`: clean.
- `clippy --workspace --all-targets -D warnings`, default and `victory-gpu/diagnostics`: clean.
- `vram_fit::tests` and `placement::tests` pass on the CPU under Wine.
- **Device tests** through the gate (`placement:: shadow:: resource:: vram_fit:: memory:: texture::`; logs `scratch/SharedMem/logs/gpu-fit{base,off,on}.log`):
  - fork main (`6f0aeaefb` + counter): 98 passed, 9 failed;
  - this branch with the switch off: 99 passed, 9 failed;
  - this branch with `VICTORY_VRAM_FIT=1`: 99 passed, 9 failed.
  - The 9 failures are identical in all three arms, so they fail on fork main too. The extra pass is the new `vram_fit` test.

**Diag exe:** `/tmp/vexe-vram-fit/victory.exe`, sha256 `b4f580a9e02d73f867b374fbe7db3c696e5ce18dd1ae8b4491001ddf1c128cfd` (`78781c0ed`). It is built in its own target directory: worktrees sharing one target dir can hand out a stale binary.

## 5. Run recipe for DieQuelle (4080)

Build `perf/vram-budget` @ `78781c0ed` with `--profile diag --features diagnostics`. Same build for both arms, warm cache, 420 s with `-StopAfterWorld 350`, the input build's keys file. Arms interleaved, 2 runs each:
- **A:** switch unset (control, counter only).
- **B:** `VICTORY_VRAM_FIT=1`.

Report, median of the world window (+200 to +350 s), as in #2:
- VRAM peak used / budget / non-local ("spilled");
- the fps line: frame, preparing, GPU, copying back;
- the `video memory fit` line;
- `host textures: …; heaps of textures N (used, free)`;
- device-removed / driver resets;
- a screenshot pair at the same frame for the picture.

Optional:
- one run per arm with `VICTORY_GPU_TIMING=1`, for GPU ms per command;
- a 2060 run of B, to see whether the heaps alone keep it under budget.
