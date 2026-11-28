# Page-table barriers (lever 1c): what they are, and a weaker exact order

Worktree `scratch/PageTables/wt`, branch `perf/pt-barrier`, based on ChainTimers'
rebase `116f56e81` (= fork main `4bf60796e` + `diag/chain-timers`). Line numbers
are at that commit (`crates/victory-gpu/src/` unless named).

## 1. Mechanism (progress note: step 1 done)

**Not a GPU mapping change.** Nothing here touches D3D12/Vulkan page tables,
tiled resources (`UpdateTileMappings` appears nowhere in the tree), xmem remaps or
placeholders. "Page tables" is VIctory's *software* page table for **title memory at
raw GPU addresses** (decision 0005, `memory.rs:1-33`): translated shaders that load
or store at addresses they compute (BVH node fetches, pointer-chasing compute) look
each 64 KiB page up in a per-view-heap table (root SRV `t0` of `MEMORY_SPACE`) that
names the heap slot pair (SRV/UAV) of the host copy holding the page. Each list
also gets a 256 KiB bitmap (root UAV `u0`): **stores and atomics mark the page they
write, loads mark only pages/256 MiB regions no copy holds ("misses")**
(`victory-shader/src/emit/lower.rs:6517`).

**Title action that triggers it:** any draw/dispatch whose pipeline reads title
memory through the table: `draw.rs:1057` → `State::memory_map` (`draw.rs:576`) →
`memory::bind` (`memory.rs:772`). In 1003a that is ~350-370 dispatches/frame of
~25 compute pipelines (`3b90165a77d95cdf` 82-88, `5062bc7a…`/`5279ae36…`/`c42a3195…`
63 each, …) in **4.0 submissions/frame** ("heavy submissions … 4.0 title memory
through page tables").

**What `memory::bind` registers on the list** (`memory.rs:772-967`):
- `defer_memory(key)` → `Step::Memory(heap)` (`host.rs:2096`, `defer_memory`
  `host.rs:5092`): at submission `memory::prepare` (`memory.rs:716`) rewrites the
  table **in place** if copies came or went (`Shared::rewrite`, `memory.rs:643`,
  documented "Only while no submission runs (the GPU reads the table and the
  views)", `memory.rs:641`; it also nulls and frees the view pairs of gone copies,
  `memory.rs:651`), then offers every copy (~5,300) to the refresh
  (`shadow::refresh_where_needed`, `host.rs:5578`).
- `defer_vram_probe` (`memory.rs:802`): read back, for copies in VRAM, every page
  marked at an *owned* entry.
- `defer_missed` (`memory.rs:850`): did the list touch memory no copy holds.
- `defer_after` (`memory.rs:857`): copy each marked owned page back into title
  memory (`Shadow::copy_back_page`, `shadow.rs:467`), or, if the list missed or
  **any** list missed this frame or the last (`memory::withholding`, `memory.rs:395`),
  withhold it (`drop_withheld`, `memory.rs:414`: retake immediately if *this*
  submission missed, else at the frame's end).

These set heavy bits 5 (`Step::Memory`, `REASON_MEMORY`), 3 (missed checks) and 1
(unnamed work after) in `reasons_of` (`host.rs:2444`); all three are in
`BARRIER_REASONS` (`host.rs:2194`), so `barrier()` (`host.rs:2208`) is true and the
`InFlight` record carries `barrier: true`.

**What the host does:** the barrier submission P itself is prepared with earlier work
left in flight (`VICTORY_EARLY_BARRIERS`: "4.9 prepared with earlier ones left in
flight"). The **next** submission N, in `Host::submit`, calls `complete_while`
(`host.rs:3108`) with `r.barrier` matching (`host.rs:2977`): it blocks on the fence
for P (and so everything before P), runs P's whole completion (`Host::finish`,
`host.rs:3708`: read-backs, marked pages, missed checks `host.rs:3792`, withholding
`host.rs:3796-3806`, the after-work), and only then starts preparing. Nothing about
remapping is done in that wait; it is a full pipe drain plus P's copy-back.

**Why it drains the whole pipe (what the barrier protects today):**
1. *Unbounded writes.* A table program may store to any page of any of the ~5,300
   copies the table names; which ones is known only from the bitmap after P ran.
   N's refreshes (title → copy), title reads at submission and alias copies could
   otherwise meet them unseen; for VRAM copies a refresh of a page P also wrote,
   followed by P's read-back into the mirror, would copy stale bytes back over a newer
   title write.
2. *Miss → withhold → retake.* If P missed, its declared view writes of table programs
   (and its table writes) are withheld and **retaken** by every copy at its next
   refresh (`host.rs:3806`, `shadow.rs:1981-1988`, `memory.rs:414-430`). N prepared
   before P completes could pass those copies by as clean and run on P's results
   computed from zeros (runs shards20/drops1).
3. *The table is rewritten in place* (and view pairs freed/reused) by the next user's
   `prepare` (`memory.rs:643-700`) while P's shaders may still read it (UMA reads the
   upload table directly; on dGPU P's prelude copy `sync_vram_pre`, `memory.rs:235`,
   may not have run yet).
4. *P's completion reads the live layout* (`layout(key)`, `memory.rs:704`), which a
   later rewrite would have replaced.

**Evidence (run 1003a, old main + timers; `log.txt:25715/25812/25915`):**
- `chain per frame: 69.9 ms on the GPU …: 50.5 waiting for a barrier's page tables,
  6.1 another barrier, 2.6 the upload list, 7.8 settles, 1.7 other completions; …
  31 submissions: 13 prepared with work in flight, 18 with the pipe empty` (50.1/50.7
  in the next blocks).
- `barriers per frame by kind: 4.0 title memory through page tables, 0.0 missed-memory
  checks, 1.0 writes through the heap's views` (each counted once by its first kind,
  `host.rs:3437`), `light submissions …: waited … 5.0 behind a barrier` → **~12.5 ms
  per page-table wait**, 4 a frame.
- `heavy submissions per frame by why: … 5.0 work after, … 4.0 missed-memory checks,
  … 4.0 title memory through page tables` (the 4 table submissions carry bits 1, 3, 5).
- `title memory this frame: … 2 page tables, 0 pages copied back, 0 withheld and
  341.5 MB of views kept (their list read memory with no copy), missed 11 pages and
  5 256 MiB regions` on every frame line: the table is rewritten ~2×/frame, **table
  programs wrote no owned page in steady state** (all bitmap bits are read misses),
  and withholding is permanently on, so the table programs' declared view writes are
  always withheld.

## 2. Design: the weakest exact order (branch `perf/pt-barrier`, `ee65a10d9` + `2b613cbf8`)

Switch `VICTORY_PT_BARRIER=1` (default 0 = today's order, bit for bit; one exe for both arms).

A submission P whose lists use page tables **and in which no program stores through a table**
is no longer a barrier (`light_reasons`/`table_bits`, host.rs). Why storing matters: the new
translator flag `memory_stores` (`Lowered`/`HostShader`, set in `Lower::memory` for every store or
atomic, carried by the shader cache as bit 6; the cache key changes with the translator source,
so the first run retranslates). Without stores the bitmap holds only misses, the probes read back
nothing, and the table copy-back (now `defer_after_table`) writes nothing. A list with stores, or
with any other barrier reason (heap-view writes, other unnamed work after, marks not captured),
keeps the barrier.

The hazards the barrier covered, and what replaces each:
1. **Unbounded writes**: none without stores. P's declared ranges stay registered in flight as
   for any light submission (`Settled::Same/Alias/Copy`).
2. **Miss → withhold → retake**: a miss of P would retake P's table-program ranges into every
   copy. Those ranges (`kept_of`: memory-flagged declared ranges, plus same-copy ranges that meet
   them) are flown as `shadow::KEPT`, which every copy (its own too) and every title read crosses.
   So a later preparing that would pass such a copy by as clean settles P first, as in the barrier
   order.
3. **In-place table rewrite / pair free**: `memory::prepare` settles the table's last user before
   N (`Users`, `Settled::Table`) before `Shared::rewrite`. This happens only when copies came or
   went (~2 a frame), and is skipped and uncounted if that user already completed.
4. **Live layout at completion**: this follows from 3, because no rewrite runs while a user is in
   flight.

Orderings kept: GPU work on one queue stays in order; completions stay in order (so withholding
and missed-state stay ordered for later completions); every CPU read or rewrite of what P may
change waits for P where it happens.

Known differences, all of the kind light submissions already have:
- a system-memory copy refreshed while P reads it (a title write that races on the console too);
- the frame index a late miss is recorded in can move by one frame (it only withholds longer).

Expected limit: N settles P at its first flush that meets a KEPT range. That is most N's, because
table lists declare ~340 MB/frame of view writes. The overlap is then N's preparing up to its
flush.

Counters (diagnostics): copystat line `page tables per frame (VICTORY_PT_BARRIER)`:
- barriers off / left in flight / kept for stores / kept for another reason;
- KEPT ranges;
- how the light ones completed (settled, `Table` settles, waited, completion thread);
- their fence ms.

## 3. Tests (all pass on the RTX 5090, see §5)

- Pure (host.rs `vram_async::light`): `page_tables_alone_make_no_barrier_with_the_switch`,
  `kept_ranges_are_what_a_miss_withholds`.
- shadow.rs `kept_ranges_cross_every_copy`.
- memory.rs `a_table_knows_its_last_user_before_the_one_preparing`.
- victory-shader `only_stores_and_atomics_through_the_table_count_as_stores` (`program()` split into
  `lowered()`).
- Device:
  - `page_tables_stay_in_flight_with_the_switch` (both arms);
  - `a_table_is_rewritten_only_once_its_last_user_completed`: the remap waits for earlier work,
    and the next submission's table names the new copy and is not dirty;
  - `a_range_a_miss_may_withhold_is_settled_before_it_is_read`;
  - `a_program_storing_through_a_table_keeps_the_barrier`.

## 4. Checks / exe

Commits `ee65a10d9` (the change) and `2b613cbf8` (`defer_after` is now used only by
diagnostics: `allow(dead_code)` without them). Both are formatted with `cargo fmt --all`; the
tree is clean.

- `clippy -p victory-gpu -p victory-shader --all-targets --locked -D warnings`: passes, default
  features and `--all-features` (`scratch/PageTables/logs/build.log`).
- `cargo xtask policy`: one violation, the `lower.rs` size (617,682 bytes), which was there
  before this change (`logs/policy.log`).
- Diag exe (`cargo build --profile diag -p victory --features victory/diagnostics`):
  `/tmp/vexe-pagetables-0b9710a6/victory.exe`, sha256
  `0b9710a646b5338f61220b84695638049b38277367e6d8f267f2ea44506a8409`. It carries the chain
  timers and the new `VICTORY_PT_BARRIER` line. The first run retranslates every shader
  (translator source hash).
- Device and CPU tests: see §5.

## 5. Test runs

Runs went through gate `/tmp/qpagetables.sh`, which takes `gpu-gate.lock` and the build lock,
uses its own prefix `scratch/PageTables/q-prefix` with GE-Proton vkd3d, and passes
`--test-threads=1`. Logs are in `scratch/PageTables/logs/gpu-*.log`. The branch's test exes were
built from `2b613cbf8`. The control is ChainTimers' plain test exe for `116f56e81` (copied to
`scratch/PageTables/base/`) plus a victory-shader test exe built from that commit; both ran in
the same prefix.

| filter | branch | base `116f56e81` |
|---|---|---|
| `vram_async` (in-flight, barrier, settle tests + the 6 new host tests) | **ok, 21 passed, 1 ignored** | — |
| `shadow::tests::` (incl. `kept_ranges_cross_every_copy`) | **ok, 22 passed** | — |
| `memory::` (incl. `a_table_knows_its_last_user…`) | 8 passed, 5 failed | 7 passed, **same 5 failed** |
| `host::tests::` | 65 passed, 7 failed | 65 passed, **same 7 failed** |
| victory-shader `memory_tests` (incl. `only_stores_and_atomics…`) | 12 passed, 1 failed | 11 passed, **same 1 failed** |

- The new device tests all passed:
  - `page_tables_stay_in_flight_with_the_switch`: with the switch on, the next submission
    returned in under 400 ms with 2 in flight; with it off, it waited for the gate.
  - `a_table_is_rewritten_only_once_its_last_user_completed`: a remap while earlier work still
    reads the old table waits for that work; the submission after it sees the new copy's
    views, and its table is copied to the GPU.
  - `a_range_a_miss_may_withhold_is_settled_before_it_is_read`: the memory-flagged range is
    settled and goes back before the reader is prepared; the unflagged control stays in flight.
  - `a_program_storing_through_a_table_keeps_the_barrier`.
- The pre-existing failures also fail on base in the same prefix:
  - the 5 `memory::tests`: `D3DCompile` returns `E_NOTIMPL` in this prefix (test HLSL compute);
  - `host::tests` and the others: `read_marks_of_both_bitmaps_are_taken`,
    `capped_heap_keeps_full_dynamic_range`, `a_host_pipeline_refused…`, `forgotten_slot_read…`,
    `unsupported_same_family…`, `out_of_range_static_sampler…`, `the_host_takes_a_pipeline_blending…`;
  - `code_table_buffer_loads_read_zeros_out_of_range`.
- Not measured yet: frame time. Main queued warm-up `1007w` (arm 1), then `1007a-d` (1/0/1/0).
