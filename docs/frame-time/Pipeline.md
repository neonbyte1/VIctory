# Pipeline: completion of submission N off the submission lock while N+1 is prepared

Worktree `scratch/Pipeline/wt`, branch `wip/perf/pipeline` from `wip/perf/pt-barrier` `2b613cbf8`
(= fork main `4bf60796e` + chain timers `116f56e81` + `VICTORY_PT_BARRIER`). Line numbers are at
`2b613cbf8`, `crates/victory-gpu/src/` unless named.

## 1. The submission lock (step 1)

`Host::submitted: Mutex<Submitted>` (host.rs:73; `Submitted` host.rs:2571: fence `value`, `flight:
VecDeque<InFlight>`, `owners`). Taken by:

| taker | where | holds it across |
|---|---|---|
| `Host::submit` (title queue piece, `queue.rs` pump) | host.rs:3089 via `complete_while` | waits before preparing (`complete_while` drops it **only** for the fence wait, host.rs:3237-3244, then completes what was reached **under** it), then all of `run` (prepare + ECL + Signal), push to `flight` |
| `Host::execute` / `execute_locked` (our lists, tests, `execute_texture_read`) | host.rs:2981/2987 | `complete_while(WAIT_OTHER)` + `run` + `finish` (whole synchronous submission) |
| completion thread `complete_in_background` | host.rs:3281 (peek), 3301 | fence wait **without** it; then `finish` of every reached record **under** it |
| `complete_all` / `complete_held` | host.rs:3198/3205 | as `complete_while` |
| `signal_after` (title Signal behind a queue) | host.rs:3158 | push onto the queue's last record in flight, else set now |
| `hold_submissions` (page remaps, `queue.rs:863`) | host.rs:2974 | "no submission's reads/writes of title memory run" while a title thread remaps pages |
| present resize | host.rs:7686 | fence bump |

State touched under it, by stage:

**Preparing (`Host::run`, host.rs:3385-3854)** — `submitted` is `&mut` for the whole call.
- `resource::evict_unused` only if no record in flight marks (host.rs:3402): the pass clears the read marks.
- `descriptor::sync_heaps` (host.rs:3410): title-copied heap slots → host heaps (`SHADER_HEAPS`).
- `bindless::uploads` (host.rs:3461; bindless.rs:436-452): texture staging over title memory, `shadow::crossing` → `settle(Texture)`.
- `run_deferred` per list (host.rs:5658-5802): `Step::Copy/CopyMemory` (title→title at submission, `crossing` of src/dst → `settle(Copy)`, host.rs:5687-5692), `Step::Fill` (`crossing` → `settle(Fill)`, 5721), `Step::Refresh` → `Batch::add` (`crossing` with own copy excluded, 5872), `Step::Host` (arbitrary host work, e.g. `draw.rs:2052` writes title memory), `Step::GpuWrite` (`watch::writing_over`, `Shadow::writing`: per-owner snapshot `subs` + arm, 5767-5775; no crossing), `Step::Heap` (`bindless::refresh` → `refresh_where_needed` → `s.wrote_over()` + `add_reached`, 5784), `Step::Memory` (`memory::prepare`: table rewrite, `Settled::Table`, `refresh_where_needed`, 5792; memory.rs:798-843).
- `Batch::flush` (host.rs:5929-6014): `settle(Alias/Label)` for the latest crossing owner, `shadow::plan` (Refused(o) → `settle(Same)`), `refresh_planned` (guard lock, title→copy, re-arm, `queue_vram_upload`).
- `shadow::take_vram_uploads`, `vram_before` (mirror → staging memcpy, host.rs:6570), `vram_after` (`PREDICTED`, read-back staging, host.rs:6635), `shadow::fly` (host.rs:3631), `early::fly`, ECL + Signal (`submitted.value`, host.rs:3693-3729), `RETIRED` taken into the record (3849).

**Settle** (host.rs:3339-3376): from inside preparing (the `settle` closure, 3450): completes `flight` front up to `owner` (fence wait + whole `finish`) with the lock still held.

**Completion (`Host::finish`, host.rs:3862-4139)**:
1. fence wait (`wait_for`, `self.event` — the event "used under the submission lock", 3321), device-removed check;
2. `vram_completed` (6758-6892): write marks (`WriteMarks::staged`/`take_ram`), `vram_marked_reads` (MARKS), list **probes** (page tables, `memory` layout), predicted pages `into_mirror`, unpredicted + past-`STAGED_MAX` pages read back through `run_copy_list` (its own CMD mutex/fence but `self.event`, 6996) and the 32 MiB ring, declared ranges' staged bytes into mirrors, `PREDICTED`, `COPY_LISTS`, `STAGING`;
3. missed checks + `memory::note_submission` + `withholding` (3962-3967; globals `SUBMISSION_MISSED`, `MISSED_IN`: "completions run one at a time");
4. `back`/`withheld` lists (3968-4014), `verify_lazy` (diag);
5. `shadow::copy_back_marks` (shadow.rs:1954-2102): held marks → `memory::drop_withheld` (watch::withheld + retake / `DROPPED`), `wrote_over`, per run `copy_back_marked_run`/`copy_back_marked` (guard lease, title read, merge into title **and copy bytes**, `write_mapped`, protections, `taken`, `rearm_for_later` reading later owners' `subs`), then dropped pages → `drop_withheld`;
6. `take_reads` + `descriptor::make_read` (4040-4048: READS bitmap, heap views);
7. declared ranges: `Shadow::copy_back` (compare with the owner's snapshot, merge, write title) or `keep_writes` + `drop_withheld` (4052-4066);
8. lists' after-work (4072-4076; page-table copy-back `defer_after_table`, texture read-backs `defer_after_writing`);
9. bottom-of-pipe writes (labels/timestamps, 4097-4113) — title-visible "GPU done";
10. `shadow::land(owner)`, `early::land` (4116-4117): the ranges leave `FLIGHT`;
11. `release_later` (batch, back, heaps, retired), `log_messages`, title fences in order (4131-4133), `COMPLETED` (4138).

**What the one lock guarantees today**
- G1 Atomicity of a completion against a preparing: a completion is wholly before or wholly after any preparing (or at a settle point inside it). The completion thread's completion can land between two preparings in either order, never inside one except via `settle`.
- G2 Copy-back before later refreshes of the same pages: `FLIGHT` ranges (declared incl. marked pages as whole pages, labels, after-writes, KEPT) are crossed by every reader in a preparing and settled (`shadow.rs:2643-2814`); same-copy pages the title changed under a range in flight → `plan` Refused → `settle(Same)`; overlapping declared ranges wait before preparing (`declared_overlap`, host.rs:2189).
- G3 Labels and title fences after the copy-back and after-work of the same submission (steps 5-9-11 in one call), fences in order (host.rs:4129-4133).
- G4 Completions in submission order (all pop `flight.front()`); globals written per completion (`SUBMISSION_MISSED`, `READS`, `quiet`) assume one at a time.
- G5 Guard/copy state: copy bytes ("only the submitting thread writes it", shadow.rs:919/973), `subs`, trackers (`arm/unarm/rearm/taken/disarm`) are changed only under the lock (title fault handlers use the guard lock/lease, not the submission lock); `hold_submissions` keeps page remaps away from copy-back writes.
- G6 `self.event` is used by one waiter at a time.

## 2. Design (step 2)

### 2.1 Split of `Host::finish` into three phases
- **A (begin, under the lock)**: fence wait, device check, write marks taken + probes (named pages), missed checks, `note_submission`/`withholding`, `back`/`withheld`, `verify_lazy`. Cheap (≈ 1-2 ms/frame).
- **B (middle)**: everything that moves bytes and is local to N's copies: `vram_completed`'s data movement (predicted pages into mirrors, waited read-backs, declared ranges' staged bytes, `PREDICTED`/pools) and `copy_back_marks` whole (incl. its `drop_withheld`s, in their order). Chain timers: I (22-26 ms) + K (21-24 ms) ≈ **44-48 ms/frame of the 50-54 ms copy-back**.
- **C (end, under the lock)**: `take_reads`/`make_read`, declared ranges (2.5 ms), after-work, labels, `land`, `release_later`, fences, `COMPLETED`.
`finish` = A;B;C in one call stays the path for everyone but the completion thread, and for every completion with `VICTORY_PIPELINE=0`. Order inside the completion is today's except that A's missed checks and `verify_lazy` move before B's data movement (they read the lists' RAM bitmaps and lazy pages, not mirrors) and `PREDICTED`/pool returns move after the marks are taken (pools and a heuristic).

### 2.2 Pipeline (`VICTORY_PIPELINE=1`, needs `VICTORY_GPU_OVERLAP`)
The completion thread, for the reached front record N, runs A under the lock, then — if N's B is *local* (no probe named a page: page-table probes name nothing without stores) — registers N's **busy footprint** in `FLIGHT` and parks N as `Submitted::completing`, lets the lock go, runs B, signals a handoff (`Mutex<bool>` + `Condvar`), retakes the lock and runs C unless someone already did.

Busy footprint: the whole title range of every copy B touches (copies of N's write-mark bits, both kinds, and every copy N's lists declare `GpuWrite` on — the read-back mirrors are theirs), flown for owner N with copy `shadow::BUSY`, which no copy equals, so even the copy itself crosses it. `land(N)` in C removes it with N's other ranges.

Exactness argument — the split completion is equivalent to today's atomic completion at the moment C runs:
- Anything a preparing (or another lock holder) does that could observe or change B's state goes through a crossing/settle of N, and a settle of N waits for B and runs C (`end_completing`) before going on — exactly today's settle of N at that point, which is legal by construction (settles happen at arbitrary points of a preparing):
  - refresh of any copy over the footprint (`Batch::add_as` crossing, `refresh_where_needed`), title copies/fills/texture uploads reading or writing title memory there (crossing with copy 0) — existing crossing sites;
  - `Step::GpuWrite` (snapshot of a copy + `watch::writing_over`): new busy check over its range;
  - `Step::Heap`, `Step::Memory` (reach thousands of copies and bump `wrote_over` before the batch flushes), `Step::Host` (arbitrary host work): settle a busy N first;
  - `Shadow::writing` of a later owner is then never interleaved with N's `rearm_for_later`.
- What B touches outside its footprint: pools (`STAGING`, `COPY_LISTS`, `PREDICTED`: mutexes, order-insensitive), `run_copy_list` (its own mutex; now waits with the thread's own event instead of `self.event`, G6), the read-back ring (only one B or inline completion at a time: completions stay in order), `DROPPED` (mutex; `log_frame` already runs off the lock today), copystat atomics. Title fault handlers race B exactly as they race today's completion thread (guard lease).
- G4: only the completion thread starts a B, one at a time; every other completion (`complete_while`, `settle`, the completion thread's own next one) first ends a parked N (`end_completing`), so completions stay in order and `SUBMISSION_MISSED` is N's from A to C.
- G3: labels, `land`, fences stay in C, after B; `signal_after` onto the queue's last record also looks at the parked one.
- G1 for others: `complete_while` treats the parked record like the front of `flight` (barrier, upload list, list again, eviction marks, declared overlap → end it first); `run`'s eviction check counts it as marking; `hold_submissions` ends it before returning (no page remap under B's title writes); `complete_held`/`complete_all`/`execute` end it.
- Deadlock: B takes no lock a lock holder can hold while it waits on the handoff (B: MARKS, guard lease, copy `subs`, pools, CMD, DROPPED; waiters wait only at settle/`complete_while`/`hold_submissions` points, where they hold the submission lock and at most bindless' upload guard, which B never takes).
- Known widening (same class as today): `log_frame`'s `retake_dropped` (present, off the lock) can now fall between A and C of one completion; today it can already fall between two completions or inside one on the completion thread (it takes no lock).

### 2.3 Counters (diagnostics, appended to the `chain per frame:` line)
- copying back by who: the completion thread / settles / waits before preparing / others (both arms);
- pipeline: completions split, B ms off the lock, of it ms overlapping a preparing (intersection of B spans with `run` spans, minus a preparing's own waits for B), waits for a B in progress (count, ms).

### 2.4 Expected gain from the chain timers
Frame today ≈ P + G + K_back (1007a/c, pt1: 93 + 64 + 51 ≈ 208; 1007b, pt0: 82 + 58 + 54 ≈ 199).
With the pipeline: frame ≈ P + G + K_back − O, where O = B time on the completion thread that runs while the submitting thread prepares (not waiting for it). Bounds: 0 ≤ O ≤ B ≈ I + K = 44-48 ms.
- Settled completions (G = 30 ms of settles in pt1, ~8/frame) are reached first by the settler, which does A;B;C inline: no gain there. Barrier waits (pt0: 48 ms page-table waits) likewise complete inline in the waiter.
- Gain comes from completions the completion thread reaches between preparings: their B now runs beside the next preparing instead of blocking its lock. 14 of 28-31 submissions are prepared with the pipe empty, i.e. their predecessors were completed before; those are the candidates.
- [INFERENCE] realistic O ≈ 10-25 ms/frame (frame 185-200 ms) if ≈ ½ of B runs on the completion thread; the new "by who" split measures that share directly in the first run (arm 0 tells it without the switch). Reaching 33 ms needs more than this lever (P ≈ 85 ms alone).

### 2.5 Upstream PRs in this area (read before designing; reviews: `scratch/Pipeline/pr*.diff`, read-only scouts)
All three are on upstream main `63bcca43c`, 1,573 commits behind fork main `4bf60796e` (host.rs +7.3k lines since: flights, `settle`, VRAM mirrors, marks); none applies to the fork.

| PR | what it splits | exact under G1-G6? | A/B as is on the 5090? |
|---|---|---|---|
| #3431 (legayclightning22) flights/late waits/relay/async submit+present | `LATE_WAIT`: prepare+ECL on the caller, no GPU wait; flights in a global line, settled **all** before the next preparing (diff ~536). `RELAY`: a thread does GPU wait + copy-back + labels + fences **under the same `submitted` lock**. `ASYNC_SUBMIT`: one worker runs the pump/prepare. | Labels/fences after copy-back and in order: yes. Copy-back never overlaps the next preparing (settle-all, or relay holding the lock) — so it does not do what we need. Gaps: held waits test labels before settling (stale reads), non-origin waits never settle without relay, relay copy-back concurrent with title writes (read-merge-write window). | No. The fork already has its flights (`Submitted.flight`, per-owner `FLIGHT` ranges, `settle` only where met, completion thread waiting without the lock). Nothing to port for this lever; its async-submit worker only moves title-thread time, which is off our chain. |
| #3449 (same author) thin-list translation + async submit lane | Recording into word chunks replayed by 4-8 workers; `ExecuteCommandLists` enqueues to a lane thread that calls the synchronous `Host::execute` (prepare, GPU wait, copy-back) serially. | Order kept on the lane, but title memory is read at lane time (later than ECL: inputs written after ECL become visible, unmapped-after-ECL reads), forced waits when heads block, cross-queue order weaker than the fork's global flight FIFO. Copy-back untouched. | No as is: it replaces flights with a serial lane (P+G+C per run, worse than the fork's overlap). Thin recording could be ported later onto the fork's pump; it shortens title CPU, not the chain. |
| #2960 (ynnmuraii) async submission + copy queue (from fxceless/victory-fork) | `ASYNC_SUBMIT`: a thread runs the unchanged synchronous execute; `QUEUE2`: texture→buffer read-backs recorded by `list.rs read_back` deferred to a COPY queue after the submission ran (waited per batch; the "up-lane" bit unimplemented); relay lands flights on a thread. | Several races (scout): snapshot release vs a new flight's arm (disarm under no lock), refreshes skipping busy pages (later preparing misses title writes made after ECL), marked pages merged only in settle (fences may precede them), copy queue unordered against the next direct submission. | No. Its 125→7 ms is a 4060 with 17 s stalls; the fork's read-backs are already staged by a list after each submission (`vram_after`, `PREDICTED`), only 5-9 ms/frame are waited for. A copy queue would only shave those waited read-backs; this design takes B (which contains them) off the chain instead. |

Credit: the "copy-back on its own thread, in order, labels after" shape is #3431's relay and #2960's landing thread; this design differs in running B **without** the submission lock while keeping C under it, with exactness from the fork's `FLIGHT` crossings plus a busy footprint.

## 3. Implementation (step 3) — commit `60bda3836` on `wip/perf/pipeline`

Files: `crates/victory-gpu/src/host.rs`, new `host/pipeline.rs`, `shadow.rs`, `copystat.rs`.

- `Host::finish` (host.rs) = `finish_begin` (A) + `finish_middle` (B) + `finish_end` (C), same steps as before; `vram_completed` split into `vram_named` (A: marks taken, `vram_marked_reads`, probes, dedupe) and `vram_rest` (B: predicted into mirrors, waited read-backs, declared ranges' staged bytes, `PREDICTED`, pools). New types beside `VramSync`: `VramRest`, `WriteBack` (was local), `Middle` (+ `footprint`), `Begun`.
- `host/pipeline.rs`: switch `pipeline()` (`VICTORY_PIPELINE`, logged once, needs `overlap()`); `Completing` (parked record + `Begun` + handoff `Mutex<bool>`/`Condvar`); `Host::complete_split` (completion thread: A, park + `shadow::fly_busy`, B without the lock, handoff, C unless ended meanwhile; no footprint → whole under the lock, counted `PipeWhole`); `Host::end_completing` (wait for B holding the submission lock, then C); `copied_back` (COPY_BACK + who); overlap spans.
- `Submitted.completing`; `complete_while` ends it first where it or any later record matches, and before popping reached records; `settle` (also `Table`'s "in flight?" test); `signal_after`; `hold_submissions`; `run`'s eviction test counts it as marking; the completion thread calls `complete_split` per reached record with the switch.
- Preparing hooks: `Settled::Busy` for `Step::GpuWrite` over a busy range, `Step::Host` while anything is busy, `Step::Heap`/`Step::Memory` reaching a busy copy (their entries are collected, the busy one settled, then `wrote_over` + `add_reached` as before; unchanged path when nothing is busy). Everything else meets the footprint through the existing crossings (`Batch::add_as`, `refresh_where_needed`, `Step::Copy/Fill`, `bindless::uploads`, early release).
- `shadow.rs`: `BUSY` (= `usize::MAX - 3`, crossed by every copy and title read), `fly_busy`/`busy`/`busy_owner` (`BUSY_OWNER` atomic + ranges; cleared by `land`), `marked_ranges` (whole ranges of the copies the marks name), `checks_stale` (diag: `VICTORY_STALE_CHECK` ends a parked completion before comparing).
- `run_copy_list` waits on the thread's own event (`wait_unlocked`) instead of `self.event`, which the lock holder uses (G6).
- Default `VICTORY_PIPELINE=0`: the same steps in one call; the only reorderings are A's missed checks/`verify_lazy` before the mirror fills and the pool/`PREDICTED` returns after the marks are taken (independent state, see §2.1); every new check is an atomic load that is false.

Counters (diagnostics), appended to `chain per frame:` (copystat.rs):
`…; copied back by the completion thread X ms, in settles Y, before preparing Z, elsewhere W; pipeline (VICTORY_PIPELINE): S completions with the copy-back off the lock (H whole under it), B ms copied back off the lock, O of it beside a preparing, N waits for one in progress (T ms), U settles of one`.
- by who: `BackThreadUs/BackSettleUs/BackWaitUs/BackOtherUs` (thread-local "who": completion thread, `settle`, `submit`'s `complete_while`, else), both arms — arm 0 tells how much copy-back the completion thread does today (the ceiling of O).
- `PipeOverlapUs` = intersection of middle spans with `run` spans, the preparing's own waits for the middle cut out; `PipeWaits/PipeWaitUs` = lock holders that found the middle still running (settles, `complete_while`, `hold_submissions`); `SettleBusy` = settles from the new step hooks.
- Read: frame ≈ preparing + GPU + copying back − O (copying back still includes the middle's time).

## 4. Tests (step 4)

Gate `/tmp/qpipeline.sh` (shared `gpu-gate.lock`, own prefix `scratch/Pipeline/q-prefix` copied from PageTables', GE-Proton vkd3d, `--test-threads=1`, headless gamescope; `pkill -f "winedbg[ ]--auto"` per Main's fix). Logs: `scratch/Pipeline/logs/gpu-*.log`. Branch test exe `victory_gpu-r2` (`dfc9a9697`, sha256 `460b796e…`; `r1` = `60bda3836`, `7ede6760…`), base = PageTables' test exe of `2b613cbf8` (`ad686222…`), same prefix.

New tests (`host/pipeline.rs`, all **pass**, `gpu-r2-pipe.log`: 6 passed):
- `a_busy_footprint_is_crossed_by_every_copy_until_it_lands` (no GPU): `fly_busy` → `busy`/`busy_owner`, crossed by any copy (its own too) and by title reads (copy 0); gone after `land`.
- `a_title_read_of_a_page_going_back_waits_for_its_bytes`: marked writer N; the completion thread is stopped before N's middle (test hook); a later submission copying that title page at submission (`Step::Copy`) waits for the copy-back and copies the bytes that went back.
- `a_preparing_touching_a_copy_going_back_waits_and_others_do_not`: while N's middle is held, refreshing another copy does not wait, refreshing a page of N's copy that no range in flight names (title left it alone) waits (only the busy footprint makes it).
- `a_label_follows_its_copy_back_off_the_lock`: N's bottom-of-pipe label: absent while the middle is held (page too); a title thread spinning on the label then reads the page and finds the copied-back bytes.
- `holding_submissions_waits_for_a_copy_back_off_the_lock`: a page remap's `hold_submissions` waits for the middle and returns with it ended and the page back.
- `guards_after_a_copy_back_off_the_lock_are_as_under_it` (pipeline off, then on): a second marking submission of the same copy prepared while the first's middle is held waits for it (its snapshot/arming of the copy); both pages go back; a title write afterwards reaches the copy at the next refresh, the shaders' bytes stay.

Do they catch wrong order? Two mutants of `dfc9a9697`, same gate:
| mutant | result |
|---|---|
| no busy footprint (`fly_busy` dropped) | 4 of 6 fail (`gpu-mut-nobusy2.log`): busy assertions, and "the second waited for the first's copy-back (120.5µs)" |
| a settle/hold does not wait for the middle (`end_completing` skips the wait) | 4 of 6 fail (`gpu-mut-probe.log`): "the reader waited for the copy-back (2.08ms)", "the busy copy's second page waited… (278µs)", "the second waited… (1.89ms)", "back before the hold" |
(The label test targets a third order — the end before the middle on the completion thread — not mutated.) A first round of mutants (`gpu-mut-nobusy.log`, `gpu-mut-nowait.log`) passed because their exes did not carry the edits; discarded.

Existing tests, branch vs base in the same prefix:
| filter | branch | base `2b613cbf8` |
|---|---|---|
| `vram_async` | ok, 21 passed, 1 ignored (`gpu-r1-vram.log`) | ok (part of `gpu-base-all.log`) |
| `vram_async` with `VICTORY_PIPELINE=1` | **ok, 21 passed, 1 ignored** (`gpu-r1-vram-pipe.log`) | — |
| `shadow::tests::` | ok, 22 passed | ok |
| `watch::` + `host::tests::` | 100 passed, 7 failed, 3 ignored | same |
| all four together (base) | 143 passed, 7 failed, 4 ignored | 143 passed, **same 7 failed**, 4 ignored |
The 7 are the known machine failures (ChainTimers/PageTables §5): `descriptor::host::tests::{capped_heap_keeps_full_dynamic_range, unsupported_same_family_and_default_uavs_are_refused}`, `host::tests::{forgotten_slot_read_in_the_next_submission_is_made, read_marks_of_both_bitmaps_are_taken}`, `pipeline::host::tests::{a_host_pipeline_refused_by_the_library_is_not_listed_as_prebuilt, out_of_range_static_sampler_mip_lod_bias_serializes_for_host, the_host_takes_a_pipeline_blending_into_an_integer_target}`.

## 5. Checks and exe (step 5)

- `cargo fmt --all --check` clean; `clippy -p victory-gpu --all-targets --locked --target x86_64-pc-windows-msvc -D warnings`: pass, default and `--all-features` (`logs/clippy3.log`, `clippy4.log`, after the last test).
- `cargo xtask policy`: only the pre-existing `victory-shader/src/emit/lower.rs` size violation (`logs/policy.log`); host.rs stays under 512 KiB.
- Diag exe (`cargo build --profile diag -p victory --features victory/diagnostics`), built from `60bda3836` (`dfc9a9697` adds only a test): `/tmp/vexe-pipeline-082f039e/victory.exe`, sha256 `082f039e3788371d15b1fbd30a53289b73586bf4c46fe371dd08b027ad5f3014` (also in `victory.exe.sha256` beside it). It carries the chain timers, `VICTORY_PT_BARRIER` and `VICTORY_PIPELINE`; one exe for both arms.
- Suggested runs: `VICTORY_PIPELINE=0/1` A-B-A-B with `VICTORY_PT_BARRIER=1` and the usual switches (`VICTORY_GPU_OVERLAP` is required: without it nothing completes on the completion thread). Read on the chain line: "copied back by the completion thread" in arm 0 is the ceiling of the gain; in arm 1 "of it beside a preparing" is the gain itself, "waits for one in progress" and "settles of one" what the whole-copy footprint costs.

## 6. Limits and next steps
- The footprint is whole copies: a later preparing touching any page of a copy N writes settles N. If arm 1 shows many busy settles/waits, the next step is a page-level footprint (needs an audit of `watch::Tracked` for a refresh of page Q beside a copy-back of page P of the same copy, never concurrent today except with title faults).
- Settled and barrier-waited completions are done inline by the waiter as before; the pipeline only helps completions the completion thread reaches first. Lever 1 (fewer barriers/settles) and this compound: each drain removed leaves a completion for the completion thread.
- Completions whose page-table probes named pages run whole under the lock (`PipeWhole`).
