# Chain timers: levers 1 and 3 split

Fork main `4bf60796e` + branch `diag/chain-timers` (commit `116f56e81`, rebased
from `ca482e4d2`). Diagnostics only (`--features diagnostics`): no behaviour
change, no new switch, nothing timed in a plain build. Every counter goes to
`copystat` (already diagnostics-only) and is printed in the existing five-second
block as **one new line**: `chain per frame:`.

Files: `crates/victory-gpu/src/host.rs`, `crates/victory-gpu/src/copystat.rs`.
Line numbers are the commit's, after `cargo fmt --all`.

## 1. The new line

`copystat::log` (`copystat.rs:545`) now takes the frame line's two totals and
prints them split as its first line (`copystat.rs:568`, called at `host.rs:7332`
with `gpu_ms` = `SUBMISSION_US[1]` and `back_ms` = `COPY_BACK[0]` — the same two
numbers the frame line at `host.rs:7250-7266` prints):

```
chain per frame: <gpu> ms on the GPU or in ExecuteCommandLists: A waiting for a barrier's page
tables, B another barrier, C the upload list, D a list run again, E the eviction pass,
F overlapping declared ranges, G settles, H other completions; copying back <back> ms:
I read-backs (J waited for), K marked pages (L reading and comparing, M waiting for the lease,
N protecting, O writing, P verifying), Q declared ranges (R comparing), S slots to read,
T work after, U lazy-page check; <n> submissions: <o> prepared with work in flight,
<e> with the pipe empty
```

- `A..H` sum to at most `<gpu>`; the difference is `ExecuteCommandLists` (the ECL
  groups in `Host::run`) plus the `Instant` reads.
- `I..U` sum to at most `<back>`; the difference is the steps in `Host::finish`
  around them (the `missed` checks, `host.rs:3792`, and the declared/withheld
  walk, `host.rs:3840`). `K ⊇ L..P` and `Q ⊇ R`.

## 2. What each new field measures

The wait reason is now carried, not just counted: `Host::complete_while`
(`host.rs:3108`) takes `causes` returning `WAIT_*` bits instead of a `bool`
(`host.rs:2232-2243`; the `submit` closure builds them, `host.rs:2975-3000`,
same bits and order, so the existing count lines are unchanged), and attributes
its wait to the matched submission's `Waited` (`host.rs:3127`, `waited_for` at
`host.rs:2251`). `Host::finish`'s own wait (`host.rs:3712`) is attributed by the
`Waited` its caller passes (`Host::complete`'s argument, `host.rs:3145`).

| field | counter, measured at | what it is |
|---|---|---|
| A page tables | `BlockBarrierMemoryUs` (`copystat.rs:333`), `host.rs:3127` via `waited_for` + `InFlight::reasons` (`host.rs:2544`, set `host.rs:3686-3689` from the fold at `host.rs:3417`) + `REASON_MEMORY` (`host.rs:2185`) | Blocked on the fence for a submission that is a barrier for title memory through page tables (`Step::Memory`, `reasons_of` bit 5). The lever-1 target: ~4 of ~5 barrier waits a frame. |
| B another barrier | `BlockBarrierOtherUs` (`copystat.rs:334`) | Barrier waits for the other barrier reasons: heap-view writes, unnamed work after, missed-memory checks, marks not captured apart. A submission waited for only because it must be alone is one of these too: `alone()` implies `barrier()`, so it has no bucket of its own (its count stays `WaitAlone`). |
| C uploads | `BlockUploadsUs` (`copystat.rs:335`) | The upload list's previous use (`uploading && r.uploads`): the upload-ring drain. |
| D a list run again | `BlockListAgainUs` (`copystat.rs:336`) | A host list submitted again before its last run completed. |
| E eviction | `BlockEvictionUs` (`copystat.rs:337`) | The once-a-frame eviction pass waiting for the submissions whose shaders mark reads. |
| F declared ranges | `BlockDeclaredUs` (`copystat.rs:338`) | Waiting for a submission whose declared range this one declares too (`declared_overlap`). |
| G settles | `BlockSettleUs` (`copystat.rs:339`), `host.rs:3712` with `Waited::Settle` (`host.rs:3246`) | The fence wait inside `Host::settle` (`host.rs:3225`): a completion a preparing needed mid-way. The settle's whole time (its copy-back included) is still `SettleUs`, which is taken out of `preparing`. |
| H other completions | `BlockOtherUs` (`copystat.rs:340`) | Every other fence wait: `complete_all` (`host.rs:3084`) / `complete_held` (`host.rs:3091`) / `execute` (`host.rs:2869`-`2877`), the completion thread's own `finish` (fence already reached, ~0). |
| I read-backs | `BackVramUs` (`copystat.rs:348`), `host.rs:3769` | `Host::vram_completed` (`host.rs:6545`) for the copies in video memory: the read-back lists, their waits, page-bitmap probes and the bytes put in the mirrors. |
| J waited for | `VramWaitedUs` (existing, `host.rs:6630`) | Of I, the part that waited on a read-back (unpredicted pages, past `STAGED_MAX`). |
| K marked pages | `BackMarksUs` (`copystat.rs:350`), `host.rs:3858` | `shadow::copy_back_marks` (its total is still `MarkedUs`). |
| L..P | `MarkPreUs`/`MarkWaitUs`/`MarkProtectUs`/`MarkWriteUs`/`MarkVerifyUs` (existing, filled `shadow.rs:2070-2076`) | Of K: reading and comparing, waiting for the guard lease, changing protections, writing title memory, checking after. This is the compare / write-to-title-memory / protection-changes split. |
| Q declared ranges | `BackRangesUs` (`copystat.rs:352`), `host.rs:3881` | The loop that copies back or keeps each declared range (`shadow::copy_back`, `shadow.rs:734`). |
| R comparing | `CompareUs` (existing, `shadow.rs:766`) | Of Q, the compare-with-the-snapshot loop; the title write inside it (`merge_writeback` + `write_mapped`, `shadow.rs:809-811`) is inside this number. |
| S slots to read | `BackReadUs` (`copystat.rs:351`), `host.rs:3873` | Budget's first timer gap: `shadow::take_reads` (`shadow.rs:1860`) + `descriptor::make_read` (`descriptor.rs:316`), the heap slots whose texture views the submission's shaders first read. |
| T work after | `BackAfterUs` (`copystat.rs:353`), `host.rs:3901` | Budget's second timer gap: the lists' after-work callbacks (`host.rs:3903-3907`). |
| U lazy-page check | `BackVerifyUs` (`copystat.rs:349`), `host.rs:3850` | `watch::verify_lazy(32)` (`host.rs:3851`) — the rest of the unattributed copy-back; it exists only with diagnostics. |
| n submissions | `Submissions` (existing, `host.rs:3272`) | Submissions prepared per frame (ours included). |
| o work in flight | `PreparedOverlapping` (existing, `host.rs:3493`) | Of n, prepared while an earlier submission was still in flight. |
| e pipe empty | `PreparedEmpty` (`copystat.rs:357`), `host.rs:3498` | Of n, prepared with nothing in flight after the waits above: their GPU work could not overlap anything. `o + e = n`. |

`waited()` (`copystat.rs:397`) maps `Waited` (`host.rs:2502`) to the counter. In
a plain build `waited_for` returns a constant and nothing is timed
(`host.rs:2273`); the enum is carried but never read
(`#[cfg_attr(not(feature = "diagnostics"), allow(dead_code))]`).

## 3. Checks

Commit `116f56e81` on `4bf60796e` (rebased from `ca482e4d2`; the only conflicts
were `copystat::take` added by upstream beside `waited()`, `execute` split into
`execute`/`execute_locked`, and `back` becoming a struct in `finish`'s
copy-back loop — resolved keeping upstream's code and this change's timers).
Tree clean, `cargo fmt --all --check` clean.

- `cargo clippy -p victory-gpu --all-targets --locked --target x86_64-pc-windows-msvc -- -D warnings`:
  pass (`logs/build2.log`).
- Same with `--all-features` (diagnostics): pass.
- `cargo xtask policy` on the rebased tree: **1 violation**,
  `crates/victory-shader/src/emit/lower.rs` size — which the task's rules allow.
  (The `.git-blame-ignore-revs` one of `ca482e4d2` is fixed by `4bf60796e`.)
- Diag build: `cargo build --profile diag -p victory --features victory/diagnostics`.
- Tests, branch against a control built from `4bf60796e` in the same worktree
  layout and environment (gate `/tmp/ct-runtests2.sh`: `flock
  /tmp/gpu-gate.lock`, headless gamescope, GE-Proton vkd3d-proton,
  `--test-threads=1`; run under `/tmp/build.sh` as the machine-wide lock rule
  asks). Exe digests: branch diag `1ed24790…`, plain `c5a9179e…`, control
  `695bb46c…`:

| filter | branch (diag) | branch (plain) | control `4bf60796e` (diag) |
|---|---|---|---|
| `host::vram_async` (the in-flight / barrier / declared / settle tests, `complete_while` + `finish`) | **ok, 15 passed, 1 ignored** | **ok, 15 passed, 1 ignored** | ok, 15 passed, 1 ignored |
| `host::tests::` | 67 passed, **7 failed**, 1 ignored | 65 passed, **7 failed**, 1 ignored | 67 passed, **7 failed**, 1 ignored |
| `shadow::tests::` | **ok, 23 passed** | — | — |
| three of the failures, one by one | fail | — | fail |

The seven failures are the **same seven, with the same messages, on
`4bf60796e`** — nothing this change touches (`shadow::take_reads` bitmaps,
vkd3d's dynamic view-heap range, vkd3d's pipeline library and pipeline-state
limits): `host::tests::{read_marks_of_both_bitmaps_are_taken,
forgotten_slot_read_in_the_next_submission_is_made}`,
`descriptor::host::tests::{capped_heap_keeps_full_dynamic_range,
unsupported_same_family_and_default_uavs_are_refused}`,
`pipeline::host::tests::{a_host_pipeline_refused_by_the_library_is_not_listed_as_prebuilt,
out_of_range_static_sampler_mip_lod_bias_serializes_for_host,
the_host_takes_a_pipeline_blending_into_an_integer_target}`. They are this
machine's vkd3d/Wine behaviour, not a regression: the control fails them
identically, both in the suite and alone.

Not run: `shadow::run_tests::` (the `lazy_views_*` CPU tests, unrelated and
already failing in `scratch/Combined/logs/tests-cpu.log`), and any game run (the
user queues those). No census: no shader changed.

## 4. Diag exe

`/tmp/vexe-chaintimers-86c24b1e/victory.exe`

```
86c24b1eeda175c9cf880c84b95cf5e2fbb20651920ada77342e44ccd3fd9450
```

Built from `116f56e81` (`cargo build --profile diag -p victory --features
victory/diagnostics`); `strings` shows the `chain per frame:` line. Control exe
for the test matrix: `target-base2/.../victory_gpu-50a48ca2bec1bfc5.exe` from
`4bf60796e`, `695bb46c193d59b68b8d3d002b28969b863ab1aedfe790db1b47848e3baf03e2`.

## 5. How to read it

- **A against B**: A is the page-table barriers lever 1(c) would turn into
  bounded writes; B is what would remain (heap views, missed-memory checks,
  marks not captured apart). If A is a large share of the blocked time, 1(c) is
  the drain to remove first.
- **C / D / F**: the drains 1(d) (upload ring) and the declared/eviction rules
  would remove; each is that rule's whole cost, since a wait is attributed to
  the first rule that made it.
- **G (`settles`)**: GPU time spent inside a preparing. It is already counted in
  `SettleUs` (taken out of `preparing`) and its copy-back in `copying back`; G
  only says how much of it was the fence.
- **e against o** (`pipe empty` against `work in flight`): the fraction of
  submissions whose GPU work could not overlap anything. Budget had 13-14 of
  28-30 in flight; if e stays large while A..H shrink, the drains were not what
  kept the pipe empty.
- **Copy-back**: I/J is lever 3(b) (predict read-backs); S is where Budget's
  first ~7 ms of unattributed copy-back went; T is the second; K's L..P split
  shows whether the marked pages' cost is the compare (L), the guard lease (M),
  the protection changes (N) or the write (O), i.e. what lever 3(a) (lazy pages)
  would remove.
- `I..U` not summing to `<back>`: the leftover is the `missed` checks and the
  range-walk bookkeeping around them (`host.rs:3792-3840`), not a missing timer.
