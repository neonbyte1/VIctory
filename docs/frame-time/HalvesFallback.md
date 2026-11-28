# Halves as a fallback in the fork's `Lower` (shape (a)) — design note

Status: design only, nothing implemented. Scope: #2633's 32-lane halves
(`lower/{halves,wide,confine}.rs`) ported onto fork main's `Lower`, chosen
**only** for programs the console flow already refuses, behind
`VICTORY_HALVES=1` (default off), with our UAV fence applied inside the halves.
Numbers are from `local://Bench2633.md` (same pack, RTX 5090).

Why: the console flow (`fix/uav-order-main`, fork main `6f0aeaefb`) runs
2629/2682 compute programs of the pack; it refuses 42 that #2633 translates
(15 whole-wave section, 13 implicit-derivative sample in the console flow, 12
title barrier not met once per console wave, 1 value before the halves'
meeting, 1 `v_permlanex16_b32`). #2633 refuses 20 the fork translates, so a
fallback-only switch moves 2629 -> at most 2671 (+1.6%) and changes nothing
else.

## 1. Selection: halves are the last attempt, never the first

Both forms live in one `Lower`; the decision stays where the fork makes it
today (`lower.rs` ~1155-1266: `console_lanes`, `split`, `needs_wave64`,
`blocked()`, `wants_console_flow`, then `console_cf` and `s.wave_size`).

Add `halves: bool` (a field and a local in `new`), computed after the existing
decision:

```
halves = HALVES.get()                      // VICTORY_HALVES=1
      && x.native.wave64 && !x.target.runs(64) && x.target.runs(32)
      && needs_wave64 && !split
      && no_console_flow.is_some()         // the console flow is blocked
```

Then, where `console_cf` is chosen today, `halves` wins over "refuse":
`console_lanes = true`, `split = false`, `pairs = None`, `console_cf = false`,
`s.wave_size = Some(32)`. If the halves attempt also refuses, the console
flow's reason is still what the program reports (today's message), with the
halves reason logged at `debug!` for the census. Net effect on a 64-lane
adapter or on `VICTORY_HALVES=0`: byte-identical emission.

## 2. What is ported, and where it lands

| from #2633 | lines | landing |
|---|---|---|
| `lower/halves.rs` | 1438 | copied; compiles against the fork's `Lower` after the API deltas below |
| `lower/wide.rs` | 1587 | copied (wave-wide regions) |
| `lower/confine.rs` | 221 | copied (one-half regions) |
| `lower.rs` +336/-1361 | -- | **only the additions** are taken (fields, `relabelled`/`alike`, the plan/literal retry loop). The 1361 deleted lines are upstream's per-lane code the fork still needs. |
| `lower/tests.rs` +1298 | | not taken (upstream's unit tests); #2633's `halves_tests` (33) and `halves_random_tests` (1) are, as the oracle |

API deltas measured from the three modules (the only names they use that the
fork's `lower.rs` does not have): fields `halves`, `literal`, `literal_wide`,
`plan`, `wide`; helpers `relabelled` (console flat id / wave / lane from the
host wave) and `alike`; emitter constructors `s.shared(name, dwords)` and
`s.lds()` (the trade's group-shared slots -- #2633's `slots()`/`slot_dwords()`
use them). The fork's console flow allocates pair slots by hand
(`pair_slots`, `meet_fenced` at lower.rs:13461); the halves keep their own
allocator so the PR's code stays recognizable: ~20 lines in the emitter for
the two constructors, `halves_dwords` accounted beside `lds_dwords` so the
32 KiB limit is still respected (the head commit `00a023481` is exactly the
"borrow slots from the program's LDS when full" rule).

`Lower` fields to add (beside `console_lanes`/`split`/`pairs`, lower.rs:472-600):
`halves`, `exchange: halves::Exchange`, `wide: wide::Wide`, `plan: &wide::Plan`,
`literal: &halves::Literal`, `halves_refused: Option<Why>`, and the existing
`uav_order: HashSet<(usize, usize)>` (lower.rs:1417) stays as the one UAV site
set.

Retry loop: the fork's `lower()` already retries (`Stop::Retry` for values,
console-flow attempts). #2633's loop (plan growth for wave-wide regions,
literal steps for whole-wave sections, retry-without-value) must wrap **only**
the halves attempt, so the console-flow path keeps its own retry order. Entry:
the same `lower()` in `lower.rs`; a `halves` plan/literal is created before the
attempt and dropped when halves are not chosen.

## 3. The UAV fence inside the halves (our fix, carried over)

The analysis is already ours and form-independent:
`lower/uav_order.rs::meetings(p, &Group{..}, raw)` gives `(block, inst)` sites
where a UAV access conflicts with an earlier one across the halves; the fork
consumes it at `lower.rs:5061` (`if self.uav_order.contains(&(b, i)) {
self.order_uavs() }`) and `order_uavs()` (uav_order.rs:1158) emits
`barrier 1|4|8` for a group of one console wave, or a pair meeting fenced `4|8`.

Change for halves:
1. Set `self.uav_order = uav_sites` when the halves form is active (today only
   `console_cf && split`), computing `uav_sites` with the same call at
   `lower.rs:1248` -- it is independent of the form.
2. At a site the halves must be at a point where every thread of the group
   meets, which is exactly what #2633 enforces for its exchanges (`meets()`);
   so feed the sites into the halves' "needs a meeting" requirement (the
   `for_want_of_a_meeting` path that already pushes a divergent region into
   `wide`), and emit `order_uavs()` there. In halves `pairs` is `None`, so
   `order_uavs()` takes the group-barrier arm: `barrier 1|4|8`, a group-scope
   UAV fence -- correct for a group of one console wave and a superset for a
   group of several (it syncs the whole group).
3. `VICTORY_UAV_ORDER=off` must keep working: with it, `meetings` is not
   consulted and the halves emit today's bytes (this is the A/B arm).

Cost: one field assignment, one push into the meeting requirement, one hook
call. The risk is not the fence, it is (2): a barrier in a divergent region
hangs the GPU, and the `wide` plan must grow for the UAV site as it does for
LDS and exchanges. The 4 program classes the console flow refuses
(whole-wave section, implicit derivatives, title barrier, v_permlanex16) are
mostly wave-wide/divergent themselves, so this is the part to test first.

## 4. Options, switch and cache

Follow `pair_meetings` exactly (mod.rs:516-530, victory-gpu/shader_cache.rs:108-170):
- `emit::Options { halves: bool }`, default false; a process static
  `HALVES: Cell<bool>` with a `Halves(bool)` guard, as `PairMeetings` does
  (lower.rs:238-250).
- `crate::shader_cache::halves()` from `VICTORY_HALVES` (on unless `0`), passed
  into `emit::Options` and into `host_key(build, target, pair_meetings, uav)`
  (shader_cache.rs:160) -- otherwise a cache filled with the switch off serves
  console-flow-only DXIL with it on. The key already carries `pair_meetings` and
  `uav`, so this is one more byte and no format change.
- `apps/victory translate` gains `--halves` beside `--wave32` for the census.

## 5. Acceptance

1. `VICTORY_HALVES=0`: census and game bytes identical to fork main today.
2. `VICTORY_HALVES=1`: 34/34 of #2633's halves tests pass on the 5090 (they are
   `cfg(test, windows)` GPU tests, 51 s for the set); they call
   `lowered(&n, host_wave32)` -- map that to the fork's `Target` (32-only) or add
   a test helper.
3. Census: 2629 -> 2671 of 2682 compute emitted; the 20 programs the fork
   translates stay on their current forms (assert by label).
4. UAV: `split_tests::uav_writes_are_seen_by_the_other_half_*` (our program,
   `Halves::ConsoleFlow`) with `VICTORY_HALVES=1` on a program the console flow
   refuses must show 0 stale reads; the same run with `VICTORY_UAV_ORDER=off`
   must show stale reads (the fence is load-bearing, as measured for the
   console flow). My ported #2633-side test (`halves_uav_tests`) is the
   "no fence" control: 0 `dx.op.barrier` in the emitted halves, and 100 % of the
   other half's later reads stale (256/256, 1024/1024, 4096/4096).
5. Game A-B with the switch on/off, 420 s runs, frame time and refusals.

## 6. Work estimate [INFERENCE: engineering estimate from the measured surface]

| step | work | estimate |
|---|---|---|
| 0 | merge shell: fork main + PR's non-lowering parts (Options, cache key, CLI), 78 conflict hunks minus `lower.rs` | 0.5-1 d |
| 1 | port `halves`/`wide`/`confine` until they compile: 5 fields, 2 helpers, 2 emitter constructors, `meets()`/`slot_dwords()` accounted beside LDS | 2-3 d |
| 2 | selection + retry-loop integration in the fork's `lower()` (halves last, only when `blocked()`) | 1-2 d |
| 3 | UAV fence: `uav_order` in halves, sites pushed into the meeting requirement, wave-wide growth for a site in a divergent region | 1-2 d |
| 4 | tests, census, GPU A/B, docs (incl. the note in `docs/`) | 1-1.5 d |
| | **total** | **5.5-9.5 d** |

Largest uncertainties: step 3 (barrier legality in divergent regions -- the
part with a hang as its failure mode) and step 1 (the modules are imperative
code written against a differently shaped `Lower`; the compiler finds the rest
of the surface, and the 34 GPU tests are the oracle).

Out of scope here: picking halves *before* the console flow (that would change
the other 2629 programs and needs its own A/B), and #2633's `lower/tests.rs`
(upstream unit tests for its own per-lane code).
