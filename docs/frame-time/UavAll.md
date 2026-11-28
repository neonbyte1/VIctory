# UavAll: UAV order between the halves outside the console flow

Branch `fix/uav-order-everywhere` @ **`d7fb97cf3`**: one commit on `fix/console-flow-uav-order` 5df170e23, in worktree `scratch/UavAll/wt`. Not pushed. Fork main has moved on to 4bf60796e; this branch has not been rebased onto it.

## Exposure per form, before and after
**The test.** `uav_write_then_read` (split_tests.rs) now takes a form, `Halves`:
- `ConsoleFlow`: a v_readlane of lane 40.
- `PerLane`: no lane reads, so no wave size is declared.
- `Split`: a `v_mov_b32_dpp row_mirror`, so lane reads stay inside each half. Wave size 32, plain split.

The tests are `split_tests::uav_writes_are_seen_by_the_other_half_{in_program_order,per_lane,split}`. Setup: RTX 5090, gate `/tmp/quavall.sh`, 1024 groups × 16 dispatches per shape. Logs are `scratch/UavAll/gate/base-uav.log` (before) and `final-group.log` (after).

Stale reads, before (5df170e23) → after (d7fb97cf3):

| shape | console flow | per-lane | split |
|---|---|---|---|
| 5→40, 64 | 0 → 0 | 996/16384 → 0 | 536/16384 → 0 |
| 5→40, 128 | 0 → 0 | 6869/32768 → 0 | 5819/32768 → 0 |
| 5→40, 256 | 0 → 0 | 16200/65536 → 0 | 14114/65536 → 0 |
| 40→5, 64 | 0 → 0 | 1282/16384 → 0 | 1042/16384 → 0 |
| 40→5, 128 | 0 → 0 | 8091/32768 → 0 | 6892/32768 → 0 |
| 40→5, 256 | 0 → 0 | 22294/65536 → 0 | 22732/65536 → 0 |

- **Before.** Both forms were exposed in every shape. Neither form had a barrier: per-lane declared `wave size any`, split `wave size 32`.
- **After.** Both now take the console flow: wave size 32, with a barrier or pair meeting. The tests assert this.

## Design and the evidence for it
**(a) Routing, chosen.** A wave64 compute program on a 32-only host (group > 32 threads, writes UAVs) that has a conflicting UAV access now takes the console flow wherever the console flow can lower it. There the existing UAV meetings apply.
- The gate is the same one the console flow already uses: several waves without pair meetings, barriers not met once, ds_permute, whole-wave sections in groups other than 64. It is factored into a `blocked()` closure in `Lower::new`.
- A blocked program keeps its form. If the console-flow lowering fails, the program falls back to the per-lane form as before, so nothing new is refused.
- Programs without a conflict emit the same bytes as before.

**(b) Meetings inside the existing forms, rejected.**
- The per-lane form and split halves branch per lane, and the conflicting accesses mostly sit under per-lane EXEC (a bounds-check `if`). The census leaf diagnostics show this was the dominant "unknown" cause before EXEC tracking.
- A group barrier or pair meeting there would be in divergent flow, which is UB or a hang. Only the console flow keeps both halves at the same program point.

**(c) Address analysis, kept, applied to the console-flow rule too.**
- **When two accesses are judged apart.** An access is lane-private only when:
  - it goes through the same descriptor: the same table slot or root-parameter address, invariant;
  - its address differs from the other access's only by constants and by a per-lane function of the thread id. That function is computed concretely per thread, through add, sub, shifts, multiplies, u24 multiplies, and, or, bfe, min/max, med3, mbcnt and the like;
  - shared terms are atoms (SGPRs or uniform values defined outside loops) that must match;
  - per lane of the console wave, the byte ranges (or texel tuples) of different lanes never meet.
- **Structured buffers** use the stride the host addresses them with: a root view's own, or `UsedResource.stride`. Typed buffers count by element.
- **Values computed under EXEC** carry that EXEC as a guard. The access's EXEC must be provably inside it, through ANDs, saveexec, `v_cmp`, ORs and phis.
- **Images.** Non-spatial coordinates (slice, mip, sample) must be uniform.
- **Supporting change.** `used_kind` became a free `used_resource()` that both lowering and the analysis use. Its logic is identical, and the off arm proves the emitted bytes did not change.
- **What limits (c)** (diagnostic census, instrumentation not committed): most remaining conflicts involve an access whose address is unknown. Those are data-dependent indices loaded from memory, float→int coordinates, coordinates clamped with a uniform bound, loop-varying soffset, and v_cndmask. The next largest group is different descriptors: read X, write Y. That is conservative by design, because descriptors may alias.

## Census
`translate shaders.rpf --list --host-lanes 32-32 --jobs 16`, `VICTORY_WAVE32_APPROX=1`, compared with `/tmp/vexe-uavorder-5df170e2` (list `census-base.txt`, dumps `cd-base`).
- **Outcomes.** Identical in all three arms (all, raw, off): **0 lost, 0 gained, refusals unchanged.**
- **DXIL changed (all arm): 540 of 25622 dumps.**
  - 462 per-lane programs routed to the console flow;
  - 17 split programs routed;
  - 61 console-flow programs with fewer meeting sites.
- **off arm.** All 25622 dumps are byte-identical to main 2a2b052df (UavOrder's `cd-main`). So one exe gives the off/raw/all A/B.
- **Before vs after (c).** The per-form counts come from the instrumented run. Sites are static and per distinct program; final dumps are byte-identical to the instrumented run's.

| | programs with ≥1 site (old rule → with (c)) | sites (old → with (c)) |
|---|---|---|
| console flow, all | 328 → 326 | 6483 → 6153 (−330) |
| console flow, raw | 178 → 176 | 2333 → 2257 |
| per-lane, all | 549 → 517 | 3541 → 3080 |
| split, all | 17 → 17 | 30 → 27 |

- **Routed, all arm (487 distinct programs).** By threads per group: 64: 326, 128: 52, 256: 86, 1024: 19, 48: 2, 384: 1, 512: 1. In the raw arm, 155 are routed.
- **Still exposed (all arm), about 55 programs.**
  - 47 are kept out by the console-flow gate: 36 with a title barrier not met once, 11 with a whole-wave section in a group that isn't 64 threads.
  - 8 fall back to the per-lane form because the console flow cannot lower them.
- **Performance cost (not measured).** About 480 programs now run in the console flow, where every lane runs every block, plus their meetings. The census cannot measure that; it needs frame A/B runs: `VICTORY_UAV_ORDER=off|raw|all`.
- **Files.** Lists and dumps are in `scratch/UavAll/`: `census-{base,final,finalraw,finaloff}.txt`, `cd-*`, and the diagnostic runs `diag-instr5*.txt`.

## Checks (d7fb97cf3, cargo +1.98.1, x86_64-pc-windows-msvc)
- **clippy** `--workspace --all-targets --locked -D warnings`: clean, both with and without `--all-features`.
- **rustfmt hunks (repo `rustfmt.toml`), base → new:** lower.rs 43/43, uav_order.rs 0/0, mod.rs 14/14, split_tests.rs 1/1, shader_cache.rs 3/3. So 0 new hunks.
- **policy:** only the existing violation, lower.rs over 512 KiB: 556,615 → 558,050 bytes.
- **CPU tests:**
  - `uav_order::tests` 6/6: three are the old rule tests, three are new: own dwords and tiles, lane functions of the thread id, address under a branch EXEC.
  - `shader_cache::tests::keys_tell_hosts_wave_sizes_apart` passes, covering the off and raw keys.
- **Device tests (gate):** `pair_meeting_tests:: split_tests:: uav_order::` **46 passed, 0 failed**. All 18 shapes of the three UAV tests read 0 stale.

## Diag exe
`/tmp/vexe-uavall-d7fb97cf/victory.exe` (`--profile diag --features diagnostics`), sha256 `b4fddc9f95c7b4dd0e8656dc3ae79e1b78d355547044fd1c56019b5176cdf331`. This is the same binary the census used.

`VICTORY_UAV_ORDER`:
- `all` (default): routing, meetings and address analysis.
- `raw`: only reads after writes.
- `off`: none of it; byte-identical to main. Added in this commit.

The shader cache keys on `off` and `raw`.

## Port to fork main 4bf60796e: branch `fix/uav-order-main` @ **`0c0e85ef6`**
- **Worktree.** `scratch/UavAll/main`. Not pushed.
- **Commits.** Three commits on 4bf60796e:
  - `b91430335` (= 7797c99d9);
  - `a0c88f2d7` (= 5df170e23);
  - `0c0e85ef6` (= d7fb97cf3, with `cargo fmt --all` folded in).
- **Not needed.** No c9e91a1b9 equivalent: main already contains dea001245.
- **Conflict.** Only in a0c88f2d7, in `shader_cache.rs` `host_key` (the call site and the test). I took the branch side, and `cargo fmt --all --check` is clean.
- **Exe:** `/tmp/vexe-uavall-main-0c0e85ef/victory.exe`, sha256 `74ecb6dfc66417881bd77b3205d0109b17f5f40acfd881d362e9fd666b4fd38d`.
- **Census vs `/tmp/vexe-main-next4-4bf60796`** (`cd-m4base`). The censuses after the base one ran under `/tmp/build.sh`.
  - **Outcomes.** Every program keeps its outcome in all three arms: 0 lost, 0 gained.
  - **Wave size column.** Main's list now shows each program's wave size, and only that column changes: 462 programs (all) and 150 (raw) go from `unrestricted` to `wave32`. These are the routed per-lane programs, the same 462 as on the old base. The split ones were already wave32.
  - **off:** byte-identical to main on all 25622 dumps.
  - **DXIL changed vs main:** 785 (all), 315 (raw). These counts include the console-flow meetings from 7797c99d9, which main didn't have. The 540 programs that changed on the old base are part of the 785, except 2.
- **Device tests (gate, `TREE=main`):** `pair_meeting_tests:: split_tests:: uav_order::` 46 passed, 0 failed. All 18 UAV shapes read 0 stale.
- **clippy** `-D warnings`: clean, with and without `--all-features`.
- **policy:** only main's existing violation, lower.rs over 512 KiB: 617,490 → 619,765 bytes.
