# GPU time per program and pass class (`VICTORY_GPU_PASSES`)

Branch `gpupasses` in `~/projects/gta6-recomp/scratch/GpuPasses/wt`, two commits on 4bf60796e:
- edf1bc916: Diagnostics: GPU time per program and kind of pass (VICTORY_GPU_PASSES)
- bfb22b92e: GPU passes: each program's longest single command and its groups

**Exe:** `/tmp/vexe-gpupasses-bfb22b92/victory.exe` (diag profile, `victory/diagnostics`).
sha256 `237666dfe4baa0b5dd0e731ac3d92e4b3f015f69f6651aee633471d073b5b7b0` (also in `victory.exe.sha256` beside it).

**Run with:** `VICTORY_GPU_PASSES=1` to time every frame. `=n` times only the lists begun in one frame of every n. Leave `VICTORY_GPU_TIMING` off: that mode runs and waits for each list on its own.
**Shader cache:** this build changes `victory-shader`, so the translation cache key changes and the first run translates from cold.

## What is measured and how

- **Where it plugs in.** It extends the existing `VICTORY_GPU_TIMING` machinery rather than adding a second one. That machinery already gives each `HostList` its own timestamp heap of 4096 entries and a read-back buffer, one stamp per noted command, and resolves them at `close()`. New pieces:
  - Pass timing switches the stamps on per list recording, chosen at reset (`note_start`).
  - A key is stored with each note.
  - Lists run as they normally do (overlapped, batched). Nothing is serialised or waited for.
- **Stamps.** Each timed list takes a GPU timestamp before every command it notes, plus `list start` and one at its end. A command is charged the time from its stamp to the next one, so the barriers and state changes recorded after it count towards it.
- **Command classes:**
  - `draw`, `indexed draw`, `mesh`, `dispatch`
  - `indirect draw/indexed/mesh/dispatch`
  - `argument clamp` and `argument clamp window` (always split out when passes are on; `VICTORY_TIMED_CLAMP` is no longer needed)
  - `copy`: copy resource, texture copy, read-back copy, texture upload, depth twin copy
  - `clear`: RTV, DSV and UAV clears
  - `resolve`, `query resolve`
  - `title barrier`: the barrier itself and the drain it waits for
  - `dropped`
  - `list start`: state moves, twins and hoisted clamps before the first noted command
  - `list of ours`: upload lists
- **No stall, no shared ring.** Each list in flight has its own heap. Stamps are read in `Host::finish` once that submission's fence has completed, so every heap is reused only after its list has run. This is a ring of query heaps in effect, one per pooled list.
- **What a command is keyed by.** Its title pipeline's host pipeline (`Arc<Pending>`, held until the report), whether that pipeline is the BVH builder's (`pipeline.builder`), and for dispatch and mesh commands their group counts.
- **Lowering form per program.** A new `victory_shader::emit::form_with(container, options)` runs the same front end and lowering choice as `translate_with` and reports which branch it took. The rows are:
  - `per-lane`
  - `per-lane (console flow refused)`
  - `console flow`
  - `console flow in halves`
  - `pair meetings`
  - extra flags: `split`, `waterfall`, `killing`

  This runs on its own thread, once per program the report names, with the same options the shader cache uses. A program whose form is not known yet shows `form pending`. A unit test checks that it reports pair meetings for a 256-thread group and console flow in halves for a 64-thread group on a 32-lane host.

### Report: one block every 5 s, normalised per timed frame

1. `GPU passes (one frame in n, F timed): lists X ms/frame from first stamp to last (N lists/frame), submissions Y ms/frame from first timed stamp to last (…; Y−X ms in lists not timed and between lists); C commands/frame timed, U past the stamps; adding the stamps Z ms/frame on the completing thread`
   - **Y** is the GPU busy span of the timed submissions. Compare it with the ~60 ms/frame of GPU command time measured earlier.
   - **Y−X** is GPU time in lists that are not timed: state preludes (including hoisted clamps), the VRAM copy lists run around each submission, the upload list, and the gaps between lists.
2. `GPU passes kind <class>: ms/frame, count/frame, µs each, longest <ms>`: totals by class.
3. `GPU passes BVH builder: ms/frame, commands/frame`.
4. `GPU passes form <compute|graphics> <form>: ms/frame, commands/frame`: totals by lowering form.
5. `GPU passes program k: ms/frame, n/frame (per-class counts), µs each, longest <ms> (XxYxZ groups), <form>[, BVH builder]: compute <hash>` (or `vertex <hash>, pixel <hash>`): the top 25 by ms/frame.
6. `GPU passes longest command, below the 25 k: …`: up to 10 more programs whose longest single command took ≥ 1 ms.
   - Added for Main's RTX 2060 outliers: a 2.5 s single dispatch of `6f6a359d1166bb27` (18x1x1) would show here, with its groups and form.
   - The hashes are the same 16-hex BLAKE3 prefixes the logs use.

## Overhead (bounded)

- **GPU: one timestamp per noted command, one per list.**
  - On vkd3d-proton a timestamp is written once all earlier work has finished (bottom of pipe). Consecutive commands without barriers then no longer overlap, and each command is charged its incremental completion time. Long dispatches are measured correctly; streams of tiny commands are inflated.
  - Measured on the RTX 5090 by the device test `passes::tests::a_stamp_costs_the_gpu_little`: 2000 empty dispatches took 684 µs timed one by one and 228 µs untimed, so **0.23 µs per stamp**. At roughly 5k timed commands per frame that bounds the GPU overhead at about **1-2 ms/frame** for compute-like commands [INFERENCE: extrapolated from empty dispatches].
  - **[INFERENCE]** vkd3d-proton may end or suspend the current render pass for a timestamp. Draw-heavy passes can then cost more under timing than without it. The draw rows are best read relative to each other.
- **Upper bound on stamps.** At most 4096 per list (`host::STAMPS`). Commands past that are counted (`past the stamps`) and charged to the last timed command.
- **CPU, recording thread.** Per command: one `EndQuery`, one `Arc` clone and one push. No note texts are built when only passes are on; the device-removed and `VICTORY_GPU_TIMING` texts read title memory, and those are skipped.
- **CPU, completing thread** (on the chain, under the submission lock): mapping each list's read-back buffer, plus one hash-map update per command. The report prints this cost as `adding the stamps … ms/frame`.
- **Bounding the frame impact.** `VICTORY_GPU_PASSES=n` times only the lists begun in one frame of every n, so the other frames run untimed. The report normalises by timed frames, `report_frames / n`.
- **Memory.** One 4096-entry query heap plus a 32 KB read-back buffer per pooled list, made on first use.

**With the switch off, nothing changes.**
- `note_start` sets the per-list `passes` flag only when the switch is on.
- Every new note site (`note_class`: clears, texture uploads, twin copies) returns at once when the flag is off.
- `note_pass` behaves exactly as `note_op` did. The old `note_op` was replaced by `note_pass`; every caller was migrated, and `note_read` is now keyed as a copy.
- `finish` and the report check `passes::on()` first.
- All of this is behind the `diagnostics` feature. Release builds do not contain it.

## Checks

- **Clippy**, `--workspace --all-targets --locked -D warnings`: clean both without features and with `--all-features` (windows-msvc target).
- **Policy and formatting:** `cargo xtask policy --staged` passes for both commits, and `cargo fmt --all` was run.
  - The first attempt also added fields to `emit/lower.rs`. That file is already over the policy's 512 KB limit, so the attempt was dropped, and `split` and pair meetings are now derived in `emit/mod.rs` from the same conditions `lower` uses.
- **victory-shader:** `pair_meeting_tests` passes, 5 of 5 including the new `the_form_names_pair_meetings_and_halves`.
- **victory-gpu unit tests:**
  - `passes::tests::the_switch_says_how_often`
  - `each_command_is_charged_to_its_stamp_and_the_next` (stamp intervals, longest command and its groups, lists of ours, unresolved lists, submission span)
- **Device tests, through `/tmp/qgpupasses.sh`:**
  - `a_timed_dispatch_takes_a_plausible_time`: a loop dispatch of 65535×64 threads at 4096 and at 16384 steps. Each must take between 5 µs and 1 s, and the 4× dispatch 2–8× as long.
  - `a_stamp_costs_the_gpu_little`: the per-stamp overhead.
  - Result (gate log `scratch/GpuPasses/logs/gpu-passes2.log`, RTX 5090): **4 passed**. The timed dispatches took 0.144 ms and 0.858 ms (4× the steps took 5.96× as long, within the 2–8 bound), and a stamp cost 0.23 µs.
  - Gate waiters were being killed by the run queue's `pkill -xf 'sleep N'` (Main is fixing that). My retry wrapper is `scratch/GpuPasses/retry-gate.sh`.
