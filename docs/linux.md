# VIctory on a Linux host

This is the Linux/Wine harness: build the Windows program on Linux, run
measurements in a queue, and run GPU device tests without a visible window.
It is the same setup we measure on, written down so a second Linux machine
produces comparable numbers.
Testing without coding is [testing.md](testing.md); how the fork works is
[fork-guide.md](fork-guide.md).

Everything here stays on your machine. Game files, logs with game text,
captured frames and shader dumps never go into the repository or into an
issue (see [asset-policy.md](asset-policy.md)).

## 1. What you need

- **NVIDIA GPU** and driver (we measure on one; AMD works for builds and
  tests, but `run-game.sh` logs VRAM with `nvidia-smi` only).
- **Wine 11.x or GE-Proton 11-14+**, plus its **vkd3d-proton**. Wine's own
  Direct3D 12 has no shader model 6.6, so it cannot run the title.
- **gamescope 3.17+** (`--backend headless`).
- **Rust** (the version in `rust-toolchain.toml`) and the Windows target:
  `rustup target add x86_64-pc-windows-msvc`.
- **LLVM**: `clang-cl`, `lld-link`, `llvm-lib` on `PATH`.
- **xwin** (`cargo install xwin`) and the MSVC CRT/Windows SDK it downloads.
- `python3` (the log summarizer), `flock`, `setsid`, `timeout`.
- `lspci` (pciutils) so `hidden.sh` can pick the GPU itself; otherwise set
  `VICTORY_VK_DEVICE` to your card's `vendor:device` (for example
  `10de:2b85`).
- A **user systemd** (`systemd-run --user`) for the RAM cap, 34 GB of RAM
  and about 30 GB of disk for build, caches and the game.

Add the Windows target once; everything else is distribution packages.

## 2. One-time setup

**Cross-compile SDK.** `xwin` unpacks Microsoft's CRT and Windows SDK into
one folder; `tools/linux/env.sh` points clang-cl and lld-link at it.

```sh
xwin --accept-license splat --output "$HOME/.xwin"
```

The other route is `cargo xtask wine-test`, which uses the same folder and a
prefix of its own under `target/`; `env.sh` is for building the game and
measurement binaries by hand.

**Wine prefix with vkd3d-proton.** Keep one prefix for the title, separate
from your normal `~/.wine`:

```sh
export WINEPREFIX=$HOME/victory-prefix
wineboot -u
SYS=$WINEPREFIX/drive_c/windows/system32
VKD3D=$HOME/GE-Proton11-14/files/lib/wine/vkd3d-proton/x86_64-windows
cp "$VKD3D/d3d12.dll" "$VKD3D/d3d12core.dll" "$SYS/"
```

`run-game.sh` unsets an inherited `VKD3D_SHADER_CACHE_PATH`: vkd3d-proton's
pipeline cache must stay on during game runs (with it off every pipeline is
a driver compile, minutes of backlog per run instead of seconds).

**Paths.** Put the game outside the repository and point
`VICTORY_GAME_DIR` at it; `victory.toml` uses the same folder for
`victory run`. Never copy game files into the checkout.

## 3. Building

Use **one `CARGO_TARGET_DIR` per worktree** (short, absolute) so parallel
worktrees never overwrite each other's binaries, and keep each measured
`victory.exe` at its own path:

```sh
export CARGO_TARGET_DIR=/w/vt-linux
. tools/linux/env.sh
tools/linux/build.sh cargo build --profile diag -p victory --features diagnostics
cp "$CARGO_TARGET_DIR/x86_64-pc-windows-msvc/diag/victory.exe" /w/exe/victory-$(git rev-parse --short HEAD).exe
```

`build.sh` takes a machine-wide `flock`, runs at `nice 19`/`ionice -c3` and
uses 4 jobs. Run every build, census and big test suite through it:
two parallel builds are what pushes the machine into swap. The queue takes
the same lock, so a build cannot start during a measured run.

Tests:

```sh
cargo xtask wine-test -- --vkd3d-proton "$VKD3D"
# or, with env.sh sourced:
tools/linux/build.sh cargo test --target x86_64-pc-windows-msvc -p victory-gpu --features diagnostics
```

## 4. A measurement, through the queue

Measurements are 420 s runs of the yard scene with the Series X profile and
ray tracing on. Compare only runs from the same build, machine and settings,
always with an adjacent control (A-B-A-B).

**Start the queue once** in its own terminal:

```sh
export VICTORY_RUNS_DIR=$HOME/victory-runs
export VICTORY_GAME_DIR=/games/gta6
export WINEPREFIX=$HOME/victory-prefix
export VICTORY_VK_DEVICE=10de:2b85   # your card, from `lspci -nn`
tools/linux/queue.sh
```

**Queue a run** by writing a job file. Line 1 is the exe, line 2 the
seconds, line 3 the environment for that run:

```sh
mkdir -p "$HOME/victory-queue"
ENV='VICTORY_GPU_OVERLAP=1 VICTORY_CLAMP_WINDOW=1 VICTORY_LOG=info VICTORY_EARLY_RELEASE=1 VICTORY_EARLY_BARRIERS=1'
printf '%s\n' /w/exe/victory-abc1234.exe 420 "$ENV" > "$HOME/victory-queue/0100-yard-a.job"
printf '%s\n' /w/exe/victory-abc1234.exe 420 "$ENV VICTORY_CLAMP_WINDOW=0" > "$HOME/victory-queue/0101-yard-b.job"
printf '%s\n' /w/exe/victory-abc1234.exe 420 "$ENV" > "$HOME/victory-queue/0102-yard-a.job"
```

- **Warm-up first.** The first run builds caches and is slow; queue it, let
  it finish, and do not use its numbers.
- The queue runs one job at a time, holds the GPU gate for the whole run,
  and prints `RUNQ done <tag>: frame=… gpu=…` from the log. The run's folder
  `$VICTORY_RUNS_DIR/<tag>/` holds `log.txt`, `frames/`, `vram.csv` and
  `result.txt`.
- `run-game.sh` sets the comparability defaults (`VICTORY_SHADER_LOOP_LIMIT`,
  and `VICTORY_SKIP_PIPELINES` for one known-slow pipeline). Leave them
  alone for measurements; change them only to explain a difference.
- **No builds or heavy jobs while a run is queued.** The queue's build lock
  stops new builds from starting, but the machine should still be quiet:
  a compile next to the game added 13-60 ms per frame.
- Job files are moved to `$HOME/victory-queue/done/` when they start; use
  the same tag only after the old run folder is gone.

## 5. Device tests, through a gate

GPU tests need a display for DXGI/D3D12 and must not overlap a game run.
A *gate* gives them a hidden one and serialises them:

```sh
export VICTORY_VKD3D_DIR=$HOME/GE-Proton11-14/files/lib/wine/vkd3d-proton/x86_64-windows
cp tools/linux/gate-template.sh /tmp/qgpupasses.sh

cargo test --target x86_64-pc-windows-msvc -p victory-gpu --features diagnostics --no-run
# ... take the printed .exe path, then:
setsid /tmp/qgpupasses.sh passes /w/vt/target/x86_64-pc-windows-msvc/debug/deps/victory_gpu-<hash>.exe \
  >/dev/null 2>&1 & disown
```

Rules that the template and the queue rely on:

- **Gate scripts are named `q<name>.sh` in `/tmp`** (`VICTORY_GATE_DIR`,
  `VICTORY_GATE_PATTERN`). The queue waits while one matches, so it does not
  start a game under a waiting test.
- **The gate takes `flock /tmp/victory-gpu-gate.lock`** (`VICTORY_GATE_LOCK`)
  and holds it for the whole test; the queue holds the same lock for a whole
  game run. Waiting on the lock alone is correct.
- **Start the gate detached** (`setsid … & disown`). A tool timeout killing
  a waiter that has not yet entered the lock looks like "no test running" to
  everyone else.
- **Never `pkill` a pattern that matches other waiters.** The template uses
  `pkill -f "winedbg[ ]--auto"`: the bracket keeps the pattern from matching
  its own command line. A bare `pkill wine` or `pkill -f winedbg` would hit
  the game run or another gate.
- **Never `wineserver -k` the game's prefix.** The gate has a prefix of its
  own (`$VICTORY_RUNS_DIR/gate/q-prefix`); shutting down the game's prefix
  during a run kills that run.
- The gate runs the exe through `hidden.sh` (fresh headless gamescope, Wine
  crash dialogs off) and kills its gamescope group afterwards, so nothing
  outlives the test and holds the gate.
- Tests run with `--test-threads=1`: several tests sharing one GPU collide.

The log for a tag is `$VICTORY_RUNS_DIR/gate/logs/gpu-<tag>.log`.

## 6. Memory, VRAM and driver resets

- `capped.sh` wraps a command in a user scope with `MemoryMax` (34 GB) and no
  swap, so an out-of-memory event kills the run, not the desktop. The queue
  uses it for game runs.
- `ramguard.sh` watches RAM: above 53 GB it pauses our builds
  (`cargo`, `rustc`, `clang-cl`, `victory.exe translate`) with `SIGSTOP`,
  below 48 GB it resumes them. It never touches the game or other users'
  processes. Run it next to the queue on a shared machine.
- `run-game.sh` samples VRAM with `nvidia-smi` into `vram.csv` (peak in
  `result.txt`) and watches the kernel log for an **Xid**. On a reset it
  waits `XID_GRACE` (15 s) and stops only that run's gamescope, so the
  desktop recovers instead of hanging until the time limit.
- A run that disappears with `SIGKILL` and no message was most likely killed
  by the memory cap.

## 7. Reading results

The table for a test report comes from the log:

```sh
sh .github/scripts/summarize-run.sh "$VICTORY_RUNS_DIR/0100-yard-a/log.txt"
tools/linux/runstat.sh 0100-yard-a 0101-yard-b 0102-yard-a   # one line each
```

`runstat.sh` resolves tags under `VICTORY_RUNS_DIR` and calls the same
summarizer, so the numbers match the template. It prints frame/min/fps,
preparing, GPU, copy-back, dropped draws, VRAM, refused shaders and fault
counts. Logs carry ANSI colours; both scripts strip them.

The frame number is the **median of the last 30** `victory_gpu::host` lines.
Picture checks use frames 1700/2048 against the control, not against older
runs.

## 8. Reporting

Fill the fork's result template (or the "Test report (no coding needed)"
issue template for an end-user report) with:

- hardware and driver, distribution, Wine/GE-Proton and vkd3d-proton
  version, commit (`git rev-parse --short HEAD`), the measured exe's hash;
- env from the job's line 3 and the run length (420 s);
- the summarizer table for the measured run and its control, adjacent;
- VRAM peak, Xid/driver-reset lines, refused shader count;
- what you changed between A and B; for A-B-A-B, all four lines.

A number without an adjacent control is a note, not a result. Say what is a
guess. Disclose AI help at the end of the post. Never attach the log,
frames, shader dumps or game files.
