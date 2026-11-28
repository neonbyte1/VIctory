#!/bin/bash
# Sequential game-run queue: runs one queued game at a time, holds the GPU
# gate for the whole run, and pauses while a device-test gate script is
# running. Start it once in its own terminal (or detached) and leave it:
#
#   tools/linux/queue.sh
#
# A job is a file <NNNN>-<tag>.job in VICTORY_QUEUE_DIR with three lines:
#   line 1  path to the victory.exe for this run
#   line 2  seconds the run may take (420 for a measurement)
#   line 3  extra environment, KEY=VAL KEY=VAL (may be empty)
# The job is moved to done/ when it starts, so it never runs twice.
# Example:
#   printf '%s\n' '/w/vt/target/x86_64-pc-windows-msvc/diag/victory.exe' 420 \
#     'VICTORY_GPU_OVERLAP=1 VICTORY_CLAMP_WINDOW=1 VICTORY_LOG=info' \
#     > "$HOME/victory-queue/0100-ab-main.job"
#
# Inputs:
#   VICTORY_RUNS_DIR     $HOME/victory-runs (logs and vram.csv land here)
#   VICTORY_QUEUE_DIR    $HOME/victory-queue (job files, done/)
#   VICTORY_GATE_LOCK    /tmp/victory-gpu-gate.lock; must be the gate scripts'
#   VICTORY_GATE_DIR     /tmp; gate scripts live here (see gate-template.sh)
#   VICTORY_GATE_PATTERN q[a-z0-9]+\.sh; the queue waits while one matches
#   VICTORY_QUEUE_POLL   20 s between checks
#   VICTORY_BUILD_LOCK   $VICTORY_RUNS_DIR/build.lock; the same lock build.sh
#                        takes, so no build starts during a run and a run waits
#                        for a build in flight
#   VICTORY_BUILD_WAIT   7200 s to wait for a build before putting the job back
#   VICTORY_CAPTURE_FRAMES  1500,1700 for perf runs unless set
# The environment the queue was started with (WINEPREFIX, VICTORY_GAME_DIR,
# VICTORY_EXE default, VICTORY_VK_DRIVER_FILES, ...) is inherited by every run;
# a job's line 3 overrides it for that run.
set -u
S=$(dirname "$(readlink -f "$0")")
: "${VICTORY_RUNS_DIR:=$HOME/victory-runs}"
: "${VICTORY_QUEUE_DIR:=$HOME/victory-queue}"
: "${VICTORY_GATE_LOCK:=/tmp/victory-gpu-gate.lock}"
: "${VICTORY_GATE_DIR:=/tmp}"
: "${VICTORY_GATE_PATTERN:=q[a-z0-9]+\.sh}"
: "${VICTORY_QUEUE_POLL:=20}"
: "${VICTORY_BUILD_LOCK:=$VICTORY_RUNS_DIR/build.lock}"
: "${VICTORY_BUILD_WAIT:=7200}"
mkdir -p "$VICTORY_QUEUE_DIR/done" "$VICTORY_RUNS_DIR" "$(dirname "$VICTORY_BUILD_LOCK")"

while :; do
  # A gate script has priority: it holds the same lock, and starting a game
  # while one waits would make the gate wait longer for no reason.
  while pgrep -f "$VICTORY_GATE_DIR/$VICTORY_GATE_PATTERN" >/dev/null 2>&1; do
    sleep "$VICTORY_QUEUE_POLL"
  done
  j=$(ls "$VICTORY_QUEUE_DIR"/*.job 2>/dev/null | sort | head -n1)
  if [ -z "$j" ]; then
    sleep "$VICTORY_QUEUE_POLL"
    continue
  fi
  tag=$(basename "$j" .job | sed 's/^[0-9]*-//')
  exe=$(sed -n 1p "$j")
  dur=$(sed -n 2p "$j")
  envs=$(sed -n 3p "$j")
  case $dur in
    '' | *[!0-9]*)
      echo "RUNQ skip $j: line 2 must be seconds" >&2
      mv "$j" "$VICTORY_QUEUE_DIR/done/" 2>/dev/null || rm -f "$j"
      continue
      ;;
  esac
  if [ -n "$envs" ] && ! printf '%s' "$envs" | grep -qE '^([A-Za-z_][A-Za-z0-9_]*=[^ ]* *)+$'; then
    echo "RUNQ skip $j: line 3 must be KEY=VAL KEY=VAL (values without spaces)" >&2
    mv "$j" "$VICTORY_QUEUE_DIR/done/" 2>/dev/null || rm -f "$j"
    continue
  fi

  # Measurements need a quiet machine: the same exclusive lock build.sh takes
  # means no build starts during a run, and a run waits for a build in flight.
  # Take it before the gate lock so a long build cannot block a gate test.
  exec 9>"$VICTORY_BUILD_LOCK"
  if ! flock -w "$VICTORY_BUILD_WAIT" 9; then
    echo "RUNQ waiting: a build holds $VICTORY_BUILD_LOCK, retrying" >&2
    exec 9>&-
    sleep "$VICTORY_QUEUE_POLL"
    continue
  fi
  # Taking the lock here serialises with the gate scripts and any other queue.
  exec 8>"$VICTORY_GATE_LOCK"
  if ! flock -w 3600 8; then
    echo "RUNQ waiting: GPU gate held for over an hour, retrying" >&2
    flock -u 9
    exec 9>&-
    exec 8>&-
    continue
  fi
  mv "$j" "$VICTORY_QUEUE_DIR/done/"

  echo "RUNQ start $tag exe=$exe sha=$(sha256sum "$exe" 2>/dev/null | cut -c1-12) env=[$envs]"
  # envs is split on purpose: it is validated as KEY=VAL words above.
  # shellcheck disable=SC2086
  env $envs VICTORY_EXE="$exe" VICTORY_CAPTURE_FRAMES="${VICTORY_CAPTURE_FRAMES:-1500,1700}" \
    "$S/capped.sh" "$S/run-game.sh" "$tag" "$dur" | grep -E '^(RESULT|RUN )'
  if [ -f "$VICTORY_RUNS_DIR/$tag/log.txt" ]; then
    "$S/runstat.sh" "$tag" | sed 's/^/RUNQ done /'
  else
    echo "RUNQ done $tag: no log (run did not start; check VICTORY_GAME_DIR/VICTORY_EXE)"
  fi

  flock -u 8
  exec 8>&-
  flock -u 9
  exec 9>&-
done
