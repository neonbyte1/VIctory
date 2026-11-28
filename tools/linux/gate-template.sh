#!/bin/bash
# Device-test gate: runs one GPU test so that it cannot overlap a game run,
# cannot appear on the desktop, and cannot leave Wine or gamescope behind.
#
# What it does
#   1. waits for / takes the GPU gate: `flock $VICTORY_GATE_LOCK`. queue.sh
#      holds exactly this lock for a whole game run, so the flock alone
#      serialises a test with game runs.
#   2. keeps a private Wine prefix and puts the vkd3d-proton d3d12.dll /
#      d3d12core.dll in it (Wine's own D3D12 has no shader model 6.6).
#   3. runs the test exe through hidden.sh: fresh headless gamescope, no
#      window on the desktop, Wine crash dialogs disabled.
#   4. afterwards stops Wine with the *safe* winedbg pattern and shuts down
#      only its own prefix.
#
# Copy it to a name the queue watches, then start it detached. Both matter:
# the name keeps the queue from starting a game while the test waits for the
# lock, and `setsid ... & disown` keeps a tool timeout from killing a waiter
# that has not entered the gate yet (the flock is only held inside the gate,
# so a killed waiter looks like "no test running" to everyone else).
#
#   cp tools/linux/gate-template.sh /tmp/qgpupasses.sh
#   setsid /tmp/qgpupasses.sh <tag> "$exe" -- --test-threads=1 \
#     >/dev/null 2>&1 & disown
#
# Usage: gate-template.sh <tag> <test exe> [extra test args...]
# Example: VICTORY_VKD3D_DIR=$HOME/GE-Proton/files/lib/wine/vkd3d-proton/x86_64-windows \
#          tools/linux/gate-template.sh passes target/x86_64-pc-windows-msvc/debug/victory-gpu.exe
#
# Inputs (required):
#   VICTORY_VKD3D_DIR   folder with vkd3d-proton's x86_64-windows d3d12 DLLs
# Inputs (defaults):
#   VICTORY_RUNS_DIR      $HOME/victory-runs
#   VICTORY_SCRATCH       $VICTORY_RUNS_DIR/gate (prefix, logs, caches)
#   VICTORY_GATE_LOCK     /tmp/victory-gpu-gate.lock (same as queue.sh)
#   VICTORY_GATE_WAIT     7200 s to wait for the gate
#   VICTORY_TEST_TIMEOUT  900 s for the test itself
#   VICTORY_GATE_PREFIX   $VICTORY_SCRATCH/q-prefix. The gate's own Wine prefix;
#                         an exported WINEPREFIX (your game prefix) is ignored, because
#                         the gate ends with `wineserver -k` on its prefix.
#   VICTORY_VK_DRIVER_FILES / VICTORY_GPU_NAME  Vulkan ICD / DXVK filter
#   VICTORY_HIDDEN        tools/linux/hidden.sh (found from this repo if unset)
#   VICTORY_LOG           warn
# A log per tag is written to $VICTORY_SCRATCH/logs/gpu-<tag>.log.
set -u
tag=${1:?usage: $0 <tag> <test exe> [extra test args...]}
exe=${2:?usage: $0 <tag> <test exe> [extra test args...]}
shift 2
[ -f "$exe" ] || { echo "$0: no test exe at $exe" >&2; exit 2; }
exe=$(readlink -f "$exe")

: "${VICTORY_RUNS_DIR:=$HOME/victory-runs}"
: "${VICTORY_SCRATCH:=$VICTORY_RUNS_DIR/gate}"
: "${VICTORY_GATE_LOCK:=/tmp/victory-gpu-gate.lock}"
: "${VICTORY_GATE_WAIT:=7200}"
: "${VICTORY_TEST_TIMEOUT:=900}"
: "${VICTORY_LOG:=warn}"
: "${VICTORY_VKD3D_DIR:?set VICTORY_VKD3D_DIR to the vkd3d-proton x86_64-windows folder}"
: "${VICTORY_GATE_PREFIX:=$VICTORY_SCRATCH/q-prefix}"
WINEPREFIX=$VICTORY_GATE_PREFIX

H=${VICTORY_HIDDEN:-}
if [ -z "$H" ]; then
  if [ -x "$(dirname "$(readlink -f "$0")")/hidden.sh" ]; then
    H=$(dirname "$(readlink -f "$0")")/hidden.sh
  else
    H=$(git -C "$PWD" rev-parse --show-toplevel 2>/dev/null)/tools/linux/hidden.sh
  fi
fi
[ -x "$H" ] || { echo "$0: hidden.sh not found; set VICTORY_HIDDEN" >&2; exit 2; }
export VICTORY_HIDDEN=$H

export VICTORY_SCRATCH VICTORY_TEST_TIMEOUT VICTORY_LOG WINEPREFIX VICTORY_VKD3D_DIR
mkdir -p "$VICTORY_SCRATCH/logs" "$VICTORY_SCRATCH/vkd3d-cache"

echo "gate: waiting for $VICTORY_GATE_LOCK (up to ${VICTORY_GATE_WAIT}s)" >&2
SELF=$(cd "$(dirname "$0")" && pwd)/$(basename "$0")
# The script path is passed on to the shell below on purpose: after the exec, the script's
# own path is what keeps `pgrep -f` (the queue's "a gate is waiting" check)
# matching this process while it waits for the lock.
exec flock -w "$VICTORY_GATE_WAIT" "$VICTORY_GATE_LOCK" bash -c '
set -u
echo "gate: body entered" >&2
tag=$1; exe=$2; shift 2
S=$VICTORY_SCRATCH
export WINEDEBUG=-all WINEDLLOVERRIDES="d3d12,d3d12core=n;winedbg.exe=d" VICTORY_LOG
[ -n "${VICTORY_VK_DRIVER_FILES:-}" ] && export VK_DRIVER_FILES=$VICTORY_VK_DRIVER_FILES
[ -n "${VICTORY_GPU_NAME:-}" ] && export DXVK_FILTER_DEVICE_NAME=$VICTORY_GPU_NAME
echo "gate: tag=$tag exe=$exe prefix=$WINEPREFIX OWNER=${VICTORY_OWNER:-default}" >&2
mkdir -p "$S/logs"
if [ ! -d "$WINEPREFIX/drive_c/windows/system32" ]; then
  timeout 240 wineboot -u >"$S/logs/wineboot.log" 2>&1 ||
    timeout 240 wineboot -i >>"$S/logs/wineboot.log" 2>&1
fi
SYS=$WINEPREFIX/drive_c/windows/system32
mkdir -p "$SYS" "$S/vkd3d-cache"
for dll in d3d12.dll d3d12core.dll; do
  rm -f "$SYS/$dll"
  cp "$VICTORY_VKD3D_DIR/$dll" "$SYS/$dll"
done
export VKD3D_SHADER_CACHE_PATH=$(winepath -w "$S/vkd3d-cache" 2>/dev/null)
LOG=$S/logs/gpu-$tag.log
echo "### $tag $exe $(sha256sum "$exe" | cut -c1-16) OWNER=${VICTORY_OWNER:-default} TRACE=${MARKCOPY_TRACE:-0}" >"$LOG"
(
  cd "$S" || exit 1
  # One test thread at a time: several tests sharing the device collide.
  timeout --foreground "$VICTORY_TEST_TIMEOUT" "$VICTORY_HIDDEN" wine "$exe" --test-threads=1 "$@" >>"$LOG" 2>&1
)
code=$?
echo "exit=$code" >>"$LOG"
# The bracket keeps this pkill from matching its own command line. Never
# pkill a bare "winedbg" or "wine": that would also hit the game run, the
# gate template must only clean up after itself.
pkill -f "winedbg[ ]--auto" >/dev/null 2>&1
# WINEPREFIX here is the gate prefix. Never point this at the game prefix:
# a wineserver -k during a run kills that run.
wineserver -k >/dev/null 2>&1
echo "gate: done $tag exit=$code log=$LOG" >&2
exit "$code"
' "$SELF" "$tag" "$exe" "$@"
