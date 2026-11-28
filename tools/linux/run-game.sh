#!/bin/bash
# One game run: a fresh output folder, a private Wine server, headless
# gamescope at 1920x1080, a hard time limit, an Xid watch that stops the run
# if the driver resets, and a VRAM log next to it.
#
# Usage: tools/linux/run-game.sh <tag> <seconds>
# Example: VICTORY_GAME_DIR=/games/gta6 VICTORY_EXE=$PWD/target/.../victory.exe \
#          WINEPREFIX=$HOME/victory-prefix tools/linux/run-game.sh ab-main 420
#
# Measurements go through queue.sh, which adds the GPU gate and the RAM cap;
# run this directly only while nothing else uses the GPU.
#
# Inputs (required):
#   VICTORY_GAME_DIR   folder holding the game files
#   VICTORY_EXE        the victory.exe to run (keep every build at its own path,
#                      so a log names exactly the binary that produced it)
# Env (defaults):
#   VICTORY_RUNS_DIR        $HOME/victory-runs; the run goes to <dir>/<tag>/
#   VICTORY_GAME_EXE        game_scarlett_master.exe
#   VICTORY_GAMESCOPE_RES   1920x1080
#   VICTORY_VK_DEVICE       vendor:device for --prefer-vk-device, e.g.
#                           10de:2b85; unset or empty = the loader's default
#   VICTORY_VK_DRIVER_FILES override the loader's Vulkan ICD choice (e.g.
#                           /usr/share/vulkan/icd.d/nvidia_icd.json)
#   VICTORY_GPU_NAME        DXVK_FILTER_DEVICE_NAME, e.g. "RTX 5090"
#   VICTORY_WINESERVER      wineserver binary (wineserver)
#   VICTORY_KEYS            button presses: frame:key,... (Enter every 8 s
#                           from 20 to 900)
#   VICTORY_CAPTURE_FRAMES  frames to capture (300,600,...,60000)
#   VICTORY_CAPTURE         overrides the frame capture folder
#   VICTORY_WINEDEBUG       err+vulkan,err+virtual,err+seh
#   VICTORY_SHADER_LOOP_LIMIT / VICTORY_SKIP_PIPELINES  see docs/linux.md
#   XID_GRACE               seconds between an Xid and stopping the run (15)
#   VICTORY_VRAM_INTERVAL   nvidia-smi sample interval (5)
#
# Output: <tag>/log.txt, <tag>/frames/, <tag>/vram.csv, <tag>/xid.txt,
# <tag>/run.pid, <tag>/watch.pid, <tag>/server.txt.
set -u
tag=${1:?usage: $0 <tag> <seconds>}
secs=${2:?usage: $0 <tag> <seconds>}
case $tag in
  '' | . | */* | *..*)
    echo "$0: bad tag '$tag' (use a plain name: ab-main, 1002b-q)" >&2
    exit 2
    ;;
esac

: "${VICTORY_RUNS_DIR:=$HOME/victory-runs}"
: "${VICTORY_GAME_EXE:=game_scarlett_master.exe}"
: "${VICTORY_GAMESCOPE_RES:=1920x1080}"
: "${VICTORY_WINESERVER:=wineserver}"
: "${VICTORY_WINEDEBUG:=err+vulkan,err+virtual,err+seh}"
: "${VICTORY_SHADER_LOOP_LIMIT:=1048576}"
: "${VICTORY_SKIP_PIPELINES:=04b9be3756e8f41c}"
: "${VICTORY_VRAM_INTERVAL:=5}"
: "${XID_GRACE:=15}"
: "${VICTORY_GAME_DIR:?set VICTORY_GAME_DIR to the game folder}"
: "${VICTORY_EXE:?set VICTORY_EXE to the victory.exe to run}"

# vkd3d-proton's pipeline cache must stay on: an inherited
# VKD3D_SHADER_CACHE_PATH=0 (from offline experiments) makes every pipeline a
# driver compile (~250 ms each, ~5 min of backlog per run).
unset VKD3D_SHADER_CACHE_PATH

# Press Enter on the first-start screens and the start screen; the world
# loads about a minute after launch.
K=""
for t in $(seq 20 8 900); do K+="$t:Enter,"; done
K=${K%,}
export VICTORY_KEYS=${VICTORY_KEYS:-$K}
export VICTORY_CAPTURE_FRAMES=${VICTORY_CAPTURE_FRAMES:-$(seq -s, 300 300 60000)}
export VICTORY_SHADER_LOOP_LIMIT VICTORY_SKIP_PIPELINES
export VICTORY_NO_WATCHDOG=1
export WINEDEBUG=$VICTORY_WINEDEBUG

OUT=$VICTORY_RUNS_DIR/$tag
rm -rf -- "$OUT"
mkdir -p "$OUT/frames"
if [ -z "${VICTORY_CAPTURE:-}" ]; then
  # Victory expects a Windows path for the capture folder; Wine maps / to Z:.
  export VICTORY_CAPTURE="Z:$(printf '%s' "$OUT/frames" | tr / '\\')"
fi
RES=$VICTORY_GAMESCOPE_RES
W=${RES%x*}
H=${RES#*x}
[ -n "${VICTORY_VK_DRIVER_FILES:-}" ] && export VK_DRIVER_FILES=$VICTORY_VK_DRIVER_FILES
[ -n "${VICTORY_GPU_NAME:-}" ] && export DXVK_FILTER_DEVICE_NAME=$VICTORY_GPU_NAME
DEV=${VICTORY_VK_DEVICE-}
DEVARG=()
[ -n "$DEV" ] && DEVARG=(--prefer-vk-device "$DEV")

run=""
watch=""
vram=""
cleanup() {
  for p in "$watch" "$vram" "$run"; do
    [ -n "$p" ] && kill -TERM -- "-$p" 2>/dev/null
  done
}
trap cleanup EXIT INT TERM

echo "RUN $tag: exe=$VICTORY_EXE sha=$(sha256sum -- "$VICTORY_EXE" 2>/dev/null | cut -c1-12) game=$VICTORY_GAME_DIR/$VICTORY_GAME_EXE ${secs}s -> $OUT"

cd "$OUT" || exit 1

if command -v "$VICTORY_WINESERVER" >/dev/null 2>&1; then
  "$VICTORY_WINESERVER" -k >/dev/null 2>&1
  sleep 1
  "$VICTORY_WINESERVER" -p10 >/dev/null 2>&1
  echo "wineserver: $VICTORY_WINESERVER" >"$OUT/server.txt"
fi

# Xid watch: if the kernel logs a GPU reset, give the driver XID_GRACE seconds
# and stop this run's gamescope (only its gamescope, by PID) so the desktop
# comes back. The journalctl pipeline runs in the background and the watcher
# polls its output file: waiting for the pipeline would delay the kill until
# the run's time limit, because journalctl at the pipe's head is not killed by
# grep exiting. A stale waiter is worse than none; the group is killed below.
setsid bash -c '
  echo $$ >"$1"
  timeout --foreground "$2" journalctl -kf -n0 --no-pager -o cat |
    grep --line-buffered -m1 Xid >"$3" &
  jl=$!
  while [ ! -s "$3" ] && kill -0 "$jl" 2>/dev/null; do sleep 0.2; done
  if [ -s "$3" ]; then
    sleep "$4"
    # run.pid is written a moment after this watcher starts; look at it only
    # after the grace sleep, and give up quietly if the run never started.
    [ -s "$5" ] || exit 0
    gs=$(pgrep -P "$(cat "$5")" -x gamescope)
    [ -n "$gs" ] && kill -INT "$gs"
  fi
  kill "$jl" 2>/dev/null
' _ "$OUT/watch.pid" "$secs" "$OUT/xid.txt" "$XID_GRACE" "$OUT/run.pid" &
watch=$!
for _ in $(seq 1 50); do [ -s "$OUT/watch.pid" ] && break; sleep 0.1; done
watch=$(cat "$OUT/watch.pid" 2>/dev/null || echo "$watch")

# VRAM every VICTORY_VRAM_INTERVAL seconds, next to the log (NVIDIA only;
# use radeontop or amdgpu's mem_info on AMD instead).
if command -v nvidia-smi >/dev/null 2>&1; then
  setsid bash -c 'echo $$ >"$1"; exec timeout --foreground "$2" nvidia-smi --query-gpu=timestamp,memory.used,memory.total,utilization.gpu --format=csv,noheader -l "$3"' \
    _ "$OUT/vram.pid" "$secs" "$VICTORY_VRAM_INTERVAL" >"$OUT/vram.csv" 2>/dev/null &
  vram=$!
  for _ in $(seq 1 50); do [ -s "$OUT/vram.pid" ] && break; sleep 0.1; done
  vram=$(cat "$OUT/vram.pid" 2>/dev/null || echo "$vram")
fi

# The game itself, in its own session so a kill reaches gamescope and Wine.
setsid bash -c 'echo $$ >"$1"; shift; exec timeout --foreground -s INT "$@"' _ "$OUT/run.pid" "$secs" \
  gamescope --backend headless -W "$W" -H "$H" -w "$W" -h "$H" "${DEVARG[@]}" --force-windows-fullscreen -- \
  sh -c 'wine "$0" run "$1" --execute; echo "wine-exit=$?"' "$VICTORY_EXE" "$VICTORY_GAME_DIR\\$VICTORY_GAME_EXE" \
  >"$OUT/log.txt" 2>&1 &
launcher=$!
for _ in $(seq 1 100); do [ -s "$OUT/run.pid" ] && break; sleep 0.1; done
run=$(cat "$OUT/run.pid" 2>/dev/null || echo "")
wait "$launcher" 2>/dev/null
if [ -n "$run" ]; then
  while kill -0 "$run" 2>/dev/null; do sleep 1; done
fi

cleanup
trap - EXIT INT TERM
"$VICTORY_WINESERVER" -k >/dev/null 2>&1
sleep 2

peak=$(awk -F, '{gsub(/[^0-9]/, "", $2); if ($2 + 0 > m) m = $2 + 0} END {print m + 0}' "$OUT/vram.csv" 2>/dev/null)
line=$(
  printf 'RESULT %s: xid=%s %s lastframe=%s vram_peak=%s' "$tag" \
    "$(grep -c Xid "$OUT/xid.txt" 2>/dev/null || true)" \
    "$(grep -o 'wine-exit=[0-9]*' "$OUT/log.txt" 2>/dev/null | tail -n1)" \
    "$(grep -E 'first bind' "$OUT/log.txt" 2>/dev/null | tail -n1 | grep -oE 'frame=[0-9]+')" \
    "${peak:-n/a}"
)
echo "$line"
echo "$line" >"$OUT/result.txt"
