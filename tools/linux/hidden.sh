#!/bin/bash
# Run one command inside a fresh headless gamescope: nothing appears on the
# desktop, but Wine gets a display and DXGI/D3D12 work (what GPU device tests
# and game runs need). Wine crash dialogs are disabled.
#
# gamescope runs in its own process group and the group is killed when the
# command ends, so no gamescope, Wine or gamescope's Vulkan device outlives
# the call and keeps the GPU gate.
#
# Usage: tools/linux/hidden.sh <command> [args...]
# Example: tools/linux/hidden.sh wine target/x86_64-pc-windows-msvc/debug/victory-gpu-*.exe --test-threads=1
#
# This does NOT take the GPU gate by itself. Anything that touches the GPU
# must run under a gate script (tools/linux/gate-template.sh); the queue holds
# the same lock for a whole game run.
# Inputs:
#   VICTORY_VK_DEVICE     vendor:device for gamescope --prefer-vk-device, e.g.
#                         10de:2b85. Default: the first NVIDIA GPU lspci finds,
#                         else the first display device. Empty = no preference.
#   VICTORY_HEADLESS_RES  gamescope resolution (1280x720)
#   WINEDEBUG             default -all
#   WINEDLLOVERRIDES      winedbg.exe=d is appended, so crashes cannot open a
#                         dialog that would wait forever for input.
set -u
if [ $# -eq 0 ]; then
  echo "usage: $0 <command> [args...]" >&2
  exit 2
fi

DEV=${VICTORY_VK_DEVICE-$("$(dirname "$(readlink -f "$0")")/vk-device.sh")}
RES=${VICTORY_HEADLESS_RES:-1280x720}
W=${RES%x*}
H=${RES#*x}
rc=$(mktemp)

setsid env -u DISPLAY -u WAYLAND_DISPLAY WINEDEBUG="${WINEDEBUG:--all}" \
  WINEDLLOVERRIDES="${WINEDLLOVERRIDES:+$WINEDLLOVERRIDES;}winedbg.exe=d" \
  gamescope --backend headless -W "$W" -H "$H" -w "$W" -h "$H" ${DEV:+--prefer-vk-device "$DEV"} -- \
  sh -c '"$@"; echo $? > "$0"' "$rc" "$@" > >(grep --line-buffered -vE '^\[gamescope\]|^wlserver|^xwm|^vulkan:|^\(EE\) failed to read Wayland|^No CAP_SYS_NICE|^Performance will be affected|^X connection to|^XIO:|after [0-9]+ requests') 2>&1 &
gs=$!
trap 'kill -TERM -- -$gs 2>/dev/null' INT TERM
# The command's exit status is written to $rc; poll it so a command that
# daemonises still cannot hold gamescope open.
while kill -0 "$gs" 2>/dev/null && [ ! -s "$rc" ]; do sleep 0.5; done
sleep 1; kill -TERM -- "-$gs" 2>/dev/null; sleep 2; kill -KILL -- "-$gs" 2>/dev/null
wait "$gs" 2>/dev/null
code=$(cat "$rc" 2>/dev/null); rm -f "$rc"; exit "${code:-1}"
