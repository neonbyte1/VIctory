#!/bin/bash
# RAM watchdog for a shared machine: when used RAM goes above a high mark,
# pause our own builds (cargo/rustc/rust-lld/lld-link/clang-cl and
# `victory.exe translate`) with SIGSTOP; resume them below the low mark. It
# never touches the game, the desktop or other users' processes.
#
# Start it once for the session, next to the queue:
#   tools/linux/ramguard.sh
# Stop it with Ctrl+C (builds keep their state; stopped ones are resumed
# before exit).
#
# Inputs:
#   VICTORY_RAM_PAUSE_GB   pause above this (53)
#   VICTORY_RAM_RESUME_GB  resume below this (48)
#   VICTORY_RAM_INTERVAL   sample interval in seconds (3)
#   VICTORY_RAM_PATTERN    pgrep -f pattern of processes it may pause
set -u
: "${VICTORY_RAM_PAUSE_GB:=53}"
: "${VICTORY_RAM_RESUME_GB:=48}"
: "${VICTORY_RAM_INTERVAL:=3}"
: "${VICTORY_RAM_PATTERN:=cargo |rustc|rust-lld|lld-link|clang-cl|victory.exe translate}"

paused=0
resume() {
  [ "$paused" -eq 1 ] || return 0
  # shellcheck disable=SC2046
  kill -CONT $(pgrep -f "$VICTORY_RAM_PATTERN" 2>/dev/null) 2>/dev/null
  paused=0
}
trap 'resume; echo "$(date +%T) ramguard stopped"; exit 0' INT TERM

echo "$(date +%T) ramguard: pause above ${VICTORY_RAM_PAUSE_GB}G, resume below ${VICTORY_RAM_RESUME_GB}G, every ${VICTORY_RAM_INTERVAL}s"
while :; do
  avail=$(awk '/MemAvailable/{print $2}' /proc/meminfo)
  total=$(awk '/MemTotal/{print $2}' /proc/meminfo)
  used_gb=$(((total - avail) / 1048576))
  pids=$(pgrep -f "$VICTORY_RAM_PATTERN" 2>/dev/null | tr '\n' ' ')
  if [ "$used_gb" -ge "$VICTORY_RAM_PAUSE_GB" ] && [ -n "$pids" ] && [ "$paused" -eq 0 ]; then
    # shellcheck disable=SC2086
    kill -STOP $pids 2>/dev/null
    paused=1
    echo "$(date +%T) used=${used_gb}G paused builds: $pids"
  elif [ "$used_gb" -lt "$VICTORY_RAM_RESUME_GB" ] && [ "$paused" -eq 1 ]; then
    resume
    echo "$(date +%T) used=${used_gb}G resumed builds"
  fi
  sleep "$VICTORY_RAM_INTERVAL"
done
