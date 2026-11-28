#!/bin/bash
# Run one command in a user scope with a hard RAM cap and no swap, so an
# out-of-memory event kills only that command and never the desktop. Wrap a
# game run or a test that loads real data:
#
#   tools/linux/capped.sh tools/linux/run-game.sh <tag> 420
#
# queue.sh already wraps its runs in this. A command that dies with SIGKILL
# and no message was most likely killed here.
# Inputs:
#   VICTORY_MEM_MAX   memory cap (34G; a run in the world needs ~33 GB)
#   VICTORY_SWAP_MAX  swap cap (0)
# Needs a user systemd (systemd-run --user). Without one, say so and use the
# desktop's own limits instead; failing loudly beats measuring uncapped.
set -u
if [ $# -eq 0 ]; then
  echo "usage: $0 <command> [args...]" >&2
  exit 2
fi
if ! command -v systemd-run >/dev/null 2>&1; then
  echo "$0: systemd-run not found; cannot cap memory without a user systemd" >&2
  exit 1
fi
: "${VICTORY_MEM_MAX:=34G}"
: "${VICTORY_SWAP_MAX:=0}"
exec systemd-run --user --scope --quiet -p "MemoryMax=$VICTORY_MEM_MAX" -p "MemorySwapMax=$VICTORY_SWAP_MAX" "$@"
