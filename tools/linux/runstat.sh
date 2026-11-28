#!/bin/bash
# One line of statistics per run, for comparing runs at a glance (A-B-A-B).
# The numbers come from .github/scripts/summarize-run.sh (same parser as the
# test-report table); this only resolves run tags to their log files.
#
# Usage: tools/linux/runstat.sh <tag|log.txt> [...]
# Example: tools/linux/runstat.sh 0100-ab-main 0101-ab-switch 0102-ab-main
#
# Inputs:
#   VICTORY_RUNS_DIR  $HOME/victory-runs; a bare tag means <dir>/<tag>/log.txt
#                      (a path to an existing file is used as given)
# Output, per run: tag: n=… frame=… min=… fps=… prep=… gpu=… copyback=…
#                  dropped=… vram=… refused=… faults=…
set -u
if [ $# -eq 0 ]; then
  echo "usage: $0 <tag|log.txt> [...]" >&2
  exit 2
fi
ROOT=$(CDPATH= cd -- "$(dirname -- "$0")/../.." && pwd)
RUNS=${VICTORY_RUNS_DIR:-$HOME/victory-runs}
args=()
for a in "$@"; do
  case $a in
    */* | *.txt)
      args+=("$a")
      ;;
    *)
      args+=("$RUNS/$a/log.txt")
      ;;
  esac
done
exec sh "$ROOT/.github/scripts/summarize-run.sh" --brief "${args[@]}"
