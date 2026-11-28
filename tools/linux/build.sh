#!/bin/bash
# One build at a time machine-wide, niced and with a fixed job count, so a
# build never lands in the middle of someone else's measurement. Use it for
# every build, census and big test suite on a shared machine:
#
#   tools/linux/build.sh cargo build --profile diag -p victory --features diagnostics
#   tools/linux/build.sh cargo test -p victory-gpu --features diagnostics
#
# Do not pass -j; the job count is below. Drop `cargo` for other tools
# (`tools/linux/build.sh ninja ...`). The lock is held only while the command runs.
# Inputs:
#   VICTORY_BUILD_LOCK  lock file ($VICTORY_RUNS_DIR/build.lock)
#   VICTORY_BUILD_JOBS  cargo jobs (4); also exported as CARGO_BUILD_JOBS
#   VICTORY_NICE        niceness 19 (measurements assume nothing heavy runs)
set -u
if [ $# -eq 0 ]; then
  echo "usage: $0 <command> [args...]   e.g. $0 cargo build --profile diag -p victory --features diagnostics" >&2
  exit 2
fi
RUNS=${VICTORY_RUNS_DIR:-$HOME/victory-runs}
: "${VICTORY_BUILD_LOCK:=$RUNS/build.lock}"
: "${VICTORY_BUILD_JOBS:=4}"
: "${VICTORY_NICE:=19}"
mkdir -p "$(dirname "$VICTORY_BUILD_LOCK")"
export CARGO_BUILD_JOBS=$VICTORY_BUILD_JOBS
exec flock "$VICTORY_BUILD_LOCK" nice -n "$VICTORY_NICE" ionice -c3 "$@"
