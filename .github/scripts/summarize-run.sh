#!/bin/sh
# Turns a victory log into the table of the test-report template.
# Usage: summarize-run.sh <log.txt>   (needs python3; no other installs)
# Windows: use summarize-run.ps1 instead.
# The asset policy allows no .py files, so the Python lives in this shell script.
exec python3 - "$@" <<'PY'
import re, statistics, sys

if len(sys.argv) != 2:
    sys.exit("usage: summarize-run.sh <log.txt>")
ansi = re.compile(r"\x1b\[[0-9;]*m")
frame = re.compile(
    r"([\d.]+) fps, ([\d.]+) ms/title frame \(([\d.]+) preparing.*?; ([\d.]+) on the GPU; "
    r"([\d.]+) copying back.*?dropped (\d+) without"
)
vram = re.compile(r"video memory: (\d+) MB of a (\d+) MB budget")
gpu = re.compile(r"host GPU name=(.+?) vendor=\S+ driver=\"?([^\" ]+)\"? dedicated_mib=(\d+)")
crash = re.compile(r"device removed|panicked|access violation|unhandled exception|crashes\.log|fatal", re.I)
build = re.compile(r"\b(?:commit|git)[ =:]+([0-9a-f]{7,40})\b|victory (\d+\.\d+\.\d+\S*)", re.I)
frames, vrams, refused, first_crash, gpu_line, build_line = [], [], set(), None, None, None
with open(sys.argv[1], encoding="utf-8", errors="replace") as f:
    for raw in f:
        line = ansi.sub("", raw).rstrip()
        m = frame.search(line)
        if m:
            frames.append([float(x) for x in m.groups()])
            continue
        m = vram.search(line)
        if m:
            vrams.append(int(m.group(1)))
        if "shader does not translate" in line:
            h = re.search(r"hash=(\w+)", line)
            if h:
                refused.add(h.group(1))
        if "victory" in line or "panicked" in line:
            if first_crash is None and crash.search(line) and "pipeline-journal" not in line:
                first_crash = line[:300]
        m = gpu.search(line)
        if m and gpu_line is None:
            gpu_line = "%s, driver %s, %s MiB" % m.groups()
        m = build.search(line)
        if m and build_line is None and "gamescope" not in line:
            build_line = m.group(1) or m.group(2)

if not frames:
    sys.exit("no frame lines ('N fps, X ms/title frame ...') found; was VICTORY_LOG=info set?")
last = frames[-30:]
med = lambda i: statistics.median(r[i] for r in last)
print("Frame lines found: %d (using the last %d)" % (len(frames), len(last)))
print("| Frame ms | Min ms | fps | Preparing ms | GPU ms | Copy-back ms | Dropped | VRAM MB |")
print("|---|---|---|---|---|---|---|---|")
print("| %.1f | %.1f | %.1f | %.1f | %.1f | %.1f | %d | %s |" % (
    med(1), min(r[1] for r in last), med(0), med(2), med(3), med(4), med(5),
    int(statistics.median(vrams[-30:])) if vrams else "n/a"))
print("Shaders refused (unique): %d" % len(refused))
print("Crash/error line: %s" % (first_crash or "none found"))
print("Host GPU: %s" % (gpu_line or "not logged"))
print("Build/commit in log: %s" % (build_line or "not logged"))
PY
