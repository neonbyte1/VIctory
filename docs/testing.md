# Help test VIctory (no coding needed)

VIctory runs the Xbox build of a certain big game on a PC. It is not playable yet, so we need many different PCs to answer three questions: **does it run, how fast, does it crash?** You do not need to know how to code. You need patience, about two hours (most of it waiting), and a terminal you can copy and paste into.

## What you need

- **Your own copy of the game**, dumped legally from your own console. We cannot help with getting one, and we never ask for any game file.
- A **PC** with an NVIDIA or AMD graphics card (GPU) that supports DirectX 12, a CPU from roughly 2015 or later (it must support "AVX2"), and **32 GB of RAM** (more is better; see below). Linux with Wine/Proton works too, but see "Linux" below.
- About **30 GB of free disk space** for the tools, the build and the caches, plus the game itself.
- Windows 10/11 64-bit is the main target.

## 1. Install the tools (once)

A *terminal* is a window where you type commands. On Windows open **PowerShell** (Start menu, type "PowerShell"). Copy a command, right-click in the window to paste it, press Enter.

**Windows**

1. Install Rust (the language the program is written in): `winget install --id Rustlang.Rustup -e`, or download from <https://rustup.rs>.
2. Install **Visual Studio Build Tools** from <https://visualstudio.microsoft.com/downloads/> (scroll to "Tools for Visual Studio"). In the installer tick **"Desktop development with C++"**. This is a few GB; it provides the compiler Rust needs.
3. Install **Git** (`winget install --id Git.Git -e`) or skip it and use "Download ZIP" below.
4. Close PowerShell and open it again so it finds the new tools.

**Linux (Wine/Proton)**: the program itself is a Windows program. You build the Windows version with the cross-compile steps in the README ("Tests on a Linux host, under Wine") and run it in Wine with vkd3d-proton (Wine's own Direct3D 12 is too old). This is harder; if you get stuck, open a "Hardware" issue and tell us your distribution and what failed, that is useful by itself. The summarizer for logs is `.github/scripts/summarize-run.sh` (needs `python3`).

## 2. Get the code

With Git:

```powershell
cd C:\
git clone https://github.com/neonbyte1/VIctory.git vt
cd vt
```

Without Git: on the project page click the green **Code** button, then **Download ZIP**, unzip it to a short folder such as `C:\vt` and `cd C:\vt`. Keep the folder name short; long paths break the build.

## 3. Build the "diagnostics" build

This is a version that writes the log we need. It takes 10-30 minutes the first time.

```powershell
cargo build --profile diag -p victory --features diagnostics
```

The program is then `C:\vt\target\diag\victory.exe`. The first `cargo` command also downloads the exact Rust version this project needs; that is normal. If it fails, copy the last 20 lines of red text into an issue.

## 4. Tell it where your game is, then "translate"

```powershell
copy victory.example.toml victory.toml
notepad victory.toml
```

Set the paths in `victory.toml` to your game folder as the comments in the file explain (keep the game files where they are; do not put them into the VIctory folder). Check it with `target\diag\victory.exe config`.

Now the long one-time step: it converts the game's graphics programs ("shaders") for your GPU and stores them in a cache on your disk (about 1.3 GB, minutes to over an hour). Replace `<game>` with your game folder and `<exe>` with the game's .exe name:

```powershell
target\diag\victory.exe translate --cache --pipelines <game>\shaders.rpf <game>\<exe> <game>\XboxBvhBuilder.dll <game>\amd_fidelityfx_upscaler_xbox.dll
```

You can stop it (Ctrl+C) and run it again later; it keeps what it did.

## 5. Do a measurement

We compare numbers only between runs on **your own** PC, so one result is two kinds of run: first a **warm-up**, then the **measured run**.

**Set the settings** (they last until you close the window). In PowerShell:

```powershell
$env:VICTORY_GPU_OVERLAP = '1'
$env:VICTORY_CLAMP_WINDOW = '1'
$env:VICTORY_LOG = 'info'
$env:VICTORY_EARLY_RELEASE = '1'
$env:VICTORY_EARLY_BARRIERS = '1'
```

In bash (Linux): `export VICTORY_GPU_OVERLAP=1 VICTORY_CLAMP_WINDOW=1 VICTORY_LOG=info VICTORY_EARLY_RELEASE=1 VICTORY_EARLY_BARRIERS=1`

**Warm-up run.** The very first start is slow because the program builds more cache (see Troubleshooting). Start the game once and let it get to the world (below); then close it. Don't use these numbers.

**Measured run.** Start it and save the log to a file. In PowerShell use `cmd` for this, it keeps the file readable:

```powershell
cmd /c "target\diag\victory.exe run <game>\<exe> --execute 2> run1.txt"
```

(bash: `target/diag/victory.exe run <game>/<exe> --execute 2> run1.txt`)

- Press **Enter** on each first-start screen and on the start screen. The world loads about a minute after launch (65-80 seconds); after that touch nothing.
- Leave it running for **420 seconds (7 minutes)** counted from launch, then close the game window. A crash before that is also a result; report it.
- Close other heavy programs (browsers, games) during the run. Don't change anything and don't watch the numbers in the title bar changing while you work on something else on that PC.

The log is `run1.txt` in the folder you were in (the `2>` part). It may contain text from the game's startup; do not post the whole file, only the summary below. Crash details, if any, are also in `crashes.log` in `%LOCALAPPDATA%\victory\data`.

**Get the numbers.** One command turns the log into a table. On Windows:

```powershell
powershell -ExecutionPolicy Bypass -File .github\scripts\summarize-run.ps1 run1.txt
```

On Linux: `.github/scripts/summarize-run.sh run1.txt`

## 6. Report it

Open a new issue at <https://github.com/neonbyte1/VIctory/issues/new/choose> and pick **"Test report (no coding needed)"**. Fill in your hardware (Task Manager → Performance shows GPU, CPU, memory; the GPU driver version is in the GPU's page), what happened, and paste the summarizer's output. If you did a second run with something changed (another GPU driver, another setting), put it in the optional "comparison" box, always next to the first run, ideally A-B-A-B (first, changed, first, changed) so a busy PC can't fool us.

Reference workload: the Xbox **Series X** profile with ray tracing **on** (the game's default here). Other profiles or ray tracing off measure something else; say so if you changed it.

### What the numbers mean

- **Frame (ms)**: how long one game frame takes, from start to finish. Lower is better; 1000 divided by it is the frames per second.
- **Min**: the shortest frame in the run (the best the machine can do).
- **Preparing**: time spent getting the frame's data ready for the GPU (copying textures, buffers) before it can draw.
- **GPU**: time the graphics card is actually busy drawing.
- **Copy-back**: time spent copying results from the GPU back to the CPU memory because the game asks for them.
- **Dropped**: number of draws per frame skipped because the matching graphics program wasn't ready or can't be translated; high numbers mean parts of the picture are missing.
- **VRAM**: video memory in use, in MB. If it is near your card's limit, expect slowdowns or crashes.
- **Shaders refused**: how many graphics programs could not be translated at all on your GPU.

### What NOT to upload

Never post game files, shader dumps, captured frames or screenshots of the game, or the full log file. The summary (numbers, hashes) is enough. You don't need to use your real name. If an AI helped you write the post, say so at the end of it.

## Troubleshooting

- **The first run is very slow and the picture has holes.** Normal: caches are being built. Run the `translate --cache --pipelines` step to completion and do the warm-up run; compare only runs after that.
- **"device removed" in the log, or the screen goes black.** The GPU driver reset itself. Report it with the crash line the summarizer shows and your driver version; try the newest driver once.
- **It stays on the start screen or on black after the legal screens.** Press Enter a few more times on each screen; the world takes a minute to load and the window can look frozen meanwhile. If nothing changes after 3 minutes, report it with what you saw.
- **The program says it ran out of memory (error 1455).** Close other programs, or enlarge the Windows paging file (Settings → System → About → Advanced system settings → Performance → Advanced → Virtual memory), then retry. A run in the world needs about 33 GB RAM + paging file together and about 10 GB of video memory.
- **The summarizer says "no frame lines".** The log has no lines like `... fps, ... ms/title frame`: the game didn't reach the point where they appear, or `VICTORY_LOG` was not set to `info` in that window. Report that too.
