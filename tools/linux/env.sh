#!/bin/sh
# Cross-compile environment for the Windows target x86_64-pc-windows-msvc on a
# Linux host, using the MSVC CRT/SDK that `xwin splat` unpacks. Source it, do
# not run it, from the repository root:
#
#   . tools/linux/env.sh
#   cargo build --target x86_64-pc-windows-msvc --profile diag -p victory --features diagnostics
#
# Inputs (export before sourcing to override):
#   XWIN_DIR    xwin splat output ($HOME/.xwin). Make it once with:
#               xwin --accept-license splat --output "$HOME/.xwin"
#   WINEPREFIX  prefix for `cargo test`/`cargo run` under Wine
#               ($PWD/target/wine-prefix, i.e. relative to where you source it)
#   WINE        Wine binary used as the cargo runner (wine)
# Needs clang-cl, llvm-lib and lld-link on PATH (LLVM >= 18).
#
# Example: . tools/linux/env.sh && cargo test --target x86_64-pc-windows-msvc -p victory-gpu

X=${XWIN_DIR:-$HOME/.xwin}
if [ ! -d "$X/crt" ] || [ ! -d "$X/sdk" ]; then
  echo "env.sh: no xwin splat at $X (xwin --accept-license splat --output $X)" >&2
fi

export CC_x86_64_pc_windows_msvc=clang-cl
export CXX_x86_64_pc_windows_msvc=clang-cl
export AR_x86_64_pc_windows_msvc=llvm-lib
export CFLAGS_x86_64_pc_windows_msvc="--target=x86_64-pc-windows-msvc -Wno-unused-command-line-argument /imsvc$X/crt/include /imsvc$X/sdk/include/ucrt /imsvc$X/sdk/include/um /imsvc$X/sdk/include/shared"
export CXXFLAGS_x86_64_pc_windows_msvc="$CFLAGS_x86_64_pc_windows_msvc"
export CARGO_TARGET_X86_64_PC_WINDOWS_MSVC_LINKER=lld-link
export CARGO_TARGET_X86_64_PC_WINDOWS_MSVC_RUSTFLAGS="-Lnative=$X/crt/lib/x86_64 -Lnative=$X/sdk/lib/um/x86_64 -Lnative=$X/sdk/lib/ucrt/x86_64"
export CARGO_TARGET_X86_64_PC_WINDOWS_MSVC_RUNNER=${WINE:-wine}
export WINEPREFIX=${WINEPREFIX:-$PWD/target/wine-prefix}
export WINEDEBUG=${WINEDEBUG:--all}
