#!/bin/bash
# Print the vendor:device id of the GPU to prefer (for gamescope
# --prefer-vk-device): the first NVIDIA GPU lspci finds, else the first display
# device, else nothing. hidden.sh and run-game.sh use it when VICTORY_VK_DEVICE
# is not set; an empty VICTORY_VK_DEVICE means "no preference".
command -v lspci >/dev/null 2>&1 || exit 0
ids=$(lspci -nn 2>/dev/null |
  grep -E 'VGA compatible controller|3D controller|Display controller' |
  sed -nE 's/.*\[([0-9a-fA-F]{4}:[0-9a-fA-F]{4})\].*/\1/p')
[ -n "$ids" ] || exit 0
nv=$(printf '%s\n' "$ids" | grep -i -m1 '^10de:')
printf '%s' "${nv:-$(printf '%s\n' "$ids" | head -n1)}"
