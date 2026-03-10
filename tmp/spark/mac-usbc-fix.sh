#!/usr/bin/env bash
set -Eeuo pipefail

msg() {
  printf '\n==> %s\n' "$*"
}

msg "Bringing down old Apple USB fallback interfaces"
sudo ifconfig en4 down 2>/dev/null || true
sudo ifconfig en5 down 2>/dev/null || true
sudo ifconfig en6 down 2>/dev/null || true

msg "Bringing down current anpi interfaces"
sudo ifconfig anpi0 down 2>/dev/null || true
sudo ifconfig anpi1 down 2>/dev/null || true
sudo ifconfig anpi2 down 2>/dev/null || true

msg "Stopping likely bad exporter services"
sudo launchctl bootout system/com.apple.usbmuxd 2>/dev/null || true
sudo launchctl bootout system/com.apple.remoted 2>/dev/null || true
launchctl bootout "gui/$(id -u)/com.apple.remoted" 2>/dev/null || true

sudo pkill -x usbmuxd 2>/dev/null || true
sudo pkill -x remoted 2>/dev/null || true

cat <<'EOF'

Now unplug and replug the cable.

After reconnect, press Enter to continue.
EOF

read -r _

msg "Bringing up anpi interfaces and requesting DHCP"
for ifn in anpi0 anpi1 anpi2; do
  sudo ifconfig "$ifn" up 2>/dev/null || true
  sudo ipconfig set "$ifn" DHCP 2>/dev/null || true
done

sleep 3

msg "Resulting interface state"
for ifn in anpi0 anpi1 anpi2; do
  echo
  echo "--- $ifn ---"
  ifconfig "$ifn" 2>/dev/null || true
  ipconfig getifaddr "$ifn" 2>/dev/null || true
done

echo
echo "Use whichever interface got 10.42.0.x or 10.43.0.x"
