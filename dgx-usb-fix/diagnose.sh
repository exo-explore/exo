#!/usr/bin/env bash
set -Eeuo pipefail

section() { printf '\n== %s ==\n' "$*"; }

section "Kernel"
uname -a
dpkg-query -W -f='${Package} ${Version} ${source:Package} ${source:Version}\n' \
  "linux-image-$(uname -r)" 2>/dev/null || true
test -d "/lib/modules/$(uname -r)/build" && echo "headers: present" || echo "headers: missing"

section "Secure Boot"
mokutil --sb-state 2>/dev/null || true
if [[ $EUID -eq 0 ]]; then
  ls -l /root/MOK.priv /root/MOK.der 2>/dev/null || true
else
  echo "MOK key file visibility requires root"
fi

section "cdc_ncm Module"
modinfo -n cdc_ncm 2>/dev/null || true
modinfo cdc_ncm 2>/dev/null | grep -E '^(filename|version|description|alias|depends|intree|name|vermagic|signer|dgx_usb_fix):' || true
lsmod | grep -E '^(cdc_ncm|cdc_mbim|cdc_ether|usbnet)' || true

section "Apple USB Device"
lsusb | grep -i '05ac:1905\|Apple' || true
lsusb -t || true

section "Kernel Log"
journalctl -k --no-pager | grep -E 'cdc_ncm|05ac|1905|bind\(\) failure' | tail -n 80 || true
