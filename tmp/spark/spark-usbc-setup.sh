#!/usr/bin/env bash
set -Eeuo pipefail

PATCHED_CDC_NCM="/root/apple1905-cdcncm/linux-nvidia-6.14-6.14.0/buildmod/cdc_ncm.ko"

need() {
  command -v "$1" >/dev/null 2>&1 || {
    echo "Missing command: $1" >&2
    exit 1
  }
}

msg() {
  printf '\n==> %s\n' "$*"
}

[[ $EUID -eq 0 ]] || {
  echo "Run with sudo." >&2
  exit 1
}

need modprobe
need insmod
need nmcli
need lsusb
need ip
need awk
need grep

[[ -f $PATCHED_CDC_NCM ]] || {
  echo "Patched module not found: $PATCHED_CDC_NCM" >&2
  exit 1
}

msg "Removing old hard-block config if present"
rm -f /etc/modprobe.d/zz-kill-mac-usb-net.conf
depmod -a

msg "Loading USB4 / Type-C / Thunderbolt support"
modprobe typec || true
modprobe typec_ucsi || true
modprobe ucsi_acpi || true
modprobe thunderbolt || true
modprobe typec_thunderbolt || true
modprobe thunderbolt_net || true

msg "Unloading stock USB network stack"
modprobe -r cdc_mbim cdc_wdm cdc_ncm cdc_ether usbnet 2>/dev/null || true

msg "Loading dependency modules"
modprobe usbnet
modprobe cdc_ether

msg "Loading patched cdc_ncm"
if lsmod | grep -q '^cdc_ncm '; then
  echo "cdc_ncm already loaded"
else
  insmod "$PATCHED_CDC_NCM"
fi

msg "Current module state"
lsmod | grep -E 'typec|ucsi|thunderbolt|cdc_ncm|cdc_ether|usbnet' || true

cat <<'EOF'

Now do this on the Mac:
  1. Run the Mac script below
  2. Replug the cable
  3. Wait for the Mac to land on the fast bus

Then run this same script again with:
  sudo ./spark-usbc-setup.sh finalize

EOF

if [[ ${1:-} != "finalize" ]]; then
  exit 0
fi

msg "Waiting for Apple Mac on fast bus (Bus 004 @ 10000M or equivalent)"
for _ in $(seq 1 60); do
  if lsusb -t | grep -q '05ac:1905\|Driver=\[none\], 10000M'; then
    break
  fi
  sleep 1
done

msg "lsusb -t snapshot"
lsusb -t
echo
lsusb | grep '05ac:1905' || true

msg "Waiting for new Apple USB NICs"
for _ in $(seq 1 30); do
  mapfile -t APPLE_IFS < <(
    ip -o link | awk -F': ' '{print $2}' |
      grep '^enx36be1bab12' || true
  )
  if [[ ${#APPLE_IFS[@]} -ge 2 ]]; then
    break
  fi
  sleep 1
done

if [[ ${#APPLE_IFS[@]} -lt 2 ]]; then
  echo "Did not find two Apple USB interfaces." >&2
  ip -br link
  exit 1
fi

IF0="${APPLE_IFS[0]}"
IF1="${APPLE_IFS[1]}"

msg "Found interfaces: $IF0 and $IF1"

msg "Resetting old NetworkManager profiles"
nmcli connection delete mac-usb-dhcp-0 2>/dev/null || true
nmcli connection delete mac-usb-dhcp-1 2>/dev/null || true

msg "Bringing up shared DHCP on both interfaces"
nmcli connection add type ethernet ifname "$IF0" con-name mac-usb-dhcp-0
nmcli connection modify mac-usb-dhcp-0 \
  ipv4.method shared \
  ipv6.method disabled \
  ipv4.addresses 10.42.0.1/24
nmcli connection up mac-usb-dhcp-0

nmcli connection add type ethernet ifname "$IF1" con-name mac-usb-dhcp-1
nmcli connection modify mac-usb-dhcp-1 \
  ipv4.method shared \
  ipv6.method disabled \
  ipv4.addresses 10.43.0.1/24
nmcli connection up mac-usb-dhcp-1

msg "Current addresses"
ip -br addr show dev "$IF0"
ip -br addr show dev "$IF1"

msg "Recent NetworkManager log"
journalctl -u NetworkManager -n 60 --no-pager || true

cat <<EOF

Done.

DGX shared-DHCP interfaces:
  $IF0 -> 10.42.0.1/24
  $IF1 -> 10.43.0.1/24

Next on the Mac:
  sudo ipconfig set anpi0 DHCP
  sudo ipconfig set anpi1 DHCP
  sudo ipconfig set anpi2 DHCP

EOF
