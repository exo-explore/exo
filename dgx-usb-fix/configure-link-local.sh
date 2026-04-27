#!/usr/bin/env bash
set -Eeuo pipefail

DRY_RUN=0
BRING_UP=1
INTERFACES=()

die() { printf 'ERROR: %s\n' "$*" >&2; exit 1; }
log() { printf '\n==> %s\n' "$*"; }
need() { command -v "$1" >/dev/null 2>&1 || die "Missing command: $1. Run from: nix develop .#dgx-usb-fix"; }

usage() {
  cat <<'EOF'
Usage: dgx-usb-fix-configure-link-local [options]

Configure NetworkManager so Apple 05ac:1905 CDC-NCM interfaces stay usable with
IPv6 link-local addresses even when DHCP/RA/global address assignment is absent.

Options:
  --iface IFACE   Configure a specific interface instead of auto-detecting.
                 May be passed more than once.
  --no-up         Modify profiles but do not bring the connections up.
  --dry-run       Print the changes without applying them.
  -h, --help      Show this help.

Expected effect:
  - ipv4.method disabled
  - ipv6.method link-local
  - ipv6.addr-gen-mode stable-privacy
  - connection.autoconnect yes

Build tools are provided by the repo dev shell:

  nix develop .#dgx-usb-fix
  dgx-usb-fix-configure-link-local
EOF
}

while (($#)); do
  case "$1" in
    --iface)
      shift
      [[ $# -gt 0 ]] || die "--iface requires an interface name"
      INTERFACES+=("$1")
      ;;
    --no-up)
      BRING_UP=0
      ;;
    --dry-run)
      DRY_RUN=1
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      die "Unknown argument: $1"
      ;;
  esac
  shift
done

[[ $EUID -eq 0 ]] || die "Run as root, preferably through: dgx-usb-fix-configure-link-local"
need ip
need nmcli
need udevadm

run() {
  printf '+'
  printf ' %q' "$@"
  printf '\n'
  if ((DRY_RUN == 0)); then
    "$@"
  fi
}

is_target_iface() {
  local iface="$1" props

  [[ -e /sys/class/net/$iface ]] || return 1
  props="$(udevadm info -q property -p "/sys/class/net/${iface}" 2>/dev/null || true)"

  grep -q '^ID_NET_DRIVER=cdc_ncm$' <<<"$props" || return 1
  grep -q '^ID_USB_VENDOR_ID=05ac$' <<<"$props" || return 1
  grep -q '^ID_USB_MODEL_ID=1905$' <<<"$props" || return 1
}

detect_interfaces() {
  local path iface

  for path in /sys/class/net/*; do
    iface="${path##*/}"
    if is_target_iface "$iface"; then
      INTERFACES+=("$iface")
    fi
  done
}

connection_for_iface() {
  local iface="$1" conn uuid type candidate_ifname

  conn="$(nmcli -g GENERAL.CONNECTION device show "$iface" 2>/dev/null | head -n1 || true)"
  if [[ -n $conn && $conn != "--" ]]; then
    printf '%s\n' "$conn"
    return 0
  fi

  while IFS=: read -r uuid type; do
    [[ $type == "802-3-ethernet" ]] || continue
    candidate_ifname="$(nmcli -g connection.interface-name connection show uuid "$uuid" 2>/dev/null || true)"
    if [[ $candidate_ifname == "$iface" ]]; then
      nmcli -g connection.id connection show uuid "$uuid"
      return 0
    fi
  done < <(nmcli -t -f UUID,TYPE connection show)

  return 1
}

configure_iface() {
  local iface="$1" conn

  is_target_iface "$iface" || die "$iface is not an Apple 05ac:1905 cdc_ncm interface"

  if conn="$(connection_for_iface "$iface")"; then
    log "Using NetworkManager profile '${conn}' for ${iface}"
  else
    conn="DGX USB link-local ${iface}"
    log "Creating NetworkManager profile '${conn}' for ${iface}"
    run nmcli connection add type ethernet ifname "$iface" con-name "$conn"
  fi

  run nmcli connection modify "$conn" \
    connection.interface-name "$iface" \
    connection.autoconnect yes \
    connection.autoconnect-priority 100 \
    ipv4.method disabled \
    ipv6.method link-local \
    ipv6.may-fail no \
    ipv6.addr-gen-mode stable-privacy

  if ((BRING_UP != 0)); then
    run nmcli connection up "$conn"
  fi

  if ((DRY_RUN == 0)); then
    ip -br addr show "$iface"
  fi
}

if ((${#INTERFACES[@]} == 0)); then
  detect_interfaces
fi

((${#INTERFACES[@]} > 0)) || die "No Apple 05ac:1905 cdc_ncm interfaces found"

for iface in "${INTERFACES[@]}"; do
  configure_iface "$iface"
done

cat <<'EOF'

Done.

Expected checks:
  ip -br addr show
  ping -6 -I <spark-interface> ff02::1

EOF
