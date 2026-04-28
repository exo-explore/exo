#!/usr/bin/env bash
set -Eeuo pipefail

SELF_IP="169.254.100.1"
PEER_IP="169.254.100.2"
PREFIX="16"
IFACE="enP7s7"
USE_NM="auto"
DRY_RUN=0

usage() {
  cat <<EOF
Usage: sudo $(basename "$0") [options]

Configure a Linux Ethernet interface with a static IPv4 for a host-to-host
link to a Mac peer.

Defaults: this host = ${SELF_IP}/${PREFIX}, peer = ${PEER_IP}, iface = ${IFACE}.

Options:
  --iface IFACE     Default: ${IFACE}
  --self-ip IP      Default: ${SELF_IP}
  --peer-ip IP      For verification ping. Default: ${PEER_IP}
  --prefix N        Default: ${PREFIX}
  --no-nm           Use 'ip addr' directly (transient, no NetworkManager).
  --dry-run         Print actions without applying.
  -h, --help        Show this help.
EOF
}

while (($#)); do
  case "$1" in
    --iface) shift; IFACE="${1:?}" ;;
    --self-ip) shift; SELF_IP="${1:?}" ;;
    --peer-ip) shift; PEER_IP="${1:?}" ;;
    --prefix) shift; PREFIX="${1:?}" ;;
    --no-nm) USE_NM=no ;;
    --dry-run) DRY_RUN=1 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown arg: $1" >&2; usage >&2; exit 1 ;;
  esac
  shift
done

[[ $EUID -eq 0 ]] || { echo "Run as root." >&2; exit 1; }

run() {
  printf '+'; printf ' %q' "$@"; printf '\n'
  ((DRY_RUN)) || "$@"
}

ip link show "$IFACE" >/dev/null 2>&1 || { echo "Interface $IFACE does not exist." >&2; exit 1; }

if [[ "$USE_NM" == "auto" ]]; then
  if command -v nmcli >/dev/null 2>&1 && systemctl is-active --quiet NetworkManager 2>/dev/null; then
    USE_NM=yes
  else
    USE_NM=no
  fi
fi

if [[ "$USE_NM" == "yes" ]]; then
  CONN="$(nmcli -g GENERAL.CONNECTION device show "$IFACE" 2>/dev/null | head -n1 || true)"
  if [[ -z "$CONN" || "$CONN" == "--" ]]; then
    CONN="static-${IFACE}"
    run nmcli connection add type ethernet ifname "$IFACE" con-name "$CONN"
  fi
  run nmcli connection modify "$CONN" \
    connection.interface-name "$IFACE" \
    connection.autoconnect yes \
    connection.autoconnect-priority 100 \
    ipv4.method manual \
    ipv4.addresses "${SELF_IP}/${PREFIX}" \
    ipv4.gateway "" \
    ipv4.dns "" \
    ipv4.never-default yes \
    ipv6.method link-local \
    ipv6.addr-gen-mode stable-privacy
  run nmcli connection up "$CONN"
else
  run ip link set "$IFACE" up
  run ip addr flush dev "$IFACE"
  run ip addr add "${SELF_IP}/${PREFIX}" dev "$IFACE"
fi

if ((!DRY_RUN)); then
  printf '\n'
  ip -br addr show "$IFACE"
  printf '\n'
  if ping -c2 -W2 "$PEER_IP" >/dev/null 2>&1; then
    echo "OK: $PEER_IP reachable on $IFACE."
  else
    echo "WARN: $PEER_IP not reachable yet."
    echo "      Verify the peer is configured (run setup_linklocal_mac.sh on the Mac)."
    echo "      ip neigh show dev $IFACE  # check for the peer MAC"
  fi
fi
