#!/usr/bin/env bash
set -Eeuo pipefail

SELF_IP="169.254.100.2"
PEER_IP="169.254.100.1"
NETMASK="255.255.0.0"
IFACE=""
DRY_RUN=0

usage() {
  cat <<EOF
Usage: sudo $(basename "$0") [options]

Configure a Mac Ethernet interface with a static IPv4 for a host-to-host link
to the DGX/GX10 peer.

Defaults: this Mac = ${SELF_IP}, peer = ${PEER_IP}, mask = ${NETMASK}.

Options:
  --iface IFACE     Interface (e.g. en12). Default: auto-detect.
  --self-ip IP      This Mac's address. Default: ${SELF_IP}.
  --peer-ip IP      Peer for verification ping. Default: ${PEER_IP}.
  --netmask MASK    Default: ${NETMASK}.
  --dry-run         Print actions without applying.
  -h, --help        Show this help.
EOF
}

while (($#)); do
  case "$1" in
  --iface)
    shift
    IFACE="${1:?}"
    ;;
  --self-ip)
    shift
    SELF_IP="${1:?}"
    ;;
  --peer-ip)
    shift
    PEER_IP="${1:?}"
    ;;
  --netmask)
    shift
    NETMASK="${1:?}"
    ;;
  --dry-run) DRY_RUN=1 ;;
  -h | --help)
    usage
    exit 0
    ;;
  *)
    echo "Unknown arg: $1" >&2
    usage >&2
    exit 1
    ;;
  esac
  shift
done

[[ $EUID -eq 0 ]] || {
  echo "Run with sudo." >&2
  exit 1
}

run() {
  printf '+'
  printf ' %q' "$@"
  printf '\n'
  ((DRY_RUN)) || "$@"
}

target_subnet_prefix() {
  local ip="$1"
  printf '%s.' "${ip%.*}"
}

iface_score() {
  local iface="$1" info subnet
  info="$(ifconfig "$iface" 2>/dev/null || true)"
  [[ -n $info ]] || {
    echo 0
    return
  }
  grep -q 'status: active' <<<"$info" || {
    echo 0
    return
  }
  subnet="$(target_subnet_prefix "$SELF_IP")"
  if grep -qE "inet ${subnet//./\\.}" <<<"$info"; then
    echo 100
    return
  fi
  if grep -qE 'inet 169\.254\.' <<<"$info"; then
    echo 80
    return
  fi
  if ! grep -qE '^[[:space:]]*inet ' <<<"$info"; then
    echo 60
    return
  fi
  echo 10
}

detect_iface() {
  local best="" best_score=0 iface score
  for iface in $(ifconfig -l); do
    [[ $iface =~ ^en[0-9]+$ ]] || continue
    score="$(iface_score "$iface")"
    if ((score > best_score)); then
      best="$iface"
      best_score="$score"
    fi
  done
  ((best_score >= 60)) || return 1
  printf '%s\n' "$best"
}

iface_to_service() {
  local iface="$1" line port=""
  while IFS= read -r line; do
    if [[ $line == "Hardware Port: "* ]]; then
      port="${line#Hardware Port: }"
    elif [[ $line == "Device: $iface" ]]; then
      printf '%s\n' "$port"
      return 0
    fi
  done < <(networksetup -listallhardwareports)
  return 1
}

if [[ -z $IFACE ]]; then
  IFACE="$(detect_iface || true)"
  [[ -n $IFACE ]] || {
    echo "Could not auto-detect a wired interface. Pass --iface enX." >&2
    echo "Active interfaces:" >&2
    ifconfig -l | tr ' ' '\n' | grep -E '^en[0-9]+$' | while read -r i; do
      printf '  %-6s %s\n' "$i" "$(ifconfig "$i" | grep -E 'status:|inet ' | tr '\n' ' ')" >&2
    done
    exit 1
  }
  echo "Auto-detected interface: $IFACE"
fi

ifconfig "$IFACE" >/dev/null 2>&1 || {
  echo "Interface $IFACE does not exist." >&2
  exit 1
}

SERVICE="$(iface_to_service "$IFACE" || true)"
[[ -n $SERVICE ]] || {
  echo "No network service maps to $IFACE. Check System Settings -> Network." >&2
  exit 1
}
echo "Network service: $SERVICE"

run networksetup -setmanual "$SERVICE" "$SELF_IP" "$NETMASK" ""

if ((!DRY_RUN)); then
  printf '\n'
  ifconfig "$IFACE" | grep -E 'inet |status:'
  printf '\n'
  if ping -c2 -t3 "$PEER_IP" >/dev/null 2>&1; then
    echo "OK: $PEER_IP reachable on $IFACE."
  else
    echo "WARN: $PEER_IP not reachable yet."
    echo "      Verify the peer is configured (run setup_linklocal_dgx.sh on the GX10)."
    echo "      arp -an -i $IFACE  # check for the peer MAC"
  fi
fi
