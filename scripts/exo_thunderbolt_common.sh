#!/usr/bin/env bash

set -euo pipefail

EXO_MODE="${EXO_MODE:-ThunderboltPair}"
EXO_BRIDGE_DEVICE="${EXO_BRIDGE_DEVICE:-bridge0}"
EXO_LOCAL_BRIDGE_IP="${EXO_LOCAL_BRIDGE_IP:-10.0.0.2}"
EXO_PEER_BRIDGE_IP="${EXO_PEER_BRIDGE_IP:-10.0.0.1}"
EXO_THUNDERBOLT_MEMBERS="${EXO_THUNDERBOLT_MEMBERS:-en1 en2 en3}"
EXO_EXPECTED_LOCATION="${EXO_EXPECTED_LOCATION:-Automatic}"
EXO_EXPECTED_REMOTE_NODE_ID="${EXO_EXPECTED_REMOTE_NODE_ID:-}"
EXO_NAMESPACE="${EXO_NAMESPACE:-leo-m2}"
EXO_API_PORT="${EXO_API_PORT:-52415}"
EXO_DEFAULT_MODEL="${EXO_DEFAULT_MODEL:-mlx-community/Meta-Llama-3.1-8B-Instruct-4bit}"
EXO_WORKDIR="${EXO_WORKDIR:-$HOME/exo}"
EXO_UV_BIN="${EXO_UV_BIN:-/opt/homebrew/bin/uv}"
EXO_REMOTE_HOST="${EXO_REMOTE_HOST:-zealous}"
EXO_REMOTE_WORKDIR="${EXO_REMOTE_WORKDIR:-$HOME/exo}"
EXO_REMOTE_UV_BIN="${EXO_REMOTE_UV_BIN:-/opt/homebrew/bin/uv}"
EXO_REMOTE_START_ARGS="${EXO_REMOTE_START_ARGS:-}"
EXO_START_ARGS="${EXO_START_ARGS:-}"
EXO_LOG_DIR="${EXO_LOG_DIR:-$HOME/Library/Logs/exo-thunderbolt}"
EXO_RUN_LOG="${EXO_RUN_LOG:-$EXO_LOG_DIR/exo-run.log}"
EXO_WATCHDOG_LOG="${EXO_WATCHDOG_LOG:-$EXO_LOG_DIR/watchdog.log}"
EXO_LOCK_DIR="${EXO_LOCK_DIR:-$EXO_LOG_DIR/recovery.lock}"
EXO_LOG_MAX_BYTES="${EXO_LOG_MAX_BYTES:-5242880}"
EXO_LOG_TAIL_LINES="${EXO_LOG_TAIL_LINES:-4000}"

mkdir -p "$EXO_LOG_DIR"

log() {
  printf '[%s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*"
}

warn() {
  printf '[%s] WARN: %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*" >&2
}

die() {
  printf '[%s] ERROR: %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*" >&2
  exit 1
}

need_cmd() {
  command -v "$1" >/dev/null 2>&1 || die "missing required command: $1"
}

rotate_log() {
  local file="$1"
  local max_bytes="${2:-$EXO_LOG_MAX_BYTES}"
  local tail_lines="${3:-$EXO_LOG_TAIL_LINES}"
  [[ -f "$file" ]] || return 0
  local size
  size=$(stat -f%z "$file" 2>/dev/null || echo 0)
  if [[ "$size" -le "$max_bytes" ]]; then
    return 0
  fi
  local tmp
  tmp=$(mktemp "${file}.XXXX")
  tail -n "$tail_lines" "$file" >"$tmp" 2>/dev/null || true
  mv "$tmp" "$file"
}

rotate_all_logs() {
  rotate_log "$EXO_RUN_LOG"
  rotate_log "$EXO_WATCHDOG_LOG"
  rotate_log "/tmp/exo-thunderbolt-local-script.log"
  rotate_log "/tmp/exo-thunderbolt-script.log"
  rotate_log "/tmp/exo-thunderbolt-run.log"
}

acquire_recovery_lock() {
  if mkdir "$EXO_LOCK_DIR" 2>/dev/null; then
    trap 'rm -rf "$EXO_LOCK_DIR"' EXIT
    return 0
  fi
  return 1
}

get_current_location() {
  scselect | awk -F '[()]' '/^[[:space:]]*\*/ {print $2; exit}'
}

find_bridge_service_name() {
  networksetup -listnetworkserviceorder 2>/dev/null | awk '
    BEGIN { service = ""; bridge = 0 }
    /^\([0-9]+\)/ {
      service = $0
      sub(/^\([0-9]+\)[[:space:]]*/, "", service)
      bridge = 0
      next
    }
    /Device: bridge0\)/ {
      bridge = 1
    }
    bridge == 1 && service != "" {
      print service
      exit
    }
  '
}

bridge_exists() {
  ifconfig "$EXO_BRIDGE_DEVICE" >/dev/null 2>&1
}

bridge_ip() {
  ifconfig "$EXO_BRIDGE_DEVICE" 2>/dev/null | awk '
    /inet / {
      if ($2 == expected) {
        print $2
        exit
      }
      if ($2 !~ /^169\.254\./ && preferred == "") {
        preferred = $2
      }
      if (fallback == "") {
        fallback = $2
      }
    }
    END {
      if (preferred != "") {
        print preferred
      } else if (fallback != "") {
        print fallback
      }
    }
  ' expected="$EXO_LOCAL_BRIDGE_IP"
}

bridge_members() {
  ifconfig "$EXO_BRIDGE_DEVICE" 2>/dev/null | awk '/member:/ {print $2}'
}

route_interface_to_peer() {
  route -n get "$EXO_PEER_BRIDGE_IP" 2>/dev/null | awk '/interface:/{print $2; exit}'
}

can_repair_bridge_non_interactive() {
  sudo -n /usr/sbin/scselect "$EXO_EXPECTED_LOCATION" >/dev/null 2>&1
}

current_exo_pid() {
  lsof -tiTCP:"$EXO_API_PORT" -sTCP:LISTEN 2>/dev/null | head -n1
}

wait_for_api() {
  local attempts="${1:-30}"
  local delay="${2:-1}"
  local i
  for ((i = 1; i <= attempts; i++)); do
    if curl -fsS "http://localhost:${EXO_API_PORT}/node_id" >/dev/null 2>&1; then
      return 0
    fi
    sleep "$delay"
  done
  return 1
}

local_state() {
  curl -fsS "http://localhost:${EXO_API_PORT}/state"
}

local_node_id() {
  curl -fsS "http://localhost:${EXO_API_PORT}/node_id" | tr -d '"'
}

topology_node_count() {
  local_state | jq '.topology.nodes | length'
}

wait_for_cluster() {
  local attempts="${1:-30}"
  local delay="${2:-1}"
  local i
  for ((i = 1; i <= attempts; i++)); do
    if [[ "$(topology_node_count 2>/dev/null || echo 0)" -ge 2 ]]; then
      if [[ -n "$EXO_EXPECTED_REMOTE_NODE_ID" ]]; then
        if local_state 2>/dev/null | jq -e --arg id "$EXO_EXPECTED_REMOTE_NODE_ID" '.topology.nodes | index($id)' >/dev/null; then
          return 0
        fi
      else
        return 0
      fi
    fi
    sleep "$delay"
  done
  return 1
}

mode_requires_bridge() {
  [[ "$EXO_MODE" == "ThunderboltPair" ]]
}

start_args_resolved() {
  if [[ -n "$EXO_START_ARGS" ]]; then
    printf '%s' "$EXO_START_ARGS"
  elif mode_requires_bridge; then
    printf '%s' "--no-downloads"
  else
    printf '%s' ""
  fi
}

remote_start_args_resolved() {
  if [[ -n "$EXO_REMOTE_START_ARGS" ]]; then
    printf '%s' "$EXO_REMOTE_START_ARGS"
  elif mode_requires_bridge; then
    printf '%s' "--no-downloads"
  else
    printf '%s' ""
  fi
}

start_local_exo() {
  local args
  args="$(start_args_resolved)"
  (
    cd "$EXO_WORKDIR"
    nohup /usr/bin/script -q "$EXO_RUN_LOG" /bin/zsh -lc \
      "cd '$EXO_WORKDIR' && env EXO_LIBP2P_NAMESPACE='$EXO_NAMESPACE' '$EXO_UV_BIN' run exo -v ${args}" \
      </dev/null >/tmp/exo-thunderbolt-local-script.log 2>&1 &
  )
}

restart_local_exo() {
  local pid
  pid="$(current_exo_pid || true)"
  if [[ -n "$pid" ]]; then
    kill "$pid" >/dev/null 2>&1 || true
    sleep 1
  fi
  start_local_exo
  wait_for_api 45 1 || return 1
}

restart_remote_exo() {
  local remote_args
  remote_args="$(remote_start_args_resolved)"
  ssh -tt -o BatchMode=yes -o ConnectTimeout=10 "$EXO_REMOTE_HOST" \
    "set -euo pipefail; \
     pid=\$(lsof -tiTCP:${EXO_API_PORT} -sTCP:LISTEN 2>/dev/null | head -n1 || true); \
     if [ -n \"\$pid\" ]; then kill \"\$pid\" >/dev/null 2>&1 || true; fi; \
     for _ in \$(seq 1 20); do \
       if ! lsof -tiTCP:${EXO_API_PORT} -sTCP:LISTEN >/dev/null 2>&1; then break; fi; \
       sleep 1; \
     done; \
     cd ${EXO_REMOTE_WORKDIR}; \
     nohup /usr/bin/script -q /tmp/exo-thunderbolt-run.log /bin/zsh -lc 'cd ${EXO_REMOTE_WORKDIR} && env EXO_LIBP2P_NAMESPACE=${EXO_NAMESPACE} ${EXO_REMOTE_UV_BIN} run exo -v ${remote_args}' </dev/null >/tmp/exo-thunderbolt-script.log 2>&1 & \
     for _ in \$(seq 1 45); do \
       if curl -fsS http://localhost:${EXO_API_PORT}/node_id >/dev/null 2>&1; then exit 0; fi; \
       sleep 1; \
     done; \
     exit 1"
}
