#!/usr/bin/env bash
set -euo pipefail

# copy_model.sh: clone ~/.exo/models from SOURCE to one or more TARGETS using scp -3.
# Username defaults:
#   - If host is "aN" and no user given, username defaults to "aN".
#   - Otherwise defaults to $(whoami), unless you pass user@host.
#
# Examples:
#   ./copy_model.sh a1 a2 a3
#   ./copy_model.sh a1 frank@a2 192.168.1.3

if [ $# -lt 2 ]; then
  echo "Usage: $0 SOURCE TARGET [TARGET...]" >&2
  exit 2
fi

SOURCE="$1"
shift
TARGETS=("$@")

DEFAULT_USER="$(whoami)"
MODELS_REL=".exo/models" # relative under $HOME

timestamp() { date "+%Y-%m-%d %H:%M:%S"; }

split_user_host() {
  local in="$1"
  if [[ "$in" == *"@"* ]]; then
    printf "%s|%s" "${in%%@*}" "${in#*@}"
  else
    printf "|%s" "$in"
  fi
}

resolve_ip() {
  local hostish="$1"
  if [[ "$hostish" =~ ^a([0-9]+)$ ]]; then
    echo "192.168.1.${BASH_REMATCH[1]}"
  else
    echo "$hostish"
  fi
}

default_user_for() {
  local hostish="$1"
  if [[ "$hostish" =~ ^a([0-9]+)$ ]]; then
    echo "$hostish"
  else
    echo "$DEFAULT_USER"
  fi
}

SSH_OPTS=(-o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o LogLevel=ERROR -o ConnectTimeout=10)
SSHPASS_BIN="$(command -v sshpass || true)"
SCP_BIN="${SCP_BIN:-scp}"

read -s -p "Password for all hosts: " PASS
echo
if [ -n "$SSHPASS_BIN" ]; then
  echo "$(timestamp) sshpass found: will provide the password non-interactively."
else
  echo "$(timestamp) WARNING: sshpass not found — you’ll be prompted by scp/ssh per hop unless keys are set up."
fi

# Build source endpoint (default username logic)
IFS='|' read -r SRC_USER_RAW SRC_HOSTISH <<<"$(split_user_host "$SOURCE")"
SRC_USER="${SRC_USER_RAW:-$(default_user_for "$SRC_HOSTISH")}"
SRC_IP="$(resolve_ip "$SRC_HOSTISH")"
SRC_HOST="${SRC_USER}@${SRC_IP}"

echo "$(timestamp) Source: ${SRC_HOST}:~/${MODELS_REL}"
echo "$(timestamp) Targets: ${#TARGETS[@]}"

# Helper to run a simple remote command via ssh (for mkdir -p checks)
ssh_run() {
  local host="$1"
  shift
  if [ -n "$SSHPASS_BIN" ]; then
    sshpass -p "$PASS" ssh "${SSH_OPTS[@]}" "$host" "$@"
  else
    ssh "${SSH_OPTS[@]}" "$host" "$@"
  fi
}

# Ensure source dir exists (create if missing, per your request)
ssh_run "$SRC_HOST" "mkdir -p ~/${MODELS_REL}"

failures=0
count=0
for T in "${TARGETS[@]}"; do
  count=$((count + 1))
  IFS='|' read -r T_USER_RAW T_HOSTISH <<<"$(split_user_host "$T")"
  T_USER="${T_USER_RAW:-$(default_user_for "$T_HOSTISH")}"
  T_IP="$(resolve_ip "$T_HOSTISH")"
  T_HOST="${T_USER}@${T_IP}"

  echo "============================================================"
  echo "$(timestamp) [${count}/${#TARGETS[@]}] ${SRC_HOST}  ==>  ${T_HOST}"
  echo "$(timestamp) Ensuring destination directory exists…"
  ssh_run "$T_HOST" "mkdir -p ~/${MODELS_REL%/*}" # ~/.exo

  # Copy the whole "models" directory into ~/.exo on the target.
  # scp -3 = copy between two remotes via local; -r recursive; -p preserve times/modes
  if [ -n "$SSHPASS_BIN" ]; then
    echo "$(timestamp) Running: scp -3 -rp ${SRC_HOST}:~/${MODELS_REL} ${T_HOST}:~/.exo/"
    if sshpass -p "$PASS" "$SCP_BIN" "${SSH_OPTS[@]}" -3 -rp \
      "${SRC_HOST}:~/${MODELS_REL}" \
      "${T_HOST}:~/.exo/"; then
      echo "$(timestamp) [${count}] Done: ${T_HOST}"
    else
      echo "$(timestamp) [${count}] ERROR during scp to ${T_HOST}" >&2
      failures=$((failures + 1))
    fi
  else
    echo "$(timestamp) Running: scp -3 -rp ${SRC_HOST}:~/${MODELS_REL} ${T_HOST}:~/.exo/"
    if "$SCP_BIN" "${SSH_OPTS[@]}" -3 -rp \
      "${SRC_HOST}:~/${MODELS_REL}" \
      "${T_HOST}:~/.exo/"; then
      echo "$(timestamp) [${count}] Done: ${T_HOST}"
    else
      echo "$(timestamp) [${count}] ERROR during scp to ${T_HOST}" >&2
      failures=$((failures + 1))
    fi
  fi
done

echo "============================================================"
if [ "$failures" -eq 0 ]; then
  echo "$(timestamp) All transfers completed successfully."
else
  echo "$(timestamp) Completed with ${failures} failure(s)."
fi
