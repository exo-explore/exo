#!/usr/bin/env bash
# Test distributed RKLLM inference across 3 LXC containers on one RK3588.
#
# Creates 3 Incus containers, each pinned to a single NPU core via
# RKNN_CORE_MASK. All three run exo and discover each other via mDNS
# on the bridge network. A small model is split across the 3 "nodes"
# to test the full pipeline parallelism path.
#
# Prerequisites:
#   - Incus installed on the host
#   - /dev/rknpu accessible (RK3588 board)
#   - exo repo cloned at ~/exo (this repo)
#
# Usage:
#   scripts/test-3node-lxc.sh [setup|run|teardown|all]
#
# "all" runs setup, waits for the cluster to form, sends a test
# prompt, and tears down.
set -euo pipefail

BASE_IMAGE="${EXO_TEST_IMAGE:-images:debian/12}"
CONTAINER_PREFIX="exo-npu"
EXO_REPO="${EXO_REPO:-$(cd "$(dirname "$0")/.." && pwd)}"

log() { printf '\033[1;34m[3node-test]\033[0m %s\n' "$*"; }
die() { printf '\033[1;31m[3node-test]\033[0m %s\n' "$*" >&2; exit 1; }

setup_containers() {
  log "creating 3 LXC containers for distributed NPU test"

  for i in 0 1 2; do
    local name="${CONTAINER_PREFIX}-${i}"
    local mask=$((1 << i))  # 0x1, 0x2, 0x4
    local port=$((52415 + i))

    if incus info "$name" >/dev/null 2>&1; then
      log "$name already exists, skipping create"
    else
      log "creating $name (core_mask=0x$(printf '%x' $mask), api_port=$port)"
      incus launch "$BASE_IMAGE" "$name"
      sleep 3
    fi

    # Pass the NPU device into the container
    incus config device add "$name" npu unix-char path=/dev/rknpu 2>/dev/null || true

    # Mount the exo repo read-only so we don't need to clone inside
    incus config device add "$name" exo-src disk \
      source="$EXO_REPO" path=/opt/exo 2>/dev/null || true

    # Install Python and deps inside the container
    incus exec "$name" -- bash -c '
      apt-get update -qq
      apt-get install -y -qq python3 python3-venv python3-pip git curl
      if ! command -v uv >/dev/null 2>&1; then
        curl -LsSf https://astral.sh/uv/install.sh | sh
      fi
    ' 2>/dev/null

    # Write the startup env vars
    incus exec "$name" -- bash -c "
      cat > /opt/exo-env.sh <<'ENVEOF'
export RKNN_CORE_MASK=$mask
export RKLLM_SERVER_PORT=8080
export EXO_PORT=$port
export PATH=\$HOME/.local/bin:\$PATH
ENVEOF
    "

    log "$name ready (core $i, mask=0x$(printf '%x' $mask))"
  done

  log "all 3 containers created"
}

run_cluster() {
  log "starting exo on all 3 containers"

  for i in 0 1 2; do
    local name="${CONTAINER_PREFIX}-${i}"
    incus exec "$name" -- bash -c '
      source /opt/exo-env.sh
      cd /opt/exo
      nohup $HOME/.local/bin/uv run exo --port $EXO_PORT > /tmp/exo.log 2>&1 &
      echo $! > /tmp/exo.pid
    '
    log "started exo on $name (pid written to /tmp/exo.pid)"
  done

  log "waiting 15s for peer discovery"
  sleep 15

  # Check cluster health via the first node's API
  local api_ip
  api_ip=$(incus exec "${CONTAINER_PREFIX}-0" -- hostname -I | awk '{print $1}')
  log "checking cluster at http://$api_ip:52415"

  local cluster_info
  cluster_info=$(curl -sS "http://$api_ip:52415/v1/models" 2>/dev/null || echo "FAILED")
  echo "$cluster_info"

  if echo "$cluster_info" | grep -q "FAILED"; then
    die "cluster API not responding"
  fi

  log "cluster is up, sending test prompt"

  local response
  response=$(curl -sS "http://$api_ip:52415/v1/chat/completions" \
    -H "Content-Type: application/json" \
    -d '{
      "model": "qwen2.5-1.5b-rkllm",
      "messages": [{"role": "user", "content": "What is 2+2?"}],
      "max_tokens": 32
    }' 2>/dev/null || echo "FAILED")

  echo "$response" | head -20
  if echo "$response" | grep -q "choices"; then
    log "distributed inference test PASSED"
  else
    log "distributed inference test FAILED (no choices in response)"
  fi
}

teardown_containers() {
  log "tearing down test containers"
  for i in 0 1 2; do
    local name="${CONTAINER_PREFIX}-${i}"
    incus delete --force "$name" 2>/dev/null || true
    log "deleted $name"
  done
  log "teardown complete"
}

case "${1:-all}" in
  setup)    setup_containers ;;
  run)      run_cluster ;;
  teardown) teardown_containers ;;
  all)
    setup_containers
    run_cluster
    teardown_containers
    ;;
  *)
    echo "usage: $0 [setup|run|teardown|all]"
    exit 1
    ;;
esac
