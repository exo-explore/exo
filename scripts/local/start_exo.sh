#!/usr/bin/env bash
# Start exo for the F002 stage-5 two-node setup (MSI 5060 + APU-TPNB04 4060, WSL2).
#
# Host-aware on purpose: the two nodes need genuinely different settings, and
# keeping one script in git is what stops them drifting apart (we lost a day to
# a cuda12/cuda13 split and another to a CPU-vs-CUDA split).
#
#   LD_PRELOAD          both  - anaconda ships libstdc++ 3.4.29; transformers loads it
#                              first and libmlx then fails on GLIBCXX_3.4.30
#   OVERRIDE_MEMORY_MB  both  - exo reports system RAM (profiling.py), but CUDA runs
#                              out of VRAM; keeps placement's budget honest
#   CUDA_HOME           APU   - its /usr/local/cuda is 12.6, whose cuda_fp8.h lacks
#                              __nv_fp8_e8m0, so nvrtc fails to JIT mlx kernels
#   LD_LIBRARY_PATH     APU   - same reason: system lib64 only has libcublasLt.so.12
#   EXO_ZENOH_CONNECT   MSI   - Wi-Fi APs drop IPv6 link-local multicast; one side
#                              dialling the other over unicast is enough
#
# Optional: EXO_PREFILL_STEP_SIZE (see the local patch in batch_generate.py).
# Leave it unset for upstream behaviour (4096). Lowering it does NOT fix the
# stage-18 prefill OOM — 512 still OOMs on a 1k-token prompt.
set -euo pipefail

SESSION="${SESSION:-exo_test}"
LOG="${LOG:-/tmp/exo_run.log}"
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

env_common=(
  "LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6"
  "OVERRIDE_MEMORY_MB=${OVERRIDE_MEMORY_MB:-7000}"
)

case "$(hostname)" in
  MSI)
    node_env=("EXO_ZENOH_CONNECT=tcp/${PEER_IP:-10.156.19.41}:52414")
    ;;
  APU-TPNB04)
    sp="$REPO/.venv/lib/python3.13/site-packages/nvidia"
    node_env=(
      "CUDA_HOME=$HOME/cuda-13.0"
      "LD_LIBRARY_PATH=$sp/cu13/lib:$sp/cudnn/lib:$sp/nccl/lib"
    )
    ;;
  *)
    echo "unknown host $(hostname); add its stanza before running" >&2
    exit 1
    ;;
esac

[ -n "${EXO_PREFILL_STEP_SIZE:-}" ] &&
  node_env+=("EXO_PREFILL_STEP_SIZE=$EXO_PREFILL_STEP_SIZE")

uv_bin="$(command -v uv || echo "$HOME/.local/bin/uv")"

tmux kill-session -t "$SESSION" 2>/dev/null || true
sleep 2
tmux new-session -d -s "$SESSION" \
  "cd '$REPO' && env ${env_common[*]} ${node_env[*]} '$uv_bin' run exo 2>&1 | tee -a '$LOG'"

echo "started on $(hostname): ${node_env[*]}"
for _ in $(seq 1 60); do
  curl -s -m 2 http://localhost:52415/state >/dev/null 2>&1 && { echo "API up"; exit 0; }
  sleep 2
done
echo "API did not come up within 120s; tmux capture-pane -p -t $SESSION" >&2
exit 1
