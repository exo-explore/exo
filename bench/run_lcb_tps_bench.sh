#!/usr/bin/env bash
#
# Run exo_bench.py for each model/mode from bench_params.json.
#
# For each entry, runs with:
#   --pp 800 (fixed, representative LCB prompt length)
#   --tg <mean completion tokens from vLLM>
#   --sharding tensor --instance-meta jaccl
#   --min-nodes 1 --max-nodes 4
#   --repeat 1
#   --danger-delete-downloads
#   --settle-timeout 300
#
# Results go to bench/eval_results/<model_dir>/tps_<mode>.json
#
# Usage:
#   bash bench/run_lcb_tps_bench.sh              # run all
#   bash bench/run_lcb_tps_bench.sh --dry-run    # show what would run

set -euo pipefail
cd "$(dirname "$0")"

PARAMS_FILE="eval_results/bench_params.json"
PP=800
HOST="${EXO_HOST:-s9}"
DRY_RUN=false

if [[ "${1:-}" == "--dry-run" ]]; then
    DRY_RUN=true
fi

if [[ ! -f "$PARAMS_FILE" ]]; then
    echo "ERROR: $PARAMS_FILE not found. Run compute_bench_params.py first."
    exit 1
fi

# Parse bench_params.json and run each entry
python3 -c "
import json, sys
data = json.load(open('$PARAMS_FILE'))
for entry in data:
    mlx_id = entry['mlx_model_id']
    mode = entry['mode']
    tg = entry['bench_params']['tg']
    vllm_name = entry['vllm_name']
    # Output dir: replace / with _
    out_dir = 'eval_results/' + mlx_id.replace('/', '_')
    out_file = out_dir + '/tps_' + mode + '.json'
    print(f'{mlx_id}\t{mode}\t{tg}\t{out_file}\t{vllm_name}')
" | while IFS=$'\t' read -r model mode tg out_file vllm_name; do
    out_dir="$(dirname "$out_file")"
    mkdir -p "$out_dir"

    echo ""
    echo "============================================================"
    echo "Model:  $model"
    echo "Mode:   $mode"
    echo "vLLM:   $vllm_name"
    echo "PP:     $PP"
    echo "TG:     $tg"
    echo "Output: $out_file"
    echo "============================================================"

    if [[ -f "$out_file" ]]; then
        echo "SKIP: $out_file already exists"
        continue
    fi

    if [[ "$DRY_RUN" == "true" ]]; then
        echo "DRY-RUN: would run exo_bench.py"
        continue
    fi

    uv run python exo_bench.py \
        --host "$HOST" \
        --model "$model" \
        --pp "$PP" \
        --tg "$tg" \
        --repeat 1 \
        --sharding tensor \
        --instance-meta jaccl \
        --min-nodes 1 \
        --max-nodes 4 \
        --settle-timeout 300 \
        --force-download \
        --danger-delete-downloads \
        --json-out "$out_file" || echo "FAILED: $model ($mode)"
done

echo ""
echo "All benchmarks complete."
