#!/usr/bin/env bash
set -euo pipefail

# 8-hour thermal throttling bench loop.
# Runs bench/exo_bench.py repeatedly, one timestamped JSON per iteration.
# Alternates --use-prefix-cache: odd iters cold, even iters cached.
# Requires exo to already be running in another terminal (`uv run exo -v`).

DURATION_HOURS="${DURATION_HOURS:-8}"
MODEL="${MODEL:-mlx-community/Qwen3.5-27B-4bit}"
PP="${PP:-8192 16384}"
TG="${TG:-512}"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

RUN_DIR="bench/thermal_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$RUN_DIR"
LOG="$RUN_DIR/loop.log"

END=$(( $(date +%s) + DURATION_HOURS * 3600 ))
i=0

{
  echo "run_dir      = $RUN_DIR"
  echo "duration_hrs = $DURATION_HOURS"
  echo "model        = $MODEL"
  echo "pp / tg      = $PP / $TG"
  echo "prefix cache = alternating (odd=cold, even=cached)"
  echo "started      = $(date -Iseconds)"
} | tee "$LOG"

while [ "$(date +%s)" -lt "$END" ]; do
  i=$((i + 1))
  ts=$(date +%Y%m%d_%H%M%S)

  cache_flag=()
  cache_label="cold"
  if (( i % 2 == 0 )); then
    cache_flag=(--use-prefix-cache)
    cache_label="cached"
  fi

  echo "=== iter $i ($cache_label) @ $ts ===" | tee -a "$LOG"

  caffeinate -i uv run python bench/exo_bench.py \
    --model "$MODEL" \
    --pp $PP --tg "$TG" \
    --repeat 1 \
    "${cache_flag[@]}" \
    --json-out "$RUN_DIR/iter_${i}_${cache_label}_${ts}.json" \
    2>&1 | tee -a "$LOG" || echo "iter $i FAILED (exit $?)" | tee -a "$LOG"
done

echo "finished     = $(date -Iseconds) after $i iterations" | tee -a "$LOG"
