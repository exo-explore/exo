#!/bin/bash
set -e

export PATH="/opt/homebrew/bin:$PATH"

echo "=== Starting overnight bench runs at $(date) ==="

echo "--- [4/8] Qwen3.5-122B-A10B-GPTQ-Int4 ---"
echo "Skipping because Int 4"
#uv run bench/exo_bench.py --force-download --model "Qwen/Qwen3.5-122B-A10B-GPTQ-Int4" --pp 700 --tg 36000 --repeat 1

echo "--- [5/8] Qwen3.5-27B-FP8 ---"
#uv run bench/exo_bench.py --force-download --model "Qwen/Qwen3.5-27B-FP8" --pp 700 --tg 35133 --repeat 1

echo "--- [6/8] GLM-4.7-Flash-bf16 ---"
uv run bench/exo_bench.py --force-download --model "mlx-community/GLM-4.7-Flash-bf16" --pp 700 --tg 29000 --repeat 1

echo "--- [7/8] NVIDIA-Nemotron-3-Nano-30B-A3B (23000,1200) ---"
uv run bench/exo_bench.py --force-download --model "mlx-community/NVIDIA-Nemotron-3-Nano-30B-A3B-MLX-BF16" --pp 700 --tg 23000,1200 --repeat 1

echo "--- [8/8] Qwen3.5-27B-bf16 ---"
uv run bench/exo_bench.py --force-download --model "mlx-community/Qwen3.5-27B-bf16" --pp 700 --tg 35400 --repeat 1

echo "=== All bench runs complete at $(date) ==="
