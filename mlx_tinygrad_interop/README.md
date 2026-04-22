# MLX Tinygrad Interop

Private code for benchmarking MLX `<->` tinygrad tensor conversions.

## Workflow

Use the repo devshell and top-level dependency graph. Do not install ad-hoc
build dependencies or patch around them with one-off environment setups.

1. Change code locally.
2. Push the `mlx` and `tinygrad` fork changes.
3. On the remote Mac, pull the updated repos.
4. Enter the devshell with `nix develop`.
5. Refresh dependencies with `uv lock && uv sync`.
6. Run tests or benchmarks with `uv run ...`.

## Benchmark

The current benchmark script isolates the raw transformation only.

- Inputs are assumed to already be synchronized.
- Inputs are assumed to already be allocated.
- Inputs are assumed to already be materialized / realized.
- Setup stays outside the timed loop.

Example:

```bash
uv run python mlx_tinygrad_interop/bench_raw_conversion.py --dtype float32 --sizes 256,512,1024,2048,4096,7168
```

## Current Scope

- Private / unsafe helpers only.
- Metal / unified-memory path only.
- Dense contiguous tensors only.
- Same-dtype conversions only.
