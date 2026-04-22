# MLX Tinygrad Interop

Private code for benchmarking MLX `<->` tinygrad tensor conversions.

## Workflow

Use the repo devshell and top-level dependency graph. Do not install ad-hoc
build dependencies or patch around them with one-off environment setups.

1. Change code locally.
2. Push the `mlx` and `tinygrad` fork changes.
3. In local `exo`, enter the devshell with `nix develop`.
4. Refresh `uv.lock` against the new fork heads with:
   `uv lock --upgrade-package mlx --upgrade-package tinygrad`
5. Commit and push the updated `exo` branch.
6. On the remote Mac, pull the updated repos.
7. Enter the devshell with `nix develop`.
8. Refresh the environment with `uv sync`.
9. Run tests or benchmarks with `uv run ...`.

Plain `uv lock` was not enough to move these git-based dependency SHAs during
testing; the explicit `--upgrade-package` form was required.

## Benchmark

The current benchmark script isolates the raw transformation only.

- Inputs are assumed to already be synchronized.
- Inputs are assumed to already be allocated.
- Inputs are assumed to already be materialized / realized.
- Setup stays outside the timed loop.
- The current unsafe helper bridge is asymmetric:
  - `MLX -> tinygrad` adopts an existing `MTLBuffer*`
  - `tinygrad -> MLX` rebuilds an MLX array from a raw pointer
- `mx.array(memoryview(...))` is a native copy path in current MLX, not an
  aliasing import path.

Example:

```bash
uv run python mlx_tinygrad_interop/bench_raw_conversion.py --dtype float32 --sizes 256,512,1024,2048,4096,7168
```

Validated remote command on `e16`:

```bash
uv run python mlx_tinygrad_interop/bench_raw_conversion.py --dtype float32 --sizes 7168 --warmup 32 --samples 5 --min-batch-us 1000
```

Observed `7168`-byte results on that run:

- `unsafe_helper_bridge`
  - `mlx_to_tinygrad`: `35.875 us` min, `36.553 us` median
  - `tinygrad_to_mlx`: `30.005 us` min, `30.197 us` median
- `memoryview_copy`
  - `mlx_to_tinygrad`: `38.960 us` min, `39.242 us` median
  - `tinygrad_to_mlx`: `2.719 us` min, `2.738 us` median
- `numpy_fallback`
  - `mlx_to_tinygrad`: `290.448 us` min, `290.927 us` median
  - `tinygrad_to_mlx`: `13.637 us` min, `13.698 us` median

## Current Scope

- Private / unsafe helpers only.
- Metal / unified-memory path only.
- Dense contiguous tensors only.
- Same-dtype conversions only.
- Current exporter/importer microbenchmarks are intended to separate helper
  overhead from end-to-end bridge cost.
