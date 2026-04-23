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

The current benchmark keeps source tensor construction and explicit pre-sync
outside the timed loop, but many rows still include per-call helper, binding,
owner pinning, and wrapper-construction overhead.

- Inputs are assumed to already be synchronized.
- Inputs are assumed to already be allocated.
- Inputs are assumed to already be materialized / realized.
- Setup stays outside the timed loop.
- The current unsafe helper bridge is asymmetric:
  - `MLX -> tinygrad` adopts an existing `MTLBuffer*`
  - `tinygrad -> MLX` rebuilds an MLX array from a raw pointer
- Newer `MLX -> tinygrad` rows also cover:
  - a single MLX-side entrypoint that calls into tinygrad without the exporter
    dict round-trip
  - a reusable tinygrad borrower that rebinds the borrowed `MTLBuffer*`
- `mx.array(memoryview(...))` is a native copy path in current MLX, not an
  aliasing import path.

Test command:

```bash
uv run python -m unittest mlx_tinygrad_interop.test_interop
```

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
  - `mlx_to_tinygrad`: `22.212 us` min, `22.372 us` median
  - `tinygrad_to_mlx`: `28.437 us` min, `28.548 us` median
- `unsafe_helper_legacy`
  - `mlx_to_tinygrad`: `35.296 us` min, `35.483 us` median
- `unsafe_helper_maybe_copy`
  - `tinygrad_to_mlx`: `28.500 us` min, `28.723 us` median
- `memoryview_copy`
  - `mlx_to_tinygrad`: `38.712 us` min, `39.033 us` median
  - `tinygrad_to_mlx`: `2.603 us` min, `2.615 us` median
- `numpy_baseline`
  - `mlx_to_tinygrad`: `285.083 us` min, `286.635 us` median
  - `tinygrad_to_mlx`: `13.249 us` min, `13.265 us` median

Later remote microbench runs showed the split more clearly:

- `MLX -> tinygrad` is dominated by tinygrad import / wrapper creation.
- The lower-overhead tinygrad import helper cuts `MLX -> tinygrad` by about
  `1.6x`, but still leaves it around `22 us`.
- The strict alias-only `tinygrad -> MLX` helper succeeds on `e16`; its timing
  is effectively the same as the maybe-copy helper on that host.
- `tinygrad -> MLX` is dominated by tinygrad export in the unsafe helper path.
- `tinygrad -> MLX memoryview_copy` is already the practical low-latency path
  for small tensors.
- Offsetted MLX slices now export both logical bytes and backing-buffer bytes,
  and a nonzero-offset slice was validated successfully into tinygrad.

## Current Scope

- Private / unsafe helpers only.
- Metal / unified-memory path only.
- Dense contiguous tensors only.
- Same-dtype conversions only.
- Current exporter/importer microbenchmarks are intended to separate helper
  overhead from end-to-end bridge cost.
