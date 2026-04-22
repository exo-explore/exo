# Tensor Conversion Benchmark Notes

## Current Goal

Benchmark raw tinygrad `<->` MLX tensor transformation latency on Apple Silicon
for tensors that are already:

- synchronized
- allocated
- materialized / realized

The timed region should measure only the transformation itself.

## Repo Layout

- Root notes file:
  - `CONVERSION_BENCH_NOTES.md`
- Interop code:
  - `mlx_tinygrad_interop/`
- Historical benchmark kept as-is:
  - `tmp/bench_pingpong.py`

## Current Fast-Path Design

The first direct benchmark path is intentionally narrow.

- `MLX -> tinygrad`
  - export MLX Metal storage metadata
  - import into tinygrad by aliasing the existing `MTLBuffer*`
- `tinygrad -> MLX`
  - export tinygrad Metal storage metadata
  - import into MLX by wrapping the underlying unified-memory pointer with a
    no-copy MLX array constructor path

This is asymmetric internally, but both directions aim to avoid copying tensor
bytes.

## Implemented Helper Surface

- MLX
  - `mx.metal._unsafe_export_storage(array)`
  - `mx.metal._unsafe_array_from_ptr(raw_ptr, shape, dtype, owner=None)`
- tinygrad
  - `Tensor._unsafe_metal_storage()`
  - `Tensor._unsafe_from_metal_buffer(mtl_buffer_ptr, shape, dtype=..., byte_offset=0, owner=None)`

These helpers are intentionally private and unsafe.

## Temporary Eligibility Rules

The current direct path should only accept tensors that are:

- backed by Metal storage
- already realized / available
- single-device
- dense row-major contiguous
- concrete-shaped
- dtype-compatible without conversion

Non-contiguous views, broadcasts, dtype casts, and multi-device tensors should
fall back to slower paths.

## Workflow

Use the `exo` devshell and `uv` workflow.

1. Change code locally.
2. Push the updated `mlx` and `tinygrad` fork branches.
3. On the remote Mac, pull the updated repos.
4. Enter the devshell with `nix develop`.
5. Refresh dependency resolution with `uv lock && uv sync`.
6. Run tests and benchmarks with `uv run ...`.

Do not rely on ad-hoc per-host build environments when the flake / devshell can
carry the needed toolchain.

## Known Nuances / Footguns

- Unified memory does not mean both frameworks consume shared storage in the
  same way. Metal kernels still bind `MTLBuffer` objects.
- Synchronization can dominate measured latency if it leaks into the timed path.
- Python overhead matters at the `1-10 us` scale, so helper calls and wrapper
  construction can dominate tiny tensors even when no tensor bytes are copied.
- tinygrad tensors are graph objects, but once realized they do have concrete
  underlying storage.
- External mutation and aliasing can bypass autograd expectations in both
  frameworks.
- The current fast path is asymmetric:
  - MLX exports `MTLBuffer*` for the `MLX -> tinygrad` direction.
  - tinygrad exports raw unified-memory pointer for the `tinygrad -> MLX`
    direction.
- The first tinygrad import helper supports byte offsets.
- The first MLX import helper is raw-pointer based rather than foreign
  `MTLBuffer*` based.
- `mx.metal._unsafe_export_storage(...)` currently expects an MLX array that is
  already in the C++ `available` state. In practice, `mx.array(np_array)` met
  that precondition for local smoke testing, while `mx.arange(...)` did not.

## Current Findings

The unsafe bridge was smoke-tested successfully on a remote Mac:

- `MLX -> tinygrad` direct alias path returned correct values.
- `tinygrad -> MLX` direct alias path returned correct values.

Preliminary latency measurements for `float32` and `7168` bytes were:

- `direct_alias`
  - `mlx_to_tinygrad`: about `33.5 us`
  - `tinygrad_to_mlx`: about `28.0 us`
- `numpy_fallback`
  - `mlx_to_tinygrad`: about `269 us`
  - `tinygrad_to_mlx`: about `12.9 us`

These numbers were gathered before switching fully to the repo-standard
`nix develop` + `uv run` flow, so they should be treated as provisional rather
than canonical benchmark results.

## Near-Term Plan

1. Push the current helper changes in `mlx` and `tinygrad`.
2. Re-run the benchmark from `mlx_tinygrad_interop/` using the repo-standard
   remote flow.
3. Separate true alias-path cost from Python wrapper overhead more aggressively
   if the current numbers remain too high.

## Open Questions

- Whether MLX should eventually import foreign `MTLBuffer*` handles directly,
  rather than only raw pointers, for a more symmetric bridge.
- Whether the first fast path should support contiguous slices with byte
  offsets, or only base-contiguous tensors.
- Whether a native-copy middle path should be benchmarked immediately, or only
  the direct path and Python / NumPy fallback.
