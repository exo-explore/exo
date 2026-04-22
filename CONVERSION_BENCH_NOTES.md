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
3. In local `exo`, enter the devshell with `nix develop`.
4. Regenerate the lockfile against the new fork heads with:
   `uv lock --upgrade-package mlx --upgrade-package tinygrad`
5. Commit and push the updated `exo` branch, including the regenerated
   `uv.lock`.
6. On the remote Mac, pull the updated `exo` branch.
7. Enter the devshell with `nix develop`.
8. Refresh the environment with `uv sync`.
9. Run tests and benchmarks with `uv run ...`.

Do not rely on ad-hoc per-host build environments when the flake / devshell can
carry the needed toolchain.

### Important Lockfile Note

For these branch-based git dependencies, plain `uv lock` was not sufficient to
advance the pinned SHAs in `uv.lock` during testing. The working command was:

`uv lock --upgrade-package mlx --upgrade-package tinygrad`

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

The unsafe bridge was validated through the repo-standard remote flow on `e16`:

1. `git pull --ff-only`
2. `nix develop`
3. `uv sync`
4. `uv run python mlx_tinygrad_interop/bench_raw_conversion.py ...`

The direct helpers worked in both directions:

- `MLX -> tinygrad` direct alias path returned correct values.
- `tinygrad -> MLX` direct alias path returned correct values.

Canonical remote latency measurements for `float32` and `7168` bytes were:

- `direct_alias`
  - `mlx_to_tinygrad`: `35.875 us` min, `36.553 us` median
  - `tinygrad_to_mlx`: `30.005 us` min, `30.197 us` median
- `memoryview_copy`
  - `mlx_to_tinygrad`: `38.960 us` min, `39.242 us` median
  - `tinygrad_to_mlx`: `2.719 us` min, `2.738 us` median
- `numpy_fallback`
  - `mlx_to_tinygrad`: `290.448 us` min, `290.927 us` median
  - `tinygrad_to_mlx`: `13.637 us` min, `13.698 us` median

Interpretation:

- The current Python-exposed direct alias path is functional, but still far
  above the target `1-10 us` range for a `7 kB` tensor.
- `MLX -> tinygrad` is dominated by fixed overhead even when aliasing, because
  the direct path is only marginally faster than the `memoryview_copy` path.
- `tinygrad -> MLX` currently has a very cheap copy path because `mx.array()`
  over a zero-copy tinygrad `memoryview` is much cheaper than the private
  raw-pointer helper that constructs a new MLX wrapper object.
- At this tensor size, Python call overhead and wrapper construction matter
  much more than raw byte movement.

## Near-Term Plan

1. Reduce Python wrapper overhead around the direct path before changing the
   storage model again.
2. Add narrower microbenchmarks that time only the helper calls, separate from
   end-to-end tensor wrapper creation.
3. Decide whether the next iteration should make MLX import a foreign
   `MTLBuffer*` directly instead of constructing from a raw pointer.

## Open Questions

- Whether MLX should eventually import foreign `MTLBuffer*` handles directly,
  rather than only raw pointers, for a more symmetric bridge.
- Whether the first fast path should support contiguous slices with byte
  offsets, or only base-contiguous tensors.
- Whether a native-copy middle path should be benchmarked immediately, or only
  the direct path and Python / NumPy fallback.
