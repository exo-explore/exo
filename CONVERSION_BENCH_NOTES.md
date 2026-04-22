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

This is asymmetric internally:

- `MLX -> tinygrad` aliases an existing `MTLBuffer*`
- `tinygrad -> MLX` rebuilds an MLX array from a raw unified-memory pointer

So the current bridge benchmark is not a symmetric measure of pure storage
adoption cost. That is acceptable for now: the working goal is "good enough"
bidirectional latency, not symmetry for its own sake.

## Implemented Helper Surface

- MLX
  - `mx.metal._unsafe_export_storage(array)`
  - `mx.metal._unsafe_array_from_ptr(raw_ptr, shape, dtype, owner=None)`
- tinygrad
  - `Tensor._unsafe_metal_storage()`
  - `Tensor._unsafe_from_metal_buffer(mtl_buffer_ptr, shape, dtype=..., byte_offset=0, owner=None)`
  - `Tensor._unsafe_from_metal_buffer_fast(mtl_buffer_ptr, shape, dtype=..., byte_offset=0, owner=None)`

These helpers are intentionally private and unsafe.

They also currently pay Python marshalling overhead because the exporters return
Python dicts and the benchmark unpacks them before calling the import helper.

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

For the current MLX exporter, the array must also already be in MLX's
`available` state. The benchmark currently satisfies that with
`mx.array(np_array)`, which is a workaround for the current helper rather than a
claim that arbitrary lazy MLX outputs are already supported by the same path.

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
- `mx.array(memoryview(...))` is not an aliasing import path in current MLX.
  It goes through MLX's native CPU ndarray conversion path and copies the
  bytes.
- The first tinygrad import helper supports byte offsets.
- The first tinygrad import helper now optionally accepts `buffer_nbytes` for
  a bounds check. If that metadata is omitted, the helper still cannot prove
  the requested view fits inside the borrowed buffer.
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

- `MLX -> tinygrad` unsafe helper bridge returned correct values.
- `tinygrad -> MLX` unsafe helper bridge returned correct values.

Updated remote latency measurements for `float32` and `7168` bytes were:

- `unsafe_helper_bridge`
  - `mlx_to_tinygrad`: `20.241 us` min, `20.747 us` median
  - `tinygrad_to_mlx`: `26.738 us` min, `27.130 us` median
- `unsafe_helper_legacy`
  - `mlx_to_tinygrad`: `33.001 us` min, `33.353 us` median
- `memoryview_copy`
  - `mlx_to_tinygrad`: `36.328 us` min, `36.490 us` median
  - `tinygrad_to_mlx`: `2.535 us` min, `2.560 us` median
- `numpy_baseline`
  - `mlx_to_tinygrad`: `263.666 us` min, `265.771 us` median
  - `tinygrad_to_mlx`: `12.457 us` min, `12.549 us` median

Interpretation:

- The lower-overhead tinygrad import helper cut `MLX -> tinygrad` from about
  `33 us` to about `20 us` at `7 kB`, so the old `Tensor.empty(...)` based
  helper was a real source of overhead.
- The current `MLX -> tinygrad` path is still above the target `1-10 us`
  range for a `7 kB` tensor.
- `MLX -> tinygrad` is now dominated by the remaining tinygrad import / wrapper
  creation cost, not by MLX export or Python dict marshalling.
- `tinygrad -> MLX` currently has a very cheap copy path because `mx.array()`
  over a tinygrad `memoryview` is implemented efficiently in MLX's native C++
  import path, even though it still copies.
- At this tensor size, Python call overhead and wrapper construction matter
  much more than raw byte movement.
- These numbers do not establish that "aliasing costs ~20 us". They establish
  that the current lower-overhead asymmetric helper stack costs that much.

Additional remote microbench sweep on `e16` for `256`, `7168`, `65536`, and
`1048576` bytes showed:

- `MLX -> tinygrad`
  - `unsafe_helper_bridge`: roughly `20-21 us`
  - `unsafe_helper_legacy`: roughly `32-34 us`
  - `memoryview_copy`: roughly `35-52 us`
  - `numpy_baseline`: roughly `264-298 us`
  - `export_helper_only`: roughly `0.55 us`
  - `import_helper_fast_only`: roughly `20-21 us`
  - `import_helper_legacy_only`: roughly `32-33 us`
- `tinygrad -> MLX`
  - `unsafe_helper_bridge`: roughly `27 us`
  - `memoryview_copy`: roughly `2.4 us` at `256 B`, `2.6 us` at `7168 B`,
    `3.6 us` at `64 KiB`, and `17.4 us` at `1 MiB`
  - `numpy_baseline`: roughly `12.1 us` at `256 B`, `12.5 us` at `7168 B`,
    `16.4 us` at `64 KiB`, and `55.1 us` at `1 MiB`
  - `export_helper_only`: roughly `23-24 us`
  - `import_helper_only`: roughly `2.2 us`

What this means:

- The MLX exporter is already cheap. Replacing its Python dict with a tuple or
  capsule is unlikely to change `MLX -> tinygrad` materially, because the
  dominant cost is tinygrad import / wrapper creation.
- The MLX importer from raw pointer is also already cheap.
- The lower-overhead tinygrad import helper bought a real speedup, but the
  expensive pieces are still both on the tinygrad side:
  - importing a borrowed `MTLBuffer*` into a new tinygrad `Tensor`
  - exporting tinygrad storage metadata through the current Python helper
- For `tinygrad -> MLX`, the native copy path is already in the desired latency
  class for small tensors and remains competitive well past `7 kB`.
- For `MLX -> tinygrad`, the new lower-overhead helper is materially better
  than the old helper and better than copy, but still not close to the desired
  `1-10 us` range at `7 kB`.

## Near-Term Plan

1. Treat `tinygrad -> MLX memoryview_copy` as the current practical fast path.
2. If `MLX -> tinygrad` must get materially faster, focus on a lower-level
   tinygrad import constructor, wrapper reuse, or non-standard reusable wrapper
   design rather than MLX-side marshalling changes.
3. Avoid spending time on symmetry unless it becomes necessary for a specific
   downstream use case.

## Open Questions

- Whether the first fast path should support contiguous slices with byte
  offsets, or only base-contiguous tensors.
- Whether a lower-level tinygrad import path can cut `MLX -> tinygrad` wrapper
  creation overhead enough to matter at `~7 kB`.
