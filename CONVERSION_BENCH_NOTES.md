# Tensor Conversion Benchmark Notes

## Current Goal

Benchmark raw tinygrad `<->` MLX tensor transformation latency on Apple Silicon
for tensors that are already:

- synchronized
- allocated
- materialized / realized

The timed region should keep source creation and explicit synchronization
outside the loop, while making it clear when helper, binding, owner-pinning,
and wrapper-construction overhead are still inside it.

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
  - `mx.metal._unsafe_to_tinygrad_fast(array, tg_dtype, owner=None)`
  - `mx.metal._unsafe_rebind_tinygrad(array, borrower, owner=None)`
  - `mx.metal._unsafe_array_from_ptr(raw_ptr, shape, dtype, owner=None)`
  - `mx.metal._unsafe_array_from_ptr_alias_only(raw_ptr, shape, dtype, owner=None)`
- tinygrad
  - `Tensor._unsafe_metal_storage()`
  - `Tensor._unsafe_from_metal_buffer(mtl_buffer_ptr, shape, dtype=..., byte_offset=0, owner=None)`
  - `Tensor._unsafe_from_metal_buffer_fast(mtl_buffer_ptr, shape, dtype=..., byte_offset=0, owner=None)`
  - `Tensor._unsafe_metal_borrower(mtl_buffer_ptr, shape, dtype=..., byte_offset=0, owner=None)`

These helpers are intentionally private and unsafe.

The current benchmark still pays Python and binding overhead in several rows.
The older helper rows return Python dicts and unpack them before calling the
import helper, while the newer MLX-side single-entry rows still include the
tinygrad-side wrapper construction they trigger.

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
- MLX export now distinguishes:
  - `logical_nbytes`: the logical bytes in the exported array view
  - `buffer_nbytes`: the backing buffer capacity
- Offsetted MLX views must use `buffer_nbytes` semantics for bounds checks.
- The legacy MLX export field `nbytes` was removed to avoid accidental use of
  logical-size semantics where backing-buffer-size semantics are required.
- The first MLX import helper is raw-pointer based rather than foreign
  `MTLBuffer*` based.
- `mx.metal._unsafe_array_from_ptr(...)` may still copy if MLX cannot alias the
  pointer directly.
- `mx.metal._unsafe_array_from_ptr_alias_only(...)` fails instead of silently
  copying, so it is the right helper for proving aliasing in benchmarks.
- `mx.metal._unsafe_to_tinygrad_fast(...)` is a single MLX binding entrypoint
  for `MLX -> tinygrad`, but it still includes tinygrad-side tensor creation.
- `Tensor._unsafe_metal_borrower(...)` is a mutable slot primitive. It reuses
  the same tinygrad tensor wrapper and rebinds its borrowed `MTLBuffer*`.
  Older references to that tensor are not snapshots.
- The borrower now updates its internal `external_ptr` metadata on rebind so
  the stored buffer metadata matches the live Metal handle.
- `mx.metal._unsafe_export_storage(...)` currently expects an MLX array that is
  already in the C++ `available` state. In practice, `mx.array(np_array)` met
  that precondition for local smoke testing, while `mx.arange(...)` did not.
- `tinygrad -> MLX memoryview_copy` also includes per-call runtime ceremony,
  because tinygrad's zero-copy Metal memoryview export synchronizes before
  exposing the buffer.
- Bench rows named `rebindable_slot_*` measure rebindable-slot cost, not fresh
  conversion cost.
- Bench rows named `borrower_ring*` rotate through multiple independent slots.
  They are intended to approximate a practical pool/ring design with fewer
  footguns than a single slot.
- Bench rows named `*_then_use_sum` measure rebinding or conversion followed by
  immediate tinygrad consumption through a realized reduction kernel.

## Current Findings

The unsafe bridge was validated through the repo-standard remote flow on `e16`:

1. `git pull --ff-only`
2. `nix develop`
3. `uv sync`
4. `uv run python mlx_tinygrad_interop/bench_raw_conversion.py ...`

The direct helpers worked in both directions:

- `MLX -> tinygrad` unsafe helper bridge returned correct values.
- `tinygrad -> MLX` unsafe helper bridge returned correct values.
- The expanded slot / ring tests passed on `e16`:
  - nonzero-offset MLX slice import
  - single-entry `MLX -> tinygrad`
  - mutable-slot rebinding semantics
  - slot metadata update on rebind
  - two-slot independence until a slot is reused

Updated remote latency measurements for `float32` and `7168` bytes were:

- `unsafe_helper_bridge`
  - `mlx_to_tinygrad`: `21.202 us` min, `21.532 us` median
  - `tinygrad_to_mlx`: `28.109 us` min, `28.372 us` median
- `single_entry_bridge`
  - `mlx_to_tinygrad`: `21.388 us` min, `21.542 us` median
- `fresh_wrapper_then_use_sum`
  - `mlx_to_tinygrad`: `601.812 us` min, `611.458 us` median
- `rebindable_slot_bridge`
  - `mlx_to_tinygrad`: `1.505 us` min, `1.542 us` median
- `rebindable_slot_then_use_sum`
  - `mlx_to_tinygrad`: `579.583 us` min, `581.833 us` median
- `borrower_ring4_bridge`
  - `mlx_to_tinygrad`: `1.531 us` min, `1.573 us` median
- `borrower_ring4_then_use_sum`
  - `mlx_to_tinygrad`: `577.730 us` min, `581.000 us` median
- `unsafe_helper_legacy`
  - `mlx_to_tinygrad`: `31.938 us` min, `32.214 us` median
- `unsafe_helper_maybe_copy`
  - `tinygrad_to_mlx`: `28.153 us` min, `28.277 us` median
- `memoryview_copy`
  - `mlx_to_tinygrad`: `35.191 us` min, `35.668 us` median
  - `tinygrad_to_mlx`: `2.596 us` min, `2.662 us` median
- `numpy_baseline`
  - `mlx_to_tinygrad`: `272.323 us` min, `275.104 us` median
  - `tinygrad_to_mlx`: `12.817 us` min, `13.005 us` median

Interpretation:

- The lower-overhead tinygrad import helper cut `MLX -> tinygrad` from about
  `32 us` to about `21 us` at `7 kB`, so the old `Tensor.empty(...)` based
  helper was a real source of overhead.
- Replacing the exporter dict/unpack stack with a single MLX binding entrypoint
  still barely moved `MLX -> tinygrad`: about `21.2 us -> 21.4 us` at `7 kB`.
- That means the remaining fixed cost was not materially in MLX export or
  Python exporter marshalling. It was overwhelmingly on the tinygrad side.
- Rebinding a single mutable slot dropped `MLX -> tinygrad` to about `1.5 us`
  at `7 kB`. Rotating through a ring of four slots landed at about `1.53 us`,
  so the ring preserved essentially all of the latency win.
- Those slot / ring rows are not fresh conversion results. They are
  rebind-and-return-slot results.
- That is still inside the target range and strongly indicates that tinygrad
  wrapper construction, not storage adoption itself, was the dominant cost.
- On `e16`, the strict alias-only `tinygrad -> MLX` helper succeeded. Its
  timings were effectively the same as the maybe-copy helper, so the benchmark
  can now report a proven aliasing path in that direction on this host.
- `tinygrad -> MLX` currently has a very cheap copy path because `mx.array()`
  over a tinygrad `memoryview` is implemented efficiently in MLX's native C++
  import path, even though it still copies.
- At this tensor size, Python call overhead and wrapper construction matter
  much more than raw byte movement.
- The `*_then_use_sum` rows are dominated by the tinygrad reduction kernel
  itself. They are useful as end-to-end "convert then immediately consume"
  probes, not as pure conversion timings.
- Those end-to-end rows still show the same direction: at `7 kB`, slot / ring
  rebinding saves roughly `20-25 us` versus the fresh-wrapper path, but the
  absolute runtime is around `580-600 us` because the reduction dominates.
- These numbers do not establish that "aliasing costs ~21-28 us". They
  establish that creating a fresh tinygrad wrapper through the current helper
  stack costs that much, while rebinding a pre-existing slot costs about
  `1.5 us` on this host.
- An offsetted MLX slice was also validated through the new export semantics:
  `offset_bytes=64`, `logical_nbytes=7168`, `buffer_nbytes=16384`, and the
  borrowed tinygrad tensor matched the expected values.
- The rebindable slot is narrower than a normal conversion helper:
  - it returns the same tinygrad `Tensor` object each time
  - it assumes fixed shape / dtype / byte-offset semantics
  - older references are not snapshots
  - it is therefore best understood as a dangerous but very informative lower
    bound and a candidate building block for a specialized converter API
- A ring of multiple slots is the more practical extension of that idea because
  it preserves most of the latency win while reducing the worst single-slot
  footgun.

Additional remote microbench sweep on `e16` for `256`, `7168`, `65536`, and
`1048576` bytes showed:

- `MLX -> tinygrad`
  - `unsafe_helper_bridge`: roughly `21-23 us`
  - `single_entry_bridge`: roughly `21-22 us`
  - `fresh_wrapper_then_use_sum`: roughly `601-693 us`
  - `rebindable_slot_bridge`: roughly `1.49-1.53 us`
  - `rebindable_slot_then_use_sum`: roughly `573-670 us`
  - `borrower_ring4_bridge`: roughly `1.53-1.62 us`
  - `borrower_ring4_then_use_sum`: roughly `566-669 us`
  - `unsafe_helper_legacy`: roughly `31-33 us`
  - `memoryview_copy`: roughly `34-51 us`
  - `numpy_baseline`: roughly `269-309 us`
  - `export_helper_only`: roughly `0.58-0.63 us`
  - `import_helper_fast_only`: roughly `20.2-21.4 us`
  - `rebindable_slot_import_only`: roughly `1.40-1.47 us`
  - `borrower_ring4_import_only`: roughly `1.45-1.49 us`
  - `import_helper_legacy_only`: roughly `31-33 us`
- `tinygrad -> MLX`
  - `unsafe_helper_bridge`: roughly `27-29 us`
  - `unsafe_helper_maybe_copy`: roughly `27-29 us`
  - `memoryview_copy`: roughly `2.5 us` at `256 B`, `2.5 us` at `7168 B`,
    `3.5 us` at `64 KiB`, and `17.4 us` at `1 MiB`
  - `numpy_baseline`: roughly `12.7 us` at `256 B`, `12.7 us` at `7168 B`,
    `16.6 us` at `64 KiB`, and `56.7 us` at `1 MiB`
  - `export_helper_only`: roughly `23.1-24.4 us`
  - `import_helper_only`: roughly `2.12-2.22 us`
  - `import_helper_maybe_copy_only`: roughly `2.16-2.29 us`

What this means:

- The MLX exporter is already cheap, and even a single MLX binding entrypoint
  did not change `MLX -> tinygrad` materially. That closes out the
  "Python exporter ceremony" hypothesis for the current bridge.
- The MLX importer from raw pointer is also already cheap, whether measured in
  strict alias-only mode or maybe-copy mode on this host.
- The lower-overhead tinygrad import helper bought a real speedup, but the
  expensive piece for `MLX -> tinygrad` was still constructing a fresh tinygrad
  wrapper around the borrowed storage.
- Rebinding a slot or rotating through a ring changes the latency class
  completely. The "rebind one pre-existing slot" lower bound is about
  `1.5 us` on this host for the measured sizes, and a ring of four slots keeps
  essentially the same latency.
- For `tinygrad -> MLX`, the native copy path is already in the desired latency
  class for small tensors and remains competitive well past `7 kB`.
- For `MLX -> tinygrad`, a fresh-wrapper helper is still not close to the
  desired `1-10 us` range at `7 kB`, but a slot / ring helper is.
- For end-to-end "convert then immediately use" measurements, the tinygrad
  compute dominates. The relevant signal is the delta versus the fresh-wrapper
  path, not the absolute `~580-700 us` number.

## Near-Term Plan

1. Treat `tinygrad -> MLX memoryview_copy` as the current practical fast path.
2. Treat `MLX -> tinygrad` slot/ring rebinding as the current latency floor and
   likely practical fast path when rebindable slot semantics are acceptable.
3. Prefer a ring/pool of slots over a single slot for any practical design.
4. If `MLX -> tinygrad` must return a fresh tinygrad tensor each time and still
   stay under `10 us`, the remaining work is entirely on the tinygrad-side
   construction path.
5. Avoid spending time on symmetry unless it becomes necessary for a specific
   downstream use case.

## Open Questions

- Whether the first fast path should support contiguous slices with byte
  offsets, or only base-contiguous tensors.
- Whether the slot/ring primitive should stay benchmark-only or be surfaced as
  a deliberate specialized converter API.
