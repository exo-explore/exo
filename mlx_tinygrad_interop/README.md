# MLX Tinygrad Interop

Private code for benchmarking MLX `<->` tinygrad tensor conversions.

## Workflow

Use the repo devshell and top-level dependency graph. Do not install ad-hoc
build dependencies or patch around them with one-off environment setups.

1. Change code locally.
2. Push the `mlx` and `tinygrad` fork changes.
3. In local `exo`, enter the devshell with `nix develop`.
4. Refresh `uv.lock` against the new fork heads with:
   `uv lock --upgrade-package mlx --refresh-package mlx --upgrade-package tinygrad --refresh-package tinygrad`
5. Commit and push the updated `exo` branch.
6. On the remote Mac, pull the updated repos.
7. Enter the devshell with `nix develop`.
8. Refresh the environment with `uv sync`.
9. Run tests or benchmarks with `uv run ...`.

Plain `uv lock` was not enough to move these git-based dependency SHAs during
testing, and `--upgrade-package` alone still left one stale git revision in a
later pass. The working command was the explicit `--upgrade-package` plus
`--refresh-package` form above.

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
  - a rebindable tinygrad slot that reuses one wrapper and rebinds the
    borrowed `MTLBuffer*`
  - a small ring of such slots
  - a keyed lease pool that owns `mx.eval(...)` on acquire and explicit lease
    release on the tinygrad side
  - `*_then_use_sum` rows that immediately consume the converted tensor through
    a realized tinygrad reduction
- `mx.array(memoryview(...))` is a native copy path in current MLX, not an
  aliasing import path.
- Rebindable slots enforce fixed shape/dtype contracts on rebind.
- Do not rebind a slot until all work derived from its previous contents has
  been realized and synchronized.
- The practical `MLX -> tinygrad` path is now a lease-managed pool keyed by
  `(shape, dtype, byte_offset)`, not a bare mutable borrower.
- This fast path is only valid for same-process, same-address-space Apple
  Silicon unified-memory handoff. It does not cross process or machine
  boundaries, and it does not remove any later Metal/host -> CUDA transfer.

Test command:

```bash
uv run python -m unittest mlx_tinygrad_interop.test_interop mlx_tinygrad_interop.test_handoff
```

Stress command:

```bash
uv run python mlx_tinygrad_interop/stress_interop.py --cases 64 --soak-iterations 512
```

Example:

```bash
uv run python mlx_tinygrad_interop/bench_raw_conversion.py --dtype float32 --sizes 256,512,1024,2048,4096,7168
```

Validated remote command on `e16`:

```bash
uv run python mlx_tinygrad_interop/bench_raw_conversion.py --dtype float32 --sizes 7168 --warmup 64 --samples 7 --min-batch-us 1000
```

Observed `7168`-byte results on that run:

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

Later remote microbench runs showed the split more clearly:

- `MLX -> tinygrad single_entry_bridge` barely changes the fresh-wrapper cost,
  so exporter dict marshalling was never the main problem.
- `MLX -> tinygrad` is dominated by tinygrad import / wrapper construction when
  a fresh tensor is created each time.
- The rebindable tinygrad slot drops `MLX -> tinygrad` to about `1.5 us`, and
  a ring of four slots stays at essentially the same latency. That means
  wrapper reuse, not exporter marshalling, is the decisive optimization on this
  host.
- After adding hardened shape/dtype contract checks, a spot-check at `7168`
  bytes moved those rows to about `2.46 us` for the single slot and `2.55 us`
  for the ring. That is still comfortably inside the target latency range.
- The strict alias-only `tinygrad -> MLX` helper succeeds on `e16`; its timing
  is effectively the same as the maybe-copy helper on that host.
- `tinygrad -> MLX` is dominated by tinygrad export in the unsafe helper path.
- `tinygrad -> MLX memoryview_copy` is already the practical low-latency path
  for small tensors.
- Offsetted MLX slices now export both logical bytes and backing-buffer bytes,
  and a nonzero-offset slice was validated successfully into tinygrad.
- The randomized stress suite also caught and fixed the zero-offset variant of
  that problem: oversized backing buffers now import through a logical tinygrad
  buffer view instead of reshaping the entire backing allocation.
- The same stress suite also found that MLX backing-buffer capacity is a raw
  byte count, not necessarily a dtype-aligned element count. The fast tinygrad
  path now handles that with byte-level bounds checks and `ceildiv`.
- The rebindable slot returns the same tinygrad `Tensor` object rebound to new
  Metal storage, so it is narrower than an ordinary "new tensor each call"
  conversion helper.
- The current ring rows are still benchmark primitives, not a production lease
  API. If this path is used in the real disaggregated MLX/tinygrad runtime, it
  should grow into an explicit pool/lease abstraction with generation
  semantics.
- The `*_then_use_sum` rows are dominated by the realized tinygrad reduction.
  They should be read as end-to-end "convert then immediately consume" probes.
  They still show the same relative story: slot/ring rebinding saves about
  `20-25 us` versus the fresh-wrapper path at `7 kB`.

## Current Scope

- Private / unsafe helpers only.
- Metal / unified-memory path only.
- Dense contiguous tensors only.
- Same-dtype conversions only.
- Current exporter/importer microbenchmarks are intended to separate helper
  overhead from end-to-end bridge cost.
