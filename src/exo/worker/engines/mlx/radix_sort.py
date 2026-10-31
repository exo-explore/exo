"""Ascending sort of float32 rows on the GPU, by least-significant-digit radix sort.

MLX sorts on the GPU with a multi-block merge sort, which at a 100k-250k vocabulary costs 100-200
us per row on Apple GPUs. This is a radix sort of the same rows: four passes over 8-bit digits of
an order-preserving key, each a tile histogram, a prefix scan of the histograms and a stable
scatter. The sorted values are the same as ``mx.sort``'s, except that -0.0 and 0.0, which compare
equal, may come out in a different order between themselves, and NaN, sorted last by both, comes
out as one NaN bit pattern.
"""

from functools import cache
from typing import Final

import mlx.core as mx

_TILE: Final = 4096
_TILE_THREADS: Final = 256
_SCAN_THREADS: Final = 1024
_DIGITS: Final = 256

_HEADER = """
// A uint that orders like the float: NaN last, as in mx.sort.
inline uint sort_key(float f) {
  if (metal::isnan(f)) {
    return 0xFFFFFFFFu;
  }
  uint u = as_type<uint>(f);
  return (u & 0x80000000u) ? ~u : (u | 0x80000000u);
}

inline uint sort_key(uint k) {
  return k;
}

inline void store_key(device uint* out, size_t i, uint k) {
  out[i] = k;
}

inline void store_key(device float* out, size_t i, uint k) {
  out[i] = as_type<float>((k & 0x80000000u) ? (k ^ 0x80000000u) : ~k);
}

inline uint ballot(bool b) {
  return static_cast<uint>(static_cast<simd_vote::vote_t>(simd_ballot(b)));
}
"""

# One threadgroup per (tile, row): how many keys of the tile have each value of the digit.
# Written digit-major, so that a prefix sum over the row gives each tile's first slot per digit.
_HISTOGRAM_SOURCE = """
  constexpr int NT = TILE_THREADS;
  threadgroup atomic_uint hist[256];
  const uint tid = thread_index_in_threadgroup;
  const uint tile = threadgroup_position_in_grid.x;
  const uint row = threadgroup_position_in_grid.y;
  for (uint d = tid; d < 256; d += NT) {
    atomic_store_explicit(&hist[d], 0u, memory_order_relaxed);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  const uint start = tile * TILE;
  const uint end = min(start + uint(TILE), uint(V));
  auto src = keys_in + size_t(row) * V;
  for (uint i = start + tid; i < end; i += NT) {
    uint d = (sort_key(src[i]) >> SHIFT) & 255u;
    atomic_fetch_add_explicit(&hist[d], 1u, memory_order_relaxed);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  auto dst = counts + size_t(row) * 256 * NTILES;
  for (uint d = tid; d < 256; d += NT) {
    dst[d * NTILES + tile] = atomic_load_explicit(&hist[d], memory_order_relaxed);
  }
"""

# One threadgroup per row: exclusive prefix sum of the M = 256 * NTILES histogram entries.
_SCAN_SOURCE = """
  constexpr int NT = SCAN_THREADS;
  threadgroup uint sg_totals[NT / 32];
  const uint tid = thread_index_in_threadgroup;
  const uint lane = thread_index_in_simdgroup;
  const uint sg = simdgroup_index_in_threadgroup;
  const uint row = threadgroup_position_in_grid.y;
  auto c = counts + size_t(row) * M;
  auto o = offsets + size_t(row) * M;
  const uint per = (uint(M) + NT - 1) / NT;
  const uint begin = min(tid * per, uint(M));
  const uint stop = min(begin + per, uint(M));
  uint local = 0;
  for (uint i = begin; i < stop; ++i) {
    local += c[i];
  }
  uint incl = simd_prefix_inclusive_sum(local);
  if (lane == 31) {
    sg_totals[sg] = incl;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (sg == 0) {
    uint t = lane < uint(NT / 32) ? sg_totals[lane] : 0u;
    uint ex = simd_prefix_exclusive_sum(t);
    if (lane < uint(NT / 32)) {
      sg_totals[lane] = ex;
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  uint run = incl - local + sg_totals[sg];
  for (uint i = begin; i < stop; ++i) {
    uint v = c[i];
    o[i] = run;
    run += v;
  }
"""

# One threadgroup per (tile, row): writes every key of the tile to its slot, keeping the tile's
# order among keys with the same digit. Each simdgroup takes a contiguous segment of the tile in
# rounds of 32 keys; a key's rank among its digit's keys is the count in earlier segments, plus
# earlier rounds of its own segment, plus lower lanes of its own round.
_SCATTER_SOURCE = """
  constexpr int NT = TILE_THREADS;
  constexpr int NSG = NT / 32;
  constexpr int ROUNDS = TILE / NT;
  threadgroup uint sg_count[NSG * 256];
  threadgroup uint tile_offset[256];
  const uint tid = thread_index_in_threadgroup;
  const uint lane = thread_index_in_simdgroup;
  const uint sg = simdgroup_index_in_threadgroup;
  const uint tile = threadgroup_position_in_grid.x;
  const uint row = threadgroup_position_in_grid.y;
  for (uint i = tid; i < uint(NSG * 256); i += NT) {
    sg_count[i] = 0;
  }
  auto offs = offsets + size_t(row) * 256 * NTILES;
  for (uint d = tid; d < 256; d += NT) {
    tile_offset[d] = offs[d * NTILES + tile];
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  const uint start = tile * TILE;
  const uint len = min(uint(TILE), uint(V) - start);
  auto src = keys_in + size_t(row) * V;
  const uint below = (1u << lane) - 1u;
  uint keys[ROUNDS];
  uint ranks[ROUNDS];
  for (int r = 0; r < ROUNDS; ++r) {
    const uint i = sg * (ROUNDS * 32) + r * 32 + lane;
    const bool valid = i < len;
    const uint key = valid ? sort_key(src[start + i]) : 0u;
    const uint d = (key >> SHIFT) & 255u;
    uint peers = ballot(valid);
    for (int b = 0; b < 8; ++b) {
      const bool bit = (d >> b) & 1u;
      const uint vote = ballot(bit);
      peers &= bit ? vote : ~vote;
    }
    const uint before = popcount(peers & below);
    const uint base = sg_count[sg * 256 + d];
    simdgroup_barrier(mem_flags::mem_threadgroup);
    if (valid && before == 0) {
      sg_count[sg * 256 + d] = base + popcount(peers);
    }
    simdgroup_barrier(mem_flags::mem_threadgroup);
    keys[r] = key;
    ranks[r] = valid ? ((d << 16) | (base + before)) : 0xFFFFFFFFu;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (tid < 256) {
    uint run = 0;
    for (int s = 0; s < NSG; ++s) {
      uint c = sg_count[s * 256 + tid];
      sg_count[s * 256 + tid] = run;
      run += c;
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  auto dst = out + size_t(row) * V;
  for (int r = 0; r < ROUNDS; ++r) {
    const uint packed = ranks[r];
    if (packed != 0xFFFFFFFFu) {
      const uint d = packed >> 16;
      const uint pos = tile_offset[d] + sg_count[sg * 256 + d] + (packed & 0xFFFFu);
      store_key(dst, pos, keys[r]);
    }
  }
"""


@cache
def _kernels() -> tuple[
    "mx.fast.MetalKernel", "mx.fast.MetalKernel", "mx.fast.MetalKernel"
]:
    histogram = mx.fast.metal_kernel(
        name="exo_radix_histogram",
        input_names=["keys_in"],
        output_names=["counts"],
        source=_HISTOGRAM_SOURCE,
        header=_HEADER,
    )
    scan = mx.fast.metal_kernel(
        name="exo_radix_scan",
        input_names=["counts"],
        output_names=["offsets"],
        source=_SCAN_SOURCE,
        header=_HEADER,
    )
    scatter = mx.fast.metal_kernel(
        name="exo_radix_scatter",
        input_names=["keys_in", "offsets"],
        output_names=["out"],
        source=_SCATTER_SOURCE,
        header=_HEADER,
    )
    return histogram, scan, scatter


def can_sort_rows(x: mx.array) -> bool:
    return mx.metal.is_available() and x.dtype == mx.float32 and x.ndim == 2


def sort_rows(x: mx.array) -> mx.array:
    """mx.sort(x, axis=-1) for [rows, n] float32 when can_sort_rows(x), up to the order of
    -0.0 and 0.0 between themselves and NaN bit patterns."""
    rows, n = x.shape
    histogram, scan, scatter = _kernels()
    n_tiles = (n + _TILE - 1) // _TILE
    entries = _DIGITS * n_tiles
    keys = x
    for shift in (0, 8, 16, 24):
        (counts,) = histogram(
            inputs=[keys],
            output_shapes=[(rows, entries)],
            output_dtypes=[mx.uint32],
            grid=(n_tiles * _TILE_THREADS, rows, 1),
            threadgroup=(_TILE_THREADS, 1, 1),
            template=[
                ("V", n),
                ("SHIFT", shift),
                ("NTILES", n_tiles),
                ("TILE", _TILE),
                ("TILE_THREADS", _TILE_THREADS),
            ],
        )
        (offsets,) = scan(
            inputs=[counts],
            output_shapes=[(rows, entries)],
            output_dtypes=[mx.uint32],
            grid=(_SCAN_THREADS, rows, 1),
            threadgroup=(_SCAN_THREADS, 1, 1),
            template=[("M", entries), ("SCAN_THREADS", _SCAN_THREADS)],
        )
        (keys,) = scatter(
            inputs=[keys, offsets],
            output_shapes=[(rows, n)],
            output_dtypes=[mx.float32 if shift == 24 else mx.uint32],
            grid=(n_tiles * _TILE_THREADS, rows, 1),
            threadgroup=(_TILE_THREADS, 1, 1),
            template=[
                ("V", n),
                ("SHIFT", shift),
                ("NTILES", n_tiles),
                ("TILE", _TILE),
                ("TILE_THREADS", _TILE_THREADS),
            ],
        )
    return keys
