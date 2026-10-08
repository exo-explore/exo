"""Sparse DeepSeek V4 attention for prefill.

During prefill mlx_lm attends every query to every key of the chunk: the
sliding-window keys (``window`` 128, but the chunk's whole key range) and the
whole compressed pool, then masks most of them out. For ratio-4 layers it
also builds the Indexer's top-k mask with a one-hot comparison of shape
``[S, top_k, pool]``. With 4096-token chunks and long contexts nearly all of
that work is masked away.

The kernel here visits only the keys each query can see: its window of
``window`` raw keys and the visible compressed rows (only the Indexer's
selected rows on ratio-4 layers). Scores, softmax and the weighted sum stay in
float32, so the result is the same attention as mlx_lm's masked SDPA, with
less rounding. Decode, per-row offsets and unsupported shapes keep the
existing path.
"""

from typing import cast

import mlx.core as mx
import mlx.nn as nn
from mlx_lm.models.cache import RotatingKVCache
from mlx_lm.models.deepseek_v4 import (
    _K_COMP,  # pyright: ignore[reportPrivateUsage]
    _K_IDX,  # pyright: ignore[reportPrivateUsage]
    DeepseekV4Cache,
    V4Attention,
    _attn_inv_rope_flatten,  # pyright: ignore[reportPrivateUsage]
    _attn_q_proj_norm,  # pyright: ignore[reportPrivateUsage]
    _attn_qkv_partial_rope,  # pyright: ignore[reportPrivateUsage]
    _attn_qkv_split_norm,  # pyright: ignore[reportPrivateUsage]
)

from exo.worker.engines.mlx.patches.deepseek_v4_decode_kernels import (
    MAX_DECODE_SEQUENCE,
)

# Keys staged in threadgroup memory per step, heads per threadgroup, and heads
# per simdgroup: every head reads the same single KV head, so each staged key
# value is used for several heads per load.
_TILE = 8
_HEADS_PER_THREADGROUP = 16
_HEADS_PER_SIMDGROUP = 2

# One threadgroup per (batch, query, block of heads). q and out are
# [B, H, S, D]. Window row r holds raw position offset + S - window_len + r;
# compressed row c is visible once its ratio-token window is complete.
_PREFILL_ATTENTION_SOURCE = """
    uint threadgroup_index = threadgroup_position_in_grid.x;
    uint head_block = threadgroup_index % HEAD_BLOCKS;
    uint query_row = threadgroup_index / HEAD_BLOCKS;
    int s = int(query_row % SEQUENCE);
    uint batch = query_row / SEQUENCE;
    uint lane = thread_index_in_simdgroup;
    uint first_head = head_block * HEADS + simdgroup_index_in_threadgroup * HPS;
    uint tid = thread_position_in_threadgroup.x;
    constexpr uint PER_LANE = D / 32;
    constexpr uint THREADS = HEADS / HPS * 32;

    int offset = params[0];
    int window_len = params[1];
    int window = params[2];
    int ratio = params[3];
    int compressed = params[4];
    int topk_count = params[5];

    int position = offset + s;
    int window_hi = position - (offset + SEQUENCE - window_len);
    int window_lo = metal::max(0, window_hi - window + 1);
    int window_count = window_hi - window_lo + 1;
    int compressed_visible = ratio > 0
        ? metal::min(compressed, (position + 1) / ratio)
        : compressed;
    int total = window_count + (HAS_TOPK ? topk_count : compressed_visible);

    threadgroup OUT_T tile[TILE * D];
    threadgroup int rows[TILE];

    float query[HPS][PER_LANE];
    float acc[HPS][PER_LANE];
    float running_max[HPS];
    float running_sum[HPS];
    for (uint j = 0; j < HPS; ++j) {
        uint q_row = ((batch * H + first_head + j) * SEQUENCE + s) * D;
        for (uint i = 0; i < PER_LANE; ++i) {
            query[j][i] = static_cast<float>(q[q_row + i * 32 + lane]) * scale[0];
            acc[j][i] = 0.0f;
        }
        running_max[j] = -INFINITY;
        running_sum[j] = 0.0f;
    }

    for (int t0 = 0; t0 < total; t0 += TILE) {
        if (tid < TILE) {
            int t = t0 + int(tid);
            int r = -1;
            if (t < window_count) {
                r = window_lo + t;
            } else if (t < total) {
                int c = t - window_count;
                if (HAS_TOPK) {
                    c = topk[query_row * topk_count + c];
                    if (c >= compressed_visible) {
                        c = -1;
                    }
                }
                r = c < 0 ? -1 : window_len + c;
            }
            rows[tid] = r;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint e = tid; e < TILE * D / 4; e += THREADS) {
            uint k = e / (D / 4);
            uint d4 = e % (D / 4);
            int r = rows[k];
            vec<OUT_T, 4> v = vec<OUT_T, 4>(0);
            if (r >= 0 && r < window_len) {
                v = ((const device vec<OUT_T, 4>*)(window_kv + (batch * window_len + r) * D))[d4];
            } else if (r >= window_len) {
                v = ((const device vec<OUT_T, 4>*)(compressed_kv + (batch * compressed + r - window_len) * D))[d4];
            }
            ((threadgroup vec<OUT_T, 4>*)(tile + k * D))[d4] = v;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        float scores[HPS][TILE];
        float tile_max[HPS];
        for (uint j = 0; j < HPS; ++j) {
            tile_max[j] = -INFINITY;
        }
        for (uint k = 0; k < TILE; ++k) {
            float dot[HPS];
            for (uint j = 0; j < HPS; ++j) {
                dot[j] = 0.0f;
            }
            for (uint i = 0; i < PER_LANE; ++i) {
                float key = static_cast<float>(tile[k * D + i * 32 + lane]);
                for (uint j = 0; j < HPS; ++j) {
                    dot[j] += query[j][i] * key;
                }
            }
            bool visible = rows[k] >= 0;
            for (uint j = 0; j < HPS; ++j) {
                float score = visible ? simd_sum(dot[j]) : -INFINITY;
                scores[j][k] = score;
                tile_max[j] = metal::max(tile_max[j], score);
            }
        }
        for (uint j = 0; j < HPS; ++j) {
            float new_max = metal::max(running_max[j], tile_max[j]);
            float correction = metal::exp(running_max[j] - new_max);
            running_sum[j] *= correction;
            for (uint i = 0; i < PER_LANE; ++i) {
                acc[j][i] *= correction;
            }
            running_max[j] = new_max;
        }
        for (uint k = 0; k < TILE; ++k) {
            float weight[HPS];
            for (uint j = 0; j < HPS; ++j) {
                weight[j] = metal::exp(scores[j][k] - running_max[j]);
                running_sum[j] += weight[j];
            }
            for (uint i = 0; i < PER_LANE; ++i) {
                float value = static_cast<float>(tile[k * D + i * 32 + lane]);
                for (uint j = 0; j < HPS; ++j) {
                    acc[j][i] += weight[j] * value;
                }
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    for (uint j = 0; j < HPS; ++j) {
        uint head = first_head + j;
        uint q_row = ((batch * H + head) * SEQUENCE + s) * D;
        float sink = static_cast<float>(sinks[head]);
        float total_max = metal::max(running_max[j], sink);
        float keys_scale = metal::exp(running_max[j] - total_max);
        float factor = keys_scale / (running_sum[j] * keys_scale + metal::exp(sink - total_max));
        for (uint i = 0; i < PER_LANE; ++i) {
            out[q_row + i * 32 + lane] = static_cast<OUT_T>(acc[j][i] * factor);
        }
    }
"""

_prefill_attention_kernel = mx.fast.metal_kernel(
    name="dsv4_prefill_attention",
    input_names=[
        "q",
        "window_kv",
        "compressed_kv",
        "topk",
        "sinks",
        "scale",
        "params",
    ],
    output_names=["out"],
    source=_PREFILL_ATTENTION_SOURCE,
)


def prefill_attention(
    q: mx.array,
    window_kv: mx.array,
    compressed: mx.array | None,
    topk: mx.array | None,
    sinks: mx.array,
    scale: float,
    offset: int,
    window: int,
    ratio: int,
) -> mx.array:
    """Sink-aware attention of q ``[B, H, S, D]`` (queries at positions
    ``offset .. offset + S - 1``) over one shared KV head: ``window_kv``
    ``[B, W, D]`` holds the raw keys ending at the last query, each query
    seeing the last ``window`` of them up to itself; ``compressed`` ``[B, C,
    D]`` rows are visible once their ``ratio``-token window is complete
    (``ratio`` 0: always), restricted to ``topk`` ``[B, S, K]`` when given.
    Returns ``[B, H, S, D]``."""
    batch, heads, sequence, head_dim = q.shape
    heads_per_threadgroup = (
        _HEADS_PER_THREADGROUP if heads % _HEADS_PER_THREADGROUP == 0 else heads
    )
    heads_per_simdgroup = (
        _HEADS_PER_SIMDGROUP if heads_per_threadgroup % _HEADS_PER_SIMDGROUP == 0 else 1
    )
    compressed_len = 0 if compressed is None else compressed.shape[1]
    topk_count = 0 if topk is None else topk.shape[-1]
    return _prefill_attention_kernel(
        inputs=[
            q,
            window_kv,
            compressed if compressed is not None else window_kv,
            topk if topk is not None else mx.zeros((1,), dtype=mx.int32),
            sinks,
            mx.array([scale], dtype=mx.float32),
            mx.array(
                [offset, window_kv.shape[1], window, ratio, compressed_len, topk_count],
                dtype=mx.int32,
            ),
        ],
        template=[
            ("H", heads),
            ("D", head_dim),
            ("HEADS", heads_per_threadgroup),
            ("HPS", heads_per_simdgroup),
            ("HEAD_BLOCKS", heads // heads_per_threadgroup),
            ("SEQUENCE", sequence),
            ("TILE", _TILE),
            ("HAS_TOPK", topk is not None),
            ("OUT_T", q.dtype),
        ],
        grid=(batch * sequence * heads // heads_per_simdgroup * 32, 1, 1),
        threadgroup=(heads_per_threadgroup // heads_per_simdgroup * 32, 1, 1),
        output_shapes=[q.shape],
        output_dtypes=[q.dtype],
    )[0]


def _prefill_supported(attention: V4Attention, x: mx.array, cache: object) -> bool:
    if x.shape[1] <= MAX_DECODE_SEQUENCE:
        return False
    if not isinstance(cache, DeepseekV4Cache):
        return False
    # Batched caches swap in a BatchRotatingKVCache with per-row offsets.
    local = cast(object, cache.local)
    if not isinstance(local, RotatingKVCache):
        return False
    if cache.pooled_lengths(_K_COMP) is not None:
        return False
    if attention.compress_ratio == 4 and cache.pooled_lengths(_K_IDX) is not None:
        return False
    for linear in (attention.wqkv_a, attention.wq_b, attention.wo_a, attention.wo_b):
        if isinstance(linear, nn.QuantizedLinear) and linear.mode == "mxfp4":
            return False
    heads, head_dim = attention.n_heads, attention.head_dim
    if head_dim % 128 != 0:
        return False
    return heads % _HEADS_PER_THREADGROUP == 0 or heads * 32 <= 1024


def _prefill_attention_call(
    self: V4Attention, x: mx.array, cache: DeepseekV4Cache
) -> mx.array:
    """``V4Attention.__call__`` for a uniform prefill chunk, with the masked
    SDPA replaced by :func:`prefill_attention`."""
    rope_dims = self.rope_head_dim
    qr, kv = _attn_qkv_split_norm(
        self.wqkv_a(x),
        self.q_norm.weight,
        self.kv_norm.weight,
        self.q_lora_rank,
        self.eps,
    )
    q = _attn_q_proj_norm(self.wq_b(qr), self.n_heads, self.head_dim, self.eps)
    win_cache = cache.local
    offset = win_cache.offset
    q, kv = _attn_qkv_partial_rope(q, kv, offset, rope_dims, self.rope.freqs)

    k4 = kv[:, None, :, :]
    window_keys, _ = cast(
        tuple[mx.array, mx.array],
        win_cache.update_and_fetch(k4, k4),  # pyright: ignore[reportUnknownMemberType]
    )
    window_kv = window_keys.squeeze(1)
    compressed: mx.array | None = None
    indexer_topk: mx.array | None = None
    if self.compress_ratio:
        self.compressor(x, cache, offset, key=_K_COMP)
        if self.compress_ratio == 4:
            indexer_topk = self.indexer(x, qr, cache, offset)
        compressed = cache.get_branch(_K_COMP).pool
        if compressed is not None and compressed.shape[1] == 0:
            compressed = None
    if compressed is None:
        indexer_topk = None

    o = prefill_attention(
        q,
        window_kv,
        compressed,
        indexer_topk,
        self._sink_for(q.dtype),
        self.scale,
        offset,
        self.window,
        self.compress_ratio,
    )
    o = _attn_inv_rope_flatten(
        o, offset, rope_dims, self.rope.freqs, self.n_heads * self.head_dim
    )
    return self.wo_b(self._grouped_output_projection(o))


def patch_deepseek_v4_prefill_attention() -> None:
    previous = V4Attention.__call__

    def patched(self: V4Attention, x: mx.array, cache: object = None) -> mx.array:
        if not _prefill_supported(self, x, cache):
            return previous(self, x, cache=cache)
        return _prefill_attention_call(self, x, cast(DeepseekV4Cache, cache))

    V4Attention.__call__ = patched
