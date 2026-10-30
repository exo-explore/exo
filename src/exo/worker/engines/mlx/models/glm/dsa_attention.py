# type: ignore
# basedpyright skips this file (MLX array code against untyped mlx-lm internals and
# mx.fast.metal_kernel, which the repo's mlx stubs don't cover); dsa_attention.pyi types
# what other modules use. Same arrangement as deepseek_v4/model.py and its .pyi.
"""GLM-5.x (DeepSeek sparse attention with a shared indexer) attention for exo: mlx-lm's
GlmMoeDsaAttention computed faster, and the indexer's head reduction in float32.

- Dense prefill scores the 192-wide no-rope part and the 64-wide rope part of each head in
  one 256-wide product with a boolean mask, instead of a separate rope-part product
  materialised as an additive float mask over every [head, query, key].
- Sparse (top-k) prefill starts at SPARSE_PREFILL_MIN_CONTEXT keys of context instead of
  mlx-lm's 16384, where it overtakes dense attention on a Mac Studio (M3 Ultra), and
  gathers the selected keys with 16-byte vector copies (kernels.gather_rows).
- Single-token decode treats the heads as the rows of one product with the shared latent
  keys (scores and values), instead of one matrix-vector product per head.
- The indexer's per-key score (relu(q.k) times each head's weight, summed over its heads)
  is summed in float32 in one pass (kernels.indexer_scores). mlx-lm adds the heads one at a
  time in bfloat16, which rounds 31 times and leaves many keys tied at the top-k boundary.

Where the Metal kernels can't run (no Metal GPU as the default device) every call goes to
mlx-lm's own implementation.
"""

from functools import partial
from typing import Any, cast

import mlx.core as mx
from mlx_lm.models.base import scaled_dot_product_attention
from mlx_lm.models.deepseek_v32 import Indexer
from mlx_lm.models.glm_moe_dsa import GlmMoeDsaAttention, ModelArgs

from exo.worker.engines.mlx.models.glm.kernels import (
    gather_rows,
    indexer_scores,
    metal_kernels_available,
)

# Context length (cached plus new keys) from which a single sequence's prefill chunk attends
# sparsely (gathering each query's top-k keys) rather than densely with a mask.
SPARSE_PREFILL_MIN_CONTEXT = 8192
SPARSE_QUERY_BLOCK = 1024


@partial(mx.compile, shapeless=True)
def _latent_softmax(
    latent_scores: mx.array, rope_scores: mx.array, mask: mx.array, scale: mx.array
) -> mx.array:
    scores = (latent_scores.astype(mx.float32) + rope_scores.astype(mx.float32)) * scale
    return mx.softmax(mx.where(mask, scores, -mx.inf), axis=-1)


def _indexer_inputs(
    indexer: Indexer, x: mx.array, qr: mx.array, cache: Any
) -> tuple[mx.array, mx.array, mx.array]:
    """mlx-lm's Indexer up to its scores: roped queries [B, H, L, D], the cached and new
    roped keys [B, 1, N, D], and the heads' weights [B, H, L, 1]."""
    batch, length, _ = x.shape
    q = indexer.wq_b(qr)
    q = q.reshape(batch, length, indexer.n_heads, indexer.head_dim).swapaxes(1, 2)
    k = indexer.k_norm(indexer.wk(x))
    k = mx.reshape(k, (batch, 1, length, indexer.head_dim))
    offset = cache.offset if cache is not None else 0
    q = indexer.rope(q, offset=offset)
    k = indexer.rope(k, offset=offset)
    if cache is not None:
        k, _ = cache.update_and_fetch(k, k)
        # As mlx-lm: keep the unused value buffer from chaining every step's graph.
        cache.values = mx.zeros_like(cache.keys)
    weights = indexer.weights_proj(x) * (indexer.n_heads**-0.5 * indexer.softmax_scale)
    return q, k, weights.swapaxes(-1, -2)[..., None]


def indexer_topk(
    indexer: Indexer, x: mx.array, qr: mx.array, mask: mx.array | None, cache: Any
) -> mx.array | None:
    """mlx-lm's Indexer.__call__ with the head reduction in float32."""
    q, k, weights = _indexer_inputs(indexer, x, qr, cache)
    if k.shape[2] <= indexer.index_topk:
        return None
    scores = indexer_scores(q @ k.swapaxes(-1, -2), weights, mask)
    return mx.argpartition(scores, kth=-indexer.index_topk, axis=-1)[
        ..., -indexer.index_topk :
    ]


def blocked_indexer_topk(
    indexer: Indexer,
    x: mx.array,
    qr: mx.array,
    mask: mx.array | None,
    cache: Any,
    block: int = 4096,
) -> mx.array | None:
    """mlx-lm's dsa.blocked_topk (scores one key block at a time, keeping a running top-k)
    with the head reduction in float32."""
    q, k, weights = _indexer_inputs(indexer, x, qr, cache)
    n_keys = k.shape[2]
    if n_keys <= indexer.index_topk:
        return None
    best_scores: mx.array | None = None
    best_ids: mx.array | None = None
    for start in range(0, n_keys, block):
        stop = min(n_keys, start + block)
        scores = indexer_scores(
            q @ k[:, :, start:stop].swapaxes(-1, -2),
            weights,
            None if mask is None else mask[..., start:stop],
        )
        ids = mx.broadcast_to(mx.arange(start, stop), scores.shape)
        if best_scores is not None and best_ids is not None:
            scores = mx.concatenate([best_scores, scores], axis=-1)
            ids = mx.concatenate([best_ids, ids], axis=-1)
        keep = mx.stop_gradient(
            mx.argpartition(scores, kth=-indexer.index_topk, axis=-1)[
                ..., -indexer.index_topk :
            ]
        )
        best_scores = mx.take_along_axis(scores, keep, axis=-1)
        best_ids = mx.take_along_axis(ids, keep, axis=-1)
    return best_ids


class GlmAttention(GlmMoeDsaAttention):
    """mlx-lm's GlmMoeDsaAttention with faster prefill and decode paths (module docstring)."""

    def __init__(self, config: ModelArgs, layer_idx: int) -> None:
        super().__init__(config, layer_idx)
        self.sparse_prefill_min_context = SPARSE_PREFILL_MIN_CONTEXT

    def __call__(
        self,
        x: mx.array,
        mask: mx.array | None = None,
        cache: Any = None,
        prev_topk_indices: mx.array | None = None,
    ) -> tuple[mx.array, mx.array | None]:
        if not metal_kernels_available():
            return cast(
                tuple[mx.array, mx.array | None],
                super().__call__(x, mask, cache, prev_topk_indices),
            )
        batch, length, _ = x.shape
        qr = self.q_a_layernorm(self.q_a_proj(x))
        q = self.q_b_proj(qr)
        q = q.reshape(batch, length, self.num_heads, self.q_head_dim).transpose(
            0, 2, 1, 3
        )
        q_nope, q_pe = mx.split(q, [self.qk_nope_head_dim], axis=-1)
        compressed_kv = self.kv_a_proj_with_mqa(x)
        compressed_kv, k_pe = mx.split(compressed_kv, [self.kv_lora_rank], axis=-1)
        k_pe = k_pe.reshape(batch, length, 1, self.qk_rope_head_dim).transpose(
            0, 2, 1, 3
        )
        kv_latent = self.kv_a_layernorm(compressed_kv)
        offset = cache[0].offset if cache is not None else 0
        q_pe = self.rope(q_pe, offset)
        k_pe = self.rope(k_pe, offset)
        kv_latent = mx.expand_dims(kv_latent, axis=1)
        if cache is not None:
            kv_latent, k_pe = cache[0].update_and_fetch(kv_latent, k_pe)
        else:
            cache = [None, None]

        sparse = (
            self.sparse_prefill
            and batch == 1
            and length > 1
            and kv_latent.shape[2] >= self.sparse_prefill_min_context
        )
        indexer = cast(Indexer | None, self.indexer)
        if indexer is not None:
            find_topk = blocked_indexer_topk if sparse else indexer_topk
            topk_indices = find_topk(indexer, x, qr, mask, cache[1])
            if cache[0] is not None:
                # Keep the indexer cache evaluated with the attention cache (as mlx-lm).
                cache[0].keys = mx.depends(
                    cache[0].keys, (cache[1].keys, cache[1].values)
                )
        else:
            topk_indices = prev_topk_indices

        if sparse and topk_indices is not None:
            output = self._sparse_prefill(
                q_nope, q_pe, kv_latent, k_pe, topk_indices, mask
            )
        elif length == 1:
            output = self._decode(q_nope, q_pe, kv_latent, k_pe, topk_indices, mask)
        else:
            output = self._dense_prefill(
                q_nope, q_pe, kv_latent, k_pe, topk_indices, mask, cache
            )
        return self.o_proj(output), topk_indices

    def _dense_prefill(
        self,
        q_nope: mx.array,
        q_pe: mx.array,
        kv_latent: mx.array,
        k_pe: mx.array,
        topk_indices: mx.array | None,
        mask: mx.array | None,
        cache: Any,
    ) -> mx.array:
        """All heads' keys and values decompressed from the latent cache; the 192 no-rope
        and 64 rope dimensions of each head scored together, with a boolean mask."""
        batch, heads, length, _ = q_nope.shape
        n_keys = kv_latent.shape[2]
        if topk_indices is not None:
            selected = mx.zeros((*topk_indices.shape[:-1], n_keys), dtype=mx.bool_)
            selected = mx.put_along_axis(
                selected, topk_indices, mx.array(True), axis=-1
            )
            mask = selected if mask is None else selected & mask
        keys = mx.concatenate(
            [
                self.embed_q(kv_latent, transpose=False),
                mx.broadcast_to(k_pe, (batch, heads, n_keys, k_pe.shape[-1])),
            ],
            axis=-1,
        )
        values = self.unembed_out(kv_latent)
        queries = mx.concatenate([q_nope, q_pe], axis=-1)
        output = scaled_dot_product_attention(
            queries, keys, values, cache=cache, scale=self.scale, mask=mask
        )
        return output.transpose(0, 2, 1, 3).reshape(batch, length, -1)

    def _sparse_prefill(
        self,
        q_nope: mx.array,
        q_pe: mx.array,
        kv_latent: mx.array,
        k_pe: mx.array,
        topk_indices: mx.array,
        mask: mx.array | None,
    ) -> mx.array:
        """mlx-lm's dsa.sparse_attention (one sequence; absorbed form: 576-wide keys, 512-wide
        values, shared by all heads), gathering with kernels.gather_rows, with the latent keys
        concatenated once per call and the value up-projection run once for all queries."""
        _, heads, length, _ = q_nope.shape
        keys_full = mx.concatenate([kv_latent, k_pe], axis=-1)[0, 0]  # [N, 576]
        queries_full = mx.concatenate([self.embed_q(q_nope), q_pe], axis=-1)[0]
        mask_rows = None if mask is None else mask.reshape(length, -1)
        latent_blocks: list[mx.array] = []
        for start in range(0, length, SPARSE_QUERY_BLOCK):
            stop = min(length, start + SPARSE_QUERY_BLOCK)
            ids = topk_indices[0, 0, start:stop]  # [l, K]
            gathered = gather_rows(keys_full, ids)  # [l, K, 576]
            block_queries = queries_full[:, start:stop].transpose(
                1, 0, 2
            )  # [l, H, 576]
            scores = (block_queries @ gathered.swapaxes(-1, -2)).astype(
                mx.float32
            ) * self.scale
            if mask_rows is not None:
                valid = mx.take_along_axis(mask_rows[start:stop], ids, axis=-1)
                scores = mx.where(valid[:, None, :], scores, -mx.inf)
            probs = mx.softmax(scores, axis=-1).astype(gathered.dtype)
            latent_blocks.append(probs @ gathered[..., : self.kv_lora_rank])
        latent = (
            mx.concatenate(latent_blocks, axis=0)
            if len(latent_blocks) > 1
            else latent_blocks[0]
        )
        output = self.unembed_out(latent.transpose(1, 0, 2)[None])  # [1, H, L, v]
        return output.transpose(0, 2, 1, 3).reshape(1, length, heads * self.v_head_dim)

    def _decode(
        self,
        q_nope: mx.array,
        q_pe: mx.array,
        kv_latent: mx.array,
        k_pe: mx.array,
        topk_indices: mx.array | None,
        mask: mx.array | None,
    ) -> mx.array:
        """One token per sequence, absorbed form: the heads are the rows of one product
        with the shared latent keys, scores summed and normalised in float32."""
        batch, heads, _, _ = q_nope.shape
        if topk_indices is not None:
            idx = topk_indices[:, :, 0, :, None]
            kv_latent = mx.take_along_axis(
                kv_latent,
                mx.broadcast_to(idx, (*idx.shape[:-1], kv_latent.shape[-1])),
                axis=2,
            )
            k_pe = mx.take_along_axis(
                k_pe, mx.broadcast_to(idx, (*idx.shape[:-1], k_pe.shape[-1])), axis=2
            )
            if mask is not None:
                mask = mx.take_along_axis(mask, topk_indices, axis=-1)
        q_latent = self.embed_q(q_nope).reshape(batch, 1, heads, self.kv_lora_rank)
        latent_scores = q_latent @ kv_latent.swapaxes(-1, -2)  # [B, 1, H, N]
        rope_scores = q_pe.reshape(batch, 1, heads, -1) @ k_pe.swapaxes(-1, -2)
        probs = _latent_softmax(
            latent_scores,
            rope_scores,
            mx.array(True) if mask is None else mask,
            mx.array(self.scale, mx.float32),
        ).astype(kv_latent.dtype)
        latent = (probs @ kv_latent).reshape(batch, heads, 1, self.kv_lora_rank)
        output = self.unembed_out(latent)  # [B, H, 1, v]
        return output.transpose(0, 2, 1, 3).reshape(batch, 1, -1)
