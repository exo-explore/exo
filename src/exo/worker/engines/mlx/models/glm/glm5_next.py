# type: ignore
# GLM-5.3-Flash (config model_type "glm5_next") for MLX.
#
# exo's mlx-lm (ml-explore/mlx-lm @ 6b572ab1b531dd8fd2d13f6f36a89c2eccda18ea) has no glm5_next,
# so the model is defined here and loaded through the glm family's ModelClasses.
#
# Sources:
#   - Hugging Face transformers v5.22.1,
#     src/transformers/models/glm5_next/modeling_glm5_next.py. Apache License 2.0,
#     Copyright 2026 the HuggingFace Team. The arithmetic of every block follows this file:
#     the KDA linear attention (bounded forget gate, L2-normalised q/k, sigmoid-gated
#     output norm), the NoPE MLA attention with the k-pool DSA indexer (pooled keys,
#     complete pools only, the always-selected incomplete tail), the hyper-connections
#     (Sinkhorn mixing, mean collapse at the end), the clamped SwiGLU and the float32 router.
#   - mlx-vlm, mlx_vlm/models/glm5_next/language.py at
#     035d6abb16a780794a5af6685163323e05b72dc5 (2026-10-12). MIT License, Copyright (c) 2024
#     Prince Canuma. The idea of folding each query's selected keys into
#     the batch dimension for MLX's fused attention, and the checkpoint key names.
#   - mlx-lm (above revision), MIT License, Copyright (c) 2023 Apple Inc.: the building blocks
#     imported below, and the absorbed-MLA decode / gathered sparse prefill pattern of
#     mlx_lm/models/deepseek_v32.py and dsa.py.
#   - exo's DeepSeek V4 model (models/deepseek_v4/model.py): its fused Metal kernel for the
#     hyper-connection Sinkhorn iterations and collapse, which compute the same formula.
# Written for exo on 2026-10-30. Differences from transformers, all deliberate:
#   - MLA decode runs in the absorbed form (queries projected into the 512-wide latent) and
#     sparse prefill attends to each query's gathered latents, both as in mlx-lm's DeepSeek
#     V3.2. Dense prefill expands keys and values per head, as transformers does.
#   - The indexer stores its keys and pooling gates in the cache's dtype (bf16 for these
#     checkpoints), but pools and scores them in float32; the running top-k over pool blocks
#     selects the same pools as transformers' single top-k (up to the order of equal scores).
#   - Hyper-connection mixing, the KDA convolution, recurrence and output norm, and the
#     router run in float32, which is at least as precise as transformers in bf16.
#   - Quantized projections run in the dtype of their quantization scales (float16 in the
#     mlx-community conversion), not in float32 (see _matmul_dtype).
#   - Only the language model: the vision tower (model.visual.*) and the multi-token
#     prediction layer (layer 45) are dropped in sanitize.
# exo loads this file for config model_type "glm5_next" through
# exo.worker.engines.mlx.models.registry.get_model_classes; glm5_next.pyi declares what exo uses.

from dataclasses import dataclass, field
from functools import lru_cache, partial
from typing import Any, Dict, List, Optional

import mlx.core as mx
import mlx.nn as nn
from mlx.nn.layers.distributed import AllToShardedLinear, QuantizedAllToShardedLinear
from mlx_lm.models.base import BaseModelArgs
from mlx_lm.models.cache import ArraysCache, BatchKVCache, CacheList, KVCache
from mlx_lm.models.deepseek_v32 import group_expert_select
from mlx_lm.models.gated_delta import gated_delta_kernel, gated_delta_update
from mlx_lm.models.mla import MultiLinear
from mlx_lm.models.switch_layers import (
    QuantizedSwitchLinear,
    SwitchGLU,
    _gather_sort,
    _scatter_unsort,
)

from exo.worker.engines.mlx.models.deepseek_v4.model import hc_sinkhorn_collapse

LINEAR_ATTENTION = "linear_attention"
SPARSE_ATTENTION = "deepseek_sparse_attention"
# The names config.json files and transformers use for the MLA + DSA layers.
_SPARSE_ATTENTION_NAMES = (SPARSE_ATTENTION, "indexed_attention", "full_attention")

# Memory bounds for the indexer's score blocks ([B, queries, heads, pools] float32) and for
# the gathered latents of sparse prefill ([queries, top-k, 512]).
_INDEX_SCORE_ELEMENTS = 1 << 27
_SPARSE_PREFILL_QUERY_BLOCK = 256


@dataclass
class ModelArgs(BaseModelArgs):
    model_type: str = "glm5_next"
    vocab_size: int = 154880
    hidden_size: int = 4096
    num_hidden_layers: int = 45
    rms_norm_eps: float = 1e-5
    layer_types: Optional[List[str]] = None
    mlp_layer_types: Optional[List[str]] = None
    indexer_types: Optional[List[str]] = None
    linear_attn_config: Dict[str, Any] = field(default_factory=dict)
    intermediate_size: int = 12288
    moe_intermediate_size: int = 2048
    n_routed_experts: int = 288
    n_shared_experts: int = 1
    num_experts_per_tok: int = 8
    routed_scaling_factor: float = 2.5
    norm_topk_prob: bool = True
    n_group: int = 1
    topk_group: int = 1
    topk_method: str = "noaux_tc"
    scoring_func: str = "sigmoid"
    swiglu_limit: float = 10.0
    first_k_dense_replace: int = 3
    num_attention_heads: int = 64
    q_lora_rank: int = 1536
    kv_lora_rank: int = 512
    qk_nope_head_dim: int = 256
    qk_rope_head_dim: int = 0
    v_head_dim: int = 256
    attention_bias: bool = False
    mla_use_nope: bool = True
    index_n_heads: int = 32
    index_head_dim: int = 128
    index_topk: int = 2048
    index_kpool: int = 4
    index_kpool_compress: bool = True
    index_kpool_always_select_tail: bool = True
    hc_mult: int = 4
    hc_sinkhorn_iters: int = 20
    hc_eps: float = 1e-6
    tie_word_embeddings: bool = False

    @classmethod
    def from_dict(cls, params):
        # config.json nests the language model under text_config (model_type
        # glm5_next_text); the top level names the checkpoint (glm5_next).
        text = params.get("text_config") or {}
        merged = {**params, **text}
        merged["model_type"] = params.get("model_type", text.get("model_type"))
        merged["tie_word_embeddings"] = params.get(
            "tie_word_embeddings", text.get("tie_word_embeddings", False)
        )
        return super().from_dict(merged)

    def __post_init__(self):
        n = self.num_hidden_layers
        if self.layer_types is None:
            self.layer_types = [
                SPARSE_ATTENTION if i % 4 == 3 else LINEAR_ATTENTION for i in range(n)
            ]
        if self.mlp_layer_types is None:
            self.mlp_layer_types = [
                "dense" if i < self.first_k_dense_replace else "sparse"
                for i in range(n)
            ]
        if self.indexer_types is None:
            self.indexer_types = ["full"] * n
        self.layer_types = [
            SPARSE_ATTENTION if t in _SPARSE_ATTENTION_NAMES else t
            for t in self.layer_types[:n]
        ]
        self.mlp_layer_types = list(self.mlp_layer_types[:n])
        self.indexer_types = list(self.indexer_types[:n])
        unsupported = []
        if self.qk_rope_head_dim != 0 or not self.mla_use_nope:
            unsupported.append("MLA with RoPE")
        if not self.index_kpool_compress or not self.index_kpool_always_select_tail:
            unsupported.append("an indexer without k-pool compression and its tail")
        if any(t != "full" for t in self.indexer_types):
            unsupported.append("shared indexer layers")
        if self.scoring_func != "sigmoid" or self.topk_method != "noaux_tc":
            unsupported.append(f"router {self.scoring_func}/{self.topk_method}")
        if any(t not in (LINEAR_ATTENTION, SPARSE_ATTENTION) for t in self.layer_types):
            unsupported.append(f"layer types {sorted(set(self.layer_types))}")
        if unsupported:
            raise ValueError(f"glm5_next: unsupported config: {', '.join(unsupported)}")


@lru_cache(maxsize=None)
def _hyper_connection_pre(hc: int, iters: int, hc_eps: float, norm_eps: float):
    """transformers' Glm5NextTextHyperConnection.forward: the (post, comb) mixing weights
    and the input streams collapsed with the pre weights, all in float32."""

    def pre(x, fn, base, scale):
        flat = x.reshape(*x.shape[:-2], -1).astype(mx.float32)
        flat = flat * mx.rsqrt(
            mx.mean(mx.square(flat), axis=-1, keepdims=True) + norm_eps
        )
        mixes = flat @ fn.astype(mx.float32).T
        base = base.astype(mx.float32)
        scale = scale.astype(mx.float32)
        pre_w = mixes[..., :hc]
        post_w = mixes[..., hc : 2 * hc]
        comb_w = mixes[..., 2 * hc :].reshape(*mixes.shape[:-1], hc, hc)
        pre_mix = mx.sigmoid(pre_w * scale[0] + base[:hc]) + hc_eps
        post_mix = 2.0 * mx.sigmoid(post_w * scale[1] + base[hc : 2 * hc])
        comb = comb_w * scale[2] + base[2 * hc :].reshape(hc, hc)
        comb = mx.softmax(comb, axis=-1, precise=True) + hc_eps
        comb = comb / (comb.sum(axis=-2, keepdims=True) + hc_eps)
        for _ in range(iters - 1):
            comb = comb / (comb.sum(axis=-1, keepdims=True) + hc_eps)
            comb = comb / (comb.sum(axis=-2, keepdims=True) + hc_eps)
        collapsed = (pre_mix[..., None] * x.astype(mx.float32)).sum(axis=-2)
        return post_mix, comb, collapsed.astype(x.dtype)

    return mx.compile(pre)


@mx.compile
def _hyper_connection_post(out, residual, post_mix, comb):
    # new stream j = post[j] * out + sum_i comb[i, j] * residual stream i
    mixed = post_mix[..., None] * out[..., None, :].astype(mx.float32)
    mixed = mixed + mx.swapaxes(comb, -1, -2) @ residual.astype(mx.float32)
    return mixed.astype(residual.dtype)


class HyperConnection(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        n = args.hc_mult
        self.hc_mult = n
        self.sinkhorn_iters = args.hc_sinkhorn_iters
        self.norm_eps = args.rms_norm_eps
        self.fn = mx.zeros(((2 + n) * n, n * args.hidden_size))
        self.base = mx.zeros(((2 + n) * n,), dtype=mx.float32)
        self.scale = mx.ones((3,), dtype=mx.float32)
        self._eps = mx.array([args.hc_eps], dtype=mx.float32)
        self._pre = _hyper_connection_pre(
            n, args.hc_sinkhorn_iters, args.hc_eps, args.rms_norm_eps
        )

    def __call__(self, x):
        if not mx.metal.is_available():
            return self._pre(x, self.fn, self.base, self.scale)
        # The same arithmetic as _hyper_connection_pre (the same formula as DeepSeek V4's
        # hyper-connections), with the Sinkhorn iterations and the collapse in one Metal
        # dispatch: exo's DeepSeek V4 kernel. The compiled ops take ~50 dispatches.
        flat = mx.fast.rms_norm(
            x.reshape(*x.shape[:-2], -1).astype(mx.float32), None, self.norm_eps
        )
        mixes = flat @ self.fn.astype(mx.float32).T
        collapsed, post_mix, comb = hc_sinkhorn_collapse(
            mixes,
            self.scale,
            self.base,
            x,
            self.hc_mult,
            self.sinkhorn_iters,
            self._eps,
        )
        return post_mix, comb, collapsed


def _make_hc_kernels():
    """A residual update and the next hyper-connection's mixing, in two dispatches instead
    of about ten:
      1. hc_update (one threadgroup per row and 256 hidden positions): the new streams
         x'_j = post_j * out + sum_i comb[i, j] * x_i (float32, stored in the streams' dtype)
         and, from the stored values, partial sums of squares and partial products with
         the next hyper-connection's 24 mixing rows (fn);
      2. hc_mix (one threadgroup per row): the partials reduced in order; the RMS-normalised
         mixes (the RMS scale times the products: the same value as normalising first);
         the Sinkhorn iterations of exo's DeepSeek V4 kernel; the pre-weighted collapse of
         the streams; and the RMS norm of the collapse with the next block's norm weight.
    Same formulas as _hyper_connection_post, HyperConnection.__call__ and nn.RMSNorm, in
    float32; sums run in a different order, so results can differ in the last bits."""
    if not mx.metal.is_available():
        return None, None
    update = mx.fast.metal_kernel(
        name="glm5_next_hc_update",
        input_names=["out", "res", "post", "comb", "fn"],
        output_names=["new_x", "partials"],
        source="""
        constexpr int HC = 4;
        constexpr int MIX = (2 + HC) * HC;
        constexpr int NP = MIX + 1;  // 24 products, then the sum of squares
        constexpr int NS = D / 256;
        threadgroup float red[8][NP];
        uint tid = thread_index_in_threadgroup;
        uint lane = thread_index_in_simdgroup;
        uint sg = simdgroup_index_in_threadgroup;
        uint slice = threadgroup_position_in_grid.x;
        uint row = threadgroup_position_in_grid.y;
        uint d = slice * 256 + tid;
        float v[HC];
        float sq = 0.0f;
        if (UPDATE) {
            float o = float(out[size_t(row) * D + d]);
            float r[HC];
            for (int i = 0; i < HC; ++i) {
                r[i] = float(res[(size_t(row) * HC + i) * D + d]);
            }
            for (int j = 0; j < HC; ++j) {
                float acc = 0.0f;
                for (int i = 0; i < HC; ++i) {
                    acc = metal::fma(comb[(row * HC + i) * HC + j], r[i], acc);
                }
                T stored = static_cast<T>(post[row * HC + j] * o + acc);
                new_x[(size_t(row) * HC + j) * D + d] = stored;
                v[j] = float(stored);
            }
        } else {
            for (int j = 0; j < HC; ++j) {
                v[j] = float(res[(size_t(row) * HC + j) * D + d]);
            }
        }
        if (!MIXES) {
            return;
        }
        for (int j = 0; j < HC; ++j) {
            sq += v[j] * v[j];
        }
        float part[NP];
        for (int m = 0; m < MIX; ++m) {
            float acc = 0.0f;
            for (int j = 0; j < HC; ++j) {
                acc += v[j] * fn[size_t(m) * HC * D + j * D + d];
            }
            part[m] = simd_sum(acc);
        }
        part[MIX] = simd_sum(sq);
        if (lane == 0) {
            for (int m = 0; m < NP; ++m) {
                red[sg][m] = part[m];
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (tid < NP) {
            float total = 0.0f;
            for (int g = 0; g < 8; ++g) {
                total += red[g][tid];
            }
            partials[(size_t(row) * NS + slice) * NP + tid] = total;
        }
        """,
    )
    mix = mx.fast.metal_kernel(
        name="glm5_next_hc_mix",
        input_names=["partials", "x_in", "scale", "base", "eps", "norm_w", "params"],
        output_names=["post", "comb", "normed"],
        source="""
        constexpr int HC = 4;
        constexpr int MIX = (2 + HC) * HC;
        constexpr int NP = MIX + 1;
        constexpr int NS = D / 256;
        constexpr int BASE_OFF = 2 * HC;
        threadgroup float mixes[NP];
        threadgroup float pre_shared[HC];
        threadgroup float collapsed[D];
        threadgroup float red[8];
        uint tid = thread_index_in_threadgroup;
        uint lane = thread_index_in_simdgroup;
        uint sg = simdgroup_index_in_threadgroup;
        uint row = threadgroup_position_in_grid.y;
        const float hc_norm_eps = params[0];
        const float norm_eps = params[1];
        if (tid < NP) {
            float total = 0.0f;
            for (int s = 0; s < NS; ++s) {
                total += partials[(size_t(row) * NS + s) * NP + tid];
            }
            mixes[tid] = total;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        const float inv_rms = metal::precise::rsqrt(mixes[MIX] / float(HC * D) + hc_norm_eps);
        if (sg == 0) {
            // exo's DeepSeek V4 Sinkhorn kernel (hc_sinkhorn_collapse), phase 1
            const float pre_scale = scale[0];
            const float post_scale = scale[1];
            const float comb_scale = scale[2];
            const float epsv = eps[0];
            const float active = (lane < (uint)HC) ? 1.0f : 0.0f;
            const uint llane = metal::min(lane, (uint)(HC - 1));
            float pre_z = mixes[llane] * inv_rms * pre_scale + base[llane];
            float post_z = mixes[HC + llane] * inv_rms * post_scale + base[HC + llane];
            float pre_v = 1.0f / (1.0f + metal::fast::exp(-pre_z)) + epsv;
            float post_v = 2.0f / (1.0f + metal::fast::exp(-post_z));
            if (lane < (uint)HC) {
                pre_shared[lane] = pre_v;
                post[row * HC + lane] = post_v;
            }
            float4 m4 = float4(
                mixes[BASE_OFF + llane * HC + 0], mixes[BASE_OFF + llane * HC + 1],
                mixes[BASE_OFF + llane * HC + 2], mixes[BASE_OFF + llane * HC + 3]) * inv_rms;
            float4 b4 = float4(
                base[BASE_OFF + llane * HC + 0], base[BASE_OFF + llane * HC + 1],
                base[BASE_OFF + llane * HC + 2], base[BASE_OFF + llane * HC + 3]);
            float4 v = (m4 * comb_scale + b4) * active;
            float row_max = metal::max(metal::max(v.x, v.y), metal::max(v.z, v.w));
            float4 e = metal::fast::exp(v - row_max) * active;
            float4 rr = e * (1.0f / (e.x + e.y + e.z + e.w + epsv)) + epsv * active;
            float4 col_inv = 1.0f / (float4(
                simd_sum(rr.x), simd_sum(rr.y), simd_sum(rr.z), simd_sum(rr.w)) + epsv);
            rr *= col_inv;
            for (int iter = 1; iter < ITERS; ++iter) {
                rr *= (1.0f / (rr.x + rr.y + rr.z + rr.w + epsv)) * active;
                col_inv = 1.0f / (float4(
                    simd_sum(rr.x), simd_sum(rr.y), simd_sum(rr.z), simd_sum(rr.w)) + epsv);
                rr *= col_inv;
            }
            if (lane < (uint)HC) {
                comb[(row * HC + lane) * HC + 0] = rr.x;
                comb[(row * HC + lane) * HC + 1] = rr.y;
                comb[(row * HC + lane) * HC + 2] = rr.z;
                comb[(row * HC + lane) * HC + 3] = rr.w;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        const float p0 = pre_shared[0], p1 = pre_shared[1], p2 = pre_shared[2], p3 = pre_shared[3];
        float sq = 0.0f;
        for (uint d = tid; d < uint(D); d += 256) {
            float x0 = float(x_in[(size_t(row) * HC + 0) * D + d]);
            float x1 = float(x_in[(size_t(row) * HC + 1) * D + d]);
            float x2 = float(x_in[(size_t(row) * HC + 2) * D + d]);
            float x3 = float(x_in[(size_t(row) * HC + 3) * D + d]);
            float c = float(static_cast<T>(
                metal::fma(p0, x0, metal::fma(p1, x1, metal::fma(p2, x2, p3 * x3)))));
            collapsed[d] = c;
            sq += c * c;
        }
        sq = simd_sum(sq);
        if (lane == 0) {
            red[sg] = sq;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        float total = 0.0f;
        for (int g = 0; g < 8; ++g) {
            total += red[g];
        }
        const float inv = metal::precise::rsqrt(total / float(D) + norm_eps);
        for (uint d = tid; d < uint(D); d += 256) {
            // mx.fast.rms_norm: w * T(x * inv), the product of two T values rounded once
            T normalized = static_cast<T>(collapsed[d] * inv);
            normed[size_t(row) * D + d] = static_cast<T>(float(normalized) * float(norm_w[d]));
        }
        """,
    )
    return update, mix


_hc_update_kernel, _hc_mix_kernel = _make_hc_kernels()


@lru_cache(maxsize=None)
def _float_params(*values):
    """Scalar kernel parameters as one float32 array, made once per distinct value set."""
    params = mx.array(values, dtype=mx.float32)
    mx.eval(params)
    return params


def _hc_kernels_apply(x, hc) -> bool:
    """Whether the hyper-connection kernels run for streams x (else the ops). The same for
    every layer of a model, so a residual stream is computed the same way whichever
    layer boundary (fused or a pipeline rank's) it crosses."""
    n, dim = x.shape[-2:]
    return (
        _hc_update_kernel is not None
        and n == 4
        and dim % 256 == 0
        and dim <= 4096
        and x.dtype in (mx.bfloat16, mx.float16)
        and hc.fn.dtype == mx.float32
        and hc.sinkhorn_iters >= 1
    )


def _hc_update(x, out=None, post_mix=None, comb=None, fn=None):
    """hc_update: the new streams if out is given (else x is kept), and the mixing
    partials against fn if fn is given."""
    *lead, _, dim = x.shape
    rows = 1
    for size in lead:
        rows *= size
    slices = dim // 256
    update, mixes = out is not None, fn is not None
    unused = _float_params(0.0)  # float32 stand-in for inputs this variant doesn't read
    new_x, partials = _hc_update_kernel(
        inputs=[
            out if update else x,
            x,
            post_mix if update else unused,
            comb if update else unused,
            fn if mixes else unused,
        ],
        template=[("T", x.dtype), ("D", dim), ("UPDATE", update), ("MIXES", mixes)],
        grid=(slices * 256, rows, 1),
        threadgroup=(256, 1, 1),
        output_shapes=[
            x.shape if update else (1,),
            (rows, slices, 25) if mixes else (1,),
        ],
        output_dtypes=[x.dtype, mx.float32],
    )
    return (new_x if update else x), (partials if mixes else None)


def _hc_mix(partials, x, hc, norm):
    *lead, _, dim = x.shape
    rows = partials.shape[0]
    params = _float_params(hc.norm_eps, norm.eps)
    return _hc_mix_kernel(
        inputs=[partials, x, hc.scale, hc.base, hc._eps, norm.weight, params],
        template=[("T", x.dtype), ("D", dim), ("ITERS", hc.sinkhorn_iters)],
        grid=(256, rows, 1),
        threadgroup=(256, 1, 1),
        output_shapes=[(*lead, 4), (*lead, 4, 4), (*lead, dim)],
        output_dtypes=[mx.float32, mx.float32, x.dtype],
    )


def _hc_pre(x, hc, norm):
    """hc(x) and norm of the collapse: (post_mix, comb, normed block input)."""
    if not _hc_kernels_apply(x, hc):
        post_mix, comb, h = hc(x)
        return post_mix, comb, norm(h)
    _, partials = _hc_update(x, fn=hc.fn)
    post_mix, comb, normed = _hc_mix(partials, x, hc, norm)
    return post_mix, comb, normed


def _hc_post(out, x, post_mix, comb, hc):
    """The new streams after a block (hc: the block's hyper-connection, which decides
    between the kernels and the ops)."""
    if not _hc_kernels_apply(x, hc) or out.dtype != x.dtype:
        return _hyper_connection_post(out, x, post_mix, comb)
    new_x, _ = _hc_update(x, out, post_mix, comb)
    return new_x


def _hc_post_then_pre(out, x, post_mix, comb, hc, norm):
    """_hc_post, then _hc_pre with the next block's hc and norm on the new streams, with
    the mixing partials computed while the streams are written: returns (new streams,
    (post_mix, comb, normed input) of the next block)."""
    if not _hc_kernels_apply(x, hc) or out.dtype != x.dtype:
        x = _hyper_connection_post(out, x, post_mix, comb)
        post_mix, comb, h = hc(x)
        return x, (post_mix, comb, norm(h))
    new_x, partials = _hc_update(x, out, post_mix, comb, hc.fn)
    return new_x, _hc_mix(partials, new_x, hc, norm)


@partial(mx.compile, shapeless=True)
def _clamped_swiglu(gate, up, limit):
    return nn.silu(mx.minimum(gate, limit)) * mx.clip(up, -limit, limit)


class ClampedSwiGLU(nn.Module):
    """SwitchGLU's activation with GLM-5.3-Flash's clamps (gate <= limit, |up| <= limit)."""

    def __init__(self, limit: float):
        super().__init__()
        self.limit = limit

    def __call__(self, x, gate):
        return _clamped_swiglu(gate, x, self.limit)


def _matmul_dtype(layer, x):
    """The dtype to run a (possibly quantized) projection of x in.

    The mlx-community conversion stores the quantization scales and biases in float16 next
    to bf16 activations. MLX then runs every quantized matmul in float32 (and returns
    float32), at about four times the cost in decode. Running them in the scales' dtype
    instead keeps the checkpoint's values exact; the inputs are normalized or clamped
    activations, far inside float16's range, and float16 has 3 more mantissa bits than
    bf16."""
    scales = layer.get("scales")
    if (
        scales is not None
        and scales.dtype in (mx.float16, mx.bfloat16)
        and x.dtype in (mx.float16, mx.bfloat16)
    ):
        return scales.dtype
    return x.dtype


def _project(layer, x):
    dtype = _matmul_dtype(layer, x)
    if dtype == x.dtype:
        return layer(x)
    return layer(x.astype(dtype)).astype(x.dtype)


# Up to how many input rows one matmul over concatenated weights replaces the parts' own
# matmuls. MLX picks its matmul variant and tiling from the row count and the output width,
# so a wider matmul can sum in another order than the parts' (checked on MLX 0.32.4, M3
# Ultra, every row count 1-17):
# - quantized matmuls: the fused ones were bitwise the parts' at every row count, for these
#   shapes and the tests' (tests/test_glm5_next_kernels.py);
# - bf16 matmuls: bitwise for one row (a decode step); from 2 rows on, small inner
#   dimensions (the tests' 256) already sum in another order.
# Past the limit, prompt-sized inputs run one matmul per part on row slices of the fused
# weight, which are bitwise the parts' own.
_FUSED_QUANTIZED_MAX_ROWS = 16
_FUSED_GEMM_MAX_ROWS = 1
# Whether modules fuse their projections on first call (tests compare against False).
_FUSE_PROJECTIONS = True


_PLAIN_LINEAR = (nn.Linear, AllToShardedLinear)
_QUANTIZED_LINEAR = (nn.QuantizedLinear, QuantizedAllToShardedLinear)


class _FusedLinear:
    """Projections of the same input, run as one matmul over their concatenated output rows.

    Built once, on a module's first call (so after tensor-parallel sharding has sliced the
    parts), from layers of one kind: all unquantized with one dtype, or all quantized with the
    same bits, group size and mode. Each output row is the same dot product as in its own
    layer, so the outputs are the parts' outputs. A plain object, not an nn.Module: its
    arrays are not model parameters and the parts it replaces are removed from the module."""

    def __init__(self, layers):
        first = layers[0]
        self.sizes = [layer.weight.shape[0] for layer in layers]
        self.weight = mx.concatenate([layer.weight for layer in layers], axis=0)
        self.quantized = type(first) in _QUANTIZED_LINEAR
        arrays = [self.weight]
        if self.quantized:
            self.scales = mx.concatenate([layer.scales for layer in layers], axis=0)
            biases = [layer.get("biases") for layer in layers]
            self.biases = None if biases[0] is None else mx.concatenate(biases, axis=0)
            self.group_size, self.bits, self.mode = (
                first.group_size,
                first.bits,
                first.mode,
            )
            arrays += [self.scales] + ([] if self.biases is None else [self.biases])
        mx.eval(arrays)

    @staticmethod
    def fusable(layers) -> bool:
        if any("bias" in layer for layer in layers):
            return False
        # The tensor-parallel all-to-sharded layers compute their forward like the plain
        # ones (sum_gradients is the identity there).
        if all(type(layer) in _PLAIN_LINEAR for layer in layers):
            return len({layer.weight.dtype for layer in layers}) == 1
        if all(type(layer) in _QUANTIZED_LINEAR for layer in layers):
            keys = {
                (
                    layer.group_size,
                    layer.bits,
                    layer.mode,
                    layer.scales.dtype,
                    layer.get("biases") is None,
                )
                for layer in layers
            }
            return len(keys) == 1
        return False

    def matmul_dtype(self, x):
        """_matmul_dtype for the fused layers."""
        if (
            self.quantized
            and self.scales.dtype in (mx.float16, mx.bfloat16)
            and x.dtype in (mx.float16, mx.bfloat16)
        ):
            return self.scales.dtype
        return x.dtype

    def _matmul(self, x, start, end):
        """x times rows [start, end) of the fused weight (a contiguous row slice)."""
        if not self.quantized:
            return x @ self.weight[start:end].T
        return mx.quantized_matmul(
            x,
            self.weight[start:end],
            scales=self.scales[start:end],
            biases=None if self.biases is None else self.biases[start:end],
            transpose=True,
            group_size=self.group_size,
            bits=self.bits,
            mode=self.mode,
        )

    def __call__(self, x, groups=None):
        """The parts' outputs, as slices of one output; ``groups`` (counts of consecutive
        parts) returns some adjacent parts as one slice."""
        out = None
        limit = _FUSED_QUANTIZED_MAX_ROWS if self.quantized else _FUSED_GEMM_MAX_ROWS
        if x.size // x.shape[-1] <= limit:
            out = self._matmul(x, 0, self.weight.shape[0])
        groups = groups or (1,) * len(self.sizes)
        parts, start, i = [], 0, 0
        for count in groups:
            size = sum(self.sizes[i : i + count])
            if out is None:
                # Prompt-sized inputs: one matmul per part over its rows of the fused
                # weight (concatenated for a group). MLX's GEMM tiling depends on the
                # output width, so a wider matmul can round some outputs differently from
                # the parts' own; matmuls on the parts' row slices are bitwise theirs.
                pieces, offset = [], start
                for part_size in self.sizes[i : i + count]:
                    pieces.append(self._matmul(x, offset, offset + part_size))
                    offset += part_size
                parts.append(
                    pieces[0] if len(pieces) == 1 else mx.concatenate(pieces, axis=-1)
                )
            else:
                parts.append(out[..., start : start + size])
            start, i = start + size, i + count
        return parts


def _fuse(module, names):
    """Replace module.<names> by one _FusedLinear, or return None if they can't be fused."""
    if not _FUSE_PROJECTIONS:
        return None
    layers = [module[name] for name in names]
    if not _FusedLinear.fusable(layers):
        return None
    fused = _FusedLinear(layers)
    for name in names:
        del module[name]
    return fused


class MLP(nn.Module):
    def __init__(self, args: ModelArgs, intermediate_size: int):
        super().__init__()
        self.swiglu_limit = args.swiglu_limit
        self.gate_proj = nn.Linear(args.hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(args.hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, args.hidden_size, bias=False)
        self._gate_up = None

    def __call__(self, x):
        if self._gate_up is None and "gate_proj" in self:
            self._gate_up = _fuse(self, ("gate_proj", "up_proj"))
        if self._gate_up is not None:
            gate, up = self._gate_up(x.astype(self._gate_up.matmul_dtype(x)))
        else:
            h = x.astype(_matmul_dtype(self.gate_proj, x))
            gate, up = self.gate_proj(h), self.up_proj(h)
        h = _clamped_swiglu(gate, up, self.swiglu_limit)
        return self.down_proj(h).astype(x.dtype)


def _make_router_select_kernel():
    """group_expert_select for n_group == 1 with normalised top-k weights, one simdgroup
    per token. It takes the sigmoid of the float32 logits, adds the correction bias to
    choose the top k (by repeated simdgroup maxima, ties to the lower expert), and returns
    the unbiased sigmoids of the chosen experts normalised to sum 1, times the routed
    scaling factor. The k experts come out in descending order of their biased score
    (argpartition leaves them in no particular order), so their weighted sum adds in
    another order."""
    if not mx.metal.is_available():
        return None
    return mx.fast.metal_kernel(
        name="glm5_next_router_select",
        input_names=["logits", "bias", "params"],
        output_names=["inds", "scores"],
        source="""
        constexpr int PER_LANE = (E + 31) / 32;
        uint lane = thread_index_in_simdgroup;
        uint row = threadgroup_position_in_grid.y;
        auto l = logits + size_t(row) * E;
        float score[PER_LANE];
        float biased[PER_LANE];
        for (int i = 0; i < PER_LANE; ++i) {
            int e = i * 32 + int(lane);
            if (e < E) {
                score[i] = 1.0f / (1.0f + metal::exp(-l[e]));
                biased[i] = score[i] + bias[e];
            } else {
                score[i] = 0.0f;
                biased[i] = -INFINITY;
            }
        }
        float chosen[K];
        uint chosen_ids[K];
        float total = 0.0f;
        for (int k = 0; k < K; ++k) {
            float best = -INFINITY;
            int best_e = E;
            for (int i = 0; i < PER_LANE; ++i) {
                int e = i * 32 + int(lane);
                if (biased[i] > best || (biased[i] == best && e < best_e)) {
                    best = biased[i];
                    best_e = e;
                }
            }
            float top = simd_max(best);
            int top_e = simd_min(best == top ? best_e : E);
            float w = 0.0f;
            for (int i = 0; i < PER_LANE; ++i) {
                if (i * 32 + int(lane) == top_e) {
                    w = score[i];
                    biased[i] = -INFINITY;
                }
            }
            w = simd_sum(w);
            chosen[k] = w;
            chosen_ids[k] = uint(top_e);
            total += w;
        }
        if (lane == 0) {
            for (int k = 0; k < K; ++k) {
                inds[size_t(row) * K + k] = chosen_ids[k];
                scores[size_t(row) * K + k] = (chosen[k] / total) * params[0];
            }
        }
        """,
    )


_router_select_kernel = _make_router_select_kernel()


class MoEGate(nn.Module):
    """noaux_tc sigmoid router with the logits in float32 (transformers and the config's
    moe_router_dtype: float32)."""

    def __init__(self, args: ModelArgs):
        super().__init__()
        self.top_k = args.num_experts_per_tok
        self.norm_topk_prob = args.norm_topk_prob
        self.routed_scaling_factor = args.routed_scaling_factor
        self.n_group = args.n_group
        self.topk_group = args.topk_group
        self.weight = mx.zeros((args.n_routed_experts, args.hidden_size))
        self.e_score_correction_bias = mx.zeros(
            (args.n_routed_experts,), dtype=mx.float32
        )

    def __call__(self, x):
        # sanitize stores the weight in float32 (exactly), so the cast is a no-op
        logits = x.astype(mx.float32) @ self.weight.T.astype(mx.float32)
        n_experts = logits.shape[-1]
        if (
            _router_select_kernel is not None
            and self.n_group == 1
            and self.norm_topk_prob
            and 1 < self.top_k <= 32
            and n_experts <= 1024
            and logits.size // n_experts <= 64
        ):
            # few tokens (decode): group_expert_select's steps in one dispatch
            rows = logits.size // n_experts
            inds, scores = _router_select_kernel(
                inputs=[
                    logits,
                    self.e_score_correction_bias,
                    _float_params(self.routed_scaling_factor),
                ],
                template=[("E", n_experts), ("K", self.top_k)],
                grid=(32, rows, 1),
                threadgroup=(32, 1, 1),
                output_shapes=[(*logits.shape[:-1], self.top_k)] * 2,
                output_dtypes=[mx.uint32, mx.float32],
            )
            return mx.stop_gradient(inds), scores
        return group_expert_select(
            logits,
            self.e_score_correction_bias,
            self.top_k,
            self.n_group,
            self.topk_group,
            self.routed_scaling_factor,
            self.norm_topk_prob,
        )


class MoE(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.gate = MoEGate(args)
        self.switch_mlp = SwitchGLU(
            args.hidden_size,
            args.moe_intermediate_size,
            args.n_routed_experts,
            activation=ClampedSwiGLU(args.swiglu_limit),
        )
        self.shared_experts = MLP(
            args, args.moe_intermediate_size * args.n_shared_experts
        )
        self.swiglu_limit = args.swiglu_limit
        self._gate_up = None

    def _fuse_gate_up(self):
        """The routed experts' gate and up projections as one gather over their
        concatenated output rows (each expert's [gate; up]), like _FusedLinear."""
        if not _FUSE_PROJECTIONS:
            return None
        switch = self.switch_mlp
        gate, up = switch.gate_proj, switch.up_proj
        if not (
            isinstance(gate, QuantizedSwitchLinear)
            and isinstance(up, QuantizedSwitchLinear)
            and (gate.group_size, gate.bits, gate.mode)
            == (up.group_size, up.bits, up.mode)
            and gate.get("biases") is not None
            and up.get("biases") is not None
            and gate.global_scale is None
            and up.global_scale is None
            and "bias" not in gate
            and "bias" not in up
        ):
            return None
        arrays = [
            mx.concatenate([gate[name], up[name]], axis=1)
            for name in ("weight", "scales", "biases")
        ]
        mx.eval(arrays)
        del switch["gate_proj"], switch["up_proj"]
        return (*arrays, gate.group_size, gate.bits, gate.mode, gate.weight.shape[1])

    def _experts(self, x, inds):
        if self._gate_up is None and "gate_proj" in self.switch_mlp:
            self._gate_up = self._fuse_gate_up()
        if self._gate_up is None:
            return self.switch_mlp(
                x.astype(_matmul_dtype(self.switch_mlp.gate_proj, x)), inds
            )
        weight, scales, biases, group_size, bits, mode, hidden = self._gate_up
        if scales.dtype in (mx.float16, mx.bfloat16) and x.dtype in (
            mx.float16,
            mx.bfloat16,
        ):
            x = x.astype(scales.dtype)
        # SwitchGLU.__call__ with the gate and up gathers as one
        x = mx.expand_dims(x, (-2, -3))
        inds = mx.stop_gradient(inds)
        do_sort = inds.size >= 64
        idx, inv_order = inds, None
        if do_sort:
            x, idx, inv_order = _gather_sort(x, inds)
        gate_up = mx.gather_qmm(
            x,
            weight,
            scales,
            biases,
            rhs_indices=idx,
            transpose=True,
            group_size=group_size,
            bits=bits,
            mode=mode,
            sorted_indices=do_sort,
        )
        h = _clamped_swiglu(
            gate_up[..., :hidden], gate_up[..., hidden:], self.swiglu_limit
        )
        y = self.switch_mlp.down_proj(h, idx, sorted_indices=do_sort)
        if do_sort:
            y = _scatter_unsort(y, inv_order, inds.shape)
        return y.squeeze(-2)

    def __call__(self, x):
        inds, scores = self.gate(x)
        y = self._experts(x, inds)
        return _combine_experts(y, scores, self.shared_experts(x))


@mx.compile
def _combine_experts(routed, scores, shared):
    routed = (routed * scores[..., None]).sum(axis=-2).astype(shared.dtype)
    return routed + shared


class GatedRMSNorm(nn.Module):
    """RMS norm with a weight, times sigmoid(gate), in float32 (transformers'
    Glm5NextTextRMSNormGated)."""

    def __init__(self, dims: int, eps: float):
        super().__init__()
        self.eps = eps
        self.weight = mx.ones((dims,))

    def __call__(self, x, gate):
        return _gated_rms_norm(x, gate, self.weight, self.eps)


@partial(mx.compile, shapeless=True)
def _gated_rms_norm(x, gate, weight, eps):
    x = x.astype(mx.float32)
    x = x * mx.rsqrt(mx.mean(mx.square(x), axis=-1, keepdims=True) + eps)
    return x * weight.astype(mx.float32) * mx.sigmoid(gate.astype(mx.float32))


@lru_cache(maxsize=None)
def _kda_conv(kernel: int, heads: int, width: int):
    """The KDA short convolution (depthwise, causal, in float32) over the concatenated
    q, k, v channels, the SiLU, and the L2 normalisation of q (with the head_dim**-0.5
    readout scale) and k: transformers' arithmetic, compiled into few dispatches."""

    def conv_qkv(conv_input, weight):
        batch = conv_input.shape[0]
        length = conv_input.shape[1] - kernel + 1
        conv = conv_input[:, :length].astype(mx.float32) * weight[:, 0]
        for j in range(1, kernel):
            conv = (
                conv + conv_input[:, j : j + length].astype(mx.float32) * weight[:, j]
            )
        conv = nn.silu(conv)
        q, k, v = (
            t.reshape(batch, length, heads, width) for t in mx.split(conv, 3, axis=-1)
        )
        q = q * (
            mx.rsqrt(mx.sum(mx.square(q), axis=-1, keepdims=True) + 1e-6) * width**-0.5
        )
        k = k * mx.rsqrt(mx.sum(mx.square(k), axis=-1, keepdims=True) + 1e-6)
        return q, k, v

    return mx.compile(conv_qkv)


@partial(mx.compile, shapeless=True)
def _kda_gates(a_log, a, dt_bias, b, lower_bound):
    """mlx-lm's gated_delta_update gates: the bounded forget gate and beta = sigmoid(b)."""
    g = mx.exp(
        lower_bound
        * mx.sigmoid(
            mx.exp(a_log.astype(mx.float32)) * (a.astype(mx.float32) + dt_bias)
        )
    )
    return g, mx.sigmoid(b)


def _make_kda_decode_kernel():
    """KimiDeltaAttention's decode step after the projections, for one (sequence, head) per
    threadgroup of 32 simdgroups:
      1. the head's q, k and v channels: depthwise causal convolution over the 3 cached
         inputs and the new one (float32), SiLU; the forget gate
         exp(lower_bound * sigmoid(exp(A_log) * (a + dt_bias))) and beta = sigmoid(b);
      2. L2 norms of q (times W**-0.5) and k;
      3. the delta rule (as mlx-lm's gated_delta vector-gate kernel: lane = 4 key
         elements, simdgroup = value rows);
      4. RMS norm of the output times the weight and sigmoid(gate), in T.
    The convolution window shifted by one is the new convolution state."""
    if not mx.metal.is_available():
        return None
    source = """
        constexpr int P = H * W;          // channels per q / k / v
        constexpr int ROWS = W / 32;      // value rows per simdgroup
        constexpr int PER_LANE = W / 32;  // key elements per lane
        threadgroup float qs[W];
        threadgroup float ks[W];
        threadgroup float vs[W];
        threadgroup float gs[W];
        threadgroup float ys[W];
        threadgroup float scalars[4];     // beta, q scale, k scale, output norm scale
        uint tid = thread_index_in_threadgroup;
        uint lane = thread_index_in_simdgroup;
        uint sg = simdgroup_index_in_threadgroup;
        uint h = threadgroup_position_in_grid.y;
        uint b = threadgroup_position_in_grid.z;
        const float lower_bound = params[0];
        const float eps = params[1];

        if (tid < 3 * W) {
            uint part = tid / W, d = tid % W;
            uint c = part * P + h * W + d;
            const size_t row = size_t(3 * P);
            float x0 = float(conv_state[(b * 3 + 0) * row + c]);
            float x1 = float(conv_state[(b * 3 + 1) * row + c]);
            float x2 = float(conv_state[(b * 3 + 2) * row + c]);
            float x3 = float(qkv[b * row + c]);
            float conv = x0 * conv_w[c * 4 + 0];
            conv = conv + x1 * conv_w[c * 4 + 1];
            conv = conv + x2 * conv_w[c * 4 + 2];
            conv = conv + x3 * conv_w[c * 4 + 3];
            conv = conv * (1.0f / (1.0f + metal::exp(-conv)));
            (part == 0 ? qs : (part == 1 ? ks : vs))[d] = conv;
            new_conv_state[(b * 3 + 0) * row + c] = conv_state[(b * 3 + 1) * row + c];
            new_conv_state[(b * 3 + 1) * row + c] = conv_state[(b * 3 + 2) * row + c];
            new_conv_state[(b * 3 + 2) * row + c] = qkv[b * row + c];
        } else if (tid < 4 * W) {
            uint d = tid - 3 * W;
            float z = metal::exp(A_log[h]) * (float(a[(b * H + h) * W + d]) + dt_bias[h * W + d]);
            gs[d] = metal::exp(lower_bound * (1.0f / (1.0f + metal::exp(-z))));
        } else if (tid == 4 * W) {
            scalars[0] = 1.0f / (1.0f + metal::exp(-float(beta_in[b * H + h])));
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (sg < 2) {
            threadgroup float* t = sg == 0 ? qs : ks;
            float sq = 0.0f;
            for (int i = 0; i < PER_LANE; ++i) {
                float v = t[lane * PER_LANE + i];
                sq += v * v;
            }
            sq = simd_sum(sq);
            if (lane == 0) {
                scalars[1 + sg] = metal::precise::rsqrt(sq + 1e-6f) * (sg == 0 ? metal::precise::rsqrt(float(W)) : 1.0f);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        const float beta = scalars[0];
        float q_l[PER_LANE], k_l[PER_LANE], g_l[PER_LANE];
        for (int i = 0; i < PER_LANE; ++i) {
            int dk = lane * PER_LANE + i;
            q_l[i] = qs[dk] * scalars[1];
            k_l[i] = ks[dk] * scalars[2];
            g_l[i] = gs[dk];
        }
        const bool row_active = active[b];
        for (int r = 0; r < ROWS; ++r) {
            uint dv = sg * ROWS + r;
            size_t base = ((size_t(b) * H + h) * W + dv) * W + lane * PER_LANE;
            if (!row_active) {
                for (int i = 0; i < PER_LANE; ++i) {
                    state_out[base + i] = state_in[base + i];
                }
                if (lane == 0) {
                    ys[dv] = 0.0f;
                }
                continue;
            }
            float st[PER_LANE];
            float kv_mem = 0.0f;
            for (int i = 0; i < PER_LANE; ++i) {
                st[i] = state_in[base + i] * g_l[i];
                kv_mem += st[i] * k_l[i];
            }
            kv_mem = simd_sum(kv_mem);
            float delta = (vs[dv] - kv_mem) * beta;
            float o = 0.0f;
            for (int i = 0; i < PER_LANE; ++i) {
                st[i] = st[i] + k_l[i] * delta;
                o += st[i] * q_l[i];
                state_out[base + i] = st[i];
            }
            o = simd_sum(o);
            if (lane == 0) {
                ys[dv] = o;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (sg == 0) {
            float sq = 0.0f;
            for (int i = 0; i < PER_LANE; ++i) {
                float v = ys[lane * PER_LANE + i];
                sq += v * v;
            }
            sq = simd_sum(sq);
            if (lane == 0) {
                scalars[3] = metal::precise::rsqrt(sq / float(W) + eps);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (tid < W) {
            float g = float(gate[(b * H + h) * W + tid]);
            float y = ys[tid] * scalars[3] * float(o_weight[tid]) * (1.0f / (1.0f + metal::exp(-g)));
            out[(b * H + h) * W + tid] = static_cast<T>(y);
        }
    """
    return mx.fast.metal_kernel(
        name="glm5_next_kda_decode",
        input_names=[
            "active",
            "qkv",
            "conv_state",
            "conv_w",
            "a",
            "beta_in",
            "A_log",
            "dt_bias",
            "state_in",
            "gate",
            "o_weight",
            "params",
        ],
        output_names=["out", "state_out", "new_conv_state"],
        source=source,
    )


_kda_decode_kernel = _make_kda_decode_kernel()


class KimiDeltaAttention(nn.Module):
    """Linear-attention layer (Kimi Delta Attention, transformers'
    Glm5NextTextLinearAttention). Cache: ArraysCache [conv state, recurrent state]."""

    def __init__(self, args: ModelArgs):
        super().__init__()
        cfg = args.linear_attn_config
        self.num_heads = cfg["num_heads"]
        self.head_dim = cfg["head_dim"]
        self.conv_kernel = cfg.get("short_conv_kernel_size", 4)
        self.lower_bound = cfg.get("gate_lower_bound")
        projection = self.num_heads * self.head_dim
        hidden = args.hidden_size
        self.q_proj = nn.Linear(hidden, projection, bias=False)
        self.k_proj = nn.Linear(hidden, projection, bias=False)
        self.v_proj = nn.Linear(hidden, projection, bias=False)
        # Depthwise causal convolutions (weights [channels, kernel, 1]); run as one
        # convolution over the concatenated q, k, v channels.
        self.q_conv = nn.Conv1d(
            projection, projection, self.conv_kernel, groups=projection, bias=False
        )
        self.k_conv = nn.Conv1d(
            projection, projection, self.conv_kernel, groups=projection, bias=False
        )
        self.v_conv = nn.Conv1d(
            projection, projection, self.conv_kernel, groups=projection, bias=False
        )
        self.f_a_proj = nn.Linear(hidden, self.head_dim, bias=False)
        self.f_b_proj = nn.Linear(self.head_dim, projection, bias=False)
        self.dt_bias = mx.zeros((projection,), dtype=mx.float32)
        self.A_log = mx.zeros((self.num_heads,), dtype=mx.float32)
        self.b_proj = nn.Linear(hidden, self.num_heads, bias=False)
        self.g_a_proj = nn.Linear(hidden, self.head_dim, bias=False)
        self.g_b_proj = nn.Linear(self.head_dim, projection, bias=False)
        self.o_norm = GatedRMSNorm(self.head_dim, args.rms_norm_eps)
        self.o_proj = nn.Linear(projection, hidden, bias=False)
        self._inputs = None
        self._conv_weight = None

    def _input_projections(self, x):
        """q, k, v (concatenated), f_a and g_a of x: two matmuls once fused. (Fusing b_proj
        or all six would change MLX's GEMV tiling for the narrow parts, so their float32
        sums; these groups keep every output bitwise the separate layers'.)"""
        if self._inputs is None and "q_proj" in self:
            self._inputs = (
                _fuse(self, ("q_proj", "k_proj", "v_proj")),
                _fuse(self, ("f_a_proj", "g_a_proj")),
            )
        qkv_proj, fg_proj = self._inputs or (None, None)
        if qkv_proj is None:
            qkv = mx.concatenate(
                [self.q_proj(x), self.k_proj(x), self.v_proj(x)], axis=-1
            )
        else:
            # q, k and v are adjacent rows of the fused weight: one slice
            (qkv,) = qkv_proj(x, groups=(3,))
        if fg_proj is None:
            f_a, g_a = self.f_a_proj(x), self.g_a_proj(x)
        else:
            f_a, g_a = fg_proj(x)
        return qkv, f_a, g_a, self.b_proj(x)

    def __call__(self, x, cache=None):
        batch, length, _ = x.shape
        heads, width = self.num_heads, self.head_dim
        mask = cache.make_mask(length) if cache is not None else None
        lengths = cache.lengths if cache is not None else None
        if mask is not None:
            x = mx.where(mask[..., None], x, 0)

        qkv, f_a, g_a, b = self._input_projections(x)
        channels = qkv.shape[-1]
        n_keep = self.conv_kernel - 1
        conv_state = cache[0] if cache is not None else None
        if conv_state is None:
            conv_state = mx.zeros((batch, n_keep, channels), dtype=qkv.dtype)
        if self._conv_weight is None:
            # [channels, kernel] in float32; built once, after tensor-parallel sharding
            self._conv_weight = mx.concatenate(
                [self.q_conv.weight, self.k_conv.weight, self.v_conv.weight], axis=0
            )[..., 0].astype(mx.float32)
            mx.eval(self._conv_weight)
        if (
            length == 1
            and lengths is None
            and _kda_decode_kernel is not None
            and self.conv_kernel == 4
            and width == 128
            and self.lower_bound is not None
            and not self.training
            and qkv.dtype == conv_state.dtype
            and x.dtype in (mx.bfloat16, mx.float16)
        ):
            return self._decode_step(x, qkv, f_a, g_a, b, conv_state, cache, mask)
        conv_input = mx.concatenate([conv_state.astype(qkv.dtype), qkv], axis=1)
        # Depthwise causal convolution (a cross-correlation, as F.conv1d), in float32,
        # SiLU, and the L2-normalised q and k.
        q, k, v = _kda_conv(self.conv_kernel, heads, width)(
            conv_input, self._conv_weight
        )
        if lengths is not None:
            ends = mx.clip(lengths, 0, length)
            positions = (ends[:, None] + mx.arange(n_keep))[..., None]
            new_conv_state = mx.take_along_axis(conv_input, positions, axis=1)
        else:
            new_conv_state = mx.contiguous(conv_input[:, -n_keep:, :])

        a = self.f_b_proj(f_a).reshape(batch, length, heads, width)
        b = b.astype(mx.float32)
        state = cache[1] if cache is not None else None
        out, state = self._recurrence(q, k, v, a, b, state, mask)
        if cache is not None:
            cache[0] = new_conv_state
            cache[1] = state
            cache.advance(length)

        gate = self.g_b_proj(g_a).reshape(batch, length, heads, width)
        out = (
            self.o_norm(out, gate).astype(x.dtype).reshape(batch, length, heads * width)
        )
        return self.o_proj(out)

    def _decode_step(self, x, qkv, f_a, g_a, b, conv_state, cache, mask=None):
        """One token per sequence: the convolution, SiLU, L2 norms, gates, the delta-rule
        recurrence and the gated output norm in one Metal kernel (_kda_decode_kernel), from
        the projections' outputs. The arithmetic is __call__'s, in float32; reductions sum
        in a different order, so results can differ in the last bits."""
        batch = x.shape[0]
        heads, width = self.num_heads, self.head_dim
        a = self.f_b_proj(f_a)
        gate = self.g_b_proj(g_a)
        state = cache[1] if cache is not None else None
        if state is None:
            state = mx.zeros((batch, heads, width, width), dtype=mx.float32)
        params = _float_params(self.lower_bound, self.o_norm.eps)
        # rows the cache masks out (left padding) keep their recurrent state and output 0,
        # as in gated_delta's masked kernel; x is already zeroed there
        active = (
            mx.ones((batch,), dtype=mx.bool_)
            if mask is None
            else mask.reshape(batch, -1)[:, -1]
        )
        out, new_state, new_conv_state = _kda_decode_kernel(
            inputs=[
                active,
                qkv,
                conv_state,
                self._conv_weight,
                a,
                b,
                self.A_log,
                self.dt_bias,
                state,
                gate,
                self.o_norm.weight,
                params,
            ],
            template=[("T", x.dtype), ("H", heads), ("W", width)],
            grid=(1024, heads, batch),
            threadgroup=(1024, 1, 1),
            output_shapes=[(batch, 1, heads * width), state.shape, conv_state.shape],
            output_dtypes=[x.dtype, mx.float32, conv_state.dtype],
        )
        if cache is not None:
            cache[0] = new_conv_state
            cache[1] = new_state
            cache.advance(1)
        return self.o_proj(out)

    def _recurrence(self, q, k, v, a, b, state, mask):
        """mlx-lm's gated_delta_update, with the gates computed in one compiled call."""
        heads, width = self.num_heads, self.head_dim
        a_log = self.A_log.reshape(heads, 1)
        dt_bias = self.dt_bias.reshape(heads, width)
        use_kernel = (
            not self.training
            and mx.default_device() == mx.gpu
            and mx.metal.is_available()
            and width >= 32
            and width % 32 == 0
        )
        if self.lower_bound is None or not use_kernel:
            return gated_delta_update(
                q,
                k,
                v,
                a,
                b,
                a_log,
                dt_bias,
                state,
                mask,
                use_kernel=not self.training,
                lower_bound=self.lower_bound,
            )
        g, beta = _kda_gates(a_log, a, dt_bias, b, self.lower_bound)
        if state is None:
            state = mx.zeros((q.shape[0], heads, width, width), dtype=mx.float32)
        return gated_delta_kernel(q, k, v, g, beta, state, mask)


def _make_pool_scores_kernel():
    """Indexer scores of every complete pool for a few queries.

    Threadgroups of 8 simdgroups, one pool per simdgroup. Each lane pools D/32 of the key
    dimensions (softmax over the pool's KPOOL gate logits plus the position embedding,
    weighting the KPOOL keys), then lane h takes head h (32 heads): relu(q_h . pooled)
    times the head weight, summed over the simdgroup. Pools that end past the key count or
    past the query's position score -inf, as in Indexer._select."""
    if not mx.metal.is_available():
        return None
    source = """
        constexpr int H = 32;
        constexpr int PER_LANE = D / 32;
        threadgroup float qs[D * H];      // [d][h]: lanes read consecutive words
        threadgroup float pooled_tg[8][D];
        uint lane = thread_index_in_simdgroup;
        uint sg = simdgroup_index_in_threadgroup;
        uint tid = thread_index_in_threadgroup;
        uint l = threadgroup_position_in_grid.y;
        uint b = threadgroup_position_in_grid.z;
        int n_keys = sizes[0];
        int n_pools = sizes[1];

        auto q_row = q + ((b * L + l) * H) * D;
        for (uint i = tid; i < uint(H * D); i += 256) {
            qs[i] = q_row[(i % H) * D + i / H];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        const float w_lane = w[(b * L + l) * H + lane];
        const int position = positions[l];
        auto k_rows = keys + size_t(b) * n_keys * D;
        auto g_rows = gates + size_t(b) * n_keys * D;
        const int base = (int(threadgroup_position_in_grid.x) * 8 + int(sg)) * POOLS;
        for (int n = 0; n < POOLS; ++n) {
            int p = base + n;
            if (p >= n_pools) {
                break;
            }
            int start = first[b] + KPOOL * p;
            int end = start + KPOOL - 1;
            bool ok = end < n_keys && end <= position;
            if (ok) {
                for (int i = 0; i < PER_LANE; ++i) {
                    int d = int(lane) * PER_LANE + i;
                    float logit[KPOOL];
                    float m = -INFINITY;
                    for (int j = 0; j < KPOOL; ++j) {
                        int row = metal::clamp(start + j, 0, n_keys - 1);
                        logit[j] = float(g_rows[size_t(row) * D + d]) + float(ape[j * D + d]);
                        m = metal::max(m, logit[j]);
                    }
                    float denom = 0.0f;
                    for (int j = 0; j < KPOOL; ++j) {
                        logit[j] = metal::exp(logit[j] - m);
                        denom += logit[j];
                    }
                    float acc = 0.0f;
                    for (int j = 0; j < KPOOL; ++j) {
                        int row = metal::clamp(start + j, 0, n_keys - 1);
                        acc += (logit[j] / denom) * float(k_rows[size_t(row) * D + d]);
                    }
                    pooled_tg[sg][d] = acc;
                }
            }
            simdgroup_barrier(mem_flags::mem_threadgroup);
            float score = -INFINITY;
            if (ok) {
                float dot = 0.0f;
                for (int d = 0; d < D; ++d) {
                    dot += qs[d * H + lane] * pooled_tg[sg][d];
                }
                score = simd_sum(w_lane * metal::max(dot, 0.0f));
            }
            if (lane == 0) {
                out[(size_t(b) * L + l) * n_pools + p] = score;
            }
            simdgroup_barrier(mem_flags::mem_threadgroup);
        }
    """
    return mx.fast.metal_kernel(
        name="glm5_next_pool_scores",
        input_names=["keys", "gates", "ape", "q", "w", "first", "positions", "sizes"],
        output_names=["out"],
        source=source,
    )


_pool_scores_kernel = _make_pool_scores_kernel()
# Queries per call up to which the indexer uses _pool_scores_kernel (decode steps and short
# chunks); longer chunks pool once and score with matmuls.
_POOL_KERNEL_MAX_QUERIES = 16


class Indexer(nn.Module):
    """DSA indexer with k-pool compression (transformers' Glm5NextTextIndexer).

    Each query attends to the 2048 / index_kpool best-scoring complete pools of
    index_kpool consecutive keys that end at or before it, plus the incomplete pool it
    sits in (the tail). Pools start at the sequence's first token, so a batch row's left
    padding shifts its pools. While a query sees at most index_topk + index_kpool - 1 keys
    that selects every key it sees, so the indexer returns None (dense attention).
    Cache: KVCache with the normalised keys as keys and the pooling gate logits as values.
    """

    def __init__(self, args: ModelArgs):
        super().__init__()
        self.n_heads = args.index_n_heads
        self.head_dim = args.index_head_dim
        self.index_topk = args.index_topk
        self.kpool = args.index_kpool
        self.wq_b = nn.Linear(
            args.q_lora_rank, self.n_heads * self.head_dim, bias=False
        )
        self.wk = nn.Linear(args.hidden_size, self.head_dim, bias=False)
        self.k_norm = nn.LayerNorm(self.head_dim, eps=1e-6)
        self.weights_proj = nn.Linear(args.hidden_size, self.n_heads, bias=False)
        self.index_kpool_compress_ape = mx.zeros((self.kpool, self.head_dim))
        self.index_kpool_compress_gate = mx.zeros((self.head_dim, args.hidden_size))
        self.softmax_scale = self.head_dim**-0.5
        self._x_proj = None

    @property
    def dense_limit(self) -> int:
        return self.index_topk + self.kpool - 1

    def _fuse_x_projections(self):
        """wk and the pooling gate (both of x, both unquantized, 128 rows each) as one
        weight. (weights_proj stays apart: a 288-row GEMV sums differently from its 32-row
        one, see _input_projections.)"""
        if not _FUSE_PROJECTIONS:
            return None
        if type(self.wk) is not nn.Linear or "bias" in self.wk:
            return None
        weights = [self.wk.weight, self.index_kpool_compress_gate]
        if len({w.dtype for w in weights}) != 1:
            return None
        fused = mx.concatenate(weights, axis=0)
        mx.eval(fused)
        del self["wk"], self["index_kpool_compress_gate"]
        return fused

    def _x_projections(self, x):
        """(wk(x), x @ index_kpool_compress_gate.T, weights_proj(x))."""
        if self._x_proj is None and "wk" in self:
            self._x_proj = self._fuse_x_projections()
        if self._x_proj is None:
            keys, gates = self.wk(x), x @ self.index_kpool_compress_gate.T
        else:
            width = self.head_dim
            if x.size // x.shape[-1] <= _FUSED_GEMM_MAX_ROWS:
                out = x @ self._x_proj.T
                keys, gates = out[..., :width], out[..., width:]
            else:
                # see _FusedLinear: row slices of the fused weight for prompt-sized inputs
                keys = x @ self._x_proj[:width].T
                gates = x @ self._x_proj[width:].T
        return keys, gates, self.weights_proj(x)

    def __call__(self, x, qr, cache=None):
        keys, gates, weights = self._x_projections(x)
        keys = self.k_norm(keys)
        gates = gates.astype(keys.dtype)
        first = None
        if cache is not None:
            keys, gates = cache.update_and_fetch(keys[:, None], gates[:, None])
            keys, gates = keys[:, 0], gates[:, 0]
            if isinstance(cache, BatchKVCache):
                first = cache.left_padding.astype(mx.int32)
        n_keys = keys.shape[1]
        if n_keys <= self.dense_limit:
            return None
        if (
            _pool_scores_kernel is not None
            and x.shape[1] <= _POOL_KERNEL_MAX_QUERIES
            and self.n_heads == 32
            and self.head_dim % 32 == 0
            and keys.dtype == gates.dtype
        ):
            return self._select_few(qr, keys, gates, weights, first, n_keys)
        return self._select(x, qr, keys, gates, weights, first, n_keys)

    def _select(self, x, qr, keys, gates, weights, first, n_keys):
        batch, length, _ = x.shape
        heads, width, kpool = self.n_heads, self.head_dim, self.kpool
        n_pools = n_keys // kpool
        if first is None:
            first = mx.zeros((batch,), dtype=mx.int32)
            grouped_keys = keys[:, : n_pools * kpool].reshape(
                batch, n_pools, kpool, width
            )
            grouped_gates = gates[:, : n_pools * kpool].reshape(
                batch, n_pools, kpool, width
            )
        else:
            idx = first[:, None, None] + kpool * mx.arange(n_pools)[None, :, None]
            idx = mx.clip(idx + mx.arange(kpool), 0, n_keys - 1).reshape(
                batch, n_pools * kpool, 1
            )
            grouped_keys = mx.take_along_axis(keys, idx, axis=1).reshape(
                batch, n_pools, kpool, width
            )
            grouped_gates = mx.take_along_axis(gates, idx, axis=1).reshape(
                batch, n_pools, kpool, width
            )
        pool_start = first[:, None] + kpool * mx.arange(n_pools)[None]
        pool_end = pool_start + kpool - 1
        pool_ok = pool_end < n_keys
        probs = mx.softmax(
            grouped_gates.astype(mx.float32)
            + self.index_kpool_compress_ape.astype(mx.float32),
            axis=2,
            precise=True,
        )
        pooled = (probs * grouped_keys.astype(mx.float32)).sum(axis=2)

        q = self.wq_b(qr).reshape(batch, length, heads, width).astype(mx.float32)
        weights = weights.astype(mx.float32) * (heads**-0.5 * self.softmax_scale)
        positions = n_keys - length + mx.arange(length)
        select_k = self.index_topk // kpool

        pool_block = min(n_pools, 32768)
        query_block = max(
            1, min(length, _INDEX_SCORE_ELEMENTS // (batch * heads * pool_block))
        )
        chunks = []
        for s in range(0, length, query_block):
            e = min(length, s + query_block)
            q_c, w_c, pos_c = q[:, s:e], weights[:, s:e, None, :], positions[s:e]
            best_v = best_i = None
            for p0 in range(0, n_pools, pool_block):
                p1 = min(n_pools, p0 + pool_block)
                scores = mx.maximum(
                    q_c @ pooled[:, None, p0:p1].swapaxes(-1, -2), 0
                )  # [B, Lc, H, Pb]
                scores = (w_c @ scores)[:, :, 0]  # [B, Lc, Pb]
                visible = pool_ok[:, None, p0:p1] & (
                    pool_end[:, None, p0:p1] <= pos_c[None, :, None]
                )
                scores = mx.where(visible, scores, -mx.inf)
                ids = mx.broadcast_to(mx.arange(p0, p1), scores.shape)
                if best_v is not None:
                    scores = mx.concatenate([best_v, scores], axis=-1)
                    ids = mx.concatenate([best_i, ids], axis=-1)
                top = mx.argpartition(-scores, kth=select_k - 1, axis=-1)[
                    ..., :select_k
                ]
                top = mx.stop_gradient(top)
                best_v = mx.take_along_axis(scores, top, axis=-1)
                best_i = mx.take_along_axis(ids, top, axis=-1)
            chunk = self._selected_tokens(best_v, best_i, first, pos_c)
            if length > query_block:
                mx.eval(chunk)
            chunks.append(chunk)
        topk = chunks[0] if len(chunks) == 1 else mx.concatenate(chunks, axis=1)
        return mx.stop_gradient(topk)

    def _selected_tokens(self, best_v, best_i, first, positions):
        """The key positions of the selected pools (-1 where fewer pools are visible),
        then the incomplete pool each query sits in."""
        batch, n_queries, select_k = best_i.shape
        kpool = self.kpool
        valid = best_v > -mx.inf
        tokens = (
            first[:, None, None, None] + kpool * best_i[..., None] + mx.arange(kpool)
        )
        tokens = mx.where(valid[..., None], tokens, -1).reshape(
            batch, n_queries, select_k * kpool
        )
        # The incomplete pool the query sits in.
        visible_count = positions[None] - first[:, None] + 1  # [B, Lc]
        tail_count = mx.where(visible_count > 0, visible_count % kpool, 0)
        tail = (positions[None] + 1 - tail_count)[..., None] + mx.arange(kpool - 1)
        tail = mx.where(mx.arange(kpool - 1) < tail_count[..., None], tail, -1)
        return mx.concatenate([tokens, tail], axis=-1).astype(mx.int32)

    def _select_few(self, qr, keys, gates, weights, first, n_keys):
        """_select for a few queries (decode): the pooling and the pool scores in one Metal
        kernel that reads the cached keys and gates once, instead of pooling every complete
        pool again into float32 arrays at every step. Same formulas in float32; the sums
        run in a different order, so scores can differ in the last bits."""
        batch, length = qr.shape[:2]
        heads, width, kpool = self.n_heads, self.head_dim, self.kpool
        n_pools = n_keys // kpool
        q = self.wq_b(qr).reshape(batch, length, heads, width).astype(mx.float32)
        weights = weights.astype(mx.float32) * (heads**-0.5 * self.softmax_scale)
        positions = n_keys - length + mx.arange(length)
        if first is None:
            first = mx.zeros((batch,), dtype=mx.int32)
        # Pools per simdgroup: amortise each threadgroup's load of the queries while
        # keeping a few hundred threadgroups in flight.
        pools_per_simdgroup = max(1, min(16, n_pools // (8 * 256)))
        (scores,) = _pool_scores_kernel(
            inputs=[
                keys,
                gates,
                self.index_kpool_compress_ape,
                q,
                weights,
                first,
                positions.astype(mx.int32),
                mx.array([n_keys, n_pools], dtype=mx.int32),
            ],
            template=[
                ("D", width),
                ("KPOOL", kpool),
                ("L", length),
                ("POOLS", pools_per_simdgroup),
            ],
            grid=(-(-n_pools // (8 * pools_per_simdgroup)) * 256, length, batch),
            threadgroup=(256, 1, 1),
            output_shapes=[(batch, length, n_pools)],
            output_dtypes=[mx.float32],
        )
        select_k = self.index_topk // kpool
        top = mx.stop_gradient(
            mx.argpartition(-scores, kth=select_k - 1, axis=-1)[..., :select_k]
        )
        best_v = mx.take_along_axis(scores, top, axis=-1)
        return mx.stop_gradient(self._selected_tokens(best_v, top, first, positions))


class Glm5NextAttention(nn.Module):
    """NoPE MLA attention restricted to the indexer's selection (transformers'
    Glm5NextTextAttention). Cache: CacheList(latent KVCache, indexer KVCache)."""

    def __init__(self, args: ModelArgs):
        super().__init__()
        self.num_heads = args.num_attention_heads
        self.q_lora_rank = args.q_lora_rank
        self.kv_lora_rank = args.kv_lora_rank
        self.qk_head_dim = args.qk_nope_head_dim + args.qk_rope_head_dim
        self.v_head_dim = args.v_head_dim
        self.scale = self.qk_head_dim**-0.5
        hidden = args.hidden_size
        self.q_a_proj = nn.Linear(hidden, args.q_lora_rank, bias=args.attention_bias)
        self.q_a_layernorm = nn.RMSNorm(args.q_lora_rank, eps=args.rms_norm_eps)
        self.q_b_proj = nn.Linear(
            args.q_lora_rank, self.num_heads * self.qk_head_dim, bias=False
        )
        self.kv_a_proj_with_mqa = nn.Linear(
            hidden,
            args.kv_lora_rank + args.qk_rope_head_dim,
            bias=args.attention_bias,
        )
        self.kv_a_layernorm = nn.RMSNorm(args.kv_lora_rank, eps=args.rms_norm_eps)
        self.embed_q = MultiLinear(
            args.qk_nope_head_dim, args.kv_lora_rank, self.num_heads
        )
        self.unembed_out = MultiLinear(
            args.kv_lora_rank, args.v_head_dim, self.num_heads
        )
        self.o_proj = nn.Linear(
            self.num_heads * self.v_head_dim, hidden, bias=args.attention_bias
        )
        self.indexer = Indexer(args)
        self._down = None

    def _down_projections(self, x):
        """q_a_proj(x) and kv_a_proj_with_mqa(x): one matmul once fused."""
        if self._down is None and "q_a_proj" in self:
            self._down = _fuse(self, ("q_a_proj", "kv_a_proj_with_mqa"))
        if self._down is None:
            return _project(self.q_a_proj, x), _project(self.kv_a_proj_with_mqa, x)
        q_a, kv_a = self._down(x.astype(self._down.matmul_dtype(x)))
        return q_a.astype(x.dtype), kv_a.astype(x.dtype)

    def __call__(self, x, cache=None):
        batch, length, _ = x.shape
        q_a, kv_a = self._down_projections(x)
        qr = self.q_a_layernorm(q_a)
        q = (
            _project(self.q_b_proj, qr)
            .reshape(batch, length, self.num_heads, -1)
            .transpose(0, 2, 1, 3)
        )
        latent = self.kv_a_layernorm(kv_a)[:, None]
        if cache is not None:
            mask = cache[0].make_mask(length, return_array=False, window_size=None)
            latent, _ = cache[0].update_and_fetch(
                latent, mx.zeros((batch, 1, length, 0), dtype=latent.dtype)
            )
            topk = self.indexer(x, qr, cache[1])
        else:
            mask = "causal" if length > 1 else None
            topk = self.indexer(x, qr)

        if topk is None:
            out = self._dense(q, latent, mask)
        elif length == 1:
            out = self._sparse_decode(q, latent, topk)
        elif batch == 1:
            out = self._sparse_prefill(q, latent, topk)
        else:
            out = self._masked(q, latent, topk)
        out = out.transpose(0, 2, 1, 3).reshape(batch, length, -1)
        return _project(self.o_proj, out)

    def _dense(self, q, latent, mask):
        if q.shape[2] == 1:
            out = mx.fast.scaled_dot_product_attention(
                self.embed_q(q), latent, latent, scale=self.scale, mask=mask
            )
            return self.unembed_out(out)
        keys = self.embed_q(latent, transpose=False)
        values = self.unembed_out(latent)
        return mx.fast.scaled_dot_product_attention(
            q, keys, values, scale=self.scale, mask=mask
        )

    def _sparse_decode(self, q, latent, topk):
        batch, _, width = topk.shape
        valid = topk >= 0
        safe = mx.maximum(topk, 0)
        if batch == 1:
            # a row gather (mx.take) is about 3x faster than take_along_axis here
            selected = mx.take(latent[0, 0], safe[0, 0], axis=0)[None, None]
        else:
            selected = mx.take_along_axis(
                latent, safe.reshape(batch, 1, width, 1), axis=2
            )
        query = self.embed_q(q)
        # an additive mask: the same result as the boolean one, which MLX converts to an
        # additive mask on every call
        mask = mx.where(
            valid.reshape(batch, 1, 1, width),
            mx.array(0.0, dtype=query.dtype),
            mx.array(-float("inf"), dtype=query.dtype),
        )
        out = mx.fast.scaled_dot_product_attention(
            query, selected, selected, scale=self.scale, mask=mask
        )
        return self.unembed_out(out)

    def _sparse_prefill(self, q, latent, topk):
        # One sequence: each query attends to its own selected latents (absorbed form), in
        # blocks of queries. The heads are the rows of one matmul per query, so the gathered
        # latents are read once per query, not once per head; scores and softmax in float32.
        # Each block's gathered latents ([Lb, K, C] float32, 1 GB for 256 queries) are
        # evaluated before the next block is built: left to one lazy graph, MLX would gather
        # every block before attending to the first (about 20 GB for a 4096-token chunk).
        length = q.shape[2]
        q_absorbed = self.embed_q(q)[0].transpose(1, 0, 2)  # [L, H, C]
        # The latents in float32 once (exact), gathered per query from that
        rows = latent[0, 0].astype(mx.float32)  # [N, C]
        outs = []
        for s in range(0, length, _SPARSE_PREFILL_QUERY_BLOCK):
            e = min(length, s + _SPARSE_PREFILL_QUERY_BLOCK)
            idx = topk[0, s:e]  # [Lb, K]
            # mx.take along axis 0 (a row gather) is about twice as fast as rows[idx]
            gathered = mx.take(rows, mx.maximum(idx, 0), axis=0)  # [Lb, K, C]
            scores = (
                q_absorbed[s:e].astype(mx.float32) * self.scale
            ) @ gathered.swapaxes(-1, -2)  # [Lb, H, K]
            scores = mx.where((idx >= 0)[:, None, :], scores, -mx.inf)
            probs = mx.softmax(scores, axis=-1, precise=True)
            outs.append((probs @ gathered).astype(q.dtype))  # [Lb, H, C]
            if length > _SPARSE_PREFILL_QUERY_BLOCK:
                mx.eval(outs[-1])
        out = outs[0] if len(outs) == 1 else mx.concatenate(outs, axis=0)
        return self.unembed_out(out.transpose(1, 0, 2)[None])

    def _masked(self, q, latent, topk):
        # Several sequences: dense attention masked to each query's selection.
        batch, _, length, _ = q.shape
        n_keys = latent.shape[2]
        safe = mx.where(topk >= 0, topk, n_keys)
        mask = mx.zeros((batch, length, n_keys + 1), dtype=mx.bool_)
        mask = mx.put_along_axis(mask, safe, mx.array(True), axis=-1)[..., :n_keys]
        keys = self.embed_q(latent, transpose=False)
        values = self.unembed_out(latent)
        return mx.fast.scaled_dot_product_attention(
            q, keys, values, scale=self.scale, mask=mask[:, None]
        )


class DecoderLayer(nn.Module):
    def __init__(self, args: ModelArgs, layer_idx: int):
        super().__init__()
        self.is_linear = args.layer_types[layer_idx] == LINEAR_ATTENTION
        self.self_attn = (
            KimiDeltaAttention(args) if self.is_linear else Glm5NextAttention(args)
        )
        if args.mlp_layer_types[layer_idx] == "sparse":
            self.mlp = MoE(args)
        else:
            self.mlp = MLP(args, args.intermediate_size)
        self.input_layernorm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        self.post_attention_layernorm = nn.RMSNorm(
            args.hidden_size, eps=args.rms_norm_eps
        )
        self.attn_hc = HyperConnection(args)
        self.ffn_hc = HyperConnection(args)

    def __call__(self, x, cache=None):
        # x: [B, L, hc_mult, hidden] residual streams
        return self.forward(x, cache)[0]

    def forward(self, x, cache=None, attn_input=None, next_layer=None):
        """The layer, with the hyper-connection mixing fused across block boundaries:
        attn_input is this layer's (post_mix, comb, normed attention input) if the previous
        layer computed it, and with next_layer the result also carries the next layer's.
        Returns (streams, next layer's attention input or None)."""
        if attn_input is None:
            post_mix, comb, h = _hc_pre(x, self.attn_hc, self.input_layernorm)
        else:
            post_mix, comb, h = attn_input
        h = self.self_attn(h, cache)
        x, (post_mix, comb, h) = _hc_post_then_pre(
            h, x, post_mix, comb, self.ffn_hc, self.post_attention_layernorm
        )
        h = self.mlp(h)
        if next_layer is None:
            return _hc_post(h, x, post_mix, comb, self.ffn_hc), None
        return _hc_post_then_pre(
            h, x, post_mix, comb, next_layer.attn_hc, next_layer.input_layernorm
        )


class Glm5NextModel(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.hc_mult = args.hc_mult
        self.embed_tokens = nn.Embedding(args.vocab_size, args.hidden_size)
        self.layers = [DecoderLayer(args, i) for i in range(args.num_hidden_layers)]
        self.norm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)

    def __call__(self, inputs, cache=None, input_embeddings=None):
        h = (
            input_embeddings
            if input_embeddings is not None
            else self.embed_tokens(inputs)
        )
        h = mx.repeat(h[:, :, None, :], self.hc_mult, axis=2)
        if cache is None:
            cache = [None] * len(self.layers)
        layers = self.layers
        attn_input = None
        for i, (layer, layer_cache) in enumerate(zip(layers, cache, strict=True)):
            if not isinstance(layer, DecoderLayer):
                # a pipeline-parallel wrapper: plain call
                h, attn_input = layer(h, layer_cache), None
                continue
            following = layers[i + 1] if i + 1 < len(layers) else None
            h, attn_input = layer.forward(
                h,
                layer_cache,
                attn_input,
                following if isinstance(following, DecoderLayer) else None,
            )
        # The final collapse is an unweighted mean of the streams.
        h = h.astype(mx.float32).mean(axis=2).astype(h.dtype)
        return self.norm(h)


class Model(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.model_type = args.model_type
        self.model = Glm5NextModel(args)
        if not args.tie_word_embeddings:
            self.lm_head = nn.Linear(args.hidden_size, args.vocab_size, bias=False)

    def __call__(self, inputs, cache=None, input_embeddings=None):
        h = self.model(inputs, cache, input_embeddings)
        if self.args.tie_word_embeddings:
            return self.model.embed_tokens.as_linear(h)
        logits = self.lm_head(h)
        group = self.__dict__.get("_vocab_group")
        if group is not None:
            # tensor parallel: each rank holds a slice of the vocabulary (rank order)
            logits = mx.distributed.all_gather(mx.moveaxis(logits, -1, 0), group=group)
            logits = mx.moveaxis(logits, 0, -1)
        return logits

    def shard_vocabulary(self, lm_head, group):
        """Use a tensor-parallel lm_head that holds this rank's rows of the vocabulary; the
        logits are gathered from every rank."""
        self.lm_head = lm_head
        self.__dict__["_vocab_group"] = group

    @property
    def layers(self):
        return self.model.layers

    def make_cache(self):
        caches = []
        for layer in self.layers:
            if layer.is_linear:
                caches.append(ArraysCache(size=2))
            else:
                caches.append(CacheList(KVCache(), KVCache()))
        return caches

    def sanitize(self, weights):
        if any(key.endswith("weight_scale_inv") for key in weights):
            raise ValueError(
                "glm5_next: FP8 checkpoints are not supported; use an MLX conversion"
            )
        n_layers = self.args.num_hidden_layers
        sanitized = {}
        for key, value in weights.items():
            if key.startswith(("model.visual.", "visual.")):
                continue
            if key.startswith("model.language_model."):
                key = "model." + key[len("model.language_model.") :]
            elif key.startswith("language_model.model."):
                key = "model." + key[len("language_model.model.") :]
            elif key.startswith("language_model."):
                key = key[len("language_model.") :]
            parts = key.split(".")
            if parts[:2] == ["model", "layers"] and int(parts[2]) >= n_layers:
                continue  # the multi-token-prediction layer
            for site in ("attn", "ffn"):
                for name in ("fn", "base", "scale"):
                    if key.endswith(f".hc_{site}_{name}"):
                        key = key[: -len(f"hc_{site}_{name}")] + f"{site}_hc.{name}"
            # The hyper-connection mixing and the router run in float32: store their weights
            # in float32 (exactly; bf16 in the checkpoint) instead of casting every call.
            if key.endswith(("_hc.fn", ".mlp.gate.weight")):
                value = value.astype(mx.float32)
            for name in ("q", "k", "v"):
                if key.endswith(f".self_attn.{name}_conv1d.weight"):
                    key = key.replace(f"{name}_conv1d.weight", f"{name}_conv.weight")
                    if value.ndim == 3 and value.shape[1] == 1:
                        value = value.moveaxis(2, 1)  # [C, 1, K] -> [C, K, 1]
            sanitized[key] = value
        weights = sanitized

        for i, layer in enumerate(self.layers):
            prefix = f"model.layers.{i}"
            if isinstance(layer.mlp, MoE):
                for name in ("gate_proj", "up_proj", "down_proj"):
                    for suffix in ("weight", "scales", "biases"):
                        first = f"{prefix}.mlp.experts.0.{name}.{suffix}"
                        if first not in weights:
                            continue
                        weights[f"{prefix}.mlp.switch_mlp.{name}.{suffix}"] = mx.stack(
                            [
                                weights.pop(f"{prefix}.mlp.experts.{e}.{name}.{suffix}")
                                for e in range(self.args.n_routed_experts)
                            ]
                        )
            if not layer.is_linear:
                self._split_kv_b(weights, f"{prefix}.self_attn")
        return weights

    def _split_kv_b(self, weights, prefix):
        key = f"{prefix}.kv_b_proj.weight"
        if key not in weights:
            return
        args = self.args
        value = weights.pop(key)
        scales = weights.pop(f"{prefix}.kv_b_proj.scales", None)
        biases = weights.pop(f"{prefix}.kv_b_proj.biases", None)
        bits = group_size = None
        if scales is not None:
            bits = value.shape[-1] * 32 // args.kv_lora_rank
            group_size = args.kv_lora_rank // scales.shape[-1]
            value = mx.dequantize(
                value, scales, biases, group_size=group_size, bits=bits
            )
        value = value.reshape(
            args.num_attention_heads,
            args.qk_nope_head_dim + args.v_head_dim,
            args.kv_lora_rank,
        )
        wk = mx.contiguous(value[:, : args.qk_nope_head_dim].swapaxes(-1, -2))
        wv = mx.contiguous(value[:, args.qk_nope_head_dim :])
        for name, w in (("embed_q", wk), ("unembed_out", wv)):
            if scales is not None:
                w, s, b = mx.quantize(w, group_size=group_size, bits=bits)
                weights[f"{prefix}.{name}.scales"] = s
                weights[f"{prefix}.{name}.biases"] = b
            weights[f"{prefix}.{name}.weight"] = w

    @property
    def cast_predicate(self):
        def predicate(key):
            return not key.endswith(
                (
                    "A_log",
                    "dt_bias",
                    "e_score_correction_bias",
                    ".attn_hc.base",
                    ".attn_hc.scale",
                    ".ffn_hc.base",
                    ".ffn_hc.scale",
                )
            )

        return predicate
