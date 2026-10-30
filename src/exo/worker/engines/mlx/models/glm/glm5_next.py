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
from mlx_lm.models.base import BaseModelArgs
from mlx_lm.models.cache import ArraysCache, BatchKVCache, CacheList, KVCache
from mlx_lm.models.deepseek_v32 import group_expert_select
from mlx_lm.models.gated_delta import gated_delta_update
from mlx_lm.models.mla import MultiLinear
from mlx_lm.models.switch_layers import SwitchGLU

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


class MLP(nn.Module):
    def __init__(self, args: ModelArgs, intermediate_size: int):
        super().__init__()
        self.swiglu_limit = args.swiglu_limit
        self.gate_proj = nn.Linear(args.hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(args.hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, args.hidden_size, bias=False)

    def __call__(self, x):
        h = x.astype(_matmul_dtype(self.gate_proj, x))
        h = _clamped_swiglu(self.gate_proj(h), self.up_proj(h), self.swiglu_limit)
        return self.down_proj(h).astype(x.dtype)


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
        logits = x.astype(mx.float32) @ self.weight.T.astype(mx.float32)
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

    def __call__(self, x):
        inds, scores = self.gate(x)
        y = self.switch_mlp(x.astype(_matmul_dtype(self.switch_mlp.gate_proj, x)), inds)
        y = (y * scores[..., None]).sum(axis=-2).astype(x.dtype)
        return y + self.shared_experts(x)


class GatedRMSNorm(nn.Module):
    """RMS norm with a weight, times sigmoid(gate), in float32 (transformers'
    Glm5NextTextRMSNormGated)."""

    def __init__(self, dims: int, eps: float):
        super().__init__()
        self.eps = eps
        self.weight = mx.ones((dims,))

    def __call__(self, x, gate):
        x = x.astype(mx.float32)
        x = x * mx.rsqrt(mx.mean(mx.square(x), axis=-1, keepdims=True) + self.eps)
        return x * self.weight.astype(mx.float32) * mx.sigmoid(gate.astype(mx.float32))


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

    def __call__(self, x, cache=None):
        batch, length, _ = x.shape
        heads, width = self.num_heads, self.head_dim
        mask = cache.make_mask(length) if cache is not None else None
        lengths = cache.lengths if cache is not None else None
        if mask is not None:
            x = mx.where(mask[..., None], x, 0)

        qkv = mx.concatenate([self.q_proj(x), self.k_proj(x), self.v_proj(x)], axis=-1)
        channels = qkv.shape[-1]
        n_keep = self.conv_kernel - 1
        conv_state = cache[0] if cache is not None else None
        if conv_state is None:
            conv_state = mx.zeros((batch, n_keep, channels), dtype=qkv.dtype)
        conv_input = mx.concatenate([conv_state.astype(qkv.dtype), qkv], axis=1)
        weight = mx.concatenate(
            [self.q_conv.weight, self.k_conv.weight, self.v_conv.weight], axis=0
        )[..., 0].astype(mx.float32)  # [channels, kernel]
        # Depthwise causal convolution (a cross-correlation, as F.conv1d), in float32.
        conv = conv_input[:, :length].astype(mx.float32) * weight[:, 0]
        for j in range(1, self.conv_kernel):
            conv = (
                conv + conv_input[:, j : j + length].astype(mx.float32) * weight[:, j]
            )
        conv = nn.silu(conv)
        if lengths is not None:
            ends = mx.clip(lengths, 0, length)
            positions = (ends[:, None] + mx.arange(n_keep))[..., None]
            new_conv_state = mx.take_along_axis(conv_input, positions, axis=1)
        else:
            new_conv_state = mx.contiguous(conv_input[:, -n_keep:, :])

        q, k, v = (
            t.reshape(batch, length, heads, width) for t in mx.split(conv, 3, axis=-1)
        )
        q = q * (
            mx.rsqrt(mx.sum(mx.square(q), axis=-1, keepdims=True) + 1e-6) * width**-0.5
        )
        k = k * mx.rsqrt(mx.sum(mx.square(k), axis=-1, keepdims=True) + 1e-6)

        a = self.f_b_proj(self.f_a_proj(x)).reshape(batch, length, heads, width)
        b = self.b_proj(x).astype(mx.float32)
        state = cache[1] if cache is not None else None
        out, state = gated_delta_update(
            q,
            k,
            v,
            a,
            b,
            self.A_log.reshape(heads, 1),
            self.dt_bias.reshape(heads, width),
            state,
            mask,
            use_kernel=not self.training,
            lower_bound=self.lower_bound,
        )
        if cache is not None:
            cache[0] = new_conv_state
            cache[1] = state
            cache.advance(length)

        gate = self.g_b_proj(self.g_a_proj(x)).reshape(batch, length, heads, width)
        out = (
            self.o_norm(out, gate).astype(x.dtype).reshape(batch, length, heads * width)
        )
        return self.o_proj(out)


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

    @property
    def dense_limit(self) -> int:
        return self.index_topk + self.kpool - 1

    def __call__(self, x, qr, cache=None):
        keys = self.k_norm(self.wk(x))
        gates = (x @ self.index_kpool_compress_gate.T).astype(keys.dtype)
        first = None
        if cache is not None:
            keys, gates = cache.update_and_fetch(keys[:, None], gates[:, None])
            keys, gates = keys[:, 0], gates[:, 0]
            if isinstance(cache, BatchKVCache):
                first = cache.left_padding.astype(mx.int32)
        n_keys = keys.shape[1]
        if n_keys <= self.dense_limit:
            return None
        return self._select(x, qr, keys, gates, first, n_keys)

    def _select(self, x, qr, keys, gates, first, n_keys):
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
        weights = self.weights_proj(x).astype(mx.float32) * (
            heads**-0.5 * self.softmax_scale
        )
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
            valid = best_v > -mx.inf
            tokens = (
                first[:, None, None, None]
                + kpool * best_i[..., None]
                + mx.arange(kpool)
            )
            tokens = mx.where(valid[..., None], tokens, -1).reshape(
                batch, e - s, select_k * kpool
            )
            # The incomplete pool the query sits in.
            visible_count = pos_c[None] - first[:, None] + 1  # [B, Lc]
            tail_count = mx.where(visible_count > 0, visible_count % kpool, 0)
            tail = (pos_c[None] + 1 - tail_count)[..., None] + mx.arange(kpool - 1)
            tail = mx.where(mx.arange(kpool - 1) < tail_count[..., None], tail, -1)
            chunk = mx.concatenate([tokens, tail], axis=-1).astype(mx.int32)
            if length > query_block:
                mx.eval(chunk)
            chunks.append(chunk)
        topk = chunks[0] if len(chunks) == 1 else mx.concatenate(chunks, axis=1)
        return mx.stop_gradient(topk)


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

    def __call__(self, x, cache=None):
        batch, length, _ = x.shape
        qr = self.q_a_layernorm(_project(self.q_a_proj, x))
        q = (
            _project(self.q_b_proj, qr)
            .reshape(batch, length, self.num_heads, -1)
            .transpose(0, 2, 1, 3)
        )
        latent = self.kv_a_layernorm(_project(self.kv_a_proj_with_mqa, x))[:, None]
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
        selected = mx.take_along_axis(
            latent, mx.maximum(topk, 0).reshape(batch, 1, width, 1), axis=2
        )
        out = mx.fast.scaled_dot_product_attention(
            self.embed_q(q),
            selected,
            selected,
            scale=self.scale,
            mask=valid.reshape(batch, 1, 1, width),
        )
        return self.unembed_out(out)

    def _sparse_prefill(self, q, latent, topk):
        # One sequence: each query attends to its own selected latents (absorbed form), in
        # blocks of queries. The heads are the rows of one matmul per query, so the gathered
        # latents are read once per query, not once per head; scores and softmax in float32.
        length = q.shape[2]
        q_absorbed = self.embed_q(q)[0].transpose(1, 0, 2)  # [L, H, C]
        rows = latent[0, 0]  # [N, C]
        outs = []
        for s in range(0, length, _SPARSE_PREFILL_QUERY_BLOCK):
            e = min(length, s + _SPARSE_PREFILL_QUERY_BLOCK)
            idx = topk[0, s:e]  # [Lb, K]
            gathered = rows[mx.maximum(idx, 0)].astype(mx.float32)  # [Lb, K, C]
            scores = (
                q_absorbed[s:e].astype(mx.float32) * self.scale
            ) @ gathered.swapaxes(-1, -2)  # [Lb, H, K]
            scores = mx.where((idx >= 0)[:, None, :], scores, -mx.inf)
            probs = mx.softmax(scores, axis=-1, precise=True)
            outs.append((probs @ gathered).astype(q.dtype))  # [Lb, H, C]
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
        post_mix, comb, h = self.attn_hc(x)
        h = self.self_attn(self.input_layernorm(h), cache)
        x = _hyper_connection_post(h, x, post_mix, comb)
        post_mix, comb, h = self.ffn_hc(x)
        h = self.mlp(self.post_attention_layernorm(h))
        return _hyper_connection_post(h, x, post_mix, comb)


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
        for layer, layer_cache in zip(self.layers, cache, strict=True):
            h = layer(h, layer_cache)
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
        return self.lm_head(h)

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
