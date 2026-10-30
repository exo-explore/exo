"""Tencent Hy3 (``hy_v3``, ``HYV3ForCausalLM``) for MLX.

Provenance
----------
- Ported from Hugging Face transformers' ``hy_v3`` model:
  https://github.com/huggingface/transformers/blob/v5.22.1/src/transformers/models/hy_v3/modeling_hy_v3.py
  (generated from ``modular_hy_v3.py``) and ``configuration_hy_v3.py``, tag v5.22.1, commit
  446a570e62b04a89ea001604ada7366baac2ea67. The checkpoint key renames are transformers'
  ``conversion_mapping.py`` entry for ``hy_v3`` at the same commit.
- Licence: Apache License 2.0. "Copyright 2026 Tencent HunYuan Team and The HuggingFace Inc.
  team. All rights reserved. Licensed under the Apache License, Version 2.0 (the "License");
  you may not use this file except in compliance with the License. You may obtain a copy of
  the License at http://www.apache.org/licenses/LICENSE-2.0. Unless required by applicable
  law or agreed to in writing, software distributed under the License is distributed on an
  "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied."
- Ported on 2026-10-30. This file is a modified version: a re-implementation in MLX.

Local changes against the reference
-----------------------------------
- MLX building blocks in place of transformers' modules: ``nn.RMSNorm`` for ``HYV3RMSNorm``
  (both normalise in float32 and round before applying the weight), ``nn.RoPE`` (rotate-half)
  for the default rotary embedding, mlx-lm's ``scaled_dot_product_attention``, ``KVCache``
  and ``SwitchGLU`` for the routed experts (``HYV3Experts``).
- Module names follow the MLX conversions of Hy3 on the Hugging Face Hub, so they load:
  ``mlp.router.gate`` (``HYV3TopKRouter``'s weight, ``mlp.gate`` in transformers),
  ``mlp.router.expert_bias`` (``e_score_correction_bias``), ``mlp.shared_mlp``
  (``shared_experts``) and ``mlp.switch_mlp`` (the experts, stacked).
  ``sanitize`` maps the official checkpoint's names (``mlp.router.gate``,
  ``mlp.expert_bias``, ``mlp.shared_mlp``, ``mlp.experts.{e}.*``) to these.
- The multi-token-prediction layer (``model.layers.{num_hidden_layers}``) is dropped, as
  transformers does (``_keys_to_ignore_on_load_unexpected``).
- The routed experts' weighted sum is accumulated in float32 and rounded once. transformers
  rounds each expert's weighted output to the model dtype and accumulates in that dtype.
- The dense/sparse layer pattern comes from ``mlp_layer_types`` when the config has it, else
  from ``first_k_dense_replace`` (transformers: one dense layer; the official config sets
  ``first_k_dense_replace`` to 1).
- Only the ``default`` rotary embedding is implemented: that is what the checkpoints use.
"""

import math
import re
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, cast

import mlx.core as mx
import mlx.nn as nn
from mlx_lm.models.activations import swiglu
from mlx_lm.models.base import (
    BaseModelArgs,
    create_attention_mask,
    scaled_dot_product_attention,
)
from mlx_lm.models.cache import KVCache
from mlx_lm.models.switch_layers import SwitchGLU

__all__ = [
    "Attention",
    "DecoderLayer",
    "HYV3Model",
    "MLP",
    "Model",
    "ModelArgs",
    "MoE",
    "Router",
    "float32_router_logits",
]

# transformers' HYV3Config.default_theta
_DEFAULT_ROPE_THETA = 11_158_840.0
_LAYER_INDEX = re.compile(r"^model\.layers\.(\d+)\.")


@dataclass
class ModelArgs(BaseModelArgs):
    model_type: str
    vocab_size: int
    hidden_size: int
    intermediate_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    rms_norm_eps: float
    num_experts: int
    num_experts_per_tok: int
    num_shared_experts: int
    moe_intermediate_size: int
    router_scaling_factor: float
    rope_parameters: dict[str, Any] | None = None
    mlp_layer_types: list[str] | None = None
    first_k_dense_replace: int = 1
    max_position_embeddings: int = 131072
    tie_word_embeddings: bool = False
    attention_bias: bool = False
    mlp_bias: bool = False
    # transformers' default; the official config sets it to false.
    enable_moe_fp32_combine: bool = True
    num_nextn_predict_layers: int = 0

    def __post_init__(self) -> None:
        if self.mlp_layer_types is None:
            dense = min(self.first_k_dense_replace, self.num_hidden_layers)
            self.mlp_layer_types = ["dense"] * dense + ["sparse"] * (
                self.num_hidden_layers - dense
            )
        if len(self.mlp_layer_types) < self.num_hidden_layers:
            raise ValueError(
                f"mlp_layer_types has {len(self.mlp_layer_types)} entries for "
                f"{self.num_hidden_layers} layers"
            )

    @property
    def rope_theta(self) -> float:
        parameters = self.rope_parameters or {}
        rope_type = cast(str, parameters.get("rope_type", "default"))
        if rope_type != "default":
            raise NotImplementedError(f"Hy3 with rope_type {rope_type!r}")
        return float(cast(float, parameters.get("rope_theta", _DEFAULT_ROPE_THETA)))

    def is_sparse(self, layer_index: int) -> bool:
        assert self.mlp_layer_types is not None
        return self.mlp_layer_types[layer_index] == "sparse"


class Attention(nn.Module):
    """``HYV3Attention``: grouped-query attention with per-head query and key RMSNorms
    applied before the rotary embedding."""

    def __init__(self, args: ModelArgs) -> None:
        super().__init__()
        self.n_heads = args.num_attention_heads
        self.n_kv_heads = args.num_key_value_heads
        self.head_dim = args.head_dim
        # As transformers' head_dim**-0.5 (both are C pow).
        self.scale = math.pow(self.head_dim, -0.5)
        bias = args.attention_bias
        hidden = args.hidden_size
        self.q_proj = nn.Linear(hidden, self.n_heads * self.head_dim, bias=bias)
        self.k_proj = nn.Linear(hidden, self.n_kv_heads * self.head_dim, bias=bias)
        self.v_proj = nn.Linear(hidden, self.n_kv_heads * self.head_dim, bias=bias)
        self.o_proj = nn.Linear(self.n_heads * self.head_dim, hidden, bias=bias)
        self.q_norm = nn.RMSNorm(self.head_dim, eps=args.rms_norm_eps)
        self.k_norm = nn.RMSNorm(self.head_dim, eps=args.rms_norm_eps)
        self.rope = nn.RoPE(self.head_dim, traditional=False, base=args.rope_theta)

    def __call__(
        self,
        x: mx.array,
        mask: mx.array | str | None = None,
        cache: KVCache | None = None,
    ) -> mx.array:
        batch, length, _ = x.shape
        # -1 heads: tensor parallelism leaves each rank a slice of the heads.
        queries = self.q_proj(x).reshape(batch, length, -1, self.head_dim)
        keys = self.k_proj(x).reshape(batch, length, -1, self.head_dim)
        values = self.v_proj(x).reshape(batch, length, -1, self.head_dim)

        queries = self.q_norm(queries).transpose(0, 2, 1, 3)
        keys = self.k_norm(keys).transpose(0, 2, 1, 3)
        values = values.transpose(0, 2, 1, 3)

        if cache is not None:
            queries = self.rope(queries, offset=cache.offset)
            keys = self.rope(keys, offset=cache.offset)
            keys, values = cache.update_and_fetch(keys, values)
        else:
            queries = self.rope(queries)
            keys = self.rope(keys)

        output = scaled_dot_product_attention(
            queries,
            keys,
            values,
            cache=cache,
            scale=self.scale,
            mask=cast(mx.array | None, mask),
        )
        output = output.transpose(0, 2, 1, 3).reshape(batch, length, -1)
        return self.o_proj(output)


class MLP(nn.Module):
    """``HYV3MLP``: the dense layers' MLP and the shared expert."""

    def __init__(self, hidden_size: int, intermediate_size: int, bias: bool) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=bias)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=bias)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=bias)

    def __call__(self, x: mx.array) -> mx.array:
        return self.down_proj(swiglu(self.gate_proj(x), self.up_proj(x)))


@mx.compile
def _select_experts(
    logits: mx.array, expert_bias: mx.array, top_k: int, scaling_factor: float
) -> tuple[mx.array, mx.array]:
    """``HYV3TopKRouter.forward`` after the logits: sigmoid scores, top-k chosen on the
    scores plus the expert bias, weights the chosen (unbiased) scores normalised to sum 1
    and scaled by ``router_scaling_factor``. All in float32."""
    scores = mx.sigmoid(logits.astype(mx.float32))
    biased = scores + expert_bias.astype(mx.float32)
    indices = mx.stop_gradient(
        mx.argpartition(-biased, kth=top_k - 1, axis=-1)[..., :top_k]
    )
    weights = mx.take_along_axis(scores, indices, axis=-1)
    weights = weights / (weights.sum(axis=-1, keepdims=True) + 1e-20)
    return indices, weights * scaling_factor


class Router(nn.Module):
    """``HYV3TopKRouter`` with the MoE's ``e_score_correction_bias``."""

    def __init__(self, args: ModelArgs) -> None:
        super().__init__()
        self.top_k = args.num_experts_per_tok
        self.scaling_factor = args.router_scaling_factor
        self.gate = nn.Linear(args.hidden_size, args.num_experts, bias=False)
        self.expert_bias = mx.zeros((args.num_experts,), dtype=mx.float32)

    def route(self, x: mx.array) -> tuple[mx.array, mx.array]:
        """The chosen experts' indices and their (float32) weights."""
        return _select_experts(
            float32_router_logits(self.gate, x),
            self.expert_bias,
            self.top_k,
            self.scaling_factor,
        )


def float32_router_logits(gate: nn.Module, x: mx.array) -> mx.array:
    """The router logits in float32, as the reference computes them:
    ``F.linear(hidden_states.float(), weight.float())``.

    Rounding them to bf16 first (as mlx-lm's sigmoid-routed MoEs do) flips near-tied choices
    among 192 experts. Over 79 MoE layers that roughly doubled exo's bf16 drift from the
    float32 reference in the early layers and put exo's bf16 run outside transformers' own
    bf16 noise (mean KL 0.0135 vs 0.0102 at full depth); in float32, exo's per-layer drift
    equals transformers' bf16 drift. A quantized router (the 4-bit conversions keep it at 8
    bits) is multiplied with float32 scales and biases, so its dequantized weights are exact.
    """
    x = x.astype(mx.float32)
    weight = gate["weight"]
    assert isinstance(weight, mx.array)
    if isinstance(gate, nn.QuantizedLinear):
        mode = cast(str, gate.mode)
        if mode != "affine":
            return gate(x)
        scales = gate["scales"]
        biases = gate.get("biases")
        assert isinstance(scales, mx.array) and isinstance(biases, mx.array)
        return mx.quantized_matmul(
            x,
            weight,
            scales.astype(mx.float32),
            biases.astype(mx.float32),
            transpose=True,
            group_size=cast(int, gate.group_size),
            bits=cast(int, gate.bits),
            mode=mode,
        )
    return x @ weight.astype(mx.float32).T


class MoE(nn.Module):
    """``HYV3MoE``: routed experts plus an always-on shared expert."""

    def __init__(self, args: ModelArgs) -> None:
        super().__init__()
        self.router = Router(args)
        self.switch_mlp = SwitchGLU(
            args.hidden_size,
            args.moe_intermediate_size,
            args.num_experts,
            bias=args.mlp_bias,
        )
        self.shared_mlp = MLP(
            args.hidden_size,
            args.moe_intermediate_size * args.num_shared_experts,
            args.mlp_bias,
        )
        self.fp32_combine = args.enable_moe_fp32_combine

    def __call__(self, x: mx.array) -> mx.array:
        indices, weights = self.router.route(x)
        experts = self.switch_mlp(x, indices)
        routed = (experts * weights[..., None]).sum(axis=-2)
        shared = self.shared_mlp(x)
        if self.fp32_combine:
            return (routed + shared.astype(mx.float32)).astype(x.dtype)
        return routed.astype(x.dtype) + shared


class DecoderLayer(nn.Module):
    def __init__(self, args: ModelArgs, layer_index: int) -> None:
        super().__init__()
        self.self_attn = Attention(args)
        self.mlp: MoE | MLP = (
            MoE(args)
            if args.is_sparse(layer_index)
            else MLP(args.hidden_size, args.intermediate_size, args.mlp_bias)
        )
        self.input_layernorm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        self.post_attention_layernorm = nn.RMSNorm(
            args.hidden_size, eps=args.rms_norm_eps
        )

    def __call__(
        self,
        x: mx.array,
        mask: mx.array | str | None = None,
        cache: KVCache | None = None,
    ) -> mx.array:
        h = x + self.self_attn(self.input_layernorm(x), mask, cache)
        return h + self.mlp(self.post_attention_layernorm(h))


class HYV3Model(nn.Module):
    def __init__(self, args: ModelArgs) -> None:
        super().__init__()
        self.embed_tokens = nn.Embedding(args.vocab_size, args.hidden_size)
        self.layers = [DecoderLayer(args, i) for i in range(args.num_hidden_layers)]
        self.norm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)

    def __call__(
        self,
        inputs: mx.array,
        cache: Sequence[KVCache | None] | None = None,
        input_embeddings: mx.array | None = None,
    ) -> mx.array:
        h = (
            input_embeddings
            if input_embeddings is not None
            else self.embed_tokens(inputs)
        )
        layer_caches: Sequence[KVCache | None] = (
            cache if cache is not None else [None] * len(self.layers)
        )
        mask = create_attention_mask(h, layer_caches[0])
        for layer, layer_cache in zip(self.layers, layer_caches, strict=True):
            h = layer(h, mask, layer_cache)
        return self.norm(h)


class Model(nn.Module):
    def __init__(self, args: ModelArgs) -> None:
        super().__init__()
        self.args = args
        self.model_type = args.model_type
        self.model = HYV3Model(args)
        if not args.tie_word_embeddings:
            self.lm_head = nn.Linear(args.hidden_size, args.vocab_size, bias=False)

    def __call__(
        self,
        inputs: mx.array,
        cache: Sequence[KVCache | None] | None = None,
        input_embeddings: mx.array | None = None,
    ) -> mx.array:
        out = self.model(inputs, cache, input_embeddings)
        if self.args.tie_word_embeddings:
            return self.model.embed_tokens.as_linear(out)
        return self.lm_head(out)

    def sanitize(self, weights: dict[str, mx.array]) -> dict[str, mx.array]:
        """Map the official checkpoint's names to this model's; MLX conversions already
        use them. Drops the multi-token-prediction layer."""
        n_layers = self.args.num_hidden_layers
        sanitized: dict[str, mx.array] = {}
        for name, value in weights.items():
            match = _LAYER_INDEX.match(name)
            if match is not None and int(match.group(1)) >= n_layers:
                continue
            sanitized[name] = value

        for layer in range(n_layers):
            prefix = f"model.layers.{layer}.mlp"
            bias = sanitized.pop(f"{prefix}.expert_bias", None)
            if bias is not None:
                sanitized[f"{prefix}.router.expert_bias"] = bias
            for projection in ("gate_proj", "up_proj", "down_proj"):
                for part in ("weight", "scales", "biases"):
                    if f"{prefix}.experts.0.{projection}.{part}" not in sanitized:
                        continue
                    sanitized[f"{prefix}.switch_mlp.{projection}.{part}"] = mx.stack(
                        [
                            sanitized.pop(f"{prefix}.experts.{e}.{projection}.{part}")
                            for e in range(self.args.num_experts)
                        ]
                    )

        if self.args.tie_word_embeddings:
            sanitized.pop("lm_head.weight", None)
        return sanitized

    @property
    def layers(self) -> list[DecoderLayer]:
        return self.model.layers
