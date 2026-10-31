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
- On Metal, the layers run a fused path (``kernels.py``, ``fused.py``) that computes the same
  values as the code above, bit for bit, in fewer kernels: the residual add with the next
  RMSNorm, the query and key norms with RoPE, the router's selection, the MoE's combination,
  the experts' gate and up projections in one gather and SwiGLU inside the 4-bit down
  projection. Elsewhere the code above runs as written.
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

from exo.worker.engines.mlx.models.hy_v3 import kernels
from exo.worker.engines.mlx.models.hy_v3.fused import (
    ExpertStack,
    StackedProjection,
    arrays,
    gather,
    gather_uses_matrix_vector,
    quantized_affine,
)

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
        if not self._fusable(x):
            return self._reference(x, mask, cache)
        batch, length, _ = x.shape
        head_dim = self.head_dim
        q, k, v = self.q_proj(x), self.k_proj(x), self.v_proj(x)
        queries, keys, values = kernels.qk_norm_rope(
            q,
            k,
            v,
            q_weight=self.q_norm.weight,
            k_weight=self.k_norm.weight,
            eps=self.q_norm.eps,
            # -1 heads: tensor parallelism leaves each rank a slice of the heads.
            n_heads=q.shape[-1] // head_dim,
            n_kv_heads=k.shape[-1] // head_dim,
            offset=cache.offset if cache is not None else 0,
            rope_base=self.rope.base,
        )
        if cache is not None:
            keys, values = cache.update_and_fetch(keys, values)
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

    def _fusable(self, x: mx.array) -> bool:
        """The fused query/key kernel computes what the layer's norms and RoPE do."""
        return (
            self.q_norm.eps == self.k_norm.eps
            and not self.rope.traditional
            and self.rope.scale == 1.0
            and self.rope.dims == self.head_dim
            and self.k_norm.weight.dtype == self.q_norm.weight.dtype
            and kernels.qk_norm_rope_fusable(x, self.q_norm.weight, self.head_dim)
        )

    def _reference(
        self,
        x: mx.array,
        mask: mx.array | str | None,
        cache: KVCache | None,
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
        stack = self._gate_up() if kernels.fused_kernels_available() else None
        gate_up = stack(x) if stack is not None else None
        if stack is None or gate_up is None:
            return self.down_proj(swiglu(self.gate_proj(x), self.up_proj(x)))
        # The exact class: a sharded down projection all-reduces in its own call.
        down = cast(nn.Module, self.down_proj)
        if (
            x.size == x.shape[-1]
            and type(down) is nn.QuantizedLinear
            and quantized_affine(down)
            and kernels.swiglu_down_fusable(
                gate_up,
                arrays(down),
                cast(int, getattr(down, "group_size")),  # noqa: B009
                cast(int, getattr(down, "bits")),  # noqa: B009
            )
        ):
            # One row: MLX's down projection is the matrix-vector kernel swiglu_down
            # reproduces, so the activation runs inside it. (A sharded down projection,
            # which all-reduces, is another class and keeps its own call.)
            out = kernels.swiglu_down(gate_up, arrays(down))
            return out.reshape(*x.shape[:-1], out.shape[-1])
        split = stack.bounds[1]
        return down(swiglu(gate_up[..., :split], gate_up[..., split:]))

    def _gate_up(self) -> StackedProjection | None:
        """gate and up stacked at the first call, if their quantization allows it."""
        if "_gate_up_stack" not in self.__dict__:
            stack = StackedProjection.build([self.gate_proj, self.up_proj])
            object.__setattr__(self, "_gate_up_stack", stack)
            # The separate weights are freed: return them to the system, not MLX's cache.
            mx.clear_cache()
        return cast(StackedProjection | None, self.__dict__["_gate_up_stack"])


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

    def route_fused(
        self, x: mx.array, x_float32: mx.array | None
    ) -> tuple[mx.array, mx.array] | None:
        """``route`` through the fused kernel, on float32 weights converted once (exact).
        None where the kernel doesn't apply."""
        logits = self._float32_logits(
            x.astype(mx.float32) if x_float32 is None else x_float32
        )
        if logits is None or not kernels.route_fusable(logits, self.top_k):
            return None
        bias, scaling = self._float32_constants()
        return kernels.route(logits, bias, self.top_k, scaling)

    def _float32_logits(self, x_float32: mx.array) -> mx.array | None:
        """``float32_router_logits`` with the float32 weights (or scales and biases) kept
        from the first call instead of converted at every call: the same values, the same
        matmul."""
        if "_float32_weights" not in self.__dict__:
            gate = self.gate
            weights: tuple[mx.array, ...] | None = None
            if isinstance(gate, nn.QuantizedLinear):
                if quantized_affine(gate):
                    weight, scales, biases = arrays(gate)
                    weights = (
                        weight,
                        scales.astype(mx.float32),
                        biases.astype(mx.float32),
                    )
            elif type(gate) is nn.Linear and "bias" not in gate:
                weights = (gate.weight.astype(mx.float32).T,)
            if weights is not None:
                mx.eval(*weights)
            object.__setattr__(self, "_float32_weights", weights)
        weights = cast(tuple[mx.array, ...] | None, self.__dict__["_float32_weights"])
        if weights is None:
            return None
        if len(weights) == 1:
            return x_float32 @ weights[0]
        gate: nn.Module = self.gate
        return mx.quantized_matmul(
            x_float32,
            weights[0],
            weights[1],
            weights[2],
            transpose=True,
            group_size=cast(int, getattr(gate, "group_size")),  # noqa: B009
            bits=cast(int, getattr(gate, "bits")),  # noqa: B009
            mode="affine",
        )

    def _float32_constants(self) -> tuple[mx.array, mx.array]:
        if "_float32_constants_value" not in self.__dict__:
            bias = self.expert_bias.astype(mx.float32)
            scaling = mx.array(self.scaling_factor, dtype=mx.float32)
            mx.eval(bias, scaling)
            object.__setattr__(self, "_float32_constants_value", (bias, scaling))
        return cast(
            tuple[mx.array, mx.array], self.__dict__["_float32_constants_value"]
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

    def __call__(
        self,
        x: mx.array,
        x_float32: mx.array | None = None,
        residual: mx.array | None = None,
    ) -> mx.array:
        """The block's output, plus ``residual`` when given (the decoder layer passes its
        residual stream when the block is not sharded). ``x_float32`` is
        ``x.astype(float32)`` when the caller has it."""
        parts = self._fused_parts(x, x_float32)
        if parts is None:
            out = self._reference(x)
            return out if residual is None else residual + out
        y, weights, shared = parts
        return kernels.moe_combine(
            y, weights, shared, residual, float32_combine=self.fp32_combine
        )

    def with_next_norm(
        self,
        x: mx.array,
        x_float32: mx.array | None,
        residual: mx.array,
        next_norm: nn.RMSNorm,
    ) -> tuple[mx.array, mx.array]:
        """``residual + self(x)`` and the next layer's input norm of it, in one kernel."""
        parts = self._fused_parts(x, x_float32)
        if parts is None:
            out = residual + self._reference(x)
            return out, next_norm(out)
        y, weights, shared = parts
        return kernels.moe_combine_norm(
            y,
            weights,
            shared,
            residual,
            next_norm.weight,
            next_norm.eps,
            float32_combine=self.fp32_combine,
        )

    def _fused_parts(
        self, x: mx.array, x_float32: mx.array | None
    ) -> tuple[mx.array, mx.array, mx.array] | None:
        """The experts' outputs, their weights and the shared expert's output, through the
        fused path; None where it doesn't apply."""
        stack = self._stack() if kernels.fused_kernels_available() else None
        if stack is None:
            return None
        routing = self.router.route_fused(x, x_float32)
        indices, weights = routing if routing is not None else self.router.route(x)
        return self._experts(x, indices, stack), weights, self.shared_mlp(x)

    def _reference(self, x: mx.array) -> mx.array:
        indices, weights = self.router.route(x)
        experts = self.switch_mlp(x, indices)
        routed = (experts * weights[..., None]).sum(axis=-2)
        shared = self.shared_mlp(x)
        if self.fp32_combine:
            return (routed + shared.astype(mx.float32)).astype(x.dtype)
        return routed.astype(x.dtype) + shared

    def _stack(self) -> ExpertStack | None:
        """The experts' gate and up projections stacked at the first call."""
        if "_expert_stack" not in self.__dict__:
            glu = self.switch_mlp
            object.__setattr__(
                self, "_expert_stack", ExpertStack.build(glu.gate_proj, glu.up_proj)
            )
            # The separate weights are freed: return them to the system, not MLX's cache
            # (over 79 layers it would hold as much as the stacks).
            mx.clear_cache()
        return cast(ExpertStack | None, self.__dict__["_expert_stack"])

    def _experts(self, x: mx.array, indices: mx.array, stack: ExpertStack) -> mx.array:
        """``self.switch_mlp(x, indices)`` (mlx-lm's ``SwitchGLU``) with gate and up in one
        gather. From 64 index pairs it sorts them by expert, as mlx-lm does."""
        down = self.switch_mlp.down_proj
        x = mx.expand_dims(x, (-2, -3))
        sort = indices.size >= 64
        rows = indices
        inverse: mx.array | None = None
        if sort:
            # mlx-lm's _gather_sort
            top_k = indices.shape[-1]
            flat = indices.flatten()
            order = mx.argsort(flat)
            inverse = mx.argsort(order)
            x = x.flatten(0, -3)[order // top_k]
            rows = flat[order]
        both = gather(x, stack.gate_up, stack.group_size, stack.bits, rows, sort)
        if quantized_affine(down):
            down_arrays = arrays(down)
            down_group_size = cast(int, down.group_size)
            down_bits = cast(int, down.bits)
            if gather_uses_matrix_vector(
                indices.size, down_arrays[0].shape[0], sort
            ) and kernels.swiglu_down_fusable(
                both, down_arrays, down_group_size, down_bits
            ):
                # The down gather multiplies one row per token-expert pair with MLX's
                # matrix-vector kernel, which swiglu_down reproduces with the activation
                # inside.
                out = kernels.swiglu_down(both, down_arrays, rows)
                if inverse is not None:
                    out = out[inverse]
                return out.reshape(*indices.shape, out.shape[-1])
        split = stack.intermediate
        hidden = swiglu(both[..., :split], both[..., split:])
        out = cast(mx.array, down(hidden, rows, sorted_indices=sort))
        if inverse is not None:
            # mlx-lm's _scatter_unsort
            out = mx.unflatten(out[inverse], 0, indices.shape)
        return out.squeeze(-2)


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
        normed_input = self._take_handed(x)
        if normed_input is None:
            normed_input = self.input_layernorm(x)
        attention = self.self_attn(normed_input, mask, cache)
        norm = self.post_attention_layernorm
        mlp = self.mlp
        following = cast("DecoderLayer | None", self.__dict__.get("_next_layer"))
        # Tensor parallelism wraps the MoE to all-reduce its output: then add here.
        if isinstance(mlp, MoE):
            h, normed, normed_float32 = kernels.add_rms_norm(
                x, attention, norm.weight, norm.eps, float32_copy=True
            )
            if following is None:
                return mlp(normed, x_float32=normed_float32, residual=h)
            out, next_normed = mlp.with_next_norm(
                normed, normed_float32, h, following.input_layernorm
            )
            following.hand(out, next_normed)
            return out
        h, normed, _ = kernels.add_rms_norm(x, attention, norm.weight, norm.eps)
        if following is None:
            return h + mlp(normed)
        next_norm = following.input_layernorm
        out, next_normed, _ = kernels.add_rms_norm(
            h, mlp(normed), next_norm.weight, next_norm.eps
        )
        following.hand(out, next_normed)
        return out

    def link(self, following: "DecoderLayer | None") -> None:
        """The layer called next on this rank, which then takes its input norm from this
        layer's last kernel."""
        object.__setattr__(self, "_next_layer", following)

    def hand(self, x: mx.array, normed: mx.array) -> None:
        """``normed`` is ``self.input_layernorm(x)``, for the call whose input is ``x``."""
        object.__setattr__(self, "_handed", (x, normed))

    def _take_handed(self, x: mx.array) -> mx.array | None:
        handed = cast("tuple[mx.array, mx.array] | None", self.__dict__.get("_handed"))
        if handed is None:
            return None
        object.__setattr__(self, "_handed", None)
        return handed[1] if handed[0] is x else None


def _decoder_layer(layer: nn.Module) -> DecoderLayer | None:
    """The decoder layer, also inside a pipeline wrapper."""
    inner = layer
    while not isinstance(inner, DecoderLayer):
        wrapped = cast(nn.Module | None, getattr(inner, "original_layer", None))
        if wrapped is None or wrapped is inner:
            return None
        inner = wrapped
    return inner


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
        # Each layer hands the next one on this rank its input norm (see DecoderLayer).
        fused = kernels.fused_kernels_available()
        inner = [_decoder_layer(layer) for layer in self.layers]
        for layer, following in zip(inner, [*inner[1:], None], strict=True):
            if layer is not None:
                layer.link(following if fused else None)
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
