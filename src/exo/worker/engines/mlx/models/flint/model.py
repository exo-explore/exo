"""Recursive Flint (``flint``) model classes: mlx-lm's, with every RMSNorm computed the way
Flint's reference implementation does, and the decoder layer's small ops fused.

Hugging Face transformers' ``FlintRMSNorm`` (``GptOssRMSNorm``) normalises in float32 and
multiplies by the weight in float32, rounding to bf16 once:
``(weight * normalized).to(input_dtype)``. MLX's ``nn.RMSNorm`` rounds the normalised
activations to bf16 first and multiplies by the weight in bf16. Flint has four RMSNorms per
layer, two of them the per-head query and key norms in front of attention logits that reach
tens, so the extra rounding shows.

Measured in bf16 against transformers in float32 on 30 prompts (5,343 teacher-forced
positions, up to 7.4k tokens), mean KL goes from 0.0482 to 0.0454 and top-1 agreement where
the reference is at least 1 nat confident from 0.9897 to 0.9921. transformers' own bf16 run
scores 0.0452 and 0.9916 on the same positions.

Speed: a decode step at batch 1 is ~1,900 small kernels, and their launches, not the
weights' bandwidth, set its time. On Metal the layer runs fused kernels
(``kernels.py``) that compute exactly what the ops they replace compute, bit for bit:
- every norm in one kernel instead of four (casts to and from float32 around the norm);
- the residual add with the norm after it;
- the query and key norms with the rotary embedding;
- the MoE's weighted sum of the experts, the shared expert and the residual add;
- the router's weight kept in float32 (the router runs in float32), not cast every step;
- the experts' row indices built once per shape, not per matmul;
- q/k/v, and every gate/up pair, as one quantized matmul where that gives the same bits.
Prefill: a sliding-window layer (35 of 49) attends, per block of queries, only to the key
tiles its 513-token window reaches, instead of to every key under a mask.
Off Metal (CUDA, CPU) the layer runs mlx-lm's ops with float32 norms.
"""

from functools import cache, lru_cache
from typing import Any, cast

import mlx.core as mx
import mlx.nn as nn
from mlx.nn.layers.distributed import sum_gradients
from mlx_lm.kernels import expert_routing
from mlx_lm.models.activations import swiglu
from mlx_lm.models.base import scaled_dot_product_attention
from mlx_lm.models.cache import RotatingKVCache
from mlx_lm.models.flint import MLP, Attention, DecoderLayer, ModelArgs, SparseMoeBlock
from mlx_lm.models.flint import Model as MlxLmModel

from exo.worker.engines.mlx.models.flint import kernels

__all__ = ["Float32RMSNorm", "Model", "ModelArgs"]


class Float32RMSNorm(nn.RMSNorm):
    """``nn.RMSNorm`` that normalises and applies its weight in float32, then rounds once."""

    def __call__(self, x: mx.array) -> mx.array:
        return kernels.rms_norm_float32(x, self.weight, self.eps)


def _float32_norm(norm: nn.RMSNorm) -> Float32RMSNorm:
    return Float32RMSNorm(norm.weight.shape[0], eps=norm.eps)


# A stacked projection runs as one quantized matmul only while every part, and the stack,
# take MLX's per-row matrix-vector kernel: below its batch limit, 6 rows or more on every
# GPU generation. Larger inputs run each part on its rows of the stacked weight, which is
# the original matmul on the original values. Either way the result is bit for bit the same.
_STACKED_MAX_ROWS = 4


def _quantized_affine(linear: nn.Module) -> bool:
    """A 4/8-bit affine-quantized linear (or switch linear) without a bias, whose forward
    is one ``quantized_matmul``/``gather_qmm``: ``QuantizedLinear``, its all-to-sharded
    variant, ``QuantizedSwitchLinear``."""
    return (
        all(name in linear for name in ("weight", "scales", "biases"))
        and "bias" not in linear
        and getattr(linear, "mode", None) == "affine"
        and getattr(linear, "global_scale", None) is None
        and isinstance(getattr(linear, "group_size", None), int)
        and isinstance(getattr(linear, "bits", None), int)
    )


class StackedQuantizedLinear(nn.Module):
    """Quantized linears with one input, their weights stacked along the output rows."""

    def __init__(self, parts: list[nn.Module]) -> None:
        super().__init__()
        self.weight = mx.concatenate([cast(mx.array, p["weight"]) for p in parts], -2)
        self.scales = mx.concatenate([cast(mx.array, p["scales"]) for p in parts], -2)
        self.biases = mx.concatenate([cast(mx.array, p["biases"]) for p in parts], -2)
        mx.eval(self.weight, self.scales, self.biases)
        self.group_size = cast(int, getattr(parts[0], "group_size"))  # noqa: B009
        self.bits = cast(int, getattr(parts[0], "bits"))  # noqa: B009
        rows = [cast(mx.array, p["weight"]).shape[-2] for p in parts]
        self.bounds = [sum(rows[:i]) for i in range(len(rows) + 1)]

    @staticmethod
    def stackable(parts: list[nn.Module]) -> bool:
        if not all(_quantized_affine(p) for p in parts):
            return False
        first = parts[0]
        weights = [cast(mx.array, p["weight"]) for p in parts]
        return all(
            getattr(p, "group_size") == getattr(first, "group_size")  # noqa: B009
            and getattr(p, "bits") == getattr(first, "bits")  # noqa: B009
            and w.shape[:-2] == weights[0].shape[:-2]
            and w.shape[-1] == weights[0].shape[-1]
            and w.dtype == weights[0].dtype
            for p, w in zip(parts, weights, strict=True)
        )

    def _matmul(self, x: mx.array, start: int, stop: int) -> mx.array:
        return mx.quantized_matmul(
            x,
            self.weight[start:stop],
            self.scales[start:stop],
            self.biases[start:stop],
            transpose=True,
            group_size=self.group_size,
            bits=self.bits,
            mode="affine",
        )

    def stacked(self, x: mx.array) -> mx.array | None:
        """All parts' outputs side by side, when one matmul gives the same bits."""
        if x.size // x.shape[-1] > _STACKED_MAX_ROWS:
            return None
        return self._matmul(x, 0, self.bounds[-1])

    def split(self, x: mx.array) -> list[mx.array]:
        """Every part's output."""
        stacked = self.stacked(x)
        if stacked is not None:
            return [
                stacked[..., start:stop]
                for start, stop in zip(self.bounds[:-1], self.bounds[1:], strict=True)
            ]
        return [
            self._matmul(x, start, stop)
            for start, stop in zip(self.bounds[:-1], self.bounds[1:], strict=True)
        ]


def _stack(module: nn.Module, names: list[str]) -> StackedQuantizedLinear | None:
    """Replace ``module``'s linears ``names`` by one stacked projection, if they allow it
    and the fused kernels run (Metal)."""
    if not kernels.fused_kernels_available():
        return None
    parts = [cast(nn.Module, module[name]) for name in names]
    if not StackedQuantizedLinear.stackable(parts):
        return None
    stacked = StackedQuantizedLinear(parts)
    for name in names:
        del module[name]
    return stacked


class FlintMLP(MLP):
    """mlx-lm's MLP, with gate_proj and up_proj stacked at the first call."""

    def __call__(self, x: mx.array) -> mx.array:
        if "gate_up_proj" not in self and "gate_proj" in self:
            stacked = _stack(self, ["gate_proj", "up_proj"])
            if stacked is not None:
                self.gate_up_proj = stacked
        if "gate_up_proj" not in self:
            return super().__call__(x)
        gate, up = cast(StackedQuantizedLinear, self["gate_up_proj"]).split(x)
        return self.down_proj(swiglu(gate, up))


# Sliding-window prefill: query blocks of this many rows, each attending only to the key
# tiles its window reaches.
_SLIDING_QUERY_BLOCK = 256
# Key slices start on a multiple of this, so they cover the same key tiles (16 wide for
# 128-value heads) as attention over all the keys does.
_KEY_TILE_ALIGN = 64


@cache
def _tiled_attention_kernel() -> bool:
    """MLX's prefill attention (``steel_attention``) runs on Apple GPUs up to generation 16
    (M1 to M4); newer ones use another kernel. In it a fully masked key tile changes
    nothing: its scores are the float minimum, so it adds exp2(min - max) = 0 to the row sums
    and scales the running output by exp2(0) = 1 (the sinks keep the running max finite).
    Leaving such tiles out is exact there."""
    architecture = str(mx.metal.device_info().get("architecture", ""))
    digits = "".join(c for c in architecture.removeprefix("applegpu_g") if c.isdigit())
    return digits != "" and int(digits) <= 16


class FlintAttention(Attention):
    def __init__(self, args: ModelArgs, layer_idx: int) -> None:
        super().__init__(args, layer_idx)
        self.q_norm = _float32_norm(self.q_norm)
        self.k_norm = _float32_norm(self.k_norm)
        sliding = args.layer_types[layer_idx] == "sliding_attention"
        self.sliding_window: int | None = args.sliding_window if sliding else None

    def _sliding_attention(
        self,
        queries: mx.array,
        keys: mx.array,
        values: mx.array,
        mask: object,
        cache: object,
    ) -> mx.array | None:
        """Prefill attention of a sliding-window layer over only the key tiles each block of
        queries reaches, or None to attend over all keys.

        A rotating cache hands back its last window - 1 keys and the new ones, in order,
        with the mask ``create_causal_mask(L, offset, window)``: query i attends keys
        offset + i - window + 1 .. offset + i. Over all keys, every other tile is fully
        masked, and the attention kernel spends most of a long prefill on them."""
        window = self.sliding_window
        length, key_length = queries.shape[2], keys.shape[2]
        if (
            window is None
            or not isinstance(mask, mx.array)
            or mask.ndim != 2
            or mask.dtype != mx.bool_
            or mask.shape != (length, key_length)
            or type(cache) is not RotatingKVCache
            or length < 2 * _SLIDING_QUERY_BLOCK
            or not _tiled_attention_kernel()
        ):
            return None
        offset = key_length - length
        blocks: list[mx.array] = []
        for start in range(0, length, _SLIDING_QUERY_BLOCK):
            stop = min(length, start + _SLIDING_QUERY_BLOCK)
            first = max(0, offset + start - window + 1)
            first -= first % _KEY_TILE_ALIGN
            last = offset + stop
            blocks.append(
                mx.fast.scaled_dot_product_attention(
                    queries[:, :, start:stop],
                    keys[:, :, first:last],
                    values[:, :, first:last],
                    scale=self.scale,
                    mask=mask[start:stop, first:last],
                    sinks=self.softmax_offset,
                )
            )
        return mx.concatenate(blocks, axis=2)

    def _rope_base(self) -> float | None:
        """The RoPE base the fused kernel applies, or None without RoPE."""
        return None if self.rope is None else self.rope.base

    def _fusable(self, x: mx.array) -> bool:
        rope = self.rope
        return (
            kernels.qk_norm_rope_fusable(x, self.q_norm.weight.shape[0])
            and self.q_norm.eps == self.k_norm.eps
            and (
                rope is None
                or (
                    not rope.traditional
                    and rope.scale == 1.0
                    and rope.dims == self.q_norm.weight.shape[0]
                )
            )
        )

    def _projections(
        self, x: mx.array
    ) -> tuple[mx.array, mx.array, mx.array, int] | None:
        """(q, k, v, k's first column in k): from one stacked matmul when it gives the same
        bits (then q, k and v are one array and its column slices), else from three."""
        if "qkv_proj" not in self and "q_proj" in self:
            stacked = _stack(self, ["q_proj", "k_proj", "v_proj"])
            if stacked is not None:
                self.qkv_proj = stacked
        if "qkv_proj" not in self:
            return None
        projection = cast(StackedQuantizedLinear, self["qkv_proj"])
        qkv = projection.stacked(x)
        if qkv is not None:
            start, stop = projection.bounds[2], projection.bounds[3]
            return qkv, qkv, qkv[..., start:stop], projection.bounds[1]
        q, k, v = projection.split(x)
        return q, k, v, 0

    def __call__(
        self,
        x: mx.array,
        mask: mx.array | None = None,
        cache: Any | None = None,
    ) -> mx.array:
        if not self._fusable(x):
            return super().__call__(x, mask, cache)
        batch, length, _ = x.shape
        projections = self._projections(x)
        if projections is None:
            q, k, v, k_column = self.q_proj(x), self.k_proj(x), self.v_proj(x), 0
        else:
            q, k, v, k_column = projections
        offset = cast(int | mx.array, cache.offset) if cache is not None else 0
        queries, keys = kernels.qk_norm_rope(
            q,
            k,
            q_weight=self.q_norm.weight,
            k_weight=self.k_norm.weight,
            eps=self.q_norm.eps,
            n_heads=self.n_heads,
            n_kv_heads=self.n_kv_heads,
            offset=offset,
            rope_base=self._rope_base(),
            k_column=k_column,
        )
        values = v.reshape(batch, length, self.n_kv_heads, -1).transpose(0, 2, 1, 3)
        if cache is not None:
            keys, values = cast(
                tuple[mx.array, mx.array],
                cache.update_and_fetch(keys, values),  # pyright: ignore[reportAny]
            )
        output = self._sliding_attention(queries, keys, values, mask, cache)
        if output is None:
            output = scaled_dot_product_attention(
                queries,
                keys,
                values,
                cache=cache,
                scale=self.scale,
                mask=mask,
                sinks=self.softmax_offset,
            )
        return self.o_proj(output.transpose(0, 2, 1, 3).reshape(batch, length, -1))


@lru_cache(maxsize=32)
def _row_indices(shape: tuple[int, ...]) -> mx.array:
    """The lhs indices ``mx.gather_qmm`` builds when it gets none: one per row of x. Only
    for decode-sized inputs, so few shapes."""
    size = 1
    for dim in shape:
        size *= dim
    rows = mx.arange(size, dtype=mx.uint32).reshape(*shape)
    mx.eval(rows)
    return rows


def _gather(
    x: mx.array,
    weights: tuple[mx.array, mx.array, mx.array],
    group_size: int,
    bits: int,
    indices: mx.array,
    sorted_indices: bool,
) -> mx.array:
    """``QuantizedSwitchLinear.__call__`` (affine, no bias); unsorted (few rows), with
    cached row indices."""
    weight, scales, biases = weights
    return mx.gather_qmm(
        x,
        weight,
        scales,
        biases,
        lhs_indices=None if sorted_indices else _row_indices(tuple(x.shape[:-2])),
        rhs_indices=indices,
        transpose=True,
        group_size=group_size,
        bits=bits,
        mode="affine",
        sorted_indices=sorted_indices,
    )


def _arrays(linear: nn.Module) -> tuple[mx.array, mx.array, mx.array]:
    return (
        cast(mx.array, linear["weight"]),
        cast(mx.array, linear["scales"]),
        cast(mx.array, linear["biases"]),
    )


class FlintSparseMoeBlock(SparseMoeBlock):
    def __init__(self, args: ModelArgs) -> None:
        super().__init__(args)
        self.shared_expert = FlintMLP(
            args.hidden_size, args.shared_expert_intermediate_size
        )

    def _gate_up(self) -> StackedQuantizedLinear | None:
        """The experts' gate and up projections stacked along their rows, at the first call:
        one gather instead of two."""
        glu = self.switch_mlp
        if "gate_up_proj" not in glu and "gate_proj" in glu:
            stacked = _stack(glu, ["gate_proj", "up_proj"])
            if stacked is not None:
                glu.gate_up_proj = stacked
        return (
            cast(StackedQuantizedLinear, glu["gate_up_proj"])
            if "gate_up_proj" in glu
            else None
        )

    def _experts(self, x: mx.array, indices: mx.array) -> mx.array:
        """``self.switch_mlp(x, indices)`` (mlx-lm's SwitchGLU), with the gate and up
        projections in one gather and, for a few tokens, cached row indices. From 64 rows
        it sorts the rows by expert, as mlx-lm does."""
        glu = self.switch_mlp
        gate_up = self._gate_up()
        down = glu.down_proj
        if gate_up is None or not _quantized_affine(down):
            return glu(x, indices)
        group_size, bits = gate_up.group_size, gate_up.bits
        x = mx.expand_dims(x, (-2, -3))
        sort = indices.size >= 64
        inverse = None
        rows = indices
        if sort:
            # mlx-lm's _gather_sort
            flat = indices.flatten()
            order = mx.argsort(flat)
            inverse = mx.argsort(order)
            x = x.flatten(0, -3)[order // indices.shape[-1]]
            rows = flat[order]
        split = gate_up.bounds[1]
        # Every gathered row is one token-expert pair (M = 1): the matrix-vector gather, or,
        # sorted, the gather that tiles N by 64 without splitting K. Both give each output
        # the same reduction for the stack as for gate and up alone.
        stack = (gate_up.weight, gate_up.scales, gate_up.biases)
        both = _gather(x, stack, group_size, bits, rows, sort)
        x_gate, x_up = both[..., :split], both[..., split:]
        hidden = cast(mx.array, glu.activation(x_up, x_gate))
        out = _gather(
            hidden,
            _arrays(down),
            cast(int, down.group_size),
            cast(int, down.bits),
            rows,
            sort,
        )  # noqa: B009
        if inverse is not None:
            # mlx-lm's _scatter_unsort
            out = mx.unflatten(out[inverse], 0, indices.shape)
        return out.squeeze(-2)

    def __call__(
        self,
        x: mx.array,
        residual: mx.array | None = None,
        x_float32: mx.array | None = None,
    ) -> mx.array:
        """The block's output, plus ``residual`` when given (the decoder layer passes its
        residual stream when the block is not sharded). ``x_float32`` is
        ``x.astype(float32)`` when the caller has it."""
        if not kernels.fused_kernels_available():
            out = super().__call__(x)
            return out if residual is None else residual + out
        group = self.sharding_group
        if group is not None:
            x = sum_gradients(group)(x)
        logits = self.gate(x.astype(mx.float32) if x_float32 is None else x_float32)
        indices, weights = expert_routing(
            logits,
            self._zero_bias,
            top_k=self.top_k,
            multiplier=1.0,
            score_eps=0.0,
            route_scale=1.0,
        )
        routed = self._experts(x, indices)
        shared = self.shared_expert(x)
        if group is None:
            return kernels.moe_combine(routed, weights, shared, residual)
        out = mx.distributed.all_sum(
            kernels.moe_combine(routed, weights, shared), group=group
        )
        return out if residual is None else residual + out


class FlintDecoderLayer(DecoderLayer):
    def __init__(self, args: ModelArgs, layer_idx: int) -> None:
        super().__init__(args, layer_idx)
        self.self_attn = FlintAttention(args, layer_idx)
        if isinstance(self.mlp, SparseMoeBlock):
            self.mlp = FlintSparseMoeBlock(args)
        else:
            self.mlp = FlintMLP(args.hidden_size, args.intermediate_size)
        self.input_layernorm = _float32_norm(self.input_layernorm)
        self.post_attention_layernorm = _float32_norm(self.post_attention_layernorm)

    def __call__(
        self,
        x: mx.array,
        mask: mx.array | None = None,
        cache: Any | None = None,
    ) -> mx.array:
        attention = self.self_attn(self.input_layernorm(x), mask, cache)
        norm = self.post_attention_layernorm
        mlp = self.mlp
        # Tensor parallelism wraps the block to all-reduce its output: then add here.
        if isinstance(mlp, FlintSparseMoeBlock) and mlp.sharding_group is None:
            h, normed, normed_float32 = kernels.add_rms_norm_float32_with_copy(
                x, attention, norm.weight, norm.eps
            )
            return mlp(normed, residual=h, x_float32=normed_float32)
        h, normed = kernels.add_rms_norm_float32(x, attention, norm.weight, norm.eps)
        return h + mlp(normed)


class Model(MlxLmModel):
    def __init__(self, args: ModelArgs) -> None:
        super().__init__(args)
        self.model.layers = [
            FlintDecoderLayer(args, i) for i in range(args.num_hidden_layers)
        ]
        self.model.norm = _float32_norm(self.model.norm)

    def sanitize(self, weights: dict[str, mx.array]) -> dict[str, mx.array]:
        """The router runs in float32 (``gate(x.astype(float32))``): keep its weight in
        float32 instead of casting it at every step. Same values, same matmul."""
        weights = super().sanitize(weights)
        return {
            name: value.astype(mx.float32)
            if name.endswith(".mlp.gate.weight") and value.dtype != mx.float32
            else value
            for name, value in weights.items()
        }
