"""
Asymmetric Tensor Parallelism for heterogeneous clusters.

When nodes have different amounts of RAM, standard 50/50 tensor parallelism
fails because the smaller node can't hold half the weights. Asymmetric TP
splits each weight tensor proportionally to available memory (e.g. 75/25)
so both nodes compute every layer simultaneously.

Mathematical correctness:
  Column parallel: y = x @ [W_a; W_b]^T = [x @ W_a^T, x @ W_b^T]
  Row parallel:    y = x_a @ W_a^T + x_b @ W_b^T = x @ W^T  (via all_sum)
  Both hold regardless of the split ratio.

Usage:
  asymmetric_tensor_auto_parallel(model, group, ratios=[0.75, 0.25])
"""
# pyright: reportAny=false, reportUnknownMemberType=false, reportUnknownArgumentType=false, reportUnknownVariableType=false

from __future__ import annotations

from typing import Any

import mlx.core as mx
import mlx.nn as nn
from mlx_lm.models.qwen3_5 import DecoderLayer as Qwen3_5DecoderLayer
from mlx_lm.models.qwen3_5 import GatedDeltaNet
from mlx_lm.models.qwen3_5 import SparseMoeBlock as Qwen3_5SparseMoeBlock
from mlx_lm.models.qwen3_next import Qwen3NextAttention as Attention
from mlx_lm.models.qwen3_next import Qwen3NextMLP as Qwen3NextMLP
from mlx_lm.models.qwen3_next import Qwen3NextSparseMoeBlock as SparseMoeBlock

try:
    from exo.shared.logging import logger
except ImportError:
    import logging

    logger = logging.getLogger(__name__)


def find_valid_ratios(
    memory_fractions: list[float],
    hidden_size: int,
    num_attention_heads: int,
    num_key_value_heads: int,
    num_experts: int = 0,
    moe_intermediate_size: int = 0,
    linear_num_value_heads: int = 0,
    linear_num_key_heads: int = 0,
    quantization_group_size: int = 64,
) -> list[float] | None:
    """
    Find valid split ratios for asymmetric TP given model dimensions and memory fractions.

    A valid ratio must produce integer dimensions for all split tensors,
    and all split dimensions must be divisible by the quantization group size.

    Returns a list of ratios (one per node) that sum to 1.0, or None if no valid
    ratio exists. Currently supports 2 nodes only.
    """
    if len(memory_fractions) != 2:
        logger.warning("Asymmetric TP currently only supports 2 nodes")
        return None

    # Key dimensions that must split cleanly
    key_dims = [
        num_attention_heads,
        hidden_size,
    ]
    if linear_num_value_heads > 0:
        key_dims.extend([linear_num_value_heads, linear_num_key_heads])
    if num_experts > 0 and moe_intermediate_size > 0:
        key_dims.append(moe_intermediate_size)

    target_ratio = memory_fractions[0]

    # Try ratios of the form n/d where d is a power of 2 or common denominator
    # that produces clean splits. Test denominators 2..32.
    best_ratio = None
    best_distance = float("inf")

    for denom in [2, 4, 8, 16, 32]:
        for numer in range(1, denom):
            ratio = numer / denom
            if ratio <= 0.5 or ratio > 0.95:
                continue

            # Check all dimensions split cleanly
            valid = True
            for dim in key_dims:
                # dim * ratio must be EXACTLY integer (for head counts)
                exact = dim * ratio
                if exact != int(exact):
                    valid = False
                    break
                a = int(exact)
                b = dim - a
                if a <= 0 or b <= 0:
                    valid = False
                    break
                # For quantized weights, split dims must be divisible by 8
                if dim > quantization_group_size and (a % 8 != 0 or b % 8 != 0):
                    valid = False
                    break

            if valid:
                distance = abs(ratio - target_ratio)
                if distance < best_distance:
                    best_distance = distance
                    best_ratio = ratio

    if best_ratio is None:
        return None

    return [best_ratio, 1.0 - best_ratio]


def _split_at(tensor: mx.array, axis: int, ratio: float) -> tuple[mx.array, mx.array]:
    """Split tensor at ratio point along axis."""
    sp = int(tensor.shape[axis] * ratio)
    parts = mx.split(tensor, [sp], axis=axis)
    return mx.contiguous(parts[0]), mx.contiguous(parts[1])


def _my_shard(tensor: mx.array, axis: int, rank: int, ratio: float) -> mx.array:
    """Get rank's portion of an asymmetric split."""
    parts = _split_at(tensor, axis, ratio)
    return parts[0] if rank == 0 else parts[1]


def _shard_quantized_ats(
    layer: Any,
    axis: int,
    rank: int,
    ratio: float,
    segments: list[int] | None = None,
) -> None:
    """Shard quantized linear all-to-sharded (output dim split)."""
    if segments is not None:
        w: mx.array = layer.weight
        seg_parts_w = mx.split(w, segments, axis=axis)
        my_w_parts = [_my_shard(p, axis, rank, ratio) for p in seg_parts_w]
        layer.weight = mx.contiguous(mx.concatenate(my_w_parts, axis=axis))
        for attr in ["scales", "biases"]:
            t: mx.array | None = getattr(layer, attr, None)
            if t is None:
                continue
            t_seg = [int(s * t.shape[axis] / w.shape[axis]) for s in segments]
            t_parts = mx.split(t, t_seg, axis=axis)
            my_parts = [_my_shard(p, axis, rank, ratio) for p in t_parts]
            setattr(layer, attr, mx.contiguous(mx.concatenate(my_parts, axis=axis)))
    else:
        for attr in ["weight", "scales", "biases"]:
            t_val: mx.array | None = getattr(layer, attr, None)
            if t_val is None:
                continue
            setattr(layer, attr, _my_shard(t_val, axis, rank, ratio))


def _shard_quantized_sta(layer: Any, rank: int, ratio: float) -> None:
    """Shard quantized linear sharded-to-all (input dim, axis -1)."""
    for attr in ["weight", "scales", "biases"]:
        t: mx.array | None = getattr(layer, attr, None)
        if t is None:
            continue
        setattr(layer, attr, _my_shard(t, -1, rank, ratio))


def _shard_gated_delta_net(
    gdn: GatedDeltaNet, rank: int, ratio: float, group: mx.distributed.Group
) -> None:
    """Asymmetric shard for GatedDeltaNet (linear attention) layers."""
    kd = gdn.key_dim
    _shard_quantized_ats(gdn.in_proj_qkv, 0, rank, ratio, segments=[kd, 2 * kd])
    _shard_quantized_ats(gdn.in_proj_z, 0, rank, ratio)
    _shard_quantized_ats(gdn.in_proj_b, 0, rank, ratio)
    _shard_quantized_ats(gdn.in_proj_a, 0, rank, ratio)
    _shard_quantized_sta(gdn.out_proj, rank, ratio)

    # conv1d: segmented split along channel dim
    conv_w = gdn.conv1d.weight
    seg_parts = mx.split(conv_w, [kd, 2 * kd], axis=0)
    my_parts = [_my_shard(p, 0, rank, ratio) for p in seg_parts]
    gdn.conv1d.weight = mx.contiguous(mx.concatenate(my_parts, axis=0))

    gdn.dt_bias = _my_shard(gdn.dt_bias, 0, rank, ratio)
    gdn.A_log = _my_shard(gdn.A_log, 0, rank, ratio)

    r = ratio if rank == 0 else (1 - ratio)
    gdn.num_k_heads = int(gdn.num_k_heads * r)
    gdn.num_v_heads = int(gdn.num_v_heads * r)
    gdn.key_dim = int(gdn.key_dim * r)
    gdn.value_dim = int(gdn.value_dim * r)
    gdn.conv_dim = int(gdn.conv_dim * r)
    gdn.conv1d.groups = gdn.conv_dim
    gdn.sharding_group = group


# Patching must happen at the class level since nn.Module.__call__ ignores instance overrides
_attention_class_patched: set[type] = set()


def _patch_attention_class(attn_cls: type) -> None:
    """Patch an attention class to add all_sum when _asymmetric_tp_group is set."""
    if attn_cls in _attention_class_patched:
        return

    original_call = attn_cls.__call__

    def patched_call(
        self: nn.Module,
        x: mx.array,
        mask: mx.array | None = None,
        cache: object | None = None,
    ) -> mx.array:
        result = original_call(self, x, mask=mask, cache=cache)
        grp = getattr(self, "_asymmetric_tp_group", None)
        if grp is not None:
            result = mx.distributed.all_sum(result, group=grp)
        return result

    attn_cls.__call__ = patched_call
    _attention_class_patched.add(attn_cls)


def _shard_attention(
    attn: Attention, rank: int, ratio: float, group: mx.distributed.Group
) -> None:
    """Asymmetric shard for self-attention layers."""
    _patch_attention_class(type(attn))
    _shard_quantized_ats(attn.q_proj, 0, rank, ratio)
    _shard_quantized_sta(attn.o_proj, rank, ratio)
    # k_proj, v_proj: replicated (too few KV heads to split in most models)

    r = ratio if rank == 0 else (1 - ratio)
    attn.num_attention_heads = int(attn.num_attention_heads * r)
    attn._asymmetric_tp_group = group



def _shard_sparse_moe(
    moe: SparseMoeBlock | Qwen3_5SparseMoeBlock,
    rank: int,
    ratio: float,
    group: mx.distributed.Group,
) -> None:
    """Asymmetric shard for SparseMoeBlock (MoE layers)."""
    # switch_mlp: split expert intermediate dims (axis 1 for 3D expert weights)
    _shard_quantized_ats(moe.switch_mlp.gate_proj, 1, rank, ratio)
    _shard_quantized_ats(moe.switch_mlp.up_proj, 1, rank, ratio)
    _shard_quantized_sta(moe.switch_mlp.down_proj, rank, ratio)

    # shared_expert: standard MLP split
    _shard_quantized_ats(moe.shared_expert.gate_proj, 0, rank, ratio)
    _shard_quantized_ats(moe.shared_expert.up_proj, 0, rank, ratio)
    _shard_quantized_sta(moe.shared_expert.down_proj, rank, ratio)

    # gate + shared_expert_gate: replicated (tiny routing weights)
    moe.sharding_group = group


_mlp_class_patched: set[type] = set()


def _patch_mlp_class(mlp_cls: type) -> None:
    """Patch a dense MLP class to add all_sum when _asymmetric_tp_group is set."""
    if mlp_cls in _mlp_class_patched:
        return

    original_call = mlp_cls.__call__

    def patched_call(self: nn.Module, x: mx.array) -> mx.array:
        result = original_call(self, x)
        grp = getattr(self, "_asymmetric_tp_group", None)
        if grp is not None:
            result = mx.distributed.all_sum(result, group=grp)
        return result

    mlp_cls.__call__ = patched_call
    _mlp_class_patched.add(mlp_cls)


def _shard_dense_mlp(
    mlp: Qwen3NextMLP, rank: int, ratio: float, group: mx.distributed.Group
) -> None:
    """Asymmetric shard for dense (non-MoE) MLP layers."""
    _patch_mlp_class(type(mlp))
    _shard_quantized_ats(mlp.gate_proj, 0, rank, ratio)
    _shard_quantized_ats(mlp.up_proj, 0, rank, ratio)
    _shard_quantized_sta(mlp.down_proj, rank, ratio)
    mlp._asymmetric_tp_group = group


def asymmetric_tensor_auto_parallel(
    model: nn.Module,
    group: mx.distributed.Group,
    ratios: list[float],
) -> nn.Module:
    """
    Apply asymmetric tensor parallelism to a model.

    Args:
        model: The model to parallelize (must have .layers property)
        group: MLX distributed group
        ratios: Per-rank weight fractions, e.g. [0.75, 0.25] for 2 nodes.
                ratios[group.rank()] is this node's fraction.

    Returns:
        The model with asymmetric sharding applied.
    """
    rank = group.rank()
    ratio = ratios[0]  # ratio for rank 0; rank 1 gets 1-ratio

    # Get the inner model's layers
    inner = model
    for attr in ["language_model", "model"]:
        candidate = getattr(inner, attr, None)
        if candidate is not None and hasattr(candidate, "layers"):
            inner = candidate

    layers: list[Any] = inner.layers if hasattr(inner, "layers") else model.layers

    for layer in layers:
        if isinstance(layer, Qwen3_5DecoderLayer):
            # Qwen3.5 hybrid: linear_attn or self_attn per layer
            if layer.is_linear:
                _shard_gated_delta_net(layer.linear_attn, rank, ratio, group)
            else:
                _shard_attention(layer.self_attn, rank, ratio, group)

            mlp = layer.mlp
            if isinstance(mlp, (SparseMoeBlock, Qwen3_5SparseMoeBlock)):
                _shard_sparse_moe(mlp, rank, ratio, group)
            else:
                _shard_dense_mlp(mlp, rank, ratio, group)
        else:
            raise ValueError(
                f"Asymmetric TP does not yet support layer type {type(layer).__name__}. "
                f"Currently supported: Qwen3.5 (GatedDeltaNet + Attention + MoE). "
                f"Contributions for other architectures welcome."
            )

    logger.info(
        f"Asymmetric TP applied: rank {rank} gets "
        f"{ratios[rank] * 100:.0f}% of each weight tensor"
    )
    return model
