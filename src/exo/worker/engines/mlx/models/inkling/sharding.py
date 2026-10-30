from collections.abc import Generator
from typing import cast

import mlx.core as mx
import mlx.nn as nn

from exo.shared.types.worker.runner_response import ModelLoadingResponse
from exo.worker.engines.mlx.models.inkling.model import (
    MLP,
    Attention,
    Model,
    MoE,
    ShortConvolution,
)
from exo.worker.engines.mlx.sharding.blocks import ShardedMoE
from exo.worker.engines.mlx.sharding.tensor import TensorParallelShardingStrategy


class InklingShardingStrategy(TensorParallelShardingStrategy):
    """Splits the attention heads, the dense MLPs and every routed and shared expert along
    its intermediate dimension.

    - Queries, keys, values and the relative-position mixing weights (one ``d_rel``
      block per head), fused in ``qkvr_proj``, are split by head; the key and value
      short convolutions are per channel, so each rank keeps its heads' channels and
      conv state. The bias profiles (``rel_proj``) and the per-head norms are the same
      for every head.
    - The attention-output and MLP-output convolutions run on the full hidden state after
      the all-reduce, on every rank.
    - The router stays whole on every rank; the routed and shared experts' partial outputs
      are summed in one all-reduce.
    """

    def shard_model(
        self,
        model: nn.Module,
    ) -> Generator[ModelLoadingResponse, None, nn.Module]:
        model = cast(Model, model)
        rank = self.group.rank()
        total = len(model.layers)
        for i, layer in enumerate(model.layers):
            mx.eval(layer.parameters())

            attention = layer.self_attn
            _keep_heads(attention, rank, self.N)
            attention.o_proj = self.sharded_to_all_linear(attention.o_proj)
            attention.num_heads //= self.N
            attention.num_kv_heads //= self.N
            for conv in (attention.k_sconv, attention.v_sconv):
                _keep_channels(conv, rank, self.N)

            if isinstance(layer.mlp, MLP):
                layer.mlp.gate_proj = self.all_to_sharded_linear(layer.mlp.gate_proj)
                layer.mlp.up_proj = self.all_to_sharded_linear(layer.mlp.up_proj)
                layer.mlp.down_proj = self.sharded_to_all_linear(layer.mlp.down_proj)
            else:
                moe: MoE = layer.mlp
                for experts in (moe.experts, moe.shared_experts):
                    _keep_gate_up_rows(experts.gate_up_proj, rank, self.N)
                    self.sharded_to_all_linear_in_place(experts.down_proj)
                sharded = ShardedMoE(moe)  # pyright: ignore[reportArgumentType]
                sharded.sharding_group = self.group
                layer.mlp = sharded  # pyright: ignore[reportAttributeAccessIssue]

            mx.eval(layer)
            mx.clear_cache()

            yield ModelLoadingResponse(layers_loaded=i, total=total)
        return model


def _keep_heads(attention: Attention, rank: int, ranks: int) -> None:
    """This rank's block of heads from each of the fused input projections (queries,
    keys, values, relative-bias mixing weights): rows of the weight, and of the
    quantization scales and biases if any, which are per output row."""
    projection = attention.qkvr_proj
    for part in ("weight", "scales", "biases"):
        if part not in projection:
            continue
        rows = cast(mx.array, projection[part])
        pieces: list[mx.array] = []
        start = 0
        for width in attention.projection_widths():
            share = width // ranks
            pieces.append(rows[start + rank * share : start + (rank + 1) * share])
            start += width
        projection[part] = mx.contiguous(mx.concatenate(pieces, axis=0))


def _keep_gate_up_rows(projection: nn.Module, rank: int, ranks: int) -> None:
    """This rank's share of the intermediate dimension in a fused gate/up projection
    (output rows: gate, then up), for every expert: the same rows of the gate and of the
    up half, and of the quantization scales and biases if any."""
    for part in ("weight", "scales", "biases"):
        if part not in projection:
            continue
        rows = cast(mx.array, projection[part])
        width = rows.shape[-2] // 2
        share = width // ranks
        gate = rows[..., rank * share : (rank + 1) * share, :]
        up = rows[..., width + rank * share : width + (rank + 1) * share, :]
        projection[part] = mx.contiguous(mx.concatenate([gate, up], axis=-2))


def _keep_channels(conv: ShortConvolution, rank: int, ranks: int) -> None:
    """This rank's contiguous block of channels, copied so the full weight can be freed."""
    channels = conv.weight.shape[0] // ranks
    conv.weight = mx.contiguous(conv.weight[rank * channels : (rank + 1) * channels])
