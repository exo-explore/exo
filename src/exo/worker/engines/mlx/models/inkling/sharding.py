from collections.abc import Generator
from typing import cast

import mlx.core as mx
import mlx.nn as nn

from exo.shared.types.worker.runner_response import ModelLoadingResponse
from exo.worker.engines.mlx.models.inkling.model import (
    MLP,
    Model,
    MoE,
    ShortConvolution,
)
from exo.worker.engines.mlx.sharding.blocks import ShardedMoE
from exo.worker.engines.mlx.sharding.tensor import TensorParallelShardingStrategy


class InklingShardingStrategy(TensorParallelShardingStrategy):
    """Splits the attention heads, the dense MLPs and every routed and shared expert along
    its intermediate dimension.

    - Queries, keys, values and the relative-position mixing weights (``r_proj``, one
      ``d_rel`` block per head) are split by head; the key and value short convolutions
      are per channel, so each rank keeps its heads' channels and conv state. The bias
      profiles (``rel_proj``) and the per-head norms are the same for every head.
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
            attention.q_proj = self.all_to_sharded_linear(attention.q_proj)
            attention.k_proj = self.all_to_sharded_linear(attention.k_proj)
            attention.v_proj = self.all_to_sharded_linear(attention.v_proj)
            attention.r_proj = self.all_to_sharded_linear(attention.r_proj)
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
                    self.all_to_sharded_linear_in_place(experts.gate_proj)
                    self.all_to_sharded_linear_in_place(experts.up_proj)
                    self.sharded_to_all_linear_in_place(experts.down_proj)
                sharded = ShardedMoE(moe)  # pyright: ignore[reportArgumentType]
                sharded.sharding_group = self.group
                layer.mlp = sharded  # pyright: ignore[reportAttributeAccessIssue]

            mx.eval(layer)
            mx.clear_cache()

            yield ModelLoadingResponse(layers_loaded=i, total=total)
        return model


def _keep_channels(conv: ShortConvolution, rank: int, ranks: int) -> None:
    """This rank's contiguous block of channels, copied so the full weight can be freed."""
    channels = conv.weight.shape[0] // ranks
    conv.weight = mx.contiguous(conv.weight[rank * channels : (rank + 1) * channels])
