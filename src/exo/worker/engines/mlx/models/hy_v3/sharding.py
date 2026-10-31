import os
from collections.abc import Generator
from typing import cast

import mlx.core as mx
import mlx.nn as nn

from exo.shared.types.worker.runner_response import ModelLoadingResponse
from exo.worker.engines.mlx.models.hy_v3.model import MLP, Model, MoE
from exo.worker.engines.mlx.sharding.blocks import ShardedMoE
from exo.worker.engines.mlx.sharding.tensor import TensorParallelShardingStrategy


def replicate_attention() -> bool:
    """Whether tensor parallelism keeps attention (and the dense MLP) whole on every rank.

    Over MLX's TCP ring (Ethernet) every all-reduce costs about a millisecond, so the
    per-layer collectives set the speed: with attention whole, a layer has one all-reduce
    (the MoE's) instead of two, at the cost of every rank reading all attention weights
    and keeping the whole KV cache. Over RDMA (jaccl) an all-reduce costs tens of
    microseconds and attention stays split by heads."""
    return (
        "MLX_IBV_DEVICES" not in os.environ
        and "MLX_JACCL_COORDINATOR" not in os.environ
    )


class HyV3ShardingStrategy(TensorParallelShardingStrategy):
    """Attention split by heads (64 query and 8 key/value heads, so up to 8 ranks), or
    whole on every rank over the TCP ring (see ``replicate_attention``); the dense MLP split
    along its intermediate dimension (whole with attention over the ring); every routed
    expert and the shared expert split along their intermediate dimension; the router and
    the norms whole on every rank."""

    def shard_model(
        self,
        model: nn.Module,
    ) -> Generator[ModelLoadingResponse, None, nn.Module]:
        model = cast(Model, model)
        total = len(model.layers)
        whole = replicate_attention()
        for i, layer in enumerate(model.layers):
            mx.eval(layer.parameters())

            attention = layer.self_attn
            if not whole:
                attention.q_proj = self.all_to_sharded_linear(attention.q_proj)
                attention.k_proj = self.all_to_sharded_linear(attention.k_proj)
                attention.v_proj = self.all_to_sharded_linear(attention.v_proj)
                attention.o_proj = self.sharded_to_all_linear(attention.o_proj)
                attention.n_heads //= self.N
                attention.n_kv_heads //= self.N

            mlp = layer.mlp
            if isinstance(mlp, MoE):
                self.all_to_sharded_linear_in_place(mlp.switch_mlp.gate_proj)
                self.all_to_sharded_linear_in_place(mlp.switch_mlp.up_proj)
                self.sharded_to_all_linear_in_place(mlp.switch_mlp.down_proj)
                self.all_to_sharded_linear_in_place(mlp.shared_mlp.gate_proj)
                self.all_to_sharded_linear_in_place(mlp.shared_mlp.up_proj)
                self.sharded_to_all_linear_in_place(mlp.shared_mlp.down_proj)
                # The routed and shared partial sums are added before ShardedMoE's
                # all-reduce: one all-reduce per MoE layer.
                sharded = ShardedMoE(mlp)  # pyright: ignore[reportArgumentType]
                sharded.sharding_group = self.group
                layer.mlp = sharded  # pyright: ignore[reportAttributeAccessIssue]
            elif not whole:
                assert isinstance(mlp, MLP)
                mlp.gate_proj = self.all_to_sharded_linear(mlp.gate_proj)
                mlp.up_proj = self.all_to_sharded_linear(mlp.up_proj)
                mlp.down_proj = self.sharded_to_all_linear(mlp.down_proj)

            mx.eval(layer)
            mx.clear_cache()
            yield ModelLoadingResponse(layers_loaded=i, total=total)
        return model
