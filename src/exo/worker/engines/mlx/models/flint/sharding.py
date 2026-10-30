from collections.abc import Generator
from typing import cast

import mlx.core as mx
import mlx.nn as nn
from mlx_lm.models.flint import MLP, FlintModel
from mlx_lm.models.flint import Model as FlintCausalModel

from exo.shared.types.worker.runner_response import ModelLoadingResponse
from exo.worker.engines.mlx.sharding.blocks import LayerCallable, ShardedMoE
from exo.worker.engines.mlx.sharding.tensor import TensorParallelShardingStrategy


class FlintShardingStrategy(TensorParallelShardingStrategy):
    """Splits the attention heads (with their sinks), the dense MLPs and every expert's
    intermediate dimension, like mlx-lm's own ``Model.shard``."""

    def shard_model(
        self,
        model: nn.Module,
    ) -> Generator[ModelLoadingResponse, None, nn.Module]:
        model = cast(FlintCausalModel, model)
        rank = self.group.rank()
        total = len(model.layers)
        for i, layer in enumerate(model.layers):
            mx.eval(layer.parameters())

            attention = layer.self_attn
            attention.q_proj = self.all_to_sharded_linear(attention.q_proj)
            attention.k_proj = self.all_to_sharded_linear(attention.k_proj)
            attention.v_proj = self.all_to_sharded_linear(attention.v_proj)
            attention.o_proj = self.sharded_to_all_linear(attention.o_proj)
            attention.n_heads //= self.N
            attention.n_kv_heads //= self.N
            # One learned sink per query head: keep this rank's heads' sinks.
            attention.softmax_offset = attention.softmax_offset[
                attention.n_heads * rank : attention.n_heads * (rank + 1)
            ]

            if isinstance(layer.mlp, MLP):
                layer.mlp.gate_proj = self.all_to_sharded_linear(layer.mlp.gate_proj)
                layer.mlp.down_proj = self.sharded_to_all_linear(layer.mlp.down_proj)
                layer.mlp.up_proj = self.all_to_sharded_linear(layer.mlp.up_proj)
            else:
                # The router stays whole on every rank; the routed experts and the shared
                # expert are split along their intermediate dimension and summed once.
                for experts in (layer.mlp.switch_mlp, layer.mlp.shared_expert):
                    self.all_to_sharded_linear_in_place(experts.gate_proj)
                    self.sharded_to_all_linear_in_place(experts.down_proj)
                    self.all_to_sharded_linear_in_place(experts.up_proj)
                layer.mlp = ShardedMoE(layer.mlp)  # pyright: ignore[reportAttributeAccessIssue, reportArgumentType]
                layer.mlp.sharding_group = self.group

            mx.eval(layer)
            mx.clear_cache()

            yield ModelLoadingResponse(layers_loaded=i, total=total)
        return model


def fix_pipeline_shard(
    model: nn.Module,
    inner_model_instance: nn.Module,
    layers: list[LayerCallable],
    start_layer: int,
    end_layer: int,
) -> None:
    """Flint builds its full-attention and sliding-window masks from the cache entries of
    the first layer of each kind, found in the whole model's layer list. Find them in the
    shard's layers instead (or none, when the shard has no layer of that kind)."""
    if isinstance(inner_model_instance, FlintModel):
        sliding = [cast(bool, getattr(layer, "use_sliding")) for layer in layers]  # noqa: B009
        inner_model_instance.swa_idx = sliding.index(True) if True in sliding else None
        inner_model_instance.fa_idx = sliding.index(False) if False in sliding else None
