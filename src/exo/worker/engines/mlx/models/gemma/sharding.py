from collections.abc import Generator
from typing import cast

import mlx.core as mx
import mlx.nn as nn
from mlx.nn.layers.distributed import sum_gradients
from mlx_lm.models.gemma4 import Model as Gemma4Model

from exo.shared.types.worker.runner_response import ModelLoadingResponse
from exo.worker.engines.mlx.sharding.blocks import CustomMlxLayer, LayerCallable
from exo.worker.engines.mlx.sharding.tensor import TensorParallelShardingStrategy


class WrappedGemma4Experts(CustomMlxLayer):
    def __init__(self, layer: LayerCallable):
        super().__init__(layer)
        self.sharding_group: mx.distributed.Group | None = None

    def __call__(
        self, x: mx.array, top_k_indices: mx.array, top_k_weights: mx.array
    ) -> mx.array:
        if self.sharding_group is not None:
            x = sum_gradients(self.sharding_group)(x)
        y: mx.array = self.original_layer(x, top_k_indices, top_k_weights)
        if self.sharding_group is not None:
            y = mx.distributed.all_sum(y, group=self.sharding_group)
        return y


class Gemma4ShardingStrategy(TensorParallelShardingStrategy):
    def shard_model(
        self,
        model: nn.Module,
    ) -> Generator[ModelLoadingResponse, None, nn.Module]:
        model = cast(Gemma4Model, model)
        layers = model.language_model.model.layers
        total = len(layers)
        for i, layer in enumerate(layers):
            mx.eval(layer.parameters())

            attn = layer.self_attn
            attn.q_proj = self.all_to_sharded_linear(attn.q_proj)
            has_kv: bool = cast(bool, attn.has_kv)
            if has_kv:
                attn.k_proj = self.all_to_sharded_linear(attn.k_proj)
                if not attn.use_k_eq_v:
                    attn.v_proj = self.all_to_sharded_linear(attn.v_proj)
            attn.o_proj = self.sharded_to_all_linear(attn.o_proj)
            attn.n_heads //= self.N
            attn.n_kv_heads //= self.N

            layer.mlp.gate_proj = self.all_to_sharded_linear(layer.mlp.gate_proj)
            layer.mlp.down_proj = self.sharded_to_all_linear(layer.mlp.down_proj)
            layer.mlp.up_proj = self.all_to_sharded_linear(layer.mlp.up_proj)

            if layer.enable_moe:
                self.all_to_sharded_linear_in_place(layer.experts.switch_glu.gate_proj)
                self.sharded_to_all_linear_in_place(layer.experts.switch_glu.down_proj)
                self.all_to_sharded_linear_in_place(layer.experts.switch_glu.up_proj)
                layer.experts = WrappedGemma4Experts(layer.experts)  # pyright: ignore[reportAttributeAccessIssue,reportArgumentType]
                layer.experts.sharding_group = self.group

            mx.eval(layer)
            mx.clear_cache()
            yield ModelLoadingResponse(layers_loaded=i, total=total)
        return model
