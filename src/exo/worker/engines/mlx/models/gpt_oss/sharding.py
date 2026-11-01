from collections.abc import Generator
from typing import cast

import mlx.core as mx
import mlx.nn as nn
from mlx_lm.models.gpt_oss import GptOssMoeModel

from exo.shared.types.worker.runner_response import ModelLoadingResponse
from exo.worker.engines.mlx.sharding.blocks import LayerCallable, ShardedMoE
from exo.worker.engines.mlx.sharding.tensor import TensorParallelShardingStrategy


class GptOssShardingStrategy(TensorParallelShardingStrategy):
    def shard_model(
        self,
        model: nn.Module,
    ) -> Generator[ModelLoadingResponse, None, nn.Module]:
        model = cast(GptOssMoeModel, model)
        total = len(model.layers)

        for i, layer in enumerate(model.layers):
            mx.eval(layer.parameters())
            layer.self_attn.q_proj = self.all_to_sharded_linear(layer.self_attn.q_proj)
            layer.self_attn.k_proj = self.all_to_sharded_linear(layer.self_attn.k_proj)
            layer.self_attn.v_proj = self.all_to_sharded_linear(layer.self_attn.v_proj)
            layer.self_attn.o_proj = self.sharded_to_all_linear(layer.self_attn.o_proj)

            layer.self_attn.num_attention_heads //= self.N
            layer.self_attn.num_key_value_heads //= self.N
            layer.self_attn.num_key_value_groups = (
                layer.self_attn.num_attention_heads
                // layer.self_attn.num_key_value_heads
            )

            layer.self_attn.sinks = layer.self_attn.sinks[
                layer.self_attn.num_attention_heads
                * self.group.rank() : layer.self_attn.num_attention_heads
                * (self.group.rank() + 1)
            ]

            self.all_to_sharded_linear_in_place(layer.mlp.experts.gate_proj)
            self.sharded_to_all_linear_in_place(layer.mlp.experts.down_proj)
            self.all_to_sharded_linear_in_place(layer.mlp.experts.up_proj)

            layer.mlp = ShardedMoE(layer.mlp)  # type: ignore
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
    if isinstance(inner_model_instance, GptOssMoeModel):
        inner_model_instance.layer_types = inner_model_instance.layer_types[
            start_layer:end_layer
        ]
        # We can assume the model has at least one layer thanks to placement.
        # If a layer type doesn't exist, we can set it to 0.
        inner_model_instance.swa_idx = (
            0
            if "sliding_attention" not in inner_model_instance.layer_types
            else inner_model_instance.layer_types.index("sliding_attention")
        )
        inner_model_instance.ga_idx = (
            0
            if "full_attention" not in inner_model_instance.layer_types
            else inner_model_instance.layer_types.index("full_attention")
        )
