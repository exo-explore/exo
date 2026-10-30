from collections.abc import Generator
from typing import cast

import mlx.core as mx
import mlx.nn as nn
from mlx_lm.models.llama import Model as LlamaModel

from exo.shared.types.worker.runner_response import ModelLoadingResponse
from exo.worker.engines.mlx.sharding.tensor import TensorParallelShardingStrategy


class LlamaShardingStrategy(TensorParallelShardingStrategy):
    def shard_model(
        self,
        model: nn.Module,
    ) -> Generator[ModelLoadingResponse, None, nn.Module]:
        model = cast(LlamaModel, model)
        total = len(model.layers)
        for i, layer in enumerate(model.layers):
            # Force load weights before sharding to avoid FAST_SYNCH deadlock
            mx.eval(layer.parameters())
            layer.self_attn.q_proj = self.all_to_sharded_linear(layer.self_attn.q_proj)
            layer.self_attn.k_proj = self.all_to_sharded_linear(layer.self_attn.k_proj)
            layer.self_attn.v_proj = self.all_to_sharded_linear(layer.self_attn.v_proj)
            layer.self_attn.o_proj = self.sharded_to_all_linear(layer.self_attn.o_proj)
            layer.self_attn.n_heads //= self.N
            if layer.self_attn.n_kv_heads is not None:
                layer.self_attn.n_kv_heads //= self.N

            layer.mlp.gate_proj = self.all_to_sharded_linear(layer.mlp.gate_proj)
            layer.mlp.down_proj = self.sharded_to_all_linear(layer.mlp.down_proj)
            layer.mlp.up_proj = self.all_to_sharded_linear(layer.mlp.up_proj)
            mx.eval(layer)

            yield ModelLoadingResponse(layers_loaded=i, total=total)
        return model
