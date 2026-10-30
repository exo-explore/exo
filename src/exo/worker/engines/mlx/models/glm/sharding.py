from collections.abc import Generator
from typing import cast

import mlx.core as mx
import mlx.nn as nn
from mlx_lm.models.glm4_moe import Model as Glm4MoeModel
from mlx_lm.models.glm4_moe import MoE
from mlx_lm.models.glm4_moe_lite import Glm4MoeLiteDecoderLayer, Glm4MoeLiteMLP
from mlx_lm.models.glm4_moe_lite import Model as GLM4MoeLiteModel

from exo.shared.types.worker.runner_response import ModelLoadingResponse
from exo.worker.engines.mlx.sharding.blocks import ShardedMoE
from exo.worker.engines.mlx.sharding.tensor import TensorParallelShardingStrategy


class GLM4MoeLiteShardingStrategy(TensorParallelShardingStrategy):
    def shard_model(
        self,
        model: nn.Module,
    ) -> Generator[ModelLoadingResponse, None, nn.Module]:
        model = cast(GLM4MoeLiteModel, model)
        total = len(model.layers)  # type: ignore
        for i, layer in enumerate(model.layers):  # type: ignore
            layer = cast(Glm4MoeLiteDecoderLayer, layer)
            mx.eval(layer.parameters())
            if layer.self_attn.q_lora_rank is None:  # type: ignore
                layer.self_attn.q_proj = self.all_to_sharded_linear(
                    layer.self_attn.q_proj
                )
            else:
                layer.self_attn.q_b_proj = self.all_to_sharded_linear(
                    layer.self_attn.q_b_proj
                )

            layer.self_attn.o_proj = self.sharded_to_all_linear(layer.self_attn.o_proj)
            layer.self_attn.num_heads //= self.N

            # Logic from upstream mlx
            num_heads = layer.self_attn.num_heads
            sh = self.group.rank() * num_heads
            eh = sh + num_heads

            def shard_heads(w: mx.array, sh: int = sh, eh: int = eh) -> mx.array:
                return w[sh:eh]

            layer.self_attn.embed_q.apply(shard_heads)
            layer.self_attn.unembed_out.apply(shard_heads)

            if isinstance(layer.mlp, Glm4MoeLiteMLP):
                layer.mlp.gate_proj = self.all_to_sharded_linear(layer.mlp.gate_proj)
                layer.mlp.down_proj = self.sharded_to_all_linear(layer.mlp.down_proj)
                layer.mlp.up_proj = self.all_to_sharded_linear(layer.mlp.up_proj)

            else:
                if getattr(layer.mlp, "shared_experts", None) is not None:
                    self.all_to_sharded_linear_in_place(
                        layer.mlp.shared_experts.gate_proj
                    )
                    self.sharded_to_all_linear_in_place(
                        layer.mlp.shared_experts.down_proj
                    )
                    self.all_to_sharded_linear_in_place(
                        layer.mlp.shared_experts.up_proj
                    )
                self.all_to_sharded_linear_in_place(layer.mlp.switch_mlp.gate_proj)
                self.sharded_to_all_linear_in_place(layer.mlp.switch_mlp.down_proj)
                self.all_to_sharded_linear_in_place(layer.mlp.switch_mlp.up_proj)
                layer.mlp = ShardedMoE(layer.mlp)  # type: ignore
                layer.mlp.sharding_group = self.group  # type: ignore
            mx.eval(layer)
            mx.clear_cache()

            yield ModelLoadingResponse(layers_loaded=i, total=total)

        return model


class Glm4MoeShardingStrategy(TensorParallelShardingStrategy):
    def shard_model(
        self,
        model: nn.Module,
    ) -> Generator[ModelLoadingResponse, None, nn.Module]:
        model = cast(Glm4MoeModel, model)
        total = len(model.layers)
        for i, layer in enumerate(model.layers):
            mx.eval(layer.parameters())

            layer.self_attn.q_proj = self.all_to_sharded_linear(layer.self_attn.q_proj)
            layer.self_attn.k_proj = self.all_to_sharded_linear(layer.self_attn.k_proj)
            layer.self_attn.v_proj = self.all_to_sharded_linear(layer.self_attn.v_proj)
            layer.self_attn.o_proj = self.sharded_to_all_linear(layer.self_attn.o_proj)
            layer.self_attn.n_heads //= self.N
            layer.self_attn.n_kv_heads //= self.N

            if isinstance(layer.mlp, MoE):
                self.all_to_sharded_linear_in_place(layer.mlp.switch_mlp.gate_proj)
                self.sharded_to_all_linear_in_place(layer.mlp.switch_mlp.down_proj)
                self.all_to_sharded_linear_in_place(layer.mlp.switch_mlp.up_proj)
                if getattr(layer.mlp, "shared_experts", None) is not None:
                    self.all_to_sharded_linear_in_place(
                        layer.mlp.shared_experts.gate_proj
                    )
                    self.sharded_to_all_linear_in_place(
                        layer.mlp.shared_experts.down_proj
                    )
                    self.all_to_sharded_linear_in_place(
                        layer.mlp.shared_experts.up_proj
                    )
                layer.mlp = ShardedMoE(layer.mlp)  # pyright: ignore[reportAttributeAccessIssue, reportArgumentType]
                layer.mlp.sharding_group = self.group

            else:
                layer.mlp.gate_proj = self.all_to_sharded_linear(layer.mlp.gate_proj)
                layer.mlp.down_proj = self.sharded_to_all_linear(layer.mlp.down_proj)
                layer.mlp.up_proj = self.all_to_sharded_linear(layer.mlp.up_proj)

            mx.eval(layer)
            mx.clear_cache()

            yield ModelLoadingResponse(layers_loaded=i, total=total)
        return model
