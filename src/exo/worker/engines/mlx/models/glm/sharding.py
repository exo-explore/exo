from collections.abc import Generator
from typing import cast

import mlx.core as mx
import mlx.nn as nn
from mlx_lm.models.glm4_moe import Model as Glm4MoeModel
from mlx_lm.models.glm4_moe import MoE
from mlx_lm.models.glm4_moe_lite import Glm4MoeLiteDecoderLayer, Glm4MoeLiteMLP
from mlx_lm.models.glm4_moe_lite import Model as GLM4MoeLiteModel

from exo.shared.types.worker.runner_response import ModelLoadingResponse
from exo.worker.engines.mlx.models.glm.glm5_next import (
    Glm5NextAttention,
    KimiDeltaAttention,
)
from exo.worker.engines.mlx.models.glm.glm5_next import Model as Glm5NextModel
from exo.worker.engines.mlx.models.glm.glm5_next import MoE as Glm5NextMoE
from exo.worker.engines.mlx.sharding.blocks import (
    CustomMlxLayer,
    LayerCallable,
    ShardedMoE,
)
from exo.worker.engines.mlx.sharding.pipeline import (
    PipelineFirstLayer,
    PipelineLastLayer,
)
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


def forward_shared_topk(
    model: nn.Module,
    inner_model_instance: nn.Module,
    layers: list[LayerCallable],
    start_layer: int,
    end_layer: int,
) -> None:
    """GLM-5.2+ checkpoints list ``indexer_types``: "shared" layers have no indexer and
    reuse the top-k selection of the previous "full" layer. Have the pipeline wrappers
    pass that selection from one rank to the next, so a rank that starts on a shared
    layer attends over the same keys as a single device would."""
    args: object = getattr(model, "args", None)
    indexer_types = cast(list[str] | None, getattr(args, "indexer_types", None))
    if not indexer_types or "shared" not in indexer_types:
        return
    topk_size = cast(int, getattr(args, "index_topk", 0))
    layer: object = layers[-1]
    while isinstance(layer, CustomMlxLayer):
        if isinstance(layer, (PipelineFirstLayer, PipelineLastLayer)):
            layer.shared_topk_size = topk_size
        layer = layer.original_layer
    first = layers[0]
    if isinstance(first, PipelineFirstLayer):
        first.shared_topk_size = topk_size


class Glm5NextShardingStrategy(TensorParallelShardingStrategy):
    """GLM-5.3-Flash: heads split across ranks in both attention kinds, experts and MLPs
    split on their intermediate dimension. The hyper-connections, the routers and the DSA
    indexers (and their caches) are replicated: every rank computes the same top-k."""

    def shard_model(
        self,
        model: nn.Module,
    ) -> Generator[ModelLoadingResponse, None, nn.Module]:
        model = cast(Glm5NextModel, model)
        rank = self.group.rank()
        total = len(model.layers)
        for i, layer in enumerate(model.layers):
            mx.eval(layer.parameters())
            attention = layer.self_attn
            if isinstance(attention, KimiDeltaAttention):
                heads = attention.num_heads // self.N
                start, end = rank * heads, (rank + 1) * heads
                width = attention.head_dim
                attention.q_proj = self.all_to_sharded_linear(attention.q_proj)
                attention.k_proj = self.all_to_sharded_linear(attention.k_proj)
                attention.v_proj = self.all_to_sharded_linear(attention.v_proj)
                for conv in (attention.q_conv, attention.k_conv, attention.v_conv):
                    conv.weight = conv.weight[start * width : end * width]
                attention.f_b_proj = self.all_to_sharded_linear(attention.f_b_proj)
                attention.g_b_proj = self.all_to_sharded_linear(attention.g_b_proj)
                attention.b_proj = self.all_to_sharded_linear(attention.b_proj)
                attention.dt_bias = attention.dt_bias[start * width : end * width]
                attention.A_log = attention.A_log[start:end]
                attention.o_proj = self.sharded_to_all_linear(attention.o_proj)
                attention.num_heads = heads
            else:
                assert isinstance(attention, Glm5NextAttention)
                heads = attention.num_heads // self.N
                start, end = rank * heads, (rank + 1) * heads

                def shard_heads(
                    w: mx.array, start: int = start, end: int = end
                ) -> mx.array:
                    return w[start:end]

                attention.q_b_proj = self.all_to_sharded_linear(attention.q_b_proj)
                attention.embed_q.apply(shard_heads)
                attention.unembed_out.apply(shard_heads)
                attention.o_proj = self.sharded_to_all_linear(attention.o_proj)
                attention.num_heads = heads

            mlp = layer.mlp
            if isinstance(mlp, Glm5NextMoE):
                self.all_to_sharded_linear_in_place(mlp.switch_mlp.gate_proj)
                self.all_to_sharded_linear_in_place(mlp.switch_mlp.up_proj)
                self.sharded_to_all_linear_in_place(mlp.switch_mlp.down_proj)
                self.all_to_sharded_linear_in_place(mlp.shared_experts.gate_proj)
                self.all_to_sharded_linear_in_place(mlp.shared_experts.up_proj)
                self.sharded_to_all_linear_in_place(mlp.shared_experts.down_proj)
                sharded = ShardedMoE(cast(LayerCallable, cast(object, mlp)))
                sharded.sharding_group = self.group
                layer.mlp = cast(Glm5NextMoE, cast(object, sharded))
            else:
                mlp.gate_proj = self.all_to_sharded_linear(mlp.gate_proj)
                mlp.up_proj = self.all_to_sharded_linear(mlp.up_proj)
                mlp.down_proj = self.sharded_to_all_linear(mlp.down_proj)

            mx.eval(layer)
            mx.clear_cache()
            yield ModelLoadingResponse(layers_loaded=i, total=total)
        return model
