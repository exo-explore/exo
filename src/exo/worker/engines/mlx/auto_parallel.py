from collections.abc import Generator
from functools import partial
from typing import TYPE_CHECKING, cast

import mlx.core as mx
import mlx.nn as nn
from mlx.nn.layers.distributed import (
    shard_inplace,
    shard_linear,
    sum_gradients,
)
from mlx_lm.models.deepseek_v3 import Model as DeepseekV3Model
from mlx_lm.models.deepseek_v4 import Model as DeepseekV4Model
from mlx_lm.models.deepseek_v32 import Model as DeepseekV32Model
from mlx_lm.models.gemma4 import Model as Gemma4Model
from mlx_lm.models.glm4_moe import Model as Glm4MoeModel
from mlx_lm.models.glm4_moe_lite import Model as GLM4MoeLiteModel
from mlx_lm.models.gpt_oss import Model as GptOssModel
from mlx_lm.models.kimi_k25 import Model as KimiK25Model
from mlx_lm.models.llama import Model as LlamaModel
from mlx_lm.models.minimax import Model as MiniMaxModel
from mlx_lm.models.ministral3 import Model as Ministral3Model
from mlx_lm.models.nemotron_h import Model as NemotronHModel
from mlx_lm.models.qwen3 import Model as Qwen3Model
from mlx_lm.models.qwen3_5 import Model as Qwen3_5TextModel
from mlx_lm.models.qwen3_5_moe import Model as Qwen3_5MoeModel
from mlx_lm.models.qwen3_moe import Model as Qwen3MoeModel
from mlx_lm.models.qwen3_next import Model as Qwen3NextModel
from mlx_lm.models.qwen3_vl import Model as Qwen3VLModel
from mlx_lm.models.step3p5 import Model as Step35Model
from mlx_lm.models.step3p5 import Step3p5MLP as Step35MLP
from mlx_lm.models.step3p5 import Step3p5Model as Step35InnerModel

from exo.shared.types.worker.runner_response import ModelLoadingResponse
from exo.shared.types.worker.shards import PipelineShardMetadata
from exo.worker.engines.mlx.models.deepseek_v3.sharding import DeepSeekShardingStrategy
from exo.worker.engines.mlx.models.deepseek_v4.sharding import (
    DeepseekV4ShardingStrategy,
)
from exo.worker.engines.mlx.models.glm.sharding import (
    GLM4MoeLiteShardingStrategy,
    Glm4MoeShardingStrategy,
)
from exo.worker.engines.mlx.models.gpt_oss.sharding import GptOssShardingStrategy
from exo.worker.engines.mlx.models.gpt_oss.sharding import (
    fix_pipeline_shard as fix_gpt_oss_pipeline_shard,
)
from exo.worker.engines.mlx.models.llama.sharding import LlamaShardingStrategy
from exo.worker.engines.mlx.models.minimax.sharding import MiniMaxShardingStrategy
from exo.worker.engines.mlx.models.nemotron_h.sharding import (
    NemotronHShardingStrategy,
)
from exo.worker.engines.mlx.models.nemotron_h.sharding import (
    fix_pipeline_shard as fix_nemotron_h_pipeline_shard,
)
from exo.worker.engines.mlx.models.qwen3.sharding import QwenShardingStrategy
from exo.worker.engines.mlx.models.qwen3.sharding import (
    fix_pipeline_shard as fix_qwen3_pipeline_shard,
)
from exo.worker.engines.mlx.sharding.blocks import (
    CustomMlxLayer,
    LayerCallable,
)
from exo.worker.engines.mlx.sharding.pipeline import (
    PipelineFirstLayer,
    PipelineLastLayer,
    get_inner_model,
    get_layers,
    patch_pipeline_model,
)
from exo.worker.engines.mlx.sharding.tensor import (
    TensorParallelShardingStrategy,
    patch_tensor_model,
)
from exo.worker.runner.bootstrap import logger

if TYPE_CHECKING:
    pass


def pipeline_auto_parallel(
    model: nn.Module,
    group: mx.distributed.Group,
    model_shard_meta: PipelineShardMetadata,
) -> Generator[ModelLoadingResponse, None, nn.Module]:
    """
    Automatically parallelize a model across multiple devices.
    Args:
    model: The model to parallelize (must have a 'layers' or 'h' property)
    model_shard_meta: The metadata for the model shard
    Returns:
    The parallelized model
    """
    inner_model_instance: nn.Module = get_inner_model(model)

    layers = get_layers(inner_model_instance)

    start_layer, end_layer = model_shard_meta.start_layer, model_shard_meta.end_layer
    device_rank, world_size = model_shard_meta.device_rank, model_shard_meta.world_size

    layers = layers[start_layer:end_layer]
    total = len(layers)
    for i, layer in enumerate(layers):
        mx.eval(layer)  # type: ignore
        mx.clear_cache()
        yield ModelLoadingResponse(layers_loaded=i, total=total)

    layers[0] = PipelineFirstLayer(layers[0], device_rank, group=group)
    layers[-1] = PipelineLastLayer(
        layers[-1],
        device_rank,
        world_size,
        group=group,
    )

    fix_gpt_oss_pipeline_shard(
        model, inner_model_instance, layers, start_layer, end_layer
    )

    if isinstance(inner_model_instance, Step35InnerModel):
        inner_model_instance.num_layers = len(layers)
        sliding_layers = [
            i for i, layer in enumerate(layers) if getattr(layer, "is_sliding", False)
        ]
        full_layers = [
            i
            for i, layer in enumerate(layers)
            if not getattr(layer, "is_sliding", True)
        ]
        inner_model_instance._swa_idx = 0 if not sliding_layers else sliding_layers[0]
        inner_model_instance._full_idx = 0 if not full_layers else full_layers[0]

    fix_qwen3_pipeline_shard(
        model, inner_model_instance, layers, start_layer, end_layer
    )

    fix_nemotron_h_pipeline_shard(
        model, inner_model_instance, layers, start_layer, end_layer
    )

    _set_layers(model, layers)

    assert isinstance(layers, list), (
        "Expected a list of layers after auto-parallel initialisation"
    )

    return patch_pipeline_model(model, group)


def tensor_auto_parallel(
    model: nn.Module,
    group: mx.distributed.Group,
) -> Generator[ModelLoadingResponse, None, nn.Module]:
    all_to_sharded_linear = partial(
        shard_linear,
        sharding="all-to-sharded",
        group=group,
    )
    sharded_to_all_linear = partial(
        shard_linear,
        sharding="sharded-to-all",
        group=group,
    )

    segments: int = 1

    def _all_to_sharded(path: str, weight: mx.array):
        if path.endswith("bias"):
            logger.info(f"Sharding bias for {path} - all to sharded")
            return weight.ndim - 1, segments
        return max(weight.ndim - 2, 0), segments

    all_to_sharded_linear_in_place = partial(
        shard_inplace,
        sharding=_all_to_sharded,  # type: ignore
        group=group,
    )

    n = group.size()

    def _sharded_to_all(path: str, weight: mx.array):
        if path.endswith("bias"):
            logger.info(f"Sharding bias for {path} - sharded to all")
            weight /= n
            return None
        return -1, segments

    sharded_to_all_linear_in_place = partial(
        shard_inplace,
        sharding=_sharded_to_all,  # type: ignore
        group=group,
    )

    if isinstance(model, (LlamaModel, Ministral3Model)):
        tensor_parallel_sharding_strategy = LlamaShardingStrategy(
            group,
            all_to_sharded_linear,
            sharded_to_all_linear,
            all_to_sharded_linear_in_place,
            sharded_to_all_linear_in_place,
        )
    elif isinstance(model, (DeepseekV3Model, DeepseekV32Model, KimiK25Model)):
        tensor_parallel_sharding_strategy = DeepSeekShardingStrategy(
            group,
            all_to_sharded_linear,
            sharded_to_all_linear,
            all_to_sharded_linear_in_place,
            sharded_to_all_linear_in_place,
        )
    elif isinstance(model, DeepseekV4Model):
        tensor_parallel_sharding_strategy = DeepseekV4ShardingStrategy(
            group,
            all_to_sharded_linear,
            sharded_to_all_linear,
            all_to_sharded_linear_in_place,
            sharded_to_all_linear_in_place,
        )
    elif isinstance(model, MiniMaxModel):
        tensor_parallel_sharding_strategy = MiniMaxShardingStrategy(
            group,
            all_to_sharded_linear,
            sharded_to_all_linear,
            all_to_sharded_linear_in_place,
            sharded_to_all_linear_in_place,
        )
    elif isinstance(model, GLM4MoeLiteModel):
        tensor_parallel_sharding_strategy = GLM4MoeLiteShardingStrategy(
            group,
            all_to_sharded_linear,
            sharded_to_all_linear,
            all_to_sharded_linear_in_place,
            sharded_to_all_linear_in_place,
        )
    elif isinstance(model, Glm4MoeModel):
        tensor_parallel_sharding_strategy = Glm4MoeShardingStrategy(
            group,
            all_to_sharded_linear,
            sharded_to_all_linear,
            all_to_sharded_linear_in_place,
            sharded_to_all_linear_in_place,
        )
    elif isinstance(
        model,
        (
            Qwen3Model,
            Qwen3MoeModel,
            Qwen3NextModel,
            Qwen3_5TextModel,
            Qwen3_5MoeModel,
            Qwen3VLModel,
        ),
    ):
        tensor_parallel_sharding_strategy = QwenShardingStrategy(
            group,
            all_to_sharded_linear,
            sharded_to_all_linear,
            all_to_sharded_linear_in_place,
            sharded_to_all_linear_in_place,
        )
    elif isinstance(model, GptOssModel):
        tensor_parallel_sharding_strategy = GptOssShardingStrategy(
            group,
            all_to_sharded_linear,
            sharded_to_all_linear,
            all_to_sharded_linear_in_place,
            sharded_to_all_linear_in_place,
        )
    elif isinstance(model, Step35Model):
        tensor_parallel_sharding_strategy = Step35ShardingStrategy(
            group,
            all_to_sharded_linear,
            sharded_to_all_linear,
            all_to_sharded_linear_in_place,
            sharded_to_all_linear_in_place,
        )
    elif isinstance(model, NemotronHModel):
        tensor_parallel_sharding_strategy = NemotronHShardingStrategy(
            group,
            all_to_sharded_linear,
            sharded_to_all_linear,
            all_to_sharded_linear_in_place,
            sharded_to_all_linear_in_place,
        )
    elif isinstance(model, Gemma4Model):
        tensor_parallel_sharding_strategy = Gemma4ShardingStrategy(
            group,
            all_to_sharded_linear,
            sharded_to_all_linear,
            all_to_sharded_linear_in_place,
            sharded_to_all_linear_in_place,
        )
    else:
        raise ValueError(f"Unsupported model type: {type(model)}")

    model = yield from tensor_parallel_sharding_strategy.shard_model(model)
    return patch_tensor_model(model)


def _set_layers(model: nn.Module, layers: list[LayerCallable]) -> None:
    inner_model_instance = get_inner_model(model)
    if hasattr(inner_model_instance, "layers"):
        inner_model_instance.layers = layers

        # Update DeepSeek V3 specific parameters when layers are shrunk
        if isinstance(
            model,
            (
                DeepseekV3Model,
                DeepseekV32Model,
                DeepseekV4Model,
                Glm4MoeModel,
                KimiK25Model,
            ),
        ) and hasattr(inner_model_instance, "num_layers"):
            logger.info(
                f"Setting num_layers to {len(layers)} for model {model.model.__class__.__name__}"
            )
            inner_model_instance.start_idx = 0
            inner_model_instance.end_idx = len(layers)
            inner_model_instance.num_layers = len(layers)
        elif isinstance(model, Qwen3MoeModel):
            logger.info(
                f"Setting num_hidden_layers to {len(layers)} for model {model.model.__class__.__name__}"
            )
            inner_model_instance.num_hidden_layers = len(layers)
    elif hasattr(inner_model_instance, "h"):
        inner_model_instance.h = layers
    else:
        raise ValueError("Model must have either a 'layers' or 'h' attribute")


class Step35ShardingStrategy(TensorParallelShardingStrategy):
    def shard_model(
        self,
        model: nn.Module,
    ) -> Generator[ModelLoadingResponse, None, nn.Module]:
        model = cast(Step35Model, model)
        total = len(model.layers)

        for i, layer in enumerate(model.layers):
            mx.eval(layer.parameters())
            layer.self_attn.q_proj = self.all_to_sharded_linear(layer.self_attn.q_proj)
            layer.self_attn.k_proj = self.all_to_sharded_linear(layer.self_attn.k_proj)
            layer.self_attn.v_proj = self.all_to_sharded_linear(layer.self_attn.v_proj)
            layer.self_attn.o_proj = self.sharded_to_all_linear(layer.self_attn.o_proj)

            layer.self_attn.num_heads //= self.N
            layer.self_attn.num_kv_heads //= self.N

            if getattr(layer.self_attn, "use_head_wise_attn_gate", False):
                layer.self_attn.g_proj = self.all_to_sharded_linear(
                    layer.self_attn.g_proj
                )

            if isinstance(layer.mlp, Step35MLP):
                layer.mlp.gate_proj = self.all_to_sharded_linear(layer.mlp.gate_proj)
                layer.mlp.up_proj = self.all_to_sharded_linear(layer.mlp.up_proj)
                layer.mlp.down_proj = self.sharded_to_all_linear(layer.mlp.down_proj)
            else:
                layer.mlp.sharding_group = self.group
                self.all_to_sharded_linear_in_place(layer.mlp.share_expert.gate_proj)
                self.all_to_sharded_linear_in_place(layer.mlp.share_expert.up_proj)
                self.sharded_to_all_linear_in_place(layer.mlp.share_expert.down_proj)
                self.all_to_sharded_linear_in_place(layer.mlp.switch_mlp.gate_proj)
                self.all_to_sharded_linear_in_place(layer.mlp.switch_mlp.up_proj)
                self.sharded_to_all_linear_in_place(layer.mlp.switch_mlp.down_proj)

            mx.eval(layer)
            mx.clear_cache()

            yield ModelLoadingResponse(layers_loaded=i, total=total)
        return model


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
