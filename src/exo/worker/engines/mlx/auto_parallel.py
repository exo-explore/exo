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
from mlx_lm.models.base import (
    scaled_dot_product_attention,
)
from mlx_lm.models.deepseek_v3 import Model as DeepseekV3Model
from mlx_lm.models.deepseek_v4 import Model as DeepseekV4Model
from mlx_lm.models.deepseek_v32 import Model as DeepseekV32Model
from mlx_lm.models.gemma4 import Model as Gemma4Model
from mlx_lm.models.glm4_moe import Model as Glm4MoeModel
from mlx_lm.models.glm4_moe_lite import Model as GLM4MoeLiteModel
from mlx_lm.models.gpt_oss import GptOssMoeModel
from mlx_lm.models.gpt_oss import Model as GptOssModel
from mlx_lm.models.kimi_k25 import Model as KimiK25Model
from mlx_lm.models.llama import Model as LlamaModel
from mlx_lm.models.minimax import MiniMaxAttention
from mlx_lm.models.minimax import Model as MiniMaxModel
from mlx_lm.models.ministral3 import Model as Ministral3Model
from mlx_lm.models.nemotron_h import Model as NemotronHModel
from mlx_lm.models.nemotron_h import (
    NemotronHAttention,
    NemotronHMamba2Mixer,
    NemotronHMoE,
)
from mlx_lm.models.nemotron_h import NemotronHModel as NemotronHInnerModel
from mlx_lm.models.qwen3 import Model as Qwen3Model
from mlx_lm.models.qwen3 import TransformerBlock as Qwen3TransformerBlock
from mlx_lm.models.qwen3_5 import DecoderLayer as Qwen3_5DecoderLayer
from mlx_lm.models.qwen3_5 import Model as Qwen3_5TextModel
from mlx_lm.models.qwen3_5 import Qwen3_5TextModel as Qwen3_5TextModelInner
from mlx_lm.models.qwen3_5 import SparseMoeBlock as Qwen3_5SparseMoeBlock
from mlx_lm.models.qwen3_5_moe import Model as Qwen3_5MoeModel
from mlx_lm.models.qwen3_moe import Model as Qwen3MoeModel
from mlx_lm.models.qwen3_moe import Qwen3MoeDecoderLayer, Qwen3MoeSparseMoeBlock
from mlx_lm.models.qwen3_next import Model as Qwen3NextModel
from mlx_lm.models.qwen3_next import (
    Qwen3NextDecoderLayer,
    Qwen3NextGatedDeltaNet,
    Qwen3NextSparseMoeBlock,
)
from mlx_lm.models.qwen3_next import Qwen3NextModel as Qwen3NextInnerModel
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
from exo.worker.engines.mlx.models.llama.sharding import LlamaShardingStrategy
from exo.worker.engines.mlx.sharding.blocks import (
    CustomMlxLayer,
    LayerCallable,
    ShardedMoE,
)
from exo.worker.engines.mlx.sharding.pipeline import (
    PipelineFirstLayer,
    PipelineLastLayer,
    get_inner_model,
    get_layers,
    patch_hybrid_cache,
    patch_pipeline_model,
)
from exo.worker.engines.mlx.sharding.tensor import (
    TensorParallelShardingStrategy,
    patch_tensor_model,
)
from exo.worker.runner.bootstrap import logger

if TYPE_CHECKING:
    from mlx_lm.models.cache import Cache


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

    if isinstance(inner_model_instance, (Qwen3_5TextModelInner, Qwen3NextInnerModel)):
        full_attn_layers = [
            i for i, layer in enumerate(layers) if not getattr(layer, "is_linear", True)
        ]
        linear_layers = [
            i for i, layer in enumerate(layers) if getattr(layer, "is_linear", False)
        ]
        inner_model_instance.fa_idx = full_attn_layers[0] if full_attn_layers else 0
        inner_model_instance.ssm_idx = linear_layers[0] if linear_layers else 0
        if not full_attn_layers or not linear_layers:
            patch_hybrid_cache(
                cast(Qwen3_5TextModel | Qwen3NextModel, model),
                fa_idx=inner_model_instance.fa_idx,
                has_full_attn=bool(full_attn_layers),
                ssm_idx=inner_model_instance.ssm_idx,
                has_linear=bool(linear_layers),
            )

    if isinstance(inner_model_instance, NemotronHInnerModel):
        # NemotronH uses block_type: "M" (Mamba/SSM), "*" (Attention), "E" (MoE), "-" (MLP)
        # Only "M" and "*" blocks have cache entries.
        # Recompute fa_idx and ssm_idx as cache-array indices for the shard's layers.
        cache_idx = 0
        fa_idx: int | None = None
        ssm_idx: int | None = None
        for layer in layers:
            block_type = getattr(layer, "block_type", None)
            if block_type == "*":
                if fa_idx is None:
                    fa_idx = cache_idx
                cache_idx += 1
            elif block_type == "M":
                if ssm_idx is None:
                    ssm_idx = cache_idx
                cache_idx += 1
        has_attn = fa_idx is not None
        has_mamba = ssm_idx is not None
        inner_model_instance.fa_idx = fa_idx if fa_idx is not None else 0
        inner_model_instance.ssm_idx = ssm_idx if ssm_idx is not None else 0
        if not has_attn or not has_mamba:
            patch_hybrid_cache(
                cast(NemotronHModel, model),
                fa_idx=inner_model_instance.fa_idx,
                has_full_attn=has_attn,
                ssm_idx=inner_model_instance.ssm_idx,
                has_linear=has_mamba,
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


class WrappedMiniMaxAttention(CustomMlxLayer):
    def __init__(self, layer: LayerCallable, group: mx.distributed.Group):
        super().__init__(layer)
        self.group = group

    def __call__(
        self,
        x: mx.array,
        mask: mx.array | None = None,
        cache: "Cache | None" = None,
    ) -> mx.array:
        batch_dim, seq_dim, _ = x.shape

        self._original_layer = cast(MiniMaxAttention, self.original_layer)  # type: ignore

        queries: mx.array = self._original_layer.q_proj(x)
        keys: mx.array = self._original_layer.k_proj(x)
        values: mx.array = self._original_layer.v_proj(x)

        if getattr(self, "use_qk_norm", False):
            q_dim = queries.shape[-1]
            k_dim = keys.shape[-1]
            n = self.group.size()

            qk = mx.concatenate(
                [queries, keys], axis=-1
            )  # (batch_dim, seq_dim, q_dim + k_dim)
            qk = mx.distributed.all_gather(
                qk, group=self.group
            )  # (n*batch_dim, seq_dim, q_dim + k_dim)

            qk = qk.reshape(n, batch_dim, seq_dim, q_dim + k_dim).transpose(1, 2, 0, 3)
            queries = qk[..., :q_dim].reshape(
                batch_dim, seq_dim, -1
            )  # (batch_dim, seq_dim, n * q_dim)
            keys = qk[..., q_dim:].reshape(
                batch_dim, seq_dim, -1
            )  # (batch_dim, seq_dim, n * k_dim)

            queries = self._original_layer.q_norm(queries)
            keys = self._original_layer.k_norm(keys)

            # Split back and take this rank's portion
            queries = mx.split(queries, n, axis=-1)[self.group.rank()]
            keys = mx.split(keys, n, axis=-1)[self.group.rank()]

        queries = queries.reshape(
            batch_dim, seq_dim, self._original_layer.num_attention_heads, -1
        ).transpose(0, 2, 1, 3)
        keys = keys.reshape(
            batch_dim, seq_dim, self._original_layer.num_key_value_heads, -1
        ).transpose(0, 2, 1, 3)
        values = values.reshape(
            batch_dim, seq_dim, self._original_layer.num_key_value_heads, -1
        ).transpose(0, 2, 1, 3)

        if cache is not None:
            queries = self._original_layer.rope(queries, offset=cache.offset)
            keys = self._original_layer.rope(keys, offset=cache.offset)
            keys, values = cache.update_and_fetch(keys, values)
        else:
            queries = self._original_layer.rope(queries)
            keys = self._original_layer.rope(keys)

        output = scaled_dot_product_attention(
            queries,
            keys,
            values,
            cache=cache,
            scale=self._original_layer.scale,
            mask=mask,
        )

        output = output.transpose(0, 2, 1, 3).reshape(batch_dim, seq_dim, -1)

        return self._original_layer.o_proj(output)


class MiniMaxShardingStrategy(TensorParallelShardingStrategy):
    def shard_model(
        self,
        model: nn.Module,
    ) -> Generator[ModelLoadingResponse, None, nn.Module]:
        model = cast(MiniMaxModel, model)
        total = len(model.layers)
        for i, layer in enumerate(model.layers):
            mx.eval(layer.parameters())
            # Shard the self attention
            layer.self_attn.q_proj = self.all_to_sharded_linear(layer.self_attn.q_proj)
            layer.self_attn.k_proj = self.all_to_sharded_linear(layer.self_attn.k_proj)
            layer.self_attn.v_proj = self.all_to_sharded_linear(layer.self_attn.v_proj)
            layer.self_attn.o_proj = self.sharded_to_all_linear(layer.self_attn.o_proj)

            layer.self_attn.num_attention_heads //= self.N
            layer.self_attn.num_key_value_heads //= self.N

            layer.self_attn = WrappedMiniMaxAttention(layer.self_attn, self.group)  # pyright: ignore[reportAttributeAccessIssue,reportArgumentType]

            # Shard the MoE.
            self.all_to_sharded_linear_in_place(
                layer.block_sparse_moe.switch_mlp.gate_proj
            )
            self.sharded_to_all_linear_in_place(
                layer.block_sparse_moe.switch_mlp.down_proj
            )
            self.all_to_sharded_linear_in_place(
                layer.block_sparse_moe.switch_mlp.up_proj
            )
            layer.block_sparse_moe = ShardedMoE(layer.block_sparse_moe)  # type: ignore
            layer.block_sparse_moe.sharding_group = self.group
            mx.eval(layer)
            mx.clear_cache()

            yield ModelLoadingResponse(layers_loaded=i, total=total)
        return model


class QwenShardingStrategy(TensorParallelShardingStrategy):
    def shard_model(
        self,
        model: nn.Module,
    ) -> Generator[ModelLoadingResponse, None, nn.Module]:
        model = cast(
            Qwen3Model
            | Qwen3MoeModel
            | Qwen3NextModel
            | Qwen3_5TextModel
            | Qwen3_5MoeModel
            | Qwen3VLModel,
            model,
        )
        total = len(model.layers)
        for i, layer in enumerate(model.layers):
            mx.eval(layer.parameters())
            # Shard the self attention
            if isinstance(layer, (Qwen3MoeDecoderLayer, Qwen3TransformerBlock)):
                layer.self_attn.q_proj = self.all_to_sharded_linear(
                    layer.self_attn.q_proj
                )
                layer.self_attn.k_proj = self.all_to_sharded_linear(
                    layer.self_attn.k_proj
                )
                layer.self_attn.v_proj = self.all_to_sharded_linear(
                    layer.self_attn.v_proj
                )
                layer.self_attn.o_proj = self.sharded_to_all_linear(
                    layer.self_attn.o_proj
                )
                layer.self_attn.n_heads //= self.N
                layer.self_attn.n_kv_heads //= self.N
            else:
                assert isinstance(layer, (Qwen3NextDecoderLayer, Qwen3_5DecoderLayer))
                if hasattr(layer, "linear_attn"):
                    linear_attn = layer.linear_attn

                    if isinstance(linear_attn, Qwen3NextGatedDeltaNet):
                        # Qwen3-Next: combined projections
                        linear_attn.in_proj_qkvz = self.all_to_sharded_linear(
                            linear_attn.in_proj_qkvz
                        )
                        linear_attn.in_proj_ba = self.all_to_sharded_linear(
                            linear_attn.in_proj_ba
                        )
                    else:
                        # Qwen3.5: separate projections
                        # in_proj_qkv has sections [q(key_dim), k(key_dim), v(value_dim)]
                        # that must be split section-aware, not as a contiguous block
                        key_dim = linear_attn.key_dim
                        value_dim = linear_attn.value_dim
                        linear_attn.in_proj_qkv = shard_linear(
                            linear_attn.in_proj_qkv,
                            "all-to-sharded",
                            segments=[key_dim, key_dim + key_dim],
                            group=self.group,
                        )
                        linear_attn.in_proj_z = self.all_to_sharded_linear(
                            linear_attn.in_proj_z
                        )
                        linear_attn.in_proj_b = self.all_to_sharded_linear(
                            linear_attn.in_proj_b
                        )
                        linear_attn.in_proj_a = self.all_to_sharded_linear(
                            linear_attn.in_proj_a
                        )
                    linear_attn.out_proj = self.sharded_to_all_linear(
                        linear_attn.out_proj
                    )

                    # Shard conv1d: depthwise conv with non-contiguous channel slicing.
                    # Channel layout is [q(key_dim), k(key_dim), v(value_dim)].
                    # Each rank takes its head-slice from each of the three sections.
                    rank = self.group.rank()
                    key_dim = linear_attn.key_dim
                    value_dim = linear_attn.value_dim
                    key_dim_shard = key_dim // self.N
                    value_dim_shard = value_dim // self.N

                    q_idx = mx.arange(rank * key_dim_shard, (rank + 1) * key_dim_shard)
                    k_idx = mx.arange(
                        key_dim + rank * key_dim_shard,
                        key_dim + (rank + 1) * key_dim_shard,
                    )
                    v_idx = mx.arange(
                        2 * key_dim + rank * value_dim_shard,
                        2 * key_dim + (rank + 1) * value_dim_shard,
                    )
                    conv_indices = mx.concatenate([q_idx, k_idx, v_idx])
                    linear_attn.conv1d.weight = linear_attn.conv1d.weight[conv_indices]
                    new_conv_dim = key_dim_shard * 2 + value_dim_shard
                    linear_attn.conv1d.groups = new_conv_dim

                    num_v_shard = linear_attn.num_v_heads // self.N
                    v_start = rank * num_v_shard
                    v_end = v_start + num_v_shard
                    linear_attn.A_log = linear_attn.A_log[v_start:v_end]
                    linear_attn.dt_bias = linear_attn.dt_bias[v_start:v_end]

                    linear_attn.num_k_heads //= self.N
                    linear_attn.num_v_heads //= self.N
                    linear_attn.key_dim = (
                        linear_attn.head_k_dim * linear_attn.num_k_heads
                    )
                    linear_attn.value_dim = (
                        linear_attn.head_v_dim * linear_attn.num_v_heads
                    )
                    linear_attn.conv_dim = (
                        linear_attn.key_dim * 2 + linear_attn.value_dim
                    )
                else:
                    layer.self_attn.q_proj = self.all_to_sharded_linear(
                        layer.self_attn.q_proj
                    )
                    layer.self_attn.k_proj = self.all_to_sharded_linear(
                        layer.self_attn.k_proj
                    )
                    layer.self_attn.v_proj = self.all_to_sharded_linear(
                        layer.self_attn.v_proj
                    )
                    layer.self_attn.o_proj = self.sharded_to_all_linear(
                        layer.self_attn.o_proj
                    )
                    layer.self_attn.num_attention_heads //= self.N
                    layer.self_attn.num_key_value_heads //= self.N

            # Shard the MoE.
            if isinstance(
                layer.mlp,
                (
                    Qwen3MoeSparseMoeBlock,
                    Qwen3NextSparseMoeBlock,
                    Qwen3_5SparseMoeBlock,
                ),
            ):
                self.all_to_sharded_linear_in_place(layer.mlp.switch_mlp.gate_proj)
                self.sharded_to_all_linear_in_place(layer.mlp.switch_mlp.down_proj)
                self.all_to_sharded_linear_in_place(layer.mlp.switch_mlp.up_proj)
                if isinstance(
                    layer.mlp, (Qwen3NextSparseMoeBlock, Qwen3_5SparseMoeBlock)
                ):
                    self.all_to_sharded_linear_in_place(
                        layer.mlp.shared_expert.gate_proj
                    )
                    self.sharded_to_all_linear_in_place(
                        layer.mlp.shared_expert.down_proj
                    )
                    self.all_to_sharded_linear_in_place(layer.mlp.shared_expert.up_proj)
                layer.mlp = ShardedMoE(layer.mlp)  # pyright: ignore[reportAttributeAccessIssue, reportArgumentType]
                layer.mlp.sharding_group = self.group

            # Shard the MLP
            else:
                layer.mlp.gate_proj = self.all_to_sharded_linear(layer.mlp.gate_proj)
                layer.mlp.down_proj = self.sharded_to_all_linear(layer.mlp.down_proj)
                layer.mlp.up_proj = self.all_to_sharded_linear(layer.mlp.up_proj)

            mx.eval(layer)
            mx.clear_cache()

            yield ModelLoadingResponse(layers_loaded=i, total=total)
        return model


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


class NemotronHShardingStrategy(TensorParallelShardingStrategy):
    def shard_model(
        self,
        model: nn.Module,
    ) -> Generator[ModelLoadingResponse, None, nn.Module]:
        model = cast(NemotronHModel, model)
        rank = self.group.rank()
        total = len(model.layers)
        for i, layer in enumerate(model.layers):
            mx.eval(layer.parameters())

            mixer = layer.mixer

            if isinstance(mixer, NemotronHAttention):
                mixer.q_proj = self.all_to_sharded_linear(mixer.q_proj)
                mixer.k_proj = self.all_to_sharded_linear(mixer.k_proj)
                mixer.v_proj = self.all_to_sharded_linear(mixer.v_proj)
                mixer.o_proj = self.sharded_to_all_linear(mixer.o_proj)
                mixer.num_heads //= self.N
                mixer.num_key_value_heads //= self.N

            elif isinstance(mixer, NemotronHMamba2Mixer):
                self._shard_mamba2_mixer(mixer, rank)

            elif isinstance(mixer, NemotronHMoE):
                # Shard routed experts (SwitchMLP uses fc1/fc2)
                self.all_to_sharded_linear_in_place(mixer.switch_mlp.fc1)
                self.sharded_to_all_linear_in_place(mixer.switch_mlp.fc2)
                # Shard shared expert in-place (no all-reduce — ShardedMoE handles that)
                if hasattr(mixer, "shared_experts"):
                    self.all_to_sharded_linear_in_place(mixer.shared_experts.up_proj)
                    self.sharded_to_all_linear_in_place(mixer.shared_experts.down_proj)
                mixer = ShardedMoE(mixer)  # pyright: ignore[reportArgumentType]
                mixer.sharding_group = self.group
                layer.mixer = mixer  # pyright: ignore[reportAttributeAccessIssue]

            mx.eval(layer)
            mx.clear_cache()
            yield ModelLoadingResponse(layers_loaded=i, total=total)
        return model

    def _shard_mamba2_mixer(self, mixer: NemotronHMamba2Mixer, rank: int) -> None:
        """Shard the Mamba2 mixer along the head dimension."""
        world_size = self.N
        num_heads = mixer.num_heads
        head_dim = mixer.head_dim
        n_groups = mixer.n_groups
        ssm_state_size = mixer.ssm_state_size
        intermediate_size = mixer.intermediate_size  # = num_heads * head_dim

        # Per-rank sizes
        heads_per_rank = num_heads // world_size
        groups_per_rank = n_groups // world_size
        is_per_rank = heads_per_rank * head_dim
        bc_per_rank = groups_per_rank * ssm_state_size

        # === in_proj: output layout is [gate:IS | conv_ssm:IS | B:NG*SS | C:NG*SS | dt:NH] ===
        gate_start = 0
        conv_ssm_start = intermediate_size
        b_start = 2 * intermediate_size
        c_start = b_start + n_groups * ssm_state_size
        dt_start = c_start + n_groups * ssm_state_size

        # Build index tensor for this rank's slice of each section
        gate_idx = mx.arange(
            gate_start + rank * is_per_rank, gate_start + (rank + 1) * is_per_rank
        )
        conv_ssm_idx = mx.arange(
            conv_ssm_start + rank * is_per_rank,
            conv_ssm_start + (rank + 1) * is_per_rank,
        )
        b_idx = mx.arange(
            b_start + rank * bc_per_rank, b_start + (rank + 1) * bc_per_rank
        )
        c_idx = mx.arange(
            c_start + rank * bc_per_rank, c_start + (rank + 1) * bc_per_rank
        )
        dt_idx = mx.arange(
            dt_start + rank * heads_per_rank, dt_start + (rank + 1) * heads_per_rank
        )

        indices = mx.concatenate([gate_idx, conv_ssm_idx, b_idx, c_idx, dt_idx])
        mixer.in_proj.weight = mixer.in_proj.weight[indices]

        # === out_proj: input is intermediate_size (sharded) → hidden_size (reduce) ===
        mixer.out_proj = self.sharded_to_all_linear(mixer.out_proj)

        # === conv1d: depthwise conv on conv_dim channels ===
        # conv_dim layout: [ssm_hidden:IS | B:NG*SS | C:NG*SS]
        conv_ssm_idx_local = mx.arange(rank * is_per_rank, (rank + 1) * is_per_rank)
        conv_b_idx = mx.arange(
            intermediate_size + rank * bc_per_rank,
            intermediate_size + (rank + 1) * bc_per_rank,
        )
        conv_c_idx = mx.arange(
            intermediate_size + n_groups * ssm_state_size + rank * bc_per_rank,
            intermediate_size + n_groups * ssm_state_size + (rank + 1) * bc_per_rank,
        )
        conv_indices = mx.concatenate([conv_ssm_idx_local, conv_b_idx, conv_c_idx])
        mixer.conv1d.weight = mixer.conv1d.weight[conv_indices]
        new_conv_dim = is_per_rank + 2 * bc_per_rank
        mixer.conv1d.groups = new_conv_dim
        if mixer.conv1d.bias is not None:
            mixer.conv1d.bias = mixer.conv1d.bias[conv_indices]

        # === Per-head parameters ===
        h_start = rank * heads_per_rank
        h_end = h_start + heads_per_rank
        mixer.dt_bias = mixer.dt_bias[h_start:h_end]
        mixer.A_log = mixer.A_log[h_start:h_end]
        mixer.D = mixer.D[h_start:h_end]

        # === Norm: weight is intermediate_size ===
        mixer.norm.weight = mixer.norm.weight[
            rank * is_per_rank : (rank + 1) * is_per_rank
        ]

        # === Update dimensions ===
        mixer.num_heads = heads_per_rank
        mixer.n_groups = groups_per_rank
        mixer.intermediate_size = is_per_rank
        mixer.conv_dim = new_conv_dim
        mixer.heads_per_group = heads_per_rank // groups_per_rank


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
