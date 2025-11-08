from abc import ABC, abstractmethod
from functools import partial
from typing import TYPE_CHECKING, Callable, Protocol, cast, override

from mlx_lm.models.deepseek_v3 import DeepseekV3MLP
from mlx_lm.models.deepseek_v3 import Model as DeepseekV3Model
from mlx_lm.models.llama import Model as LlamaModel
from mlx_lm.models.qwen3_moe import Model as Qwen3MoeModel
from mlx_lm.models.qwen3_moe import Qwen3MoeSparseMoeBlock

import mlx.core as mx
import mlx.nn as nn
from exo.shared.types.worker.shards import (
    PipelineShardMetadata,
    ShardMetadata,
    TensorShardMetadata,
)
from mlx.nn.layers.distributed import (
    shard_inplace,
    shard_linear,
    sum_gradients,
)


class _LayerCallable(Protocol):
    """Structural type that any compatible layer must satisfy.

    We require a single positional input of type ``mx.array`` and an
    ``mx.array`` output, while permitting arbitrary *args / **kwargs so this
    protocol matches the vast majority of `mlx.nn.Module` subclasses.
    """

    def __call__(self, x: mx.array, *args: object, **kwargs: object) -> mx.array: ...


class CustomMlxLayer(nn.Module):
    """Base class for replacing an MLX layer with a custom implementation."""

    def __init__(self, original_layer: _LayerCallable):
        super().__init__()
        # Set twice to avoid __setattr__ recursion
        object.__setattr__(self, "_original_layer", original_layer)
        self.original_layer: _LayerCallable = original_layer

    # Calls __getattr__ for any attributes not found on nn.Module (e.g. use_sliding)
    if not TYPE_CHECKING:

        def __getattr__(self, name):
            try:
                return super().__getattr__(name)
            except AttributeError:
                original_layer = object.__getattribute__(self, "_original_layer")
                return object.__getattribute__(original_layer, name)


class PipelineFirstLayer(CustomMlxLayer):
    def __init__(
        self,
        original_layer: _LayerCallable,
        r: int,
        s: int,
        group: mx.distributed.Group | None = None,
    ):
        super().__init__(original_layer)
        self.r: int = r
        self.s: int = s
        self.group = group

    @override
    def __call__(self, x: mx.array, *args: object, **kwargs: object) -> mx.array:
        if self.r != 0:
            x = mx.distributed.recv_like(x, (self.r - 1), group=self.group)
        return self.original_layer(x, *args, **kwargs)


class PipelineLastLayer(CustomMlxLayer):
    def __init__(
        self,
        original_layer: _LayerCallable,
        r: int,
        s: int,
        group: mx.distributed.Group | None = None,
    ):
        super().__init__(original_layer)
        self.r: int = r
        self.s: int = s
        self.group = group

    @override
    def __call__(
        self, x: mx.array, *args: object, cache: object = None, **kwargs: object
    ) -> mx.array:
        output: mx.array = self.original_layer(x, *args, **kwargs)
        if self.r != self.s - 1:
            output = mx.distributed.send(
                output, (self.r + 1) % self.s, group=self.group
            )
            if (
                cache is not None
                and hasattr(cache, "keys")
                and getattr(cache, "keys", None) is not None
            ):
                cache.keys = mx.depends(cache.keys, output)  # type: ignore[reportUnknownMemberType]

        output = mx.distributed.all_gather(output, group=self.group)[-output.shape[0] :]
        return output


class ParallelisationShardStrategy(Protocol):
    def auto_parallel(
        self, model: nn.Module, model_shard_meta: ShardMetadata
    ) -> nn.Module: ...


class PipelineParallelisationStrategy(ParallelisationShardStrategy):
    def __init__(self, group: mx.distributed.Group):
        self.group = group

    def auto_parallel(
        self, model: nn.Module, model_shard_meta: ShardMetadata
    ) -> nn.Module:
        """
        Automatically parallelize a model across multiple devices.
        Args:
        model: The model to parallelize (must have a 'layers' or 'h' property)
        model_shard_meta: The metadata for the model shard
        Returns:
        The parallelized model
        """
        assert isinstance(model_shard_meta, PipelineShardMetadata)

        inner_model_instance: nn.Module = PipelineParallelisationStrategy._inner_model(
            model
        )

        # Handle both model.layers and model.h cases
        layers: list[_LayerCallable]
        if hasattr(inner_model_instance, "layers"):
            layers = cast(list[_LayerCallable], inner_model_instance.layers)
        elif hasattr(inner_model_instance, "h"):
            layers = cast(list[_LayerCallable], inner_model_instance.h)
        else:
            raise ValueError("Model must have either a 'layers' or 'h' attribute")

        layers = layers[model_shard_meta.start_layer : model_shard_meta.end_layer]
        layers[0] = PipelineFirstLayer(
            layers[0],
            model_shard_meta.device_rank,
            model_shard_meta.world_size,
            group=self.group,
        )
        layers[-1] = PipelineLastLayer(
            layers[-1],
            model_shard_meta.device_rank,
            model_shard_meta.world_size,
            group=self.group,
        )

        PipelineParallelisationStrategy._set_layers(model, layers)

        return model

    @staticmethod
    def _inner_model(model: nn.Module) -> nn.Module:
        inner = getattr(model, "model", None)
        if isinstance(inner, nn.Module):
            return inner

        inner = getattr(model, "transformer", None)
        if isinstance(inner, nn.Module):
            return inner

        raise ValueError("Model must either have a 'model' or 'transformer' attribute")

    @staticmethod
    def _set_layers(model: nn.Module, layers: list[_LayerCallable]) -> None:
        inner_model_instance = PipelineParallelisationStrategy._inner_model(model)
        if hasattr(inner_model_instance, "layers"):
            inner_model_instance.layers = layers

            # Update DeepSeek V3 specific parameters when layers are shrunk
            if isinstance(model, DeepseekV3Model) and hasattr(
                inner_model_instance, "num_layers"
            ):
                inner_model_instance.start_idx = 0
                inner_model_instance.end_idx = len(layers)
                inner_model_instance.num_layers = len(layers)
        elif hasattr(inner_model_instance, "h"):
            inner_model_instance.h = layers
        else:
            raise ValueError("Model must have either a 'layers' or 'h' attribute")


class TensorParallelisationStrategy(ParallelisationShardStrategy):
    def __init__(self, group: mx.distributed.Group):
        self.group = group

    def auto_parallel(
        self, model: nn.Module, model_shard_meta: ShardMetadata
    ) -> nn.Module:
        assert isinstance(model_shard_meta, TensorShardMetadata)

        all_to_sharded_linear = partial(
            shard_linear,
            sharding="all-to-sharded",
            group=self.group,
        )
        sharded_to_all_linear = partial(
            shard_linear,
            sharding="sharded-to-all",
            group=self.group,
        )

        all_to_sharded_linear_in_place = partial(
            shard_inplace,
            sharding="all-to-sharded",
            group=self.group,
        )
        sharded_to_all_linear_in_place = partial(
            shard_inplace,
            sharding="sharded-to-all",
            group=self.group,
        )

        if isinstance(model, LlamaModel):
            tensor_parallel_sharding_strategy = LlamaShardingStrategy(
                self.group,
                all_to_sharded_linear,
                sharded_to_all_linear,
                all_to_sharded_linear_in_place,
                sharded_to_all_linear_in_place,
            )
        elif isinstance(model, DeepseekV3Model):
            tensor_parallel_sharding_strategy = DeepSeekShardingStrategy(
                self.group,
                all_to_sharded_linear,
                sharded_to_all_linear,
                all_to_sharded_linear_in_place,
                sharded_to_all_linear_in_place,
            )
        elif isinstance(model, Qwen3MoeModel):
            tensor_parallel_sharding_strategy = QwenShardingStrategy(
                self.group,
                all_to_sharded_linear,
                sharded_to_all_linear,
                all_to_sharded_linear_in_place,
                sharded_to_all_linear_in_place,
            )
        else:
            raise ValueError(f"Unsupported model type: {type(model)}")

        return tensor_parallel_sharding_strategy.shard_model(model)


class TensorParallelShardingStrategy(ABC):
    def __init__(
        self,
        group: mx.distributed.Group,
        all_to_sharded_linear: Callable[..., nn.Linear],
        sharded_to_all_linear: Callable[..., nn.Linear],
        all_to_sharded_linear_in_place: Callable[..., None],
        sharded_to_all_linear_in_place: Callable[..., None],
    ):
        self.all_to_sharded_linear = all_to_sharded_linear
        self.sharded_to_all_linear = sharded_to_all_linear
        self.all_to_sharded_linear_in_place = all_to_sharded_linear_in_place
        self.sharded_to_all_linear_in_place = sharded_to_all_linear_in_place
        self.group = group
        self.N = group.size()

    @abstractmethod
    def shard_model(self, model: nn.Module) -> nn.Module: ...


class LlamaShardingStrategy(TensorParallelShardingStrategy):
    def shard_model(self, model: nn.Module) -> nn.Module:
        model = cast(LlamaModel, model)
        for layer in model.layers:
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

        return model


class DeepSeekShardingStrategy(TensorParallelShardingStrategy):
    def shard_model(self, model: nn.Module) -> nn.Module:
        model = cast(DeepseekV3Model, model)
        for layer in model.layers:
            # Shard the self attention
            if layer.self_attn.q_lora_rank is None:  # pyright: ignore[reportUnnecessaryComparison]
                # Unfortunately, q_lora_rank can be None despite typing hints.
                layer.self_attn.q_proj = self.all_to_sharded_linear(
                    layer.self_attn.q_proj
                )
            else:
                layer.self_attn.q_b_proj = self.all_to_sharded_linear(
                    layer.self_attn.q_b_proj
                )
            layer.self_attn.kv_b_proj = self.all_to_sharded_linear(
                layer.self_attn.kv_b_proj
            )
            layer.self_attn.o_proj = self.sharded_to_all_linear(layer.self_attn.o_proj)
            layer.self_attn.num_heads //= self.N

            # Shard the MLP
            if isinstance(layer.mlp, DeepseekV3MLP):
                layer.mlp.gate_proj = self.all_to_sharded_linear(layer.mlp.gate_proj)
                layer.mlp.down_proj = self.sharded_to_all_linear(layer.mlp.down_proj)
                layer.mlp.up_proj = self.all_to_sharded_linear(layer.mlp.up_proj)

            # Shard the MoE. Shard in place since the MoE should be responsible
            # for aggregating the results.
            else:
                self.all_to_sharded_linear_in_place(layer.mlp.shared_experts.gate_proj)
                self.sharded_to_all_linear_in_place(layer.mlp.shared_experts.down_proj)
                self.all_to_sharded_linear_in_place(layer.mlp.shared_experts.up_proj)
                self.all_to_sharded_linear_in_place(layer.mlp.switch_mlp.gate_proj)
                self.sharded_to_all_linear_in_place(layer.mlp.switch_mlp.down_proj)
                self.all_to_sharded_linear_in_place(layer.mlp.switch_mlp.up_proj)
                layer.mlp = ShardedDeepseekV3MoE(layer.mlp)  # type: ignore
                layer.mlp.sharding_group = self.group

        return model


class ShardedDeepseekV3MoE(CustomMlxLayer):
    def __init__(self, layer: _LayerCallable):
        super().__init__(layer)
        self.sharding_group: mx.distributed.Group | None = None

    def __call__(self, x: mx.array) -> mx.array:
        if self.sharding_group is not None:
            x = sum_gradients(self.sharding_group)(x)
        y = self.original_layer.__call__(x)
        if self.sharding_group is not None:
            y = mx.distributed.all_sum(y, group=self.sharding_group)
        return y


class QwenShardingStrategy(TensorParallelShardingStrategy):
    def shard_model(self, model: nn.Module) -> nn.Module:
        model = cast(Qwen3MoeModel, model)
        for layer in model.layers:
            # Shard the self attention
            layer.self_attn.q_proj = self.all_to_sharded_linear(layer.self_attn.q_proj)
            layer.self_attn.k_proj = self.all_to_sharded_linear(layer.self_attn.k_proj)
            layer.self_attn.v_proj = self.all_to_sharded_linear(layer.self_attn.v_proj)
            layer.self_attn.o_proj = self.sharded_to_all_linear(layer.self_attn.o_proj)
            layer.self_attn.n_heads //= self.N
            layer.self_attn.n_kv_heads //= self.N

            # Shard the MoE. Shard in place since the MoE should be responsible
            # for aggregating the results.
            if isinstance(layer.mlp, Qwen3MoeSparseMoeBlock):
                self.all_to_sharded_linear_in_place(layer.mlp.switch_mlp.gate_proj)
                self.sharded_to_all_linear_in_place(layer.mlp.switch_mlp.down_proj)
                self.all_to_sharded_linear_in_place(layer.mlp.switch_mlp.up_proj)
                layer.mlp = ShardedQwenMoE(layer.mlp)  # type: ignore
                layer.mlp.sharding_group = self.group

            # Shard the MLP
            else:
                layer.mlp.gate_proj = self.all_to_sharded_linear(layer.mlp.gate_proj)
                layer.mlp.down_proj = self.sharded_to_all_linear(layer.mlp.down_proj)
                layer.mlp.up_proj = self.all_to_sharded_linear(layer.mlp.up_proj)

        return model


class ShardedQwenMoE(CustomMlxLayer):
    def __init__(self, layer: _LayerCallable):
        super().__init__(layer)
        self.sharding_group: mx.distributed.Group | None = None

    def __call__(self, x: mx.array) -> mx.array:
        if self.sharding_group is not None:
            x = sum_gradients(self.sharding_group)(x)
        y = self.original_layer.__call__(x)
        if self.sharding_group is not None:
            y = mx.distributed.all_sum(y, group=self.sharding_group)
        return y
