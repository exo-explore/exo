from collections.abc import Generator
from functools import partial

import mlx.core as mx
import mlx.nn as nn
from mlx.nn.layers.distributed import (
    shard_inplace,
    shard_linear,
)

from exo.shared.types.worker.runner_response import ModelLoadingResponse
from exo.shared.types.worker.shards import PipelineShardMetadata
from exo.worker.engines.mlx.models.family import Architecture
from exo.worker.engines.mlx.models.registry import architecture_for_model
from exo.worker.engines.mlx.sharding.blocks import LayerCallable
from exo.worker.engines.mlx.sharding.pipeline import (
    PipelineFirstLayer,
    PipelineLastLayer,
    get_inner_model,
    get_layers,
    patch_pipeline_model,
)
from exo.worker.engines.mlx.sharding.tensor import patch_tensor_model
from exo.worker.runner.bootstrap import logger


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

    architecture = architecture_for_model(model)
    if architecture is not None and architecture.fix_pipeline_shard is not None:
        architecture.fix_pipeline_shard(
            model, inner_model_instance, layers, start_layer, end_layer
        )

    _set_layers(model, layers, architecture)

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
        sharding=_all_to_sharded,
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
        sharding=_sharded_to_all,
        group=group,
    )

    architecture = architecture_for_model(model)
    if architecture is None or architecture.tensor_parallel is None:
        raise ValueError(f"Unsupported model type: {type(model)}")
    tensor_parallel_sharding_strategy = architecture.tensor_parallel(
        group,
        all_to_sharded_linear,
        sharded_to_all_linear,
        all_to_sharded_linear_in_place,
        sharded_to_all_linear_in_place,
    )

    model = yield from tensor_parallel_sharding_strategy.shard_model(model)
    return patch_tensor_model(model)


def _set_layers(
    model: nn.Module, layers: list[LayerCallable], architecture: Architecture | None
) -> None:
    inner_model_instance = get_inner_model(model)
    layer_count = None if architecture is None else architecture.pipeline_layer_count
    if hasattr(inner_model_instance, "layers"):
        inner_model_instance.layers = layers

        # Update DeepSeek V3 specific parameters when layers are shrunk
        if layer_count == "num_layers" and hasattr(inner_model_instance, "num_layers"):
            logger.info(
                f"Setting num_layers to {len(layers)} for model {inner_model_instance.__class__.__name__}"
            )
            inner_model_instance.start_idx = 0
            inner_model_instance.end_idx = len(layers)
            inner_model_instance.num_layers = len(layers)
        elif layer_count == "num_hidden_layers":
            logger.info(
                f"Setting num_hidden_layers to {len(layers)} for model {inner_model_instance.__class__.__name__}"
            )
            inner_model_instance.num_hidden_layers = len(layers)
    elif hasattr(inner_model_instance, "h"):
        inner_model_instance.h = layers
    else:
        raise ValueError("Model must have either a 'layers' or 'h' attribute")
