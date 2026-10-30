from collections.abc import Generator

import mlx.core as mx
import mlx.nn as nn

from exo.shared.types.worker.runner_response import ModelLoadingResponse
from exo.shared.types.worker.shards import PipelineShardMetadata
from exo.worker.engines.mlx.models.deepseek_v41.model import Model, PipelineStage
from exo.worker.engines.mlx.sharding.tensor import TensorParallelShardingStrategy


class DeepseekV41ShardingStrategy(TensorParallelShardingStrategy):
    """The model's own split (Model.shard): whole output groups and their heads per rank, a
    slice of every expert's intermediate dimension; the compressor, indexer,
    hyper-connections and engram replicated."""

    def shard_model(
        self, model: nn.Module
    ) -> Generator[ModelLoadingResponse, None, nn.Module]:
        assert isinstance(model, Model)
        model.shard(self.group)
        layers = model.layers
        for i, layer in enumerate(layers):
            mx.eval(layer.parameters())
            mx.clear_cache()
            yield ModelLoadingResponse(layers_loaded=i, total=len(layers))
        return model


def pipeline_parallel(
    model: nn.Module,
    group: mx.distributed.Group,
    model_shard_meta: PipelineShardMetadata,
) -> Generator[ModelLoadingResponse, None, nn.Module]:
    """Keep the node's layers and let the model pass its streams, pre_mix and the shared
    compressed KV / index keys / top-k state to the next stage itself (rank r runs the
    layers just before rank r + 1's)."""
    assert isinstance(model, Model)
    rank, size = model_shard_meta.device_rank, model_shard_meta.world_size
    assert rank == group.rank() and size == group.size()
    model.model.set_pipeline_stage(
        PipelineStage(
            group=group,
            start=model_shard_meta.start_layer,
            end=model_shard_meta.end_layer,
            previous_rank=rank - 1 if rank > 0 else None,
            next_rank=rank + 1 if rank < size - 1 else None,
            last_rank=size - 1,
        )
    )
    layers = model.layers
    for i, layer in enumerate(layers):
        mx.eval(layer.parameters())
        mx.clear_cache()
        yield ModelLoadingResponse(layers_loaded=i, total=len(layers))
    return model
