import mlx.core as mx
from mflux.models.flux.variants.txt2img.flux import Flux1

from exo.shared.types.worker.shards import (
    PipelineShardMetadata,
    ShardMetadata,
)
from exo.worker.engines.mflux.pipefusion import apply_pipefusion_transformer
from exo.worker.engines.mlx.utils_mlx import mx_barrier


def shard_flux_transformer(
    model: Flux1,
    group: mx.distributed.Group,
    shard_metadata: ShardMetadata,
) -> Flux1:
    if not isinstance(shard_metadata, PipelineShardMetadata):
        raise ValueError("Unsupported parallelism")

    model = apply_pipefusion_transformer(model, group, shard_metadata)

    mx.eval(model.parameters())

    # TODO: Do we need this?
    mx.eval(model)

    # Synchronize processes before generation to avoid timeout
    mx_barrier(group)

    return model
