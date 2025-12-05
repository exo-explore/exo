import mlx.core as mx
from mflux.models.flux.variants.txt2img.flux import Flux1

from exo.shared.types.worker.shards import PipelineShardMetadata
from exo.worker.engines.mflux.pipefusion.pipefusion import apply_pipefusion_transformer
from exo.worker.engines.mlx.utils_mlx import mx_barrier


def shard_flux_transformer(
    model: Flux1,
    group: mx.distributed.Group,
    shard_metadata: PipelineShardMetadata,
) -> Flux1:
    """
    Apply distributed sharding to the Flux1 transformer.

    Wraps the transformer with DistributedTransformer for pipeline parallelism,
    then synchronizes all nodes before returning.

    Args:
        model: The Flux1 model to shard
        group: The MLX distributed group
        shard_metadata: Pipeline shard metadata with layer assignments

    Returns:
        The model with sharded transformer
    """
    model = apply_pipefusion_transformer(model, group, shard_metadata)

    mx.eval(model.parameters())

    # TODO: Do we need this?
    mx.eval(model)

    # Synchronize processes before generation to avoid timeout
    mx_barrier(group)

    return model
