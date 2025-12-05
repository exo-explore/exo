"""
Pipeline parallelism for Flux transformer.

This module provides the DistributedTransformer wrapper that handles distributed
communication at the transformer level. Block slicing and send/recv operations
are managed by the wrapper, not per-block wrappers.
"""

import mlx.core as mx
from mflux.models.flux.variants.txt2img.flux import Flux1

from exo.shared.types.worker.shards import PipelineShardMetadata
from exo.worker.engines.mflux.pipefusion.distributed_transformer import (
    DistributedTransformer,
)


def apply_pipefusion_transformer(
    model: Flux1,
    group: mx.distributed.Group,
    shard_metadata: PipelineShardMetadata,
) -> Flux1:
    """
    Apply pipeline parallelism to the Flux1 model's transformer.

    This wraps the transformer with DistributedTransformer, which handles:
    - Block slicing (each stage runs only its assigned blocks)
    - Communication (send/recv at phase boundaries)
    - Final output synchronization (all_gather)

    The original transformer's blocks are NOT modified - DistributedTransformer
    accesses them by index based on shard_metadata.

    Args:
        model: The Flux1 model to parallelize
        group: The MLX distributed group
        shard_metadata: Metadata containing layer assignments for this stage

    Returns:
        The model with its transformer wrapped in DistributedTransformer
    """
    # Wrap the transformer with distributed communication handling
    # The wrapper reimplements forward() with inline send/recv
    model.transformer = DistributedTransformer(  # type: ignore[assignment]
        transformer=model.transformer,
        group=group,
        shard_metadata=shard_metadata,
    )

    return model
