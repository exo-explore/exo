from loguru import logger
from mflux.config.model_config import ModelConfig
from mflux.models.flux.variants.txt2img.flux import Flux1

from exo.shared.types.worker.instances import BoundInstance
from exo.shared.types.worker.shards import PipelineShardMetadata
from exo.worker.download.download_utils import build_model_path
from exo.worker.engines.mflux.distributed_flux import DistributedFlux1
from exo.worker.engines.mflux.shard_mflux import shard_flux_transformer
from exo.worker.engines.mlx.utils_mlx import mlx_distributed_init


def initialize_mflux(bound_instance: BoundInstance) -> Flux1 | DistributedFlux1:
    model_id = bound_instance.bound_shard.model_meta.model_id
    model_path = build_model_path(model_id)

    # TODO: generalise
    model = Flux1(
        model_config=ModelConfig.from_name(model_name=model_id, base_model=None),
        local_path=str(model_path),
        # quantize=8,
    )

    is_distributed = len(bound_instance.instance.shard_assignments.node_to_runner) > 1

    if is_distributed:
        # Multi-node: initialize distributed and shard transformer
        logger.info("Starting distributed init for Flux")
        group = mlx_distributed_init(bound_instance)

        shard_metadata = bound_instance.bound_shard
        if not isinstance(shard_metadata, PipelineShardMetadata):
            raise ValueError("Expected PipelineShardMetadata for distributed Flux")

        model = shard_flux_transformer(
            model=model,
            group=group,
            shard_metadata=shard_metadata,
        )
        logger.info(f"Flux transformer sharded for rank {group.rank()}")

        # Wrap with distributed context for runtime access
        return DistributedFlux1(
            model=model,
            group=group,
            shard_metadata=shard_metadata,
        )

    return model
