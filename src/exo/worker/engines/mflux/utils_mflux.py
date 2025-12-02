from loguru import logger
from mflux.config.model_config import ModelConfig
from mflux.models.flux.variants.txt2img.flux import Flux1

from exo.shared.types.worker.instances import BoundInstance
from exo.worker.download.download_utils import build_model_path
from exo.worker.engines.mflux.shard_mflux import shard_flux_transformer
from exo.worker.engines.mlx.utils_mlx import mlx_distributed_init


def initialize_mflux(bound_instance: BoundInstance) -> Flux1:
    model_id = bound_instance.bound_shard.model_meta.model_id
    model_path = build_model_path(model_id)

    is_distributed = len(bound_instance.instance.shard_assignments.node_to_runner) > 1

    if not is_distributed:
        # Single-node: load full model normally
        logger.info(f"Single device used for {bound_instance.instance}")
        model = Flux1(
            model_config=ModelConfig.from_name(model_name=model_id, base_model=None),
            local_path=str(model_path),
            # quantize=8,
        )
    else:
        # Multi-node: initialize distributed and shard transformer
        logger.info("Starting distributed init for Flux")
        group = mlx_distributed_init(bound_instance)

        logger.info("Loading Flux model for distributed inference")
        model = Flux1(
            model_config=ModelConfig.from_name(model_name=model_id, base_model=None),
            local_path=str(model_path),
            # quantize=8,
        )

        logger.info("Applying pipeline parallelism to Flux transformer")
        model = shard_flux_transformer(
            model=model,
            group=group,
            shard_metadata=bound_instance.bound_shard,
        )
        logger.info(f"Flux transformer sharded for rank {group.rank()}")

    return model
