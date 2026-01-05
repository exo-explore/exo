from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, Optional

import mlx.core as mx
from mflux.config.config import Config
from PIL import Image

from exo.shared.types.worker.instances import BoundInstance
from exo.shared.types.worker.shards import PipelineShardMetadata
from exo.worker.download.download_utils import build_model_path
from exo.worker.engines.image.config import ImageModelConfig
from exo.worker.engines.image.models import (
    create_adapter_for_model,
    get_config_for_model,
)
from exo.worker.engines.image.models.base import BaseModelAdapter
from exo.worker.engines.image.pipeline import DiffusionRunner
from exo.worker.engines.mlx.utils_mlx import mlx_distributed_init, mx_barrier
from exo.worker.runner.bootstrap import logger


class DistributedImageModel:
    __slots__ = (
        "_config",
        "_adapter",
        "_group",
        "_shard_metadata",
        "_runner",
    )

    _config: ImageModelConfig
    _adapter: BaseModelAdapter
    _group: Optional[mx.distributed.Group]
    _shard_metadata: PipelineShardMetadata
    _runner: DiffusionRunner

    def __init__(
        self,
        model_id: str,
        local_path: Path,
        shard_metadata: PipelineShardMetadata,
        group: Optional[mx.distributed.Group] = None,
        quantize: int | None = None,
    ):
        # Get model config and create adapter (adapter owns the model)
        config = get_config_for_model(model_id)
        adapter = create_adapter_for_model(config, model_id, local_path, quantize)

        # Create diffusion runner (handles both single-node and distributed modes)
        num_sync_steps = config.get_num_sync_steps("medium") if group else 0
        runner = DiffusionRunner(
            config=config,
            adapter=adapter,
            group=group,
            shard_metadata=shard_metadata,
            num_sync_steps=num_sync_steps,
        )

        if group is not None:
            logger.info("Initialized distributed diffusion runner")

            mx.eval(adapter.model.parameters())

            # TODO: Do we need this?
            mx.eval(adapter.model)

            # Synchronize processes before generation to avoid timeout
            mx_barrier(group)
            logger.info(f"Transformer sharded for rank {group.rank()}")
        else:
            logger.info("Single-node initialization")

        object.__setattr__(self, "_config", config)
        object.__setattr__(self, "_adapter", adapter)
        object.__setattr__(self, "_group", group)
        object.__setattr__(self, "_shard_metadata", shard_metadata)
        object.__setattr__(self, "_runner", runner)

    @classmethod
    def from_bound_instance(
        cls, bound_instance: BoundInstance
    ) -> "DistributedImageModel":
        model_id = bound_instance.bound_shard.model_meta.model_id
        model_path = build_model_path(model_id)

        shard_metadata = bound_instance.bound_shard
        if not isinstance(shard_metadata, PipelineShardMetadata):
            raise ValueError("Expected PipelineShardMetadata for image generation")

        is_distributed = (
            len(bound_instance.instance.shard_assignments.node_to_runner) > 1
        )

        if is_distributed:
            logger.info("Starting distributed init for image model")
            group = mlx_distributed_init(bound_instance)
        else:
            group = None

        return cls(
            model_id=model_id,
            local_path=model_path,
            shard_metadata=shard_metadata,
            group=group,
        )

    @property
    def model(self) -> Any:
        """Return the underlying mflux model via the adapter."""
        return self._adapter.model

    @property
    def config(self) -> ImageModelConfig:
        return self._config

    @property
    def adapter(self) -> BaseModelAdapter:
        return self._adapter

    @property
    def group(self) -> Optional[mx.distributed.Group]:
        return self._group

    @property
    def shard_metadata(self) -> PipelineShardMetadata:
        return self._shard_metadata

    @property
    def rank(self) -> int:
        return self._shard_metadata.device_rank

    @property
    def world_size(self) -> int:
        return self._shard_metadata.world_size

    @property
    def is_first_stage(self) -> bool:
        return self._shard_metadata.device_rank == 0

    @property
    def is_last_stage(self) -> bool:
        return self._shard_metadata.device_rank == self._shard_metadata.world_size - 1

    @property
    def is_distributed(self) -> bool:
        return self._shard_metadata.world_size > 1

    @property
    def runner(self) -> DiffusionRunner:
        return self._runner

    # Delegate attribute access to the underlying model via the adapter.
    # Guarded with TYPE_CHECKING to prevent type checker complaints
    # while still providing full delegation at runtime.
    if not TYPE_CHECKING:

        def __getattr__(self, name: str) -> Any:
            return getattr(self._adapter.model, name)

        def __setattr__(self, name: str, value: Any) -> None:
            if name in (
                "_config",
                "_adapter",
                "_group",
                "_shard_metadata",
                "_runner",
            ):
                object.__setattr__(self, name, value)
            else:
                setattr(self._adapter.model, name, value)

    def generate(
        self,
        prompt: str,
        height: int,
        width: int,
        quality: Literal["low", "medium", "high"] = "medium",
        seed: int = 2,
    ) -> Optional[Image.Image]:
        # Determine number of inference steps based on quality
        steps = self._config.get_steps_for_quality(quality)

        config = Config(num_inference_steps=steps, height=height, width=width)
        image = self._generate_image(settings=config, prompt=prompt, seed=seed)
        logger.info("generated image")

        # Only rank 0 returns the actual image
        if self.is_first_stage:
            return image.image

    def _generate_image(self, settings: Config, prompt: str, seed: int) -> Any:
        """Generate image by delegating to the runner."""
        return self._runner.generate_image(
            settings=settings,
            prompt=prompt,
            seed=seed,
        )


def initialize_image_model(bound_instance: BoundInstance) -> DistributedImageModel:
    """Initialize DistributedImageModel from a BoundInstance."""
    return DistributedImageModel.from_bound_instance(bound_instance)
