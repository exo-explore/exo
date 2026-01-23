from collections.abc import Generator
from pathlib import Path
from typing import Any, Literal, Optional

import mlx.core as mx
from mflux.models.common.config.config import Config
from PIL import Image

from exo.shared.types.api import AdvancedImageParams
from exo.shared.types.worker.instances import BoundInstance
from exo.shared.types.worker.shards import PipelineShardMetadata
from exo.worker.download.download_utils import build_model_path
from exo.worker.engines.image.config import ImageModelConfig
from exo.worker.engines.image.models import (
    create_adapter_for_model,
    get_config_for_model,
)
from exo.worker.engines.image.models.base import ModelAdapter
from exo.worker.engines.image.pipeline import DiffusionRunner
from exo.worker.engines.mlx.utils_mlx import mlx_distributed_init, mx_barrier
from exo.worker.runner.bootstrap import logger


class DistributedImageModel:
    _config: ImageModelConfig
    _adapter: ModelAdapter[Any, Any]
    _runner: DiffusionRunner

    def __init__(
        self,
        model_id: str,
        local_path: Path,
        shard_metadata: PipelineShardMetadata,
        group: Optional[mx.distributed.Group] = None,
        quantize: int | None = None,
    ):
        config = get_config_for_model(model_id)
        adapter = create_adapter_for_model(config, model_id, local_path, quantize)

        if group is not None:
            adapter.slice_transformer_blocks(
                start_layer=shard_metadata.start_layer,
                end_layer=shard_metadata.end_layer,
            )

        runner = DiffusionRunner(
            config=config,
            adapter=adapter,
            group=group,
            shard_metadata=shard_metadata,
        )

        if group is not None:
            logger.info("Initialized distributed diffusion runner")

            mx.eval(adapter.model.parameters())  # pyright: ignore[reportAny]

            # TODO(ciaran): Do we need this?
            mx.eval(adapter.model)  # pyright: ignore[reportAny]

            mx_barrier(group)
            logger.info(f"Transformer sharded for rank {group.rank()}")
        else:
            logger.info("Single-node initialization")

        self._config = config
        self._adapter = adapter
        self._runner = runner

    @classmethod
    def from_bound_instance(
        cls, bound_instance: BoundInstance
    ) -> "DistributedImageModel":
        model_id = bound_instance.bound_shard.model_card.model_id
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

    def get_steps_for_quality(self, quality: Literal["low", "medium", "high"]) -> int:
        """Get the number of inference steps for a quality level."""
        return self._config.get_steps_for_quality(quality)

    def generate(
        self,
        prompt: str,
        height: int,
        width: int,
        quality: Literal["low", "medium", "high"] = "medium",
        seed: int = 2,
        image_path: Path | None = None,
        partial_images: int = 0,
        advanced_params: AdvancedImageParams | None = None,
    ) -> Generator[Image.Image | tuple[Image.Image, int, int], None, None]:
        if (
            advanced_params is not None
            and advanced_params.num_inference_steps is not None
        ):
            steps = advanced_params.num_inference_steps
        else:
            steps = self._config.get_steps_for_quality(quality)

        guidance_override: float | None = None
        if advanced_params is not None and advanced_params.guidance is not None:
            guidance_override = advanced_params.guidance

        negative_prompt: str | None = None
        if advanced_params is not None and advanced_params.negative_prompt is not None:
            negative_prompt = advanced_params.negative_prompt

        # For edit mode: compute dimensions from input image
        # This also stores image_paths in the adapter for encode_prompt()
        if image_path is not None:
            computed_dims = self._adapter.set_image_dimensions(image_path)
            if computed_dims is not None:
                # Override user-provided dimensions with computed ones
                width, height = computed_dims

        config = Config(
            num_inference_steps=steps,
            height=height,
            width=width,
            image_path=image_path,
            model_config=self._adapter.model.model_config,  # pyright: ignore[reportAny]
        )

        num_sync_steps = self._config.get_num_sync_steps(steps)

        for result in self._runner.generate_image(
            runtime_config=config,
            prompt=prompt,
            seed=seed,
            partial_images=partial_images,
            guidance_override=guidance_override,
            negative_prompt=negative_prompt,
            num_sync_steps=num_sync_steps,
        ):
            if isinstance(result, tuple):
                # Partial image: (GeneratedImage, partial_index, total_partials)
                image, partial_idx, total_partials = result
                yield (image, partial_idx, total_partials)
            else:
                logger.info("generated image")
                yield result


def initialize_image_model(bound_instance: BoundInstance) -> DistributedImageModel:
    return DistributedImageModel.from_bound_instance(bound_instance)
