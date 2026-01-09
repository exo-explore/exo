from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import mlx.core as mx
from mflux.config.runtime_config import RuntimeConfig
from mflux.models.common.latent_creator.latent_creator import Img2Img, LatentCreator
from mflux.utils.array_util import ArrayUtil
from mflux.utils.image_util import ImageUtil


class BaseModelAdapter(ABC):
    """Base class for model adapters with shared utilities.

    Provides common implementations for latent creation and decoding.
    Subclasses implement model-specific prompt encoding and noise computation.
    """

    def create_latents(self, seed: int, runtime_config: RuntimeConfig) -> mx.array:
        """Create initial latents. Uses model-specific latent creator."""
        return LatentCreator.create_for_txt2img_or_img2img(
            seed=seed,
            height=runtime_config.height,
            width=runtime_config.width,
            img2img=Img2Img(
                vae=self.model.vae,
                latent_creator=self._get_latent_creator(),
                sigmas=runtime_config.scheduler.sigmas,
                init_time_step=runtime_config.init_time_step,
                image_path=runtime_config.image_path,
            ),
        )

    def decode_latents(
        self,
        latents: mx.array,
        runtime_config: RuntimeConfig,
        seed: int,
        prompt: str,
    ) -> Any:
        """Decode latents to image. Shared implementation."""
        latents = ArrayUtil.unpack_latents(
            latents=latents,
            height=runtime_config.height,
            width=runtime_config.width,
        )
        decoded = self.model.vae.decode(latents)
        return ImageUtil.to_image(
            decoded_latents=decoded,
            config=runtime_config,
            seed=seed,
            prompt=prompt,
            quantization=self.model.bits,
            lora_paths=self.model.lora_paths,
            lora_scales=self.model.lora_scales,
            image_path=runtime_config.image_path,
            image_strength=runtime_config.image_strength,
            generation_time=0,
        )

    # Abstract methods - subclasses must implement

    @property
    @abstractmethod
    def model(self) -> Any:
        """Return the underlying mflux model."""
        ...

    @abstractmethod
    def _get_latent_creator(self) -> type:
        """Return the latent creator class for this model."""
        ...

    @abstractmethod
    def slice_transformer_blocks(
        self,
        start_layer: int,
        end_layer: int,
        total_joint_blocks: int,
        total_single_blocks: int,
    ):
        """Remove transformer blocks outside the assigned range.

        This should be called BEFORE mx.eval() to avoid loading unused weights
        in distributed mode.

        Args:
            start_layer: First layer index (inclusive) assigned to this node
            end_layer: Last layer index (exclusive) assigned to this node
            total_joint_blocks: Total number of joint blocks in the model
            total_single_blocks: Total number of single blocks in the model
        """
        ...

    def set_image_dimensions(self, image_path: Path) -> tuple[int, int] | None:
        """Default implementation: no dimension computation needed.

        Override in edit adapters to compute dimensions from input image.

        Returns:
            None (use user-specified dimensions)
        """
        return None
