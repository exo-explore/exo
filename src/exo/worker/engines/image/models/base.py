from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

import mlx.core as mx
from mflux.config.config import Config
from mflux.config.runtime_config import RuntimeConfig
from mflux.models.common.latent_creator.latent_creator import Img2Img, LatentCreator
from mflux.utils.array_util import ArrayUtil
from mflux.utils.image_util import ImageUtil

if TYPE_CHECKING:
    from exo.worker.engines.image.pipeline.runner import DiffusionRunner


class BaseModelAdapter(ABC):
    """Base class for model adapters with shared generation logic.

    Uses the template method pattern to share common generation flow
    while allowing subclasses to implement model-specific steps.
    """

    def generate_image(
        self,
        settings: Config,
        prompt: str,
        seed: int,
        runner: "DiffusionRunner | None" = None,
    ) -> Any:
        """Generate an image using the template method pattern.

        Args:
            settings: Generation config (steps, height, width)
            prompt: Text prompt
            seed: Random seed
            runner: Optional DiffusionRunner for distributed mode

        Returns:
            GeneratedImage result
        """
        # 1. Create runtime config (shared)
        runtime_config = RuntimeConfig(settings, self.model.model_config)

        # 2. Create initial latents (uses model-specific latent creator)
        latents = self._create_latents(seed, runtime_config)

        # 3. Encode prompt (model-specific)
        prompt_data = self._encode_prompt(prompt)

        # 4. Run denoising loop (model-specific)
        latents = self._run_denoising(latents, prompt_data, runtime_config, runner)

        # 5. Decode and return (shared)
        return self._decode_latents(latents, runtime_config, seed, prompt)

    def _create_latents(self, seed: int, runtime_config: RuntimeConfig) -> mx.array:
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

    def _decode_latents(
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
    def _encode_prompt(self, prompt: str) -> Any:
        """Encode the prompt. Returns model-specific prompt data."""
        ...

    @abstractmethod
    def _run_denoising(
        self,
        latents: mx.array,
        prompt_data: Any,
        runtime_config: RuntimeConfig,
        runner: "DiffusionRunner | None",
    ) -> mx.array:
        """Run the denoising loop. Model-specific implementation."""
        ...
