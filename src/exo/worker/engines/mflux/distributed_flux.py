from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, Optional

import mlx.core as mx
from mflux.callbacks.callbacks import Callbacks
from mflux.config.config import Config
from mflux.config.model_config import ModelConfig
from mflux.config.runtime_config import RuntimeConfig
from mflux.models.common.latent_creator.latent_creator import Img2Img, LatentCreator
from mflux.models.flux.latent_creator.flux_latent_creator import FluxLatentCreator
from mflux.models.flux.model.flux_text_encoder.prompt_encoder import PromptEncoder
from mflux.models.flux.variants.txt2img.flux import Flux1
from mflux.utils.array_util import ArrayUtil
from mflux.utils.exceptions import StopImageGenerationException
from mflux.utils.image_util import ImageUtil
from PIL import Image
from tqdm import tqdm

from exo.shared.types.worker.instances import BoundInstance
from exo.shared.types.worker.shards import PipelineShardMetadata
from exo.worker.download.download_utils import build_model_path
from exo.worker.engines.mflux.pipefusion.distributed_denoising import (
    DistributedDenoising,
)
from exo.worker.engines.mlx.utils_mlx import mlx_distributed_init, mx_barrier
from exo.worker.runner.bootstrap import logger


class DistributedFlux1:
    """
    Wrapper for Flux1 that attaches distributed group and shard metadata.

    This wrapper enables the generation runtime to access distributed context
    (group, rank, world_size, shard boundaries).
    """

    __slots__ = ("_model", "_group", "_shard_metadata", "_variant")

    _model: Flux1
    _group: Optional[mx.distributed.Group]
    _shard_metadata: PipelineShardMetadata
    _variant: str

    _steps_map: dict[str, dict[str, int]] = {
        "schnell": {"low": 1, "medium": 2, "high": 4},
        "dev": {"low": 10, "medium": 25, "high": 50},
    }

    def __init__(
        self,
        model_id: str,
        local_path: Path,
        shard_metadata: PipelineShardMetadata,
        group: Optional[mx.distributed.Group] = None,
        quantize: int | None = None,
    ):
        """
        Initialize DistributedFlux1 directly from configuration.

        Args:
            model_id: The model identifier (e.g., "black-forest-labs/FLUX.1-schnell")
            local_path: Path to the local model weights
            shard_metadata: Pipeline shard metadata with layer assignments
            group: MLX distributed group for multi-node coordination (None for single-node)
            quantize: Optional quantization bit width
        """
        variant = model_id.split("-")[-1]
        model = Flux1(
            model_config=ModelConfig.from_name(model_name=model_id, base_model=None),
            local_path=str(local_path),
            quantize=quantize,
        )

        if group is not None:
            # Apply pipeline parallelism by wrapping the transformer
            model.transformer = DistributedDenoising(  # type: ignore[assignment]
                transformer=model.transformer,
                group=group,
                shard_metadata=shard_metadata,
            )
            logger.info("Applied pipefusion transformations")

            mx.eval(model.parameters())

            # TODO: Do we need this?
            mx.eval(model)

            # Synchronize processes before generation to avoid timeout
            mx_barrier(group)
            logger.info(f"Flux transformer sharded for rank {group.rank()}")
        else:
            logger.info("Single-node Flux initialization")

        object.__setattr__(self, "_model", model)
        object.__setattr__(self, "_group", group)
        object.__setattr__(self, "_shard_metadata", shard_metadata)
        object.__setattr__(self, "_variant", variant)

    @classmethod
    def from_bound_instance(cls, bound_instance: BoundInstance) -> "DistributedFlux1":
        """
        Create DistributedFlux1 from a BoundInstance.

        This factory method extracts model configuration from the bound instance
        and handles distributed initialization if needed.

        Args:
            bound_instance: The bound instance containing model and shard info

        Returns:
            Initialized DistributedFlux1 ready for inference
        """
        model_id = bound_instance.bound_shard.model_meta.model_id
        model_path = build_model_path(model_id)

        shard_metadata = bound_instance.bound_shard
        if not isinstance(shard_metadata, PipelineShardMetadata):
            raise ValueError("Expected PipelineShardMetadata for Flux")

        is_distributed = (
            len(bound_instance.instance.shard_assignments.node_to_runner) > 1
        )

        if is_distributed:
            logger.info("Starting distributed init for Flux")
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
    def model(self) -> Flux1:
        """The underlying Flux1 model."""
        return self._model

    @property
    def group(self) -> Optional[mx.distributed.Group]:
        """The MLX distributed group for this model (None for single-node)."""
        return self._group

    @property
    def shard_metadata(self) -> PipelineShardMetadata:
        """Shard metadata containing layer assignments and device info."""
        return self._shard_metadata

    @property
    def rank(self) -> int:
        """This device's rank in the distributed group."""
        return self._shard_metadata.device_rank

    @property
    def world_size(self) -> int:
        """Total number of devices in the distributed group."""
        return self._shard_metadata.world_size

    @property
    def is_first_stage(self) -> bool:
        """True if this device is the first stage in the pipeline."""
        return self._shard_metadata.device_rank == 0

    @property
    def is_last_stage(self) -> bool:
        """True if this device is the last stage in the pipeline."""
        return self._shard_metadata.device_rank == self._shard_metadata.world_size - 1

    @property
    def is_distributed(self) -> bool:
        """True if running in distributed mode (world_size > 1)."""
        return self._shard_metadata.world_size > 1

    # Delegate attribute access to the underlying model.
    # Guarded with TYPE_CHECKING to prevent type checker complaints
    # while still providing full delegation at runtime.
    if not TYPE_CHECKING:

        def __getattr__(self, name: str) -> Any:
            return getattr(self._model, name)

        def __setattr__(self, name: str, value: Any) -> None:
            if name in ("_model", "_group", "_shard_metadata"):
                object.__setattr__(self, name, value)
            else:
                setattr(self._model, name, value)

    def generate(
        self,
        prompt: str,
        height: int,
        width: int,
        quality: Literal["low", "medium", "high"] = "medium",
        seed: int = 2,
    ) -> Optional[Image.Image]:
        """
        Generate an image using the Flux1 model.

        For distributed inference, only the first stage (rank 0) returns the image.
        Other stages return None after participating in the pipeline.

        Args:
            prompt: Text description of the image to generate
            height: Image height in pixels
            width: Image width in pixels
            quality: Generation quality ("low", "medium", "high")
            seed: Random seed for reproducibility

        Returns:
            Generated PIL Image (rank 0) or None (other ranks)
        """
        # Determine number of inference steps based on quality
        steps = self._steps_map[self._variant][quality]

        config = Config(num_inference_steps=steps, height=height, width=width)
        image = self._generate_image(settings=config, prompt=prompt, seed=seed)
        logger.info("generated image")

        # Only rank 0 returns the actual image
        if self.is_first_stage:
            return image.image

    def _generate_image(self, settings: Config, prompt: str, seed: int) -> Any:
        """
        Internal image generation with the diffusion loop.

        This method implements the core diffusion process with distributed
        communication handled at the transformer level (via block wrappers
        for now, will be moved to loop level for async pipeline).
        """
        model = self._model

        # 0. Create runtime config
        config = RuntimeConfig(settings, model.model_config)
        time_steps = tqdm(range(config.init_time_step, config.num_inference_steps))

        # 1. Create initial latents (all nodes create the same latents with same seed)
        latents = LatentCreator.create_for_txt2img_or_img2img(
            seed=seed,
            height=config.height,
            width=config.width,
            img2img=Img2Img(
                vae=model.vae,
                latent_creator=FluxLatentCreator,
                image_path=config.image_path,
                sigmas=config.scheduler.sigmas,
                init_time_step=config.init_time_step,
            ),
        )

        # 2. Encode the prompt (all nodes encode to get consistent embeddings)
        prompt_embeds, pooled_prompt_embeds = PromptEncoder.encode_prompt(
            prompt=prompt,
            prompt_cache=model.prompt_cache,
            t5_tokenizer=model.t5_tokenizer,
            clip_tokenizer=model.clip_tokenizer,
            t5_text_encoder=model.t5_text_encoder,
            clip_text_encoder=model.clip_text_encoder,
        )

        # (Optional) Call subscribers for beginning of loop
        Callbacks.before_loop(
            seed=seed,
            prompt=prompt,
            latents=latents,
            config=config,
        )

        # 3. Main diffusion loop
        for t in time_steps:
            try:
                latents = self._diffusion_step(
                    t=t,
                    config=config,
                    latents=latents,
                    prompt_embeds=prompt_embeds,
                    pooled_prompt_embeds=pooled_prompt_embeds,
                )

                # (Optional) Call subscribers in-loop
                Callbacks.in_loop(
                    t=t,
                    seed=seed,
                    prompt=prompt,
                    latents=latents,
                    config=config,
                    time_steps=time_steps,
                )

                mx.eval(latents)

            except KeyboardInterrupt:  # noqa: PERF203
                Callbacks.interruption(
                    t=t,
                    seed=seed,
                    prompt=prompt,
                    latents=latents,
                    config=config,
                    time_steps=time_steps,
                )
                raise StopImageGenerationException(
                    f"Stopping image generation at step {t + 1}/{len(time_steps)}"
                ) from None

        # (Optional) Call subscribers after loop
        Callbacks.after_loop(
            seed=seed,
            prompt=prompt,
            latents=latents,
            config=config,
        )

        # 4. Decode latents to image (all nodes decode for now)
        latents = ArrayUtil.unpack_latents(
            latents=latents, height=config.height, width=config.width
        )
        decoded = model.vae.decode(latents)
        return ImageUtil.to_image(
            decoded_latents=decoded,
            config=config,
            seed=seed,
            prompt=prompt,
            quantization=model.bits,
            lora_paths=model.lora_paths,
            lora_scales=model.lora_scales,
            image_path=config.image_path,
            image_strength=config.image_strength,
            generation_time=time_steps.format_dict["elapsed"],
        )

    def _diffusion_step(
        self,
        t: int,
        config: RuntimeConfig,
        latents: mx.array,
        prompt_embeds: mx.array,
        pooled_prompt_embeds: mx.array,
    ) -> mx.array:
        if self._group is None:
            latents = config.scheduler.scale_model_input(latents, t)

        noise = self._model.transformer(
            t=t,
            config=config,
            hidden_states=latents,
            prompt_embeds=prompt_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds,
        )

        if self._group is None:
            latents = config.scheduler.step(
                model_output=noise,
                timestep=t,
                sample=latents,
            )
        else:
            latents = noise

        return latents
