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
from exo.worker.engines.mflux.config import get_config_for_model
from exo.worker.engines.mflux.config.model_config import ImageModelConfig
from exo.worker.engines.mflux.pipefusion import get_adapter_for_model
from exo.worker.engines.mflux.pipefusion.adapter import ModelAdapter
from exo.worker.engines.mflux.pipefusion.distributed_denoising import (
    DistributedDenoising,
)
from exo.worker.engines.mlx.utils_mlx import mlx_distributed_init, mx_barrier
from exo.worker.runner.bootstrap import logger


class DistributedImageModel:
    """
    Model-agnostic wrapper for distributed image generation.

    This wrapper enables the generation runtime to access distributed context
    (group, rank, world_size, shard boundaries) and works with any mflux model
    (Flux, Fibo, Qwen, etc.) through configuration and adapters.
    """

    __slots__ = ("_model", "_config", "_adapter", "_group", "_shard_metadata")

    _model: Flux1  # Will be generalized to support other model types
    _config: ImageModelConfig
    _adapter: ModelAdapter
    _group: Optional[mx.distributed.Group]
    _shard_metadata: PipelineShardMetadata

    def __init__(
        self,
        model_id: str,
        local_path: Path,
        shard_metadata: PipelineShardMetadata,
        group: Optional[mx.distributed.Group] = None,
        quantize: int | None = None,
    ):
        """
        Initialize DistributedImageModel directly from configuration.

        Args:
            model_id: The model identifier (e.g., "black-forest-labs/FLUX.1-schnell")
            local_path: Path to the local model weights
            shard_metadata: Pipeline shard metadata with layer assignments
            group: MLX distributed group for multi-node coordination (None for single-node)
            quantize: Optional quantization bit width
        """
        # Get model config and adapter
        config = get_config_for_model(model_id)
        adapter = get_adapter_for_model(config)

        # Create the appropriate mflux model based on family
        if config.model_family == "flux":
            model = Flux1(
                model_config=ModelConfig.from_name(
                    model_name=model_id, base_model=None
                ),
                local_path=str(local_path),
                quantize=quantize,
            )
        else:
            raise ValueError(f"Unsupported model family: {config.model_family}")

        if group is not None:
            # Apply pipeline parallelism by wrapping the transformer
            num_sync_steps = config.get_num_sync_steps("medium")
            model.transformer = DistributedDenoising(
                transformer=model.transformer,
                config=config,
                adapter=adapter,
                group=group,
                shard_metadata=shard_metadata,
                num_sync_steps=num_sync_steps,
            )
            logger.info("Applied pipefusion transformations")

            mx.eval(model.parameters())

            # TODO: Do we need this?
            mx.eval(model)

            # Synchronize processes before generation to avoid timeout
            mx_barrier(group)
            logger.info(f"Transformer sharded for rank {group.rank()}")
        else:
            logger.info("Single-node initialization")

        object.__setattr__(self, "_model", model)
        object.__setattr__(self, "_config", config)
        object.__setattr__(self, "_adapter", adapter)
        object.__setattr__(self, "_group", group)
        object.__setattr__(self, "_shard_metadata", shard_metadata)

    @classmethod
    def from_bound_instance(
        cls, bound_instance: BoundInstance
    ) -> "DistributedImageModel":
        """
        Create DistributedImageModel from a BoundInstance.

        This factory method extracts model configuration from the bound instance
        and handles distributed initialization if needed.

        Args:
            bound_instance: The bound instance containing model and shard info

        Returns:
            Initialized DistributedImageModel ready for inference
        """
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
    def model(self) -> Flux1:
        """The underlying mflux model."""
        return self._model

    @property
    def config(self) -> ImageModelConfig:
        """The model configuration."""
        return self._config

    @property
    def adapter(self) -> ModelAdapter:
        """The model adapter for model-specific operations."""
        return self._adapter

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
            if name in ("_model", "_config", "_adapter", "_group", "_shard_metadata"):
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
        Generate an image using the model.

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
        steps = self._config.get_steps_for_quality(quality)

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
        communication handled at the transformer level.
        """
        model = self._model

        # 0. Create runtime config
        runtime_config = RuntimeConfig(settings, model.model_config)
        time_steps = tqdm(
            range(runtime_config.init_time_step, runtime_config.num_inference_steps)
        )

        # 1. Create initial latents (all nodes create the same latents with same seed)
        latents = LatentCreator.create_for_txt2img_or_img2img(
            seed=seed,
            height=runtime_config.height,
            width=runtime_config.width,
            img2img=Img2Img(
                vae=model.vae,
                latent_creator=FluxLatentCreator,
                image_path=runtime_config.image_path,
                sigmas=runtime_config.scheduler.sigmas,
                init_time_step=runtime_config.init_time_step,
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
            config=runtime_config,
        )

        # 3. Main diffusion loop
        for t in time_steps:
            try:
                latents = self._diffusion_step(
                    t=t,
                    runtime_config=runtime_config,
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
                    config=runtime_config,
                    time_steps=time_steps,
                )

                mx.eval(latents)

            except KeyboardInterrupt:  # noqa: PERF203
                Callbacks.interruption(
                    t=t,
                    seed=seed,
                    prompt=prompt,
                    latents=latents,
                    config=runtime_config,
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
            config=runtime_config,
        )

        # 4. Decode latents to image (all nodes decode for now)
        latents = ArrayUtil.unpack_latents(
            latents=latents, height=runtime_config.height, width=runtime_config.width
        )
        decoded = model.vae.decode(latents)
        return ImageUtil.to_image(
            decoded_latents=decoded,
            config=runtime_config,
            seed=seed,
            prompt=prompt,
            quantization=model.bits,
            lora_paths=model.lora_paths,
            lora_scales=model.lora_scales,
            image_path=runtime_config.image_path,
            image_strength=runtime_config.image_strength,
            generation_time=time_steps.format_dict["elapsed"],
        )

    def _diffusion_step(
        self,
        t: int,
        runtime_config: RuntimeConfig,
        latents: mx.array,
        prompt_embeds: mx.array,
        pooled_prompt_embeds: mx.array,
    ) -> mx.array:
        if self._group is None:
            latents = runtime_config.scheduler.scale_model_input(latents, t)

        noise = self._model.transformer(
            t=t,
            config=runtime_config,
            hidden_states=latents,
            prompt_embeds=prompt_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds,
        )

        if self._group is None:
            latents = runtime_config.scheduler.step(
                model_output=noise,
                timestep=t,
                sample=latents,
            )
        else:
            latents = noise

        return latents
