from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, Optional

import mlx.core as mx
from mflux.config.config import Config
from mflux.config.runtime_config import RuntimeConfig
from mflux.models.common.latent_creator.latent_creator import Img2Img, LatentCreator
from mflux.models.flux.latent_creator.flux_latent_creator import FluxLatentCreator
from mflux.models.flux.model.flux_text_encoder.prompt_encoder import PromptEncoder
from mflux.models.flux.variants.txt2img.flux import Flux1
from mflux.utils.array_util import ArrayUtil
from mflux.utils.image_util import ImageUtil
from PIL import Image

from exo.shared.types.worker.instances import BoundInstance
from exo.shared.types.worker.shards import PipelineShardMetadata
from exo.worker.download.download_utils import build_model_path
from exo.worker.engines.mflux.config import get_config_for_model
from exo.worker.engines.mflux.config.model_config import ImageModelConfig
from exo.worker.engines.mflux.pipefusion import create_adapter_for_model
from exo.worker.engines.mflux.pipefusion.adapter import ModelAdapter
from exo.worker.engines.mflux.pipefusion.diffusion_runner import DiffusionRunner
from exo.worker.engines.mlx.utils_mlx import mlx_distributed_init, mx_barrier
from exo.worker.runner.bootstrap import logger


class DistributedImageModel:
    __slots__ = (
        "_model",
        "_config",
        "_adapter",
        "_group",
        "_shard_metadata",
        "_runner",
    )

    _model: Flux1  # Will be generalized to support other model types
    _config: ImageModelConfig
    _adapter: ModelAdapter
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

        # Get model from adapter
        model = adapter.model

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
    def model(self) -> Flux1:
        return self._model

    @property
    def config(self) -> ImageModelConfig:
        return self._config

    @property
    def adapter(self) -> ModelAdapter:
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

    # Delegate attribute access to the underlying model.
    # Guarded with TYPE_CHECKING to prevent type checker complaints
    # while still providing full delegation at runtime.
    if not TYPE_CHECKING:

        def __getattr__(self, name: str) -> Any:
            return getattr(self._model, name)

        def __setattr__(self, name: str, value: Any) -> None:
            if name in (
                "_model",
                "_config",
                "_adapter",
                "_group",
                "_shard_metadata",
                "_runner",
            ):
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
        # Determine number of inference steps based on quality
        steps = self._config.get_steps_for_quality(quality)

        config = Config(num_inference_steps=steps, height=height, width=width)
        image = self._generate_image(settings=config, prompt=prompt, seed=seed)
        logger.info("generated image")

        # Only rank 0 returns the actual image
        if self.is_first_stage:
            return image.image

    def _generate_image(self, settings: Config, prompt: str, seed: int) -> Any:
        model = self._model

        # Create runtime config
        runtime_config = RuntimeConfig(settings, model.model_config)

        # Create initial latents (all nodes create the same latents with same seed)
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

        # Encode the prompt (all nodes encode to get consistent embeddings)
        prompt_embeds, pooled_prompt_embeds = PromptEncoder.encode_prompt(
            prompt=prompt,
            prompt_cache=model.prompt_cache,
            t5_tokenizer=model.t5_tokenizer,
            clip_tokenizer=model.clip_tokenizer,
            t5_text_encoder=model.t5_text_encoder,
            clip_text_encoder=model.clip_text_encoder,
        )

        # Run the diffusion loop (runner handles callbacks internally)
        latents = self._runner.run(
            latents=latents,
            prompt_embeds=prompt_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds,
            runtime_config=runtime_config,
            seed=seed,
            prompt=prompt,
        )

        # Decode latents to image (all nodes decode for now)
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
            generation_time=0,  # TODO: Track time in runner if needed
        )
