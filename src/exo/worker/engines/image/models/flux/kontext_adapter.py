import math
from pathlib import Path
from typing import Any, final

import mlx.core as mx
from mflux.models.common.config.config import Config
from mflux.models.common.config.model_config import ModelConfig
from mflux.models.flux.latent_creator.flux_latent_creator import FluxLatentCreator
from mflux.models.flux.model.flux_text_encoder.prompt_encoder import PromptEncoder
from mflux.models.flux.model.flux_transformer.transformer import Transformer
from mflux.models.flux.variants.kontext.flux_kontext import Flux1Kontext
from mflux.models.flux.variants.kontext.kontext_util import KontextUtil

from exo.worker.engines.image.config import ImageModelConfig
from exo.worker.engines.image.models.base import (
    ModelAdapter,
    PromptData,
    RotaryEmbeddings,
)
from exo.worker.engines.image.models.flux.wrappers import (
    FluxJointBlockWrapper,
    FluxSingleBlockWrapper,
)
from exo.worker.engines.image.pipeline.block_wrapper import (
    JointBlockWrapper,
    SingleBlockWrapper,
)


@final
class FluxKontextPromptData(PromptData):
    """Prompt data for FLUX.1-Kontext image editing.

    Stores text embeddings along with conditioning latents and position IDs
    for the input image.
    """

    def __init__(
        self,
        prompt_embeds: mx.array,
        pooled_prompt_embeds: mx.array,
        conditioning_latents: mx.array,
        kontext_image_ids: mx.array,
    ):
        self._prompt_embeds = prompt_embeds
        self._pooled_prompt_embeds = pooled_prompt_embeds
        self._conditioning_latents = conditioning_latents
        self._kontext_image_ids = kontext_image_ids

    @property
    def prompt_embeds(self) -> mx.array:
        return self._prompt_embeds

    @property
    def pooled_prompt_embeds(self) -> mx.array:
        return self._pooled_prompt_embeds

    @property
    def negative_prompt_embeds(self) -> mx.array | None:
        return None

    @property
    def negative_pooled_prompt_embeds(self) -> mx.array | None:
        return None

    def get_encoder_hidden_states_mask(self, positive: bool = True) -> mx.array | None:
        return None

    @property
    def cond_image_grid(
        self,
    ) -> tuple[int, int, int] | list[tuple[int, int, int]] | None:
        return None

    @property
    def conditioning_latents(self) -> mx.array | None:
        """VAE-encoded input image latents for Kontext conditioning."""
        return self._conditioning_latents

    @property
    def kontext_image_ids(self) -> mx.array | None:
        """Position IDs for Kontext conditioning (first_coord=1)."""
        return self._kontext_image_ids

    def get_cfg_branch_data(
        self, positive: bool
    ) -> tuple[mx.array, mx.array | None, mx.array | None, mx.array | None]:
        """Kontext doesn't use CFG, but we return positive data for compatibility."""
        return (
            self._prompt_embeds,
            None,
            self._pooled_prompt_embeds,
            self._conditioning_latents,
        )

    def get_batched_cfg_data(
        self,
    ) -> tuple[mx.array, mx.array, mx.array | None, mx.array | None] | None:
        # Kontext doesn't use CFG
        return None


@final
class FluxKontextModelAdapter(ModelAdapter[Flux1Kontext, Transformer]):
    """Adapter for FLUX.1-Kontext image editing model.

    Key differences from standard FluxModelAdapter:
    - Takes an input image and computes output dimensions from it
    - Creates conditioning latents from the input image via VAE
    - Creates special position IDs (kontext_image_ids) for conditioning tokens
    - Creates pure noise latents (not img2img blending)
    """

    def __init__(
        self,
        config: ImageModelConfig,
        model_id: str,
        local_path: Path,
        quantize: int | None = None,
    ):
        self._config = config
        self._model = Flux1Kontext(
            model_config=ModelConfig.from_name(model_name=model_id, base_model=None),
            model_path=str(local_path),
            quantize=quantize,
        )
        self._transformer = self._model.transformer

        # Stores image path and computed dimensions after set_image_dimensions
        self._image_path: str | None = None
        self._output_height: int | None = None
        self._output_width: int | None = None

    @property
    def hidden_dim(self) -> int:
        return self._transformer.x_embedder.weight.shape[0]  # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType]

    @property
    def needs_cfg(self) -> bool:
        return False

    def _get_latent_creator(self) -> type:
        return FluxLatentCreator

    def get_joint_block_wrappers(
        self,
        text_seq_len: int,
        encoder_hidden_states_mask: mx.array | None = None,
    ) -> list[JointBlockWrapper[Any]]:
        """Create wrapped joint blocks for Flux Kontext."""
        return [
            FluxJointBlockWrapper(block, text_seq_len)
            for block in self._transformer.transformer_blocks
        ]

    def get_single_block_wrappers(
        self,
        text_seq_len: int,
    ) -> list[SingleBlockWrapper[Any]]:
        """Create wrapped single blocks for Flux Kontext."""
        return [
            FluxSingleBlockWrapper(block, text_seq_len)
            for block in self._transformer.single_transformer_blocks
        ]

    def slice_transformer_blocks(
        self,
        start_layer: int,
        end_layer: int,
    ):
        all_joint = list(self._transformer.transformer_blocks)
        all_single = list(self._transformer.single_transformer_blocks)
        total_joint_blocks = len(all_joint)
        if end_layer <= total_joint_blocks:
            # All assigned are joint blocks
            joint_start, joint_end = start_layer, end_layer
            single_start, single_end = 0, 0
        elif start_layer >= total_joint_blocks:
            # All assigned are single blocks
            joint_start, joint_end = 0, 0
            single_start = start_layer - total_joint_blocks
            single_end = end_layer - total_joint_blocks
        else:
            # Spans both joint and single
            joint_start, joint_end = start_layer, total_joint_blocks
            single_start = 0
            single_end = end_layer - total_joint_blocks

        self._transformer.transformer_blocks = all_joint[joint_start:joint_end]
        self._transformer.single_transformer_blocks = all_single[
            single_start:single_end
        ]

    def set_image_dimensions(self, image_path: Path) -> tuple[int, int]:
        """Compute and store dimensions from input image.

        Also stores image_path for use in encode_prompt().

        Args:
            image_path: Path to the input image

        Returns:
            (output_width, output_height) for runtime config
        """
        from mflux.utils.image_util import ImageUtil

        pil_image = ImageUtil.load_image(str(image_path)).convert("RGB")
        image_size = pil_image.size

        # Compute output dimensions from input image aspect ratio
        # Target area of 1024x1024 = ~1M pixels
        target_area = 1024 * 1024
        ratio = image_size[0] / image_size[1]
        output_width = math.sqrt(target_area * ratio)
        output_height = output_width / ratio
        output_width = round(output_width / 32) * 32
        output_height = round(output_height / 32) * 32

        # Ensure multiple of 16 for VAE
        vae_scale_factor = 8
        multiple_of = vae_scale_factor * 2
        output_width = output_width // multiple_of * multiple_of
        output_height = output_height // multiple_of * multiple_of

        self._image_path = str(image_path)
        self._output_width = int(output_width)
        self._output_height = int(output_height)

        return self._output_width, self._output_height

    def create_latents(self, seed: int, runtime_config: Config) -> mx.array:
        """Create initial noise latents for Kontext.

        Unlike standard img2img which blends noise with encoded input,
        Kontext uses pure noise latents. The input image is provided
        separately as conditioning.
        """
        return FluxLatentCreator.create_noise(
            seed=seed,
            height=runtime_config.height,
            width=runtime_config.width,
        )

    def encode_prompt(
        self, prompt: str, negative_prompt: str | None = None
    ) -> FluxKontextPromptData:
        """Encode prompt and create conditioning from stored input image.

        Must call set_image_dimensions() before this method.

        Args:
            prompt: Text prompt for editing
            negative_prompt: Ignored (Kontext doesn't use CFG)

        Returns:
            FluxKontextPromptData with text embeddings and image conditioning
        """
        del negative_prompt  # Kontext doesn't support negative prompts or CFG

        if (
            self._image_path is None
            or self._output_height is None
            or self._output_width is None
        ):
            raise RuntimeError(
                "set_image_dimensions() must be called before encode_prompt() "
                "for FluxKontextModelAdapter"
            )

        assert isinstance(self.model.prompt_cache, dict)
        assert isinstance(self.model.tokenizers, dict)

        # Encode text prompt
        prompt_embeds, pooled_prompt_embeds = PromptEncoder.encode_prompt(
            prompt=prompt,
            prompt_cache=self.model.prompt_cache,
            t5_tokenizer=self.model.tokenizers["t5"],  # pyright: ignore[reportAny]
            clip_tokenizer=self.model.tokenizers["clip"],  # pyright: ignore[reportAny]
            t5_text_encoder=self.model.t5_text_encoder,
            clip_text_encoder=self.model.clip_text_encoder,
        )

        # Create conditioning latents from input image
        conditioning_latents, kontext_image_ids = (
            KontextUtil.create_image_conditioning_latents(
                vae=self.model.vae,
                height=self._output_height,
                width=self._output_width,
                image_path=self._image_path,
            )
        )

        return FluxKontextPromptData(
            prompt_embeds=prompt_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds,
            conditioning_latents=conditioning_latents,
            kontext_image_ids=kontext_image_ids,
        )

    def compute_embeddings(
        self,
        hidden_states: mx.array,
        prompt_embeds: mx.array,
    ) -> tuple[mx.array, mx.array]:
        embedded_hidden = self._transformer.x_embedder(hidden_states)
        embedded_encoder = self._transformer.context_embedder(prompt_embeds)
        return embedded_hidden, embedded_encoder

    def compute_text_embeddings(
        self,
        t: int,
        runtime_config: Config,
        pooled_prompt_embeds: mx.array | None = None,
        hidden_states: mx.array | None = None,
    ) -> mx.array:
        if pooled_prompt_embeds is None:
            raise ValueError(
                "pooled_prompt_embeds is required for Flux Kontext text embeddings"
            )

        return Transformer.compute_text_embeddings(
            t, pooled_prompt_embeds, self._transformer.time_text_embed, runtime_config
        )

    def compute_rotary_embeddings(
        self,
        prompt_embeds: mx.array,
        runtime_config: Config,
        encoder_hidden_states_mask: mx.array | None = None,
        cond_image_grid: tuple[int, int, int]
        | list[tuple[int, int, int]]
        | None = None,
        kontext_image_ids: mx.array | None = None,
    ) -> RotaryEmbeddings:
        return Transformer.compute_rotary_embeddings(
            prompt_embeds,
            self._transformer.pos_embed,
            runtime_config,
            kontext_image_ids,
        )

    def apply_guidance(
        self,
        noise_positive: mx.array,
        noise_negative: mx.array,
        guidance_scale: float,
    ) -> mx.array:
        raise NotImplementedError("Flux Kontext does not use classifier-free guidance")
