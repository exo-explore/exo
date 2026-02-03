from abc import ABC, abstractmethod
from pathlib import Path
from typing import TYPE_CHECKING, Any, Generic, TypeVar

import mlx.core as mx
from mflux.models.common.config.config import Config
from mflux.models.common.latent_creator.latent_creator import Img2Img, LatentCreator
from mflux.utils.image_util import ImageUtil
from PIL import Image

from exo.worker.engines.image.config import ImageModelConfig

if TYPE_CHECKING:
    from exo.worker.engines.image.pipeline.block_wrapper import (
        JointBlockWrapper,
        SingleBlockWrapper,
    )

ModelT = TypeVar("ModelT")
TransformerT = TypeVar("TransformerT")

RotaryEmbeddings = mx.array | tuple[mx.array, mx.array]


class PromptData(ABC):
    @property
    @abstractmethod
    def prompt_embeds(self) -> mx.array: ...

    @property
    @abstractmethod
    def pooled_prompt_embeds(self) -> mx.array: ...

    @property
    @abstractmethod
    def negative_prompt_embeds(self) -> mx.array | None: ...

    @property
    @abstractmethod
    def negative_pooled_prompt_embeds(self) -> mx.array | None: ...

    @abstractmethod
    def get_encoder_hidden_states_mask(
        self, positive: bool = True
    ) -> mx.array | None: ...

    @property
    @abstractmethod
    def cond_image_grid(
        self,
    ) -> tuple[int, int, int] | list[tuple[int, int, int]] | None:
        """Conditioning image grid dimensions for edit mode.

        Returns:
            Grid dimensions (edit) or None (standard generation).
        """
        ...

    @property
    @abstractmethod
    def conditioning_latents(self) -> mx.array | None:
        """Conditioning latents for edit mode.

        Returns:
            Conditioning latents array for image editing, None for standard generation.
        """
        ...

    @abstractmethod
    def get_batched_cfg_data(
        self,
    ) -> tuple[mx.array, mx.array, mx.array | None, mx.array | None] | None:
        """Get embeddings for CFG with batch_size=2.

        Combines positive and negative embeddings into batched tensors for
        a single forward pass. Pads shorter sequences to max length. Attention
        mask is used to mask padding.

        Returns:
            None if model doesn't support CFG, otherwise tuple of:
            - batched_embeds: [2, max_seq, hidden] (positive then negative)
            - batched_mask: [2, max_seq] attention mask
            - batched_pooled: [2, hidden] pooled embeddings or None
            - conditioning_latents: [2, latent_seq, latent_dim] or None
            TODO(ciaran): type this
        """
        ...


class ModelAdapter(ABC, Generic[ModelT, TransformerT]):
    _config: ImageModelConfig
    _model: ModelT
    _transformer: TransformerT

    @property
    def config(self) -> ImageModelConfig:
        return self._config

    @property
    def model(self) -> ModelT:
        return self._model

    @property
    def transformer(self) -> TransformerT:
        return self._transformer

    @property
    @abstractmethod
    def hidden_dim(self) -> int: ...

    @property
    @abstractmethod
    def needs_cfg(self) -> bool:
        """Whether this model uses classifier-free guidance."""
        ...

    @abstractmethod
    def _get_latent_creator(self) -> type: ...

    @abstractmethod
    def get_joint_block_wrappers(
        self,
        text_seq_len: int,
        encoder_hidden_states_mask: mx.array | None = None,
    ) -> list["JointBlockWrapper[Any]"]:
        """Create wrapped joint transformer blocks with pipefusion support.

        Args:
            text_seq_len: Number of text tokens (constant for generation)
            encoder_hidden_states_mask: Attention mask for text (Qwen only)

        Returns:
            List of wrapped joint blocks ready for pipefusion
        """
        ...

    @abstractmethod
    def get_single_block_wrappers(
        self,
        text_seq_len: int,
    ) -> list["SingleBlockWrapper[Any]"]:
        """Create wrapped single transformer blocks with pipefusion support.

        Args:
            text_seq_len: Number of text tokens (constant for generation)

        Returns:
            List of wrapped single blocks ready for pipefusion
        """
        ...

    @abstractmethod
    def slice_transformer_blocks(
        self,
        start_layer: int,
        end_layer: int,
    ):
        """Remove transformer blocks outside the assigned range.

        This should be called BEFORE mx.eval() to avoid loading unused weights
        in distributed mode.

        Args:
            start_layer: First layer index (inclusive) assigned to this node
            end_layer: Last layer index (exclusive) assigned to this node
        """
        ...

    def set_image_dimensions(self, image_path: Path) -> tuple[int, int] | None:
        """Default implementation: no dimension computation needed.

        Override in edit adapters to compute dimensions from input image.
        TODO(ciaran): this is a hack

        Returns:
            None (use user-specified dimensions)
        """
        return None

    def create_latents(self, seed: int, runtime_config: Config) -> mx.array:
        """Create initial latents. Uses model-specific latent creator."""
        model: Any = self.model
        return LatentCreator.create_for_txt2img_or_img2img(
            seed=seed,
            height=runtime_config.height,
            width=runtime_config.width,
            img2img=Img2Img(
                vae=model.vae,  # pyright: ignore[reportAny]
                latent_creator=self._get_latent_creator(),
                sigmas=runtime_config.scheduler.sigmas,  # pyright: ignore[reportAny]
                init_time_step=runtime_config.init_time_step,
                image_path=runtime_config.image_path,
            ),
        )

    def decode_latents(
        self,
        latents: mx.array,
        runtime_config: Config,
        seed: int,
        prompt: str,
    ) -> Image.Image:
        model: Any = self.model  # Allow attribute access on model
        latents = self._get_latent_creator().unpack_latents(  # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType]
            latents=latents,
            height=runtime_config.height,
            width=runtime_config.width,
        )
        decoded = model.vae.decode(latents)  # pyright: ignore[reportAny]
        # TODO(ciaran):
        # from mflux.models.common.vae.vae_util import VAEUtil
        # VAEUtil.decode(vae=model.vae, latents=latents, tiling_config=self.tiling_config)
        generated_image = ImageUtil.to_image(
            decoded_latents=decoded,  # pyright: ignore[reportAny]
            config=runtime_config,
            seed=seed,
            prompt=prompt,
            quantization=model.bits,  # pyright: ignore[reportAny]
            lora_paths=model.lora_paths,  # pyright: ignore[reportAny]
            lora_scales=model.lora_scales,  # pyright: ignore[reportAny]
            image_path=runtime_config.image_path,
            image_strength=runtime_config.image_strength,
            generation_time=0,
        )
        return generated_image.image

    @abstractmethod
    def encode_prompt(
        self, prompt: str, negative_prompt: str | None = None
    ) -> "PromptData": ...

    @abstractmethod
    def compute_embeddings(
        self,
        hidden_states: mx.array,
        prompt_embeds: mx.array,
    ) -> tuple[mx.array, mx.array]: ...

    @abstractmethod
    def compute_text_embeddings(
        self,
        t: int,
        runtime_config: Config,
        pooled_prompt_embeds: mx.array | None = None,
        hidden_states: mx.array | None = None,
    ) -> mx.array: ...

    @abstractmethod
    def compute_rotary_embeddings(
        self,
        prompt_embeds: mx.array,
        runtime_config: Config,
        encoder_hidden_states_mask: mx.array | None = None,
        cond_image_grid: tuple[int, int, int]
        | list[tuple[int, int, int]]
        | None = None,
        kontext_image_ids: mx.array | None = None,
    ) -> RotaryEmbeddings: ...

    def merge_streams(
        self,
        hidden_states: mx.array,
        encoder_hidden_states: mx.array,
    ) -> mx.array:
        return mx.concatenate([encoder_hidden_states, hidden_states], axis=1)

    @abstractmethod
    def apply_guidance(
        self,
        noise_positive: mx.array,
        noise_negative: mx.array,
        guidance_scale: float,
    ) -> mx.array:
        """Apply classifier-free guidance to combine positive/negative predictions.

        Only called when needs_cfg is True.

        Args:
            noise_positive: Noise prediction from positive prompt
            noise_negative: Noise prediction from negative prompt
            guidance_scale: Guidance strength

        Returns:
            Guided noise prediction
        """
        ...

    def final_projection(
        self,
        hidden_states: mx.array,
        text_embeddings: mx.array,
    ) -> mx.array:
        transformer: Any = self.transformer
        hidden_states = transformer.norm_out(hidden_states, text_embeddings)  # pyright: ignore[reportAny]
        return transformer.proj_out(hidden_states)  # pyright: ignore[reportAny]
