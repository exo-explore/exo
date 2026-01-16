from abc import ABC, abstractmethod
from pathlib import Path
from typing import TYPE_CHECKING, Any

import mlx.core as mx
from mflux.models.common.config.config import Config
from mflux.models.common.latent_creator.latent_creator import Img2Img, LatentCreator
from mflux.utils.image_util import ImageUtil

from exo.worker.engines.image.config import ImageModelConfig

if TYPE_CHECKING:
    from exo.worker.engines.image.pipeline.adapter import (
        BlockWrapperMode,
        JointBlockInterface,
        SingleBlockInterface,
    )
    from exo.worker.engines.image.pipeline.kv_cache import ImagePatchKVCache


class PromptData(ABC):
    """Abstract base class for encoded prompt data.

    All adapters must return prompt data that inherits from this class.
    Model-specific prompt data classes can add additional attributes
    (e.g., attention masks for Qwen).
    """

    @property
    @abstractmethod
    def prompt_embeds(self) -> mx.array:
        """Text embeddings from encoder."""
        ...

    @property
    @abstractmethod
    def pooled_prompt_embeds(self) -> mx.array:
        """Pooled text embeddings (for Flux) or placeholder (for Qwen)."""
        ...

    @property
    @abstractmethod
    def negative_prompt_embeds(self) -> mx.array | None:
        """Negative prompt embeddings for CFG (None if not using CFG)."""
        ...

    @property
    @abstractmethod
    def negative_pooled_prompt_embeds(self) -> mx.array | None:
        """Negative pooled embeddings for CFG (None if not using CFG)."""
        ...

    @abstractmethod
    def get_encoder_hidden_states_mask(self, positive: bool = True) -> mx.array | None:
        """Get encoder hidden states mask for attention.

        Args:
            positive: If True, return mask for positive prompt pass.
                     If False, return mask for negative prompt pass.

        Returns:
            Attention mask array (Qwen) or None (Flux).
        """
        ...

    @property
    @abstractmethod
    def cond_image_grid(
        self,
    ) -> tuple[int, int, int] | list[tuple[int, int, int]] | None:
        """Conditioning image grid dimensions for edit mode.

        Returns:
            Grid dimensions (Qwen edit) or None (standard generation).
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


class ModelAdapter(ABC):
    """Base class for model adapters with shared utilities."""

    _config: ImageModelConfig
    _model: Any
    _transformer: Any

    @property
    def config(self) -> ImageModelConfig:
        return self._config

    @property
    def model(self) -> Any:
        return self._model

    @property
    def transformer(self) -> Any:
        return self._transformer

    @property
    @abstractmethod
    def hidden_dim(self) -> int:
        """Return the size of hidden_dim."""
        ...

    @property
    @abstractmethod
    def needs_cfg(self) -> bool:
        """Whether this model uses classifier-free guidance.

        Returns:
            True if model requires two forward passes with guidance (e.g., Qwen)
            False if model uses a single forward pass (e.g., Flux)
        """
        ...

    @abstractmethod
    def _get_latent_creator(self) -> type:
        """Return the latent creator class for this model."""
        ...

    @abstractmethod
    def get_joint_blocks(self) -> list["JointBlockInterface"]:
        """Get the list of joint transformer blocks from the model."""
        ...

    @abstractmethod
    def get_single_blocks(self) -> list["SingleBlockInterface"]:
        """Get the list of single transformer blocks from the model."""
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

        Returns:
            None (use user-specified dimensions)
        """
        return None

    def create_latents(self, seed: int, runtime_config: Config) -> mx.array:
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
        runtime_config: Config,
        seed: int,
        prompt: str,
    ) -> Any:
        """Decode latents to image. Shared implementation."""
        latents = self._get_latent_creator().unpack_latents(
            latents=latents,
            height=runtime_config.height,
            width=runtime_config.width,
        )
        decoded = self.model.vae.decode(latents)
        # TODO(ciaran):
        # from mflux.models.common.vae.vae_util import VAEUtil
        # VAEUtil.decode(vae=self.model.vae, latents=latents, tiling_config=self.tiling_config)
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

    @abstractmethod
    def encode_prompt(self, prompt: str) -> "PromptData":
        """Encode prompt into model-specific prompt data.

        Args:
            prompt: Text prompt

        Returns:
            PromptData containing embeddings (and model-specific extras)
        """
        ...

    @abstractmethod
    def compute_embeddings(
        self,
        hidden_states: mx.array,
        prompt_embeds: mx.array,
    ) -> tuple[mx.array, mx.array]:
        """Compute x_embedder and context_embedder outputs.

        Args:
            hidden_states: Input latent states
            prompt_embeds: Text embeddings from encoder

        Returns:
            Tuple of (embedded_hidden_states, embedded_encoder_states)
        """
        ...

    @abstractmethod
    def compute_text_embeddings(
        self,
        t: int,
        runtime_config: Config,
        pooled_prompt_embeds: mx.array | None = None,
        hidden_states: mx.array | None = None,
    ) -> mx.array:
        """Compute time/text embeddings for conditioning.

        Args:
            t: Current timestep
            runtime_config: Runtime configuration
            pooled_prompt_embeds: Pooled text embeddings (used by Flux)
            hidden_states: Image hidden states

        Returns:
            Text embeddings tensor
        """
        ...

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
    ) -> Any:
        """Compute rotary position embeddings.

        Args:
            prompt_embeds: Text embeddings
            runtime_config: Runtime configuration
            encoder_hidden_states_mask: Attention mask for text (Qwen)
            cond_image_grid: Conditioning image grid dimensions (Qwen edit)
            kontext_image_ids: Kontext image position IDs (Flux)

        Returns:
            Flux: mx.array
            Qwen: tuple[mx.array, mx.array]
        """
        ...

    @abstractmethod
    def apply_joint_block(
        self,
        block: "JointBlockInterface",
        hidden_states: mx.array,
        encoder_hidden_states: mx.array,
        text_embeddings: mx.array,
        rotary_embeddings: Any,
        kv_cache: "ImagePatchKVCache | None",
        mode: "BlockWrapperMode",
        text_seq_len: int,
        patch_start: int | None = None,
        patch_end: int | None = None,
        encoder_hidden_states_mask: mx.array | None = None,
        block_idx: int | None = None,
    ) -> tuple[mx.array, mx.array]:
        """Apply a joint transformer block.

        Args:
            block: The joint transformer block
            hidden_states: Image hidden states
            encoder_hidden_states: Text hidden states
            text_embeddings: Conditioning embeddings
            rotary_embeddings: Rotary position embeddings (format varies by model)
            kv_cache: KV cache (None if not using cache)
            mode: CACHING or PATCHED mode
            text_seq_len: Text sequence length
            patch_start: Start index for patched mode
            patch_end: End index for patched mode
            encoder_hidden_states_mask: Attention mask for text (Qwen)
            block_idx: Block index for debugging (Qwen)

        Returns:
            Tuple of (encoder_hidden_states, hidden_states)
        """
        ...

    def merge_streams(
        self,
        hidden_states: mx.array,
        encoder_hidden_states: mx.array,
    ) -> mx.array:
        return mx.concatenate([encoder_hidden_states, hidden_states], axis=1)

    @abstractmethod
    def apply_single_block(
        self,
        block: "SingleBlockInterface",
        hidden_states: mx.array,
        text_embeddings: mx.array,
        rotary_embeddings: mx.array,
        kv_cache: "ImagePatchKVCache | None",
        mode: "BlockWrapperMode",
        text_seq_len: int,
        patch_start: int | None = None,
        patch_end: int | None = None,
    ) -> mx.array:
        """Apply a single transformer block.

        Args:
            block: The single transformer block
            hidden_states: Concatenated [text + image] hidden states
            text_embeddings: Conditioning embeddings
            rotary_embeddings: Rotary position embeddings
            kv_cache: KV cache (None if not using cache)
            mode: CACHING or PATCHED mode
            text_seq_len: Text sequence length
            patch_start: Start index for patched mode
            patch_end: End index for patched mode

        Returns:
            Output hidden states
        """
        ...

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
        """Apply final norm and projection.

        Args:
            hidden_states: Hidden states (image only, text already removed)
            text_embeddings: Conditioning embeddings

        Returns:
            Projected output
        """
        hidden_states = self._transformer.norm_out(hidden_states, text_embeddings)
        return self._transformer.proj_out(hidden_states)
