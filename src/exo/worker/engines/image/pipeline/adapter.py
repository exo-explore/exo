from enum import Enum
from typing import Any, Protocol

import mlx.core as mx
from mflux.config.runtime_config import RuntimeConfig

from exo.worker.engines.image.config import ImageModelConfig
from exo.worker.engines.image.pipeline.kv_cache import ImagePatchKVCache


class AttentionInterface(Protocol):
    num_heads: int
    head_dimension: int
    to_q: Any
    to_k: Any
    to_v: Any
    norm_q: Any
    norm_k: Any
    to_out: list[Any]


class JointAttentionInterface(AttentionInterface, Protocol):
    add_q_proj: Any
    add_k_proj: Any
    add_v_proj: Any
    norm_added_q: Any
    norm_added_k: Any
    to_add_out: Any


class JointBlockInterface(Protocol):
    attn: JointAttentionInterface
    norm1: Any  # Callable module: (hidden_states, text_embeddings) -> tuple[5 arrays]
    norm1_context: (
        Any  # Callable module: (hidden_states, text_embeddings) -> tuple[5 arrays]
    )
    norm2: Any
    norm2_context: Any
    ff: Any
    ff_context: Any


class SingleBlockInterface(Protocol):
    attn: AttentionInterface
    norm: Any  # Callable module: (hidden_states, text_embeddings) -> tuple[2 arrays]

    def _apply_feed_forward_and_projection(
        self, norm_hidden_states: mx.array, attn_output: mx.array, gate: mx.array
    ) -> mx.array:
        """Apply feed forward network and projection."""
        ...


class BlockWrapperMode(Enum):
    CACHING = "caching"  # Sync mode: compute full attention, populate cache
    PATCHED = "patched"  # Async mode: compute patch attention, use cached KV


class PromptData(Protocol):
    """Protocol for encoded prompt data.

    All adapters must return prompt data that conforms to this protocol.
    Model-specific prompt data classes can add additional attributes
    (e.g., attention masks for Qwen).
    """

    @property
    def prompt_embeds(self) -> mx.array:
        """Text embeddings from encoder."""
        ...

    @property
    def pooled_prompt_embeds(self) -> mx.array:
        """Pooled text embeddings (for Flux) or placeholder (for Qwen)."""
        ...

    @property
    def negative_prompt_embeds(self) -> mx.array | None:
        """Negative prompt embeddings for CFG (None if not using CFG)."""
        ...

    @property
    def negative_pooled_prompt_embeds(self) -> mx.array | None:
        """Negative pooled embeddings for CFG (None if not using CFG)."""
        ...

    def get_extra_forward_kwargs(self, positive: bool = True) -> dict[str, Any]:
        """Return model-specific kwargs for forward pass.

        Args:
            positive: If True, return kwargs for positive prompt pass.
                     If False, return kwargs for negative prompt pass.

        Returns:
            Dict of extra kwargs (e.g., {"encoder_hidden_states_mask": ...} for Qwen)
        """
        ...


class ModelAdapter(Protocol):
    @property
    def config(self) -> ImageModelConfig:
        """Return the model configuration."""
        ...

    @property
    def model(self) -> Any:
        """Return the underlying mflux model instance (e.g., Flux1, Fibo, Qwen)."""
        ...

    @property
    def transformer(self) -> Any:
        """Return the transformer component of the model."""
        ...

    @property
    def hidden_dim(self) -> int:
        """Return the hidden dimension of the transformer."""
        ...

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

    def compute_text_embeddings(
        self,
        t: int,
        runtime_config: RuntimeConfig,
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

    def compute_rotary_embeddings(
        self,
        prompt_embeds: mx.array,
        runtime_config: RuntimeConfig,
        **kwargs: Any,
    ) -> Any:
        """Compute rotary position embeddings.

        Args:
            prompt_embeds: Text embeddings
            runtime_config: Runtime configuration
            **kwargs: Model-specific arguments (e.g., encoder_hidden_states_mask for Qwen)

        Returns:
            Flux: mx.array
            Qwen: tuple[tuple[mx.array, mx.array], tuple[mx.array, mx.array]]
        """
        ...

    def apply_joint_block(
        self,
        block: JointBlockInterface,
        hidden_states: mx.array,
        encoder_hidden_states: mx.array,
        text_embeddings: mx.array,
        rotary_embeddings: Any,  # Format varies: mx.array (Flux) or nested tuple (Qwen)
        kv_cache: ImagePatchKVCache | None,
        mode: "BlockWrapperMode",
        text_seq_len: int,
        patch_start: int | None = None,
        patch_end: int | None = None,
        **kwargs: Any,
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
            **kwargs: Additional model-specific arguments (e.g., encoder_hidden_states_mask,
                block_idx for Qwen)

        Returns:
            Tuple of (encoder_hidden_states, hidden_states)
        """
        ...

    def apply_single_block(
        self,
        block: SingleBlockInterface,
        hidden_states: mx.array,
        text_embeddings: mx.array,
        rotary_embeddings: mx.array,
        kv_cache: ImagePatchKVCache | None,
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
        ...

    def get_joint_blocks(self) -> list[JointBlockInterface]:
        """Get the list of joint transformer blocks from the model."""
        ...

    def get_single_blocks(self) -> list[SingleBlockInterface]:
        """Get the list of single transformer blocks from the model."""
        ...

    def merge_streams(
        self,
        hidden_states: mx.array,
        encoder_hidden_states: mx.array,
    ) -> mx.array:
        """Merge image and text streams for transition to single blocks.

        This is called at the transition point from joint blocks (which process
        image and text separately) to single blocks (which process them
        together). Override to customize the merge strategy.

        Args:
            hidden_states: Image hidden states
            encoder_hidden_states: Text hidden states

        Returns:
            Merged hidden states (default: concatenate [text, image])
        """
        ...

    # -------------------------------------------------------------------------
    # High-level generation methods (used by DiffusionRunner)
    # -------------------------------------------------------------------------

    def create_latents(self, seed: int, runtime_config: RuntimeConfig) -> mx.array:
        """Create initial noise latents for generation.

        Args:
            seed: Random seed
            runtime_config: Runtime configuration with dimensions

        Returns:
            Initial latent tensor
        """
        ...

    def encode_prompt(self, prompt: str) -> PromptData:
        """Encode prompt into model-specific prompt data.

        Args:
            prompt: Text prompt

        Returns:
            PromptData containing embeddings (and model-specific extras)
        """
        ...

    @property
    def needs_cfg(self) -> bool:
        """Whether this model uses classifier-free guidance.

        Returns:
            True if model requires two forward passes with guidance (e.g., Qwen)
            False if model uses a single forward pass (e.g., Flux)
        """
        ...

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

    def decode_latents(
        self,
        latents: mx.array,
        runtime_config: RuntimeConfig,
        seed: int,
        prompt: str,
    ) -> Any:
        """Decode latents to final image.

        Args:
            latents: Final denoised latents
            runtime_config: Runtime configuration
            seed: Random seed (for metadata)
            prompt: Text prompt (for metadata)

        Returns:
            GeneratedImage result
        """
        ...
