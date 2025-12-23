from enum import Enum
from typing import Any, Protocol

import mlx.core as mx
from mflux.config.runtime_config import RuntimeConfig

from exo.worker.engines.image.config import BlockType, ImageModelConfig
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
        pooled_prompt_embeds: mx.array,
        runtime_config: RuntimeConfig,
    ) -> mx.array:
        """Compute time/text embeddings for conditioning.

        Args:
            t: Current timestep
            pooled_prompt_embeds: Pooled text embeddings
            runtime_config: Runtime configuration

        Returns:
            Text embeddings tensor
        """
        ...

    def compute_rotary_embeddings(
        self,
        prompt_embeds: mx.array,
        runtime_config: RuntimeConfig,
        **kwargs: Any,
    ) -> mx.array:
        """Compute rotary position embeddings.

        Args:
            prompt_embeds: Text embeddings
            runtime_config: Runtime configuration

        Returns:
            Rotary embeddings tensor
        """
        ...

    def apply_joint_block(
        self,
        block: JointBlockInterface,
        hidden_states: mx.array,
        encoder_hidden_states: mx.array,
        text_embeddings: mx.array,
        rotary_embeddings: mx.array,
        kv_cache: ImagePatchKVCache | None,
        mode: "BlockWrapperMode",
        text_seq_len: int,
        patch_start: int | None = None,
        patch_end: int | None = None,
    ) -> tuple[mx.array, mx.array]:
        """Apply a joint transformer block.

        Args:
            block: The joint transformer block
            hidden_states: Image hidden states
            encoder_hidden_states: Text hidden states
            text_embeddings: Conditioning embeddings
            rotary_embeddings: Rotary position embeddings
            kv_cache: KV cache (None if not using cache)
            mode: CACHING or PATCHED mode
            text_seq_len: Text sequence length
            patch_start: Start index for patched mode
            patch_end: End index for patched mode

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

    def get_blocks(self) -> list[tuple[Any, BlockType]]:
        """Get all transformer blocks in execution order with their types.

        This method provides a combined view of all blocks, regardless of their
        specific type (joint or single). New model adapters can override
        this to return blocks in their native arrangement.

        Returns:
            List of (block, block_type) tuples in execution order
        """
        ...

    def apply_block(
        self,
        block: Any,
        block_type: BlockType,
        hidden_states: mx.array,
        encoder_hidden_states: mx.array | None,
        text_embeddings: mx.array,
        rotary_embeddings: mx.array,
        kv_cache: ImagePatchKVCache | None,
        mode: "BlockWrapperMode",
        text_seq_len: int,
        patch_start: int | None = None,
        patch_end: int | None = None,
    ) -> tuple[mx.array, mx.array | None]:
        """Apply any transformer block type.

        This method dispatches to the appropriate block-specific logic
        based on block_type. New model adapters can implement this directly
        without needing separate apply_joint_block/apply_single_block methods.

        Args:
            block: The transformer block
            block_type: Type of block (JOINT or SINGLE)
            hidden_states: Image hidden states (or concatenated for SINGLE)
            encoder_hidden_states: Text hidden states (None for SINGLE)
            text_embeddings: Conditioning embeddings
            rotary_embeddings: Rotary position embeddings
            kv_cache: KV cache (None if not using cache)
            mode: CACHING or PATCHED mode
            text_seq_len: Text sequence length
            patch_start: Start index for patched mode
            patch_end: End index for patched mode

        Returns:
            Tuple of (hidden_states, encoder_hidden_states or None)
            - For JOINT blocks: (image_hidden, text_hidden)
            - For SINGLE blocks: (concatenated_hidden, None)
        """
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
