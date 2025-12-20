from enum import Enum
from pathlib import Path
from typing import Any, Protocol

import mlx.core as mx
from mflux.config.runtime_config import RuntimeConfig

from exo.worker.engines.mflux.config.model_config import BlockType, ImageModelConfig
from exo.worker.engines.mflux.pipefusion.kv_cache import ImagePatchKVCache


class AttentionInterface(Protocol):
    """Protocol defining the interface for attention modules used in transformer blocks."""

    num_heads: int
    head_dimension: int
    to_q: Any
    to_k: Any
    to_v: Any
    norm_q: Any
    norm_k: Any
    to_out: list[Any]


class JointAttentionInterface(AttentionInterface, Protocol):
    """Protocol for attention modules in joint transformer blocks.

    Extends AttentionInterface with additional projections for the context stream.
    """

    add_q_proj: Any
    add_k_proj: Any
    add_v_proj: Any
    norm_added_q: Any
    norm_added_k: Any
    to_add_out: Any


class JointBlockInterface(Protocol):
    """Protocol defining the interface for joint transformer blocks.

    Joint blocks process both image and text streams separately,
    then combine them in attention.
    """

    attn: JointAttentionInterface
    norm2: Any
    norm2_context: Any
    ff: Any
    ff_context: Any

    def norm1(
        self, hidden_states: mx.array, text_embeddings: mx.array
    ) -> tuple[mx.array, mx.array, mx.array, mx.array, mx.array]:
        """Normalize image hidden states."""
        ...

    def norm1_context(
        self, hidden_states: mx.array, text_embeddings: mx.array
    ) -> tuple[mx.array, mx.array, mx.array, mx.array, mx.array]:
        """Normalize encoder hidden states."""
        ...


class SingleBlockInterface(Protocol):
    """Protocol defining the interface for single transformer blocks.

    Single blocks process concatenated [text + image] hidden states
    in a unified stream.
    """

    attn: AttentionInterface

    def norm(
        self, hidden_states: mx.array, text_embeddings: mx.array
    ) -> tuple[mx.array, mx.array]:
        """Normalize hidden states."""
        ...

    def _apply_feed_forward_and_projection(
        self, norm_hidden_states: mx.array, attn_output: mx.array, gate: mx.array
    ) -> mx.array:
        """Apply feed forward network and projection."""
        ...


class BlockWrapperMode(Enum):
    """Mode for block wrapper operation."""

    CACHING = "caching"  # Sync mode: compute full attention, populate cache
    PATCHED = "patched"  # Async mode: compute patch attention, use cached KV


class ModelAdapter(Protocol):
    """Protocol for model-specific operations in PipeFusion.

    Adapters handle the differences between mflux model architectures:
    - Flux: JointAttention + SingleBlockAttention
    - Fibo: FiboJointAttention with attention masks
    - Qwen: Unified blocks with different RoPE
    """

    @property
    def config(self) -> ImageModelConfig:
        """Return the model configuration."""
        ...

    def create_model(
        self,
        model_id: str,
        local_path: Path,
        quantize: int | None = None,
    ) -> Any:
        """Create the underlying mflux model instance.

        Args:
            model_id: The model identifier (e.g., "black-forest-labs/FLUX.1-schnell")
            local_path: Path to the local model weights
            quantize: Optional quantization bit width

        Returns:
            The mflux model instance (e.g., Flux1, Fibo, Qwen)
        """
        ...

    def compute_embeddings(
        self,
        hidden_states: mx.array,
        prompt_embeds: mx.array,
        transformer: Any,
    ) -> tuple[mx.array, mx.array]:
        """Compute x_embedder and context_embedder outputs.

        Args:
            hidden_states: Input latent states
            prompt_embeds: Text embeddings from encoder
            transformer: The transformer model

        Returns:
            Tuple of (embedded_hidden_states, embedded_encoder_states)
        """
        ...

    def compute_text_embeddings(
        self,
        t: int,
        pooled_prompt_embeds: mx.array,
        transformer: Any,
        runtime_config: RuntimeConfig,
    ) -> mx.array:
        """Compute time/text embeddings for conditioning.

        Args:
            t: Current timestep
            pooled_prompt_embeds: Pooled text embeddings
            transformer: The transformer model
            runtime_config: Runtime configuration

        Returns:
            Text embeddings tensor
        """
        ...

    def compute_rotary_embeddings(
        self,
        prompt_embeds: mx.array,
        transformer: Any,
        runtime_config: RuntimeConfig,
        **kwargs: Any,
    ) -> mx.array:
        """Compute rotary position embeddings.

        Args:
            prompt_embeds: Text embeddings
            transformer: The transformer model
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
        transformer: Any,
    ) -> mx.array:
        """Apply final norm and projection.

        Args:
            hidden_states: Hidden states (image only, text already removed)
            text_embeddings: Conditioning embeddings
            transformer: The transformer model

        Returns:
            Projected output
        """
        ...

    def get_joint_blocks(self, transformer: Any) -> list[JointBlockInterface]:
        """Get the list of joint transformer blocks from the model.

        Args:
            transformer: The transformer model

        Returns:
            List of joint transformer blocks
        """
        ...

    def get_single_blocks(self, transformer: Any) -> list[SingleBlockInterface]:
        """Get the list of single transformer blocks from the model.

        Args:
            transformer: The transformer model

        Returns:
            List of single transformer blocks
        """
        ...

    def get_blocks(self, transformer: Any) -> list[tuple[Any, BlockType]]:
        """Get all transformer blocks in execution order with their types.

        This method provides a unified view of all blocks, regardless of their
        specific type (joint, single, unified). New model adapters can override
        this to return blocks in their native arrangement.

        Args:
            transformer: The transformer model

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

        This unified method dispatches to the appropriate block-specific logic
        based on block_type. New model adapters can implement this directly
        without needing separate apply_joint_block/apply_single_block methods.

        Args:
            block: The transformer block
            block_type: Type of block (JOINT, SINGLE, or UNIFIED)
            hidden_states: Image hidden states (or concatenated for SINGLE/UNIFIED)
            encoder_hidden_states: Text hidden states (None for SINGLE/UNIFIED)
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
            - For SINGLE/UNIFIED blocks: (concatenated_hidden, None)
        """
        ...

    def merge_streams(
        self,
        hidden_states: mx.array,
        encoder_hidden_states: mx.array,
    ) -> mx.array:
        """Merge image and text streams for transition to single/unified blocks.

        This is called at the transition point from joint blocks (which process
        image and text separately) to single/unified blocks (which process them
        together). Override to customize the merge strategy.

        Args:
            hidden_states: Image hidden states
            encoder_hidden_states: Text hidden states

        Returns:
            Merged hidden states (default: concatenate [text, image])
        """
        ...
