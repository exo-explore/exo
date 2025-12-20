from enum import Enum
from pathlib import Path
from typing import Any, Protocol

import mlx.core as mx
from mflux.config.runtime_config import RuntimeConfig

from exo.worker.engines.mflux.config.model_config import ImageModelConfig
from exo.worker.engines.mflux.pipefusion.kv_cache import ImagePatchKVCache


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
        block: Any,
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
        block: Any,
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
