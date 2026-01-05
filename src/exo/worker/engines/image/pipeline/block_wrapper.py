from typing import Any

import mlx.core as mx

from exo.worker.engines.image.pipeline.adapter import (
    BlockWrapperMode,
    JointBlockInterface,
    ModelAdapter,
    SingleBlockInterface,
)
from exo.worker.engines.image.pipeline.kv_cache import ImagePatchKVCache


class JointBlockWrapper:
    """Unified wrapper for joint transformer blocks.

    Handles both CACHING (sync) and PATCHED (async) modes by delegating
    to the model adapter for model-specific attention computation.

    The wrapper is created once at initialization and reused across calls.
    Mode and KV cache are passed at call time to support switching between
    sync and async pipelines.
    """

    def __init__(
        self,
        block: JointBlockInterface,
        adapter: ModelAdapter,
    ):
        """Initialize the joint block wrapper.

        Args:
            block: The joint transformer block to wrap
            adapter: Model adapter for model-specific operations
        """
        self.block = block
        self.adapter = adapter

    def __call__(
        self,
        hidden_states: mx.array,
        encoder_hidden_states: mx.array,
        text_embeddings: mx.array,
        rotary_embeddings: mx.array,
        text_seq_len: int,
        kv_cache: ImagePatchKVCache | None,
        mode: BlockWrapperMode,
        patch_start: int | None = None,
        patch_end: int | None = None,
        **kwargs: Any,
    ) -> tuple[mx.array, mx.array]:
        """Apply the joint block.

        Args:
            hidden_states: Image hidden states (full or patch depending on mode)
            encoder_hidden_states: Text hidden states
            text_embeddings: Conditioning embeddings
            rotary_embeddings: Rotary position embeddings
            text_seq_len: Text sequence length
            kv_cache: KV cache for storing/retrieving image K/V (None if not using cache)
            mode: CACHING (populate cache) or PATCHED (use cached K/V)
            patch_start: Start index for patched mode (required if mode=PATCHED)
            patch_end: End index for patched mode (required if mode=PATCHED)
            **kwargs: Additional model-specific arguments (e.g., encoder_hidden_states_mask,
                block_idx for Qwen)

        Returns:
            Tuple of (encoder_hidden_states, hidden_states)
        """
        return self.adapter.apply_joint_block(
            block=self.block,
            hidden_states=hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            text_embeddings=text_embeddings,
            rotary_embeddings=rotary_embeddings,
            kv_cache=kv_cache,
            mode=mode,
            text_seq_len=text_seq_len,
            patch_start=patch_start,
            patch_end=patch_end,
            **kwargs,
        )


class SingleBlockWrapper:
    """Unified wrapper for single transformer blocks.

    Handles both CACHING (sync) and PATCHED (async) modes by delegating
    to the model adapter for model-specific attention computation.

    The wrapper is created once at initialization and reused across calls.
    Mode and KV cache are passed at call time to support switching between
    sync and async pipelines.
    """

    def __init__(
        self,
        block: SingleBlockInterface,
        adapter: ModelAdapter,
    ):
        """Initialize the single block wrapper.

        Args:
            block: The single transformer block to wrap
            adapter: Model adapter for model-specific operations
        """
        self.block = block
        self.adapter = adapter

    def __call__(
        self,
        hidden_states: mx.array,
        text_embeddings: mx.array,
        rotary_embeddings: mx.array,
        text_seq_len: int,
        kv_cache: ImagePatchKVCache | None,
        mode: BlockWrapperMode,
        patch_start: int | None = None,
        patch_end: int | None = None,
    ) -> mx.array:
        """Apply the single block.

        Args:
            hidden_states: [text + image] hidden states (full or patch depending on mode)
            text_embeddings: Conditioning embeddings
            rotary_embeddings: Rotary position embeddings
            text_seq_len: Text sequence length
            kv_cache: KV cache for storing/retrieving image K/V (None if not using cache)
            mode: CACHING (populate cache) or PATCHED (use cached K/V)
            patch_start: Start index for patched mode (required if mode=PATCHED)
            patch_end: End index for patched mode (required if mode=PATCHED)

        Returns:
            Output hidden states
        """
        return self.adapter.apply_single_block(
            block=self.block,
            hidden_states=hidden_states,
            text_embeddings=text_embeddings,
            rotary_embeddings=rotary_embeddings,
            kv_cache=kv_cache,
            mode=mode,
            text_seq_len=text_seq_len,
            patch_start=patch_start,
            patch_end=patch_end,
        )
