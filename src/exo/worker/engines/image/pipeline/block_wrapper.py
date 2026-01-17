from abc import ABC, abstractmethod
from enum import Enum
from typing import Any, Self

import mlx.core as mx

from exo.worker.engines.image.pipeline.kv_cache import ImagePatchKVCache


class BlockWrapperMode(Enum):
    CACHING = "caching"  # Sync mode: compute full attention, populate cache
    PATCHED = "patched"  # Async mode: compute patch attention, use cached KV


class JointBlockWrapper(ABC):
    """Base class for joint transformer block wrappers with pipefusion support.

    Subclass this to add pipefusion support to any model's joint blocks.
    The wrapper:
    - Owns its KV cache (created lazily on first CACHING forward)
    - Controls the forward pass flow (CACHING vs PATCHED mode)
    - Handles patch slicing and cache operations

    Model subclass provides:
    - _compute_qkv: Compute Q, K, V tensors (norms, projections, RoPE)
    - _compute_attention: Run scaled dot-product attention
    - _apply_output: Apply output projection, feed-forward, residuals
    """

    def __init__(self, block: Any, text_seq_len: int):
        """Initialize the joint block wrapper.

        Args:
            block: The joint transformer block to wrap
            text_seq_len: Number of text tokens (constant for entire generation)
        """
        self.block = block
        self._text_seq_len = text_seq_len
        self._kv_cache: ImagePatchKVCache | None = None  # Primary (or positive for CFG)
        self._kv_cache_negative: ImagePatchKVCache | None = None  # Only for CFG
        self._mode = BlockWrapperMode.CACHING
        self._patch_start: int = 0
        self._patch_end: int = 0
        self._use_negative_cache: bool = False

    def set_patch(
        self,
        mode: BlockWrapperMode,
        patch_start: int = 0,
        patch_end: int = 0,
    ) -> Self:
        """Set mode and patch range.

        Args:
            mode: CACHING (full attention) or PATCHED (use cached KV)
            patch_start: Start token index within image (for PATCHED mode)
            patch_end: End token index within image (for PATCHED mode)

        Returns:
            Self for method chaining
        """
        self._mode = mode
        self._patch_start = patch_start
        self._patch_end = patch_end
        return self

    def set_use_negative_cache(self, use_negative: bool) -> None:
        """Switch to negative cache for CFG. False = primary cache."""
        self._use_negative_cache = use_negative

    def set_text_seq_len(self, text_seq_len: int) -> None:
        """Update text sequence length for CFG passes with different prompt lengths."""
        self._text_seq_len = text_seq_len

    def _get_active_cache(self) -> ImagePatchKVCache | None:
        """Get the active KV cache based on current CFG pass."""
        if self._use_negative_cache:
            return self._kv_cache_negative
        return self._kv_cache

    def _ensure_cache(self, img_key: mx.array) -> None:
        """Create cache on first CACHING forward using actual dimensions."""
        batch, num_heads, img_seq_len, head_dim = img_key.shape
        if self._use_negative_cache:
            if self._kv_cache_negative is None:
                self._kv_cache_negative = ImagePatchKVCache(
                    batch_size=batch,
                    num_heads=num_heads,
                    image_seq_len=img_seq_len,
                    head_dim=head_dim,
                )
        else:
            if self._kv_cache is None:
                self._kv_cache = ImagePatchKVCache(
                    batch_size=batch,
                    num_heads=num_heads,
                    image_seq_len=img_seq_len,
                    head_dim=head_dim,
                )

    def _cache_full_image_kv(self, img_key: mx.array, img_value: mx.array) -> None:
        """Store full image K/V during CACHING mode."""
        self._ensure_cache(img_key)
        cache = self._get_active_cache()
        assert cache is not None
        cache.update_image_patch(0, img_key.shape[2], img_key, img_value)

    def _cache_patch_kv(self, img_key: mx.array, img_value: mx.array) -> None:
        """Store current patch's K/V during PATCHED mode."""
        cache = self._get_active_cache()
        assert cache is not None
        cache.update_image_patch(self._patch_start, self._patch_end, img_key, img_value)

    def _get_full_kv(
        self, text_key: mx.array, text_value: mx.array
    ) -> tuple[mx.array, mx.array]:
        """Get full K/V by combining fresh text with cached image."""
        cache = self._get_active_cache()
        assert cache is not None
        return cache.get_full_kv(text_key, text_value)

    def reset_cache(self) -> None:
        """Reset all KV caches. Call at the start of a new generation."""
        self._kv_cache = None
        self._kv_cache_negative = None

    def set_encoder_mask(self, mask: mx.array | None) -> None:  # noqa: B027
        """Set the encoder hidden states mask for attention.

        Override in subclasses that use attention masks (e.g., Qwen).
        Default is a no-op for models that don't use masks (e.g., Flux).
        """
        del mask  # Unused in base class

    def __call__(
        self,
        hidden_states: mx.array,
        encoder_hidden_states: mx.array,
        text_embeddings: mx.array,
        rotary_embeddings: Any,
    ) -> tuple[mx.array, mx.array]:
        """Apply the joint block.

        Args:
            hidden_states: Image hidden states [B, num_img_tokens, D]
            encoder_hidden_states: Text hidden states [B, text_seq_len, D]
            text_embeddings: Conditioning embeddings [B, D]
            rotary_embeddings: Rotary position embeddings (model-specific format)

        Returns:
            Tuple of (encoder_hidden_states, hidden_states) - text and image outputs
        """
        if self._mode == BlockWrapperMode.CACHING:
            return self._forward_caching(
                hidden_states, encoder_hidden_states, text_embeddings, rotary_embeddings
            )
        return self._forward_patched(
            hidden_states, encoder_hidden_states, text_embeddings, rotary_embeddings
        )

    def _forward_caching(
        self,
        hidden_states: mx.array,
        encoder_hidden_states: mx.array,
        text_embeddings: mx.array,
        rotary_embeddings: Any,
    ) -> tuple[mx.array, mx.array]:
        """CACHING mode: Full attention, store image K/V in cache."""
        # Model computes Q/K/V for full sequence
        query, key, value = self._compute_qkv(
            hidden_states, encoder_hidden_states, text_embeddings, rotary_embeddings
        )

        img_key = key[:, :, self._text_seq_len :, :]
        img_value = value[:, :, self._text_seq_len :, :]
        self._cache_full_image_kv(img_key, img_value)

        attn_out = self._compute_attention(query, key, value)

        return self._apply_output(
            attn_out, hidden_states, encoder_hidden_states, text_embeddings
        )

    def _forward_patched(
        self,
        hidden_states: mx.array,
        encoder_hidden_states: mx.array,
        text_embeddings: mx.array,
        rotary_embeddings: Any,
    ) -> tuple[mx.array, mx.array]:
        """PATCHED mode: Compute patch Q/K/V, use cached image K/V for attention."""
        # hidden_states is already the patch (provided by runner)
        patch_hidden = hidden_states

        query, key, value = self._compute_patch_qkv(
            patch_hidden, encoder_hidden_states, text_embeddings, rotary_embeddings
        )

        text_key = key[:, :, : self._text_seq_len, :]
        text_value = value[:, :, : self._text_seq_len, :]
        img_key = key[:, :, self._text_seq_len :, :]
        img_value = value[:, :, self._text_seq_len :, :]

        self._cache_patch_kv(img_key, img_value)
        full_key, full_value = self._get_full_kv(text_key, text_value)

        attn_out = self._compute_attention(query, full_key, full_value)

        return self._apply_output(
            attn_out, patch_hidden, encoder_hidden_states, text_embeddings
        )

    @abstractmethod
    def _compute_qkv(
        self,
        hidden_states: mx.array,
        encoder_hidden_states: mx.array,
        text_embeddings: mx.array,
        rotary_embeddings: Any,
    ) -> tuple[mx.array, mx.array, mx.array]:
        """Compute Q, K, V tensors for full sequence.

        Includes normalization, projections, concatenation, and RoPE.

        Args:
            hidden_states: Image hidden states [B, num_img_tokens, D]
            encoder_hidden_states: Text hidden states [B, text_seq_len, D]
            text_embeddings: Conditioning embeddings [B, D]
            rotary_embeddings: Rotary position embeddings

        Returns:
            Tuple of (query, key, value) with shape [B, H, text+img, head_dim]
        """
        ...

    @abstractmethod
    def _compute_patch_qkv(
        self,
        patch_hidden: mx.array,
        encoder_hidden_states: mx.array,
        text_embeddings: mx.array,
        rotary_embeddings: Any,
    ) -> tuple[mx.array, mx.array, mx.array]:
        """Compute Q, K, V tensors for [text + patch].

        Similar to _compute_qkv but for patch mode - may need to slice RoPE.

        Args:
            patch_hidden: Patch hidden states [B, patch_len, D]
            encoder_hidden_states: Text hidden states [B, text_seq_len, D]
            text_embeddings: Conditioning embeddings [B, D]
            rotary_embeddings: Rotary position embeddings

        Returns:
            Tuple of (query, key, value) with shape [B, H, text+patch, head_dim]
        """
        ...

    @abstractmethod
    def _compute_attention(
        self, query: mx.array, key: mx.array, value: mx.array
    ) -> mx.array:
        """Compute scaled dot-product attention.

        Args:
            query: Query tensor [B, H, Q_len, head_dim]
            key: Key tensor [B, H, KV_len, head_dim]
            value: Value tensor [B, H, KV_len, head_dim]

        Returns:
            Attention output [B, Q_len, D]
        """
        ...

    @abstractmethod
    def _apply_output(
        self,
        attn_out: mx.array,
        hidden_states: mx.array,
        encoder_hidden_states: mx.array,
        text_embeddings: mx.array,
    ) -> tuple[mx.array, mx.array]:
        """Apply output projection, feed-forward, and residuals.

        Args:
            attn_out: Attention output [B, text+img, D]
            hidden_states: Original image hidden states (for residual)
            encoder_hidden_states: Original text hidden states (for residual)
            text_embeddings: Conditioning embeddings

        Returns:
            Tuple of (encoder_hidden_states, hidden_states) - updated text and image
        """
        ...


class SingleBlockWrapper(ABC):
    """Base class for single-stream transformer block wrappers.

    Similar to JointBlockWrapper but for blocks that operate on a single
    concatenated [text, image] stream rather than separate streams.
    """

    def __init__(self, block: Any, text_seq_len: int):
        """Initialize the single block wrapper.

        Args:
            block: The single transformer block to wrap
            text_seq_len: Number of text tokens (constant for entire generation)
        """
        self.block = block
        self._text_seq_len = text_seq_len
        self._kv_cache: ImagePatchKVCache | None = None  # Primary (or positive for CFG)
        self._kv_cache_negative: ImagePatchKVCache | None = None  # Only for CFG
        self._mode = BlockWrapperMode.CACHING
        self._patch_start: int = 0
        self._patch_end: int = 0
        self._use_negative_cache: bool = False

    def set_patch(
        self,
        mode: BlockWrapperMode,
        patch_start: int = 0,
        patch_end: int = 0,
    ) -> Self:
        """Set mode and patch range. Only call when these change."""
        self._mode = mode
        self._patch_start = patch_start
        self._patch_end = patch_end
        return self

    def set_use_negative_cache(self, use_negative: bool) -> None:
        """Switch to negative cache for CFG. False = primary cache."""
        self._use_negative_cache = use_negative

    def set_text_seq_len(self, text_seq_len: int) -> None:
        """Update text sequence length for CFG passes with different prompt lengths."""
        self._text_seq_len = text_seq_len

    def _get_active_cache(self) -> ImagePatchKVCache | None:
        """Get the active KV cache based on current CFG pass."""
        if self._use_negative_cache:
            return self._kv_cache_negative
        return self._kv_cache

    def _ensure_cache(self, img_key: mx.array) -> None:
        """Create cache on first CACHING forward using actual dimensions."""
        batch, num_heads, img_seq_len, head_dim = img_key.shape
        if self._use_negative_cache:
            if self._kv_cache_negative is None:
                self._kv_cache_negative = ImagePatchKVCache(
                    batch_size=batch,
                    num_heads=num_heads,
                    image_seq_len=img_seq_len,
                    head_dim=head_dim,
                )
        else:
            if self._kv_cache is None:
                self._kv_cache = ImagePatchKVCache(
                    batch_size=batch,
                    num_heads=num_heads,
                    image_seq_len=img_seq_len,
                    head_dim=head_dim,
                )

    def _cache_full_image_kv(self, img_key: mx.array, img_value: mx.array) -> None:
        """Store full image K/V during CACHING mode."""
        self._ensure_cache(img_key)
        cache = self._get_active_cache()
        assert cache is not None
        cache.update_image_patch(0, img_key.shape[2], img_key, img_value)

    def _cache_patch_kv(self, img_key: mx.array, img_value: mx.array) -> None:
        """Store current patch's K/V during PATCHED mode."""
        cache = self._get_active_cache()
        assert cache is not None
        cache.update_image_patch(self._patch_start, self._patch_end, img_key, img_value)

    def _get_full_kv(
        self, text_key: mx.array, text_value: mx.array
    ) -> tuple[mx.array, mx.array]:
        """Get full K/V by combining fresh text with cached image."""
        cache = self._get_active_cache()
        assert cache is not None
        return cache.get_full_kv(text_key, text_value)

    def reset_cache(self) -> None:
        """Reset all KV caches. Call at the start of a new generation."""
        self._kv_cache = None
        self._kv_cache_negative = None

    def __call__(
        self,
        hidden_states: mx.array,
        text_embeddings: mx.array,
        rotary_embeddings: Any,
    ) -> mx.array:
        """Apply the single block.

        Args:
            hidden_states: Concatenated [text, image] hidden states
            text_embeddings: Conditioning embeddings [B, D]
            rotary_embeddings: Rotary position embeddings

        Returns:
            Updated hidden states [B, text+img, D]
        """
        if self._mode == BlockWrapperMode.CACHING:
            return self._forward_caching(
                hidden_states, text_embeddings, rotary_embeddings
            )
        return self._forward_patched(hidden_states, text_embeddings, rotary_embeddings)

    def _forward_caching(
        self,
        hidden_states: mx.array,
        text_embeddings: mx.array,
        rotary_embeddings: Any,
    ) -> mx.array:
        """CACHING mode: Full attention, store image K/V in cache."""
        query, key, value = self._compute_qkv(
            hidden_states, text_embeddings, rotary_embeddings
        )

        img_key = key[:, :, self._text_seq_len :, :]
        img_value = value[:, :, self._text_seq_len :, :]
        self._cache_full_image_kv(img_key, img_value)

        attn_out = self._compute_attention(query, key, value)

        return self._apply_output(attn_out, hidden_states, text_embeddings)

    def _forward_patched(
        self,
        hidden_states: mx.array,
        text_embeddings: mx.array,
        rotary_embeddings: Any,
    ) -> mx.array:
        """PATCHED mode: Compute patch Q/K/V, use cached image K/V for attention."""
        # hidden_states is already [text, patch] - extract both parts
        text_hidden = hidden_states[:, : self._text_seq_len, :]
        patch_hidden = hidden_states[:, self._text_seq_len :, :]
        patch_states = mx.concatenate([text_hidden, patch_hidden], axis=1)

        query, key, value = self._compute_patch_qkv(
            patch_states, text_embeddings, rotary_embeddings
        )

        text_key = key[:, :, : self._text_seq_len, :]
        text_value = value[:, :, : self._text_seq_len, :]
        img_key = key[:, :, self._text_seq_len :, :]
        img_value = value[:, :, self._text_seq_len :, :]

        self._cache_patch_kv(img_key, img_value)
        full_key, full_value = self._get_full_kv(text_key, text_value)

        attn_out = self._compute_attention(query, full_key, full_value)

        return self._apply_output(attn_out, patch_states, text_embeddings)

    @abstractmethod
    def _compute_qkv(
        self,
        hidden_states: mx.array,
        text_embeddings: mx.array,
        rotary_embeddings: Any,
    ) -> tuple[mx.array, mx.array, mx.array]:
        """Compute Q, K, V tensors for full sequence."""
        ...

    @abstractmethod
    def _compute_patch_qkv(
        self,
        patch_states: mx.array,
        text_embeddings: mx.array,
        rotary_embeddings: Any,
    ) -> tuple[mx.array, mx.array, mx.array]:
        """Compute Q, K, V tensors for [text + patch]."""
        ...

    @abstractmethod
    def _compute_attention(
        self, query: mx.array, key: mx.array, value: mx.array
    ) -> mx.array:
        """Compute scaled dot-product attention."""
        ...

    @abstractmethod
    def _apply_output(
        self,
        attn_out: mx.array,
        hidden_states: mx.array,
        text_embeddings: mx.array,
    ) -> mx.array:
        """Apply output projection, feed-forward, and residuals."""
        ...
