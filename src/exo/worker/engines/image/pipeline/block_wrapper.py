from abc import ABC, abstractmethod
from enum import Enum
from typing import Generic, Self, TypeVar

import mlx.core as mx

from exo.worker.engines.image.models.base import RotaryEmbeddings
from exo.worker.engines.image.pipeline.kv_cache import ImagePatchKVCache

BlockT = TypeVar("BlockT")


class BlockWrapperMode(Enum):
    CACHING = "caching"  # Sync mode: compute full attention, populate cache
    PATCHED = "patched"  # Async mode: compute patch attention, use cached KV


class BlockWrapperMixin:
    """Common cache management logic for block wrappers.

    Including:
    - KV cache creation and management
    - Mode
    - Patch range setting
    """

    _text_seq_len: int
    _kv_cache: ImagePatchKVCache | None
    _mode: BlockWrapperMode
    _patch_start: int
    _patch_end: int

    def _init_cache_state(self, text_seq_len: int) -> None:
        self._text_seq_len = text_seq_len
        self._kv_cache = None
        self._mode = BlockWrapperMode.CACHING
        self._patch_start = 0
        self._patch_end = 0

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

    def set_text_seq_len(self, text_seq_len: int) -> None:
        self._text_seq_len = text_seq_len

    def _get_active_cache(self) -> ImagePatchKVCache | None:
        return self._kv_cache

    def _ensure_cache(self, img_key: mx.array) -> None:
        if self._kv_cache is None:
            batch, num_heads, img_seq_len, head_dim = img_key.shape
            self._kv_cache = ImagePatchKVCache(
                batch_size=batch,
                num_heads=num_heads,
                image_seq_len=img_seq_len,
                head_dim=head_dim,
            )

    def _cache_full_image_kv(self, img_key: mx.array, img_value: mx.array) -> None:
        self._ensure_cache(img_key)
        cache = self._get_active_cache()
        assert cache is not None
        cache.update_image_patch(0, img_key.shape[2], img_key, img_value)

    def _cache_patch_kv(self, img_key: mx.array, img_value: mx.array) -> None:
        cache = self._get_active_cache()
        assert cache is not None
        cache.update_image_patch(self._patch_start, self._patch_end, img_key, img_value)

    def _get_full_kv(
        self, text_key: mx.array, text_value: mx.array
    ) -> tuple[mx.array, mx.array]:
        cache = self._get_active_cache()
        assert cache is not None
        return cache.get_full_kv(text_key, text_value)

    def reset_cache(self) -> None:
        self._kv_cache = None


class JointBlockWrapper(BlockWrapperMixin, ABC, Generic[BlockT]):
    """Base class for joint transformer block wrappers with pipefusion support.

    The wrapper:
    - Owns its KV cache (created lazily on first CACHING forward)
    - Controls the forward pass flow (CACHING vs PATCHED mode)
    - Handles patch slicing and cache operations
    """

    block: BlockT

    def __init__(self, block: BlockT, text_seq_len: int):
        self.block = block
        self._init_cache_state(text_seq_len)

    def set_encoder_mask(self, mask: mx.array | None) -> None:  # noqa: B027
        """Set the encoder hidden states mask for attention.

        Override in subclasses that use attention masks
        Default is a no-op for models that don't use masks
        """
        del mask  # Unused in base class

    def __call__(
        self,
        hidden_states: mx.array,
        encoder_hidden_states: mx.array,
        text_embeddings: mx.array,
        rotary_embeddings: RotaryEmbeddings,
    ) -> tuple[mx.array, mx.array]:
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
        rotary_embeddings: RotaryEmbeddings,
    ) -> tuple[mx.array, mx.array]:
        """CACHING mode: Full attention, store image K/V in cache."""
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
        rotary_embeddings: RotaryEmbeddings,
    ) -> tuple[mx.array, mx.array]:
        # hidden_states is already the patch (provided by runner)
        patch_hidden = hidden_states

        query, key, value = self._compute_qkv(
            patch_hidden,
            encoder_hidden_states,
            text_embeddings,
            rotary_embeddings,
            patch_mode=True,
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
        rotary_embeddings: RotaryEmbeddings,
        patch_mode: bool = False,
    ) -> tuple[mx.array, mx.array, mx.array]: ...

    @abstractmethod
    def _compute_attention(
        self, query: mx.array, key: mx.array, value: mx.array
    ) -> mx.array: ...

    @abstractmethod
    def _apply_output(
        self,
        attn_out: mx.array,
        hidden_states: mx.array,
        encoder_hidden_states: mx.array,
        text_embeddings: mx.array,
    ) -> tuple[mx.array, mx.array]: ...


class SingleBlockWrapper(BlockWrapperMixin, ABC, Generic[BlockT]):
    """Base class for single-stream transformer block wrappers.

    Similar to JointBlockWrapper but for blocks that operate on a single
    concatenated [text, image] stream rather than separate streams.
    """

    block: BlockT

    def __init__(self, block: BlockT, text_seq_len: int):
        self.block = block
        self._init_cache_state(text_seq_len)

    def __call__(
        self,
        hidden_states: mx.array,
        text_embeddings: mx.array,
        rotary_embeddings: RotaryEmbeddings,
    ) -> mx.array:
        if self._mode == BlockWrapperMode.CACHING:
            return self._forward_caching(
                hidden_states, text_embeddings, rotary_embeddings
            )
        return self._forward_patched(hidden_states, text_embeddings, rotary_embeddings)

    def _forward_caching(
        self,
        hidden_states: mx.array,
        text_embeddings: mx.array,
        rotary_embeddings: RotaryEmbeddings,
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
        rotary_embeddings: RotaryEmbeddings,
    ) -> mx.array:
        """PATCHED mode: Compute patch Q/K/V, use cached image K/V for attention."""
        query, key, value = self._compute_qkv(
            hidden_states, text_embeddings, rotary_embeddings, patch_mode=True
        )

        text_key = key[:, :, : self._text_seq_len, :]
        text_value = value[:, :, : self._text_seq_len, :]
        img_key = key[:, :, self._text_seq_len :, :]
        img_value = value[:, :, self._text_seq_len :, :]

        self._cache_patch_kv(img_key, img_value)
        full_key, full_value = self._get_full_kv(text_key, text_value)

        attn_out = self._compute_attention(query, full_key, full_value)

        return self._apply_output(attn_out, hidden_states, text_embeddings)

    @abstractmethod
    def _compute_qkv(
        self,
        hidden_states: mx.array,
        text_embeddings: mx.array,
        rotary_embeddings: RotaryEmbeddings,
        patch_mode: bool = False,
    ) -> tuple[mx.array, mx.array, mx.array]: ...

    @abstractmethod
    def _compute_attention(
        self, query: mx.array, key: mx.array, value: mx.array
    ) -> mx.array: ...

    @abstractmethod
    def _apply_output(
        self,
        attn_out: mx.array,
        hidden_states: mx.array,
        text_embeddings: mx.array,
    ) -> mx.array: ...
