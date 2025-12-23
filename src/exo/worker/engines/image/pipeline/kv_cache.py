import mlx.core as mx


class ImagePatchKVCache:
    """KV cache that stores only IMAGE K/V with patch-level updates.

    Only caches image K/V since:
    - Text K/V is always computed fresh (same for all patches)
    - Only image portion needs stale/fresh cache management across patches
    """

    def __init__(
        self,
        batch_size: int,
        num_heads: int,
        image_seq_len: int,
        head_dim: int,
        dtype: mx.Dtype = mx.float32,
    ):
        self.batch_size = batch_size
        self.num_heads = num_heads
        self.image_seq_len = image_seq_len
        self.head_dim = head_dim
        self._dtype = dtype

        self.key_cache = mx.zeros(
            (batch_size, num_heads, image_seq_len, head_dim), dtype=dtype
        )
        self.value_cache = mx.zeros(
            (batch_size, num_heads, image_seq_len, head_dim), dtype=dtype
        )

    def update_image_patch(
        self, patch_start: int, patch_end: int, key: mx.array, value: mx.array
    ) -> None:
        """Update cache with fresh K/V for an image patch slice.

        Args:
            patch_start: Start token index within image portion (0-indexed)
            patch_end: End token index within image portion
            key: Fresh key tensor [batch, heads, patch_seq_len, head_dim]
            value: Fresh value tensor [batch, heads, patch_seq_len, head_dim]
        """
        self.key_cache[:, :, patch_start:patch_end, :] = key
        self.value_cache[:, :, patch_start:patch_end, :] = value

    def get_full_kv(
        self, text_key: mx.array, text_value: mx.array
    ) -> tuple[mx.array, mx.array]:
        """Return full K/V by concatenating fresh text K/V with cached image K/V.

        Args:
            text_key: Fresh text key tensor [batch, heads, text_seq_len, head_dim]
            text_value: Fresh text value tensor [batch, heads, text_seq_len, head_dim]

        Returns:
            Tuple of (full_key, full_value) with shape [batch, heads, text+image, head_dim]
        """
        full_key = mx.concatenate([text_key, self.key_cache], axis=2)
        full_value = mx.concatenate([text_value, self.value_cache], axis=2)
        return full_key, full_value

    def reset(self) -> None:
        """Reset cache to zeros."""
        self.key_cache = mx.zeros(
            (self.batch_size, self.num_heads, self.image_seq_len, self.head_dim),
            dtype=self._dtype,
        )
        self.value_cache = mx.zeros(
            (self.batch_size, self.num_heads, self.image_seq_len, self.head_dim),
            dtype=self._dtype,
        )
