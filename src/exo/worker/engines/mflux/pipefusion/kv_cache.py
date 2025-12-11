import mlx.core as mx


class JointPatchKVCache:
    """KV cache for joint attention where text and image are processed separately.

    Used for joint transformer blocks (19 double blocks in Flux).
    Separates text and image portions:
    - Text K/V is always "fresh" (updated each patch since we have full text)
    - Image K/V uses stale values for non-current patches
    """

    def __init__(
        self,
        batch_size: int,
        num_heads: int,
        text_seq_len: int,
        image_seq_len: int,
        head_dim: int,
        dtype: mx.Dtype = mx.float32,
    ):
        self.batch_size = batch_size
        self.num_heads = num_heads
        self.text_seq_len = text_seq_len
        self.image_seq_len = image_seq_len
        self.head_dim = head_dim
        self.total_seq_len = text_seq_len + image_seq_len

        self.key_cache = mx.zeros(
            (batch_size, num_heads, self.total_seq_len, head_dim), dtype=dtype
        )
        self.value_cache = mx.zeros(
            (batch_size, num_heads, self.total_seq_len, head_dim), dtype=dtype
        )

    def update_text(self, key: mx.array, value: mx.array) -> None:
        """Update text portion (always fresh, not patched).

        Args:
            key: Text key tensor [batch, heads, text_seq_len, head_dim]
            value: Text value tensor [batch, heads, text_seq_len, head_dim]
        """
        self.key_cache[:, :, : self.text_seq_len, :] = key
        self.value_cache[:, :, : self.text_seq_len, :] = value

    def update_image_patch(
        self, patch_start: int, patch_end: int, key: mx.array, value: mx.array
    ) -> None:
        """Update image patch portion.

        Args:
            patch_start: Start token index within image portion (0-indexed)
            patch_end: End token index within image portion
            key: Image patch key tensor [batch, heads, patch_len, head_dim]
            value: Image patch value tensor [batch, heads, patch_len, head_dim]
        """
        start = self.text_seq_len + patch_start
        end = self.text_seq_len + patch_end
        self.key_cache[:, :, start:end, :] = key
        self.value_cache[:, :, start:end, :] = value

    def get_full_kv(self) -> tuple[mx.array, mx.array]:
        """Return full cached K/V (text + image with fresh/stale mix)."""
        return self.key_cache, self.value_cache


class PatchKVCache:
    """KV cache that stores full sequence K/V with patch-level updates.

    Used for single transformer blocks where text and image tokens are concatenated.
    The cache stores K/V for the full sequence [text + image] and allows
    updating individual image patch slices while keeping stale values for others.
    """

    def __init__(
        self,
        batch_size: int,
        num_heads: int,
        total_seq_len: int,
        head_dim: int,
        dtype: mx.Dtype = mx.float32,
    ):
        self.batch_size = batch_size
        self.num_heads = num_heads
        self.total_seq_len = total_seq_len
        self.head_dim = head_dim

        self.key_cache = mx.zeros(
            (batch_size, num_heads, total_seq_len, head_dim), dtype=dtype
        )
        self.value_cache = mx.zeros(
            (batch_size, num_heads, total_seq_len, head_dim), dtype=dtype
        )

    def update(
        self, patch_start: int, patch_end: int, key: mx.array, value: mx.array
    ) -> None:
        """Update cache with fresh K/V for a patch slice.

        Args:
            patch_start: Start token index in the full sequence
            patch_end: End token index in the full sequence
            key: Fresh key tensor [batch, heads, patch_seq_len, head_dim]
            value: Fresh value tensor [batch, heads, patch_seq_len, head_dim]
        """
        self.key_cache[:, :, patch_start:patch_end, :] = key
        self.value_cache[:, :, patch_start:patch_end, :] = value

    def get_full_kv(self) -> tuple[mx.array, mx.array]:
        """Return full cached K/V (mix of fresh current patch + stale others)."""
        return self.key_cache, self.value_cache
