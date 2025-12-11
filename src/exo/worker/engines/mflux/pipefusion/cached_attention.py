import mlx.core as mx
from mflux.models.flux.model.flux_transformer.common.attention_utils import (
    AttentionUtils,
)
from mflux.models.flux.model.flux_transformer.joint_attention import JointAttention
from mflux.models.flux.model.flux_transformer.single_block_attention import (
    SingleBlockAttention,
)

from exo.worker.engines.mflux.pipefusion.kv_cache import JointPatchKVCache, PatchKVCache


class CachedJointAttention:
    """JointAttention that uses KV cache for PipeFusion.

    Processes attention for a single image patch while attending to the full
    sequence (text + all image patches) using cached K/V values.
    """

    head_dim = 128
    num_heads = 24

    def __init__(self, attn: JointAttention):
        """Wrap an existing JointAttention module.

        Args:
            attn: The original JointAttention module (for weight access)
        """
        self.attn = attn
        self.num_heads = attn.num_heads
        self.head_dim = attn.head_dimension

    def __call__(
        self,
        norm_hidden: mx.array,
        norm_encoder: mx.array,
        image_rotary_emb: mx.array,
        kv_cache: JointPatchKVCache,
        patch_start: int,
        patch_end: int,
        text_seq_len: int,
    ) -> tuple[mx.array, mx.array]:
        """Forward pass with KV caching for patch-based attention.

        Args:
            norm_hidden: Normalized image patch hidden states [B, patch_img_len, D]
            norm_encoder: Normalized full text hidden states [B, text_len, D]
            image_rotary_emb: Full rotary embeddings for [text + full_image]
            kv_cache: KV cache to update and read from
            patch_start: Start token index of this patch in image sequence
            patch_end: End token index of this patch in image sequence
            text_seq_len: Length of text sequence

        Returns:
            Tuple of (hidden_states_out, encoder_hidden_states_out) for the patch
        """
        # 1. Compute Q, K, V for current image patch
        img_query, img_key, img_value = AttentionUtils.process_qkv(
            hidden_states=norm_hidden,
            to_q=self.attn.to_q,
            to_k=self.attn.to_k,
            to_v=self.attn.to_v,
            norm_q=self.attn.norm_q,
            norm_k=self.attn.norm_k,
            num_heads=self.num_heads,
            head_dim=self.head_dim,
        )

        # 2. Compute Q, K, V for text (always full)
        txt_query, txt_key, txt_value = AttentionUtils.process_qkv(
            hidden_states=norm_encoder,
            to_q=self.attn.add_q_proj,
            to_k=self.attn.add_k_proj,
            to_v=self.attn.add_v_proj,
            norm_q=self.attn.norm_added_q,
            norm_k=self.attn.norm_added_k,
            num_heads=self.num_heads,
            head_dim=self.head_dim,
        )

        # 3. Concatenate Q for attention output: [text, image_patch]
        query = mx.concatenate([txt_query, img_query], axis=2)

        # 4. Concatenate K, V for this patch: [text, image_patch]
        patch_key = mx.concatenate([txt_key, img_key], axis=2)
        patch_value = mx.concatenate([txt_value, img_value], axis=2)

        # 5. Extract RoPE for [text + current_patch] positions
        # RoPE shape is (1, 1, seq_len, head_dim/2, 2, 2), sequence is on axis 2
        text_rope = image_rotary_emb[:, :, :text_seq_len, ...]
        patch_img_rope = image_rotary_emb[
            :, :, text_seq_len + patch_start : text_seq_len + patch_end, ...
        ]
        patch_rope = mx.concatenate([text_rope, patch_img_rope], axis=2)

        # 6. Apply RoPE to Q and K for current patch
        query, patch_key = AttentionUtils.apply_rope(
            xq=query, xk=patch_key, freqs_cis=patch_rope
        )

        # 7. Update cache with this patch's K, V (after RoPE)
        kv_cache.update_text(
            key=patch_key[:, :, :text_seq_len, :],
            value=patch_value[:, :, :text_seq_len, :],
        )
        kv_cache.update_image_patch(
            patch_start=patch_start,
            patch_end=patch_end,
            key=patch_key[:, :, text_seq_len:, :],
            value=patch_value[:, :, text_seq_len:, :],
        )

        # 8. Get full K, V from cache (fresh current patch + stale others)
        full_key, full_value = kv_cache.get_full_kv()

        # 9. Compute attention: patch query attends to full K, V
        # Query shape: [B, H, text_seq_len + patch_len, D]
        # Key/Value shape: [B, H, text_seq_len + full_img_len, D]
        # Output shape: [B, text_seq_len + patch_len, D]
        batch_size = norm_hidden.shape[0]
        attn_output = AttentionUtils.compute_attention(
            query=query,
            key=full_key,
            value=full_value,
            batch_size=batch_size,
            num_heads=self.num_heads,
            head_dim=self.head_dim,
        )

        # 10. Extract outputs for text and image patch
        encoder_hidden_states_out = attn_output[:, :text_seq_len, :]
        hidden_states_out = attn_output[:, text_seq_len:, :]

        # 11. Project outputs
        hidden_states_out = self.attn.to_out[0](hidden_states_out)
        encoder_hidden_states_out = self.attn.to_add_out(encoder_hidden_states_out)

        return hidden_states_out, encoder_hidden_states_out


class CachedSingleBlockAttention:
    """SingleBlockAttention that uses KV cache for PipeFusion.

    Processes attention for a single patch of concatenated [text + image]
    while attending to the full sequence using cached K/V values.
    """

    head_dim = 128
    num_heads = 24

    def __init__(self, attn: SingleBlockAttention):
        """Wrap an existing SingleBlockAttention module.

        Args:
            attn: The original SingleBlockAttention module (for weight access)
        """
        self.attn = attn
        self.head_dim = attn.head_dimension
        self.num_heads = attn.num_heads

    def __call__(
        self,
        norm_hidden: mx.array,
        image_rotary_emb: mx.array,
        kv_cache: PatchKVCache,
        patch_start: int,
        patch_end: int,
        text_seq_len: int,
    ) -> mx.array:
        """Forward pass with KV caching for patch-based attention.

        Args:
            norm_hidden: Normalized patch of [text + image] hidden states [B, text_len + patch_img_len, D]
            image_rotary_emb: Full rotary embeddings for [text + full_image]
            kv_cache: KV cache to update and read from
            patch_start: Start token index of image patch (within image portion)
            patch_end: End token index of image patch (within image portion)
            text_seq_len: Length of text portion in hidden_states

        Returns:
            Attention output for the patch [B, text_len + patch_img_len, D]
        """
        # 1. Compute Q, K, V for current patch
        query, key, value = AttentionUtils.process_qkv(
            hidden_states=norm_hidden,
            to_q=self.attn.to_q,
            to_k=self.attn.to_k,
            to_v=self.attn.to_v,
            norm_q=self.attn.norm_q,
            norm_k=self.attn.norm_k,
            num_heads=self.num_heads,
            head_dim=self.head_dim,
        )

        # 2. Extract RoPE for [text + current_image_patch] positions
        # RoPE shape is (1, 1, seq_len, head_dim/2, 2, 2), sequence is on axis 2
        text_rope = image_rotary_emb[:, :, :text_seq_len, ...]
        patch_img_rope = image_rotary_emb[
            :, :, text_seq_len + patch_start : text_seq_len + patch_end, ...
        ]
        patch_rope = mx.concatenate([text_rope, patch_img_rope], axis=2)

        # 3. Apply RoPE to Q and K
        query, key = AttentionUtils.apply_rope(xq=query, xk=key, freqs_cis=patch_rope)

        # 4. Update cache with this patch's K, V (after RoPE)
        # Cache stores full [text + image] sequence
        # Text portion: indices 0 to text_seq_len
        # Image portion: indices text_seq_len to text_seq_len + full_img_len
        kv_cache.update(
            patch_start=0,
            patch_end=text_seq_len,
            key=key[:, :, :text_seq_len, :],
            value=value[:, :, :text_seq_len, :],
        )
        kv_cache.update(
            patch_start=text_seq_len + patch_start,
            patch_end=text_seq_len + patch_end,
            key=key[:, :, text_seq_len:, :],
            value=value[:, :, text_seq_len:, :],
        )

        # 5. Get full K, V from cache
        full_key, full_value = kv_cache.get_full_kv()

        # 6. Compute attention: patch query attends to full K, V
        batch_size = norm_hidden.shape[0]
        attn_output = AttentionUtils.compute_attention(
            query=query,
            key=full_key,
            value=full_value,
            batch_size=batch_size,
            num_heads=self.num_heads,
            head_dim=self.head_dim,
        )

        return attn_output
