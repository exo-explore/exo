import mlx.core as mx
from mflux.models.flux.model.flux_transformer.common.attention_utils import (
    AttentionUtils,
)
from mflux.models.flux.model.flux_transformer.joint_attention import JointAttention
from mflux.models.flux.model.flux_transformer.joint_transformer_block import (
    JointTransformerBlock,
)
from mflux.models.flux.model.flux_transformer.single_block_attention import (
    SingleBlockAttention,
)
from mflux.models.flux.model.flux_transformer.single_transformer_block import (
    SingleTransformerBlock,
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


class PatchedJointTransformerBlock:
    """Joint transformer block with KV caching for patch-based processing.

    Wraps a JointTransformerBlock to process image patches while using
    cached K/V values for the full sequence (text + all image patches).
    """

    def __init__(self, block: JointTransformerBlock):
        """Wrap an existing JointTransformerBlock.

        Args:
            block: The original JointTransformerBlock (for weight access)
        """
        self.block = block
        self.cached_attn = CachedJointAttention(block.attn)

    def __call__(
        self,
        patch_hidden: mx.array,
        encoder_hidden_states: mx.array,
        text_embeddings: mx.array,
        image_rotary_emb: mx.array,
        kv_cache: JointPatchKVCache,
        patch_start: int,
        patch_end: int,
        text_seq_len: int,
    ) -> tuple[mx.array, mx.array]:
        """Forward pass with KV caching for patch-based processing.

        Args:
            patch_hidden: Image patch hidden states [B, patch_img_len, D]
            encoder_hidden_states: Full text hidden states [B, text_len, D]
            text_embeddings: Time + pooled text conditioning
            image_rotary_emb: Full rotary embeddings for [text + full_image]
            kv_cache: KV cache to update and read from
            patch_start: Start token index of this patch in image sequence
            patch_end: End token index of this patch in image sequence
            text_seq_len: Length of text sequence

        Returns:
            Tuple of (encoder_hidden_states, patch_hidden) after block processing
        """
        # 1. Compute norms
        norm_hidden, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.block.norm1(
            hidden_states=patch_hidden,
            text_embeddings=text_embeddings,
        )
        (
            norm_encoder,
            c_gate_msa,
            c_shift_mlp,
            c_scale_mlp,
            c_gate_mlp,
        ) = self.block.norm1_context(
            hidden_states=encoder_hidden_states,
            text_embeddings=text_embeddings,
        )

        # 2. Compute attention with KV cache
        attn_output, context_attn_output = self.cached_attn(
            norm_hidden=norm_hidden,
            norm_encoder=norm_encoder,
            image_rotary_emb=image_rotary_emb,
            kv_cache=kv_cache,
            patch_start=patch_start,
            patch_end=patch_end,
            text_seq_len=text_seq_len,
        )

        # 3. Apply norm and feed forward
        patch_hidden = JointTransformerBlock.apply_norm_and_feed_forward(
            hidden_states=patch_hidden,
            attn_output=attn_output,
            gate_mlp=gate_mlp,
            gate_msa=gate_msa,
            scale_mlp=scale_mlp,
            shift_mlp=shift_mlp,
            norm_layer=self.block.norm2,
            ff_layer=self.block.ff,
        )
        encoder_hidden_states = JointTransformerBlock.apply_norm_and_feed_forward(
            hidden_states=encoder_hidden_states,
            attn_output=context_attn_output,
            gate_mlp=c_gate_mlp,
            gate_msa=c_gate_msa,
            scale_mlp=c_scale_mlp,
            shift_mlp=c_shift_mlp,
            norm_layer=self.block.norm2_context,
            ff_layer=self.block.ff_context,
        )

        return encoder_hidden_states, patch_hidden


class PatchedSingleTransformerBlock:
    """Single transformer block with KV caching for patch-based processing.

    Wraps a SingleTransformerBlock to process patches of concatenated [text + image]
    while using cached K/V values for the full sequence.
    """

    def __init__(self, block: SingleTransformerBlock):
        """Wrap an existing SingleTransformerBlock.

        Args:
            block: The original SingleTransformerBlock (for weight access)
        """
        self.block = block
        self.cached_attn = CachedSingleBlockAttention(block.attn)

    def __call__(
        self,
        patch_hidden: mx.array,
        text_embeddings: mx.array,
        image_rotary_emb: mx.array,
        kv_cache: PatchKVCache,
        patch_start: int,
        patch_end: int,
        text_seq_len: int,
    ) -> mx.array:
        """Forward pass with KV caching for patch-based processing.

        Args:
            patch_hidden: Patch of [text + image] hidden states [B, text_len + patch_img_len, D]
            text_embeddings: Time + pooled text conditioning
            image_rotary_emb: Full rotary embeddings for [text + full_image]
            kv_cache: KV cache to update and read from
            patch_start: Start token index of image patch (within image portion)
            patch_end: End token index of image patch (within image portion)
            text_seq_len: Length of text portion in hidden_states

        Returns:
            Output hidden states [B, text_len + patch_img_len, D]
        """
        # 0. Establish residual connection
        residual = patch_hidden

        # 1. Compute norm
        norm_hidden, gate = self.block.norm(
            hidden_states=patch_hidden,
            text_embeddings=text_embeddings,
        )

        # 2. Compute attention with KV cache
        attn_output = self.cached_attn(
            norm_hidden=norm_hidden,
            image_rotary_emb=image_rotary_emb,
            kv_cache=kv_cache,
            patch_start=patch_start,
            patch_end=patch_end,
            text_seq_len=text_seq_len,
        )

        # 3. Apply feed forward and projection
        hidden_states = self.block._apply_feed_forward_and_projection(
            norm_hidden_states=norm_hidden,
            attn_output=attn_output,
            gate=gate,
        )

        return residual + hidden_states


class CachingJointTransformerBlock:
    """Joint transformer block that captures K/V for cache during sync mode.

    Runs full (non-patched) attention but stores K/V in the cache for
    subsequent async timesteps to use as stale values.
    """

    def __init__(self, block: JointTransformerBlock, kv_cache: JointPatchKVCache):
        """Wrap an existing JointTransformerBlock with a KV cache.

        Args:
            block: The original JointTransformerBlock
            kv_cache: KV cache to populate during forward pass
        """
        self.block = block
        self.kv_cache = kv_cache

    def __call__(
        self,
        hidden_states: mx.array,
        encoder_hidden_states: mx.array,
        text_embeddings: mx.array,
        rotary_embeddings: mx.array,
    ) -> tuple[mx.array, mx.array]:
        """Forward pass that also populates the KV cache.

        Args:
            hidden_states: Full image hidden states [B, img_len, D]
            encoder_hidden_states: Full text hidden states [B, text_len, D]
            text_embeddings: Time + pooled text conditioning
            rotary_embeddings: Full rotary embeddings for [text + full_image]

        Returns:
            Tuple of (encoder_hidden_states, hidden_states) after block processing
        """
        # Run standard block (computes full attention)
        encoder_out, hidden_out = self.block(
            hidden_states=hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            text_embeddings=text_embeddings,
            rotary_embeddings=rotary_embeddings,
        )

        # Populate KV cache for async pipeline warmstart
        self._populate_cache(
            hidden_states, encoder_hidden_states, text_embeddings, rotary_embeddings
        )

        return encoder_out, hidden_out

    def _populate_cache(
        self,
        hidden_states: mx.array,
        encoder_hidden_states: mx.array,
        text_embeddings: mx.array,
        rotary_embeddings: mx.array,
    ) -> None:
        """Compute and store K/V in cache for async pipeline warmstart."""
        attn = self.block.attn
        text_seq_len = encoder_hidden_states.shape[1]
        num_img_tokens = hidden_states.shape[1]

        # Get normalized inputs (same as what attention would see)
        norm_hidden, *_ = self.block.norm1(
            hidden_states=hidden_states,
            text_embeddings=text_embeddings,
        )
        norm_encoder, *_ = self.block.norm1_context(
            hidden_states=encoder_hidden_states,
            text_embeddings=text_embeddings,
        )

        # Compute K, V for image (full, not patched)
        _, img_key, img_value = AttentionUtils.process_qkv(
            hidden_states=norm_hidden,
            to_q=attn.to_q,
            to_k=attn.to_k,
            to_v=attn.to_v,
            norm_q=attn.norm_q,
            norm_k=attn.norm_k,
            num_heads=attn.num_heads,
            head_dim=attn.head_dimension,
        )

        # Compute K, V for text
        _, txt_key, txt_value = AttentionUtils.process_qkv(
            hidden_states=norm_encoder,
            to_q=attn.add_q_proj,
            to_k=attn.add_k_proj,
            to_v=attn.add_v_proj,
            norm_q=attn.norm_added_q,
            norm_k=attn.norm_added_k,
            num_heads=attn.num_heads,
            head_dim=attn.head_dimension,
        )

        # Concatenate and apply RoPE
        full_key = mx.concatenate([txt_key, img_key], axis=2)
        full_value = mx.concatenate([txt_value, img_value], axis=2)
        _, full_key = AttentionUtils.apply_rope(
            xq=full_key, xk=full_key, freqs_cis=rotary_embeddings
        )

        # Store full sequence in cache
        self.kv_cache.update_text(
            key=full_key[:, :, :text_seq_len, :],
            value=full_value[:, :, :text_seq_len, :],
        )
        self.kv_cache.update_image_patch(
            patch_start=0,
            patch_end=num_img_tokens,
            key=full_key[:, :, text_seq_len:, :],
            value=full_value[:, :, text_seq_len:, :],
        )


class CachingSingleTransformerBlock:
    """Single transformer block that captures K/V for cache during sync mode.

    Runs full (non-patched) attention but stores K/V in the cache for
    subsequent async timesteps to use as stale values.
    """

    def __init__(self, block: SingleTransformerBlock, kv_cache: PatchKVCache):
        """Wrap an existing SingleTransformerBlock with a KV cache.

        Args:
            block: The original SingleTransformerBlock
            kv_cache: KV cache to populate during forward pass
        """
        self.block = block
        self.kv_cache = kv_cache

    def __call__(
        self,
        hidden_states: mx.array,
        text_embeddings: mx.array,
        rotary_embeddings: mx.array,
    ) -> mx.array:
        """Forward pass that also populates the KV cache.

        Args:
            hidden_states: Full [text + image] hidden states [B, text_len + img_len, D]
            text_embeddings: Time + pooled text conditioning
            rotary_embeddings: Full rotary embeddings for [text + full_image]

        Returns:
            Output hidden states after block processing
        """
        # Run standard block
        hidden_out = self.block(
            hidden_states=hidden_states,
            text_embeddings=text_embeddings,
            rotary_embeddings=rotary_embeddings,
        )

        # Populate KV cache for async pipeline warmstart
        self._populate_cache(hidden_states, text_embeddings, rotary_embeddings)

        return hidden_out

    def _populate_cache(
        self,
        hidden_states: mx.array,
        text_embeddings: mx.array,
        rotary_embeddings: mx.array,
    ) -> None:
        """Compute and store K/V in cache for async pipeline warmstart."""
        attn = self.block.attn
        total_seq_len = hidden_states.shape[1]

        # Get normalized inputs
        norm_hidden, _ = self.block.norm(
            hidden_states=hidden_states,
            text_embeddings=text_embeddings,
        )

        # Compute K, V
        _, key, value = AttentionUtils.process_qkv(
            hidden_states=norm_hidden,
            to_q=attn.to_q,
            to_k=attn.to_k,
            to_v=attn.to_v,
            norm_q=attn.norm_q,
            norm_k=attn.norm_k,
            num_heads=attn.num_heads,
            head_dim=attn.head_dimension,
        )

        # Apply RoPE
        _, key = AttentionUtils.apply_rope(
            xq=key, xk=key, freqs_cis=rotary_embeddings
        )

        # Store full sequence in cache
        self.kv_cache.update(
            patch_start=0,
            patch_end=total_seq_len,
            key=key,
            value=value,
        )
