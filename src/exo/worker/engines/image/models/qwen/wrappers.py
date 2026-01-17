import mlx.core as mx
from mflux.models.qwen.model.qwen_transformer.qwen_attention import QwenAttention
from mflux.models.qwen.model.qwen_transformer.qwen_transformer_block import (
    QwenTransformerBlock,
)

from exo.worker.engines.image.pipeline.block_wrapper import JointBlockWrapper


class QwenJointBlockWrapper(JointBlockWrapper):
    """Qwen-specific joint block wrapper with pipefusion support.

    Qwen differs from Flux in several ways:
    - Uses modulation parameters computed from text_embeddings
    - Uses 3D RoPE with separate (cos, sin) for image and text
    - Uses attention mask for variable-length text
    """

    def __init__(
        self,
        block: QwenTransformerBlock,
        text_seq_len: int,
        encoder_hidden_states_mask: mx.array | None = None,
    ):
        super().__init__(block, text_seq_len)
        self._encoder_hidden_states_mask = encoder_hidden_states_mask

        # Cache attention parameters from block
        self._num_heads = block.attn.num_heads
        self._head_dim = block.attn.head_dim

        # Intermediate state stored between _compute_qkv and _apply_output
        self._img_mod1: mx.array | None = None
        self._img_mod2: mx.array | None = None
        self._txt_mod1: mx.array | None = None
        self._txt_mod2: mx.array | None = None
        self._img_gate1: mx.array | None = None
        self._txt_gate1: mx.array | None = None

    def set_encoder_mask(self, mask: mx.array | None) -> None:
        """Set the encoder hidden states mask for attention."""
        self._encoder_hidden_states_mask = mask

    def _compute_qkv(
        self,
        hidden_states: mx.array,
        encoder_hidden_states: mx.array,
        text_embeddings: mx.array,
        rotary_embeddings: tuple[tuple[mx.array, mx.array], tuple[mx.array, mx.array]],
    ) -> tuple[mx.array, mx.array, mx.array]:
        """Compute Q, K, V for full sequence with Qwen-specific logic."""
        batch_size = hidden_states.shape[0]
        num_img_tokens = hidden_states.shape[1]
        attn = self.block.attn

        # 1. Compute modulation parameters
        img_mod_params = self.block.img_mod_linear(
            self.block.img_mod_silu(text_embeddings)
        )
        txt_mod_params = self.block.txt_mod_linear(
            self.block.txt_mod_silu(text_embeddings)
        )

        self._img_mod1, self._img_mod2 = mx.split(img_mod_params, 2, axis=-1)
        self._txt_mod1, self._txt_mod2 = mx.split(txt_mod_params, 2, axis=-1)

        # 2. Apply normalization and modulation
        img_normed = self.block.img_norm1(hidden_states)
        img_modulated, self._img_gate1 = QwenTransformerBlock._modulate(
            img_normed, self._img_mod1
        )

        txt_normed = self.block.txt_norm1(encoder_hidden_states)
        txt_modulated, self._txt_gate1 = QwenTransformerBlock._modulate(
            txt_normed, self._txt_mod1
        )

        # 3. Compute Q, K, V for image
        img_query = attn.to_q(img_modulated)
        img_key = attn.to_k(img_modulated)
        img_value = attn.to_v(img_modulated)

        # 4. Compute Q, K, V for text
        txt_query = attn.add_q_proj(txt_modulated)
        txt_key = attn.add_k_proj(txt_modulated)
        txt_value = attn.add_v_proj(txt_modulated)

        # 5. Reshape to [B, S, H, D]
        img_query = mx.reshape(
            img_query, (batch_size, num_img_tokens, self._num_heads, self._head_dim)
        )
        img_key = mx.reshape(
            img_key, (batch_size, num_img_tokens, self._num_heads, self._head_dim)
        )
        img_value = mx.reshape(
            img_value, (batch_size, num_img_tokens, self._num_heads, self._head_dim)
        )

        txt_query = mx.reshape(
            txt_query,
            (batch_size, self._text_seq_len, self._num_heads, self._head_dim),
        )
        txt_key = mx.reshape(
            txt_key, (batch_size, self._text_seq_len, self._num_heads, self._head_dim)
        )
        txt_value = mx.reshape(
            txt_value, (batch_size, self._text_seq_len, self._num_heads, self._head_dim)
        )

        # 6. Apply RMSNorm to Q, K
        img_query = attn.norm_q(img_query)
        img_key = attn.norm_k(img_key)
        txt_query = attn.norm_added_q(txt_query)
        txt_key = attn.norm_added_k(txt_key)

        # 7. Apply RoPE (Qwen uses 3D RoPE with separate embeddings)
        (img_cos, img_sin), (txt_cos, txt_sin) = rotary_embeddings

        img_query = QwenAttention._apply_rope_qwen(img_query, img_cos, img_sin)
        img_key = QwenAttention._apply_rope_qwen(img_key, img_cos, img_sin)
        txt_query = QwenAttention._apply_rope_qwen(txt_query, txt_cos, txt_sin)
        txt_key = QwenAttention._apply_rope_qwen(txt_key, txt_cos, txt_sin)

        # 8. Transpose to [B, H, S, D] for attention
        img_query = mx.transpose(img_query, (0, 2, 1, 3))
        img_key = mx.transpose(img_key, (0, 2, 1, 3))
        img_value = mx.transpose(img_value, (0, 2, 1, 3))

        txt_query = mx.transpose(txt_query, (0, 2, 1, 3))
        txt_key = mx.transpose(txt_key, (0, 2, 1, 3))
        txt_value = mx.transpose(txt_value, (0, 2, 1, 3))

        # 9. Concatenate [text, image]
        query = mx.concatenate([txt_query, img_query], axis=2)
        key = mx.concatenate([txt_key, img_key], axis=2)
        value = mx.concatenate([txt_value, img_value], axis=2)

        return query, key, value

    def _compute_patch_qkv(
        self,
        patch_hidden: mx.array,
        encoder_hidden_states: mx.array,
        text_embeddings: mx.array,
        rotary_embeddings: tuple[tuple[mx.array, mx.array], tuple[mx.array, mx.array]],
    ) -> tuple[mx.array, mx.array, mx.array]:
        """Compute Q, K, V for [text + patch] with sliced RoPE."""
        batch_size = patch_hidden.shape[0]
        patch_len = patch_hidden.shape[1]
        attn = self.block.attn

        # 1. Compute modulation parameters
        img_mod_params = self.block.img_mod_linear(
            self.block.img_mod_silu(text_embeddings)
        )
        txt_mod_params = self.block.txt_mod_linear(
            self.block.txt_mod_silu(text_embeddings)
        )

        self._img_mod1, self._img_mod2 = mx.split(img_mod_params, 2, axis=-1)
        self._txt_mod1, self._txt_mod2 = mx.split(txt_mod_params, 2, axis=-1)

        # 2. Apply normalization and modulation
        img_normed = self.block.img_norm1(patch_hidden)
        img_modulated, self._img_gate1 = QwenTransformerBlock._modulate(
            img_normed, self._img_mod1
        )

        txt_normed = self.block.txt_norm1(encoder_hidden_states)
        txt_modulated, self._txt_gate1 = QwenTransformerBlock._modulate(
            txt_normed, self._txt_mod1
        )

        # 3. Compute Q, K, V for image patch
        img_query = attn.to_q(img_modulated)
        img_key = attn.to_k(img_modulated)
        img_value = attn.to_v(img_modulated)

        # 4. Compute Q, K, V for text
        txt_query = attn.add_q_proj(txt_modulated)
        txt_key = attn.add_k_proj(txt_modulated)
        txt_value = attn.add_v_proj(txt_modulated)

        # 5. Reshape to [B, S, H, D]
        img_query = mx.reshape(
            img_query, (batch_size, patch_len, self._num_heads, self._head_dim)
        )
        img_key = mx.reshape(
            img_key, (batch_size, patch_len, self._num_heads, self._head_dim)
        )
        img_value = mx.reshape(
            img_value, (batch_size, patch_len, self._num_heads, self._head_dim)
        )

        txt_query = mx.reshape(
            txt_query,
            (batch_size, self._text_seq_len, self._num_heads, self._head_dim),
        )
        txt_key = mx.reshape(
            txt_key, (batch_size, self._text_seq_len, self._num_heads, self._head_dim)
        )
        txt_value = mx.reshape(
            txt_value, (batch_size, self._text_seq_len, self._num_heads, self._head_dim)
        )

        # 6. Apply RMSNorm to Q, K
        img_query = attn.norm_q(img_query)
        img_key = attn.norm_k(img_key)
        txt_query = attn.norm_added_q(txt_query)
        txt_key = attn.norm_added_k(txt_key)

        # 7. Extract RoPE for patch: slice image RoPE, keep full text RoPE
        (img_cos, img_sin), (txt_cos, txt_sin) = rotary_embeddings
        patch_img_cos = img_cos[self._patch_start : self._patch_end]
        patch_img_sin = img_sin[self._patch_start : self._patch_end]

        # 8. Apply RoPE
        img_query = QwenAttention._apply_rope_qwen(
            img_query, patch_img_cos, patch_img_sin
        )
        img_key = QwenAttention._apply_rope_qwen(img_key, patch_img_cos, patch_img_sin)
        txt_query = QwenAttention._apply_rope_qwen(txt_query, txt_cos, txt_sin)
        txt_key = QwenAttention._apply_rope_qwen(txt_key, txt_cos, txt_sin)

        # 9. Transpose to [B, H, S, D] for attention
        img_query = mx.transpose(img_query, (0, 2, 1, 3))
        img_key = mx.transpose(img_key, (0, 2, 1, 3))
        img_value = mx.transpose(img_value, (0, 2, 1, 3))

        txt_query = mx.transpose(txt_query, (0, 2, 1, 3))
        txt_key = mx.transpose(txt_key, (0, 2, 1, 3))
        txt_value = mx.transpose(txt_value, (0, 2, 1, 3))

        # 10. Concatenate [text, patch]
        query = mx.concatenate([txt_query, img_query], axis=2)
        key = mx.concatenate([txt_key, img_key], axis=2)
        value = mx.concatenate([txt_value, img_value], axis=2)

        return query, key, value

    def _compute_attention(
        self, query: mx.array, key: mx.array, value: mx.array
    ) -> mx.array:
        """Compute scaled dot-product attention with Qwen-specific mask."""
        attn = self.block.attn

        # Build attention mask
        mask = QwenAttention._convert_mask_for_qwen(
            mask=self._encoder_hidden_states_mask,
            joint_seq_len=key.shape[2],
            txt_seq_len=self._text_seq_len,
        )

        # Transpose back to [B, S, H, D] for Qwen's attention
        query_bshd = mx.transpose(query, (0, 2, 1, 3))
        key_bshd = mx.transpose(key, (0, 2, 1, 3))
        value_bshd = mx.transpose(value, (0, 2, 1, 3))

        return attn._compute_attention_qwen(
            query=query_bshd,
            key=key_bshd,
            value=value_bshd,
            mask=mask,
            block_idx=None,
        )

    def _apply_output(
        self,
        attn_out: mx.array,
        hidden_states: mx.array,
        encoder_hidden_states: mx.array,
        text_embeddings: mx.array,
    ) -> tuple[mx.array, mx.array]:
        """Apply output projection, feed-forward, and residuals."""
        attn = self.block.attn

        # 1. Extract text and image attention outputs
        txt_attn_output = attn_out[:, : self._text_seq_len, :]
        img_attn_output = attn_out[:, self._text_seq_len :, :]

        # 2. Project outputs
        img_attn_output = attn.attn_to_out[0](img_attn_output)
        txt_attn_output = attn.to_add_out(txt_attn_output)

        # 3. Apply residual + gate for attention
        hidden_states = hidden_states + self._img_gate1 * img_attn_output
        encoder_hidden_states = (
            encoder_hidden_states + self._txt_gate1 * txt_attn_output
        )

        # 4. Apply feed-forward for image
        img_normed2 = self.block.img_norm2(hidden_states)
        img_modulated2, img_gate2 = QwenTransformerBlock._modulate(
            img_normed2, self._img_mod2
        )
        img_mlp_output = self.block.img_ff(img_modulated2)
        hidden_states = hidden_states + img_gate2 * img_mlp_output

        # 5. Apply feed-forward for text
        txt_normed2 = self.block.txt_norm2(encoder_hidden_states)
        txt_modulated2, txt_gate2 = QwenTransformerBlock._modulate(
            txt_normed2, self._txt_mod2
        )
        txt_mlp_output = self.block.txt_ff(txt_modulated2)
        encoder_hidden_states = encoder_hidden_states + txt_gate2 * txt_mlp_output

        return encoder_hidden_states, hidden_states
