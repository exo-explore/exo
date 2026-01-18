import mlx.core as mx
from mflux.models.flux.model.flux_transformer.common.attention_utils import (
    AttentionUtils,
)
from mflux.models.flux.model.flux_transformer.joint_transformer_block import (
    JointTransformerBlock,
)
from mflux.models.flux.model.flux_transformer.single_transformer_block import (
    SingleTransformerBlock,
)

from exo.worker.engines.image.pipeline.block_wrapper import (
    JointBlockWrapper,
    SingleBlockWrapper,
)


class FluxJointBlockWrapper(JointBlockWrapper):
    """Flux-specific joint block wrapper with pipefusion support."""

    def __init__(self, block: JointTransformerBlock, text_seq_len: int):
        super().__init__(block, text_seq_len)
        # Cache attention parameters from block
        self._num_heads = block.attn.num_heads
        self._head_dim = block.attn.head_dimension

        # Intermediate state stored between _compute_qkv and _apply_output
        self._gate_msa: mx.array | None = None
        self._shift_mlp: mx.array | None = None
        self._scale_mlp: mx.array | None = None
        self._gate_mlp: mx.array | None = None
        self._c_gate_msa: mx.array | None = None
        self._c_shift_mlp: mx.array | None = None
        self._c_scale_mlp: mx.array | None = None
        self._c_gate_mlp: mx.array | None = None

    def _compute_qkv(
        self,
        hidden_states: mx.array,
        encoder_hidden_states: mx.array,
        text_embeddings: mx.array,
        rotary_embeddings: mx.array,
        patch_mode: bool = False,
    ) -> tuple[mx.array, mx.array, mx.array]:
        """Compute Q, K, V for sequence with Flux-specific logic.

        Args:
            hidden_states: Image hidden states [B, num_img_tokens, D] or patch [B, patch_len, D]
            encoder_hidden_states: Text hidden states [B, text_seq_len, D]
            text_embeddings: Conditioning embeddings [B, D]
            rotary_embeddings: Rotary position embeddings
            patch_mode: If True, slice RoPE for current patch range
        """
        attn = self.block.attn

        # 1. Compute norms (store gates for _apply_output)
        (
            norm_hidden,
            self._gate_msa,
            self._shift_mlp,
            self._scale_mlp,
            self._gate_mlp,
        ) = self.block.norm1(
            hidden_states=hidden_states,
            text_embeddings=text_embeddings,
        )
        (
            norm_encoder,
            self._c_gate_msa,
            self._c_shift_mlp,
            self._c_scale_mlp,
            self._c_gate_mlp,
        ) = self.block.norm1_context(
            hidden_states=encoder_hidden_states,
            text_embeddings=text_embeddings,
        )

        # 2. Compute Q, K, V for image
        img_query, img_key, img_value = AttentionUtils.process_qkv(
            hidden_states=norm_hidden,
            to_q=attn.to_q,
            to_k=attn.to_k,
            to_v=attn.to_v,
            norm_q=attn.norm_q,
            norm_k=attn.norm_k,
            num_heads=self._num_heads,
            head_dim=self._head_dim,
        )

        # 3. Compute Q, K, V for text
        txt_query, txt_key, txt_value = AttentionUtils.process_qkv(
            hidden_states=norm_encoder,
            to_q=attn.add_q_proj,
            to_k=attn.add_k_proj,
            to_v=attn.add_v_proj,
            norm_q=attn.norm_added_q,
            norm_k=attn.norm_added_k,
            num_heads=self._num_heads,
            head_dim=self._head_dim,
        )

        # 4. Concatenate Q, K, V: [text, image/patch]
        query = mx.concatenate([txt_query, img_query], axis=2)
        key = mx.concatenate([txt_key, img_key], axis=2)
        value = mx.concatenate([txt_value, img_value], axis=2)

        # 5. Apply RoPE (slice for patch mode)
        if patch_mode:
            text_rope = rotary_embeddings[:, :, : self._text_seq_len, ...]
            patch_img_rope = rotary_embeddings[
                :,
                :,
                self._text_seq_len + self._patch_start : self._text_seq_len
                + self._patch_end,
                ...,
            ]
            rope = mx.concatenate([text_rope, patch_img_rope], axis=2)
        else:
            rope = rotary_embeddings

        query, key = AttentionUtils.apply_rope(xq=query, xk=key, freqs_cis=rope)

        return query, key, value

    def _compute_attention(
        self, query: mx.array, key: mx.array, value: mx.array
    ) -> mx.array:
        """Compute scaled dot-product attention."""
        batch_size = query.shape[0]
        return AttentionUtils.compute_attention(
            query=query,
            key=key,
            value=value,
            batch_size=batch_size,
            num_heads=self._num_heads,
            head_dim=self._head_dim,
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
        context_attn_output = attn_out[:, : self._text_seq_len, :]
        hidden_attn_output = attn_out[:, self._text_seq_len :, :]

        # 2. Project outputs
        hidden_attn_output = attn.to_out[0](hidden_attn_output)
        context_attn_output = attn.to_add_out(context_attn_output)

        # 3. Apply norm and feed forward (using stored gates)
        hidden_states = JointTransformerBlock.apply_norm_and_feed_forward(
            hidden_states=hidden_states,
            attn_output=hidden_attn_output,
            gate_mlp=self._gate_mlp,
            gate_msa=self._gate_msa,
            scale_mlp=self._scale_mlp,
            shift_mlp=self._shift_mlp,
            norm_layer=self.block.norm2,
            ff_layer=self.block.ff,
        )
        encoder_hidden_states = JointTransformerBlock.apply_norm_and_feed_forward(
            hidden_states=encoder_hidden_states,
            attn_output=context_attn_output,
            gate_mlp=self._c_gate_mlp,
            gate_msa=self._c_gate_msa,
            scale_mlp=self._c_scale_mlp,
            shift_mlp=self._c_shift_mlp,
            norm_layer=self.block.norm2_context,
            ff_layer=self.block.ff_context,
        )

        return encoder_hidden_states, hidden_states


class FluxSingleBlockWrapper(SingleBlockWrapper):
    """Flux-specific single block wrapper with pipefusion support."""

    def __init__(self, block: SingleTransformerBlock, text_seq_len: int):
        super().__init__(block, text_seq_len)
        # Cache attention parameters from block
        self._num_heads = block.attn.num_heads
        self._head_dim = block.attn.head_dimension

        # Intermediate state stored between _compute_qkv and _apply_output
        self._gate: mx.array | None = None
        self._norm_hidden: mx.array | None = None

    def _compute_qkv(
        self,
        hidden_states: mx.array,
        text_embeddings: mx.array,
        rotary_embeddings: mx.array,
        patch_mode: bool = False,
    ) -> tuple[mx.array, mx.array, mx.array]:
        """Compute Q, K, V for [text, image] sequence.

        Args:
            hidden_states: Concatenated [text, image] hidden states
            text_embeddings: Conditioning embeddings [B, D]
            rotary_embeddings: Rotary position embeddings
            patch_mode: If True, slice RoPE for current patch range
        """
        attn = self.block.attn

        # 1. Compute norm (store for _apply_output)
        self._norm_hidden, self._gate = self.block.norm(
            hidden_states=hidden_states,
            text_embeddings=text_embeddings,
        )

        # 2. Compute Q, K, V
        query, key, value = AttentionUtils.process_qkv(
            hidden_states=self._norm_hidden,
            to_q=attn.to_q,
            to_k=attn.to_k,
            to_v=attn.to_v,
            norm_q=attn.norm_q,
            norm_k=attn.norm_k,
            num_heads=self._num_heads,
            head_dim=self._head_dim,
        )

        # 3. Apply RoPE (slice for patch mode)
        if patch_mode:
            text_rope = rotary_embeddings[:, :, : self._text_seq_len, ...]
            patch_img_rope = rotary_embeddings[
                :,
                :,
                self._text_seq_len + self._patch_start : self._text_seq_len
                + self._patch_end,
                ...,
            ]
            rope = mx.concatenate([text_rope, patch_img_rope], axis=2)
        else:
            rope = rotary_embeddings

        query, key = AttentionUtils.apply_rope(xq=query, xk=key, freqs_cis=rope)

        return query, key, value

    def _compute_attention(
        self, query: mx.array, key: mx.array, value: mx.array
    ) -> mx.array:
        """Compute scaled dot-product attention."""
        batch_size = query.shape[0]
        return AttentionUtils.compute_attention(
            query=query,
            key=key,
            value=value,
            batch_size=batch_size,
            num_heads=self._num_heads,
            head_dim=self._head_dim,
        )

    def _apply_output(
        self,
        attn_out: mx.array,
        hidden_states: mx.array,
        text_embeddings: mx.array,
    ) -> mx.array:
        """Apply feed forward and projection with residual."""
        # Residual from original hidden_states
        residual = hidden_states

        # Apply feed forward and projection (using stored norm and gate)
        output = self.block._apply_feed_forward_and_projection(
            norm_hidden_states=self._norm_hidden,
            attn_output=attn_out,
            gate=self._gate,
        )

        return residual + output
