from typing import final

import mlx.core as mx
from mflux.models.qwen.model.qwen_transformer.qwen_attention import QwenAttention
from mflux.models.qwen.model.qwen_transformer.qwen_transformer_block import (
    QwenTransformerBlock,
)
from pydantic import BaseModel, ConfigDict

from exo.worker.engines.image.models.base import RotaryEmbeddings
from exo.worker.engines.image.pipeline.block_wrapper import JointBlockWrapper


@final
class QwenStreamModulation(BaseModel):
    model_config = ConfigDict(frozen=True, strict=True, arbitrary_types_allowed=True)

    mod1: mx.array
    mod2: mx.array
    gate1: mx.array


class QwenJointBlockWrapper(JointBlockWrapper[QwenTransformerBlock]):
    def __init__(
        self,
        block: QwenTransformerBlock,
        text_seq_len: int,
        encoder_hidden_states_mask: mx.array | None = None,
    ):
        super().__init__(block, text_seq_len)
        self._encoder_hidden_states_mask = encoder_hidden_states_mask

        self._num_heads = block.attn.num_heads
        self._head_dim = block.attn.head_dim

        # Intermediate state stored between _compute_qkv and _apply_output
        self._img_mod: QwenStreamModulation | None = None
        self._txt_mod: QwenStreamModulation | None = None

    def set_encoder_mask(self, mask: mx.array | None) -> None:
        """Set the encoder hidden states mask for attention."""
        self._encoder_hidden_states_mask = mask

    def _compute_qkv(
        self,
        hidden_states: mx.array,
        encoder_hidden_states: mx.array,
        text_embeddings: mx.array,
        rotary_embeddings: RotaryEmbeddings,
        patch_mode: bool = False,
    ) -> tuple[mx.array, mx.array, mx.array]:
        assert isinstance(rotary_embeddings, tuple)

        batch_size = hidden_states.shape[0]
        img_seq_len = hidden_states.shape[1]
        attn = self.block.attn

        img_mod_params = self.block.img_mod_linear(
            self.block.img_mod_silu(text_embeddings)  # pyright: ignore[reportUnknownArgumentType]
        )
        txt_mod_params = self.block.txt_mod_linear(
            self.block.txt_mod_silu(text_embeddings)  # pyright: ignore[reportUnknownArgumentType]
        )

        img_mod1, img_mod2 = mx.split(img_mod_params, 2, axis=-1)
        txt_mod1, txt_mod2 = mx.split(txt_mod_params, 2, axis=-1)

        img_normed = self.block.img_norm1(hidden_states)
        img_modulated, img_gate1 = QwenTransformerBlock._modulate(img_normed, img_mod1)
        self._img_mod = QwenStreamModulation(
            mod1=img_mod1, mod2=img_mod2, gate1=img_gate1
        )

        txt_normed = self.block.txt_norm1(encoder_hidden_states)
        txt_modulated, txt_gate1 = QwenTransformerBlock._modulate(txt_normed, txt_mod1)
        self._txt_mod = QwenStreamModulation(
            mod1=txt_mod1, mod2=txt_mod2, gate1=txt_gate1
        )

        img_query = attn.to_q(img_modulated)
        img_key = attn.to_k(img_modulated)
        img_value = attn.to_v(img_modulated)

        txt_query = attn.add_q_proj(txt_modulated)
        txt_key = attn.add_k_proj(txt_modulated)
        txt_value = attn.add_v_proj(txt_modulated)

        img_query = mx.reshape(
            img_query, (batch_size, img_seq_len, self._num_heads, self._head_dim)
        )
        img_key = mx.reshape(
            img_key, (batch_size, img_seq_len, self._num_heads, self._head_dim)
        )
        img_value = mx.reshape(
            img_value, (batch_size, img_seq_len, self._num_heads, self._head_dim)
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

        img_query = attn.norm_q(img_query)
        img_key = attn.norm_k(img_key)
        txt_query = attn.norm_added_q(txt_query)
        txt_key = attn.norm_added_k(txt_key)

        (img_cos, img_sin), (txt_cos, txt_sin) = rotary_embeddings

        if patch_mode:
            # Slice image RoPE for patch, keep full text RoPE
            img_cos = img_cos[self._patch_start : self._patch_end]
            img_sin = img_sin[self._patch_start : self._patch_end]

        img_query = QwenAttention._apply_rope_qwen(img_query, img_cos, img_sin)
        img_key = QwenAttention._apply_rope_qwen(img_key, img_cos, img_sin)
        txt_query = QwenAttention._apply_rope_qwen(txt_query, txt_cos, txt_sin)
        txt_key = QwenAttention._apply_rope_qwen(txt_key, txt_cos, txt_sin)

        img_query = mx.transpose(img_query, (0, 2, 1, 3))
        img_key = mx.transpose(img_key, (0, 2, 1, 3))
        img_value = mx.transpose(img_value, (0, 2, 1, 3))

        txt_query = mx.transpose(txt_query, (0, 2, 1, 3))
        txt_key = mx.transpose(txt_key, (0, 2, 1, 3))
        txt_value = mx.transpose(txt_value, (0, 2, 1, 3))

        query = mx.concatenate([txt_query, img_query], axis=2)
        key = mx.concatenate([txt_key, img_key], axis=2)
        value = mx.concatenate([txt_value, img_value], axis=2)

        return query, key, value

    def _compute_attention(
        self, query: mx.array, key: mx.array, value: mx.array
    ) -> mx.array:
        attn = self.block.attn

        mask = QwenAttention._convert_mask_for_qwen(
            mask=self._encoder_hidden_states_mask,
            joint_seq_len=key.shape[2],
            txt_seq_len=self._text_seq_len,
        )

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
        attn = self.block.attn

        assert self._img_mod is not None
        assert self._txt_mod is not None

        txt_attn_output = attn_out[:, : self._text_seq_len, :]
        img_attn_output = attn_out[:, self._text_seq_len :, :]

        img_attn_output = attn.attn_to_out[0](img_attn_output)  # pyright: ignore[reportAny]
        txt_attn_output = attn.to_add_out(txt_attn_output)

        hidden_states = hidden_states + self._img_mod.gate1 * img_attn_output  # pyright: ignore[reportAny]
        encoder_hidden_states = (
            encoder_hidden_states + self._txt_mod.gate1 * txt_attn_output
        )

        img_normed2 = self.block.img_norm2(hidden_states)
        img_modulated2, img_gate2 = QwenTransformerBlock._modulate(
            img_normed2, self._img_mod.mod2
        )
        img_mlp_output = self.block.img_ff(img_modulated2)  # pyright: ignore[reportAny]
        hidden_states = hidden_states + img_gate2 * img_mlp_output  # pyright: ignore[reportAny]

        txt_normed2 = self.block.txt_norm2(encoder_hidden_states)
        txt_modulated2, txt_gate2 = QwenTransformerBlock._modulate(
            txt_normed2, self._txt_mod.mod2
        )
        txt_mlp_output = self.block.txt_ff(txt_modulated2)  # pyright: ignore[reportAny]
        encoder_hidden_states = encoder_hidden_states + txt_gate2 * txt_mlp_output  # pyright: ignore[reportAny]

        return encoder_hidden_states, hidden_states
