from typing import final

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
from pydantic import BaseModel, ConfigDict

from exo.worker.engines.image.models.base import RotaryEmbeddings
from exo.worker.engines.image.pipeline.block_wrapper import (
    JointBlockWrapper,
    SingleBlockWrapper,
)


@final
class FluxModulationParams(BaseModel):
    model_config = ConfigDict(frozen=True, strict=True, arbitrary_types_allowed=True)

    gate_msa: mx.array
    shift_mlp: mx.array
    scale_mlp: mx.array
    gate_mlp: mx.array


@final
class FluxNormGateState(BaseModel):
    model_config = ConfigDict(frozen=True, strict=True, arbitrary_types_allowed=True)

    norm_hidden: mx.array
    gate: mx.array


class FluxJointBlockWrapper(JointBlockWrapper[JointTransformerBlock]):
    def __init__(self, block: JointTransformerBlock, text_seq_len: int):
        super().__init__(block, text_seq_len)
        self._num_heads = block.attn.num_heads
        self._head_dim = block.attn.head_dimension

        # Intermediate state stored between _compute_qkv and _apply_output
        self._hidden_mod: FluxModulationParams | None = None
        self._context_mod: FluxModulationParams | None = None

    def _compute_qkv(
        self,
        hidden_states: mx.array,
        encoder_hidden_states: mx.array,
        text_embeddings: mx.array,
        rotary_embeddings: RotaryEmbeddings,
        patch_mode: bool = False,
    ) -> tuple[mx.array, mx.array, mx.array]:
        assert isinstance(rotary_embeddings, mx.array)

        attn = self.block.attn

        (
            norm_hidden,
            gate_msa,
            shift_mlp,
            scale_mlp,
            gate_mlp,
        ) = self.block.norm1(
            hidden_states=hidden_states,
            text_embeddings=text_embeddings,
        )
        self._hidden_mod = FluxModulationParams(
            gate_msa=gate_msa,
            shift_mlp=shift_mlp,
            scale_mlp=scale_mlp,
            gate_mlp=gate_mlp,
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
        self._context_mod = FluxModulationParams(
            gate_msa=c_gate_msa,
            shift_mlp=c_shift_mlp,
            scale_mlp=c_scale_mlp,
            gate_mlp=c_gate_mlp,
        )

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

        query = mx.concatenate([txt_query, img_query], axis=2)
        key = mx.concatenate([txt_key, img_key], axis=2)
        value = mx.concatenate([txt_value, img_value], axis=2)

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
        attn = self.block.attn

        context_attn_output = attn_out[:, : self._text_seq_len, :]
        hidden_attn_output = attn_out[:, self._text_seq_len :, :]

        hidden_attn_output = attn.to_out[0](hidden_attn_output)  # pyright: ignore[reportAny]
        context_attn_output = attn.to_add_out(context_attn_output)

        assert self._hidden_mod is not None
        assert self._context_mod is not None

        hidden_states = JointTransformerBlock.apply_norm_and_feed_forward(
            hidden_states=hidden_states,
            attn_output=hidden_attn_output,  # pyright: ignore[reportAny]
            gate_mlp=self._hidden_mod.gate_mlp,
            gate_msa=self._hidden_mod.gate_msa,
            scale_mlp=self._hidden_mod.scale_mlp,
            shift_mlp=self._hidden_mod.shift_mlp,
            norm_layer=self.block.norm2,
            ff_layer=self.block.ff,
        )
        encoder_hidden_states = JointTransformerBlock.apply_norm_and_feed_forward(
            hidden_states=encoder_hidden_states,
            attn_output=context_attn_output,
            gate_mlp=self._context_mod.gate_mlp,
            gate_msa=self._context_mod.gate_msa,
            scale_mlp=self._context_mod.scale_mlp,
            shift_mlp=self._context_mod.shift_mlp,
            norm_layer=self.block.norm2_context,
            ff_layer=self.block.ff_context,
        )

        return encoder_hidden_states, hidden_states


class FluxSingleBlockWrapper(SingleBlockWrapper[SingleTransformerBlock]):
    """Flux-specific single block wrapper with pipefusion support."""

    def __init__(self, block: SingleTransformerBlock, text_seq_len: int):
        super().__init__(block, text_seq_len)
        self._num_heads = block.attn.num_heads
        self._head_dim = block.attn.head_dimension

        # Intermediate state stored between _compute_qkv and _apply_output
        self._norm_state: FluxNormGateState | None = None

    def _compute_qkv(
        self,
        hidden_states: mx.array,
        text_embeddings: mx.array,
        rotary_embeddings: RotaryEmbeddings,
        patch_mode: bool = False,
    ) -> tuple[mx.array, mx.array, mx.array]:
        assert isinstance(rotary_embeddings, mx.array)

        attn = self.block.attn

        norm_hidden, gate = self.block.norm(
            hidden_states=hidden_states,
            text_embeddings=text_embeddings,
        )
        self._norm_state = FluxNormGateState(norm_hidden=norm_hidden, gate=gate)

        query, key, value = AttentionUtils.process_qkv(
            hidden_states=norm_hidden,
            to_q=attn.to_q,
            to_k=attn.to_k,
            to_v=attn.to_v,
            norm_q=attn.norm_q,
            norm_k=attn.norm_k,
            num_heads=self._num_heads,
            head_dim=self._head_dim,
        )

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
        residual = hidden_states

        assert self._norm_state is not None

        output = self.block._apply_feed_forward_and_projection(
            norm_hidden_states=self._norm_state.norm_hidden,
            attn_output=attn_out,
            gate=self._norm_state.gate,
        )

        return residual + output
