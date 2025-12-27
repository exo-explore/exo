from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import mlx.core as mx
from mflux.config.model_config import ModelConfig
from mflux.config.runtime_config import RuntimeConfig
from mflux.models.flux.latent_creator.flux_latent_creator import FluxLatentCreator
from mflux.models.flux.model.flux_text_encoder.prompt_encoder import PromptEncoder
from mflux.models.flux.model.flux_transformer.common.attention_utils import (
    AttentionUtils,
)
from mflux.models.flux.model.flux_transformer.joint_transformer_block import (
    JointTransformerBlock,
)
from mflux.models.flux.model.flux_transformer.transformer import Transformer
from mflux.models.flux.variants.txt2img.flux import Flux1

from exo.worker.engines.image.config import ImageModelConfig
from exo.worker.engines.image.models.base import BaseModelAdapter
from exo.worker.engines.image.pipeline.adapter import (
    BlockWrapperMode,
    JointBlockInterface,
    SingleBlockInterface,
)
from exo.worker.engines.image.pipeline.kv_cache import ImagePatchKVCache

if TYPE_CHECKING:
    from exo.worker.engines.image.pipeline.runner import DiffusionRunner


class FluxModelAdapter(BaseModelAdapter):
    def __init__(
        self,
        config: ImageModelConfig,
        model_id: str,
        local_path: Path,
        quantize: int | None = None,
    ):
        self._config = config
        self._model = Flux1(
            model_config=ModelConfig.from_name(model_name=model_id, base_model=None),
            local_path=str(local_path),
            quantize=quantize,
        )
        # Store original transformer reference BEFORE it may be replaced by DistributedDenoising
        self._transformer = self._model.transformer

    @property
    def config(self) -> ImageModelConfig:
        return self._config

    @property
    def model(self) -> Flux1:
        return self._model

    @property
    def transformer(self) -> Transformer:
        return self._transformer

    @property
    def hidden_dim(self) -> int:
        return self._transformer.x_embedder.weight.shape[0]

    # -------------------------------------------------------------------------
    # BaseModelAdapter abstract method implementations
    # -------------------------------------------------------------------------

    def _get_latent_creator(self) -> type:
        return FluxLatentCreator

    def _encode_prompt(self, prompt: str) -> tuple[mx.array, mx.array]:
        return PromptEncoder.encode_prompt(
            prompt=prompt,
            prompt_cache=self._model.prompt_cache,
            t5_tokenizer=self._model.t5_tokenizer,
            clip_tokenizer=self._model.clip_tokenizer,
            t5_text_encoder=self._model.t5_text_encoder,
            clip_text_encoder=self._model.clip_text_encoder,
        )

    def _run_denoising(
        self,
        latents: mx.array,
        prompt_data: tuple[mx.array, mx.array],
        runtime_config: RuntimeConfig,
        runner: "DiffusionRunner | None",
    ) -> mx.array:
        prompt_embeds, pooled_prompt_embeds = prompt_data
        if runner:
            # Distributed mode - use DiffusionRunner
            return runner.run(
                latents=latents,
                prompt_embeds=prompt_embeds,
                pooled_prompt_embeds=pooled_prompt_embeds,
                runtime_config=runtime_config,
                seed=0,  # Not used by runner
                prompt="",  # Not used by runner
            )
        else:
            # Single-node mode - use DiffusionRunner with no distribution
            # This path shouldn't be hit in practice since we always have a runner
            raise NotImplementedError(
                "Single-node FLUX generation requires a DiffusionRunner"
            )

    # -------------------------------------------------------------------------
    # ModelAdapter protocol implementations (for distributed inference)
    # -------------------------------------------------------------------------

    def compute_embeddings(
        self,
        hidden_states: mx.array,
        prompt_embeds: mx.array,
    ) -> tuple[mx.array, mx.array]:
        embedded_hidden = self._transformer.x_embedder(hidden_states)
        embedded_encoder = self._transformer.context_embedder(prompt_embeds)
        return embedded_hidden, embedded_encoder

    def compute_text_embeddings(
        self,
        t: int,
        pooled_prompt_embeds: mx.array,
        runtime_config: RuntimeConfig,
    ) -> mx.array:
        return Transformer.compute_text_embeddings(
            t, pooled_prompt_embeds, self._transformer.time_text_embed, runtime_config
        )

    def compute_rotary_embeddings(
        self,
        prompt_embeds: mx.array,
        runtime_config: RuntimeConfig,
        **kwargs: Any,
    ) -> mx.array:
        kontext_image_ids = kwargs.get("kontext_image_ids")
        return Transformer.compute_rotary_embeddings(
            prompt_embeds,
            self._transformer.pos_embed,
            runtime_config,
            kontext_image_ids,
        )

    def apply_joint_block(
        self,
        block: JointBlockInterface,
        hidden_states: mx.array,
        encoder_hidden_states: mx.array,
        text_embeddings: mx.array,
        rotary_embeddings: mx.array,
        kv_cache: ImagePatchKVCache | None,
        mode: BlockWrapperMode,
        text_seq_len: int,
        patch_start: int | None = None,
        patch_end: int | None = None,
    ) -> tuple[mx.array, mx.array]:
        if mode == BlockWrapperMode.CACHING:
            return self._apply_joint_block_caching(
                block=block,
                hidden_states=hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                text_embeddings=text_embeddings,
                rotary_embeddings=rotary_embeddings,
                kv_cache=kv_cache,
                text_seq_len=text_seq_len,
            )
        else:
            assert patch_start is not None and patch_end is not None
            assert kv_cache is not None
            return self._apply_joint_block_patched(
                block=block,
                patch_hidden=hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                text_embeddings=text_embeddings,
                rotary_embeddings=rotary_embeddings,
                kv_cache=kv_cache,
                text_seq_len=text_seq_len,
                patch_start=patch_start,
                patch_end=patch_end,
            )

    def apply_single_block(
        self,
        block: SingleBlockInterface,
        hidden_states: mx.array,
        text_embeddings: mx.array,
        rotary_embeddings: mx.array,
        kv_cache: ImagePatchKVCache | None,
        mode: BlockWrapperMode,
        text_seq_len: int,
        patch_start: int | None = None,
        patch_end: int | None = None,
    ) -> mx.array:
        if mode == BlockWrapperMode.CACHING:
            return self._apply_single_block_caching(
                block=block,
                hidden_states=hidden_states,
                text_embeddings=text_embeddings,
                rotary_embeddings=rotary_embeddings,
                kv_cache=kv_cache,
                text_seq_len=text_seq_len,
            )
        else:
            assert patch_start is not None and patch_end is not None
            assert kv_cache is not None
            return self._apply_single_block_patched(
                block=block,
                patch_hidden=hidden_states,
                text_embeddings=text_embeddings,
                rotary_embeddings=rotary_embeddings,
                kv_cache=kv_cache,
                text_seq_len=text_seq_len,
                patch_start=patch_start,
                patch_end=patch_end,
            )

    def final_projection(
        self,
        hidden_states: mx.array,
        text_embeddings: mx.array,
    ) -> mx.array:
        hidden_states = self._transformer.norm_out(hidden_states, text_embeddings)
        return self._transformer.proj_out(hidden_states)

    def get_joint_blocks(self) -> list[JointBlockInterface]:
        return cast(
            list[JointBlockInterface], list(self._transformer.transformer_blocks)
        )

    def get_single_blocks(self) -> list[SingleBlockInterface]:
        return cast(
            list[SingleBlockInterface],
            list(self._transformer.single_transformer_blocks),
        )

    def merge_streams(
        self,
        hidden_states: mx.array,
        encoder_hidden_states: mx.array,
    ) -> mx.array:
        return mx.concatenate([encoder_hidden_states, hidden_states], axis=1)

    # -------------------------------------------------------------------------
    # Joint block implementations
    # -------------------------------------------------------------------------

    def _apply_joint_block_caching(
        self,
        block: JointBlockInterface,
        hidden_states: mx.array,
        encoder_hidden_states: mx.array,
        text_embeddings: mx.array,
        rotary_embeddings: mx.array,
        kv_cache: ImagePatchKVCache | None,
        text_seq_len: int,
    ) -> tuple[mx.array, mx.array]:
        num_img_tokens = hidden_states.shape[1]
        batch_size = hidden_states.shape[0]
        attn = block.attn
        num_heads = attn.num_heads
        head_dim = attn.head_dimension

        # 1. Compute norms
        norm_hidden, gate_msa, shift_mlp, scale_mlp, gate_mlp = block.norm1(
            hidden_states=hidden_states,
            text_embeddings=text_embeddings,
        )
        norm_encoder, c_gate_msa, c_shift_mlp, c_scale_mlp, c_gate_mlp = (
            block.norm1_context(
                hidden_states=encoder_hidden_states,
                text_embeddings=text_embeddings,
            )
        )

        # 2. Compute Q, K, V for full image
        img_query, img_key, img_value = AttentionUtils.process_qkv(
            hidden_states=norm_hidden,
            to_q=attn.to_q,
            to_k=attn.to_k,
            to_v=attn.to_v,
            norm_q=attn.norm_q,
            norm_k=attn.norm_k,
            num_heads=num_heads,
            head_dim=head_dim,
        )

        # 3. Compute Q, K, V for text
        txt_query, txt_key, txt_value = AttentionUtils.process_qkv(
            hidden_states=norm_encoder,
            to_q=attn.add_q_proj,
            to_k=attn.add_k_proj,
            to_v=attn.add_v_proj,
            norm_q=attn.norm_added_q,
            norm_k=attn.norm_added_k,
            num_heads=num_heads,
            head_dim=head_dim,
        )

        # 4. Concatenate Q, K, V: [text, image]
        query = mx.concatenate([txt_query, img_query], axis=2)
        key = mx.concatenate([txt_key, img_key], axis=2)
        value = mx.concatenate([txt_value, img_value], axis=2)

        # 5. Apply RoPE
        query, key = AttentionUtils.apply_rope(
            xq=query, xk=key, freqs_cis=rotary_embeddings
        )

        # 6. Store IMAGE K/V in cache for async pipeline
        if kv_cache is not None:
            kv_cache.update_image_patch(
                patch_start=0,
                patch_end=num_img_tokens,
                key=key[:, :, text_seq_len:, :],
                value=value[:, :, text_seq_len:, :],
            )

        # 7. Compute full attention
        attn_output = AttentionUtils.compute_attention(
            query=query,
            key=key,
            value=value,
            batch_size=batch_size,
            num_heads=num_heads,
            head_dim=head_dim,
        )

        # 8. Extract and project outputs
        context_attn_output = attn_output[:, :text_seq_len, :]
        attn_output = attn_output[:, text_seq_len:, :]

        attn_output = attn.to_out[0](attn_output)
        context_attn_output = attn.to_add_out(context_attn_output)

        # 9. Apply norm and feed forward
        hidden_states = JointTransformerBlock.apply_norm_and_feed_forward(
            hidden_states=hidden_states,
            attn_output=attn_output,
            gate_mlp=gate_mlp,
            gate_msa=gate_msa,
            scale_mlp=scale_mlp,
            shift_mlp=shift_mlp,
            norm_layer=block.norm2,
            ff_layer=block.ff,
        )
        encoder_hidden_states = JointTransformerBlock.apply_norm_and_feed_forward(
            hidden_states=encoder_hidden_states,
            attn_output=context_attn_output,
            gate_mlp=c_gate_mlp,
            gate_msa=c_gate_msa,
            scale_mlp=c_scale_mlp,
            shift_mlp=c_shift_mlp,
            norm_layer=block.norm2_context,
            ff_layer=block.ff_context,
        )

        return encoder_hidden_states, hidden_states

    def _apply_joint_block_patched(
        self,
        block: JointBlockInterface,
        patch_hidden: mx.array,
        encoder_hidden_states: mx.array,
        text_embeddings: mx.array,
        rotary_embeddings: mx.array,
        kv_cache: ImagePatchKVCache,
        text_seq_len: int,
        patch_start: int,
        patch_end: int,
    ) -> tuple[mx.array, mx.array]:
        batch_size = patch_hidden.shape[0]
        attn = block.attn
        num_heads = attn.num_heads
        head_dim = attn.head_dimension

        # 1. Compute norms
        norm_hidden, gate_msa, shift_mlp, scale_mlp, gate_mlp = block.norm1(
            hidden_states=patch_hidden,
            text_embeddings=text_embeddings,
        )
        norm_encoder, c_gate_msa, c_shift_mlp, c_scale_mlp, c_gate_mlp = (
            block.norm1_context(
                hidden_states=encoder_hidden_states,
                text_embeddings=text_embeddings,
            )
        )

        # 2. Compute Q, K, V for image patch
        img_query, img_key, img_value = AttentionUtils.process_qkv(
            hidden_states=norm_hidden,
            to_q=attn.to_q,
            to_k=attn.to_k,
            to_v=attn.to_v,
            norm_q=attn.norm_q,
            norm_k=attn.norm_k,
            num_heads=num_heads,
            head_dim=head_dim,
        )

        # 3. Compute Q, K, V for text
        txt_query, txt_key, txt_value = AttentionUtils.process_qkv(
            hidden_states=norm_encoder,
            to_q=attn.add_q_proj,
            to_k=attn.add_k_proj,
            to_v=attn.add_v_proj,
            norm_q=attn.norm_added_q,
            norm_k=attn.norm_added_k,
            num_heads=num_heads,
            head_dim=head_dim,
        )

        # 4. Concatenate Q, K, V for patch: [text, patch]
        query = mx.concatenate([txt_query, img_query], axis=2)
        patch_key = mx.concatenate([txt_key, img_key], axis=2)
        patch_value = mx.concatenate([txt_value, img_value], axis=2)

        # 5. Extract RoPE for [text + current_patch]
        text_rope = rotary_embeddings[:, :, :text_seq_len, ...]
        patch_img_rope = rotary_embeddings[
            :, :, text_seq_len + patch_start : text_seq_len + patch_end, ...
        ]
        patch_rope = mx.concatenate([text_rope, patch_img_rope], axis=2)

        # 6. Apply RoPE
        query, patch_key = AttentionUtils.apply_rope(
            xq=query, xk=patch_key, freqs_cis=patch_rope
        )

        # 7. Update cache with this patch's IMAGE K/V
        kv_cache.update_image_patch(
            patch_start=patch_start,
            patch_end=patch_end,
            key=patch_key[:, :, text_seq_len:, :],
            value=patch_value[:, :, text_seq_len:, :],
        )

        # 8. Get full K, V from cache
        full_key, full_value = kv_cache.get_full_kv(
            text_key=patch_key[:, :, :text_seq_len, :],
            text_value=patch_value[:, :, :text_seq_len, :],
        )

        # 9. Compute attention
        attn_output = AttentionUtils.compute_attention(
            query=query,
            key=full_key,
            value=full_value,
            batch_size=batch_size,
            num_heads=num_heads,
            head_dim=head_dim,
        )

        # 10. Extract and project outputs
        context_attn_output = attn_output[:, :text_seq_len, :]
        hidden_attn_output = attn_output[:, text_seq_len:, :]

        hidden_attn_output = attn.to_out[0](hidden_attn_output)
        context_attn_output = attn.to_add_out(context_attn_output)

        # 11. Apply norm and feed forward
        patch_hidden = JointTransformerBlock.apply_norm_and_feed_forward(
            hidden_states=patch_hidden,
            attn_output=hidden_attn_output,
            gate_mlp=gate_mlp,
            gate_msa=gate_msa,
            scale_mlp=scale_mlp,
            shift_mlp=shift_mlp,
            norm_layer=block.norm2,
            ff_layer=block.ff,
        )
        encoder_hidden_states = JointTransformerBlock.apply_norm_and_feed_forward(
            hidden_states=encoder_hidden_states,
            attn_output=context_attn_output,
            gate_mlp=c_gate_mlp,
            gate_msa=c_gate_msa,
            scale_mlp=c_scale_mlp,
            shift_mlp=c_shift_mlp,
            norm_layer=block.norm2_context,
            ff_layer=block.ff_context,
        )

        return encoder_hidden_states, patch_hidden

    # -------------------------------------------------------------------------
    # Single block implementations
    # -------------------------------------------------------------------------

    def _apply_single_block_caching(
        self,
        block: SingleBlockInterface,
        hidden_states: mx.array,
        text_embeddings: mx.array,
        rotary_embeddings: mx.array,
        kv_cache: ImagePatchKVCache | None,
        text_seq_len: int,
    ) -> mx.array:
        total_seq_len = hidden_states.shape[1]
        num_img_tokens = total_seq_len - text_seq_len
        batch_size = hidden_states.shape[0]
        attn = block.attn
        num_heads = attn.num_heads
        head_dim = attn.head_dimension

        # Residual connection
        residual = hidden_states

        # 1. Compute norm
        norm_hidden, gate = block.norm(
            hidden_states=hidden_states,
            text_embeddings=text_embeddings,
        )

        # 2. Compute Q, K, V
        query, key, value = AttentionUtils.process_qkv(
            hidden_states=norm_hidden,
            to_q=attn.to_q,
            to_k=attn.to_k,
            to_v=attn.to_v,
            norm_q=attn.norm_q,
            norm_k=attn.norm_k,
            num_heads=num_heads,
            head_dim=head_dim,
        )

        # 3. Apply RoPE
        query, key = AttentionUtils.apply_rope(
            xq=query, xk=key, freqs_cis=rotary_embeddings
        )

        # 4. Store IMAGE K/V in cache
        if kv_cache is not None:
            kv_cache.update_image_patch(
                patch_start=0,
                patch_end=num_img_tokens,
                key=key[:, :, text_seq_len:, :],
                value=value[:, :, text_seq_len:, :],
            )

        # 5. Compute attention
        attn_output = AttentionUtils.compute_attention(
            query=query,
            key=key,
            value=value,
            batch_size=batch_size,
            num_heads=num_heads,
            head_dim=head_dim,
        )

        # 6. Apply feed forward and projection
        hidden_states = block._apply_feed_forward_and_projection(
            norm_hidden_states=norm_hidden,
            attn_output=attn_output,
            gate=gate,
        )

        return residual + hidden_states

    def _apply_single_block_patched(
        self,
        block: SingleBlockInterface,
        patch_hidden: mx.array,
        text_embeddings: mx.array,
        rotary_embeddings: mx.array,
        kv_cache: ImagePatchKVCache,
        text_seq_len: int,
        patch_start: int,
        patch_end: int,
    ) -> mx.array:
        batch_size = patch_hidden.shape[0]
        attn = block.attn
        num_heads = attn.num_heads
        head_dim = attn.head_dimension

        # Residual connection
        residual = patch_hidden

        # 1. Compute norm
        norm_hidden, gate = block.norm(
            hidden_states=patch_hidden,
            text_embeddings=text_embeddings,
        )

        # 2. Compute Q, K, V
        query, key, value = AttentionUtils.process_qkv(
            hidden_states=norm_hidden,
            to_q=attn.to_q,
            to_k=attn.to_k,
            to_v=attn.to_v,
            norm_q=attn.norm_q,
            norm_k=attn.norm_k,
            num_heads=num_heads,
            head_dim=head_dim,
        )

        # 3. Extract RoPE for [text + current_patch]
        text_rope = rotary_embeddings[:, :, :text_seq_len, ...]
        patch_img_rope = rotary_embeddings[
            :, :, text_seq_len + patch_start : text_seq_len + patch_end, ...
        ]
        patch_rope = mx.concatenate([text_rope, patch_img_rope], axis=2)

        # 4. Apply RoPE
        query, key = AttentionUtils.apply_rope(xq=query, xk=key, freqs_cis=patch_rope)

        # 5. Update cache with this patch's IMAGE K/V
        kv_cache.update_image_patch(
            patch_start=patch_start,
            patch_end=patch_end,
            key=key[:, :, text_seq_len:, :],
            value=value[:, :, text_seq_len:, :],
        )

        # 6. Get full K, V from cache
        full_key, full_value = kv_cache.get_full_kv(
            text_key=key[:, :, :text_seq_len, :],
            text_value=value[:, :, :text_seq_len, :],
        )

        # 7. Compute attention
        attn_output = AttentionUtils.compute_attention(
            query=query,
            key=full_key,
            value=full_value,
            batch_size=batch_size,
            num_heads=num_heads,
            head_dim=head_dim,
        )

        # 8. Apply feed forward and projection
        hidden_states = block._apply_feed_forward_and_projection(
            norm_hidden_states=norm_hidden,
            attn_output=attn_output,
            gate=gate,
        )

        return residual + hidden_states
