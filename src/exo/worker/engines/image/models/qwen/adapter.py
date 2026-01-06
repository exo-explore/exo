from pathlib import Path
from typing import Any, cast

import mlx.core as mx
from mflux.config.model_config import ModelConfig
from mflux.config.runtime_config import RuntimeConfig
from mflux.models.qwen.latent_creator.qwen_latent_creator import QwenLatentCreator
from mflux.models.qwen.model.qwen_text_encoder.qwen_prompt_encoder import (
    QwenPromptEncoder,
)
from mflux.models.qwen.model.qwen_transformer.qwen_attention import QwenAttention
from mflux.models.qwen.model.qwen_transformer.qwen_transformer import QwenTransformer
from mflux.models.qwen.model.qwen_transformer.qwen_transformer_block import (
    QwenTransformerBlock,
)
from mflux.models.qwen.variants.txt2img.qwen_image import QwenImage

from exo.worker.engines.image.config import ImageModelConfig
from exo.worker.engines.image.models.base import BaseModelAdapter
from exo.worker.engines.image.pipeline.adapter import (
    BlockWrapperMode,
    JointBlockInterface,
    SingleBlockInterface,
)
from exo.worker.engines.image.pipeline.kv_cache import ImagePatchKVCache


class QwenPromptData:
    """Container for Qwen prompt encoding results.

    Implements PromptData protocol with additional Qwen-specific attributes.
    """

    def __init__(
        self,
        prompt_embeds: mx.array,
        prompt_mask: mx.array,
        negative_prompt_embeds: mx.array,
        negative_prompt_mask: mx.array,
    ):
        self._prompt_embeds = prompt_embeds
        self.prompt_mask = prompt_mask
        self._negative_prompt_embeds = negative_prompt_embeds
        self.negative_prompt_mask = negative_prompt_mask

    @property
    def prompt_embeds(self) -> mx.array:
        """Text embeddings from encoder."""
        return self._prompt_embeds

    @property
    def pooled_prompt_embeds(self) -> mx.array:
        """Placeholder for protocol compliance - Qwen doesn't use pooled embeds."""
        return self._prompt_embeds  # Use prompt_embeds as placeholder

    @property
    def negative_prompt_embeds(self) -> mx.array:
        """Negative prompt embeddings for CFG."""
        return self._negative_prompt_embeds

    @property
    def negative_pooled_prompt_embeds(self) -> mx.array:
        """Placeholder - Qwen doesn't use pooled embeds."""
        return self._negative_prompt_embeds

    def get_extra_forward_kwargs(self, positive: bool = True) -> dict[str, Any]:
        """Return encoder_hidden_states_mask for the appropriate prompt."""
        if positive:
            return {"encoder_hidden_states_mask": self.prompt_mask}
        else:
            return {"encoder_hidden_states_mask": self.negative_prompt_mask}


class QwenModelAdapter(BaseModelAdapter):
    """Adapter for Qwen-Image model.

    Key differences from Flux:
    - Single text encoder (vs dual T5+CLIP)
    - 60 joint-style blocks, no single blocks
    - 3D RoPE returning ((img_cos, img_sin), (txt_cos, txt_sin))
    - Norm-preserving CFG with negative prompts
    - Uses attention mask for variable-length text
    """

    def __init__(
        self,
        config: ImageModelConfig,
        model_id: str,
        local_path: Path,
        quantize: int | None = None,
    ):
        self._config = config
        self._model = QwenImage(
            model_config=ModelConfig.from_name(model_name=model_id, base_model=None),
            local_path=str(local_path),
            quantize=quantize,
        )
        self._transformer = self._model.transformer

    @property
    def config(self) -> ImageModelConfig:
        return self._config

    @property
    def model(self) -> QwenImage:
        return self._model

    @property
    def transformer(self) -> QwenTransformer:
        return self._transformer

    @property
    def hidden_dim(self) -> int:
        return self._transformer.inner_dim

    def _get_latent_creator(self) -> type:
        return QwenLatentCreator

    def encode_prompt(self, prompt: str) -> QwenPromptData:
        """Encode prompt into QwenPromptData.

        Qwen uses classifier-free guidance with explicit negative prompts.
        Returns a QwenPromptData container with all 4 tensors.
        """
        # TODO(ciaran): empty string as default negative prompt
        negative_prompt = ""

        prompt_embeds, prompt_mask, neg_embeds, neg_mask = (
            QwenPromptEncoder.encode_prompt(
                prompt=prompt,
                negative_prompt=negative_prompt,
                prompt_cache=self._model.prompt_cache,
                qwen_tokenizer=self._model.qwen_tokenizer,
                qwen_text_encoder=self._model.text_encoder,
            )
        )

        return QwenPromptData(
            prompt_embeds=prompt_embeds,
            prompt_mask=prompt_mask,
            negative_prompt_embeds=neg_embeds,
            negative_prompt_mask=neg_mask,
        )

    @property
    def needs_cfg(self) -> bool:
        gs = self._config.guidance_scale
        return gs is not None and gs > 1.0

    def apply_guidance(
        self,
        noise_positive: mx.array,
        noise_negative: mx.array,
        guidance_scale: float,
    ) -> mx.array:
        return self._model.compute_guided_noise(
            noise=noise_positive,
            noise_negative=noise_negative,
            guidance=guidance_scale,
        )

    def compute_embeddings(
        self,
        hidden_states: mx.array,
        prompt_embeds: mx.array,
    ) -> tuple[mx.array, mx.array]:
        """Compute image and text embeddings."""
        # Image embedding
        embedded_hidden = self._transformer.img_in(hidden_states)
        # Text embedding: first normalize, then project
        encoder_hidden_states = self._transformer.txt_norm(prompt_embeds)
        embedded_encoder = self._transformer.txt_in(encoder_hidden_states)
        return embedded_hidden, embedded_encoder

    def compute_text_embeddings(
        self,
        t: int,
        runtime_config: RuntimeConfig,
        pooled_prompt_embeds: mx.array | None = None,
        hidden_states: mx.array | None = None,
    ) -> mx.array:
        """Compute time/text embeddings.

        For Qwen, the time_text_embed only uses hidden_states for:
        - batch_size (shape[0])
        - dtype

        This allows us to pass any tensor (latents, prompt_embeds) as a fallback
        when embedded hidden_states are not yet available.
        """
        # Use hidden_states if provided, otherwise fall back to pooled_prompt_embeds
        # (which for Qwen is the same as prompt_embeds)
        ref_tensor = (
            hidden_states if hidden_states is not None else pooled_prompt_embeds
        )
        if ref_tensor is None:
            raise ValueError(
                "Either hidden_states or pooled_prompt_embeds is required "
                "for Qwen text embeddings"
            )

        timestep = QwenTransformer._compute_timestep(t, runtime_config)  # noqa: SLF001
        batch_size = ref_tensor.shape[0]
        timestep = mx.broadcast_to(timestep, (batch_size,)).astype(mx.float32)
        return self._transformer.time_text_embed(timestep, ref_tensor)

    def compute_rotary_embeddings(
        self,
        prompt_embeds: mx.array,
        runtime_config: RuntimeConfig,
        **kwargs: Any,
    ) -> Any:
        """Compute 3D rotary embeddings for Qwen.

        Qwen uses video-aware 3D RoPE with separate embeddings for image and text.

        Returns:
            tuple[tuple[mx.array, mx.array], tuple[mx.array, mx.array]]:
                ((img_cos, img_sin), (txt_cos, txt_sin))
        """
        encoder_hidden_states_mask = kwargs.get("encoder_hidden_states_mask")
        cond_image_grid = kwargs.get("cond_image_grid")

        if encoder_hidden_states_mask is None:
            raise ValueError(
                "encoder_hidden_states_mask is required for Qwen RoPE computation"
            )

        return QwenTransformer._compute_rotary_embeddings(  # noqa: SLF001
            encoder_hidden_states_mask=encoder_hidden_states_mask,
            pos_embed=self._transformer.pos_embed,
            config=runtime_config,
            cond_image_grid=cond_image_grid,
        )

    def apply_joint_block(
        self,
        block: JointBlockInterface,
        hidden_states: mx.array,
        encoder_hidden_states: mx.array,
        text_embeddings: mx.array,
        rotary_embeddings: Any,  # tuple[tuple[mx.array, mx.array], tuple[mx.array, mx.array]] for Qwen
        kv_cache: ImagePatchKVCache | None,
        mode: BlockWrapperMode,
        text_seq_len: int,
        patch_start: int | None = None,
        patch_end: int | None = None,
        **kwargs: Any,
    ) -> tuple[mx.array, mx.array]:
        """Apply Qwen joint block.

        For caching mode, we run the full block and optionally populate the KV cache.
        For patched mode, we use the cached KV values (not yet implemented).
        """
        encoder_hidden_states_mask = kwargs.get("encoder_hidden_states_mask")
        block_idx = kwargs.get("block_idx")

        if mode == BlockWrapperMode.CACHING:
            return self._apply_joint_block_caching(
                block=block,
                hidden_states=hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                text_embeddings=text_embeddings,
                rotary_embeddings=rotary_embeddings,
                kv_cache=kv_cache,
                text_seq_len=text_seq_len,
                encoder_hidden_states_mask=encoder_hidden_states_mask,
                block_idx=block_idx,
            )
        else:
            # mode == BlockWrapperMode.PATCHED
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
                encoder_hidden_states_mask=encoder_hidden_states_mask,
                block_idx=block_idx,
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
        """Qwen has no single blocks."""
        raise NotImplementedError("Qwen does not have single blocks")

    def final_projection(
        self,
        hidden_states: mx.array,
        text_embeddings: mx.array,
    ) -> mx.array:
        """Apply final normalization and projection."""
        hidden_states = self._transformer.norm_out(hidden_states, text_embeddings)
        return self._transformer.proj_out(hidden_states)

    def get_joint_blocks(self) -> list[JointBlockInterface]:
        """Return all 60 transformer blocks."""
        return cast(
            list[JointBlockInterface], list(self._transformer.transformer_blocks)
        )

    def get_single_blocks(self) -> list[SingleBlockInterface]:
        """Qwen has no single blocks."""
        return []

    def slice_transformer_blocks(
        self,
        start_layer: int,
        end_layer: int,
        total_joint_blocks: int,
        total_single_blocks: int,
    ) -> None:
        all_blocks = list(self._transformer.transformer_blocks)
        assigned_blocks = all_blocks[start_layer:end_layer]
        self._transformer.transformer_blocks = assigned_blocks

    def merge_streams(
        self,
        hidden_states: mx.array,
        encoder_hidden_states: mx.array,
    ) -> mx.array:
        """Merge image and text streams.

        For Qwen, this is called before final projection.
        The streams remain separate through all blocks.
        """
        return mx.concatenate([encoder_hidden_states, hidden_states], axis=1)

    def _apply_joint_block_caching(
        self,
        block: Any,  # QwenTransformerBlock
        hidden_states: mx.array,
        encoder_hidden_states: mx.array,
        text_embeddings: mx.array,
        rotary_embeddings: tuple[tuple[mx.array, mx.array], tuple[mx.array, mx.array]],
        kv_cache: ImagePatchKVCache | None,
        text_seq_len: int,
        encoder_hidden_states_mask: mx.array | None = None,
        block_idx: int | None = None,
    ) -> tuple[mx.array, mx.array]:
        """Apply joint block in caching mode (full attention, optionally populate cache).

        Delegates to the QwenTransformerBlock's forward pass.
        """
        # Call the block directly - it handles all the modulation and attention internally
        return block(
            hidden_states=hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            encoder_hidden_states_mask=encoder_hidden_states_mask,
            text_embeddings=text_embeddings,
            image_rotary_emb=rotary_embeddings,
            block_idx=block_idx,
        )

    def _apply_joint_block_patched(
        self,
        block: Any,  # QwenTransformerBlock
        patch_hidden: mx.array,
        encoder_hidden_states: mx.array,
        text_embeddings: mx.array,
        rotary_embeddings: tuple[tuple[mx.array, mx.array], tuple[mx.array, mx.array]],
        kv_cache: ImagePatchKVCache,
        text_seq_len: int,
        patch_start: int,
        patch_end: int,
        encoder_hidden_states_mask: mx.array | None = None,
        block_idx: int | None = None,
    ) -> tuple[mx.array, mx.array]:
        batch_size = patch_hidden.shape[0]
        attn = block.attn
        num_heads = attn.num_heads
        head_dim = attn.head_dim

        # 1. Compute modulation parameters
        img_mod_params = block.img_mod_linear(block.img_mod_silu(text_embeddings))
        txt_mod_params = block.txt_mod_linear(block.txt_mod_silu(text_embeddings))

        img_mod1, img_mod2 = mx.split(img_mod_params, 2, axis=-1)
        txt_mod1, txt_mod2 = mx.split(txt_mod_params, 2, axis=-1)

        # 2. Apply normalization and modulation
        img_normed = block.img_norm1(patch_hidden)
        img_modulated, img_gate1 = QwenTransformerBlock._modulate(img_normed, img_mod1)

        txt_normed = block.txt_norm1(encoder_hidden_states)
        txt_modulated, txt_gate1 = QwenTransformerBlock._modulate(txt_normed, txt_mod1)

        # 3. Compute Q, K, V for image patch
        img_query = attn.to_q(img_modulated)
        img_key = attn.to_k(img_modulated)
        img_value = attn.to_v(img_modulated)

        # 4. Compute Q, K, V for text
        txt_query = attn.add_q_proj(txt_modulated)
        txt_key = attn.add_k_proj(txt_modulated)
        txt_value = attn.add_v_proj(txt_modulated)

        # 5. Reshape to [B, S, H, D]
        patch_len = patch_hidden.shape[1]
        img_query = mx.reshape(img_query, (batch_size, patch_len, num_heads, head_dim))
        img_key = mx.reshape(img_key, (batch_size, patch_len, num_heads, head_dim))
        img_value = mx.reshape(img_value, (batch_size, patch_len, num_heads, head_dim))

        txt_query = mx.reshape(
            txt_query, (batch_size, text_seq_len, num_heads, head_dim)
        )
        txt_key = mx.reshape(txt_key, (batch_size, text_seq_len, num_heads, head_dim))
        txt_value = mx.reshape(
            txt_value, (batch_size, text_seq_len, num_heads, head_dim)
        )

        # 6. Apply RMSNorm to Q, K
        img_query = attn.norm_q(img_query)
        img_key = attn.norm_k(img_key)
        txt_query = attn.norm_added_q(txt_query)
        txt_key = attn.norm_added_k(txt_key)

        # 7. Extract RoPE for patch: slice image RoPE, keep full text RoPE
        (img_cos, img_sin), (txt_cos, txt_sin) = rotary_embeddings
        patch_img_cos = img_cos[patch_start:patch_end]
        patch_img_sin = img_sin[patch_start:patch_end]

        # 8. Apply RoPE to Q, K
        img_query = QwenAttention._apply_rope_qwen(
            img_query, patch_img_cos, patch_img_sin
        )
        img_key = QwenAttention._apply_rope_qwen(img_key, patch_img_cos, patch_img_sin)
        txt_query = QwenAttention._apply_rope_qwen(txt_query, txt_cos, txt_sin)
        txt_key = QwenAttention._apply_rope_qwen(txt_key, txt_cos, txt_sin)

        # 9. Transpose to [B, H, S, D] for cache operations
        img_key_bhsd = mx.transpose(img_key, (0, 2, 1, 3))
        img_value_bhsd = mx.transpose(img_value, (0, 2, 1, 3))

        # 10. Update cache with this patch's IMAGE K/V
        kv_cache.update_image_patch(
            patch_start=patch_start,
            patch_end=patch_end,
            key=img_key_bhsd,
            value=img_value_bhsd,
        )

        # 11. Get full K, V from cache (text + full image)
        txt_key_bhsd = mx.transpose(txt_key, (0, 2, 1, 3))
        txt_value_bhsd = mx.transpose(txt_value, (0, 2, 1, 3))
        full_key, full_value = kv_cache.get_full_kv(
            text_key=txt_key_bhsd,
            text_value=txt_value_bhsd,
        )

        # 12. Build query: [text, patch]
        joint_query = mx.concatenate([txt_query, img_query], axis=1)

        # 13. Build attention mask for [text + patch] query attending to [text + full_image] KV
        mask = QwenAttention._convert_mask_for_qwen(
            mask=encoder_hidden_states_mask,
            joint_seq_len=full_key.shape[2],  # text + full_image
            txt_seq_len=text_seq_len,
        )

        # 14. Compute attention
        hidden_states = attn._compute_attention_qwen(
            query=joint_query,
            key=mx.transpose(full_key, (0, 2, 1, 3)),  # Back to [B, S, H, D]
            value=mx.transpose(full_value, (0, 2, 1, 3)),
            mask=mask,
            block_idx=block_idx,
        )

        # 15. Extract text and image attention outputs
        txt_attn_output = hidden_states[:, :text_seq_len, :]
        img_attn_output = hidden_states[:, text_seq_len:, :]

        # 16. Project outputs
        img_attn_output = attn.attn_to_out[0](img_attn_output)
        txt_attn_output = attn.to_add_out(txt_attn_output)

        # 17. Apply residual + gate for attention
        patch_hidden = patch_hidden + img_gate1 * img_attn_output
        encoder_hidden_states = encoder_hidden_states + txt_gate1 * txt_attn_output

        # 18. Apply feed-forward for image
        img_normed2 = block.img_norm2(patch_hidden)
        img_modulated2, img_gate2 = QwenTransformerBlock._modulate(
            img_normed2, img_mod2
        )
        img_mlp_output = block.img_ff(img_modulated2)
        patch_hidden = patch_hidden + img_gate2 * img_mlp_output

        # 19. Apply feed-forward for text
        txt_normed2 = block.txt_norm2(encoder_hidden_states)
        txt_modulated2, txt_gate2 = QwenTransformerBlock._modulate(
            txt_normed2, txt_mod2
        )
        txt_mlp_output = block.txt_ff(txt_modulated2)
        encoder_hidden_states = encoder_hidden_states + txt_gate2 * txt_mlp_output

        return encoder_hidden_states, patch_hidden
