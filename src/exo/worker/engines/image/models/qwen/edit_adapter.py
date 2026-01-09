import math
from pathlib import Path
from typing import Any, cast

import mlx.core as mx
from mflux.config.runtime_config import RuntimeConfig
from mflux.models.qwen.latent_creator.qwen_latent_creator import QwenLatentCreator
from mflux.models.qwen.model.qwen_transformer.qwen_attention import QwenAttention
from mflux.models.qwen.model.qwen_transformer.qwen_transformer import QwenTransformer
from mflux.models.qwen.model.qwen_transformer.qwen_transformer_block import (
    QwenTransformerBlock,
)
from mflux.models.qwen.variants.edit.qwen_image_edit import QwenImageEdit
from mflux.models.qwen.variants.edit.utils.qwen_edit_util import QwenEditUtil

from exo.worker.engines.image.config import ImageModelConfig
from exo.worker.engines.image.models.base import BaseModelAdapter
from exo.worker.engines.image.pipeline.adapter import (
    BlockWrapperMode,
    JointBlockInterface,
    SingleBlockInterface,
)
from exo.worker.engines.image.pipeline.kv_cache import ImagePatchKVCache


class QwenEditPromptData:
    """Container for Qwen edit prompt encoding results.

    Includes vision-language encoded embeddings and edit-specific conditioning.
    """

    def __init__(
        self,
        prompt_embeds: mx.array,
        prompt_mask: mx.array,
        negative_prompt_embeds: mx.array,
        negative_prompt_mask: mx.array,
        conditioning_latents: mx.array,
        qwen_image_ids: mx.array,
        cond_image_grid: tuple[int, int, int] | list[tuple[int, int, int]],
    ):
        self._prompt_embeds = prompt_embeds
        self.prompt_mask = prompt_mask
        self._negative_prompt_embeds = negative_prompt_embeds
        self.negative_prompt_mask = negative_prompt_mask
        self._conditioning_latents = conditioning_latents
        self._qwen_image_ids = qwen_image_ids
        self._cond_image_grid = cond_image_grid

    @property
    def prompt_embeds(self) -> mx.array:
        """Text embeddings from vision-language encoder."""
        return self._prompt_embeds

    @property
    def pooled_prompt_embeds(self) -> mx.array:
        """Placeholder for protocol compliance - Qwen doesn't use pooled embeds."""
        return self._prompt_embeds

    @property
    def negative_prompt_embeds(self) -> mx.array:
        """Negative prompt embeddings for CFG."""
        return self._negative_prompt_embeds

    @property
    def negative_pooled_prompt_embeds(self) -> mx.array:
        """Placeholder - Qwen doesn't use pooled embeds."""
        return self._negative_prompt_embeds

    @property
    def conditioning_latents(self) -> mx.array:
        """Static image conditioning latents to concatenate with generated latents."""
        return self._conditioning_latents

    @property
    def qwen_image_ids(self) -> mx.array:
        """Spatial position IDs for conditioning images."""
        return self._qwen_image_ids

    @property
    def cond_image_grid(self) -> tuple[int, int, int] | list[tuple[int, int, int]]:
        """Conditioning image grid dimensions."""
        return self._cond_image_grid

    def get_extra_forward_kwargs(self, positive: bool = True) -> dict[str, Any]:
        """Return encoder_hidden_states_mask and edit-specific params."""
        if positive:
            return {
                "encoder_hidden_states_mask": self.prompt_mask,
                "qwen_image_ids": self._qwen_image_ids,
                "cond_image_grid": self._cond_image_grid,
            }
        else:
            return {
                "encoder_hidden_states_mask": self.negative_prompt_mask,
                "qwen_image_ids": self._qwen_image_ids,
                "cond_image_grid": self._cond_image_grid,
            }

    @property
    def is_edit_mode(self) -> bool:
        """Indicates this is edit mode with conditioning latents."""
        return True


class QwenEditModelAdapter(BaseModelAdapter):
    """Adapter for Qwen-Image-Edit model.

    Key differences from standard QwenModelAdapter:
    - Uses QwenImageEdit model with vision-language components
    - Encodes prompts WITH input images via VL tokenizer/encoder
    - Creates conditioning latents from input images
    - Supports image editing with concatenated latents during diffusion
    """

    def __init__(
        self,
        config: ImageModelConfig,
        model_id: str,
        local_path: Path,
        quantize: int | None = None,
    ):
        self._config = config
        self._model = QwenImageEdit(
            quantize=quantize,
            local_path=str(local_path),
        )
        self._transformer = self._model.transformer

        # Store dimensions computed from input image (set during encode_prompt_with_images)
        self._vl_width: int | None = None
        self._vl_height: int | None = None
        self._vae_width: int | None = None
        self._vae_height: int | None = None

    @property
    def config(self) -> ImageModelConfig:
        return self._config

    @property
    def model(self) -> QwenImageEdit:
        return self._model

    @property
    def transformer(self) -> QwenTransformer:
        return self._transformer

    @property
    def hidden_dim(self) -> int:
        return self._transformer.inner_dim

    def _get_latent_creator(self) -> type:
        return QwenLatentCreator

    def _compute_dimensions_from_image(
        self, image_path: Path
    ) -> tuple[int, int, int, int, int, int]:
        """Compute VL and VAE dimensions from input image.

        Returns:
            (vl_width, vl_height, vae_width, vae_height, output_width, output_height)
        """
        from mflux.utils.image_util import ImageUtil

        pil_image = ImageUtil.load_image(str(image_path)).convert("RGB")
        image_size = pil_image.size

        # Vision-language dimensions (384x384 target area)
        condition_image_size = 384 * 384
        condition_ratio = image_size[0] / image_size[1]
        vl_width = math.sqrt(condition_image_size * condition_ratio)
        vl_height = vl_width / condition_ratio
        vl_width = round(vl_width / 32) * 32
        vl_height = round(vl_height / 32) * 32

        # VAE dimensions (1024x1024 target area)
        vae_image_size = 1024 * 1024
        vae_ratio = image_size[0] / image_size[1]
        vae_width = math.sqrt(vae_image_size * vae_ratio)
        vae_height = vae_width / vae_ratio
        vae_width = round(vae_width / 32) * 32
        vae_height = round(vae_height / 32) * 32

        # Output dimensions from input image aspect ratio
        target_area = 1024 * 1024
        ratio = image_size[0] / image_size[1]
        output_width = math.sqrt(target_area * ratio)
        output_height = output_width / ratio
        output_width = round(output_width / 32) * 32
        output_height = round(output_height / 32) * 32

        # Ensure multiple of 16 for VAE
        vae_scale_factor = 8
        multiple_of = vae_scale_factor * 2
        output_width = output_width // multiple_of * multiple_of
        output_height = output_height // multiple_of * multiple_of

        return (
            int(vl_width),
            int(vl_height),
            int(vae_width),
            int(vae_height),
            int(output_width),
            int(output_height),
        )

    def create_latents(self, seed: int, runtime_config: RuntimeConfig) -> mx.array:
        """Create initial noise latents (pure noise for edit mode)."""
        return QwenLatentCreator.create_noise(
            seed=seed,
            height=runtime_config.height,
            width=runtime_config.width,
        )

    def encode_prompt(self, prompt: str) -> QwenEditPromptData:
        """Not supported - use encode_prompt_with_images for edit mode."""
        raise NotImplementedError(
            "QwenEditModelAdapter requires encode_prompt_with_images()"
        )

    def encode_prompt_with_images(
        self,
        prompt: str,
        image_paths: list[str],
        runtime_config: RuntimeConfig,
    ) -> QwenEditPromptData:
        """Encode prompt with input images using vision-language encoder.

        Args:
            prompt: Text prompt for editing
            image_paths: List of paths to input images
            runtime_config: Runtime configuration with dimensions

        Returns:
            QwenEditPromptData with VL embeddings and conditioning latents
        """
        negative_prompt = ""

        # Use stored dimensions (computed from input image)
        vl_width = self._vl_width
        vl_height = self._vl_height
        vae_width = self._vae_width
        vae_height = self._vae_height

        # Encode prompts with images via vision-language components
        tokenizer = self._model.qwen_vl_tokenizer
        pos_input_ids, pos_attention_mask, pos_pixel_values, pos_image_grid_thw = (
            tokenizer.tokenize_with_image(
                prompt, image_paths, vl_width=vl_width, vl_height=vl_height
            )
        )

        pos_hidden_states = self._model.qwen_vl_encoder(
            input_ids=pos_input_ids,
            attention_mask=pos_attention_mask,
            pixel_values=pos_pixel_values,
            image_grid_thw=pos_image_grid_thw,
        )
        mx.eval(pos_hidden_states[0])
        mx.eval(pos_hidden_states[1])

        # Encode negative prompt with images
        neg_input_ids, neg_attention_mask, neg_pixel_values, neg_image_grid_thw = (
            tokenizer.tokenize_with_image(
                negative_prompt, image_paths, vl_width=vl_width, vl_height=vl_height
            )
        )

        neg_hidden_states = self._model.qwen_vl_encoder(
            input_ids=neg_input_ids,
            attention_mask=neg_attention_mask,
            pixel_values=neg_pixel_values,
            image_grid_thw=neg_image_grid_thw,
        )
        mx.eval(neg_hidden_states[0])
        mx.eval(neg_hidden_states[1])

        # Create conditioning latents from input images
        # Ensure dimensions are set (should have been set via set_image_dimensions)
        assert vl_width is not None and vl_height is not None
        assert vae_width is not None and vae_height is not None

        (
            conditioning_latents,
            qwen_image_ids,
            cond_h_patches,
            cond_w_patches,
            num_images,
        ) = QwenEditUtil.create_image_conditioning_latents(
            vae=self._model.vae,
            height=vae_height,
            width=vae_width,
            image_paths=image_paths,
            vl_width=vl_width,
            vl_height=vl_height,
        )

        # Build cond_image_grid
        if num_images > 1:
            cond_image_grid: tuple[int, int, int] | list[tuple[int, int, int]] = [
                (1, cond_h_patches, cond_w_patches) for _ in range(num_images)
            ]
        else:
            cond_image_grid = (1, cond_h_patches, cond_w_patches)

        return QwenEditPromptData(
            prompt_embeds=pos_hidden_states[0].astype(mx.float16),
            prompt_mask=pos_hidden_states[1].astype(mx.float16),
            negative_prompt_embeds=neg_hidden_states[0].astype(mx.float16),
            negative_prompt_mask=neg_hidden_states[1].astype(mx.float16),
            conditioning_latents=conditioning_latents,
            qwen_image_ids=qwen_image_ids,
            cond_image_grid=cond_image_grid,
        )

    def set_image_dimensions(self, image_path: Path) -> tuple[int, int]:
        """Compute and store dimensions from input image.

        Returns:
            (output_width, output_height) for runtime config
        """
        vl_w, vl_h, vae_w, vae_h, out_w, out_h = self._compute_dimensions_from_image(
            image_path
        )
        self._vl_width = vl_w
        self._vl_height = vl_h
        self._vae_width = vae_w
        self._vae_height = vae_h
        return out_w, out_h

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
        from mflux.models.qwen.variants.txt2img.qwen_image import QwenImage

        return QwenImage.compute_guided_noise(
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
        embedded_hidden = self._transformer.img_in(hidden_states)
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
        """Compute time/text embeddings."""
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
        """Compute 3D rotary embeddings for Qwen edit."""
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
        rotary_embeddings: Any,
        kv_cache: ImagePatchKVCache | None,
        mode: BlockWrapperMode,
        text_seq_len: int,
        patch_start: int | None = None,
        patch_end: int | None = None,
        **kwargs: Any,
    ) -> tuple[mx.array, mx.array]:
        """Apply Qwen joint block."""
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
        """Merge image and text streams."""
        return mx.concatenate([encoder_hidden_states, hidden_states], axis=1)

    def _apply_joint_block_caching(
        self,
        block: Any,
        hidden_states: mx.array,
        encoder_hidden_states: mx.array,
        text_embeddings: mx.array,
        rotary_embeddings: tuple[tuple[mx.array, mx.array], tuple[mx.array, mx.array]],
        kv_cache: ImagePatchKVCache | None,
        text_seq_len: int,
        encoder_hidden_states_mask: mx.array | None = None,
        block_idx: int | None = None,
    ) -> tuple[mx.array, mx.array]:
        """Apply joint block in caching mode."""
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
        block: Any,
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

        # Modulation parameters
        img_mod_params = block.img_mod_linear(block.img_mod_silu(text_embeddings))
        txt_mod_params = block.txt_mod_linear(block.txt_mod_silu(text_embeddings))

        img_mod1, img_mod2 = mx.split(img_mod_params, 2, axis=-1)
        txt_mod1, txt_mod2 = mx.split(txt_mod_params, 2, axis=-1)

        # Normalization and modulation
        img_normed = block.img_norm1(patch_hidden)
        img_modulated, img_gate1 = QwenTransformerBlock._modulate(img_normed, img_mod1)

        txt_normed = block.txt_norm1(encoder_hidden_states)
        txt_modulated, txt_gate1 = QwenTransformerBlock._modulate(txt_normed, txt_mod1)

        # Q, K, V for image patch
        img_query = attn.to_q(img_modulated)
        img_key = attn.to_k(img_modulated)
        img_value = attn.to_v(img_modulated)

        # Q, K, V for text
        txt_query = attn.add_q_proj(txt_modulated)
        txt_key = attn.add_k_proj(txt_modulated)
        txt_value = attn.add_v_proj(txt_modulated)

        # Reshape to [B, S, H, D]
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

        # RMSNorm to Q, K
        img_query = attn.norm_q(img_query)
        img_key = attn.norm_k(img_key)
        txt_query = attn.norm_added_q(txt_query)
        txt_key = attn.norm_added_k(txt_key)

        # Extract RoPE for patch
        (img_cos, img_sin), (txt_cos, txt_sin) = rotary_embeddings
        patch_img_cos = img_cos[patch_start:patch_end]
        patch_img_sin = img_sin[patch_start:patch_end]

        # Apply RoPE
        img_query = QwenAttention._apply_rope_qwen(
            img_query, patch_img_cos, patch_img_sin
        )
        img_key = QwenAttention._apply_rope_qwen(img_key, patch_img_cos, patch_img_sin)
        txt_query = QwenAttention._apply_rope_qwen(txt_query, txt_cos, txt_sin)
        txt_key = QwenAttention._apply_rope_qwen(txt_key, txt_cos, txt_sin)

        # Transpose to [B, H, S, D]
        img_key_bhsd = mx.transpose(img_key, (0, 2, 1, 3))
        img_value_bhsd = mx.transpose(img_value, (0, 2, 1, 3))

        # Update cache
        kv_cache.update_image_patch(
            patch_start=patch_start,
            patch_end=patch_end,
            key=img_key_bhsd,
            value=img_value_bhsd,
        )

        # Get full K, V from cache
        txt_key_bhsd = mx.transpose(txt_key, (0, 2, 1, 3))
        txt_value_bhsd = mx.transpose(txt_value, (0, 2, 1, 3))
        full_key, full_value = kv_cache.get_full_kv(
            text_key=txt_key_bhsd,
            text_value=txt_value_bhsd,
        )

        # Build query
        joint_query = mx.concatenate([txt_query, img_query], axis=1)

        # Build attention mask
        mask = QwenAttention._convert_mask_for_qwen(
            mask=encoder_hidden_states_mask,
            joint_seq_len=full_key.shape[2],
            txt_seq_len=text_seq_len,
        )

        # Compute attention
        hidden_states = attn._compute_attention_qwen(
            query=joint_query,
            key=mx.transpose(full_key, (0, 2, 1, 3)),
            value=mx.transpose(full_value, (0, 2, 1, 3)),
            mask=mask,
            block_idx=block_idx,
        )

        # Extract outputs
        txt_attn_output = hidden_states[:, :text_seq_len, :]
        img_attn_output = hidden_states[:, text_seq_len:, :]

        # Project
        img_attn_output = attn.attn_to_out[0](img_attn_output)
        txt_attn_output = attn.to_add_out(txt_attn_output)

        # Residual + gate
        patch_hidden = patch_hidden + img_gate1 * img_attn_output
        encoder_hidden_states = encoder_hidden_states + txt_gate1 * txt_attn_output

        # Feed-forward for image
        img_normed2 = block.img_norm2(patch_hidden)
        img_modulated2, img_gate2 = QwenTransformerBlock._modulate(
            img_normed2, img_mod2
        )
        img_mlp_output = block.img_ff(img_modulated2)
        patch_hidden = patch_hidden + img_gate2 * img_mlp_output

        # Feed-forward for text
        txt_normed2 = block.txt_norm2(encoder_hidden_states)
        txt_modulated2, txt_gate2 = QwenTransformerBlock._modulate(
            txt_normed2, txt_mod2
        )
        txt_mlp_output = block.txt_ff(txt_modulated2)
        encoder_hidden_states = encoder_hidden_states + txt_gate2 * txt_mlp_output

        return encoder_hidden_states, patch_hidden
