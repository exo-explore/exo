import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import mlx.core as mx
from mflux.models.common.config.config import Config
from mflux.models.qwen.latent_creator.qwen_latent_creator import QwenLatentCreator
from mflux.models.qwen.model.qwen_transformer.qwen_transformer import QwenTransformer
from mflux.models.qwen.variants.edit.qwen_edit_util import QwenEditUtil
from mflux.models.qwen.variants.edit.qwen_image_edit import QwenImageEdit

from exo.worker.engines.image.config import ImageModelConfig
from exo.worker.engines.image.models.base import (
    ModelAdapter,
    PromptData,
    RotaryEmbeddings,
)
from exo.worker.engines.image.models.qwen.wrappers import QwenJointBlockWrapper
from exo.worker.engines.image.pipeline.block_wrapper import (
    JointBlockWrapper,
    SingleBlockWrapper,
)


@dataclass(frozen=True)
class EditImageDimensions:
    vl_width: int
    vl_height: int
    vae_width: int
    vae_height: int
    image_paths: list[str]


class QwenEditPromptData(PromptData):
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
        self._prompt_mask = prompt_mask
        self._negative_prompt_embeds = negative_prompt_embeds
        self._negative_prompt_mask = negative_prompt_mask
        self._conditioning_latents = conditioning_latents
        self._qwen_image_ids = qwen_image_ids
        self._cond_image_grid = cond_image_grid

    @property
    def prompt_embeds(self) -> mx.array:
        return self._prompt_embeds

    @property
    def pooled_prompt_embeds(self) -> mx.array:
        return self._prompt_embeds

    @property
    def negative_prompt_embeds(self) -> mx.array:
        return self._negative_prompt_embeds

    @property
    def negative_pooled_prompt_embeds(self) -> mx.array:
        return self._negative_prompt_embeds

    def get_encoder_hidden_states_mask(self, positive: bool = True) -> mx.array:
        if positive:
            return self._prompt_mask
        else:
            return self._negative_prompt_mask

    @property
    def cond_image_grid(self) -> tuple[int, int, int] | list[tuple[int, int, int]]:
        return self._cond_image_grid

    @property
    def conditioning_latents(self) -> mx.array:
        return self._conditioning_latents

    @property
    def qwen_image_ids(self) -> mx.array:
        return self._qwen_image_ids

    @property
    def is_edit_mode(self) -> bool:
        return True

    def get_batched_cfg_data(
        self,
    ) -> tuple[mx.array, mx.array, mx.array | None, mx.array | None] | None:
        """Batch positive and negative embeddings for CFG with batch_size=2.

        Pads shorter sequence to max length using zeros for embeddings
        and zeros (masked) for attention mask. Duplicates conditioning
        latents for both positive and negative passes.

        Returns:
            Tuple of (batched_embeds, batched_mask, None, batched_cond_latents)
            - batched_embeds: [2, max_seq, hidden]
            - batched_mask: [2, max_seq]
            - None for pooled (Qwen doesn't use it)
            - batched_cond_latents: [2, latent_seq, latent_dim]
            TODO(ciaran): type this
        """
        pos_embeds = self._prompt_embeds
        neg_embeds = self._negative_prompt_embeds
        pos_mask = self._prompt_mask
        neg_mask = self._negative_prompt_mask

        pos_seq_len = pos_embeds.shape[1]
        neg_seq_len = neg_embeds.shape[1]
        max_seq_len = max(pos_seq_len, neg_seq_len)
        hidden_dim = pos_embeds.shape[2]

        if pos_seq_len < max_seq_len:
            pad_len = max_seq_len - pos_seq_len
            pos_embeds = mx.concatenate(
                [
                    pos_embeds,
                    mx.zeros((1, pad_len, hidden_dim), dtype=pos_embeds.dtype),
                ],
                axis=1,
            )
            pos_mask = mx.concatenate(
                [pos_mask, mx.zeros((1, pad_len), dtype=pos_mask.dtype)],
                axis=1,
            )

        if neg_seq_len < max_seq_len:
            pad_len = max_seq_len - neg_seq_len
            neg_embeds = mx.concatenate(
                [
                    neg_embeds,
                    mx.zeros((1, pad_len, hidden_dim), dtype=neg_embeds.dtype),
                ],
                axis=1,
            )
            neg_mask = mx.concatenate(
                [neg_mask, mx.zeros((1, pad_len), dtype=neg_mask.dtype)],
                axis=1,
            )

        batched_embeds = mx.concatenate([pos_embeds, neg_embeds], axis=0)
        batched_mask = mx.concatenate([pos_mask, neg_mask], axis=0)

        batched_cond_latents = mx.concatenate(
            [self._conditioning_latents, self._conditioning_latents], axis=0
        )

        return batched_embeds, batched_mask, None, batched_cond_latents


class QwenEditModelAdapter(ModelAdapter[QwenImageEdit, QwenTransformer]):
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
            model_path=str(local_path),
        )
        self._transformer = self._model.transformer

        self._edit_dimensions: EditImageDimensions | None = None

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

    @property
    def needs_cfg(self) -> bool:
        gs = self._config.guidance_scale
        return gs is not None and gs > 1.0

    def _get_latent_creator(self) -> type[QwenLatentCreator]:
        return QwenLatentCreator

    def get_joint_block_wrappers(
        self,
        text_seq_len: int,
        encoder_hidden_states_mask: mx.array | None = None,
    ) -> list[JointBlockWrapper[Any]]:
        """Create wrapped joint blocks for Qwen Edit."""
        return [
            QwenJointBlockWrapper(block, text_seq_len, encoder_hidden_states_mask)
            for block in self._transformer.transformer_blocks
        ]

    def get_single_block_wrappers(
        self,
        text_seq_len: int,
    ) -> list[SingleBlockWrapper[Any]]:
        """Qwen has no single blocks."""
        return []

    def slice_transformer_blocks(
        self,
        start_layer: int,
        end_layer: int,
    ):
        self._transformer.transformer_blocks = self._transformer.transformer_blocks[
            start_layer:end_layer
        ]

    def set_image_dimensions(self, image_path: Path) -> tuple[int, int]:
        """Compute and store dimensions from input image.

        Also stores image_paths for use in encode_prompt().

        Returns:
            (output_width, output_height) for runtime config
        """
        vl_w, vl_h, vae_w, vae_h, out_w, out_h = self._compute_dimensions_from_image(
            image_path
        )
        self._edit_dimensions = EditImageDimensions(
            vl_width=vl_w,
            vl_height=vl_h,
            vae_width=vae_w,
            vae_height=vae_h,
            image_paths=[str(image_path)],
        )
        return out_w, out_h

    def create_latents(self, seed: int, runtime_config: Config) -> mx.array:
        """Create initial noise latents (pure noise for edit mode)."""
        return QwenLatentCreator.create_noise(
            seed=seed,
            height=runtime_config.height,
            width=runtime_config.width,
        )

    def encode_prompt(
        self, prompt: str, negative_prompt: str | None = None
    ) -> QwenEditPromptData:
        dims = self._edit_dimensions
        if dims is None:
            raise RuntimeError(
                "set_image_dimensions() must be called before encode_prompt() "
                "for QwenEditModelAdapter"
            )

        if negative_prompt is None or negative_prompt == "":
            negative_prompt = " "

        # TODO(ciaran): config is untyped and unused, unsure if Config or RuntimeConfig is intended
        (
            prompt_embeds,
            prompt_mask,
            negative_prompt_embeds,
            negative_prompt_mask,
        ) = self._model._encode_prompts_with_images(
            prompt,
            negative_prompt,
            dims.image_paths,
            self._config,  # pyright: ignore[reportArgumentType]
            dims.vl_width,
            dims.vl_height,
        )

        (
            conditioning_latents,
            qwen_image_ids,
            cond_h_patches,
            cond_w_patches,
            num_images,
        ) = QwenEditUtil.create_image_conditioning_latents(  # pyright: ignore[reportUnknownMemberType]
            vae=self._model.vae,
            height=dims.vae_height,
            width=dims.vae_width,
            image_paths=dims.image_paths,
            vl_width=dims.vl_width,
            vl_height=dims.vl_height,
        )

        if num_images > 1:
            cond_image_grid: tuple[int, int, int] | list[tuple[int, int, int]] = [
                (1, cond_h_patches, cond_w_patches) for _ in range(num_images)
            ]
        else:
            cond_image_grid = (1, cond_h_patches, cond_w_patches)

        return QwenEditPromptData(
            prompt_embeds=prompt_embeds,
            prompt_mask=prompt_mask,
            negative_prompt_embeds=negative_prompt_embeds,
            negative_prompt_mask=negative_prompt_mask,
            conditioning_latents=conditioning_latents,
            qwen_image_ids=qwen_image_ids,
            cond_image_grid=cond_image_grid,
        )

    def compute_embeddings(
        self,
        hidden_states: mx.array,
        prompt_embeds: mx.array,
    ) -> tuple[mx.array, mx.array]:
        embedded_hidden = self._transformer.img_in(hidden_states)
        encoder_hidden_states = self._transformer.txt_norm(prompt_embeds)
        embedded_encoder = self._transformer.txt_in(encoder_hidden_states)
        return embedded_hidden, embedded_encoder

    def compute_text_embeddings(
        self,
        t: int,
        runtime_config: Config,
        pooled_prompt_embeds: mx.array | None = None,
        hidden_states: mx.array | None = None,
    ) -> mx.array:
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
        return self._transformer.time_text_embed(timestep, ref_tensor)  # pyright: ignore[reportAny]

    def compute_rotary_embeddings(
        self,
        prompt_embeds: mx.array,
        runtime_config: Config,
        encoder_hidden_states_mask: mx.array | None = None,
        cond_image_grid: tuple[int, int, int]
        | list[tuple[int, int, int]]
        | None = None,
        kontext_image_ids: mx.array | None = None,
    ) -> RotaryEmbeddings:
        if encoder_hidden_states_mask is None:
            raise ValueError(
                "encoder_hidden_states_mask is required for Qwen RoPE computation"
            )

        return QwenTransformer._compute_rotary_embeddings(
            encoder_hidden_states_mask=encoder_hidden_states_mask,
            pos_embed=self._transformer.pos_embed,  # pyright: ignore[reportAny]
            config=runtime_config,
            cond_image_grid=cond_image_grid,
        )

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

    def _compute_dimensions_from_image(
        self, image_path: Path
    ) -> tuple[int, int, int, int, int, int]:
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
