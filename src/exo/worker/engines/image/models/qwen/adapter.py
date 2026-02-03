from pathlib import Path
from typing import Any

import mlx.core as mx
from mflux.models.common.config import ModelConfig
from mflux.models.common.config.config import Config
from mflux.models.qwen.latent_creator.qwen_latent_creator import QwenLatentCreator
from mflux.models.qwen.model.qwen_text_encoder.qwen_prompt_encoder import (
    QwenPromptEncoder,
)
from mflux.models.qwen.model.qwen_transformer.qwen_transformer import QwenTransformer
from mflux.models.qwen.variants.txt2img.qwen_image import QwenImage

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


class QwenPromptData(PromptData):
    def __init__(
        self,
        prompt_embeds: mx.array,
        prompt_mask: mx.array,
        negative_prompt_embeds: mx.array,
        negative_prompt_mask: mx.array,
    ):
        self._prompt_embeds = prompt_embeds
        self._prompt_mask = prompt_mask
        self._negative_prompt_embeds = negative_prompt_embeds
        self._negative_prompt_mask = negative_prompt_mask

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
    def cond_image_grid(
        self,
    ) -> tuple[int, int, int] | list[tuple[int, int, int]] | None:
        return None

    @property
    def conditioning_latents(self) -> mx.array | None:
        return None

    def get_batched_cfg_data(
        self,
    ) -> tuple[mx.array, mx.array, mx.array | None, mx.array | None] | None:
        """Batch positive and negative embeddings for CFG with batch_size=2.

        Pads shorter sequence to max length using zeros for embeddings
        and zeros (masked) for attention mask.

        Returns:
            Tuple of (batched_embeds, batched_mask, None, conditioning_latents)
            - batched_embeds: [2, max_seq, hidden]
            - batched_mask: [2, max_seq]
            - None for pooled (Qwen doesn't use it)
            - conditioning_latents: [2, latent_seq, latent_dim] or None
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

        elif neg_seq_len < max_seq_len:
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

        # TODO(ciaran): currently None but maybe we will deduplicate with edit
        # adapter
        cond_latents = self.conditioning_latents
        if cond_latents is not None:
            cond_latents = mx.concatenate([cond_latents, cond_latents], axis=0)

        return batched_embeds, batched_mask, None, cond_latents


class QwenModelAdapter(ModelAdapter[QwenImage, QwenTransformer]):
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
            model_path=str(local_path),
            quantize=quantize,
        )
        self._transformer = self._model.transformer

    @property
    def hidden_dim(self) -> int:
        return self._transformer.inner_dim

    @property
    def needs_cfg(self) -> bool:
        gs = self._config.guidance_scale
        return gs is not None and gs > 1.0

    def _get_latent_creator(self) -> type:
        return QwenLatentCreator

    def get_joint_block_wrappers(
        self,
        text_seq_len: int,
        encoder_hidden_states_mask: mx.array | None = None,
    ) -> list[JointBlockWrapper[Any]]:
        """Create wrapped joint blocks for Qwen."""
        return [
            QwenJointBlockWrapper(block, text_seq_len, encoder_hidden_states_mask)
            for block in self._transformer.transformer_blocks
        ]

    def get_single_block_wrappers(
        self,
        text_seq_len: int,
    ) -> list[SingleBlockWrapper[Any]]:
        return []

    def slice_transformer_blocks(
        self,
        start_layer: int,
        end_layer: int,
    ):
        self._transformer.transformer_blocks = self._transformer.transformer_blocks[
            start_layer:end_layer
        ]

    def encode_prompt(
        self, prompt: str, negative_prompt: str | None = None
    ) -> QwenPromptData:
        assert isinstance(self.model.prompt_cache, dict)
        assert isinstance(self.model.tokenizers, dict)

        if negative_prompt is None or negative_prompt == "":
            negative_prompt = " "

        prompt_embeds, prompt_mask, neg_embeds, neg_mask = (
            QwenPromptEncoder.encode_prompt(
                prompt=prompt,
                negative_prompt=negative_prompt,
                prompt_cache=self.model.prompt_cache,
                qwen_tokenizer=self.model.tokenizers["qwen"],  # pyright: ignore[reportAny]
                qwen_text_encoder=self.model.text_encoder,
            )
        )

        return QwenPromptData(
            prompt_embeds=prompt_embeds,
            prompt_mask=prompt_mask,
            negative_prompt_embeds=neg_embeds,
            negative_prompt_mask=neg_mask,
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
        return self._model.compute_guided_noise(
            noise=noise_positive,
            noise_negative=noise_negative,
            guidance=guidance_scale,
        )
