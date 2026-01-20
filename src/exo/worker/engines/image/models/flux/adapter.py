from pathlib import Path
from typing import Any

import mlx.core as mx
from mflux.models.common.config.config import Config
from mflux.models.common.config.model_config import ModelConfig
from mflux.models.flux.latent_creator.flux_latent_creator import FluxLatentCreator
from mflux.models.flux.model.flux_text_encoder.prompt_encoder import PromptEncoder
from mflux.models.flux.model.flux_transformer.transformer import Transformer
from mflux.models.flux.variants.txt2img.flux import Flux1

from exo.worker.engines.image.config import ImageModelConfig
from exo.worker.engines.image.models.base import (
    ModelAdapter,
    PromptData,
    RotaryEmbeddings,
)
from exo.worker.engines.image.models.flux.wrappers import (
    FluxJointBlockWrapper,
    FluxSingleBlockWrapper,
)
from exo.worker.engines.image.pipeline.block_wrapper import (
    JointBlockWrapper,
    SingleBlockWrapper,
)


class FluxPromptData(PromptData):
    def __init__(self, prompt_embeds: mx.array, pooled_prompt_embeds: mx.array):
        self._prompt_embeds = prompt_embeds
        self._pooled_prompt_embeds = pooled_prompt_embeds

    @property
    def prompt_embeds(self) -> mx.array:
        return self._prompt_embeds

    @property
    def pooled_prompt_embeds(self) -> mx.array:
        return self._pooled_prompt_embeds

    @property
    def negative_prompt_embeds(self) -> mx.array | None:
        return None

    @property
    def negative_pooled_prompt_embeds(self) -> mx.array | None:
        return None

    def get_encoder_hidden_states_mask(self, positive: bool = True) -> mx.array | None:
        return None

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
        return None


class FluxModelAdapter(ModelAdapter[Flux1, Transformer]):
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
            model_path=str(local_path),
            quantize=quantize,
        )
        self._transformer = self._model.transformer

    @property
    def hidden_dim(self) -> int:
        return self._transformer.x_embedder.weight.shape[0]  # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType]

    @property
    def needs_cfg(self) -> bool:
        return False

    def _get_latent_creator(self) -> type:
        return FluxLatentCreator

    def get_joint_block_wrappers(
        self,
        text_seq_len: int,
        encoder_hidden_states_mask: mx.array | None = None,
    ) -> list[JointBlockWrapper[Any]]:
        """Create wrapped joint blocks for Flux."""
        return [
            FluxJointBlockWrapper(block, text_seq_len)
            for block in self._transformer.transformer_blocks
        ]

    def get_single_block_wrappers(
        self,
        text_seq_len: int,
    ) -> list[SingleBlockWrapper[Any]]:
        """Create wrapped single blocks for Flux."""
        return [
            FluxSingleBlockWrapper(block, text_seq_len)
            for block in self._transformer.single_transformer_blocks
        ]

    def slice_transformer_blocks(
        self,
        start_layer: int,
        end_layer: int,
    ):
        all_joint = list(self._transformer.transformer_blocks)
        all_single = list(self._transformer.single_transformer_blocks)
        total_joint_blocks = len(all_joint)
        if end_layer <= total_joint_blocks:
            # All assigned are joint blocks
            joint_start, joint_end = start_layer, end_layer
            single_start, single_end = 0, 0
        elif start_layer >= total_joint_blocks:
            # All assigned are single blocks
            joint_start, joint_end = 0, 0
            single_start = start_layer - total_joint_blocks
            single_end = end_layer - total_joint_blocks
        else:
            # Spans both joint and single
            joint_start, joint_end = start_layer, total_joint_blocks
            single_start = 0
            single_end = end_layer - total_joint_blocks

        self._transformer.transformer_blocks = all_joint[joint_start:joint_end]

        self._transformer.single_transformer_blocks = all_single[
            single_start:single_end
        ]

    def encode_prompt(
        self, prompt: str, negative_prompt: str | None = None
    ) -> FluxPromptData:
        del negative_prompt

        assert isinstance(self.model.prompt_cache, dict)
        assert isinstance(self.model.tokenizers, dict)

        prompt_embeds, pooled_prompt_embeds = PromptEncoder.encode_prompt(
            prompt=prompt,
            prompt_cache=self.model.prompt_cache,
            t5_tokenizer=self.model.tokenizers["t5"],  # pyright: ignore[reportAny]
            clip_tokenizer=self.model.tokenizers["clip"],  # pyright: ignore[reportAny]
            t5_text_encoder=self.model.t5_text_encoder,
            clip_text_encoder=self.model.clip_text_encoder,
        )
        return FluxPromptData(
            prompt_embeds=prompt_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds,
        )

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
        runtime_config: Config,
        pooled_prompt_embeds: mx.array | None = None,
        hidden_states: mx.array | None = None,  # Ignored by Flux
    ) -> mx.array:
        if pooled_prompt_embeds is None:
            raise ValueError(
                "pooled_prompt_embeds is required for Flux text embeddings"
            )

        # hidden_states is ignored - Flux uses pooled_prompt_embeds instead
        return Transformer.compute_text_embeddings(
            t, pooled_prompt_embeds, self._transformer.time_text_embed, runtime_config
        )

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
        return Transformer.compute_rotary_embeddings(
            prompt_embeds,
            self._transformer.pos_embed,
            runtime_config,
            kontext_image_ids,
        )

    def apply_guidance(
        self,
        noise_positive: mx.array,
        noise_negative: mx.array,
        guidance_scale: float,
    ) -> mx.array:
        raise NotImplementedError("Flux does not use classifier-free guidance")
