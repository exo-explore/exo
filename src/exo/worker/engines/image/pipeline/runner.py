from math import ceil
from typing import Any, Optional

import mlx.core as mx
from mflux.models.common.config.config import Config
from mflux.utils.exceptions import StopImageGenerationException
from tqdm import tqdm

from exo.shared.types.worker.shards import PipelineShardMetadata
from exo.worker.engines.image.config import ImageModelConfig
from exo.worker.engines.image.models.base import (
    ModelAdapter,
    PromptData,
    RotaryEmbeddings,
)
from exo.worker.engines.image.pipeline.block_wrapper import (
    BlockWrapperMode,
    JointBlockWrapper,
    SingleBlockWrapper,
)


def calculate_patch_heights(
    latent_height: int, num_patches: int
) -> tuple[list[int], int]:
    patch_height = ceil(latent_height / num_patches)

    actual_num_patches = ceil(latent_height / patch_height)
    patch_heights = [patch_height] * (actual_num_patches - 1)

    last_height = latent_height - patch_height * (actual_num_patches - 1)
    patch_heights.append(last_height)

    return patch_heights, actual_num_patches


def calculate_token_indices(
    patch_heights: list[int], latent_width: int
) -> list[tuple[int, int]]:
    tokens_per_row = latent_width

    token_ranges: list[tuple[int, int]] = []
    cumulative_height = 0

    for h in patch_heights:
        start_token = tokens_per_row * cumulative_height
        end_token = tokens_per_row * (cumulative_height + h)

        token_ranges.append((start_token, end_token))
        cumulative_height += h

    return token_ranges


class DiffusionRunner:
    """Orchestrates the diffusion loop for image generation.

    In distributed mode, it implements PipeFusion with:
    - Sync pipeline for initial timesteps (full image, all devices in lockstep)
    - Async pipeline for later timesteps (patches processed independently)
    """

    def __init__(
        self,
        config: ImageModelConfig,
        adapter: ModelAdapter[Any, Any],
        group: Optional[mx.distributed.Group],
        shard_metadata: PipelineShardMetadata,
        num_patches: Optional[int] = None,
    ):
        self.config = config
        self.adapter = adapter
        self.group = group

        if group is None:
            self.rank = 0
            self.world_size = 1
            self.next_rank = 0
            self.prev_rank = 0
            self.start_layer = 0
            self.end_layer = config.total_blocks
        else:
            self.rank = shard_metadata.device_rank
            self.world_size = shard_metadata.world_size
            self.next_rank = (self.rank + 1) % self.world_size
            self.prev_rank = (self.rank - 1 + self.world_size) % self.world_size
            self.start_layer = shard_metadata.start_layer
            self.end_layer = shard_metadata.end_layer

        self.num_patches = num_patches if num_patches else max(1, self.world_size)

        self.total_joint = config.joint_block_count
        self.total_single = config.single_block_count
        self.total_layers = config.total_blocks

        self._guidance_override: float | None = None

        self._compute_assigned_blocks()

    def _compute_assigned_blocks(self) -> None:
        """Determine which joint/single blocks this stage owns."""
        start = self.start_layer
        end = self.end_layer

        if end <= self.total_joint:
            self.joint_start = start
            self.joint_end = end
            self.single_start = 0
            self.single_end = 0
        elif start >= self.total_joint:
            self.joint_start = 0
            self.joint_end = 0
            self.single_start = start - self.total_joint
            self.single_end = end - self.total_joint
        else:
            self.joint_start = start
            self.joint_end = self.total_joint
            self.single_start = 0
            self.single_end = end - self.total_joint

        self.has_joint_blocks = self.joint_end > self.joint_start
        self.has_single_blocks = self.single_end > self.single_start

        self.owns_concat_stage = self.has_joint_blocks and (
            self.has_single_blocks or self.end_layer == self.total_joint
        )

        # Wrappers created lazily on first forward (need text_seq_len)
        self.joint_block_wrappers: list[JointBlockWrapper[Any]] | None = None
        self.single_block_wrappers: list[SingleBlockWrapper[Any]] | None = None
        self._wrappers_initialized = False
        self._current_text_seq_len: int | None = None

    @property
    def is_first_stage(self) -> bool:
        return self.rank == 0

    @property
    def is_last_stage(self) -> bool:
        return self.rank == self.world_size - 1

    @property
    def is_distributed(self) -> bool:
        return self.group is not None

    def _get_effective_guidance_scale(self) -> float | None:
        if self._guidance_override is not None:
            return self._guidance_override
        return self.config.guidance_scale

    def _ensure_wrappers(
        self,
        text_seq_len: int,
        encoder_hidden_states_mask: mx.array | None = None,
    ) -> None:
        """Lazily create block wrappers on first forward pass.

        Wrappers need text_seq_len which is only known after prompt encoding.
        Re-initializes if text_seq_len changes (e.g., warmup vs real generation).
        """
        if self._wrappers_initialized and self._current_text_seq_len == text_seq_len:
            return

        self.joint_block_wrappers = self.adapter.get_joint_block_wrappers(
            text_seq_len=text_seq_len,
            encoder_hidden_states_mask=encoder_hidden_states_mask,
        )
        self.single_block_wrappers = self.adapter.get_single_block_wrappers(
            text_seq_len=text_seq_len,
        )
        self._wrappers_initialized = True
        self._current_text_seq_len = text_seq_len

    def _reset_all_caches(self) -> None:
        """Reset KV caches on all wrappers for a new generation."""
        if self.joint_block_wrappers:
            for wrapper in self.joint_block_wrappers:
                wrapper.reset_cache()
        if self.single_block_wrappers:
            for wrapper in self.single_block_wrappers:
                wrapper.reset_cache()

    def _set_text_seq_len(self, text_seq_len: int) -> None:
        if self.joint_block_wrappers:
            for wrapper in self.joint_block_wrappers:
                wrapper.set_text_seq_len(text_seq_len)
        if self.single_block_wrappers:
            for wrapper in self.single_block_wrappers:
                wrapper.set_text_seq_len(text_seq_len)

    def _calculate_capture_steps(
        self,
        partial_images: int,
        init_time_step: int,
        num_inference_steps: int,
    ) -> set[int]:
        """Calculate which timesteps should produce partial images.

        Places the first partial after step 1 for fast initial feedback,
        then evenly spaces remaining partials with equal gaps between them
        and from the last partial to the final image.

        Args:
            partial_images: Number of partial images to capture
            init_time_step: Starting timestep (for img2img this may not be 0)
            num_inference_steps: Total inference steps

        Returns:
            Set of timestep indices to capture
        """
        if partial_images <= 0:
            return set()

        total_steps = num_inference_steps - init_time_step
        if total_steps <= 1:
            return set()

        if partial_images >= total_steps - 1:
            return set(range(init_time_step, num_inference_steps - 1))

        capture_steps: set[int] = set()

        first_capture = init_time_step + 1
        capture_steps.add(first_capture)

        if partial_images == 1:
            return capture_steps

        final_step = num_inference_steps - 1
        remaining_range = final_step - first_capture

        for i in range(1, partial_images):
            step_idx = first_capture + int(i * remaining_range / partial_images)
            capture_steps.add(step_idx)

        return capture_steps

    def generate_image(
        self,
        runtime_config: Config,
        prompt: str,
        seed: int,
        partial_images: int = 0,
        guidance_override: float | None = None,
        negative_prompt: str | None = None,
        num_sync_steps: int = 1,
    ):
        """Primary entry point for image generation.

        Orchestrates the full generation flow:
        1. Create runtime config
        2. Create initial latents
        3. Encode prompt
        4. Run diffusion loop (yielding partials if requested)
        5. Decode to image

        Args:
            settings: Generation config (steps, height, width)
            prompt: Text prompt
            seed: Random seed
            partial_images: Number of intermediate images to yield (0 for none)
            guidance_override: Optional override for guidance scale (CFG)

        Yields:
            Partial images as (GeneratedImage, partial_index, total_partials) tuples
            Final GeneratedImage
        """
        self._guidance_override = guidance_override
        latents = self.adapter.create_latents(seed, runtime_config)
        prompt_data = self.adapter.encode_prompt(prompt, negative_prompt)

        capture_steps = self._calculate_capture_steps(
            partial_images=partial_images,
            init_time_step=runtime_config.init_time_step,
            num_inference_steps=runtime_config.num_inference_steps,
        )

        diffusion_gen = self._run_diffusion_loop(
            latents=latents,
            prompt_data=prompt_data,
            runtime_config=runtime_config,
            seed=seed,
            prompt=prompt,
            capture_steps=capture_steps,
            num_sync_steps=num_sync_steps,
        )

        partial_index = 0
        total_partials = len(capture_steps)

        if capture_steps:
            try:
                while True:
                    partial_latents, _step = next(diffusion_gen)
                    if self.is_last_stage:
                        partial_image = self.adapter.decode_latents(
                            partial_latents, runtime_config, seed, prompt
                        )
                        yield (partial_image, partial_index, total_partials)
                        partial_index += 1
            except StopIteration as e:
                latents = e.value  # pyright: ignore[reportAny]
        else:
            try:
                while True:
                    next(diffusion_gen)
            except StopIteration as e:
                latents = e.value  # pyright: ignore[reportAny]

        if self.is_last_stage:
            yield self.adapter.decode_latents(latents, runtime_config, seed, prompt)  # pyright: ignore[reportAny]

    def _run_diffusion_loop(
        self,
        latents: mx.array,
        prompt_data: PromptData,
        runtime_config: Config,
        seed: int,
        prompt: str,
        num_sync_steps: int,
        capture_steps: set[int] | None = None,
    ):
        if capture_steps is None:
            capture_steps = set()

        self._reset_all_caches()

        time_steps = tqdm(range(runtime_config.num_inference_steps))

        ctx = self.adapter.model.callbacks.start(  # pyright: ignore[reportAny]
            seed=seed, prompt=prompt, config=runtime_config
        )

        ctx.before_loop(  # pyright: ignore[reportAny]
            latents=latents,
        )

        for t in time_steps:
            try:
                latents = self._diffusion_step(
                    t=t,
                    config=runtime_config,
                    latents=latents,
                    prompt_data=prompt_data,
                    num_sync_steps=num_sync_steps,
                )

                ctx.in_loop(  # pyright: ignore[reportAny]
                    t=t,
                    latents=latents,
                )

                mx.eval(latents)

                if t in capture_steps and self.is_last_stage:
                    yield (latents, t)

            except KeyboardInterrupt:  # noqa: PERF203
                ctx.interruption(t=t, latents=latents)  # pyright: ignore[reportAny]
                raise StopImageGenerationException(
                    f"Stopping image generation at step {t + 1}/{len(time_steps)}"
                ) from None

        ctx.after_loop(latents=latents)  # pyright: ignore[reportAny]

        return latents

    def _forward_pass(
        self,
        latents: mx.array,
        prompt_embeds: mx.array,
        pooled_prompt_embeds: mx.array,
        t: int,
        config: Config,
        encoder_hidden_states_mask: mx.array | None = None,
        cond_image_grid: tuple[int, int, int]
        | list[tuple[int, int, int]]
        | None = None,
        conditioning_latents: mx.array | None = None,
    ) -> mx.array:
        """Run a single forward pass through the transformer.
        Args:
            latents: Input latents (already scaled by caller)
            prompt_embeds: Text embeddings
            pooled_prompt_embeds: Pooled text embeddings (Flux) or placeholder (Qwen)
            t: Current timestep
            config: Runtime configuration
            encoder_hidden_states_mask: Attention mask for text (Qwen)
            cond_image_grid: Conditioning image grid dimensions (Qwen edit)
            conditioning_latents: Conditioning latents for edit mode

        Returns:
            Noise prediction tensor
        """
        text_seq_len = prompt_embeds.shape[1]

        self._ensure_wrappers(text_seq_len, encoder_hidden_states_mask)

        if self.joint_block_wrappers and encoder_hidden_states_mask is not None:
            for wrapper in self.joint_block_wrappers:
                wrapper.set_encoder_mask(encoder_hidden_states_mask)

        scaled_latents = config.scheduler.scale_model_input(latents, t)  # pyright: ignore[reportAny]

        # For edit mode: concatenate with conditioning latents
        original_latent_tokens: int = scaled_latents.shape[1]  # pyright: ignore[reportAny]
        if conditioning_latents is not None:
            scaled_latents = mx.concatenate(
                [scaled_latents, conditioning_latents], axis=1
            )

        hidden_states, encoder_hidden_states = self.adapter.compute_embeddings(
            scaled_latents, prompt_embeds
        )
        text_embeddings = self.adapter.compute_text_embeddings(
            t, config, pooled_prompt_embeds, hidden_states=hidden_states
        )
        rotary_embeddings = self.adapter.compute_rotary_embeddings(
            prompt_embeds,
            config,
            encoder_hidden_states_mask=encoder_hidden_states_mask,
            cond_image_grid=cond_image_grid,
        )

        assert self.joint_block_wrappers is not None
        for wrapper in self.joint_block_wrappers:
            encoder_hidden_states, hidden_states = wrapper(
                hidden_states=hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                text_embeddings=text_embeddings,
                rotary_embeddings=rotary_embeddings,
            )

        if self.joint_block_wrappers:
            hidden_states = self.adapter.merge_streams(
                hidden_states, encoder_hidden_states
            )

        assert self.single_block_wrappers is not None
        for wrapper in self.single_block_wrappers:
            hidden_states = wrapper(
                hidden_states=hidden_states,
                text_embeddings=text_embeddings,
                rotary_embeddings=rotary_embeddings,
            )

        # Extract image portion
        hidden_states = hidden_states[:, text_seq_len:, ...]

        # For edit mode: extract only the generated portion (exclude conditioning latents)
        if conditioning_latents is not None:
            hidden_states = hidden_states[:, :original_latent_tokens, ...]

        return self.adapter.final_projection(hidden_states, text_embeddings)

    def _diffusion_step(
        self,
        t: int,
        config: Config,
        latents: mx.array,
        prompt_data: PromptData,
        num_sync_steps: int,
    ) -> mx.array:
        if self.group is None:
            return self._single_node_step(t, config, latents, prompt_data)
        elif t < config.init_time_step + num_sync_steps:
            return self._sync_pipeline_step(
                t,
                config,
                latents,
                prompt_data,
            )
        else:
            return self._async_pipeline_step(
                t,
                config,
                latents,
                prompt_data,
                is_first_async_step=t == config.init_time_step + num_sync_steps,
            )

    def _single_node_step(
        self,
        t: int,
        config: Config,
        latents: mx.array,
        prompt_data: PromptData,
    ) -> mx.array:
        cond_image_grid = prompt_data.cond_image_grid
        needs_cfg = self.adapter.needs_cfg

        if needs_cfg:
            batched_data = prompt_data.get_batched_cfg_data()
            assert batched_data is not None, "CFG model must provide batched data"
            prompt_embeds, encoder_mask, batched_pooled, cond_latents = batched_data
            pooled_embeds = (
                batched_pooled if batched_pooled is not None else prompt_embeds
            )
            step_latents = mx.concatenate([latents, latents], axis=0)
        else:
            prompt_embeds = prompt_data.prompt_embeds
            pooled_embeds = prompt_data.pooled_prompt_embeds
            encoder_mask = prompt_data.get_encoder_hidden_states_mask(positive=True)
            cond_latents = prompt_data.conditioning_latents
            step_latents = latents

        noise = self._forward_pass(
            step_latents,
            prompt_embeds,
            pooled_embeds,
            t=t,
            config=config,
            encoder_hidden_states_mask=encoder_mask,
            cond_image_grid=cond_image_grid,
            conditioning_latents=cond_latents,
        )

        if needs_cfg:
            noise_pos, noise_neg = mx.split(noise, 2, axis=0)
            guidance_scale = self._get_effective_guidance_scale()
            assert guidance_scale is not None
            noise = self.adapter.apply_guidance(
                noise_pos, noise_neg, guidance_scale=guidance_scale
            )

        return config.scheduler.step(noise=noise, timestep=t, latents=latents)  # pyright: ignore[reportAny]

    def _create_patches(
        self,
        latents: mx.array,
        config: Config,
    ) -> tuple[list[mx.array], list[tuple[int, int]]]:
        latent_height = config.height // 16
        latent_width = config.width // 16

        patch_heights, _ = calculate_patch_heights(latent_height, self.num_patches)
        token_indices = calculate_token_indices(patch_heights, latent_width)

        patch_latents = [latents[:, start:end, :] for start, end in token_indices]

        return patch_latents, token_indices

    def _run_sync_pass(
        self,
        t: int,
        config: Config,
        scaled_hidden_states: mx.array,
        prompt_embeds: mx.array,
        pooled_prompt_embeds: mx.array,
        encoder_hidden_states_mask: mx.array | None,
        cond_image_grid: tuple[int, int, int] | list[tuple[int, int, int]] | None,
        kontext_image_ids: mx.array | None,
        num_img_tokens: int,
        original_latent_tokens: int,
        conditioning_latents: mx.array | None,
    ) -> mx.array | None:
        hidden_states = scaled_hidden_states
        batch_size = hidden_states.shape[0]
        text_seq_len = prompt_embeds.shape[1]
        hidden_dim = self.adapter.hidden_dim
        dtype = scaled_hidden_states.dtype

        self._set_text_seq_len(text_seq_len)

        if self.joint_block_wrappers:
            for wrapper in self.joint_block_wrappers:
                wrapper.set_encoder_mask(encoder_hidden_states_mask)

        encoder_hidden_states: mx.array | None = None
        if self.is_first_stage:
            hidden_states, encoder_hidden_states = self.adapter.compute_embeddings(
                hidden_states, prompt_embeds
            )

        text_embeddings = self.adapter.compute_text_embeddings(
            t, config, pooled_prompt_embeds
        )
        image_rotary_embeddings = self.adapter.compute_rotary_embeddings(
            prompt_embeds,
            config,
            encoder_hidden_states_mask=encoder_hidden_states_mask,
            cond_image_grid=cond_image_grid,
            kontext_image_ids=kontext_image_ids,
        )

        if self.has_joint_blocks:
            if not self.is_first_stage:
                hidden_states = mx.distributed.recv(
                    (batch_size, num_img_tokens, hidden_dim),
                    dtype,
                    self.prev_rank,
                    group=self.group,
                )
                encoder_hidden_states = mx.distributed.recv(
                    (batch_size, text_seq_len, hidden_dim),
                    dtype,
                    self.prev_rank,
                    group=self.group,
                )
                mx.eval(hidden_states, encoder_hidden_states)

            assert self.joint_block_wrappers is not None
            assert encoder_hidden_states is not None
            for wrapper in self.joint_block_wrappers:
                wrapper.set_patch(BlockWrapperMode.CACHING)
                encoder_hidden_states, hidden_states = wrapper(
                    hidden_states=hidden_states,
                    encoder_hidden_states=encoder_hidden_states,
                    text_embeddings=text_embeddings,
                    rotary_embeddings=image_rotary_embeddings,
                )

        if self.owns_concat_stage:
            assert encoder_hidden_states is not None
            concatenated = self.adapter.merge_streams(
                hidden_states, encoder_hidden_states
            )

            if self.has_single_blocks or self.is_last_stage:
                hidden_states = concatenated
            else:
                concatenated = mx.distributed.send(
                    concatenated, self.next_rank, group=self.group
                )
                mx.async_eval(concatenated)

        elif self.has_joint_blocks and not self.is_last_stage:
            assert encoder_hidden_states is not None
            hidden_states = mx.distributed.send(
                hidden_states, self.next_rank, group=self.group
            )
            encoder_hidden_states = mx.distributed.send(
                encoder_hidden_states, self.next_rank, group=self.group
            )
            mx.async_eval(hidden_states, encoder_hidden_states)

        if self.has_single_blocks:
            if not self.owns_concat_stage and not self.is_first_stage:
                hidden_states = mx.distributed.recv(
                    (batch_size, text_seq_len + num_img_tokens, hidden_dim),
                    dtype,
                    self.prev_rank,
                    group=self.group,
                )
                mx.eval(hidden_states)

            assert self.single_block_wrappers is not None
            for wrapper in self.single_block_wrappers:
                wrapper.set_patch(BlockWrapperMode.CACHING)
                hidden_states = wrapper(
                    hidden_states=hidden_states,
                    text_embeddings=text_embeddings,
                    rotary_embeddings=image_rotary_embeddings,
                )

            if not self.is_last_stage:
                hidden_states = mx.distributed.send(
                    hidden_states, self.next_rank, group=self.group
                )
                mx.async_eval(hidden_states)

        hidden_states = hidden_states[:, text_seq_len:, ...]

        if conditioning_latents is not None:
            hidden_states = hidden_states[:, :original_latent_tokens, ...]

        if self.is_last_stage:
            return self.adapter.final_projection(hidden_states, text_embeddings)

        return None

    def _sync_pipeline_step(
        self,
        t: int,
        config: Config,
        hidden_states: mx.array,
        prompt_data: PromptData,
        kontext_image_ids: mx.array | None = None,
    ) -> mx.array:
        prev_latents = hidden_states
        needs_cfg = self.adapter.needs_cfg
        cond_image_grid = prompt_data.cond_image_grid

        scaled_hidden_states = config.scheduler.scale_model_input(hidden_states, t)  # pyright: ignore[reportAny]
        original_latent_tokens: int = scaled_hidden_states.shape[1]  # pyright: ignore[reportAny]

        if needs_cfg:
            batched_data = prompt_data.get_batched_cfg_data()
            assert batched_data is not None, "CFG model must provide batched data"
            prompt_embeds, encoder_mask, batched_pooled, cond_latents = batched_data
            pooled_embeds = (
                batched_pooled if batched_pooled is not None else prompt_embeds
            )
            step_latents = mx.concatenate(
                [scaled_hidden_states, scaled_hidden_states], axis=0
            )
        else:
            prompt_embeds = prompt_data.prompt_embeds
            pooled_embeds = prompt_data.pooled_prompt_embeds
            encoder_mask = prompt_data.get_encoder_hidden_states_mask(positive=True)
            cond_latents = prompt_data.conditioning_latents
            step_latents = scaled_hidden_states  # pyright: ignore[reportAny]

        if cond_latents is not None:
            num_img_tokens: int = original_latent_tokens + cond_latents.shape[1]
        else:
            num_img_tokens = original_latent_tokens

        if self.is_first_stage and cond_latents is not None:
            step_latents = mx.concatenate([step_latents, cond_latents], axis=1)

        text_seq_len = prompt_embeds.shape[1]
        self._ensure_wrappers(text_seq_len, encoder_mask)

        noise = self._run_sync_pass(
            t,
            config,
            step_latents,
            prompt_embeds,
            pooled_embeds,
            encoder_mask,
            cond_image_grid,
            kontext_image_ids,
            num_img_tokens,
            original_latent_tokens,
            cond_latents,
        )

        if self.is_last_stage:
            assert noise is not None
            if needs_cfg:
                noise_pos, noise_neg = mx.split(noise, 2, axis=0)
                guidance_scale = self._get_effective_guidance_scale()
                assert guidance_scale is not None
                noise = self.adapter.apply_guidance(
                    noise_pos, noise_neg, guidance_scale
                )

            hidden_states = config.scheduler.step(  # pyright: ignore[reportAny]
                noise=noise, timestep=t, latents=prev_latents
            )

            if not self.is_first_stage:
                hidden_states = mx.distributed.send(hidden_states, 0, group=self.group)
                mx.async_eval(hidden_states)

        elif self.is_first_stage:
            hidden_states = mx.distributed.recv_like(
                prev_latents, src=self.world_size - 1, group=self.group
            )
            mx.eval(hidden_states)

        else:
            hidden_states = prev_latents

        return hidden_states

    def _async_pipeline_step(
        self,
        t: int,
        config: Config,
        latents: mx.array,
        prompt_data: PromptData,
        is_first_async_step: bool,
        kontext_image_ids: mx.array | None = None,
    ) -> mx.array:
        patch_latents, token_indices = self._create_patches(latents, config)
        needs_cfg = self.adapter.needs_cfg
        cond_image_grid = prompt_data.cond_image_grid

        if needs_cfg:
            batched_data = prompt_data.get_batched_cfg_data()
            assert batched_data is not None, "CFG model must provide batched data"
            prompt_embeds, encoder_mask, batched_pooled, _ = batched_data
            pooled_embeds = (
                batched_pooled if batched_pooled is not None else prompt_embeds
            )
        else:
            prompt_embeds = prompt_data.prompt_embeds
            pooled_embeds = prompt_data.pooled_prompt_embeds
            encoder_mask = prompt_data.get_encoder_hidden_states_mask(positive=True)

        text_seq_len = prompt_embeds.shape[1]
        self._ensure_wrappers(text_seq_len, encoder_mask)
        self._set_text_seq_len(text_seq_len)

        if self.joint_block_wrappers:
            for wrapper in self.joint_block_wrappers:
                wrapper.set_encoder_mask(encoder_mask)

        text_embeddings = self.adapter.compute_text_embeddings(t, config, pooled_embeds)
        image_rotary_embeddings = self.adapter.compute_rotary_embeddings(
            prompt_embeds,
            config,
            encoder_hidden_states_mask=encoder_mask,
            cond_image_grid=cond_image_grid,
            kontext_image_ids=kontext_image_ids,
        )

        prev_patch_latents = [p for p in patch_latents]
        encoder_hidden_states: mx.array | None = None

        for patch_idx in range(len(patch_latents)):
            patch = patch_latents[patch_idx]

            if (
                self.is_first_stage
                and not self.is_last_stage
                and not is_first_async_step
            ):
                patch = mx.distributed.recv_like(
                    patch, src=self.prev_rank, group=self.group
                )
                mx.eval(patch)

            step_patch = mx.concatenate([patch, patch], axis=0) if needs_cfg else patch

            noise, encoder_hidden_states = self._run_single_patch_pass(
                patch=step_patch,
                patch_idx=patch_idx,
                token_indices=token_indices[patch_idx],
                prompt_embeds=prompt_embeds,
                text_embeddings=text_embeddings,
                image_rotary_embeddings=image_rotary_embeddings,
                encoder_hidden_states=encoder_hidden_states,
            )

            if self.is_last_stage:
                assert noise is not None
                if needs_cfg:
                    noise_pos, noise_neg = mx.split(noise, 2, axis=0)
                    guidance_scale = self._get_effective_guidance_scale()
                    assert guidance_scale is not None
                    noise = self.adapter.apply_guidance(
                        noise_pos, noise_neg, guidance_scale
                    )

                patch_latents[patch_idx] = config.scheduler.step(  # pyright: ignore[reportAny]
                    noise=noise,
                    timestep=t,
                    latents=prev_patch_latents[patch_idx],
                )

                if not self.is_first_stage and t != config.num_inference_steps - 1:
                    patch_latents[patch_idx] = mx.distributed.send(
                        patch_latents[patch_idx], self.next_rank, group=self.group
                    )
                    mx.async_eval(patch_latents[patch_idx])

        return mx.concatenate(patch_latents, axis=1)

    def _run_single_patch_pass(
        self,
        patch: mx.array,
        patch_idx: int,
        token_indices: tuple[int, int],
        prompt_embeds: mx.array,
        text_embeddings: mx.array,
        image_rotary_embeddings: RotaryEmbeddings,
        encoder_hidden_states: mx.array | None,
    ) -> tuple[mx.array | None, mx.array | None]:
        """Process a single patch through the forward pipeline.

        Handles stage-to-stage communication (stage i -> stage i+1).
        Ring communication (last stage -> first stage) is handled by the caller.

        Args:
            patch: The patch latents to process
            patch_idx: Index of this patch (0-indexed)
            token_indices: (start_token, end_token) for this patch
            prompt_embeds: Text embeddings (for compute_embeddings on first stage)
            text_embeddings: Precomputed text embeddings
            image_rotary_embeddings: Precomputed rotary embeddings
            encoder_hidden_states: Encoder hidden states (passed between patches)

        Returns:
            (noise_prediction, encoder_hidden_states) - noise is None for non-last stages
        """
        start_token, end_token = token_indices
        batch_size = patch.shape[0]
        text_seq_len = prompt_embeds.shape[1]
        hidden_dim = self.adapter.hidden_dim

        if self.has_joint_blocks:
            if not self.is_first_stage:
                patch_len = patch.shape[1]
                patch = mx.distributed.recv(
                    (batch_size, patch_len, hidden_dim),
                    patch.dtype,
                    self.prev_rank,
                    group=self.group,
                )
                mx.eval(patch)

                if patch_idx == 0:
                    encoder_hidden_states = mx.distributed.recv(
                        (batch_size, text_seq_len, hidden_dim),
                        patch.dtype,
                        self.prev_rank,
                        group=self.group,
                    )
                    mx.eval(encoder_hidden_states)

            if self.is_first_stage:
                patch, encoder_hidden_states = self.adapter.compute_embeddings(
                    patch, prompt_embeds
                )

            assert self.joint_block_wrappers is not None
            assert encoder_hidden_states is not None
            for wrapper in self.joint_block_wrappers:
                wrapper.set_patch(BlockWrapperMode.PATCHED, start_token, end_token)
                encoder_hidden_states, patch = wrapper(
                    hidden_states=patch,
                    encoder_hidden_states=encoder_hidden_states,
                    text_embeddings=text_embeddings,
                    rotary_embeddings=image_rotary_embeddings,
                )

        if self.owns_concat_stage:
            assert encoder_hidden_states is not None
            patch_concat = self.adapter.merge_streams(patch, encoder_hidden_states)

            if self.has_single_blocks or self.is_last_stage:
                patch = patch_concat
            else:
                patch_concat = mx.distributed.send(
                    patch_concat, self.next_rank, group=self.group
                )
                mx.async_eval(patch_concat)

        elif self.has_joint_blocks and not self.is_last_stage:
            patch = mx.distributed.send(patch, self.next_rank, group=self.group)
            mx.async_eval(patch)

            if patch_idx == 0:
                assert encoder_hidden_states is not None
                encoder_hidden_states = mx.distributed.send(
                    encoder_hidden_states, self.next_rank, group=self.group
                )
                mx.async_eval(encoder_hidden_states)

        if self.has_single_blocks:
            if not self.owns_concat_stage and not self.is_first_stage:
                patch_len = patch.shape[1]
                patch = mx.distributed.recv(
                    (batch_size, text_seq_len + patch_len, hidden_dim),
                    patch.dtype,
                    self.prev_rank,
                    group=self.group,
                )
                mx.eval(patch)

            assert self.single_block_wrappers is not None
            for wrapper in self.single_block_wrappers:
                wrapper.set_patch(BlockWrapperMode.PATCHED, start_token, end_token)
                patch = wrapper(
                    hidden_states=patch,
                    text_embeddings=text_embeddings,
                    rotary_embeddings=image_rotary_embeddings,
                )

            if not self.is_last_stage:
                patch = mx.distributed.send(patch, self.next_rank, group=self.group)
                mx.async_eval(patch)

        noise: mx.array | None = None
        if self.is_last_stage:
            patch_img_only = patch[:, text_seq_len:, :]
            noise = self.adapter.final_projection(patch_img_only, text_embeddings)

        return noise, encoder_hidden_states
