from math import ceil
from typing import Optional

import mlx.core as mx
from mflux.models.common.config.config import Config
from mflux.utils.exceptions import StopImageGenerationException
from tqdm import tqdm

from exo.shared.types.worker.shards import PipelineShardMetadata
from exo.worker.engines.image.config import ImageModelConfig
from exo.worker.engines.image.models.base import ModelAdapter, PromptData
from exo.worker.engines.image.pipeline.adapter import BlockWrapperMode
from exo.worker.engines.image.pipeline.block_wrapper import (
    JointBlockWrapper,
    SingleBlockWrapper,
)
from exo.worker.engines.image.pipeline.kv_cache import ImagePatchKVCache


def calculate_patch_heights(latent_height: int, num_patches: int):
    patch_height = ceil(latent_height / num_patches)

    actual_num_patches = ceil(latent_height / patch_height)
    patch_heights = [patch_height] * (actual_num_patches - 1)

    last_height = latent_height - patch_height * (actual_num_patches - 1)
    patch_heights.append(last_height)

    return patch_heights, actual_num_patches


def calculate_token_indices(patch_heights: list[int], latent_width: int):
    tokens_per_row = latent_width

    token_ranges = []
    cumulative_height = 0

    for h in patch_heights:
        start_token = tokens_per_row * cumulative_height
        end_token = tokens_per_row * (cumulative_height + h)

        token_ranges.append((start_token, end_token))
        cumulative_height += h

    return token_ranges


class DiffusionRunner:
    """Orchestrates the diffusion loop for image generation.

    This class owns the entire diffusion process, handling both single-node
    and distributed (PipeFusion) modes.

    In distributed mode, it implements PipeFusion with:
    - Sync pipeline for initial timesteps (full image, all devices in lockstep)
    - Async pipeline for later timesteps (patches processed independently)
    """

    def __init__(
        self,
        config: ImageModelConfig,
        adapter: ModelAdapter,
        group: Optional[mx.distributed.Group],
        shard_metadata: PipelineShardMetadata,
        num_sync_steps: int = 1,
        num_patches: Optional[int] = None,
    ):
        """Initialize the diffusion runner.

        Args:
            config: Model configuration (architecture, block counts, etc.)
            adapter: Model adapter for model-specific operations
            group: MLX distributed group (None for single-node mode)
            shard_metadata: Pipeline shard metadata with layer assignments
            num_sync_steps: Number of synchronous timesteps before async mode
            num_patches: Number of patches for async mode (defaults to world_size)
        """
        self.config = config
        self.adapter = adapter
        self.group = group

        # Handle single-node vs distributed mode
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

        self.num_sync_steps = num_sync_steps
        self.num_patches = num_patches if num_patches else max(1, self.world_size)

        # Persistent KV caches (initialized on first async timestep, reused across timesteps)
        self.joint_kv_caches: list[ImagePatchKVCache] | None = None
        self.single_kv_caches: list[ImagePatchKVCache] | None = None

        # Get block counts from config (model-agnostic)
        self.total_joint = config.joint_block_count
        self.total_single = config.single_block_count
        self.total_layers = config.total_blocks

        self._compute_assigned_blocks()

    def _compute_assigned_blocks(self) -> None:
        """Determine which joint/single blocks this stage owns."""
        start = self.start_layer
        end = self.end_layer

        if end <= self.total_joint:
            # All assigned blocks are joint blocks
            self.joint_start = start
            self.joint_end = end
            self.single_start = 0
            self.single_end = 0
        elif start >= self.total_joint:
            # All assigned blocks are single blocks
            self.joint_start = 0
            self.joint_end = 0
            self.single_start = start - self.total_joint
            self.single_end = end - self.total_joint
        else:
            # Stage spans joint→single transition
            self.joint_start = start
            self.joint_end = self.total_joint
            self.single_start = 0
            self.single_end = end - self.total_joint

        self.has_joint_blocks = self.joint_end > self.joint_start
        self.has_single_blocks = self.single_end > self.single_start

        self.owns_concat_stage = self.has_joint_blocks and (
            self.has_single_blocks or self.end_layer == self.total_joint
        )

        joint_blocks = self.adapter.get_joint_blocks()
        single_blocks = self.adapter.get_single_blocks()

        # Wrap blocks at initialization (reused across all calls)
        self.joint_block_wrappers = [
            JointBlockWrapper(block=block, adapter=self.adapter)
            for block in joint_blocks
        ]
        self.single_block_wrappers = [
            SingleBlockWrapper(block=block, adapter=self.adapter)
            for block in single_blocks
        ]

    @property
    def is_first_stage(self) -> bool:
        return self.rank == 0

    @property
    def is_last_stage(self) -> bool:
        return self.rank == self.world_size - 1

    @property
    def is_distributed(self) -> bool:
        return self.group is not None

    def _calculate_capture_steps(
        self,
        partial_images: int,
        init_time_step: int,
        num_inference_steps: int,
    ) -> set[int]:
        """Calculate which timesteps should produce partial images.

        Evenly spaces `partial_images` captures across the diffusion loop.
        Does NOT include the final timestep (that's the complete image).

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
            # Capture every step except final
            return set(range(init_time_step, num_inference_steps - 1))

        # Evenly space partial captures
        step_interval = total_steps / (partial_images + 1)
        capture_steps: set[int] = set()
        for i in range(1, partial_images + 1):
            step_idx = int(init_time_step + i * step_interval)
            # Ensure we don't capture the final step
            if step_idx < num_inference_steps - 1:
                capture_steps.add(step_idx)

        return capture_steps

    def generate_image(
        self,
        runtime_config: Config,
        prompt: str,
        seed: int,
        partial_images: int = 0,
    ):
        """Primary entry point for image generation.

        Orchestrates the full generation flow:
        1. Create runtime config
        2. Create initial latents
        3. Encode prompt
        4. Run diffusion loop (yielding partials if requested)
        5. Decode to image

        When partial_images > 0, yields (GeneratedImage, partial_index, total_partials)
        tuples for intermediate images, then yields the final GeneratedImage.

        Args:
            settings: Generation config (steps, height, width)
            prompt: Text prompt
            seed: Random seed
            partial_images: Number of intermediate images to yield (0 for none)

        Yields:
            Partial images as (GeneratedImage, partial_index, total_partials) tuples
            Final GeneratedImage
        """
        latents = self.adapter.create_latents(seed, runtime_config)
        prompt_data = self.adapter.encode_prompt(prompt)

        # Calculate which steps to capture
        capture_steps = self._calculate_capture_steps(
            partial_images=partial_images,
            init_time_step=runtime_config.init_time_step,
            num_inference_steps=runtime_config.num_inference_steps,
        )

        # Run diffusion loop - may yield partial latents
        diffusion_gen = self._run_diffusion_loop(
            latents=latents,
            prompt_data=prompt_data,
            runtime_config=runtime_config,
            seed=seed,
            prompt=prompt,
            capture_steps=capture_steps,
        )

        # Process partial yields and get final latents
        partial_index = 0
        total_partials = len(capture_steps)

        if capture_steps:
            # Generator mode - iterate to get partials and final latents
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
                latents = e.value
        else:
            # No partials - just consume generator to get final latents
            try:
                while True:
                    next(diffusion_gen)
            except StopIteration as e:
                latents = e.value

        # Yield final image (only on last stage)
        if self.is_last_stage:
            yield self.adapter.decode_latents(latents, runtime_config, seed, prompt)

    def _run_diffusion_loop(
        self,
        latents: mx.array,
        prompt_data: PromptData,
        runtime_config: Config,
        seed: int,
        prompt: str,
        capture_steps: set[int] | None = None,
    ):
        """Execute the diffusion loop, optionally yielding at capture steps.

        When capture_steps is provided and non-empty, this becomes a generator
        that yields (latents, step_index) tuples at the specified timesteps.
        Only the last stage yields (others have incomplete latents).

        Args:
            latents: Initial noise latents
            prompt_data: Encoded prompt data
            runtime_config: RuntimeConfig with scheduler, steps, dimensions
            seed: Random seed (for callbacks)
            prompt: Text prompt (for callbacks)
            capture_steps: Set of timestep indices to capture (None = no captures)

        Yields:
            (latents, step_index) tuples at capture steps (last stage only)

        Returns:
            Final denoised latents ready for VAE decoding
        """
        if capture_steps is None:
            capture_steps = set()

        time_steps = tqdm(range(runtime_config.num_inference_steps))

        ctx = self.adapter.model.callbacks.start(
            seed=seed, prompt=prompt, config=runtime_config
        )

        ctx.before_loop(
            latents=latents,
        )

        for t in time_steps:
            try:
                latents = self._diffusion_step(
                    t=t,
                    config=runtime_config,
                    latents=latents,
                    prompt_data=prompt_data,
                )

                # Call subscribers in-loop
                ctx.in_loop(
                    t=t,
                    latents=latents,
                )

                mx.eval(latents)

                # Yield partial latents at capture steps (only on last stage)
                if t in capture_steps and self.is_last_stage:
                    yield (latents, t)

            except KeyboardInterrupt:  # noqa: PERF203
                ctx.interruption(t=t, latents=latents)
                raise StopImageGenerationException(
                    f"Stopping image generation at step {t + 1}/{len(time_steps)}"
                ) from None

        # Call subscribers after loop
        ctx.after_loop(latents=latents)

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

        This is the internal method called by adapters via compute_step_noise.
        Returns noise prediction without applying scheduler step.

        For edit mode, concatenates conditioning latents with generated latents
        before the transformer, and extracts only the generated portion after.

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
        scaled_latents = config.scheduler.scale_model_input(latents, t)

        # For edit mode: concatenate with conditioning latents
        original_latent_tokens = scaled_latents.shape[1]
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

        text_seq_len = prompt_embeds.shape[1]

        # Run through all joint blocks
        for block_idx, wrapper in enumerate(self.joint_block_wrappers):
            encoder_hidden_states, hidden_states = wrapper(
                hidden_states=hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                text_embeddings=text_embeddings,
                rotary_embeddings=rotary_embeddings,
                text_seq_len=text_seq_len,
                kv_cache=None,
                mode=BlockWrapperMode.CACHING,
                block_idx=block_idx,
                encoder_hidden_states_mask=encoder_hidden_states_mask,
            )

        # Merge streams
        if self.joint_block_wrappers:
            hidden_states = self.adapter.merge_streams(
                hidden_states, encoder_hidden_states
            )

        # Run through single blocks
        for wrapper in self.single_block_wrappers:
            hidden_states = wrapper(
                hidden_states=hidden_states,
                text_embeddings=text_embeddings,
                rotary_embeddings=rotary_embeddings,
                text_seq_len=text_seq_len,
                kv_cache=None,
                mode=BlockWrapperMode.CACHING,
            )

        # Extract image portion and project
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
    ) -> mx.array:
        """Execute a single diffusion step.

        Routes to single-node, sync pipeline, or async pipeline based on
        configuration and current timestep.
        """
        if self.group is None:
            return self._single_node_step(t, config, latents, prompt_data)
        elif t < config.init_time_step + self.num_sync_steps:
            return self._sync_pipeline(
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
            )

    def _single_node_step(
        self,
        t: int,
        config: Config,
        latents: mx.array,
        prompt_data: PromptData,
    ) -> mx.array:
        """Execute a single diffusion step on a single node (no distribution)."""
        conditioning_latents = prompt_data.conditioning_latents
        cond_image_grid = prompt_data.cond_image_grid

        if self.adapter.needs_cfg:
            # Two forward passes + guidance for CFG models (e.g., Qwen)
            noise_pos = self._forward_pass(
                latents,
                prompt_data.prompt_embeds,
                prompt_data.pooled_prompt_embeds,
                t=t,
                config=config,
                encoder_hidden_states_mask=prompt_data.get_encoder_hidden_states_mask(
                    positive=True
                ),
                cond_image_grid=cond_image_grid,
                conditioning_latents=conditioning_latents,
            )

            negative_prompt_embeds = prompt_data.negative_prompt_embeds
            negative_pooled_prompt_embeds = prompt_data.negative_pooled_prompt_embeds
            assert negative_prompt_embeds is not None, (
                "CFG requires negative_prompt_embeds"
            )
            assert negative_pooled_prompt_embeds is not None, (
                "CFG requires negative_pooled_prompt_embeds"
            )

            noise_neg = self._forward_pass(
                latents,
                negative_prompt_embeds,
                negative_pooled_prompt_embeds,
                t=t,
                config=config,
                encoder_hidden_states_mask=prompt_data.get_encoder_hidden_states_mask(
                    positive=False
                ),
                cond_image_grid=cond_image_grid,
                conditioning_latents=conditioning_latents,
            )

            assert self.config.guidance_scale is not None
            noise = self.adapter.apply_guidance(
                noise_pos, noise_neg, guidance_scale=self.config.guidance_scale
            )
        else:
            # Single forward pass for non-CFG models (e.g., Flux)
            noise = self._forward_pass(
                latents,
                prompt_data.prompt_embeds,
                prompt_data.pooled_prompt_embeds,
                t=t,
                config=config,
                encoder_hidden_states_mask=prompt_data.get_encoder_hidden_states_mask(),
                cond_image_grid=cond_image_grid,
                conditioning_latents=conditioning_latents,
            )

        return config.scheduler.step(noise=noise, timestep=t, latents=latents)

    def _initialize_kv_caches(
        self,
        batch_size: int,
        num_img_tokens: int,
        dtype: mx.Dtype,
    ) -> None:
        """Initialize KV caches for both sync and async pipelines.

        Note: Caches only store IMAGE K/V, not text K/V. Text K/V is always
        computed fresh and doesn't need caching (it's the same for all patches).
        """
        self.joint_kv_caches = [
            ImagePatchKVCache(
                batch_size=batch_size,
                num_heads=self.config.num_heads,
                image_seq_len=num_img_tokens,
                head_dim=self.config.head_dim,
                dtype=dtype,
            )
            for _ in range(len(self.joint_block_wrappers))
        ]
        self.single_kv_caches = [
            ImagePatchKVCache(
                batch_size=batch_size,
                num_heads=self.config.num_heads,
                image_seq_len=num_img_tokens,
                head_dim=self.config.head_dim,
                dtype=dtype,
            )
            for _ in range(len(self.single_block_wrappers))
        ]

    def _create_patches(
        self,
        latents: mx.array,
        config: Config,
    ) -> tuple[list[mx.array], list[tuple[int, int]]]:
        """Split latents into patches for async pipeline."""
        # Use 16 to match FluxLatentCreator.create_noise formula
        latent_height = config.height // 16
        latent_width = config.width // 16

        patch_heights, _ = calculate_patch_heights(latent_height, self.num_patches)
        token_indices = calculate_token_indices(patch_heights, latent_width)

        # Split latents into patches
        patch_latents = [latents[:, start:end, :] for start, end in token_indices]

        return patch_latents, token_indices

    def _sync_pipeline(
        self,
        t: int,
        config: Config,
        hidden_states: mx.array,
        prompt_data: PromptData,
        kontext_image_ids: mx.array | None = None,
    ) -> mx.array:
        prev_latents = hidden_states

        # Extract embeddings and model-specific data
        prompt_embeds = prompt_data.prompt_embeds
        pooled_prompt_embeds = prompt_data.pooled_prompt_embeds
        encoder_hidden_states_mask = prompt_data.get_encoder_hidden_states_mask()
        cond_image_grid = prompt_data.cond_image_grid

        hidden_states = config.scheduler.scale_model_input(hidden_states, t)

        # For edit mode: handle conditioning latents
        # All stages need to know the total token count for correct recv templates
        conditioning_latents = prompt_data.conditioning_latents
        original_latent_tokens = hidden_states.shape[1]
        if conditioning_latents is not None:
            num_img_tokens = original_latent_tokens + conditioning_latents.shape[1]
        else:
            num_img_tokens = original_latent_tokens

        # First stage: concatenate conditioning latents before embedding
        if self.is_first_stage and conditioning_latents is not None:
            hidden_states = mx.concatenate(
                [hidden_states, conditioning_latents], axis=1
            )

        # === PHASE 1: Embeddings ===
        if self.is_first_stage:
            hidden_states, encoder_hidden_states = self.adapter.compute_embeddings(
                hidden_states, prompt_embeds
            )

        # All stages need these for their blocks
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

        # === Initialize KV caches to populate during sync for async warmstart ===
        batch_size = prev_latents.shape[0]
        text_seq_len = prompt_embeds.shape[1]
        hidden_dim = self.adapter.hidden_dim

        if t == config.init_time_step:
            self._initialize_kv_caches(
                batch_size=batch_size,
                num_img_tokens=num_img_tokens,
                dtype=prev_latents.dtype,
            )

        # === PHASE 2: Joint Blocks with Communication and Caching ===
        if self.has_joint_blocks:
            # Receive from previous stage (if not first stage)
            if not self.is_first_stage:
                recv_template = mx.zeros(
                    (batch_size, num_img_tokens, hidden_dim), dtype=prev_latents.dtype
                )
                hidden_states = mx.distributed.recv_like(
                    recv_template, self.prev_rank, group=self.group
                )
                enc_template = mx.zeros(
                    (batch_size, text_seq_len, hidden_dim), dtype=prev_latents.dtype
                )
                encoder_hidden_states = mx.distributed.recv_like(
                    enc_template, self.prev_rank, group=self.group
                )
                mx.eval(hidden_states, encoder_hidden_states)

            # Run assigned joint blocks with caching mode
            for block_idx, wrapper in enumerate(self.joint_block_wrappers):
                encoder_hidden_states, hidden_states = wrapper(
                    hidden_states=hidden_states,
                    encoder_hidden_states=encoder_hidden_states,
                    text_embeddings=text_embeddings,
                    rotary_embeddings=image_rotary_embeddings,
                    text_seq_len=text_seq_len,
                    kv_cache=self.joint_kv_caches[block_idx],
                    mode=BlockWrapperMode.CACHING,
                    encoder_hidden_states_mask=encoder_hidden_states_mask,
                )

        # === PHASE 3: Joint→Single Transition ===
        if self.owns_concat_stage:
            # Merge encoder and hidden states using adapter hook
            concatenated = self.adapter.merge_streams(
                hidden_states, encoder_hidden_states
            )

            if self.has_single_blocks or self.is_last_stage:
                # Keep locally: either for single blocks or final projection
                hidden_states = concatenated
            else:
                # Send concatenated state to next stage (which has single blocks)
                mx.eval(
                    mx.distributed.send(concatenated, self.next_rank, group=self.group)
                )

        elif self.has_joint_blocks and not self.is_last_stage:
            # Send joint block outputs to next stage (which has more joint blocks)
            mx.eval(
                mx.distributed.send(hidden_states, self.next_rank, group=self.group),
                mx.distributed.send(
                    encoder_hidden_states, self.next_rank, group=self.group
                ),
            )

        # === PHASE 4: Single Blocks with Communication and Caching ===
        if self.has_single_blocks:
            # Receive from previous stage if we didn't do concatenation
            if not self.owns_concat_stage and not self.is_first_stage:
                recv_template = mx.zeros(
                    (batch_size, text_seq_len + num_img_tokens, hidden_dim),
                    dtype=prev_latents.dtype,
                )
                hidden_states = mx.distributed.recv_like(
                    recv_template, self.prev_rank, group=self.group
                )
                mx.eval(hidden_states)

            # Run assigned single blocks with caching mode
            for block_idx, wrapper in enumerate(self.single_block_wrappers):
                hidden_states = wrapper(
                    hidden_states=hidden_states,
                    text_embeddings=text_embeddings,
                    rotary_embeddings=image_rotary_embeddings,
                    text_seq_len=text_seq_len,
                    kv_cache=self.single_kv_caches[block_idx],
                    mode=BlockWrapperMode.CACHING,
                )

            # Send to next stage if not last
            if not self.is_last_stage:
                mx.eval(
                    mx.distributed.send(hidden_states, self.next_rank, group=self.group)
                )

        # === PHASE 5: Last Stage - Final Projection + Scheduler ===
        # Extract image portion (remove text embeddings prefix)
        hidden_states = hidden_states[:, text_seq_len:, ...]

        # For edit mode: extract only the generated portion (exclude conditioning latents)
        if conditioning_latents is not None:
            hidden_states = hidden_states[:, :original_latent_tokens, ...]

        if self.is_last_stage:
            hidden_states = self.adapter.final_projection(
                hidden_states, text_embeddings
            )

            hidden_states = config.scheduler.step(
                noise=hidden_states,
                timestep=t,
                latents=prev_latents,
            )

            if not self.is_first_stage:
                mx.eval(mx.distributed.send(hidden_states, 0, group=self.group))

        elif self.is_first_stage:
            hidden_states = mx.distributed.recv_like(
                prev_latents, src=self.world_size - 1, group=self.group
            )

            mx.eval(hidden_states)

        else:
            # For shape correctness
            hidden_states = prev_latents

        return hidden_states

    def _async_pipeline_step(
        self,
        t: int,
        config: Config,
        latents: mx.array,
        prompt_data: PromptData,
        kontext_image_ids: mx.array | None = None,
    ) -> mx.array:
        patch_latents, token_indices = self._create_patches(latents, config)

        patch_latents = self._async_pipeline(
            t,
            config,
            patch_latents,
            token_indices,
            prompt_data,
            kontext_image_ids,
        )

        return mx.concatenate(patch_latents, axis=1)

    def _async_pipeline(
        self,
        t: int,
        config: Config,
        patch_latents: list[mx.array],
        token_indices: list[tuple[int, int]],
        prompt_data: PromptData,
        kontext_image_ids: mx.array | None = None,
    ) -> list[mx.array]:
        """Execute async pipeline for all patches."""
        assert self.joint_kv_caches is not None
        assert self.single_kv_caches is not None

        # Extract embeddings and model-specific data
        prompt_embeds = prompt_data.prompt_embeds
        pooled_prompt_embeds = prompt_data.pooled_prompt_embeds
        encoder_hidden_states_mask = prompt_data.get_encoder_hidden_states_mask()
        cond_image_grid = prompt_data.cond_image_grid

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

        batch_size = patch_latents[0].shape[0]
        text_seq_len = prompt_embeds.shape[1]
        hidden_dim = self.adapter.hidden_dim

        for patch_idx, patch in enumerate(patch_latents):
            patch_prev = patch

            start_token, end_token = token_indices[patch_idx]

            if self.has_joint_blocks:
                if (
                    not self.is_first_stage
                    or t != config.init_time_step + self.num_sync_steps
                ):
                    if self.is_first_stage:
                        # First stage receives latent-space from last stage (scheduler output)
                        recv_template = patch
                    else:
                        # Other stages receive hidden-space from previous stage
                        patch_len = patch.shape[1]
                        recv_template = mx.zeros(
                            (batch_size, patch_len, hidden_dim),
                            dtype=patch.dtype,
                        )
                    patch = mx.distributed.recv_like(
                        recv_template, src=self.prev_rank, group=self.group
                    )
                    mx.eval(patch)
                    patch_latents[patch_idx] = patch

                    if not self.is_first_stage and patch_idx == 0:
                        enc_template = mx.zeros(
                            (batch_size, text_seq_len, hidden_dim),
                            dtype=patch_latents[0].dtype,
                        )
                        encoder_hidden_states = mx.distributed.recv_like(
                            enc_template, src=self.prev_rank, group=self.group
                        )
                        mx.eval(encoder_hidden_states)

                if self.is_first_stage:
                    patch, encoder_hidden_states = self.adapter.compute_embeddings(
                        patch, prompt_embeds
                    )

                # Run assigned joint blocks with patched mode
                for block_idx, wrapper in enumerate(self.joint_block_wrappers):
                    encoder_hidden_states, patch = wrapper(
                        hidden_states=patch,
                        encoder_hidden_states=encoder_hidden_states,
                        text_embeddings=text_embeddings,
                        rotary_embeddings=image_rotary_embeddings,
                        text_seq_len=text_seq_len,
                        kv_cache=self.joint_kv_caches[block_idx],
                        mode=BlockWrapperMode.PATCHED,
                        patch_start=start_token,
                        patch_end=end_token,
                        encoder_hidden_states_mask=encoder_hidden_states_mask,
                    )

            if self.owns_concat_stage:
                patch_concat = self.adapter.merge_streams(patch, encoder_hidden_states)

                if self.has_single_blocks or self.is_last_stage:
                    # Keep locally: either for single blocks or final projection
                    patch = patch_concat
                else:
                    mx.eval(
                        mx.distributed.send(
                            patch_concat, self.next_rank, group=self.group
                        )
                    )

            elif self.has_joint_blocks and not self.is_last_stage:
                mx.eval(mx.distributed.send(patch, self.next_rank, group=self.group))

                if patch_idx == 0:
                    mx.eval(
                        mx.distributed.send(
                            encoder_hidden_states, self.next_rank, group=self.group
                        )
                    )

            if self.has_single_blocks:
                if not self.owns_concat_stage and not self.is_first_stage:
                    recv_template = mx.zeros(
                        [
                            batch_size,
                            text_seq_len + patch_latents[patch_idx].shape[1],
                            hidden_dim,
                        ],
                        dtype=patch_latents[0].dtype,
                    )

                    patch = mx.distributed.recv_like(
                        recv_template, src=self.prev_rank, group=self.group
                    )
                    mx.eval(patch)
                    patch_latents[patch_idx] = patch

                # Run assigned single blocks with patched mode
                for block_idx, wrapper in enumerate(self.single_block_wrappers):
                    patch = wrapper(
                        hidden_states=patch,
                        text_embeddings=text_embeddings,
                        rotary_embeddings=image_rotary_embeddings,
                        text_seq_len=text_seq_len,
                        kv_cache=self.single_kv_caches[block_idx],
                        mode=BlockWrapperMode.PATCHED,
                        patch_start=start_token,
                        patch_end=end_token,
                    )

                if not self.is_last_stage:
                    mx.eval(
                        mx.distributed.send(patch, self.next_rank, group=self.group)
                    )

            if self.is_last_stage:
                patch_img_only = patch[:, text_seq_len:, :]

                patch_img_only = self.adapter.final_projection(
                    patch_img_only, text_embeddings
                )

                patch = config.scheduler.step(
                    noise=patch_img_only,
                    timestep=t,
                    latents=patch_prev,
                )

                if not self.is_first_stage and t != config.num_inference_steps - 1:
                    mx.eval(
                        mx.distributed.send(patch, self.next_rank, group=self.group)
                    )

                patch_latents[patch_idx] = patch

        return patch_latents
