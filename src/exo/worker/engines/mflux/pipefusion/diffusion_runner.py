from math import ceil
from typing import Optional

import mlx.core as mx
from mflux.callbacks.callbacks import Callbacks
from mflux.config.runtime_config import RuntimeConfig
from mflux.utils.exceptions import StopImageGenerationException
from tqdm import tqdm

from exo.shared.types.worker.shards import PipelineShardMetadata
from exo.worker.engines.mflux.config.model_config import ImageModelConfig
from exo.worker.engines.mflux.pipefusion.adapter import BlockWrapperMode, ModelAdapter
from exo.worker.engines.mflux.pipefusion.block_wrapper import (
    JointBlockWrapper,
    SingleBlockWrapper,
)
from exo.worker.engines.mflux.pipefusion.kv_cache import ImagePatchKVCache


def calculate_patch_heights(latent_height: int, num_patches: int, patch_size: int):
    patch_height = ceil(latent_height / num_patches)
    patch_height = ceil(patch_height / patch_size) * patch_size

    actual_num_patches = ceil(latent_height / patch_height)
    patch_heights = [patch_height] * (actual_num_patches - 1)

    last_height = latent_height - patch_height * (actual_num_patches - 1)
    patch_heights.append(last_height)

    return patch_heights, actual_num_patches


def calculate_token_indices(
    patch_heights: list[int], latent_width: int, patch_size: int
):
    tokens_per_row = latent_width // patch_size

    token_ranges = []
    cumulative_height = 0

    for h in patch_heights:
        start_row = cumulative_height // patch_size
        end_row = (cumulative_height + h) // patch_size

        start_token = tokens_per_row * start_row
        end_token = tokens_per_row * end_row

        token_ranges.append((start_token, end_token))
        cumulative_height += h

    return token_ranges  # List of (start, end) token indices


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

        # Slice blocks to only those assigned to this stage
        all_joint_blocks = self.adapter.get_joint_blocks()
        all_single_blocks = self.adapter.get_single_blocks()

        assigned_joint_blocks = all_joint_blocks[self.joint_start : self.joint_end]
        assigned_single_blocks = all_single_blocks[self.single_start : self.single_end]

        # Wrap blocks at initialization (reused across all calls)
        self.joint_block_wrappers = [
            JointBlockWrapper(block=block, adapter=self.adapter)
            for block in assigned_joint_blocks
        ]
        self.single_block_wrappers = [
            SingleBlockWrapper(block=block, adapter=self.adapter)
            for block in assigned_single_blocks
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

    def run(
        self,
        latents: mx.array,
        prompt_embeds: mx.array,
        pooled_prompt_embeds: mx.array,
        runtime_config: RuntimeConfig,
        seed: int,
        prompt: str,
    ) -> mx.array:
        """Execute the full diffusion loop.

        Args:
            latents: Initial noise latents
            prompt_embeds: T5 text embeddings
            pooled_prompt_embeds: CLIP pooled embeddings
            runtime_config: RuntimeConfig with scheduler, steps, dimensions
            seed: Random seed (for callbacks)
            prompt: Text prompt (for callbacks)

        Returns:
            Final denoised latents ready for VAE decoding
        """
        time_steps = tqdm(
            range(runtime_config.init_time_step, runtime_config.num_inference_steps)
        )

        # Call subscribers for beginning of loop
        Callbacks.before_loop(
            seed=seed,
            prompt=prompt,
            latents=latents,
            config=runtime_config,
        )

        for t in time_steps:
            try:
                latents = self._diffusion_step(
                    t=t,
                    config=runtime_config,
                    latents=latents,
                    prompt_embeds=prompt_embeds,
                    pooled_prompt_embeds=pooled_prompt_embeds,
                )

                # Call subscribers in-loop
                Callbacks.in_loop(
                    t=t,
                    seed=seed,
                    prompt=prompt,
                    latents=latents,
                    config=runtime_config,
                    time_steps=time_steps,
                )

                mx.eval(latents)

            except KeyboardInterrupt:  # noqa: PERF203
                Callbacks.interruption(
                    t=t,
                    seed=seed,
                    prompt=prompt,
                    latents=latents,
                    config=runtime_config,
                    time_steps=time_steps,
                )
                raise StopImageGenerationException(
                    f"Stopping image generation at step {t + 1}/{len(time_steps)}"
                ) from None

        # Call subscribers after loop
        Callbacks.after_loop(
            seed=seed,
            prompt=prompt,
            latents=latents,
            config=runtime_config,
        )

        return latents

    def _diffusion_step(
        self,
        t: int,
        config: RuntimeConfig,
        latents: mx.array,
        prompt_embeds: mx.array,
        pooled_prompt_embeds: mx.array,
    ) -> mx.array:
        """Execute a single diffusion step.

        Routes to single-node, sync pipeline, or async pipeline based on
        configuration and current timestep.
        """
        if self.group is None:
            return self._single_node_step(
                t, config, latents, prompt_embeds, pooled_prompt_embeds
            )
        elif t < self.num_sync_steps:
            return self._sync_pipeline(
                t, config, latents, prompt_embeds, pooled_prompt_embeds
            )
        else:
            return self._async_pipeline_step(
                t, config, latents, prompt_embeds, pooled_prompt_embeds
            )

    def _single_node_step(
        self,
        t: int,
        config: RuntimeConfig,
        latents: mx.array,
        prompt_embeds: mx.array,
        pooled_prompt_embeds: mx.array,
    ) -> mx.array:
        """Execute a single diffusion step on a single node (no distribution)."""
        latents = config.scheduler.scale_model_input(latents, t)

        # Compute embeddings
        hidden_states, encoder_hidden_states = self.adapter.compute_embeddings(
            latents, prompt_embeds
        )
        text_embeddings = self.adapter.compute_text_embeddings(
            t, pooled_prompt_embeds, config
        )
        rotary_embeddings = self.adapter.compute_rotary_embeddings(
            prompt_embeds, config
        )

        text_seq_len = prompt_embeds.shape[1]

        # Run through all joint blocks
        for wrapper in self.joint_block_wrappers:
            encoder_hidden_states, hidden_states = wrapper(
                hidden_states=hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                text_embeddings=text_embeddings,
                rotary_embeddings=rotary_embeddings,
                text_seq_len=text_seq_len,
                kv_cache=None,
                mode=BlockWrapperMode.CACHING,
            )

        # Merge streams for single blocks
        hidden_states = self.adapter.merge_streams(hidden_states, encoder_hidden_states)

        # Run through all single blocks
        for wrapper in self.single_block_wrappers:
            hidden_states = wrapper(
                hidden_states=hidden_states,
                text_embeddings=text_embeddings,
                rotary_embeddings=rotary_embeddings,
                text_seq_len=text_seq_len,
                kv_cache=None,
                mode=BlockWrapperMode.CACHING,
            )

        # Extract image portion (remove text embeddings prefix)
        hidden_states = hidden_states[:, text_seq_len:, ...]

        # Final projection
        noise = self.adapter.final_projection(hidden_states, text_embeddings)

        # Scheduler step
        return config.scheduler.step(model_output=noise, timestep=t, sample=latents)

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
        config: RuntimeConfig,
    ) -> tuple[list[mx.array], list[tuple[int, int]]]:
        """Split latents into patches for async pipeline."""
        latent_height = config.height // self.config.vae_scale_factor
        latent_width = config.width // self.config.vae_scale_factor
        patch_size = self.config.patch_size

        patch_heights, _ = calculate_patch_heights(
            latent_height, self.num_patches, patch_size
        )
        token_indices = calculate_token_indices(patch_heights, latent_width, patch_size)

        # Split latents into patches
        patch_latents = [latents[:, start:end, :] for start, end in token_indices]

        return patch_latents, token_indices

    def _sync_pipeline(
        self,
        t: int,
        config: RuntimeConfig,
        hidden_states: mx.array,
        prompt_embeds: mx.array,
        pooled_prompt_embeds: mx.array,
        kontext_image_ids: mx.array | None = None,
    ) -> mx.array:
        prev_latents = hidden_states

        hidden_states = config.scheduler.scale_model_input(hidden_states, t)

        # === PHASE 1: Embeddings ===
        if self.is_first_stage:
            hidden_states, encoder_hidden_states = self.adapter.compute_embeddings(
                hidden_states, prompt_embeds
            )

        # All stages need these for their blocks
        text_embeddings = self.adapter.compute_text_embeddings(
            t, pooled_prompt_embeds, config
        )
        image_rotary_embeddings = self.adapter.compute_rotary_embeddings(
            prompt_embeds,
            config,
            kontext_image_ids=kontext_image_ids,
        )

        # === Initialize KV caches to populate during sync for async warmstart ===
        batch_size = prev_latents.shape[0]
        num_img_tokens = prev_latents.shape[1]
        text_seq_len = prompt_embeds.shape[1]
        hidden_dim = self.adapter.hidden_dim

        if t == 0:
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
                )

        # === PHASE 3: Joint→Single Transition ===
        if self.owns_concat_stage:
            # Merge encoder and hidden states using adapter hook
            concatenated = self.adapter.merge_streams(
                hidden_states, encoder_hidden_states
            )

            if self.has_single_blocks:
                # We continue with single blocks on this stage
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

        if self.is_last_stage:
            hidden_states = self.adapter.final_projection(
                hidden_states, text_embeddings
            )

            hidden_states = config.scheduler.step(
                model_output=hidden_states,
                timestep=t,
                sample=prev_latents,
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
        config: RuntimeConfig,
        latents: mx.array,
        prompt_embeds: mx.array,
        pooled_prompt_embeds: mx.array,
        kontext_image_ids: mx.array | None = None,
    ) -> mx.array:
        patch_latents, token_indices = self._create_patches(latents, config)

        patch_latents = self._async_pipeline(
            t,
            config,
            patch_latents,
            token_indices,
            prompt_embeds,
            pooled_prompt_embeds,
            kontext_image_ids,
        )

        # Receive final patches from last rank
        if (
            t == config.num_inference_steps - 1
            and self.is_first_stage
            and not self.is_last_stage
        ):
            for patch_idx in range(len(patch_latents)):
                patch_latents[patch_idx] = mx.distributed.recv_like(
                    patch_latents[patch_idx], src=self.prev_rank, group=self.group
                )
                mx.eval(patch_latents[patch_idx])

        return mx.concatenate(patch_latents, axis=1)

    def _async_pipeline(
        self,
        t: int,
        config: RuntimeConfig,
        patch_latents: list[mx.array],
        token_indices: list[tuple[int, int]],
        prompt_embeds: mx.array,
        pooled_prompt_embeds: mx.array,
        kontext_image_ids: mx.array | None = None,
    ) -> list[mx.array]:
        """Execute async pipeline for all patches."""
        assert self.joint_kv_caches is not None
        assert self.single_kv_caches is not None

        text_embeddings = self.adapter.compute_text_embeddings(
            t, pooled_prompt_embeds, config
        )
        image_rotary_embeddings = self.adapter.compute_rotary_embeddings(
            prompt_embeds,
            config,
            kontext_image_ids=kontext_image_ids,
        )

        batch_size = patch_latents[0].shape[0]
        text_seq_len = prompt_embeds.shape[1]
        hidden_dim = self.adapter.hidden_dim

        for patch_idx, patch in enumerate(patch_latents):
            patch_prev = patch

            start_token, end_token = token_indices[patch_idx]

            if self.has_joint_blocks:
                if not self.is_first_stage or t != self.num_sync_steps:
                    patch = mx.distributed.recv_like(
                        patch, src=self.prev_rank, group=self.group
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
                    )

            if self.owns_concat_stage:
                patch_concat = self.adapter.merge_streams(patch, encoder_hidden_states)

                if self.has_single_blocks:
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
                    model_output=patch_img_only,
                    timestep=t,
                    sample=patch_prev,
                )

                if not self.is_first_stage:
                    mx.eval(
                        mx.distributed.send(patch, self.next_rank, group=self.group)
                    )

                patch_latents[patch_idx] = patch

        return patch_latents
