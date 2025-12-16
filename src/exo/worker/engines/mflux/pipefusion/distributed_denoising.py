from math import ceil
from typing import Any, Optional

import mlx.core as mx
from mflux.config.runtime_config import RuntimeConfig
from mflux.models.flux.model.flux_transformer.transformer import Transformer

from exo.shared.types.worker.shards import PipelineShardMetadata
from exo.worker.engines.mflux.pipefusion.kv_cache import JointPatchKVCache, PatchKVCache
from exo.worker.engines.mflux.pipefusion.patched_blocks import (
    CachingJointTransformerBlock,
    CachingSingleTransformerBlock,
    PatchedJointTransformerBlock,
    PatchedSingleTransformerBlock,
)
from exo.worker.runner.bootstrap import logger


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


class DistributedDenoising:
    def __init__(
        self,
        transformer: Transformer,
        group: mx.distributed.Group,
        shard_metadata: PipelineShardMetadata,
        num_sync_steps: int = 1,
        num_patches: Optional[int] = None,
    ):
        self.transformer = transformer
        self.group = group
        self.rank = shard_metadata.device_rank
        self.world_size = shard_metadata.world_size
        self.next_rank = (self.rank + 1) % self.world_size
        self.prev_rank = (self.rank - 1 + self.world_size) % self.world_size
        self.start_layer = shard_metadata.start_layer
        self.end_layer = shard_metadata.end_layer

        self.num_sync_steps = num_sync_steps
        self.num_patches = num_patches if num_patches else group.size()

        # Persistent KV caches (initialized on first async timestep, reused across timesteps)
        self.joint_kv_caches: list[JointPatchKVCache] | None = None
        self.single_kv_caches: list[PatchKVCache] | None = None

        # Get block counts from the original transformer (before slicing)
        # Note: These are the ORIGINAL counts, not the sliced counts
        self.total_joint = 19  # Flux has 19 joint blocks
        self.total_single = 38  # Flux has 38 single blocks
        self.total_layers = self.total_joint + self.total_single

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

        self.transformer_blocks = self.transformer_blocks[
            self.joint_start : self.joint_end
        ]
        self.single_transformer_blocks = self.single_transformer_blocks[
            self.single_start : self.single_end
        ]

    @property
    def is_first_stage(self) -> bool:
        return self.rank == 0

    @property
    def is_last_stage(self) -> bool:
        return self.rank == self.world_size - 1

    def _initialize_kv_caches(
        self,
        batch_size: int,
        num_img_tokens: int,
        dtype: mx.Dtype,
    ) -> None:
        """Initialize KV caches for both sync and async pipelines.

        Note: Caches only store IMAGE K/V, not text K/V. Text K/V is always
        computed fresh and doesn't need caching (it's the same for all patches).

        Args:
            batch_size: Batch size
            num_img_tokens: Number of image tokens
            dtype: Data type for cache tensors
        """
        self.joint_kv_caches = [
            JointPatchKVCache(
                batch_size=batch_size,
                num_heads=24,
                image_seq_len=num_img_tokens,
                head_dim=128,
                dtype=dtype,
            )
            for _ in range(len(self.transformer_blocks))
        ]
        self.single_kv_caches = [
            PatchKVCache(
                batch_size=batch_size,
                num_heads=24,
                image_seq_len=num_img_tokens,
                head_dim=128,
                dtype=dtype,
            )
            for _ in range(len(self.single_transformer_blocks))
        ]

    def _create_patches(
        self,
        latents: mx.array,
        config: RuntimeConfig,
    ) -> tuple[list[mx.array], list[tuple[int, int]]]:
        # Calculate patch metadata
        # TODO(ciaran): generalise
        latent_height = config.height // 8
        latent_width = config.width // 8
        patch_size = 2  # Flux uses 2x2 patches

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
        # First stage: compute embeddings
        # Non-first stages: will receive embedded values
        if self.is_first_stage:
            hidden_states = self.transformer.x_embedder(hidden_states)
            encoder_hidden_states = self.transformer.context_embedder(prompt_embeds)

        # All stages need these for their blocks
        text_embeddings = Transformer.compute_text_embeddings(
            t, pooled_prompt_embeds, self.transformer.time_text_embed, config
        )
        image_rotary_embeddings = Transformer.compute_rotary_embeddings(
            prompt_embeds, self.transformer.pos_embed, config, kontext_image_ids
        )

        # === Initialize KV caches to populate during sync for async warmstart ===
        batch_size = prev_latents.shape[0]
        num_img_tokens = prev_latents.shape[1]
        text_seq_len = prompt_embeds.shape[1]
        hidden_dim = self.transformer.x_embedder.weight.shape[0]

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

            # Run assigned joint blocks with caching wrappers
            for block_idx, block in enumerate(self.transformer_blocks):
                caching_block = CachingJointTransformerBlock(
                    block, self.joint_kv_caches[block_idx]
                )
                encoder_hidden_states, hidden_states = caching_block(
                    hidden_states=hidden_states,
                    encoder_hidden_states=encoder_hidden_states,
                    text_embeddings=text_embeddings,
                    rotary_embeddings=image_rotary_embeddings,
                )

        # === PHASE 3: Joint→Single Transition ===
        if self.owns_concat_stage:
            # Concatenate encoder and hidden states
            concatenated = mx.concatenate(
                [encoder_hidden_states, hidden_states], axis=1
            )

            if self.has_single_blocks:
                # We continue with single blocks on this stage
                hidden_states = concatenated
            else:
                # Send concatenated state to next stage (which has single blocks)
                mx.distributed.send(concatenated, self.next_rank, group=self.group)

        elif self.has_joint_blocks and not self.is_last_stage:
            # Send joint block outputs to next stage (which has more joint blocks)
            mx.distributed.send(hidden_states, self.next_rank, group=self.group)
            mx.distributed.send(encoder_hidden_states, self.next_rank, group=self.group)

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

            # Run assigned single blocks with caching wrappers
            for block_idx, block in enumerate(self.single_transformer_blocks):
                caching_block = CachingSingleTransformerBlock(
                    block, self.single_kv_caches[block_idx]
                )
                hidden_states = caching_block(
                    hidden_states=hidden_states,
                    text_embeddings=text_embeddings,
                    rotary_embeddings=image_rotary_embeddings,
                    text_seq_len=text_seq_len,
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
            hidden_states = self.transformer.norm_out(hidden_states, text_embeddings)
            hidden_states = self.transformer.proj_out(hidden_states)

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
        # TODO(ciaran): needed in general?
        # hidden_states = config.scheduler.scale_model_input(hidden_states, t)

        text_embeddings = Transformer.compute_text_embeddings(
            t, pooled_prompt_embeds, self.transformer.time_text_embed, config
        )
        image_rotary_embeddings = Transformer.compute_rotary_embeddings(
            prompt_embeds, self.transformer.pos_embed, config, kontext_image_ids
        )

        batch_size = patch_latents[0].shape[0]
        text_seq_len = prompt_embeds.shape[1]
        hidden_dim = self.transformer.x_embedder.weight.shape[0]

        for patch_idx, patch in enumerate(patch_latents):
            patch_prev = patch

            start_token, end_token = token_indices[patch_idx]

            if self.has_joint_blocks:
                if not self.is_first_stage or t != self.num_sync_steps:
                    # rank 0 already has first iteration inputs
                    # TODO(ciaran): correct shapes
                    patch = mx.distributed.recv_like(
                        patch, src=self.prev_rank, group=self.group
                    )
                    mx.eval(patch)
                    logger.info(
                        "==============================================================\n\n"
                        + f"rank {self.rank}, t = {t}, patch_idx = {patch_idx}, received patch: {patch.shape}"
                        + "\n\n=============================================================="
                    )

                    if not self.is_first_stage and patch_idx == 0:
                        enc_template = mx.zeros(
                            (batch_size, text_seq_len, hidden_dim),
                            dtype=patch_latents[0].dtype,
                        )
                        encoder_hidden_states = mx.distributed.recv_like(
                            enc_template, src=self.prev_rank, group=self.group
                        )
                        mx.eval(encoder_hidden_states)
                        logger.info(
                            "==============================================================\n\n"
                            + f"rank {self.rank}, t = {t}, patch_idx = {patch_idx}, received encoder_hidden_states: {encoder_hidden_states.shape}"
                            + "\n\n=============================================================="
                        )

                if self.is_first_stage:
                    patch = self.transformer.x_embedder(patch)
                    encoder_hidden_states = self.transformer.context_embedder(
                        prompt_embeds
                    )

                # Run assigned joint blocks with KV cache
                for block_idx, block in enumerate(self.transformer_blocks):
                    patched_block = PatchedJointTransformerBlock(block)
                    encoder_hidden_states, patch = patched_block(
                        patch_hidden=patch,
                        encoder_hidden_states=encoder_hidden_states,
                        text_embeddings=text_embeddings,
                        image_rotary_emb=image_rotary_embeddings,
                        kv_cache=self.joint_kv_caches[block_idx],
                        patch_start=start_token,
                        patch_end=end_token,
                        text_seq_len=text_seq_len,
                    )

            if self.owns_concat_stage:
                patch_concat = mx.concatenate([encoder_hidden_states, patch], axis=1)

                if self.has_single_blocks:
                    patch = patch_concat
                else:
                    mx.eval(
                        mx.distributed.send(
                            patch_concat, self.next_rank, group=self.group
                        )
                    )
                    logger.info(
                        "==============================================================\n\n"
                        + f"rank {self.rank}, t = {t}, patch_idx = {patch_idx}, sent patch_concat: {patch_concat.shape}"
                        + "\n\n=============================================================="
                    )

            elif self.has_joint_blocks and not self.is_last_stage:
                mx.eval(mx.distributed.send(patch, self.next_rank, group=self.group))
                logger.info(
                    "==============================================================\n\n"
                    + f"rank {self.rank}, t = {t}, patch_idx = {patch_idx}, sent patch: {patch.shape}"
                    + "\n\n=============================================================="
                )

                if patch_idx == 0:
                    mx.eval(
                        mx.distributed.send(
                            encoder_hidden_states, self.next_rank, group=self.group
                        )
                    )
                    logger.info(
                        "==============================================================\n\n"
                        + f"rank {self.rank}, t = {t}, patch_idx = {patch_idx}, sent encoder_hidden_states: {encoder_hidden_states.shape}"
                        + "\n\n=============================================================="
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
                    logger.info(
                        "==============================================================\n\n"
                        + f"rank {self.rank}, t = {t}, patch_idx = {patch_idx}, received patch: {patch.shape}"
                        + "\n\n=============================================================="
                    )

                # Run assigned single blocks with KV cache
                for block_idx, block in enumerate(self.single_transformer_blocks):
                    patched_block = PatchedSingleTransformerBlock(block)
                    patch = patched_block(
                        patch_hidden=patch,
                        text_embeddings=text_embeddings,
                        image_rotary_emb=image_rotary_embeddings,
                        kv_cache=self.single_kv_caches[block_idx],
                        patch_start=start_token,
                        patch_end=end_token,
                        text_seq_len=text_seq_len,
                    )

                if not self.is_last_stage:
                    mx.eval(
                        mx.distributed.send(patch, self.next_rank, group=self.group)
                    )
                    logger.info(
                        "==============================================================\n\n"
                        + f"rank {self.rank}, t = {t}, patch_idx = {patch_idx}, sent patch: {patch.shape}"
                        + "\n\n=============================================================="
                    )

            if self.is_last_stage:
                patch_img_only = patch[:, text_seq_len:, :]

                patch_img_only = self.transformer.norm_out(
                    patch_img_only, text_embeddings
                )
                patch_img_only = self.transformer.proj_out(patch_img_only)

                patch = config.scheduler.step(
                    model_output=patch_img_only,
                    timestep=t,
                    sample=patch_prev,
                )

                if not self.is_first_stage:
                    mx.eval(
                        mx.distributed.send(patch, self.next_rank, group=self.group)
                    )
                    logger.info(
                        "==============================================================\n\n"
                        + f"rank {self.rank}, t = {t}, patch_idx = {patch_idx}, sent patch: {patch.shape}"
                        + "\n\n=============================================================="
                    )

                patch_latents[patch_idx] = patch

        return patch_latents

    def __call__(
        self,
        t: int,
        config: RuntimeConfig,
        hidden_states: mx.array,
        prompt_embeds: mx.array,
        pooled_prompt_embeds: mx.array,
        controlnet_block_samples: list[mx.array] | None = None,
        controlnet_single_block_samples: list[mx.array] | None = None,
        kontext_image_ids: mx.array | None = None,
    ) -> mx.array:
        if t < self.num_sync_steps:
            latents = self._sync_pipeline(
                t,
                config,
                hidden_states,
                prompt_embeds,
                pooled_prompt_embeds,
                kontext_image_ids,
            )
        else:
            patch_latents, token_indices = self._create_patches(hidden_states, config)

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

            latents = mx.concatenate(patch_latents, axis=1)

        return latents

    # Delegate attribute access to the underlying transformer for compatibility
    def __getattr__(self, name: str) -> Any:
        return getattr(self.transformer, name)
