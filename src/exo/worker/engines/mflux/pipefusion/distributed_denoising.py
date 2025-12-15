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
from exo.worker.engines.mlx.utils_mlx import mx_barrier


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
        self.start_layer = shard_metadata.start_layer
        self.end_layer = shard_metadata.end_layer

        self.num_sync_steps = num_sync_steps
        self.num_patches = num_patches if num_patches else group.size()

        # Persistent KV caches (initialized on first async timestep, reused across timesteps)
        self._joint_kv_caches: list[JointPatchKVCache] | None = None
        self._single_kv_caches: list[PatchKVCache] | None = None

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
        self._joint_kv_caches = [
            JointPatchKVCache(
                batch_size=batch_size,
                num_heads=24,
                image_seq_len=num_img_tokens,
                head_dim=128,
                dtype=dtype,
            )
            for _ in range(len(self.transformer_blocks))
        ]
        self._single_kv_caches = [
            PatchKVCache(
                batch_size=batch_size,
                num_heads=24,
                image_seq_len=num_img_tokens,
                head_dim=128,
                dtype=dtype,
            )
            for _ in range(len(self.single_transformer_blocks))
        ]

    def _sync_pipeline(
        self,
        t: int,
        config: RuntimeConfig,
        hidden_states: mx.array,
        prompt_embeds: mx.array,
        pooled_prompt_embeds: mx.array,
        kontext_image_ids: mx.array | None = None,
    ):
        prev_latents = hidden_states

        hidden_states = config.scheduler.scale_model_input(hidden_states, t)

        # === PHASE 1: Create Embeddings (all stages compute, for consistency) ===
        hidden_states = self.transformer.x_embedder(hidden_states)
        encoder_hidden_states = self.transformer.context_embedder(prompt_embeds)
        text_embeddings = Transformer.compute_text_embeddings(
            t, pooled_prompt_embeds, self.transformer.time_text_embed, config
        )
        image_rotary_embeddings = Transformer.compute_rotary_embeddings(
            prompt_embeds, self.transformer.pos_embed, config, kontext_image_ids
        )

        # === Initialize KV caches to populate during sync for async warmstart ===
        batch_size = hidden_states.shape[0]
        num_img_tokens = hidden_states.shape[1]
        text_seq_len = encoder_hidden_states.shape[1]

        if self._joint_kv_caches is None:
            self._initialize_kv_caches(
                batch_size=batch_size,
                num_img_tokens=num_img_tokens,
                dtype=hidden_states.dtype,
            )

        # === PHASE 2: Joint Blocks with Communication and Caching ===
        if self.has_joint_blocks:
            # Receive from previous stage (if not first stage)
            if not self.is_first_stage:
                hidden_states = mx.distributed.recv_like(
                    hidden_states, self.rank - 1, group=self.group
                )
                encoder_hidden_states = mx.distributed.recv_like(
                    encoder_hidden_states, self.rank - 1, group=self.group
                )

            # Run assigned joint blocks with caching wrappers
            for block_idx, block in enumerate(self.transformer_blocks):
                caching_block = CachingJointTransformerBlock(
                    block, self._joint_kv_caches[block_idx]
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
                mx.distributed.send(concatenated, self.rank + 1, group=self.group)

        elif self.has_joint_blocks and not self.is_last_stage:
            # Send joint block outputs to next stage (which has more joint blocks)
            mx.distributed.send(hidden_states, self.rank + 1, group=self.group)
            mx.distributed.send(encoder_hidden_states, self.rank + 1, group=self.group)

        # === PHASE 4: Single Blocks with Communication and Caching ===
        if self.has_single_blocks:
            # Receive from previous stage if we didn't do concatenation
            if not self.owns_concat_stage and not self.is_first_stage:
                concatenated = mx.concatenate(
                    [encoder_hidden_states, hidden_states], axis=1
                )
                hidden_states = mx.distributed.recv_like(
                    concatenated, self.rank - 1, group=self.group
                )
                mx.eval(hidden_states)

            # Run assigned single blocks with caching wrappers
            for block_idx, block in enumerate(self.single_transformer_blocks):
                caching_block = CachingSingleTransformerBlock(
                    block, self._single_kv_caches[block_idx]
                )
                hidden_states = caching_block(
                    hidden_states=hidden_states,
                    text_embeddings=text_embeddings,
                    rotary_embeddings=image_rotary_embeddings,
                    text_seq_len=text_seq_len,
                )

            # Send to next stage if not last
            if not self.is_last_stage:
                hidden_states = mx.distributed.send(
                    hidden_states, self.rank + 1, group=self.group
                )

        #
        # === PHASE 5: All-gather Final Output ===
        # All stages participate to receive the final output
        mx.eval(hidden_states)
        mx_barrier(group=self.group)
        hidden_states = mx.distributed.all_gather(hidden_states, group=self.group)[
            -hidden_states.shape[0] :
        ]

        # === PHASE 6: Final Projection (last stage only) ===
        # Extract image portion (remove text embeddings prefix)
        hidden_states = hidden_states[:, encoder_hidden_states.shape[1] :, ...]
        hidden_states = self.transformer.norm_out(hidden_states, text_embeddings)
        hidden_states = self.transformer.proj_out(hidden_states)

        latents = config.scheduler.step(
            model_output=hidden_states,
            timestep=t,
            sample=prev_latents,
        )

        return latents

    def _async_pipeline(
        self,
        t: int,
        config: RuntimeConfig,
        hidden_states: mx.array,
        prompt_embeds: mx.array,
        pooled_prompt_embeds: mx.array,
        kontext_image_ids: mx.array | None = None,
    ):
        prev_latents = hidden_states
        hidden_states = config.scheduler.scale_model_input(hidden_states, t)

        # === PHASE 1: Create Embeddings (all stages compute, for consistency) ===
        full_hidden = self.transformer.x_embedder(hidden_states)
        encoder_hidden_states = self.transformer.context_embedder(prompt_embeds)
        text_embeddings = Transformer.compute_text_embeddings(
            t, pooled_prompt_embeds, self.transformer.time_text_embed, config
        )
        image_rotary_embeddings = Transformer.compute_rotary_embeddings(
            prompt_embeds, self.transformer.pos_embed, config, kontext_image_ids
        )

        # === Calculate patch info ===
        batch_size = full_hidden.shape[0]
        num_img_tokens = full_hidden.shape[1]
        text_seq_len = encoder_hidden_states.shape[1]
        total_seq_len = text_seq_len + num_img_tokens

        # Latent dimensions (image size / 8 for VAE)
        latent_height = config.height // 8
        latent_width = config.width // 8
        patch_size = 2  # Flux uses 2x2 patches

        patch_heights, actual_num_patches = calculate_patch_heights(
            latent_height, self.num_patches, patch_size
        )
        token_indices = calculate_token_indices(patch_heights, latent_width, patch_size)

        # === Initialize KV caches if not already done (reused across timesteps) ===
        # This enables true PipeFusion behavior: at timestep T, patches not yet processed
        # have stale K/V from timestep T-1. If sync pipeline ran first, caches already
        # contain valid K/V from the last sync timestep.
        if self._joint_kv_caches is None:
            self._initialize_kv_caches(
                batch_size=batch_size,
                num_img_tokens=num_img_tokens,
                dtype=full_hidden.dtype,
            )

        # Use persistent caches (stale K/V from previous timestep for unprocessed patches)
        joint_kv_caches = self._joint_kv_caches
        single_kv_caches = self._single_kv_caches

        # === Process each patch ===
        # Encoder hidden states are the same for all patches in a timestep,
        # so we only need to receive them once (with the first patch)
        output_patches = []
        for patch_idx, (start_token, end_token) in enumerate(token_indices):
            # Extract current patch from full hidden states
            patch_hidden = full_hidden[:, start_token:end_token, :]

            # === PHASE 2: Joint Blocks with KV Cache ===
            if self.has_joint_blocks:
                # Receive from previous stage (if not first stage)
                if not self.is_first_stage:
                    patch_hidden = mx.distributed.recv_like(
                        patch_hidden, self.rank - 1, group=self.group
                    )
                    # Only receive encoder_hidden_states once per timestep (with first patch)
                    if patch_idx == 0:
                        encoder_hidden_states = mx.distributed.recv_like(
                            encoder_hidden_states, self.rank - 1, group=self.group
                        )

                # Run assigned joint blocks with KV cache
                for block_idx, block in enumerate(self.transformer_blocks):
                    patched_block = PatchedJointTransformerBlock(block)
                    encoder_hidden_states, patch_hidden = patched_block(
                        patch_hidden=patch_hidden,
                        encoder_hidden_states=encoder_hidden_states,
                        text_embeddings=text_embeddings,
                        image_rotary_emb=image_rotary_embeddings,
                        kv_cache=joint_kv_caches[block_idx],
                        patch_start=start_token,
                        patch_end=end_token,
                        text_seq_len=text_seq_len,
                    )

            # === PHASE 3: Joint→Single Transition ===
            if self.owns_concat_stage:
                # Concatenate encoder and hidden states for this patch
                patch_concat = mx.concatenate(
                    [encoder_hidden_states, patch_hidden], axis=1
                )

                if self.has_single_blocks:
                    # We continue with single blocks on this stage
                    patch_hidden = patch_concat
                else:
                    # Send concatenated state to next stage
                    mx.eval(
                        mx.distributed.send(
                            patch_concat, self.rank + 1, group=self.group
                        )
                    )

            elif self.has_joint_blocks and not self.is_last_stage:
                # Send joint block outputs to next stage
                # Only send encoder_hidden_states once per timestep (with first patch)
                if patch_idx == 0:
                    mx.distributed.send(
                        encoder_hidden_states, self.rank + 1, group=self.group
                    )
                else:
                    mx.distributed.send(patch_hidden, self.rank + 1, group=self.group)

            # === PHASE 4: Single Blocks with KV Cache ===
            if self.has_single_blocks:
                # Receive from previous stage if we didn't do concatenation
                if not self.owns_concat_stage and not self.is_first_stage:
                    # Create template for recv
                    recv_template = mx.concatenate(
                        [encoder_hidden_states, patch_hidden], axis=1
                    )
                    patch_hidden = mx.distributed.recv_like(
                        recv_template, self.rank - 1, group=self.group
                    )
                    mx.eval(patch_hidden)

                # Run assigned single blocks with KV cache
                for block_idx, block in enumerate(self.single_transformer_blocks):
                    patched_block = PatchedSingleTransformerBlock(block)
                    patch_hidden = patched_block(
                        patch_hidden=patch_hidden,
                        text_embeddings=text_embeddings,
                        image_rotary_emb=image_rotary_embeddings,
                        kv_cache=single_kv_caches[block_idx],
                        patch_start=start_token,
                        patch_end=end_token,
                        text_seq_len=text_seq_len,
                    )

                # Send to next stage if not last
                if not self.is_last_stage:
                    mx.eval(
                        mx.distributed.send(
                            patch_hidden, self.rank + 1, group=self.group
                        )
                    )

            # Extract only image portion from this patch (remove text prefix)
            patch_img_only = patch_hidden[:, text_seq_len:, :]
            output_patches.append(patch_img_only)

        # Reconstruct full sequence from patches (already image-only)
        hidden_states = mx.concatenate(output_patches, axis=1)

        # === PHASE 5: All-gather Final Output ===
        mx.eval(hidden_states)
        mx_barrier(group=self.group)
        hidden_states = mx.distributed.all_gather(hidden_states, group=self.group)[
            -hidden_states.shape[0] :
        ]

        # === PHASE 6: Final Projection ===
        hidden_states = self.transformer.norm_out(hidden_states, text_embeddings)
        hidden_states = self.transformer.proj_out(hidden_states)

        latents = config.scheduler.step(
            model_output=hidden_states,
            timestep=t,
            sample=prev_latents,
        )

        return latents

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
        if t == 0:
            self._joint_kv_caches = None
            self._single_kv_caches = None

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
            latents = self._async_pipeline(
                t,
                config,
                hidden_states,
                prompt_embeds,
                pooled_prompt_embeds,
                kontext_image_ids,
            )

        return latents

    # Delegate attribute access to the underlying transformer for compatibility
    def __getattr__(self, name: str) -> Any:
        return getattr(self.transformer, name)
