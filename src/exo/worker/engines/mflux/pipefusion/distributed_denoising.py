from math import ceil
from typing import Any, Optional

import mlx.core as mx
from mflux.config.runtime_config import RuntimeConfig
from mflux.models.flux.model.flux_transformer.transformer import Transformer

from exo.shared.types.worker.shards import PipelineShardMetadata
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
        self.num_patches = num_patches if num_patches else group.size

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

    def _sync_pipeline(
        self,
        t: int,
        config: RuntimeConfig,
        hidden_states: mx.array,
        prompt_embeds: mx.array,
        pooled_prompt_embeds: mx.array,
        kontext_image_ids: mx.array | None = None,
    ):
        # === PHASE 1: Create Embeddings (all stages compute, for consistency) ===
        hidden_states = self.transformer.x_embedder(hidden_states)
        encoder_hidden_states = self.transformer.context_embedder(prompt_embeds)
        text_embeddings = Transformer.compute_text_embeddings(
            t, pooled_prompt_embeds, self.transformer.time_text_embed, config
        )
        image_rotary_embeddings = Transformer.compute_rotary_embeddings(
            prompt_embeds, self.transformer.pos_embed, config, kontext_image_ids
        )

        # === PHASE 2: Joint Blocks with Communication ===
        if self.has_joint_blocks:
            # Receive from previous stage (if not first stage)
            if not self.is_first_stage:
                hidden_states = mx.distributed.recv_like(
                    hidden_states, self.rank - 1, group=self.group
                )
                encoder_hidden_states = mx.distributed.recv_like(
                    encoder_hidden_states, self.rank - 1, group=self.group
                )

            # Run assigned joint blocks
            for block in self.transformer_blocks:
                encoder_hidden_states, hidden_states = block(
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
                # This stage is done with blocks, but will participate in all_gather
        elif self.has_joint_blocks and not self.is_last_stage:
            # Send joint block outputs to next stage (which has more joint blocks)
            mx.distributed.send(hidden_states, self.rank + 1, group=self.group)
            mx.distributed.send(encoder_hidden_states, self.rank + 1, group=self.group)

        # === PHASE 4: Single Blocks with Communication ===
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

            # Run assigned single blocks
            for block in self.single_transformer_blocks:
                hidden_states = block(
                    hidden_states=hidden_states,
                    text_embeddings=text_embeddings,
                    rotary_embeddings=image_rotary_embeddings,
                )

            # Send to next stage if not last
            if not self.is_last_stage:
                hidden_states = mx.distributed.send(
                    hidden_states, self.rank + 1, group=self.group
                )

        mx.eval(hidden_states)
        mx_barrier(group=self.group)

        #
        # === PHASE 5: All-gather Final Output ===
        # All stages participate to receive the final output
        hidden_states = mx.distributed.all_gather(hidden_states, group=self.group)[
            -hidden_states.shape[0] :
        ]

        # === PHASE 6: Final Projection (last stage only) ===
        # Extract image portion (remove text embeddings prefix)
        hidden_states = hidden_states[:, encoder_hidden_states.shape[1] :, ...]
        hidden_states = self.transformer.norm_out(hidden_states, text_embeddings)
        hidden_states = self.transformer.proj_out(hidden_states)

        return hidden_states

    def _async_pipeline(
        self,
        t: int,
        config: RuntimeConfig,
        hidden_states: mx.array,
        prompt_embeds: mx.array,
        pooled_prompt_embeds: mx.array,
        kontext_image_ids: mx.array | None = None,
    ):
        return self._sync_pipeline(
            t,
            config,
            hidden_states,
            prompt_embeds,
            pooled_prompt_embeds,
            kontext_image_ids,
        )

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
        prev_latents = hidden_states
        hidden_states = config.scheduler.scale_model_input(hidden_states, t)

        if t < self.num_sync_steps:
            hidden_states = self._sync_pipeline(
                t,
                config,
                hidden_states,
                prompt_embeds,
                pooled_prompt_embeds,
                kontext_image_ids,
            )
        else:
            hidden_states = self._async_pipeline(
                t,
                config,
                hidden_states,
                prompt_embeds,
                pooled_prompt_embeds,
                kontext_image_ids,
            )

        latents = config.scheduler.step(
            model_output=hidden_states,
            timestep=t,
            sample=prev_latents,
        )

        return latents

    # Delegate attribute access to the underlying transformer for compatibility
    def __getattr__(self, name: str) -> Any:
        return getattr(self.transformer, name)
