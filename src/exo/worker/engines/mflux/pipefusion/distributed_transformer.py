from typing import Any

import mlx.core as mx
from mflux.config.runtime_config import RuntimeConfig
from mflux.models.flux.model.flux_transformer.transformer import Transformer

from exo.shared.types.worker.shards import PipelineShardMetadata
from exo.worker.engines.mlx.utils_mlx import mx_barrier
from exo.worker.runner.bootstrap import logger


class DistributedTransformer:
    """
    Reimplements Transformer forward with distributed communication.

    Each stage runs assigned blocks with send/recv at phase boundaries

    This design:
    - All stages compute embeddings (deterministic, same on all nodes)
        TODO: only perform on rank 0?
    - Communication happens at joint→single transition and end of phases
    - Final projection only on last stage
    - All-gather synchronizes final output to all nodes
    """

    def __init__(
        self,
        transformer: Transformer,
        group: mx.distributed.Group,
        shard_metadata: PipelineShardMetadata,
    ):
        self.transformer = transformer
        self.group = group
        self.rank = shard_metadata.device_rank
        self.world_size = shard_metadata.world_size
        self.start_layer = shard_metadata.start_layer
        self.end_layer = shard_metadata.end_layer

        # Get block counts from the original transformer (before slicing)
        # Note: These are the ORIGINAL counts, not the sliced counts
        self.total_joint = 19  # Flux has 19 joint blocks
        self.total_single = 38  # Flux has 38 single blocks
        self.total_layers = self.total_joint + self.total_single

        # Compute which blocks this stage owns
        self._compute_assigned_blocks()

    def _compute_assigned_blocks(self) -> None:
        """Determine which joint/single blocks this stage owns."""
        start = self.start_layer
        end = self.end_layer

        # Joint block range for this stage
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

        # Convenience properties
        self.has_joint_blocks = self.joint_end > self.joint_start
        self.has_single_blocks = self.single_end > self.single_start

        # Determine if this stage handles the joint→single concatenation
        # This happens when we have joint blocks and either:
        # - We also have single blocks (transition within this stage), or
        # - We're the last stage to have joint blocks and next stage has single
        self.owns_concat_stage = self.has_joint_blocks and (
            self.has_single_blocks or self.end_layer == self.total_joint
        )

    @property
    def is_first_stage(self) -> bool:
        return self.rank == 0

    @property
    def is_last_stage(self) -> bool:
        return self.rank == self.world_size - 1

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
        """Forward pass with inline distributed communication."""
        transformer = self.transformer

        # === PHASE 1: Create Embeddings (all stages compute, for consistency) ===
        hidden_states = transformer.x_embedder(hidden_states)
        encoder_hidden_states = transformer.context_embedder(prompt_embeds)
        text_embeddings = Transformer.compute_text_embeddings(
            t, pooled_prompt_embeds, transformer.time_text_embed, config
        )
        image_rotary_embeddings = Transformer.compute_rotary_embeddings(
            prompt_embeds, transformer.pos_embed, config, kontext_image_ids
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
            for idx in range(self.joint_start, self.joint_end):
                block = transformer.transformer_blocks[idx]
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
                hidden_states = mx.concatenate(
                    [encoder_hidden_states, hidden_states], axis=1
                )
                mx_barrier(self.group)
                hidden_states = mx.distributed.recv_like(
                    hidden_states, self.rank - 1, group=self.group
                )

            # Run assigned single blocks
            for idx in range(self.single_start, self.single_end):
                block = transformer.single_transformer_blocks[idx]
                hidden_states = block(
                    hidden_states=hidden_states,
                    text_embeddings=text_embeddings,
                    rotary_embeddings=image_rotary_embeddings,
                )

            # Send to next stage if not last
            if not self.is_last_stage:
                mx_barrier(self.group)
                hidden_states = mx.distributed.send(
                    hidden_states, self.rank + 1, group=self.group
                )

        # === PHASE 5: All-gather Final Output ===
        # All stages participate to receive the final output
        mx_barrier(self.group)
        hidden_states = mx.distributed.all_gather(hidden_states, group=self.group)[
            -hidden_states.shape[0] :
        ]

        # === PHASE 6: Final Projection (last stage only) ===
        # Extract image portion (remove text embeddings prefix)
        hidden_states = hidden_states[:, encoder_hidden_states.shape[1] :, ...]
        hidden_states = transformer.norm_out(hidden_states, text_embeddings)
        hidden_states = transformer.proj_out(hidden_states)

        return hidden_states

    # Delegate attribute access to the underlying transformer for compatibility
    def __getattr__(self, name: str) -> Any:
        return getattr(self.transformer, name)
