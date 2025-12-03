from typing import TYPE_CHECKING, Protocol, cast

import mlx.core as mx
from mflux.models.flux.model.flux_transformer.transformer import (
    JointTransformerBlock,
    SingleTransformerBlock,
    Transformer,
)
from mflux.models.flux.variants.txt2img.flux import Flux1

from exo.shared.types.worker.shards import (
    PipelineShardMetadata,
    ShardMetadata,
    TensorShardMetadata,
)
from exo.worker.engines.mlx.utils_mlx import mx_barrier


class _JointBlock(Protocol):
    def __call__(
        self,
        hidden_states: mx.array,
        encoder_hidden_states: mx.array,
        text_embeddings: mx.array,
        rotary_embeddings: mx.array,
    ) -> tuple[mx.array, mx.array]: ...


class _SingleBlock(Protocol):
    def __call__(
        self,
        hidden_states: mx.array,
        text_embeddings: mx.array,
        rotary_embeddings: mx.array,
    ) -> mx.array: ...


class CustomMlxJointBlock(JointTransformerBlock):
    """Base class for replacing an MLX layer with a custom implementation."""

    def __init__(self, original_layer: _JointBlock):
        super().__init__(None)
        # Set twice to avoid __setattr__ recursion
        object.__setattr__(self, "_original_layer", original_layer)
        self.original_layer: _JointBlock = original_layer

    # Calls __getattr__ for any attributes not found on nn.Module (e.g. use_sliding)
    if not TYPE_CHECKING:

        def __getattr__(self, name):
            try:
                return super().__getattr__(name)
            except AttributeError:
                original_layer = object.__getattribute__(self, "_original_layer")
                return object.__getattribute__(original_layer, name)


class CustomMlxSingleBlock(SingleTransformerBlock):
    """Base class for replacing an MLX layer with a custom implementation."""

    def __init__(self, original_layer: _SingleBlock):
        super().__init__(None)
        # Set twice to avoid __setattr__ recursion
        object.__setattr__(self, "_original_layer", original_layer)
        self.original_layer: _SingleBlock = original_layer

    # Calls __getattr__ for any attributes not found on nn.Module (e.g. use_sliding)
    if not TYPE_CHECKING:

        def __getattr__(self, name):
            try:
                return super().__getattr__(name)
            except AttributeError:
                original_layer = object.__getattribute__(self, "_original_layer")
                return object.__getattribute__(original_layer, name)


class FluxJointPipelineFirstBlock(CustomMlxJointBlock):
    def __init__(
        self,
        original_block: _JointBlock,
        rank: int,
        group: mx.distributed.Group,
    ):
        super().__init__(original_block)
        self.original_block = original_block
        self.rank = rank
        self.group = group

    def __call__(
        self,
        hidden_states: mx.array,
        encoder_hidden_states: mx.array,
        text_embeddings: mx.array,
        rotary_embeddings: mx.array,
    ) -> tuple[mx.array, mx.array]:
        if self.rank != 0:
            encoder_hidden_states = mx.distributed.recv_like(
                encoder_hidden_states, self.rank - 1, group=self.group
            )
            hidden_states = mx.distributed.recv_like(
                hidden_states, self.rank - 1, group=self.group
            )

        return self.original_block(
            hidden_states=hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            text_embeddings=text_embeddings,
            rotary_embeddings=rotary_embeddings,
        )


class FluxJointPipelineLastBlock(CustomMlxJointBlock):
    def __init__(
        self,
        original_block: _JointBlock,
        rank: int,
        world_size: int,
        group: mx.distributed.Group,
    ):
        super().__init__(original_block)
        self.original_block = original_block
        self.rank = rank
        self.world_size = world_size
        self.group = group

    def __call__(
        self,
        hidden_states: mx.array,
        encoder_hidden_states: mx.array,
        text_embeddings: mx.array,
        rotary_embeddings: mx.array,
    ) -> tuple[mx.array, mx.array]:
        encoder_hidden_states, hidden_states = self.original_block(
            hidden_states=hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            text_embeddings=text_embeddings,
            rotary_embeddings=rotary_embeddings,
        )

        if self.rank != self.world_size - 1:
            encoder_hidden_states = mx.distributed.send(
                encoder_hidden_states, self.rank + 1, group=self.group
            )
            hidden_states = mx.distributed.send(
                hidden_states, self.rank + 1, group=self.group
            )

        return encoder_hidden_states, hidden_states


class FluxJointToSingleTransition(CustomMlxJointBlock):
    def __init__(
        self,
        original_block: _JointBlock,
        rank: int,
        group: mx.distributed.Group,
    ):
        super().__init__(original_block)
        self.original_block = original_block
        self.rank = rank
        self.group = group

    def __call__(
        self,
        hidden_states: mx.array,
        encoder_hidden_states: mx.array,
        text_embeddings: mx.array,
        rotary_embeddings: mx.array,
    ) -> tuple[mx.array, mx.array]:
        encoder_hidden_states, hidden_states = self.original_block(
            hidden_states=hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            text_embeddings=text_embeddings,
            rotary_embeddings=rotary_embeddings,
        )

        concatenated = mx.concat([encoder_hidden_states, hidden_states], axis=-2)
        mx.distributed.send(concatenated, self.rank + 1, group=self.group)

        return encoder_hidden_states, hidden_states


class FluxSinglePipelineFirstBlock(CustomMlxSingleBlock):
    def __init__(
        self,
        original_block: _SingleBlock,
        rank: int,
        group: mx.distributed.Group,
    ):
        super().__init__(original_block)
        self.original_block = original_block
        self.rank = rank
        self.group = group

    def __call__(
        self,
        hidden_states: mx.array,
        text_embeddings: mx.array,
        rotary_embeddings: mx.array,
    ) -> mx.array:
        hidden_states = mx.distributed.recv_like(
            hidden_states, self.rank - 1, group=self.group
        )

        return self.original_block(
            hidden_states=hidden_states,
            text_embeddings=text_embeddings,
            rotary_embeddings=rotary_embeddings,
        )


class FluxSinglePipelineLastBlock(CustomMlxSingleBlock):
    def __init__(
        self,
        original_block: _SingleBlock,
        rank: int,
        world_size: int,
        group: mx.distributed.Group,
    ):
        super().__init__(original_block)
        self.original_block = original_block
        self.rank = rank
        self.world_size = world_size
        self.group = group

    def __call__(
        self,
        hidden_states: mx.array,
        text_embeddings: mx.array,
        rotary_embeddings: mx.array,
    ) -> mx.array:
        hidden_states = self.original_block(
            hidden_states=hidden_states,
            text_embeddings=text_embeddings,
            rotary_embeddings=rotary_embeddings,
        )

        if self.rank != self.world_size - 1:
            hidden_states = mx.distributed.send(
                hidden_states, self.rank + 1, group=self.group
            )

        hidden_states = mx.distributed.all_gather(hidden_states, group=self.group)[
            -hidden_states.shape[0] :
        ]

        return hidden_states


class FluxSingleSyncBlock(CustomMlxSingleBlock):
    """Fake single block for nodes without single blocks.

    Participates in all_gather to receive the final hidden_states from the last node.
    """

    def __init__(self, group: mx.distributed.Group):
        super().__init__(self._dummy_block)
        self.group = group

    @staticmethod
    def _dummy_block(
        hidden_states: mx.array,
        text_embeddings: mx.array,
        rotary_embeddings: mx.array,
    ) -> mx.array:
        return hidden_states

    def __call__(
        self,
        hidden_states: mx.array,
        text_embeddings: mx.array,
        rotary_embeddings: mx.array,
    ) -> mx.array:
        hidden_states = mx.distributed.all_gather(hidden_states, group=self.group)[
            -hidden_states.shape[0] :
        ]

        return hidden_states


def pipeline_transformer(
    model: Flux1, group: mx.distributed.Group, shard_metadata: ShardMetadata
):
    transformer: Transformer = model.transformer

    # Total = joint blocks + single blocks
    total_joint_blocks = len(transformer.transformer_blocks)
    total_single_blocks = len(transformer.single_transformer_blocks)
    total_layers = total_joint_blocks + total_single_blocks

    start_layer = shard_metadata.start_layer
    end_layer = shard_metadata.end_layer
    rank = shard_metadata.device_rank
    world_size = shard_metadata.world_size

    if end_layer <= total_joint_blocks:
        assigned_joint_blocks = cast(
            list[_JointBlock], transformer.transformer_blocks[start_layer:end_layer]
        )
        assigned_single_blocks = []
    elif start_layer >= total_joint_blocks:
        assigned_joint_blocks = []
        single_start = start_layer - total_joint_blocks
        single_end = end_layer - total_joint_blocks
        assigned_single_blocks = cast(
            list[_SingleBlock],
            transformer.single_transformer_blocks[single_start:single_end],
        )
    else:
        assigned_joint_blocks = cast(
            list[_JointBlock], transformer.transformer_blocks[start_layer:]
        )
        single_end = end_layer - total_joint_blocks
        assigned_single_blocks = cast(
            list[_SingleBlock], transformer.single_transformer_blocks[:single_end]
        )

    if assigned_joint_blocks:
        if rank > 0:
            assigned_joint_blocks[0] = FluxJointPipelineFirstBlock(
                original_block=assigned_joint_blocks[0],
                rank=rank,
                group=group,
            )

        if rank < world_size - 1:
            if end_layer == total_joint_blocks:
                # This node has the last joint block, next node has single blocks
                assigned_joint_blocks[-1] = FluxJointToSingleTransition(
                    original_block=assigned_joint_blocks[-1],
                    rank=rank,
                    group=group,
                )
            elif end_layer < total_joint_blocks:
                # Next node has more joint blocks
                assigned_joint_blocks[-1] = FluxJointPipelineLastBlock(
                    original_block=assigned_joint_blocks[-1],
                    rank=rank,
                    world_size=world_size,
                    group=group,
                )

    # Single blocks
    if assigned_single_blocks:
        # Wrap first single block if receiving from previous node
        # (either from joint blocks on previous node or single blocks on previous node)
        has_joint_on_this_node = len(assigned_joint_blocks) > 0
        is_first_single_globally = start_layer == total_joint_blocks

        if not has_joint_on_this_node and not is_first_single_globally:
            # Receiving from previous node (which had single blocks)
            assigned_single_blocks[0] = FluxSinglePipelineFirstBlock(
                original_block=assigned_single_blocks[0],
                rank=rank,
                group=group,
            )

        # Wrap last single block (always do all_gather)
        assigned_single_blocks[-1] = FluxSinglePipelineLastBlock(
            original_block=assigned_single_blocks[-1],
            rank=rank,
            world_size=world_size,
            group=group,
        )
    else:
        # No single blocks on this node - add sync block to participate in all_gather
        assigned_single_blocks = [FluxSingleSyncBlock(group)]

    # Replace transformer blocks with sharded versions
    transformer.transformer_blocks = assigned_joint_blocks
    transformer.single_transformer_blocks = assigned_single_blocks

    return model


def shard_flux_transformer(
    model: Flux1,
    group: mx.distributed.Group,
    shard_metadata: ShardMetadata,
) -> Flux1:
    match shard_metadata:
        case TensorShardMetadata():
            raise NotImplementedError(
                "Tensor parallelism is not yet supported for Flux models. "
                "Use pipeline parallelism instead."
            )
        case PipelineShardMetadata():
            model = pipeline_transformer(model, group, shard_metadata)

    mx.eval(model.parameters())

    # TODO: Do we need this?
    mx.eval(model)

    # Synchronize processes before generation to avoid timeout
    mx_barrier(group)

    return model
