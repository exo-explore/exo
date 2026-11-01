"""Tensor and pipeline parallelism for mlx-lm's Muse Spark (``muse_spark``)."""

from collections.abc import Generator
from typing import cast

import mlx.core as mx
import mlx.nn as nn
from mlx_lm.kernels import pre_norm
from mlx_lm.models.muse_spark import Model, TransformerBlock, shard_experts_inplace

from exo.shared.types.worker.runner_response import ModelLoadingResponse
from exo.shared.types.worker.shards import PipelineShardMetadata
from exo.worker.engines.mlx.sharding.blocks import CustomMlxLayer, LayerCallable
from exo.worker.engines.mlx.sharding.pipeline import (
    PipelineFirstLayer,
    PipelineLastLayer,
    patch_pipeline_model,
)
from exo.worker.engines.mlx.sharding.tensor import TensorParallelShardingStrategy


class MuseSparkShardingStrategy(TensorParallelShardingStrategy):
    """mlx-lm's own ``Model.shard``, one layer at a time:

    - Attention: the query, key, value and output-gate heads (one fused projection, split
      segment by segment) and the output projection. The QK norms and the gated output
      norm are per head, so they need no communication.
    - The latent MoE: every expert's intermediate dimension. The router and the
      projections into and out of the experts' compressed width stay whole. The MoE block
      sums the experts' outputs across ranks before its post-expert norm and, for a few
      tokens (decoding), splits the rows of those two projections itself.
    - The LM head along the vocabulary, gathered after the head.
    """

    def shard_model(
        self,
        model: nn.Module,
    ) -> Generator[ModelLoadingResponse, None, nn.Module]:
        assert isinstance(model, Model)
        text = model.args.text
        # Check before the first change, so that a failure leaves the model whole.
        for name, count in (
            ("key/value heads", text.num_key_value_heads),
            ("vocabulary", text.vocab_size),
        ):
            if count % self.N != 0:
                raise ValueError(
                    f"Muse Spark's {count} {name} don't divide by {self.N} ranks"
                )

        model.lm_head = self.all_to_sharded_linear(model.lm_head)
        model.sharding_group = self.group
        layers = model.layers
        for i, layer in enumerate(layers):
            attention = layer.self_attn
            q_dim = attention.n_heads * attention.head_dim
            kv_dim = attention.n_kv_heads * attention.head_dim
            attention.qkvg_proj = self.all_to_sharded_linear(
                attention.qkvg_proj,
                segments=[q_dim, q_dim + kv_dim, q_dim + 2 * kv_dim],
            )
            attention.o_proj = self.sharded_to_all_linear(attention.o_proj)
            attention.n_heads //= self.N
            attention.n_kv_heads //= self.N

            # mlx-lm's own expert split: it copies large slices in chunks (the bf16 experts
            # of the first and last layers are too large for one) and keeps per-expert
            # tensor scales (nvfp4) whole.
            experts = layer.mlp.experts
            shard_experts_inplace(experts.gate_proj, "all-to-sharded", self.group)
            shard_experts_inplace(experts.down_proj, "sharded-to-all", self.group)
            shard_experts_inplace(experts.up_proj, "all-to-sharded", self.group)
            layer.mlp.sharding_group = self.group

            mx.eval(layer)
            mx.clear_cache()
            yield ModelLoadingResponse(layers_loaded=i, total=len(layers))
        return model


class _PreNormedInput(CustomMlxLayer):
    """The first layer of every pipeline stage after the first.

    Each Muse Spark layer computes the next layer's attention input in the same kernel as
    its residual add (``norm_add_norm``), and only the first layer of the model norms its
    own input. A stage receives the residual stream alone, so its first layer norms it
    with ``pre_norm``: the same sums as the fused kernel, as in mlx-lm's own pipeline.
    A split therefore gives the same values as a single device."""

    def __call__(self, x: mx.array, *args: object, **kwargs: object) -> mx.array:
        layer = cast(TransformerBlock, cast(object, self.original_layer))
        if kwargs.get("attn_input") is None:
            norm = layer.input_layernorm
            kwargs["attn_input"] = pre_norm(x, norm, dtype=norm.weight.dtype)
        return cast(mx.array, cast(object, layer(x, *args, **kwargs)))


def pipeline_parallel(
    model: nn.Module,
    group: mx.distributed.Group,
    model_shard_meta: PipelineShardMetadata,
) -> Generator[ModelLoadingResponse, None, nn.Module]:
    """The generic split (PipelineFirstLayer / PipelineLastLayer around the node's layers)
    on the language model, whose layer list the outer model only exposes as a read-only
    property, with the first layer of each later stage norming the stream it receives."""
    assert isinstance(model, Model)
    rank, size = model_shard_meta.device_rank, model_shard_meta.world_size
    text_model = model.model.language_model
    layers = cast(
        list[LayerCallable],
        text_model.layers[model_shard_meta.start_layer : model_shard_meta.end_layer],
    )
    for i, layer in enumerate(layers):
        mx.eval(layer)  # pyright: ignore[reportArgumentType]
        mx.clear_cache()
        yield ModelLoadingResponse(layers_loaded=i, total=len(layers))

    if rank > 0:
        layers[0] = _PreNormedInput(layers[0])
    layers[0] = PipelineFirstLayer(layers[0], rank, group=group)
    layers[-1] = PipelineLastLayer(layers[-1], rank, size, group=group)
    text_model.layers = cast(list[TransformerBlock], cast(object, layers))
    return patch_pipeline_model(model, group)
