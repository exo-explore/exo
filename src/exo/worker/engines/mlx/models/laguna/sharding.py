from collections.abc import Generator
from typing import cast

import mlx.core as mx
import mlx.nn as nn
from mlx_lm.models.laguna import MLP, LagunaModel
from mlx_lm.models.laguna import Model as LagunaCausalModel

from exo.shared.types.worker.runner_response import ModelLoadingResponse
from exo.worker.engines.mlx.sharding.blocks import LayerCallable, ShardedMoE
from exo.worker.engines.mlx.sharding.tensor import TensorParallelShardingStrategy


class LagunaShardingStrategy(TensorParallelShardingStrategy):
    """Splits the attention heads together with their output gates, the dense MLP, and
    every routed expert and the shared expert along their intermediate dimension."""

    def shard_model(
        self,
        model: nn.Module,
    ) -> Generator[ModelLoadingResponse, None, nn.Module]:
        model = cast(LagunaCausalModel, model)
        total = len(model.layers)
        for i, layer in enumerate(model.layers):
            mx.eval(layer.parameters())

            # Head counts differ per layer (e.g. 48 on full-attention and 72 on sliding
            # layers); each rank keeps a contiguous block of query heads and the KV heads
            # they read.
            attention = layer.self_attn
            attention.q_proj = self.all_to_sharded_linear(attention.q_proj)
            attention.k_proj = self.all_to_sharded_linear(attention.k_proj)
            attention.v_proj = self.all_to_sharded_linear(attention.v_proj)
            if attention.gating:
                # One softplus gate per query head: the rows of the rank's heads.
                attention.g_proj = self.all_to_sharded_linear(attention.g_proj)
            attention.o_proj = self.sharded_to_all_linear(attention.o_proj)
            attention.n_heads //= self.N
            attention.n_kv_heads //= self.N

            if isinstance(layer.mlp, MLP):
                layer.mlp.gate_proj = self.all_to_sharded_linear(layer.mlp.gate_proj)
                layer.mlp.down_proj = self.sharded_to_all_linear(layer.mlp.down_proj)
                layer.mlp.up_proj = self.all_to_sharded_linear(layer.mlp.up_proj)
            else:
                # The router (and its score-correction bias) stays whole on every rank, so
                # every rank picks the same experts. The routed experts and the shared
                # expert are split along their intermediate dimension; the routed scaling
                # and the sum with the shared expert are linear, so one all-reduce of the
                # block's output completes it.
                for experts in (layer.mlp.switch_mlp, layer.mlp.shared_expert):
                    self.all_to_sharded_linear_in_place(experts.gate_proj)
                    self.sharded_to_all_linear_in_place(experts.down_proj)
                    self.all_to_sharded_linear_in_place(experts.up_proj)
                layer.mlp = ShardedMoE(layer.mlp)  # pyright: ignore[reportAttributeAccessIssue, reportArgumentType]
                layer.mlp.sharding_group = self.group

            mx.eval(layer)
            mx.clear_cache()

            yield ModelLoadingResponse(layers_loaded=i, total=total)
        return model


def fix_pipeline_shard(
    model: nn.Module,
    inner_model_instance: nn.Module,
    layers: list[LayerCallable],
    start_layer: int,
    end_layer: int,
) -> None:
    """Laguna builds its full-attention and sliding-window masks from the cache entries of
    the first layer of each kind, found in the whole model's layer list. Find them in the
    shard's layers instead.

    A shard without a full-attention layer never uses the full-attention mask, so it is
    built from the first sliding layer's entry (mlx-lm indexes the cache with it
    unconditionally). A shard without a sliding layer gets ``swa_idx = None``, which
    mlx-lm already handles."""
    if isinstance(inner_model_instance, LagunaModel):
        sliding = [cast(bool, getattr(layer, "use_sliding")) for layer in layers]  # noqa: B009
        swa_idx = sliding.index(True) if True in sliding else None
        fa_idx = sliding.index(False) if False in sliding else swa_idx
        assert fa_idx is not None
        inner_model_instance.swa_idx = swa_idx
        inner_model_instance.fa_idx = fa_idx
