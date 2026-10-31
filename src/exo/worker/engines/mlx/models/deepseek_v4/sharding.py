from collections.abc import Callable, Generator
from typing import cast

import mlx.core as mx
import mlx.nn as nn
from mlx.nn.layers.distributed import sum_gradients

from exo.shared.types.worker.runner_response import ModelLoadingResponse
from exo.worker.engines.mlx.models.deepseek_v4.model import DeepseekV4MoE, V4Attention
from exo.worker.engines.mlx.models.deepseek_v4.model import Model as DeepseekV4Model
from exo.worker.engines.mlx.sharding.blocks import CustomMlxLayer, LayerCallable
from exo.worker.engines.mlx.sharding.tensor import TensorParallelShardingStrategy


class ShardedMoEV4(CustomMlxLayer):
    """Same as ShardedMoE but for DeepseekV4MoE which takes (x, input_ids)."""

    def __init__(self, layer: DeepseekV4MoE):
        super().__init__(cast(LayerCallable, cast(object, layer)))
        self._v4_inner = layer
        self.sharding_group: mx.distributed.Group | None = None

    def __call__(self, x: mx.array, input_ids: mx.array) -> mx.array:
        if self.sharding_group is not None:
            x = sum_gradients(self.sharding_group)(x)
        y = self._v4_inner(x, input_ids)
        if self.sharding_group is not None:
            y = mx.distributed.all_sum(y, group=self.sharding_group)
        return y


def _shard_quantized_rows(
    q: nn.QuantizedLinear,
    head_dim: int,
    slicer: Callable[[mx.array, int], mx.array],
) -> None:
    weight = q["weight"]
    scales = q["scales"]
    assert isinstance(weight, mx.array)
    assert isinstance(scales, mx.array)
    q.weight = slicer(weight, head_dim)
    q.scales = slicer(scales, head_dim)
    biases = q.get("biases")
    if isinstance(biases, mx.array):
        q.biases = slicer(biases, head_dim)


class _AllSumLinear(nn.Module):
    """Wraps an unsharded wo_b that takes a head-sharded partial wo_a output.

    Flow per rank:
      1. all_sum the incoming partial wo_a output (summed across the head
         input shards → full wo_a_out on every rank)
      2. apply the unsharded wo_b → full hidden on every rank

    One collective per layer on the smaller of (n_groups * o_lora_rank) vs
    hidden. wo_b compute is replicated, but at decode B=1 it's only ~30M FLOPs
    per layer and 61 extra all_gathers/token cost more than running wo_b on
    every rank.
    """

    def __init__(self, inner: nn.Module, group: mx.distributed.Group):
        super().__init__()
        self.inner = inner
        self._group = group

    def __call__(self, x: mx.array) -> mx.array:
        x = mx.distributed.all_sum(x, group=self._group)
        return cast(Callable[[mx.array], mx.array], self.inner)(x)


def _shard_v4_attention_heads(
    attn: V4Attention,
    world_size: int,
    rank: int,
) -> None:
    """Interleaved-per-group head sharding for V4Attention.

    V4 uses a grouped low-rank output projection: `_grouped_output_projection`
    reshapes the flat `n_heads * head_dim` dim into `(o_groups, heads_per_group,
    head_dim)`, so group g owns heads `[g * heads_per_group : (g+1) * heads_per_group]`.

    A naive contiguous `shard_linear("all-to-sharded")` on wq_b puts whole
    original groups on each rank — the per-rank "group g" ends up containing
    heads that don't belong to original group g. That breaks the wo_a grouped
    weight mapping. We instead slice heads interleaved-by-group: each rank
    owns `heads_per_group / N` heads *from every original group*, kept in
    group-major order so SDPA → reshape → wo_a preserves the group mapping.

    Affects `wq_b.weight` / `wq_b.bias`, `attn_sink`. wo_a is sharded via a
    normal input-dim block split (the default axis-(-1) behavior of
    shard_inplace), which now correctly aligns with the interleaved head
    layout because the last dim of out after reshape is `heads_per_group/N *
    head_dim` per group.
    """
    n_heads: int = attn.n_heads
    head_dim: int = attn.head_dim
    o_groups: int = attn.n_groups
    assert n_heads % o_groups == 0, "n_heads must be divisible by o_groups"
    heads_per_group = n_heads // o_groups
    assert heads_per_group % world_size == 0, (
        f"heads_per_group ({heads_per_group}) must be divisible by world_size "
        f"({world_size}) for interleaved per-group head sharding"
    )
    hpg_per_rank = heads_per_group // world_size
    start = rank * hpg_per_rank
    end = start + hpg_per_rank

    def _slice_head_major_flat(arr: mx.array, stride: int) -> mx.array:
        """Slice arr on axis 0 where the flat 0-axis is (o_groups *
        heads_per_group * stride), returning a fresh contiguous allocation
        so the full unsharded array can be freed. Without the contiguous
        copy the slice is a view and the original weight stays resident —
        OOM on large V4. Quantized packed weights don't round-trip through
        numpy so we use mx.contiguous directly."""
        rest = arr.shape[1:]
        reshaped = arr.reshape(o_groups, heads_per_group, stride, *rest)
        sliced = reshaped[:, start:end].reshape(o_groups * hpg_per_rank * stride, *rest)
        detached = mx.contiguous(sliced)
        mx.eval(detached)
        return detached

    wq_b: nn.Module = attn.wq_b
    if isinstance(wq_b, nn.QuantizedLinear):
        # Packed weight: (n_heads*head_dim, q_lora_rank/el_per_int).
        # scales/biases: (n_heads*head_dim, q_lora_rank/group_size).
        # Slice axis 0 interleaved-by-group with head_dim stride.
        _shard_quantized_rows(wq_b, head_dim, _slice_head_major_flat)
    else:
        dense = wq_b
        assert isinstance(dense, nn.Linear)
        w = dense.weight
        q_lora_rank = w.shape[-1]
        w_sharded = _slice_head_major_flat(w, head_dim)
        has_bias = "bias" in dense
        new_wq_b = nn.Linear(q_lora_rank, w_sharded.shape[0], bias=has_bias)
        new_wq_b.weight = w_sharded
        if has_bias:
            b = dense.bias
            assert b is not None
            new_wq_b.bias = _slice_head_major_flat(b[:, None], head_dim).reshape(-1)
        attn.wq_b = new_wq_b

    sink = attn.attn_sink
    reshaped = sink.reshape(o_groups, heads_per_group)[:, start:end].reshape(-1)
    detached_sink = mx.contiguous(reshaped)
    mx.eval(detached_sink)
    attn.attn_sink = detached_sink
    attn.n_heads = o_groups * hpg_per_rank


class DeepseekV4ShardingStrategy(TensorParallelShardingStrategy):
    def shard_model(
        self,
        model: nn.Module,
    ) -> Generator[ModelLoadingResponse, None, nn.Module]:
        model = cast(DeepseekV4Model, model)
        total = len(model.layers)

        for i, layer in enumerate(model.layers):
            mx.eval(layer.parameters())

            # Head-parallel attention with interleaved-per-group sharding.
            _shard_v4_attention_heads(layer.attn, self.N, self.group.rank())
            self.sharded_to_all_linear_in_place(layer.attn.wo_a)
            layer.attn.wo_b = _AllSumLinear(layer.attn.wo_b, self.group)  # type: ignore

            ffn = layer.ffn
            if getattr(ffn, "shared_experts", None) is not None:
                self.all_to_sharded_linear_in_place(ffn.shared_experts.gate_proj)
                self.sharded_to_all_linear_in_place(ffn.shared_experts.down_proj)
                self.all_to_sharded_linear_in_place(ffn.shared_experts.up_proj)
            self.all_to_sharded_linear_in_place(ffn.switch_mlp.gate_proj)
            self.sharded_to_all_linear_in_place(ffn.switch_mlp.down_proj)
            self.all_to_sharded_linear_in_place(ffn.switch_mlp.up_proj)
            wrapped = ShardedMoEV4(ffn)
            wrapped.sharding_group = self.group
            layer.ffn = wrapped  # type: ignore

            mx.eval(layer)
            mx.clear_cache()
            yield ModelLoadingResponse(layers_loaded=i, total=total)

        return model
