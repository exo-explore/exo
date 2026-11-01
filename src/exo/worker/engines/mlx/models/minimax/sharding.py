from collections.abc import Generator
from typing import TYPE_CHECKING, cast

import mlx.core as mx
import mlx.nn as nn
from mlx_lm.models.base import (
    scaled_dot_product_attention,
)
from mlx_lm.models.minimax import MiniMaxAttention
from mlx_lm.models.minimax import Model as MiniMaxModel
from mlx_lm.models.minimax_m3_vl import Model as MiniMaxM3Model

from exo.shared.types.worker.runner_response import ModelLoadingResponse
from exo.worker.engines.mlx.sharding.blocks import (
    CustomMlxLayer,
    LayerCallable,
    ShardedMoE,
)
from exo.worker.engines.mlx.sharding.tensor import TensorParallelShardingStrategy

if TYPE_CHECKING:
    from mlx_lm.models.cache import Cache


class WrappedMiniMaxAttention(CustomMlxLayer):
    def __init__(self, layer: LayerCallable, group: mx.distributed.Group):
        super().__init__(layer)
        self.group = group

    def __call__(
        self,
        x: mx.array,
        mask: mx.array | None = None,
        cache: "Cache | None" = None,
    ) -> mx.array:
        batch_dim, seq_dim, _ = x.shape

        self._original_layer = cast(MiniMaxAttention, self.original_layer)  # type: ignore

        queries: mx.array = self._original_layer.q_proj(x)
        keys: mx.array = self._original_layer.k_proj(x)
        values: mx.array = self._original_layer.v_proj(x)

        if getattr(self, "use_qk_norm", False):
            q_dim = queries.shape[-1]
            k_dim = keys.shape[-1]
            n = self.group.size()

            qk = mx.concatenate(
                [queries, keys], axis=-1
            )  # (batch_dim, seq_dim, q_dim + k_dim)
            qk = mx.distributed.all_gather(
                qk, group=self.group
            )  # (n*batch_dim, seq_dim, q_dim + k_dim)

            qk = qk.reshape(n, batch_dim, seq_dim, q_dim + k_dim).transpose(1, 2, 0, 3)
            queries = qk[..., :q_dim].reshape(
                batch_dim, seq_dim, -1
            )  # (batch_dim, seq_dim, n * q_dim)
            keys = qk[..., q_dim:].reshape(
                batch_dim, seq_dim, -1
            )  # (batch_dim, seq_dim, n * k_dim)

            queries = self._original_layer.q_norm(queries)
            keys = self._original_layer.k_norm(keys)

            # Split back and take this rank's portion
            queries = mx.split(queries, n, axis=-1)[self.group.rank()]
            keys = mx.split(keys, n, axis=-1)[self.group.rank()]

        queries = queries.reshape(
            batch_dim, seq_dim, self._original_layer.num_attention_heads, -1
        ).transpose(0, 2, 1, 3)
        keys = keys.reshape(
            batch_dim, seq_dim, self._original_layer.num_key_value_heads, -1
        ).transpose(0, 2, 1, 3)
        values = values.reshape(
            batch_dim, seq_dim, self._original_layer.num_key_value_heads, -1
        ).transpose(0, 2, 1, 3)

        if cache is not None:
            queries = self._original_layer.rope(queries, offset=cache.offset)
            keys = self._original_layer.rope(keys, offset=cache.offset)
            keys, values = cache.update_and_fetch(keys, values)
        else:
            queries = self._original_layer.rope(queries)
            keys = self._original_layer.rope(keys)

        output = scaled_dot_product_attention(
            queries,
            keys,
            values,
            cache=cache,
            scale=self._original_layer.scale,
            mask=mask,
        )

        output = output.transpose(0, 2, 1, 3).reshape(batch_dim, seq_dim, -1)

        return self._original_layer.o_proj(output)


class MiniMaxShardingStrategy(TensorParallelShardingStrategy):
    def shard_model(
        self,
        model: nn.Module,
    ) -> Generator[ModelLoadingResponse, None, nn.Module]:
        model = cast(MiniMaxModel, model)
        total = len(model.layers)
        for i, layer in enumerate(model.layers):
            mx.eval(layer.parameters())
            # Shard the self attention
            layer.self_attn.q_proj = self.all_to_sharded_linear(layer.self_attn.q_proj)
            layer.self_attn.k_proj = self.all_to_sharded_linear(layer.self_attn.k_proj)
            layer.self_attn.v_proj = self.all_to_sharded_linear(layer.self_attn.v_proj)
            layer.self_attn.o_proj = self.sharded_to_all_linear(layer.self_attn.o_proj)

            layer.self_attn.num_attention_heads //= self.N
            layer.self_attn.num_key_value_heads //= self.N

            layer.self_attn = WrappedMiniMaxAttention(layer.self_attn, self.group)  # pyright: ignore[reportAttributeAccessIssue,reportArgumentType]

            # Shard the MoE.
            self.all_to_sharded_linear_in_place(
                layer.block_sparse_moe.switch_mlp.gate_proj
            )
            self.sharded_to_all_linear_in_place(
                layer.block_sparse_moe.switch_mlp.down_proj
            )
            self.all_to_sharded_linear_in_place(
                layer.block_sparse_moe.switch_mlp.up_proj
            )
            layer.block_sparse_moe = ShardedMoE(layer.block_sparse_moe)  # type: ignore
            layer.block_sparse_moe.sharding_group = self.group
            mx.eval(layer)
            mx.clear_cache()

            yield ModelLoadingResponse(layers_loaded=i, total=total)
        return model


class MiniMaxM3ShardingStrategy(TensorParallelShardingStrategy):
    """MiniMax M3 / M3.1 (minimax_m3_vl), like mlx-lm's own ``Model.shard``.

    - Attention: query, key and value heads. The QK norms are per head, so they need no
      communication.
    - Sparse attention: the lightning indexer has one index head per KV group, so each
      rank keeps the index heads of its own KV heads. Its key projection is a single head
      that every group shares, so it stays whole.
    - MLPs: the dense MLPs, and the routed experts and the shared expert along their
      intermediate dimension. The router stays whole on every rank; the MoE block sums the
      routed and shared outputs across ranks once (its own ``sharding_group``).
    """

    def shard_model(
        self,
        model: nn.Module,
    ) -> Generator[ModelLoadingResponse, None, nn.Module]:
        model = cast(MiniMaxM3Model, model)
        total = len(model.layers)
        for i, layer in enumerate(model.layers):
            attention = layer.self_attn
            # Check before the first change, so that a failure leaves no layer half split.
            heads = [attention.num_attention_heads, attention.num_key_value_heads]
            if attention.is_sparse_attn:
                heads.append(attention.index_n_heads)
            if any(count % self.N != 0 for count in heads):
                raise ValueError(
                    f"MiniMax M3 head counts {heads} don't divide by {self.N} ranks"
                )

            mx.eval(layer.parameters())

            attention.q_proj = self.all_to_sharded_linear(attention.q_proj)
            attention.k_proj = self.all_to_sharded_linear(attention.k_proj)
            attention.v_proj = self.all_to_sharded_linear(attention.v_proj)
            attention.o_proj = self.sharded_to_all_linear(attention.o_proj)
            attention.num_attention_heads //= self.N
            attention.num_key_value_heads //= self.N
            if attention.is_sparse_attn:
                attention.index_q_proj = self.all_to_sharded_linear(
                    attention.index_q_proj
                )
                attention.index_n_heads //= self.N

            if layer.is_moe:
                moe = layer.block_sparse_moe
                for experts in (moe.switch_mlp, moe.shared_experts):
                    self.all_to_sharded_linear_in_place(experts.gate_proj)
                    self.all_to_sharded_linear_in_place(experts.up_proj)
                    self.sharded_to_all_linear_in_place(experts.down_proj)
                moe.sharding_group = self.group
            else:
                layer.mlp.gate_proj = self.all_to_sharded_linear(layer.mlp.gate_proj)
                layer.mlp.up_proj = self.all_to_sharded_linear(layer.mlp.up_proj)
                layer.mlp.down_proj = self.sharded_to_all_linear(layer.mlp.down_proj)

            mx.eval(layer)
            mx.clear_cache()

            yield ModelLoadingResponse(layers_loaded=i, total=total)
        return model
