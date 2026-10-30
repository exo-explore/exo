from collections.abc import Generator
from typing import cast

import mlx.core as mx
import mlx.nn as nn
from mlx.nn.layers.distributed import shard_linear
from mlx_lm.models.qwen3 import Model as Qwen3Model
from mlx_lm.models.qwen3 import TransformerBlock as Qwen3TransformerBlock
from mlx_lm.models.qwen3_5 import DecoderLayer as Qwen3_5DecoderLayer
from mlx_lm.models.qwen3_5 import GatedDeltaNet as Qwen3_5GatedDeltaNet
from mlx_lm.models.qwen3_5 import Model as Qwen3_5TextModel
from mlx_lm.models.qwen3_5 import Qwen3_5TextModel as Qwen3_5TextModelInner
from mlx_lm.models.qwen3_5 import SparseMoeBlock as Qwen3_5SparseMoeBlock
from mlx_lm.models.qwen3_5_moe import Model as Qwen3_5MoeModel
from mlx_lm.models.qwen3_moe import Model as Qwen3MoeModel
from mlx_lm.models.qwen3_moe import Qwen3MoeDecoderLayer, Qwen3MoeSparseMoeBlock
from mlx_lm.models.qwen3_next import Model as Qwen3NextModel
from mlx_lm.models.qwen3_next import (
    Qwen3NextDecoderLayer,
    Qwen3NextGatedDeltaNet,
    Qwen3NextSparseMoeBlock,
)
from mlx_lm.models.qwen3_next import Qwen3NextModel as Qwen3NextInnerModel
from mlx_lm.models.qwen3_vl import Model as Qwen3VLModel

from exo.shared.types.worker.runner_response import ModelLoadingResponse
from exo.worker.engines.mlx.models.qwen3.qwen4_exp import Model as Qwen4ExpModel
from exo.worker.engines.mlx.models.qwen3.qwen4_exp import (
    Qwen4ExpModel as Qwen4ExpInnerModel,
)
from exo.worker.engines.mlx.models.qwen3.qwen4_exp import (
    SparseMoeBlock as Qwen4ExpSparseMoeBlock,
)
from exo.worker.engines.mlx.sharding.blocks import LayerCallable, ShardedMoE
from exo.worker.engines.mlx.sharding.pipeline import patch_hybrid_cache
from exo.worker.engines.mlx.sharding.tensor import TensorParallelShardingStrategy


class QwenShardingStrategy(TensorParallelShardingStrategy):
    def shard_model(
        self,
        model: nn.Module,
    ) -> Generator[ModelLoadingResponse, None, nn.Module]:
        model = cast(
            Qwen3Model
            | Qwen3MoeModel
            | Qwen3NextModel
            | Qwen3_5TextModel
            | Qwen3_5MoeModel
            | Qwen3VLModel,
            model,
        )
        total = len(model.layers)
        for i, layer in enumerate(model.layers):
            mx.eval(layer.parameters())
            # Shard the self attention
            if isinstance(layer, (Qwen3MoeDecoderLayer, Qwen3TransformerBlock)):
                layer.self_attn.q_proj = self.all_to_sharded_linear(
                    layer.self_attn.q_proj
                )
                layer.self_attn.k_proj = self.all_to_sharded_linear(
                    layer.self_attn.k_proj
                )
                layer.self_attn.v_proj = self.all_to_sharded_linear(
                    layer.self_attn.v_proj
                )
                layer.self_attn.o_proj = self.sharded_to_all_linear(
                    layer.self_attn.o_proj
                )
                layer.self_attn.n_heads //= self.N
                layer.self_attn.n_kv_heads //= self.N
            else:
                assert isinstance(layer, (Qwen3NextDecoderLayer, Qwen3_5DecoderLayer))
                if hasattr(layer, "linear_attn"):
                    linear_attn = layer.linear_attn

                    if isinstance(linear_attn, Qwen3NextGatedDeltaNet):
                        # Qwen3-Next: combined projections
                        linear_attn.in_proj_qkvz = self.all_to_sharded_linear(
                            linear_attn.in_proj_qkvz
                        )
                        linear_attn.in_proj_ba = self.all_to_sharded_linear(
                            linear_attn.in_proj_ba
                        )
                        shard_gated_delta_net_heads(self, linear_attn)
                    else:
                        shard_qwen3_5_gated_delta_net(self, linear_attn)
                else:
                    layer.self_attn.q_proj = self.all_to_sharded_linear(
                        layer.self_attn.q_proj
                    )
                    layer.self_attn.k_proj = self.all_to_sharded_linear(
                        layer.self_attn.k_proj
                    )
                    layer.self_attn.v_proj = self.all_to_sharded_linear(
                        layer.self_attn.v_proj
                    )
                    layer.self_attn.o_proj = self.sharded_to_all_linear(
                        layer.self_attn.o_proj
                    )
                    layer.self_attn.num_attention_heads //= self.N
                    layer.self_attn.num_key_value_heads //= self.N

            # Shard the MoE.
            if isinstance(
                layer.mlp,
                (
                    Qwen3MoeSparseMoeBlock,
                    Qwen3NextSparseMoeBlock,
                    Qwen3_5SparseMoeBlock,
                ),
            ):
                self.all_to_sharded_linear_in_place(layer.mlp.switch_mlp.gate_proj)
                self.sharded_to_all_linear_in_place(layer.mlp.switch_mlp.down_proj)
                self.all_to_sharded_linear_in_place(layer.mlp.switch_mlp.up_proj)
                if isinstance(
                    layer.mlp, (Qwen3NextSparseMoeBlock, Qwen3_5SparseMoeBlock)
                ):
                    self.all_to_sharded_linear_in_place(
                        layer.mlp.shared_expert.gate_proj
                    )
                    self.sharded_to_all_linear_in_place(
                        layer.mlp.shared_expert.down_proj
                    )
                    self.all_to_sharded_linear_in_place(layer.mlp.shared_expert.up_proj)
                layer.mlp = ShardedMoE(layer.mlp)  # pyright: ignore[reportAttributeAccessIssue, reportArgumentType]
                layer.mlp.sharding_group = self.group

            # Shard the MLP
            else:
                layer.mlp.gate_proj = self.all_to_sharded_linear(layer.mlp.gate_proj)
                layer.mlp.down_proj = self.sharded_to_all_linear(layer.mlp.down_proj)
                layer.mlp.up_proj = self.all_to_sharded_linear(layer.mlp.up_proj)

            mx.eval(layer)
            mx.clear_cache()

            yield ModelLoadingResponse(layers_loaded=i, total=total)
        return model


def shard_qwen3_5_gated_delta_net(
    strategy: TensorParallelShardingStrategy, linear_attn: Qwen3_5GatedDeltaNet
) -> None:
    """Shard a Qwen3.5 Gated DeltaNet (separate q/k/v, z, b, a projections) by heads."""
    # in_proj_qkv has sections [q(key_dim), k(key_dim), v(value_dim)]
    # that must be split section-aware, not as a contiguous block
    key_dim = linear_attn.key_dim
    linear_attn.in_proj_qkv = shard_linear(
        linear_attn.in_proj_qkv,
        "all-to-sharded",
        segments=[key_dim, key_dim + key_dim],
        group=strategy.group,
    )
    linear_attn.in_proj_z = strategy.all_to_sharded_linear(linear_attn.in_proj_z)
    linear_attn.in_proj_b = strategy.all_to_sharded_linear(linear_attn.in_proj_b)
    linear_attn.in_proj_a = strategy.all_to_sharded_linear(linear_attn.in_proj_a)
    shard_gated_delta_net_heads(strategy, linear_attn)


def shard_gated_delta_net_heads(
    strategy: TensorParallelShardingStrategy,
    linear_attn: Qwen3NextGatedDeltaNet | Qwen3_5GatedDeltaNet,
) -> None:
    """Shard what Qwen3-Next's and Qwen3.5's Gated DeltaNets share once their input
    projections are sharded: the output projection, the convolution channels, the
    per-head parameters and the head counts."""
    linear_attn.out_proj = strategy.sharded_to_all_linear(linear_attn.out_proj)

    # Shard conv1d: depthwise conv with non-contiguous channel slicing.
    # Channel layout is [q(key_dim), k(key_dim), v(value_dim)].
    # Each rank takes its head-slice from each of the three sections.
    n = strategy.N
    rank = strategy.group.rank()
    key_dim = linear_attn.key_dim
    value_dim = linear_attn.value_dim
    key_dim_shard = key_dim // n
    value_dim_shard = value_dim // n

    q_idx = mx.arange(rank * key_dim_shard, (rank + 1) * key_dim_shard)
    k_idx = mx.arange(
        key_dim + rank * key_dim_shard,
        key_dim + (rank + 1) * key_dim_shard,
    )
    v_idx = mx.arange(
        2 * key_dim + rank * value_dim_shard,
        2 * key_dim + (rank + 1) * value_dim_shard,
    )
    conv_indices = mx.concatenate([q_idx, k_idx, v_idx])
    linear_attn.conv1d.weight = linear_attn.conv1d.weight[conv_indices]
    new_conv_dim = key_dim_shard * 2 + value_dim_shard
    linear_attn.conv1d.groups = new_conv_dim

    num_v_shard = linear_attn.num_v_heads // n
    v_start = rank * num_v_shard
    v_end = v_start + num_v_shard
    linear_attn.A_log = linear_attn.A_log[v_start:v_end]
    linear_attn.dt_bias = linear_attn.dt_bias[v_start:v_end]

    linear_attn.num_k_heads //= n
    linear_attn.num_v_heads //= n
    linear_attn.key_dim = linear_attn.head_k_dim * linear_attn.num_k_heads
    linear_attn.value_dim = linear_attn.head_v_dim * linear_attn.num_v_heads
    linear_attn.conv_dim = linear_attn.key_dim * 2 + linear_attn.value_dim


def fix_pipeline_shard(
    model: nn.Module,
    inner_model_instance: nn.Module,
    layers: list[LayerCallable],
    start_layer: int,
    end_layer: int,
) -> None:
    if isinstance(inner_model_instance, (Qwen3_5TextModelInner, Qwen3NextInnerModel)):
        full_attn_layers = [
            i for i, layer in enumerate(layers) if not getattr(layer, "is_linear", True)
        ]
        linear_layers = [
            i for i, layer in enumerate(layers) if getattr(layer, "is_linear", False)
        ]
        inner_model_instance.fa_idx = full_attn_layers[0] if full_attn_layers else 0
        inner_model_instance.ssm_idx = linear_layers[0] if linear_layers else 0
        if not full_attn_layers or not linear_layers:
            patch_hybrid_cache(
                cast(Qwen3_5TextModel | Qwen3NextModel, model),
                fa_idx=inner_model_instance.fa_idx,
                has_full_attn=bool(full_attn_layers),
                ssm_idx=inner_model_instance.ssm_idx,
                has_linear=bool(linear_layers),
            )


class Qwen4ExpShardingStrategy(TensorParallelShardingStrategy):
    """Qwen3.8-Flash-Next (qwen4_exp): the Gated DeltaNets, attention heads and experts
    are split as Qwen3.5's. The QSA indexer, hyper-connections, n-gram embedding,
    embeddings and LM head are replicated (they are small, or read on the host)."""

    def shard_model(
        self,
        model: nn.Module,
    ) -> Generator[ModelLoadingResponse, None, nn.Module]:
        assert isinstance(model, Qwen4ExpModel)
        kv_heads = model.language_model.args.num_key_value_heads
        if kv_heads % self.N:
            raise ValueError(
                f"Tensor parallelism over {self.N} nodes needs {self.N} to divide the "
                f"{kv_heads} KV heads"
            )
        model.model.tensor_group = self.group
        total = len(model.layers)
        for i, layer in enumerate(model.layers):
            mx.eval(layer.parameters())
            if layer.is_linear:
                shard_qwen3_5_gated_delta_net(self, layer.linear_attn)
            else:
                attention = layer.self_attn
                attention.q_proj = self.all_to_sharded_linear(attention.q_proj)
                attention.k_proj = self.all_to_sharded_linear(attention.k_proj)
                attention.v_proj = self.all_to_sharded_linear(attention.v_proj)
                attention.o_proj = self.sharded_to_all_linear(attention.o_proj)
                attention.n_heads //= self.N
                attention.n_kv_heads //= self.N

            moe = layer.mlp
            assert isinstance(moe, Qwen4ExpSparseMoeBlock)
            self.all_to_sharded_linear_in_place(moe.switch_mlp.gate_proj)
            self.sharded_to_all_linear_in_place(moe.switch_mlp.down_proj)
            self.all_to_sharded_linear_in_place(moe.switch_mlp.up_proj)
            self.all_to_sharded_linear_in_place(moe.shared_expert.gate_proj)
            self.sharded_to_all_linear_in_place(moe.shared_expert.down_proj)
            self.all_to_sharded_linear_in_place(moe.shared_expert.up_proj)
            sharded_moe = ShardedMoE(moe)  # pyright: ignore[reportArgumentType]
            sharded_moe.sharding_group = self.group
            layer.mlp = sharded_moe

            mx.eval(layer)
            mx.clear_cache()
            yield ModelLoadingResponse(layers_loaded=i, total=total)
        return model


def fix_qwen4_exp_pipeline_shard(
    model: nn.Module,
    inner_model_instance: nn.Module,
    layers: list[LayerCallable],
    start_layer: int,
    end_layer: int,
) -> None:
    """Point the attention and linear-attention masks at this stage's own layers (None
    when the stage has no layer of that kind)."""
    assert isinstance(inner_model_instance, Qwen4ExpInnerModel)
    kinds = [cast(bool, getattr(layer, "is_linear")) for layer in layers]  # noqa: B009
    inner_model_instance.fa_idx = next(
        (i for i, linear in enumerate(kinds) if not linear), None
    )
    inner_model_instance.ssm_idx = next(
        (i for i, linear in enumerate(kinds) if linear), None
    )
