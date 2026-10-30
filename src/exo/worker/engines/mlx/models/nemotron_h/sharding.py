from collections.abc import Generator
from typing import cast

import mlx.core as mx
import mlx.nn as nn
from mlx_lm.models.nemotron_h import Model as NemotronHModel
from mlx_lm.models.nemotron_h import (
    NemotronHAttention,
    NemotronHMamba2Mixer,
    NemotronHMoE,
)
from mlx_lm.models.nemotron_h import NemotronHModel as NemotronHInnerModel

from exo.shared.types.worker.runner_response import ModelLoadingResponse
from exo.worker.engines.mlx.sharding.blocks import LayerCallable, ShardedMoE
from exo.worker.engines.mlx.sharding.pipeline import patch_hybrid_cache
from exo.worker.engines.mlx.sharding.tensor import TensorParallelShardingStrategy


class NemotronHShardingStrategy(TensorParallelShardingStrategy):
    def shard_model(
        self,
        model: nn.Module,
    ) -> Generator[ModelLoadingResponse, None, nn.Module]:
        model = cast(NemotronHModel, model)
        rank = self.group.rank()
        total = len(model.layers)
        for i, layer in enumerate(model.layers):
            mx.eval(layer.parameters())

            mixer = layer.mixer

            if isinstance(mixer, NemotronHAttention):
                mixer.q_proj = self.all_to_sharded_linear(mixer.q_proj)
                mixer.k_proj = self.all_to_sharded_linear(mixer.k_proj)
                mixer.v_proj = self.all_to_sharded_linear(mixer.v_proj)
                mixer.o_proj = self.sharded_to_all_linear(mixer.o_proj)
                mixer.num_heads //= self.N
                mixer.num_key_value_heads //= self.N

            elif isinstance(mixer, NemotronHMamba2Mixer):
                self._shard_mamba2_mixer(mixer, rank)

            elif isinstance(mixer, NemotronHMoE):
                # Routed experts (SwitchMLP's fc1/fc2) and the shared expert keep a slice
                # of their hidden dimension; ShardedMoE all-reduces the partial sums.
                shard_hidden_dimension(
                    mixer.switch_mlp.fc1, mixer.switch_mlp.fc2, rank, self.N
                )
                if hasattr(mixer, "shared_experts"):
                    shard_hidden_dimension(
                        mixer.shared_experts.up_proj,
                        mixer.shared_experts.down_proj,
                        rank,
                        self.N,
                    )
                mixer = ShardedMoE(mixer)  # pyright: ignore[reportArgumentType]
                mixer.sharding_group = self.group
                layer.mixer = mixer  # pyright: ignore[reportAttributeAccessIssue]

            mx.eval(layer)
            mx.clear_cache()
            yield ModelLoadingResponse(layers_loaded=i, total=total)
        return model

    def _shard_mamba2_mixer(self, mixer: NemotronHMamba2Mixer, rank: int) -> None:
        """Shard the Mamba2 mixer along the head dimension."""
        world_size = self.N
        num_heads = mixer.num_heads
        head_dim = mixer.head_dim
        n_groups = mixer.n_groups
        ssm_state_size = mixer.ssm_state_size
        intermediate_size = mixer.intermediate_size  # = num_heads * head_dim

        # Per-rank sizes
        heads_per_rank = num_heads // world_size
        groups_per_rank = n_groups // world_size
        is_per_rank = heads_per_rank * head_dim
        bc_per_rank = groups_per_rank * ssm_state_size

        # === in_proj: output layout is [gate:IS | conv_ssm:IS | B:NG*SS | C:NG*SS | dt:NH] ===
        gate_start = 0
        conv_ssm_start = intermediate_size
        b_start = 2 * intermediate_size
        c_start = b_start + n_groups * ssm_state_size
        dt_start = c_start + n_groups * ssm_state_size

        # Build index tensor for this rank's slice of each section
        gate_idx = mx.arange(
            gate_start + rank * is_per_rank, gate_start + (rank + 1) * is_per_rank
        )
        conv_ssm_idx = mx.arange(
            conv_ssm_start + rank * is_per_rank,
            conv_ssm_start + (rank + 1) * is_per_rank,
        )
        b_idx = mx.arange(
            b_start + rank * bc_per_rank, b_start + (rank + 1) * bc_per_rank
        )
        c_idx = mx.arange(
            c_start + rank * bc_per_rank, c_start + (rank + 1) * bc_per_rank
        )
        dt_idx = mx.arange(
            dt_start + rank * heads_per_rank, dt_start + (rank + 1) * heads_per_rank
        )

        indices = mx.concatenate([gate_idx, conv_ssm_idx, b_idx, c_idx, dt_idx])
        take_output_rows(mixer.in_proj, indices)

        # === out_proj: input is intermediate_size (sharded) → hidden_size (reduce) ===
        mixer.out_proj = self.sharded_to_all_linear(mixer.out_proj)

        # === conv1d: depthwise conv on conv_dim channels ===
        # conv_dim layout: [ssm_hidden:IS | B:NG*SS | C:NG*SS]
        conv_ssm_idx_local = mx.arange(rank * is_per_rank, (rank + 1) * is_per_rank)
        conv_b_idx = mx.arange(
            intermediate_size + rank * bc_per_rank,
            intermediate_size + (rank + 1) * bc_per_rank,
        )
        conv_c_idx = mx.arange(
            intermediate_size + n_groups * ssm_state_size + rank * bc_per_rank,
            intermediate_size + n_groups * ssm_state_size + (rank + 1) * bc_per_rank,
        )
        conv_indices = mx.concatenate([conv_ssm_idx_local, conv_b_idx, conv_c_idx])
        mixer.conv1d.weight = mixer.conv1d.weight[conv_indices]
        new_conv_dim = is_per_rank + 2 * bc_per_rank
        mixer.conv1d.groups = new_conv_dim
        if mixer.conv1d.bias is not None:
            mixer.conv1d.bias = mixer.conv1d.bias[conv_indices]

        # === Per-head parameters ===
        h_start = rank * heads_per_rank
        h_end = h_start + heads_per_rank
        mixer.dt_bias = mixer.dt_bias[h_start:h_end]
        mixer.A_log = mixer.A_log[h_start:h_end]
        mixer.D = mixer.D[h_start:h_end]

        # === Norm: weight is intermediate_size ===
        mixer.norm.weight = mixer.norm.weight[
            rank * is_per_rank : (rank + 1) * is_per_rank
        ]

        # === Update dimensions ===
        mixer.num_heads = heads_per_rank
        mixer.n_groups = groups_per_rank
        mixer.intermediate_size = is_per_rank
        mixer.conv_dim = new_conv_dim
        mixer.heads_per_group = heads_per_rank // groups_per_rank


def take_output_rows(linear: nn.Module, rows: mx.array) -> None:
    """Keep only the given output rows of a linear layer, in place.

    A quantized linear layer stores its scales and biases per output row, next to the
    packed weight. All of them, and the bias, must be indexed together: MLX's quantized
    matmul does not check that the scales have as many rows as the weight, so indexing
    the weight alone silently dequantizes most rows with another row's scales.
    """
    for name in ("weight", "scales", "biases", "bias"):
        parameter = cast(object, linear.get(name))
        if isinstance(parameter, mx.array):
            setattr(linear, name, parameter[rows])


def _parameter(layer: nn.Module, name: str) -> mx.array | None:
    parameter = cast(object, layer.get(name))
    return parameter if isinstance(parameter, mx.array) else None


def hidden_dimension_bounds(
    hidden: int, step: int, rank: int, world_size: int
) -> tuple[int, int]:
    """Rank `rank`'s slice [start, end) of a hidden dimension split into `world_size`
    parts on multiples of `step`, as evenly as the steps allow."""
    steps, remainder = divmod(hidden, step)
    assert remainder == 0, f"hidden size {hidden} is not a multiple of {step}"
    base, extra = divmod(steps, world_size)
    start = rank * base + min(rank, extra)
    end = start + base + (1 if rank < extra else 0)
    return start * step, end * step


def shard_hidden_dimension(
    up: nn.Module, down: nn.Module, rank: int, world_size: int
) -> None:
    """Keep this rank's slice of an MLP's hidden dimension, in place: rows of `up` and the
    matching input columns of `down`, for linear layers and per-expert SwitchLinear ones,
    quantized or not. The caller all-reduces the partial outputs. The slices are copied
    (`mx.contiguous`): a slice is a view that would keep the whole tensor in memory.

    A quantized `down` stores one scale (and bias) per group of its inputs, so the slices
    fall on group boundaries and may be uneven: Nemotron 3 Nano and 3.5 Lightning have
    1856 = 29 x 64 expert hidden units, which no even split of their 64-group 4-bit and
    8-bit conversions can divide between two ranks.
    """
    weight = _parameter(down, "weight")
    assert weight is not None
    scales = _parameter(down, "scales")
    if scales is None:
        hidden, step, packing = weight.shape[-1], 1, 1
    else:
        group_size = cast(int, getattr(down, "group_size"))  # noqa: B009
        bits = cast(int, getattr(down, "bits"))  # noqa: B009
        hidden, step, packing = scales.shape[-1] * group_size, group_size, 32 // bits
    start, end = hidden_dimension_bounds(hidden, step, rank, world_size)

    for name in ("weight", "scales", "biases"):
        parameter = _parameter(up, name)
        if parameter is not None:
            setattr(up, name, mx.contiguous(parameter[..., start:end, :]))
    up_bias = _parameter(up, "bias")
    if up_bias is not None:
        setattr(up, "bias", mx.contiguous(up_bias[..., start:end]))  # noqa: B010

    setattr(  # noqa: B010
        down, "weight", mx.contiguous(weight[..., start // packing : end // packing])
    )
    if scales is not None:
        group_size = hidden // scales.shape[-1]
        groups = slice(start // group_size, end // group_size)
        setattr(down, "scales", mx.contiguous(scales[..., groups]))  # noqa: B010
        biases = _parameter(down, "biases")
        if biases is not None:
            setattr(down, "biases", mx.contiguous(biases[..., groups]))  # noqa: B010
    down_bias = _parameter(down, "bias")
    if down_bias is not None:
        # Every rank adds it before the all-reduce.
        setattr(down, "bias", down_bias / world_size)  # noqa: B010


def fix_pipeline_shard(
    model: nn.Module,
    inner_model_instance: nn.Module,
    layers: list[LayerCallable],
    start_layer: int,
    end_layer: int,
) -> None:
    if isinstance(inner_model_instance, NemotronHInnerModel):
        # NemotronH uses block_type: "M" (Mamba/SSM), "*" (Attention), "E" (MoE), "-" (MLP)
        # Only "M" and "*" blocks have cache entries.
        # Recompute fa_idx and ssm_idx as cache-array indices for the shard's layers.
        cache_idx = 0
        fa_idx: int | None = None
        ssm_idx: int | None = None
        for layer in layers:
            block_type = getattr(layer, "block_type", None)
            if block_type == "*":
                if fa_idx is None:
                    fa_idx = cache_idx
                cache_idx += 1
            elif block_type == "M":
                if ssm_idx is None:
                    ssm_idx = cache_idx
                cache_idx += 1
        has_attn = fa_idx is not None
        has_mamba = ssm_idx is not None
        inner_model_instance.fa_idx = fa_idx if fa_idx is not None else 0
        inner_model_instance.ssm_idx = ssm_idx if ssm_idx is not None else 0
        if not has_attn or not has_mamba:
            patch_hybrid_cache(
                cast(NemotronHModel, model),
                has_full_attn=has_attn,
                ssm_idx=inner_model_instance.ssm_idx,
                has_linear=has_mamba,
            )
