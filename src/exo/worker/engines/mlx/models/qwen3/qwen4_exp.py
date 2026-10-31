"""Qwen3.8-Flash-Next (config model_type ``qwen4_exp``), text only: MLX model definition.

Provenance:
- Ported from ``mlx_lm/models/qwen4_exp.py`` in rltakashige/mlx-lm at commit
  ca64b72634fe4862a041878a6a16d353bf86f7cf
  (https://github.com/rltakashige/mlx-lm/blob/ca64b72634fe4862a041878a6a16d353bf86f7cf/mlx_lm/models/qwen4_exp.py),
  MIT licence, Copyright (c) 2026 Apple Inc. Copied 2026-10-30. Upstream ml-explore/mlx-lm has
  no qwen4_exp model.
- The arithmetic follows Hugging Face transformers' ``qwen4_exp`` model
  (src/transformers/models/qwen4_exp/modeling_qwen4_exp.py at v5.22.0, Apache-2.0,
  Copyright 2026 The Qwen Team and The HuggingFace Inc. team), the reference this port is
  checked against.

What the model adds to Qwen3.5 / Qwen3-Next:
- Four residual streams mixed by gated hyper-connections; no layer norms and no final norm
  (a final mixer takes its place).
- A hashed n-gram embedding (PLE) at one linear-attention layer. Its table (51B parameters)
  stays in the checkpoint files: rows are read through memory maps on the host.
- Qwen Sparse Attention on the full-attention layers: an indexer picks 512 blocks of 4 keys
  per query, applied as a boolean mask on dense attention. While a query sees no more than
  512 complete blocks (2048 tokens) every key is picked and attention is plain causal.
- A sigmoid (not SiLU) output gate in the Gated DeltaNet layers.
- Zero-centred norm gains, applied as ``norm(x) * (1 + w)`` (in float32, as the reference).

Local changes against the fork's file:
- Plain MLX ops instead of the fork's Metal kernels and fusions (fused_ops, hc_small,
  qmv_small, moe_small, the atomic MoE sum, fused q/k/v, z/b/a and gate/up projections,
  hyper-connection injection rows fused into the mix projection, the deferred block
  combine). Projections keep the checkpoint's names and shapes, so tensor parallelism shards
  them like Qwen3.5's.
- The Gated DeltaNet, MoE experts and shared expert are upstream mlx-lm's qwen3_5 / qwen3_next
  modules. The router takes its softmax and top-k in float32, as the reference.
- Norm gains, the group norms and the indexer pooling are computed in float32 as the
  reference does, then cast back.
- The n-gram table reader handles 2/4/8-bit affine tables (mlx-community layout,
  ``ngram_embedding.shards.N``) and unquantised ones (the official layout,
  ``ngram_embedding.shard_N``). The table is attached after loading
  (``Model.attach_ngram_table``) instead of through a model_path config field.
- The hash multipliers, prime table sizes and offsets are checked against the buffers the
  checkpoint stores.
- No speculative (MTP) rows, no row cache: the page cache keeps the table warm.
- Caches are mlx-lm's ArraysCache (the PLE layer's has four entries: the Gated DeltaNet's
  two, the PLE conv state and the n-gram context) and a KVCache that also keeps the
  indexer's keys.
- Strict typing.

Inference path (Model.pack_for_inference, run after loading and sharding): projections
that share an input are packed into one matmul, the MoE's shared expert joins the routed
experts, and decode steps of one token run fused Metal kernels that reproduce the plain
ops bit for bit. The n-gram rows of a decode step
are read after its graph is built. Prefill keeps the plain ops.
"""

import json
import math
import mmap
import os
import re
import struct
import threading
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, Self, cast

import mlx.core as mx
import mlx.nn as nn
import numpy as np
import numpy.typing as npt
from mlx_lm.models.activations import swiglu
from mlx_lm.models.base import (
    BaseModelArgs,
    create_attention_mask,
    create_ssm_mask,
    scaled_dot_product_attention,
)
from mlx_lm.models.cache import ArraysCache, KVCache, QuantizedKVCache
from mlx_lm.models.gated_delta import (
    gated_delta_kernel,
    gated_delta_update,
    normalize_qk,
)
from mlx_lm.models.qwen3_5 import GatedDeltaNet as Qwen3_5GatedDeltaNet
from mlx_lm.models.qwen3_5 import TextModelArgs as Qwen3_5TextModelArgs
from mlx_lm.models.qwen3_next import Qwen3NextMLP, Qwen3NextRMSNormGated
from mlx_lm.models.switch_layers import QuantizedSwitchLinear, SwitchGLU

from exo.worker.engines.mlx.models.qwen3 import qwen4_exp_kernels as kernels

IntArray = npt.NDArray[np.int64]
FloatArray = npt.NDArray[np.float32]
AttentionMask = mx.array | Literal["causal"] | None


@dataclass
class TextArgs(BaseModelArgs):
    model_type: str = "qwen4_exp_text"
    hidden_size: int = 2560
    num_hidden_layers: int = 48
    num_attention_heads: int = 24
    num_key_value_heads: int = 2
    head_dim: int = 256
    vocab_size: int = 248320
    rms_norm_eps: float = 1e-6
    layer_types: list[str] = field(default_factory=list[str])
    full_attention_interval: int = 4
    tie_word_embeddings: bool = False
    max_position_embeddings: int = 262144
    # MoE
    num_experts: int = 512
    num_experts_per_tok: int = 10
    moe_intermediate_size: int = 640
    shared_expert_intermediate_size: int = 640
    norm_topk_prob: bool = True
    # Gated DeltaNet
    linear_num_key_heads: int = 16
    linear_num_value_heads: int = 48
    linear_key_head_dim: int = 128
    linear_value_head_dim: int = 128
    linear_conv_kernel_dim: int = 4
    output_gate_type: str = "sigmoid"
    # Hyper-connections
    hc_count: int = 4
    hc_lowrank: int = 320
    # Qwen Sparse Attention
    indexer_n_heads: int = 4
    indexer_kv_heads: int = 1
    indexer_head_dim: int = 128
    indexer_budget: int = 2048
    indexer_compress_ratio: int = 4
    # Hashed n-gram embedding (PLE). ple_layer_ids are one-indexed.
    ngram_size: int = 3
    heads_per_ngram: int = 8
    ngram_vocab_size_base: int = 20_000_000
    make_ngram_vocab_size_divisible_by: int = 128
    ple_embed_dim: int | None = None
    ple_layer_ids: list[int] = field(default_factory=lambda: [2])
    ple_conv_kernel_size: int = 4
    # Not in the released config.json: the reference's default
    seed: int = 1234
    eos_token_id: int | list[int] = 248044
    # RoPE
    rope_parameters: dict[str, object] | None = None
    rope_theta: float = 10_000_000.0
    partial_rotary_factor: float = 0.25

    def __post_init__(self) -> None:
        rope = self.rope_parameters or {}
        theta = rope.get("rope_theta", self.rope_theta)
        partial = rope.get("partial_rotary_factor", self.partial_rotary_factor)
        assert isinstance(theta, (int, float)) and isinstance(partial, (int, float))
        self.rope_theta = float(theta)
        self.partial_rotary_factor = float(partial)
        rope_type = rope.get("rope_type", rope.get("type", "default"))
        if rope_type != "default":
            raise ValueError(f"qwen4_exp: unsupported rope_type {rope_type!r}")
        if self.ple_embed_dim is None:
            self.ple_embed_dim = self.hidden_size
        if not self.layer_types:
            interval = self.full_attention_interval
            self.layer_types = [
                "full_attention" if (i + 1) % interval == 0 else "linear_attention"
                for i in range(self.num_hidden_layers)
            ]
        # A truncated config keeps the first num_hidden_layers entries
        self.layer_types = self.layer_types[: self.num_hidden_layers]
        self.ple_layer_ids = sorted(set(self.ple_layer_ids))
        for layer_id in self.ple_layer_ids:
            if layer_id <= self.num_hidden_layers and (
                self.layer_types[layer_id - 1] != "linear_attention"
            ):
                raise ValueError(
                    "qwen4_exp: PLE layers must be linear-attention layers"
                )

    @classmethod
    def from_dict(cls, params: dict[str, object]) -> Self:
        return cast(Self, super().from_dict(params))  # pyright: ignore[reportUnknownMemberType]

    @property
    def eos_id(self) -> int:
        eos = self.eos_token_id
        return eos[0] if isinstance(eos, list) else eos

    @property
    def ple_dim(self) -> int:
        assert self.ple_embed_dim is not None
        return self.ple_embed_dim

    def gated_delta_net_args(self) -> Qwen3_5TextModelArgs:
        """The fields mlx-lm's Qwen3.5 GatedDeltaNet reads."""
        return Qwen3_5TextModelArgs.from_dict(
            {
                "model_type": "qwen3_5_text",
                "hidden_size": self.hidden_size,
                "intermediate_size": 0,
                "num_hidden_layers": self.num_hidden_layers,
                "num_attention_heads": self.num_attention_heads,
                "rms_norm_eps": self.rms_norm_eps,
                "vocab_size": self.vocab_size,
                "num_key_value_heads": self.num_key_value_heads,
                "max_position_embeddings": self.max_position_embeddings,
                "linear_num_value_heads": self.linear_num_value_heads,
                "linear_num_key_heads": self.linear_num_key_heads,
                "linear_key_head_dim": self.linear_key_head_dim,
                "linear_value_head_dim": self.linear_value_head_dim,
                "linear_conv_kernel_dim": self.linear_conv_kernel_dim,
                "head_dim": self.head_dim,
            }
        )


@dataclass
class ModelArgs(BaseModelArgs):
    model_type: str
    text_config: dict[str, object]

    @classmethod
    def from_dict(cls, params: dict[str, object]) -> Self:
        if "text_config" not in params:
            params = {**params, "text_config": params}
        return cast(Self, super().from_dict(params))  # pyright: ignore[reportUnknownMemberType]


# Steps of one token (exo's decode) take the fused kernels. With more rows, MLX's
# quantized matmul sums in another order (qmv_wide on M3 and later GPUs, then qmm),
# which the kernels' matrix-vector products don't reproduce: such steps take the
# plain ops.
_FUSED_TOKENS = 1


# ----------------------------------------------------------------------------- norms


class ZeroCenteredRMSNorm(nn.Module):
    """RMSNorm with a zero-centred gain: ``norm(x) * (1 + weight)``, in float32.

    With ``group_size`` the statistic is taken over each group of the last axis (one per
    hyper-connection stream); the gain spans the whole axis.
    """

    def __init__(self, dims: int, eps: float, group_size: int | None = None):
        super().__init__()
        self.weight = mx.zeros((dims,))
        self.eps = eps
        self.group_size = group_size

    def __call__(self, x: mx.array) -> mx.array:
        y = x.astype(mx.float32)
        if self.group_size is not None:
            y = y.reshape(*x.shape[:-1], -1, self.group_size)
        y = mx.fast.rms_norm(y, None, self.eps).reshape(*x.shape)
        return (y * (1.0 + self.weight.astype(mx.float32))).astype(x.dtype)


class SigmoidGatedRMSNorm(Qwen3NextRMSNormGated):
    """The Gated DeltaNet output norm with a sigmoid gate (Qwen3.5's is SiLU):
    ``norm(x) * weight * sigmoid(gate)``. The gain is conventional (not zero-centred)."""

    def __call__(
        self, hidden_states: mx.array, gate: mx.array | None = None
    ) -> mx.array:
        y = mx.fast.rms_norm(hidden_states, self.weight, self.eps)
        if gate is None:
            return y
        gated = mx.sigmoid(gate.astype(mx.float32)) * y.astype(mx.float32)
        return gated.astype(hidden_states.dtype)


# ------------------------------------------------------------------ hyper-connections


class GatedResidual(nn.Module):
    """Mixes the ``hc_count`` residual streams into one block input and, with
    ``use_combine``, gives the weights that add the block's output back to each stream."""

    def __init__(self, args: TextArgs, use_combine: bool = True):
        super().__init__()
        self.hc_count = args.hc_count
        self.hidden_size = args.hidden_size
        hc_dims = self.hc_count * self.hidden_size
        self.hc_norm = ZeroCenteredRMSNorm(
            hc_dims, args.rms_norm_eps, group_size=self.hidden_size
        )
        self.input_mix_weight_down = nn.Linear(hc_dims, args.hc_lowrank, bias=False)
        self.input_mix_weight_up = nn.Linear(args.hc_lowrank, hc_dims, bias=False)
        self.block_inject_weight: nn.Linear | None = (
            nn.Linear(hc_dims, self.hc_count, bias=False) if use_combine else None
        )
        self.projection: PackedLinear | None = None

    def mix(self, hyper: mx.array) -> tuple[mx.array, mx.array | None]:
        """(block input, injection weights or None) for the streams ``hyper``."""
        normed = self.hc_norm(hyper)
        weights = nn.silu(self.input_mix_weight_down(normed) / self.hc_count)
        weights = mx.sigmoid(self.input_mix_weight_up(weights))
        streams_shape = (*hyper.shape[:-1], self.hc_count, self.hidden_size)
        mixed = (weights.reshape(*streams_shape) * normed.reshape(*streams_shape)).mean(
            axis=-2
        )
        if self.block_inject_weight is None:
            return mixed, None
        inject = 2 * mx.sigmoid(self.block_inject_weight(normed) / self.hc_count)
        return mixed, inject

    # Packed path (see Model.pack_for_inference): the down and injection projections are
    # one matmul, and the norm, the injection and the stream update are fused kernels.

    def pack(self) -> bool:
        down, up = self.input_mix_weight_down, self.input_mix_weight_up
        parts = (
            [down]
            if self.block_inject_weight is None
            else [down, self.block_inject_weight]
        )
        if not (
            _packable(parts)
            and _affine(up)
            and cast(int, up.bits) == 4
            and self.hc_count == 4
            and kernels.norm_supported(self.hidden_size)
        ):
            return False
        self.projection = PackedLinear(parts)
        del self["input_mix_weight_down"]
        if self.block_inject_weight is not None:
            del self["block_inject_weight"]
        return True

    def packed_norm(self, hyper: mx.array) -> mx.array:
        return kernels.hc_norm(
            hyper, self.hc_norm.weight, self.hc_norm.eps, self.hc_count
        )

    def packed_mix(self, normed: mx.array) -> tuple[mx.array, mx.array, int]:
        """(block input, injection logits, their first column) for the normed
        streams: the logits are a column range of a larger array at decode."""
        assert self.projection is not None
        up = self.input_mix_weight_up
        low_rank = up.weight.shape[1] * 32 // cast(int, up.bits)
        if normed.size // normed.shape[-1] <= _FUSED_TOKENS:
            projection = self.projection(normed)
            mixed = kernels.hc_mix_up(
                projection,
                up.weight,
                cast(mx.array, up.scales),
                cast(mx.array, up.biases),
                cast(int, up.group_size),
                cast(int, up.bits),
                normed,
                self.hc_count,
                low_rank,
            )
            return mixed, projection, low_rank
        outputs = self.projection.outputs(normed)
        weights = nn.silu(outputs[0] / self.hc_count)
        weights = mx.sigmoid(up(weights))
        streams_shape = (*normed.shape[:-1], self.hc_count, self.hidden_size)
        mixed = (weights.reshape(*streams_shape) * normed.reshape(*streams_shape)).mean(
            axis=-2
        )
        return mixed, outputs[-1], 0


def add_to_streams(hyper: mx.array, block_out: mx.array, inject: mx.array) -> mx.array:
    """streams + block output scaled per stream by the injection weights."""
    injection = block_out[..., None, :] * inject[..., None]
    return hyper + injection.reshape(*hyper.shape)


def _affine(module: nn.Module) -> bool:
    """An affine-quantised linear layer with scales and biases (exo's tensor-parallel
    linears included)."""
    return (
        "weight" in module
        and "scales" in module
        and "biases" in module
        and "bias" not in module
        and getattr(module, "mode", None) == "affine"
        and isinstance(getattr(module, "bits", None), int)
        and isinstance(getattr(module, "group_size", None), int)
    )


def _packable(parts: Sequence[nn.Module]) -> bool:
    """Affine-quantised layers with one bit width and group size between them."""
    return (
        all(_affine(part) for part in parts)
        and len({cast(int, part.bits) for part in parts}) == 1
        and len({cast(int, part.group_size) for part in parts}) == 1
    )


class PackedLinear(nn.Module):
    """Affine-quantised linear layers that share their input, as one matmul: their
    weights stacked along the output dimension. Each output column is computed as the
    separate layer computes it."""

    sizes: tuple[int, ...]

    def __init__(self, parts: Sequence[nn.Module]):
        super().__init__()
        bits = {cast(int, part.bits) for part in parts}
        group_sizes = {cast(int, part.group_size) for part in parts}
        assert len(bits) == 1 and len(group_sizes) == 1
        self.bits = bits.pop()
        self.group_size = group_sizes.pop()
        arrays = [cast(dict[str, mx.array], part) for part in parts]
        self.weight = mx.concatenate([a["weight"] for a in arrays])
        self.scales = mx.concatenate([a["scales"] for a in arrays])
        self.biases = mx.concatenate([a["biases"] for a in arrays])
        # (a tuple attribute would join the parameter tree)
        object.__setattr__(
            self, "sizes", tuple(int(a["weight"].shape[0]) for a in arrays)
        )
        mx.eval(self.weight, self.scales, self.biases)

    def __call__(self, x: mx.array) -> mx.array:
        """All the layers' outputs, concatenated."""
        return self._matmul(x, self.weight, self.scales, self.biases)

    def _matmul(
        self, x: mx.array, weight: mx.array, scales: mx.array, biases: mx.array
    ) -> mx.array:
        return mx.quantized_matmul(
            x,
            weight,
            scales,
            biases,
            transpose=True,
            group_size=self.group_size,
            bits=self.bits,
        )

    def split(self, y: mx.array) -> list[mx.array]:
        outputs: list[mx.array] = []
        start = 0
        for size in self.sizes:
            outputs.append(y[..., start : start + size])
            start += size
        return outputs

    def outputs(self, x: mx.array) -> list[mx.array]:
        """Each layer's output. For a few tokens one matmul does them all; for more,
        MLX's matrix kernel can round an output differently with a different number
        of columns, so each layer is a matmul of its own (on row slices of the
        packed weights, which are views)."""
        if x.size // x.shape[-1] <= _FUSED_TOKENS:
            return self.split(self(x))
        outputs: list[mx.array] = []
        start = 0
        for size in self.sizes:
            end = start + size
            outputs.append(
                self._matmul(
                    x,
                    self.weight[start:end],
                    self.scales[start:end],
                    self.biases[start:end],
                )
            )
            start = end
        return outputs


# -------------------------------------------------------------- hashed n-gram (PLE)

_MASK64 = (1 << 64) - 1
_SPLITMIX_GAMMA = 0x9E3779B97F4A7C15
_SPLITMIX_M1 = 0xBF58476D1CE4E5B9
_SPLITMIX_M2 = 0x94D049BB133111EB
_PRIME_1 = 10007


def _splitmix64(value: int) -> int:
    value = (value + _SPLITMIX_GAMMA) & _MASK64
    value = ((value ^ (value >> 30)) * _SPLITMIX_M1) & _MASK64
    value = ((value ^ (value >> 27)) * _SPLITMIX_M2) & _MASK64
    return (value ^ (value >> 31)) & _MASK64


def layer_multipliers(
    vocab_size: int, ngram_size: int, ple_layer_index: int, seed: int
) -> list[int]:
    """The odd hash multipliers of one PLE layer, derived as the reference does."""
    half_bound = max(1, (((1 << 63) - 1) // max(vocab_size, 1)) // 2)
    base_seed = seed + _PRIME_1 * ple_layer_index
    return [
        2
        * (_splitmix64((base_seed + _SPLITMIX_GAMMA * (i + 1)) & _MASK64) % half_bound)
        + 1
        for i in range(ngram_size)
    ]


def _is_prime(value: int) -> bool:
    if value < 2:
        return False
    if value % 2 == 0:
        return value == 2
    return all(value % d for d in range(3, math.isqrt(value) + 1, 2))


def _nth_prime_after(start: int, count: int) -> int:
    prime = start
    for _ in range(count):
        prime += 1
        while not _is_prime(prime):
            prime += 1
    return prime


class NGramHasher:
    """Row ids of the n-gram table for a token history, computed on the host as the
    reference computes them (int64 arithmetic that wraps, history reset at EOS)."""

    def __init__(self, args: TextArgs, ple_layer_index: int):
        self.ngram_size = args.ngram_size
        self.context_len = args.ngram_size - 1
        self.heads_per_ngram = args.heads_per_ngram
        self.ngram_heads = self.context_len * args.heads_per_ngram
        self.eos = args.eos_id
        sizes: list[int] = []
        offsets: list[int] = []
        total = 0
        for head in range(self.ngram_heads):
            global_head = ple_layer_index * self.ngram_heads + head
            size = _nth_prime_after(args.ngram_vocab_size_base - 1, global_head + 1)
            sizes.append(size)
            offsets.append(total)
            total += size
        divisor = args.make_ngram_vocab_size_divisible_by
        self.rows = -(-total // divisor) * divisor
        self.sizes: IntArray = np.array(sizes, dtype=np.int64)
        self.offsets: IntArray = np.array(offsets, dtype=np.int64)
        self.multipliers = layer_multipliers(
            args.vocab_size, args.ngram_size, ple_layer_index, args.seed
        )

    def check_buffers(
        self,
        multipliers: Sequence[int],
        vocab_sizes: Sequence[int],
        offsets: Sequence[int],
    ) -> None:
        """Raise if the checkpoint's stored hash buffers differ from the derived ones (a
        wrong seed or table layout gives plausible but wrong embeddings)."""
        if (
            list(multipliers) != self.multipliers
            or list(vocab_sizes) != self.sizes.tolist()
            or list(offsets) != self.offsets.tolist()
        ):
            raise ValueError(
                "qwen4_exp: the checkpoint's n-gram hash buffers do not match the ones "
                "derived from the config (seed, vocabulary sizes)"
            )

    def _shift(self, ids: IntArray, shift: int) -> IntArray:
        """Shift right by ``shift`` without reaching back past an EOS."""
        if shift == 0:
            return ids
        batch, length = np.size(ids, 0), np.size(ids, 1)
        positions = np.arange(length, dtype=np.int64)
        eos_positions = np.where(np.equal(ids, self.eos), positions[None], -1)
        previous_eos = np.concatenate(
            [
                np.full((batch, 1), -1, dtype=np.int64),
                np.maximum.accumulate(eos_positions, axis=1)[:, :-1],
            ],
            axis=1,
        )
        in_segment = positions[None] - (previous_eos + 1)
        source = positions - shift
        gather = np.broadcast_to(np.maximum(source, 0)[None], (batch, length))
        shifted = np.take_along_axis(ids, gather, axis=1)
        valid = (in_segment >= shift) & (source[None] >= 0)
        return np.where(valid, shifted, np.int64(self.eos))

    def __call__(self, history: IntArray) -> IntArray:
        """(B, T) token history -> (B, T, ngram_heads) row ids."""
        shifted = [self._shift(history, s) for s in range(self.ngram_size)]
        blocks: list[IntArray] = []
        with np.errstate(over="ignore"):
            for ngram in range(2, self.ngram_size + 1):
                lo = (ngram - 2) * self.heads_per_ngram
                hi = lo + self.heads_per_ngram
                mixed: IntArray = shifted[0] * np.int64(self.multipliers[0])
                for position in range(1, ngram):
                    term: IntArray = shifted[position] * np.int64(
                        self.multipliers[position]
                    )
                    mixed = np.bitwise_xor(mixed, term)
                row_ids: IntArray = (
                    np.expand_dims(mixed, -1) % self.sizes[lo:hi] + self.offsets[lo:hi]
                )
                blocks.append(row_ids)
        return np.concatenate(blocks, axis=-1)


_SHARD_KEY = re.compile(
    r"^(?P<prefix>.*\.ple\.ple_embedding\.ngram_embedding)\.(?:shards\.|shard_)"
    r"(?P<shard>\d+)\.(?P<name>weight|scales|biases)$"
)
_FLOAT_DTYPES: dict[str, np.dtype[np.generic]] = {
    "BF16": np.dtype("<u2"),
    "F16": np.dtype("<f2"),
    "F32": np.dtype("<f4"),
}


def _to_float32(raw: npt.NDArray[np.generic], dtype: str) -> FloatArray:
    if dtype == "BF16":
        bits = raw.astype(np.uint32) << np.uint32(16)
        return bits.view(np.float32)
    return raw.astype(np.float32)


@dataclass(frozen=True)
class _TableShard:
    weight: npt.NDArray[np.generic]
    weight_dtype: str
    scales: npt.NDArray[np.generic] | None
    biases: npt.NDArray[np.generic] | None
    scale_dtype: str


class NGramTable:
    """The n-gram table, left in the checkpoint's safetensors files and read row by row
    through memory maps. Takes 2/4/8-bit affine shards (with scales and biases) or
    unquantised BF16/F16/F32 shards."""

    def __init__(self, model_path: Path, prefix: str, width: int):
        self.width = width
        index = cast(
            dict[str, object],
            json.loads((model_path / "model.safetensors.index.json").read_text()),
        )
        weight_map = cast(dict[str, str], index["weight_map"])
        found: dict[int, dict[str, str]] = {}
        for key in weight_map:
            match = _SHARD_KEY.match(key)
            if match is not None and match["prefix"] == prefix:
                found.setdefault(int(match["shard"]), {})[match["name"]] = key
        if not found or sorted(found) != list(range(len(found))):
            raise ValueError(f"qwen4_exp: no complete n-gram table under {prefix}")
        if f"{prefix}.weight_scale" in weight_map:
            raise ValueError(
                "qwen4_exp: this n-gram table layout (with a weight_scale) is not supported"
            )
        headers: dict[str, tuple[int, dict[str, dict[str, object]]]] = {}
        # One read-only map per file; the tensors are numpy views into it
        self._maps: dict[str, mmap.mmap] = {}
        # (map, byte offset of the tensor, bytes per row) for each view, for read-ahead
        self._layout: dict[int, tuple[mmap.mmap, int, int]] = {}

        def memmap(key: str) -> tuple[npt.NDArray[np.generic], str]:
            file_name = weight_map[key]
            if file_name not in headers:
                with open(model_path / file_name, "rb") as fh:
                    length = cast(int, struct.unpack("<Q", fh.read(8))[0])
                    header = cast(
                        dict[str, dict[str, object]], json.loads(fh.read(length))
                    )
                    self._maps[file_name] = mmap.mmap(
                        fh.fileno(), 0, access=mmap.ACCESS_READ
                    )
                headers[file_name] = (8 + length, header)
            base, header = headers[file_name]
            info = header[key]
            dtype_name = cast(str, info["dtype"])
            if dtype_name == "U32":
                dtype = np.dtype("<u4")
            elif dtype_name in _FLOAT_DTYPES:
                dtype = _FLOAT_DTYPES[dtype_name]
            else:
                raise ValueError(
                    f"qwen4_exp: n-gram tensor {key} has dtype {dtype_name}"
                )
            start = cast(list[int], info["data_offsets"])[0]
            shape = tuple(cast(list[int], info["shape"]))
            mapped = self._maps[file_name]
            count = math.prod(shape)
            array = np.frombuffer(
                mapped, dtype=dtype, count=count, offset=base + start
            ).reshape(shape)
            row_bytes = count * dtype.itemsize // shape[0]
            self._layout[id(array)] = (mapped, base + start, row_bytes)
            return array, dtype_name

        self.shards: list[_TableShard] = []
        starts = [0]
        for shard in range(len(found)):
            names = found[shard]
            weight, weight_dtype = memmap(names["weight"])
            scales = biases = None
            scale_dtype = ""
            if weight_dtype == "U32":
                if "scales" not in names or "biases" not in names:
                    raise ValueError("qwen4_exp: quantised n-gram shards need biases")
                scales, scale_dtype = memmap(names["scales"])
                biases, _ = memmap(names["biases"])
            self.shards.append(
                _TableShard(weight, weight_dtype, scales, biases, scale_dtype)
            )
            starts.append(starts[-1] + int(np.size(weight, 0)))
        self.starts: IntArray = np.array(starts, dtype=np.int64)
        self.rows = starts[-1]
        first = self.shards[0]
        if first.weight_dtype == "U32":
            assert first.scales is not None
            self.bits = int(np.size(first.weight, 1)) * 32 // width
            self.group_size = width // int(np.size(first.scales, 1))
            if self.bits not in (2, 4, 8) or width % self.group_size:
                raise ValueError(
                    f"qwen4_exp: unsupported n-gram quantisation ({self.bits} bits)"
                )
        else:
            self.bits = 0
            self.group_size = 0
            if np.size(first.weight, 1) != width:
                raise ValueError("qwen4_exp: the n-gram table width does not match")
        self.files = sorted(
            {
                model_path / weight_map[key]
                for names in found.values()
                for key in names.values()
            }
        )

    def prewarm(self) -> None:
        """Read the table's files once so that row reads hit the page cache (a cold row
        is a random disk read)."""
        buffer = bytearray(64 << 20)
        for file in self.files:
            with open(file, "rb", buffering=0) as fh:
                while fh.readinto(buffer):
                    pass

    def _dequantize(self, shard: _TableShard, rows: IntArray) -> FloatArray:
        weight = np.asarray(shard.weight[rows])
        if shard.weight_dtype != "U32":
            return _to_float32(weight, shard.weight_dtype)
        assert shard.scales is not None and shard.biases is not None
        per_word = 32 // self.bits
        shifts = (self.bits * np.arange(per_word, dtype=np.uint32))[None, None]
        mask = np.uint32((1 << self.bits) - 1)
        values = (weight.astype(np.uint32)[..., None] >> shifts) & mask
        values = values.reshape(len(rows), -1, self.group_size).astype(np.float32)
        scales = _to_float32(np.asarray(shard.scales[rows]), shard.scale_dtype)
        biases = _to_float32(np.asarray(shard.biases[rows]), shard.scale_dtype)
        return (values * scales[..., None] + biases[..., None]).reshape(len(rows), -1)

    def _read_ahead(self, shard: _TableShard, rows: IntArray) -> None:
        """Ask the OS to start reading the pages of ``rows`` of every tensor of the shard
        at once. Row reads that miss the page cache are random disk reads; issued one at a
        time by the copy below they take ~0.2 ms each (a cold token is 48 of them), issued
        together they overlap."""
        for array in (shard.weight, shard.scales, shard.biases):
            if array is None:
                continue
            mapped, offset, row_bytes = self._layout[id(array)]
            pages = np.unique((offset + rows * row_bytes) // mmap.PAGESIZE)
            for page in cast(list[int], pages.tolist()):
                mapped.madvise(mmap.MADV_WILLNEED, page * mmap.PAGESIZE, mmap.PAGESIZE)

    def __call__(self, ids: IntArray) -> FloatArray:
        """Rows (n, width) in float32 of the row ids ``ids`` (n,)."""
        unique, inverse = np.unique(ids, return_inverse=True)
        out = np.empty((len(unique), self.width), dtype=np.float32)
        shard_of = np.searchsorted(self.starts, unique, side="right") - 1
        starts = cast(list[int], self.starts.tolist())
        shards = cast(list[int], np.unique(shard_of).tolist())
        for shard in shards:
            ahead = np.flatnonzero(np.equal(shard_of, shard))
            rows: IntArray = np.take(unique, ahead) - np.int64(starts[shard])
            self._read_ahead(self.shards[shard], rows)
        for shard in shards:
            selected = np.flatnonzero(np.equal(shard_of, shard))
            local: IntArray = np.take(unique, selected) - np.int64(starts[shard])
            out[selected] = self._dequantize(self.shards[shard], local)
        return out[inverse]


class NGramEmbedding(nn.Module):
    """The hashed n-gram embedding: row ids from the hasher, rows from the table."""

    def __init__(self, args: TextArgs, ple_layer_index: int):
        super().__init__()
        self.ple_layer_index = ple_layer_index
        self.hasher = NGramHasher(args, ple_layer_index)
        self.width = args.ple_dim // self.hasher.ngram_heads
        self._table: NGramTable | None = None

    @property
    def table(self) -> NGramTable | None:
        return self._table

    def attach(self, table: NGramTable) -> None:
        if table.rows != self.hasher.rows or table.width != self.width:
            raise ValueError(
                f"qwen4_exp: the n-gram table has {table.rows} rows of width "
                f"{table.width}, the config gives {self.hasher.rows} of {self.width}"
            )
        self._table = table

    def rows(self, ids: IntArray, dtype: mx.Dtype) -> mx.array:
        """Embeddings (..., ple_embed_dim) of row ids (..., ngram_heads)."""
        if self._table is None:
            raise RuntimeError(
                "qwen4_exp: the n-gram table is not attached "
                "(Model.attach_ngram_table after loading)"
            )
        rows = self._table(ids.reshape(-1))
        shape = [np.size(ids, axis) for axis in range(ids.ndim)]
        shape[-1] *= self.width
        return mx.array(rows).reshape(*shape).astype(dtype)


# The PLE layer's cache entries, after the Gated DeltaNet's (0: conv state, 1: state)
_PLE_CONV = 2
_PLE_CONTEXT = 3


class PLELayer(nn.Module):
    """Adds the hashed n-gram features of each token to the residual streams."""

    def __init__(self, args: TextArgs, ple_layer_index: int):
        super().__init__()
        self.hidden_size = args.hidden_size
        self.hc_count = args.hc_count
        hc_dims = self.hidden_size * self.hc_count
        self.ple_embedding = NGramEmbedding(args, ple_layer_index)
        self.state_len = (args.ple_conv_kernel_size - 1) * args.ngram_size
        self.key_proj = nn.Linear(args.ple_dim, hc_dims, bias=False)
        self.value_proj = nn.Linear(args.ple_dim, self.hidden_size, bias=False)
        eps = args.rms_norm_eps
        self.norm_key = ZeroCenteredRMSNorm(hc_dims, eps, group_size=self.hidden_size)
        self.norm_query = ZeroCenteredRMSNorm(hc_dims, eps, group_size=self.hidden_size)
        self.norm_conv = ZeroCenteredRMSNorm(hc_dims, eps, group_size=self.hidden_size)
        self.conv1d = nn.Conv1d(
            hc_dims,
            hc_dims,
            kernel_size=args.ple_conv_kernel_size,
            dilation=args.ngram_size,
            groups=hc_dims,
            bias=False,
        )

    def _ngram_ids(self, ids: IntArray, cache: ArraysCache | None) -> IntArray:
        """Row ids (B, L, heads) of the tokens ``ids`` (B, L), after the cached context."""
        hasher = self.ple_embedding.hasher
        batch = int(np.size(ids, 0))
        context = (
            cast(mx.array | None, cache[_PLE_CONTEXT]) if cache is not None else None
        )
        if context is not None:
            previous = np.array(context, dtype=np.int64)
        else:
            previous = np.full((batch, hasher.context_len), hasher.eos, dtype=np.int64)
        history = np.concatenate([previous, ids], axis=1)
        if cache is not None:
            cache[_PLE_CONTEXT] = mx.array(history[:, -hasher.context_len :])
        return hasher(history)[:, -ids.shape[1] :]

    def embedding(
        self, ids: IntArray, cache: ArraysCache | None, dtype: mx.Dtype
    ) -> mx.array:
        """The n-gram embeddings (B, L, ple_embed_dim) of the tokens ``ids`` (B, L)."""
        return self.ple_embedding.rows(self._ngram_ids(ids, cache), dtype)

    def deferred_embedding(
        self, inputs: mx.array, cache: ArraysCache | None, dtype: mx.Dtype
    ) -> tuple[mx.array, Callable[[], None]]:
        """The n-gram embeddings of the (not yet computed) tokens ``inputs`` (B, L), and
        the function that reads them. The embeddings are a host buffer the graph reads:
        build the rest of the graph first, then call the function (it waits for the
        tokens and fills the buffer) before evaluating anything that depends on them.
        At decode this keeps building the next step's graph off the critical path."""
        batch, length = inputs.shape[0], inputs.shape[-1]
        hasher = self.ple_embedding.hasher
        width = self.ple_embedding.width
        buffer = mx.array(
            np.zeros((batch * length * hasher.ngram_heads, width), dtype=np.float32)
        )
        embedding = buffer.reshape(batch, length, hasher.ngram_heads * width).astype(
            dtype
        )

        def fill() -> None:
            ids = np.array(inputs, dtype=np.int64).reshape(batch, length)
            row_ids = self._ngram_ids(ids, cache)
            table = self.ple_embedding.table
            if table is None:
                raise RuntimeError("qwen4_exp: the n-gram table is not attached")
            # A view of the buffer (NumPy 2 raises rather than copy with copy=False)
            view = cast(FloatArray, np.array(buffer, copy=False))
            view[...] = table(row_ids.reshape(-1))

        return embedding, fill

    def _short_conv(self, x: mx.array, cache: ArraysCache | None) -> mx.array:
        state = cast(mx.array | None, cache[_PLE_CONV]) if cache is not None else None
        if state is None:
            state = mx.zeros((x.shape[0], self.state_len, x.shape[-1]), dtype=x.dtype)
        conv_input = mx.concatenate([state, x], axis=1)
        if cache is not None:
            cache[_PLE_CONV] = mx.contiguous(conv_input[:, -self.state_len :])
        return nn.silu(self.conv1d(conv_input))

    def __call__(
        self, hyper: mx.array, embedding: mx.array, cache: ArraysCache | None
    ) -> mx.array:
        """``embedding``: the tokens' n-gram embeddings (``embedding`` or
        ``deferred_embedding``)."""
        batch, length, _ = hyper.shape
        streams = (batch, length, self.hc_count, self.hidden_size)
        key = self.norm_key(self.key_proj(embedding)).reshape(*streams)
        value = self.value_proj(embedding)
        query = self.norm_query(hyper).reshape(*streams)
        gate = (key * query).sum(axis=-1, keepdims=True) / math.sqrt(self.hidden_size)
        gate = mx.sqrt(mx.maximum(mx.abs(gate), 1e-6)) * mx.sign(gate)
        gated = (mx.sigmoid(gate) * value[..., None, :]).reshape(*hyper.shape)
        return gated + self._short_conv(self.norm_conv(gated), cache)


# -------------------------------------------------------------------------- attention


class QSAKVCache(KVCache):
    """A KVCache that also keeps the indexer's raw keys (B, L, index_head_dim)."""

    def __init__(self) -> None:
        super().__init__()
        self.index_keys: mx.array | None = None

    def update_index(self, keys: mx.array) -> mx.array:
        """Store ``keys`` (B, S, D) at the current offset (the attention keys of the same
        tokens are stored after) and return the keys of every token so far."""
        previous, length = self.offset, keys.shape[1]
        step = self.step
        if self.index_keys is None or previous + length > self.index_keys.shape[1]:
            batch, _, dims = keys.shape
            grow = ((step + length - 1) // step) * step
            new = mx.zeros((batch, grow, dims), keys.dtype)
            if self.index_keys is not None:
                kept = self.index_keys[:, :previous]
                new = mx.concatenate([kept, new], axis=1)
            self.index_keys = new
        self.index_keys[:, previous : previous + length] = keys
        return self.index_keys[:, : previous + length]

    @property
    def nbytes(self) -> int:
        extra = 0 if self.index_keys is None else self.index_keys.nbytes
        return super().nbytes + extra

    def to_quantized(self, group_size: int = 64, bits: int = 4) -> QuantizedKVCache:
        # A quantized cache would drop the indexer's keys and silently attend densely
        raise ValueError("Qwen Sparse Attention does not support a quantized KV cache")


class QSAIndexer(nn.Module):
    """Picks, per query, ``indexer_budget / compress_ratio`` blocks of
    ``compress_ratio`` keys. Returns the boolean mask of the keys each query attends to,
    or None while every query sees no more complete blocks than that (dense causal
    attention is then exact)."""

    def __init__(self, args: TextArgs, rotary_dims: int):
        super().__init__()
        self.n_heads = args.indexer_n_heads
        self.head_dim = args.indexer_head_dim
        self.ratio = args.indexer_compress_ratio
        self.block_topk = args.indexer_budget // self.ratio
        self.rotary_dims = rotary_dims
        self.rope_theta = args.rope_theta
        self.index_qk_proj = nn.Linear(
            args.hidden_size,
            (self.n_heads + args.indexer_kv_heads) * self.head_dim,
            bias=False,
        )
        self.q_layernorm = ZeroCenteredRMSNorm(self.head_dim, args.rms_norm_eps)
        self.k_layernorm = ZeroCenteredRMSNorm(self.head_dim, args.rms_norm_eps)

    def _rope(self, x: mx.array, offset: int = 0, scale: float = 1.0) -> mx.array:
        return mx.fast.rope(
            x,
            self.rotary_dims,
            traditional=False,
            base=self.rope_theta,
            scale=scale,
            offset=offset,
        )

    def select(
        self, x: mx.array, cache: QSAKVCache | None, qk: mx.array | None = None
    ) -> mx.array | None:
        """The keys mask (None: all keys, causally); ``qk`` is ``index_qk_proj(x)``
        when already computed. Updates the cache's indexer keys."""
        batch, length, _ = x.shape
        if qk is None:
            qk = self.index_qk_proj(x)
        split = self.n_heads * self.head_dim
        queries = qk[..., :split].reshape(batch, length, self.n_heads, self.head_dim)
        raw_keys = qk[..., split:].reshape(batch, length, self.head_dim)
        offset = 0
        if cache is not None:
            offset = cache.offset
            raw_keys = cache.update_index(raw_keys)
        kv_len = raw_keys.shape[1]
        n_blocks = kv_len // self.ratio
        if n_blocks <= self.block_topk:
            return None

        # Blocks of `ratio` keys, mean-pooled in float32, normed, and rotated to the
        # position of their first token (block i starts at i * ratio).
        pooled = raw_keys[:, : n_blocks * self.ratio].reshape(
            batch, n_blocks, self.ratio, self.head_dim
        )
        pooled = pooled.astype(mx.float32).mean(axis=2).astype(raw_keys.dtype)
        pooled = self._rope(self.k_layernorm(pooled)[:, None], scale=float(self.ratio))
        queries = self.q_layernorm(queries).transpose(0, 2, 1, 3)
        queries = self._rope(queries, offset=offset)
        scores = queries.astype(mx.float32) @ pooled.astype(mx.float32).swapaxes(-1, -2)
        scores = mx.maximum(scores, 0).sum(axis=1) / math.sqrt(self.head_dim)

        # A query can pick the complete blocks that end at or before it
        positions = offset + mx.arange(length)
        complete = (positions + 1) // self.ratio
        visible = mx.arange(n_blocks)[None, None] < complete[None, :, None]
        visible = mx.broadcast_to(visible, scores.shape)
        scores = mx.where(visible, scores, -mx.inf)
        top = mx.argpartition(-scores, self.block_topk - 1, axis=-1)[
            ..., : self.block_topk
        ]
        picked = mx.take_along_axis(visible, top, axis=-1)
        # Unpicked slots scatter to an extra column that is dropped
        keep = mx.put_along_axis(
            mx.zeros((batch, length, n_blocks + 1), dtype=mx.bool_),
            mx.where(picked, top, n_blocks),
            mx.array(True),
            axis=-1,
        )[..., :n_blocks]
        keep = mx.repeat(keep, self.ratio, axis=-1)
        rest = kv_len - n_blocks * self.ratio
        if rest:
            keep = mx.concatenate(
                [keep, mx.zeros((batch, length, rest), dtype=mx.bool_)], axis=-1
            )
        # The query's own incomplete block (up to and including itself) is always kept
        tokens = mx.arange(kv_len)
        own = (tokens[None] >= (complete * self.ratio)[:, None]) & (
            tokens[None] <= positions[:, None]
        )
        return (keep | own[None])[:, None]


class Attention(nn.Module):
    """Gated full attention (as Qwen3.5's) with the Qwen Sparse Attention indexer."""

    def __init__(self, args: TextArgs):
        super().__init__()
        self.n_heads = args.num_attention_heads
        self.n_kv_heads = args.num_key_value_heads
        self.head_dim = args.head_dim
        self.scale = 1.0 / math.sqrt(self.head_dim)
        self.rotary_dims = int(self.head_dim * args.partial_rotary_factor)
        self.rope_theta = args.rope_theta
        dims = args.hidden_size
        # The query projection carries each head's output gate after its query
        self.q_proj = nn.Linear(dims, 2 * self.n_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(dims, self.n_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(dims, self.n_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.n_heads * self.head_dim, dims, bias=False)
        self.q_norm = ZeroCenteredRMSNorm(self.head_dim, args.rms_norm_eps)
        self.k_norm = ZeroCenteredRMSNorm(self.head_dim, args.rms_norm_eps)
        self.indexer = QSAIndexer(args, self.rotary_dims)
        self.projection: PackedLinear | None = None

    def _rope(self, x: mx.array, offset: int) -> mx.array:
        return mx.fast.rope(
            x,
            self.rotary_dims,
            traditional=False,
            base=self.rope_theta,
            scale=1.0,
            offset=offset,
        )

    # Packed path (see Model.pack_for_inference): q, k, v and the indexer's projection
    # are one matmul.

    def pack(self) -> bool:
        parts = [self.q_proj, self.k_proj, self.v_proj, self.indexer.index_qk_proj]
        if not _packable(parts):
            return False
        self.projection = PackedLinear(parts)
        for name in ("q_proj", "k_proj", "v_proj"):
            del self[name]
        del self.indexer["index_qk_proj"]
        return True

    def __call__(
        self, x: mx.array, mask: AttentionMask, cache: QSAKVCache | None
    ) -> mx.array:
        batch, length, _ = x.shape
        if self.projection is not None:
            q_out, k_out, v_out, index_qk = self.projection.outputs(x)
            selected = self.indexer.select(x, cache, index_qk)
        else:
            q_out, k_out, v_out = self.q_proj(x), self.k_proj(x), self.v_proj(x)
            selected = self.indexer.select(x, cache)
        offset = cache.offset if cache is not None else 0

        packed = self.projection is not None and kernels.norm_supported(self.head_dim)
        queries, gate = mx.split(
            q_out.reshape(batch, length, self.n_heads, -1), 2, axis=-1
        )
        gate = gate.reshape(batch, length, -1)
        if packed:
            # The same norms as one kernel each, straight from the projection
            queries = kernels.head_norm(
                q_out,
                self.q_norm.weight,
                self.q_norm.eps,
                self.n_heads,
                2 * self.head_dim,
            ).transpose(0, 2, 1, 3)
            keys = kernels.head_norm(
                k_out,
                self.k_norm.weight,
                self.k_norm.eps,
                self.n_kv_heads,
                self.head_dim,
            ).transpose(0, 2, 1, 3)
        else:
            queries = self.q_norm(queries).transpose(0, 2, 1, 3)
            keys = self.k_norm(
                k_out.reshape(batch, length, self.n_kv_heads, -1)
            ).transpose(0, 2, 1, 3)
        values = v_out.reshape(batch, length, self.n_kv_heads, -1).transpose(0, 2, 1, 3)
        queries = self._rope(queries, offset)
        keys = self._rope(keys, offset)
        if cache is not None:
            keys, values = cache.update_and_fetch(keys, values)

        attention_mask: mx.array | None
        if selected is not None:
            # The selection is causal by construction
            attention_mask = (
                selected if not isinstance(mask, mx.array) else mask & selected
            )
        elif isinstance(mask, str):
            attention_mask = cast(mx.array, cast(object, mask))
        else:
            attention_mask = mask
        out = scaled_dot_product_attention(
            queries, keys, values, cache=cache, scale=self.scale, mask=attention_mask
        )
        if packed:
            return self.o_proj(kernels.gate_heads(out, q_out))
        out = out.transpose(0, 2, 1, 3).reshape(batch, length, -1)
        return self.o_proj(out * mx.sigmoid(gate))


# ------------------------------------------------------------------- Gated DeltaNet


def _training(module: nn.Module) -> bool:
    return cast(bool, module.training)


class GatedDeltaNet(Qwen3_5GatedDeltaNet):
    """mlx-lm's Qwen3.5 Gated DeltaNet with a sigmoid output gate."""

    def __init__(self, args: TextArgs):
        super().__init__(args.gated_delta_net_args())
        if args.output_gate_type != "sigmoid":
            raise ValueError(
                f"qwen4_exp: unsupported output_gate_type {args.output_gate_type!r}"
            )
        self.norm = SigmoidGatedRMSNorm(self.head_v_dim, args.rms_norm_eps)
        self.in_proj: PackedLinear | None = None

    # Packed path (see Model.pack_for_inference): the four input projections are one
    # matmul; at decode the conv, SiLU, q/k norms, decay and beta are one kernel and the
    # gated output norm another.

    def pack(self) -> bool:
        parts = [self.in_proj_qkv, self.in_proj_z, self.in_proj_b, self.in_proj_a]
        if not _packable(cast(list[nn.Module], parts)):
            return False
        self.in_proj = PackedLinear(cast(list[nn.Module], parts))
        for name in ("in_proj_qkv", "in_proj_z", "in_proj_b", "in_proj_a"):
            del self[name]
        return True

    def __call__(
        self,
        inputs: mx.array,
        mask: mx.array | None = None,
        cache: ArraysCache | None = None,
    ) -> mx.array:
        if self.in_proj is None:
            return super().__call__(inputs, mask, cache)
        batch, length, _ = inputs.shape
        conv_dim = self.conv_dim
        kernel_size = self.conv_kernel_size
        if cache is not None and cache[0] is not None:
            conv_state = cast(mx.array, cache[0])
        else:
            conv_state = mx.zeros(
                (batch, kernel_size - 1, conv_dim), dtype=inputs.dtype
            )
        state = cast(mx.array | None, cache[1]) if cache is not None else None
        if state is None:
            state = mx.zeros(
                (batch, self.num_v_heads, self.head_v_dim, self.head_k_dim),
                dtype=mx.float32,
            )
        if (
            batch == 1
            and length <= _FUSED_TOKENS
            and not _training(self)
            and mask is None
            and (cache is None or cache.lengths is None)
            and self.head_k_dim == self.head_v_dim
        ):
            projection = self.in_proj(inputs)
            q, k, v, g, beta, conv_state = kernels.gdn_prepare(
                projection,
                conv_state,
                self.conv1d.weight.reshape(conv_dim, kernel_size),
                self.A_log,
                self.dt_bias,
                self.num_k_heads,
                self.num_v_heads,
                self.head_k_dim,
                self.head_v_dim,
            )
            out, state = gated_delta_kernel(q, k, v, g, beta, state)
            if cache is not None:
                cache[0] = conv_state
                cache[1] = state
                cache.advance(length)
            y = kernels.gdn_gated_norm(
                out, projection, conv_dim, self.norm.weight, self.norm.eps
            )
            return self.out_proj(y)

        # As mlx-lm's Qwen3.5 Gated DeltaNet, on the packed projection
        qkv, z, b, a = self.in_proj.outputs(inputs)
        z = z.reshape(batch, length, self.num_v_heads, self.head_v_dim)
        if mask is not None:
            qkv = mx.where(mask[..., None], qkv, 0)
        conv_input = mx.concatenate([conv_state, qkv], axis=1)
        if cache is not None:
            n_keep = kernel_size - 1
            if cache.lengths is not None:
                ends = mx.clip(cache.lengths, 0, length)
                positions = (ends[:, None] + mx.arange(n_keep))[..., None]
                cache[0] = mx.take_along_axis(conv_input, positions, axis=1)
            else:
                cache[0] = mx.contiguous(conv_input[:, -n_keep:, :])
        conv_out = nn.silu(self.conv1d(conv_input))
        q, k, v = [
            t.reshape(batch, length, h, d)
            for t, h, d in zip(
                mx.split(conv_out, [self.key_dim, 2 * self.key_dim], -1),
                [self.num_k_heads, self.num_k_heads, self.num_v_heads],
                [self.head_k_dim, self.head_k_dim, self.head_v_dim],
                strict=True,
            )
        ]
        q, k = normalize_qk(
            q, k, inv_scale=cast(float, self.head_k_dim**-0.5), eps=1e-6
        )
        out, state = gated_delta_update(
            q,
            k,
            v,
            a,
            b,
            self.A_log,
            self.dt_bias,
            state,
            mask,
            use_kernel=not _training(self),
        )
        if cache is not None:
            cache[1] = state
            cache.advance(length)
        out = self.norm(out, z)
        return self.out_proj(out.reshape(batch, length, -1))


# --------------------------------------------------------------------------------- MoE


class SparseMoeBlock(nn.Module):
    """Top-k routed experts plus a sigmoid-gated shared expert (Qwen3-Next's), with the
    router's softmax, top-k and renormalisation in float32 as the reference."""

    def __init__(self, args: TextArgs):
        super().__init__()
        dims = args.hidden_size
        self.top_k = args.num_experts_per_tok
        self.norm_topk_prob = args.norm_topk_prob
        self.gate = nn.Linear(dims, args.num_experts, bias=False)
        self.switch_mlp = SwitchGLU(dims, args.moe_intermediate_size, args.num_experts)
        self.shared_expert = Qwen3NextMLP(dims, args.shared_expert_intermediate_size)
        self.shared_expert_gate = nn.Linear(dims, 1, bias=False)
        self.router: PackedLinear | None = None
        self.experts: PackedExperts | None = None

    def __call__(self, x: mx.array) -> mx.array:
        if self.router is not None:
            return self._packed_call(x)
        logits = self.gate(x)
        probs = mx.softmax(logits.astype(mx.float32), axis=-1)
        k = self.top_k
        indices = mx.stop_gradient(mx.argpartition(-probs, kth=k - 1, axis=-1)[..., :k])
        scores = mx.take_along_axis(probs, indices, axis=-1)
        if self.norm_topk_prob:
            scores = scores / scores.sum(axis=-1, keepdims=True)
        scores = scores.astype(logits.dtype)
        routed = self.switch_mlp(x, indices)
        routed = (routed * scores[..., None]).sum(axis=-2)
        shared = mx.sigmoid(self.shared_expert_gate(x)) * self.shared_expert(x)
        return routed + shared

    # Packed path (see Model.pack_for_inference): the router and the shared expert's
    # gate are one matmul; each expert's gate and up projections are one, and the
    # shared expert is expert ``num_experts`` of the packed expert weights. At decode it
    # runs with the routed experts in one gather; longer inputs take it dense, as above.

    def pack(self) -> bool:
        switch = self.switch_mlp
        projections = (switch.gate_proj, switch.up_proj, switch.down_proj)
        shared = self.shared_expert
        if not (
            _packable([self.gate, self.shared_expert_gate])
            and all(_switch_affine(p) for p in projections)
            and _packable(
                [switch.gate_proj, switch.up_proj, shared.gate_proj, shared.up_proj]
            )
            and _packable([switch.down_proj, shared.down_proj])
            and kernels.router_supported(self.gate.weight.shape[0])
        ):
            return False
        self.router = PackedLinear([self.gate, self.shared_expert_gate])
        self.experts = PackedExperts(
            switch.gate_proj, switch.up_proj, switch.down_proj, shared
        )
        for name in ("gate", "shared_expert_gate", "switch_mlp", "shared_expert"):
            del self[name]
        return True

    def _packed_call(self, x: mx.array) -> mx.array:
        assert self.router is not None and self.experts is not None
        if x.size // x.shape[-1] <= _FUSED_TOKENS:
            logits = self.router(x)
            indices, weights, gate_up_rows = kernels.moe_route(
                logits, logits.shape[-1] - 1, self.top_k, self.norm_topk_prob
            )
            # The shared expert rides along as the last expert
            experts = self.experts
            if experts.vector_down:
                outputs = kernels.moe_down(
                    experts.gate_up(x, indices, gate_up_rows),
                    indices,
                    experts.down_weight,
                    experts.down_scales,
                    experts.down_biases,
                    experts.down_group_size,
                )
                return kernels.moe_combine(outputs, weights)
            return kernels.moe_combine(experts(x, indices), weights)
        # Longer inputs: the plain path's routing and sums (MLX's partition and
        # reductions order ties and terms their own way at these sizes)
        logits, shared_logit = self.router.outputs(x)
        probs = mx.softmax(logits.astype(mx.float32), axis=-1)
        k = self.top_k
        indices = mx.stop_gradient(mx.argpartition(-probs, kth=k - 1, axis=-1)[..., :k])
        scores = mx.take_along_axis(probs, indices, axis=-1)
        if self.norm_topk_prob:
            scores = scores / scores.sum(axis=-1, keepdims=True)
        scores = scores.astype(logits.dtype)
        routed = (self.experts(x, indices) * scores[..., None]).sum(axis=-2)
        return routed + mx.sigmoid(shared_logit) * self.experts.shared(x)


def _switch_affine(module: nn.Module) -> bool:
    return (
        isinstance(module, QuantizedSwitchLinear)
        and module.mode == "affine"
        and "biases" in module
        and "bias" not in module
    )


def _gather_sort(x: mx.array, indices: mx.array) -> tuple[mx.array, mx.array, mx.array]:
    # As mlx-lm's switch_layers: group the (token, expert) pairs by expert
    k = indices.shape[-1]
    flat = indices.flatten()
    order = mx.argsort(flat)
    inverse = mx.argsort(order)
    return x.flatten(0, -3)[order // k], flat[order], inverse


class PackedExperts(nn.Module):
    """The routed experts plus the shared expert (as the last expert), affine-quantised.
    The gate and up projections of all experts are one array: the experts' gate
    projections, then their up projections, so that one gather does both at decode and
    each is its own contiguous block for longer inputs."""

    _lhs_cache: dict[tuple[int, ...], tuple[mx.array, mx.array]]

    def __init__(
        self,
        gate_proj: nn.Module,
        up_proj: nn.Module,
        down_proj: nn.Module,
        shared: nn.Module,
    ):
        super().__init__()
        gate = cast(dict[str, mx.array], gate_proj)
        up = cast(dict[str, mx.array], up_proj)
        down = cast(dict[str, mx.array], down_proj)
        shared_gate = cast(dict[str, mx.array], shared["gate_proj"])
        shared_up = cast(dict[str, mx.array], shared["up_proj"])
        shared_down = cast(dict[str, mx.array], shared["down_proj"])
        self.group_size = cast(int, gate_proj.group_size)
        self.bits = cast(int, gate_proj.bits)
        self.down_group_size = cast(int, down_proj.group_size)
        self.down_bits = cast(int, down_proj.bits)
        # routed experts + the shared one
        self.experts = int(gate["weight"].shape[0]) + 1
        packed: dict[str, mx.array] = {}
        for name in ("weight", "scales", "biases"):
            packed["gate_up_" + name] = mx.concatenate(
                [gate[name], shared_gate[name][None], up[name], shared_up[name][None]]
            )
            packed["down_" + name] = mx.concatenate(
                [down[name], shared_down[name][None]]
            )
            mx.eval(packed["gate_up_" + name], packed["down_" + name])
        self.gate_up_weight = packed["gate_up_weight"]
        self.gate_up_scales = packed["gate_up_scales"]
        self.gate_up_biases = packed["gate_up_biases"]
        self.down_weight = packed["down_weight"]
        self.down_scales = packed["down_scales"]
        self.down_biases = packed["down_biases"]
        # (not a module attribute: a dict would join the parameter tree)
        object.__setattr__(self, "_lhs_cache", {})

    def _half(self, which: int) -> tuple[mx.array, mx.array, mx.array]:
        """The gate (0) or up (1) projections of every expert (views)."""
        start, end = which * self.experts, (which + 1) * self.experts
        return (
            self.gate_up_weight[start:end],
            self.gate_up_scales[start:end],
            self.gate_up_biases[start:end],
        )

    def _gather(
        self,
        x: mx.array,
        arrays: tuple[mx.array, mx.array, mx.array],
        indices: mx.array,
        group_size: int,
        bits: int,
        sorted_indices: bool = False,
        lhs: mx.array | None = None,
    ) -> mx.array:
        weight, scales, biases = arrays
        return mx.gather_qmm(
            x,
            weight,
            scales,
            biases,
            lhs_indices=lhs,
            rhs_indices=indices,
            transpose=True,
            group_size=group_size,
            bits=bits,
            sorted_indices=sorted_indices,
        )

    def _down_arrays(self) -> tuple[mx.array, mx.array, mx.array]:
        return self.down_weight, self.down_scales, self.down_biases

    def _routed(
        self, arrays: tuple[mx.array, mx.array, mx.array]
    ) -> tuple[mx.array, mx.array, mx.array]:
        """The routed experts alone (views): MLX's gather_qmm picks its kernel from
        the rows per expert, so a gather of routed experts must see their count."""
        routed = self.experts - 1
        weight, scales, biases = arrays
        return weight[:routed], scales[:routed], biases[:routed]

    def _lhs_indices(self, shape: tuple[int, ...]) -> tuple[mx.array, mx.array]:
        """For (..., k) experts per token, which token each of the 2k gate/up rows
        reads, and the (token, expert) row of each of the k down rows."""
        cached = self._lhs_cache.get(shape)
        if cached is None:
            *batch, k = shape
            tokens = math.prod(batch)
            token_of_row = np.repeat(np.arange(tokens, dtype=np.uint32), 2 * k)
            row = np.arange(tokens * k, dtype=np.uint32)
            cached = (
                mx.array(token_of_row.reshape(*batch, 2 * k)),
                mx.array(row.reshape(shape)),
            )
            self._lhs_cache[shape] = cached
        return cached

    @property
    def vector_down(self) -> bool:
        """The down projections take qwen4_exp_kernels.moe_down (4-bit, groups of 32)."""
        return (
            self.down_bits == 4
            and self.down_group_size == 32
            and self.gate_up_weight.shape[1] % 8 == 0
        )

    def gate_up(
        self, x: mx.array, indices: mx.array, rows: mx.array | None = None
    ) -> mx.array:
        """(..., D) tokens, (..., k) experts -> (..., 2k, inner): the experts' gate
        projections, then their up projections, in one unsorted gather (few tokens).
        ``rows``: the 2k gate/up rows of the experts, when already known."""
        lhs, _ = self._lhs_indices(indices.shape)
        both = (
            mx.concatenate([indices, indices + self.experts], axis=-1)
            if rows is None
            else rows
        )
        out = self._gather(
            mx.expand_dims(x, (-2, -3)),
            (self.gate_up_weight, self.gate_up_scales, self.gate_up_biases),
            both,
            self.group_size,
            self.bits,
            lhs=lhs,
        )
        return out.squeeze(-2)

    def _shared_projection(
        self,
        x: mx.array,
        arrays: tuple[mx.array, mx.array, mx.array],
        group: int,
        bits: int,
    ) -> mx.array:
        weight, scales, biases = arrays
        last = self.experts - 1
        return mx.quantized_matmul(
            x,
            weight[last],
            scales[last],
            biases[last],
            transpose=True,
            group_size=group,
            bits=bits,
        )

    def shared(self, x: mx.array) -> mx.array:
        """The shared expert alone, dense (as Qwen3NextMLP computes it)."""
        gate = self._shared_projection(x, self._half(0), self.group_size, self.bits)
        up = self._shared_projection(x, self._half(1), self.group_size, self.bits)
        return self._shared_projection(
            swiglu(gate, up), self._down_arrays(), self.down_group_size, self.down_bits
        )

    def __call__(self, x: mx.array, indices: mx.array) -> mx.array:
        """(..., D) tokens, (..., k) expert indices -> (..., k, D), as SwitchGLU."""
        if indices.size >= 64:
            # As SwitchGLU: group the (token, expert) rows by expert
            expanded = mx.expand_dims(x, (-2, -3))
            x, idx, inverse = _gather_sort(expanded, indices)
            gate_arrays, up_arrays = (
                self._routed(self._half(0)),
                self._routed(self._half(1)),
            )
            gate = self._gather(x, gate_arrays, idx, self.group_size, self.bits, True)
            up = self._gather(x, up_arrays, idx, self.group_size, self.bits, True)
            y = self._gather(
                swiglu(gate, up),
                self._routed(self._down_arrays()),
                idx,
                self.down_group_size,
                self.down_bits,
                True,
            )
            y = mx.unflatten(y[inverse], 0, indices.shape)
        else:
            k = indices.shape[-1]
            both = self.gate_up(x, indices)
            gate, up = both[..., :k, :], both[..., k:, :]
            _, down_lhs = self._lhs_indices(indices.shape)
            y = self._gather(
                swiglu(gate, up)[..., None, :],
                self._down_arrays(),
                indices,
                self.down_group_size,
                self.down_bits,
                lhs=down_lhs,
            )
        return y.squeeze(-2)


# ----------------------------------------------------------------------------- layers


class DecoderLayer(nn.Module):
    next_norm: "ZeroCenteredRMSNorm | None"
    handoff: "NormHandoff"

    def __init__(self, args: TextArgs, layer_idx: int):
        super().__init__()
        self.is_linear = args.layer_types[layer_idx] == "linear_attention"
        if self.is_linear:
            self.linear_attn = GatedDeltaNet(args)
        else:
            self.self_attn = Attention(args)
        self.mlp: nn.Module = SparseMoeBlock(args)
        # ple_layer_ids are one-indexed
        self.ple: PLELayer | None = (
            PLELayer(args, args.ple_layer_ids.index(layer_idx + 1))
            if layer_idx + 1 in args.ple_layer_ids
            else None
        )
        self.attn_hyper_connection = GatedResidual(args)
        self.mlp_hyper_connection = GatedResidual(args)
        self.packed = False
        # Set by Model.pack_for_inference (plain attributes: not part of the parameters)
        object.__setattr__(self, "next_norm", None)
        object.__setattr__(self, "handoff", NormHandoff())

    def __call__(
        self,
        h: mx.array,
        mask: AttentionMask = None,
        cache: ArraysCache | QSAKVCache | None = None,
        ngram_embedding: mx.array | None = None,
    ) -> mx.array:
        if self.ple is not None:
            if ngram_embedding is None:
                raise ValueError("qwen4_exp: the PLE layer needs the n-gram embeddings")
            assert cache is None or isinstance(cache, ArraysCache)
            h = h + self.ple(h, ngram_embedding, cache)

        if self.packed:
            return self._packed_call(h, mask, cache)

        x, inject = self.attn_hyper_connection.mix(h)
        assert inject is not None
        x = self._block(x, mask, cache)
        h = add_to_streams(h, x, inject)

        x, inject = self.mlp_hyper_connection.mix(h)
        assert inject is not None
        return add_to_streams(h, self.mlp(x), inject)

    def _block(
        self, x: mx.array, mask: AttentionMask, cache: ArraysCache | QSAKVCache | None
    ) -> mx.array:
        if self.is_linear:
            assert cache is None or isinstance(cache, ArraysCache)
            linear_mask = mask if isinstance(mask, mx.array) else None
            return self.linear_attn(x, linear_mask, cache)
        assert cache is None or isinstance(cache, QSAKVCache)
        return self.self_attn(x, mask, cache)

    def pack(self) -> None:
        """Pack the hyper-connections, the block and the MoE for inference; the layer
        takes the fused stream update only when both hyper-connections are packed."""
        hyper_packed = [
            self.attn_hyper_connection.pack(),
            self.mlp_hyper_connection.pack(),
        ]
        if self.is_linear:
            self.linear_attn.pack()
        else:
            self.self_attn.pack()
        moe = self.mlp
        if isinstance(moe, SparseMoeBlock):
            moe.pack()
        else:
            # exo's tensor parallelism wraps the MoE (all-reduce after it)
            original = getattr(moe, "original_layer", None)
            if isinstance(original, SparseMoeBlock):
                original.pack()
        self.packed = all(hyper_packed)
        if not self.packed:
            for hyper in (self.attn_hyper_connection, self.mlp_hyper_connection):
                assert hyper.projection is None, "hyper-connections packed partially"

    def _packed_call(
        self, h: mx.array, mask: AttentionMask, cache: ArraysCache | QSAKVCache | None
    ) -> mx.array:
        attn_hc, mlp_hc = self.attn_hyper_connection, self.mlp_hyper_connection
        # One-token steps fuse each stream update with the norm after it. Longer inputs
        # compute the norm right before its use: computed with the update, it stayed in
        # memory longer (0.2 GB more at the peak of a 4096-token prefill)
        fuse = h.size // h.shape[-1] <= _FUSED_TOKENS
        normed = self.handoff.take(h)
        if normed is None:
            normed = attn_hc.packed_norm(h)
        x, inject, inject_at = attn_hc.packed_mix(normed)
        x = self._block(x, mask, cache)
        h, normed = kernels.hc_add_norm(
            h,
            x,
            inject,
            inject_at,
            attn_hc.hc_count,
            mlp_hc.hc_norm.weight if fuse else None,
            mlp_hc.hc_norm.eps,
        )
        if normed is None:
            normed = mlp_hc.packed_norm(h)
        x, inject, inject_at = mlp_hc.packed_mix(normed)
        # The stream update and, when the next layer (or the final mixer) norms these
        # streams first, that norm, in one kernel
        following = self.next_norm if fuse else None
        h, normed = kernels.hc_add_norm(
            h,
            self.mlp(x),
            inject,
            inject_at,
            mlp_hc.hc_count,
            None if following is None else following.weight,
            0.0 if following is None else following.eps,
        )
        if normed is not None:
            self.handoff.give(h, normed)
        return h


class NormHandoff:
    """The streams one packed layer returns and their norm for the module after it, so
    that the next layer takes the norm instead of recomputing it. Only taken for the
    very array it was computed from."""

    def __init__(self) -> None:
        self._value: tuple[mx.array, mx.array] | None = None

    def give(self, streams: mx.array, normed: mx.array) -> None:
        self._value = (streams, normed)

    def take(self, streams: mx.array) -> mx.array | None:
        value, self._value = self._value, None
        if value is not None and value[0] is streams:
            return value[1]
        return None


class Qwen4ExpModel(nn.Module):
    handoff: NormHandoff

    def __init__(self, args: TextArgs):
        super().__init__()
        self.args = args
        self.embed_tokens = nn.Embedding(args.vocab_size, args.hidden_size)
        self.layers = [DecoderLayer(args, i) for i in range(args.num_hidden_layers)]
        # The final mixer stands in for a final norm
        self.hyper_connection_mixer = GatedResidual(args, use_combine=False)
        self.fa_idx: int | None = next(
            (i for i, layer in enumerate(self.layers) if not layer.is_linear), None
        )
        self.ssm_idx: int | None = next(
            (i for i, layer in enumerate(self.layers) if layer.is_linear), None
        )
        # Set by tensor parallelism: the ranks meet after their host n-gram reads
        self.tensor_group: mx.distributed.Group | None = None
        # Set by Model.pack_for_inference: decode steps read the n-gram rows after
        # building their graph (see PLELayer.deferred_embedding)
        self.defer_ngram_reads = False
        # Set by pipeline parallelism: the layer whose call evaluates this stage's output
        self.stage_output_layer: int | None = None
        object.__setattr__(self, "handoff", NormHandoff())

    def __call__(
        self,
        inputs: mx.array,
        cache: Sequence[ArraysCache | QSAKVCache | None] | None = None,
        input_embeddings: mx.array | None = None,
    ) -> mx.array:
        h = self.embed_tokens(inputs) if input_embeddings is None else input_embeddings
        layer_caches: Sequence[ArraysCache | QSAKVCache | None] = (
            cache if cache is not None else [None] * len(self.layers)
        )
        fa_mask: AttentionMask = None
        ssm_mask: AttentionMask = None
        if self.fa_idx is not None:
            fa_mask = create_attention_mask(h, layer_caches[self.fa_idx])
        if self.ssm_idx is not None:
            ssm_mask = create_ssm_mask(h, layer_caches[self.ssm_idx])
        # The n-gram hash and table read run on the host: they need the token ids there
        ple_index = next(
            (i for i, layer in enumerate(self.layers) if layer.ple is not None), None
        )
        ngram_embedding: mx.array | None = None
        fill: Callable[[], None] | None = None
        if ple_index is not None:
            ple = self.layers[ple_index].ple
            assert ple is not None
            ple_cache = layer_caches[ple_index]
            assert ple_cache is None or isinstance(ple_cache, ArraysCache)
            if self.defer_ngram_reads and inputs.shape[-1] <= _FUSED_TOKENS:
                ngram_embedding, fill = ple.deferred_embedding(
                    inputs, ple_cache, h.dtype
                )
            else:
                ids = np.array(inputs, dtype=np.int64).reshape(inputs.shape[0], -1)
                ngram_embedding = ple.embedding(ids, ple_cache, h.dtype)

        h = mx.tile(h, (1, 1, self.args.hc_count))
        for index, (layer, layer_cache) in enumerate(
            zip(self.layers, layer_caches, strict=True)
        ):
            if fill is not None and index == self.stage_output_layer:
                # A pipeline stage evaluates its output in its last layer's call: read
                # the rows before it
                fill()
                fill = None
            mask = ssm_mask if layer.is_linear else fa_mask
            h = layer(h, mask, layer_cache, ngram_embedding)
        mixer = self.hyper_connection_mixer
        if mixer.projection is not None:
            normed = self.handoff.take(h)
            if normed is None:
                normed = mixer.packed_norm(h)
            mixed, _, _ = mixer.packed_mix(normed)
        else:
            mixed, _ = mixer.mix(h)
        if fill is not None:
            # The whole graph is built: now wait for the tokens and read their rows
            fill()
        if self.tensor_group is not None and ple_index is not None:
            # The n-gram rows are read on the host while the graph is built, which can
            # take one rank much longer than the other (a cold page cache). Let every rank
            # finish before any starts the graph: a rank that starts early waits in the
            # first all-reduce on the GPU, where a long wait fails the command buffer.
            mx.eval(
                mx.distributed.all_sum(
                    mx.array(1.0),
                    group=self.tensor_group,
                    stream=mx.default_stream(mx.Device(mx.cpu)),
                )
            )
        return mixed


class TextModel(nn.Module):
    def __init__(self, args: TextArgs):
        super().__init__()
        self.args = args
        self.model_type = args.model_type
        self.model = Qwen4ExpModel(args)
        if not args.tie_word_embeddings:
            self.lm_head = nn.Linear(args.hidden_size, args.vocab_size, bias=False)

    def __call__(
        self,
        inputs: mx.array,
        cache: Sequence[ArraysCache | QSAKVCache | None] | None = None,
        input_embeddings: mx.array | None = None,
    ) -> mx.array:
        out = self.model(inputs, cache, input_embeddings)
        if self.args.tie_word_embeddings:
            return self.model.embed_tokens.as_linear(out)
        return self.lm_head(out)


def _rename(key: str) -> str:
    if key.startswith("model.language_model."):
        return "language_model.model." + key.removeprefix("model.language_model.")
    if key.startswith("lm_head."):
        return "language_model." + key
    if key.startswith("language_model."):
        return key
    return "language_model.model." + key.removeprefix("model.")


_DROPPED_PREFIXES = ("vision_tower.", "model.visual.", "visual.", "mtp.")
_PER_EXPERT_KEY = re.compile(
    r"^(?P<mlp>.*\.mlp)\.experts\.(?P<expert>\d+)\.(?P<proj>gate_proj|up_proj|down_proj)\.weight$"
)
_HASH_BUFFERS = ("layer_multipliers", "ngram_heads_vocab_sizes", "ngram_heads_offsets")


class Model(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.model_type = args.model_type
        self.language_model = TextModel(TextArgs.from_dict(args.text_config))

    def __call__(
        self,
        inputs: mx.array,
        cache: Sequence[ArraysCache | QSAKVCache | None] | None = None,
        input_embeddings: mx.array | None = None,
    ) -> mx.array:
        return self.language_model(inputs, cache, input_embeddings)

    @property
    def model(self) -> Qwen4ExpModel:
        return self.language_model.model

    @property
    def layers(self) -> list[DecoderLayer]:
        return self.language_model.model.layers

    def make_cache(self) -> list[ArraysCache | QSAKVCache]:
        caches: list[ArraysCache | QSAKVCache] = []
        for layer in self.layers:
            if layer.ple is not None:
                caches.append(ArraysCache(size=4))
            elif layer.is_linear:
                caches.append(ArraysCache(size=2))
            else:
                caches.append(QSAKVCache())
        return caches

    def pack_for_inference(self) -> None:
        """Repack the loaded (and sharded) weights for exo's inference path, without
        changing what the model computes:
        - linear layers that share an input become one matmul (each output column
          computed as before);
        - the MoE's shared expert joins the routed experts as one more expert;
        - the hyper-connections, the router and the Gated DeltaNet run fused kernels.
        A layer whose weights aren't affine-quantised keeps the plain path. Runs after
        tensor/pipeline sharding (exo's post_load). EXO_QWEN4_EXP_PACK=0 turns it off."""
        if not kernels.available() or os.environ.get("EXO_QWEN4_EXP_PACK", "1") == "0":
            return
        wrapped = False
        originals: list[DecoderLayer] = []
        for layer in self.layers:
            original = cast(DecoderLayer, getattr(layer, "original_layer", layer))
            wrapped = wrapped or original is not layer
            original.pack()
            originals.append(original)
            mx.clear_cache()
        mixer = self.model.hyper_connection_mixer
        mixer.pack()
        # A layer computes the norm its successor starts with, when the successor's input
        # is its output: not across a pipeline stage boundary (the stage's last layer is
        # wrapped), nor into the n-gram layer (its embedding is added first).
        handoff = self.model.handoff
        last = len(self.layers) - 1
        for index, (layer, original) in enumerate(
            zip(self.layers, originals, strict=True)
        ):
            object.__setattr__(original, "handoff", handoff)
            if not original.packed:
                continue
            if index < last:
                successor = originals[index + 1]
                if layer is original and successor.packed and successor.ple is None:
                    object.__setattr__(
                        original, "next_norm", successor.attn_hyper_connection.hc_norm
                    )
            elif layer is original and mixer.projection is not None:
                object.__setattr__(original, "next_norm", mixer.hc_norm)
        # A pipeline stage evaluates its output inside its last layer's call: the deferred
        # n-gram read fills its buffer before that call (stage_output_layer). Other
        # wrapped layers: read the rows up front.
        self.model.defer_ngram_reads = (
            not wrapped or self.model.stage_output_layer is not None
        )

    def attach_ngram_table(self, model_path: Path, prewarm: bool = True) -> None:
        """Memory-map the n-gram tables in the checkpoint at ``model_path`` (both the
        mlx-community and the official key layouts) for this model's PLE layers. With
        ``prewarm``, the tables' files are read once in the background so that row
        reads come from the page cache."""
        text_args = self.language_model.args
        index = cast(
            dict[str, object],
            json.loads((model_path / "model.safetensors.index.json").read_text()),
        )
        prefixes = {
            match["prefix"]
            for key in cast(dict[str, str], index["weight_map"])
            if (match := _SHARD_KEY.match(key)) is not None
            and not key.startswith("mtp.")
        }
        for layer in self.layers:
            if layer.ple is None:
                continue
            embedding = layer.ple.ple_embedding
            layer_id = text_args.ple_layer_ids[embedding.ple_layer_index] - 1
            # language_model.model.layers.N (mlx-community), model.language_model.layers.N
            # (official) or model.layers.N (a text-only transformers checkpoint)
            suffix = f"layers.{layer_id}.ple.ple_embedding.ngram_embedding"
            matching = sorted(p for p in prefixes if p.endswith("." + suffix))
            if len(matching) != 1:
                raise ValueError(
                    f"qwen4_exp: no single n-gram table for layer {layer_id} in "
                    f"{model_path} (found {matching})"
                )
            table = NGramTable(model_path, matching[0], embedding.width)
            embedding.attach(table)
            if prewarm:
                threading.Thread(target=table.prewarm, daemon=True).start()

    def sanitize(self, weights: dict[str, mx.array]) -> dict[str, mx.array]:
        """Map the official (Hugging Face; fused or per-expert experts) or mlx-community keys
        onto this model. Drops
        vision, MTP and the n-gram tables (read from the files, see
        attach_ngram_table) and checks the stored n-gram hash buffers."""
        text_args = self.language_model.args
        n_layers = text_args.num_hidden_layers
        out: dict[str, mx.array] = {}
        hash_buffers: dict[tuple[int, str], list[int]] = {}
        per_expert: dict[tuple[str, str], dict[int, mx.array]] = {}
        for key, value in weights.items():
            if key.startswith(_DROPPED_PREFIXES) or ".mtp." in key:
                continue
            name = _rename(key)
            layer_match = re.match(r"language_model\.model\.layers\.(\d+)\.", name)
            if layer_match is not None and int(layer_match[1]) >= n_layers:
                continue
            if ".ple.ple_embedding." in name:
                buffer = name.rsplit(".", 1)[-1]
                if buffer in _HASH_BUFFERS and layer_match is not None:
                    hash_buffers[(int(layer_match[1]), buffer)] = cast(
                        list[int], value.tolist()
                    )
                continue
            if name.endswith("mlp.experts.gate_up_proj"):
                base = name.removesuffix("experts.gate_up_proj")
                gate, up = mx.split(value, 2, axis=-2)
                out[base + "switch_mlp.gate_proj.weight"] = gate
                out[base + "switch_mlp.up_proj.weight"] = up
                continue
            if name.endswith("mlp.experts.down_proj"):
                out[
                    name.removesuffix("experts.down_proj")
                    + "switch_mlp.down_proj.weight"
                ] = value
                continue
            expert = _PER_EXPERT_KEY.match(name)
            if expert is not None:
                # Per-expert tensors (what transformers' save_pretrained writes)
                per_expert.setdefault((expert["mlp"], expert["proj"]), {})[
                    int(expert["expert"])
                ] = value
                continue
            if (
                name.endswith("conv1d.weight")
                and value.ndim == 3
                and value.shape[1] == 1
            ):
                value = value.swapaxes(1, 2)
            out[name] = value
        for (mlp, proj), experts in per_expert.items():
            stacked = [experts[index] for index in sorted(experts)]
            if sorted(experts) != list(range(text_args.num_experts)):
                raise ValueError(f"qwen4_exp: {mlp} has {len(experts)} experts' {proj}")
            out[f"{mlp}.switch_mlp.{proj}.weight"] = mx.stack(stacked)
        if text_args.tie_word_embeddings:
            out.pop("language_model.lm_head.weight", None)
        for index, layer in enumerate(self.layers):
            if layer.ple is None:
                continue
            stored = [hash_buffers.get((index, name)) for name in _HASH_BUFFERS]
            if all(buffer is not None for buffer in stored):
                multipliers, vocab_sizes, offsets = cast(list[list[int]], stored)
                layer.ple.ple_embedding.hasher.check_buffers(
                    multipliers, vocab_sizes, offsets
                )
        return out

    @property
    def cast_predicate(self) -> Callable[[str], bool]:
        def predicate(path: str) -> bool:
            return not path.endswith("A_log")

        return predicate
