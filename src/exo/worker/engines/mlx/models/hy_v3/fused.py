"""Stacked quantized projections and expert weights for Hy3's fused paths.

Stacks are built once, at a layer's first call (after loading and sharding), and the
original modules' weights then point into them (slices of the stack: no copy, no extra
memory). Every module keeps its own forward, which the fallback paths call.

Exactness: MLX picks its quantized-matmul kernel by shape. A stack is multiplied in one
call only where every output row keeps the reduction it has alone:
- dense projections, below MLX's matrix-vector batch limit for the stack and for every part
  (``qmv`` for one row, ``qmv_wide`` for a few: kernels that give each output row its own
  reduction over K, independent of how many rows the matrix has); above it each part runs
  on its own slice of the stack, the original matmul on the original values;
- expert gathers: every gathered row is one token-expert pair, multiplied by MLX's
  matrix-vector gather or, sorted, by the gather that tiles N without splitting K; the
  choice between them depends on the number of pairs and of experts, which a stack of
  gate and up (along N) leaves unchanged.
"""

from dataclasses import dataclass
from functools import cache
from typing import cast

import mlx.core as mx
import mlx.nn as nn


@cache
def _gpu_architecture() -> tuple[int, str] | None:
    """(generation, size letter) of the Apple GPU, e.g. (15, "d") for an M3 Ultra."""
    if not mx.metal.is_available():
        return None
    name = str(mx.device_info().get("architecture", ""))
    if not name.startswith("applegpu_g") or len(name) < len("applegpu_g") + 2:
        return None
    digits = "".join(c for c in name.removeprefix("applegpu_g") if c.isdigit())
    if not digits:
        return None
    return int(digits), name[-1]


def qmv_batch_limit(in_dims: int, out_dims: int) -> int | None:
    """MLX 0.32.4's ``get_qmv_batch_limit`` (backend/metal/quantized.cpp): from this many
    rows a transposed quantized matmul runs the matrix-matrix kernels. None if the GPU is
    unknown."""
    architecture = _gpu_architecture()
    if architecture is None:
        return None
    generation, size = architecture
    small = in_dims <= 2048 and out_dims <= 2048
    medium = in_dims <= 4096 and out_dims <= 4096
    if generation >= 17 and size != "d":
        return 33 if small else 25 if medium else 13
    if generation >= 15 and size != "d":
        return 13 if small else 15 if medium else 13
    if size == "d":
        return 32 if small else 18 if medium else 12
    if generation >= 13:
        if small:
            return 14
        if medium:
            return 10
        return 7 if (generation == 13 and size == "s") else 6
    return 18 if small else 12 if medium else 10


def quantized_affine(linear: nn.Module) -> bool:
    """A quantized linear (or switch linear) without a bias whose forward is one affine
    ``quantized_matmul`` / ``gather_qmm``."""
    return (
        all(name in linear for name in ("weight", "scales", "biases"))
        and "bias" not in linear
        and getattr(linear, "mode", None) == "affine"
        and getattr(linear, "global_scale", None) is None
        and isinstance(getattr(linear, "group_size", None), int)
        and isinstance(getattr(linear, "bits", None), int)
    )


def arrays(linear: nn.Module) -> tuple[mx.array, mx.array, mx.array]:
    return (
        cast(mx.array, linear["weight"]),
        cast(mx.array, linear["scales"]),
        cast(mx.array, linear["biases"]),
    )


def _same_quantization(parts: list[nn.Module]) -> bool:
    if not all(quantized_affine(p) for p in parts):
        return False
    first = parts[0]
    weight0, scales0, _ = arrays(first)
    for part in parts:
        weight, scales, biases = arrays(part)
        if (
            getattr(part, "group_size") != getattr(first, "group_size")  # noqa: B009
            or getattr(part, "bits") != getattr(first, "bits")  # noqa: B009
            or weight.shape[:-2] != weight0.shape[:-2]
            or weight.shape[-1] != weight0.shape[-1]
            or weight.dtype != weight0.dtype
            or scales.dtype != scales0.dtype
            or biases.dtype != scales0.dtype
        ):
            return False
    return True


def _point_into(
    linear: nn.Module, stack: tuple[mx.array, ...], index: tuple[slice | int, ...]
) -> None:
    weight, scales, biases = stack
    linear["weight"] = weight[index]
    linear["scales"] = scales[index]
    linear["biases"] = biases[index]


@dataclass(frozen=True)
class StackedProjection:
    """Dense quantized linears with one input, stacked along their output rows."""

    weight: mx.array
    scales: mx.array
    biases: mx.array
    group_size: int
    bits: int
    bounds: tuple[int, ...]
    max_rows: int

    @staticmethod
    def build(parts: list[nn.Module]) -> "StackedProjection | None":
        """Stack ``parts`` (each then pointing into the stack), or None if they can't be."""
        if not _same_quantization(parts):
            return None
        if any(cast(mx.array, p["weight"]).ndim != 2 for p in parts):
            return None
        rows = [cast(mx.array, p["weight"]).shape[0] for p in parts]
        bits = cast(int, getattr(parts[0], "bits"))  # noqa: B009
        in_dims = cast(mx.array, parts[0]["weight"]).shape[-1] * 32 // bits
        limits = [qmv_batch_limit(in_dims, n) for n in [*rows, sum(rows)]]
        if any(limit is None for limit in limits):
            return None
        stack = tuple(
            mx.concatenate([cast(mx.array, p[name]) for p in parts], axis=0)
            for name in ("weight", "scales", "biases")
        )
        mx.eval(*stack)
        bounds = tuple(sum(rows[:i]) for i in range(len(rows) + 1))
        for part, start, stop in zip(parts, bounds[:-1], bounds[1:], strict=True):
            _point_into(part, stack, (slice(start, stop),))
        return StackedProjection(
            weight=stack[0],
            scales=stack[1],
            biases=stack[2],
            group_size=cast(int, getattr(parts[0], "group_size")),  # noqa: B009
            bits=cast(int, getattr(parts[0], "bits")),  # noqa: B009
            bounds=bounds,
            max_rows=min(cast(int, limit) for limit in limits) - 1,
        )

    def __call__(self, x: mx.array) -> mx.array | None:
        """All parts' outputs side by side, when one matmul gives the same bits."""
        if x.size // x.shape[-1] > self.max_rows:
            return None
        return mx.quantized_matmul(
            x,
            self.weight,
            self.scales,
            self.biases,
            transpose=True,
            group_size=self.group_size,
            bits=self.bits,
            mode="affine",
        )


@dataclass(frozen=True)
class ExpertStack:
    """The routed experts' gate and up projections stacked along N: (E, 2I, ...)."""

    gate_up: tuple[mx.array, mx.array, mx.array]
    intermediate: int
    group_size: int
    bits: int

    @staticmethod
    def build(gate: nn.Module, up: nn.Module) -> "ExpertStack | None":
        """Stack gate and up (both then pointing into the stack), or None if they can't
        be."""
        if not _same_quantization([gate, up]):
            return None
        gate_weight, _, _ = arrays(gate)
        if gate_weight.ndim != 3:
            return None
        intermediate = gate_weight.shape[1]
        stack = tuple(
            mx.concatenate([g, u], axis=1)
            for g, u in zip(arrays(gate), arrays(up), strict=True)
        )
        mx.eval(*stack)
        experts = slice(0, gate_weight.shape[0])
        _point_into(gate, stack, (experts, slice(0, intermediate)))
        _point_into(up, stack, (experts, slice(intermediate, 2 * intermediate)))
        return ExpertStack(
            gate_up=cast(tuple[mx.array, mx.array, mx.array], stack),
            intermediate=intermediate,
            group_size=cast(int, getattr(gate, "group_size")),  # noqa: B009
            bits=cast(int, getattr(gate, "bits")),  # noqa: B009
        )


@cache
def row_indices(rows: int) -> mx.array:
    """The lhs indices ``mx.gather_qmm`` builds when it gets none (one per row of x), for a
    decode-sized number of rows, built once."""
    indices = mx.arange(rows, dtype=mx.uint32)
    mx.eval(indices)
    return indices


def gather(
    x: mx.array,
    weights: tuple[mx.array, mx.array, mx.array],
    group_size: int,
    bits: int,
    indices: mx.array,
    sorted_indices: bool,
) -> mx.array:
    """``QuantizedSwitchLinear.__call__`` (affine, no bias), with cached row indices when
    unsorted and few."""
    weight, scales, biases = weights
    lhs: mx.array | None = None
    rows = x.size // (x.shape[-1] * x.shape[-2])
    if not sorted_indices and rows <= 64:
        lhs = row_indices(rows).reshape(*x.shape[:-2])
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
        mode="affine",
        sorted_indices=sorted_indices,
    )


def gather_uses_matrix_vector(pairs: int, experts: int, sorted_indices: bool) -> bool:
    """Whether MLX's ``GatherQMM`` takes its matrix-vector kernel for ``pairs`` one-row
    token-expert pairs over ``experts`` experts (not ``gather_qmm_rhs``, which it takes
    for sorted pairs from 16 and 4 per expert)."""
    return not (sorted_indices and pairs >= 16 and pairs // experts >= 4)
