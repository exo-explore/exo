"""Pass precomputed ``lhs_indices`` to ``gather_qmm`` in quantized MoE experts.

Without ``lhs_indices``, ``mx.gather_qmm`` builds ``arange(...)`` for the
input's batch dims inside every call, which is an extra GPU kernel (plus a
cast) in front of each expert projection: three per MoE layer per decode
step. Those indices only depend on the input's batch shape, so for the small
shapes seen at decode we build them once, keep them evaluated, and pass them
in. Larger (prefill) shapes keep the default path.
"""

from typing import cast

import mlx.core as mx
from mlx_lm.models.switch_layers import QuantizedSwitchLinear

_MAX_CACHED_BATCH = 64

_lhs_indices: dict[tuple[int, ...], mx.array] = {}


def _cached_lhs_indices(batch_shape: tuple[int, ...]) -> mx.array | None:
    size = 1
    for dim in batch_shape:
        size *= dim
    if size > _MAX_CACHED_BATCH:
        return None
    indices = _lhs_indices.get(batch_shape)
    if indices is None:
        indices = mx.reshape(mx.array(list(range(size)), dtype=mx.uint32), batch_shape)
        _lhs_indices[batch_shape] = indices
    return indices


def _patched_call(
    self: QuantizedSwitchLinear,
    x: mx.array,
    indices: mx.array,
    sorted_indices: bool = False,
) -> mx.array:
    weight = cast(mx.array, self["weight"])
    scales = cast(mx.array, self["scales"])
    biases = cast(mx.array | None, self.get("biases"))
    y = mx.gather_qmm(
        x,
        weight,
        scales,
        biases,
        lhs_indices=_cached_lhs_indices(tuple(x.shape[:-2])),
        rhs_indices=indices,
        transpose=True,
        group_size=self.group_size,
        bits=self.bits,
        mode=self.mode,
        sorted_indices=sorted_indices,
    )
    if "bias" in self:
        bias = cast(mx.array, self["bias"])
        y = y + mx.expand_dims(bias[indices], -2)
    return y


def patch_switch_lhs_indices() -> None:
    QuantizedSwitchLinear.__call__ = _patched_call
