"""Byte buffers that cross a pipeline hop, and their on-device tensors.

Exo owns the network. These functions perform one host-to-device copy onto
``Device.DEFAULT``, or one copy back to row-major bytes. The realized tensor
does not keep the previous hop's graph.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Literal, final

from pydantic import ConfigDict

from exo.backends.tinygrad_weights import TinygradWeightError
from exo.utils.pydantic_ext import FrozenModel

if TYPE_CHECKING:
    from tinygrad.tensor import Tensor

type HiddenStateDTypeName = Literal["float16", "bfloat16", "float32"]

_ITEMSIZE_BY_DTYPE: dict[HiddenStateDTypeName, int] = {
    "float16": 2,
    "bfloat16": 2,
    "float32": 4,
}
_TOKEN_ID_ITEMSIZE = 4

_BUFFER_CONFIG = ConfigDict(
    extra="forbid",
    frozen=True,
    strict=True,
)


@final
class HiddenStateBuffer(FrozenModel):
    """Row-major hidden state or logits carried between ranks."""

    model_config = _BUFFER_CONFIG

    dtype: HiddenStateDTypeName
    shape: tuple[int, ...]
    data: bytes


@final
class TokenIdBuffer(FrozenModel):
    """Row-major int32 token ids for the first pipeline rank."""

    model_config = _BUFFER_CONFIG

    shape: tuple[int, ...]
    data: bytes


def _expected_byte_count(shape: tuple[int, ...], itemsize: int) -> int:
    return math.prod(shape) * itemsize


def hidden_state_to_tensor(hidden_state: HiddenStateBuffer) -> Tensor:
    """Copy ``hidden_state`` onto ``Device.DEFAULT`` and realize it.

    Raises:
        TinygradWeightError: The runner entrypoint handles this when the byte
            length does not match ``shape`` and ``dtype``. The import error
            for a missing tinygrad install is left to propagate as
            ``TinygradDeviceSelectionError`` from engine construction, which
            the same entrypoint already handles.
    """
    from tinygrad import dtypes
    from tinygrad.tensor import Tensor

    expected = _expected_byte_count(
        hidden_state.shape, _ITEMSIZE_BY_DTYPE[hidden_state.dtype]
    )
    if len(hidden_state.data) != expected:
        raise TinygradWeightError(
            f"Hidden state has {len(hidden_state.data)} bytes, expected {expected}"
        )
    tinygrad_dtype = {
        "float16": dtypes.float16,
        "bfloat16": dtypes.bfloat16,
        "float32": dtypes.float32,
    }[hidden_state.dtype]
    return (
        Tensor(hidden_state.data, dtype=dtypes.uint8)
        .bitcast(tinygrad_dtype)
        .reshape(*hidden_state.shape)
        .contiguous()
        .realize()
    )


def tensor_to_hidden_state(tensor: Tensor) -> HiddenStateBuffer:
    """Realize ``tensor`` and copy its bytes back to the host.

    Raises:
        TinygradWeightError: The runner entrypoint handles this when the
            realized dtype is not float16, bfloat16, or float32.
    """
    from tinygrad import dtypes

    realized = tensor.contiguous().realize()
    dtype_name: HiddenStateDTypeName
    if realized.dtype == dtypes.float16:
        dtype_name = "float16"
    elif realized.dtype == dtypes.bfloat16:
        dtype_name = "bfloat16"
    elif realized.dtype == dtypes.float32:
        dtype_name = "float32"
    else:
        raise TinygradWeightError(
            f"Unsupported hidden-state dtype {realized.dtype.name}"
        )
    raw = realized.bitcast(dtypes.uint8).contiguous().realize().numpy().tobytes()
    shape = tuple(int(dimension) for dimension in realized.shape)
    return HiddenStateBuffer(dtype=dtype_name, shape=shape, data=raw)


def token_ids_to_tensor(token_ids: TokenIdBuffer) -> Tensor:
    """Copy int32 token ids onto ``Device.DEFAULT`` and realize them.

    Raises:
        TinygradWeightError: The runner entrypoint handles this when the byte
            length does not match ``shape``.
    """
    from tinygrad import dtypes
    from tinygrad.tensor import Tensor

    expected = _expected_byte_count(token_ids.shape, _TOKEN_ID_ITEMSIZE)
    if len(token_ids.data) != expected:
        raise TinygradWeightError(
            f"Token ids have {len(token_ids.data)} bytes, expected {expected}"
        )
    return (
        Tensor(token_ids.data, dtype=dtypes.uint8)
        .bitcast(dtypes.int32)
        .reshape(*token_ids.shape)
        .contiguous()
        .realize()
    )
