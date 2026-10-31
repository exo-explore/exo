"""Stubs for mlx.core.fast (fused kernels)."""

from collections.abc import Sequence
from typing import Literal, Protocol

from mlx.core import Device, Dtype, Stream, array

def rope(
    a: array,
    dims: int,
    *,
    traditional: bool,
    base: float | None,
    scale: float,
    offset: int | array,
    freqs: array | None = ...,
    stream: Stream | Device | None = ...,
) -> array:
    """Apply rotary positional encoding to the input."""

def rms_norm(
    x: array, weight: array | None, eps: float, *, stream: Stream | Device | None = ...
) -> array:
    """Root Mean Square normalization (RMS norm)."""

def layer_norm(
    x: array,
    weight: array | None,
    bias: array | None,
    eps: float,
    *,
    stream: Stream | Device | None = ...,
) -> array:
    """Layer normalization."""

def scaled_dot_product_attention(
    q: array,
    k: array,
    v: array,
    *,
    scale: float,
    mask: Literal["causal"] | array | None = ...,
    sinks: array | None = ...,
    stream: Stream | Device | None = ...,
) -> array:
    """A fast implementation of multi-head attention: O = softmax(Q @ K.T, dim=-1) @ V."""

class MetalKernel(Protocol):
    def __call__(
        self,
        *,
        inputs: Sequence[array | int | float | bool],
        output_shapes: Sequence[Sequence[int]],
        output_dtypes: Sequence[Dtype],
        grid: tuple[int, int, int],
        threadgroup: tuple[int, int, int],
        template: Sequence[tuple[str, int | bool | Dtype]] | None = ...,
        init_value: float | None = ...,
        verbose: bool = ...,
        stream: Stream | Device | None = ...,
    ) -> list[array]: ...

def metal_kernel(
    name: str,
    input_names: Sequence[str],
    output_names: Sequence[str],
    source: str,
    header: str = ...,
    ensure_row_contiguous: bool = ...,
    atomic_outputs: bool = ...,
) -> MetalKernel:
    """A jit-compiled custom Metal kernel defined from a source string."""
