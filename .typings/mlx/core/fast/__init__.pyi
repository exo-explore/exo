from collections.abc import Sequence
from typing import Protocol

from mlx.core import Device, Dtype, Stream, array

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
    """A custom Metal kernel from the body of its source (see the MLX documentation)."""

def rms_norm(
    x: array,
    weight: array | None,
    eps: float,
    *,
    stream: Stream | Device | None = ...,
) -> array: ...
