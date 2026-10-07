from typing import Protocol, Sequence

from mlx.core import Device, Dtype, Stream, array

def rms_norm(
    x: array,
    weight: array | None,
    eps: float,
    *,
    stream: Stream | Device | None = ...,
) -> array: ...
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
) -> array: ...
def scaled_dot_product_attention(
    q: array,
    k: array,
    v: array,
    *,
    scale: float,
    mask: array | str | None = ...,
    sinks: array | None = ...,
    stream: Stream | Device | None = ...,
) -> array: ...

class MetalKernel(Protocol):
    def __call__(
        self,
        *,
        inputs: Sequence[array | float | int],
        output_shapes: Sequence[Sequence[int]],
        output_dtypes: Sequence[Dtype],
        grid: tuple[int, int, int],
        threadgroup: tuple[int, int, int],
        template: Sequence[tuple[str, bool | int | Dtype]] | None = ...,
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
) -> MetalKernel: ...
