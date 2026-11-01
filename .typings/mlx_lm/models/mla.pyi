import mlx.core as mx
import mlx.nn as nn

class MultiLinear(nn.Module):
    weight: mx.array
    def __init__(self, input_dims: int, output_dims: int, num_heads: int) -> None: ...
    def __call__(self, x: mx.array, transpose: bool = ...) -> mx.array: ...
    def to_quantized(
        self, group_size: int, bits: int, mode: str = ...
    ) -> QuantizedMultiLinear: ...

class QuantizedMultiLinear(nn.Module):
    group_size: int
    bits: int
    mode: str
    weight: mx.array
    scales: mx.array
    biases: mx.array | None
    def __init__(
        self,
        input_dims: int,
        output_dims: int,
        num_heads: int,
        group_size: int,
        bits: int,
        mode: str,
    ) -> None: ...
    def __call__(self, x: mx.array, transpose: bool = ...) -> mx.array: ...
