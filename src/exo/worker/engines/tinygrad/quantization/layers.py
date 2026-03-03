from typing import final

from tinygrad.tensor import Tensor

from .dequantization import affine_dequantize
from .packing import PackedTensor, unpack_bits


@final
class QuantizedLinear:
    """
        This is a drop-in replacement for tinygrad's Linear class while
        supporting quantization.

        Weights are eagerly dequantized in __init__ so that tinygrad's kernel
        cache can reuse compiled kernels for same-shape weights, avoiding
        redundant HIP/CUDA compilations during the first forward pass.
    """

    def __init__(
        self,
        weight_q: PackedTensor,
        scales: Tensor,
        biases: Tensor,
        group_size: int = 64,
        bias: Tensor | None = None,
    ) -> None:
        self.weight_q = weight_q
        self.scales = scales
        self.biases = biases
        self.group_size = group_size
        self.bias = bias

    def __call__(self, x: Tensor) -> Tensor:
        result = x @ self.dequantize().T
        if self.bias is not None:
            result = result + self.bias
        return result

    def dequantize(self) -> Tensor:
        unpacked = unpack_bits(self.weight_q)
        return affine_dequantize(
            unpacked,  self.scales,
            self.biases, self.group_size)

    @property
    def in_features(self) -> int:
        return self.weight_q.original_shape[1]

    @property
    def out_features(self) -> int:
        return self.weight_q.original_shape[0]

@final
class QuantizedEmbedding:
    """
        This is a drop-in replacement for tinygrad's Embedding class providing
        the embedding support.
    """

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        weight_q: PackedTensor,
        scales: Tensor,
        biases: Tensor,
        group_size: int = 64,
    ) -> None:
        self.num_embeddings = num_embeddings
        self.weight_q = weight_q
        self.embedding_dim = embedding_dim
        self.scales = scales
        self.biases = biases
        self.group_size = group_size

    def __call__(self, indices: Tensor) -> Tensor:
        return self.dequantize()[indices]

    def dequantize(self) -> Tensor:
        unpacked = unpack_bits(self.weight_q)
        return affine_dequantize(
            unpacked, self.scales, self.biases, self.group_size
        )
