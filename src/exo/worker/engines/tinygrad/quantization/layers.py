from typing import final

from tinygrad import Tensor

from .dequantization import affine_dequantize
from .packing import PackedTensor, unpack_bits

@final
class QuantizedLinear:
    """
        This is a drop-in replacedment for tinygrad's Linear class while
        supporting quantization.

        Since the implementation relies purely on tensor ops, we're using lazy
        dequantization to optimize performance. The first forward pass will
        dequantize the weights and caching it for subsequent calls.
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
        self._dequantized_weight: Tensor | None = None

    def __call__(self, x: Tensor) -> Tensor:
        if self._dequantized_weight is None:
            self._dequantized_weight = self._dequantize()

        result = x @ self._dequantized_weight.T
        if self.bias is not None:
            result = result + bias

        return result

    def _dequantize(self) -> Tensor:
        unpacked = unpack_bits(self.weight_q)
        return affine_dequantize(
            unpacked,  self.scales,
            self.biases, self.group_size)

    def clear_cache(self) -> None:
        self._dequantized_weight = None

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
        self._dequantized_weight: Tensor | None = None

    def __call__(self, indices: Tensor) -> Tensor:
        if self._dequantized_weight is None:
            self._dequantized_weight = self._dequantize()
        return self._dequantized_weight[indices]

    def _dequantize(self) -> Tensor:
        unpacked = unpack_bits(self.weight_q)
        return affine_dequantize(
            unpacked, self.scales, self.biases, self.group_size
        )

    def clear_cache(self) -> None:
        self._dequantized_weight = None
