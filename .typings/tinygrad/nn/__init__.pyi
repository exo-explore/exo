"""Tinygrad neural-network helpers."""

from tinygrad.tensor import Tensor


class Linear:
    weight: Tensor
    bias: Tensor | None

    def __init__(
        self, in_features: int, out_features: int, bias: bool = True
    ) -> None: ...
    def __call__(self, hidden_state: Tensor) -> Tensor: ...
