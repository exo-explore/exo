import torch
from _typeshed import Incomplete
from collections.abc import Callable
from typing import Any
from vllm.model_executor.layers.quantization.quark.schemes import QuarkScheme

__all__ = ["QuarkW8A8Fp8"]

class QuarkW8A8Fp8(QuarkScheme):
    weight_qscheme: Incomplete
    is_static_input_scheme: bool
    input_qscheme: str | None
    activation_quant_key: Incomplete
    weight_quant_key: Incomplete
    out_dtype: Incomplete
    def __init__(
        self, weight_config: dict[str, Any], input_config: dict[str, Any] | None
    ) -> None: ...
    @classmethod
    def get_min_capability(cls) -> int: ...
    def process_weights_after_loading(self, layer) -> None: ...
    fp8_linear: Incomplete
    def create_weights(
        self,
        layer: torch.nn.Module,
        output_partition_sizes: list[int],
        input_size_per_partition: int,
        params_dtype: torch.dtype,
        weight_loader: Callable,
        **kwargs,
    ): ...
    def apply_weights(
        self, layer: torch.nn.Module, x: torch.Tensor, bias: torch.Tensor | None = None
    ) -> torch.Tensor: ...
