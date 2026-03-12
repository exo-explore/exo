import torch
from _typeshed import Incomplete
from typing import Literal
from vllm.model_executor.custom_op import CustomOp as CustomOp
from vllm.utils.torch_utils import is_torch_equal as is_torch_equal

class ConvLayerBase(CustomOp):
    num_dim: int
    in_channels: Incomplete
    out_channels: Incomplete
    kernel_size: Incomplete
    stride: Incomplete
    padding: Incomplete
    dilation: Incomplete
    groups: Incomplete
    padding_mode: Incomplete
    enable_linear: Incomplete
    input_size: Incomplete
    weight: Incomplete
    bias: Incomplete
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int | tuple[int, ...],
        stride: int | tuple[int, ...] = 1,
        padding: int | tuple[int, ...] | Literal["same", "valid"] = 0,
        dilation: int | tuple[int, ...] = 1,
        groups: int = 1,
        bias: bool = True,
        padding_mode: Literal["zeros", "reflect", "replicate", "circular"] = "zeros",
        *,
        params_dtype: torch.dtype | None = None,
    ) -> None: ...
    def extra_repr(self) -> str: ...

class Conv2dLayer(ConvLayerBase):
    num_dim: int
    def forward_native(self, x: torch.Tensor) -> torch.Tensor: ...
    def forward_cuda(self, x: torch.Tensor) -> torch.Tensor: ...

class CausalConv2dLayer(Conv2dLayer):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int,
        padding: int = 0,
        dilation: int = 1,
        groups: int = 1,
        bias: bool = True,
        padding_mode: str = "zeros",
        *,
        params_dtype: torch.dtype | None = None,
    ) -> None: ...
    def forward(self, x: torch.Tensor) -> torch.Tensor: ...

class Conv3dLayer(ConvLayerBase):
    num_dim: int
    def forward_native(self, x: torch.Tensor) -> torch.Tensor: ...
    def forward_cuda(self, x: torch.Tensor) -> torch.Tensor: ...
