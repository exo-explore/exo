import torch
from _typeshed import Incomplete
from torch.nn.parameter import Parameter
from vllm.model_executor.custom_op import PluggableLayer as PluggableLayer
from vllm.model_executor.layers.linear import ReplicatedLinear as ReplicatedLinear
from vllm.platforms import current_platform as current_platform

class GateLinear(ReplicatedLinear):
    DSV3_SUPPORTED_NUM_EXPERTS: Incomplete
    DSV3_SUPPORTED_HIDDEN_SIZES: Incomplete
    out_dtype: Incomplete
    allow_specialized_router_gemm: Incomplete
    allow_dsv3_router_gemm: Incomplete
    allow_cublas_router_gemm: Incomplete
    def __init__(
        self,
        input_size: int,
        output_size: int,
        bias: bool = False,
        out_dtype: torch.dtype | None = None,
        params_dtype: torch.dtype | None = None,
        force_fp32_compute: bool = False,
        prefix: str = "",
    ) -> None: ...
    def set_out_dtype(self, out_dtype: torch.dtype) -> None: ...
    def forward(
        self, x: torch.Tensor
    ) -> torch.Tensor | tuple[torch.Tensor, Parameter | None]: ...
