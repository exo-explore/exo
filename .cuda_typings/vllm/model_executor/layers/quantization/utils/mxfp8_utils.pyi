import torch
from _typeshed import Incomplete
from enum import Enum
from vllm.logger import init_logger as init_logger
from vllm.utils.torch_utils import (
    direct_register_custom_op as direct_register_custom_op,
)

logger: Incomplete

class Mxfp8LinearBackend(Enum):
    EMULATION = "emulation"
    FLASHINFER_CUTLASS = "flashinfer-cutlass"

MXFP8_VALUE_DTYPE: Incomplete
MXFP8_SCALE_DTYPE: Incomplete
MXFP8_BLOCK_SIZE: int

def swizzle_mxfp8_scale(sf: torch.Tensor, M: int, K: int) -> torch.Tensor: ...
def mxfp8_e4m3_quantize(
    x: torch.Tensor, is_sf_swizzled_layout: bool = False
) -> tuple[torch.Tensor, torch.Tensor]: ...
def dequant_mxfp8_to_bf16(x: torch.Tensor, scales: torch.Tensor) -> torch.Tensor: ...
def mxfp8_e4m3_quantize_fake(
    x: torch.Tensor, is_sf_swizzled_layout: bool = False
) -> tuple[torch.Tensor, torch.Tensor]: ...

class Mxfp8LinearOp:
    backend: Incomplete
    def __init__(self, backend: Mxfp8LinearBackend) -> None: ...
    def apply(
        self,
        input: torch.Tensor,
        weight: torch.Tensor,
        weight_scale: torch.Tensor,
        out_dtype: torch.dtype,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor: ...
