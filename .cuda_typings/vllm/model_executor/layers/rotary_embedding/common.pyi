import torch
from _typeshed import Incomplete
from vllm.logger import init_logger as init_logger
from vllm.model_executor.custom_op import CustomOp as CustomOp
from vllm.utils.torch_utils import (
    direct_register_custom_op as direct_register_custom_op,
)

logger: Incomplete

def rotate_neox(x: torch.Tensor) -> torch.Tensor: ...
def rotate_gptj(x: torch.Tensor) -> torch.Tensor: ...
def yarn_find_correction_dim(
    num_rotations: int,
    dim: int,
    base: float = 10000,
    max_position_embeddings: int = 2048,
) -> float: ...
def yarn_find_correction_range(
    low_rot: int,
    high_rot: int,
    dim: int,
    base: float = 10000,
    max_position_embeddings: int = 2048,
    truncate: bool = True,
) -> tuple[float | int, float | int]: ...
def yarn_linear_ramp_mask(
    low: float, high: float, dim: int, dtype: torch.dtype
) -> torch.Tensor: ...
def yarn_get_mscale(scale: float = 1) -> float: ...

class ApplyRotaryEmb(CustomOp):
    is_neox_style: Incomplete
    enable_fp32_compute: Incomplete
    apply_rotary_emb_flash_attn: Incomplete
    def __init__(
        self,
        enforce_enable: bool = False,
        is_neox_style: bool = True,
        enable_fp32_compute: bool = False,
    ) -> None: ...
    @staticmethod
    def forward_static(
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        is_neox_style: bool = True,
        enable_fp32_compute: bool = False,
    ) -> torch.Tensor: ...
    def forward_native(
        self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
    ) -> torch.Tensor: ...
    def forward_cuda(
        self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
    ) -> torch.Tensor: ...
    def forward_hip(
        self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
    ) -> torch.Tensor: ...
    def forward_cpu(
        self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
    ) -> torch.Tensor: ...
    def extra_repr(self) -> str: ...
