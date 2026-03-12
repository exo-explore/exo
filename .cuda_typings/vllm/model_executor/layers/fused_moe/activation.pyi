import torch
from enum import Enum

class MoEActivation(Enum):
    SILU = "silu"
    GELU = "gelu"
    RELU2 = "relu2"
    SWIGLUOAI = "swigluoai"
    SWIGLUSTEP = "swiglustep"
    SILU_NO_MUL = "silu_no_mul"
    GELU_NO_MUL = "gelu_no_mul"
    RELU2_NO_MUL = "relu2_no_mul"
    @property
    def is_gated(self) -> bool: ...
    @property
    def custom_op_name(self) -> str: ...
    def without_mul(self) -> MoEActivation: ...
    @classmethod
    def from_str(cls, s: str) -> MoEActivation: ...

def activation_without_mul(activation: str) -> str: ...
def apply_moe_activation(
    activation: MoEActivation, output: torch.Tensor, input: torch.Tensor
) -> torch.Tensor: ...
