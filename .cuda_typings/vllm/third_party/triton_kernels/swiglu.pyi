import torch
from _typeshed import Incomplete
from dataclasses import dataclass
from triton_kernels.numerics import InFlexData, OutFlexData

@dataclass(frozen=True)
class FlexCtx:
    out_data: OutFlexData = ...
    inp_data: InFlexData = ...
    saturate_inf: bool = ...

@dataclass(frozen=True)
class PrecisionConfig:
    limit: float
    flex_ctx: FlexCtx = ...

swiglu_fn: Incomplete

class SwiGLU(torch.autograd.Function):
    @staticmethod
    def forward(ctx, a, alpha, precision_config, routing_data): ...

def swiglu(a, alpha, precision_config, routing_data=None): ...
def swiglu_torch(a, alpha, precision_config): ...
