import torch
from .specialize import (
    ClosureArg as ClosureArg,
    FnSpecs as FnSpecs,
    SpecializationModule as SpecializationModule,
)
from _typeshed import Incomplete
from dataclasses import dataclass
from triton_kernels.numerics import InFlexData, OutFlexData

@dataclass(frozen=True)
class PostprocessFn:
    specs: FnSpecs = ...
    fn_args: tuple[object] = ...

specializations: Incomplete

def reduce(
    x: torch.Tensor,
    dim: int,
    mask: torch.Tensor | None = None,
    scale: torch.Tensor | None = None,
    x_mxscale: torch.Tensor | None = None,
    x_flex: InFlexData | None = ...,
    y_dtype: torch.dtype | None = None,
    y_flex: OutFlexData | None = ...,
    y_flex_saturate_inf: bool = False,
    y_has_mx: bool | None = None,
    y: torch.Tensor | None = None,
    postprocess_fn1: PostprocessFn | None = None,
    postprocess_fn2: PostprocessFn | None = None,
) -> tuple[torch.Tensor, torch.Tensor | None]: ...
def compute_actual_scale(x, dtype, per_batch_scale: bool = False): ...
def reduce_torch(
    x: torch.Tensor,
    dim: int,
    mask: torch.Tensor | None = None,
    scale: torch.Tensor | None = None,
    x_mxscale: torch.Tensor | None = None,
    x_flex: InFlexData | None = ...,
    y_flex: OutFlexData | None = ...,
    y_flex_saturate_inf: bool = False,
    postprocess_fn1: callable | None = None,
): ...
