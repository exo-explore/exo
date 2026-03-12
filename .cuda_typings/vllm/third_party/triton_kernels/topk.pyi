import torch
from triton_kernels.tensor import SparseMatrix, Tensor

def make_empty(offset, shape, dtype, device, all_gather): ...
def topk_forward(
    x,
    k,
    apply_softmax: bool = True,
    dim: int = 1,
    y_indx=None,
    n_rows=None,
    all_gather: bool = False,
): ...
def topk_backward(x, y_indx, dy_vals, k, n_rows, apply_softmax): ...

class TopK(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, k, apply_softmax, dim, y_indx, n_rows, all_gather): ...
    @staticmethod
    def backward(ctx, dy_vals, _0, _1): ...

def topk(
    x: Tensor | torch.Tensor,
    k: int,
    apply_softmax: bool = True,
    dim: int = 1,
    y_indx: torch.Tensor | None = None,
    n_rows: int | None = None,
    all_gather: bool = False,
): ...
def topk_torch(
    x,
    k,
    apply_softmax: bool = True,
    dim: int = 1,
    y_indx: torch.Tensor | None = None,
    n_rows: int | None = None,
) -> SparseMatrix: ...
