import torch
from _typeshed import Incomplete
from vllm.distributed.parallel_state import GroupCoordinator as GroupCoordinator
from vllm.triton_utils import tl as tl, triton as triton

class CPTritonContext:
    inner_kernel: Incomplete
    def __init__(self) -> None: ...
    def call_kernel(self, kernel, grid, *regular_args, **const_args) -> None: ...

def correct_attn_out(
    out: torch.Tensor,
    lses: torch.Tensor,
    cp_rank: int,
    ctx: CPTritonContext,
    is_lse_base_on_e: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]: ...
def cp_lse_ag_out_rs(
    cp_attn_out: torch.Tensor,
    cp_attn_lse: torch.Tensor,
    cp_group: GroupCoordinator,
    ctx: CPTritonContext | None = None,
    return_lse: bool = False,
    is_lse_base_on_e: bool = True,
): ...
def cp_lse_ag_out_ar(
    cp_attn_out: torch.Tensor,
    cp_attn_lse: torch.Tensor,
    cp_group: GroupCoordinator,
    ctx: CPTritonContext | None = None,
    return_lse: bool = False,
    is_lse_base_on_e: bool = True,
): ...
def pack_seq_triton(
    x: torch.Tensor,
    lengths: torch.Tensor,
    pad_value: float = ...,
    block_t: int = 64,
    block_d: int = 64,
) -> torch.Tensor: ...
def unpack_seq_triton(
    packed_tensor: torch.Tensor,
    lengths: torch.Tensor,
    block_t: int = 64,
    block_d: int = 64,
) -> torch.Tensor: ...
