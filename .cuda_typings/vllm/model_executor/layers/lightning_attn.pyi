import torch
from _typeshed import Incomplete
from vllm.triton_utils import tl as tl, triton as triton

class _attention(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, s, kv_history): ...

lightning_attention_: Incomplete

def lightning_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    ed: torch.Tensor,
    block_size: int = 256,
    kv_history: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]: ...
def linear_decode_forward_triton(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    kv_caches: torch.Tensor,
    slope_rate: torch.Tensor,
    slot_idx: torch.Tensor,
    BLOCK_SIZE: int = 32,
) -> torch.Tensor: ...
