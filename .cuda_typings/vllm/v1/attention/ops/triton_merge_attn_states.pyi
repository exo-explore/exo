import torch
from vllm.triton_utils import tl as tl, triton as triton

def merge_attn_states(
    output: torch.Tensor,
    prefix_output: torch.Tensor,
    prefix_lse: torch.Tensor,
    suffix_output: torch.Tensor,
    suffix_lse: torch.Tensor,
    output_lse: torch.Tensor | None = None,
) -> None: ...
@triton.jit
def merge_attn_states_kernel(
    output,
    output_lse,
    prefix_output,
    prefix_lse,
    suffix_output,
    suffix_lse,
    prefix_head_stride,
    output_head_stride,
    HEAD_SIZE: tl.constexpr,
    PADDED_HEAD_SIZE: tl.constexpr,
    OUTPUT_LSE: tl.constexpr,
): ...
