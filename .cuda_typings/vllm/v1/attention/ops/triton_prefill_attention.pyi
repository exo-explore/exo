import torch
from vllm.platforms import current_platform as current_platform
from vllm.triton_utils import tl as tl, triton as triton
from vllm.utils.math_utils import RCP_LN2 as RCP_LN2

def get_block_size(dtype: torch.dtype) -> int: ...
def context_attention_fwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    o: torch.Tensor,
    b_start_loc: torch.Tensor,
    b_seq_len: torch.Tensor,
    max_input_len: int,
    is_causal: bool = True,
    softmax_scale: float | None = None,
    sliding_window_q: int | None = None,
    sliding_window_k: int | None = None,
): ...
