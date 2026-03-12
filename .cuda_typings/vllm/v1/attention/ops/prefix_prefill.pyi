import torch
from _typeshed import Incomplete
from vllm.platforms import current_platform as current_platform
from vllm.triton_utils import tl as tl, triton as triton

BASE_BLOCK: Incomplete
NUM_WARPS: Incomplete
IS_TURING: Incomplete
float8_info: Incomplete

def context_attention_fwd(
    q,
    k,
    v,
    o,
    kv_cache_dtype: str,
    k_cache,
    v_cache,
    b_loc,
    b_start_loc,
    b_seq_len,
    max_seq_len,
    max_input_len,
    k_scale: torch.Tensor,
    v_scale: torch.Tensor,
    alibi_slopes=None,
    sliding_window=None,
    sm_scale=None,
    skip_decode: bool = False,
    fp8_out_scale=None,
    sinks=None,
    is_block_table_ptr: bool = False,
): ...
