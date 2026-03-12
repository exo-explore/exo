import torch
from _typeshed import Incomplete
from vllm.model_executor.layers.mamba.ops.triton_helpers import fast_exp as fast_exp
from vllm.triton_utils import HAS_TRITON as HAS_TRITON, tl as tl, triton as triton
from vllm.v1.attention.backends.utils import PAD_SLOT_ID as PAD_SLOT_ID

TRITON3: Incomplete

@triton.jit
def softplus(dt): ...
def selective_state_update(
    state,
    x,
    dt,
    A,
    B,
    C,
    D=None,
    z=None,
    dt_bias=None,
    dt_softplus: bool = False,
    state_batch_indices=None,
    dst_state_batch_indices=None,
    pad_slot_id=...,
    out=None,
    num_accepted_tokens=None,
    cu_seqlens=None,
    is_blackwell: bool = False,
): ...
def selective_scan_fn(
    u,
    ssm_states,
    delta,
    A,
    B,
    C,
    D=None,
    z=None,
    delta_bias=None,
    delta_softplus: bool = False,
    query_start_loc=None,
    cache_indices=None,
    has_initial_state=None,
    pad_slot_id=...,
    block_size: int = 1024,
    block_idx_first_scheduled_token=None,
    block_idx_last_scheduled_token=None,
    initial_state_idx=None,
    cu_chunk_seqlen=None,
    last_chunk_indices=None,
) -> torch.Tensor: ...
