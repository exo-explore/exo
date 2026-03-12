from _typeshed import Incomplete
from vllm.triton_utils import triton as triton

TRITON_22: Incomplete

def is_int_pow_2(n): ...
def mamba_chunk_scan_combined_varlen(
    x,
    dt,
    A,
    B,
    C,
    chunk_size,
    cu_seqlens,
    cu_chunk_seqlens,
    last_chunk_indices,
    seq_idx,
    out,
    D=None,
    z=None,
    dt_bias=None,
    initial_states=None,
    dt_softplus: bool = False,
    dt_limit=...,
    return_intermediate_states: bool = False,
    state_dtype=None,
): ...
