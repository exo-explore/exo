from _typeshed import Incomplete
from vllm.platforms import current_platform as current_platform
from vllm.triton_utils import tl as tl, triton as triton

is_hip_: Incomplete
logger: Incomplete

@triton.jit
def tanh(x): ...
def decode_attention_fwd_normal(
    q,
    k_buffer,
    v_buffer,
    o,
    lse,
    req_to_token,
    b_seq_len,
    attn_logits,
    num_kv_splits,
    sm_scale,
    page_size,
    logit_cap: float = 0.0,
) -> None: ...
def decode_attention_fwd_grouped(
    q,
    k_buffer,
    v_buffer,
    o,
    lse,
    req_to_token,
    b_seq_len,
    attn_logits,
    num_kv_splits,
    sm_scale,
    page_size,
    logit_cap: float = 0.0,
) -> None: ...
def decode_attention_fwd(
    q,
    k_buffer,
    v_buffer,
    o,
    lse,
    req_to_token,
    b_seq_len,
    attn_logits,
    num_kv_splits,
    sm_scale,
    page_size: int = 1,
    logit_cap: float = 0.0,
) -> None: ...
