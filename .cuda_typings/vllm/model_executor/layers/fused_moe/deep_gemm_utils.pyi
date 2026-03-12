import torch
import vllm.model_executor.layers.fused_moe.modular_kernel as mk
from vllm.model_executor.layers.fused_moe.utils import (
    count_expert_num_tokens as count_expert_num_tokens,
)
from vllm.triton_utils import tl as tl, triton as triton
from vllm.utils.deep_gemm import (
    get_mk_alignment_for_contiguous_layout as get_mk_alignment_for_contiguous_layout,
)
from vllm.utils.math_utils import round_up as round_up

def expert_num_tokens_round_up_and_sum(
    expert_num_tokens: torch.Tensor, alignment: int
) -> int: ...
def compute_aligned_M(
    M: int,
    num_topk: int,
    local_num_experts: int,
    alignment: int,
    expert_tokens_meta: mk.ExpertTokensMetadata | None,
): ...
@triton.jit
def apply_expert_map(expert_id, expert_map): ...
@triton.jit
def round_up_128(x: int) -> int: ...
def ep_scatter(
    recv_x: torch.Tensor,
    recv_x_scale: torch.Tensor,
    recv_topk: torch.Tensor,
    num_recv_tokens_per_expert: torch.Tensor,
    expert_map: torch.Tensor | None,
    expert_start_loc: torch.Tensor,
    output_tensor: torch.Tensor,
    output_tensor_scale: torch.Tensor,
    m_indices: torch.Tensor,
    output_index: torch.Tensor,
): ...
def ep_gather(
    input_tensor: torch.Tensor,
    recv_topk_ids: torch.Tensor,
    recv_topk_weight: torch.Tensor,
    input_index: torch.Tensor,
    expert_map: torch.Tensor | None,
    output_tensor: torch.Tensor,
): ...
def deepgemm_moe_permute(
    aq: torch.Tensor,
    aq_scale: torch.Tensor,
    topk_ids: torch.Tensor,
    local_num_experts: int,
    expert_map: torch.Tensor | None,
    expert_tokens_meta: mk.ExpertTokensMetadata | None,
    aq_out: torch.Tensor | None = None,
): ...
def deepgemm_unpermute_and_reduce(
    a: torch.Tensor,
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    inv_perm: torch.Tensor,
    expert_map: torch.Tensor | None,
    output: torch.Tensor,
): ...
