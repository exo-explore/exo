import torch
import torch.nn as nn
from _typeshed import Incomplete
from collections.abc import Sequence
from vllm.logger import init_logger as init_logger
from vllm.triton_utils import tl as tl, triton as triton
from vllm.v1.outputs import (
    LogprobsLists as LogprobsLists,
    LogprobsTensors as LogprobsTensors,
    SamplerOutput as SamplerOutput,
)
from vllm.v1.sample.logits_processor.builtin import (
    MinTokensLogitsProcessor as MinTokensLogitsProcessor,
)
from vllm.v1.sample.metadata import SamplingMetadata as SamplingMetadata
from vllm.v1.sample.ops.bad_words import (
    apply_bad_words_with_drafts as apply_bad_words_with_drafts,
)
from vllm.v1.sample.ops.penalties import apply_all_penalties as apply_all_penalties
from vllm.v1.sample.ops.topk_topp_sampler import apply_top_k_top_p as apply_top_k_top_p
from vllm.v1.sample.sampler import Sampler as Sampler
from vllm.v1.spec_decode.metadata import SpecDecodeMetadata as SpecDecodeMetadata

logger: Incomplete
PLACEHOLDER_TOKEN_ID: tl.constexpr
GREEDY_TEMPERATURE: tl.constexpr
MAX_SPEC_LEN: int

class RejectionSampler(nn.Module):
    sampler: Incomplete
    is_processed_logprobs_mode: Incomplete
    is_logits_logprobs_mode: Incomplete
    def __init__(self, sampler: Sampler) -> None: ...
    def forward(
        self,
        metadata: SpecDecodeMetadata,
        draft_probs: torch.Tensor | None,
        logits: torch.Tensor,
        sampling_metadata: SamplingMetadata,
    ) -> SamplerOutput: ...
    @staticmethod
    def parse_output(
        output_token_ids: torch.Tensor,
        vocab_size: int,
        discard_req_indices: Sequence[int] = (),
        logprobs_tensors: LogprobsTensors | None = None,
    ) -> tuple[list[list[int]], LogprobsLists | None]: ...
    def apply_logits_processors(
        self,
        logits: torch.Tensor,
        sampling_metadata: SamplingMetadata,
        metadata: SpecDecodeMetadata,
    ) -> torch.Tensor: ...
    @staticmethod
    def apply_penalties(
        logits: torch.Tensor,
        sampling_metadata: SamplingMetadata,
        metadata: SpecDecodeMetadata,
        repeat_indices: torch.Tensor,
        output_token_ids: list[list[int]],
    ) -> torch.Tensor: ...

def rejection_sample(
    draft_token_ids: torch.Tensor,
    num_draft_tokens: list[int],
    max_spec_len: int,
    cu_num_draft_tokens: torch.Tensor,
    draft_probs: torch.Tensor | None,
    target_logits: torch.Tensor,
    bonus_token_ids: torch.Tensor,
    sampling_metadata: SamplingMetadata,
) -> torch.Tensor: ...
def apply_sampling_constraints(
    logits: torch.Tensor,
    cu_num_draft_tokens: torch.Tensor,
    sampling_metadata: SamplingMetadata,
) -> torch.Tensor: ...
def expand_batch_to_tokens(
    x: torch.Tensor,
    cu_num_tokens: torch.Tensor,
    num_tokens: int,
    replace_from: int = 0,
    replace_to: int = 0,
) -> torch.Tensor: ...
def generate_uniform_probs(
    num_tokens: int,
    num_draft_tokens: list[int],
    generators: dict[int, torch.Generator],
    device: torch.device,
) -> torch.Tensor: ...
def sample_recovered_tokens(
    max_spec_len: int,
    num_draft_tokens: list[int],
    cu_num_draft_tokens: torch.Tensor,
    draft_token_ids: torch.Tensor,
    draft_probs: torch.Tensor | None,
    target_probs: torch.Tensor,
    sampling_metadata: SamplingMetadata,
    device: torch.device,
) -> torch.Tensor: ...
def rejection_greedy_sample_kernel(
    output_token_ids_ptr,
    cu_num_draft_tokens_ptr,
    draft_token_ids_ptr,
    target_argmax_ptr,
    bonus_token_ids_ptr,
    is_greedy_ptr,
    max_spec_len,
) -> None: ...
def rejection_random_sample_kernel(
    output_token_ids_ptr,
    cu_num_draft_tokens_ptr,
    draft_token_ids_ptr,
    draft_probs_ptr,
    target_probs_ptr,
    bonus_token_ids_ptr,
    recovered_token_ids_ptr,
    uniform_probs_ptr,
    is_greedy_ptr,
    max_spec_len,
    vocab_size,
    NO_DRAFT_PROBS: tl.constexpr,
): ...
def expand_kernel(
    output_ptr,
    input_ptr,
    cu_num_tokens_ptr,
    replace_from,
    replace_to,
    MAX_NUM_TOKENS: tl.constexpr,
): ...
@triton.jit
def sample_recovered_tokens_kernel(
    output_token_ids_ptr,
    cu_num_draft_tokens_ptr,
    draft_token_ids_ptr,
    draft_probs_ptr,
    target_probs_ptr,
    inv_q_ptr,
    vocab_size,
    BLOCK_SIZE: tl.constexpr,
    NO_DRAFT_PROBS: tl.constexpr,
): ...
