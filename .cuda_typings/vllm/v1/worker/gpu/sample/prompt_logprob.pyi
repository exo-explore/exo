import numpy as np
import torch
from _typeshed import Incomplete
from collections.abc import Callable as Callable
from vllm.sampling_params import SamplingParams as SamplingParams
from vllm.triton_utils import tl as tl, triton as triton
from vllm.v1.outputs import LogprobsTensors as LogprobsTensors
from vllm.v1.worker.gpu.input_batch import InputBatch as InputBatch
from vllm.v1.worker.gpu.sample.logprob import (
    compute_topk_logprobs as compute_topk_logprobs,
)

class PromptLogprobsWorker:
    max_num_reqs: Incomplete
    uses_prompt_logprobs: Incomplete
    in_progress_prompt_logprobs: dict[str, list[LogprobsTensors]]
    def __init__(self, max_num_reqs: int) -> None: ...
    def add_request(
        self, req_id: str, req_idx: int, sampling_params: SamplingParams
    ): ...
    def remove_request(self, req_id: str) -> None: ...
    def compute_prompt_logprobs(
        self,
        logits_fn: Callable[[torch.Tensor], torch.Tensor],
        hidden_states: torch.Tensor,
        input_batch: InputBatch,
        all_token_ids: torch.Tensor,
        num_computed_tokens: torch.Tensor,
        prompt_lens: np.ndarray,
        prefill_lens: np.ndarray,
        num_computed_prefill_tokens: np.ndarray,
    ) -> dict[str, LogprobsTensors]: ...

def get_prompt_logprobs_token_ids(
    num_tokens: int,
    query_start_loc: torch.Tensor,
    idx_mapping: torch.Tensor,
    num_computed_tokens: torch.Tensor,
    all_token_ids: torch.Tensor,
) -> torch.Tensor: ...
def compute_prompt_logprobs_with_chunking(
    prompt_token_ids: torch.Tensor,
    prompt_hidden_states: torch.Tensor,
    logits_fn: Callable[[torch.Tensor], torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor]: ...
