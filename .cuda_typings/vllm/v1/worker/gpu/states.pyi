import numpy as np
import torch
from _typeshed import Incomplete
from vllm.v1.worker.gpu.buffer_utils import (
    StagedWriteTensor as StagedWriteTensor,
    UvaBackedTensor as UvaBackedTensor,
)

class RequestState:
    max_num_reqs: Incomplete
    max_model_len: Incomplete
    max_num_batched_tokens: Incomplete
    num_speculative_steps: Incomplete
    vocab_size: Incomplete
    device: Incomplete
    req_id_to_index: dict[str, int]
    index_to_req_id: dict[int, str]
    free_indices: Incomplete
    all_token_ids: Incomplete
    prompt_len: Incomplete
    prefill_len: Incomplete
    total_len: Incomplete
    num_computed_prefill_tokens: Incomplete
    num_computed_tokens: Incomplete
    last_sampled_tokens: Incomplete
    draft_tokens: Incomplete
    next_prefill_tokens: Incomplete
    def __init__(
        self,
        max_num_reqs: int,
        max_model_len: int,
        max_num_batched_tokens: int,
        num_speculative_steps: int,
        vocab_size: int,
        device: torch.device,
    ) -> None: ...
    @property
    def num_reqs(self) -> int: ...
    def add_request(
        self,
        req_id: str,
        prompt_len: int,
        all_token_ids: list[int],
        num_computed_tokens: int,
    ) -> None: ...
    def apply_staged_writes(self) -> None: ...
    def remove_request(self, req_id: str) -> None: ...
    def any_prefills(self, idx_mapping_np: np.ndarray) -> bool: ...
