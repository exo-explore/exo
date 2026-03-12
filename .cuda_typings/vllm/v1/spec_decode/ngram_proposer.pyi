import numpy as np
import torch
from _typeshed import Incomplete
from vllm.config import VllmConfig as VllmConfig

class NgramProposer:
    min_n: Incomplete
    max_n: Incomplete
    k: Incomplete
    max_model_len: Incomplete
    valid_ngram_draft: Incomplete
    valid_ngram_num_drafts: Incomplete
    num_tokens_threshold: int
    num_numba_thread_available: Incomplete
    def __init__(self, vllm_config: VllmConfig) -> None: ...
    def batch_propose(
        self,
        num_requests: int,
        valid_ngram_requests: list,
        num_tokens_no_spec: np.ndarray,
        token_ids_cpu: np.ndarray,
    ) -> list[list[int]]: ...
    def propose(
        self,
        sampled_token_ids: list[list[int]],
        num_tokens_no_spec: np.ndarray,
        token_ids_cpu: np.ndarray,
        slot_mappings: dict[str, torch.Tensor]
        | list[dict[str, torch.Tensor]]
        | None = None,
    ) -> list[list[int]]: ...
    def load_model(self, *args, **kwargs) -> None: ...

def batch_propose_numba(
    valid_ngram_requests: list,
    num_tokens_no_spec: np.ndarray,
    token_ids_cpu: np.ndarray,
    min_n: int,
    max_n: int,
    max_model_len: int,
    k: int,
    valid_ngram_draft: np.ndarray,
    valid_ngram_num_drafts: np.ndarray,
): ...
