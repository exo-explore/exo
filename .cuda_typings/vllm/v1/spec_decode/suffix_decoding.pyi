import torch
from _typeshed import Incomplete
from vllm.config import VllmConfig as VllmConfig
from vllm.v1.worker.gpu_input_batch import InputBatch as InputBatch

class SuffixDecodingProposer:
    num_speculative_tokens: Incomplete
    max_tree_depth: Incomplete
    max_spec_factor: Incomplete
    min_token_prob: Incomplete
    max_model_len: Incomplete
    suffix_cache: Incomplete
    def __init__(self, vllm_config: VllmConfig) -> None: ...
    def propose(
        self,
        input_batch: InputBatch,
        sampled_token_ids: list[list[int]],
        slot_mappings: dict[str, torch.Tensor]
        | list[dict[str, torch.Tensor]]
        | None = None,
    ) -> list[list[int]]: ...
    def load_model(self, *args, **kwargs) -> None: ...
