import torch
from _typeshed import Incomplete
from vllm.model_executor.models.interfaces import SupportsMRoPE as SupportsMRoPE
from vllm.triton_utils import tl as tl, triton as triton
from vllm.v1.worker.gpu.buffer_utils import (
    StagedWriteTensor as StagedWriteTensor,
    UvaBackedTensor as UvaBackedTensor,
)

class MRopeState:
    max_num_reqs: Incomplete
    max_num_tokens: Incomplete
    max_model_len: Incomplete
    device: Incomplete
    prefill_mrope_positions: Incomplete
    prefill_mrope_delta: Incomplete
    mrope_positions: Incomplete
    def __init__(
        self,
        max_num_reqs: int,
        max_num_tokens: int,
        max_model_len: int,
        device: torch.device,
    ) -> None: ...
    def init_prefill_mrope_positions(
        self,
        req_idx: int,
        mrope_model: SupportsMRoPE,
        prefill_token_ids: list[int],
        mm_features: list,
    ) -> None: ...
    def apply_staged_writes(self) -> None: ...
    def prepare_mrope_positions(
        self,
        idx_mapping: torch.Tensor,
        query_start_loc: torch.Tensor,
        prefill_lens: torch.Tensor,
        num_computed_tokens: torch.Tensor,
    ) -> None: ...
