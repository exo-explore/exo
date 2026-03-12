import numpy as np
import torch
from _typeshed import Incomplete
from vllm.v1.outputs import DraftTokenIds as DraftTokenIds
from vllm.v1.worker.gpu.async_utils import async_copy_to_np as async_copy_to_np
from vllm.v1.worker.gpu.input_batch import InputBatch as InputBatch

class DraftTokensHandler:
    device: Incomplete
    copy_stream: Incomplete
    copy_event: Incomplete
    req_ids: list[str]
    draft_tokens_np: np.ndarray | None
    num_draft_tokens: int
    def __init__(self, device: torch.device | None = None) -> None: ...
    def set_draft_tokens(
        self, input_batch: InputBatch, draft_tokens: torch.Tensor
    ) -> None: ...
    def get_draft_tokens(self) -> DraftTokenIds | None: ...
