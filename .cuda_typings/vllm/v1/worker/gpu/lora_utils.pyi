import numpy as np
from _typeshed import Incomplete
from vllm.lora.request import LoRARequest as LoRARequest

NO_LORA_ID: int

class LoraState:
    lora_ids: Incomplete
    lora_requests: dict[str, LoRARequest]
    def __init__(self, max_num_reqs: int) -> None: ...
    def add_request(
        self, req_id: str, req_index: int, lora_request: LoRARequest | None
    ) -> None: ...
    def remove_request(self, req_id: str) -> None: ...
    def make_lora_inputs(
        self,
        req_ids: list[str],
        idx_mapping: np.ndarray,
        num_scheduled_tokens: np.ndarray,
    ) -> tuple[tuple[int, ...], tuple[int, ...], set[LoRARequest]]: ...
