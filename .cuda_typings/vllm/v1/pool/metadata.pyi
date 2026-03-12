import numpy as np
import torch
from _typeshed import Incomplete
from dataclasses import dataclass
from vllm.pooling_params import PoolingParams as PoolingParams
from vllm.tasks import PoolingTask as PoolingTask
from vllm.utils.platform_utils import is_pin_memory_available as is_pin_memory_available

pin_memory: Incomplete

@dataclass
class PoolingCursor:
    index: list[int]
    first_token_indices_gpu: torch.Tensor
    last_token_indices_gpu: torch.Tensor
    prompt_lens_cpu: torch.Tensor
    seq_lens_cpu: torch.Tensor
    num_scheduled_tokens_cpu: torch.Tensor
    def __getitem__(self, indices: slice): ...
    def is_partial_prefill(self): ...
    def is_finished(self): ...

class PoolingStates:
    hidden_states_cache: list[torch.Tensor]
    def __init__(self) -> None: ...
    def clean(self) -> None: ...

@dataclass
class PoolingMetadata:
    prompt_lens: torch.Tensor
    prompt_token_ids: torch.Tensor | None
    pooling_params: list[PoolingParams]
    pooling_states: list[PoolingStates]
    pooling_cursor: PoolingCursor | None = ...
    tasks = ...
    def __post_init__(self) -> None: ...
    def __getitem__(self, indices: slice): ...
    def get_prompt_token_ids(self) -> list[torch.Tensor]: ...
    def get_pooling_cursor(self) -> PoolingCursor: ...
    def build_pooling_cursor(
        self,
        num_scheduled_tokens_np: np.ndarray,
        seq_lens_cpu: torch.Tensor,
        device: torch.device,
    ): ...
