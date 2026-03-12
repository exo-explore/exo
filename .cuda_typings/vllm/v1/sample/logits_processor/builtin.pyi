import torch
from _typeshed import Incomplete
from collections.abc import Callable as Callable, Sequence
from typing import TypeVar
from vllm import SamplingParams as SamplingParams
from vllm.config import VllmConfig as VllmConfig
from vllm.v1.sample.logits_processor.interface import (
    BatchUpdate as BatchUpdate,
    LogitsProcessor as LogitsProcessor,
    MoveDirectionality as MoveDirectionality,
)

T = TypeVar("T")

class MinPLogitsProcessor(LogitsProcessor):
    min_p_count: int
    min_p_cpu_tensor: Incomplete
    min_p_cpu: Incomplete
    use_double_tensor: Incomplete
    min_p_device: torch.Tensor
    min_p: torch.Tensor
    def __init__(
        self, vllm_config: VllmConfig, device: torch.device, is_pin_memory: bool
    ) -> None: ...
    def is_argmax_invariant(self) -> bool: ...
    def get_min_p_by_index(self, index: int) -> float: ...
    def update_state(self, batch_update: BatchUpdate | None): ...
    def apply(self, logits: torch.Tensor) -> torch.Tensor: ...

class LogitBiasLogitsProcessor(LogitsProcessor):
    device: Incomplete
    pin_memory: Incomplete
    biases: dict[int, dict[int, float]]
    bias_tensor: torch.Tensor
    logits_slice: Incomplete
    def __init__(self, _, device: torch.device, is_pin_memory: bool) -> None: ...
    def is_argmax_invariant(self) -> bool: ...
    def update_state(self, batch_update: BatchUpdate | None): ...
    def apply(self, logits: torch.Tensor) -> torch.Tensor: ...

class MinTokensLogitsProcessor(LogitsProcessor):
    device: Incomplete
    pin_memory: Incomplete
    min_toks: dict[int, tuple[int, Sequence[int], set[int]]]
    logits_slice: tuple[torch.Tensor, torch.Tensor]
    neg_inf_tensor: Incomplete
    def __init__(
        self, vllm_config: VllmConfig, device: torch.device, is_pin_memory: bool
    ) -> None: ...
    def is_argmax_invariant(self) -> bool: ...
    @staticmethod
    def add_request(
        params: SamplingParams, _: list[int] | None, output_tok_ids: list[int]
    ) -> tuple[int, Sequence[int], set[int]] | None: ...
    def update_state(self, batch_update: BatchUpdate | None): ...
    def apply(self, logits: torch.Tensor) -> torch.Tensor: ...
    def apply_with_spec_decode(
        self, logits: torch.Tensor, num_draft_tokens: list[int]
    ) -> torch.Tensor: ...

def process_dict_updates(
    req_entries: dict[int, T],
    batch_update: BatchUpdate | None,
    new_state: Callable[[SamplingParams, list[int] | None, list[int]], T | None],
) -> bool: ...
