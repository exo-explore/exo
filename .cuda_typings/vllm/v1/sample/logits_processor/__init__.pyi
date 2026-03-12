import abc
import torch
from abc import abstractmethod
from collections.abc import Sequence
from functools import partial
from vllm.config import VllmConfig
from vllm.logits_process import LogitsProcessor as RequestLogitsProcessor
from vllm.sampling_params import SamplingParams
from vllm.v1.sample.logits_processor.builtin import (
    LogitBiasLogitsProcessor as LogitBiasLogitsProcessor,
    MinPLogitsProcessor as MinPLogitsProcessor,
    MinTokensLogitsProcessor as MinTokensLogitsProcessor,
)
from vllm.v1.sample.logits_processor.interface import (
    BatchUpdate as BatchUpdate,
    LogitsProcessor as LogitsProcessor,
    MoveDirectionality as MoveDirectionality,
)
from vllm.v1.sample.logits_processor.state import (
    BatchUpdateBuilder as BatchUpdateBuilder,
    LogitsProcessors as LogitsProcessors,
)

__all__ = [
    "LogitsProcessor",
    "LogitBiasLogitsProcessor",
    "MinPLogitsProcessor",
    "MinTokensLogitsProcessor",
    "BatchUpdate",
    "BatchUpdateBuilder",
    "MoveDirectionality",
    "LogitsProcessors",
    "build_logitsprocs",
    "STR_POOLING_REJECTS_LOGITSPROCS",
    "LOGITSPROCS_GROUP",
    "AdapterLogitsProcessor",
]

STR_POOLING_REJECTS_LOGITSPROCS: str
LOGITSPROCS_GROUP: str

def build_logitsprocs(
    vllm_config: VllmConfig,
    device: torch.device,
    is_pin_memory: bool,
    is_pooling_model: bool,
    custom_logitsprocs: Sequence[str | type[LogitsProcessor]] = (),
) -> LogitsProcessors: ...

class AdapterLogitsProcessor(LogitsProcessor, metaclass=abc.ABCMeta):
    req_info: dict[int, partial[torch.Tensor]]
    def __init__(
        self, vllm_config: VllmConfig, device: torch.device, is_pin_memory: bool
    ) -> None: ...
    @abstractmethod
    def new_req_logits_processor(
        self, params: SamplingParams
    ) -> RequestLogitsProcessor | None: ...
    def update_state(self, batch_update: BatchUpdate | None): ...
    def apply(self, logits: torch.Tensor) -> torch.Tensor: ...
