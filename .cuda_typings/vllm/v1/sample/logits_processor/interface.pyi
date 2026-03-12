import abc
import torch
from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass
from enum import Enum
from vllm import SamplingParams as SamplingParams
from vllm.config import VllmConfig as VllmConfig

class MoveDirectionality(Enum):
    UNIDIRECTIONAL = ...
    SWAP = ...

RemovedRequest = int
AddedRequest = tuple[int, SamplingParams, list[int] | None, list[int]]
MovedRequest = tuple[int, int, MoveDirectionality]

@dataclass(frozen=True)
class BatchUpdate:
    batch_size: int
    removed: Sequence[RemovedRequest]
    added: Sequence[AddedRequest]
    moved: Sequence[MovedRequest]

class LogitsProcessor(ABC, metaclass=abc.ABCMeta):
    @classmethod
    def validate_params(cls, sampling_params: SamplingParams): ...
    @abstractmethod
    def __init__(
        self, vllm_config: VllmConfig, device: torch.device, is_pin_memory: bool
    ): ...
    @abstractmethod
    def apply(self, logits: torch.Tensor) -> torch.Tensor: ...
    @abstractmethod
    def is_argmax_invariant(self) -> bool: ...
    @abstractmethod
    def update_state(self, batch_update: BatchUpdate | None) -> None: ...
