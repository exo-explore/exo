import numpy as np
import torch
from _typeshed import Incomplete
from collections.abc import Sequence as GenericSequence
from dataclasses import dataclass
from typing import Any, Generic
from vllm.logger import init_logger as init_logger
from vllm.logprobs import (
    PromptLogprobs as PromptLogprobs,
    SampleLogprobs as SampleLogprobs,
)
from vllm.lora.request import LoRARequest as LoRARequest
from vllm.v1.metrics.stats import RequestStateStats as RequestStateStats

logger: Incomplete

@dataclass
class CompletionOutput:
    index: int
    text: str
    token_ids: GenericSequence[int]
    cumulative_logprob: float | None
    logprobs: SampleLogprobs | None
    routed_experts: np.ndarray | None = ...
    finish_reason: str | None = ...
    stop_reason: int | str | None = ...
    lora_request: LoRARequest | None = ...
    def finished(self) -> bool: ...

@dataclass
class PoolingOutput:
    data: torch.Tensor
    def __eq__(self, other: object) -> bool: ...

class RequestOutput:
    request_id: str
    prompt: str | None
    prompt_token_ids: list[int] | None
    prompt_logprobs: PromptLogprobs | None
    outputs: list[CompletionOutput]
    finished: bool
    metrics: RequestStateStats | None
    lora_request: LoRARequest | None
    encoder_prompt: str | None
    encoder_prompt_token_ids: list[int] | None
    num_cached_tokens: int | None
    kv_transfer_params: dict[str, Any] | None
    def __init__(
        self,
        request_id: str,
        prompt: str | None,
        prompt_token_ids: list[int] | None,
        prompt_logprobs: PromptLogprobs | None,
        outputs: list[CompletionOutput],
        finished: bool,
        metrics: RequestStateStats | None = None,
        lora_request: LoRARequest | None = None,
        encoder_prompt: str | None = None,
        encoder_prompt_token_ids: list[int] | None = None,
        num_cached_tokens: int | None = None,
        *,
        kv_transfer_params: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None: ...
    def add(self, next_output: RequestOutput, aggregate: bool) -> None: ...

STREAM_FINISHED: Incomplete

class PoolingRequestOutput(Generic[_O]):
    request_id: Incomplete
    prompt_token_ids: Incomplete
    num_cached_tokens: Incomplete
    finished: Incomplete
    outputs: Incomplete
    def __init__(
        self,
        request_id: str,
        outputs: _O,
        prompt_token_ids: list[int],
        num_cached_tokens: int,
        finished: bool,
    ) -> None: ...

@dataclass
class EmbeddingOutput:
    embedding: list[float]
    @staticmethod
    def from_base(pooling_output: PoolingOutput): ...
    @property
    def hidden_size(self) -> int: ...

class EmbeddingRequestOutput(PoolingRequestOutput[EmbeddingOutput]):
    @staticmethod
    def from_base(request_output: PoolingRequestOutput): ...

@dataclass
class ClassificationOutput:
    probs: list[float]
    @staticmethod
    def from_base(pooling_output: PoolingOutput): ...
    @property
    def num_classes(self) -> int: ...

class ClassificationRequestOutput(PoolingRequestOutput[ClassificationOutput]):
    @staticmethod
    def from_base(request_output: PoolingRequestOutput): ...

@dataclass
class ScoringOutput:
    score: float
    @staticmethod
    def from_base(pooling_output: PoolingOutput): ...

class ScoringRequestOutput(PoolingRequestOutput[ScoringOutput]):
    @staticmethod
    def from_base(request_output: PoolingRequestOutput): ...
