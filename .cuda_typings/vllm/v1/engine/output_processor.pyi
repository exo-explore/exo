import numpy as np
import torch
from _typeshed import Incomplete
from collections import defaultdict, deque
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any
from vllm.lora.request import LoRARequest as LoRARequest
from vllm.outputs import (
    CompletionOutput as CompletionOutput,
    PoolingOutput as PoolingOutput,
    PoolingRequestOutput as PoolingRequestOutput,
    RequestOutput as RequestOutput,
    STREAM_FINISHED as STREAM_FINISHED,
)
from vllm.sampling_params import RequestOutputKind as RequestOutputKind
from vllm.tokenizers import TokenizerLike as TokenizerLike
from vllm.tracing import (
    SpanAttributes as SpanAttributes,
    SpanKind as SpanKind,
    extract_trace_context as extract_trace_context,
    instrument_manual as instrument_manual,
)
from vllm.utils import (
    length_from_prompt_token_ids_or_embeds as length_from_prompt_token_ids_or_embeds,
)
from vllm.v1.engine import (
    EngineCoreOutput as EngineCoreOutput,
    EngineCoreRequest as EngineCoreRequest,
    FinishReason as FinishReason,
)
from vllm.v1.engine.detokenizer import IncrementalDetokenizer as IncrementalDetokenizer
from vllm.v1.engine.logprobs import LogprobsProcessor as LogprobsProcessor
from vllm.v1.engine.parallel_sampling import ParentRequest as ParentRequest
from vllm.v1.metrics.stats import (
    IterationStats as IterationStats,
    LoRARequestStates as LoRARequestStates,
    RequestStateStats as RequestStateStats,
    SchedulerStats as SchedulerStats,
)

EMPTY_CPU_TENSOR: Incomplete

class RequestOutputCollector:
    aggregate: Incomplete
    request_id: Incomplete
    output: RequestOutput | PoolingRequestOutput | Exception | None
    ready: Incomplete
    def __init__(self, output_kind: RequestOutputKind, request_id: str) -> None: ...
    def put(self, output: RequestOutput | PoolingRequestOutput | Exception) -> None: ...
    async def get(self) -> RequestOutput | PoolingRequestOutput: ...
    def get_nowait(self) -> RequestOutput | PoolingRequestOutput | None: ...
    def close(self) -> None: ...
    def __del__(self) -> None: ...

@dataclass
class OutputProcessorOutput:
    request_outputs: list[RequestOutput | PoolingRequestOutput]
    reqs_to_abort: list[str]

@dataclass
class StreamingUpdate:
    prompt: str | None
    prompt_token_ids: list[int] | None
    arrival_time: float
    final: bool = ...

class RequestState:
    request_id: Incomplete
    external_req_id: Incomplete
    parent_req: Incomplete
    request_index: Incomplete
    lora_request: Incomplete
    lora_name: Incomplete
    output_kind: Incomplete
    prompt: Incomplete
    prompt_token_ids: Incomplete
    prompt_embeds: Incomplete
    prompt_len: Incomplete
    logprobs_processor: Incomplete
    detokenizer: Incomplete
    max_tokens_param: Incomplete
    top_p: Incomplete
    n: Incomplete
    temperature: Incomplete
    is_prefilling: bool
    queue: Incomplete
    num_cached_tokens: int
    stats: Incomplete
    stream_interval: Incomplete
    sent_tokens_offset: int
    streaming_input: Incomplete
    input_chunk_queue: deque[StreamingUpdate] | None
    def __init__(
        self,
        request_id: str,
        external_req_id: str,
        parent_req: ParentRequest | None,
        request_index: int,
        lora_request: LoRARequest | None,
        output_kind: RequestOutputKind,
        prompt: str | None,
        prompt_token_ids: list[int] | None,
        prompt_embeds: torch.Tensor | None,
        logprobs_processor: LogprobsProcessor | None,
        detokenizer: IncrementalDetokenizer | None,
        max_tokens_param: int | None,
        arrival_time: float,
        queue: RequestOutputCollector | None,
        log_stats: bool,
        stream_interval: int,
        top_p: float | None = None,
        n: int | None = None,
        temperature: float | None = None,
        stream_input: bool = False,
    ) -> None: ...
    def apply_streaming_update(self, update: StreamingUpdate) -> None: ...
    @classmethod
    def from_new_request(
        cls,
        tokenizer: TokenizerLike | None,
        request: EngineCoreRequest,
        prompt: str | None,
        parent_req: ParentRequest | None,
        request_index: int,
        queue: RequestOutputCollector | None,
        log_stats: bool,
        stream_interval: int,
    ) -> RequestState: ...
    def make_request_output(
        self,
        new_token_ids: list[int],
        pooling_output: torch.Tensor | None,
        finish_reason: FinishReason | None,
        stop_reason: int | str | None,
        kv_transfer_params: dict[str, Any] | None = None,
        routed_experts: np.ndarray | None = None,
    ) -> RequestOutput | PoolingRequestOutput | None: ...

class OutputProcessor:
    log_stats: Incomplete
    tokenizer: Incomplete
    stream_interval: Incomplete
    request_states: dict[str, RequestState]
    parent_requests: dict[str, ParentRequest]
    external_req_ids: defaultdict[str, list[str]]
    lora_states: Incomplete
    tracing_enabled: Incomplete
    def __init__(
        self,
        tokenizer: TokenizerLike | None,
        *,
        log_stats: bool,
        stream_interval: int = 1,
        tracing_enabled: bool = False,
    ) -> None: ...
    def get_num_unfinished_requests(self): ...
    def has_unfinished_requests(self) -> bool: ...
    def propagate_error(self, e: Exception): ...
    def abort_requests(
        self, request_ids: Iterable[str], internal: bool
    ) -> list[str]: ...
    def add_request(
        self,
        request: EngineCoreRequest,
        prompt: str | None,
        parent_req: ParentRequest | None = None,
        request_index: int = 0,
        queue: RequestOutputCollector | None = None,
    ) -> None: ...
    def process_outputs(
        self,
        engine_core_outputs: list[EngineCoreOutput],
        engine_core_timestamp: float | None = None,
        iteration_stats: IterationStats | None = None,
    ) -> OutputProcessorOutput: ...
    def update_scheduler_stats(self, scheduler_stats: SchedulerStats | None): ...
    def do_tracing(
        self,
        engine_core_output: EngineCoreOutput,
        req_state: RequestState,
        iteration_stats: IterationStats | None,
    ) -> None: ...
