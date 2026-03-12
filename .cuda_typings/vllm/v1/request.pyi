import enum
import torch
from _typeshed import Incomplete
from collections import deque
from collections.abc import Callable as Callable, Mapping
from dataclasses import dataclass
from typing import Any
from vllm.lora.request import LoRARequest as LoRARequest
from vllm.multimodal.inputs import MultiModalFeatureSpec as MultiModalFeatureSpec
from vllm.pooling_params import PoolingParams as PoolingParams
from vllm.sampling_params import SamplingParams as SamplingParams
from vllm.utils import (
    length_from_prompt_token_ids_or_embeds as length_from_prompt_token_ids_or_embeds,
)
from vllm.v1.core.kv_cache_utils import BlockHash as BlockHash
from vllm.v1.engine import (
    EngineCoreEvent as EngineCoreEvent,
    EngineCoreEventType as EngineCoreEventType,
    EngineCoreRequest as EngineCoreRequest,
    FinishReason as FinishReason,
)
from vllm.v1.structured_output.request import (
    StructuredOutputRequest as StructuredOutputRequest,
)
from vllm.v1.utils import ConstantList as ConstantList

@dataclass
class StreamingUpdate:
    mm_features: list[MultiModalFeatureSpec] | None
    prompt_token_ids: list[int] | None
    max_tokens: int
    arrival_time: float
    sampling_params: SamplingParams | None
    @classmethod
    def from_request(cls, request: Request) -> StreamingUpdate | None: ...

class Request:
    request_id: Incomplete
    client_index: Incomplete
    priority: Incomplete
    sampling_params: Incomplete
    pooling_params: Incomplete
    lora_request: Incomplete
    structured_output_request: Incomplete
    arrival_time: Incomplete
    status: Incomplete
    events: list[EngineCoreEvent]
    stop_reason: int | str | None
    kv_transfer_params: dict[str, Any] | None
    max_tokens: int
    prompt_token_ids: Incomplete
    prompt_embeds: Incomplete
    num_prompt_tokens: Incomplete
    num_output_placeholders: int
    discard_latest_async_tokens: bool
    spec_token_ids: list[int]
    num_computed_tokens: int
    cache_salt: str | None
    mm_features: Incomplete
    output_token_ids: Incomplete
    all_token_ids: Incomplete
    trace_headers: Incomplete
    num_cached_tokens: int
    is_prefill_chunk: bool
    num_nans_in_logits: int
    num_preemptions: int
    num_external_computed_tokens: int
    block_hashes: list[BlockHash]
    skip_reading_prefix_cache: Incomplete
    resumable: Incomplete
    streaming_queue: deque[StreamingUpdate | None] | None
    def __init__(
        self,
        request_id: str,
        prompt_token_ids: list[int] | None,
        sampling_params: SamplingParams | None,
        pooling_params: PoolingParams | None,
        client_index: int = 0,
        arrival_time: float | None = None,
        prompt_embeds: torch.Tensor | None = None,
        mm_features: list[MultiModalFeatureSpec] | None = None,
        lora_request: LoRARequest | None = None,
        cache_salt: str | None = None,
        priority: int = 0,
        trace_headers: Mapping[str, str] | None = None,
        block_hasher: Callable[[Request], list["BlockHash"]] | None = None,
        resumable: bool = False,
        reasoning_ended: bool | None = None,
    ) -> None: ...
    @classmethod
    def from_engine_core_request(
        cls,
        request: EngineCoreRequest,
        block_hasher: Callable[[Request], list["BlockHash"]] | None,
    ) -> Request: ...
    def append_output_token_ids(self, token_ids: int | list[int]) -> None: ...
    def update_block_hashes(self) -> None: ...
    @property
    def use_structured_output(self) -> bool: ...
    @property
    def num_tokens(self) -> int: ...
    @property
    def num_tokens_with_spec(self) -> int: ...
    @property
    def num_output_tokens(self) -> int: ...
    @property
    def num_encoder_inputs(self) -> int: ...
    @property
    def has_encoder_inputs(self) -> bool: ...
    def get_skip_reading_prefix_cache(self) -> bool: ...
    def is_finished(self) -> bool: ...
    def get_finished_reason(self) -> FinishReason | None: ...
    def get_num_encoder_embeds(self, input_id: int) -> int: ...
    def record_event(
        self, event_type: EngineCoreEventType, timestamp: float | None = None
    ) -> None: ...
    def take_events(self) -> list[EngineCoreEvent] | None: ...
    def __lt__(self, other: Request) -> bool: ...

class RequestStatus(enum.IntEnum):
    WAITING = ...
    WAITING_FOR_FSM = ...
    WAITING_FOR_REMOTE_KVS = ...
    WAITING_FOR_STREAMING_REQ = ...
    RUNNING = ...
    PREEMPTED = ...
    FINISHED_STOPPED = ...
    FINISHED_LENGTH_CAPPED = ...
    FINISHED_ABORTED = ...
    FINISHED_IGNORED = ...
    FINISHED_ERROR = ...
    FINISHED_REPETITION = ...
    @staticmethod
    def is_finished(status: RequestStatus) -> bool: ...
    @staticmethod
    def get_finished_reason(status: RequestStatus) -> FinishReason | None: ...
