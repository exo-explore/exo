import enum
import msgspec
import numpy as np
import torch
from _typeshed import Incomplete
from collections.abc import Mapping
from typing import Any
from vllm.lora.request import LoRARequest as LoRARequest
from vllm.multimodal.inputs import MultiModalFeatureSpec as MultiModalFeatureSpec
from vllm.pooling_params import PoolingParams as PoolingParams
from vllm.sampling_params import SamplingParams as SamplingParams
from vllm.v1.metrics.stats import SchedulerStats as SchedulerStats
from vllm.v1.outputs import (
    LogprobsLists as LogprobsLists,
    LogprobsTensors as LogprobsTensors,
)
from vllm.v1.serial_utils import UtilityResult as UtilityResult

PauseMode: Incomplete
FINISH_REASON_STRINGS: Incomplete
EEP_NOTIFICATION_CALL_ID: int

class EEPNotificationType(enum.Enum):
    NEW_CORE_ENGINES_INIT_READY = "NEW_CORE_ENGINES_INIT_READY"
    NEW_CORE_ENGINES_WEIGHTS_INIT_READY = "NEW_CORE_ENGINES_WEIGHTS_INIT_READY"
    RECONFIGURE_FINISHED = "RECONFIGURE_FINISHED"
    SHUTDOWN_COMPLETE = "SHUTDOWN_COMPLETE"

class FinishReason(enum.IntEnum):
    STOP = 0
    LENGTH = 1
    ABORT = 2
    ERROR = 3
    REPETITION = 4

class EngineCoreRequest(msgspec.Struct, array_like=True, omit_defaults=True, gc=False):
    request_id: str
    prompt_token_ids: list[int] | None
    mm_features: list[MultiModalFeatureSpec] | None
    sampling_params: SamplingParams | None
    pooling_params: PoolingParams | None
    arrival_time: float
    lora_request: LoRARequest | None
    cache_salt: str | None
    data_parallel_rank: int | None
    prompt_embeds: torch.Tensor | None = ...
    client_index: int = ...
    current_wave: int = ...
    priority: int = ...
    trace_headers: Mapping[str, str] | None = ...
    resumable: bool = ...
    external_req_id: str | None = ...
    reasoning_ended: bool | None = ...
    @property
    def params(self) -> SamplingParams | PoolingParams: ...

class EngineCoreEventType(enum.IntEnum):
    QUEUED = 1
    SCHEDULED = 2
    PREEMPTED = 3

class EngineCoreEvent(msgspec.Struct):
    type: EngineCoreEventType
    timestamp: float
    @classmethod
    def new_event(
        cls, event_type: EngineCoreEventType, timestamp: float | None = None
    ) -> EngineCoreEvent: ...

class EngineCoreOutput(msgspec.Struct, array_like=True, omit_defaults=True, gc=False):
    request_id: str
    new_token_ids: list[int]
    new_logprobs: LogprobsLists | None = ...
    new_prompt_logprobs_tensors: LogprobsTensors | None = ...
    pooling_output: torch.Tensor | None = ...
    finish_reason: FinishReason | None = ...
    stop_reason: int | str | None = ...
    events: list[EngineCoreEvent] | None = ...
    kv_transfer_params: dict[str, Any] | None = ...
    trace_headers: Mapping[str, str] | None = ...
    num_cached_tokens: int = ...
    num_external_computed_tokens: int = ...
    routed_experts: np.ndarray | None = ...
    num_nans_in_logits: int = ...
    @property
    def finished(self) -> bool: ...

class UtilityOutput(msgspec.Struct, array_like=True, gc=False):
    call_id: int
    failure_message: str | None = ...
    result: UtilityResult | None = ...

class EngineCoreOutputs(msgspec.Struct, array_like=True, omit_defaults=True, gc=False):
    engine_index: int = ...
    outputs: list[EngineCoreOutput] = ...
    scheduler_stats: SchedulerStats | None = ...
    timestamp: float = ...
    utility_output: UtilityOutput | None = ...
    finished_requests: set[str] | None = ...
    wave_complete: int | None = ...
    start_wave: int | None = ...
    def __post_init__(self) -> None: ...

class EngineCoreRequestType(enum.Enum):
    ADD = b"\x00"
    ABORT = b"\x01"
    START_DP_WAVE = b"\x02"
    UTILITY = b"\x03"
    EXECUTOR_FAILED = b"\x04"

class ReconfigureDistributedRequest(msgspec.Struct):
    new_data_parallel_size: int
    new_data_parallel_rank: int
    new_data_parallel_rank_local: int
    new_data_parallel_master_ip: str
    new_data_parallel_master_port: int
    new_data_parallel_master_port_list: list[int]
    new_stateless_world_group_port_list: list[list[int]]
    new_stateless_dp_group_port_list: list[list[int]]
    new_stateless_ep_group_port_list: list[list[int]]
    new_stateless_eplb_group_port_list: list[list[int]]

class ReconfigureRankType(enum.IntEnum):
    KEEP_CURRENT_RANK = -1
    SHUTDOWN_CURRENT_RANK = -2
