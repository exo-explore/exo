import abc
from abc import ABC, abstractmethod
from collections.abc import AsyncGenerator, Iterable, Mapping
from dataclasses import dataclass
from typing import Any
from vllm.config import ModelConfig as ModelConfig, VllmConfig as VllmConfig
from vllm.distributed.weight_transfer.base import (
    WeightTransferInitRequest as WeightTransferInitRequest,
    WeightTransferUpdateRequest as WeightTransferUpdateRequest,
)
from vllm.inputs.data import (
    ProcessorInputs as ProcessorInputs,
    PromptType as PromptType,
)
from vllm.lora.request import LoRARequest as LoRARequest
from vllm.outputs import (
    PoolingRequestOutput as PoolingRequestOutput,
    RequestOutput as RequestOutput,
)
from vllm.plugins.io_processors import IOProcessor as IOProcessor
from vllm.pooling_params import PoolingParams as PoolingParams
from vllm.renderers import BaseRenderer as BaseRenderer
from vllm.sampling_params import SamplingParams as SamplingParams
from vllm.tasks import SupportedTask as SupportedTask
from vllm.v1.engine import (
    EngineCoreRequest as EngineCoreRequest,
    PauseMode as PauseMode,
)
from vllm.v1.engine.input_processor import InputProcessor as InputProcessor

@dataclass
class StreamingInput:
    prompt: ProcessorInputs
    sampling_params: SamplingParams | None = ...

class EngineClient(ABC, metaclass=abc.ABCMeta):
    vllm_config: VllmConfig
    model_config: ModelConfig
    renderer: BaseRenderer
    io_processor: IOProcessor | None
    input_processor: InputProcessor
    @property
    @abstractmethod
    def is_running(self) -> bool: ...
    @property
    @abstractmethod
    def is_stopped(self) -> bool: ...
    @property
    @abstractmethod
    def errored(self) -> bool: ...
    @property
    @abstractmethod
    def dead_error(self) -> BaseException: ...
    @abstractmethod
    def generate(
        self,
        prompt: EngineCoreRequest
        | PromptType
        | ProcessorInputs
        | AsyncGenerator[StreamingInput, None],
        sampling_params: SamplingParams,
        request_id: str,
        *,
        prompt_text: str | None = None,
        lora_request: LoRARequest | None = None,
        tokenization_kwargs: dict[str, Any] | None = None,
        trace_headers: Mapping[str, str] | None = None,
        priority: int = 0,
        data_parallel_rank: int | None = None,
        reasoning_ended: bool | None = None,
    ) -> AsyncGenerator[RequestOutput, None]: ...
    @abstractmethod
    def encode(
        self,
        prompt: PromptType | ProcessorInputs,
        pooling_params: PoolingParams,
        request_id: str,
        lora_request: LoRARequest | None = None,
        trace_headers: Mapping[str, str] | None = None,
        priority: int = 0,
        tokenization_kwargs: dict[str, Any] | None = None,
        reasoning_ended: bool | None = None,
    ) -> AsyncGenerator[PoolingRequestOutput, None]: ...
    @abstractmethod
    async def abort(self, request_id: str | Iterable[str]) -> None: ...
    @abstractmethod
    async def is_tracing_enabled(self) -> bool: ...
    @abstractmethod
    async def do_log_stats(self) -> None: ...
    @abstractmethod
    async def check_health(self) -> None: ...
    @abstractmethod
    async def start_profile(self) -> None: ...
    @abstractmethod
    async def stop_profile(self) -> None: ...
    @abstractmethod
    async def reset_mm_cache(self) -> None: ...
    @abstractmethod
    async def reset_encoder_cache(self) -> None: ...
    @abstractmethod
    async def reset_prefix_cache(
        self, reset_running_requests: bool = False, reset_connector: bool = False
    ) -> bool: ...
    @abstractmethod
    async def sleep(self, level: int = 1, mode: PauseMode = "abort") -> None: ...
    @abstractmethod
    async def wake_up(self, tags: list[str] | None = None) -> None: ...
    @abstractmethod
    async def is_sleeping(self) -> bool: ...
    @abstractmethod
    async def add_lora(self, lora_request: LoRARequest) -> bool: ...
    @abstractmethod
    async def pause_generation(
        self,
        *,
        mode: PauseMode = "abort",
        wait_for_inflight_requests: bool = False,
        clear_cache: bool = True,
    ) -> None: ...
    @abstractmethod
    async def resume_generation(self) -> None: ...
    @abstractmethod
    async def is_paused(self) -> bool: ...
    async def scale_elastic_ep(
        self, new_data_parallel_size: int, drain_timeout: int = 300
    ) -> None: ...
    async def collective_rpc(
        self,
        method: str,
        timeout: float | None = None,
        args: tuple = (),
        kwargs: dict | None = None,
    ): ...
    async def get_supported_tasks(self) -> tuple[SupportedTask, ...]: ...
    async def init_weight_transfer_engine(
        self, init_request: WeightTransferInitRequest
    ) -> None: ...
    async def update_weights(self, request: WeightTransferUpdateRequest) -> None: ...
