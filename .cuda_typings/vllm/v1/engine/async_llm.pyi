import asyncio
from _typeshed import Incomplete
from collections.abc import AsyncGenerator, Iterable, Mapping
from typing import Any
from vllm import TokensPrompt as TokensPrompt
from vllm.config import VllmConfig as VllmConfig
from vllm.distributed.weight_transfer.base import (
    WeightTransferInitRequest as WeightTransferInitRequest,
    WeightTransferUpdateRequest as WeightTransferUpdateRequest,
)
from vllm.engine.arg_utils import AsyncEngineArgs as AsyncEngineArgs
from vllm.engine.protocol import (
    EngineClient as EngineClient,
    StreamingInput as StreamingInput,
)
from vllm.entrypoints.serve.elastic_ep.middleware import (
    set_scaling_elastic_ep as set_scaling_elastic_ep,
)
from vllm.inputs import ProcessorInputs as ProcessorInputs, PromptType as PromptType
from vllm.logger import init_logger as init_logger
from vllm.lora.request import LoRARequest as LoRARequest
from vllm.multimodal import (
    MULTIMODAL_REGISTRY as MULTIMODAL_REGISTRY,
    MultiModalRegistry as MultiModalRegistry,
)
from vllm.outputs import (
    PoolingRequestOutput as PoolingRequestOutput,
    RequestOutput as RequestOutput,
    STREAM_FINISHED as STREAM_FINISHED,
)
from vllm.plugins.io_processors import get_io_processor as get_io_processor
from vllm.pooling_params import PoolingParams as PoolingParams
from vllm.renderers import renderer_from_config as renderer_from_config
from vllm.renderers.inputs.preprocess import (
    extract_prompt_components as extract_prompt_components,
)
from vllm.sampling_params import (
    RequestOutputKind as RequestOutputKind,
    SamplingParams as SamplingParams,
)
from vllm.tasks import SupportedTask as SupportedTask
from vllm.tokenizers import TokenizerLike as TokenizerLike
from vllm.tracing import init_tracer as init_tracer
from vllm.transformers_utils.config import (
    maybe_register_config_serialize_by_value as maybe_register_config_serialize_by_value,
)
from vllm.usage.usage_lib import UsageContext as UsageContext
from vllm.utils.async_utils import cancel_task_threadsafe as cancel_task_threadsafe
from vllm.utils.collection_utils import as_list as as_list
from vllm.v1.engine import (
    EngineCoreRequest as EngineCoreRequest,
    PauseMode as PauseMode,
)
from vllm.v1.engine.core_client import EngineCoreClient as EngineCoreClient
from vllm.v1.engine.exceptions import (
    EngineDeadError as EngineDeadError,
    EngineGenerateError as EngineGenerateError,
)
from vllm.v1.engine.input_processor import InputProcessor as InputProcessor
from vllm.v1.engine.output_processor import (
    OutputProcessor as OutputProcessor,
    RequestOutputCollector as RequestOutputCollector,
)
from vllm.v1.engine.parallel_sampling import ParentRequest as ParentRequest
from vllm.v1.executor import Executor as Executor
from vllm.v1.metrics.loggers import (
    StatLoggerFactory as StatLoggerFactory,
    StatLoggerManager as StatLoggerManager,
    load_stat_logger_plugin_factories as load_stat_logger_plugin_factories,
)
from vllm.v1.metrics.prometheus import shutdown_prometheus as shutdown_prometheus
from vllm.v1.metrics.stats import IterationStats as IterationStats

logger: Incomplete

class InputStreamError(Exception):
    cause: Incomplete
    def __init__(self, cause: Exception) -> None: ...

class AsyncLLM(EngineClient):
    vllm_config: Incomplete
    model_config: Incomplete
    observability_config: Incomplete
    log_requests: Incomplete
    log_stats: Incomplete
    renderer: Incomplete
    io_processor: Incomplete
    input_processor: Incomplete
    output_processor: Incomplete
    engine_core: Incomplete
    logger_manager: StatLoggerManager | None
    output_handler: asyncio.Task | None
    profiler: Incomplete
    def __init__(
        self,
        vllm_config: VllmConfig,
        executor_class: type[Executor],
        log_stats: bool,
        usage_context: UsageContext = ...,
        mm_registry: MultiModalRegistry = ...,
        use_cached_outputs: bool = False,
        log_requests: bool = True,
        start_engine_loop: bool = True,
        stat_loggers: list[StatLoggerFactory] | None = None,
        aggregate_engine_logging: bool = False,
        client_addresses: dict[str, str] | None = None,
        client_count: int = 1,
        client_index: int = 0,
    ) -> None: ...
    @classmethod
    def from_vllm_config(
        cls,
        vllm_config: VllmConfig,
        start_engine_loop: bool = True,
        usage_context: UsageContext = ...,
        stat_loggers: list[StatLoggerFactory] | None = None,
        enable_log_requests: bool = False,
        aggregate_engine_logging: bool = False,
        disable_log_stats: bool = False,
        client_addresses: dict[str, str] | None = None,
        client_count: int = 1,
        client_index: int = 0,
    ) -> AsyncLLM: ...
    @classmethod
    def from_engine_args(
        cls,
        engine_args: AsyncEngineArgs,
        start_engine_loop: bool = True,
        usage_context: UsageContext = ...,
        stat_loggers: list[StatLoggerFactory] | None = None,
    ) -> AsyncLLM: ...
    def __del__(self) -> None: ...
    def shutdown(self) -> None: ...
    async def get_supported_tasks(self) -> tuple[SupportedTask, ...]: ...
    async def add_request(
        self,
        request_id: str,
        prompt: EngineCoreRequest
        | PromptType
        | ProcessorInputs
        | AsyncGenerator[StreamingInput, None],
        params: SamplingParams | PoolingParams,
        arrival_time: float | None = None,
        lora_request: LoRARequest | None = None,
        tokenization_kwargs: dict[str, Any] | None = None,
        trace_headers: Mapping[str, str] | None = None,
        priority: int = 0,
        data_parallel_rank: int | None = None,
        prompt_text: str | None = None,
        reasoning_ended: bool | None = None,
    ) -> RequestOutputCollector: ...
    async def generate(
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
    async def abort(
        self, request_id: str | Iterable[str], internal: bool = False
    ) -> None: ...
    async def pause_generation(
        self,
        *,
        mode: PauseMode = "abort",
        wait_for_inflight_requests: bool | None = None,
        clear_cache: bool = True,
    ) -> None: ...
    async def resume_generation(self) -> None: ...
    async def is_paused(self) -> bool: ...
    async def encode(
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
    @property
    def tokenizer(self) -> TokenizerLike | None: ...
    def get_tokenizer(self) -> TokenizerLike: ...
    async def is_tracing_enabled(self) -> bool: ...
    async def do_log_stats(self) -> None: ...
    async def check_health(self) -> None: ...
    async def start_profile(self, profile_prefix: str | None = None) -> None: ...
    async def stop_profile(self) -> None: ...
    async def reset_mm_cache(self) -> None: ...
    async def reset_prefix_cache(
        self, reset_running_requests: bool = False, reset_connector: bool = False
    ) -> bool: ...
    async def reset_encoder_cache(self) -> None: ...
    async def sleep(self, level: int = 1, mode: PauseMode = "abort") -> None: ...
    async def wake_up(self, tags: list[str] | None = None) -> None: ...
    async def is_sleeping(self) -> bool: ...
    async def add_lora(self, lora_request: LoRARequest) -> bool: ...
    async def remove_lora(self, lora_id: int) -> bool: ...
    async def list_loras(self) -> set[int]: ...
    async def pin_lora(self, lora_id: int) -> bool: ...
    async def collective_rpc(
        self,
        method: str,
        timeout: float | None = None,
        args: tuple = (),
        kwargs: dict | None = None,
    ): ...
    async def wait_for_requests_to_drain(self, drain_timeout: int = 300): ...
    async def scale_elastic_ep(
        self, new_data_parallel_size: int, drain_timeout: int = 300
    ): ...
    @property
    def is_running(self) -> bool: ...
    @property
    def is_stopped(self) -> bool: ...
    @property
    def errored(self) -> bool: ...
    @property
    def dead_error(self) -> BaseException: ...
    async def init_weight_transfer_engine(
        self, request: WeightTransferInitRequest
    ) -> None: ...
    async def update_weights(self, request: WeightTransferUpdateRequest) -> None: ...
