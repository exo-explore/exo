import torch.nn as nn
from _typeshed import Incomplete
from collections.abc import Callable as Callable, Mapping
from typing import Any
from vllm.config import (
    ModelConfig as ModelConfig,
    ParallelConfig as ParallelConfig,
    VllmConfig as VllmConfig,
)
from vllm.distributed import (
    stateless_destroy_torch_distributed_process_group as stateless_destroy_torch_distributed_process_group,
)
from vllm.distributed.parallel_state import get_dp_group as get_dp_group
from vllm.engine.arg_utils import EngineArgs as EngineArgs
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
)
from vllm.plugins.io_processors import get_io_processor as get_io_processor
from vllm.pooling_params import PoolingParams as PoolingParams
from vllm.renderers import renderer_from_config as renderer_from_config
from vllm.renderers.inputs.preprocess import (
    extract_prompt_components as extract_prompt_components,
)
from vllm.sampling_params import SamplingParams as SamplingParams
from vllm.tasks import SupportedTask as SupportedTask
from vllm.tokenizers import TokenizerLike as TokenizerLike
from vllm.tracing import init_tracer as init_tracer
from vllm.usage.usage_lib import UsageContext as UsageContext
from vllm.v1.engine import (
    EngineCoreRequest as EngineCoreRequest,
    PauseMode as PauseMode,
)
from vllm.v1.engine.core_client import EngineCoreClient as EngineCoreClient
from vllm.v1.engine.input_processor import InputProcessor as InputProcessor
from vllm.v1.engine.output_processor import OutputProcessor as OutputProcessor
from vllm.v1.engine.parallel_sampling import ParentRequest as ParentRequest
from vllm.v1.executor import Executor as Executor
from vllm.v1.metrics.loggers import (
    StatLoggerFactory as StatLoggerFactory,
    StatLoggerManager as StatLoggerManager,
)
from vllm.v1.metrics.reader import (
    Metric as Metric,
    get_metrics_snapshot as get_metrics_snapshot,
)
from vllm.v1.metrics.stats import IterationStats as IterationStats
from vllm.v1.utils import (
    record_function_or_nullcontext as record_function_or_nullcontext,
)
from vllm.v1.worker.worker_base import WorkerBase as WorkerBase

logger: Incomplete

class LLMEngine:
    vllm_config: VllmConfig
    model_config: ModelConfig
    observability_config: Incomplete
    log_stats: Incomplete
    external_launcher_dp: Incomplete
    dp_group: Incomplete
    should_execute_dummy_batch: bool
    renderer: Incomplete
    io_processor: Incomplete
    input_processor: Incomplete
    output_processor: Incomplete
    engine_core: Incomplete
    logger_manager: StatLoggerManager | None
    model_executor: Incomplete
    def __init__(
        self,
        vllm_config: VllmConfig,
        executor_class: type[Executor],
        log_stats: bool,
        aggregate_engine_logging: bool = False,
        usage_context: UsageContext = ...,
        stat_loggers: list[StatLoggerFactory] | None = None,
        mm_registry: MultiModalRegistry = ...,
        use_cached_outputs: bool = False,
        multiprocess_mode: bool = False,
    ) -> None: ...
    @classmethod
    def from_vllm_config(
        cls,
        vllm_config: VllmConfig,
        usage_context: UsageContext = ...,
        stat_loggers: list[StatLoggerFactory] | None = None,
        disable_log_stats: bool = False,
    ) -> LLMEngine: ...
    @classmethod
    def from_engine_args(
        cls,
        engine_args: EngineArgs,
        usage_context: UsageContext = ...,
        stat_loggers: list[StatLoggerFactory] | None = None,
        enable_multiprocessing: bool = False,
    ) -> LLMEngine: ...
    def get_num_unfinished_requests(self) -> int: ...
    def has_unfinished_requests(self) -> bool: ...
    def has_unfinished_requests_dp(self, has_unfinished: bool) -> bool: ...
    def get_supported_tasks(self) -> tuple[SupportedTask, ...]: ...
    def abort_request(self, request_ids: list[str], internal: bool = False) -> None: ...
    def add_request(
        self,
        request_id: str,
        prompt: EngineCoreRequest | PromptType | ProcessorInputs,
        params: SamplingParams | PoolingParams,
        arrival_time: float | None = None,
        lora_request: LoRARequest | None = None,
        tokenization_kwargs: dict[str, Any] | None = None,
        trace_headers: Mapping[str, str] | None = None,
        priority: int = 0,
        prompt_text: str | None = None,
    ) -> str: ...
    def step(self) -> list[RequestOutput | PoolingRequestOutput]: ...
    def start_profile(self, profile_prefix: str | None = None): ...
    def stop_profile(self) -> None: ...
    def reset_mm_cache(self) -> None: ...
    def reset_prefix_cache(
        self, reset_running_requests: bool = False, reset_connector: bool = False
    ) -> bool: ...
    def reset_encoder_cache(self) -> None: ...
    def sleep(self, level: int = 1, mode: PauseMode = "abort"): ...
    def wake_up(self, tags: list[str] | None = None): ...
    def is_sleeping(self) -> bool: ...
    def get_metrics(self) -> list[Metric]: ...
    @property
    def tokenizer(self) -> TokenizerLike | None: ...
    def get_tokenizer(self) -> TokenizerLike: ...
    def do_log_stats(self) -> None: ...
    def do_log_stats_with_interval(self) -> None: ...
    def add_lora(self, lora_request: LoRARequest) -> bool: ...
    def remove_lora(self, lora_id: int) -> bool: ...
    def list_loras(self) -> set[int]: ...
    def pin_lora(self, lora_id: int) -> bool: ...
    def collective_rpc(
        self,
        method: str | Callable[[WorkerBase], _R],
        timeout: float | None = None,
        args: tuple = (),
        kwargs: dict[str, Any] | None = None,
    ) -> list[_R]: ...
    def apply_model(self, func: Callable[[nn.Module], _R]) -> list[_R]: ...
    def __del__(self) -> None: ...
