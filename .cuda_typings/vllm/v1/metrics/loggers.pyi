import abc
from _typeshed import Incomplete
from abc import ABC, abstractmethod
from prometheus_client import Counter, Gauge, Histogram
from typing import TypeAlias
from vllm.compilation.cuda_graph import CUDAGraphLogging as CUDAGraphLogging
from vllm.config import (
    SupportsMetricsInfo as SupportsMetricsInfo,
    VllmConfig as VllmConfig,
)
from vllm.distributed.kv_transfer.kv_connector.v1.metrics import (
    KVConnectorLogging as KVConnectorLogging,
    KVConnectorPrometheus as KVConnectorPrometheus,
)
from vllm.logger import init_logger as init_logger
from vllm.plugins import (
    STAT_LOGGER_PLUGINS_GROUP as STAT_LOGGER_PLUGINS_GROUP,
    load_plugins_by_group as load_plugins_by_group,
)
from vllm.v1.engine import FinishReason as FinishReason
from vllm.v1.metrics.perf import (
    PerfMetricsLogging as PerfMetricsLogging,
    PerfMetricsProm as PerfMetricsProm,
)
from vllm.v1.metrics.prometheus import (
    unregister_vllm_metrics as unregister_vllm_metrics,
)
from vllm.v1.metrics.stats import (
    CachingMetrics as CachingMetrics,
    IterationStats as IterationStats,
    MultiModalCacheStats as MultiModalCacheStats,
    PromptTokenStats as PromptTokenStats,
    SchedulerStats as SchedulerStats,
)
from vllm.v1.spec_decode.metrics import (
    SpecDecodingLogging as SpecDecodingLogging,
    SpecDecodingProm as SpecDecodingProm,
)

logger: Incomplete
PerEngineStatLoggerFactory: Incomplete
AggregateStatLoggerFactory: Incomplete
StatLoggerFactory = AggregateStatLoggerFactory | PerEngineStatLoggerFactory

class StatLoggerBase(ABC, metaclass=abc.ABCMeta):
    @abstractmethod
    def __init__(self, vllm_config: VllmConfig, engine_index: int = 0): ...
    @abstractmethod
    def record(
        self,
        scheduler_stats: SchedulerStats | None,
        iteration_stats: IterationStats | None,
        mm_cache_stats: MultiModalCacheStats | None = None,
        engine_idx: int = 0,
    ): ...
    @abstractmethod
    def log_engine_initialized(self): ...
    def log(self) -> None: ...
    def record_sleep_state(self, is_awake: int, level: int): ...

def load_stat_logger_plugin_factories() -> list[StatLoggerFactory]: ...

class AggregateStatLoggerBase(StatLoggerBase, metaclass=abc.ABCMeta):
    @abstractmethod
    def __init__(self, vllm_config: VllmConfig, engine_indexes: list[int]): ...

class LoggingStatLogger(StatLoggerBase):
    engine_index: Incomplete
    vllm_config: Incomplete
    last_scheduler_stats: Incomplete
    prefix_caching_metrics: Incomplete
    connector_prefix_caching_metrics: Incomplete
    mm_caching_metrics: Incomplete
    spec_decoding_logging: Incomplete
    kv_connector_logging: Incomplete
    cudagraph_logging: Incomplete
    last_prompt_throughput: float
    last_generation_throughput: float
    engine_is_idle: bool
    aggregated: bool
    perf_metrics_logging: Incomplete
    def __init__(self, vllm_config: VllmConfig, engine_index: int = 0) -> None: ...
    @property
    def log_prefix(self): ...
    def record(
        self,
        scheduler_stats: SchedulerStats | None,
        iteration_stats: IterationStats | None,
        mm_cache_stats: MultiModalCacheStats | None = None,
        engine_idx: int = 0,
    ): ...
    def aggregate_scheduler_stats(self) -> None: ...
    def log(self) -> None: ...
    def log_engine_initialized(self) -> None: ...

class AggregatedLoggingStatLogger(LoggingStatLogger, AggregateStatLoggerBase):
    engine_indexes: Incomplete
    last_scheduler_stats_dict: dict[int, SchedulerStats]
    aggregated: bool
    def __init__(self, vllm_config: VllmConfig, engine_indexes: list[int]) -> None: ...
    @property
    def log_prefix(self): ...
    def record(
        self,
        scheduler_stats: SchedulerStats | None,
        iteration_stats: IterationStats | None,
        mm_cache_stats: MultiModalCacheStats | None = None,
        engine_idx: int = 0,
    ): ...
    last_scheduler_stats: Incomplete
    def aggregate_scheduler_stats(self) -> None: ...
    def log(self) -> None: ...
    def log_engine_initialized(self) -> None: ...

class PerEngineStatLoggerAdapter(AggregateStatLoggerBase):
    per_engine_stat_loggers: Incomplete
    engine_indexes: Incomplete
    def __init__(
        self,
        vllm_config: VllmConfig,
        engine_indexes: list[int],
        per_engine_stat_logger_factory: PerEngineStatLoggerFactory,
    ) -> None: ...
    def record(
        self,
        scheduler_stats: SchedulerStats | None,
        iteration_stats: IterationStats | None,
        mm_cache_stats: MultiModalCacheStats | None = None,
        engine_idx: int = 0,
    ): ...
    def log(self) -> None: ...
    def log_engine_initialized(self) -> None: ...

class PrometheusStatLogger(AggregateStatLoggerBase):
    engine_indexes: Incomplete
    vllm_config: Incomplete
    show_hidden_metrics: Incomplete
    kv_cache_metrics_enabled: Incomplete
    spec_decoding_prom: Incomplete
    kv_connector_prom: Incomplete
    perf_metrics_prom: Incomplete
    gauge_scheduler_running: Incomplete
    gauge_scheduler_waiting: Incomplete
    gauge_engine_sleep_state: Incomplete
    gauge_kv_cache_usage: Incomplete
    counter_corrupted_requests: Incomplete
    counter_prefix_cache_queries: Incomplete
    counter_prefix_cache_hits: Incomplete
    counter_connector_prefix_cache_queries: Incomplete
    counter_connector_prefix_cache_hits: Incomplete
    counter_mm_cache_queries: Incomplete
    counter_mm_cache_hits: Incomplete
    counter_num_preempted_reqs: Incomplete
    counter_prompt_tokens: Incomplete
    counter_prompt_tokens_by_source: dict[str, dict[int, Counter]]
    counter_prompt_tokens_cached: Incomplete
    counter_prompt_tokens_recomputed: Incomplete
    counter_generation_tokens: Incomplete
    counter_request_success: dict[FinishReason, dict[int, Counter]]
    histogram_num_prompt_tokens_request: Incomplete
    histogram_num_generation_tokens_request: Incomplete
    histogram_iteration_tokens: Incomplete
    histogram_max_num_generation_tokens_request: Incomplete
    histogram_n_request: Incomplete
    histogram_max_tokens_request: Incomplete
    histogram_time_to_first_token: Incomplete
    histogram_inter_token_latency: Incomplete
    histogram_request_time_per_output_token: Incomplete
    histogram_e2e_time_request: Incomplete
    histogram_queue_time_request: Incomplete
    histogram_inference_time_request: Incomplete
    histogram_prefill_time_request: Incomplete
    histogram_decode_time_request: Incomplete
    histogram_prefill_kv_computed_request: Incomplete
    histogram_kv_block_lifetime: Incomplete
    histogram_kv_block_idle_before_evict: Incomplete
    histogram_kv_block_reuse_gap: Incomplete
    gauge_lora_info: Gauge | None
    labelname_max_lora: str
    labelname_waiting_lora_adapters: str
    labelname_running_lora_adapters: str
    max_lora: Incomplete
    def __init__(
        self, vllm_config: VllmConfig, engine_indexes: list[int] | None = None
    ) -> None: ...
    def log_metrics_info(self, type: str, config_obj: SupportsMetricsInfo): ...
    def record(
        self,
        scheduler_stats: SchedulerStats | None,
        iteration_stats: IterationStats | None,
        mm_cache_stats: MultiModalCacheStats | None = None,
        engine_idx: int = 0,
    ): ...
    def record_sleep_state(self, sleep: int = 0, level: int = 0): ...
    def log_engine_initialized(self) -> None: ...

PromMetric: TypeAlias = Gauge | Counter | Histogram

def make_per_engine(
    metric: PromMetric, engine_idxs: list[int], model_name: object
) -> dict[int, PromMetric]: ...
def build_buckets(mantissa_lst: list[int], max_value: int) -> list[int]: ...
def build_1_2_5_buckets(max_value: int) -> list[int]: ...

class StatLoggerManager:
    engine_indexes: Incomplete
    stat_loggers: list[AggregateStatLoggerBase]
    def __init__(
        self,
        vllm_config: VllmConfig,
        engine_idxs: list[int] | None = None,
        custom_stat_loggers: list[StatLoggerFactory] | None = None,
        enable_default_loggers: bool = True,
        aggregate_engine_logging: bool = False,
        client_count: int = 1,
    ) -> None: ...
    def record(
        self,
        scheduler_stats: SchedulerStats | None,
        iteration_stats: IterationStats | None,
        mm_cache_stats: MultiModalCacheStats | None = None,
        engine_idx: int | None = None,
    ): ...
    def record_sleep_state(self, sleep: int = 0, level: int = 0): ...
    def log(self) -> None: ...
    def log_engine_initialized(self) -> None: ...
