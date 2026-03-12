from _typeshed import Incomplete
from dataclasses import dataclass, field
from prometheus_client import Counter, Gauge, Histogram
from typing import Any, TypeAlias, TypeVar
from vllm.config import KVTransferConfig as KVTransferConfig, VllmConfig as VllmConfig
from vllm.distributed.kv_transfer.kv_connector.factory import (
    KVConnectorFactory as KVConnectorFactory,
)
from vllm.logger import init_logger as init_logger

PromMetric: TypeAlias = Gauge | Counter | Histogram
PromMetricT = TypeVar("PromMetricT", bound=PromMetric)
logger: Incomplete

@dataclass
class KVConnectorStats:
    data: dict[str, Any] = field(default_factory=dict)
    def reset(self) -> None: ...
    def aggregate(self, other: KVConnectorStats) -> KVConnectorStats: ...
    def reduce(self) -> dict[str, int | float]: ...
    def is_empty(self) -> bool: ...

class KVConnectorLogging:
    connector_cls: Incomplete
    def __init__(self, kv_transfer_config: KVTransferConfig | None) -> None: ...
    transfer_stats_accumulator: KVConnectorStats | None
    def reset(self) -> None: ...
    def observe(self, transfer_stats_data: dict[str, Any]): ...
    def log(self, log_fn=...) -> None: ...

class KVConnectorPromMetrics:
    per_engine_labelvalues: Incomplete
    def __init__(
        self,
        vllm_config: VllmConfig,
        metric_types: dict[type[PromMetric], type[PromMetricT]],
        labelnames: list[str],
        per_engine_labelvalues: dict[int, list[object]],
    ) -> None: ...
    def make_per_engine(self, metric: PromMetric) -> dict[int, PromMetric]: ...
    def observe(self, transfer_stats_data: dict[str, Any], engine_idx: int = 0): ...

class KVConnectorPrometheus:
    prom_metrics: KVConnectorPromMetrics | None
    def __init__(
        self,
        vllm_config: VllmConfig,
        labelnames: list[str],
        per_engine_labelvalues: dict[int, list[object]],
    ) -> None: ...
    def observe(self, transfer_stats_data: dict[str, Any], engine_idx: int = 0): ...
