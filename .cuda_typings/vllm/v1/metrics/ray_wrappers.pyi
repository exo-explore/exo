from _typeshed import Incomplete
from ray.util.metrics import Metric as Metric
from vllm.distributed.kv_transfer.kv_connector.v1.metrics import (
    KVConnectorPrometheus as KVConnectorPrometheus,
)
from vllm.v1.metrics.loggers import PrometheusStatLogger as PrometheusStatLogger
from vllm.v1.metrics.perf import PerfMetricsProm as PerfMetricsProm
from vllm.v1.spec_decode.metrics import SpecDecodingProm as SpecDecodingProm

class RayPrometheusMetric:
    metric: Metric
    def __init__(self) -> None: ...
    def labels(self, *labels, **labelskwargs): ...

class RayGaugeWrapper(RayPrometheusMetric):
    metric: Incomplete
    def __init__(
        self,
        name: str,
        documentation: str | None = "",
        labelnames: list[str] | None = None,
        multiprocess_mode: str | None = "",
    ) -> None: ...
    def set(self, value: int | float): ...
    def set_to_current_time(self): ...

class RayCounterWrapper(RayPrometheusMetric):
    metric: Incomplete
    def __init__(
        self,
        name: str,
        documentation: str | None = "",
        labelnames: list[str] | None = None,
    ) -> None: ...
    def inc(self, value: int | float = 1.0): ...

class RayHistogramWrapper(RayPrometheusMetric):
    metric: Incomplete
    def __init__(
        self,
        name: str,
        documentation: str | None = "",
        labelnames: list[str] | None = None,
        buckets: list[float] | None = None,
    ) -> None: ...
    def observe(self, value: int | float): ...

class RaySpecDecodingProm(SpecDecodingProm): ...
class RayKVConnectorPrometheus(KVConnectorPrometheus): ...
class RayPerfMetricsProm(PerfMetricsProm): ...
class RayPrometheusStatLogger(PrometheusStatLogger): ...
