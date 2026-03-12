from _typeshed import Incomplete
from collections.abc import Mapping
from vllm.logger import init_logger as init_logger
from vllm.v1.metrics.reader import (
    Gauge as Gauge,
    get_metrics_snapshot as get_metrics_snapshot,
)

logger: Incomplete

def create_orca_header(
    metrics_format: str, named_metrics: list[tuple[str, float]]
) -> Mapping[str, str] | None: ...
def get_named_metrics_from_prometheus() -> list[tuple[str, float]]: ...
def metrics_header(metrics_format: str) -> Mapping[str, str] | None: ...
