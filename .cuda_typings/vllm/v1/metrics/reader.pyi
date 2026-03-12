from dataclasses import dataclass
from prometheus_client.samples import Sample as Sample

@dataclass
class Metric:
    name: str
    labels: dict[str, str]

@dataclass
class Counter(Metric):
    value: int

@dataclass
class Vector(Metric):
    values: list[int]

@dataclass
class Gauge(Metric):
    value: float

@dataclass
class Histogram(Metric):
    count: int
    sum: float
    buckets: dict[str, int]

def get_metrics_snapshot() -> list[Metric]: ...
