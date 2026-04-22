"""Prometheus-compatible metrics for exo.

Each node's API exposes a `/metrics` endpoint rendered from the default
registry. To avoid double-counting when VictoriaMetrics scrapes every node:

* Per-generation metrics (counters, histograms) are recorded only by the
  current elected master — the one node whose `_apply_state` is treated as
  authoritative for the cluster. Callers gate on `API.is_master`.
* Per-node system metrics are recorded only when the event's `node_id`
  matches the observer — each node reports its own macmon/memory/disk.
* Cluster gauges are refreshed on scrape against the master's view of state.
"""

from collections.abc import Mapping
from typing import Final

from prometheus_client import (
    CollectorRegistry,
    Counter,
    Gauge,
    Histogram,
    generate_latest,
)
from prometheus_client.exposition import CONTENT_TYPE_LATEST as _PROM_CONTENT_TYPE

from exo.api.types import GenerationStats
from exo.shared.types.chunks import GenerationChunk
from exo.shared.types.common import NodeId
from exo.shared.types.profiling import MemoryUsage
from exo.shared.types.worker.instances import Instance, InstanceId
from exo.shared.types.worker.runners import RunnerId, RunnerStatus
from exo.utils.info_gatherer.info_gatherer import (
    GatheredInfo,
    NodeDiskUsage,
    RdmaCtlStatus,
)
from exo.utils.info_gatherer.macmon import MacmonMetrics

CONTENT_TYPE_LATEST: Final[str] = _PROM_CONTENT_TYPE

_registry: Final[CollectorRegistry] = CollectorRegistry()

_INF: Final[float] = float("inf")
_TPS_BUCKETS_PROMPT: Final[tuple[float, ...]] = (
    50.0,
    100.0,
    200.0,
    400.0,
    800.0,
    1600.0,
    3200.0,
    6400.0,
    _INF,
)
_TPS_BUCKETS_GEN: Final[tuple[float, ...]] = (
    5.0,
    10.0,
    20.0,
    40.0,
    60.0,
    80.0,
    120.0,
    200.0,
    _INF,
)

# --- Process / election ---

_up: Final[Gauge] = Gauge(
    "exo_up",
    "1 when the exo node process is serving metrics.",
    ["node_id"],
    registry=_registry,
)

_is_master_gauge: Final[Gauge] = Gauge(
    "exo_is_master",
    "1 if this node is the currently elected master.",
    ["node_id"],
    registry=_registry,
)

# --- Per-generation (master only) ---

_generation_requests: Final[Counter] = Counter(
    "exo_generation_requests_total",
    "Text-generation requests that reached a finish state.",
    ["instance_id", "model_id", "finish_reason"],
    registry=_registry,
)

_prompt_tokens: Final[Counter] = Counter(
    "exo_prompt_tokens_total",
    "Total prompt (prefill) tokens processed.",
    ["instance_id", "model_id"],
    registry=_registry,
)

_generation_tokens: Final[Counter] = Counter(
    "exo_generation_tokens_total",
    "Total tokens generated (decode output).",
    ["instance_id", "model_id"],
    registry=_registry,
)

_prompt_tps: Final[Histogram] = Histogram(
    "exo_prompt_tps",
    "Prompt (prefill) tokens per second observed at completion.",
    ["instance_id", "model_id"],
    buckets=_TPS_BUCKETS_PROMPT,
    registry=_registry,
)

_generation_tps: Final[Histogram] = Histogram(
    "exo_generation_tps",
    "Decode tokens per second observed at completion.",
    ["instance_id", "model_id"],
    buckets=_TPS_BUCKETS_GEN,
    registry=_registry,
)

_prefix_cache_hits: Final[Counter] = Counter(
    "exo_prefix_cache_hits_total",
    "Text-generation requests bucketed by prefix-cache outcome.",
    ["instance_id", "model_id", "hit_kind"],
    registry=_registry,
)

_peak_memory: Final[Gauge] = Gauge(
    "exo_peak_memory_bytes",
    "Peak memory reported at completion of the most recent request.",
    ["instance_id", "model_id"],
    registry=_registry,
)

_chunk_events: Final[Counter] = Counter(
    "exo_chunk_events_total",
    "Count of chunks observed at the master, bucketed by chunk kind.",
    ["kind"],
    registry=_registry,
)

# --- System (per node) ---

_gpu_usage: Final[Gauge] = Gauge(
    "exo_gpu_usage_ratio",
    "GPU usage fraction (0.0–1.0).",
    ["node_id"],
    registry=_registry,
)
_gpu_temp: Final[Gauge] = Gauge(
    "exo_gpu_temp_celsius",
    "Average GPU temperature in Celsius.",
    ["node_id"],
    registry=_registry,
)
_sys_power: Final[Gauge] = Gauge(
    "exo_system_power_watts",
    "Total system power draw in watts.",
    ["node_id"],
    registry=_registry,
)
_pcpu_usage: Final[Gauge] = Gauge(
    "exo_pcpu_usage_ratio",
    "Performance-core CPU usage fraction (0.0–1.0).",
    ["node_id"],
    registry=_registry,
)
_ecpu_usage: Final[Gauge] = Gauge(
    "exo_ecpu_usage_ratio",
    "Efficiency-core CPU usage fraction (0.0–1.0).",
    ["node_id"],
    registry=_registry,
)
_ram_used: Final[Gauge] = Gauge(
    "exo_memory_ram_used_bytes",
    "RAM bytes in use.",
    ["node_id"],
    registry=_registry,
)
_ram_total: Final[Gauge] = Gauge(
    "exo_memory_ram_total_bytes",
    "Total RAM bytes.",
    ["node_id"],
    registry=_registry,
)
_swap_used: Final[Gauge] = Gauge(
    "exo_memory_swap_used_bytes",
    "Swap bytes in use.",
    ["node_id"],
    registry=_registry,
)
_swap_total: Final[Gauge] = Gauge(
    "exo_memory_swap_total_bytes",
    "Total swap bytes.",
    ["node_id"],
    registry=_registry,
)
_disk_available: Final[Gauge] = Gauge(
    "exo_disk_available_bytes",
    "Available bytes on the models directory filesystem.",
    ["node_id"],
    registry=_registry,
)
_disk_total: Final[Gauge] = Gauge(
    "exo_disk_total_bytes",
    "Total bytes on the models directory filesystem.",
    ["node_id"],
    registry=_registry,
)
_rdma_enabled: Final[Gauge] = Gauge(
    "exo_rdma_enabled",
    "1 if rdma_ctl reports enabled on this node, 0 if disabled.",
    ["node_id"],
    registry=_registry,
)

# --- Cluster gauges (refreshed on scrape; master only) ---

_instances_by_model: Final[Gauge] = Gauge(
    "exo_instances",
    "Number of instances registered in the cluster, by model.",
    ["model_id"],
    registry=_registry,
)

_runners_by_status: Final[Gauge] = Gauge(
    "exo_runners",
    "Number of runners in the cluster, by status tag.",
    ["status"],
    registry=_registry,
)


def set_up(node_id: NodeId) -> None:
    _up.labels(node_id=str(node_id)).set(1.0)


def set_is_master(node_id: NodeId, is_master: bool) -> None:
    _is_master_gauge.labels(node_id=str(node_id)).set(1.0 if is_master else 0.0)


def record_chunk_generated(chunk: GenerationChunk) -> None:
    """Cheap counter on every ChunkGenerated. Call only from master."""
    _chunk_events.labels(kind=type(chunk).__name__).inc()


def record_generation_complete(
    instance_id: InstanceId,
    model_id: str,
    stats: GenerationStats,
    finish_reason: str,
) -> None:
    """Record completion of a text-generation request. Call only from master."""
    iid = str(instance_id)
    _generation_requests.labels(
        instance_id=iid, model_id=model_id, finish_reason=finish_reason
    ).inc()
    _prompt_tokens.labels(instance_id=iid, model_id=model_id).inc(stats.prompt_tokens)
    _generation_tokens.labels(instance_id=iid, model_id=model_id).inc(
        stats.generation_tokens
    )
    if stats.prompt_tps > 0.0:
        _prompt_tps.labels(instance_id=iid, model_id=model_id).observe(stats.prompt_tps)
    if stats.generation_tps > 0.0:
        _generation_tps.labels(instance_id=iid, model_id=model_id).observe(
            stats.generation_tps
        )
    _prefix_cache_hits.labels(
        instance_id=iid, model_id=model_id, hit_kind=stats.prefix_cache_hit
    ).inc()
    _peak_memory.labels(instance_id=iid, model_id=model_id).set(
        float(stats.peak_memory_usage.in_bytes)
    )


def record_node_gathered_info(node_id: NodeId, info: GatheredInfo) -> None:
    """Update per-node gauges from a NodeGatheredInfo event. Call only when
    `node_id == self.node_id` on the observer so each node reports its own.
    """
    nid = str(node_id)
    if isinstance(info, MacmonMetrics):
        profile = info.system_profile
        _gpu_usage.labels(node_id=nid).set(float(profile.gpu_usage))
        _gpu_temp.labels(node_id=nid).set(float(profile.temp))
        _sys_power.labels(node_id=nid).set(float(profile.sys_power))
        _pcpu_usage.labels(node_id=nid).set(float(profile.pcpu_usage))
        _ecpu_usage.labels(node_id=nid).set(float(profile.ecpu_usage))
        _update_memory(nid, info.memory)
    elif isinstance(info, MemoryUsage):
        _update_memory(nid, info)
    elif isinstance(info, NodeDiskUsage):
        usage = info.disk_usage
        _disk_available.labels(node_id=nid).set(float(usage.available.in_bytes))
        _disk_total.labels(node_id=nid).set(float(usage.total.in_bytes))
    elif isinstance(info, RdmaCtlStatus):
        _rdma_enabled.labels(node_id=nid).set(1.0 if info.enabled else 0.0)
    # Other GatheredInfo variants are static inventory; nothing to export.


def _update_memory(node_id: str, mem: MemoryUsage) -> None:
    _ram_total.labels(node_id=node_id).set(float(mem.ram_total.in_bytes))
    _ram_used.labels(node_id=node_id).set(
        float(mem.ram_total.in_bytes - mem.ram_available.in_bytes)
    )
    _swap_total.labels(node_id=node_id).set(float(mem.swap_total.in_bytes))
    _swap_used.labels(node_id=node_id).set(
        float(mem.swap_total.in_bytes - mem.swap_available.in_bytes)
    )


def refresh_cluster_gauges(
    instances: Mapping[InstanceId, Instance],
    runners: Mapping[RunnerId, RunnerStatus],
) -> None:
    """Refresh scrape-time cluster gauges from master state."""
    _instances_by_model.clear()
    per_model: dict[str, int] = {}
    for inst in instances.values():
        mid = str(inst.shard_assignments.model_id)
        per_model[mid] = per_model.get(mid, 0) + 1
    for mid, count in per_model.items():
        _instances_by_model.labels(model_id=mid).set(float(count))

    _runners_by_status.clear()
    per_status: dict[str, int] = {}
    for status in runners.values():
        tag = type(status).__name__
        per_status[tag] = per_status.get(tag, 0) + 1
    for tag, count in per_status.items():
        _runners_by_status.labels(status=tag).set(float(count))


def render_latest() -> bytes:
    """Render current registry state as Prometheus text exposition."""
    return generate_latest(_registry)
