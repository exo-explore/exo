from exo.api.types import GenerationStats
from exo.metrics import (
    CONTENT_TYPE_LATEST,
    record_generation_complete,
    record_node_gathered_info,
    render_latest,
    set_is_master,
)
from exo.metrics.metrics import set_up as metrics_set_up
from exo.shared.types.common import NodeId
from exo.shared.types.memory import Memory
from exo.shared.types.profiling import (
    DiskUsage,
    MemoryUsage,
    SystemPerformanceProfile,
)
from exo.shared.types.worker.instances import InstanceId
from exo.utils.info_gatherer.info_gatherer import NodeDiskUsage, RdmaCtlStatus
from exo.utils.info_gatherer.macmon import MacmonMetrics


def _node_id(value: str) -> NodeId:
    return NodeId(value)


def test_content_type_is_prom_text():
    assert "text/plain" in CONTENT_TYPE_LATEST
    assert "version=" in CONTENT_TYPE_LATEST


def test_render_is_bytes_and_includes_exo_up():
    node = _node_id("test-node-1")
    metrics_set_up(node)
    output = render_latest()
    assert isinstance(output, bytes)
    text = output.decode("utf-8")
    assert "exo_up" in text
    assert f'node_id="{node}"' in text


def test_set_is_master_flips_gauge():
    node = _node_id("test-node-2")
    set_is_master(node, True)
    assert f'exo_is_master{{node_id="{node}"}} 1.0' in render_latest().decode()
    set_is_master(node, False)
    assert f'exo_is_master{{node_id="{node}"}} 0.0' in render_latest().decode()


def test_record_generation_complete_populates_counters():
    instance = InstanceId("inst-abc")
    model_id = "mlx-community/fake-model"
    stats = GenerationStats(
        prompt_tps=123.4,
        generation_tps=56.7,
        prompt_tokens=100,
        generation_tokens=50,
        peak_memory_usage=Memory(in_bytes=1024 * 1024 * 1024),
        prefix_cache_hit="partial",
    )
    record_generation_complete(
        instance_id=instance,
        model_id=model_id,
        stats=stats,
        finish_reason="stop",
    )
    text = render_latest().decode()
    assert 'exo_generation_requests_total{finish_reason="stop"' in text
    assert f'instance_id="{instance}"' in text
    assert f'model_id="{model_id}"' in text
    assert "exo_prompt_tokens_total" in text
    assert "exo_generation_tokens_total" in text
    assert 'hit_kind="partial"' in text


def test_record_node_gathered_info_macmon_sets_gauges():
    node = _node_id("test-node-macmon")
    mem = MemoryUsage(
        ram_total=Memory(in_bytes=128 * 1024 * 1024 * 1024),
        ram_available=Memory(in_bytes=100 * 1024 * 1024 * 1024),
        swap_total=Memory(in_bytes=0),
        swap_available=Memory(in_bytes=0),
    )
    info = MacmonMetrics(
        system_profile=SystemPerformanceProfile(
            gpu_usage=0.42,
            temp=55.0,
            sys_power=180.0,
            pcpu_usage=0.30,
            ecpu_usage=0.05,
        ),
        memory=mem,
    )
    record_node_gathered_info(node, info)
    text = render_latest().decode()
    assert f'exo_gpu_usage_ratio{{node_id="{node}"}} 0.42' in text
    assert f'exo_system_power_watts{{node_id="{node}"}} 180.0' in text
    assert f'exo_memory_ram_total_bytes{{node_id="{node}"}}' in text


def test_record_node_gathered_info_rdma_and_disk():
    node = _node_id("test-node-disk")
    record_node_gathered_info(node, RdmaCtlStatus(enabled=True))
    record_node_gathered_info(
        node,
        NodeDiskUsage(
            disk_usage=DiskUsage(
                total=Memory(in_bytes=1_000_000_000),
                available=Memory(in_bytes=500_000_000),
            )
        ),
    )
    text = render_latest().decode()
    assert f'exo_rdma_enabled{{node_id="{node}"}} 1.0' in text
    assert f'exo_disk_available_bytes{{node_id="{node}"}} 5e+08' in text
