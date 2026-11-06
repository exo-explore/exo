import time
from datetime import UTC, datetime

from exo.shared.apply import apply_node_gathered_info, apply_node_timed_out
from exo.shared.types.common import NodeId
from exo.shared.types.events import NodeGatheredInfo, NodeTimedOut
from exo.shared.types.profiling import (
    LinkTelemetry,
    NetworkInterfaceInfo,
    NodeLinkTelemetry,
    NodeNetworkInfo,
)
from exo.shared.types.state import State
from exo.utils.info_gatherer.info_gatherer import NodeLinkTelemetryData
from exo.utils.info_gatherer.net_profile import (
    InterfaceBandwidthTracker,
    LinkTelemetryRegistry,
)


def test_link_telemetry_registry_ema_smoothing():
    registry = LinkTelemetryRegistry()
    node_id = NodeId("node-peer-1")
    ip = "192.168.1.10"

    # First sample: 10.0 ms
    s1 = registry.record_latency(ip, node_id, 10.0)
    assert s1 == 10.0
    assert registry.get_latency(ip, node_id) == 10.0

    # Second sample: 20.0 ms -> smooth = 0.3 * 20 + 0.7 * 10 = 13.0 ms
    s2 = registry.record_latency(ip, node_id, 20.0)
    assert abs(s2 - 13.0) < 0.01

    # Third sample: 5.0 ms -> smooth = 0.3 * 5 + 0.7 * 13 = 10.6 ms
    s3 = registry.record_latency(ip, node_id, 5.0)
    assert abs(s3 - 10.6) < 0.01


def test_capacity_estimation():
    assert LinkTelemetryRegistry.estimate_capacity_mbps("tb5") == 10000.0
    assert LinkTelemetryRegistry.estimate_capacity_mbps("Thunderbolt 5 Port") == 10000.0
    assert LinkTelemetryRegistry.estimate_capacity_mbps("thunderbolt") == 5000.0
    assert LinkTelemetryRegistry.estimate_capacity_mbps("rdma") == 12500.0
    assert LinkTelemetryRegistry.estimate_capacity_mbps("ethernet") == 125.0
    assert LinkTelemetryRegistry.estimate_capacity_mbps("wifi") == 50.0


def test_generate_node_telemetry():
    registry = LinkTelemetryRegistry()
    self_nid = NodeId("self-node")
    peer_nid = NodeId("peer-node")

    registry.record_latency("10.0.0.2", peer_nid, 1.5)

    node_network = {
        peer_nid: NodeNetworkInfo(
            interfaces=[
                NetworkInterfaceInfo(
                    name="tb0",
                    ip_address="10.0.0.2",
                    interface_type="thunderbolt",
                )
            ]
        )
    }

    telemetry = registry.generate_node_telemetry(self_nid, node_network)
    assert isinstance(telemetry, NodeLinkTelemetry)
    assert len(telemetry.links) == 1

    link = telemetry.links[0]
    assert link.source == self_nid
    assert link.target == peer_nid
    assert link.target_ip == "10.0.0.2"
    assert link.latency_ms == 1.5
    assert link.bandwidth_capacity_mbps == 5000.0


def test_apply_node_link_telemetry_state():
    state = State()
    node_a = NodeId("node-a")
    node_b = NodeId("node-b")

    telemetry_data = NodeLinkTelemetryData(
        telemetry=NodeLinkTelemetry(
            links=[
                LinkTelemetry(
                    source=node_a,
                    target=node_b,
                    target_ip="192.168.1.20",
                    latency_ms=2.4,
                    bandwidth_capacity_mbps=10000.0,
                    used_bandwidth_mbps=45.6,
                    last_updated=time.time(),
                )
            ]
        )
    )

    event = NodeGatheredInfo(
        node_id=node_a,
        when=datetime.now(UTC).isoformat(),
        info=telemetry_data,
    )

    new_state = apply_node_gathered_info(event, state)
    assert node_a in new_state.node_link_telemetry
    assert len(new_state.node_link_telemetry[node_a].links) == 1
    assert new_state.node_link_telemetry[node_a].links[0].used_bandwidth_mbps == 45.6

    # Test node timeout cleanup
    timed_out_state = apply_node_timed_out(NodeTimedOut(node_id=node_a), new_state)
    assert node_a not in timed_out_state.node_link_telemetry


def test_interface_bandwidth_tracker():
    tracker = InterfaceBandwidthTracker()
    # First measurement initializes baseline
    rates_init = tracker.sample_throughput()
    assert isinstance(rates_init, dict)
