from ipaddress import ip_address
from itertools import count

from exo.shared.types.common import NodeId
from exo.shared.types.profiling import (
    MemoryPerformanceProfile,
    NodePerformanceProfile,
    SystemPerformanceProfile,
)
from exo.shared.types.topology import Connection, ConnectionProfile, NodeInfo

ip_octet_iter = count()


def create_node(memory: int, node_id: NodeId | None = None) -> NodeInfo:
    if node_id is None:
        node_id = NodeId()
    return NodeInfo(
        node_id=node_id,
        node_profile=NodePerformanceProfile(
            model_id="test",
            chip_id="test",
            friendly_name="test",
            memory=MemoryPerformanceProfile.from_bytes(
                ram_total=1000,
                ram_available=memory,
                swap_total=1000,
                swap_available=1000,
            ),
            network_interfaces=[],
            system=SystemPerformanceProfile(),
        ),
    )


def create_connection(
    source_node_id: NodeId,
    sink_node_id: NodeId,
    *,
    ip_octet: int | None = None,
) -> Connection:
    global ip_octet_iter

    return Connection(
        source_id=source_node_id,
        sink_id=sink_node_id,
        sink_addr=ip_address(
            f"169.254.0.{ip_octet if ip_octet is not None else next(ip_octet_iter)}"
        ),
        connection_profile=ConnectionProfile(
            throughput=1000, latency=1000, jitter=1000
        ),
    )
