from ipaddress import ip_address
from itertools import count

from exo.routing.connection_message import SocketAddress
from exo.shared.types.common import NodeId
from exo.shared.types.profiling import (
    MemoryPerformanceProfile,
    NodePerformanceProfile,
    SystemPerformanceProfile,
)
from exo.shared.types.topology import Connection, ConnectionProfile, NodeInfo

ip_octet_iter = count()
port_iter = count(5000)


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
    port: int | None = None,
    ip_octet: int | None = None,
) -> Connection:
    global ip_octet_iter
    global port_iter

    return Connection(
        source_id=source_node_id,
        sink_id=sink_node_id,
        sink_addr=SocketAddress(
            ip=ip_address(
                f"169.254.0.{ip_octet if ip_octet is not None else next(ip_octet_iter)}"
            ),
            port=port if port is not None else next(port_iter),
            zone_id=None,
        ),
        connection_profile=ConnectionProfile(
            throughput=1000, latency=1000, jitter=1000
        ),
    )
