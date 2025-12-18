from typing import Callable

import pytest

from exo.shared.types.common import NodeId
from exo.shared.types.multiaddr import Multiaddr
from exo.shared.types.profiling import (
    MemoryPerformanceProfile,
    NodePerformanceProfile,
    SystemPerformanceProfile,
)
from exo.shared.types.topology import Connection, ConnectionProfile, NodeInfo


@pytest.fixture
def create_node():
    def _create_node(memory: int, node_id: NodeId | None = None) -> NodeInfo:
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

    return _create_node


# TODO: this is a hack to get the port for the send_back_multiaddr
@pytest.fixture
def create_connection() -> Callable[[NodeId, NodeId, int | None], Connection]:
    port_counter = 1235
    ip_counter = 1

    def _create_connection(
        source_node_id: NodeId, sink_node_id: NodeId, send_back_port: int | None = None
    ) -> Connection:
        nonlocal port_counter
        nonlocal ip_counter
        # assign unique ips
        ip_counter += 1
        if send_back_port is None:
            send_back_port = port_counter
            port_counter += 1
        return Connection(
            local_node_id=source_node_id,
            send_back_node_id=sink_node_id,
            send_back_multiaddr=Multiaddr(
                address=f"/ip4/169.254.0.{ip_counter}/tcp/{send_back_port}"
            ),
            connection_profile=ConnectionProfile(
                throughput=1000, latency=1000, jitter=1000
            ),
        )

    return _create_connection
