import pytest

from shared.types.common import NodeId
from shared.types.profiling import (
    MemoryPerformanceProfile,
    NodePerformanceProfile,
    SystemPerformanceProfile,
)
from shared.types.topology import Connection, ConnectionProfile, Node


@pytest.fixture
def create_node():
    def _create_node(memory: int, node_id: NodeId | None = None) -> Node:
        if node_id is None:
            node_id = NodeId()
        return Node(
            node_id=node_id, 
            node_profile=NodePerformanceProfile(
                model_id="test", 
                chip_id="test", 
                memory=MemoryPerformanceProfile(
                    ram_total=1000, 
                    ram_available=memory, 
                    swap_total=1000, 
                    swap_available=1000
                ), 
                network_interfaces=[], 
                system=SystemPerformanceProfile(flops_fp16=1000)
            )
        )

    return _create_node


@pytest.fixture
def create_connection():
    def _create_connection(source_node_id: NodeId, sink_node_id: NodeId) -> Connection:
        return Connection(
            source_node_id=source_node_id,
            sink_node_id=sink_node_id,
            source_multiaddr="/ip4/127.0.0.1/tcp/1234",
            sink_multiaddr="/ip4/127.0.0.1/tcp/1235",
            connection_profile=ConnectionProfile(throughput=1000, latency=1000, jitter=1000)
        )
    return _create_connection