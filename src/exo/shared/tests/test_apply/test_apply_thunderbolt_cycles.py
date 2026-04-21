from exo.shared.apply import apply_topology_edge_created
from exo.shared.topology import Topology
from exo.shared.types.common import NodeId
from exo.shared.types.events import TopologyEdgeCreated
from exo.shared.types.profiling import (
    NetworkInterfaceInfo,
    NodeNetworkInfo,
    ThunderboltBridgeStatus,
)
from exo.shared.types.state import State
from exo.shared.types.topology import Connection, RDMAConnection


ZEALOUS = NodeId("zealous")
GANDALF = NodeId("gandalf")
SARUMAN = NodeId("saruman")


def _tb_status(*node_ids: NodeId) -> dict[NodeId, ThunderboltBridgeStatus]:
    return {
        node_id: ThunderboltBridgeStatus(
            enabled=True, exists=True, service_name="Thunderbolt Bridge"
        )
        for node_id in node_ids
    }


def _network(*pairs: tuple[NodeId, str]) -> dict[NodeId, NodeNetworkInfo]:
    return {
        node_id: NodeNetworkInfo(
            interfaces=[
                NetworkInterfaceInfo(
                    name="bridge0",
                    ip_address=ip,
                    interface_type="thunderbolt",
                )
            ]
        )
        for node_id, ip in pairs
    }


def test_apply_topology_edge_created_recomputes_thunderbolt_cycles():
    topology = Topology()
    topology.add_connection(
        Connection(
            source=ZEALOUS,
            sink=GANDALF,
            edge=RDMAConnection(source_rdma_iface="rdma_en2", sink_rdma_iface="rdma_en3"),
        )
    )
    topology.add_connection(
        Connection(
            source=GANDALF,
            sink=SARUMAN,
            edge=RDMAConnection(source_rdma_iface="rdma_en4", sink_rdma_iface="rdma_en2"),
        )
    )

    state = State(
        topology=topology,
        node_thunderbolt_bridge=_tb_status(ZEALOUS, GANDALF, SARUMAN),
        node_network=_network(
            (ZEALOUS, "10.0.0.1"),
            (GANDALF, "10.0.0.2"),
            (SARUMAN, "10.0.0.3"),
        ),
    )

    new_state = apply_topology_edge_created(
        TopologyEdgeCreated(
            conn=Connection(
                source=SARUMAN,
                sink=ZEALOUS,
                edge=RDMAConnection(
                    source_rdma_iface="rdma_en2", sink_rdma_iface="rdma_en4"
                ),
            )
        ),
        state,
    )

    assert len(new_state.thunderbolt_bridge_cycles) == 1
    assert set(new_state.thunderbolt_bridge_cycles[0]) == {ZEALOUS, GANDALF, SARUMAN}
