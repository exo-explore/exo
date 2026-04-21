from exo.shared.topology import Topology
from exo.shared.types.common import NodeId
from exo.shared.types.multiaddr import Multiaddr
from exo.shared.types.profiling import (
    NetworkInterfaceInfo,
    NodeNetworkInfo,
    ThunderboltBridgeStatus,
)
from exo.shared.types.topology import Connection, RDMAConnection, SocketConnection


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


def test_bidirectional_two_node_rdma_pair_is_not_flagged_as_tb_cycle():
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
            sink=ZEALOUS,
            edge=RDMAConnection(source_rdma_iface="rdma_en3", sink_rdma_iface="rdma_en2"),
        )
    )

    cycles = topology.get_thunderbolt_bridge_cycles(
        _tb_status(ZEALOUS, GANDALF),
        _network((ZEALOUS, "10.0.0.1"), (GANDALF, "10.0.0.2")),
    )

    assert cycles == []


def test_three_node_tb_loop_is_flagged():
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
    topology.add_connection(
        Connection(
            source=SARUMAN,
            sink=ZEALOUS,
            edge=RDMAConnection(source_rdma_iface="rdma_en2", sink_rdma_iface="rdma_en4"),
        )
    )

    cycles = topology.get_thunderbolt_bridge_cycles(
        _tb_status(ZEALOUS, GANDALF, SARUMAN),
        _network(
            (ZEALOUS, "10.0.0.1"),
            (GANDALF, "10.0.0.2"),
            (SARUMAN, "10.0.0.3"),
        ),
    )

    assert len(cycles) == 1
    assert set(cycles[0]) == {ZEALOUS, GANDALF, SARUMAN}


def test_socket_reachability_over_tb_subnet_does_not_create_false_triangle():
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
            sink=ZEALOUS,
            edge=RDMAConnection(source_rdma_iface="rdma_en3", sink_rdma_iface="rdma_en2"),
        )
    )
    topology.add_connection(
        Connection(
            source=ZEALOUS,
            sink=SARUMAN,
            edge=SocketConnection(
                sink_multiaddr=Multiaddr(address="/ip4/10.0.0.3/tcp/52415")
            ),
        )
    )
    topology.add_connection(
        Connection(
            source=SARUMAN,
            sink=ZEALOUS,
            edge=SocketConnection(
                sink_multiaddr=Multiaddr(address="/ip4/10.0.0.1/tcp/52415")
            ),
        )
    )
    topology.add_connection(
        Connection(
            source=GANDALF,
            sink=SARUMAN,
            edge=SocketConnection(
                sink_multiaddr=Multiaddr(address="/ip4/10.0.0.3/tcp/52415")
            ),
        )
    )
    topology.add_connection(
        Connection(
            source=SARUMAN,
            sink=GANDALF,
            edge=SocketConnection(
                sink_multiaddr=Multiaddr(address="/ip4/10.0.0.2/tcp/52415")
            ),
        )
    )

    cycles = topology.get_thunderbolt_bridge_cycles(
        _tb_status(ZEALOUS, GANDALF, SARUMAN),
        _network(
            (ZEALOUS, "10.0.0.1"),
            (GANDALF, "10.0.0.2"),
            (SARUMAN, "10.0.0.3"),
        ),
    )

    assert cycles == []
