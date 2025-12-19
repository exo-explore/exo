from exo.shared.types.common import NodeId
from exo.shared.types.multiaddr import Multiaddr
from exo.shared.types.profiling import (
    MemoryUsage,
    NodePerformanceProfile,
    SystemPerformanceProfile,
)
from exo.shared.types.topology import SocketConnection


def create_node(memory: int, node_id: NodeId | None = None) -> tuple[NodeId, NodePerformanceProfile]:
        if node_id is None:
            node_id = NodeId()
        return (node_id,
            NodePerformanceProfile(
                model_id="test",
                chip_id="test",
                friendly_name="test",
                memory=MemoryUsage.from_bytes(
                    ram_total=1000,
                    ram_available=memory,
                    swap_total=1000,
                    swap_available=1000,
                ),
                network_interfaces=[],
                system=SystemPerformanceProfile(),
            ),
        )


# TODO: this is a hack to get the port for the send_back_multiaddr
def create_connection(sink_port: int, ip: int) -> SocketConnection:
    return SocketConnection(
        sink_multiaddr=Multiaddr(
            address=f"/ip4/169.254.0.{ip}/tcp/{sink_port}"
        ),
    )

