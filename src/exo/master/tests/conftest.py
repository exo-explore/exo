from exo.shared.types.multiaddr import Multiaddr
from exo.shared.types.profiling import (
    MemoryUsage,
    NetworkInterfaceInfo,
    NodePerformanceProfile,
    SystemPerformanceProfile,
)
from exo.shared.types.topology import RDMAConnection, SocketConnection


def create_node_profile(memory: int) -> NodePerformanceProfile:
    return NodePerformanceProfile(
        model_id="test",
        chip_id="test",
        friendly_name="test",
        memory=MemoryUsage.from_bytes(
            ram_total=1000,
            ram_available=memory,
            swap_total=1000,
            swap_available=1000,
        ),
        network_interfaces=[
            NetworkInterfaceInfo(name="en0", ip_address=f"169.254.0.{i}")
            for i in range(10)
        ],
        system=SystemPerformanceProfile(),
    )


def create_socket_connection(ip: int, sink_port: int = 1234) -> SocketConnection:
    return SocketConnection(
        sink_multiaddr=Multiaddr(address=f"/ip4/169.254.0.{ip}/tcp/{sink_port}"),
    )


def create_rdma_connection(iface: int) -> RDMAConnection:
    return RDMAConnection(
        source_rdma_iface=f"rdma_en{iface}", sink_rdma_iface=f"rdma_en{iface}"
    )
