import socket
from collections.abc import Mapping

from anyio import create_task_group, to_thread

from exo.shared.topology import Topology
from exo.shared.types.common import NodeId
from exo.shared.types.profiling import NodePerformanceProfile


# TODO: ref. api port
async def check_reachability(
    target_ip: str, target_node_id: NodeId, out: dict[NodeId, set[str]]
) -> None:
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.settimeout(1)  # 1 second timeout
    try:
        result = await to_thread.run_sync(sock.connect_ex, (target_ip, 52415))
    except socket.gaierror:
        # seems to throw on ipv6 loopback. oh well
        # logger.warning(f"invalid {target_ip=}")
        return
    finally:
        sock.close()

    if result == 0:
        if target_node_id not in out:
            out[target_node_id] = set()
        out[target_node_id].add(target_ip)


async def check_reachable(
    topology: Topology, profiles: Mapping[NodeId, NodePerformanceProfile]
) -> dict[NodeId, set[str]]:
    reachable: dict[NodeId, set[str]] = {}
    async with create_task_group() as tg:
        for node in topology.list_nodes():
            if node not in profiles:
                continue
            for iface in profiles[node].network_interfaces:
                tg.start_soon(check_reachability, iface.ip_address, node, reachable)

    return reachable
