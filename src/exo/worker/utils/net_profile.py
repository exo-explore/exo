import socket
from ipaddress import ip_address

from anyio import create_task_group, to_thread

from exo.routing.connection_message import IpAddress
from exo.shared.topology import Topology
from exo.shared.types.common import NodeId


# TODO: ref. api port
async def check_reachability(
    target_ip: IpAddress, target_node_id: NodeId, out: dict[NodeId, set[IpAddress]]
) -> None:
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.settimeout(1)  # 1 second timeout
    try:
        result = await to_thread.run_sync(sock.connect_ex, (str(target_ip), 52415))
    except socket.gaierror:
        # seems to throw on ipv6 loopback. oh well
        # logger.warning(f"invalid {target_ip=}")
        return
    finally:
        sock.close()

    if result == 0:
        if target_node_id not in out:
            out[target_node_id] = set()
        out[target_node_id].add(ip_address(target_ip))


async def check_reachable(topology: Topology) -> dict[NodeId, set[IpAddress]]:
    reachable: dict[NodeId, set[IpAddress]] = {}
    async with create_task_group() as tg:
        for node in topology.list_nodes():
            if not node.node_profile:
                continue
            for iface in node.node_profile.network_interfaces:
                tg.start_soon(
                    check_reachability, iface.ip_address, node.node_id, reachable
                )

    return reachable
