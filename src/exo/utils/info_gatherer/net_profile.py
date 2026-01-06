import http.client
from collections.abc import Mapping

from anyio import create_task_group, to_thread
from loguru import logger

from exo.shared.topology import Topology
from exo.shared.types.common import NodeId
from exo.shared.types.profiling import NodePerformanceProfile


async def check_reachability(
    target_ip: str,
    expected_node_id: NodeId,
    self_node_id: NodeId,
    out: dict[NodeId, set[str]],
) -> None:
    """Check if a node is reachable at the given IP and verify its identity."""

    def _fetch_remote_node_id() -> NodeId | None:
        connection = http.client.HTTPConnection(target_ip, 52415, timeout=1)
        try:
            connection.request("GET", "/node_id")
            response = connection.getresponse()
            if response.status != 200:
                return None

            body = response.read().decode("utf-8").strip()

            # Strip quotes if present (JSON string response)
            if body.startswith('"') and body.endswith('"') and len(body) >= 2:
                body = body[1:-1]

            return NodeId(body) or None
        except OSError:
            return None
        except http.client.HTTPException:
            return None
        finally:
            connection.close()

    remote_node_id = await to_thread.run_sync(_fetch_remote_node_id)
    if remote_node_id is None:
        return

    if remote_node_id == self_node_id:
        return

    if remote_node_id != expected_node_id:
        logger.warning(
            f"Discovered node with unexpected node_id; "
            f"ip={target_ip}, expected_node_id={expected_node_id}, "
            f"remote_node_id={remote_node_id}"
        )
        return

    if remote_node_id not in out:
        out[remote_node_id] = set()
    out[remote_node_id].add(target_ip)


async def check_reachable(
    topology: Topology,
    profiles: Mapping[NodeId, NodePerformanceProfile],
    self_node_id: NodeId,
) -> dict[NodeId, set[str]]:
    reachable: dict[NodeId, set[str]] = {}
    async with create_task_group() as tg:
        for node_id in topology.list_nodes():
            if not node_id not in profiles:
                continue
            for iface in profiles[node_id].network_interfaces:
                tg.start_soon(
                    check_reachability,
                    iface.ip_address,
                    node_id,
                    self_node_id,
                    reachable,
                )

    return reachable
