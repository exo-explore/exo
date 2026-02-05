from collections.abc import Mapping

import anyio
import httpx
from anyio import create_task_group
from loguru import logger

from exo.shared.topology import Topology
from exo.shared.types.common import NodeId
from exo.shared.types.profiling import NodeNetworkInfo

REACHABILITY_ATTEMPTS = 3


async def check_reachability(
    target_ip: str,
    expected_node_id: NodeId,
    out: dict[NodeId, set[str]],
    client: httpx.AsyncClient,
) -> None:
    """Check if a node is reachable at the given IP and verify its identity."""
    if ":" in target_ip:
        # TODO: use real IpAddress types
        url = f"http://[{target_ip}]:52415/node_id"
    else:
        url = f"http://{target_ip}:52415/node_id"

    remote_node_id = None
    last_error = None

    for _ in range(REACHABILITY_ATTEMPTS):
        try:
            r = await client.get(url)
            if r.status_code != 200:
                await anyio.sleep(1)
                continue

            body = r.text.strip().strip('"')
            if not body:
                await anyio.sleep(1)
                continue

            remote_node_id = NodeId(body)
            break

        # expected failure cases
        except (
            httpx.TimeoutException,
            httpx.NetworkError,
        ):
            await anyio.sleep(1)

        # other failures should be logged on last attempt
        except httpx.HTTPError as e:
            last_error = e
            await anyio.sleep(1)

    if last_error is not None:
        logger.warning(
            f"connect error {type(last_error).__name__} from {target_ip} after {REACHABILITY_ATTEMPTS} attempts; treating as down"
        )

    if remote_node_id is None:
        return

    if remote_node_id != expected_node_id:
        logger.debug(
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
    self_node_id: NodeId,
    node_network: Mapping[NodeId, NodeNetworkInfo],
) -> dict[NodeId, set[str]]:
    """Check which nodes are reachable and return their IPs."""

    reachable: dict[NodeId, set[str]] = {}

    # these are intentionally httpx's defaults so we can tune them later
    timeout = httpx.Timeout(timeout=5.0)
    limits = httpx.Limits(
        max_connections=100,
        max_keepalive_connections=20,
        keepalive_expiry=5,
    )

    async with (
        httpx.AsyncClient(timeout=timeout, limits=limits) as client,
        create_task_group() as tg,
    ):
        for node_id in topology.list_nodes():
            if node_id not in node_network:
                continue
            if node_id == self_node_id:
                continue
            for iface in node_network[node_id].interfaces:
                tg.start_soon(
                    check_reachability,
                    iface.ip_address,
                    node_id,
                    reachable,
                    client,
                )

    return reachable
