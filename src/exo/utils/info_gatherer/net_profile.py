import time
from collections import defaultdict
from collections.abc import AsyncGenerator, Mapping

import anyio
import httpx
from anyio import create_task_group
from loguru import logger

from exo.shared.topology import Topology
from exo.shared.types.common import NodeId
from exo.shared.types.profiling import NodeNetworkInfo
from exo.utils.channels import Sender, channel

REACHABILITY_ATTEMPTS = 3

# Thresholds below which a latency change is treated as measurement noise
LATENCY_NOISE_FLOOR_MS = 2.0
LATENCY_CHANGE_FACTOR = 2.0


def latency_changed_materially(old_ms: float | None, new_ms: float) -> bool:
    """Whether a new latency measurement differs enough to be worth republishing.

    Requires both an absolute change above the noise floor and a factor-of-two
    change, so probe jitter on fast links never churns the topology.
    """
    if old_ms is None:
        return True
    if abs(new_ms - old_ms) <= LATENCY_NOISE_FLOOR_MS:
        return False
    return (
        new_ms < old_ms / LATENCY_CHANGE_FACTOR
        or new_ms > old_ms * LATENCY_CHANGE_FACTOR
    )


async def check_reachability(
    target_ip: str,
    expected_node_id: NodeId,
    out: dict[NodeId, set[str]],
    client: httpx.AsyncClient,
    api_port: int,
) -> float | None:
    """Check if a node is reachable at the given IP and verify its identity.

    Returns the round-trip time of the successful probe in milliseconds, or
    None if the node was not reachable at this IP.
    """
    if ":" in target_ip:
        # TODO: use real IpAddress types
        url = f"http://[{target_ip}]:{api_port}/node_id"
    else:
        url = f"http://{target_ip}:{api_port}/node_id"

    remote_node_id = None
    last_error = None
    latency_ms = None

    for _ in range(REACHABILITY_ATTEMPTS):
        try:
            probe_start = time.perf_counter()
            r = await client.get(url)
            probe_elapsed_ms = (time.perf_counter() - probe_start) * 1000
            if r.status_code != 200:
                await anyio.sleep(1)
                continue

            body = r.text.strip().strip('"')
            if not body:
                await anyio.sleep(1)
                continue

            remote_node_id = NodeId(body)
            latency_ms = probe_elapsed_ms
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
        return None

    if remote_node_id != expected_node_id:
        logger.debug(
            f"Discovered node with unexpected node_id; "
            f"ip={target_ip}, expected_node_id={expected_node_id}, "
            f"remote_node_id={remote_node_id}"
        )
        return None

    if remote_node_id not in out:
        out[remote_node_id] = set()
    out[remote_node_id].add(target_ip)
    return latency_ms


async def check_reachable(
    topology: Topology,
    self_node_id: NodeId,
    node_network: Mapping[NodeId, NodeNetworkInfo],
    api_port: int,
) -> AsyncGenerator[tuple[str, NodeId, float], None]:
    """Yield (ip, node_id, latency_ms) tuples as reachability probes complete."""

    send, recv = channel[tuple[str, NodeId, float]]()

    # these are intentionally httpx's defaults so we can tune them later
    timeout = httpx.Timeout(timeout=5.0)
    limits = httpx.Limits(
        max_connections=100,
        max_keepalive_connections=20,
        keepalive_expiry=5,
    )

    async def _probe(
        target_ip: str,
        expected_node_id: NodeId,
        client: httpx.AsyncClient,
        send: Sender[tuple[str, NodeId, float]],
    ) -> None:
        async with send:
            out: defaultdict[NodeId, set[str]] = defaultdict(set)
            latency_ms = await check_reachability(
                target_ip, expected_node_id, out, client, api_port
            )
            if expected_node_id in out and latency_ms is not None:
                await send.send((target_ip, expected_node_id, latency_ms))

    async with (
        httpx.AsyncClient(timeout=timeout, limits=limits, verify=False) as client,
        create_task_group() as tg,
    ):
        for node_id in topology.list_nodes():
            if node_id not in node_network:
                continue
            if node_id == self_node_id:
                continue
            for iface in node_network[node_id].interfaces:
                tg.start_soon(_probe, iface.ip_address, node_id, client, send.clone())
        send.close()

        with recv:
            async for item in recv:
                yield item
