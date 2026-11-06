import time
from collections import defaultdict
from collections.abc import AsyncGenerator, Mapping

import anyio
import httpx
import psutil
from anyio import create_task_group
from loguru import logger

from exo.shared.topology import Topology
from exo.shared.types.common import NodeId
from exo.shared.types.profiling import (
    LinkTelemetry,
    NodeLinkTelemetry,
    NodeNetworkInfo,
)
from exo.utils.channels import Sender, channel

REACHABILITY_ATTEMPTS = 3


class InterfaceBandwidthTracker:
    """Tracks per-interface real-time bandwidth usage (MB/s).

    Implements official TODO.md #5: monitor how much bandwidth each link is using.
    """

    def __init__(self) -> None:
        self._last_counters: dict[str, tuple[float, int, int]] = {}

    def sample_throughput(self) -> dict[str, float]:
        """Return current throughput in MB/s per network interface."""
        now = time.monotonic()
        rates: dict[str, float] = {}
        try:
            current_io = psutil.net_io_counters(pernic=True)
        except (OSError, ValueError):
            return rates

        for iface, io in current_io.items():
            if iface in self._last_counters:
                last_time, last_sent, last_recv = self._last_counters[iface]
                dt = now - last_time
                if dt > 0.1:
                    delta_bytes = (io.bytes_sent - last_sent) + (
                        io.bytes_recv - last_recv
                    )
                    # Convert to MB/s
                    mb_s = max(0.0, (delta_bytes / dt) / (1024.0 * 1024.0))
                    rates[iface] = round(mb_s, 2)
            self._last_counters[iface] = (now, io.bytes_sent, io.bytes_recv)

        return rates


class LinkTelemetryRegistry:
    """Maintains link-level profiled latency (RTT) and estimated bandwidth capacity.

    Implements official TODO.md #4 and #5.
    """

    def __init__(self) -> None:
        self._latencies: dict[tuple[str, str], float] = {}  # (ip, node_id) -> smoothed ms
        self._bandwidth_tracker = InterfaceBandwidthTracker()

    def record_latency(self, ip: str, node_id: NodeId, sample_ms: float) -> float:
        """Record a latency sample and apply exponential moving average smoothing."""
        key = (ip, str(node_id))
        sample_ms = max(0.05, sample_ms)
        if key in self._latencies:
            prev = self._latencies[key]
            # EMA smoothing factor alpha = 0.3
            smooth = 0.3 * sample_ms + 0.7 * prev
        else:
            smooth = sample_ms
        self._latencies[key] = round(smooth, 2)
        return self._latencies[key]

    def get_latency(self, ip: str, node_id: NodeId) -> float:
        return self._latencies.get((ip, str(node_id)), 0.0)

    @staticmethod
    def estimate_capacity_mbps(interface_type: str) -> float:
        """Estimate link bandwidth capacity in MB/s from interface type."""
        t = interface_type.lower()
        if "tb5" in t or "thunderbolt 5" in t:
            return 10000.0  # 80 Gbps ~ 10,000 MB/s
        if "thunderbolt" in t or "tb" in t:
            return 5000.0  # 40 Gbps ~ 5,000 MB/s
        if "rdma" in t:
            return 12500.0  # 100 Gbps ~ 12,500 MB/s
        if "ethernet" in t or "maybe_ethernet" in t:
            return 125.0  # 1 Gbps ~ 125 MB/s
        if "wifi" in t:
            return 50.0  # ~400 Mbps ~ 50 MB/s
        return 100.0

    def generate_node_telemetry(
        self,
        self_node_id: NodeId,
        node_network: Mapping[NodeId, NodeNetworkInfo],
    ) -> NodeLinkTelemetry:
        """Produce a complete NodeLinkTelemetry snapshot for all active peer links."""
        throughput_by_iface = self._bandwidth_tracker.sample_throughput()
        links: list[LinkTelemetry] = []
        now = time.time()

        for target_nid, net_info in node_network.items():
            if target_nid == self_node_id:
                continue
            for iface in net_info.interfaces:
                ip = iface.ip_address
                key = (ip, str(target_nid))
                latency = self._latencies.get(key, 0.0)
                capacity = self.estimate_capacity_mbps(iface.interface_type)
                # If we have interface-level throughput, assign it
                used_bw = throughput_by_iface.get(iface.name, 0.0)

                links.append(
                    LinkTelemetry(
                        source=self_node_id,
                        target=target_nid,
                        target_ip=ip,
                        interface_name=iface.name,
                        latency_ms=latency,
                        bandwidth_capacity_mbps=capacity,
                        used_bandwidth_mbps=used_bw,
                        last_updated=now,
                    )
                )

        return NodeLinkTelemetry(links=links)


link_telemetry_registry = LinkTelemetryRegistry()


async def check_reachability(
    target_ip: str,
    expected_node_id: NodeId,
    out: dict[NodeId, set[str]],
    client: httpx.AsyncClient,
    api_port: int,
) -> None:
    """Check if a node is reachable at the given IP, verify its identity, and measure RTT."""
    if ":" in target_ip:
        # TODO: use real IpAddress types
        url = f"http://[{target_ip}]:{api_port}/node_id"
    else:
        url = f"http://{target_ip}:{api_port}/node_id"

    remote_node_id = None
    last_error = None

    for _ in range(REACHABILITY_ATTEMPTS):
        try:
            t0 = time.perf_counter()
            r = await client.get(url)
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            if r.status_code != 200:
                await anyio.sleep(1)
                continue

            body = r.text.strip().strip('"')
            if not body:
                await anyio.sleep(1)
                continue

            remote_node_id = NodeId(body)
            # Record sampled RTT in telemetry registry (TODO #4)
            link_telemetry_registry.record_latency(target_ip, remote_node_id, elapsed_ms)
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
    api_port: int,
) -> AsyncGenerator[tuple[str, NodeId]]:
    """Yield (ip, node_id) pairs as reachability probes complete."""

    send, recv = channel[tuple[str, NodeId]]()

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
        send: Sender[tuple[str, NodeId]],
    ) -> None:
        async with send:
            out: defaultdict[NodeId, set[str]] = defaultdict(set)
            await check_reachability(target_ip, expected_node_id, out, client, api_port)
            if expected_node_id in out:
                await send.send((target_ip, expected_node_id))

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
