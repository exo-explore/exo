"""Link profilers — measure latency and bandwidth between this node and a peer.

Two transports, dispatched per topology edge:

- `SocketLinkProfile` — HTTP over the API port; measures the kernel TCP path.
- `RDMALinkProfile`   — subprocess‑isolated MLX `jaccl`; measures the RDMA
                       path actually used for inference traffic.

Both profiles carry directional bandwidth (`upload_mbps` = source→sink,
`download_mbps` = sink→source) so the dashboard can surface real-world
asymmetries (Wi-Fi up vs down, NIC tx/rx, RDMA controller path skew).
Latency is reported as round-trip (`latency_ms`) — true one-way latency
needs sub-microsecond clock sync that we don't have, so we don't fake it.

Both return `None` on any failure (timeout, peer mismatch, subprocess
crash). Missing values are a legitimate "next tick will retry" state.
"""

import statistics
import time
from typing import Self, final

import httpx
from pydantic import BaseModel, ConfigDict, ValidationError

from exo.shared.types.common import NodeId
from exo.utils.profilers.rdma_probe import (
    RdmaProbeError,
    RdmaProbeParams,
    run_rdma_probe_source_side,
)
from exo.utils.pydantic_ext import TaggedModel


class _UploadResponse(BaseModel):
    """Parsed body of a POST /profile/upload response."""

    model_config = ConfigDict(extra="ignore", strict=False)
    bytes_received: int
    recv_duration_ms: float


LATENCY_PAYLOAD_BYTES = 64
LATENCY_SAMPLES = 5
BANDWIDTH_PAYLOAD_BYTES = 8 * 1024 * 1024
PROBE_TIMEOUT_SECONDS = 30.0


@final
class SocketLinkProfile(TaggedModel):
    """Measured TCP/IP RTT plus per-direction bandwidth to a peer's API port."""

    sink_node_id: NodeId
    sink_ip: str
    latency_ms: float
    upload_mbps: float
    download_mbps: float

    @classmethod
    async def measure(
        cls,
        *,
        client: httpx.AsyncClient,
        sink_ip: str,
        expected_sink_node_id: NodeId,
        api_port: int,
    ) -> Self | None:
        if not await _peer_node_id_matches(
            client, sink_ip, api_port, expected_sink_node_id
        ):
            return None

        latency_ms = await _measure_latency_ms(client, sink_ip, api_port)
        if latency_ms is None:
            return None

        upload_mbps = await _measure_upload_mbps(client, sink_ip, api_port)
        if upload_mbps is None:
            return None

        download_mbps = await _measure_download_mbps(client, sink_ip, api_port)
        if download_mbps is None:
            return None

        return cls(
            sink_node_id=expected_sink_node_id,
            sink_ip=sink_ip,
            latency_ms=latency_ms,
            upload_mbps=upload_mbps,
            download_mbps=download_mbps,
        )


@final
class RDMALinkProfile(TaggedModel):
    """Measured RDMA bandwidth (per direction) + RTT over a Thunderbolt edge.

    All three numeric fields are None when the most recent probe was skipped
    (the local node, the peer, or both had active runners) or failed.
    """

    sink_node_id: NodeId
    source_rdma_iface: str
    sink_rdma_iface: str
    upload_mbps: float | None
    download_mbps: float | None
    payload_bytes: int | None
    latency_ms: float | None

    @classmethod
    async def measure(
        cls,
        *,
        client: httpx.AsyncClient,
        sink_ip: str,
        sink_node_id: NodeId,
        api_port: int,
        source_rdma_iface: str,
        sink_rdma_iface: str,
        coordinator_ip: str,
    ) -> Self | None:
        params = RdmaProbeParams(
            source_rdma_iface=source_rdma_iface,
            sink_rdma_iface=sink_rdma_iface,
            coordinator_ip=coordinator_ip,
        )
        try:
            result = await run_rdma_probe_source_side(
                client=client,
                params=params,
                sink_ip=sink_ip,
                api_port=api_port,
            )
        except RdmaProbeError:
            return None
        if result is None:
            return cls(
                sink_node_id=sink_node_id,
                source_rdma_iface=source_rdma_iface,
                sink_rdma_iface=sink_rdma_iface,
                upload_mbps=None,
                download_mbps=None,
                payload_bytes=None,
                latency_ms=None,
            )
        return cls(
            sink_node_id=sink_node_id,
            source_rdma_iface=source_rdma_iface,
            sink_rdma_iface=sink_rdma_iface,
            upload_mbps=result.upload_mbps,
            download_mbps=result.download_mbps,
            payload_bytes=result.payload_bytes,
            latency_ms=result.latency_ms,
        )


LinkProfile = SocketLinkProfile | RDMALinkProfile


def _bracketed(ip: str) -> str:
    return f"[{ip}]" if ":" in ip else ip


def _echo_url(sink_ip: str, api_port: int) -> str:
    return f"http://{_bracketed(sink_ip)}:{api_port}/profile/echo"


def _upload_url(sink_ip: str, api_port: int) -> str:
    return f"http://{_bracketed(sink_ip)}:{api_port}/profile/upload"


def _download_url(sink_ip: str, api_port: int, n_bytes: int) -> str:
    return f"http://{_bracketed(sink_ip)}:{api_port}/profile/download?bytes={n_bytes}"


def _node_id_url(sink_ip: str, api_port: int) -> str:
    return f"http://{_bracketed(sink_ip)}:{api_port}/node_id"


async def _peer_node_id_matches(
    client: httpx.AsyncClient,
    sink_ip: str,
    api_port: int,
    expected_sink_node_id: NodeId,
) -> bool:
    """Confirm the peer reachable at `sink_ip` is the one we expect.

    IP addresses outlive node memberships (e.g. DHCP rebind, node restart with
    a fresh node_id), so we re‑verify before attributing bandwidth to a peer.
    """
    try:
        response = await client.get(
            _node_id_url(sink_ip, api_port), timeout=PROBE_TIMEOUT_SECONDS
        )
    except httpx.HTTPError:
        return False
    if response.status_code != 200:
        return False
    return response.text.strip().strip('"') == expected_sink_node_id


async def _measure_latency_ms(
    client: httpx.AsyncClient, sink_ip: str, api_port: int
) -> float | None:
    """Round-trip latency, median over K small-payload echoes."""
    samples_ms: list[float] = []
    payload = b"\x00" * LATENCY_PAYLOAD_BYTES
    url = _echo_url(sink_ip, api_port)
    for _ in range(LATENCY_SAMPLES):
        start = time.perf_counter()
        try:
            response = await client.post(
                url, content=payload, timeout=PROBE_TIMEOUT_SECONDS
            )
        except httpx.HTTPError:
            return None
        elapsed_ms = (time.perf_counter() - start) * 1000
        if (
            response.status_code != 200
            or len(response.content) != LATENCY_PAYLOAD_BYTES
        ):
            return None
        samples_ms.append(elapsed_ms)
    return statistics.median(samples_ms)


async def _measure_upload_mbps(
    client: httpx.AsyncClient, sink_ip: str, api_port: int
) -> float | None:
    """One-way upload bandwidth (source -> sink). Server times its own
    receive duration so the small response's RTT doesn't pollute the result.
    """
    payload = b"\x00" * BANDWIDTH_PAYLOAD_BYTES
    try:
        response = await client.post(
            _upload_url(sink_ip, api_port),
            content=payload,
            timeout=PROBE_TIMEOUT_SECONDS,
        )
    except httpx.HTTPError:
        return None
    if response.status_code != 200:
        return None
    try:
        body = _UploadResponse.model_validate_json(response.content)
    except ValidationError:
        return None
    if body.bytes_received != BANDWIDTH_PAYLOAD_BYTES or body.recv_duration_ms <= 0:
        return None
    return BANDWIDTH_PAYLOAD_BYTES * 8 / (body.recv_duration_ms / 1000.0) / 1e6


async def _measure_download_mbps(
    client: httpx.AsyncClient, sink_ip: str, api_port: int
) -> float | None:
    """One-way download bandwidth (sink -> source). The request is tiny so
    the client's wall-clock receive time is dominated by the response transit.
    """
    url = _download_url(sink_ip, api_port, BANDWIDTH_PAYLOAD_BYTES)
    start = time.perf_counter()
    try:
        response = await client.get(url, timeout=PROBE_TIMEOUT_SECONDS)
    except httpx.HTTPError:
        return None
    elapsed = time.perf_counter() - start
    if (
        response.status_code != 200
        or len(response.content) != BANDWIDTH_PAYLOAD_BYTES
        or elapsed <= 0
    ):
        return None
    return BANDWIDTH_PAYLOAD_BYTES * 8 / elapsed / 1e6


__all__ = [
    "BANDWIDTH_PAYLOAD_BYTES",
    "LATENCY_PAYLOAD_BYTES",
    "LATENCY_SAMPLES",
    "LinkProfile",
    "RDMALinkProfile",
    "SocketLinkProfile",
]
