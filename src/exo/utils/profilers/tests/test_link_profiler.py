"""Tests for the socket link profiler.

We use httpx.MockTransport to simulate the peer's API. Two probes happen
inside `measure()` — a `/node_id` GET and a `/profile/echo` POST, the
latter once for latency (small payload) and once for bandwidth.
"""

import httpx
import pytest

from exo.shared.types.common import NodeId
from exo.utils.profilers.link_profiler import (
    BANDWIDTH_PAYLOAD_BYTES,
    LATENCY_PAYLOAD_BYTES,
    SocketLinkProfile,
)

EXPECTED_NODE_ID = NodeId("alice")
WRONG_NODE_ID = NodeId("mallory")
SINK_IP = "10.0.0.5"
API_PORT = 52415


def _make_mock_transport(*, node_id: str) -> httpx.MockTransport:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/node_id":
            return httpx.Response(200, text=node_id)
        if request.url.path == "/profile/echo":
            body = request.read()
            return httpx.Response(
                200,
                content=body,
                headers={"Content-Type": "application/octet-stream"},
            )
        if request.url.path == "/profile/upload":
            body = request.read()
            return httpx.Response(
                200,
                json={
                    "bytes_received": len(body),
                    "recv_duration_ms": 1.0,
                },
            )
        if request.url.path == "/profile/download":
            from typing import cast as _cast

            n_bytes_str = _cast(str, request.url.params.get("bytes", "0"))
            n_bytes = int(n_bytes_str) if n_bytes_str.isdigit() else 0
            return httpx.Response(
                200,
                content=b"\x00" * n_bytes,
                headers={"Content-Type": "application/octet-stream"},
            )
        return httpx.Response(404)

    return httpx.MockTransport(handler)


async def test_socket_profile_returns_finite_measurements():
    transport = _make_mock_transport(node_id=str(EXPECTED_NODE_ID))
    async with httpx.AsyncClient(transport=transport) as client:
        profile = await SocketLinkProfile.measure(
            client=client,
            sink_ip=SINK_IP,
            expected_sink_node_id=EXPECTED_NODE_ID,
            api_port=API_PORT,
        )
    assert profile is not None
    assert profile.sink_node_id == EXPECTED_NODE_ID
    assert profile.sink_ip == SINK_IP
    assert profile.latency_ms > 0
    assert profile.upload_mbps > 0
    assert profile.download_mbps > 0


async def test_socket_profile_rejects_node_id_mismatch():
    """If the IP is reused by a different node, we must not attribute the bandwidth."""
    transport = _make_mock_transport(node_id=str(WRONG_NODE_ID))
    async with httpx.AsyncClient(transport=transport) as client:
        profile = await SocketLinkProfile.measure(
            client=client,
            sink_ip=SINK_IP,
            expected_sink_node_id=EXPECTED_NODE_ID,
            api_port=API_PORT,
        )
    assert profile is None


async def test_socket_profile_returns_none_on_short_echo():
    """Echo endpoint must round-trip the exact payload — anything else is a bug."""

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/node_id":
            return httpx.Response(200, text=str(EXPECTED_NODE_ID))
        # Truncated echo — peer is misbehaving.
        return httpx.Response(200, content=b"")

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        profile = await SocketLinkProfile.measure(
            client=client,
            sink_ip=SINK_IP,
            expected_sink_node_id=EXPECTED_NODE_ID,
            api_port=API_PORT,
        )
    assert profile is None


async def test_socket_profile_returns_none_on_http_error():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(503)

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        profile = await SocketLinkProfile.measure(
            client=client,
            sink_ip=SINK_IP,
            expected_sink_node_id=EXPECTED_NODE_ID,
            api_port=API_PORT,
        )
    assert profile is None


@pytest.mark.parametrize("ip", ["10.0.0.5", "fe80::1"])
async def test_socket_profile_supports_v4_and_v6(ip: str):
    transport = _make_mock_transport(node_id=str(EXPECTED_NODE_ID))
    async with httpx.AsyncClient(transport=transport) as client:
        profile = await SocketLinkProfile.measure(
            client=client,
            sink_ip=ip,
            expected_sink_node_id=EXPECTED_NODE_ID,
            api_port=API_PORT,
        )
    assert profile is not None
    assert profile.sink_ip == ip


def test_payload_constants_are_consistent():
    # Latency must be tiny (well under MTU); bandwidth must be much larger.
    assert LATENCY_PAYLOAD_BYTES < 1024
    assert BANDWIDTH_PAYLOAD_BYTES > 1024 * 1024
