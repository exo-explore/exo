"""The simulated network on its own: discovery relay, proxied links, and both kinds of cut."""

import contextlib
import fcntl
import socket
import threading
import time
from collections.abc import Iterator
from dataclasses import dataclass
from typing import cast

import pytest

from exo.testkit.netsim import NetSim
from exo.testkit.ports import LEASES

_HELLO = b"EXO\x00" + b"N" * 8 + b"S" * 8


def _whats_up(nonce: bytes, port: int) -> bytes:
    return b"EXO\x01" + nonce + b"Z" * 16 + port.to_bytes(2, "little")


@dataclass
class _FakeNode:
    name: str
    discovery: socket.socket
    zenoh: socket.socket

    @property
    def discovery_port(self) -> int:
        return int(self.discovery.getsockname()[1])  # pyright: ignore[reportAny]

    @property
    def zenoh_port(self) -> int:
        return int(self.zenoh.getsockname()[1])  # pyright: ignore[reportAny]


def _echo(server: socket.socket) -> None:
    while True:
        try:
            connection = server.accept()[0]
        except OSError:
            return

        def serve(connection: socket.socket = connection) -> None:
            with connection:
                while data := connection.recv(4096):
                    connection.sendall(data)

        threading.Thread(target=serve, daemon=True).start()


def _node(name: str) -> _FakeNode:
    discovery = socket.socket(socket.AF_INET6, socket.SOCK_DGRAM)
    discovery.bind(("::1", 0))
    discovery.settimeout(2)
    zenoh = socket.socket(socket.AF_INET6, socket.SOCK_STREAM)
    zenoh.bind(("::1", 0))
    zenoh.listen()
    threading.Thread(target=_echo, args=(zenoh,), daemon=True).start()
    return _FakeNode(name, discovery, zenoh)


@pytest.fixture
def network() -> Iterator[tuple[NetSim, _FakeNode, _FakeNode]]:
    if not socket.has_ipv6:
        pytest.skip("no IPv6")
    netsim = NetSim()
    netsim.start()
    a, b = _node("a"), _node("b")
    for node in (a, b):
        netsim.add_node(node.name, node.discovery_port, node.zenoh_port)
    yield netsim, a, b
    netsim.stop()
    for node in (a, b):
        node.discovery.close()
        node.zenoh.close()


def _discover(netsim: NetSim, a: _FakeNode, b: _FakeNode) -> int:
    """a says Hello; b answers; returns the zenoh port a is told to dial for b."""
    a.discovery.sendto(_HELLO, ("::1", netsim.relay_port))
    hello, relayed_from = cast(
        tuple[bytes, tuple[str, int, int, int]], b.discovery.recvfrom(64)
    )
    assert hello == _HELLO
    b.discovery.sendto(_whats_up(hello[4:12], b.zenoh_port), relayed_from)
    reply = a.discovery.recv(64)
    assert reply[:28] == _whats_up(hello[4:12], b.zenoh_port)[:28]
    port = int.from_bytes(reply[28:30], "little")
    assert port != b.zenoh_port, "the reply should point at the proxy"
    return port


def _connect(port: int) -> socket.socket:
    connection = socket.create_connection(("::1", port), timeout=2)
    connection.sendall(b"ping")
    assert connection.recv(4) == b"ping"
    return connection


def test_discovery_goes_through_the_relay_and_links_through_a_proxy(
    network: tuple[NetSim, _FakeNode, _FakeNode],
) -> None:
    netsim, a, b = network
    connection = _connect(_discover(netsim, a, b))
    connection.close()
    assert netsim.bytes_between("a", "b") >= 8


def test_the_report_counts_what_went_through_and_what_was_dropped(
    network: tuple[NetSim, _FakeNode, _FakeNode],
) -> None:
    netsim, a, b = network
    _connect(_discover(netsim, a, b)).close()
    stranger = socket.socket(socket.AF_INET6, socket.SOCK_DGRAM)
    with stranger:
        stranger.bind(("::1", 0))
        stranger.sendto(_HELLO, ("::1", netsim.relay_port))
        stranger_port = int(stranger.getsockname()[1])  # pyright: ignore[reportAny]
        dropped = f"relay: datagram from unknown port {stranger_port}: 1"
        deadline = time.monotonic() + 2
        while (
            dropped not in (report := netsim.report()) and time.monotonic() < deadline
        ):
            time.sleep(0.01)
    assert dropped in report, report
    assert "Hellos from a: 1" in report
    assert "WhatsUps b->a: 1" in report
    assert "zenoh connections a->b: 1" in report
    assert any(line.startswith("bytes a-b: ") for line in report), report


def test_a_blackhole_holds_bytes_until_the_heal(
    network: tuple[NetSim, _FakeNode, _FakeNode],
) -> None:
    netsim, a, b = network
    connection = _connect(_discover(netsim, a, b))
    netsim.cut("a", "b", "blackhole")
    connection.sendall(b"held")
    connection.settimeout(0.5)
    with pytest.raises(TimeoutError):
        connection.recv(4)
    # Discovery doesn't cross the cut either
    a.discovery.sendto(_HELLO.replace(b"N", b"M"), ("::1", netsim.relay_port))
    b.discovery.settimeout(0.5)
    with pytest.raises(TimeoutError):
        b.discovery.recvfrom(64)
    netsim.heal("a", "b")
    connection.settimeout(2)
    assert connection.recv(4) == b"held"
    connection.close()


def test_a_reset_tears_connections_down_and_refuses_new_ones(
    network: tuple[NetSim, _FakeNode, _FakeNode],
) -> None:
    netsim, a, b = network
    port = _discover(netsim, a, b)
    connection = _connect(port)
    netsim.cut("a", "b", "reset")
    connection.settimeout(2)
    with contextlib.suppress(ConnectionResetError):
        assert connection.recv(4) == b""
    refused = socket.create_connection(("::1", port), timeout=2)
    refused.settimeout(2)
    with contextlib.suppress(ConnectionResetError):
        assert refused.recv(4) == b""
    netsim.heal("a", "b")
    _connect(port).close()
    connection.close()
    refused.close()


def _ring_port(netsim: NetSim, instance: str, rank: int) -> int:
    with socket.create_connection(
        ("::1", netsim.ring_proxy_port), timeout=2
    ) as harness:
        harness.sendall(f"PORT {instance} {rank}\n".encode())
        return int(harness.recv(16).decode().strip())


def test_stub_ring_ports_are_one_per_rank_and_never_shared_between_clusters() -> None:
    first, second = NetSim(), NetSim()
    first.start()
    second.start()
    try:
        port = _ring_port(first, "instance", 0)
        assert _ring_port(first, "instance", 0) == port
        assert _ring_port(first, "instance", 1) != port
        assert _ring_port(second, "instance", 0) != port

        # The rank's runners can listen on it, with MLX's SO_REUSEPORT...
        listener = socket.socket()
        listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEPORT, 1)
        listener.bind(("0.0.0.0", port))
        listener.listen()
        with socket.create_connection(("127.0.0.1", port), timeout=2):
            accepted = listener.accept()[0]
            accepted.close()
        listener.close()
        # ...and every other harness on the machine skips it while it is leased
        with (LEASES / f"{port}.lock").open("a") as other, pytest.raises(OSError):
            fcntl.flock(other, fcntl.LOCK_EX | fcntl.LOCK_NB)
    finally:
        first.stop()
        second.stop()


def test_a_nodes_api_is_reached_through_the_pairs_proxy_and_cut_with_it() -> None:
    """One node's HTTP to another's API (serving a request without the master) goes through
    the pair's proxy, which counts it, and a cut between the two applies to it."""
    if not socket.has_ipv6:
        pytest.skip("no IPv6")
    api = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    api.bind(("127.0.0.1", 0))
    api.listen()
    threading.Thread(target=_echo, args=(api,), daemon=True).start()
    netsim = NetSim()
    netsim.start()
    a, b = _node("a"), _node("b")
    netsim.add_node(a.name, a.discovery_port, a.zenoh_port)
    netsim.add_node(
        b.name,
        b.discovery_port,
        b.zenoh_port,
        api_port=int(api.getsockname()[1]),  # pyright: ignore[reportAny]
    )
    try:
        assert ("b", "a") not in netsim.api_proxy_ports, "a has no API"
        connection = _connect(netsim.api_proxy_ports[("a", "b")])
        assert netsim.api_connections[("a", "b")] == 1
        netsim.cut("a", "b", "reset")
        connection.settimeout(2)
        with contextlib.suppress(ConnectionResetError):
            assert connection.recv(4) == b""
        connection.close()
    finally:
        netsim.stop()
        for each in (a.discovery, a.zenoh, b.discovery, b.zenoh, api):
            each.close()
