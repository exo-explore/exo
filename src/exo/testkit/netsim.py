"""The simulated network between the harness's nodes, so a test can cut any pair without root.

All traffic between nodes passes through here, on the IPv6 loopback:

- discovery: every node sends its Hellos to the relay instead of the multicast group (the
  EXO_TESTKIT_DISCOVERY_ANNOUNCE_TO hook in rust/networking/src/discovery.rs). The relay hands each one
  to every other node, and hands each reply (WhatsUp) back with its zenoh port replaced by
  the port of a proxy for that pair. So when exo's discovery dials a peer, it dials the proxy;
- zenoh: one TCP proxy per ordered pair of nodes forwards to the peer's zenoh port;
- stub ring links: the stub runners connect through one more proxy that is told the pair.

A cut affects one pair, in one of two ways:

- blackhole: the proxies stop moving bytes in both directions and the relay drops discovery
  between the two. Connections stay open but nothing gets through, as when a cable is pulled
  or Wi-Fi drops: each side finds out only when its own timeouts expire. Bytes in flight wait
  for the heal (TCP would have retransmitted them), so nothing is corrupted;
- reset: every connection between the two is torn down at once (RST) and new ones are refused.

The reachability probes between nodes' APIs don't come through here; the nodes read the same
cuts from the network table (exo.testkit.network_table).

It runs on its own event loop in a background thread, so a test that blocks doesn't stall the
cluster's network.
"""

import asyncio
import contextlib
import socket
import struct
import threading
from collections import Counter, deque
from collections.abc import Coroutine
from dataclasses import dataclass, field
from functools import partial
from typing import cast, final

from exo.testkit.network_table import CutMode
from exo.testkit.ports import PortLease, lease_port

# rust/networking/src/discovery.rs: where a node sends its Hellos instead of the multicast group
ANNOUNCE_TO_ENV = "EXO_TESTKIT_DISCOVERY_ANNOUNCE_TO"

# rust/networking/src/discovery.rs: a 4-byte header (b"EXO", kind), then the message
_MAGIC = b"EXO"
_HELLO_KIND = 0
_WHATS_UP_KIND = 1
_HELLO_SIZE = 4 + 8 + 8
_WHATS_UP_SIZE = 4 + 8 + 16 + 2
_WHATS_UP_PORT = slice(28, 30)
_READ_SIZE = 256 * 1024
# How long the ring proxy waits for a new connection's header line (the stub sends it at once)
_RING_HEADER_TIMEOUT = 5.0
# How long a call from the test's thread waits for the network's loop (a start or a cut
# takes milliseconds)
_CALL_TIMEOUT = 10.0
# How often a connection held by a blackhole checks whether it has healed or been reset
_CUT_POLL_SECONDS = 0.05


@final
@dataclass(eq=False)
class _Link:
    """One proxied TCP connection, both of its sockets."""

    writers: tuple[asyncio.StreamWriter, asyncio.StreamWriter]
    closed: bool = False

    def abort(self, reset: bool) -> None:
        self.closed = True
        for writer in self.writers:
            if reset:
                raw = cast(object, writer.get_extra_info("socket"))
                if isinstance(raw, socket.socket):
                    with contextlib.suppress(OSError):
                        raw.setsockopt(
                            socket.SOL_SOCKET,
                            socket.SO_LINGER,
                            struct.pack("ii", 1, 0),
                        )
            writer.transport.abort()


@final
@dataclass(eq=False)
class _Pair:
    """The link between two nodes, both directions."""

    mode: CutMode | None = None
    flowing: asyncio.Event = field(default_factory=asyncio.Event)
    links: set[_Link] = field(default_factory=set)
    bytes_forwarded: int = 0

    def __post_init__(self) -> None:
        self.flowing.set()


@final
@dataclass(eq=False)
class _Node:
    name: str
    discovery_port: int
    zenoh_port: int
    node_id: str | None = None


class _RelayProtocol(asyncio.DatagramProtocol):
    def __init__(self, netsim: "NetSim", origin: str | None, peer: str | None) -> None:
        self.netsim = netsim
        # For a pair socket: the node whose Hello it forwards, and the node it forwards to
        self.origin = origin
        self.peer = peer
        self.transport: asyncio.DatagramTransport | None = None

    def connection_made(self, transport: asyncio.BaseTransport) -> None:
        if not isinstance(transport, asyncio.DatagramTransport):
            raise TypeError(f"expected a datagram transport, got {transport!r}")
        self.transport = transport

    def datagram_received(self, data: bytes, addr: tuple[str | int, ...]) -> None:
        source_port = addr[1]
        if not isinstance(source_port, int):
            return
        if self.origin is None:
            self.netsim.relay_hello(data, source_port)
        elif self.peer is not None:
            self.netsim.relay_whats_up(data, self.origin, self.peer, source_port)


@final
class NetSim:
    def __init__(self) -> None:
        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(
            target=self._loop.run_forever, name="testkit-netsim", daemon=True
        )
        self._nodes: dict[str, _Node] = {}
        self._pairs: dict[frozenset[str], _Pair] = {}
        self._proxy_ports: dict[tuple[str, str], int] = {}
        # (instance, rank) -> the lease on that rank's stub ring port
        self._ring_ports: dict[tuple[str, int], PortLease] = {}
        self._servers: list[asyncio.Server] = []
        self._pair_sockets: dict[tuple[str, str], asyncio.DatagramTransport] = {}
        self._relay: asyncio.DatagramTransport | None = None
        self._seen_hellos: deque[tuple[str, bytes]] = deque(maxlen=512)
        # What went through, and what didn't, for report(): a cluster that doesn't form
        # needs to say whether discovery, a proxy or the node itself is at fault
        self._counts: Counter[str] = Counter()
        self._connect_errors: dict[tuple[str, str], str] = {}
        self.relay_port = 0
        self.ring_proxy_port = 0

    # --- called from the harness's thread -------------------------------------------------

    def start(self) -> None:
        self._thread.start()
        self._call(self._start())

    def stop(self) -> None:
        if not self._thread.is_alive():
            return
        self._call(self._stop())
        self._loop.call_soon_threadsafe(self._loop.stop)
        self._thread.join(timeout=5)
        if not self._thread.is_alive():
            self._loop.close()

    def add_node(self, name: str, discovery_port: int, zenoh_port: int) -> None:
        self._call(self._add_node(name, discovery_port, zenoh_port))

    def set_node_id(self, name: str, node_id: str) -> None:
        def update() -> None:
            self._nodes[name].node_id = node_id

        self._loop.call_soon_threadsafe(update)

    def cut(self, a: str, b: str, mode: CutMode) -> None:
        self._call(self._cut(a, b, mode))

    def heal(self, a: str, b: str) -> None:
        self._call(self._heal(a, b))

    def bytes_between(self, a: str, b: str) -> int:
        return self._pairs[frozenset((a, b))].bytes_forwarded

    def report(self) -> list[str]:
        """What the network has passed so far: Hellos per origin, WhatsUps per pair, zenoh
        connections and bytes per pair, and what it dropped and why."""
        return self._call(self._report())

    def _call[T](self, coroutine: Coroutine[object, object, T]) -> T:
        return asyncio.run_coroutine_threadsafe(coroutine, self._loop).result(
            _CALL_TIMEOUT
        )

    # --- on the network's own loop ---------------------------------------------------------

    async def _start(self) -> None:
        relay, _ = await self._loop.create_datagram_endpoint(
            lambda: _RelayProtocol(self, None, None), local_addr=("::1", 0)
        )
        self._relay = relay
        self.relay_port = _port_of(cast(object, relay.get_extra_info("sockname")))
        ring = await asyncio.start_server(self._ring_connection, host="::1", port=0)
        self._servers.append(ring)
        self.ring_proxy_port = _server_port(ring)

    async def _stop(self) -> None:
        for pair in self._pairs.values():
            for link in list(pair.links):
                link.abort(reset=False)
        for server in self._servers:
            server.close()
        for transport in self._pair_sockets.values():
            transport.close()
        for lease in self._ring_ports.values():
            lease.release()
        if self._relay is not None:
            self._relay.close()
        others = [
            task for task in asyncio.all_tasks() if task is not asyncio.current_task()
        ]
        for task in others:
            task.cancel()
        await asyncio.gather(*others, return_exceptions=True)

    async def _add_node(self, name: str, discovery_port: int, zenoh_port: int) -> None:
        if name in self._nodes:
            raise ValueError(f"node {name} is already on the network")
        node = _Node(name, discovery_port, zenoh_port)
        for other in self._nodes.values():
            self._pairs[frozenset((name, other.name))] = _Pair()
            for dialler, target in ((node, other), (other, node)):
                server = await asyncio.start_server(
                    self._zenoh_connection_handler(dialler.name, target.name),
                    host="::1",
                    port=0,
                )
                self._servers.append(server)
                self._proxy_ports[(dialler.name, target.name)] = _server_port(server)
                transport, _ = await self._loop.create_datagram_endpoint(
                    partial(_RelayProtocol, self, dialler.name, target.name),
                    local_addr=("::1", 0),
                )
                self._pair_sockets[(dialler.name, target.name)] = transport
        self._nodes[name] = node

    async def _report(self) -> list[str]:
        lines = [f"{key}: {count}" for key, count in sorted(self._counts.items())]
        for (dialler, target), error in sorted(self._connect_errors.items()):
            lines.append(f"zenoh {dialler}->{target} last connect error: {error}")
        for key, pair in sorted(self._pairs.items(), key=lambda item: sorted(item[0])):
            a, b = sorted(key)
            lines.append(
                f"bytes {a}-{b}: {pair.bytes_forwarded}, {len(pair.links)} open, "
                f"cut {pair.mode}"
            )
        return lines

    async def _cut(self, a: str, b: str, mode: CutMode) -> None:
        pair = self._pairs[frozenset((a, b))]
        pair.mode = mode
        pair.flowing.clear()
        if mode == "reset":
            for link in list(pair.links):
                link.abort(reset=True)
            pair.links.clear()

    async def _heal(self, a: str, b: str) -> None:
        pair = self._pairs[frozenset((a, b))]
        pair.mode = None
        pair.flowing.set()

    def _node_by_discovery_port(self, port: int) -> _Node | None:
        return next(
            (node for node in self._nodes.values() if node.discovery_port == port), None
        )

    def relay_hello(self, data: bytes, source_port: int) -> None:
        origin = self._node_by_discovery_port(source_port)
        if origin is None:
            self._counts[f"relay: datagram from unknown port {source_port}"] += 1
            return
        if not _is(data, _HELLO_KIND, _HELLO_SIZE):
            self._counts[f"relay: not a Hello from {origin.name}"] += 1
            return
        # Pass each Hello on once, however many copies arrive
        seen = (origin.name, data[4:12])
        if seen in self._seen_hellos:
            return
        self._seen_hellos.append(seen)
        self._counts[f"Hellos from {origin.name}"] += 1
        for peer in self._nodes.values():
            if peer is origin or self._pairs[frozenset((origin.name, peer.name))].mode:
                continue
            self._pair_sockets[(origin.name, peer.name)].sendto(
                data, ("::1", peer.discovery_port)
            )

    def relay_whats_up(
        self, data: bytes, origin_name: str, peer_name: str, source_port: int
    ) -> None:
        origin, peer = self._nodes.get(origin_name), self._nodes.get(peer_name)
        if origin is None or peer is None:
            return
        if source_port != peer.discovery_port:
            self._counts[
                f"WhatsUp {peer_name}->{origin_name}: dropped, from port {source_port}"
            ] += 1
            return
        if not _is(data, _WHATS_UP_KIND, _WHATS_UP_SIZE):
            self._counts[f"WhatsUp {peer_name}->{origin_name}: dropped, malformed"] += 1
            return
        if self._pairs[frozenset((origin_name, peer_name))].mode:
            return
        self._counts[f"WhatsUps {peer_name}->{origin_name}"] += 1
        proxy_port = self._proxy_ports[(origin_name, peer_name)]
        reply = bytearray(data)
        reply[_WHATS_UP_PORT] = proxy_port.to_bytes(2, "little")
        self._pair_sockets[(origin_name, peer_name)].sendto(
            bytes(reply), ("::1", origin.discovery_port)
        )

    def ring_port(self, instance_id: str, rank: int) -> int:
        """The port the stub runner of `rank` of an instance listens on: the same for every
        runner of that rank (as MLX's ring gives every rank the placement's port), and
        leased (exo.testkit.ports), so clusters running side by side never share one."""
        key = (instance_id, rank)
        lease = self._ring_ports.get(key)
        if lease is None:
            lease = lease_port()
            self._ring_ports[key] = lease
        return lease.port

    def _zenoh_connection_handler(self, dialler: str, target: str):
        async def handle(
            reader: asyncio.StreamReader, writer: asyncio.StreamWriter
        ) -> None:
            pair = self._pairs[frozenset((dialler, target))]
            self._counts[f"zenoh connections {dialler}->{target}"] += 1
            error = await self._proxy(
                pair, reader, writer, "::1", self._nodes[target].zenoh_port
            )
            if error is not None:
                self._connect_errors[(dialler, target)] = error

        return handle

    async def _ring_connection(
        self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter
    ) -> None:
        try:
            header = await asyncio.wait_for(reader.readline(), _RING_HEADER_TIMEOUT)
            words = header.decode().split()
            if words and words[0] == "PORT":
                _, instance_id, rank = words
                writer.write(f"{self.ring_port(instance_id, int(rank))}\n".encode())
                await writer.drain()
                writer.close()
                return
            _, source_id, destination_id, host, port = words
        except (TimeoutError, ValueError, UnicodeDecodeError, OSError):
            writer.transport.abort()
            return
        names = {node.node_id: node.name for node in self._nodes.values()}
        source, destination = names.get(source_id), names.get(destination_id)
        if source is None or destination is None or source == destination:
            # Ranks of one node, or nodes the network doesn't know: never cut
            pair = _Pair()
        else:
            pair = self._pairs[frozenset((source, destination))]
        # Every node is this machine: a peer's own loopback address (127.0.0.k, see
        # exo.testkit.node) is reached at 127.0.0.1, which every system has
        if host.startswith("127."):
            host = "127.0.0.1"
        await self._proxy(pair, reader, writer, host, int(port), greet=b"OK\n")

    async def _proxy(
        self,
        pair: _Pair,
        reader: asyncio.StreamReader,
        writer: asyncio.StreamWriter,
        host: str,
        port: int,
        greet: bytes | None = None,
    ) -> str | None:
        """Forward one connection until either side closes it. Returns why the upstream
        connection couldn't be made, if it couldn't."""
        if pair.mode == "reset":
            _Link((writer, writer)).abort(reset=True)
            return None
        # Through a blackholed pair a connection gets nowhere until the heal
        await pair.flowing.wait()
        try:
            upstream_reader, upstream_writer = await asyncio.open_connection(host, port)
        except OSError as error:
            writer.transport.abort()
            return f"{host}:{port}: {error!r}"
        for each in (writer, upstream_writer):
            _no_delay(each)
        if greet is not None:
            writer.write(greet)
        link = _Link((writer, upstream_writer))
        pair.links.add(link)
        try:
            await asyncio.gather(
                self._pump(pair, link, reader, upstream_writer),
                self._pump(pair, link, upstream_reader, writer),
            )
        finally:
            pair.links.discard(link)
            link.abort(reset=False)
        return None

    async def _pump(
        self,
        pair: _Pair,
        link: _Link,
        reader: asyncio.StreamReader,
        writer: asyncio.StreamWriter,
    ) -> None:
        try:
            while await _until_flowing(pair, link):
                data = await reader.read(_READ_SIZE)
                # A close, like any other byte, doesn't cross a blackhole until it heals
                if not await _until_flowing(pair, link) or not data:
                    break
                writer.write(data)
                pair.bytes_forwarded += len(data)
                await writer.drain()
        except (ConnectionError, OSError):
            pass
        finally:
            writer.transport.abort()


async def _until_flowing(pair: _Pair, link: _Link) -> bool:
    """Wait while the pair is cut; False once the link has been torn down."""
    while not pair.flowing.is_set() and not link.closed:
        await asyncio.sleep(_CUT_POLL_SECONDS)
    return not link.closed


def _no_delay(writer: asyncio.StreamWriter) -> None:
    """Forward small writes at once, as the endpoints themselves do: Nagle's algorithm on the
    proxy would add delays the real network doesn't have."""
    raw = cast(object, writer.get_extra_info("socket"))
    if isinstance(raw, socket.socket):
        with contextlib.suppress(OSError):
            raw.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)


def _is(data: bytes, kind: int, size: int) -> bool:
    return len(data) == size and data[:3] == _MAGIC and data[3] == kind


def _port_of(sockname: object) -> int:
    port = sockname[1] if isinstance(sockname, tuple) else None  # pyright: ignore[reportUnknownVariableType]
    if not isinstance(port, int):
        raise TypeError(f"unexpected socket name {sockname!r}")
    return port


def _server_port(server: asyncio.Server) -> int:
    return _port_of(cast(object, server.sockets[0].getsockname()))
