"""The stub ranks' collective: a TCP ring between an instance's runners, as MLX's ring backend has.

It couples the ranks the way the real model does: each rank waits for the others at every
step, a stopped rank stops the others, and a lost rank ends the others. The addresses come
from the instance's placement (`hosts_by_node`), the same ones MLX uses.

It copies what MLX 0.32.4's ring backend was seen to do on real hardware:

- each pair of neighbouring ranks shares one connection, which the higher rank makes. A rank
  that only accepts (rank 0, and every rank but the last of a larger ring) waits for its
  connections forever; a rank that connects tries 5 times, waiting 1, 2, 4, 8 and 16 s, then
  fails with "Couldn't connect";
- it listens with SO_REUSEADDR and SO_REUSEPORT, so a second process on the same port binds
  without an error and waits; and it stops listening once the ring is up;
- whatever process accepts at the address is taken as the neighbour: there is no check that
  it is this instance's current runner;
- a rank whose neighbour goes away fails with "[ring] connection to a peer was lost".

One deviation, because every node is on one machine: MLX gives every rank the placement's
port, and here each rank gets a port of its own from the harness (exo.testkit.netsim), the
same for every runner of that rank and unique on the machine, so clusters running side by
side never share one. Without the harness (unit tests) it is the placement's port plus the
rank.
"""

import errno
import socket
import struct
import time
from dataclasses import dataclass
from typing import cast, final

from exo.shared.types.common import NodeId
from exo.shared.types.worker.instances import BoundInstance, MlxRingInstance

_FRAME_HEADER = struct.Struct("!IH")
# MLX's ring backend: the waits after each failed attempt to connect, before it gives up
DIAL_BACKOFF_SECONDS = (1.0, 2.0, 4.0, 8.0, 16.0)
LOST_PEER_MESSAGE = "[ring] connection to a peer was lost"


class RingError(RuntimeError):
    """A link to a neighbouring rank failed: the peer is gone or the link is cut."""

    def __init__(self, error_number: int, detail: str) -> None:
        super().__init__(f"{LOST_PEER_MESSAGE} ({detail})")
        self.error_number = error_number


def rank_to_node(bound_instance: BoundInstance) -> dict[int, NodeId]:
    assignments = bound_instance.instance.shard_assignments
    return {
        assignments.runner_to_shard[runner_id].device_rank: node_id
        for node_id, runner_id in assignments.node_to_runner.items()
    }


def _links(rank: int, size: int) -> tuple[list[int], list[int]]:
    """The neighbours this rank connects to (lower ranks) and accepts from (higher ones)."""
    neighbours = sorted({(rank - 1) % size, (rank + 1) % size} - {rank})
    return (
        [n for n in neighbours if n < rank],
        [n for n in neighbours if n > rank],
    )


@final
@dataclass
class Ring:
    rank: int
    size: int
    _right: socket.socket
    _left: socket.socket

    @classmethod
    def connect(cls, bound_instance: BoundInstance, proxy_port: int | None) -> "Ring":
        instance = bound_instance.instance
        if not isinstance(instance, MlxRingInstance):
            raise NotImplementedError("the stub engine only runs ring instances")
        shard = bound_instance.bound_shard
        rank, size = shard.device_rank, shard.world_size
        hosts = instance.hosts_by_node[bound_instance.bound_node_id]
        nodes = rank_to_node(bound_instance)
        dial_to, accept_from = _links(rank, size)

        links: dict[int, socket.socket] = {}
        if accept_from:
            listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEPORT, 1)
            # The harness's proxy and every peer on this machine dial 127.0.0.1: listening on
            # every interface would open LAN-reachable ports for nothing
            listener.bind(("127.0.0.1", _port(instance, rank, proxy_port)))
            listener.listen(4)
            try:
                while len(links) < len(accept_from):
                    connection = listener.accept()[0]
                    peer = _read_line(connection)
                    if peer.isdigit() and int(peer) in accept_from:
                        links[int(peer)] = connection
                    else:
                        connection.close()
            finally:
                listener.close()
        for peer in dial_to:
            links[peer] = _dial(
                host=hosts[peer].ip,
                port=_port(instance, peer, proxy_port),
                proxy_port=proxy_port,
                source=bound_instance.bound_node_id,
                destination=nodes[peer],
                rank=rank,
            )
        for connection in links.values():
            connection.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        return cls(rank, size, links[(rank + 1) % size], links[(rank - 1) % size])

    def all_gather(self, payload: bytes) -> list[bytes]:
        """Every rank's payload, indexed by rank."""
        gathered: list[bytes | None] = [None] * self.size
        gathered[self.rank] = payload
        current_rank, current = self.rank, payload
        for _ in range(self.size - 1):
            self._send(current_rank, current)
            current_rank, current = self._receive()
            gathered[current_rank] = current
        return [item if item is not None else b"" for item in gathered]

    def barrier(self) -> None:
        self.all_gather(b"")

    def close(self) -> None:
        for each in {self._right, self._left}:
            each.close()

    def _send(self, origin_rank: int, payload: bytes) -> None:
        try:
            self._right.sendall(_FRAME_HEADER.pack(len(payload), origin_rank) + payload)
        except OSError as error:
            raise RingError(error.errno or errno.EPIPE, f"send: {error}") from error

    def _receive(self) -> tuple[int, bytes]:
        header = self._receive_exactly(_FRAME_HEADER.size)
        length, origin_rank = cast(tuple[int, int], _FRAME_HEADER.unpack(header))
        return origin_rank, self._receive_exactly(length)

    def _receive_exactly(self, length: int) -> bytes:
        received = bytearray()
        while len(received) < length:
            try:
                chunk = self._left.recv(length - len(received))
            except OSError as error:
                raise RingError(
                    error.errno or errno.ECONNRESET, f"receive: {error}"
                ) from error
            if not chunk:
                raise RingError(errno.ECONNRESET, "the peer closed the connection")
            received += chunk
        return bytes(received)


def _port(instance: MlxRingInstance, rank: int, proxy_port: int | None) -> int:
    """Where `rank` of the instance listens."""
    if proxy_port is None:
        return instance.ephemeral_port + rank
    with socket.create_connection(("::1", proxy_port)) as harness:
        harness.sendall(f"PORT {instance.instance_id} {rank}\n".encode())
        return int(_read_line(harness))


def _read_line(connection: socket.socket) -> str:
    line = bytearray()
    while not line.endswith(b"\n"):
        chunk = connection.recv(1)
        if not chunk:
            break
        line += chunk
    return line.decode(errors="replace").strip()


def _dial(
    *,
    host: str,
    port: int,
    proxy_port: int | None,
    source: NodeId,
    destination: NodeId,
    rank: int,
) -> socket.socket:
    """Connect to a lower neighbour, on MLX's schedule. Through the harness's proxy, an
    attempt succeeds once the proxy has reached the neighbour's address."""
    for wait in DIAL_BACKOFF_SECONDS:
        connection: socket.socket | None = None
        try:
            if proxy_port is None:
                connection = socket.create_connection((host, port))
            else:
                connection = socket.create_connection(("::1", proxy_port))
                connection.sendall(
                    f"RING {source} {destination} {host} {port}\n".encode()
                )
                if _read_line(connection) != "OK":
                    raise ConnectionRefusedError("nothing listens at the address")
            # Says which rank this is, so a rank with two neighbours tells them apart
            connection.sendall(f"{rank}\n".encode())
            return connection
        except OSError:
            if connection is not None:
                connection.close()
        time.sleep(wait)
    raise RuntimeError(
        f"[ring] Couldn't connect to {host}:{port} "
        f"(tried {len(DIAL_BACKOFF_SECONDS)} times)"
    )
