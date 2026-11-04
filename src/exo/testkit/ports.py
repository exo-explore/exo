"""Ports for the harness's nodes and stub runners, leased so that nothing else takes them.

A port found by binding port 0 and closing the socket is free only at that moment: until the
node binds it, any socket on the machine can be handed the same port. For a node's discovery
port that failure is silent on macOS. exo binds it as [::] with SO_REUSEADDR and
SO_REUSEPORT, which succeeds next to a socket already on [::1] at that port, and the OS then
delivers every unicast datagram to [::1] to the other socket: the node never hears from its
peers, and nothing logs an error.

So every port comes from below the OS's ephemeral range (from 49152 on macOS, 32768 on
Linux), which it never hands out by itself, and is leased: an flock on a lock file per port
in the temp directory, which every harness sharing it respects.
"""

import contextlib
import fcntl
import random
import socket
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import TextIO, final

PORTS = range(20000, 32768)
LEASES = Path(tempfile.gettempdir()) / "exo-testkit-ports"
# Every way a node or a runner binds a port: an API (TCP, IPv4), zenoh (TCP, IPv6),
# discovery (UDP, IPv6) and a stub ring listener (TCP, IPv4)
_BINDS = (
    (socket.AF_INET, socket.SOCK_STREAM, "0.0.0.0"),
    (socket.AF_INET6, socket.SOCK_STREAM, "::"),
    (socket.AF_INET6, socket.SOCK_DGRAM, "::"),
)


@final
@dataclass(eq=False)
class PortLease:
    """A port this process may use until it releases it or exits."""

    port: int
    lock: TextIO

    def release(self) -> None:
        self.lock.close()


def lease_port() -> PortLease:
    """A port no other harness holds and nothing on the machine is bound to, of any kind."""
    LEASES.mkdir(exist_ok=True)
    start = random.randrange(len(PORTS))
    for offset in range(len(PORTS)):
        port = PORTS[(start + offset) % len(PORTS)]
        lock = (LEASES / f"{port}.lock").open("a")
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            lock.close()  # another harness holds it
            continue
        if not _unbound(port):
            lock.close()  # something else is using it
            continue
        return PortLease(port, lock)
    raise RuntimeError("no free port for the harness")


def _unbound(port: int) -> bool:
    for family, kind, host in _BINDS:
        with socket.socket(family, kind) as probe:
            if family == socket.AF_INET6:
                with contextlib.suppress(OSError):
                    probe.setsockopt(socket.IPPROTO_IPV6, socket.IPV6_V6ONLY, 1)
            try:
                probe.bind((host, port))
            except OSError:
                return False
    return True
