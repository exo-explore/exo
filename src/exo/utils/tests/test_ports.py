"""Tests for :mod:`exo.utils.ports`.

Coverage focuses on the kernel-assigned-free-port behaviour introduced
in PR #20 round-(N+12) to address Codex P1 (placement.py:711). The
legacy uniformly-random implementation had no test coverage at all
because there was no behaviour to assert beyond "returns an int in
range"; the rewrite has actual semantics worth pinning.
"""

import socket
from contextlib import closing

import pytest

from exo.utils.ports import DEFAULT_API_PORT, random_ephemeral_port


@pytest.mark.parametrize("invocations", [1, 16, 64])
def test_returns_port_in_ephemeral_range(invocations: int) -> None:
    """Every returned port lives in the ephemeral / dynamic range.

    The kernel pool on Linux is conventionally 32768-60999 and on
    macOS / BSD is 49152-65535; both are subsets of the IANA
    "dynamic / private" range 49152-65535 plus the upper portion of
    "registered ports". We assert the broadest acceptable range so
    the test is not platform-specific.
    """
    for _ in range(invocations):
        port = random_ephemeral_port()
        assert 1024 < port <= 65535
        assert port != DEFAULT_API_PORT


def test_kernel_assigned_port_is_actually_free() -> None:
    """A returned port can be re-bound immediately on the same host.

    The whole point of the bind-to-port-0 idiom is that the kernel
    will not reassign the port for some short window after we close
    our transient socket. Re-binding it from the same process here
    is a strong proxy for "free at the moment of return".
    """
    port = random_ephemeral_port()
    with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as sock:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind(("", port))


def test_returned_ports_are_not_all_identical() -> None:
    """Independent calls return distinct ports.

    The kernel hands out fresh ports from its free pool, so a small
    handful of consecutive calls should yield more than one distinct
    value. Asserting strict uniqueness across N calls would be
    flaky (the kernel can reuse a port if we close fast enough); we
    only require that the function is not a constant.
    """
    ports = {random_ephemeral_port() for _ in range(8)}
    assert len(ports) > 1
