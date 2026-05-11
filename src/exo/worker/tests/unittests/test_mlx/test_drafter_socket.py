"""Tests for :mod:`exo.worker.engines.mlx.generator.drafter_socket`.

Focused on bind-time address-family resolution: the asymmetric drafter
listener must accept the drafter's dial regardless of whether
``DrafterPlacement.drafter_socket_host`` resolved to an IPv4 or IPv6
address. Pre-fix the listener was hard-coded to ``AF_INET`` and an
IPv6 advertised host (Tailscale ULA, link-local IPv6, IPv6-only LAN)
could never accept the drafter's dial.
"""

from __future__ import annotations

import socket
import threading
from typing import cast

import pytest

from exo.worker.engines.mlx.generator.drafter_socket import (
    bind_target_listener,
    dial_target,
)


def _ipv4_sockname(listener: socket.socket) -> tuple[str, int]:
    """Return ``(host, port)`` from an IPv4 listener's ``getsockname``.

    ``socket.socket.getsockname`` is typed as ``Any`` in stdlib, so cast
    locally to keep tests strictly typed.
    """
    return cast(tuple[str, int], listener.getsockname())


def _ipv6_sockname(listener: socket.socket) -> tuple[str, int, int, int]:
    """Return the IPv6 sockaddr 4-tuple from ``getsockname``."""
    return cast(tuple[str, int, int, int], listener.getsockname())


def _has_ipv6_loopback() -> bool:
    """Return ``True`` if the host has a usable IPv6 loopback.

    CI runners occasionally lack IPv6 entirely (notably some container
    images and cross-platform GitHub Actions runners). Skip IPv6
    coverage in that case rather than failing the test.
    """
    try:
        with socket.socket(socket.AF_INET6, socket.SOCK_STREAM) as probe:
            probe.bind(("::1", 0))
        return True
    except OSError:
        return False


class TestBindTargetListenerFamilyResolution:
    """Codex P2 (PR #20 round-(N+9), drafter_socket.py:106): the listener
    must use family-appropriate sockets so an IPv6 advertised host can
    accept the drafter's dial.
    """

    def test_ipv4_wildcard_binds_af_inet_listener(self) -> None:
        listener = bind_target_listener("0.0.0.0", 0)
        try:
            assert listener.family == socket.AF_INET
            host, port = _ipv4_sockname(listener)
            assert host == "0.0.0.0"
            assert port > 0
        finally:
            listener.close()

    def test_ipv4_literal_binds_af_inet_listener(self) -> None:
        listener = bind_target_listener("127.0.0.1", 0)
        try:
            assert listener.family == socket.AF_INET
        finally:
            listener.close()

    def test_ipv6_wildcard_binds_af_inet6_listener(self) -> None:
        if not _has_ipv6_loopback():
            pytest.skip("host has no usable IPv6 loopback")
        listener = bind_target_listener("::", 0)
        try:
            assert listener.family == socket.AF_INET6
            # Dual-stack must be enabled (IPV6_V6ONLY=0) so an IPv6
            # wildcard bind also services IPv4-mapped connects on
            # platforms where dual-stack is off-by-default (Linux).
            v6only = listener.getsockopt(socket.IPPROTO_IPV6, socket.IPV6_V6ONLY)
            assert v6only == 0, (
                "IPv6 listener must run in dual-stack mode so IPv4-mapped "
                "connects are accepted"
            )
        finally:
            listener.close()

    def test_ipv6_literal_binds_af_inet6_listener(self) -> None:
        if not _has_ipv6_loopback():
            pytest.skip("host has no usable IPv6 loopback")
        listener = bind_target_listener("::1", 0)
        try:
            assert listener.family == socket.AF_INET6
        finally:
            listener.close()

    def test_ipv4_dial_reaches_ipv4_listener(self) -> None:
        listener = bind_target_listener("127.0.0.1", 0)
        try:
            _host, port = _ipv4_sockname(listener)
            accepted: list[socket.socket] = []

            def _accept_once() -> None:
                listener.settimeout(5.0)
                try:
                    accepted_pair = listener.accept()
                    accepted.append(accepted_pair[0])
                finally:
                    listener.settimeout(None)

            accept_thread = threading.Thread(target=_accept_once, daemon=True)
            accept_thread.start()
            client = dial_target("127.0.0.1", port, total_timeout_seconds=5.0)
            try:
                accept_thread.join(timeout=5.0)
                assert not accept_thread.is_alive()
                assert len(accepted) == 1
            finally:
                client.close()
                if accepted:
                    accepted[0].close()
        finally:
            listener.close()

    def test_dial_target_respects_total_timeout_when_no_listener(self) -> None:
        """Codex P2 (PR #20 round-(N+13), drafter_socket.py:195):
        each ``socket.create_connection`` attempt MUST use the
        remaining time until the deadline, not a fixed
        ``min(10.0, total_timeout_seconds)``.

        Pre-fix the loop pattern was:
        * Start at deadline = now + total_timeout_seconds (e.g. 1.5s).
        * Attempt 1: ``timeout=min(10.0, 1.5) = 1.5s`` -> fails fast
          on a refusing peer (ConnectionRefusedError) at ~0s.
        * Backoff sleep ~0.5s -> now ~0.5s into the budget.
        * Attempt 2: ``timeout=min(10.0, 1.5) = 1.5s`` -> on a
          black-hole peer can block the FULL 1.5s -> total elapsed
          ~2.0s -> exceeds the configured ``total_timeout_seconds``.

        This test wires up a black-hole-style refusing peer (a TCP
        listener socket that we close immediately so connects get
        ``ConnectionRefusedError`` instantaneously) and asserts the
        function raises ``ConnectionError`` close to the deadline,
        not significantly past it. We allow 0.5s slack on top of
        the configured budget for shell startup / scheduling
        jitter; the pre-fix behavior would routinely overshoot by
        the full ``min(10.0, ...)`` cap on the final attempt.
        """
        import time

        # Allocate-bind-close to get a port that will refuse
        # connects (the canonical way to reserve a known-refusing
        # endpoint without needing a real black-hole).
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as scratch:
            scratch.bind(("127.0.0.1", 0))
            _host, refusing_port = _ipv4_sockname(scratch)

        budget = 1.0
        slack = 0.6
        start = time.monotonic()
        with pytest.raises(ConnectionError, match="within"):
            dial_target(
                "127.0.0.1",
                refusing_port,
                total_timeout_seconds=budget,
                initial_backoff_seconds=0.05,
            )
        elapsed = time.monotonic() - start
        assert elapsed <= budget + slack, (
            f"dial_target exceeded total_timeout_seconds={budget:.1f}s "
            f"by more than {slack:.1f}s slack; elapsed={elapsed:.2f}s. "
            f"Pre-fix the per-attempt timeout was a fixed "
            f"min(10.0, total_timeout_seconds), so the final "
            f"attempt could block past the deadline; post-fix the "
            f"per-attempt timeout uses the remaining budget."
        )

    def test_ipv4_dial_reaches_dual_stack_ipv6_listener(self) -> None:
        """Pre-fix an IPv6 advertised host with an IPv4 drafter would
        be unreachable; with dual-stack the listener accepts the
        IPv4-mapped connect. This exercises the realistic mixed
        environment where the drafter side resolves an IPv6 host but
        falls back to an IPv4 connect.
        """
        if not _has_ipv6_loopback():
            pytest.skip("host has no usable IPv6 loopback")
        listener = bind_target_listener("::", 0)
        try:
            # IPv6 sockaddr is a 4-tuple; port is at index 1.
            _host, port, _flowinfo, _scopeid = _ipv6_sockname(listener)
            accepted: list[socket.socket] = []

            def _accept_once() -> None:
                listener.settimeout(5.0)
                try:
                    accepted_pair = listener.accept()
                    accepted.append(accepted_pair[0])
                finally:
                    listener.settimeout(None)

            accept_thread = threading.Thread(target=_accept_once, daemon=True)
            accept_thread.start()
            client = dial_target("127.0.0.1", port, total_timeout_seconds=5.0)
            try:
                accept_thread.join(timeout=5.0)
                assert not accept_thread.is_alive(), (
                    "dual-stack IPv6 listener must accept IPv4-mapped "
                    "connects so the drafter's IPv4 dial reaches the target "
                    "even when the advertised host is IPv6"
                )
                assert len(accepted) == 1
            finally:
                client.close()
                if accepted:
                    accepted[0].close()
        finally:
            listener.close()
