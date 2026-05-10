"""Tests for :func:`_bind_drafter_listener_same_port_retry`.

Covers the round-2 Codex fix for the drafter listener bind retry
(PR #20, ``utils_mlx.py:452``):

* P1 round-2: round-1's port re-roll broke the placement contract --
  the drafter dials ``DrafterPlacement.drafter_socket_port``, so
  switching ports on retry made the listener unreachable. The retry
  must keep the SAME port (TIME_WAIT residue is the realistic
  ``EADDRINUSE`` case in cross-host deploys and clears within ~100ms).
* P2 round-2: round-1 caught every ``OSError``, hiding non-collision
  errors (``EAFNOSUPPORT`` / ``EACCES``) behind a misleading
  "ephemeral port range" message. Only ``EADDRINUSE`` is transient;
  everything else surfaces immediately.

Pure-unit tests with an injected ``bind_target_listener`` -- no real
sockets bound, no sleeps observed (we patch ``time.sleep``).
"""

from __future__ import annotations

import errno
import socket
from unittest import mock

import pytest

from exo.worker.engines.mlx.utils_mlx import (
    _DRAFTER_BIND_RETRY_BUDGET,  # pyright: ignore[reportPrivateUsage]
    _bind_drafter_listener_same_port_retry,  # pyright: ignore[reportPrivateUsage]
)


def _eaddrinuse() -> OSError:
    return OSError(errno.EADDRINUSE, "Address already in use")


def _eafnosupport() -> OSError:
    return OSError(errno.EAFNOSUPPORT, "Address family not supported")


def _eacces() -> OSError:
    return OSError(errno.EACCES, "Permission denied")


class TestSamePortRetry:
    """Round-2 P1: retry must keep the placement-announced port."""

    def test_first_attempt_succeeds_returns_listener(self) -> None:
        listener = mock.Mock(spec=socket.socket)
        bind = mock.Mock(return_value=listener)
        with mock.patch("time.sleep") as sleep:
            result = _bind_drafter_listener_same_port_retry(
                bind_host="::",
                bind_target_listener=bind,
                port=12345,
                advertised_host="127.0.0.1",
            )
        assert result is listener
        assert bind.call_count == 1
        bind.assert_called_with("::", 12345)
        assert sleep.call_count == 0

    def test_transient_eaddrinuse_then_success_keeps_same_port(self) -> None:
        listener = mock.Mock(spec=socket.socket)
        bind = mock.Mock(side_effect=[_eaddrinuse(), _eaddrinuse(), listener])
        with mock.patch("time.sleep") as sleep:
            result = _bind_drafter_listener_same_port_retry(
                bind_host="0.0.0.0",
                bind_target_listener=bind,
                port=42000,
                advertised_host="10.0.0.5",
            )
        assert result is listener
        assert bind.call_count == 3
        # All attempts must use the SAME port -- changing it would break
        # the placement contract (the drafter dials port 42000).
        for call_args in bind.call_args_list:
            assert tuple(call_args.args) == ("0.0.0.0", 42000)
        assert sleep.call_count == 2

    def test_persistent_eaddrinuse_exhausts_budget(self) -> None:
        bind = mock.Mock(side_effect=_eaddrinuse())
        with mock.patch("time.sleep"), pytest.raises(OSError) as exc_info:
            _bind_drafter_listener_same_port_retry(
                bind_host="::",
                bind_target_listener=bind,
                port=42000,
                advertised_host="fd00::1",
            )
        assert exc_info.value.errno == errno.EADDRINUSE
        assert bind.call_count == _DRAFTER_BIND_RETRY_BUDGET
        # Final error must mention the port so the operator can re-place.
        assert "42000" in str(exc_info.value)


class TestNonEaddrinuseSurfacesImmediately:
    """Round-2 P2: non-collision errors must not be retried."""

    def test_eafnosupport_raises_on_first_attempt(self) -> None:
        bind = mock.Mock(side_effect=_eafnosupport())
        with mock.patch("time.sleep") as sleep, pytest.raises(OSError) as exc_info:
            _bind_drafter_listener_same_port_retry(
                bind_host="::",
                bind_target_listener=bind,
                port=12345,
                advertised_host="fd00::1",
            )
        assert exc_info.value.errno == errno.EAFNOSUPPORT
        assert bind.call_count == 1
        assert sleep.call_count == 0

    def test_eacces_raises_on_first_attempt(self) -> None:
        bind = mock.Mock(side_effect=_eacces())
        with mock.patch("time.sleep") as sleep, pytest.raises(OSError) as exc_info:
            _bind_drafter_listener_same_port_retry(
                bind_host="0.0.0.0",
                bind_target_listener=bind,
                port=80,
                advertised_host="10.0.0.5",
            )
        assert exc_info.value.errno == errno.EACCES
        assert bind.call_count == 1
        assert sleep.call_count == 0

    def test_eaddrinuse_then_eafnosupport_surfaces_eafnosupport(self) -> None:
        bind = mock.Mock(side_effect=[_eaddrinuse(), _eafnosupport()])
        with mock.patch("time.sleep"), pytest.raises(OSError) as exc_info:
            _bind_drafter_listener_same_port_retry(
                bind_host="::",
                bind_target_listener=bind,
                port=12345,
                advertised_host="fd00::1",
            )
        # The second attempt's EAFNOSUPPORT must surface, not the first
        # attempt's EADDRINUSE.
        assert exc_info.value.errno == errno.EAFNOSUPPORT
        assert bind.call_count == 2
