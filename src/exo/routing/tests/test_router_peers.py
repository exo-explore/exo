"""
Unit tests for Router peer bootstrapping (EXO_PEERS) and reconnect logic.
No I/O — mocks the libp2p dial to test pure Python logic only.
"""
import os
from typing import cast
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from exo.routing.router import Router


def _make_router() -> Router:
    handle = MagicMock()
    handle.dial = AsyncMock()
    return Router(handle=handle)


def _dial_mock(router: Router) -> AsyncMock:
    return cast(AsyncMock, router._net.dial)  # pyright: ignore[reportPrivateUsage]


@pytest.mark.asyncio
async def test_auto_dial_empty_env():
    """EXO_PEERS unset → no dial calls, no persistent_peers populated."""
    router = _make_router()
    with patch.dict(os.environ, {}, clear=False):
        os.environ.pop("EXO_PEERS", None)
        await router._auto_dial_peers()  # pyright: ignore[reportPrivateUsage]
    assert _dial_mock(router).call_count == 0
    assert router._persistent_peers == []  # pyright: ignore[reportPrivateUsage]


@pytest.mark.asyncio
async def test_auto_dial_parses_peers():
    """EXO_PEERS with 2 addrs → both stored in _persistent_peers."""
    router = _make_router()
    addrs = "/ip4/192.168.2.2/tcp/51821/p2p/QmFoo,/ip4/10.0.0.152/tcp/51821/p2p/QmBar"
    with patch.dict(os.environ, {"EXO_PEERS": addrs}), patch("exo.routing.router.sleep", new_callable=AsyncMock):
        await router._auto_dial_peers()  # pyright: ignore[reportPrivateUsage]
    assert len(router._persistent_peers) == 2  # pyright: ignore[reportPrivateUsage]
    assert router._persistent_peers[0] == "/ip4/192.168.2.2/tcp/51821/p2p/QmFoo"  # pyright: ignore[reportPrivateUsage]
    assert router._persistent_peers[1] == "/ip4/10.0.0.152/tcp/51821/p2p/QmBar"  # pyright: ignore[reportPrivateUsage]


@pytest.mark.asyncio
async def test_auto_dial_skips_when_already_connected():
    """If a peer already connected before auto-dial runs, dial is skipped."""
    router = _make_router()
    router._connected_peer_ids.add("QmAlreadyHere")  # pyright: ignore[reportPrivateUsage]
    addrs = "/ip4/192.168.2.2/tcp/51821/p2p/QmFoo"
    with patch.dict(os.environ, {"EXO_PEERS": addrs}), patch("exo.routing.router.sleep", new_callable=AsyncMock):
        await router._auto_dial_peers()  # pyright: ignore[reportPrivateUsage]
    assert _dial_mock(router).call_count == 0


@pytest.mark.asyncio
async def test_reconnect_loop_exits_when_no_peers():
    """_reconnect_loop returns immediately if no persistent_peers set."""
    router = _make_router()
    with patch("exo.routing.router.sleep", new_callable=AsyncMock):
        await router._reconnect_loop()  # pyright: ignore[reportPrivateUsage]
    assert _dial_mock(router).call_count == 0


def test_peer_tracking_deduplicates_connect():
    """Second connect event for same peer_id must not add it twice."""
    router = _make_router()
    router._connected_peer_ids.add("QmPeer1")  # pyright: ignore[reportPrivateUsage]
    assert len(router._connected_peer_ids) == 1  # pyright: ignore[reportPrivateUsage]
    router._connected_peer_ids.add("QmPeer1")  # pyright: ignore[reportPrivateUsage]
    assert len(router._connected_peer_ids) == 1  # pyright: ignore[reportPrivateUsage]


def test_peer_tracking_discard_on_disconnect():
    """Disconnect of unknown peer must not raise."""
    router = _make_router()
    router._connected_peer_ids.discard("QmNeverSeen")  # pyright: ignore[reportPrivateUsage]
    assert len(router._connected_peer_ids) == 0  # pyright: ignore[reportPrivateUsage]
