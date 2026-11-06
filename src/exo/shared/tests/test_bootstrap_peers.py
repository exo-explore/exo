import argparse
import os
import re
from unittest.mock import patch

import pytest

from exo.routing.bootstrap_connector import BootstrapConnector
from exo.routing.peer_resolver import BootstrapPeerResolver, ResolvedPeer


def test_normalize_ipv4_plain_and_with_port():
    """Verify IPv4 addresses with default port or explicit custom port."""
    peer_plain = BootstrapPeerResolver.normalize_peer("192.168.1.100")
    assert peer_plain.host == "192.168.1.100"
    assert peer_plain.port == 52414
    assert not peer_plain.is_ipv6
    assert peer_plain.zenoh_locator == "tcp/192.168.1.100:52414"

    peer_custom = BootstrapPeerResolver.normalize_peer("10.0.0.5:6000")
    assert peer_custom.host == "10.0.0.5"
    assert peer_custom.port == 6000
    assert peer_custom.zenoh_locator == "tcp/10.0.0.5:6000"


def test_normalize_ipv6_bracketed_and_bare():
    """Verify IPv6 addresses with and without brackets."""
    peer_bare = BootstrapPeerResolver.normalize_peer("::1")
    assert peer_bare.host == "::1"
    assert peer_bare.port == 52414
    assert peer_bare.is_ipv6
    assert peer_bare.zenoh_locator == "tcp/[::1]:52414"

    peer_bracketed = BootstrapPeerResolver.normalize_peer("[2001:db8::1]:8000")
    assert peer_bracketed.host == "2001:db8::1"
    assert peer_bracketed.port == 8000
    assert peer_bracketed.is_ipv6
    assert peer_bracketed.zenoh_locator == "tcp/[2001:db8::1]:8000"


def test_normalize_dns_hostnames():
    """Verify standard DNS domain names and subdomains."""
    peer_domain = BootstrapPeerResolver.normalize_peer("node1.exo.local")
    assert peer_domain.host == "node1.exo.local"
    assert peer_domain.port == 52414
    assert peer_domain.zenoh_locator == "tcp/node1.exo.local:52414"

    peer_custom_port = BootstrapPeerResolver.normalize_peer("cluster.internal:9090")
    assert peer_custom_port.host == "cluster.internal"
    assert peer_custom_port.port == 9090
    assert peer_custom_port.zenoh_locator == "tcp/cluster.internal:9090"


def test_normalize_legacy_multiaddr():
    """Verify legacy libp2p multiaddrs with or without p2p peer IDs."""
    # Plain IPv4 multiaddr
    p1 = BootstrapPeerResolver.normalize_peer("/ip4/192.168.1.100/tcp/52414")
    assert p1.host == "192.168.1.100"
    assert p1.port == 52414
    assert p1.zenoh_locator == "tcp/192.168.1.100:52414"

    # Multiaddr with /p2p/ PeerID appended
    p2 = BootstrapPeerResolver.normalize_peer(
        "/ip4/192.168.1.100/tcp/6000/p2p/12D3KooWTestPeerId"
    )
    assert p2.host == "192.168.1.100"
    assert p2.port == 6000
    assert p2.zenoh_locator == "tcp/192.168.1.100:6000"

    # DNS multiaddr
    p3 = BootstrapPeerResolver.normalize_peer("/dns4/myhost.com/tcp/7000")
    assert p3.host == "myhost.com"
    assert p3.port == 7000
    assert p3.zenoh_locator == "tcp/myhost.com:7000"


def test_normalize_raw_zenoh_locator():
    """Verify already formatted Zenoh locator string is preserved."""
    p = BootstrapPeerResolver.normalize_peer("tcp/192.168.1.50:52414")
    assert p.host == "192.168.1.50"
    assert p.port == 52414
    assert p.zenoh_locator == "tcp/192.168.1.50:52414"


def test_parse_peer_list_delimited_and_deduplication():
    """Verify parsing multi-format strings with comma, semicolon, space and deduplication."""
    raw = (
        "192.168.1.10, 10.0.0.2:6000; /ip4/172.16.0.1/tcp/52414 "
        "192.168.1.10:52414   tcp/10.0.0.2:6000"
    )
    peers = BootstrapPeerResolver.parse_peer_list(raw)
    assert len(peers) == 3
    locators = [p.zenoh_locator for p in peers]
    assert "tcp/192.168.1.10:52414" in locators
    assert "tcp/10.0.0.2:6000" in locators
    assert "tcp/172.16.0.1:52414" in locators


def test_parse_peer_list_empty_and_invalid():
    """Verify empty and invalid inputs produce empty or filtered peer lists."""
    assert BootstrapPeerResolver.parse_peer_list(None) == []
    assert BootstrapPeerResolver.parse_peer_list("") == []
    assert BootstrapPeerResolver.parse_peer_list(["   ", ""]) == []

    with pytest.raises(ValueError, match="cannot be empty"):
        BootstrapPeerResolver.normalize_peer("   ")


def test_bootstrap_connector_apply_zenoh_env():
    """Verify applying Zenoh connect endpoints to os.environ."""
    peers = [
        ResolvedPeer("192.168.1.10", 52414, False, "tcp/192.168.1.10:52414", "192.168.1.10"),
        ResolvedPeer("10.0.0.2", 6000, False, "tcp/10.0.0.2:6000", "10.0.0.2:6000"),
    ]
    BootstrapConnector.apply_zenoh_env(peers)
    assert (
        os.environ.get("ZENOH_CONNECT_ENDPOINTS")
        == "tcp/192.168.1.10:52414,tcp/10.0.0.2:6000"
    )


@pytest.mark.asyncio
async def test_bootstrap_connector_probe():
    """Verify peer reachability probe returns boolean without throwing exceptions."""
    peer = ResolvedPeer(
        host="127.0.0.1",
        port=9,  # Discard port, typically closed
        is_ipv6=False,
        zenoh_locator="tcp/127.0.0.1:9",
        original="127.0.0.1:9",
    )
    result = await BootstrapConnector.probe_peer_reachability(peer, timeout_seconds=0.1)
    assert isinstance(result, bool)

    report = await BootstrapConnector.probe_all_peers([peer], timeout_seconds=0.1)
    assert peer.zenoh_locator in report
    assert isinstance(report[peer.zenoh_locator], bool)


def test_cli_bootstrap_peers_parser_specification():
    """Verify the CLI parser specification for --bootstrap-peers correctly handles inputs."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--bootstrap-peers",
        type=lambda s: [p.strip() for p in re.split(r"[,;\s]+", s) if p.strip()],
        default=[
            p.strip()
            for p in re.split(r"[,;\s]+", os.getenv("EXO_BOOTSTRAP_PEERS", ""))
            if p.strip()
        ]
        if os.getenv("EXO_BOOTSTRAP_PEERS")
        else [],
        dest="bootstrap_peers",
    )

    # Test explicit flag
    args = parser.parse_args(["--bootstrap-peers", "192.168.1.50:52414,node2.local; 10.0.0.3"])
    assert len(args.bootstrap_peers) == 3
    assert args.bootstrap_peers == ["192.168.1.50:52414", "node2.local", "10.0.0.3"]

    # Test environment variable fallback
    with patch.dict(os.environ, {"EXO_BOOTSTRAP_PEERS": "172.16.0.1:52414, 172.16.0.2:52414"}):
        parser_env = argparse.ArgumentParser()
        parser_env.add_argument(
            "--bootstrap-peers",
            type=lambda s: [p.strip() for p in re.split(r"[,;\s]+", s) if p.strip()],
            default=[
                p.strip()
                for p in re.split(r"[,;\s]+", os.getenv("EXO_BOOTSTRAP_PEERS", ""))
                if p.strip()
            ]
            if os.getenv("EXO_BOOTSTRAP_PEERS")
            else [],
            dest="bootstrap_peers",
        )
        args_env = parser_env.parse_args([])
        assert args_env.bootstrap_peers == ["172.16.0.1:52414", "172.16.0.2:52414"]
