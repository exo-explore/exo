import ipaddress
import re
from collections.abc import Sequence
from dataclasses import dataclass

DEFAULT_ZENOH_PORT = 52414


@dataclass(frozen=True)
class ResolvedPeer:
    """Represents a standardized bootstrap peer endpoint."""

    host: str
    port: int
    is_ipv6: bool
    zenoh_locator: str
    original: str


class BootstrapPeerResolver:
    """Normalizes and resolves diverse peer address formats into standard Zenoh locators.

    Supports:
      - Plain IP: "192.168.1.100" -> tcp/192.168.1.100:52414
      - IP with Port: "192.168.1.100:6000" -> tcp/192.168.1.100:6000
      - IPv6 with/without brackets: "[::1]:52414", "::1" -> tcp/[::1]:52414
      - DNS hostnames: "node1.cluster.local:52414" -> tcp/node1.cluster.local:52414
      - Legacy libp2p multiaddrs: "/ip4/192.168.1.100/tcp/52414/p2p/12D3..." -> tcp/192.168.1.100:52414
      - Raw Zenoh locators: "tcp/192.168.1.100:52414" -> tcp/192.168.1.100:52414
    """

    MULTIADDR_IP4_REGEX = re.compile(
        r"^/ip4/(?P<host>\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})(?:/tcp/(?P<port>\d{1,5}))?(?:/p2p/\w+)?$"
    )
    MULTIADDR_IP6_REGEX = re.compile(
        r"^/ip6/(?P<host>[0-9a-fA-F:]+)(?:/tcp/(?P<port>\d{1,5}))?(?:/p2p/\w+)?$"
    )
    MULTIADDR_DNS_REGEX = re.compile(
        r"^/dns[46]?/(?P<host>[a-zA-Z0-9.-]+)(?:/tcp/(?P<port>\d{1,5}))?(?:/p2p/\w+)?$"
    )

    @classmethod
    def normalize_peer(
        cls,
        raw_peer: str,
        default_port: int = DEFAULT_ZENOH_PORT,
    ) -> ResolvedPeer:
        """Parses a single peer string into a standardized ResolvedPeer."""
        if not raw_peer or not raw_peer.strip():
            raise ValueError("Bootstrap peer address cannot be empty")

        cleaned = raw_peer.strip().strip("'\"")

        # Strip explicit tcp/ prefix if provided
        cleaned = cleaned.removeprefix("tcp/")

        # Case 1: libp2p Multiaddr format (/ip4/..., /ip6/..., /dns/...)
        if cleaned.startswith("/"):
            return cls._parse_multiaddr(cleaned, raw_peer, default_port)

        # Case 2: Bracketed IPv6 format, e.g. [2001:db8::1]:52414 or [::1]
        if cleaned.startswith("["):
            return cls._parse_bracketed_ipv6(cleaned, raw_peer, default_port)

        # Case 3: Standard host:port or pure host/ip
        return cls._parse_host_port(cleaned, raw_peer, default_port)

    @classmethod
    def _parse_multiaddr(
        cls, cleaned: str, original: str, default_port: int
    ) -> ResolvedPeer:
        for regex in (
            cls.MULTIADDR_IP4_REGEX,
            cls.MULTIADDR_IP6_REGEX,
            cls.MULTIADDR_DNS_REGEX,
        ):
            match = regex.match(cleaned)
            if match:
                host = match.group("host")
                port_str = match.group("port")
                port = int(port_str) if port_str else default_port
                is_ipv6 = cls._is_valid_ipv6(host)
                locator = cls._format_locator(host, port, is_ipv6)
                return ResolvedPeer(
                    host=host,
                    port=port,
                    is_ipv6=is_ipv6,
                    zenoh_locator=locator,
                    original=original,
                )

        raise ValueError(f"Unrecognized multiaddr format: '{original}'")

    @classmethod
    def _parse_bracketed_ipv6(
        cls, cleaned: str, original: str, default_port: int
    ) -> ResolvedPeer:
        end_bracket = cleaned.find("]")
        if end_bracket == -1:
            raise ValueError(f"Mismatched IPv6 bracket in '{original}'")

        host = cleaned[1:end_bracket]
        remainder = cleaned[end_bracket + 1 :]
        port = default_port

        if remainder:
            if not remainder.startswith(":"):
                raise ValueError(f"Invalid separator after IPv6 bracket in '{original}'")
            port = int(remainder[1:])

        cls._validate_port(port)
        locator = cls._format_locator(host, port, is_ipv6=True)
        return ResolvedPeer(
            host=host,
            port=port,
            is_ipv6=True,
            zenoh_locator=locator,
            original=original,
        )

    @classmethod
    def _parse_host_port(
        cls, cleaned: str, original: str, default_port: int
    ) -> ResolvedPeer:
        # Check if it's a bare IPv6 without brackets (e.g. "fe80::1" or "::1")
        if cls._is_valid_ipv6(cleaned):
            locator = cls._format_locator(cleaned, default_port, is_ipv6=True)
            return ResolvedPeer(
                host=cleaned,
                port=default_port,
                is_ipv6=True,
                zenoh_locator=locator,
                original=original,
            )

        # Standard IPv4 or hostname with optional port (host:port)
        if ":" in cleaned:
            parts = cleaned.rsplit(":", 1)
            host = parts[0]
            try:
                port = int(parts[1])
            except ValueError:
                raise ValueError(f"Invalid port in bootstrap peer: '{original}'") from None
        else:
            host = cleaned
            port = default_port

        cls._validate_port(port)
        is_ipv6 = cls._is_valid_ipv6(host)
        locator = cls._format_locator(host, port, is_ipv6)
        return ResolvedPeer(
            host=host,
            port=port,
            is_ipv6=is_ipv6,
            zenoh_locator=locator,
            original=original,
        )

    @classmethod
    def parse_peer_list(
        cls,
        raw_input: str | Sequence[str] | None,
        default_port: int = DEFAULT_ZENOH_PORT,
    ) -> list[ResolvedPeer]:
        """Parses a list, single string, or delimited environment variable of peers."""
        if not raw_input:
            return []

        raw_items: list[str] = []
        if isinstance(raw_input, str):
            # Split on comma, semicolon, or whitespace
            raw_items = [
                item.strip()
                for item in re.split(r"[,;\s]+", raw_input)
                if item.strip()
            ]
        else:
            for entry in raw_input:
                if entry:
                    raw_items.extend(
                        [
                            item.strip()
                            for item in re.split(r"[,;\s]+", entry)
                            if item.strip()
                        ]
                    )

        resolved_peers: list[ResolvedPeer] = []
        seen_locators = set()

        for item in raw_items:
            try:
                peer = cls.normalize_peer(item, default_port=default_port)
                if peer.zenoh_locator not in seen_locators:
                    seen_locators.add(peer.zenoh_locator)
                    resolved_peers.append(peer)
            except ValueError:
                continue

        return resolved_peers

    @classmethod
    def to_zenoh_endpoints(cls, peers: Sequence[ResolvedPeer]) -> list[str]:
        """Returns a list of Zenoh locator endpoint strings."""
        return [peer.zenoh_locator for peer in peers]

    @staticmethod
    def _is_valid_ipv6(addr: str) -> bool:
        try:
            return ipaddress.ip_address(addr).version == 6
        except ValueError:
            return False

    @staticmethod
    def _validate_port(port: int) -> None:
        if not (1 <= port <= 65535):
            raise ValueError(f"Port {port} out of valid TCP range 1..65535")

    @staticmethod
    def _format_locator(host: str, port: int, is_ipv6: bool) -> str:
        if is_ipv6 and not host.startswith("["):
            return f"tcp/[{host}]:{port}"
        return f"tcp/{host}:{port}"
