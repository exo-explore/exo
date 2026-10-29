import asyncio
import os
import socket
from collections.abc import Sequence

from loguru import logger

from exo.routing.peer_resolver import BootstrapPeerResolver, ResolvedPeer


class BootstrapConnector:
    """Manages active reachability probing and Zenoh environment configuration for bootstrap peers."""

    @classmethod
    def apply_zenoh_env(cls, peers: Sequence[ResolvedPeer]) -> None:
        """Injects normalized Zenoh connect endpoints into the process environment."""
        if not peers:
            return

        locators = BootstrapPeerResolver.to_zenoh_endpoints(peers)
        endpoints_str = ",".join(locators)

        # Set standard ZENOH_CONNECT_ENDPOINTS variable
        os.environ["ZENOH_CONNECT_ENDPOINTS"] = endpoints_str
        logger.debug(f"Configured ZENOH_CONNECT_ENDPOINTS={endpoints_str}")

    @classmethod
    async def probe_peer_reachability(
        cls,
        peer: ResolvedPeer,
        timeout_seconds: float = 1.5,
    ) -> bool:
        """Probes whether a bootstrap peer's TCP port is actively listening without blocking."""
        loop = asyncio.get_running_loop()

        def _sync_test() -> bool:
            try:
                # Support IPv6 and IPv4
                family = socket.AF_INET6 if peer.is_ipv6 else socket.AF_INET
                sock = socket.socket(family, socket.SOCK_STREAM)
                sock.settimeout(timeout_seconds)
                host = peer.host.strip("[]")
                sock.connect((host, peer.port))
                sock.close()
                return True
            except (TimeoutError, OSError):
                return False

        try:
            is_reachable = await loop.run_in_executor(None, _sync_test)
            if is_reachable:
                logger.info(
                    f"Bootstrap peer {peer.zenoh_locator} is actively reachable"
                )
            else:
                logger.warning(
                    f"Bootstrap peer {peer.zenoh_locator} did not respond within {timeout_seconds}s "
                    "(will retry in background via Zenoh gossip)"
                )
            return is_reachable
        except (OSError, TimeoutError) as exc:
            logger.debug(f"Error probing {peer.zenoh_locator}: {exc}")
            return False

    @classmethod
    async def probe_all_peers(
        cls,
        peers: Sequence[ResolvedPeer],
        timeout_seconds: float = 1.5,
    ) -> dict[str, bool]:
        """Probes all configured bootstrap peers in parallel."""
        if not peers:
            return {}

        tasks = [
            cls.probe_peer_reachability(peer, timeout_seconds=timeout_seconds)
            for peer in peers
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        reachability_map: dict[str, bool] = {}
        for peer, res in zip(peers, results, strict=False):
            reachability_map[peer.zenoh_locator] = bool(res is True)

        return reachability_map
