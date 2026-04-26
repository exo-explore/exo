"""Peer discovery for P2P model downloads.

Given the global download status and per-node network information, find a
peer that already holds a fully-downloaded copy of a model and return the
URL of its file server. The caller can then point its downloader at that
URL instead of HuggingFace.
"""

from collections.abc import Mapping, Sequence

from exo.shared.constants import EXO_FILE_SERVER_PORT
from exo.shared.types.common import NodeId
from exo.shared.types.profiling import NodeNetworkInfo
from exo.shared.types.worker.downloads import DownloadCompleted, DownloadProgress

# Lower number = preferred. Anything not in this map gets the default below.
# `maybe_ethernet` is preferred over confirmed `ethernet` because on macOS
# Thunderbolt bridges often land in the maybe_ethernet bucket; on a TB-mesh
# cluster these are the high-bandwidth paths we actually want to ride.
_INTERFACE_PRIORITY: dict[str, int] = {
    "thunderbolt": 0,
    "maybe_ethernet": 1,
    "ethernet": 2,
}
_DEFAULT_INTERFACE_PRIORITY = 10


def find_peer_repo_url(
    node_id: NodeId,
    model_id: str,
    global_download_status: Mapping[NodeId, Sequence[DownloadProgress]],
    node_network: Mapping[NodeId, NodeNetworkInfo],
) -> str | None:
    """Return ``http://<best-peer-ip>:<port>`` if a peer has the model, else None.

    Selection rules:
      - Skip the requester itself.
      - Only consider peers in DownloadCompleted for the requested model.
      - Skip peers with no network info or no interfaces.
      - Skip IPv6 addresses, link-local IPv4 (169.254.*) and loopback (127.*).
      - Prefer interface types in this order: thunderbolt > ethernet >
        maybe_ethernet > everything else.
      - Among multiple matching peers, the first one yielded by the mapping
        wins (Python dict insertion order is stable, so this is deterministic
        for a given input).
    """
    for peer_id, peer_downloads in global_download_status.items():
        if peer_id == node_id:
            continue
        if not _peer_has_model(peer_downloads, model_id):
            continue
        ip = _best_peer_ip(node_network.get(peer_id))
        if ip is not None:
            return f"http://{ip}:{EXO_FILE_SERVER_PORT}"
    return None


def _peer_has_model(
    peer_downloads: Sequence[DownloadProgress], model_id: str
) -> bool:
    return any(
        isinstance(dp, DownloadCompleted)
        and dp.shard_metadata.model_card.model_id == model_id
        for dp in peer_downloads
    )


def _best_peer_ip(net_info: NodeNetworkInfo | None) -> str | None:
    if net_info is None or not net_info.interfaces:
        return None
    best_ip: str | None = None
    best_priority = _DEFAULT_INTERFACE_PRIORITY + 1
    for iface in net_info.interfaces:
        ip = iface.ip_address
        if not _ip_is_routable(ip):
            continue
        priority = _INTERFACE_PRIORITY.get(
            iface.interface_type, _DEFAULT_INTERFACE_PRIORITY
        )
        if priority < best_priority:
            best_ip = ip
            best_priority = priority
    return best_ip


def _ip_is_routable(ip: str) -> bool:
    """True if ``ip`` is an IPv4 address we can reach from another host.

    We deliberately reject:
      - IPv6 (aiohttp/curl on some hosts won't dial fe80:: link-local)
      - 127.0.0.0/8 (loopback — peer-on-self isn't useful)
      - 169.254.0.0/16 (RFC 3927 link-local, often unconfigured)
    """
    if ":" in ip:
        return False
    return not ip.startswith(("fe80", "127.", "169.254."))
