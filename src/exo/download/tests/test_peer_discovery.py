"""Tests for peer-discovery URL selection.

Each test case constructs a small synthetic state (a couple of peers, a
download status snapshot, a network-info map) and asserts that
`find_peer_repo_url` either returns the expected URL or `None`.
"""

from collections.abc import Mapping, Sequence

from exo.download.peer_discovery import find_peer_repo_url
from exo.shared.constants import EXO_FILE_SERVER_PORT
from exo.shared.models.model_cards import ModelCard, ModelId, ModelTask
from exo.shared.types.common import NodeId
from exo.shared.types.memory import Memory
from exo.shared.types.profiling import NetworkInterfaceInfo, NodeNetworkInfo
from exo.shared.types.worker.downloads import (
    DownloadCompleted,
    DownloadOngoing,
    DownloadPending,
    DownloadProgress,
    DownloadProgressData,
)
from exo.shared.types.worker.shards import PipelineShardMetadata, ShardMetadata

NODE_SELF = NodeId("aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa")
NODE_PEER_A = NodeId("bbbbbbbb-bbbb-4bbb-8bbb-bbbbbbbbbbbb")
NODE_PEER_B = NodeId("cccccccc-cccc-4ccc-8ccc-cccccccccccc")
MODEL_ID = ModelId("test-org/test-model")
OTHER_MODEL_ID = ModelId("test-org/other-model")


def _shard(model_id: ModelId = MODEL_ID) -> ShardMetadata:
    return PipelineShardMetadata(
        model_card=ModelCard(
            model_id=model_id,
            storage_size=Memory.from_mb(100),
            n_layers=1,
            hidden_size=1,
            supports_tensor=False,
            tasks=[ModelTask.TextGeneration],
        ),
        device_rank=0,
        world_size=1,
        start_layer=0,
        end_layer=1,
        n_layers=1,
    )


def _completed(node_id: NodeId, model_id: ModelId = MODEL_ID) -> DownloadCompleted:
    return DownloadCompleted(
        node_id=node_id,
        shard_metadata=_shard(model_id),
        total=Memory.from_mb(100),
        model_directory="",
    )


def _ongoing(node_id: NodeId) -> DownloadOngoing:
    return DownloadOngoing(
        node_id=node_id,
        shard_metadata=_shard(),
        model_directory="",
        download_progress=DownloadProgressData(
            total=Memory.from_mb(100),
            downloaded=Memory.from_mb(50),
            downloaded_this_session=Memory.from_mb(50),
            completed_files=1,
            total_files=2,
            speed=0,
            eta_ms=0,
            files={},
        ),
    )


def _pending(node_id: NodeId) -> DownloadPending:
    return DownloadPending(
        node_id=node_id,
        shard_metadata=_shard(),
        model_directory="",
    )


def _net(*ifaces: tuple[str, str]) -> NodeNetworkInfo:
    return NodeNetworkInfo(
        interfaces=[
            NetworkInterfaceInfo(
                name=f"if{idx}", ip_address=ip, interface_type=iface_type  # type: ignore[arg-type]
            )
            for idx, (ip, iface_type) in enumerate(ifaces)
        ]
    )


def _state(
    downloads: Mapping[NodeId, Sequence[DownloadProgress]],
    network: Mapping[NodeId, NodeNetworkInfo],
) -> tuple[
    Mapping[NodeId, Sequence[DownloadProgress]],
    Mapping[NodeId, NodeNetworkInfo],
]:
    return downloads, network


# ---- "no peer found" cases --------------------------------------------------


def test_returns_none_when_no_peers() -> None:
    downloads, net = _state({NODE_SELF: []}, {})
    assert find_peer_repo_url(NODE_SELF, str(MODEL_ID), downloads, net) is None


def test_skips_self() -> None:
    """Even if the requester has the model completed, it must not be returned
    as its own peer."""
    downloads, net = _state(
        {NODE_SELF: [_completed(NODE_SELF)]},
        {NODE_SELF: _net(("10.0.0.1", "ethernet"))},
    )
    assert find_peer_repo_url(NODE_SELF, str(MODEL_ID), downloads, net) is None


def test_skips_peer_with_only_ongoing_download() -> None:
    """A peer that is *currently downloading* the model is not yet a source."""
    downloads, net = _state(
        {NODE_PEER_A: [_ongoing(NODE_PEER_A)]},
        {NODE_PEER_A: _net(("10.0.0.2", "ethernet"))},
    )
    assert find_peer_repo_url(NODE_SELF, str(MODEL_ID), downloads, net) is None


def test_skips_peer_with_only_pending_download() -> None:
    downloads, net = _state(
        {NODE_PEER_A: [_pending(NODE_PEER_A)]},
        {NODE_PEER_A: _net(("10.0.0.2", "ethernet"))},
    )
    assert find_peer_repo_url(NODE_SELF, str(MODEL_ID), downloads, net) is None


def test_skips_peer_with_completed_other_model() -> None:
    downloads, net = _state(
        {NODE_PEER_A: [_completed(NODE_PEER_A, OTHER_MODEL_ID)]},
        {NODE_PEER_A: _net(("10.0.0.2", "ethernet"))},
    )
    assert find_peer_repo_url(NODE_SELF, str(MODEL_ID), downloads, net) is None


def test_skips_peer_with_no_network_info() -> None:
    downloads, net = _state(
        {NODE_PEER_A: [_completed(NODE_PEER_A)]},
        {},  # no network info entry for the peer
    )
    assert find_peer_repo_url(NODE_SELF, str(MODEL_ID), downloads, net) is None


def test_skips_peer_with_empty_interfaces() -> None:
    downloads, net = _state(
        {NODE_PEER_A: [_completed(NODE_PEER_A)]},
        {NODE_PEER_A: NodeNetworkInfo(interfaces=[])},
    )
    assert find_peer_repo_url(NODE_SELF, str(MODEL_ID), downloads, net) is None


def test_skips_peer_with_only_unroutable_addresses() -> None:
    """IPv6, loopback, and link-local IPv4 are all unroutable for our purposes."""
    downloads, net = _state(
        {NODE_PEER_A: [_completed(NODE_PEER_A)]},
        {
            NODE_PEER_A: _net(
                ("fe80::1", "ethernet"),
                ("::1", "ethernet"),
                ("127.0.0.1", "ethernet"),
                ("169.254.1.1", "ethernet"),
            )
        },
    )
    assert find_peer_repo_url(NODE_SELF, str(MODEL_ID), downloads, net) is None


# ---- "peer found" cases -----------------------------------------------------


def test_returns_url_for_completed_peer() -> None:
    downloads, net = _state(
        {NODE_PEER_A: [_completed(NODE_PEER_A)]},
        {NODE_PEER_A: _net(("10.0.0.2", "ethernet"))},
    )
    url = find_peer_repo_url(NODE_SELF, str(MODEL_ID), downloads, net)
    assert url == f"http://10.0.0.2:{EXO_FILE_SERVER_PORT}"


def test_thunderbolt_beats_ethernet() -> None:
    """Both interfaces are routable; thunderbolt wins."""
    downloads, net = _state(
        {NODE_PEER_A: [_completed(NODE_PEER_A)]},
        {
            NODE_PEER_A: _net(
                ("10.0.0.2", "ethernet"),
                ("169.254.81.1", "ethernet"),  # gets skipped (link-local)
                ("10.0.1.2", "thunderbolt"),
            )
        },
    )
    url = find_peer_repo_url(NODE_SELF, str(MODEL_ID), downloads, net)
    assert url == f"http://10.0.1.2:{EXO_FILE_SERVER_PORT}"


def test_maybe_ethernet_beats_confirmed_ethernet() -> None:
    """In the adurham M4 cluster, TB-bridges show up as `maybe_ethernet`,
    so we deliberately rank them above confirmed `ethernet`."""
    downloads, net = _state(
        {NODE_PEER_A: [_completed(NODE_PEER_A)]},
        {
            NODE_PEER_A: _net(
                ("10.0.0.2", "ethernet"),
                ("10.0.5.2", "maybe_ethernet"),
            )
        },
    )
    url = find_peer_repo_url(NODE_SELF, str(MODEL_ID), downloads, net)
    assert url == f"http://10.0.5.2:{EXO_FILE_SERVER_PORT}"


def test_unknown_interface_type_is_lower_priority_than_ethernet() -> None:
    downloads, net = _state(
        {NODE_PEER_A: [_completed(NODE_PEER_A)]},
        {
            NODE_PEER_A: _net(
                ("10.0.9.2", "wifi"),  # not in the priority table
                ("10.0.0.2", "ethernet"),
            )
        },
    )
    url = find_peer_repo_url(NODE_SELF, str(MODEL_ID), downloads, net)
    assert url == f"http://10.0.0.2:{EXO_FILE_SERVER_PORT}"


def test_skips_ipv6_picks_ipv4() -> None:
    downloads, net = _state(
        {NODE_PEER_A: [_completed(NODE_PEER_A)]},
        {
            NODE_PEER_A: _net(
                ("fd00::1", "thunderbolt"),  # IPv6 — skipped
                ("10.0.1.2", "ethernet"),
            )
        },
    )
    url = find_peer_repo_url(NODE_SELF, str(MODEL_ID), downloads, net)
    assert url == f"http://10.0.1.2:{EXO_FILE_SERVER_PORT}"


def test_first_completed_peer_wins() -> None:
    """When multiple peers have the model, the first one we encounter in
    download_status iteration order is selected (Python dict iteration is
    insertion-ordered, so this is deterministic for a given input)."""
    downloads, net = _state(
        {
            NODE_PEER_A: [_completed(NODE_PEER_A)],
            NODE_PEER_B: [_completed(NODE_PEER_B)],
        },
        {
            NODE_PEER_A: _net(("10.0.0.2", "ethernet")),
            NODE_PEER_B: _net(("10.0.0.3", "thunderbolt")),
        },
    )
    url = find_peer_repo_url(NODE_SELF, str(MODEL_ID), downloads, net)
    # NODE_PEER_A came first, so we don't search further even though
    # NODE_PEER_B has a faster interface.
    assert url == f"http://10.0.0.2:{EXO_FILE_SERVER_PORT}"


def test_falls_through_to_second_peer_when_first_unreachable() -> None:
    """If the first completed peer has no routable IP, fall through to the next."""
    downloads, net = _state(
        {
            NODE_PEER_A: [_completed(NODE_PEER_A)],
            NODE_PEER_B: [_completed(NODE_PEER_B)],
        },
        {
            NODE_PEER_A: _net(("127.0.0.1", "ethernet")),  # unroutable
            NODE_PEER_B: _net(("10.0.0.3", "ethernet")),
        },
    )
    url = find_peer_repo_url(NODE_SELF, str(MODEL_ID), downloads, net)
    assert url == f"http://10.0.0.3:{EXO_FILE_SERVER_PORT}"


def test_completed_among_other_states_still_wins() -> None:
    """A peer can be in DownloadCompleted for one model and DownloadOngoing
    for another — we should still pick it for the completed one."""
    downloads, net = _state(
        {
            NODE_PEER_A: [
                _ongoing(NODE_PEER_A),  # different model, in progress
                _completed(NODE_PEER_A, MODEL_ID),  # the one we want
            ]
        },
        {NODE_PEER_A: _net(("10.0.0.2", "ethernet"))},
    )
    url = find_peer_repo_url(NODE_SELF, str(MODEL_ID), downloads, net)
    assert url == f"http://10.0.0.2:{EXO_FILE_SERVER_PORT}"
