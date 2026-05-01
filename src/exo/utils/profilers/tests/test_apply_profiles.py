"""Verify the apply layer wires GpuProfile / SocketLinkProfile / RDMALinkProfile
through to the granular state mappings."""

from datetime import datetime, timezone

from exo.shared.apply import apply
from exo.shared.types.common import NodeId
from exo.shared.types.events import (
    EventId,
    IndexedEvent,
    NodeGatheredInfo,
    NodeTimedOut,
)
from exo.shared.types.profiling import NodeSocketLinkProfile
from exo.shared.types.state import State
from exo.utils.profilers.gpu_profiler import GpuProfile
from exo.utils.profilers.link_profiler import RDMALinkProfile, SocketLinkProfile

NODE_A = NodeId("a")
NODE_B = NodeId("b")
WHEN = str(datetime(2026, 1, 1, tzinfo=timezone.utc))


def _wrap(idx: int, info: object, when: str = WHEN) -> IndexedEvent:
    return IndexedEvent(
        idx=idx,
        event=NodeGatheredInfo(
            event_id=EventId(),
            node_id=NODE_A,
            when=when,
            info=info,  # pyright: ignore[reportArgumentType]
        ),
    )


def test_apply_gpu_profile_writes_node_gpu_profile():
    state = State()
    profile = GpuProfile(engine="mlx", tflops_fp16=42.0, memory_bandwidth_gbps=400.0)
    new_state = apply(state, _wrap(0, profile))
    entry = new_state.node_gpu_profile[NODE_A]
    assert entry.tflops_fp16 == 42.0
    assert entry.memory_bandwidth_gbps == 400.0
    assert entry.engine == "mlx"


def test_apply_socket_link_profile_keys_by_source_and_sink():
    state = State()
    profile = SocketLinkProfile(
        sink_node_id=NODE_B,
        sink_ip="10.0.0.5",
        latency_ms=1.2,
        latency_jitter_ms=0.1,
        upload_mbps=420.0,
        download_mbps=900.0,
    )
    new_state = apply(state, _wrap(0, profile))
    profiles_to_b = new_state.node_link_profiles[NODE_A][NODE_B]
    socket_profiles = [p for p in profiles_to_b if isinstance(p, NodeSocketLinkProfile)]
    assert len(socket_profiles) == 1
    assert socket_profiles[0].latency_ms == 1.2
    assert socket_profiles[0].upload_mbps == 420.0
    assert socket_profiles[0].download_mbps == 900.0


def test_apply_replaces_socket_profile_for_same_transport():
    """A second socket measurement to the same peer should overwrite, not duplicate."""
    state = State()
    p1 = SocketLinkProfile(
        sink_node_id=NODE_B,
        sink_ip="10.0.0.5",
        latency_ms=1.0,
        latency_jitter_ms=0.1,
        upload_mbps=50.0,
        download_mbps=100.0,
    )
    p2 = SocketLinkProfile(
        sink_node_id=NODE_B,
        sink_ip="10.0.0.5",
        latency_ms=2.0,
        latency_jitter_ms=0.1,
        upload_mbps=120.0,
        download_mbps=200.0,
    )
    state = apply(state, _wrap(0, p1))
    state = apply(state, _wrap(1, p2))
    profiles = state.node_link_profiles[NODE_A][NODE_B]
    socket_profiles = [p for p in profiles if isinstance(p, NodeSocketLinkProfile)]
    assert len(socket_profiles) == 1
    assert socket_profiles[0].upload_mbps == 120.0
    assert socket_profiles[0].download_mbps == 200.0


def test_apply_keeps_separate_socket_profiles_per_ip():
    """A node reachable on multiple IPs (LAN + Tailscale + link-local) gets one
    row per IP — they are NOT deduped down to a single socket profile that
    bounces between paths as the reconciler probes each IP in turn.
    """
    state = State()
    lan = SocketLinkProfile(
        sink_node_id=NODE_B,
        sink_ip="10.0.0.5",
        latency_ms=1.0,
        latency_jitter_ms=0.1,
        upload_mbps=1000.0,
        download_mbps=1000.0,
    )
    tailscale = SocketLinkProfile(
        sink_node_id=NODE_B,
        sink_ip="100.88.70.34",
        latency_ms=5.0,
        latency_jitter_ms=0.1,
        upload_mbps=400.0,
        download_mbps=400.0,
    )
    state = apply(state, _wrap(0, lan))
    state = apply(state, _wrap(1, tailscale))
    profiles = state.node_link_profiles[NODE_A][NODE_B]
    socket_profiles = [p for p in profiles if isinstance(p, NodeSocketLinkProfile)]
    assert len(socket_profiles) == 2
    by_ip = {p.sink_ip: p for p in socket_profiles}
    assert by_ip["10.0.0.5"].upload_mbps == 1000.0
    assert by_ip["100.88.70.34"].upload_mbps == 400.0


def test_apply_keeps_socket_and_rdma_profiles_to_same_peer():
    """A node may have two transports to the same peer (Wi-Fi + Thunderbolt)."""
    state = State()
    socket_p = SocketLinkProfile(
        sink_node_id=NODE_B,
        sink_ip="10.0.0.5",
        latency_ms=2.0,
        latency_jitter_ms=0.1,
        upload_mbps=400.0,
        download_mbps=900.0,
    )
    rdma_p = RDMALinkProfile(
        sink_node_id=NODE_B,
        source_rdma_iface="rdma_en2",
        sink_rdma_iface="rdma_en3",
        latency_ms=0.05,
        latency_jitter_ms=0.1,
        upload_mbps=20_000.0,
        download_mbps=18_000.0,
        payload_bytes=64 * 1024 * 1024,
    )
    state = apply(state, _wrap(0, socket_p))
    state = apply(state, _wrap(1, rdma_p))
    profiles = state.node_link_profiles[NODE_A][NODE_B]
    transports = sorted(p.transport for p in profiles)
    assert transports == ["rdma", "socket"]


def test_apply_node_timed_out_drops_profiles():
    state = State()
    state = apply(
        state,
        _wrap(0, GpuProfile(engine="mlx", tflops_fp16=1, memory_bandwidth_gbps=1)),
    )
    state = apply(
        state,
        _wrap(
            1,
            SocketLinkProfile(
                sink_node_id=NODE_B,
                sink_ip="10.0.0.5",
                latency_ms=1,
        latency_jitter_ms=0.1,
                upload_mbps=1,
                download_mbps=1,
            ),
        ),
    )

    timed_out = IndexedEvent(
        idx=2,
        event=NodeTimedOut(event_id=EventId(), node_id=NODE_A),
    )
    state = apply(state, timed_out)

    assert NODE_A not in state.node_gpu_profile
    assert NODE_A not in state.node_link_profiles


def test_apply_node_timed_out_drops_inverse_link_profiles():
    """Removing node B should also remove any profiles where B is the sink."""
    state = State()
    state = apply(
        state,
        _wrap(
            0,
            SocketLinkProfile(
                sink_node_id=NODE_B,
                sink_ip="10.0.0.5",
                latency_ms=1,
        latency_jitter_ms=0.1,
                upload_mbps=1,
                download_mbps=1,
            ),
        ),
    )

    timed_out = IndexedEvent(
        idx=1,
        event=NodeTimedOut(event_id=EventId(), node_id=NODE_B),
    )
    state = apply(state, timed_out)

    profiles_from_a = state.node_link_profiles.get(NODE_A, {})
    assert NODE_B not in profiles_from_a


def test_apply_uses_event_when_for_measured_at():
    state = State()
    when = "2026-04-30T12:00:00+00:00"
    profile = GpuProfile(engine="mlx", tflops_fp16=1.0, memory_bandwidth_gbps=1.0)
    state = apply(state, _wrap(0, profile, when=when))
    assert state.node_gpu_profile[NODE_A].measured_at == datetime.fromisoformat(when)
