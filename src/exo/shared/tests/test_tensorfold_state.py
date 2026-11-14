from exo.shared.apply import apply_node_gathered_info, apply_node_timed_out
from exo.shared.types.backends import Backend
from exo.shared.types.common import NodeId
from exo.shared.types.events import NodeGatheredInfo, NodeTimedOut
from exo.shared.types.state import State
from exo.shared.types.tensorfold import NativeCapabilities, ServeCapabilities
from exo.utils.info_gatherer.info_gatherer import NodeBackends


def capabilities() -> NativeCapabilities:
    return NativeCapabilities(
        schema=1,
        engine="zig",
        version="test",
        chip="nvidia-sm121",
        backends=["cuda"],
        families={"qwen3_5": ["mlx-q4g64"]},
        token_stream_versions=[1],
        serve=ServeCapabilities(flags={"--unix-socket": {}, "--name": {}}),
    )


def event(node: NodeId, native: NativeCapabilities | None) -> NodeGatheredInfo:
    return NodeGatheredInfo(
        node_id=node,
        when="2026-11-13T12:00:00+00:00",
        info=NodeBackends(
            backends=[Backend.MlxCpu, Backend.TensorFoldCuda]
            if native is not None
            else [Backend.MlxCpu],
            tensorfold=native,
        ),
    )


def test_native_capabilities_survive_event_and_state_serialization() -> None:
    node = NodeId("native")
    original = event(node, capabilities())
    decoded = NodeGatheredInfo.model_validate_json(original.model_dump_json())
    state = apply_node_gathered_info(decoded, State())
    assert state.node_tensorfold[node] == capabilities()
    assert Backend.TensorFoldCuda in state.node_backends[node]
    restored = State.model_validate_json(state.model_dump_json())
    assert restored.node_tensorfold == state.node_tensorfold
    assert restored.topology.contains_node(node)


def test_unavailable_binary_clears_previous_capabilities() -> None:
    node = NodeId("native")
    state = apply_node_gathered_info(event(node, capabilities()), State())
    state = apply_node_gathered_info(event(node, None), state)
    assert node not in state.node_tensorfold
    assert state.node_backends[node] == [Backend.MlxCpu]


def test_node_loss_removes_native_advertisement() -> None:
    node = NodeId("native")
    state = apply_node_gathered_info(event(node, capabilities()), State())
    state = apply_node_timed_out(NodeTimedOut(node_id=node), state)
    assert node not in state.node_tensorfold
    assert node not in state.node_backends


def test_old_node_events_and_snapshots_remain_readable() -> None:
    backends = NodeBackends.model_validate({"backends": [Backend.MlxCpu]})
    assert backends.tensorfold is None
    assert State.model_validate_json("{}").node_tensorfold == {}
