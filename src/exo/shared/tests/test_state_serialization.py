from exo.shared.types.common import NodeId
from exo.shared.types.multiaddr import Multiaddr
from exo.shared.types.state import State
from exo.shared.types.topology import SocketConnection


def test_state_serialization_roundtrip() -> None:
    """Verify that State → JSON → State round-trip preserves topology."""

    # --- build a simple state ------------------------------------------------
    node_a = NodeId("node-a")
    node_b = NodeId("node-b")

    connection = SocketConnection(
        sink_multiaddr=Multiaddr(address="/ip4/127.0.0.1/tcp/10001"),
    )

    state = State()
    state.topology.add_connection(node_a, node_b, connection)

    json_repr = state.model_dump_json()
    restored_state = State.model_validate_json(json_repr)

    assert state.topology.to_snapshot() == restored_state.topology.to_snapshot()
    assert restored_state.model_dump_json() == json_repr
