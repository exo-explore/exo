from ipaddress import ip_address

from exo.routing.connection_message import SocketAddress
from exo.shared.types.common import NodeId
from exo.shared.types.state import State
from exo.shared.types.topology import Connection


def test_state_serialization_roundtrip() -> None:
    """Verify that State → JSON → State round-trip preserves topology."""

    # --- build a simple state ------------------------------------------------
    node_a = NodeId("node-a")
    node_b = NodeId("node-b")

    connection = Connection(
        sink_id=node_a,
        source_id=node_b,
        sink_addr=SocketAddress(ip=ip_address("127.0.0.1"), port=5354, zone_id=None),
    )

    state = State()
    state.topology.add_connection(connection)

    json_repr = state.model_dump_json()
    restored_state = State.model_validate_json(json_repr)

    assert state.topology.to_snapshot() == restored_state.topology.to_snapshot()
    assert restored_state.model_dump_json() == json_repr
