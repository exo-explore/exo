from datetime import datetime, timezone

from exo.shared.apply import apply_node_timed_out
from exo.shared.topology import Topology
from exo.shared.types.backends import Backend
from exo.shared.types.common import NodeId
from exo.shared.types.events import NodeTimedOut
from exo.shared.types.profiling import NodeIdentity
from exo.shared.types.state import State


def test_apply_node_timed_out_removes_node_identity() -> None:
    timed_out_node = NodeId("timed-out-node")
    live_node = NodeId("live-node")
    topology = Topology()
    topology.add_node(timed_out_node)
    topology.add_node(live_node)
    live_identity = NodeIdentity(friendly_name="Live node")

    state = State(
        topology=topology,
        last_seen={
            timed_out_node: datetime.now(tz=timezone.utc),
            live_node: datetime.now(tz=timezone.utc),
        },
        node_identities={
            timed_out_node: NodeIdentity(friendly_name="Timed-out node"),
            live_node: live_identity,
        },
        node_backends={
            timed_out_node: [Backend.MlxMetal],
            live_node: [Backend.MlxCpu],
        },
    )

    new_state = apply_node_timed_out(NodeTimedOut(node_id=timed_out_node), state)

    assert set(new_state.topology.list_nodes()) == {live_node}
    assert new_state.last_seen.keys() == {live_node}
    assert new_state.node_identities == {live_node: live_identity}
    assert new_state.node_backends == {live_node: [Backend.MlxCpu]}
