from exo.shared.apply import apply_node_timed_out
from exo.shared.types.common import NodeId
from exo.shared.types.events import NodeTimedOut
from exo.shared.types.profiling import NodeIdentity
from exo.shared.types.state import State


def test_a_timed_out_node_is_forgotten_with_its_identity():
    """A node that times out leaves no identity behind: anything counting identities (the macOS app's node
    count) otherwise kept counting it, and a restarted node, which comes back with a new id, was counted twice."""
    gone, staying = NodeId("gone"), NodeId("staying")
    state = State(
        node_identities={
            gone: NodeIdentity(friendly_name="Mac mini"),
            staying: NodeIdentity(friendly_name="Mac Studio"),
        }
    )

    new_state = apply_node_timed_out(NodeTimedOut(node_id=gone), state)

    assert dict(new_state.node_identities) == {
        staying: NodeIdentity(friendly_name="Mac Studio")
    }
