from collections.abc import Mapping

from shared.types.common import NodeId
from shared.types.events.common import (
    EventCategories,
    State,
)
from shared.types.states.shared import SharedState
from shared.types.worker.common import NodeStatus


class NodeStatusState(State[EventCategories.ControlPlaneEventTypes]):
    node_status: Mapping[NodeId, NodeStatus]


class WorkerState(SharedState):
    node_status: NodeStatusState
