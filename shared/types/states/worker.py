from collections.abc import Mapping

from shared.types.common import NodeId
from shared.types.events.common import (
    NodeStatusEventTypes,
    State,
)
from shared.types.states.shared import SharedState
from shared.types.worker.common import NodeStatus


class NodeStatusState(State[NodeStatusEventTypes]):
    node_status: Mapping[NodeId, NodeStatus]


class WorkerState(SharedState):
    node_status: NodeStatusState
