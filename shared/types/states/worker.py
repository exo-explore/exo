from collections.abc import Mapping
from typing import Literal

from shared.types.common import NodeId
from shared.types.events.common import (
    EventCategoryEnum,
    State,
)
from shared.types.states.shared import SharedState
from shared.types.worker.common import NodeStatus


class NodeStatusState(State[EventCategoryEnum.MutatesRunnerStatus]):
    event_category: Literal[EventCategoryEnum.MutatesRunnerStatus] = (
        EventCategoryEnum.MutatesRunnerStatus
    )
    node_status: Mapping[NodeId, NodeStatus]


class WorkerState(SharedState):
    node_status: NodeStatusState
