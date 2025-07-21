from queue import Queue
from typing import Mapping, Sequence

from shared.types.events.common import BaseEvent, EventCategory
from shared.types.graphs.topology import Topology
from shared.types.states.master import CachePolicy, CachePolicyType
from shared.types.tasks.common import Task
from shared.types.worker.instances import InstanceId, InstanceParams


def get_instance_placement(
    inbox: Queue[Task],
    outbox: Queue[Task],
    topology: Topology,
    current_instances: Mapping[InstanceId, InstanceParams],
    cache_policy: CachePolicy[CachePolicyType],
) -> Mapping[InstanceId, InstanceParams]: ...


def get_transition_events(
    current_instances: Mapping[InstanceId, InstanceParams],
    target_instances: Mapping[InstanceId, InstanceParams],
) -> Sequence[BaseEvent[EventCategory]]: ...
