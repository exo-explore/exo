from queue import Queue
from typing import Mapping, Sequence

from shared.topology import Topology
from shared.types.events import Event
from shared.types.state import CachePolicy
from shared.types.tasks import Task
from shared.types.worker.instances import InstanceId, InstanceParams


def get_instance_placements(
    inbox: Queue[Task],
    outbox: Queue[Task],
    topology: Topology,
    current_instances: Mapping[InstanceId, InstanceParams],
    cache_policy: CachePolicy,
) -> Mapping[InstanceId, InstanceParams]: ...



def get_transition_events(
    current_instances: Mapping[InstanceId, InstanceParams],
    target_instances: Mapping[InstanceId, InstanceParams],
) -> Sequence[Event]: ...
