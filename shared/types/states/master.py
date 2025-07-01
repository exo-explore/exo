from queue import Queue

from pydantic import BaseModel

from shared.types.common import NodeId
from shared.types.networking.topology import NetworkState
from shared.types.profiling.common import NodeProfile
from shared.types.states.shared import SharedState


class ExternalCommand(BaseModel): ...


class MasterState(SharedState):
    network_state: NetworkState
    node_profiles: dict[NodeId, NodeProfile]
    job_inbox: Queue[ExternalCommand]
    job_outbox: Queue[ExternalCommand]
