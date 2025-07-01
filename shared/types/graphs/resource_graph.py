from collections.abc import Mapping

from pydantic import BaseModel

from shared.types.common import NodeId
from shared.types.networking.topology import Topology
from shared.types.profiling.common import NodeProfile
from shared.types.worker.common import NodeState


class ResourceGraph(BaseModel): ...


def get_graph_of_compute_resources(
    network_topology: Topology,
    node_states: Mapping[NodeId, NodeState],
    node_profiles: Mapping[NodeId, NodeProfile],
) -> ResourceGraph: ...
