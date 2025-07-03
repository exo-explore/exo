from collections.abc import Mapping

from pydantic import BaseModel

from shared.types.common import NodeId
from shared.types.networking.topology import ControlPlaneTopology, DataPlaneTopology
from shared.types.profiling.common import NodePerformanceProfile


class ResourceGraph(BaseModel): ...


def get_graph_of_compute_resources(
    control_plane_topology: ControlPlaneTopology,
    data_plane_topology: DataPlaneTopology,
    node_profiles: Mapping[NodeId, NodePerformanceProfile],
) -> ResourceGraph: ...
