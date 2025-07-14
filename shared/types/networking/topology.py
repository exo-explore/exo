from shared.graphs.networkx import NetworkXGraph
from shared.types.common import NodeId
from shared.types.networking.control_plane import ControlPlaneEdgeId
from shared.types.networking.data_plane import (
    DataPlaneEdgeData,
    DataPlaneEdgeId,
)
from shared.types.worker.common import NodeStatus


class DataPlaneTopology(
    NetworkXGraph[
        DataPlaneEdgeData,
        None,
        DataPlaneEdgeId,
        NodeId,
    ]
):
    pass


class OrphanedPartOfDataPlaneTopology(
    NetworkXGraph[
        DataPlaneEdgeData,
        None,
        DataPlaneEdgeId,
        NodeId,
    ]
):
    pass


class ControlPlaneTopology(NetworkXGraph[None, NodeStatus, ControlPlaneEdgeId, NodeId]):
    pass


class OrphanedPartOfControlPlaneTopology(
    NetworkXGraph[
        None,
        NodeStatus,
        ControlPlaneEdgeId,
        NodeId,
    ]
):
    pass
