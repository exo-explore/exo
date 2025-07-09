from shared.types.common import NodeId
from shared.types.graphs.common import Graph, GraphData
from shared.types.networking.control_plane import ControlPlaneEdgeId
from shared.types.networking.data_plane import (
    DataPlaneEdgeData,
    DataPlaneEdgeId,
)
from shared.types.worker.common import NodeStatus


class DataPlaneTopology(
    Graph[
        DataPlaneEdgeData,
        None,
        DataPlaneEdgeId,
        NodeId,
    ]
):
    graph_data: GraphData[
        DataPlaneEdgeData,
        None,
        DataPlaneEdgeId,
        NodeId,
    ]


class OrphanedPartOfDataPlaneTopology(
    Graph[
        DataPlaneEdgeData,
        None,
        DataPlaneEdgeId,
        NodeId,
    ]
):
    graph_data: GraphData[
        DataPlaneEdgeData,
        None,
        DataPlaneEdgeId,
        NodeId,
    ]


class ControlPlaneTopology(
    Graph[
        None,
        NodeStatus,
        ControlPlaneEdgeId,
        NodeId,
    ]
):
    graph_data: GraphData[
        None,
        NodeStatus,
        ControlPlaneEdgeId,
        NodeId,
    ]


class OrphanedPartOfControlPlaneTopology(
    Graph[
        None,
        NodeStatus,
        ControlPlaneEdgeId,
        NodeId,
    ]
):
    graph_data: GraphData[
        None,
        NodeStatus,
        ControlPlaneEdgeId,
        NodeId,
    ]
