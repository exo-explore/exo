from shared.types.common import NodeId
from shared.types.graphs.common import Graph, GraphData
from shared.types.networking.control_plane import ControlPlaneEdgeId
from shared.types.networking.data_plane import (
    AddressingProtocol,
    ApplicationProtocol,
    DataPlaneEdge,
    DataPlaneEdgeId,
)
from shared.types.worker.common import NodeStatus


class DataPlaneTopology(
    Graph[
        DataPlaneEdge[AddressingProtocol, ApplicationProtocol],
        None,
        DataPlaneEdgeId,
        NodeId,
    ]
):
    graph_data: GraphData[
        DataPlaneEdge[AddressingProtocol, ApplicationProtocol],
        None,
        DataPlaneEdgeId,
        NodeId,
    ]


class OrphanedPartOfDataPlaneTopology(
    Graph[
        DataPlaneEdge[AddressingProtocol, ApplicationProtocol],
        None,
        DataPlaneEdgeId,
        NodeId,
    ]
):
    graph_data: GraphData[
        DataPlaneEdge[AddressingProtocol, ApplicationProtocol],
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
