from shared.types.common import NodeId
from shared.types.graphs.common import Graph, GraphData
from shared.types.networking.edges import (
    AddressingProtocol,
    ApplicationProtocol,
    EdgeId,
    NetworkEdge,
)


class Topology(
    Graph[
        NetworkEdge[AddressingProtocol, ApplicationProtocol],
        None,
        EdgeId,
        NodeId,
    ]
):
    graph_data: GraphData[
        NetworkEdge[AddressingProtocol, ApplicationProtocol],
        None,
        EdgeId,
        NodeId,
    ]


class OrphanedPartOfTopology(
    Graph[
        NetworkEdge[AddressingProtocol, ApplicationProtocol],
        None,
        EdgeId,
        NodeId,
    ]
):
    graph_data: GraphData[
        NetworkEdge[AddressingProtocol, ApplicationProtocol],
        None,
        EdgeId,
        NodeId,
    ]
