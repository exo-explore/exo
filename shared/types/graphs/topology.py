from pydantic import BaseModel, IPvAnyAddress

from shared.graphs import Graph
from shared.types.common import NewUUID, NodeId
from shared.types.profiling.common import NodePerformanceProfile


class TopologyEdgeId(NewUUID):
    pass


class TopologyEdgeProfile(BaseModel):
    throughput: float
    latency: float
    jitter: float


class TopologyEdge(BaseModel):
    source_ip: IPvAnyAddress
    sink_ip: IPvAnyAddress
    edge_profile: TopologyEdgeProfile
    

class TopologyNode(BaseModel):
    node_id: NodeId
    node_profile: NodePerformanceProfile


class Topology(
    Graph[
        TopologyEdge,
        TopologyNode,
        TopologyEdgeId,
        NodeId,
    ]
):
    pass


class OrphanedPartOfTopology(
    Graph[
        TopologyEdge,
        TopologyNode,
        TopologyEdgeId,
        NodeId,
    ]
):
    pass
