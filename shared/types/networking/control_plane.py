from typing import TypeAlias

from shared.types.common import NewUUID, NodeId
from shared.types.graphs.common import Edge


class ControlPlaneEdgeId(NewUUID):
    pass


ControlPlaneEdgeType: TypeAlias = Edge[None, ControlPlaneEdgeId, NodeId]
