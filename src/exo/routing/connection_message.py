from exo_core.models import CamelCaseModel
from exo_core.types.common import NodeId
from exo_pyo3_bindings import PyFromSwarm

"""Serialisable types for Connection Updates/Messages"""


class ConnectionMessage(CamelCaseModel):
    node_id: NodeId
    connected: bool

    @classmethod
    def from_update(cls, update: PyFromSwarm.Connection) -> "ConnectionMessage":
        return cls(node_id=NodeId(update.peer_id), connected=update.connected)
