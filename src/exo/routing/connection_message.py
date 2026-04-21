from exo_pyo3_bindings import PyFromSwarm

from exo.shared.types.common import NodeId
from exo.utils.pydantic_ext import FrozenModel

"""Serialisable types for Connection Updates/Messages"""


class ConnectionMessage(FrozenModel):
    node_id: NodeId
    connected: bool

    @classmethod
    def from_update(cls, update: PyFromSwarm.Connection) -> "ConnectionMessage":
        return cls(node_id=NodeId(update.peer_id), connected=update.connected)
