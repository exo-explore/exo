from enum import Enum

from exo_pyo3_bindings import ConnectionUpdate, ConnectionUpdateType

from exo.shared.types.common import NodeId
from exo.utils.pydantic_ext import CamelCaseModel

"""Serialisable types for Connection Updates/Messages"""


class ConnectionMessageType(Enum):
    Connected = 0
    Disconnected = 1

    @staticmethod
    def from_update_type(update_type: ConnectionUpdateType):
        match update_type:
            case ConnectionUpdateType.Connected:
                return ConnectionMessageType.Connected
            case ConnectionUpdateType.Disconnected:
                return ConnectionMessageType.Disconnected


class ConnectionMessage(CamelCaseModel):
    node_id: NodeId
    connection_type: ConnectionMessageType
    remote_ipv4: str
    remote_tcp_port: int

    @classmethod
    def from_update(cls, update: ConnectionUpdate) -> "ConnectionMessage":
        return cls(
            node_id=NodeId(update.peer_id.to_base58()),
            connection_type=ConnectionMessageType.from_update_type(update.update_type),
            remote_ipv4=update.remote_ipv4,
            remote_tcp_port=update.remote_tcp_port,
        )
