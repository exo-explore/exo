from typing import Annotated, Callable, NewType, Protocol

from pydantic import BaseModel, Field

from shared.types.common import NodeId
from shared.types.networking.edges import (
    AddressingProtocol,
    ApplicationProtocol,
    EdgeDirection,
    EdgeId,
    EdgeInfo,
)

TopicName = NewType("TopicName", str)


class WrappedMessage(BaseModel):
    node_id: NodeId
    unix_timestamp: Annotated[int, Field(gt=0)]


PubSubMessageHandler = Callable[[TopicName, WrappedMessage], None]
NodeConnectedHandler = Callable[
    [EdgeId, EdgeDirection, EdgeInfo[AddressingProtocol, ApplicationProtocol]], None
]
NodeDisconnectedHandler = Callable[[EdgeId], None]


class DiscoveryService(Protocol):
    def register_node_connected_handler(
        self, handler: NodeConnectedHandler
    ) -> None: ...
    def register_node_disconnected_handler(
        self, handler: NodeDisconnectedHandler
    ) -> None: ...


class PubSubService(Protocol):
    def register_handler(
        self, key: str, topic_name: TopicName, handler: PubSubMessageHandler
    ) -> None: ...
    def deregister_handler(self, key: str) -> None: ...
