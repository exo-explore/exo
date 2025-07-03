from typing import Callable, NewType, Protocol, TypeVar

from shared.types.networking.control_plane import (
    ControlPlaneEdgeId,
    ControlPlaneEdgeType,
)

TopicName = NewType("TopicName", str)

MessageT = TypeVar("MessageT", bound=object)


PubSubMessageHandler = Callable[[TopicName, MessageT], None]
NodeConnectedHandler = Callable[
    [
        ControlPlaneEdgeId,
        ControlPlaneEdgeType,
    ],
    None,
]
NodeDisconnectedHandler = Callable[[ControlPlaneEdgeId], None]


class DiscoveryService(Protocol):
    def register_node_connected_handler(
        self, handler: NodeConnectedHandler
    ) -> None: ...
    def register_node_disconnected_handler(
        self, handler: NodeDisconnectedHandler
    ) -> None: ...


class PubSubService(Protocol):
    def register_handler(
        self, key: str, topic_name: TopicName, handler: PubSubMessageHandler[MessageT]
    ) -> None: ...
    def deregister_handler(self, key: str) -> None: ...
