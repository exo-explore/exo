from typing import Callable, NewType, Protocol

from shared.types.networking.control_plane import (
    ControlPlaneEdgeId,
    ControlPlaneEdgeType,
)

TopicName = NewType("TopicName", str)

PubSubMessageHandler = Callable[[TopicName, object], None]
NodeConnectedHandler = Callable[
    [
        ControlPlaneEdgeId,
        ControlPlaneEdgeType,
    ],
    None,
]
NodeDisconnectedHandler = Callable[[ControlPlaneEdgeId], None]


class DiscoveryService(Protocol):
    def on_node_connected(self, handler: NodeConnectedHandler) -> None: ...
    def on_node_disconnected(self, handler: NodeDisconnectedHandler) -> None: ...


class PubSubService(Protocol):
    def on_message_received(
        self, topic_name: TopicName, handler: PubSubMessageHandler
    ) -> None: ...
