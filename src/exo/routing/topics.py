from dataclasses import dataclass
from enum import Enum

from exo.routing.connection_message import ConnectionMessage
from exo.shared.election import ElectionMessage
from exo.shared.types.commands import ForwarderCommand, ForwarderDownloadCommand
from exo.shared.types.events import (
    GlobalForwarderEvent,
    LocalForwarderEvent,
    StateSnapshot,
)
from exo.utils.pydantic_ext import FrozenModel


class PublishPolicy(str, Enum):
    Never = "Never"
    """Never publish to the network - this is a local message"""
    Minimal = "Minimal"
    """Only publish when there is no local receiver for this type of message"""
    Always = "Always"
    """Always publish to the network"""


@dataclass  # (frozen=True)
class TypedTopic[T: FrozenModel]:
    topic: str
    publish_policy: PublishPolicy

    model_type: type[
        T
    ]  # This can be worked around with evil type hacking, see https://stackoverflow.com/a/71720366 - I don't think it's necessary here.

    # Send ahead of normal traffic, e.g. so election messages don't queue behind events
    high_priority: bool = False

    @staticmethod
    def serialize(t: T) -> bytes:
        return t.model_dump_json().encode("utf-8")

    def deserialize(self, b: bytes) -> T:
        return self.model_type.model_validate_json(b.decode("utf-8"))


GLOBAL_EVENTS = TypedTopic("global_events", PublishPolicy.Always, GlobalForwarderEvent)
LOCAL_EVENTS = TypedTopic("local_events", PublishPolicy.Always, LocalForwarderEvent)
STATE_SNAPSHOTS = TypedTopic("state_snapshots", PublishPolicy.Always, StateSnapshot)
COMMANDS = TypedTopic("commands", PublishPolicy.Always, ForwarderCommand)
ELECTION_MESSAGES = TypedTopic(
    "election_messages", PublishPolicy.Always, ElectionMessage, high_priority=True
)
CONNECTION_MESSAGES = TypedTopic(
    "connection_messages", PublishPolicy.Never, ConnectionMessage
)
DOWNLOAD_COMMANDS = TypedTopic(
    "download_commands", PublishPolicy.Always, ForwarderDownloadCommand
)
