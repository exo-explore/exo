"""A message a node can't read is dropped instead of stopping the node's receive loop."""

import pytest

from exo.routing.router import TopicRouter
from exo.routing.topics import LOCAL_EVENTS
from exo.shared.types.events import LocalForwarderEvent
from exo.utils.channels import Receiver, channel


def router() -> tuple[TopicRouter[LocalForwarderEvent], Receiver[LocalForwarderEvent]]:
    networking_sender, _ = channel[tuple[str, bytes]]()
    topic_router = TopicRouter[LocalForwarderEvent](LOCAL_EVENTS, networking_sender)
    subscriber, received = channel[LocalForwarderEvent]()
    topic_router.senders.add(subscriber)
    return topic_router, received


@pytest.mark.parametrize(
    "data",
    [b"not json", b'{"origin": "x"}', "\xff".encode("latin-1")],
)
async def test_an_unreadable_message_is_dropped(data: bytes) -> None:
    topic_router, received = router()

    await topic_router.publish_bytes(data)

    assert received.collect() == []
