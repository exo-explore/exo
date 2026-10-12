"""Each topic's priority must reach the networking layer."""

from typing import cast

from exo_rs import NetworkingHandle

import exo.routing.topics as topics
from exo.routing.router import Router
from exo.routing.topics import TypedTopic
from exo.utils.pydantic_ext import FrozenModel


class RecordingHandle:
    def __init__(self) -> None:
        self.high_priority: dict[str, bool] = {}

    async def gossipsub_subscribe(self, topic: str, high_priority: bool) -> bool:
        self.high_priority[topic] = high_priority
        return True


async def test_election_messages_are_high_priority() -> None:
    handle = RecordingHandle()
    router = Router(cast(NetworkingHandle, cast(object, handle)))

    async def subscribe[T: FrozenModel](topic: TypedTopic[T]) -> None:
        await router.register_topic(topic)
        await router._networking_subscribe(topic)  # pyright: ignore[reportPrivateUsage]

    await subscribe(topics.GLOBAL_EVENTS)
    await subscribe(topics.LOCAL_EVENTS)
    await subscribe(topics.COMMANDS)
    await subscribe(topics.ELECTION_MESSAGES)
    await subscribe(topics.DOWNLOAD_COMMANDS)

    # Election rounds are time-sensitive and must not queue behind event traffic
    assert handle.high_priority == {
        "global_events": False,
        "local_events": False,
        "commands": False,
        "election_messages": True,
        "download_commands": False,
    }
