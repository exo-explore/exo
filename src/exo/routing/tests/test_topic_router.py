"""Messages from the network are only parsed when something on this node receives them."""


from exo.routing.router import TopicRouter
from exo.routing.topics import PublishPolicy, TypedTopic
from exo.utils.channels import channel
from exo.utils.pydantic_ext import FrozenModel


class Ping(FrozenModel):
    n: int


PINGS = TypedTopic("pings", PublishPolicy.Always, Ping)


def make_router() -> TopicRouter[Ping]:
    networking_sender, _ = channel[tuple[str, bytes]]()
    return TopicRouter[Ping](PINGS, networking_sender)


async def test_message_is_delivered_to_receivers() -> None:
    router = make_router()
    send, recv = channel[Ping]()
    router.senders.add(send)

    await router.publish_bytes(PINGS.serialize(Ping(n=1)))

    assert recv.collect() == [Ping(n=1)]


async def test_message_nobody_receives_is_not_parsed() -> None:
    router = make_router()

    # Would fail validation if it were parsed
    await router.publish_bytes(b"not a ping")


async def test_an_unreadable_message_is_dropped_even_when_someone_receives() -> None:
    router = make_router()
    send, recv = channel[Ping]()
    router.senders.add(send)

    await router.publish_bytes(b"not a ping")

    assert recv.collect() == []
