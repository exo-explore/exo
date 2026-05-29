from typing import cast

import anyio
import pytest
from exo_pyo3_bindings import NetworkingHandle, PyFromSwarm

from exo.routing.router import Router
from exo.routing.topics import LOCAL_EVENTS
from exo.shared.types.common import NodeId, SessionId, SystemId
from exo.shared.types.events import LocalForwarderEvent, TestEvent


def _good_event() -> LocalForwarderEvent:
    return LocalForwarderEvent(
        origin_idx=0,
        origin=SystemId("system-under-test"),
        session=SessionId(master_node_id=NodeId("master"), election_clock=0),
        event=TestEvent(),
    )


class _FakeNet:
    """Yields a fixed sequence of swarm messages, then blocks forever.

    Stands in for NetworkingHandle so we can drive _networking_recv directly.
    """

    def __init__(self, messages: list[PyFromSwarm]):
        self._messages = messages
        self._index = 0

    async def recv(self) -> PyFromSwarm:
        if self._index < len(self._messages):
            message = self._messages[self._index]
            self._index += 1
            return message
        await anyio.sleep_forever()
        raise AssertionError("unreachable")


@pytest.mark.asyncio
async def test_undeserializable_message_is_dropped_not_fatal():
    """An undeserializable gossip message (e.g. a peer on an incompatible
    version) must be dropped, leaving the receive loop alive to process
    subsequent valid messages — not crash the whole node."""
    good = _good_event()
    fake_net = _FakeNet(
        [
            PyFromSwarm.Message("peer-on-old-version", LOCAL_EVENTS.topic, b"{}"),
            PyFromSwarm.Message(
                "peer-on-this-version", LOCAL_EVENTS.topic, LOCAL_EVENTS.serialize(good)
            ),
        ]
    )

    router = Router(cast(NetworkingHandle, cast(object, fake_net)))
    await router.register_topic(LOCAL_EVENTS)
    receiver = router.receiver(LOCAL_EVENTS)

    async with anyio.create_task_group() as task_group:
        task_group.start_soon(router._networking_recv)  # pyright: ignore[reportPrivateUsage]
        with anyio.fail_after(5):
            received = await receiver.receive()
        assert received == good
        task_group.cancel_scope.cancel()
