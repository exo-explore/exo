import anyio
import pytest

from exo.routing import topics
from exo.routing.transient_router import TransientRouter
from exo.shared.types.common import NodeId, SessionId
from exo.shared.types.events import (
    GlobalForwarderTransientEvent,
    TaskAcknowledged,
)
from exo.shared.types.tasks import TaskId
from exo.utils.channels import channel


def test_transient_topic_round_trips_wrapped_event() -> None:
    wrapped = GlobalForwarderTransientEvent(
        origin=NodeId("worker"),
        session=SessionId(master_node_id=NodeId("master"), election_clock=1),
        event=TaskAcknowledged(task_id=TaskId("task-1")),
    )

    restored = topics.TRANSIENT_EVENTS.deserialize(
        topics.TRANSIENT_EVENTS.serialize(wrapped)
    )

    assert restored == wrapped


@pytest.mark.asyncio
async def test_transient_router_publishes_and_dispatches_session_events() -> None:
    node_id = NodeId("worker")
    session_id = SessionId(master_node_id=NodeId("master"), election_clock=1)
    external_outbound_sender, external_outbound_receiver = channel[
        GlobalForwarderTransientEvent
    ]()
    external_inbound_sender, external_inbound_receiver = channel[
        GlobalForwarderTransientEvent
    ]()
    router = TransientRouter(
        node_id=node_id,
        session_id=session_id,
        external_outbound=external_outbound_sender,
        external_inbound=external_inbound_receiver,
    )
    local_sender = router.sender()
    local_receiver = router.receiver()
    event = TaskAcknowledged(task_id=TaskId("task-1"))

    async with anyio.create_task_group() as tg:
        tg.start_soon(router.run)
        await local_sender.send(event)

        wrapped = await external_outbound_receiver.receive()
        assert wrapped.origin == node_id
        assert wrapped.session == session_id
        assert wrapped.event == event

        await external_inbound_sender.send(wrapped)
        assert await local_receiver.receive() == event

        stale_session = SessionId(master_node_id=NodeId("master"), election_clock=2)
        await external_inbound_sender.send(
            wrapped.model_copy(update={"session": stale_session})
        )
        await anyio.sleep(0)
        assert local_receiver.collect() == []

        router.shutdown()
        tg.cancel_scope.cancel()
