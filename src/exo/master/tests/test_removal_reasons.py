# pyright: reportPrivateUsage=false
"""Every removal of an instance says why: a user stopped it, a node was lost, or its runner kept
failing to start.

DeleteInstance used to stand for the first and the last. A worker that gave up on an instance sent the
same command as a user who stopped it, so nothing that came after could tell them apart: the
API told the instance's requests "The model was stopped", and anything that should start a
lost model again would have had to guess whether the user wanted it.
"""

from exo.master.tests.test_requests_to_deleted_instances import (
    LOST_NODE,
    NODE,
    chat_request,
    instance_on,
    master_with,
    process,
)
from exo.routing.topics import COMMANDS, GLOBAL_EVENTS
from exo.shared.types.commands import (
    DeleteInstance,
    ForwarderCommand,
    InstanceBroken,
)
from exo.shared.types.common import NodeId, SessionId, SystemId
from exo.shared.types.events import (
    Event,
    GlobalForwarderEvent,
    InstanceDeleted,
)
from exo.shared.types.worker.instances import InstanceId, InstanceRemovalReason


def removals(output: list[Event]) -> list[tuple[InstanceId, InstanceRemovalReason]]:
    return [
        (event.instance_id, event.reason)
        for event in output
        if isinstance(event, InstanceDeleted)
    ]


async def test_a_user_stop_is_a_user_stop() -> None:
    instance = instance_on(NODE)
    master, events = master_with(instance)

    output = await process(
        master, events, DeleteInstance(instance_id=instance.instance_id), chat_request()
    )

    assert removals(output) == [(instance.instance_id, "UserStopped")]


async def test_a_worker_that_gave_up_says_the_model_failed_to_start() -> None:
    instance = instance_on(NODE)
    master, events = master_with(instance)

    output = await process(
        master,
        events,
        InstanceBroken(
            instance_id=instance.instance_id,
            reason="StartFailed",
            message="its runner here failed to start 3 times",
        ),
        chat_request(),
    )

    assert removals(output) == [(instance.instance_id, "StartFailed")]


async def test_a_late_report_on_an_instance_already_gone_changes_nothing() -> None:
    """A worker asks again while its first request is on the way, or after the instance was
    removed for another reason: none of that may remove anything else."""
    other = instance_on(NODE)
    master, events = master_with(other)

    output = await process(
        master,
        events,
        InstanceBroken(instance_id=InstanceId(), reason="StartFailed"),
        DeleteInstance(instance_id=InstanceId()),
        chat_request(),
    )

    assert removals(output) == []


async def test_an_instance_removed_for_a_lost_node_is_not_removed_again() -> None:
    instance = instance_on(LOST_NODE)
    master, events = master_with(instance)
    master._remove_silent_nodes_and_broken_instances()
    lost = (await events.receive()).event

    output = await process(
        master,
        events,
        InstanceBroken(instance_id=instance.instance_id, reason="StartFailed"),
        chat_request(),
    )

    assert lost == InstanceDeleted(
        instance_id=instance.instance_id, reason="NodeLost", event_id=lost.event_id
    )
    assert removals(output) == []


def test_the_reasons_cross_the_network() -> None:
    """Nodes read events and commands as JSON from the network topics."""
    master = NodeId("master")
    for reason in ("NodeLost", "UserStopped", "StartFailed"):
        event = GlobalForwarderEvent(
            origin_idx=0,
            origin=master,
            session=SessionId(master_node_id=master, election_clock=1),
            event=InstanceDeleted(instance_id=InstanceId(), reason=reason),
        )
        assert GLOBAL_EVENTS.deserialize(GLOBAL_EVENTS.serialize(event)) == event
    command = ForwarderCommand(
        origin=SystemId(),
        command=InstanceBroken(
            instance_id=InstanceId(), reason="StartFailed", message="why"
        ),
    )
    assert COMMANDS.deserialize(COMMANDS.serialize(command)) == command
