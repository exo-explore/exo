"""The master doesn't send a request to an instance with a device it can't reach now, as its own
membership sees it: the backstop for an API whose view differs (every API checks the same before
sending). A request whose model has only such instances ends at once. Nor does it launch a model
on such a device."""

import anyio

from exo.master.tests.test_requests_to_deleted_instances import launch_of
from exo.master.tests.test_silent_nodes import (
    MASTER,
    PEER_A,
    PEER_B,
    instance_on,
    listed,
    master_with,
    membership,
)
from exo.shared.request_errors import DEVICE_UNREACHABLE
from exo.shared.types.chunks import ErrorChunk
from exo.shared.types.commands import ForwarderCommand, TextGeneration
from exo.shared.types.common import NodeId, SystemId
from exo.shared.types.events import (
    ChunkGenerated,
    Event,
    InstanceCreated,
    TaskCreated,
)
from exo.shared.types.memory import Memory
from exo.shared.types.profiling import MemoryUsage
from exo.shared.types.text_generation import (
    InputMessage,
    InputMessageContent,
    TextGenerationTaskParams,
)
from exo.shared.types.worker.instances import MlxRingInstance
from exo.shared.types.worker.runners import RunnerId, RunnerReady
from exo.utils.channels import channel


def ready(*instances: MlxRingInstance) -> dict[RunnerId, RunnerReady]:
    return {
        runner: RunnerReady()
        for instance in instances
        for runner in instance.shard_assignments.runner_to_shard
    }


async def first_event_for_a_request(
    instances: tuple[MlxRingInstance, ...], unreachable: tuple[NodeId, ...]
) -> Event:
    """What the master answers a request for the instances' model with, when its membership
    can't reach `unreachable`"""
    view = membership("Majority", alive=(PEER_A, PEER_B))
    master, broadcast = master_with(
        listed(MASTER, PEER_A, PEER_B), *instances, membership=view
    )
    for node in unreachable:
        await view.lose(node)
    master.state = master.state.model_copy(
        update={"runners": ready(*instances), "tasks": {}}
    )
    command_sender, commands = channel[ForwarderCommand]()
    master.command_receiver = commands
    request = TextGeneration(
        task_params=TextGenerationTaskParams(
            model=instances[0].shard_assignments.model_id,
            input=[InputMessage(role="user", content=InputMessageContent("hi"))],
        )
    )
    answers: list[Event] = []
    async with anyio.create_task_group() as tg:
        tg.start_soon(master._command_processor)  # pyright: ignore[reportPrivateUsage]
        await command_sender.send(ForwarderCommand(origin=SystemId(), command=request))
        with anyio.fail_after(5):
            answers.append((await broadcast.receive()).event)
        tg.cancel_scope.cancel()
    return answers[0]


async def test_a_request_for_a_model_it_cant_reach_ends_at_once() -> None:
    event = await first_event_for_a_request((instance_on(PEER_A),), (PEER_A,))

    assert not isinstance(event, TaskCreated)
    assert isinstance(event, ChunkGenerated)
    assert isinstance(event.chunk, ErrorChunk)
    assert event.chunk.error_message == DEVICE_UNREACHABLE


async def test_a_request_goes_to_an_instance_it_can_reach() -> None:
    unreachable, reachable = instance_on(PEER_A), instance_on(PEER_B)

    event = await first_event_for_a_request((unreachable, reachable), (PEER_A,))

    assert isinstance(event, TaskCreated)
    assert event.task.instance_id == reachable.instance_id


async def test_a_launch_onto_a_node_it_cant_reach_is_refused() -> None:
    """A launch with its nodes given (a placement) onto a node this master can't reach now:
    its model couldn't load there, and the instance would end once the node is removed.
    Refused, as a placement leaves such a node out (the API sends a launch again until it
    shows). One onto the nodes it reaches is launched."""
    view = membership("Majority", alive=(PEER_A, PEER_B))
    master, broadcast = master_with(listed(MASTER, PEER_A, PEER_B), membership=view)
    memory = MemoryUsage(
        ram_total=Memory.from_gb(64),
        ram_available=Memory.from_gb(64),
        swap_total=Memory(),
        swap_available=Memory(),
    )
    master.state = master.state.model_copy(
        update={"node_memory": {PEER_A: memory, PEER_B: memory}}
    )
    await view.lose(PEER_A)
    unreachable, reachable = instance_on(PEER_A), instance_on(PEER_B)
    command_sender, commands = channel[ForwarderCommand]()
    master.command_receiver = commands
    launched: list[Event] = []
    async with anyio.create_task_group() as tg:
        tg.start_soon(master._command_processor)  # pyright: ignore[reportPrivateUsage]
        for instance in (unreachable, reachable):
            await command_sender.send(
                ForwarderCommand(origin=SystemId(), command=launch_of(instance))
            )
        with anyio.fail_after(5):
            while not any(isinstance(event, InstanceCreated) for event in launched):
                launched.append((await broadcast.receive()).event)
        tg.cancel_scope.cancel()

    (event,) = [event for event in launched if isinstance(event, InstanceCreated)]
    assert event.instance.instance_id == reachable.instance_id
    assert unreachable.instance_id not in master.state.instances
    assert len(master.state.deployments) == 1
