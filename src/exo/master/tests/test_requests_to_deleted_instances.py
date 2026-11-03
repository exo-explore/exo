"""The master never sends a request to an instance it has deleted.

The master's state used to change only when its own events came back indexed, a few
milliseconds after it sent them. A request that reached it in that window, just after the
instance was deleted (by a DeleteInstance, or because a node was lost), got a task on the
deleted instance. Nothing ran that task and nothing ended its request: the stream got
keepalives until the client gave up. The master now applies its own events as it decides on
them, so the next command it handles sees the deletion.
"""

from datetime import datetime, timezone

import anyio

from exo.master.main import Master
from exo.routing.router import get_node_zid
from exo.shared.models.model_cards import ModelCard, ModelId, ModelTask
from exo.shared.topology import Topology
from exo.shared.types.backends import Backend
from exo.shared.types.chunks import ErrorChunk
from exo.shared.types.commands import (
    Command,
    DeleteInstance,
    ForwarderCommand,
    ForwarderDownloadCommand,
    TextGeneration,
)
from exo.shared.types.common import NodeId, SessionId, SystemId
from exo.shared.types.events import (
    ChunkGenerated,
    Event,
    GlobalForwarderEvent,
    InstanceDeleted,
    LocalForwarderEvent,
    StateSnapshot,
    TaskCreated,
    TaskStatusUpdated,
)
from exo.shared.types.memory import Memory
from exo.shared.types.state import State
from exo.shared.types.tasks import TaskStatus
from exo.shared.types.text_generation import (
    InputMessage,
    InputMessageContent,
    TextGenerationTaskParams,
)
from exo.shared.types.worker.instances import InstanceId, MlxRingInstance
from exo.shared.types.worker.runners import (
    RunnerId,
    RunnerLoading,
    RunnerReady,
    ShardAssignments,
)
from exo.shared.types.worker.shards import PipelineShardMetadata
from exo.utils.channels import Receiver, channel

MODEL = ModelId("test-model")
NODE = NodeId("node")
LOST_NODE = NodeId("lost-node")


def instance_on(node: NodeId) -> MlxRingInstance:
    card = ModelCard(
        model_id=MODEL,
        storage_size=Memory.from_mb(100),
        n_layers=1,
        hidden_size=64,
        supports_tensor=False,
        tasks=[ModelTask.TextGeneration],
        backends=[Backend.MlxMetal],
    )
    runner = RunnerId()
    return MlxRingInstance(
        instance_id=InstanceId(),
        shard_assignments=ShardAssignments(
            model_id=MODEL,
            runner_to_shard={
                runner: PipelineShardMetadata(
                    model_card=card,
                    device_rank=0,
                    world_size=1,
                    start_layer=0,
                    end_layer=1,
                    n_layers=1,
                )
            },
            node_to_runner={node: runner},
        ),
        hosts_by_node={},
        ephemeral_port=50000,
    )


def chat_request() -> TextGeneration:
    return TextGeneration(
        task_params=TextGenerationTaskParams(
            model=MODEL,
            input=[InputMessage(role="user", content=InputMessageContent("hi"))],
        )
    )


def master_with(
    *instances: MlxRingInstance,
) -> tuple[Master, Receiver[GlobalForwarderEvent]]:
    """A master with the instances, ready, on a node it has just heard from, and what it
    sends to every node."""
    node_id = get_node_zid()
    global_event_sender, broadcast = channel[GlobalForwarderEvent]()
    topology = Topology()
    topology.add_node(NODE)
    master = Master(
        node_id,
        SessionId(master_node_id=node_id, election_clock=0),
        command_receiver=channel[ForwarderCommand]()[1],
        local_event_receiver=channel[LocalForwarderEvent]()[1],
        global_event_sender=global_event_sender,
        snapshot_sender=channel[StateSnapshot]()[0],
        download_command_sender=channel[ForwarderDownloadCommand]()[0],
        initial_state=State(
            topology=topology,
            last_seen={NODE: datetime.now(tz=timezone.utc)},
            instances={instance.instance_id: instance for instance in instances},
            runners={
                runner: RunnerReady()
                for instance in instances
                for runner in instance.shard_assignments.runner_to_shard
            },
        ),
    )
    return master, broadcast


async def process(
    master: Master,
    broadcast: Receiver[GlobalForwarderEvent],
    *sent: Command,
    until: tuple[type[Event], ...] = (TaskCreated, ChunkGenerated),
) -> list[Event]:
    """Have the master process the commands back to back, and return what it sent until an
    event of the given types."""
    commands_sender, commands = channel[ForwarderCommand]()
    master.command_receiver = commands
    output: list[Event] = []
    async with anyio.create_task_group() as tg:
        tg.start_soon(master._command_processor)  # pyright: ignore[reportPrivateUsage]
        for command in sent:
            await commands_sender.send(
                ForwarderCommand(origin=SystemId(), command=command)
            )
        with anyio.fail_after(5):
            while not any(isinstance(event, until) for event in output):
                output.append((await broadcast.receive()).event)
        tg.cancel_scope.cancel()
    return output


async def test_a_request_right_after_its_instance_was_deleted_ends_with_an_error() -> (
    None
):
    instance = instance_on(NODE)
    master, broadcast = master_with(instance)
    request = chat_request()

    output = await process(
        master, broadcast, DeleteInstance(instance_id=instance.instance_id), request
    )

    assert [
        event.instance_id for event in output if isinstance(event, InstanceDeleted)
    ] == [instance.instance_id]
    assert not [event for event in output if isinstance(event, TaskCreated)]
    (ended,) = [event for event in output if isinstance(event, ChunkGenerated)]
    assert ended.command_id == request.command_id
    assert ended.chunk == ErrorChunk(
        model=MODEL,
        error_message="The model instance was shut down before this request could start.",
    )


async def test_a_request_right_after_a_deletion_goes_to_another_instance() -> None:
    deleted, other = instance_on(NODE), instance_on(NODE)
    master, broadcast = master_with(deleted, other)

    output = await process(
        master,
        broadcast,
        DeleteInstance(instance_id=deleted.instance_id),
        chat_request(),
    )

    (created,) = [event for event in output if isinstance(event, TaskCreated)]
    assert created.task.instance_id == other.instance_id


async def test_a_request_just_before_its_instance_was_deleted_is_cancelled_with_it() -> (
    None
):
    """The other order: the deletion cancels the instance's requests, the one just sent too."""
    instance = instance_on(NODE)
    master, broadcast = master_with(instance)

    output = await process(
        master,
        broadcast,
        chat_request(),
        DeleteInstance(instance_id=instance.instance_id),
        until=(InstanceDeleted,),
    )

    (created,) = [event for event in output if isinstance(event, TaskCreated)]
    assert created.task.instance_id == instance.instance_id
    assert [
        (event.task_id, event.task_status)
        for event in output
        if isinstance(event, TaskStatusUpdated)
    ] == [(created.task_id, TaskStatus.Cancelled)]


async def test_a_request_right_after_a_node_loss_ends_with_an_error() -> None:
    instance = instance_on(LOST_NODE)
    master, broadcast = master_with(instance)
    # The instance lost its node: the master deletes it
    master._remove_silent_nodes_and_broken_instances()  # pyright: ignore[reportPrivateUsage]
    deleted = (await broadcast.receive()).event
    assert isinstance(deleted, InstanceDeleted)
    assert deleted.instance_id == instance.instance_id
    request = chat_request()

    output = await process(master, broadcast, request)

    assert not [event for event in output if isinstance(event, TaskCreated)]
    (ended,) = [event for event in output if isinstance(event, ChunkGenerated)]
    assert ended.command_id == request.command_id
    assert isinstance(ended.chunk, ErrorChunk)


async def test_a_request_for_a_model_that_is_not_ready_ends_with_why() -> None:
    """The API waits for a model to be ready, but a runner can fail just after it looked."""
    instance = instance_on(NODE)
    master, broadcast = master_with(instance)
    (runner,) = instance.shard_assignments.runner_to_shard
    master.state = master.state.model_copy(
        update={"runners": {runner: RunnerLoading(layers_loaded=1, total_layers=2)}}
    )
    request = chat_request()

    output = await process(master, broadcast, request)

    assert not [event for event in output if isinstance(event, TaskCreated)]
    (ended,) = [event for event in output if isinstance(event, ChunkGenerated)]
    assert ended.chunk == ErrorChunk(
        model=MODEL,
        error_message="The model is not ready yet: loading (1 of 2 layers). "
        "Try again in 5 s.",
    )
