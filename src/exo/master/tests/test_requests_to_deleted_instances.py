"""The master never sends a request to an instance it is deleting.

The master's state changes only when its own events come back indexed, a few milliseconds
after it sends them. A request that reached it in that window, just after the instance was
deleted (by a DeleteInstance, or because a node was lost), used to get a task on the deleted
instance. Nothing ran that task and nothing ended its request: the stream got keepalives
until the client gave up.
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
)
from exo.shared.types.memory import Memory
from exo.shared.types.state import State
from exo.shared.types.text_generation import (
    InputMessage,
    InputMessageContent,
    TextGenerationTaskParams,
)
from exo.shared.types.worker.instances import InstanceId, MlxRingInstance
from exo.shared.types.worker.runners import RunnerId, ShardAssignments
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


def master_with(*instances: MlxRingInstance) -> tuple[Master, Receiver[Event]]:
    """A master whose events are never applied, as in the window after it sends them."""
    node_id = get_node_zid()
    event_sender, events = channel[Event]()
    topology = Topology()
    topology.add_node(NODE)
    master = Master(
        node_id,
        SessionId(master_node_id=node_id, election_clock=0),
        command_receiver=channel[ForwarderCommand]()[1],
        event_sender=event_sender,
        local_event_receiver=channel[LocalForwarderEvent]()[1],
        global_event_sender=channel[GlobalForwarderEvent]()[0],
        snapshot_sender=channel[StateSnapshot]()[0],
        download_command_sender=channel[ForwarderDownloadCommand]()[0],
        initial_state=State(
            topology=topology,
            last_seen={NODE: datetime.now(tz=timezone.utc)},
            instances={instance.instance_id: instance for instance in instances},
        ),
    )
    return master, events


async def process(
    master: Master, events: Receiver[Event], *sent: Command
) -> list[Event]:
    """Have the master process the commands back to back, applying none of its events."""
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
            while not any(
                isinstance(event, (TaskCreated, ChunkGenerated)) for event in output
            ):
                output.append(await events.receive())
        tg.cancel_scope.cancel()
    return output


async def test_a_request_right_after_its_instance_was_deleted_ends_with_an_error() -> (
    None
):
    instance = instance_on(NODE)
    master, events = master_with(instance)
    request = chat_request()

    output = await process(
        master, events, DeleteInstance(instance_id=instance.instance_id), request
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
    master, events = master_with(deleted, other)

    output = await process(
        master, events, DeleteInstance(instance_id=deleted.instance_id), chat_request()
    )

    (created,) = [event for event in output if isinstance(event, TaskCreated)]
    assert created.task.instance_id == other.instance_id


async def test_a_request_right_after_a_node_loss_ends_with_an_error() -> None:
    instance = instance_on(LOST_NODE)
    master, events = master_with(instance)
    # The instance lost its node: the master deletes it
    await master._remove_silent_nodes_and_broken_instances()  # pyright: ignore[reportPrivateUsage]
    deleted = await events.receive()
    assert isinstance(deleted, InstanceDeleted)
    assert deleted.instance_id == instance.instance_id
    request = chat_request()

    output = await process(master, events, request)

    assert not [event for event in output if isinstance(event, TaskCreated)]
    (ended,) = [event for event in output if isinstance(event, ChunkGenerated)]
    assert ended.command_id == request.command_id
    assert isinstance(ended.chunk, ErrorChunk)
