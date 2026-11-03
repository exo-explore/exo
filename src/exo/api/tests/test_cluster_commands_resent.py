"""A request that changes the cluster (launching or stopping a model, ...) is answered once the
change shows in this node's state. Its command is sent again until then, at once to a new
master, so a launch sent as the master died is no longer accepted and then lost."""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

import anyio
import pytest
from fastapi import HTTPException

import exo.api.main as api_main
from exo.api.main import API
from exo.api.recent_events import RecentEvents
from exo.api.types import (
    CreateInstanceParams,
    CreateInstanceResponse,
    ImageGenerationTaskParams,
    PlaceInstanceParams,
)
from exo.master.tests.conftest import create_node_memory, create_node_network
from exo.shared.models.model_cards import ModelCard, ModelId, ModelTask, card_cache
from exo.shared.topology import Topology
from exo.shared.types.backends import Backend
from exo.shared.types.chunks import ErrorChunk, ImageChunk
from exo.shared.types.commands import (
    Command,
    CreateInstance,
    DeleteInstance,
    ForwarderCommand,
    ImageGeneration,
    PlaceInstance,
)
from exo.shared.types.common import NodeId, SessionId, SystemId
from exo.shared.types.events import (
    IndexedEvent,
    InstanceCreated,
    InstanceDeleted,
    StateSnapshot,
)
from exo.shared.types.memory import Memory
from exo.shared.types.state import State
from exo.shared.types.worker.instances import Instance, InstanceMeta
from exo.shared.types.worker.shards import Sharding
from exo.utils.channels import Receiver, Sender, channel
from exo.utils.task_group import TaskGroup

MODEL = ModelCard(
    model_id=ModelId("test-org/test-model"),
    storage_size=Memory.from_kb(1000),
    n_layers=10,
    hidden_size=30,
    supports_tensor=True,
    tasks=[ModelTask.TextGeneration],
    backends=[Backend.MlxMetal],
)
NODE = NodeId("only-node")
NEW_MASTER = SessionId(master_node_id=NodeId("new-master"), election_clock=2)
# Long enough that within a test only a snapshot (a new master) makes the API send again
NO_TIMED_RESENDS = 60.0

type NodeEvents = IndexedEvent | StateSnapshot


@pytest.fixture(autouse=True)
def known_model(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(card_cache, "cc", {MODEL.model_id: MODEL})


def single_node_state() -> State:
    topology = Topology()
    topology.add_node(NODE)
    return State(
        topology=topology,
        node_memory={NODE: create_node_memory(10_000_000)},
        node_network={NODE: create_node_network()},
        node_backends={NODE: [Backend.MlxMetal]},
    )


@asynccontextmanager
async def running_api() -> AsyncIterator[
    tuple[API, Receiver[ForwarderCommand], Sender[NodeEvents]]
]:
    """An API applying the events it is sent, as on a node, with the commands it sends."""
    api = object.__new__(API)
    api.state = single_node_state()
    api.paused = False
    api._system_id = SystemId()  # pyright: ignore[reportPrivateUsage]
    api.command_sender, commands = channel[ForwarderCommand]()
    events, api.event_receiver = channel[NodeEvents]()
    api._recent_events = RecentEvents()  # pyright: ignore[reportPrivateUsage]
    api._text_generation_queues = {}  # pyright: ignore[reportPrivateUsage]
    api._image_generation_queues = {}  # pyright: ignore[reportPrivateUsage]
    api._tg = TaskGroup()  # pyright: ignore[reportPrivateUsage]
    async with api._tg:  # pyright: ignore[reportPrivateUsage]
        api._tg.start_soon(api._apply_state)  # pyright: ignore[reportPrivateUsage]
        yield api, commands, events
        api._tg.cancel_tasks()  # pyright: ignore[reportPrivateUsage]


def next_event(api: API, event: InstanceCreated | InstanceDeleted) -> IndexedEvent:
    return IndexedEvent(idx=api.state.last_event_applied_idx + 1, event=event)


def snapshot_from_new_master(api: API) -> StateSnapshot:
    """The state a newly elected master sends: what this node had, nothing more."""
    return StateSnapshot(session=NEW_MASTER, requester=SystemId(), state=api.state)


async def a_placement(api: API) -> Instance:
    return await api.get_placement(
        MODEL.model_id, Sharding.Pipeline, InstanceMeta.MlxRing, 1
    )


async def test_a_launch_the_dying_master_never_applied_is_sent_to_the_new_one(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(api_main, "CLUSTER_COMMAND_RESEND_INTERVAL", NO_TIMED_RESENDS)
    responses: list[CreateInstanceResponse] = []
    sent: list[Command] = []
    async with running_api() as (api, commands, events):
        instance = await a_placement(api)

        async def launch() -> None:
            responses.append(
                await api.create_instance(CreateInstanceParams(instance=instance))
            )

        async with anyio.create_task_group() as tg:
            tg.start_soon(launch)
            with anyio.fail_after(5):
                # The master dies with the command: nothing comes back
                sent.append((await commands.receive()).command)
                await anyio.sleep(0.1)
                assert responses == []
                # A new master is elected and this node catches up from its state
                await events.send(snapshot_from_new_master(api))
                sent.append((await commands.receive()).command)
                assert responses == []
                await events.send(next_event(api, InstanceCreated(instance=instance)))
    assert (
        sent == [CreateInstance(command_id=sent[0].command_id, instance=instance)] * 2
    )
    assert len(responses) == 1
    assert instance.instance_id in api.state.instances


async def test_a_launch_never_applied_fails_instead_of_being_accepted(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(api_main, "CLUSTER_COMMAND_RESEND_INTERVAL", 0.05)
    monkeypatch.setattr(api_main, "CLUSTER_COMMAND_TIMEOUT", 0.3)
    async with running_api() as (api, commands, _events):
        instance = await a_placement(api)
        with anyio.fail_after(5), pytest.raises(HTTPException) as failed:
            await api.create_instance(CreateInstanceParams(instance=instance))
        sent = [c.command for c in commands.collect()]
    assert failed.value.status_code == 503
    # Sent again every resend interval, and each time the same launch
    assert len(sent) >= 3
    assert all(command == sent[0] for command in sent)


async def test_a_launch_undone_at_once_still_counts_as_applied() -> None:
    """An instance created and then deleted straight away (its node was lost) was launched."""
    responses: list[CreateInstanceResponse] = []
    async with running_api() as (api, commands, events):
        instance = await a_placement(api)

        async def launch() -> None:
            responses.append(
                await api.create_instance(CreateInstanceParams(instance=instance))
            )

        async with anyio.create_task_group() as tg:
            tg.start_soon(launch)
            with anyio.fail_after(5):
                await commands.receive()
                last = api.state.last_event_applied_idx
                await events.send(
                    IndexedEvent(idx=last + 1, event=InstanceCreated(instance=instance))
                )
                await events.send(
                    IndexedEvent(
                        idx=last + 2,
                        event=InstanceDeleted(instance_id=instance.instance_id),
                    )
                )
    assert len(responses) == 1


async def test_a_placement_is_launched_as_the_instance_it_computed() -> None:
    """Sent again, a placement launches the same instance rather than placing it anew."""
    responses: list[CreateInstanceResponse] = []
    async with running_api() as (api, commands, events):
        expected = await a_placement(api)

        async def place() -> None:
            responses.append(
                await api.place_instance(PlaceInstanceParams(model_id=MODEL.model_id))
            )

        async with anyio.create_task_group() as tg:
            tg.start_soon(place)
            with anyio.fail_after(5):
                command = (await commands.receive()).command
                assert not isinstance(command, PlaceInstance)
                assert isinstance(command, CreateInstance)
                assert (
                    command.instance.shard_assignments.node_to_runner.keys()
                    == expected.shard_assignments.node_to_runner.keys()
                )
                await events.send(
                    next_event(api, InstanceCreated(instance=command.instance))
                )
    assert len(responses) == 1


async def test_a_stop_is_sent_to_the_new_master(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(api_main, "CLUSTER_COMMAND_RESEND_INTERVAL", NO_TIMED_RESENDS)
    async with running_api() as (api, commands, events):
        instance = await a_placement(api)
        await events.send(next_event(api, InstanceCreated(instance=instance)))
        with anyio.fail_after(5):
            while instance.instance_id not in api.state.instances:
                await anyio.sleep(0.01)

        sent: list[Command] = []
        async with anyio.create_task_group() as tg:
            tg.start_soon(api.delete_instance, instance.instance_id)
            with anyio.fail_after(5):
                sent.append((await commands.receive()).command)
                await events.send(snapshot_from_new_master(api))
                sent.append((await commands.receive()).command)
                await events.send(
                    next_event(api, InstanceDeleted(instance_id=instance.instance_id))
                )
    stop = DeleteInstance(
        command_id=sent[0].command_id, instance_id=instance.instance_id
    )
    assert sent == [stop, stop]
    assert instance.instance_id not in api.state.instances


# Image requests are sent again until accepted, like chats


@pytest.fixture
def quick_request_resends(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(api_main, "REQUEST_RESEND_WAITS", (0.05, 0.05))


@pytest.mark.usefixtures("quick_request_resends")
async def test_an_image_request_never_accepted_is_sent_again_then_ends_with_an_error() -> (
    None
):
    request = ImageGeneration(
        task_params=ImageGenerationTaskParams(prompt="a cat", model=MODEL.model_id)
    )
    async with running_api() as (api, commands, _events):
        image_send: Sender[ImageChunk | ErrorChunk]
        image_send, images = channel[ImageChunk | ErrorChunk]()
        api._image_generation_queues[request.command_id] = image_send  # pyright: ignore[reportPrivateUsage]
        await api._send(request)  # pyright: ignore[reportPrivateUsage]
        with anyio.fail_after(5):
            chunk = await images.receive()
        sent: list[Command] = [c.command for c in commands.collect()]
    assert sent == [request, request, request]
    assert isinstance(chunk, ErrorChunk)
    assert "didn't accept" in chunk.error_message
