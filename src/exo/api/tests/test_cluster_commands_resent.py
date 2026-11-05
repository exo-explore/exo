"""A request that changes the cluster (launching or stopping a model, ...) is answered once the
change shows in this node's state. Its command is sent again until then, at once to a new
master, so a launch sent as the master died is no longer accepted and then lost."""

from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from datetime import datetime, timezone

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
from exo.master.tests.test_requests_to_deleted_instances import deployment_of
from exo.routing.tests.fake_membership import FakeMembership
from exo.shared.models.model_cards import ModelCard, ModelId, ModelTask, card_cache
from exo.shared.topology import Topology
from exo.shared.types.backends import Backend
from exo.shared.types.chunks import ErrorChunk, ImageChunk
from exo.shared.types.commands import (
    Command,
    CreateDeployment,
    DeleteDeployment,
    ForwarderCommand,
    ImageGeneration,
)
from exo.shared.types.common import NodeId, SessionId, SystemId
from exo.shared.types.events import (
    DeploymentCreated,
    DeploymentRemoved,
    Event,
    IndexedEvent,
    InstanceCreated,
    InstanceDeleted,
    StateSnapshot,
)
from exo.shared.types.instance_link import InstanceLinkId
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
NOW = datetime.now(tz=timezone.utc).isoformat()

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
    # Reaching every node the state lists
    api.membership = FakeMembership(
        NodeId("api-node"), *api.state.topology.list_nodes()
    )
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


def next_event(api: API, event: Event) -> IndexedEvent:
    return IndexedEvent(idx=api.state.last_event_applied_idx + 1, event=event)


async def apply_events(api: API, events: Sender[NodeEvents], *sent: Event) -> None:
    """Send events as the master would, numbered on from this node's state, and wait until
    it has applied them."""
    first = api.state.last_event_applied_idx + 1
    for offset, event in enumerate(sent):
        await events.send(IndexedEvent(idx=first + offset, event=event))
    await until(lambda: api.state.last_event_applied_idx >= first + len(sent) - 1)


async def launched(api: API, events: Sender[NodeEvents], instance: Instance) -> None:
    """The master launched the instance, for its deployment."""
    await apply_events(
        api,
        events,
        DeploymentCreated(deployment=deployment_of(instance)),
        InstanceCreated(instance=instance),
    )


async def stopped(api: API, events: Sender[NodeEvents], instance: Instance) -> None:
    """The master stopped the instance's deployment."""
    await apply_events(
        api,
        events,
        InstanceDeleted(instance_id=instance.instance_id, reason="UserStopped", at=NOW),
        DeploymentRemoved(deployment_id=instance.deployment_id),
    )


def snapshot_from_new_master(api: API) -> StateSnapshot:
    """The state a newly elected master sends: what this node had, nothing more."""
    return StateSnapshot(session=NEW_MASTER, requester=SystemId(), state=api.state)


async def a_placement(api: API) -> Instance:
    return await api.get_placement(
        MODEL.model_id, Sharding.Pipeline, InstanceMeta.MlxRing, 1
    )


async def first_sent[T: Command](
    commands: Receiver[ForwarderCommand], kind: type[T]
) -> T:
    """The next command of this kind the API sends, skipping copies of earlier ones."""
    while True:
        command = (await commands.receive()).command
        if isinstance(command, kind):
            return command


async def until(condition: Callable[[], bool]) -> None:
    with anyio.fail_after(5):
        while not condition():
            await anyio.sleep(0.01)


def answer_quickly(monkeypatch: pytest.MonkeyPatch) -> None:
    """Shorten the wait for a change, so a request is answered 503 within the test."""
    monkeypatch.setattr(api_main, "CLUSTER_COMMAND_RESEND_INTERVAL", 0.05)
    monkeypatch.setattr(api_main, "CLUSTER_COMMAND_TIMEOUT", 0.3)


def answer_patiently(monkeypatch: pytest.MonkeyPatch) -> None:
    """Back to waiting for a change for longer than any test takes, sending it once."""
    monkeypatch.setattr(api_main, "CLUSTER_COMMAND_RESEND_INTERVAL", NO_TIMED_RESENDS)
    monkeypatch.setattr(api_main, "CLUSTER_COMMAND_TIMEOUT", NO_TIMED_RESENDS)


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
                launch_sent = sent[0]
                assert sent[1] == launch_sent
                assert isinstance(launch_sent, CreateDeployment)
                await launched(api, events, launch_sent.instance)
    # The very same launch, both times
    assert sent[0] == sent[1]
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
                launch_sent = (await commands.receive()).command
                assert isinstance(launch_sent, CreateDeployment)
                await apply_events(
                    api,
                    events,
                    DeploymentCreated(deployment=deployment_of(launch_sent.instance)),
                    InstanceCreated(instance=launch_sent.instance),
                    InstanceDeleted(
                        instance_id=instance.instance_id, reason="NodeLost", at=NOW
                    ),
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
                assert isinstance(command, CreateDeployment)
                assert (
                    command.instance.shard_assignments.node_to_runner.keys()
                    == expected.shard_assignments.node_to_runner.keys()
                )
                await launched(api, events, command.instance)
    assert len(responses) == 1


async def test_a_stop_is_sent_to_the_new_master(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(api_main, "CLUSTER_COMMAND_RESEND_INTERVAL", NO_TIMED_RESENDS)
    async with running_api() as (api, commands, events):
        instance = await a_placement(api)
        await launched(api, events, instance)

        sent: list[Command] = []
        async with anyio.create_task_group() as tg:
            tg.start_soon(api.delete_instance, instance.instance_id)
            with anyio.fail_after(5):
                sent.append((await commands.receive()).command)
                await events.send(snapshot_from_new_master(api))
                sent.append((await commands.receive()).command)
                await stopped(api, events, instance)
    stop = DeleteDeployment(
        command_id=sent[0].command_id, deployment_id=instance.deployment_id
    )
    assert sent == [stop, stop]
    assert instance.instance_id not in api.state.instances


# A request repeating a change already made is answered at once


async def test_a_repeated_launch_is_answered_at_once_without_being_sent() -> None:
    """Launching an instance that is already running changes nothing, so it is answered at
    once instead of after CLUSTER_COMMAND_TIMEOUT (46 s)."""
    async with running_api() as (api, commands, events):
        instance = await a_placement(api)
        await launched(api, events, instance)

        with anyio.fail_after(1):
            response = await api.create_instance(
                CreateInstanceParams(instance=instance)
            )
        assert commands.collect() == []
        # It names the deployment the placement runs for
        assert response.deployment_id == instance.deployment_id


async def test_unlinking_a_link_that_is_gone_is_answered_at_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(api_main, "ENABLE_DISAGGREGATION", True)
    async with running_api() as (api, commands, _events):
        with anyio.fail_after(1):
            await api.delete_instance_link(InstanceLinkId())
        assert commands.collect() == []


# After a 503 a launch is called off: a late copy of it is undone


async def test_a_launch_answered_503_is_undone_when_it_applies_after_all(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The client tries POST /place_instance again after a 503, and then a late copy of the
    first launch applies too: it is deleted again, so one instance runs, not two."""
    answer_quickly(monkeypatch)
    async with running_api() as (api, commands, events):
        with anyio.fail_after(5), pytest.raises(HTTPException) as failed:
            await api.place_instance(PlaceInstanceParams(model_id=MODEL.model_id))
        assert failed.value.status_code == 503
        first = commands.collect()[0].command
        assert isinstance(first, CreateDeployment)
        answer_patiently(monkeypatch)

        # Tried again, a new placement is launched, and this time it shows
        retries: list[CreateDeployment] = []
        async with anyio.create_task_group() as tg:
            tg.start_soon(
                api.place_instance, PlaceInstanceParams(model_id=MODEL.model_id)
            )
            with anyio.fail_after(5):
                retries.append(await first_sent(commands, CreateDeployment))
                await launched(api, events, retries[0].instance)
        (retry,) = retries
        assert retry.instance.instance_id != first.instance.instance_id
        assert retry.deployment_id != first.deployment_id

        # The first launch applies late
        await launched(api, events, first.instance)
        with anyio.fail_after(2):
            undo = await first_sent(commands, DeleteDeployment)
        assert undo.deployment_id == first.deployment_id
        await stopped(api, events, first.instance)
        assert list(api.state.instances) == [retry.instance.instance_id]
        assert list(api.state.deployments) == [retry.deployment_id]
    # The 503 said so
    assert "called off" in str(failed.value.detail)


async def test_the_same_launch_tried_again_after_a_503_is_kept(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Trying the very same instance again takes over from the failed request: when it
    shows, it is what the client asked for, not a late copy to undo. It is launched under
    the failed request's deployment, so a late copy of the first launch is the same launch,
    and a stop of it covers both (a copy under another id, no longer watched, could start
    a stopped model again)."""
    answer_quickly(monkeypatch)
    responses: list[CreateInstanceResponse] = []
    async with running_api() as (api, commands, events):
        instance = await a_placement(api)
        with anyio.fail_after(5), pytest.raises(HTTPException):
            await api.create_instance(CreateInstanceParams(instance=instance))
        answer_patiently(monkeypatch)
        (first, *_) = [
            c.command
            for c in commands.collect()
            if isinstance(c.command, CreateDeployment)
        ]
        assert isinstance(first, CreateDeployment)

        async def launch() -> None:
            responses.append(
                await api.create_instance(CreateInstanceParams(instance=instance))
            )

        retried: list[CreateDeployment] = []
        async with anyio.create_task_group() as tg:
            tg.start_soon(launch)
            with anyio.fail_after(5):
                retried.append(await first_sent(commands, CreateDeployment))
                await launched(api, events, retried[0].instance)
        await anyio.sleep(0.1)
        sent = [c.command for c in commands.collect()]
    assert len(responses) == 1
    assert not any(isinstance(command, DeleteDeployment) for command in sent)
    assert instance.instance_id in api.state.instances
    assert retried[0].deployment_id == first.deployment_id
    assert responses[0].deployment_id == first.deployment_id


async def test_the_same_launch_tried_again_while_its_late_copy_is_undone_waits_for_it(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Tried again just as a late copy of the failed launch is being undone, the request
    isn't answered from the instance about to go: it waits for the undo, then launches."""
    answer_quickly(monkeypatch)
    responses: list[CreateInstanceResponse] = []
    async with running_api() as (api, commands, events):
        instance = await a_placement(api)
        with anyio.fail_after(5), pytest.raises(HTTPException):
            await api.create_instance(CreateInstanceParams(instance=instance))
        answer_patiently(monkeypatch)

        # The late copy applies, and this node starts undoing it
        (late,) = [
            c.command
            for c in commands.collect()
            if isinstance(c.command, CreateDeployment)
        ][:1]
        assert isinstance(late, CreateDeployment)
        await launched(api, events, late.instance)
        with anyio.fail_after(2):
            await first_sent(commands, DeleteDeployment)

        async def launch() -> None:
            responses.append(
                await api.create_instance(CreateInstanceParams(instance=instance))
            )

        async with anyio.create_task_group() as tg:
            tg.start_soon(launch)
            await anyio.sleep(0.1)
            assert responses == []
            await stopped(api, events, late.instance)
            with anyio.fail_after(5):
                again = await first_sent(commands, CreateDeployment)
                assert again.instance.instance_id == instance.instance_id
                assert again.deployment_id != late.deployment_id
                await launched(api, events, again.instance)
    assert len(responses) == 1
    assert instance.instance_id in api.state.instances


# A change the old master applied, but the master that replaced it never saw, is sent again
# to the new one: the request was answered already


OLD_MASTER = SessionId(master_node_id=NodeId("old-master"), election_clock=1)


async def launched_through_api(
    api: API, commands: Receiver[ForwarderCommand], events: Sender[NodeEvents]
) -> CreateDeployment:
    """Launch an instance through the API, applied by this node's master."""
    sent: list[CreateDeployment] = []
    async with anyio.create_task_group() as tg:
        tg.start_soon(
            api.create_instance, CreateInstanceParams(instance=await a_placement(api))
        )
        with anyio.fail_after(5):
            launch = await first_sent(commands, CreateDeployment)
            await launched(api, events, launch.instance)
            sent.append(launch)
    return sent[0]


async def test_a_launch_applied_by_a_master_that_lost_a_merge_is_sent_to_the_winner() -> (
    None
):
    """Two sides of a split meet again: this side's master applied the launch, the other
    side's master wins the election and never saw it. Before, the launch was answered and
    then lost. It goes under a new command id: the winner may have seen the old one (as the
    link came back) without applying it, and it ignores an id it has processed."""
    async with running_api() as (api, commands, events):
        before = api.state
        launch = await launched_through_api(api, commands, events)
        instance = launch.instance
        await events.send(
            StateSnapshot(session=NEW_MASTER, requester=SystemId(), state=before)
        )
        with anyio.fail_after(5):
            resent = await first_sent(commands, CreateDeployment)
        await launched(api, events, instance)

    assert resent.instance == instance
    assert resent.deployment_id == launch.deployment_id
    assert resent.command_id != launch.command_id


async def test_a_launch_stopped_since_is_not_restored_by_a_new_master() -> None:
    """The user stopped the model since: a new master without it doesn't bring it back. (A
    node lost since doesn't undo a launch: the model is still wanted, and is placed again.)"""
    async with running_api() as (api, commands, events):
        before = api.state
        instance = (await launched_through_api(api, commands, events)).instance
        await stopped(api, events, instance)
        await events.send(
            StateSnapshot(session=NEW_MASTER, requester=SystemId(), state=before)
        )
        await anyio.sleep(0.2)

    assert commands.collect() == []


async def test_catching_up_from_the_same_master_restores_nothing() -> None:
    """A snapshot from the master that applied the launch, without it: the master changed
    it in events this node skipped, so it isn't this node's to restore."""
    async with running_api() as (api, commands, events):
        await events.send(
            StateSnapshot(session=OLD_MASTER, requester=SystemId(), state=api.state)
        )
        before = api.state
        await launched_through_api(api, commands, events)
        await events.send(
            StateSnapshot(session=OLD_MASTER, requester=SystemId(), state=before)
        )
        await anyio.sleep(0.2)

    assert commands.collect() == []


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
