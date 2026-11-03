# pyright: reportPrivateUsage=false
"""Launches the master refuses because their devices don't have the memory for them.

The API checks a launch against its own state, which doesn't show a launch still on its way,
so two launches at once (two tabs, two devices) both passed, and the master placed both in
memory that holds one. The master now refuses the second, in its own log. Nothing tells the API,
so the request learns of it only as the launch not showing: it is answered 503 after the
usual wait, saying the launch may not have fitted, and called off.

The API doesn't guess a refusal from its own state: its view lags the master's, so a dip in
free memory it sees while it waits may be gone by the time the master handles the launch.
"""

import time
from collections.abc import Awaitable, Callable
from datetime import datetime, timezone

import anyio
import pytest
from fastapi import HTTPException

import exo.api.main as api_main
from exo.api.main import API, WithdrawnClusterCommand
from exo.api.recent_events import RecentEvents
from exo.api.types import (
    CreateInstanceParams,
    CreateInstanceResponse,
    PlaceInstanceParams,
)
from exo.master.main import Master
from exo.master.tests.conftest import create_node_network
from exo.routing.router import get_node_zid
from exo.routing.tests.fake_membership import FakeMembership
from exo.shared.models.model_cards import ModelCard, ModelId, ModelTask, card_cache
from exo.shared.topology import Topology
from exo.shared.types.backends import Backend
from exo.shared.types.commands import (
    Command,
    CreateInstance,
    DeleteInstance,
    ForwarderCommand,
    ForwarderDownloadCommand,
)
from exo.shared.types.common import NodeId, SessionId, SystemId
from exo.shared.types.events import (
    Event,
    GlobalForwarderEvent,
    IndexedEvent,
    LocalForwarderEvent,
    NodeGatheredInfo,
    RunnerStatusUpdated,
    StateSnapshot,
)
from exo.shared.types.memory import Memory
from exo.shared.types.profiling import MemoryUsage
from exo.shared.types.state import State
from exo.shared.types.worker.instances import InstanceId, InstanceMeta, MlxRingInstance
from exo.shared.types.worker.runners import (
    RunnerId,
    RunnerLoaded,
    RunnerLoading,
    ShardAssignments,
)
from exo.shared.types.worker.shards import PipelineShardMetadata, Sharding
from exo.utils.channels import Receiver, Sender, channel
from exo.utils.task_group import TaskGroup

NODE = NodeId("only-node")
# The node has room for one copy of the model, not two
MODEL = ModelCard(
    model_id=ModelId("test-org/test-model"),
    storage_size=Memory.from_mb(600),
    n_layers=10,
    hidden_size=30,
    supports_tensor=True,
    tasks=[ModelTask.TextGeneration],
    backends=[Backend.MlxMetal],
)
NODE_MEMORY = Memory.from_mb(1000)
# Another model, still loading on the node
OTHER = ModelCard(
    model_id=ModelId("test-org/other-model"),
    storage_size=Memory.from_mb(300),
    n_layers=1,
    hidden_size=30,
    supports_tensor=False,
    tasks=[ModelTask.TextGeneration],
    backends=[Backend.MlxMetal],
)


@pytest.fixture(autouse=True)
def known_model(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(card_cache, "cc", {MODEL.model_id: MODEL})


def single_node_state() -> State:
    topology = Topology()
    topology.add_node(NODE)
    return State(
        topology=topology,
        last_seen={NODE: datetime.now(tz=timezone.utc)},
        node_memory={NODE: usage(NODE_MEMORY.in_mb)},
        node_network={NODE: create_node_network()},
        node_backends={NODE: [Backend.MlxMetal]},
    )


def usage(available_mb: float) -> MemoryUsage:
    return MemoryUsage(
        ram_total=NODE_MEMORY,
        ram_available=Memory.from_mb(available_mb),
        swap_total=Memory(),
        swap_available=Memory(),
    )


def memory_report(available_mb: float) -> NodeGatheredInfo:
    return NodeGatheredInfo(
        node_id=NODE,
        when=str(datetime.now(tz=timezone.utc)),
        info=usage(available_mb),
    )


def loading_instance() -> tuple[MlxRingInstance, RunnerId]:
    runner = RunnerId()
    instance = MlxRingInstance(
        instance_id=InstanceId(),
        shard_assignments=ShardAssignments(
            model_id=OTHER.model_id,
            runner_to_shard={
                runner: PipelineShardMetadata(
                    model_card=OTHER,
                    device_rank=0,
                    world_size=1,
                    start_layer=0,
                    end_layer=1,
                    n_layers=1,
                )
            },
            node_to_runner={NODE: runner},
        ),
        hosts_by_node={},
        ephemeral_port=50000,
    )
    return instance, runner


type Launch = Callable[[API], Awaitable[CreateInstanceResponse]]


async def given_placement(api: API) -> CreateInstanceResponse:
    """POST /instance with a placement made beforehand, as the dashboard launches a preview."""
    placement = await api.get_placement(
        MODEL.model_id, Sharding.Pipeline, InstanceMeta.MlxRing, 1
    )
    return await api.create_instance(CreateInstanceParams(instance=placement))


async def placed_by_the_api(api: API) -> CreateInstanceResponse:
    """POST /place_instance"""
    return await api.place_instance(PlaceInstanceParams(model_id=MODEL.model_id))


def an_api(state: State) -> tuple[API, Receiver[ForwarderCommand]]:
    api = object.__new__(API)
    api.state = state
    api.paused = False
    api._system_id = SystemId()
    api.command_sender, commands = channel[ForwarderCommand]()
    api._recent_events = RecentEvents()
    api._text_generation_queues = {}
    api._image_generation_queues = {}
    api._tg = TaskGroup()
    return api, commands


def a_master(
    state: State, commands: Receiver[ForwarderCommand]
) -> tuple[Master, Receiver[GlobalForwarderEvent], Sender[LocalForwarderEvent]]:
    """A master, what it sends to every node, and how nodes' events reach it."""
    node_id = get_node_zid()
    global_event_sender, broadcast = channel[GlobalForwarderEvent]()
    local_sender, local_events = channel[LocalForwarderEvent]()
    master = Master(
        node_id,
        SessionId(master_node_id=node_id, election_clock=0),
        command_receiver=commands,
        local_event_receiver=local_events,
        global_event_sender=global_event_sender,
        snapshot_sender=channel[StateSnapshot]()[0],
        download_command_sender=channel[ForwarderDownloadCommand]()[0],
        # Reaching every node the state lists
        membership=FakeMembership(node_id, *state.topology.list_nodes()),
        initial_state=state,
    )
    return master, broadcast, local_sender


@pytest.mark.parametrize("launch", [given_placement, placed_by_the_api])
async def test_of_two_launches_at_once_that_do_not_both_fit_one_starts(
    launch: Launch, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The other is answered 503, saying it may not have fitted, once it hasn't shown within
    the wait, and is called off. Nothing is placed in memory that holds one."""
    monkeypatch.setattr(api_main, "CLUSTER_COMMAND_TIMEOUT", 1.0)
    monkeypatch.setattr(api_main, "CLUSTER_COMMAND_RESEND_INTERVAL", 0.2)
    initial = single_node_state()
    api, commands = an_api(initial)
    to_api, api.event_receiver = channel[IndexedEvent | StateSnapshot]()
    master, broadcast, _ = a_master(initial, commands)

    async def deliver() -> None:
        with broadcast as messages:
            async for message in messages:
                await to_api.send(
                    IndexedEvent(idx=message.origin_idx, event=message.event)
                )

    answers: list[CreateInstanceResponse | HTTPException] = []

    async def request() -> None:
        try:
            answers.append(await launch(api))
        except HTTPException as error:
            answers.append(error)

    with anyio.fail_after(10):
        async with api._tg:
            api._tg.start_soon(api._apply_state)
            api._tg.start_soon(deliver)
            api._tg.start_soon(request)
            api._tg.start_soon(request)
            # Both launches are on their way before the master has handled either
            while commands.statistics().current_buffer_used < 2:
                await anyio.sleep(0.001)
            api._tg.start_soon(master._command_processor)
            while len(answers) < 2:
                await anyio.sleep(0.01)
            api._tg.cancel_tasks()

    started = [
        answer for answer in answers if isinstance(answer, CreateInstanceResponse)
    ]
    failed = [answer for answer in answers if isinstance(answer, HTTPException)]
    assert len(started) == 1
    assert [error.status_code for error in failed] == [503]
    assert "may not have fitted" in str(failed[0].detail)
    assert len(master.state.instances) == 1
    # The refused launch is called off: should it still apply, this node undoes it
    (withdrawn,) = api._withdrawn_cluster_commands.values()
    assert isinstance(withdrawn.command, CreateInstance)
    assert withdrawn.command.instance.instance_id not in master.state.instances
    assert isinstance(withdrawn.undo, DeleteInstance)
    assert withdrawn.undo.instance_id == withdrawn.command.instance.instance_id


async def run_with_a_lagging_api(
    initial: State, before_launch: list[Event]
) -> tuple[list[object], list[Command], Master]:
    """The master indexes `before_launch`, then handles a launch. The API sees none of it
    until the master has handled the launch, as a lagging node would, then all of it in
    order."""
    api, api_commands = an_api(initial)
    to_api, api.event_receiver = channel[IndexedEvent | StateSnapshot]()
    master_commands_sender, master_commands = channel[ForwarderCommand]()
    master, broadcast, local_sender = a_master(initial, master_commands)
    sent: list[Command] = []
    lag_over = anyio.Event()

    async def forward() -> None:
        with api_commands as stream:
            async for command in stream:
                sent.append(command.command)
                await master_commands_sender.send(command)

    async def deliver() -> None:
        await lag_over.wait()
        with broadcast as messages:
            async for message in messages:
                await to_api.send(
                    IndexedEvent(idx=message.origin_idx, event=message.event)
                )

    answers: list[object] = []

    async def request() -> None:
        try:
            answers.append(await given_placement(api))
        except HTTPException as error:
            answers.append(error)

    with anyio.fail_after(10):
        async with api._tg:
            api._tg.start_soon(api._apply_state)
            api._tg.start_soon(deliver)
            api._tg.start_soon(forward)
            api._tg.start_soon(master.run)
            origin = SystemId()
            for index, event in enumerate(before_launch):
                await local_sender.send(
                    LocalForwarderEvent(
                        origin_idx=index,
                        origin=origin,
                        session=master.session_id,
                        event=event,
                    )
                )
            while master.state.last_event_applied_idx < (
                initial.last_event_applied_idx + len(before_launch)
            ):
                await anyio.sleep(0.001)
            api._tg.start_soon(request)
            while not any(isinstance(command, CreateInstance) for command in sent):
                await anyio.sleep(0.001)
            (launched,) = [
                command for command in sent if isinstance(command, CreateInstance)
            ]
            # The master handles the launch on the state it has
            while launched.instance.instance_id not in master.state.instances:
                await anyio.sleep(0.001)
            lag_over.set()
            while not answers:
                await anyio.sleep(0.001)
            # Time for a launch called off to be undone
            await anyio.sleep(0.5)
            api._tg.cancel_tasks()
    return answers, sent, master


@pytest.mark.parametrize("dip", ["memory_report", "another_instance_loading"])
async def test_a_launch_the_master_accepted_is_kept_whatever_the_api_saw_meanwhile(
    dip: str,
) -> None:
    """The API's view of free memory dips while it waits for the launch, and recovers before
    the master handles it: the launch is answered as started and kept."""
    initial = single_node_state()
    if dip == "memory_report":
        # It fits in 1000 MB. A report of 500 MB, then 1000 MB again
        events: list[Event] = [memory_report(500), memory_report(1000)]
    else:
        # 1000 MB free less 300 MB still loading: 700 MB, so the 600 MB launch fits. The report
        # then shows the 300 MB taken while its runner still says loading (counted twice, 400
        # MB), then the runner says loaded (700 MB again)
        other, runner = loading_instance()
        initial = initial.model_copy(
            update={
                "instances": {other.instance_id: other},
                "runners": {runner: RunnerLoading()},
            }
        )
        events = [
            memory_report(700),
            RunnerStatusUpdated(runner_id=runner, runner_status=RunnerLoaded()),
        ]

    answers, sent, master = await run_with_a_lagging_api(initial, events)

    (launched,) = [command for command in sent if isinstance(command, CreateInstance)]
    assert isinstance(answers[0], CreateInstanceResponse), answers[0]
    assert not [
        command
        for command in sent
        if isinstance(command, DeleteInstance)
        and command.instance_id == launched.instance.instance_id
    ]
    assert launched.instance.instance_id in master.state.instances


def test_a_called_off_launch_is_watched_for_a_late_copy_only_as_long_as_the_wait() -> (
    None
):
    """A launch answered 503 is undone should a copy still apply, until CLUSTER_COMMAND_TIMEOUT
    after the answer. A launch the master refused never applies: without the limit it would
    be checked on every event for the life of the node."""
    api, _ = an_api(single_node_state())
    (old, _), (recent, _) = loading_instance(), loading_instance()
    for launch, answered_at in (
        (old, time.monotonic() - api_main.CLUSTER_COMMAND_TIMEOUT - 1),
        (recent, time.monotonic()),
    ):
        api._withdrawn_cluster_commands[launch.instance_id] = WithdrawnClusterCommand(
            command=CreateInstance(instance=launch),
            undo=DeleteInstance(instance_id=launch.instance_id),
            withdrawn_at=answered_at,
        )

    api._note_applied_cluster_commands()

    assert list(api._withdrawn_cluster_commands) == [recent.instance_id]
