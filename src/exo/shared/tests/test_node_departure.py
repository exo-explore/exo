"""A node that is quitting announces its departure before it stops."""

import time
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path
from typing import cast

import anyio
import pytest
from exo_rs import NetworkingHandle

import exo.download.store as download_store
import exo.main as exo_main
from exo.download.coordinator import DownloadCoordinator
from exo.download.transfer import DownloadSettings
from exo.main import Node
from exo.master.main import Master
from exo.routing import topics
from exo.routing.event_router import EventRouter
from exo.routing.router import Router
from exo.shared.apply import apply
from exo.shared.models.model_cards import ModelCard, ModelId, ModelTask, card_cache
from exo.shared.topology import Topology
from exo.shared.types.backends import Backend
from exo.shared.types.commands import ForwarderDownloadCommand, StartDownload
from exo.shared.types.common import NodeId, SessionId, SystemId
from exo.shared.types.events import (
    Event,
    GlobalForwarderEvent,
    IndexedEvent,
    NodeTimedOut,
    TestEvent,
)
from exo.shared.types.memory import Memory
from exo.shared.types.state import State
from exo.shared.types.worker.downloads import DownloadFailed
from exo.shared.types.worker.shards import PipelineShardMetadata
from exo.utils.channels import Receiver, Sender, channel
from exo.utils.task_group import TaskGroup
from exo.worker.main import Worker

NODE = NodeId("quitting-node")


def listed() -> State:
    topology = Topology()
    topology.add_node(NODE)
    return State(topology=topology, last_seen={NODE: datetime.now(tz=timezone.utc)})


class FakeWorker:
    def __init__(self) -> None:
        self.state = listed()
        self.reporting = True

    def stop_reporting(self) -> None:
        self.reporting = False


class FakeEventRouter:
    def __init__(self) -> None:
        self.send: Sender[Event]
        self.events: Receiver[Event]
        self.send, self.events = channel[Event]()

    def sender(self) -> Sender[Event]:
        return self.send.clone()


def node_with(
    worker: FakeWorker | None,
    router: FakeEventRouter,
    *,
    download_coordinator: DownloadCoordinator | None = None,
    is_master: bool = False,
) -> Node:
    node = object.__new__(Node)
    node.worker = cast(Worker, cast(object, worker))
    node.event_router = cast(EventRouter, cast(object, router))
    node.download_coordinator = download_coordinator
    # Only whether there is one matters to a quitting node
    node.master = cast(Master, object()) if is_master else None
    node.node_id = NODE
    node._tg = TaskGroup()  # pyright: ignore[reportPrivateUsage]
    node._leaving = False  # pyright: ignore[reportPrivateUsage]
    return node


async def until(condition: Callable[[], bool]) -> None:
    with anyio.fail_after(5):
        while not condition():
            await anyio.sleep(0.01)


async def run_until_stopped(node: Node, during: list[str]) -> None:
    """Runs the node's task group with one stand-in task until the node stops it."""

    async def component() -> None:
        try:
            await anyio.sleep_forever()
        finally:
            during.append("stopped")

    async with node._tg as tg:  # pyright: ignore[reportPrivateUsage]
        tg.start_soon(component)
        await anyio.sleep(0)
        node.shutdown()


async def test_a_quitting_node_announces_its_departure_before_stopping() -> None:
    worker, router = FakeWorker(), FakeEventRouter()
    node = node_with(worker, router)
    during: list[str] = []

    async def master() -> None:
        # The master receives the departure and the node sees itself removed
        event = await router.events.receive()
        assert event == NodeTimedOut(node_id=NODE, event_id=event.event_id)
        assert not worker.reporting, "the worker could add the node back"
        assert during == [], "the node stopped before its departure was applied"
        during.append("departure applied")
        worker.state = State()

    with anyio.fail_after(2):
        async with anyio.create_task_group() as tg:
            tg.start_soon(master)
            await run_until_stopped(node, during)

    assert during == ["departure applied", "stopped"]


async def test_an_unconfirmed_departure_holds_the_quit_up_only_briefly(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(exo_main, "LEAVE_ANNOUNCE_TIMEOUT", 0.2)
    node = node_with(FakeWorker(), FakeEventRouter())
    during: list[str] = []

    with anyio.fail_after(1):
        await run_until_stopped(node, during)

    assert during == ["stopped"]


async def test_a_node_without_a_worker_stops_at_once() -> None:
    router = FakeEventRouter()
    node = node_with(None, router)
    during: list[str] = []

    with anyio.fail_after(0.5):
        await run_until_stopped(node, during)

    assert during == ["stopped"]
    assert router.events.collect() == []


def test_a_second_quit_signal_exits_at_once() -> None:
    node = node_with(FakeWorker(), FakeEventRouter())
    node._leaving = True  # pyright: ignore[reportPrivateUsage]

    with pytest.raises(SystemExit):
        node.shutdown()


# Nothing the node reports after its departure puts it back into the cluster's state

MODEL = ModelCard(
    model_id=ModelId("test-org/leaving-model"),
    storage_size=Memory.from_kb(1000),
    n_layers=2,
    hidden_size=64,
    supports_tensor=False,
    tasks=[ModelTask.TextGeneration],
    backends=[Backend.MlxMetal],
)
SHARD = PipelineShardMetadata(
    model_card=MODEL,
    device_rank=0,
    world_size=1,
    start_layer=0,
    end_layer=MODEL.n_layers,
    n_layers=MODEL.n_layers,
)


@pytest.fixture
def no_models_on_disk(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Models in an empty directory, and only MODEL known to the node."""
    monkeypatch.setattr(download_store, "EXO_DEFAULT_MODELS_DIR", tmp_path)
    monkeypatch.setattr(download_store, "EXO_MODELS_DIRS", (tmp_path,))
    monkeypatch.setattr(download_store, "EXO_MODELS_READ_ONLY_DIRS", ())
    monkeypatch.setattr(card_cache, "cc", {MODEL.model_id: MODEL})


@pytest.mark.usefixtures("no_models_on_disk")
async def test_a_departed_node_leaves_no_download_entries_behind() -> None:
    """The node's download coordinator keeps running while the node leaves. A status it
    sends after the departure is applied (here: the worker, still seeing the node's
    instance, asks for its model) would list the node in the cluster's downloads again,
    without it being in the topology or last_seen."""
    worker, router = FakeWorker(), FakeEventRouter()
    download_commands, download_command_receiver = channel[ForwarderDownloadCommand]()
    coordinator = DownloadCoordinator(
        NODE,
        DownloadSettings(),
        download_command_receiver=download_command_receiver,
        event_sender=router.sender(),
        offline=True,
    )
    node = node_with(worker, router, download_coordinator=coordinator)
    cluster = worker.state

    async def master() -> None:
        """Applies what the node sends, as the master would, and tells the node."""
        nonlocal cluster
        index = 0
        with router.events as events:
            async for event in events:
                cluster = apply(cluster, IndexedEvent(idx=index, event=event))
                index += 1
                worker.state = cluster
                if isinstance(event, NodeTimedOut):
                    await download_commands.send(
                        ForwarderDownloadCommand(
                            origin=SystemId(),
                            command=StartDownload(
                                target_node_id=NODE, shard_metadata=SHARD
                            ),
                        )
                    )

    def model_status() -> object:
        model = coordinator._models.get(MODEL.model_id)  # pyright: ignore[reportPrivateUsage]
        return None if model is None else model.status

    with anyio.fail_after(5):
        async with anyio.create_task_group() as tg:
            tg.start_soon(coordinator.run)
            tg.start_soon(master)
            # The coordinator reports what is on disk as the node starts
            await until(lambda: NODE in cluster.downloads)

            await run_until_stopped(node, [])
            # The coordinator handles the request: offline, the download fails
            await until(lambda: isinstance(model_status(), DownloadFailed))
            await anyio.sleep(0.1)
            tg.cancel_scope.cancel()

    assert not cluster.topology.contains_node(NODE)
    assert NODE not in cluster.last_seen
    assert NODE not in cluster.downloads, "a download entry for a departed node"


# LEAVE_FLUSH: a quitting master gives the departure it indexed time to leave the process


async def time_from_departure_to_stop(*, is_master: bool) -> float:
    worker, router = FakeWorker(), FakeEventRouter()
    node = node_with(worker, router, is_master=is_master)
    applied_at: list[float] = []
    stopped_at: list[float] = []

    async def master() -> None:
        await router.events.receive()
        worker.state = State()
        applied_at.append(time.monotonic())

    async def component() -> None:
        try:
            await anyio.sleep_forever()
        finally:
            stopped_at.append(time.monotonic())

    with anyio.fail_after(2):
        async with anyio.create_task_group() as tg:
            tg.start_soon(master)
            async with node._tg as node_tg:  # pyright: ignore[reportPrivateUsage]
                node_tg.start_soon(component)
                await anyio.sleep(0)
                node.shutdown()
    return stopped_at[0] - applied_at[0]


async def test_a_quitting_master_stays_up_for_the_flush_and_a_follower_does_not() -> (
    None
):
    """The departure a follower sees came from the master, which has sent it to all."""
    assert await time_from_departure_to_stop(is_master=True) >= exo_main.LEAVE_FLUSH
    assert await time_from_departure_to_stop(is_master=False) < exo_main.LEAVE_FLUSH / 2


# What LEAVE_FLUSH's derivation assumes: handing a message to zenoh takes up to 1 ms, and a
# master that keeps up with its events has far fewer than this many queued for the network
HANDING_OVER = 0.001
QUEUED_AHEAD = 100


class SlowNetwork:
    """A networking handle that takes HANDING_OVER for each message it is given."""

    def __init__(self) -> None:
        self.published: list[bytes] = []

    async def gossipsub_subscribe(self, topic: str, high_priority: bool) -> bool:
        return True

    async def gossipsub_unsubscribe(self, topic: str) -> bool:
        return True

    async def gossipsub_publish(self, topic: str, data: bytes) -> None:
        await anyio.sleep(HANDING_OVER)
        self.published.append(data)

    async def recv(self) -> object:
        await anyio.sleep_forever()


async def test_the_departure_a_quitting_master_indexed_leaves_within_the_flush() -> (
    None
):
    """A master's worker sees the departure the master indexed before its router has handed
    it to the network, behind the events queued before it. With QUEUED_AHEAD of them at the
    measured cost, it is handed over within LEAVE_FLUSH, so stopping then doesn't lose it."""
    network = SlowNetwork()
    router = Router(cast(NetworkingHandle, cast(object, network)))
    await router.register_topic(topics.GLOBAL_EVENTS)
    to_worker = router.receiver(topics.GLOBAL_EVENTS)
    from_master = router.sender(topics.GLOBAL_EVENTS)
    session = SessionId(master_node_id=NODE, election_clock=0)

    def indexed(index: int, event: Event) -> GlobalForwarderEvent:
        return GlobalForwarderEvent(
            origin_idx=index, origin=NODE, session=session, event=event
        )

    departure = indexed(QUEUED_AHEAD, NodeTimedOut(node_id=NODE))
    departure_sent = topics.GLOBAL_EVENTS.serialize(departure)
    with anyio.fail_after(5):
        async with anyio.create_task_group() as tg:
            tg.start_soon(router.run)
            for index in range(QUEUED_AHEAD):
                await from_master.send(indexed(index, TestEvent()))
            await from_master.send(departure)
            while (await to_worker.receive()) != departure:
                pass
            seen = time.monotonic()
            # There is something to wait for
            assert departure_sent not in network.published
            await until(lambda: departure_sent in network.published)
            assert time.monotonic() - seen < exo_main.LEAVE_FLUSH
            tg.cancel_scope.cancel()
