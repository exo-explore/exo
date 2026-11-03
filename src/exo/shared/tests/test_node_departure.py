"""A node that is quitting announces its departure before it stops."""

from datetime import datetime, timezone
from typing import cast

import anyio
import pytest

import exo.main as exo_main
from exo.main import Node
from exo.routing.event_router import EventRouter
from exo.shared.topology import Topology
from exo.shared.types.common import NodeId
from exo.shared.types.events import Event, NodeTimedOut
from exo.shared.types.state import State
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


def node_with(worker: FakeWorker | None, router: FakeEventRouter) -> Node:
    node = object.__new__(Node)
    node.worker = cast(Worker, cast(object, worker))
    node.event_router = cast(EventRouter, cast(object, router))
    node.node_id = NODE
    node._tg = TaskGroup()  # pyright: ignore[reportPrivateUsage]
    node._leaving = False  # pyright: ignore[reportPrivateUsage]
    return node


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
