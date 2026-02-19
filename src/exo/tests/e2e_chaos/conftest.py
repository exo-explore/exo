"""Shared fixtures and helpers for E2E chaos/networking tests.

Provides a ``MiniCluster`` that wires Master + Worker(s) + Election together
using in-process channels.  No Docker, no network, no GPU -- pure async
integration testing of the coordination layer.
"""

from collections.abc import Iterator
from datetime import datetime, timezone
from typing import Final

import pytest
from _pytest.logging import LogCaptureFixture
from loguru import logger

from exo.master.main import Master
from exo.shared.models.model_cards import ModelCard, ModelTask
from exo.shared.types.commands import (
    CommandId,
    ForwarderCommand,
    ForwarderDownloadCommand,
    PlaceInstance,
    TextGeneration,
)
from exo.shared.types.common import ModelId, NodeId, SessionId
from exo.shared.types.events import (
    ForwarderEvent,
    IndexedEvent,
    NodeGatheredInfo,
    TopologyEdgeCreated,
)
from exo.shared.types.memory import Memory
from exo.shared.types.multiaddr import Multiaddr
from exo.shared.types.profiling import MemoryUsage
from exo.shared.types.text_generation import InputMessage, TextGenerationTaskParams
from exo.shared.types.topology import Connection, SocketConnection
from exo.shared.types.worker.instances import InstanceMeta
from exo.shared.types.worker.shards import Sharding
from exo.utils.channels import Receiver, Sender, channel
from exo.worker.main import Worker

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

TEST_MODEL_ID: Final[ModelId] = ModelId("test-model/chaos-test-1b")
TEST_MODEL_CARD: Final[ModelCard] = ModelCard(
    model_id=TEST_MODEL_ID,
    n_layers=16,
    storage_size=Memory.from_bytes(678_948),
    hidden_size=2048,
    supports_tensor=True,
    tasks=[ModelTask.TextGeneration],
)

FAST_ELECTION_TIMEOUT: Final[float] = 0.1


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_node_id(label: str) -> NodeId:
    return NodeId(f"node-{label}")


def make_session_id(master_node: NodeId) -> SessionId:
    return SessionId(master_node_id=master_node, election_clock=0)


def make_memory_info() -> MemoryUsage:
    return MemoryUsage(
        ram_total=Memory.from_bytes(16 * 1024 * 1024 * 1024),
        ram_available=Memory.from_bytes(8 * 1024 * 1024 * 1024),
        swap_total=Memory.from_bytes(0),
        swap_available=Memory.from_bytes(0),
    )


def make_gathered_info_event(
    node_id: NodeId, sender_id: NodeId, session_id: SessionId, origin_idx: int
) -> ForwarderEvent:
    return ForwarderEvent(
        origin_idx=origin_idx,
        origin=sender_id,
        session=session_id,
        event=NodeGatheredInfo(
            when=str(datetime.now(tz=timezone.utc)),
            node_id=node_id,
            info=make_memory_info(),
        ),
    )


def make_topology_edge_event(
    source: NodeId,
    sink: NodeId,
    sender_id: NodeId,
    session_id: SessionId,
    origin_idx: int,
    ip_suffix: int = 1,
) -> ForwarderEvent:
    """Create a ForwarderEvent wrapping a TopologyEdgeCreated event."""
    return ForwarderEvent(
        origin_idx=origin_idx,
        origin=sender_id,
        session=session_id,
        event=TopologyEdgeCreated(
            conn=Connection(
                source=source,
                sink=sink,
                edge=SocketConnection(
                    sink_multiaddr=Multiaddr(
                        address=f"/ip4/10.0.0.{ip_suffix}/tcp/52415"
                    )
                ),
            )
        ),
    )


class EventCollector:
    """Collects ForwarderEvents from a global event receiver."""

    def __init__(self, receiver: Receiver[ForwarderEvent]) -> None:
        self._receiver = receiver
        self.indexed_events: list[IndexedEvent] = []

    def collect(self) -> list[IndexedEvent]:
        raw = self._receiver.collect()
        for fe in raw:
            self.indexed_events.append(
                IndexedEvent(event=fe.event, idx=len(self.indexed_events))
            )
        return self.indexed_events

    async def wait_for_event_count(
        self, count: int, *, timeout: float = 5.0, poll_interval: float = 0.01
    ) -> list[IndexedEvent]:
        import anyio

        with anyio.fail_after(timeout):
            while len(self.collect()) < count:
                await anyio.sleep(poll_interval)
        return self.indexed_events


class MiniCluster:
    """An in-process cluster with one Master and N Workers wired via channels.

    No networking, no real model loading -- exercises the coordination logic
    (event sourcing, command routing, election) in a deterministic, fast
    test harness.
    """

    def __init__(self, node_count: int = 2) -> None:
        self.node_count = node_count
        self.master_node_id = make_node_id("master")
        self.session_id = make_session_id(self.master_node_id)

        # -- shared bus channels --
        self.global_event_sender: Sender[ForwarderEvent]
        self.global_event_internal_receiver: Receiver[ForwarderEvent]
        self.global_event_sender, self.global_event_internal_receiver = channel[
            ForwarderEvent
        ]()

        self.command_sender: Sender[ForwarderCommand]
        self.command_receiver: Receiver[ForwarderCommand]
        self.command_sender, self.command_receiver = channel[ForwarderCommand]()

        self.local_event_sender: Sender[ForwarderEvent]
        self.local_event_receiver: Receiver[ForwarderEvent]
        self.local_event_sender, self.local_event_receiver = channel[ForwarderEvent]()

        self.download_cmd_sender: Sender[ForwarderDownloadCommand]
        self._download_cmd_receiver: Receiver[ForwarderDownloadCommand]
        self.download_cmd_sender, self._download_cmd_receiver = channel[
            ForwarderDownloadCommand
        ]()

        # -- event collector (taps global events) --
        self.event_collector = EventCollector(
            self.global_event_internal_receiver.clone()
        )

        # -- master --
        self.master = Master(
            self.master_node_id,
            self.session_id,
            global_event_sender=self.global_event_sender.clone(),
            local_event_receiver=self.local_event_receiver.clone(),
            command_receiver=self.command_receiver.clone(),
            download_command_sender=self.download_cmd_sender.clone(),
        )

        # -- workers --
        self.worker_node_ids: list[NodeId] = []
        self.workers: list[Worker] = []
        for i in range(node_count):
            wid = make_node_id(f"worker-{i}")
            self.worker_node_ids.append(wid)

            counter: Iterator[int] = iter(range(1_000_000))
            worker = Worker(
                wid,
                self.session_id,
                global_event_receiver=self.global_event_internal_receiver.clone(),
                local_event_sender=self.local_event_sender.clone(),
                command_sender=self.command_sender.clone(),
                download_command_sender=self.download_cmd_sender.clone(),
                event_index_counter=counter,
            )
            self.workers.append(worker)

    async def inject_node_info(self, node_id: NodeId, sender_suffix: str = "") -> None:
        """Inject a NodeGatheredInfo event for a node into the local event bus."""
        sender_id = NodeId(f"{node_id}_sender{sender_suffix}")
        await self.local_event_sender.send(
            make_gathered_info_event(node_id, sender_id, self.session_id, 0)
        )

    async def wait_for_topology_nodes(
        self, count: int, *, timeout: float = 5.0
    ) -> None:
        import anyio

        with anyio.fail_after(timeout):
            while len(list(self.master.state.topology.list_nodes())) < count:
                await anyio.sleep(0.01)

    async def place_model(
        self,
        model_card: ModelCard | None = None,
        min_nodes: int = 1,
    ) -> None:
        card = model_card or TEST_MODEL_CARD
        await self.command_sender.send(
            ForwarderCommand(
                origin=self.master_node_id,
                command=PlaceInstance(
                    command_id=CommandId(),
                    model_card=card,
                    sharding=Sharding.Pipeline,
                    instance_meta=InstanceMeta.MlxRing,
                    min_nodes=min_nodes,
                ),
            )
        )

    async def wait_for_instances(self, count: int, *, timeout: float = 5.0) -> None:
        import anyio

        with anyio.fail_after(timeout):
            while len(self.master.state.instances) < count:
                await anyio.sleep(0.01)

    async def send_chat(
        self,
        message: str,
        model: ModelId | None = None,
    ) -> CommandId:
        cmd_id = CommandId()
        await self.command_sender.send(
            ForwarderCommand(
                origin=self.master_node_id,
                command=TextGeneration(
                    command_id=cmd_id,
                    task_params=TextGenerationTaskParams(
                        model=model or TEST_MODEL_ID,
                        input=[InputMessage(role="user", content=message)],
                    ),
                ),
            )
        )
        return cmd_id

    async def shutdown_master(self) -> None:
        await self.master.shutdown()

    def shutdown_workers(self) -> None:
        for w in self.workers:
            w.shutdown()


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def fast_election_timeout(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("exo.shared.election.DEFAULT_ELECTION_TIMEOUT", 0.1)


@pytest.fixture
def caplog(caplog: LogCaptureFixture) -> Iterator[LogCaptureFixture]:
    handler_id = logger.add(
        caplog.handler,
        format="{message}",
        level=0,
        filter=lambda record: record["level"].no >= caplog.handler.level,
        enqueue=True,
    )
    yield caplog
    logger.remove(handler_id)
