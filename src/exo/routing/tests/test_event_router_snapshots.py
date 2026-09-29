"""A node that fell too far behind catches up from a state snapshot sent by the master."""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass

import anyio

from exo.routing.event_router import EventRouter
from exo.shared.types.commands import ForwarderCommand, RequestEventLog
from exo.shared.types.common import NodeId, SessionId, SystemId
from exo.shared.types.events import (
    GlobalForwarderEvent,
    IndexedEvent,
    LocalForwarderEvent,
    StateSnapshot,
    TestEvent,
)
from exo.shared.types.state import State
from exo.utils.channels import Receiver, Sender, channel

SESSION = SessionId(master_node_id=NodeId("master"), election_clock=0)


@dataclass
class Harness:
    router: EventRouter
    events: Sender[GlobalForwarderEvent]
    snapshots: Sender[StateSnapshot]
    commands: Receiver[ForwarderCommand]
    consumer: Receiver[IndexedEvent | StateSnapshot]

    async def send_events(self, *indices: int) -> None:
        for idx in indices:
            await self.events.send(
                GlobalForwarderEvent(
                    origin_idx=idx,
                    origin=SESSION.master_node_id,
                    session=SESSION,
                    event=TestEvent(),
                )
            )
        await anyio.sleep(0.05)

    async def send_snapshot(
        self,
        idx: int,
        requester: SystemId | None = None,
        session: SessionId = SESSION,
    ) -> None:
        await self.snapshots.send(
            StateSnapshot(
                session=session,
                requester=requester or self.router._system_id,  # pyright: ignore[reportPrivateUsage]
                state=State(last_event_applied_idx=idx),
            )
        )
        await anyio.sleep(0.05)

    def received(self) -> list[str]:
        return [
            f"snapshot@{item.state.last_event_applied_idx}"
            if isinstance(item, StateSnapshot)
            else f"event@{item.idx}"
            for item in self.consumer.collect()
        ]


@asynccontextmanager
async def running_router() -> AsyncIterator[Harness]:
    command_send, command_recv = channel[ForwarderCommand]()
    events_send, events_recv = channel[GlobalForwarderEvent]()
    outbound_send, _outbound_recv = channel[LocalForwarderEvent]()
    snapshots_send, snapshots_recv = channel[StateSnapshot]()
    router = EventRouter(
        SESSION,
        command_sender=command_send,
        external_inbound=events_recv,
        external_outbound=outbound_send,
        snapshot_inbound=snapshots_recv,
    )
    consumer = router.receiver()
    async with anyio.create_task_group() as tg:
        tg.start_soon(router.run)
        yield Harness(router, events_send, snapshots_send, command_recv, consumer)
        router.shutdown()


async def test_late_joiner_asks_for_the_log_from_the_start() -> None:
    async with running_router() as harness:
        await harness.send_events(500)
        with anyio.fail_after(2):
            request = await harness.commands.receive()
        assert isinstance(request.command, RequestEventLog)
        assert request.command.since_idx == 0
        assert harness.received() == []


async def test_snapshot_then_buffered_events_are_delivered_in_order() -> None:
    async with running_router() as harness:
        await harness.send_events(10, 11, 12)
        assert harness.received() == []

        await harness.send_snapshot(10)
        assert harness.received() == ["snapshot@10", "event@11", "event@12"]

        await harness.send_events(13)
        assert harness.received() == ["event@13"]


async def test_snapshot_for_another_node_is_ignored() -> None:
    async with running_router() as harness:
        await harness.send_events(10, 11)
        await harness.send_snapshot(10, requester=SystemId())
        assert harness.received() == []


async def test_snapshot_from_another_session_is_ignored() -> None:
    async with running_router() as harness:
        await harness.send_events(10, 11)
        other = SessionId(master_node_id=NodeId("other"), election_clock=1)
        await harness.send_snapshot(10, session=other)
        assert harness.received() == []


async def test_snapshot_we_are_already_past_is_ignored() -> None:
    async with running_router() as harness:
        await harness.send_events(0, 1, 2, 3)
        assert harness.received() == ["event@0", "event@1", "event@2", "event@3"]

        await harness.send_snapshot(2)
        await harness.send_snapshot(3)
        assert harness.received() == []

        await harness.send_events(4)
        assert harness.received() == ["event@4"]
