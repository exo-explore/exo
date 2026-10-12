"""A node follows a new master without tearing down its event router or its consumers."""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass

import anyio

from exo.routing.event_router import EventRouter
from exo.shared.types.commands import ForwarderCommand, RequestEventLog
from exo.shared.types.common import NodeId, SessionId
from exo.shared.types.events import (
    GlobalForwarderEvent,
    IndexedEvent,
    LocalForwarderEvent,
    StateSnapshot,
    TestEvent,
)
from exo.shared.types.state import State
from exo.utils.channels import Receiver, Sender, channel

OLD = SessionId(master_node_id=NodeId("old-master"), election_clock=1)
NEW = SessionId(master_node_id=NodeId("new-master"), election_clock=2)


@dataclass
class Harness:
    router: EventRouter
    events: Sender[GlobalForwarderEvent]
    snapshots: Sender[StateSnapshot]
    commands: Receiver[ForwarderCommand]
    outbound: Receiver[LocalForwarderEvent]
    consumer: Receiver[IndexedEvent | StateSnapshot]

    async def send_events(self, session: SessionId, *indices: int) -> None:
        for idx in indices:
            await self.events.send(
                GlobalForwarderEvent(
                    origin_idx=idx,
                    origin=session.master_node_id,
                    session=session,
                    event=TestEvent(),
                )
            )
        await anyio.sleep(0.05)

    async def send_snapshot(self, session: SessionId, idx: int) -> None:
        await self.snapshots.send(
            StateSnapshot(
                session=session,
                requester=self.router._system_id,  # pyright: ignore[reportPrivateUsage]
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
    outbound_send, outbound_recv = channel[LocalForwarderEvent]()
    snapshots_send, snapshots_recv = channel[StateSnapshot]()
    router = EventRouter(
        OLD,
        command_sender=command_send,
        external_inbound=events_recv,
        external_outbound=outbound_send,
        snapshot_inbound=snapshots_recv,
    )
    consumer = router.receiver()
    async with anyio.create_task_group() as tg:
        tg.start_soon(router.run)
        yield Harness(
            router, events_send, snapshots_send, command_recv, outbound_recv, consumer
        )
        router.shutdown()


async def test_the_node_catches_up_with_a_new_master_from_its_snapshot() -> None:
    async with running_router() as harness:
        await harness.send_events(OLD, 0, 1, 2)
        assert harness.received() == ["event@0", "event@1", "event@2"]

        harness.router.switch_session(NEW)
        # The node asks for the new master's state straight away
        with anyio.fail_after(2):
            request = await harness.commands.receive()
        assert isinstance(request.command, RequestEventLog)
        assert request.command.snapshot

        # The new master numbers its events on from the state it carried over; they
        # wait for its snapshot, since the node's own state is the old session's
        await harness.send_events(NEW, 41, 42)
        assert harness.received() == []

        await harness.send_snapshot(NEW, 40)
        assert harness.received() == ["snapshot@40", "event@41", "event@42"]


async def test_a_new_cluster_master_s_events_from_zero_also_wait_for_its_snapshot() -> (
    None
):
    async with running_router() as harness:
        await harness.send_events(OLD, 0, 1, 2)
        harness.consumer.collect()

        harness.router.switch_session(NEW)
        await harness.send_events(NEW, 0, 1)
        # Applying them on top of the old session's state would fail
        assert harness.received() == []

        await harness.send_snapshot(NEW, -1)
        assert harness.received() == ["snapshot@-1", "event@0", "event@1"]


async def test_events_from_the_old_master_are_ignored_after_the_switch() -> None:
    async with running_router() as harness:
        harness.router.switch_session(NEW)
        await harness.send_events(OLD, 0, 1)
        await harness.send_snapshot(NEW, -1)
        await harness.send_events(NEW, 0)

        assert harness.received() == ["snapshot@-1", "event@0"]


async def test_each_sender_numbers_its_events_from_zero_for_the_new_master() -> None:
    async with running_router() as harness:
        sender = harness.router.sender()
        await sender.send(TestEvent())
        await sender.send(TestEvent())
        await anyio.sleep(0.05)
        harness.router.switch_session(NEW)
        await sender.send(TestEvent())
        await anyio.sleep(0.05)

        sent = harness.outbound.collect()
        assert [(e.session, e.origin_idx) for e in sent] == [
            (OLD, 0),
            (OLD, 1),
            (NEW, 0),
        ]
        assert sent[0].origin == sent[1].origin != sent[2].origin


async def test_returning_to_an_earlier_master_does_not_reuse_its_numbering() -> None:
    # e.g. a master that froze, was replaced, and won the cluster back when it recovered:
    # it still expects index 2 from the origin it knew, and would drop (or clash with) a
    # second index 0 from it
    async with running_router() as harness:
        sender = harness.router.sender()
        await sender.send(TestEvent())
        await sender.send(TestEvent())
        await anyio.sleep(0.05)
        harness.router.switch_session(NEW)
        await sender.send(TestEvent())
        await anyio.sleep(0.05)
        harness.router.switch_session(OLD)
        await sender.send(TestEvent())
        await anyio.sleep(0.05)

        sent = harness.outbound.collect()
        first, back = sent[0], sent[-1]
        assert back.session == OLD and back.origin_idx == 0
        assert back.origin != first.origin


async def test_events_the_old_master_never_acknowledged_are_not_resent() -> None:
    async with running_router() as harness:
        sender = harness.router.sender()
        await sender.send(TestEvent())
        await anyio.sleep(0.05)
        assert harness.router.out_for_delivery

        harness.router.switch_session(NEW)

        assert harness.router.out_for_delivery == {}
