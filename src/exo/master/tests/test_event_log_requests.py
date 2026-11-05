"""The master replays recent events to nodes that missed a few, and sends a state
snapshot to nodes too far behind to replay."""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from unittest.mock import patch

import anyio

from exo.master.main import Master
from exo.routing.router import get_node_zid
from exo.routing.tests.fake_membership import FakeMembership
from exo.shared.types.commands import (
    ForwarderCommand,
    ForwarderDownloadCommand,
    RequestEventLog,
)
from exo.shared.types.common import SessionId, SystemId
from exo.shared.types.events import (
    Event,
    GlobalForwarderEvent,
    LocalForwarderEvent,
    StateSnapshot,
    TestEvent,
)
from exo.utils.channels import Receiver, Sender, channel

N_EVENTS = 12
REPLAYABLE = 5


@dataclass
class Harness:
    master: Master
    session: SessionId
    commands: Sender[ForwarderCommand]
    global_events: Receiver[GlobalForwarderEvent]
    snapshots: Receiver[StateSnapshot]

    async def request(self, since_idx: int, requester: SystemId) -> None:
        await self.commands.send(
            ForwarderCommand(
                origin=requester, command=RequestEventLog(since_idx=since_idx)
            )
        )
        await anyio.sleep(0.05)


@asynccontextmanager
async def running_master(events: list[Event] | None = None) -> AsyncIterator[Harness]:
    events = events if events is not None else [TestEvent() for _ in range(N_EVENTS)]
    node_id = get_node_zid()
    session = SessionId(master_node_id=node_id, election_clock=0)
    global_send, global_recv = channel[GlobalForwarderEvent]()
    command_send, command_recv = channel[ForwarderCommand]()
    local_send, local_recv = channel[LocalForwarderEvent]()
    download_send, _download_recv = channel[ForwarderDownloadCommand]()
    snapshot_send, snapshot_recv = channel[StateSnapshot]()

    with patch("exo.master.main.REPLAYABLE_EVENTS", REPLAYABLE):
        master = Master(
            node_id,
            session,
            global_event_sender=global_send,
            snapshot_sender=snapshot_send,
            local_event_receiver=local_recv,
            command_receiver=command_recv,
            download_command_sender=download_send,
            membership=FakeMembership(node_id),
        )
    async with anyio.create_task_group() as tg:
        tg.start_soon(master.run)
        worker = SystemId()
        for i, event in enumerate(events):
            await local_send.send(
                LocalForwarderEvent(
                    origin_idx=i, origin=worker, session=session, event=event
                )
            )
        with anyio.fail_after(5):
            while master.state.last_event_applied_idx < len(events) - 1:
                await anyio.sleep(0.01)
        global_recv.collect()  # drop the live broadcast of those events

        yield Harness(master, session, command_send, global_recv, snapshot_recv)

        await master.shutdown()


async def test_recent_events_are_replayed() -> None:
    async with running_master() as harness:
        await harness.request(since_idx=N_EVENTS - 3, requester=SystemId())

        replayed = harness.global_events.collect()
        assert [e.origin_idx for e in replayed] == [
            N_EVENTS - 3,
            N_EVENTS - 2,
            N_EVENTS - 1,
        ]
        assert harness.snapshots.collect() == []


async def test_oldest_retained_event_is_still_replayed() -> None:
    async with running_master() as harness:
        await harness.request(since_idx=N_EVENTS - REPLAYABLE, requester=SystemId())

        replayed = harness.global_events.collect()
        assert [e.origin_idx for e in replayed] == list(
            range(N_EVENTS - REPLAYABLE, N_EVENTS)
        )
        assert harness.snapshots.collect() == []


async def test_node_too_far_behind_gets_a_snapshot() -> None:
    async with running_master() as harness:
        requester = SystemId()
        await harness.request(since_idx=0, requester=requester)

        assert harness.global_events.collect() == []
        snapshots = harness.snapshots.collect()
        assert len(snapshots) == 1
        snapshot = snapshots[0]
        assert snapshot.requester == requester
        assert snapshot.session == harness.session
        assert snapshot.state.last_event_applied_idx == N_EVENTS - 1
        assert snapshot.state is harness.master.state


async def test_request_from_the_future_is_ignored() -> None:
    async with running_master() as harness:
        await harness.request(since_idx=N_EVENTS + 10, requester=SystemId())

        assert harness.global_events.collect() == []
        assert harness.snapshots.collect() == []


async def test_a_snapshot_is_sent_when_asked_for_even_if_the_events_could_be_replayed() -> (
    None
):
    async with running_master() as harness:
        requester = SystemId()
        await harness.commands.send(
            ForwarderCommand(
                origin=requester,
                command=RequestEventLog(since_idx=0, snapshot=True),
            )
        )
        with anyio.fail_after(5):
            snapshot = await harness.snapshots.receive()

        assert snapshot.requester == requester
        assert snapshot.state.last_event_applied_idx == N_EVENTS - 1
        assert harness.global_events.collect() == []
