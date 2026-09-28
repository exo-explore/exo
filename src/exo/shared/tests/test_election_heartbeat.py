"""A master re-announces itself so nodes that ended up following a different master
(their election missed its messages) re-run the election instead of staying split."""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass

import anyio
import pytest
from anyio import create_task_group, fail_after

from exo.routing.connection_message import ConnectionMessage
from exo.shared.election import Election, ElectionMessage, ElectionResult
from exo.shared.types.commands import ForwarderCommand
from exo.shared.types.common import NodeId, SessionId
from exo.utils.channels import Receiver, Sender, channel

ME = NodeId("ME")


def em(
    clock: int,
    seniority: int,
    node_id: str,
    heartbeat: bool = False,
    election_clock: int | None = None,
) -> ElectionMessage:
    return ElectionMessage(
        clock=clock,
        seniority=seniority,
        proposed_session=SessionId(
            master_node_id=NodeId(node_id),
            election_clock=clock if election_clock is None else election_clock,
        ),
        commands_seen=0,
        heartbeat=heartbeat,
    )


@pytest.fixture(autouse=True)
def fast_timings(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr("exo.shared.election.DEFAULT_ELECTION_TIMEOUT", 0.1)
    monkeypatch.setattr("exo.shared.election.HEARTBEAT_INTERVAL", 0.05)


@dataclass
class Harness:
    election: Election
    inbound: Sender[ElectionMessage]
    outbound: Receiver[ElectionMessage]
    results: Receiver[ElectionResult]

    async def follow(self, master: ElectionMessage) -> ElectionResult:
        """Run a round that `master` wins."""
        await self.inbound.send(master)
        return await self.result_for(master.clock)

    async def result_for(self, clock: int) -> ElectionResult:
        while True:
            result = await self.results.receive()
            if result.won_clock == clock:
                return result

    def sent(self) -> list[ElectionMessage]:
        return self.outbound.collect()


@asynccontextmanager
async def running_election() -> AsyncIterator[Harness]:
    em_out_tx, em_out_rx = channel[ElectionMessage]()
    em_in_tx, em_in_rx = channel[ElectionMessage]()
    er_tx, er_rx = channel[ElectionResult]()
    cm_tx, cm_rx = channel[ConnectionMessage]()
    co_tx, co_rx = channel[ForwarderCommand]()
    election = Election(
        node_id=ME,
        election_message_receiver=em_in_rx,
        election_message_sender=em_out_tx,
        election_result_sender=er_tx,
        connection_message_receiver=cm_rx,
        command_receiver=co_rx,
        is_candidate=True,
    )
    async with create_task_group() as tg:
        with fail_after(5):
            tg.start_soon(election.run)
            yield Harness(election, em_in_tx, em_out_rx, er_rx)
            em_in_tx.close()
            cm_tx.close()
            co_tx.close()


async def test_master_sends_heartbeats() -> None:
    async with running_election() as h:
        # Every node starts as its own master
        await anyio.sleep(0.3)
        heartbeats = [m for m in h.sent() if m.heartbeat]
        assert len(heartbeats) >= 3
        assert all(m.proposed_session.master_node_id == ME for m in heartbeats)


async def test_follower_does_not_send_heartbeats() -> None:
    async with running_election() as h:
        await h.follow(em(clock=1, seniority=5, node_id="MASTER"))
        h.sent()
        await anyio.sleep(0.3)
        assert [m for m in h.sent() if m.heartbeat] == []


async def test_heartbeat_from_our_master_only_syncs_the_clock() -> None:
    async with running_election() as h:
        master = em(clock=1, seniority=5, node_id="MASTER")
        await h.follow(master)
        h.sent()

        await h.inbound.send(master.model_copy(update={"clock": 7, "heartbeat": True}))
        await anyio.sleep(0.3)

        assert h.election.clock == 7
        assert h.election.current_session == master.proposed_session
        assert h.sent() == []  # no new round


async def test_heartbeat_from_a_more_senior_master_starts_a_new_round() -> None:
    async with running_election() as h:
        await h.follow(em(clock=1, seniority=1, node_id="SPLIT"))

        # Another master, which our round never heard from, re-announces itself
        await h.inbound.send(
            em(clock=1, seniority=5, node_id="REAL", election_clock=0, heartbeat=True)
        )
        # We start round 2; the other master joins it
        while True:
            message = await h.outbound.receive()
            if message.clock == 2 and not message.heartbeat:
                break
        await h.inbound.send(em(clock=2, seniority=5, node_id="REAL", election_clock=0))

        result = await h.result_for(2)
        assert result.session_id.master_node_id == NodeId("REAL")


async def test_heartbeat_from_a_more_junior_master_is_ignored() -> None:
    async with running_election() as h:
        master = em(clock=1, seniority=5, node_id="MASTER")
        await h.follow(master)
        h.sent()

        await h.inbound.send(em(clock=1, seniority=1, node_id="SPLIT", heartbeat=True))
        await anyio.sleep(0.3)

        # That master will hear ours and rejoin; we stay put
        assert h.election.current_session == master.proposed_session
        assert h.election.clock == 1
        assert h.sent() == []


async def test_heartbeats_arriving_together_run_one_round() -> None:
    async with running_election() as h:
        await h.follow(em(clock=1, seniority=1, node_id="SPLIT"))
        h.sent()
        for _ in range(3):
            await h.inbound.send(
                em(
                    clock=1,
                    seniority=5,
                    node_id="REAL",
                    heartbeat=True,
                    election_clock=0,
                )
            )
        await anyio.sleep(0.5)
        rounds = {m.clock for m in h.sent() if not m.heartbeat}
        # Later triggers cancel earlier ones before they start: peers see a single round
        assert len(rounds) == 1
        assert rounds.pop() > 1
