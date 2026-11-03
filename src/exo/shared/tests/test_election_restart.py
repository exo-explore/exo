"""A node keeps its id across restarts. Its new run must not be mistaken for the old one by nodes
that still follow the old run as master: the new run has none of its state, and they would wait
on it forever."""

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

FOLLOWER = NodeId("1f0110e7")
MASTER = NodeId("3a57e7")


@pytest.fixture(autouse=True)
def fast_timings(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr("exo.shared.election.DEFAULT_ELECTION_TIMEOUT", 0.1)
    monkeypatch.setattr("exo.shared.election.HEARTBEAT_INTERVAL", 0.05)
    monkeypatch.setattr("exo.shared.election.MASTER_SILENCE_TIMEOUT", 0.3)


@dataclass
class Harness:
    election: Election
    inbound: Sender[ElectionMessage]
    outbound: Receiver[ElectionMessage]
    results: Receiver[ElectionResult]

    async def result_for(self, clock: int) -> ElectionResult:
        while True:
            result = await self.results.receive()
            if result.won_clock == clock:
                return result


@asynccontextmanager
async def running_election(node_id: NodeId) -> AsyncIterator[Harness]:
    em_out_tx, em_out_rx = channel[ElectionMessage]()
    em_in_tx, em_in_rx = channel[ElectionMessage]()
    er_tx, er_rx = channel[ElectionResult]()
    cm_tx, cm_rx = channel[ConnectionMessage]()
    co_tx, co_rx = channel[ForwarderCommand]()
    election = Election(
        node_id=node_id,
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


async def _start_session_of_a_run(node_id: NodeId) -> SessionId:
    """The session one run of `node_id` is master of when it starts (as it tells others)."""
    async with running_election(node_id) as run:
        first = await run.outbound.receive()
        return first.proposed_session


async def test_each_start_of_a_node_has_its_own_session() -> None:
    sessions = [await _start_session_of_a_run(MASTER) for _ in range(5)]
    assert len(set(sessions)) == len(sessions)
    assert all(session.master_node_id == MASTER for session in sessions)


async def test_a_start_session_never_equals_an_election_rounds_session() -> None:
    # A node that wins a round while not master proposes the round's clock (from 1 up)
    session = await _start_session_of_a_run(MASTER)
    assert session.election_clock < 0


async def test_followers_of_a_master_that_restarted_elect_again() -> None:
    first_run = await _start_session_of_a_run(MASTER)
    async with running_election(FOLLOWER) as follower:
        # The master's first run wins a round while master of the session it started with:
        # that session is the one the cluster then runs in
        await follower.inbound.send(
            ElectionMessage(
                clock=1,
                seniority=5,
                proposed_session=first_run,
                commands_seen=3,
            )
        )
        result = await follower.result_for(1)
        assert result.session_id == first_run

        # The master restarts with the same id and announces itself as master of its new run
        async with running_election(MASTER) as second_run:
            new_master = await second_run.outbound.receive()
            # For longer than MASTER_SILENCE_TIMEOUT
            with anyio.move_on_after(0.6):
                while True:
                    await follower.inbound.send(
                        new_master.model_copy(update={"heartbeat": True})
                    )
                    await anyio.sleep(0.05)

        # The old run is silent, so the follower runs a new election instead of treating the
        # new run as the master whose state it has
        assert follower.election.current_session != new_master.proposed_session
        rounds = [m for m in follower.outbound.collect() if not m.heartbeat]
        assert any(m.clock > 1 for m in rounds)
