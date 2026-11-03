"""Elections start a round on what they knew as a connection change: a node that becomes Alive
counts as "connected", one that leaves Alive as "disconnected", and nothing else starts one."""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import UTC, datetime

import anyio
import pytest
from anyio import create_task_group, fail_after
from exo_rs import MembershipSnapshot

from exo.routing.membership import (
    LossReason,
    Member,
    Membership,
    MembershipChange,
    MemberState,
)
from exo.shared.election import (
    Election,
    ElectionMessage,
    ElectionResult,
    connection_change,
)
from exo.shared.types.commands import ForwarderCommand
from exo.shared.types.common import NodeId
from exo.utils.channels import Receiver, Sender, channel

ME = NodeId("ME")
PEER = NodeId("PEER")


def change(
    previous: MemberState | None, current: MemberState, node: NodeId = PEER
) -> MembershipChange:
    reason: LossReason | None = (
        "Disconnected" if current in ("Suspect", "Gone") else None
    )
    return MembershipChange(
        node_id=node, previous=previous, current=current, reason=reason
    )


@pytest.mark.parametrize(
    ("previous", "current", "expected"),
    [
        # A node joins, rejoins, or is reachable again: what was "connected"
        (None, "Alive", "connected"),
        ("Suspect", "Alive", "connected"),
        ("Gone", "Alive", "connected"),
        # A node becomes unreachable, or leaves: what was "disconnected"
        ("Alive", "Suspect", "disconnected"),
        ("Alive", "Gone", "disconnected"),
        # New with membership states: these start no round
        ("Suspect", "Gone", None),
        ("Suspect", "Suspect", None),
        ("Gone", "Gone", None),
        (None, "Suspect", None),
        (None, "Gone", None),
    ],
)
def test_each_membership_change_is_mapped_to_what_elections_knew(
    previous: MemberState | None, current: MemberState, expected: str | None
) -> None:
    assert connection_change(change(previous, current)) == expected


@pytest.mark.parametrize(
    ("previous", "current", "expected"),
    [
        # This node starting is "connected", as its own token appearing was
        (None, "Alive", "connected"),
        # A stall of this node, and its end, start nothing
        ("Alive", "Resuming", None),
        ("Resuming", "Alive", None),
    ],
)
def test_this_nodes_own_states_are_mapped_too(
    previous: MemberState | None, current: MemberState, expected: str | None
) -> None:
    assert connection_change(change(previous, current, node=ME)) == expected


@pytest.fixture(autouse=True)
def fast_timings(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr("exo.shared.election.DEFAULT_ELECTION_TIMEOUT", 0.1)


@dataclass
class Harness:
    election: Election
    membership: Sender[Member]
    outbound: Receiver[ElectionMessage]

    async def report(self, node: NodeId, state: MemberState) -> None:
        reason: LossReason | None = (
            "Disconnected" if state in ("Suspect", "Gone") else None
        )
        await self.membership.send(
            Member(node_id=node, state=state, reason=reason, since=datetime.now(tz=UTC))
        )

    async def rounds_after(self, report: tuple[NodeId, MemberState]) -> list[int]:
        """The rounds this node starts within a second of a membership report"""
        self.outbound.collect()
        await self.report(*report)
        await anyio.sleep(0.6)
        return sorted({message.clock for message in self.outbound.collect()})


def no_snapshot() -> MembershipSnapshot:
    raise AssertionError("elections don't read the snapshot")


@asynccontextmanager
async def election_on_membership() -> AsyncIterator[Harness]:
    """An election fed by a running Membership, as on a node"""
    member_tx, member_rx = channel[Member]()
    membership = Membership(
        ME,
        member_receiver=member_rx,
        snapshot=no_snapshot,
    )
    em_out_tx, em_out_rx = channel[ElectionMessage]()
    em_in_tx, em_in_rx = channel[ElectionMessage]()
    er_tx, _er_rx = channel[ElectionResult]()
    co_tx, co_rx = channel[ForwarderCommand]()
    election = Election(
        node_id=ME,
        election_message_receiver=em_in_rx,
        election_message_sender=em_out_tx,
        election_result_sender=er_tx,
        membership_change_receiver=membership.changes(),
        command_receiver=co_rx,
        is_candidate=True,
    )
    async with create_task_group() as tg:
        with fail_after(10):
            tg.start_soon(membership.run)
            tg.start_soon(election.run)
            await anyio.sleep(0.2)
            yield Harness(election, member_tx, em_out_rx)
            member_tx.close()
            em_in_tx.close()
            co_tx.close()


async def test_a_node_joining_or_becoming_unreachable_starts_a_round() -> None:
    async with election_on_membership() as h:
        assert await h.rounds_after((PEER, "Alive")) == [1]
        assert await h.rounds_after((PEER, "Suspect")) == [2]
        assert await h.rounds_after((PEER, "Alive")) == [3]


async def test_a_suspect_node_becoming_gone_starts_no_round() -> None:
    async with election_on_membership() as h:
        await h.rounds_after((PEER, "Alive"))
        await h.rounds_after((PEER, "Suspect"))
        assert await h.rounds_after((PEER, "Gone")) == []


async def test_a_peer_lost_while_this_node_resumes_still_starts_a_round() -> None:
    async with election_on_membership() as h:
        await h.rounds_after((PEER, "Alive"))
        # This node was stopped (asleep, frozen): that alone starts nothing...
        assert await h.rounds_after((ME, "Resuming")) == []
        # ... but its master (or any peer) being lost meanwhile is acted on at once
        assert await h.rounds_after((PEER, "Suspect")) == [2]
        assert await h.rounds_after((ME, "Alive")) == []
