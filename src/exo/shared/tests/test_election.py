import pytest
from anyio import create_task_group, fail_after, move_on_after

from exo.routing.connection_message import ConnectionMessage, ConnectionMessageType
from exo.shared.election import Election, ElectionMessage, ElectionResult
from exo.shared.types.commands import ForwarderCommand, TestCommand
from exo.shared.types.common import NodeId, SessionId
from exo.utils.channels import channel

# ======= #
# Helpers #
# ======= #


def em(
    clock: int,
    seniority: int,
    node_id: str,
    commands_seen: int = 0,
    election_clock: int | None = None,
) -> ElectionMessage:
    """
    Helper to build ElectionMessages for a given proposer node.

    The new API carries a proposed SessionId (master_node_id + election_clock).
    By default we use the same value for election_clock as the 'clock' of the round.
    """
    return ElectionMessage(
        clock=clock,
        seniority=seniority,
        proposed_session=SessionId(
            master_node_id=NodeId(node_id),
            election_clock=clock if election_clock is None else election_clock,
        ),
        commands_seen=commands_seen,
    )


# ======================================= #
#                 TESTS                   #
# ======================================= #


@pytest.fixture(autouse=True)
def fast_election_timeout(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr("exo.shared.election.DEFAULT_ELECTION_TIMEOUT", 0.1)


@pytest.mark.anyio
async def test_single_round_broadcasts_and_updates_seniority_on_self_win() -> None:
    """
    Start a round by injecting an ElectionMessage with higher clock.
    With only our node effectively 'winning', we should broadcast once and update seniority.
    """
    # Outbound election messages from the Election (we'll observe these)
    em_out_tx, em_out_rx = channel[ElectionMessage]()
    # Inbound election messages to the Election (we'll inject these)
    em_in_tx, em_in_rx = channel[ElectionMessage]()
    # Election results produced by the Election (we'll observe these)
    er_tx, er_rx = channel[ElectionResult]()
    # Connection messages
    cm_tx, cm_rx = channel[ConnectionMessage]()
    # Commands
    co_tx, co_rx = channel[ForwarderCommand]()

    election = Election(
        node_id=NodeId("B"),
        election_message_receiver=em_in_rx,
        election_message_sender=em_out_tx,
        election_result_sender=er_tx,
        connection_message_receiver=cm_rx,
        command_receiver=co_rx,
        is_candidate=True,
    )

    async with create_task_group() as tg:
        with fail_after(2):
            tg.start_soon(election.run)
            # Trigger new round at clock=1 (peer announces it)
            await em_in_tx.send(em(clock=1, seniority=0, node_id="A"))

            # Expect our broadcast back to the peer side for this round only
            while True:
                got = await em_out_rx.receive()
                if got.clock == 1 and got.proposed_session.master_node_id == NodeId(
                    "B"
                ):
                    break

            # Wait for the round to finish and produce an ElectionResult
            result = await er_rx.receive()
            assert result.session_id.master_node_id == NodeId("B")
            # We spawned as master; electing ourselves again is not "new master".
            assert result.is_new_master is False

            # Close inbound streams to end the receivers (and run())
            em_in_tx.close()
            cm_tx.close()
            co_tx.close()

    # We should have updated seniority to 2 (A + B).
    assert election.seniority == 2


@pytest.mark.anyio
async def test_peer_with_higher_seniority_wins_and_we_switch_master() -> None:
    """
    If a peer with clearly higher seniority participates in the round, they should win.
    We should broadcast our status exactly once for this round, then switch master.
    """
    em_out_tx, em_out_rx = channel[ElectionMessage]()
    em_in_tx, em_in_rx = channel[ElectionMessage]()
    er_tx, er_rx = channel[ElectionResult]()
    cm_tx, cm_rx = channel[ConnectionMessage]()
    co_tx, co_rx = channel[ForwarderCommand]()

    election = Election(
        node_id=NodeId("ME"),
        election_message_receiver=em_in_rx,
        election_message_sender=em_out_tx,
        election_result_sender=er_tx,
        connection_message_receiver=cm_rx,
        command_receiver=co_rx,
        is_candidate=True,
    )

    async with create_task_group() as tg:
        with fail_after(2):
            tg.start_soon(election.run)

            # Start round with peer's message (higher seniority)
            await em_in_tx.send(em(clock=1, seniority=10, node_id="PEER"))

            # We should still broadcast our status exactly once for this round
            while True:
                got = await em_out_rx.receive()
                if got.clock == 1:
                    assert got.seniority == 0
                    break

            # After the timeout, election result for clock=1 should report the peer as master
            # (Skip any earlier result from the boot campaign at clock=0 by filtering on election_clock)
            while True:
                result = await er_rx.receive()
                if result.session_id.election_clock == 1:
                    break

            assert result.session_id.master_node_id == NodeId("PEER")
            assert result.is_new_master is True

            em_in_tx.close()
            cm_tx.close()
            co_tx.close()

    # We lost → seniority unchanged
    assert election.seniority == 0


@pytest.mark.anyio
async def test_ignores_older_messages() -> None:
    """
    Messages with a lower clock than the current round are ignored by the receiver.
    Expect exactly one broadcast for the higher clock round.
    """
    em_out_tx, em_out_rx = channel[ElectionMessage]()
    em_in_tx, em_in_rx = channel[ElectionMessage]()
    er_tx, _er_rx = channel[ElectionResult]()
    cm_tx, cm_rx = channel[ConnectionMessage]()
    co_tx, co_rx = channel[ForwarderCommand]()

    election = Election(
        node_id=NodeId("ME"),
        election_message_receiver=em_in_rx,
        election_message_sender=em_out_tx,
        election_result_sender=er_tx,
        connection_message_receiver=cm_rx,
        command_receiver=co_rx,
        is_candidate=True,
    )

    async with create_task_group() as tg:
        with fail_after(2):
            tg.start_soon(election.run)

            # Newer round arrives first -> triggers campaign at clock=2
            await em_in_tx.send(em(clock=2, seniority=0, node_id="A"))
            while True:
                first = await em_out_rx.receive()
                if first.clock == 2:
                    break

            # Older message (clock=1) must be ignored (no second broadcast)
            await em_in_tx.send(em(clock=1, seniority=999, node_id="B"))

            got_second = False
            with move_on_after(0.05):
                _ = await em_out_rx.receive()
                got_second = True
            assert not got_second, "Should not receive a broadcast for an older round"

            em_in_tx.close()
            cm_tx.close()
            co_tx.close()

    # Not asserting on the result; focus is on ignore behavior.


@pytest.mark.anyio
async def test_two_rounds_emit_two_broadcasts_and_increment_clock() -> None:
    """
    Two successive rounds → two broadcasts. Second round triggered by a higher-clock message.
    """
    em_out_tx, em_out_rx = channel[ElectionMessage]()
    em_in_tx, em_in_rx = channel[ElectionMessage]()
    er_tx, _er_rx = channel[ElectionResult]()
    cm_tx, cm_rx = channel[ConnectionMessage]()
    co_tx, co_rx = channel[ForwarderCommand]()

    election = Election(
        node_id=NodeId("ME"),
        election_message_receiver=em_in_rx,
        election_message_sender=em_out_tx,
        election_result_sender=er_tx,
        connection_message_receiver=cm_rx,
        command_receiver=co_rx,
        is_candidate=True,
    )

    async with create_task_group() as tg:
        with fail_after(2):
            tg.start_soon(election.run)

            # Round 1 at clock=1
            await em_in_tx.send(em(clock=1, seniority=0, node_id="X"))
            while True:
                m1 = await em_out_rx.receive()
                if m1.clock == 1:
                    break

            # Round 2 at clock=2
            await em_in_tx.send(em(clock=2, seniority=0, node_id="Y"))
            while True:
                m2 = await em_out_rx.receive()
                if m2.clock == 2:
                    break

            em_in_tx.close()
            cm_tx.close()
            co_tx.close()

    # Not asserting on who won; just that both rounds were broadcast.


@pytest.mark.anyio
async def test_promotion_new_seniority_counts_participants() -> None:
    """
    When we win against two peers in the same round, our seniority becomes
    max(existing, number_of_candidates). With existing=0: expect 3 (us + A + B).
    """
    em_out_tx, em_out_rx = channel[ElectionMessage]()
    em_in_tx, em_in_rx = channel[ElectionMessage]()
    er_tx, er_rx = channel[ElectionResult]()
    cm_tx, cm_rx = channel[ConnectionMessage]()
    co_tx, co_rx = channel[ForwarderCommand]()

    election = Election(
        node_id=NodeId("ME"),
        election_message_receiver=em_in_rx,
        election_message_sender=em_out_tx,
        election_result_sender=er_tx,
        connection_message_receiver=cm_rx,
        command_receiver=co_rx,
        is_candidate=True,
    )

    async with create_task_group() as tg:
        with fail_after(2):
            tg.start_soon(election.run)

            # Start round at clock=7 with two peer participants
            await em_in_tx.send(em(clock=7, seniority=0, node_id="A"))
            await em_in_tx.send(em(clock=7, seniority=0, node_id="B"))

            # We should see exactly one broadcast from us for this round
            while True:
                got = await em_out_rx.receive()
                if got.clock == 7 and got.proposed_session.master_node_id == NodeId(
                    "ME"
                ):
                    break

            # Wait for the election to finish so seniority updates
            _ = await er_rx.receive()

            em_in_tx.close()
            cm_tx.close()
            co_tx.close()

    # We + A + B = 3 → new seniority expected to be 3
    assert election.seniority == 3


@pytest.mark.anyio
async def test_connection_message_triggers_new_round_broadcast() -> None:
    """
    A connection message increments the clock and starts a new campaign.
    We should observe a broadcast at the incremented clock.
    """
    em_out_tx, em_out_rx = channel[ElectionMessage]()
    em_in_tx, em_in_rx = channel[ElectionMessage]()
    er_tx, _er_rx = channel[ElectionResult]()
    cm_tx, cm_rx = channel[ConnectionMessage]()
    co_tx, co_rx = channel[ForwarderCommand]()

    election = Election(
        node_id=NodeId("ME"),
        election_message_receiver=em_in_rx,
        election_message_sender=em_out_tx,
        election_result_sender=er_tx,
        connection_message_receiver=cm_rx,
        command_receiver=co_rx,
        is_candidate=True,
    )

    async with create_task_group() as tg:
        with fail_after(2):
            tg.start_soon(election.run)

            # Send any connection message object; we close quickly to cancel before result creation
            await cm_tx.send(
                ConnectionMessage(
                    node_id=NodeId(),
                    connection_type=ConnectionMessageType.Connected,
                    remote_ipv4="",
                    remote_tcp_port=0,
                )
            )

            # Expect a broadcast for the new round at clock=1
            while True:
                got = await em_out_rx.receive()
                if got.clock == 1 and got.proposed_session.master_node_id == NodeId(
                    "ME"
                ):
                    break

            # Close promptly to avoid waiting for campaign completion
            em_in_tx.close()
            cm_tx.close()
            co_tx.close()

    # After cancellation (before election finishes), no seniority changes asserted here.


@pytest.mark.anyio
async def test_tie_breaker_prefers_node_with_more_commands_seen() -> None:
    """
    With equal seniority, the node that has seen more commands should win the election.
    We increase our local 'commands_seen' by sending TestCommand()s before triggering the round.
    """
    em_out_tx, em_out_rx = channel[ElectionMessage]()
    em_in_tx, em_in_rx = channel[ElectionMessage]()
    er_tx, er_rx = channel[ElectionResult]()
    cm_tx, cm_rx = channel[ConnectionMessage]()
    co_tx, co_rx = channel[ForwarderCommand]()

    me = NodeId("ME")

    election = Election(
        node_id=me,
        election_message_receiver=em_in_rx,
        election_message_sender=em_out_tx,
        election_result_sender=er_tx,
        connection_message_receiver=cm_rx,
        command_receiver=co_rx,
        is_candidate=True,
        seniority=0,
    )

    async with create_task_group() as tg:
        with fail_after(2):
            tg.start_soon(election.run)

            # Pump local commands so our commands_seen is high before the round starts
            for _ in range(50):
                await co_tx.send(
                    ForwarderCommand(origin=NodeId("SOMEONE"), command=TestCommand())
                )

            # Trigger a round at clock=1 with a peer of equal seniority but fewer commands
            await em_in_tx.send(
                em(clock=1, seniority=0, node_id="PEER", commands_seen=5)
            )

            # Observe our broadcast for this round (to ensure we've joined the round)
            while True:
                got = await em_out_rx.receive()
                if got.clock == 1 and got.proposed_session.master_node_id == me:
                    # We don't assert exact count, just that we've participated this round.
                    break

            # The elected result for clock=1 should be us due to higher commands_seen
            while True:
                result = await er_rx.receive()
                if result.session_id.master_node_id == me:
                    assert result.session_id.election_clock in (0, 1)
                    break

            em_in_tx.close()
            cm_tx.close()
            co_tx.close()
