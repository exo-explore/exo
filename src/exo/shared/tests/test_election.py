import pytest
from anyio import create_task_group, fail_after, move_on_after

from exo.routing.connection_message import ConnectionMessage, ConnectionMessageType
from exo.shared.election import Election, ElectionMessage, ElectionResult
from exo.shared.types.common import NodeId
from exo.utils.channels import channel

# ======= #
# Helpers #
# ======= #


def em(clock: int, seniority: int, node_id: str) -> ElectionMessage:
    return ElectionMessage(clock=clock, seniority=seniority, node_id=NodeId(node_id))


@pytest.fixture
def fast_timeout(monkeypatch: pytest.MonkeyPatch):
    # Keep campaigns fast; user explicitly allows tests to shorten the timeout.
    import exo.shared.election as election_mod

    monkeypatch.setattr(election_mod, "ELECTION_TIMEOUT", 0.05, raising=True)
    yield


# ======================================= #
#                 TESTS                   #
# ======================================= #


@pytest.mark.anyio
async def test_single_round_broadcasts_and_updates_seniority_on_self_win(
    fast_timeout: None,
) -> None:
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
    # Connection messages (unused in this test but required by ctor)
    cm_tx, cm_rx = channel[ConnectionMessage]()

    election = Election(
        node_id=NodeId("B"),
        election_message_receiver=em_in_rx,
        election_message_sender=em_out_tx,
        election_result_sender=er_tx,
        connection_message_receiver=cm_rx,
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
                if got.clock == 1 and got.node_id == NodeId("B"):
                    break

            # Wait for the round to finish and produce an ElectionResult
            result = await er_rx.receive()
            assert result.node_id == NodeId("B")
            # We spawned as master; electing ourselves again is not "new master".
            assert result.is_new_master is False

            # Close inbound streams to end the receivers (and run())
            await em_in_tx.aclose()
            await cm_tx.aclose()

    # We should have updated seniority to 2 (A + B).
    assert election.seniority == 2


@pytest.mark.anyio
async def test_peer_with_higher_seniority_wins_and_we_switch_master(
    fast_timeout: None,
) -> None:
    """
    If a peer with clearly higher seniority participates in the round, they should win.
    We should broadcast our status exactly once for this round, then switch master.
    """
    em_out_tx, em_out_rx = channel[ElectionMessage]()
    em_in_tx, em_in_rx = channel[ElectionMessage]()
    er_tx, er_rx = channel[ElectionResult]()
    cm_tx, cm_rx = channel[ConnectionMessage]()

    election = Election(
        node_id=NodeId("ME"),
        election_message_receiver=em_in_rx,
        election_message_sender=em_out_tx,
        election_result_sender=er_tx,
        connection_message_receiver=cm_rx,
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

            # After the timeout, election result should report the peer as master
            result = await er_rx.receive()
            assert result.node_id == NodeId("PEER")
            assert result.is_new_master is True

            await em_in_tx.aclose()
            await cm_tx.aclose()

    # We lost → seniority unchanged
    assert election.seniority == 0


@pytest.mark.anyio
async def test_ignores_older_messages(fast_timeout: None) -> None:
    """
    Messages with a lower clock than the current round are ignored by the receiver.
    Expect exactly one broadcast for the higher clock round.
    """
    em_out_tx, em_out_rx = channel[ElectionMessage]()
    em_in_tx, em_in_rx = channel[ElectionMessage]()
    er_tx, _er_rx = channel[ElectionResult]()
    cm_tx, cm_rx = channel[ConnectionMessage]()

    election = Election(
        node_id=NodeId("ME"),
        election_message_receiver=em_in_rx,
        election_message_sender=em_out_tx,
        election_result_sender=er_tx,
        connection_message_receiver=cm_rx,
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
            with move_on_after(0.2):
                _ = await em_out_rx.receive()
                got_second = True
            assert not got_second, "Should not receive a broadcast for an older round"

            await em_in_tx.aclose()
            await cm_tx.aclose()

    # Not asserting on the result; focus is on ignore behavior.


@pytest.mark.anyio
async def test_two_rounds_emit_two_broadcasts_and_increment_clock(
    fast_timeout: None,
) -> None:
    """
    Two successive rounds → two broadcasts. Second round triggered by a higher-clock message.
    """
    em_out_tx, em_out_rx = channel[ElectionMessage]()
    em_in_tx, em_in_rx = channel[ElectionMessage]()
    er_tx, _er_rx = channel[ElectionResult]()
    cm_tx, cm_rx = channel[ConnectionMessage]()

    election = Election(
        node_id=NodeId("ME"),
        election_message_receiver=em_in_rx,
        election_message_sender=em_out_tx,
        election_result_sender=er_tx,
        connection_message_receiver=cm_rx,
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

            await em_in_tx.aclose()
            await cm_tx.aclose()

    # Not asserting on who won; just that both rounds were broadcast.


@pytest.mark.anyio
async def test_promotion_new_seniority_counts_participants(fast_timeout: None) -> None:
    """
    When we win against two peers in the same round, our seniority becomes
    max(existing, number_of_candidates). With existing=0: expect 3 (us + A + B).
    """
    em_out_tx, em_out_rx = channel[ElectionMessage]()
    em_in_tx, em_in_rx = channel[ElectionMessage]()
    er_tx, er_rx = channel[ElectionResult]()
    cm_tx, cm_rx = channel[ConnectionMessage]()

    election = Election(
        node_id=NodeId("ME"),
        election_message_receiver=em_in_rx,
        election_message_sender=em_out_tx,
        election_result_sender=er_tx,
        connection_message_receiver=cm_rx,
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
                if got.clock == 7 and got.node_id == NodeId("ME"):
                    break

            # Wait for the election to finish so seniority updates
            _ = await er_rx.receive()

            await em_in_tx.aclose()
            await cm_tx.aclose()

    # We + A + B = 3 → new seniority expected to be 3
    assert election.seniority == 3


@pytest.mark.anyio
async def test_connection_message_triggers_new_round_broadcast(
    fast_timeout: None,
) -> None:
    """
    A connection message increments the clock and starts a new campaign.
    We should observe a broadcast at the incremented clock.
    """
    em_out_tx, em_out_rx = channel[ElectionMessage]()
    em_in_tx, em_in_rx = channel[ElectionMessage]()
    er_tx, _er_rx = channel[ElectionResult]()
    cm_tx, cm_rx = channel[ConnectionMessage]()

    election = Election(
        node_id=NodeId("ME"),
        election_message_receiver=em_in_rx,
        election_message_sender=em_out_tx,
        election_result_sender=er_tx,
        connection_message_receiver=cm_rx,
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
                if got.clock == 1 and got.node_id == NodeId("ME"):
                    break

            # Close promptly to avoid waiting for campaign completion
            await em_in_tx.aclose()
            await cm_tx.aclose()

    # After cancellation (before election finishes), no seniority changes asserted here.
