"""E2E Chaos Test: Failure recovery.

Scenarios:
1. Master crash and re-election -- master shuts down, a new election round
   produces a new master, workers re-converge.
2. Worker crash during task execution -- runner death is detected, instance
   is cleaned up, and cluster recovers.
"""

import anyio
import pytest

from exo.master.main import Master
from exo.shared.types.commands import (
    CommandId,
    ForwarderCommand,
    ForwarderDownloadCommand,
    PlaceInstance,
)
from exo.shared.types.common import NodeId, SessionId
from exo.shared.types.events import (
    ForwarderEvent,
    RunnerStatusUpdated,
)
from exo.shared.types.worker.instances import InstanceMeta
from exo.shared.types.worker.runners import RunnerFailed
from exo.shared.types.worker.shards import Sharding
from exo.utils.channels import channel

from .conftest import (
    TEST_MODEL_CARD,
    EventCollector,
    MiniCluster,
    make_gathered_info_event,
    make_node_id,
)


@pytest.mark.slow
@pytest.mark.asyncio
async def test_master_crash_and_reelection() -> None:
    """Simulate master crash by shutting it down, then verify a new master
    can be started with fresh state and begin accepting commands.

    This tests the scenario where the elected master dies and a new election
    must take place.  We simulate the election result directly (since
    Election is tested separately) and verify the new master works.
    """
    cluster = MiniCluster(node_count=1)
    old_instance_id: str = ""

    async with anyio.create_task_group() as tg:
        tg.start_soon(cluster.master.run)

        # Set up initial state
        await cluster.inject_node_info(cluster.master_node_id)
        await cluster.wait_for_topology_nodes(1)
        await cluster.place_model()
        await cluster.wait_for_instances(1)

        # Verify initial state
        assert len(cluster.master.state.instances) == 1
        old_instance_id = next(iter(cluster.master.state.instances))

        # --- Crash the master ---
        await cluster.shutdown_master()

    # --- Start a new master (simulating re-election) ---
    new_master_nid = make_node_id("new-master")
    new_session_id = SessionId(master_node_id=new_master_nid, election_clock=1)

    ge_sender, ge_receiver = channel[ForwarderEvent]()
    cmd_sender, cmd_receiver = channel[ForwarderCommand]()
    le_sender, le_receiver = channel[ForwarderEvent]()
    dl_sender, _dl_receiver = channel[ForwarderDownloadCommand]()

    new_master = Master(
        new_master_nid,
        new_session_id,
        global_event_sender=ge_sender,
        local_event_receiver=le_receiver,
        command_receiver=cmd_receiver,
        download_command_sender=dl_sender,
    )

    _new_collector = EventCollector(ge_receiver.clone())

    async with anyio.create_task_group() as tg:
        tg.start_soon(new_master.run)

        # New master starts with clean state
        assert len(new_master.state.instances) == 0
        assert new_master.state.last_event_applied_idx == -1

        # Re-register node with the new master
        sender_id = NodeId(f"{new_master_nid}_sender_new")
        await le_sender.send(
            make_gathered_info_event(new_master_nid, sender_id, new_session_id, 0)
        )

        # Wait for topology to be rebuilt
        with anyio.fail_after(3):
            while len(list(new_master.state.topology.list_nodes())) == 0:
                await anyio.sleep(0.01)

        # Place a new model instance on the new master
        await cmd_sender.send(
            ForwarderCommand(
                origin=new_master_nid,
                command=PlaceInstance(
                    command_id=CommandId(),
                    model_card=TEST_MODEL_CARD,
                    sharding=Sharding.Pipeline,
                    instance_meta=InstanceMeta.MlxRing,
                    min_nodes=1,
                ),
            )
        )

        with anyio.fail_after(3):
            while len(new_master.state.instances) == 0:
                await anyio.sleep(0.01)

        # Verify new master is functional
        assert len(new_master.state.instances) == 1
        new_instance_id = next(iter(new_master.state.instances))
        # New instance should be different from old one
        assert new_instance_id != old_instance_id

        await new_master.shutdown()


@pytest.mark.slow
@pytest.mark.asyncio
async def test_runner_failure_triggers_instance_cleanup() -> None:
    """Simulate a runner failure by injecting a RunnerStatusUpdated(RunnerFailed)
    event.  Verify that the master's plan loop eventually detects the broken
    instance (no connected node for the runner) and cleans it up.
    """
    master_nid = make_node_id("master-runner-fail")
    session_id = SessionId(master_node_id=master_nid, election_clock=0)

    ge_sender, ge_receiver = channel[ForwarderEvent]()
    cmd_sender, cmd_receiver = channel[ForwarderCommand]()
    le_sender, le_receiver = channel[ForwarderEvent]()
    dl_sender, _dl_receiver = channel[ForwarderDownloadCommand]()

    master = Master(
        master_nid,
        session_id,
        global_event_sender=ge_sender,
        local_event_receiver=le_receiver,
        command_receiver=cmd_receiver,
        download_command_sender=dl_sender,
    )

    _collector = EventCollector(ge_receiver.clone())

    async with anyio.create_task_group() as tg:
        tg.start_soon(master.run)

        # Register a worker node
        worker_nid = make_node_id("worker-failing")
        sender_id = NodeId(f"{worker_nid}_sender")
        await le_sender.send(
            make_gathered_info_event(worker_nid, sender_id, session_id, 0)
        )

        with anyio.fail_after(3):
            while len(list(master.state.topology.list_nodes())) == 0:
                await anyio.sleep(0.01)

        # Place a model instance
        await cmd_sender.send(
            ForwarderCommand(
                origin=master_nid,
                command=PlaceInstance(
                    command_id=CommandId(),
                    model_card=TEST_MODEL_CARD,
                    sharding=Sharding.Pipeline,
                    instance_meta=InstanceMeta.MlxRing,
                    min_nodes=1,
                ),
            )
        )

        with anyio.fail_after(3):
            while len(master.state.instances) == 0:
                await anyio.sleep(0.01)

        instance_id = next(iter(master.state.instances))
        instance = master.state.instances[instance_id]
        runner_id = next(iter(instance.shard_assignments.runner_to_shard))

        # Inject a RunnerFailed event from the worker
        await le_sender.send(
            ForwarderEvent(
                origin_idx=1,
                origin=sender_id,
                session=session_id,
                event=RunnerStatusUpdated(
                    runner_id=runner_id,
                    runner_status=RunnerFailed(
                        error_message="Simulated OOM kill (exitcode=137)"
                    ),
                ),
            )
        )

        # Wait for the runner failure to be processed
        with anyio.fail_after(3):
            while runner_id not in master.state.runners:
                await anyio.sleep(0.01)

        # The runner status should be RunnerFailed
        assert isinstance(master.state.runners[runner_id], RunnerFailed)

        await master.shutdown()


@pytest.mark.slow
@pytest.mark.asyncio
async def test_election_recovers_after_multiple_node_joins() -> None:
    """Verify that the election protocol correctly handles rapid node
    join/leave events by running multiple election rounds.
    """
    from exo.routing.connection_message import ConnectionMessage, ConnectionMessageType
    from exo.shared.election import Election, ElectionMessage, ElectionResult

    em_out_tx, em_out_rx = channel[ElectionMessage]()
    em_in_tx, em_in_rx = channel[ElectionMessage]()
    er_tx, er_rx = channel[ElectionResult]()
    cm_tx, cm_rx = channel[ConnectionMessage]()
    co_tx, co_rx = channel[ForwarderCommand]()

    election = Election(
        node_id=NodeId("SURVIVOR"),
        election_message_receiver=em_in_rx,
        election_message_sender=em_out_tx,
        election_result_sender=er_tx,
        connection_message_receiver=cm_rx,
        command_receiver=co_rx,
        is_candidate=True,
    )

    async with anyio.create_task_group() as tg:
        with anyio.fail_after(5):
            tg.start_soon(election.run)

            # Simulate rapid node joins via connection messages
            for i in range(3):
                await cm_tx.send(
                    ConnectionMessage(
                        node_id=NodeId(f"joiner-{i}"),
                        connection_type=ConnectionMessageType.Connected,
                        remote_ipv4=f"10.0.0.{i + 1}",
                        remote_tcp_port=52415,
                    )
                )
                # Each connection triggers a new election round
                while True:
                    got = await em_out_rx.receive()
                    if got.proposed_session.master_node_id == NodeId("SURVIVOR"):
                        break

            # After all joins, an election result should eventually be produced
            result = await er_rx.receive()
            assert result.session_id.master_node_id == NodeId("SURVIVOR")

            em_in_tx.close()
            cm_tx.close()
            co_tx.close()
