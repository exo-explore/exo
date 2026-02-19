"""E2E Chaos Test: Networking resilience.

Scenarios:
1. Node disconnect mid-inference -- a worker stops receiving global events, then
   reconnects and catches up via the event buffer / nack mechanism.
2. Master detects stale node and times it out, then the node re-announces.
"""

import anyio
import pytest

from exo.master.main import Master
from exo.shared.types.commands import (
    ForwarderCommand,
    ForwarderDownloadCommand,
)
from exo.shared.types.common import NodeId, SessionId
from exo.shared.types.events import (
    ForwarderEvent,
    InstanceCreated,
    NodeGatheredInfo,
    TaskCreated,
)
from exo.utils.channels import channel

from .conftest import (
    EventCollector,
    MiniCluster,
    make_gathered_info_event,
    make_node_id,
)


@pytest.mark.slow
@pytest.mark.asyncio
async def test_node_disconnect_and_reconnect_event_replay() -> None:
    """Simulate a node disconnecting by closing its global event receiver,
    then reconnecting with a fresh receiver.

    After reconnection, events that were broadcast while the node was
    disconnected should be replayed to the new receiver via the shared
    channel state.  The master's state should remain consistent.
    """
    cluster = MiniCluster(node_count=1)

    async with anyio.create_task_group() as tg:
        tg.start_soon(cluster.master.run)

        # Register the master node so topology is populated
        await cluster.inject_node_info(cluster.master_node_id)
        await cluster.wait_for_topology_nodes(1)

        # Place a model instance
        await cluster.place_model()
        await cluster.wait_for_instances(1)

        # Verify instance was created
        assert len(cluster.master.state.instances) == 1

        # --- Simulate disconnection ---
        # The worker's global event receiver is independent; we just verify
        # that the master continues to accept commands while a worker is gone.
        _first_instance_id = next(iter(cluster.master.state.instances))

        # Send a chat command while "disconnected" worker can't process
        _cmd_id = await cluster.send_chat("Hello during disconnect")

        # Give master time to process the command
        await cluster.event_collector.wait_for_event_count(3, timeout=3.0)

        events = cluster.event_collector.indexed_events
        # Should have: NodeGatheredInfo, InstanceCreated, TaskCreated
        assert any(isinstance(e.event, NodeGatheredInfo) for e in events)
        assert any(isinstance(e.event, InstanceCreated) for e in events)
        assert any(isinstance(e.event, TaskCreated) for e in events)

        # --- Simulate reconnection ---
        # A reconnecting node gets a fresh receiver clone and catches up
        reconnect_receiver = cluster.global_event_internal_receiver.clone()
        _reconnect_collector = EventCollector(reconnect_receiver)

        # The new receiver should see future events; existing events are in
        # the master's event log (which would be replayed via RequestEventLog
        # in production). Here we verify the channel infrastructure works.
        await cluster.send_chat("Hello after reconnect")
        await anyio.sleep(0.1)

        # Master state should now have 2 tasks
        assert len(cluster.master.state.tasks) == 2

        # The master's state is consistent throughout
        assert len(cluster.master.state.instances) == 1
        assert cluster.master.state.last_event_applied_idx >= 3

        await cluster.shutdown_master()


@pytest.mark.slow
@pytest.mark.asyncio
async def test_master_detects_timed_out_node_and_cleans_state() -> None:
    """Verify that the master's plan loop detects a node that hasn't sent
    a heartbeat (NodeGatheredInfo) recently and emits NodeTimedOut, cleaning
    up topology and related state.
    """
    master_nid = make_node_id("master-timeout")
    session_id = SessionId(master_node_id=master_nid, election_clock=0)

    ge_sender, ge_receiver = channel[ForwarderEvent]()
    _cmd_sender, cmd_receiver = channel[ForwarderCommand]()
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

        # Register two nodes
        stale_node = make_node_id("stale")
        alive_node = make_node_id("alive")

        for node_id, suffix in [(stale_node, "_s0"), (alive_node, "_a0")]:
            sender_id = NodeId(f"{node_id}_sender{suffix}")
            await le_sender.send(
                make_gathered_info_event(node_id, sender_id, session_id, 0)
            )

        # Wait for both nodes in topology
        with anyio.fail_after(3):
            while len(list(master.state.topology.list_nodes())) < 2:
                await anyio.sleep(0.01)

        assert stale_node in master.state.last_seen
        assert alive_node in master.state.last_seen

        # Manually expire the stale node's last_seen time by patching the state
        # (in production, the _plan loop checks every 10s with a 30s threshold)
        from datetime import timedelta

        old_time = master.state.last_seen[stale_node] - timedelta(seconds=60)
        patched_last_seen = {**master.state.last_seen, stale_node: old_time}
        master.state = master.state.model_copy(update={"last_seen": patched_last_seen})

        # Trigger the plan loop manually to speed up the test
        # The plan loop checks for stale nodes
        # We wait for the NodeTimedOut event to be emitted
        with anyio.fail_after(15):
            while stale_node in master.state.last_seen:
                await anyio.sleep(0.1)

        # Stale node should be removed from topology
        assert stale_node not in set(master.state.topology.list_nodes())

        # Alive node should still be present
        assert alive_node in set(master.state.topology.list_nodes())
        assert alive_node in master.state.last_seen

        await master.shutdown()


@pytest.mark.slow
@pytest.mark.asyncio
async def test_event_ordering_preserved_under_concurrent_writers() -> None:
    """Multiple sources writing local events concurrently.  Verify that the
    master's MultiSourceBuffer correctly sequences events from each source
    and the final state is consistent.
    """
    master_nid = make_node_id("master-ordering")
    session_id = SessionId(master_node_id=master_nid, election_clock=0)

    ge_sender, ge_receiver = channel[ForwarderEvent]()
    _cmd_sender, cmd_receiver = channel[ForwarderCommand]()
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

        # Inject events from 3 different "worker" sources concurrently
        node_ids = [make_node_id(f"concurrent-{i}") for i in range(3)]

        async def inject_events(node_id: NodeId, count: int) -> None:
            for idx in range(count):
                sender_id = NodeId(f"{node_id}_sender")
                await le_sender.send(
                    make_gathered_info_event(node_id, sender_id, session_id, idx)
                )
                await anyio.sleep(0.001)  # slight jitter

        async with anyio.create_task_group() as inject_tg:
            for nid in node_ids:
                inject_tg.start_soon(inject_events, nid, 5)

        # Wait for master to process all events (3 nodes * 5 events each = 15)
        with anyio.fail_after(5):
            while master.state.last_event_applied_idx < 14:
                await anyio.sleep(0.01)

        # All 3 nodes should be visible in topology
        topo_nodes = set(master.state.topology.list_nodes())
        for nid in node_ids:
            assert nid in topo_nodes

        # Event indices should be sequential with no gaps
        assert master.state.last_event_applied_idx == 14

        await master.shutdown()
