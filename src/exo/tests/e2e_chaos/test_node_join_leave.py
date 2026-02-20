"""E2E Chaos Test: Node join/leave during operation.

Scenarios:
1. Add nodes dynamically -- register new nodes with the master while
   a model is already placed, verify topology grows.
2. Remove nodes -- simulate node timeout, verify instances on that node
   are cleaned up and remaining nodes are unaffected.
3. Rapid join/leave churn -- nodes join and leave quickly, verify state
   converges to a consistent snapshot.
"""

from datetime import timedelta

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
)
from exo.shared.types.worker.instances import InstanceMeta
from exo.shared.types.worker.shards import Sharding
from exo.utils.channels import channel

from .conftest import (
    TEST_MODEL_CARD,
    make_gathered_info_event,
    make_node_id,
)


@pytest.mark.slow
@pytest.mark.asyncio
async def test_dynamic_node_registration_expands_topology() -> None:
    """Start with one node, then add more dynamically.  Verify the topology
    grows and all nodes are visible in state.
    """
    master_nid = make_node_id("master-join")
    session_id = SessionId(master_node_id=master_nid, election_clock=0)

    ge_sender, _ge_receiver = channel[ForwarderEvent]()
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

    async with anyio.create_task_group() as tg:
        tg.start_soon(master.run)

        # Register initial node
        initial_node = make_node_id("initial")
        sender_id = NodeId(f"{initial_node}_sender")
        await le_sender.send(
            make_gathered_info_event(initial_node, sender_id, session_id, 0)
        )

        with anyio.fail_after(3):
            while len(list(master.state.topology.list_nodes())) < 1:
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

        # Dynamically add 3 more nodes
        new_nodes: list[NodeId] = []
        for i in range(3):
            new_nid = make_node_id(f"dynamic-{i}")
            new_nodes.append(new_nid)
            new_sender = NodeId(f"{new_nid}_sender")
            await le_sender.send(
                make_gathered_info_event(new_nid, new_sender, session_id, 0)
            )

        with anyio.fail_after(3):
            while len(list(master.state.topology.list_nodes())) < 4:
                await anyio.sleep(0.01)

        # All 4 nodes should be in topology
        topo_nodes = set(master.state.topology.list_nodes())
        assert initial_node in topo_nodes
        for nid in new_nodes:
            assert nid in topo_nodes

        # Original instance should still exist
        assert len(master.state.instances) >= 1

        await master.shutdown()


@pytest.mark.slow
@pytest.mark.asyncio
async def test_node_removal_cleans_up_instances() -> None:
    """Place a model on a specific node, then time it out.  Verify the
    instance assigned to that node is deleted by the master's plan loop.
    """
    master_nid = make_node_id("master-leave")
    session_id = SessionId(master_node_id=master_nid, election_clock=0)

    ge_sender, _ge_receiver = channel[ForwarderEvent]()
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

    async with anyio.create_task_group() as tg:
        tg.start_soon(master.run)

        # Register a worker node
        worker_nid = make_node_id("worker-leaving")
        sender_id = NodeId(f"{worker_nid}_sender")
        await le_sender.send(
            make_gathered_info_event(worker_nid, sender_id, session_id, 0)
        )

        with anyio.fail_after(3):
            while len(list(master.state.topology.list_nodes())) < 1:
                await anyio.sleep(0.01)

        # Place instance on the worker node
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

        assert len(master.state.instances) == 1

        # Simulate node leaving by expiring its last_seen
        old_time = master.state.last_seen[worker_nid] - timedelta(seconds=60)
        patched_last_seen = {**master.state.last_seen, worker_nid: old_time}
        master.state = master.state.model_copy(update={"last_seen": patched_last_seen})

        # The plan loop should detect the stale node and delete the instance
        # because the node assigned to the instance is no longer in the topology
        with anyio.fail_after(15):
            while worker_nid in master.state.last_seen:
                await anyio.sleep(0.1)

        # After timeout, the node should be removed from topology
        assert worker_nid not in set(master.state.topology.list_nodes())

        # The instance should eventually be deleted since the assigned node
        # is no longer connected (the _plan loop kills broken instances)
        with anyio.fail_after(15):
            while len(master.state.instances) > 0:
                await anyio.sleep(0.1)

        assert len(master.state.instances) == 0

        await master.shutdown()


@pytest.mark.slow
@pytest.mark.asyncio
async def test_rapid_join_leave_churn_converges() -> None:
    """Rapidly join and leave nodes.  After the churn settles, verify the
    master's state reflects only the surviving nodes.
    """
    master_nid = make_node_id("master-churn")
    session_id = SessionId(master_node_id=master_nid, election_clock=0)

    ge_sender, _ge_receiver = channel[ForwarderEvent]()
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

    async with anyio.create_task_group() as tg:
        tg.start_soon(master.run)

        # Register 5 nodes rapidly
        all_nodes: list[NodeId] = []
        for i in range(5):
            nid = make_node_id(f"churn-{i}")
            all_nodes.append(nid)
            sender_id = NodeId(f"{nid}_sender")
            await le_sender.send(
                make_gathered_info_event(nid, sender_id, session_id, 0)
            )

        with anyio.fail_after(5):
            while len(list(master.state.topology.list_nodes())) < 5:
                await anyio.sleep(0.01)

        assert len(list(master.state.topology.list_nodes())) == 5

        # Expire the first 3 nodes (simulate leaving)
        leaving_nodes = all_nodes[:3]
        surviving_nodes = all_nodes[3:]

        patched_last_seen = dict(master.state.last_seen)
        for nid in leaving_nodes:
            patched_last_seen[nid] = patched_last_seen[nid] - timedelta(seconds=60)
        master.state = master.state.model_copy(update={"last_seen": patched_last_seen})

        # Wait for master's plan loop to time out the expired nodes
        with anyio.fail_after(15):
            while any(nid in master.state.last_seen for nid in leaving_nodes):
                await anyio.sleep(0.1)

        # Verify only surviving nodes remain
        topo_nodes = set(master.state.topology.list_nodes())
        for nid in leaving_nodes:
            assert nid not in topo_nodes
        for nid in surviving_nodes:
            assert nid in topo_nodes

        assert len(list(master.state.topology.list_nodes())) == 2

        await master.shutdown()
