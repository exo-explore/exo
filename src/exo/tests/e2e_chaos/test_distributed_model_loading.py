"""E2E Chaos Test: Large model distributed loading.

Scenarios:
1. Multi-node sharding -- place a model with min_nodes > 1, verify sharding
   is distributed across multiple nodes with correct shard assignments.
2. Single-node gets all layers -- place on 1 node, verify full assignment.
3. Three-node sharding -- verify 3-way distribution.
"""

import anyio
import pytest

from exo.master.main import Master
from exo.shared.models.model_cards import ModelCard, ModelTask
from exo.shared.types.commands import (
    CommandId,
    ForwarderCommand,
    ForwarderDownloadCommand,
    PlaceInstance,
)
from exo.shared.types.common import ModelId, NodeId, SessionId
from exo.shared.types.events import (
    ForwarderEvent,
)
from exo.shared.types.memory import Memory
from exo.shared.types.worker.instances import InstanceMeta, MlxRingInstance
from exo.shared.types.worker.shards import PipelineShardMetadata, Sharding
from exo.utils.channels import Sender, channel

from .conftest import (
    TEST_MODEL_CARD,
    make_gathered_info_event,
    make_node_id,
    make_topology_edge_event,
)

# A model large enough to need sharding but small enough to fit in test node memory
# Each test node has 8GB available, so 2 nodes = 16GB, 3 nodes = 24GB.
# storage_size < total cluster memory to pass the memory filter.
LARGE_MODEL_CARD = ModelCard(
    model_id=ModelId("test-model/large-70b-4bit"),
    n_layers=80,
    storage_size=Memory.from_bytes(4 * 1024 * 1024 * 1024),
    hidden_size=8192,
    supports_tensor=True,
    tasks=[ModelTask.TextGeneration],
)


async def _register_node(
    le_sender: Sender[ForwarderEvent],
    node_id: NodeId,
    session_id: SessionId,
) -> None:
    """Register a node by injecting NodeGatheredInfo."""
    sender_id = NodeId(f"{node_id}_sender")
    await le_sender.send(make_gathered_info_event(node_id, sender_id, session_id, 0))


async def _add_bidirectional_edge(
    le_sender: Sender[ForwarderEvent],
    node_a: NodeId,
    node_b: NodeId,
    session_id: SessionId,
    sender_id: NodeId,
    origin_idx_start: int,
    ip_a: int,
    ip_b: int,
) -> None:
    """Add bidirectional topology edges between two nodes."""
    await le_sender.send(
        make_topology_edge_event(
            node_a, node_b, sender_id, session_id, origin_idx_start, ip_suffix=ip_b
        )
    )
    await le_sender.send(
        make_topology_edge_event(
            node_b, node_a, sender_id, session_id, origin_idx_start + 1, ip_suffix=ip_a
        )
    )


@pytest.mark.slow
@pytest.mark.asyncio
async def test_multi_node_sharding_distributes_layers() -> None:
    """Place a model with min_nodes=2 on a cluster with 2 connected nodes.
    Verify the resulting instance has shard assignments spanning both nodes.
    """
    master_nid = make_node_id("master-shard")
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

        worker_a = make_node_id("shard-worker-a")
        worker_b = make_node_id("shard-worker-b")

        # Register both worker nodes (each sender uses origin_idx=0)
        for nid in [worker_a, worker_b]:
            await _register_node(le_sender, nid, session_id)

        with anyio.fail_after(3):
            while len(list(master.state.topology.list_nodes())) < 2:
                await anyio.sleep(0.01)

        # Add bidirectional edges to form a 2-node cycle (A <-> B)
        edge_sender = NodeId("edge_sender")
        await _add_bidirectional_edge(
            le_sender, worker_a, worker_b, session_id, edge_sender, 0, 1, 2
        )

        # Wait for edges to be processed
        with anyio.fail_after(3):
            while len(list(master.state.topology.list_connections())) < 2:
                await anyio.sleep(0.01)

        # Place a large model requiring 2 nodes
        await cmd_sender.send(
            ForwarderCommand(
                origin=master_nid,
                command=PlaceInstance(
                    command_id=CommandId(),
                    model_card=LARGE_MODEL_CARD,
                    sharding=Sharding.Pipeline,
                    instance_meta=InstanceMeta.MlxRing,
                    min_nodes=2,
                ),
            )
        )

        with anyio.fail_after(5):
            while len(master.state.instances) == 0:
                await anyio.sleep(0.01)

        instance_id = next(iter(master.state.instances))
        instance = master.state.instances[instance_id]
        assert isinstance(instance, MlxRingInstance)

        shard_assignments = instance.shard_assignments
        runner_shards = shard_assignments.runner_to_shard

        assert len(runner_shards) == 2

        assigned_nodes = set(shard_assignments.node_to_runner.keys())
        assert worker_a in assigned_nodes
        assert worker_b in assigned_nodes

        shards = list(runner_shards.values())
        assert all(isinstance(s, PipelineShardMetadata) for s in shards)
        pipeline_shards = [s for s in shards if isinstance(s, PipelineShardMetadata)]

        assert all(s.world_size == 2 for s in pipeline_shards)
        ranks = {s.device_rank for s in pipeline_shards}
        assert ranks == {0, 1}

        sorted_shards = sorted(pipeline_shards, key=lambda s: s.device_rank)
        assert sorted_shards[0].start_layer == 0
        assert sorted_shards[-1].end_layer == LARGE_MODEL_CARD.n_layers

        total_layers = sum(s.end_layer - s.start_layer for s in sorted_shards)
        assert total_layers == LARGE_MODEL_CARD.n_layers

        await master.shutdown()


@pytest.mark.slow
@pytest.mark.asyncio
async def test_single_node_gets_all_layers() -> None:
    """Place a model with min_nodes=1 on a single node.  Verify the
    instance has one runner assigned all layers (world_size=1).
    """
    master_nid = make_node_id("master-single")
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

        worker_nid = make_node_id("single-worker")
        await _register_node(le_sender, worker_nid, session_id)

        with anyio.fail_after(3):
            while len(list(master.state.topology.list_nodes())) < 1:
                await anyio.sleep(0.01)

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
        assert isinstance(instance, MlxRingInstance)

        shards = list(instance.shard_assignments.runner_to_shard.values())
        assert len(shards) == 1

        shard = shards[0]
        assert isinstance(shard, PipelineShardMetadata)
        assert shard.world_size == 1
        assert shard.device_rank == 0
        assert shard.start_layer == 0
        assert shard.end_layer == TEST_MODEL_CARD.n_layers

        await master.shutdown()


@pytest.mark.slow
@pytest.mark.asyncio
async def test_three_node_sharding_distributes_evenly() -> None:
    """Place a model across 3 connected nodes.  Verify all 3 get shard assignments."""
    master_nid = make_node_id("master-3way")
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

        workers: list[NodeId] = []
        for i in range(3):
            nid = make_node_id(f"three-worker-{i}")
            workers.append(nid)
            await _register_node(le_sender, nid, session_id)

        with anyio.fail_after(3):
            while len(list(master.state.topology.list_nodes())) < 3:
                await anyio.sleep(0.01)

        # Add bidirectional edges to form a fully connected 3-node cycle:
        # A <-> B, B <-> C, C <-> A
        edge_sender = NodeId("edge_sender_3way")
        idx = 0
        ip_counter = 10
        for i in range(3):
            source = workers[i]
            sink = workers[(i + 1) % 3]
            # Forward edge
            await le_sender.send(
                make_topology_edge_event(
                    source,
                    sink,
                    edge_sender,
                    session_id,
                    idx,
                    ip_suffix=ip_counter,
                )
            )
            idx += 1
            ip_counter += 1
            # Reverse edge
            await le_sender.send(
                make_topology_edge_event(
                    sink,
                    source,
                    edge_sender,
                    session_id,
                    idx,
                    ip_suffix=ip_counter,
                )
            )
            idx += 1
            ip_counter += 1

        # Wait for all 6 edges (3 pairs x 2 directions)
        with anyio.fail_after(3):
            while len(list(master.state.topology.list_connections())) < 6:
                await anyio.sleep(0.01)

        await cmd_sender.send(
            ForwarderCommand(
                origin=master_nid,
                command=PlaceInstance(
                    command_id=CommandId(),
                    model_card=LARGE_MODEL_CARD,
                    sharding=Sharding.Pipeline,
                    instance_meta=InstanceMeta.MlxRing,
                    min_nodes=3,
                ),
            )
        )

        with anyio.fail_after(5):
            while len(master.state.instances) == 0:
                await anyio.sleep(0.01)

        instance = next(iter(master.state.instances.values()))
        assert isinstance(instance, MlxRingInstance)

        assignments = instance.shard_assignments
        assert len(assignments.runner_to_shard) == 3
        assert len(assignments.node_to_runner) == 3

        for w in workers:
            assert w in assignments.node_to_runner

        shards = list(assignments.runner_to_shard.values())
        ranks = {s.device_rank for s in shards if isinstance(s, PipelineShardMetadata)}
        assert ranks == {0, 1, 2}

        pipeline_shards = [s for s in shards if isinstance(s, PipelineShardMetadata)]
        total_layers = sum(s.end_layer - s.start_layer for s in pipeline_shards)
        assert total_layers == LARGE_MODEL_CARD.n_layers

        await master.shutdown()
