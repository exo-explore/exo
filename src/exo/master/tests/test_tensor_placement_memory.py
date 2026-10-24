import pytest

from exo.master.placement import place_instance
from exo.master.placement_utils import (
    get_shard_assignments,
    get_shard_assignments_for_tensor_parallel,
)
from exo.master.tests.conftest import (
    create_node_network,
    create_socket_connection,
)
from exo.shared.models.model_cards import ModelCard, ModelId, ModelTask
from exo.shared.topology import Topology
from exo.shared.types.backends import Backend
from exo.shared.types.commands import PlaceInstance
from exo.shared.types.common import CommandId, NodeId
from exo.shared.types.memory import Memory
from exo.shared.types.profiling import MemoryUsage
from exo.shared.types.topology import Connection, Cycle
from exo.shared.types.worker.instances import InstanceMeta
from exo.shared.types.worker.shards import Sharding, TensorShardMetadata
from exo.worker.runner.diagnostics import (
    RunnerDiagnosticCollector,
    RunnerOutOfMemory,
)


def _make_node_memory(ram_gb: float) -> MemoryUsage:
    return MemoryUsage.from_bytes(
        ram_total=round(ram_gb * (1024**3)),
        ram_available=round(ram_gb * (1024**3)),
        swap_total=0,
        swap_available=0,
    )


def test_tensor_parallel_rejected_when_node_cannot_fit_equal_shard():
    """Reproduce Issue #2430:
    Heterogeneous nodes (Node A has 28 GB free, Node B has 55 GB free).
    Total free RAM is 83 GB, which is more than the 70 GB model.
    However, Tensor sharding requires 70 GB / 2 = 35 GB per node.
    Node A (28 GB) cannot hold 35 GB, so placement must fail with an informative error.
    """
    model_card = ModelCard(
        model_id=ModelId("test-70b-model"),
        storage_size=Memory.from_gb(70),
        n_layers=80,
        hidden_size=8192,
        num_key_value_heads=8,
        supports_tensor=True,
        tasks=[ModelTask.TextGeneration],
        backends=[Backend.MlxMetal],
    )

    node_a = NodeId("node-36gb")
    node_b = NodeId("node-64gb")

    topology = Topology()
    topology.add_node(node_a)
    topology.add_node(node_b)
    topology.add_connection(
        Connection(source=node_a, sink=node_b, edge=create_socket_connection(1))
    )
    topology.add_connection(
        Connection(source=node_b, sink=node_a, edge=create_socket_connection(2))
    )

    node_memory = {
        node_a: _make_node_memory(28),
        node_b: _make_node_memory(55),
    }
    node_network = {
        node_a: create_node_network(),
        node_b: create_node_network(),
    }
    node_backends = {
        node_a: [Backend.MlxMetal],
        node_b: [Backend.MlxMetal],
    }

    cmd_tensor = PlaceInstance(
        command_id=CommandId(),
        model_card=model_card,
        sharding=Sharding.Tensor,
        instance_meta=InstanceMeta.MlxRing,
        min_nodes=2,
    )

    with pytest.raises(ValueError) as excinfo:
        place_instance(
            cmd_tensor,
            topology,
            {},
            node_memory,
            node_network,
            node_backends,
        )

    err_msg = str(excinfo.value)
    assert "Not enough per-device memory for Tensor sharding" in err_msg
    assert "needs 35.0 GB per node" in err_msg
    assert "node 'node-36gb' has 28.0 GB free" in err_msg
    assert (
        "Consider using Pipeline sharding to allocate layers proportionally" in err_msg
    )

    # Verify that the EXACT same cluster successfully places with Pipeline sharding!
    cmd_pipeline = PlaceInstance(
        command_id=CommandId(),
        model_card=model_card,
        sharding=Sharding.Pipeline,
        instance_meta=InstanceMeta.MlxRing,
        min_nodes=2,
    )

    placements = place_instance(
        cmd_pipeline,
        topology,
        {},
        node_memory,
        node_network,
        node_backends,
    )
    assert len(placements) == 1
    instance = next(iter(placements.values()))
    assert instance.shard_assignments.model_id == model_card.model_id


def test_tensor_parallel_accepted_when_all_nodes_have_sufficient_memory():
    """Homogeneous nodes (both 50 GB free).
    70 GB model requires 35 GB per node.
    Both nodes have 50 GB >= 35 GB, so Tensor sharding placement succeeds.
    """
    model_card = ModelCard(
        model_id=ModelId("test-70b-model"),
        storage_size=Memory.from_gb(70),
        n_layers=80,
        hidden_size=8192,
        num_key_value_heads=8,
        supports_tensor=True,
        tasks=[ModelTask.TextGeneration],
        backends=[Backend.MlxMetal],
    )

    node_a = NodeId("node-a")
    node_b = NodeId("node-b")

    topology = Topology()
    topology.add_node(node_a)
    topology.add_node(node_b)
    topology.add_connection(
        Connection(source=node_a, sink=node_b, edge=create_socket_connection(1))
    )
    topology.add_connection(
        Connection(source=node_b, sink=node_a, edge=create_socket_connection(2))
    )

    node_memory = {
        node_a: _make_node_memory(50),
        node_b: _make_node_memory(50),
    }
    node_network = {
        node_a: create_node_network(),
        node_b: create_node_network(),
    }
    node_backends = {
        node_a: [Backend.MlxMetal],
        node_b: [Backend.MlxMetal],
    }

    cmd_tensor = PlaceInstance(
        command_id=CommandId(),
        model_card=model_card,
        sharding=Sharding.Tensor,
        instance_meta=InstanceMeta.MlxRing,
        min_nodes=2,
    )

    placements = place_instance(
        cmd_tensor,
        topology,
        {},
        node_memory,
        node_network,
        node_backends,
    )
    assert len(placements) == 1
    instance = next(iter(placements.values()))
    for shard in instance.shard_assignments.runner_to_shard.values():
        assert isinstance(shard, TensorShardMetadata)
        assert shard.world_size == 2


def test_get_shard_assignments_tensor_parallel_defensive_validation():
    """Verify get_shard_assignments_for_tensor_parallel validates node_memory when provided."""
    model_card = ModelCard(
        model_id=ModelId("test-model"),
        storage_size=Memory.from_gb(40),
        n_layers=32,
        hidden_size=4096,
        supports_tensor=True,
        tasks=[ModelTask.TextGeneration],
        backends=[Backend.MlxMetal],
    )
    node_1 = NodeId("n1")
    node_2 = NodeId("n2")
    cycle = Cycle(node_ids=[node_1, node_2])

    # Insufficient on node 1
    insufficient_memory = {
        node_1: _make_node_memory(15),  # needs 20
        node_2: _make_node_memory(30),
    }

    with pytest.raises(ValueError) as excinfo:
        get_shard_assignments_for_tensor_parallel(
            model_card, cycle, node_memory=insufficient_memory
        )
    assert "insufficient memory for tensor parallel" in str(excinfo.value)

    # Also test via dispatching function get_shard_assignments
    with pytest.raises(ValueError) as excinfo_dispatch:
        get_shard_assignments(
            model_card, cycle, Sharding.Tensor, node_memory=insufficient_memory
        )
    assert "insufficient memory for tensor parallel" in str(excinfo_dispatch.value)

    # Sufficient memory
    sufficient_memory = {
        node_1: _make_node_memory(25),
        node_2: _make_node_memory(30),
    }
    assignments = get_shard_assignments(
        model_card, cycle, Sharding.Tensor, node_memory=sufficient_memory
    )
    assert len(assignments.runner_to_shard) == 2


def test_runner_diagnostic_detects_out_of_memory_patterns():
    """Verify runner diagnostics captures OOM signatures correctly."""
    collector = RunnerDiagnosticCollector()

    lines = [
        "[2026-04-12 18:24:59.049] [metal] The requested buffer size (38.16 GiB) is close to the maximum recommended size for this device (38.40 GiB). Allocations may fail.",
        "libc++abi: terminating due to uncaught exception of type std::bad_alloc: std::bad_alloc",
        "torch.cuda.OutOfMemoryError: CUDA out of memory. Tried to allocate 2.00 GiB",
    ]

    for line in lines:
        collector.record_line(line)

    diagnostics = collector.diagnostics()
    assert len(diagnostics) == 3
    for diag in diagnostics:
        assert isinstance(diag, RunnerOutOfMemory)
        assert "Process ran out of memory" in diag.message
