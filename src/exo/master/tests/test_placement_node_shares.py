"""Placement checks that every node can hold its own share of the model, in the memory a model
can use on it, not just that the nodes have enough memory between them."""

from collections.abc import Mapping

import pytest

from exo.master.placement import place_instance, shard_bytes, usable_memory
from exo.master.tests.conftest import create_node_network, create_socket_connection
from exo.shared.models.model_cards import ModelCard, ModelId, ModelTask
from exo.shared.topology import Topology
from exo.shared.types.backends import Backend
from exo.shared.types.commands import PlaceInstance
from exo.shared.types.common import CommandId, NodeId
from exo.shared.types.memory import Memory
from exo.shared.types.profiling import MemoryUsage, NodeIdentity
from exo.shared.types.topology import Connection
from exo.shared.types.worker.instances import Instance, InstanceMeta
from exo.shared.types.worker.shards import Sharding

GB = 1024**3


def _memory(free_gb: float, gpu_working_set_gb: float | None = None) -> MemoryUsage:
    return MemoryUsage(
        ram_total=Memory.from_gb(2 * free_gb),
        ram_available=Memory.from_gb(free_gb),
        swap_total=Memory(),
        swap_available=Memory(),
        gpu_working_set=None
        if gpu_working_set_gb is None
        else Memory.from_gb(gpu_working_set_gb),
    )


def _model(size_gb: float) -> ModelCard:
    return ModelCard(
        model_id=ModelId("test-org/test-model"),
        storage_size=Memory.from_gb(size_gb),
        n_layers=40,
        hidden_size=1024,
        supports_tensor=True,
        tasks=[ModelTask.TextGeneration],
        backends=[Backend.MlxMetal],
    )


def _connected(*groups: list[NodeId]) -> Topology:
    """Nodes connected both ways to every other node in their group."""
    topology = Topology()
    port = 0
    for group in groups:
        for node_id in group:
            topology.add_node(node_id)
        for source in group:
            for sink in group:
                if source != sink:
                    port += 1
                    topology.add_connection(
                        Connection(
                            source=source,
                            sink=sink,
                            edge=create_socket_connection(port),
                        )
                    )
    return topology


def _place(
    model: ModelCard,
    topology: Topology,
    node_memory: Mapping[NodeId, MemoryUsage],
    sharding: Sharding,
    min_nodes: int = 2,
    node_identities: Mapping[NodeId, NodeIdentity] | None = None,
) -> Instance:
    placement = place_instance(
        PlaceInstance(
            command_id=CommandId(),
            model_card=model,
            sharding=sharding,
            instance_meta=InstanceMeta.MlxRing,
            min_nodes=min_nodes,
        ),
        topology,
        {},
        node_memory,
        {node_id: create_node_network() for node_id in node_memory},
        {node_id: [Backend.MlxMetal] for node_id in node_memory},
        node_identities=node_identities,
    )
    (instance,) = placement.values()
    return instance


def _share_gb(instance: Instance) -> dict[NodeId, float]:
    assignments = instance.shard_assignments
    return {
        node_id: shard_bytes(assignments.runner_to_shard[runner_id]) / GB
        for node_id, runner_id in assignments.node_to_runner.items()
    }


def test_tensor_sharding_is_refused_when_a_node_cannot_hold_its_share() -> None:
    # A 16 GB Mac and a 64 GB Mac: a 27.5 GB model fits between them, but tensor sharding
    # gives each 13.75 GB, and the 16 GB Mac's GPU can use only 11.8 GB of its 12 GB free
    small, large = NodeId(), NodeId()
    node_memory = {small: _memory(12, 11.8), large: _memory(58, 51.8)}

    with pytest.raises(ValueError) as refused:
        _place(
            _model(27.5),
            _connected([small, large]),
            node_memory,
            Sharding.Tensor,
            node_identities={small: NodeIdentity(friendly_name="Small Mac")},
        )

    assert str(refused.value) == (
        "Tensor sharding gives each of 2 nodes 13.8 GB of test-org/test-model, but Small "
        "Mac can use only about 11.8 GB. Use Pipeline sharding, which splits the model by "
        "each node's memory, or nodes with more memory"
    )


def test_pipeline_sharding_places_the_same_model_by_each_nodes_memory() -> None:
    small, large = NodeId(), NodeId()
    node_memory = {small: _memory(12, 11.8), large: _memory(58, 51.8)}

    instance = _place(
        _model(27.5), _connected([small, large]), node_memory, Sharding.Pipeline
    )

    shares = _share_gb(instance)
    assert shares[small] <= 11.8
    assert shares[large] <= 51.8


def test_tensor_sharding_on_unequal_nodes_is_placed_when_each_holds_its_share() -> None:
    small, large = NodeId(), NodeId()
    node_memory = {small: _memory(11, 11.8), large: _memory(58, 51.8)}

    instance = _place(
        _model(20), _connected([small, large]), node_memory, Sharding.Tensor
    )

    assert _share_gb(instance) == {small: 10.0, large: 10.0}


def test_tensor_sharding_on_equal_nodes_is_placed() -> None:
    a, b = NodeId(), NodeId()
    node_memory = {a: _memory(20, 17.8), b: _memory(20, 17.8)}

    instance = _place(_model(30), _connected([a, b]), node_memory, Sharding.Tensor)

    assert _share_gb(instance) == {a: 15.0, b: 15.0}


def test_equal_nodes_are_refused_a_model_beyond_their_gpus() -> None:
    # 40 GB free between them, but each GPU can use only 17.8 GB, less than an 18 GB share
    a, b = NodeId(), NodeId()
    node_memory = {a: _memory(20, 17.8), b: _memory(20, 17.8)}

    with pytest.raises(ValueError, match="can use at most 35.6 GB between them"):
        _place(_model(36), _connected([a, b]), node_memory, Sharding.Tensor)


def test_tensor_sharding_uses_nodes_that_can_hold_their_shares() -> None:
    # The first pair has the most memory, but only the second can split the model evenly
    small, large, c, d = NodeId(), NodeId(), NodeId(), NodeId()
    node_memory = {
        small: _memory(2),
        large: _memory(40),
        c: _memory(11),
        d: _memory(11),
    }

    instance = _place(
        _model(20), _connected([small, large], [c, d]), node_memory, Sharding.Tensor
    )

    assert set(instance.shard_assignments.node_to_runner) == {c, d}


def test_a_single_node_is_placed_within_its_gpus_working_set() -> None:
    node = NodeId()
    topology = _connected([node])

    instance = _place(
        _model(40), topology, {node: _memory(58, 51.8)}, Sharding.Pipeline, min_nodes=1
    )

    assert _share_gb(instance) == {node: 40.0}
    with pytest.raises(ValueError, match="can use at most 51.8 GB"):
        _place(
            _model(54),
            topology,
            {node: _memory(58, 51.8)},
            Sharding.Pipeline,
            min_nodes=1,
        )


def test_pipeline_is_refused_when_the_model_fits_only_in_free_memory() -> None:
    # The pair of the report: a 36 GB and a 64 GB Mac have 88 GB free between them, but their
    # GPUs can use only 79.9 GB, and an 80 GB model would overflow the 64 GB Mac's GPU
    small, large = NodeId(), NodeId()
    node_memory = {small: _memory(30, 28.1), large: _memory(58, 51.8)}

    with pytest.raises(ValueError) as refused:
        _place(_model(80), _connected([small, large]), node_memory, Sharding.Pipeline)

    assert str(refused.value) == (
        "Not enough memory: test-org/test-model needs 80.0 GB, but connected nodes can use "
        "at most 79.9 GB between them"
    )


def test_pipeline_layers_follow_the_memory_a_model_can_use() -> None:
    # Both nodes have 24 GB free, but the first one's GPU can use only 8 GB of it
    limited, unlimited = NodeId(), NodeId()
    node_memory = {limited: _memory(24, 8), unlimited: _memory(24)}

    instance = _place(
        _model(20), _connected([limited, unlimited]), node_memory, Sharding.Pipeline
    )

    shares = _share_gb(instance)
    assert shares[limited] <= 8
    assert shares[limited] + shares[unlimited] == 20


def test_usable_memory_is_the_free_memory_up_to_the_gpus_working_set() -> None:
    no_gpu, gpu_limited, ram_limited = NodeId(), NodeId(), NodeId()

    usable = usable_memory(
        {
            no_gpu: _memory(10),
            gpu_limited: _memory(58, 51.8),
            ram_limited: _memory(9, 11.8),
        }
    )

    assert {node: round(u.ram_available.in_gb, 1) for node, u in usable.items()} == {
        no_gpu: 10.0,
        gpu_limited: 51.8,
        ram_limited: 9.0,
    }
