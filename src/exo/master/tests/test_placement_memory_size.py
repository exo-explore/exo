"""A card's optional memory_size (what the model takes once loaded) is what placement sizes
nodes by; cards without it are placed by storage_size exactly as before."""

from pathlib import Path

import pytest
from anyio import Path as AsyncPath

from exo.master.placement import place_instance, shard_bytes
from exo.master.tests.conftest import create_node_network
from exo.shared.models.model_cards import ModelCard, ModelId, ModelTask
from exo.shared.topology import Topology
from exo.shared.types.backends import Backend
from exo.shared.types.commands import PlaceInstance
from exo.shared.types.common import CommandId, NodeId
from exo.shared.types.memory import Memory
from exo.shared.types.profiling import MemoryUsage
from exo.shared.types.worker.instances import Instance, InstanceMeta
from exo.shared.types.worker.shards import Sharding

GB = 1024**3


def _card(storage_gb: float, memory_gb: float | None = None) -> ModelCard:
    return ModelCard(
        model_id=ModelId("test-org/partly-on-disk"),
        storage_size=Memory.from_gb(storage_gb),
        memory_size=None if memory_gb is None else Memory.from_gb(memory_gb),
        n_layers=40,
        hidden_size=1024,
        supports_tensor=True,
        tasks=[ModelTask.TextGeneration],
        backends=[Backend.MlxMetal],
    )


def _one_node(free_gb: float) -> tuple[Topology, NodeId, dict[NodeId, MemoryUsage]]:
    node = NodeId()
    topology = Topology()
    topology.add_node(node)
    memory = MemoryUsage(
        ram_total=Memory.from_gb(2 * free_gb),
        ram_available=Memory.from_gb(free_gb),
        swap_total=Memory(),
        swap_available=Memory(),
        gpu_working_set=None,
    )
    return topology, node, {node: memory}


def _place_single(card: ModelCard, free_gb: float) -> Instance:
    topology, _, node_memory = _one_node(free_gb)
    placement = place_instance(
        PlaceInstance(
            command_id=CommandId(),
            model_card=card,
            sharding=Sharding.Pipeline,
            instance_meta=InstanceMeta.MlxRing,
            min_nodes=1,
        ),
        topology,
        {},
        node_memory,
        {node_id: create_node_network() for node_id in node_memory},
        {node_id: [Backend.MlxMetal] for node_id in node_memory},
    )
    (instance,) = placement.values()
    return instance


def test_resident_size_defaults_to_storage_size() -> None:
    card = _card(500)
    assert card.memory_size is None
    assert card.resident_size == card.storage_size


def test_without_memory_size_placement_is_unchanged() -> None:
    with pytest.raises(ValueError) as refused:
        _place_single(_card(500), free_gb=450)
    assert "needs 500.0 GB" in str(refused.value)

    instance = _place_single(_card(400), free_gb=450)
    (shard,) = instance.shard_assignments.runner_to_shard.values()
    assert shard_bytes(shard) == Memory.from_gb(400).in_bytes


def test_memory_size_is_what_a_node_must_hold() -> None:
    card = _card(500, memory_gb=300)
    instance = _place_single(card, free_gb=450)
    (shard,) = instance.shard_assignments.runner_to_shard.values()
    assert shard_bytes(shard) == Memory.from_gb(300).in_bytes

    with pytest.raises(ValueError) as refused:
        _place_single(card, free_gb=250)
    assert "needs 300.0 GB" in str(refused.value)


async def test_memory_size_round_trips_through_a_card_file(tmp_path: Path) -> None:
    folder = AsyncPath(tmp_path)
    card = _card(500, memory_gb=300)
    await card.save(folder / "card.toml")
    loaded = await ModelCard.load_from_path(folder / "card.toml")
    assert loaded.memory_size == card.memory_size
    assert loaded.resident_size == Memory.from_gb(300)

    plain = _card(500)
    await plain.save(folder / "plain.toml")
    assert "memory_size" not in (tmp_path / "plain.toml").read_text()
    assert (await ModelCard.load_from_path(folder / "plain.toml")).memory_size is None
