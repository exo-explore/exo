"""Tests for the drafter-aware placement warning (item 10).

When a model card declares `drafter_model_ids`, the placement engine still
prefers single-node (via the existing smallest-cycle-first logic). When
single-node placement is impossible because no single node has enough RAM
for the requested quant, placement falls back to multi-node and emits a
clear warning so the operator knows speculative decoding has been silently
disabled and can re-place a smaller-quant variant.
"""

from collections.abc import Iterator

import pytest
from loguru import logger as loguru_logger

from exo.master.placement import place_instance
from exo.master.tests.conftest import (
    create_node_memory,
    create_node_network,
    create_socket_connection,
)
from exo.shared.models.model_cards import ModelCard, ModelId, ModelTask
from exo.shared.topology import Topology
from exo.shared.types.commands import PlaceInstance
from exo.shared.types.common import CommandId, NodeId
from exo.shared.types.memory import Memory
from exo.shared.types.topology import Connection
from exo.shared.types.worker.instances import InstanceMeta
from exo.shared.types.worker.shards import Sharding


@pytest.fixture
def loguru_capture() -> Iterator[list[str]]:
    """Capture loguru WARNING+ messages into a list (caplog doesn't see loguru)."""
    captured: list[str] = []
    sink_id = loguru_logger.add(
        lambda message: captured.append(str(message)), level="WARNING"
    )
    try:
        yield captured
    finally:
        loguru_logger.remove(sink_id)


def _drafter_aware_card(storage_bytes: int) -> ModelCard:
    return ModelCard(
        model_id=ModelId("mlx-community/gemma-4-31b-it-8bit"),
        storage_size=Memory.from_bytes(storage_bytes),
        n_layers=60,
        hidden_size=5376,
        num_key_value_heads=16,
        supports_tensor=True,
        tasks=[ModelTask.TextGeneration],
        family="gemma",
        base_model="Gemma 4 31B",
        drafter_model_ids=[
            ModelId("mlx-community/gemma-4-e2b-it-8bit"),
            ModelId("mlx-community/gemma-4-e4b-it-8bit"),
        ],
    )


def test_drafter_aware_card_placed_single_node_when_fits(
    loguru_capture: list[str],
) -> None:
    """When a single node has enough RAM, the model lands on that node and
    no warning is emitted -- speculative decoding is preserved."""
    big_node = NodeId()
    topology = Topology()
    topology.add_node(big_node)

    card = _drafter_aware_card(20_000_000_000)
    command = PlaceInstance(
        sharding=Sharding.Pipeline,
        instance_meta=InstanceMeta.MlxRing,
        command_id=CommandId(),
        model_card=card,
        min_nodes=1,
    )

    placements = place_instance(
        command,
        topology,
        {},
        {big_node: create_node_memory(64_000_000_000)},
        {big_node: create_node_network()},
    )
    assert len(placements) == 1
    instance = next(iter(placements.values()))
    assert len(instance.shard_assignments.node_to_runner) == 1
    joined = "\n".join(loguru_capture).lower()
    assert "speculative decoding is single-device only" not in joined


def test_drafter_aware_card_warns_when_only_multi_node_fits(
    loguru_capture: list[str],
) -> None:
    """When no single node has enough RAM, placement falls back to multi-node
    and warns the operator that the drafter will be silently disabled."""
    node_a, node_b = NodeId(), NodeId()
    topology = Topology()
    topology.add_node(node_a)
    topology.add_node(node_b)
    topology.add_connection(
        Connection(source=node_a, sink=node_b, edge=create_socket_connection(2))
    )
    topology.add_connection(
        Connection(source=node_b, sink=node_a, edge=create_socket_connection(2))
    )

    # 20 GB target with hidden_size divisible by 2 nodes; only multi-node
    # fits (16 GB each). Use Tensor sharding because Gemma 4 doesn't allow
    # multi-node Pipeline.
    card = _drafter_aware_card(20_000_000_000)
    command = PlaceInstance(
        sharding=Sharding.Tensor,
        instance_meta=InstanceMeta.MlxRing,
        command_id=CommandId(),
        model_card=card,
        min_nodes=1,
    )

    placements = place_instance(
        command,
        topology,
        {},
        {
            node_a: create_node_memory(16_000_000_000),
            node_b: create_node_memory(16_000_000_000),
        },
        {
            node_a: create_node_network(),
            node_b: create_node_network(),
        },
    )
    assert len(placements) == 1
    instance = next(iter(placements.values()))
    assert len(instance.shard_assignments.node_to_runner) == 2
    joined = "\n".join(loguru_capture).lower()
    assert "speculative decoding is single-device only" in joined
    assert "smaller quant" in joined
