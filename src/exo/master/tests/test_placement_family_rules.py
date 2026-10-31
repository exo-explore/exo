"""Placement applies each model family's rules to the built-in cards of that family."""

import tomllib
from pathlib import Path

import pytest

from exo.master.placement import place_instance
from exo.master.tests.conftest import (
    create_node_memory,
    create_node_network,
    create_socket_connection,
)
from exo.shared.constants import RESOURCES_DIR
from exo.shared.models.model_cards import ModelCard
from exo.shared.topology import Topology
from exo.shared.types.backends import Backend
from exo.shared.types.commands import PlaceInstance
from exo.shared.types.common import CommandId, NodeId
from exo.shared.types.topology import Connection
from exo.shared.types.worker.instances import InstanceMeta
from exo.shared.types.worker.shards import Sharding

_TB = 1024**4


def _card(family: str, name: str) -> ModelCard:
    path = Path(RESOURCES_DIR) / "inference_model_cards" / family / f"{name}.toml"
    return ModelCard.model_validate(tomllib.loads(path.read_text()))


def _place(card: ModelCard, sharding: Sharding, min_nodes: int) -> int:
    a, b = NodeId(), NodeId()
    topology = Topology()
    topology.add_node(a)
    topology.add_node(b)
    topology.add_connection(
        Connection(source=a, sink=b, edge=create_socket_connection(1))
    )
    topology.add_connection(
        Connection(source=b, sink=a, edge=create_socket_connection(2))
    )
    node_memory = {n: create_node_memory(_TB) for n in (a, b)}
    command = PlaceInstance(
        command_id=CommandId(),
        model_card=card,
        sharding=sharding,
        instance_meta=InstanceMeta.MlxRing,
        min_nodes=min_nodes,
    )
    placement = place_instance(
        command,
        topology,
        {},
        node_memory,
        {n: create_node_network() for n in (a, b)},
        {n: [Backend.MlxMetal] for n in (a, b)},
    )
    (instance,) = placement.values()
    return len(instance.shard_assignments.node_to_runner)


def test_deepseek_v4_tensor_parallel_ignores_its_single_kv_head() -> None:
    card = _card("deepseek_v4", "mlx-community--DeepSeek-V4-Flash")
    assert card.num_key_value_heads == 1
    assert _place(card, Sharding.Tensor, min_nodes=2) == 2


def test_other_models_need_kv_heads_that_divide_by_the_node_count() -> None:
    card = _card("deepseek_v4", "mlx-community--DeepSeek-V4-Flash").model_copy(
        update={"model_type": "deepseek_v3"}
    )
    with pytest.raises(ValueError, match="No tensor sharding found"):
        _place(card, Sharding.Tensor, min_nodes=2)


def test_deepseek_v31_8bit_has_no_pipeline_parallelism() -> None:
    card = _card("deepseek_v3", "mlx-community--DeepSeek-V3.1-8bit")
    with pytest.raises(
        ValueError,
        match=r"^Pipeline parallelism is not supported for DeepSeek V3.1 \(8-bit\)$",
    ):
        _place(card, Sharding.Pipeline, min_nodes=1)
    other = _card("deepseek_v3", "mlx-community--DeepSeek-V3.1-4bit")
    assert _place(other, Sharding.Pipeline, min_nodes=2) == 2


def test_gemma_4_pipeline_parallelism_stays_on_one_node() -> None:
    card = _card("gemma", "mlx-community--gemma-4-31b-it-4bit")
    assert _place(card, Sharding.Pipeline, min_nodes=1) == 1
    with pytest.raises(
        ValueError,
        match=r"^Pipeline parallelism is not supported for Gemma 4; use tensor parallelism instead\.$",
    ):
        _place(card, Sharding.Pipeline, min_nodes=2)
    assert _place(card, Sharding.Tensor, min_nodes=2) == 2


def _place_tensor_on_ring(card: ModelCard, nodes: int) -> int:
    ids = [NodeId() for _ in range(nodes)]
    topology = Topology()
    for node in ids:
        topology.add_node(node)
    for index, node in enumerate(ids):
        neighbour = ids[(index + 1) % nodes]
        for port, (source, sink) in enumerate(((node, neighbour), (neighbour, node))):
            topology.add_connection(
                Connection(
                    source=source,
                    sink=sink,
                    edge=create_socket_connection(2 * index + port + 1),
                )
            )
    command = PlaceInstance(
        command_id=CommandId(),
        model_card=card,
        sharding=Sharding.Tensor,
        instance_meta=InstanceMeta.MlxRing,
        min_nodes=nodes,
    )
    placement = place_instance(
        command,
        topology,
        {},
        {n: create_node_memory(_TB) for n in ids},
        {n: create_node_network() for n in ids},
        {n: [Backend.MlxMetal] for n in ids},
    )
    (instance,) = placement.values()
    return len(instance.shard_assignments.node_to_runner)


@pytest.mark.parametrize(
    "name",
    [
        "deepseek-ai--DeepSeek-V4-Pro-0813",
        "mlx-community--DeepSeek-V4-Pro",
    ],
)
def test_deepseek_v4_pro_tensor_parallelism_needs_whole_head_groups(
    name: str,
) -> None:
    """V4 shards every output group's 8 heads evenly, so 7 nodes can't hold V4-Pro even
    though its hidden size (7168) divides by 7."""
    card = _card("deepseek_v4", name)
    assert card.hidden_size % 7 == 0
    with pytest.raises(ValueError, match="No tensor sharding found"):
        _place_tensor_on_ring(card, 7)
    assert _place_tensor_on_ring(card, 8) == 8
    assert _place_tensor_on_ring(card, 4) == 4
