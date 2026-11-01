"""A family's tensor_parallel_divides rule keeps tensor parallelism off node counts that
don't divide those numbers; without the rule, placement is unchanged."""

from collections.abc import Callable

import pytest

from exo.master import placement
from exo.master.tests.conftest import (
    create_node_memory,
    create_node_network,
    create_socket_connection,
)
from exo.shared.models.model_cards import ModelCard, ModelId, ModelTask
from exo.shared.topology import Topology
from exo.shared.types.backends import Backend
from exo.shared.types.commands import PlaceInstance
from exo.shared.types.common import CommandId, NodeId
from exo.shared.types.memory import Memory
from exo.shared.types.topology import Connection
from exo.shared.types.worker.instances import InstanceMeta
from exo.shared.types.worker.shards import Sharding
from exo.worker.engines.mlx.models.spec import PlacementRules

_TB = 1024**4


def _card() -> ModelCard:
    return ModelCard(
        model_id=ModelId("test-org/grouped"),
        model_type="grouped",
        storage_size=Memory.from_gb(30),
        n_layers=12,
        hidden_size=1536,
        supports_tensor=True,
        tasks=[ModelTask.TextGeneration],
        backends=[Backend.MlxMetal],
    )


def _place_tensor(nodes: int) -> int:
    ids = [NodeId() for _ in range(nodes)]
    topology = Topology()
    for node in ids:
        topology.add_node(node)
    port = 0
    for source in ids:
        for sink in ids:
            if source != sink:
                port += 1
                topology.add_connection(
                    Connection(
                        source=source, sink=sink, edge=create_socket_connection(port)
                    )
                )
    placed = placement.place_instance(
        PlaceInstance(
            command_id=CommandId(),
            model_card=_card(),
            sharding=Sharding.Tensor,
            instance_meta=InstanceMeta.MlxRing,
            min_nodes=nodes,
        ),
        topology,
        {},
        {node: create_node_memory(_TB) for node in ids},
        {node: create_node_network() for node in ids},
        {node: [Backend.MlxMetal] for node in ids},
    )
    (instance,) = placed.values()
    return len(instance.shard_assignments.node_to_runner)


def _rules(rules: PlacementRules) -> Callable[[str | None], PlacementRules]:
    def lookup(_model_type: str | None) -> PlacementRules:
        return rules

    return lookup


def test_without_the_rule_three_nodes_are_allowed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        placement, "placement_rules_for_model_type", _rules(PlacementRules())
    )
    assert _place_tensor(3) == 3


def test_the_rule_refuses_node_counts_that_do_not_divide(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        placement,
        "placement_rules_for_model_type",
        _rules(PlacementRules(tensor_parallel_divides=(8,))),
    )
    with pytest.raises(ValueError):
        _place_tensor(3)
    assert _place_tensor(2) == 2
