"""The placement endpoints judge each node by the share of the model it would hold, in the
memory a model can use on it: a 16 GB Mac can't take half of a model that only fits in a
16 GB and a 64 GB Mac together."""

import pytest
from fastapi import HTTPException

from exo.api.main import API
from exo.api.types import CreateInstanceParams, PlaceInstanceParams
from exo.master.tests.conftest import create_node_network, create_socket_connection
from exo.shared.models.model_cards import ModelCard, ModelId, ModelTask, card_cache
from exo.shared.topology import Topology
from exo.shared.types.backends import Backend
from exo.shared.types.commands import CreateInstance, ForwarderCommand
from exo.shared.types.common import NodeId, SystemId
from exo.shared.types.memory import Memory
from exo.shared.types.profiling import MemoryUsage, NodeIdentity
from exo.shared.types.state import State
from exo.shared.types.topology import Connection
from exo.shared.types.worker.instances import InstanceMeta
from exo.shared.types.worker.shards import Sharding
from exo.utils.channels import Receiver, channel

GB = 1024**3
MODEL = ModelCard(
    model_id=ModelId("test-org/test-model"),
    storage_size=Memory.from_gb(27.5),
    n_layers=40,
    hidden_size=1024,
    supports_tensor=True,
    tasks=[ModelTask.TextGeneration],
    backends=[Backend.MlxMetal],
)
SMALL = NodeId("small-node")
LARGE = NodeId("large-node")


@pytest.fixture(autouse=True)
def known_model(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(card_cache, "cc", {MODEL.model_id: MODEL})


def _memory(free_gb: float, gpu_working_set_gb: float) -> MemoryUsage:
    return MemoryUsage(
        ram_total=Memory.from_gb(2 * free_gb),
        ram_available=Memory.from_gb(free_gb),
        swap_total=Memory(),
        swap_available=Memory(),
        gpu_working_set=Memory.from_gb(gpu_working_set_gb),
    )


def unequal_pair_api() -> tuple[API, Receiver[ForwarderCommand]]:
    """A 16 GB Mac (12 GB free, its GPU can use 11.8 GB) and a 64 GB Mac."""
    topology = Topology()
    for node in (SMALL, LARGE):
        topology.add_node(node)
    topology.add_connection(
        Connection(source=SMALL, sink=LARGE, edge=create_socket_connection(1))
    )
    topology.add_connection(
        Connection(source=LARGE, sink=SMALL, edge=create_socket_connection(2))
    )
    api = object.__new__(API)
    api.state = State(
        topology=topology,
        node_memory={SMALL: _memory(12, 11.8), LARGE: _memory(58, 51.8)},
        node_network={node: create_node_network() for node in (SMALL, LARGE)},
        node_backends={node: [Backend.MlxMetal] for node in (SMALL, LARGE)},
        node_identities={SMALL: NodeIdentity(friendly_name="Small Mac")},
    )
    api.paused = False
    api._system_id = SystemId()  # pyright: ignore[reportPrivateUsage]
    api.command_sender, commands = channel[ForwarderCommand]()
    return api, commands


async def test_tensor_sharding_a_node_cannot_hold_is_refused_naming_the_node() -> None:
    api, commands = unequal_pair_api()

    with pytest.raises(HTTPException) as refused:
        await api.place_instance(
            PlaceInstanceParams(
                model_id=MODEL.model_id, sharding=Sharding.Tensor, min_nodes=2
            )
        )

    assert refused.value.status_code == 400
    assert refused.value.detail == (
        "Tensor sharding gives each of 2 nodes 13.8 GB of test-org/test-model, but Small "
        "Mac can use only about 11.8 GB. Use Pipeline sharding, which splits the model by "
        "each node's memory, or nodes with more memory"
    )
    assert commands.collect() == []


async def test_previews_offer_pipeline_with_each_nodes_share() -> None:
    api, _ = unequal_pair_api()

    previews = (await api.get_placement_previews(MODEL.model_id)).previews

    two_nodes = [
        preview
        for preview in previews
        if preview.instance_meta == InstanceMeta.MlxRing
        and preview.instance is not None
        and len(preview.instance.shard_assignments.node_to_runner) == 2
    ]
    assert [preview.sharding for preview in two_nodes] == [Sharding.Pipeline]
    shares = two_nodes[0].memory_delta_by_node
    assert shares is not None
    # Layers in proportion to what each node can use, not half each
    assert shares[SMALL] <= 11.8 * GB
    assert shares[SMALL] + shares[LARGE] == MODEL.storage_size.in_bytes
    assert any(
        preview.sharding == Sharding.Tensor
        and preview.error is not None
        and "Small Mac can use only about 11.8 GB" in preview.error
        for preview in previews
    )


async def test_an_instance_whose_shard_a_node_cannot_hold_is_refused() -> None:
    api, commands = unequal_pair_api()
    # A tensor placement made on a pair that could hold it
    roomy, _ = unequal_pair_api()
    roomy.state = roomy.state.model_copy(
        update={"node_memory": {SMALL: _memory(58, 51.8), LARGE: _memory(58, 51.8)}}
    )
    tensor = await roomy.get_placement(
        MODEL.model_id, Sharding.Tensor, InstanceMeta.MlxRing, 2
    )

    with pytest.raises(HTTPException) as refused:
        await api.create_instance(CreateInstanceParams(instance=tensor))

    assert refused.value.status_code == 400
    assert refused.value.detail == (
        "Not enough memory: Small Mac can use only about 11.8 GB, but its shard of "
        "test-org/test-model needs 13.8 GB"
    )
    assert commands.collect() == []


async def test_an_instance_every_node_can_hold_is_created() -> None:
    api, commands = unequal_pair_api()
    pipeline = await api.get_placement(
        MODEL.model_id, Sharding.Pipeline, InstanceMeta.MlxRing, 2
    )

    await api.create_instance(CreateInstanceParams(instance=pipeline))

    assert [type(c.command) for c in commands.collect()] == [CreateInstance]
