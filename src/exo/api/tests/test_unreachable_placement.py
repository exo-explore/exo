"""The API places a model, and previews its placements, only on the nodes this node reaches now,
as its membership sees them: on a node asleep, frozen or cut off (Suspect, or Gone and kept listed
in an even split) the model couldn't load. A launch naming such a node is answered at once: sent,
the master would refuse it, and the API would wait for it to show until CLUSTER_COMMAND_TIMEOUT."""

import pytest
from fastapi import HTTPException

from exo.api.main import (
    API,
    LAUNCH_UNREACHABLE_MESSAGE,
    ModelUnreachableError,
)
from exo.api.tests.applying_master import sent_while_applying
from exo.api.types import CreateInstanceParams, PlaceInstanceParams
from exo.master.tests.conftest import create_node_network, create_socket_connection
from exo.routing.tests.fake_membership import FakeMembership
from exo.shared.models.model_cards import ModelCard, ModelId, ModelTask, card_cache
from exo.shared.topology import Topology
from exo.shared.types.backends import Backend
from exo.shared.types.commands import CreateInstance, ForwarderCommand
from exo.shared.types.common import NodeId, SystemId
from exo.shared.types.memory import Memory
from exo.shared.types.profiling import MemoryUsage
from exo.shared.types.state import State
from exo.shared.types.topology import Connection
from exo.shared.types.worker.instances import InstanceMeta
from exo.shared.types.worker.shards import Sharding
from exo.utils.channels import Receiver, channel

MODEL = ModelCard(
    model_id=ModelId("test-org/small-model"),
    storage_size=Memory.from_gb(8),
    n_layers=16,
    hidden_size=1024,
    supports_tensor=True,
    tasks=[ModelTask.TextGeneration],
    backends=[Backend.MlxMetal],
)
API_NODE = NodeId("api-node")
SMALL = NodeId("small-node")
# Placement prefers the node with the most memory: this one, while it is reachable
LARGE = NodeId("large-node")


@pytest.fixture(autouse=True)
def known_model(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(card_cache, "cc", {MODEL.model_id: MODEL})


def _memory(free_gb: float) -> MemoryUsage:
    return MemoryUsage(
        ram_total=Memory.from_gb(2 * free_gb),
        ram_available=Memory.from_gb(free_gb),
        swap_total=Memory(),
        swap_available=Memory(),
    )


def pair_api() -> tuple[API, FakeMembership, Receiver[ForwarderCommand]]:
    """Two Macs that can each hold the model, the larger one preferred, and the membership of
    the node the API runs on: it reaches both."""
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
        node_memory={SMALL: _memory(24), LARGE: _memory(96)},
        node_network={node: create_node_network() for node in (SMALL, LARGE)},
        node_backends={node: [Backend.MlxMetal] for node in (SMALL, LARGE)},
    )
    api.paused = False
    api._system_id = SystemId()  # pyright: ignore[reportPrivateUsage]
    api.command_sender, commands = channel[ForwarderCommand]()
    membership = FakeMembership(API_NODE, SMALL, LARGE)
    api.membership = membership
    return api, membership, commands


def nodes_of(instance: CreateInstance) -> set[NodeId]:
    return set(instance.instance.shard_assignments.node_to_runner)


async def test_a_placement_is_made_on_the_nodes_it_reaches() -> None:
    api, membership, commands = pair_api()
    # Reaching both, the larger node gets the model
    reachable = await api.get_placement(
        MODEL.model_id, Sharding.Pipeline, InstanceMeta.MlxRing, 1
    )
    assert set(reachable.shard_assignments.node_to_runner) == {LARGE}

    await membership.lose(LARGE)
    _, sent = await sent_while_applying(
        api,
        commands,
        lambda: api.place_instance(PlaceInstanceParams(model_id=MODEL.model_id)),
    )

    (launch,) = sent
    assert isinstance(launch, CreateInstance)
    assert nodes_of(launch) == {SMALL}


async def test_previews_leave_out_a_node_it_cant_reach() -> None:
    api, membership, _ = pair_api()
    await membership.lose(LARGE, "Gone")

    previews = (await api.get_placement_previews(MODEL.model_id)).previews

    placed = [preview.instance for preview in previews if preview.instance is not None]
    assert placed, previews
    assert all(
        set(instance.shard_assignments.node_to_runner) == {SMALL} for instance in placed
    )


async def test_a_launch_naming_a_node_it_cant_reach_is_answered_at_once() -> None:
    api, membership, commands = pair_api()
    on_large = await api.get_placement(
        MODEL.model_id, Sharding.Pipeline, InstanceMeta.MlxRing, 1
    )
    await membership.lose(LARGE)

    with pytest.raises(HTTPException) as refused:
        await api.create_instance(CreateInstanceParams(instance=on_large))

    assert isinstance(refused.value, ModelUnreachableError)
    assert refused.value.status_code == 503
    assert refused.value.code == "device_unreachable"
    assert refused.value.detail == LAUNCH_UNREACHABLE_MESSAGE
    assert commands.collect() == []
