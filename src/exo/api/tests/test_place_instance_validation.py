"""POST /place_instance refuses a placement the cluster can't hold, instead of accepting it."""

import pytest
from fastapi import HTTPException

from exo.api.main import API
from exo.api.types import PlaceInstanceParams
from exo.master.tests.conftest import create_node_memory, create_node_network
from exo.shared.models.model_cards import ModelCard, ModelId, ModelTask, card_cache
from exo.shared.topology import Topology
from exo.shared.types.backends import Backend
from exo.shared.types.commands import ForwarderCommand, PlaceInstance
from exo.shared.types.common import NodeId, SystemId
from exo.shared.types.memory import Memory
from exo.shared.types.state import State
from exo.utils.channels import Receiver, channel

MODEL = ModelCard(
    model_id=ModelId("test-org/test-model"),
    storage_size=Memory.from_kb(1000),
    n_layers=10,
    hidden_size=30,
    supports_tensor=True,
    tasks=[ModelTask.TextGeneration],
    backends=[Backend.MlxMetal],
)
NODE = NodeId("only-node")


@pytest.fixture(autouse=True)
def known_model(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(card_cache, "cc", {MODEL.model_id: MODEL})


def single_node_api() -> tuple[API, Receiver[ForwarderCommand]]:
    topology = Topology()
    topology.add_node(NODE)
    api = object.__new__(API)
    api.state = State(
        topology=topology,
        node_memory={NODE: create_node_memory(10_000_000)},
        node_network={NODE: create_node_network()},
        node_backends={NODE: [Backend.MlxMetal]},
    )
    api.paused = False
    api._system_id = SystemId()  # pyright: ignore[reportPrivateUsage]
    api.command_sender, commands = channel[ForwarderCommand]()
    return api, commands


async def test_a_placement_that_fits_is_sent_to_the_master() -> None:
    api, commands = single_node_api()

    await api.place_instance(PlaceInstanceParams(model_id=MODEL.model_id))

    assert [type(c.command) for c in commands.collect()] == [PlaceInstance]


async def test_a_placement_that_cannot_fit_is_refused_with_the_reason() -> None:
    api, commands = single_node_api()

    with pytest.raises(HTTPException) as refused:
        await api.place_instance(
            PlaceInstanceParams(model_id=MODEL.model_id, min_nodes=2)
        )

    assert refused.value.status_code == 400
    assert refused.value.detail
    assert commands.collect() == []
