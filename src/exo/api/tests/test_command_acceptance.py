"""Whether a node's state shows a cluster command applied."""

from exo.api.command_acceptance import (
    command_already_applied,
    command_applied,
    undoing_command,
)
from exo.shared.models.model_cards import ModelCard, ModelId, ModelTask
from exo.shared.types.backends import Backend
from exo.shared.types.commands import (
    AddCustomModelCard,
    CreateInstance,
    DeleteCustomModelCard,
    DeleteInstance,
    DeleteInstanceLink,
    SetInstanceLink,
)
from exo.shared.types.common import NodeId
from exo.shared.types.instance_link import InstanceLink, InstanceLinkId
from exo.shared.types.memory import Memory
from exo.shared.types.state import State
from exo.shared.types.worker.instances import InstanceId
from exo.shared.types.worker.runners import RunnerId
from exo.worker.tests.unittests.conftest import (
    get_mlx_ring_instance,
    get_pipeline_shard_metadata,
)

MODEL_ID = ModelId("test-org/test-model")
INSTANCE = get_mlx_ring_instance(
    instance_id=InstanceId("instance"),
    model_id=MODEL_ID,
    node_to_runner={NodeId("node"): RunnerId("runner")},
    runner_to_shard={RunnerId("runner"): get_pipeline_shard_metadata(MODEL_ID, 0)},
)
CARD = ModelCard(
    model_id=MODEL_ID,
    storage_size=Memory.from_kb(1000),
    n_layers=10,
    hidden_size=30,
    supports_tensor=True,
    tasks=[ModelTask.TextGeneration],
    backends=[Backend.MlxMetal],
)
LINK = InstanceLinkId("link")
PREFILL = InstanceId("prefill")
DECODE = InstanceId("decode")


def test_a_launch_is_applied_once_its_instance_is_in_the_state() -> None:
    launch = CreateInstance(instance=INSTANCE)
    assert not command_applied(launch, State())
    assert command_applied(launch, State(instances={INSTANCE.instance_id: INSTANCE}))


def test_a_stop_is_applied_once_its_instance_is_gone() -> None:
    stop = DeleteInstance(instance_id=INSTANCE.instance_id)
    assert not command_applied(stop, State(instances={INSTANCE.instance_id: INSTANCE}))
    assert command_applied(stop, State())


def test_a_link_is_applied_once_the_state_holds_it_as_sent() -> None:
    # Instances named twice are kept once, as the master does
    link = SetInstanceLink(
        link_id=LINK, prefill_instances=[PREFILL, PREFILL], decode_instances=[DECODE]
    )
    as_applied = InstanceLink(
        link_id=LINK, prefill_instances=[PREFILL], decode_instances=[DECODE]
    )
    before_the_update = InstanceLink(
        link_id=LINK, prefill_instances=[PREFILL], decode_instances=[PREFILL]
    )
    assert not command_applied(link, State())
    assert not command_applied(link, State(instance_links={LINK: before_the_update}))
    assert command_applied(link, State(instance_links={LINK: as_applied}))


def test_an_unlink_is_applied_once_the_link_is_gone() -> None:
    unlink = DeleteInstanceLink(link_id=LINK)
    held = InstanceLink(
        link_id=LINK, prefill_instances=[PREFILL], decode_instances=[DECODE]
    )
    assert not command_applied(unlink, State(instance_links={LINK: held}))
    assert command_applied(unlink, State())


def test_custom_cards_are_applied_once_added_or_gone() -> None:
    add = AddCustomModelCard(model_card=CARD)
    remove = DeleteCustomModelCard(model_id=MODEL_ID)
    with_card = State(custom_model_cards={MODEL_ID: CARD})
    assert not command_applied(add, State())
    assert command_applied(add, with_card)
    assert not command_applied(remove, with_card)
    assert command_applied(remove, State())


def test_adding_a_card_again_is_a_repeat_only_when_it_is_the_same_card() -> None:
    """Adding a card again refreshes it, so it is sent unless the very same card is there."""
    add = AddCustomModelCard(model_card=CARD)
    older = CARD.model_copy(update={"n_layers": 5})
    assert command_already_applied(add, State(custom_model_cards={MODEL_ID: CARD}))
    assert not command_already_applied(add, State(custom_model_cards={MODEL_ID: older}))
    assert not command_already_applied(add, State())


def test_launches_and_new_links_are_undone_but_nothing_else() -> None:
    """A failed request's late copy is undone where trying again would make a second one."""
    launch = CreateInstance(instance=INSTANCE)
    new_link = SetInstanceLink(
        link_id=LINK, prefill_instances=[PREFILL], decode_instances=[DECODE]
    )
    held = InstanceLink(
        link_id=LINK, prefill_instances=[PREFILL], decode_instances=[PREFILL]
    )

    undo_launch = undoing_command(launch, State())
    assert isinstance(undo_launch, DeleteInstance)
    assert undo_launch.instance_id == INSTANCE.instance_id
    undo_link = undoing_command(new_link, State())
    assert isinstance(undo_link, DeleteInstanceLink)
    assert undo_link.link_id == LINK
    # Updating a link that exists: trying again updates the same link
    assert undoing_command(new_link, State(instance_links={LINK: held})) is None
    # Stops and removals: applying them late is what trying again does
    for command in (
        DeleteInstance(instance_id=INSTANCE.instance_id),
        DeleteInstanceLink(link_id=LINK),
        AddCustomModelCard(model_card=CARD),
        DeleteCustomModelCard(model_id=MODEL_ID),
    ):
        assert undoing_command(command, State()) is None
