"""Whether a node's state shows a cluster command applied."""

from exo.api.command_acceptance import (
    changed_by,
    command_already_applied,
    command_applied,
    undoing_command,
)
from exo.master.tests.test_requests_to_deleted_instances import deployment_of
from exo.shared.models.model_cards import ModelCard, ModelId, ModelTask
from exo.shared.types.backends import Backend
from exo.shared.types.commands import (
    AddCustomModelCard,
    CreateDeployment,
    DeleteCustomModelCard,
    DeleteDeployment,
    DeleteInstanceLink,
    ResumeDeployment,
    SetInstanceLink,
)
from exo.shared.types.common import DeploymentId, NodeId
from exo.shared.types.instance_link import InstanceLink, InstanceLinkId
from exo.shared.types.memory import Memory
from exo.shared.types.state import State
from exo.shared.types.worker.instances import Instance, InstanceId
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
DEPLOYMENT = deployment_of(INSTANCE)
LINK = InstanceLinkId("link")
PREFILL = InstanceId("prefill")
DECODE = InstanceId("decode")


def launch_of(instance: Instance = INSTANCE) -> CreateDeployment:
    return CreateDeployment(
        deployment_id=instance.deployment_id,
        spec=deployment_of(instance).spec,
        model_card=CARD,
        instance=instance,
    )


def test_a_launch_is_applied_once_its_deployment_is_in_the_state() -> None:
    launch = launch_of()
    assert not command_applied(launch, State())
    assert command_applied(
        launch, State(deployments={DEPLOYMENT.deployment_id: DEPLOYMENT})
    )


def test_a_placement_launched_by_another_request_counts_as_applied() -> None:
    """The same placement posted twice at once: each request makes its own deployment, and
    the master runs the one it took first. The other request's launch is answered by it."""
    other = launch_of().model_copy(update={"deployment_id": DeploymentId()})
    running = State(
        instances={INSTANCE.instance_id: INSTANCE},
        deployments={DEPLOYMENT.deployment_id: DEPLOYMENT},
    )
    assert command_applied(other, running)
    # A retry of the very same placement takes over from a failed request for it
    assert changed_by(other) == changed_by(launch_of()) == INSTANCE.instance_id


def test_a_stop_is_applied_once_its_deployment_is_gone() -> None:
    stop = DeleteDeployment(deployment_id=DEPLOYMENT.deployment_id)
    assert not command_applied(
        stop, State(deployments={DEPLOYMENT.deployment_id: DEPLOYMENT})
    )
    assert command_applied(stop, State())


def test_a_resume_is_applied_once_the_revision_it_resumed_has_moved_on() -> None:
    resume = ResumeDeployment(deployment_id=DEPLOYMENT.deployment_id, revision=0)
    resumed = DEPLOYMENT.model_copy(update={"revision": 1})
    assert not command_applied(
        resume, State(deployments={DEPLOYMENT.deployment_id: DEPLOYMENT})
    )
    assert command_applied(
        resume, State(deployments={DEPLOYMENT.deployment_id: resumed})
    )
    # Stopped meanwhile: nothing left to resume
    assert command_applied(resume, State())


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
    launch = launch_of()
    new_link = SetInstanceLink(
        link_id=LINK, prefill_instances=[PREFILL], decode_instances=[DECODE]
    )
    held = InstanceLink(
        link_id=LINK, prefill_instances=[PREFILL], decode_instances=[PREFILL]
    )

    undo_launch = undoing_command(launch, State())
    assert isinstance(undo_launch, DeleteDeployment)
    assert undo_launch.deployment_id == launch.deployment_id
    undo_link = undoing_command(new_link, State())
    assert isinstance(undo_link, DeleteInstanceLink)
    assert undo_link.link_id == LINK
    # Updating a link that exists: trying again updates the same link
    assert undoing_command(new_link, State(instance_links={LINK: held})) is None
    # Stops and removals: applying them late is what trying again does
    for command in (
        DeleteDeployment(deployment_id=DEPLOYMENT.deployment_id),
        ResumeDeployment(deployment_id=DEPLOYMENT.deployment_id, revision=0),
        DeleteInstanceLink(link_id=LINK),
        AddCustomModelCard(model_card=CARD),
        DeleteCustomModelCard(model_id=MODEL_ID),
    ):
        assert undoing_command(command, State()) is None
