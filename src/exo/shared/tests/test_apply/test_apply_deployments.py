from exo.shared.apply import event_apply
from exo.shared.models.model_cards import ModelCard, ModelId, ModelTask
from exo.shared.types.backends import Backend
from exo.shared.types.deployments import Deployment, DeploymentId
from exo.shared.types.events import (
    DeploymentCreated,
    DeploymentDeleted,
    DeploymentPlaced,
    DeploymentPlacementFailed,
    IndexedEvent,
)
from exo.shared.types.memory import Memory
from exo.shared.types.state import State
from exo.shared.types.worker.instances import InstanceId, InstanceMeta
from exo.shared.types.worker.shards import Sharding


def _deployment(model_id: str) -> Deployment:
    return Deployment(
        deployment_id=DeploymentId(),
        model_card=ModelCard(
            model_id=ModelId(model_id),
            storage_size=Memory.from_bytes(1000),
            n_layers=10,
            hidden_size=30,
            supports_tensor=True,
            tasks=[ModelTask.TextGeneration],
            backends=[Backend.MlxMetal],
        ),
        sharding=Sharding.Pipeline,
        instance_meta=InstanceMeta.MlxRing,
        min_nodes=1,
    )


def test_created_deployment_is_added():
    deployment = _deployment("test/model-a")

    state = event_apply(DeploymentCreated(deployment=deployment), State())

    assert state.deployments == {deployment.deployment_id: deployment}


def test_a_second_deployment_of_a_model_is_ignored():
    """Two requests can both pass the API's check before either is applied: the first wins."""
    first = _deployment("test/model-a")
    state = event_apply(DeploymentCreated(deployment=first), State())

    state = event_apply(
        DeploymentCreated(deployment=_deployment("test/model-a")), state
    )
    other = _deployment("test/model-b")
    state = event_apply(DeploymentCreated(deployment=other), state)

    assert state.deployments == {
        first.deployment_id: first,
        other.deployment_id: other,
    }


def test_deleted_deployment_is_removed():
    kept, deleted = _deployment("test/model-a"), _deployment("test/model-b")
    state = State(
        deployments={kept.deployment_id: kept, deleted.deployment_id: deleted}
    )

    state = event_apply(DeploymentDeleted(deployment_id=deleted.deployment_id), state)
    state = event_apply(DeploymentDeleted(deployment_id=DeploymentId()), state)

    assert state.deployments == {kept.deployment_id: kept}


def test_placed_records_the_instance_and_clears_the_error():
    deployment = _deployment("test/model-a").model_copy(
        update={"placement_error": "No cycles found with sufficient memory"}
    )
    state = State(deployments={deployment.deployment_id: deployment})
    instance_id = InstanceId()

    state = event_apply(
        DeploymentPlaced(
            deployment_id=deployment.deployment_id, instance_id=instance_id
        ),
        state,
    )

    placed = state.deployments[deployment.deployment_id]
    assert placed.instance_id == instance_id
    assert placed.placement_error is None


def test_placement_failed_records_why_and_keeps_the_last_instance():
    instance_id = InstanceId()
    deployment = _deployment("test/model-a").model_copy(
        update={"instance_id": instance_id}
    )
    state = State(deployments={deployment.deployment_id: deployment})

    state = event_apply(
        DeploymentPlacementFailed(
            deployment_id=deployment.deployment_id, error="no room"
        ),
        state,
    )

    failed = state.deployments[deployment.deployment_id]
    assert failed.placement_error == "no room"
    assert failed.instance_id == instance_id


def test_events_for_a_deleted_deployment_change_nothing():
    """A placement can still be on its way when its deployment is deleted."""
    state = State()

    for event in (
        DeploymentPlaced(deployment_id=DeploymentId(), instance_id=InstanceId()),
        DeploymentPlacementFailed(deployment_id=DeploymentId(), error="no room"),
    ):
        assert event_apply(event, state) == state


def test_deployments_survive_serialization():
    """The state is sent to nodes that join, and the events are stored in the event log."""
    deployment = _deployment("test/model-a").model_copy(
        update={"instance_id": InstanceId(), "placement_error": "no room"}
    )
    state = State(deployments={deployment.deployment_id: deployment})

    restored = State.model_validate_json(state.model_dump_json())
    assert restored.deployments == state.deployments

    event = IndexedEvent(idx=0, event=DeploymentCreated(deployment=deployment))
    assert IndexedEvent.model_validate_json(event.model_dump_json()) == event
