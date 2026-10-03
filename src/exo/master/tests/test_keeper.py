"""The keeper as a pure function of the cluster state.

Each test builds a State through the same events the cluster applies, steps a Keeper at chosen
times, and applies what it returns, so the tests follow the keeper exactly as the master runs it.
"""

import random
from collections.abc import Iterable

import pytest

from exo.master.keeper import (
    FIRST_BACKOFF,
    MAX_BACKOFF,
    PLACEMENT_TIMEOUT,
    STARTUP_GRACE,
    UNPLACEABLE_RETRY,
    Keeper,
    deployment_status,
    has_instance,
)
from exo.master.placement import place_instance
from exo.master.tests.conftest import (
    create_node_memory,
    create_node_network,
    create_socket_connection,
)
from exo.shared.apply import event_apply
from exo.shared.models.model_cards import ModelCard, ModelId, ModelTask
from exo.shared.topology import Topology
from exo.shared.types.backends import Backend
from exo.shared.types.commands import (
    CancelDownload,
    DeleteDeployment,
    PlaceInstance,
)
from exo.shared.types.common import NodeId
from exo.shared.types.deployments import Deployment, DeploymentId
from exo.shared.types.events import (
    DeploymentCreated,
    DeploymentDeleted,
    DeploymentPlaced,
    DeploymentPlacementFailed,
    Event,
    InstanceCreated,
    InstanceDeleted,
    NodeTimedOut,
    RunnerStatusUpdated,
)
from exo.shared.types.memory import Memory
from exo.shared.types.state import State
from exo.shared.types.topology import Connection
from exo.shared.types.worker.downloads import DownloadOngoing, DownloadProgressData
from exo.shared.types.worker.instances import InstanceId, InstanceMeta
from exo.shared.types.worker.runners import (
    RunnerFailed,
    RunnerLoading,
    RunnerReady,
    RunnerRunning,
)
from exo.shared.types.worker.shards import Sharding

START = 1000.0
AFTER_GRACE = START + STARTUP_GRACE
NODE_MEMORY = 10_000_000


def _model(name: str, size_bytes: int = 1_000_000) -> ModelCard:
    return ModelCard(
        model_id=ModelId(name),
        storage_size=Memory.from_bytes(size_bytes),
        n_layers=10,
        hidden_size=30,
        supports_tensor=True,
        tasks=[ModelTask.TextGeneration],
        backends=[Backend.MlxMetal],
    )


MODEL_A = _model("test/model-a")
MODEL_B = _model("test/model-b")


def _cluster(node_count: int, memory: int = NODE_MEMORY) -> State:
    """A cluster of fully connected Metal nodes."""
    nodes = [NodeId() for _ in range(node_count)]
    topology = Topology()
    for node in nodes:
        topology.add_node(node)
    port = 0
    for source in nodes:
        for sink in nodes:
            if source != sink:
                port += 1
                topology.add_connection(
                    Connection(
                        source=source, sink=sink, edge=create_socket_connection(port)
                    )
                )
    return State(
        topology=topology,
        node_memory={node: create_node_memory(memory) for node in nodes},
        node_network={node: create_node_network() for node in nodes},
        node_backends={node: [Backend.MlxMetal] for node in nodes},
    )


def _deployment(model: ModelCard) -> Deployment:
    return Deployment(
        deployment_id=DeploymentId(),
        model_card=model,
        sharding=Sharding.Pipeline,
        instance_meta=InstanceMeta.MlxRing,
        min_nodes=1,
    )


def _apply(state: State, events: Iterable[Event]) -> State:
    for event in events:
        state = event_apply(event, state)
    return state


def _deploy(state: State, *models: ModelCard) -> tuple[State, list[Deployment]]:
    deployments = [_deployment(model) for model in models]
    state = _apply(state, [DeploymentCreated(deployment=d) for d in deployments])
    return state, deployments


def _without_ids(events: Iterable[Event]) -> list[tuple[str, dict[str, object]]]:
    """Events to compare: every event gets a fresh event_id."""
    return [(type(e).__name__, e.model_dump(exclude={"event_id"})) for e in events]


def _placed(events: list[Event]) -> InstanceId:
    """The instance a keeper step placed."""
    (placed,) = [e for e in events if isinstance(e, DeploymentPlaced)]
    return placed.instance_id


def _make_ready(state: State, instance_id: InstanceId) -> State:
    runner_ids = state.instances[instance_id].shard_assignments.node_to_runner.values()
    return _apply(
        state,
        [
            RunnerStatusUpdated(runner_id=runner_id, runner_status=RunnerReady())
            for runner_id in runner_ids
        ],
    )


def _lose(state: State, instance_id: InstanceId) -> State:
    return event_apply(InstanceDeleted(instance_id=instance_id), state)


def _instances_of(state: State, model: ModelCard) -> list[InstanceId]:
    return [
        instance_id
        for instance_id, instance in state.instances.items()
        if instance.shard_assignments.model_id == model.model_id
    ]


def _place_by_hand(state: State, model: ModelCard) -> tuple[State, InstanceId]:
    """Place an instance as a /place_instance request would, without the keeper."""
    placement = place_instance(
        PlaceInstance(
            model_card=model,
            sharding=Sharding.Pipeline,
            instance_meta=InstanceMeta.MlxRing,
            min_nodes=1,
        ),
        state.topology,
        state.instances,
        state.node_memory,
        state.node_network,
        state.node_backends,
    )
    (instance_id,) = [i for i in placement if i not in state.instances]
    return (
        event_apply(InstanceCreated(instance=placement[instance_id]), state),
        instance_id,
    )


# Placing


def test_does_nothing_without_deployments():
    keeper = Keeper(started_at=START)
    state = _cluster(2)

    assert keeper.step(state, AFTER_GRACE) == []
    assert keeper.step(state, AFTER_GRACE + 3600) == []


def test_waits_for_nodes_to_report_their_instances_after_starting():
    keeper = Keeper(started_at=START)
    state, _ = _deploy(_cluster(1), MODEL_A)

    for now in (START, START + 1, AFTER_GRACE - 0.001):
        assert keeper.step(state, now) == []
    assert keeper.step(state, AFTER_GRACE) != []


def test_places_an_instance_for_a_deployment_whose_model_has_none():
    keeper = Keeper(started_at=START)
    state, (deployment,) = _deploy(_cluster(2), MODEL_A)

    events = keeper.step(state, AFTER_GRACE)
    state = _apply(state, events)

    instance_id = _placed(events)
    created = [e for e in events if isinstance(e, InstanceCreated)]
    assert [e.instance.instance_id for e in created] == [instance_id]
    assert state.instances[instance_id].shard_assignments.model_id == MODEL_A.model_id
    assert state.deployments[deployment.deployment_id].instance_id == instance_id
    assert keeper.step(state, AFTER_GRACE + 1) == []


def test_places_with_the_deployments_options():
    keeper = Keeper(started_at=START)
    deployment = _deployment(MODEL_A).model_copy(update={"min_nodes": 3})
    state = event_apply(DeploymentCreated(deployment=deployment), _cluster(3))

    instance_id = _placed(events := keeper.step(state, AFTER_GRACE))
    state = _apply(state, events)

    assert len(state.instances[instance_id].shard_assignments.node_to_runner) == 3


def test_any_instance_of_the_model_keeps_the_deployment_satisfied():
    keeper = Keeper(started_at=START)
    state, _ = _deploy(_cluster(2), MODEL_A)
    state, _ = _place_by_hand(state, MODEL_A)

    for now in (AFTER_GRACE, AFTER_GRACE + 60, AFTER_GRACE + 3600):
        assert keeper.step(state, now) == []


def test_an_instance_of_another_model_does_not_count():
    keeper = Keeper(started_at=START)
    state, _ = _deploy(_cluster(2), MODEL_A)
    state, _ = _place_by_hand(state, MODEL_B)

    instance_id = _placed(events := keeper.step(state, AFTER_GRACE))
    state = _apply(state, events)

    assert state.instances[instance_id].shard_assignments.model_id == MODEL_A.model_id
    assert len(_instances_of(state, MODEL_B)) == 1


def test_places_for_one_deployment_per_step():
    keeper = Keeper(started_at=START)
    state, _ = _deploy(_cluster(2), MODEL_A, MODEL_B)

    first = keeper.step(state, AFTER_GRACE)
    assert len([e for e in first if isinstance(e, InstanceCreated)]) == 1
    state = _apply(state, first)

    second = keeper.step(state, AFTER_GRACE + 1)
    assert len([e for e in second if isinstance(e, InstanceCreated)]) == 1
    state = _apply(state, second)

    assert len(_instances_of(state, MODEL_A)) == 1
    assert len(_instances_of(state, MODEL_B)) == 1
    assert keeper.step(state, AFTER_GRACE + 2) == []


def test_only_ever_creates_instances():
    """A step adds one instance; it never deletes or changes one that exists."""
    keeper = Keeper(started_at=START)
    state, _ = _deploy(_cluster(3), MODEL_A, MODEL_B)
    state, _ = _place_by_hand(state, _model("test/someone-elses"))

    events = keeper.step(state, AFTER_GRACE)

    assert {type(e) for e in events} == {InstanceCreated, DeploymentPlaced}
    after = _apply(state, events)
    for instance_id, instance in state.instances.items():
        assert after.instances[instance_id] == instance


# Waiting for a placement to show up


def test_does_not_place_again_while_the_last_placement_is_on_its_way():
    keeper = Keeper(started_at=START)
    state, _ = _deploy(_cluster(2), MODEL_A)

    first = _placed(keeper.step(state, AFTER_GRACE))
    # The events haven't reached the state yet
    for now in (AFTER_GRACE + 1, AFTER_GRACE + PLACEMENT_TIMEOUT - 0.001):
        assert keeper.step(state, now) == []

    second = _placed(keeper.step(state, AFTER_GRACE + PLACEMENT_TIMEOUT))
    assert second != first


def test_a_placement_that_never_arrives_is_not_counted_as_a_failure():
    keeper = Keeper(started_at=START)
    state, _ = _deploy(_cluster(2), MODEL_A)

    keeper.step(state, AFTER_GRACE)
    now = AFTER_GRACE + PLACEMENT_TIMEOUT
    events = keeper.step(state, now)
    state = _apply(state, events)

    assert has_instance(next(iter(state.deployments.values())), state)


# Losing an instance


def test_places_again_at_once_when_a_ready_instance_is_lost():
    keeper = Keeper(started_at=START)
    state, _ = _deploy(_cluster(2), MODEL_A)
    now = AFTER_GRACE
    for _ in range(5):
        instance_id = _placed(events := keeper.step(state, now))
        state = _make_ready(_apply(state, events), instance_id)
        now += 1
        assert keeper.step(state, now) == []
        state = _lose(state, instance_id)
        now += 1


def test_places_again_when_someone_deletes_the_only_instance():
    keeper = Keeper(started_at=START)
    state, _ = _deploy(_cluster(2), MODEL_A)
    state, by_hand = _place_by_hand(state, MODEL_A)
    assert keeper.step(state, AFTER_GRACE) == []

    state = _lose(state, by_hand)

    assert _placed(keeper.step(state, AFTER_GRACE + 1)) != by_hand


def test_places_again_on_the_nodes_left_when_a_node_dies():
    keeper = Keeper(started_at=START)
    state, _ = _deploy(_cluster(2), MODEL_A)
    instance_id = _placed(events := keeper.step(state, AFTER_GRACE))
    state = _make_ready(_apply(state, events), instance_id)
    assert keeper.step(state, AFTER_GRACE + 1) == []
    (dead,) = state.instances[instance_id].shard_assignments.node_to_runner

    state = _apply(
        state, [NodeTimedOut(node_id=dead), InstanceDeleted(instance_id=instance_id)]
    )
    events = keeper.step(state, AFTER_GRACE + 2)
    state = _apply(state, events)

    (replacement,) = _instances_of(state, MODEL_A)
    assert dead not in state.instances[replacement].shard_assignments.node_to_runner


def test_an_instance_lost_between_two_steps_is_placed_again_after_one_backoff():
    """The keeper never sees this instance: it arrives and is gone before the next step. It still
    counts as placed and lost, so the keeper doesn't wait out PLACEMENT_TIMEOUT for it."""
    keeper = Keeper(started_at=START)
    state, _ = _deploy(_cluster(2), MODEL_A)
    instance_id = _placed(events := keeper.step(state, AFTER_GRACE))
    state = _lose(_apply(state, events), instance_id)

    assert keeper.step(state, AFTER_GRACE + 1) == []
    assert FIRST_BACKOFF < PLACEMENT_TIMEOUT
    assert keeper.step(state, AFTER_GRACE + 1 + FIRST_BACKOFF) != []


def test_backs_off_when_instances_are_lost_before_they_are_ready():
    keeper = Keeper(started_at=START)
    state, _ = _deploy(_cluster(2), MODEL_A)
    now = AFTER_GRACE
    # 10 s, 20 s, 40 s, then the 60 s cap
    waits = [min(FIRST_BACKOFF * 2.0**i, MAX_BACKOFF) for i in range(6)]
    assert waits[-3:] == [MAX_BACKOFF] * 3

    for wait in waits:
        instance_id = _placed(events := keeper.step(state, now))
        state = _apply(state, events)
        assert keeper.step(state, now + 1) == []  # still loading
        state = _lose(state, instance_id)
        lost_at = now + 2
        assert keeper.step(state, lost_at) == []
        assert keeper.step(state, lost_at + wait - 0.001) == []
        now = lost_at + wait


def test_a_ready_instance_clears_the_backoff():
    keeper = Keeper(started_at=START)
    state, _ = _deploy(_cluster(2), MODEL_A)

    def place_then_lose(now: float, ready: bool) -> None:
        """Place an instance at `now`, see it a second later, then lose it."""
        nonlocal state
        instance_id = _placed(events := keeper.step(state, now))
        state = _apply(state, events)
        if ready:
            state = _make_ready(state, instance_id)
        assert keeper.step(state, now + 1) == []
        state = _lose(state, instance_id)

    first = AFTER_GRACE
    place_then_lose(first, ready=False)
    assert keeper.step(state, first + 2) == []  # waits FIRST_BACKOFF
    second = first + 2 + FIRST_BACKOFF
    place_then_lose(second, ready=False)
    assert keeper.step(state, second + 2) == []  # waits twice that
    third = second + 2 + 2 * FIRST_BACKOFF
    place_then_lose(third, ready=True)

    # Lost after it was ready: placed again at once, and a failure after that waits only
    # FIRST_BACKOFF again
    place_then_lose(third + 2, ready=False)
    lost_at = third + 4
    assert keeper.step(state, lost_at) == []
    assert keeper.step(state, lost_at + FIRST_BACKOFF - 0.001) == []
    assert keeper.step(state, lost_at + FIRST_BACKOFF) != []


def test_a_runner_that_becomes_ready_late_still_counts():
    keeper = Keeper(started_at=START)
    state, _ = _deploy(_cluster(1), MODEL_A)
    instance_id = _placed(events := keeper.step(state, AFTER_GRACE))
    state = _apply(state, events)
    (runner_id,) = state.instances[
        instance_id
    ].shard_assignments.node_to_runner.values()

    state = event_apply(
        RunnerStatusUpdated(runner_id=runner_id, runner_status=RunnerLoading()), state
    )
    keeper.step(state, AFTER_GRACE + 1)
    state = event_apply(
        RunnerStatusUpdated(runner_id=runner_id, runner_status=RunnerRunning()), state
    )
    keeper.step(state, AFTER_GRACE + 60)
    state = _lose(state, instance_id)

    assert keeper.step(state, AFTER_GRACE + 61) != []


# A deployment no placement fits


def test_reports_once_why_a_deployment_cant_be_placed_and_keeps_trying():
    keeper = Keeper(started_at=START)
    too_big = _model("test/too-big", size_bytes=NODE_MEMORY * 10)
    state, (deployment,) = _deploy(_cluster(2), too_big)

    events = keeper.step(state, AFTER_GRACE)
    (failed,) = events
    assert isinstance(failed, DeploymentPlacementFailed)
    state = _apply(state, events)
    assert state.deployments[deployment.deployment_id].placement_error == failed.error

    assert keeper.step(state, AFTER_GRACE + 1) == []
    # Tried again, failed the same way: nothing new to say
    assert keeper.step(state, AFTER_GRACE + UNPLACEABLE_RETRY) == []
    assert keeper.step(state, AFTER_GRACE + 10 * UNPLACEABLE_RETRY) == []


def test_reports_when_the_reason_it_cant_be_placed_changes():
    keeper = Keeper(started_at=START)
    too_big = _model("test/too-big", size_bytes=NODE_MEMORY * 10)
    state, _ = _deploy(_cluster(2), too_big)
    state = _apply(state, keeper.step(state, AFTER_GRACE))

    roomy = _cluster(2, memory=NODE_MEMORY * 100)
    state = state.model_copy(
        update={
            "topology": roomy.topology,
            "node_memory": roomy.node_memory,
            "node_network": roomy.node_network,
            "node_backends": {},
        }
    )
    (failed,) = keeper.step(state, AFTER_GRACE + UNPLACEABLE_RETRY)

    assert isinstance(failed, DeploymentPlacementFailed)
    assert "backend" in failed.error


def test_places_once_the_cluster_can_hold_the_model():
    keeper = Keeper(started_at=START)
    big = _model("test/big", size_bytes=int(NODE_MEMORY * 2.5))
    state, (deployment,) = _deploy(_cluster(2), big)
    state = _apply(state, keeper.step(state, AFTER_GRACE))
    assert (
        deployment_status(state.deployments[deployment.deployment_id], state)
        == "cant_place"
    )

    bigger = _cluster(4)
    state = state.model_copy(
        update={
            "topology": bigger.topology,
            "node_memory": bigger.node_memory,
            "node_network": bigger.node_network,
            "node_backends": bigger.node_backends,
        }
    )
    assert keeper.step(state, AFTER_GRACE + UNPLACEABLE_RETRY - 0.001) == []
    state = _apply(state, keeper.step(state, AFTER_GRACE + UNPLACEABLE_RETRY))

    assert len(_instances_of(state, big)) == 1
    assert state.deployments[deployment.deployment_id].placement_error is None


def test_an_unplaceable_deployment_does_not_hold_up_the_others():
    keeper = Keeper(started_at=START)
    too_big = _model("test/too-big", size_bytes=NODE_MEMORY * 10)
    state, _ = _deploy(_cluster(2), too_big, MODEL_A)

    now = AFTER_GRACE
    for _ in range(3):
        state = _apply(state, keeper.step(state, now))
        now += 1

    assert len(_instances_of(state, MODEL_A)) == 1


# Deployments coming and going


def test_forgets_a_deployment_that_leaves_the_state_without_it():
    """As when another master deleted it, or the state was replaced by a newer snapshot."""
    keeper = Keeper(started_at=START)
    state, (deployment,) = _deploy(_cluster(2), MODEL_A)
    instance_id = _placed(events := keeper.step(state, AFTER_GRACE))
    state = _apply(state, events)
    keeper.step(state, AFTER_GRACE + 1)
    state = _lose(state, instance_id)
    assert keeper.step(state, AFTER_GRACE + 2) == []  # backing off

    state = event_apply(
        DeploymentDeleted(deployment_id=deployment.deployment_id), state
    )
    assert keeper.step(state, AFTER_GRACE + 3) == []
    assert keeper._attempts == {}  # pyright: ignore[reportPrivateUsage]
    state, _ = _deploy(state, MODEL_A)

    assert keeper.step(state, AFTER_GRACE + 4) != []  # a new deployment starts afresh


def test_deployments_created_and_deleted_leave_nothing_behind():
    keeper = Keeper(started_at=START)
    state = _cluster(2)
    now = AFTER_GRACE
    for _ in range(100):
        state, (deployment,) = _deploy(state, MODEL_A)
        state = _apply(state, keeper.step(state, now))
        events, _ = keeper.delete(
            DeleteDeployment(deployment_id=deployment.deployment_id), state
        )
        state = _apply(state, events)
        now += 1
    assert keeper.step(state, now) == []

    assert state.instances == {}
    assert state.deployments == {}
    assert keeper._attempts == {}  # pyright: ignore[reportPrivateUsage]
    assert keeper._deleted == set()  # pyright: ignore[reportPrivateUsage]


def test_delete_deployment_deletes_the_instance_the_keeper_placed():
    keeper = Keeper(started_at=START)
    state, (deployment, _) = _deploy(_cluster(2), MODEL_A, MODEL_B)
    state = _apply(state, keeper.step(state, AFTER_GRACE))
    state = _apply(state, keeper.step(state, AFTER_GRACE + 1))
    (kept,) = _instances_of(state, deployment.model_card)
    (other,) = _instances_of(state, MODEL_B)

    events, downloads = keeper.delete(
        DeleteDeployment(deployment_id=deployment.deployment_id), state
    )
    state = _apply(state, events)

    assert deployment.deployment_id not in state.deployments
    assert kept not in state.instances
    assert other in state.instances
    assert downloads == []
    assert keeper.step(state, AFTER_GRACE + 2) == []


def test_delete_deployment_cancels_downloads_for_the_deleted_instance():
    keeper = Keeper(started_at=START)
    state, (deployment,) = _deploy(_cluster(1), MODEL_A)
    instance_id = _placed(events := keeper.step(state, AFTER_GRACE))
    state = _apply(state, events)
    instance = state.instances[instance_id]
    ((node_id, runner_id),) = instance.shard_assignments.node_to_runner.items()
    downloading = DownloadOngoing(
        node_id=node_id,
        shard_metadata=instance.shard_assignments.runner_to_shard[runner_id],
        download_progress=DownloadProgressData(
            total=Memory.from_bytes(1000),
            downloaded=Memory.from_bytes(300),
            downloaded_this_session=Memory.from_bytes(300),
            completed_files=0,
            total_files=1,
            speed=0.0,
            eta_ms=0,
            files={},
        ),
    )
    state = state.model_copy(update={"downloads": {node_id: [downloading]}})

    _, downloads = keeper.delete(
        DeleteDeployment(deployment_id=deployment.deployment_id), state
    )

    assert downloads == [
        CancelDownload(
            target_node_id=node_id,
            model_id=MODEL_A.model_id,
            command_id=downloads[0].command_id,
        )
    ]


def test_delete_deployment_leaves_an_instance_someone_else_placed():
    keeper = Keeper(started_at=START)
    state, (deployment,) = _deploy(_cluster(2), MODEL_A)
    state, by_hand = _place_by_hand(state, MODEL_A)
    assert keeper.step(state, AFTER_GRACE) == []

    events, downloads = keeper.delete(
        DeleteDeployment(deployment_id=deployment.deployment_id), state
    )

    assert _without_ids(events) == _without_ids(
        [DeploymentDeleted(deployment_id=deployment.deployment_id)]
    )
    assert downloads == []
    assert by_hand in _apply(state, events).instances


def test_delete_deployment_after_its_instance_is_gone():
    keeper = Keeper(started_at=START)
    state, (deployment,) = _deploy(_cluster(2), MODEL_A)
    instance_id = _placed(events := keeper.step(state, AFTER_GRACE))
    state = _lose(_apply(state, events), instance_id)

    events, downloads = keeper.delete(
        DeleteDeployment(deployment_id=deployment.deployment_id), state
    )

    assert _without_ids(events) == _without_ids(
        [DeploymentDeleted(deployment_id=deployment.deployment_id)]
    )
    assert downloads == []


def test_delete_deployment_deletes_an_instance_still_on_its_way():
    keeper = Keeper(started_at=START)
    state, (deployment,) = _deploy(_cluster(2), MODEL_A)
    placing = keeper.step(state, AFTER_GRACE)  # sent, but not in the state yet

    deleting, _ = keeper.delete(
        DeleteDeployment(deployment_id=deployment.deployment_id), state
    )
    # The cluster applies the master's events in the order it sent them
    state = _apply(state, [*placing, *deleting])

    assert state.instances == {}
    assert state.deployments == {}


def test_places_nothing_for_a_deployment_being_deleted():
    keeper = Keeper(started_at=START)
    state, (deployment,) = _deploy(_cluster(2), MODEL_A)

    deleting, _ = keeper.delete(
        DeleteDeployment(deployment_id=deployment.deployment_id), state
    )
    # Until the deletion reaches the state, the state still shows the deployment
    for now in (AFTER_GRACE, AFTER_GRACE + PLACEMENT_TIMEOUT, AFTER_GRACE + 3600):
        assert keeper.step(state, now) == []
    state = _apply(state, deleting)

    assert keeper.step(state, AFTER_GRACE + 3601) == []
    assert keeper._deleted == set()  # pyright: ignore[reportPrivateUsage]


def test_deleting_a_deployment_twice_is_harmless():
    keeper = Keeper(started_at=START)
    state, (deployment,) = _deploy(_cluster(2), MODEL_A)
    placing = keeper.step(state, AFTER_GRACE)
    command = DeleteDeployment(deployment_id=deployment.deployment_id)

    first, _ = keeper.delete(command, state)
    second, _ = keeper.delete(command, state)
    state = _apply(state, [*placing, *first, *second])

    assert state.instances == {}
    assert state.deployments == {}


def test_delete_unknown_deployment_does_nothing():
    keeper = Keeper(started_at=START)
    state, _ = _deploy(_cluster(1), MODEL_A)

    assert keeper.delete(DeleteDeployment(deployment_id=DeploymentId()), state) == (
        [],
        [],
    )


# Status


def test_deployment_status_follows_its_instance():
    keeper = Keeper(started_at=START)
    state, (deployment,) = _deploy(_cluster(2), MODEL_A)

    def status() -> str:
        return deployment_status(state.deployments[deployment.deployment_id], state)

    assert status() == "placing"
    instance_id = _placed(events := keeper.step(state, AFTER_GRACE))
    state = _apply(state, events)
    assert status() == "starting"
    state = _make_ready(state, instance_id)
    assert status() == "serving"
    (runner_id,) = state.instances[
        instance_id
    ].shard_assignments.node_to_runner.values()
    state = event_apply(
        RunnerStatusUpdated(runner_id=runner_id, runner_status=RunnerRunning()), state
    )
    assert status() == "serving"
    state = event_apply(
        RunnerStatusUpdated(
            runner_id=runner_id,
            runner_status=RunnerFailed(error_message="boom", diagnostics=[]),
        ),
        state,
    )
    assert status() == "starting"
    state = _lose(state, instance_id)
    assert status() == "placing"


# Under random faults


def test_keeps_every_model_running_through_random_faults():
    """Lose instances and nodes at random for simulated hours. Throughout, the keeper only adds
    instances, never two of a model at once; once the faults stop, every model is running again
    within one backoff."""
    rng = random.Random(1234)
    keeper = Keeper(started_at=START)
    models = [_model(f"test/model-{i}") for i in range(4)]
    full = _cluster(5)
    state, _ = _deploy(full, *models)
    now = AFTER_GRACE

    def step() -> None:
        nonlocal state
        events = keeper.step(state, now)
        assert {type(e) for e in events} <= {
            InstanceCreated,
            DeploymentPlaced,
            DeploymentPlacementFailed,
        }
        assert len([e for e in events if isinstance(e, InstanceCreated)]) <= 1
        state = _apply(state, events)
        for model in models:
            assert len(_instances_of(state, model)) <= 1

    for _ in range(4 * 3600):
        step()
        fault = rng.random()
        if fault < 0.01 and state.instances:
            state = _lose(state, rng.choice(list(state.instances)))
        elif fault < 0.02 and state.instances:
            state = _make_ready(state, rng.choice(list(state.instances)))
        elif fault < 0.022 and len(list(state.topology.list_nodes())) > 1:
            dead = rng.choice(list(state.topology.list_nodes()))
            on_dead = [
                instance_id
                for instance_id, instance in state.instances.items()
                if dead in instance.shard_assignments.node_to_runner
            ]
            state = _apply(
                state,
                [
                    NodeTimedOut(node_id=dead),
                    *(InstanceDeleted(instance_id=i) for i in on_dead),
                ],
            )
        elif fault < 0.025:
            # A node comes back: the full cluster again
            state = state.model_copy(
                update={
                    "topology": full.topology,
                    "node_memory": full.node_memory,
                    "node_network": full.node_network,
                    "node_backends": full.node_backends,
                }
            )
        now += 1

    for _ in range(int(MAX_BACKOFF + PLACEMENT_TIMEOUT) + len(models)):
        step()
        now += 1
    for model in models:
        assert len(_instances_of(state, model)) == 1


@pytest.mark.parametrize("seed", range(20))
def test_deployments_come_and_go_while_the_state_trails_the_events(seed: int):
    """As in the master, the keeper's events, deletions, creations and losses reach the state
    only after a delay (here up to 10 s). Deployments are created and deleted, instances lost or
    made ready, at random. At no point does a model have two instances, and once things settle,
    exactly the models with a deployment have an instance."""
    rng = random.Random(seed)
    keeper = Keeper(started_at=START)
    models = [_model(f"test/model-{i}") for i in range(3)]
    state = _cluster(3)
    sent: list[tuple[float, Event]] = []  # sent by the master, not yet in the state
    now = START

    def send(events: Iterable[Event]) -> None:
        sent.extend((now, event) for event in events)

    def deliver(everything: bool) -> None:
        nonlocal state
        count = len(sent) if everything else rng.randint(0, 3)
        # Nothing waits longer than 10 s
        while sent and sent[0][0] < now - 10:
            state = event_apply(sent.pop(0)[1], state)
        for _ in range(min(count, len(sent))):
            state = event_apply(sent.pop(0)[1], state)
        for model in models:
            assert len(_instances_of(state, model)) <= 1

    for _ in range(2 * 3600):
        action = rng.random()
        if action < 0.02:
            missing = [
                model
                for model in models
                if not any(
                    d.model_card.model_id == model.model_id
                    for d in state.deployments.values()
                )
            ]
            if missing:
                send([DeploymentCreated(deployment=_deployment(rng.choice(missing)))])
        elif action < 0.03 and state.deployments:
            deployment_id = rng.choice(list(state.deployments))
            events, _ = keeper.delete(
                DeleteDeployment(deployment_id=deployment_id), state
            )
            send(events)
        elif action < 0.04 and state.instances:
            send([InstanceDeleted(instance_id=rng.choice(list(state.instances)))])
        elif action < 0.06 and state.instances:
            instance_id = rng.choice(list(state.instances))
            runner_ids = state.instances[
                instance_id
            ].shard_assignments.node_to_runner.values()
            send(
                RunnerStatusUpdated(runner_id=runner_id, runner_status=RunnerReady())
                for runner_id in runner_ids
            )
        send(keeper.step(state, now))
        deliver(everything=False)
        now += 1

    for _ in range(int(MAX_BACKOFF) + 10):
        send(keeper.step(state, now))
        deliver(everything=True)
        now += 1
    deployed = {d.model_card.model_id for d in state.deployments.values()}
    for model in models:
        expected = 1 if model.model_id in deployed else 0
        assert len(_instances_of(state, model)) == expected, model.model_id
