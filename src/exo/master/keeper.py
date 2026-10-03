"""The keeper: keeps every deployment's model running.

Once a second the master asks the keeper what to do. For a deployment whose model has no instance
in the cluster, the keeper places one, with the same placement a `/place_instance` request uses.

The keeper never judges or changes an existing instance. Deciding that an instance is broken stays
with the paths that delete instances today: a node going silent, runners that keep failing to start,
a user deleting it. The keeper only notices that its model has no instance any more and places a
new one. When a deployment is deleted, the keeper deletes the instance it placed for it, as
`DELETE /instance` would.

The master's state trails the events it sends: they reach the state only after a round trip
through the event router. So the keeper remembers the instance it placed until the state shows
it, and the master emits the keeper's events and a deployment's deletion under one lock, so that
a deletion can't miss an instance that is still on its way.
"""

import math
from dataclasses import dataclass, field

from exo.master.placement import (
    cancel_unnecessary_downloads,
    delete_instance,
    get_transition_events,
    place_instance,
)
from exo.shared.types.commands import (
    DeleteDeployment,
    DeleteInstance,
    DownloadCommand,
    PlaceInstance,
)
from exo.shared.types.common import NodeId
from exo.shared.types.deployments import Deployment, DeploymentId, DeploymentStatus
from exo.shared.types.events import (
    DeploymentDeleted,
    DeploymentPlaced,
    DeploymentPlacementFailed,
    Event,
    InstanceDeleted,
)
from exo.shared.types.state import State
from exo.shared.types.topology import RDMAConnection
from exo.shared.types.worker.instances import Instance, InstanceId
from exo.shared.types.worker.runners import RunnerReady, RunnerRunning

# A new master waits this long before placing anything, so nodes can reconnect and report the
# instances they already run
STARTUP_GRACE = 15.0
# How long a placement may take to appear in the state before the keeper places again
PLACEMENT_TIMEOUT = 30.0
# How often a deployment that no placement fits is tried again; it is also tried as soon as the
# nodes placement can use, or the links between them, change
UNPLACEABLE_RETRY = 30.0
# An instance lost before all its runners were ready counts as a failed placement: the next one
# waits FIRST_BACKOFF, doubling with each failure in a row up to MAX_BACKOFF. An instance lost
# after it was ready (its node died, someone deleted it) is placed again at once.
FIRST_BACKOFF = 10.0
MAX_BACKOFF = 60.0


# The nodes placement can use, and the links between them (source, sink, is RDMA)
_Shape = tuple[frozenset[NodeId], frozenset[tuple[NodeId, NodeId, bool]]]


@dataclass
class _Attempt:
    """What the keeper remembers about one deployment. It is held in memory only: a new master
    starts afresh, and the state tells it whether the model has an instance."""

    instance_id: InstanceId | None = None
    placed_at: float = -math.inf
    ready: bool = False
    failures: int = 0
    retry_at: float = -math.inf
    # The cluster's shape when placement last found nothing that fits
    unplaceable_on: _Shape | None = None


@dataclass
class Keeper:
    started_at: float
    _attempts: dict[DeploymentId, _Attempt] = field(default_factory=dict)
    # Deployments being deleted, until the state no longer shows them
    _deleted: set[DeploymentId] = field(default_factory=set)

    def step(self, state: State, now: float) -> list[Event]:
        """The events that place an instance for at most one deployment that needs one.

        One placement per step, so that each placement is made against a state that already
        holds the one before it.
        """
        for deployment_id in list(self._attempts):
            if deployment_id not in state.deployments:
                del self._attempts[deployment_id]
        self._deleted.intersection_update(state.deployments)
        if now - self.started_at < STARTUP_GRACE:
            return []
        for deployment in sorted(
            state.deployments.values(), key=lambda d: d.deployment_id
        ):
            if deployment.deployment_id in self._deleted:
                continue
            attempt = self._attempts.setdefault(deployment.deployment_id, _Attempt())
            _observe(deployment, attempt, state, now)
            if events := _keep(deployment, attempt, state, now):
                return events
        return []

    def delete(
        self, command: DeleteDeployment, state: State
    ) -> tuple[list[Event], list[DownloadCommand]]:
        """Stop keeping a model running, and delete the instance the keeper placed for it, exactly
        as `DELETE /instance` does, even if that instance hasn't reached the state yet. An instance
        someone else placed for the model is left alone."""
        deployment = state.deployments.get(command.deployment_id)
        if deployment is None:
            return [], []
        self._deleted.add(deployment.deployment_id)
        events: list[Event] = [
            DeploymentDeleted(deployment_id=deployment.deployment_id)
        ]
        attempt = self._attempts.pop(deployment.deployment_id, None)
        if (
            attempt is not None
            and attempt.instance_id is not None
            and not _arrived(deployment, attempt)
        ):
            # Its InstanceCreated was sent before this, so the deletion is applied after it
            events.append(InstanceDeleted(instance_id=attempt.instance_id))
        instance_id = deployment.instance_id
        if instance_id is None or instance_id not in state.instances:
            return events, []
        placement = delete_instance(
            DeleteInstance(instance_id=instance_id), state.instances
        )
        events.extend(get_transition_events(state.instances, placement, state.tasks))
        return events, list(cancel_unnecessary_downloads(placement, state.downloads))


def _observe(
    deployment: Deployment, attempt: _Attempt, state: State, now: float
) -> None:
    """Follow the instance the keeper placed last: on its way, there, ready, or lost."""
    if attempt.instance_id is None or not _arrived(deployment, attempt):
        return
    instance = state.instances.get(attempt.instance_id)
    if instance is not None:
        if is_ready(instance, state):
            attempt.ready = True
            attempt.failures = 0
        return
    if not attempt.ready:
        attempt.failures += 1
        attempt.retry_at = now + min(
            FIRST_BACKOFF * 2.0 ** (attempt.failures - 1), MAX_BACKOFF
        )
    attempt.instance_id = None
    attempt.ready = False


def _arrived(deployment: Deployment, attempt: _Attempt) -> bool:
    """Whether the keeper's last placement has reached the state. DeploymentPlaced follows the
    instance's InstanceCreated, so once the deployment names the instance, the instance has been
    created, even if it is already gone again."""
    return deployment.instance_id == attempt.instance_id


def _keep(
    deployment: Deployment, attempt: _Attempt, state: State, now: float
) -> list[Event]:
    if has_instance(deployment, state):
        return []
    in_flight = (
        attempt.instance_id is not None
        and not _arrived(deployment, attempt)
        and now - attempt.placed_at < PLACEMENT_TIMEOUT
    )
    shape = _shape(state)
    cluster_changed = attempt.unplaceable_on not in (None, shape)
    if in_flight or (now < attempt.retry_at and not cluster_changed):
        return []
    command = PlaceInstance(
        model_card=deployment.model_card,
        sharding=deployment.sharding,
        instance_meta=deployment.instance_meta,
        min_nodes=deployment.min_nodes,
    )
    try:
        placement = place_instance(
            command,
            state.topology,
            state.instances,
            state.node_memory,
            state.node_network,
            state.node_backends,
            download_status=state.downloads,
            node_rdma_ctl=state.node_rdma_ctl,
        )
    except ValueError as error:
        attempt.retry_at = now + UNPLACEABLE_RETRY
        attempt.unplaceable_on = shape
        if deployment.placement_error == str(error):
            return []
        return [
            DeploymentPlacementFailed(
                deployment_id=deployment.deployment_id, error=str(error)
            )
        ]
    (instance_id,) = [i for i in placement if i not in state.instances]
    attempt.instance_id, attempt.placed_at, attempt.ready = instance_id, now, False
    attempt.unplaceable_on = None
    return [
        *get_transition_events(state.instances, placement, state.tasks),
        DeploymentPlaced(
            deployment_id=deployment.deployment_id, instance_id=instance_id
        ),
    ]


def _shape(state: State) -> _Shape:
    """What placement works from, but for memory figures, which change all the time. A joining
    node arrives in pieces: it shows up, reports its memory and backends, and its links appear
    seconds later, so the keeper tries again as each piece arrives."""
    nodes = frozenset(
        node_id
        for node_id in state.topology.list_nodes()
        if node_id in state.node_memory and node_id in state.node_backends
    )
    links = frozenset(
        (link.source, link.sink, isinstance(link.edge, RDMAConnection))
        for link in state.topology.list_connections()
        if link.source in nodes and link.sink in nodes
    )
    return nodes, links


def has_instance(deployment: Deployment, state: State) -> bool:
    """Whether the cluster has an instance of the deployment's model, however it was created."""
    return any(
        instance.shard_assignments.model_id == deployment.model_card.model_id
        for instance in state.instances.values()
    )


def is_ready(instance: Instance, state: State) -> bool:
    """Whether every runner of the instance has loaded its shard and can serve."""
    runner_ids = list(instance.shard_assignments.node_to_runner.values())
    return bool(runner_ids) and all(
        isinstance(state.runners.get(runner_id), (RunnerReady, RunnerRunning))
        for runner_id in runner_ids
    )


def deployment_status(deployment: Deployment, state: State) -> DeploymentStatus:
    """What a deployment is doing, for the API: serving when an instance of its model is ready,
    starting while one is on its way up, otherwise placing or unable to place."""
    instances = [
        instance
        for instance in state.instances.values()
        if instance.shard_assignments.model_id == deployment.model_card.model_id
    ]
    if any(is_ready(instance, state) for instance in instances):
        return "serving"
    if instances:
        return "starting"
    return "cant_place" if deployment.placement_error is not None else "placing"
