"""The reconciler: the master keeps every deployment placed.

A deployment without an instance (its instance lost a node, or was removed to be started again)
is placed again here, and only here:
- with the user's constraints (model, sharding, kind, fewest nodes, pinned nodes), not with
  what placement chose last time, which is only a preference;
- only on nodes that already have the model's files, so it never starts a download by itself;
- not while the master's node is Resuming after it was stopped (its peers are still
  reconnecting), and not before its `not_before` (a launch the master learned of from a node
  waits for the nodes it last ran on, a model whose node quit waits for it) unless those nodes
  are back;
- an RDMA instance comes back as a TCP ring when the RDMA-linked nodes aren't all there.
A deployment that keeps losing its devices is paused instead (CRASH_LOOP_LIMIT).

Everything here is pure: the master calls `reconcile` from its plan loop and sends the events.
"""

import contextlib
from collections.abc import Hashable, Mapping, Sequence
from collections.abc import Set as AbstractSet
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

from exo.master.placement import place_instance
from exo.master.placement_utils import Cycle
from exo.shared.models.model_cards import ModelCard, ModelId
from exo.shared.types.commands import AnnounceDeployments
from exo.shared.types.common import DeploymentId, NodeId
from exo.shared.types.deployments import (
    CRASH_LOOP_LIMIT,
    CRASH_LOOP_WINDOW,
    WAIT_LIMIT,
    Deployment,
    DeploymentSpec,
    Hold,
    Waiting,
)
from exo.shared.types.events import (
    DeploymentCreated,
    DeploymentHeld,
    DeploymentRemoved,
    DeploymentResumed,
    DeploymentStatus,
    Event,
    InstanceCreated,
)
from exo.shared.types.placement import PlacementRequest
from exo.shared.types.state import State
from exo.shared.types.worker.downloads import (
    DownloadCompleted,
    DownloadFailed,
    DownloadOngoing,
    DownloadPending,
    DownloadProgress,
)
from exo.shared.types.worker.instances import (
    Instance,
    InstanceId,
    InstanceMeta,
    MlxJacclInstance,
    TensorFoldInstance,
)
from exo.shared.types.worker.shards import Sharding, TensorShardMetadata

RDMA_FALLBACK_NOTE = (
    "Running over TCP: the devices linked for RDMA aren't all available"
)

# What a deployment's status is: why it waits, and its note
Status = tuple[Waiting | None, str | None]


def spec_of(
    instance: Instance, min_nodes: int = 1, pinned_nodes: Sequence[NodeId] = ()
) -> DeploymentSpec:
    """The spec of a deployment launched as this instance, with the user's constraints."""
    shard = next(iter(instance.shard_assignments.runner_to_shard.values()))
    return DeploymentSpec(
        model_id=instance.shard_assignments.model_id,
        sharding=Sharding.Tensor
        if isinstance(shard, TensorShardMetadata)
        else Sharding.Pipeline,
        instance_meta=InstanceMeta.TensorFold
        if isinstance(instance, TensorFoldInstance)
        else InstanceMeta.MlxJaccl
        if isinstance(instance, MlxJacclInstance)
        else InstanceMeta.MlxRing,
        min_nodes=min_nodes,
        pinned_nodes=list(pinned_nodes),
        tensorfold_context_length=instance.context_length
        if isinstance(instance, TensorFoldInstance)
        else 8192,
        tensorfold_parallel=instance.parallel
        if isinstance(instance, TensorFoldInstance)
        else None,
    )


def _has_files(progress: DownloadProgress) -> bool:
    """Whether a node has the model's files, or is downloading them: a re-placement there
    carries on a download, never starts one. A failed or not yet started download doesn't
    count, even with bytes on disk."""
    match progress:
        case DownloadCompleted():
            return True
        case DownloadOngoing():
            return progress.download_progress.downloaded.in_bytes > 0
        case DownloadFailed() | DownloadPending():
            return False


def nodes_with_files(
    model_id: ModelId, downloads: Mapping[NodeId, Sequence[DownloadProgress]]
) -> set[NodeId]:
    return {
        node_id
        for node_id, progresses in downloads.items()
        if any(
            progress.shard_metadata.model_card.model_id == model_id
            and _has_files(progress)
            for progress in progresses
        )
    }


@dataclass(frozen=True)
class Placed:
    instance: Instance
    note: str | None = None


def place_deployment(
    deployment_id: DeploymentId,
    spec: DeploymentSpec,
    card: ModelCard,
    state: State,
    instances: Mapping[InstanceId, Instance],
    *,
    allowed_nodes: AbstractSet[NodeId],
    cycles: Sequence[Cycle] | None = None,
) -> Placed:
    """An instance of the deployment among `instances` (the state's and any being started), on
    a TCP ring when an RDMA one doesn't fit.

    Raises ValueError, saying why, when it doesn't fit."""

    def place(instance_meta: InstanceMeta) -> Instance:
        placement = place_instance(
            PlacementRequest(
                model_card=card,
                sharding=spec.sharding,
                instance_meta=instance_meta,
                min_nodes=spec.min_nodes,
                tensorfold_context_length=spec.tensorfold_context_length,
                tensorfold_parallel=spec.tensorfold_parallel,
            ),
            state.topology,
            instances,
            state.node_memory,
            state.node_network,
            state.node_backends,
            required_nodes=set(spec.pinned_nodes) or None,
            download_status=state.downloads,
            node_rdma_ctl=state.node_rdma_ctl,
            runners=state.runners,
            node_identities=state.node_identities,
            allowed_nodes=allowed_nodes,
            cycles=cycles,
            node_tensorfold=state.node_tensorfold,
        )
        (new,) = (
            instance
            for instance_id, instance in placement.items()
            if instance_id not in instances
        )
        return new.model_copy(update={"deployment_id": deployment_id})

    try:
        return Placed(place(spec.instance_meta))
    except ValueError as rdma_error:
        if spec.instance_meta != InstanceMeta.MlxJaccl:
            raise
        try:
            return Placed(place(InstanceMeta.MlxRing), note=RDMA_FALLBACK_NOTE)
        except ValueError:
            # The RDMA error says more about what the user asked for
            raise rdma_error from None


RESTORED_MESSAGE = "exo restarted on every device: resume it to load it again"


def announced(
    state: State, announcement: AnnounceDeployments, now: datetime, wait: timedelta
) -> list[Event]:
    """What a node's announcement adds to the master's state: the deployments and stops the
    state lacks, and the user's later changes (by revision) to those it has.

    A deployment the state lacks waits up to `wait` for the nodes it last ran on, so a launch
    made on the other side of a heal comes back on its own node. One the node only has from its
    disk comes back held: the whole cluster restarted, and models aren't loaded again by
    themselves at start-up. A node that has run it since it started shows this was no full
    restart, and lifts that hold.

    Holds and waits are this master's own decisions, from what its side saw: none comes from
    a node, and one this master has is lifted for a deployment the node's previous master was
    running."""
    events: list[Event] = []
    for deployment_id, at in announcement.removed.items():
        if (
            deployment_id in state.deployments
            or deployment_id not in state.removed_deployments
        ):
            events.append(DeploymentRemoved(deployment_id=deployment_id, at=at))
    restored = set(announcement.restored)
    running = set(announcement.running)
    placed = {instance.deployment_id for instance in state.instances.values()}
    for record in announcement.deployments:
        deployment_id = record.deployment_id
        if (
            deployment_id in announcement.removed
            or deployment_id in state.removed_deployments
        ):
            continue
        current = state.deployments.get(deployment_id)
        if current is None:
            hold = (
                Hold(kind="Restored", message=RESTORED_MESSAGE)
                if deployment_id in restored
                else None
            )
            events.append(
                DeploymentCreated(
                    deployment=Deployment(
                        deployment_id=deployment_id,
                        spec=record.spec,
                        model_card=record.model_card,
                        revision=record.revision,
                        hold=hold,
                        last_nodes=record.last_nodes,
                        not_before=(now + wait).isoformat()
                        if record.last_nodes and hold is None
                        else None,
                        # It has no instance here: how long it waits counts from now
                        lost_at=now.isoformat(),
                    )
                )
            )
        elif record.revision > current.revision:
            # The user changed it on the other side (resumed it): that wins
            events.append(
                DeploymentCreated(
                    deployment=Deployment(
                        deployment_id=deployment_id,
                        spec=record.spec,
                        model_card=record.model_card,
                        revision=record.revision,
                    )
                )
            )
        elif current.hold is not None and (
            (current.hold.kind == "Restored" and deployment_id not in restored)
            or (deployment_id in running and deployment_id not in placed)
        ):
            events.append(
                DeploymentResumed(
                    deployment_id=deployment_id, at=now.isoformat(), by_user=False
                )
            )
    return events


def recent_losses(deployment: Deployment, now: datetime) -> int:
    return sum(
        1
        for loss in deployment.losses
        if now - datetime.fromisoformat(loss) < CRASH_LOOP_WINDOW
    )


@dataclass(frozen=True)
class Decisions:
    events: list[Event]
    # The deployments that didn't fit, and what they were tried with
    unfit: dict[DeploymentId, Hashable]


def _tried_with(
    state: State, with_files: AbstractSet[NodeId], reachable: AbstractSet[NodeId]
) -> Hashable:
    """What a placement depends on, coarsely: memory is rounded to half a GB, so that a
    deployment that didn't fit isn't tried again each time free memory moves a little. The
    links between the nodes with its files count: a node back from an absence has its files
    before its links, and its model fits only once they are there."""
    return (
        frozenset(reachable),
        frozenset(with_files),
        frozenset(
            (connection.source, connection.sink, type(connection.edge).__name__)
            for connection in state.topology.list_connections()
            if connection.source in with_files and connection.sink in with_files
        ),
        frozenset(state.instances),
        tuple(
            sorted(
                (node_id, round(usage.ram_available.in_gb * 2))
                for node_id, usage in state.node_memory.items()
                if node_id in with_files
            )
        ),
    )


@dataclass
class _Tick:
    """What one round of the reconciler shares between deployments."""

    state: State
    reachable: AbstractSet[NodeId]
    instances: dict[InstanceId, Instance]
    _cycles: Sequence[Cycle] | None = None

    @property
    def cycles(self) -> Sequence[Cycle]:
        # Found once per round, and only if something is placed: it is the costly part
        if self._cycles is None:
            self._cycles = self.state.topology.get_cycles()
        return self._cycles


def _linked(nodes: AbstractSet[NodeId], tick: _Tick) -> bool:
    """Whether an instance can span the nodes: they form a ring in the topology."""
    return len(nodes) == 1 or any(set(cycle.node_ids) == nodes for cycle in tick.cycles)


def _waiting_or_placed(
    deployment: Deployment, tick: _Tick, with_files: AbstractSet[NodeId], now: datetime
) -> Waiting | Placed:
    """Place a deployment again, or say why it waits."""
    spec = deployment.spec
    previous = set(deployment.last_nodes)
    # Its nodes are back once they have its files and links to each other again. A node the
    # master removed reports its memory and files as soon as it is back, but its links only
    # as the nodes' connection checks find them, up to 10 s later: placed in between, a
    # model that ran on two nodes came back on one
    back = bool(previous) and previous <= with_files and _linked(previous, tick)
    if (
        deployment.not_before is not None
        and now < datetime.fromisoformat(deployment.not_before)
        and not back
    ):
        return Waiting(kind="DeviceAway", until=deployment.not_before)
    if not with_files:
        return Waiting(kind="NoDevicesWithFiles")
    if len(with_files) < spec.min_nodes:
        return Waiting(
            kind="NotEnoughDevices", needed=spec.min_nodes, have=len(with_files)
        )
    if not set(spec.pinned_nodes) <= with_files:
        return Waiting(kind="DeviceAway")
    # Back on the nodes it ran on when they are all here with its files and links: a node
    # that came back (from sleep, a cut, or a restart with its id) gets its models again
    # rather than having them moved
    if back and set(spec.pinned_nodes) <= previous:
        with contextlib.suppress(ValueError):
            return place_deployment(
                deployment.deployment_id,
                spec.model_copy(
                    update={"min_nodes": max(spec.min_nodes, len(previous))}
                ),
                deployment.model_card,
                tick.state,
                tick.instances,
                allowed_nodes=previous,
                cycles=tick.cycles,
            )
    try:
        return place_deployment(
            deployment.deployment_id,
            spec,
            deployment.model_card,
            tick.state,
            tick.instances,
            allowed_nodes=with_files,
            cycles=tick.cycles,
        )
    except ValueError:
        return Waiting(kind="DoesNotFit")


def _hold(deployment: Deployment, now: datetime) -> Hold | None:
    """Why a deployment without an instance is held now, if it is."""
    if (losses := recent_losses(deployment, now)) >= CRASH_LOOP_LIMIT:
        return Hold(
            kind="CrashLoop",
            message=f"It lost a device {losses} times in "
            f"{CRASH_LOOP_WINDOW.total_seconds() / 60:.0f} minutes",
        )
    if (
        deployment.lost_at is not None
        and now - datetime.fromisoformat(deployment.lost_at) >= WAIT_LIMIT
    ):
        hours = WAIT_LIMIT.total_seconds() / 3600
        return Hold(
            kind="WaitedTooLong",
            message=f"It couldn't start again within {hours:g} h",
        )
    return None


def reconcile(
    state: State,
    *,
    now: datetime,
    settled: bool,
    reachable: AbstractSet[NodeId],
    unfit: Mapping[DeploymentId, Hashable] = {},
) -> Decisions:
    """The events that place the deployments without an instance, or say why they wait.

    `reachable` are the nodes the master's node reaches now, as its membership sees them: only
    those are placed on. `unfit` are the deployments that didn't fit last time, and what they
    were tried with: one is only tried again when that changes. Nothing is placed while not
    `settled` (the master's node is Resuming after it was stopped).
    """
    placed = {instance.deployment_id for instance in state.instances.values()}
    tick = _Tick(state, reachable, dict(state.instances))
    events: list[Event] = []
    still_unfit: dict[DeploymentId, Hashable] = {}
    for deployment_id, deployment in state.deployments.items():
        if deployment_id in placed or deployment.hold is not None:
            continue
        if (hold := _hold(deployment, now)) is not None:
            events.append(DeploymentHeld(deployment_id=deployment_id, hold=hold))
            continue
        if not settled:
            continue
        with_files = nodes_with_files(deployment.spec.model_id, state.downloads) & set(
            reachable
        )
        tried_with = _tried_with(state, with_files, reachable)
        if unfit.get(deployment_id) == tried_with:
            still_unfit[deployment_id] = tried_with
            continue
        outcome = _waiting_or_placed(deployment, tick, with_files, now)
        if isinstance(outcome, Placed):
            tick.instances[outcome.instance.instance_id] = outcome.instance
            events.append(InstanceCreated(instance=outcome.instance))
            status: Status = (None, outcome.note)
        else:
            if outcome.kind == "DoesNotFit":
                still_unfit[deployment_id] = tried_with
            status = (outcome, deployment.note)
        if status != (deployment.waiting, deployment.note):
            events.append(
                DeploymentStatus(
                    deployment_id=deployment_id, waiting=status[0], note=status[1]
                )
            )
    return Decisions(events=events, unfit=still_unfit)


def resume_event(deployment: Deployment, state: State) -> DeploymentResumed:
    """A user resumes a deployment: placed now, on the nodes that are here. Pins that no
    connected node with the files can meet are dropped, or resuming would change nothing."""
    with_files = nodes_with_files(deployment.spec.model_id, state.downloads) & set(
        state.topology.list_nodes()
    )
    return DeploymentResumed(
        deployment_id=deployment.deployment_id,
        at=datetime.now(tz=timezone.utc).isoformat(),
        unpin=not set(deployment.spec.pinned_nodes) <= with_files,
    )
