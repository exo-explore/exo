"""Deployments, rule by rule: how the reconciler decides, how deployments change as instances
come and go, and that every deployment event and command crosses the network."""

import copy
from collections.abc import Hashable
from datetime import datetime, timedelta, timezone

import pytest

from exo.master.deployments import (
    RDMA_FALLBACK_NOTE,
    announced,
    nodes_with_files,
    reconcile,
)
from exo.master.tests.test_deployments_running_master import (
    BYSTANDER,
    JOINS,
    LEAVES,
    MODEL,
    STAYS,
    cluster,
    deployed,
    instance_on,
)
from exo.routing.topics import COMMANDS, GLOBAL_EVENTS, STATE_SNAPSHOTS
from exo.shared.apply import event_apply
from exo.shared.deployment_store import record_of
from exo.shared.types.commands import (
    AnnounceDeployments,
    CreateDeployment,
    DeleteDeployment,
    ForwarderCommand,
    ResumeDeployment,
)
from exo.shared.types.common import DeploymentId, NodeId, SessionId, SystemId
from exo.shared.types.deployments import (
    CRASH_LOOP_WINDOW,
    Deployment,
    DeploymentRecord,
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
    GlobalForwarderEvent,
    InstanceCreated,
    InstanceDeleted,
    NodeGatheredInfo,
    StateSnapshot,
)
from exo.shared.types.memory import Memory
from exo.shared.types.profiling import MemoryUsage
from exo.shared.types.state import State
from exo.shared.types.worker.downloads import (
    DownloadCompleted,
    DownloadFailed,
    DownloadOngoing,
    DownloadPending,
    DownloadProgress,
    DownloadProgressData,
)
from exo.shared.types.worker.instances import InstanceMeta, InstanceRemovalReason
from exo.shared.types.worker.shards import Sharding

NOW = datetime(2026, 11, 3, 12, 0, tzinfo=timezone.utc)


def lost(state: State, *, at: datetime = NOW) -> State:
    """The state after its one instance was lost with a node."""
    (instance,) = state.instances.values()
    return event_apply(
        InstanceDeleted(
            instance_id=instance.instance_id,
            reason="NodeLost",
            at=at.isoformat(),
        ),
        state,
    )


def without(state: State, node_id: NodeId) -> State:
    """The state after node_id left the topology (a copy: topologies are mutable)."""
    topology = copy.deepcopy(state.topology)
    topology.remove_node(node_id)
    return state.model_copy(update={"topology": topology})


def decide(
    state: State,
    *,
    now: datetime = NOW,
    settled: bool = True,
    reachable: set[NodeId] | None = None,
    unfit: dict[DeploymentId, Hashable] | None = None,
) -> list[Event]:
    """What the reconciler decides, with every node in the topology reachable."""
    return reconcile(
        state,
        now=now,
        settled=settled,
        reachable=set(state.topology.list_nodes()) if reachable is None else reachable,
        unfit=unfit or {},
    ).events


def only(state: State) -> Deployment:
    (deployment,) = state.deployments.values()
    return deployment


def statuses(events: list[Event]) -> list[Waiting | None]:
    return [event.waiting for event in events if isinstance(event, DeploymentStatus)]


def created(events: list[Event]) -> list[frozenset[NodeId]]:
    return [
        frozenset(event.instance.shard_assignments.node_to_runner)
        for event in events
        if isinstance(event, InstanceCreated)
    ]


def three_node_loss(min_nodes: int = 1) -> State:
    """A 2-node deployment on STAYS and LEAVES that lost LEAVES; BYSTANDER has no files."""
    state = cluster(
        deployed(instance_on(STAYS, LEAVES), min_nodes=min_nodes),
        STAYS,
        LEAVES,
        BYSTANDER,
        without_files=(BYSTANDER,),
    )
    return without(lost(state), LEAVES)


# The reconciler


def test_a_lost_deployment_is_placed_on_the_nodes_left_with_its_files() -> None:
    events = decide(three_node_loss())

    assert created(events) == [frozenset({STAYS})]


def test_the_users_node_count_is_kept() -> None:
    events = decide(three_node_loss(min_nodes=2))

    assert created(events) == []
    assert statuses(events) == [Waiting(kind="NotEnoughDevices", needed=2, have=1)]


def test_a_pinned_node_that_is_away_makes_it_wait() -> None:
    state = three_node_loss()
    (deployment,) = state.deployments.values()
    state = state.model_copy(
        update={
            "deployments": {
                deployment.deployment_id: deployment.model_copy(
                    update={
                        "spec": deployment.spec.model_copy(
                            update={"pinned_nodes": [LEAVES]}
                        )
                    }
                )
            }
        }
    )

    events = decide(state)

    assert created(events) == []
    assert statuses(events) == [Waiting(kind="DeviceAway")]


def test_nothing_is_placed_while_the_masters_node_is_resuming() -> None:
    assert decide(three_node_loss(), settled=False) == []


def test_a_merged_launch_waits_until_its_time_or_its_nodes() -> None:
    """A launch the master learned of from a node waits for the node it last ran on."""
    until = (NOW + WAIT).isoformat()
    state = lost(cluster(deployed(instance_on(LEAVES)), STAYS, LEAVES))
    deployment = only(state).model_copy(update={"not_before": until})
    waits = state.model_copy(
        update={"deployments": {deployment.deployment_id: deployment}}
    )
    away = without(waits, LEAVES)

    assert created(decide(away)) == []
    assert statuses(decide(away)) == [Waiting(kind="DeviceAway", until=until)]
    # Its time has come: it goes on the node that is left
    later = NOW + WAIT + timedelta(seconds=1)
    assert created(decide(away, now=later)) == [frozenset({STAYS})]
    # Or its node is back: it goes back there at once
    assert created(decide(waits)) == [frozenset({LEAVES})]


def test_it_goes_back_on_its_own_nodes_when_they_are_all_here() -> None:
    """Even when fewer nodes would hold it: a node that came back with its id gets its models
    back."""
    state = cluster(deployed(instance_on(STAYS, LEAVES)), STAYS, LEAVES, JOINS)
    (instance,) = state.instances.values()
    back = event_apply(
        InstanceDeleted(
            instance_id=instance.instance_id, reason="NodeLost", at=NOW.isoformat()
        ),
        state,
    )

    assert created(decide(back)) == [frozenset({STAYS, LEAVES})]


def test_an_rdma_model_comes_back_over_tcp_and_says_so() -> None:
    state = three_node_loss()
    (deployment,) = state.deployments.values()
    rdma = deployment.model_copy(
        update={
            "spec": deployment.spec.model_copy(
                update={"instance_meta": InstanceMeta.MlxJaccl}
            )
        }
    )
    state = state.model_copy(update={"deployments": {rdma.deployment_id: rdma}})

    events = decide(state)

    assert created(events) == [frozenset({STAYS})]
    (status,) = [event for event in events if isinstance(event, DeploymentStatus)]
    assert (status.waiting, status.note) == (None, RDMA_FALLBACK_NOTE)


def test_a_deployment_that_keeps_losing_its_device_is_held() -> None:
    state = three_node_loss()
    for minutes in (1, 2):
        (deployment,) = state.deployments.values()
        state = state.model_copy(
            update={
                "deployments": {
                    deployment.deployment_id: deployment.model_copy(
                        update={
                            "losses": [
                                *deployment.losses,
                                (NOW - timedelta(minutes=minutes)).isoformat(),
                            ]
                        }
                    )
                }
            }
        )

    events = decide(state)

    assert created(events) == []
    (held,) = [event for event in events if isinstance(event, DeploymentHeld)]
    assert held.hold == Hold(
        kind="CrashLoop", message="It lost a device 3 times in 10 minutes"
    )


def test_a_held_deployment_is_left_alone() -> None:
    state = three_node_loss()
    state = event_apply(
        DeploymentHeld(
            deployment_id=only(state).deployment_id,
            hold=Hold(kind="StartFailed", message="no"),
        ),
        state,
    )

    assert decide(state) == []


def test_a_status_the_state_has_is_not_sent_again() -> None:
    state = three_node_loss(min_nodes=2)
    (status,) = [
        event for event in decide(state) if isinstance(event, DeploymentStatus)
    ]

    assert decide(event_apply(status, state)) == []


def test_a_waiting_reason_is_not_sent_again_each_time_free_memory_moves() -> None:
    """A reason quoting free memory was a new event for every node each time
    memory moved by 0.1 GB."""

    def memory_event(megabytes: int) -> NodeGatheredInfo:
        return NodeGatheredInfo(
            node_id=STAYS,
            when=NOW.isoformat(),
            info=MemoryUsage.from_bytes(
                ram_total=64 * 1024**3,
                ram_available=megabytes * 1024**2,
                swap_total=0,
                swap_available=0,
            ),
        )

    waiting = event_apply(memory_event(10), three_node_loss())
    first = decide(waiting)
    assert statuses(first) == [Waiting(kind="DoesNotFit")]
    for event in first:
        waiting = event_apply(event, waiting)
    for megabytes in (11, 20, 50):
        assert decide(event_apply(memory_event(megabytes), waiting)) == []


SHARD = next(iter(instance_on(STAYS).shard_assignments.runner_to_shard.values()))


def _ongoing(downloaded: int) -> DownloadOngoing:
    return DownloadOngoing(
        node_id=STAYS,
        shard_metadata=SHARD,
        download_progress=DownloadProgressData(
            total=MODEL.storage_size,
            downloaded=Memory.from_bytes(downloaded),
            downloaded_this_session=Memory.from_bytes(downloaded),
            completed_files=0,
            total_files=1,
            speed=0.0,
            eta_ms=0,
            files={},
        ),
    )


@pytest.mark.parametrize(
    ("progress", "has_files"),
    [
        (
            DownloadCompleted(
                node_id=STAYS, shard_metadata=SHARD, total=MODEL.storage_size
            ),
            True,
        ),
        (_ongoing(1), True),
        (_ongoing(0), False),
        # A failed download counted as having the files
        (
            DownloadFailed(
                node_id=STAYS, shard_metadata=SHARD, error_message="disk full"
            ),
            False,
        ),
        (
            DownloadPending(
                node_id=STAYS,
                shard_metadata=SHARD,
                downloaded=Memory.from_bytes(10),
                total=MODEL.storage_size,
            ),
            False,
        ),
    ],
)
def test_which_downloads_count_as_having_the_files(
    progress: DownloadProgress, has_files: bool
) -> None:
    assert nodes_with_files(MODEL.model_id, {STAYS: [progress]}) == (
        {STAYS} if has_files else set()
    )


# How deployments change


def test_a_lost_node_counts_as_a_loss() -> None:
    state = cluster(deployed(instance_on(STAYS, LEAVES)), STAYS, LEAVES)

    deployment = only(lost(state))

    assert deployment.losses == [NOW.isoformat()]
    assert deployment.last_nodes == [STAYS, LEAVES]
    assert deployment.lost_at == NOW.isoformat()


def test_losses_older_than_the_window_are_forgotten() -> None:
    state = cluster(deployed(instance_on(STAYS)), STAYS)
    old = NOW - CRASH_LOOP_WINDOW - timedelta(seconds=1)
    state = lost(state, at=old)
    (deployment,) = state.deployments.values()
    again = instance_on(STAYS).model_copy(
        update={"deployment_id": deployment.deployment_id}
    )
    state = event_apply(InstanceCreated(instance=again), state)

    assert only(lost(state)).losses == [NOW.isoformat()]


def test_a_failed_start_is_not_a_loss_and_a_lost_node_is() -> None:
    """A lost node counts towards a crash loop: a model that takes its node down as it loads
    would otherwise be loaded forever."""
    state = cluster(deployed(instance_on(STAYS)), STAYS)
    (instance,) = state.instances.values()

    def after(reason: InstanceRemovalReason) -> list[str]:
        return only(
            event_apply(
                InstanceDeleted(
                    instance_id=instance.instance_id, reason=reason, at=NOW.isoformat()
                ),
                state,
            )
        ).losses

    assert after("StartFailed") == []
    assert after("NodeLost") == [NOW.isoformat()]


def test_placing_it_clears_the_wait_and_resuming_clears_everything() -> None:
    state = lost(cluster(deployed(instance_on(STAYS)), STAYS))
    deployment_id = only(state).deployment_id
    state = event_apply(
        DeploymentStatus(
            deployment_id=deployment_id, waiting=Waiting(kind="DeviceAway"), note=None
        ),
        state,
    )
    placed = event_apply(
        InstanceCreated(
            instance=instance_on(STAYS).model_copy(
                update={"deployment_id": deployment_id}
            )
        ),
        state,
    )
    assert (only(placed).waiting, only(placed).lost_at) == (None, None)

    held = event_apply(
        DeploymentHeld(deployment_id=deployment_id, hold=Hold(kind="CrashLoop")), state
    )
    resumed = event_apply(
        DeploymentResumed(deployment_id=deployment_id, at=NOW.isoformat()), held
    )
    # A hold is the master's: only the user's resume counts as a change of the deployment
    assert only(held).revision == 0
    assert (
        only(resumed).hold,
        only(resumed).losses,
        only(resumed).waiting,
        only(resumed).revision,
    ) == (None, [], None, 1)


def test_a_deployment_is_created_once_and_removed_by_its_id() -> None:
    state = cluster(deployed(instance_on(STAYS)), STAYS)
    deployment = only(state)
    # The same revision again, from another node: nothing changes
    again = deployment.model_copy(update={"hold": Hold(kind="CrashLoop")})

    assert only(event_apply(DeploymentCreated(deployment=again), state)) == deployment
    assert (
        event_apply(
            DeploymentRemoved(
                deployment_id=deployment.deployment_id, at=NOW.isoformat()
            ),
            state,
        ).deployments
        == {}
    )


# The network


def test_every_deployment_event_and_command_crosses_the_network() -> None:
    deployment_id = DeploymentId()
    spec = DeploymentSpec(
        model_id=MODEL.model_id,
        sharding=Sharding.Tensor,
        instance_meta=InstanceMeta.MlxJaccl,
        min_nodes=2,
        pinned_nodes=[STAYS],
    )
    deployment = Deployment(
        deployment_id=deployment_id,
        spec=spec,
        model_card=MODEL,
        revision=3,
        hold=Hold(kind="CrashLoop", message="lost"),
        last_nodes=[STAYS, LEAVES],
        not_before=NOW.isoformat(),
        lost_at=NOW.isoformat(),
        losses=[NOW.isoformat()],
        waiting=Waiting(kind="DeviceAway", until=NOW.isoformat()),
        note="a note",
    )
    master = NodeId("master")
    session = SessionId(master_node_id=master, election_clock=1)
    events: list[Event] = [
        DeploymentCreated(deployment=deployment),
        DeploymentRemoved(deployment_id=deployment_id, at=NOW.isoformat()),
        DeploymentHeld(deployment_id=deployment_id, hold=Hold(kind="StartFailed")),
        DeploymentResumed(deployment_id=deployment_id, at=NOW.isoformat(), unpin=True),
        DeploymentStatus(
            deployment_id=deployment_id, waiting=Waiting(kind="DoesNotFit"), note=None
        ),
        InstanceDeleted(
            instance_id=instance_on(STAYS).instance_id,
            reason="NodeLost",
            at=NOW.isoformat(),
        ),
    ]
    for event in events:
        sent = GlobalForwarderEvent(
            origin_idx=0, origin=master, session=session, event=event
        )
        assert GLOBAL_EVENTS.deserialize(GLOBAL_EVENTS.serialize(sent)) == sent
    for command in (
        CreateDeployment(
            deployment_id=deployment_id,
            spec=spec,
            model_card=MODEL,
            instance=instance_on(STAYS),
        ),
        DeleteDeployment(deployment_id=deployment_id),
        ResumeDeployment(deployment_id=deployment_id, revision=3),
        AnnounceDeployments(
            deployments=[record_of(deployment)],
            removed={DeploymentId(): NOW.isoformat()},
            restored=[deployment_id],
        ),
    ):
        forwarded = ForwarderCommand(origin=SystemId(), command=command)
        assert COMMANDS.deserialize(COMMANDS.serialize(forwarded)) == forwarded
    snapshot = StateSnapshot(
        session=session,
        requester=SystemId(),
        state=State(
            deployments={deployment_id: deployment},
            removed_deployments={DeploymentId(): NOW.isoformat()},
        ),
    )
    received = STATE_SNAPSHOTS.deserialize(STATE_SNAPSHOTS.serialize(snapshot))
    assert received.state.deployments == snapshot.state.deployments
    assert received.state.removed_deployments == snapshot.state.removed_deployments


# What a node tells a new master


# How long a launch learned from a node waits for the nodes it last ran on
WAIT = timedelta(seconds=15)


def _announced(state: State, announcement: AnnounceDeployments) -> list[Event]:
    return announced(state, announcement, NOW, WAIT)


def _record(deployment: Deployment) -> DeploymentRecord:
    return record_of(deployment)


def _announce(
    records: list[DeploymentRecord],
    removed: dict[DeploymentId, str] | None = None,
    restored: list[DeploymentId] | None = None,
    running: list[DeploymentId] | None = None,
) -> AnnounceDeployments:
    return AnnounceDeployments(
        deployments=records,
        removed=removed or {},
        restored=restored or [],
        running=running or [],
    )


def _apply_all(state: State, events: list[Event]) -> State:
    for event in events:
        state = event_apply(event, state)
    return state


def test_a_launch_the_master_never_saw_is_added_as_it_was() -> None:
    """Two Macs: launched on the side that lost the election. It comes with
    the nodes it ran on, and waits for them a while, so it comes back on its own node rather
    than moving to another that has the files."""
    on_leaves = cluster(deployed(instance_on(LEAVES)), LEAVES)
    (instance,) = on_leaves.instances.values()
    launched = only(on_leaves)

    events = _announced(
        State(),
        _announce([record_of(launched, [LEAVES])], running=[instance.deployment_id]),
    )

    after = _apply_all(State(), events)
    assert only(after).deployment_id == launched.deployment_id
    assert only(after).hold is None
    assert only(after).last_nodes == [LEAVES]
    assert only(after).not_before == (NOW + WAIT).isoformat()


def test_a_merged_launch_comes_back_on_its_own_node() -> None:
    """Both nodes have its files and room: it goes back where it ran, once that node has its
    files and links, within the wait. Without the nodes in the record it went to whichever
    fitted best (on the other Mac in 1 of 2 runs)."""
    on_leaves = cluster(deployed(instance_on(LEAVES)), STAYS, LEAVES)
    launched = only(on_leaves)
    lacking = on_leaves.model_copy(update={"instances": {}, "deployments": {}})
    after = _apply_all(
        lacking, _announced(lacking, _announce([record_of(launched, [LEAVES])]))
    )

    assert created(decide(after)) == [frozenset({LEAVES})]


def test_a_heal_where_one_side_held_it_and_the_other_ran_it_keeps_it_running() -> None:
    """Side 1's master held a deployment (it couldn't start it within
    WAIT_LIMIT) while side 2 ran it. Side 2 wins the election, and side 1 tells it its view.
    Its hold was its own side's: the running deployment stays as it is. A hold used to bump the
    revision, so side 1's record won the merge and paused the running model."""
    running_side = cluster(deployed(instance_on(STAYS)), STAYS)
    deployment = only(running_side)
    held_side = event_apply(
        DeploymentHeld(
            deployment_id=deployment.deployment_id,
            hold=Hold(
                kind="WaitedTooLong", message="It couldn't start again within 1 h"
            ),
        ),
        running_side,
    )

    events = _announced(running_side, _announce([record_of(only(held_side))]))

    after = _apply_all(running_side, events)
    assert events == []
    assert only(after).hold is None
    assert len(after.instances) == 1


def test_a_heal_where_the_side_that_held_it_wins_lifts_the_hold() -> None:
    """The other way round: side 1 (which held it) wins, and side 2 says its master was
    running it. The hold was side 1's own: it is lifted, without counting as a change by the
    user, and the deployment is placed again."""
    running_side = cluster(deployed(instance_on(STAYS)), STAYS)
    deployment = only(running_side)
    held = event_apply(
        DeploymentHeld(
            deployment_id=deployment.deployment_id,
            hold=Hold(
                kind="WaitedTooLong", message="It couldn't start again within 1 h"
            ),
        ),
        running_side.model_copy(update={"instances": {}}),
    )

    after = _apply_all(
        held,
        _announced(
            held,
            _announce(
                [record_of(deployment, [STAYS])], running=[deployment.deployment_id]
            ),
        ),
    )

    assert only(after).hold is None
    assert only(after).revision == deployment.revision
    assert created(decide(after)) == [frozenset({STAYS})]


def test_a_deployment_only_on_disks_comes_back_held() -> None:
    """The whole cluster restarted: models aren't loaded again at start-up."""
    launched = only(cluster(deployed(instance_on(STAYS)), STAYS))

    after = _apply_all(
        State(),
        _announced(
            State(),
            _announce([_record(launched)], restored=[launched.deployment_id]),
        ),
    )

    hold = only(after).hold
    assert hold is not None and hold.kind == "Restored"
    # A node that has run it since it started shows it was no full restart
    resumed = _apply_all(after, _announced(after, _announce([_record(launched)])))
    assert only(resumed).hold is None


def test_a_stop_beats_a_launch_and_ends_it() -> None:
    state = cluster(deployed(instance_on(STAYS)), STAYS)
    running = only(state)

    after = _apply_all(
        state,
        _announced(
            state,
            _announce(
                [_record(running)], removed={running.deployment_id: NOW.isoformat()}
            ),
        ),
    )

    assert after.deployments == {}
    assert after.removed_deployments == {running.deployment_id: NOW.isoformat()}
    # A node that still has it can't bring it back
    assert _announced(after, _announce([_record(running)])) == []


def test_the_users_later_change_wins_and_an_older_record_changes_nothing() -> None:
    """The user resumed it on the other side, dropping a pin: that change is the latest, and
    wins over this master's hold."""
    state = cluster(deployed(instance_on(STAYS)), STAYS)
    current = only(state)
    held = event_apply(
        DeploymentHeld(
            deployment_id=current.deployment_id, hold=Hold(kind="CrashLoop")
        ),
        state,
    )
    unpinned = current.spec.model_copy(update={"pinned_nodes": []})
    newer = _record(current).model_copy(update={"revision": 4, "spec": unpinned})

    after = _apply_all(held, _announced(held, _announce([newer])))

    assert (only(after).revision, only(after).hold, only(after).spec) == (
        4,
        None,
        unpinned,
    )
    assert only(after).last_nodes == current.last_nodes
    assert _announced(after, _announce([_record(current)])) == []


def test_announcing_what_the_master_has_changes_nothing() -> None:
    state = cluster(deployed(instance_on(STAYS)), STAYS)
    state = event_apply(
        DeploymentRemoved(deployment_id=DeploymentId(), at=NOW.isoformat()), state
    )

    assert (
        _announced(
            state,
            _announce([_record(only(state))], removed=dict(state.removed_deployments)),
        )
        == []
    )
