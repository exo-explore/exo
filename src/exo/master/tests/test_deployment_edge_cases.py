# pyright: reportPrivateUsage=false
"""Deployments at their edges: splits (as the master's membership reads them), the same launch
posted twice, a failed start, a node that returns, the cost of waiting deployments, and how long
a deployment may wait."""

from datetime import timedelta

import anyio
import pytest

import exo.master.main as master_main
from exo.master.deployments import reconcile
from exo.master.tests.test_deployments_running_master import (
    MODEL,
    Cluster,
    cluster,
    deployed,
    edges_between,
    instance_on,
    launch_on,
    running_master,
    wait_until,
)
from exo.routing.membership import MemberState, Quorum
from exo.shared.apply import event_apply
from exo.shared.topology import Topology
from exo.shared.types.commands import (
    CreateDeployment,
    DeleteDeployment,
    InstanceBroken,
    ResumeDeployment,
)
from exo.shared.types.common import DeploymentId, NodeId
from exo.shared.types.deployments import WAIT_LIMIT, DeploymentSpec, Hold
from exo.shared.types.events import (
    DeploymentHeld,
    DeploymentResumed,
    InstanceCreated,
    InstanceDeleted,
)
from exo.shared.types.memory import Memory
from exo.shared.types.worker.instances import InstanceMeta
from exo.shared.types.worker.shards import Sharding

A, B, C, D = NodeId("a"), NodeId("b"), NodeId("c"), NodeId("d")
# In the cluster but without the model's files
X = NodeId("x")


@pytest.fixture(autouse=True)
def fast_master(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(master_main, "PLAN_INTERVAL", 0.05)


async def record_placements(running: Cluster, seen: list[frozenset[NodeId]]) -> None:
    known: set[str] = set()
    while True:
        for instance_id, instance in running.state.instances.items():
            if instance_id not in known:
                known.add(instance_id)
                seen.append(frozenset(instance.shard_assignments.node_to_runner))
        await anyio.sleep(0.005)


# Splits


async def test_an_even_split_keeps_the_lost_nodes_models_and_places_nothing() -> None:
    """4 nodes; one model on C+D, one on C and one on D. C and D are cut off together: this
    side (A, B) reaches half of the cluster, and the other side may be serving the models.
    Their instances are kept, with C and D, for the split hold, and nothing is placed here.
    Once the hold is over the quorum read is a majority of the nodes this side reaches: C
    and D are removed, and every model is placed here at once."""
    on_both, on_c, on_d = (
        deployed(instance_on(C, D)),
        deployed(instance_on(C)),
        deployed(instance_on(D)),
    )
    initial = cluster(
        on_both.model_copy(
            update={
                "instances": {
                    **on_both.instances,
                    **on_c.instances,
                    **on_d.instances,
                },
                "deployments": {
                    **on_both.deployments,
                    **on_c.deployments,
                    **on_d.deployments,
                },
            }
        ),
        A,
        B,
        C,
        D,
    )
    originals = set(initial.instances)
    async with running_master(initial) as running:
        running.lose(C, D, quorum="Half")
        await anyio.sleep(0.5)
        assert set(running.state.instances) == originals
        assert {C, D} <= set(running.state.topology.list_nodes())

        running.lose(C, D, quorum="Majority")
        await wait_until(
            lambda: not originals & running.state.instances.keys()
            and len(running.state.instances) == 3
        )
        assert all(nodes <= {A, B} for nodes in running.model_nodes())


async def test_a_minority_places_nothing() -> None:
    """3 nodes, the model on B+C; the master's node A is cut off from both: most likely it
    is the one cut off, and the other side still runs the model. Nothing is removed or
    placed until the split hold is over."""
    initial = cluster(deployed(instance_on(B, C)), A, B, C)
    (original,) = initial.instances
    async with running_master(initial) as running:
        running.lose(B, C, quorum="Minority")
        await anyio.sleep(0.5)
        assert list(running.state.instances) == [original]

        running.lose(B, C, quorum="Majority")
        await wait_until(lambda: running.model_nodes() == [frozenset({A})])


async def test_the_hold_ends_when_this_side_is_a_majority_again() -> None:
    """2 nodes; the model's node restarts under a new id (a restart is a new node): its old
    id never comes back, but the cluster has two nodes again, so the quorum read is a
    majority, the old id is removed, and the model doesn't wait for the whole split hold."""
    initial = cluster(deployed(instance_on(C)), A, C)
    (original,) = initial.instances
    restarted = NodeId("c-again")
    async with running_master(initial) as running:
        running.lose(C, quorum="Half")
        await anyio.sleep(0.3)
        assert list(running.state.instances) == [original]

        await running.join(restarted, A)
        running.membership.quorum = "Majority"

        await wait_until(
            lambda: original not in running.state.instances
            and running.model_nodes() != []
        )
        assert C not in running.state.topology.list_nodes()


@pytest.mark.parametrize(
    ("state", "quorum"),
    [("Gone", "Half"), ("Suspect", "Majority")],
    ids=["gone_in_an_even_split", "suspect"],
)
async def test_a_launch_replaces_the_models_copy_this_node_cant_reach(
    state: MemberState, quorum: Quorum
) -> None:
    """A pair; the model on B, which A can't reach: its requests fail, and its card says so. The
    user launches it again on A. That is the model they want, so the copy kept for B goes with
    its deployment. It had an instance, so a launch didn't replace it, and when B was removed
    it was placed too: two copies on A."""
    initial = cluster(deployed(instance_on(B)), A, B)
    (kept,) = initial.instances
    async with running_master(initial) as running:
        running.membership.quorum = quorum
        running.membership.set(B, state, "Disconnected")
        await anyio.sleep(0.2)
        assert list(running.state.instances) == [kept]

        launch = launch_on(A)
        await running.command(launch)
        await wait_until(lambda: launch.deployment_id in running.state.deployments)
        assert list(running.state.deployments) == [launch.deployment_id]
        assert running.model_nodes() == [frozenset({A})]
        assert _removals(running) == [(kept, "UserStopped")]

        # B is removed, once the split hold is over
        running.lose(B, quorum="Majority")
        await wait_until(lambda: B not in running.state.topology.list_nodes())
        await anyio.sleep(0.3)
        assert running.model_nodes() == [frozenset({A})]


async def test_a_launch_that_replaced_a_kept_copy_stays_stopped_when_stopped() -> None:
    """As above, for a model that fits on A only once: the kept deployment waited, unseen next
    to the running copy, and when the user stopped that copy the model started again by
    itself."""
    big = MODEL.model_copy(update={"storage_size": Memory.from_gb(20)})
    initial = cluster(deployed(instance_on(B)), A, B)
    (kept,) = initial.deployments
    initial = initial.model_copy(
        update={
            "deployments": {
                kept: initial.deployments[kept].model_copy(update={"model_card": big})
            }
        }
    )
    launch = _launch((A,), card_gb=20)
    async with running_master(initial) as running:
        running.lose(B, quorum="Half")
        await anyio.sleep(0.2)
        await running.command(launch)
        await wait_until(lambda: launch.deployment_id in running.state.deployments)
        running.lose(B, quorum="Majority")
        await wait_until(lambda: B not in running.state.topology.list_nodes())

        await running.command(DeleteDeployment(deployment_id=launch.deployment_id))
        await wait_until(lambda: running.state.deployments == {})
        await anyio.sleep(0.3)
        assert running.model_nodes() == []


async def test_a_launch_leaves_a_copy_this_node_reaches_running() -> None:
    """Two launches of a model, both on nodes this node reaches, are two copies, as asked."""
    initial = cluster(deployed(instance_on(B)), A, B)
    (first,) = initial.deployments
    launch = launch_on(A)
    async with running_master(initial) as running:
        await running.command(launch)
        await wait_until(lambda: launch.deployment_id in running.state.deployments)
        await anyio.sleep(0.2)
        assert set(running.state.deployments) == {first, launch.deployment_id}
        assert sorted(running.model_nodes(), key=sorted) == [
            frozenset({A}),
            frozenset({B}),
        ]


async def test_a_model_on_two_nodes_comes_back_on_both_when_their_links_come_late() -> (
    None
):
    """The model on A+C, at least 2 nodes; X has no files. C sleeps past the grace period
    and is removed (2 of 3 is a majority), and the model waits for 2 nodes with its files.
    C wakes: it reports its memory and files at once, its links to the others only as the
    connection checks find them, up to 10 s later. The model comes back on A+C: it didn't fit
    before the links came, and wasn't tried again when they did, while the memo of what a
    placement that didn't fit was tried with left the links out."""
    initial = cluster(
        deployed(instance_on(A, C), min_nodes=2), A, C, X, without_files=(X,)
    )
    (original,) = initial.instances
    seen: list[frozenset[NodeId]] = []
    async with running_master(initial) as running, anyio.create_task_group() as tg:
        tg.start_soon(record_placements, running, seen)
        running.lose(C)
        await wait_until(lambda: original not in running.state.instances)
        await wait_until(lambda: running.deployment().waiting is not None)
        # C wakes: its memory and files first
        await running.join(C)
        await anyio.sleep(0.5)
        # Its links later
        for other in (A, X):
            await running.send(edges_between(other, C))
        await wait_until(lambda: len(seen) > 1)
        await anyio.sleep(0.5)
        tg.cancel_scope.cancel()
    assert seen == [frozenset({A, C}), frozenset({A, C})], seen


# The same launch posted twice


def _launch(
    instance_nodes: tuple[NodeId, ...], card_gb: float | None = None
) -> CreateDeployment:
    """A launch of the same preview (a double click, a client's retry of POST /instance)."""
    card = (
        MODEL
        if card_gb is None
        else MODEL.model_copy(update={"storage_size": Memory.from_gb(card_gb)})
    )
    template = instance_on(*instance_nodes)
    preview = template.model_copy(
        update={
            "instance_id": _PREVIEW_ID,
            "shard_assignments": template.shard_assignments.model_copy(
                update={
                    "runner_to_shard": {
                        runner: shard.model_copy(update={"model_card": card})
                        for runner, shard in template.shard_assignments.runner_to_shard.items()
                    }
                }
            ),
        }
    )
    deployment_id = DeploymentId()
    return CreateDeployment(
        deployment_id=deployment_id,
        spec=DeploymentSpec(
            model_id=MODEL.model_id,
            sharding=Sharding.Pipeline,
            instance_meta=InstanceMeta.MlxRing,
        ),
        model_card=card,
        instance=preview.model_copy(update={"deployment_id": deployment_id}),
    )


_PREVIEW_ID = instance_on(A).instance_id


async def test_the_same_placement_posted_twice_runs_once() -> None:
    """A second launch of the same placement made a second deployment, which took the
    instance over and left the first to be placed again: two copies."""
    initial = cluster(deployed(instance_on(A)), A, B, C).model_copy(
        update={"instances": {}, "deployments": {}}
    )
    async with running_master(initial) as running:
        await running.command(_launch((B,)))
        await wait_until(lambda: running.state.instances != {})
        await running.command(_launch((B,)))
        await anyio.sleep(1.0)
    assert len(running.state.instances) == 1
    assert len(running.state.deployments) == 1


async def test_a_model_posted_twice_and_stopped_stays_stopped() -> None:
    """When it fits once, the second deployment waited where nobody saw it, and started the
    model again as soon as the user stopped the one they saw."""
    initial = cluster(deployed(instance_on(A)), A, B, without_files=(A,)).model_copy(
        update={"instances": {}, "deployments": {}}
    )
    async with running_master(initial) as running:
        await running.command(_launch((B,), card_gb=20))
        await running.command(_launch((B,), card_gb=20))
        await anyio.sleep(0.5)
        (visible,) = running.state.instances.values()
        await running.command(DeleteDeployment(deployment_id=visible.deployment_id))
        await anyio.sleep(1.0)
    assert running.state.instances == {}
    assert running.state.deployments == {}


# Sweeps


def _removals(running: Cluster) -> list[tuple[str, str]]:
    return [
        (event.instance_id, event.reason)
        for event in running.master._recent_events
        if isinstance(event, InstanceDeleted)
    ]


async def test_a_second_instance_of_one_deployment_is_removed() -> None:
    """Two instances of one deployment (there was no sweep for it): the master removes the
    second as a Duplicate, and keeps the deployment and the first."""
    initial = cluster(deployed(instance_on(C)), A, C)
    ((deployment_id, _),) = initial.deployments.items()
    (first,) = initial.instances
    second = instance_on(A).model_copy(update={"deployment_id": deployment_id})
    initial = initial.model_copy(
        update={"instances": {**initial.instances, second.instance_id: second}}
    )
    async with running_master(initial) as running:
        await wait_until(lambda: len(running.state.instances) == 1)
        await anyio.sleep(0.5)
    assert list(running.state.instances) == [first]
    assert list(running.state.deployments) == [deployment_id]
    assert _removals(running) == [(second.instance_id, "Duplicate")]


async def test_an_instance_whose_deployment_was_stopped_is_removed() -> None:
    """An instance whose deployment isn't in the state (a stop that crossed its placement,
    or one merged in from another node) is removed as stopped by the user."""
    initial = cluster(deployed(instance_on(C)), A, C)
    (orphan,) = initial.instances
    initial = initial.model_copy(update={"deployments": {}})
    async with running_master(initial) as running:
        await wait_until(lambda: running.state.instances == {})
        await anyio.sleep(0.5)
    assert running.state.deployments == {}
    assert _removals(running) == [(orphan, "UserStopped")]


# A failed start


async def test_a_model_that_failed_to_start_is_held_before_anything_places_it_again(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The hold comes before the removal: a plan round between them placed the model again
    (30 of 30 runs with no delay between rounds)."""
    monkeypatch.setattr(master_main, "PLAN_INTERVAL", 0.0)
    for _ in range(10):
        initial = cluster(deployed(instance_on(A)), A, B)
        (instance,) = initial.instances.values()
        seen: list[frozenset[NodeId]] = []
        async with running_master(initial) as running, anyio.create_task_group() as tg:
            tg.start_soon(record_placements, running, seen)
            await running.command(
                InstanceBroken(instance_id=instance.instance_id, reason="StartFailed")
            )
            await wait_until(lambda: running.deployment().hold is not None)
            await anyio.sleep(0.1)
            tg.cancel_scope.cancel()
        assert len(seen) == 1, f"placed again before the hold: {seen}"


def test_resuming_a_model_that_failed_to_start_tries_other_nodes() -> None:
    state = cluster(deployed(instance_on(A)), A, B)
    (instance,) = state.instances.values()
    for event in (
        DeploymentHeld(
            deployment_id=instance.deployment_id, hold=Hold(kind="StartFailed")
        ),
        InstanceDeleted(
            instance_id=instance.instance_id,
            reason="StartFailed",
            at="2026-11-03T12:00:00+00:00",
        ),
        DeploymentResumed(
            deployment_id=instance.deployment_id, at="2026-11-03T12:00:01+00:00"
        ),
    ):
        state = event_apply(event, state)

    (deployment,) = state.deployments.values()
    assert deployment.last_nodes == []


# Waiting that ends


def _waiting_since(lost_at: str):  # noqa: ANN202
    state = cluster(deployed(instance_on(C), min_nodes=2), A, C)
    (instance,) = state.instances.values()
    state = event_apply(
        InstanceDeleted(
            instance_id=instance.instance_id, reason="NodeLost", at=lost_at
        ),
        state,
    )
    topology = Topology()
    topology.add_node(A)
    return state.model_copy(update={"topology": topology})


def test_a_deployment_that_waits_too_long_is_held_for_the_user() -> None:
    """A model that can't come back for WAIT_LIMIT waits for the user, rather than loading by
    itself when it fits at last, hours or days later."""
    lost_at = "2026-11-03T12:00:00+00:00"
    state = _waiting_since(lost_at)
    from datetime import datetime

    before = datetime.fromisoformat(lost_at) + WAIT_LIMIT - timedelta(seconds=1)
    after = datetime.fromisoformat(lost_at) + WAIT_LIMIT

    def holds(now: datetime) -> list[Hold]:
        return [
            event.hold
            for event in reconcile(state, now=now, settled=True, reachable={A}).events
            if isinstance(event, DeploymentHeld)
        ]

    assert holds(before) == []
    (hold,) = holds(after)
    assert hold.kind == "WaitedTooLong"


async def test_resuming_a_model_pinned_to_a_node_that_is_away_loads_it_on_the_others() -> (
    None
):
    """Resume did nothing for a pinned wait: the pin still couldn't be met."""
    pinned = deployed(instance_on(C))
    (deployment,) = pinned.deployments.values()
    pinned = pinned.model_copy(
        update={
            "deployments": {
                deployment.deployment_id: deployment.model_copy(
                    update={
                        "spec": deployment.spec.model_copy(update={"pinned_nodes": [C]})
                    }
                )
            }
        }
    )
    initial = cluster(pinned, A, B, C)
    async with running_master(initial) as running:
        running.lose(C)
        await wait_until(lambda: running.deployment().waiting is not None)
        await anyio.sleep(0.3)
        assert running.model_nodes() == []

        await running.command(
            ResumeDeployment(
                deployment_id=running.deployment().deployment_id,
                revision=running.deployment().revision,
            )
        )

        await wait_until(lambda: running.model_nodes() != [])
        assert running.deployment().spec.pinned_nodes == []


# The cost of waiting deployments


def test_a_round_finds_the_topologys_cycles_once_and_skips_what_still_cannot_fit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Each waiting deployment that didn't fit found every cycle of the topology 4-6 times a
    round, and their number grows quickly with the nodes (8 nodes, 3 such deployments: 606 ms
    a round). Now they are found once a round, and a deployment is tried again only when what
    it depends on changes."""
    calls = 0
    real = Topology.get_cycles

    def counted(self: Topology):  # noqa: ANN202
        nonlocal calls
        calls += 1
        return real(self)

    monkeypatch.setattr(Topology, "get_cycles", counted)
    state = cluster(deployed(instance_on(A)), A, B, C, D)
    for _ in range(2):
        extra = deployed(instance_on(A))
        state = state.model_copy(
            update={"deployments": {**state.deployments, **extra.deployments}}
        )
    # None of them fits: they need more memory than any node has
    big = MODEL.model_copy(update={"storage_size": Memory.from_gb(500)})
    state = state.model_copy(
        update={
            "instances": {},
            "deployments": {
                deployment_id: deployment.model_copy(update={"model_card": big})
                for deployment_id, deployment in state.deployments.items()
            },
        }
    )
    from datetime import datetime, timezone

    now = datetime.now(tz=timezone.utc)
    first = reconcile(state, now=now, settled=True, reachable={A, B, C, D})
    assert calls == 1
    assert len(first.unfit) == 3
    for event in first.events:
        state = event_apply(event, state)

    second = reconcile(
        state, now=now, settled=True, reachable={A, B, C, D}, unfit=first.unfit
    )

    assert calls == 1
    assert [e for e in second.events if isinstance(e, InstanceCreated)] == []
