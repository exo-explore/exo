"""Deployments on a real cluster: what a user launched keeps running through cuts, sleeps, a change of master and a full restart, and only the user
stops it.

The membership's timers these depend on: a lost node is Suspect after zenoh's 10 s lease (or the
2 s probe of a closed connection) and Gone after the 30 s grace; an even split, or the smaller
side, keeps a gone node listed, with its models, for the 5-minute split hold.
"""

import time
from collections.abc import Callable

import httpx
import pytest

from exo.api.main import CLUSTER_COMMAND_TIMEOUT
from exo.api.types import CreateInstanceParams
from exo.shared.types.common import DeploymentId, NodeId
from exo.shared.types.state import State
from exo.shared.types.worker.instances import InstanceId
from exo.shared.types.worker.runners import RunnerReady, RunnerRunning
from exo.testkit.cluster import STUB_MODEL, Cluster
from exo.testkit.tests.conftest import MakeCluster

pytestmark = [pytest.mark.cluster, pytest.mark.slow]


def serving(cluster: Cluster, via: int) -> list[frozenset[NodeId]]:
    """The node sets of the stub model's instances that are ready, in node `via`'s view."""
    state = cluster.state(via)
    if state is None:
        return []
    out: list[frozenset[NodeId]] = []
    for instance in state.instances.values():
        if instance.shard_assignments.model_id != STUB_MODEL:
            continue
        if all(
            isinstance(state.runners.get(runner), (RunnerReady, RunnerRunning))
            for runner in instance.shard_assignments.runner_to_shard
        ):
            out.append(frozenset(instance.shard_assignments.node_to_runner))
    return out


def placements_while(
    cluster: Cluster, via: int, done: Callable[[], bool], timeout: float, what: str
) -> list[frozenset[NodeId]]:
    """The node sets of every instance of the stub model seen in `via`'s view, in order, until
    `done()`; raises HarnessTimeoutError after `timeout`."""
    seen: dict[str, frozenset[NodeId]] = {}

    def look() -> bool:
        state = cluster.state(via)
        if state is not None:
            for instance_id, instance in state.instances.items():
                if instance.shard_assignments.model_id == STUB_MODEL:
                    seen.setdefault(
                        instance_id,
                        frozenset(instance.shard_assignments.node_to_runner),
                    )
        return done()

    cluster.wait_for(look, timeout=timeout, what=what)
    return list(seen.values())


def deployments(cluster: Cluster, via: int) -> set[DeploymentId]:
    state = cluster.state(via)
    return set() if state is None else set(state.deployments)


def deployment_of(cluster: Cluster, instance_id: InstanceId, via: int) -> DeploymentId:
    state = cluster.state(via)
    assert state is not None
    return state.instances[instance_id].deployment_id


def _ok(cluster: Cluster, prompt: str, via: int) -> None:
    result = cluster.chat(f"{prompt} [stub tokens=20]", via=via)
    assert result.ok and not result.problems(), result.summary()


# --- the model's devices are lost, or come back ------------------------------------------


def test_a_two_node_cut_moves_nothing_and_heals_to_one_copy(
    make_cluster: MakeCluster,
) -> None:
    """Two nodes lose each other for 45 s: each is Gone to the other, but each side is half of
    the cluster, and keeps the other listed, with its model, for the split hold. The side
    without the model doesn't load a copy, and after the heal the model runs exactly once, on
    its node, in both views. Bound 60 s for the heal."""
    cluster = make_cluster(2)
    original = cluster.launch(on=(1,))
    launched = deployments(cluster, 0)
    model_node = cluster.nodes[1].node_id

    cluster.cut(0, 1)
    time.sleep(45)
    # The other side's view keeps the model's instance, and has no other
    view = cluster.state(0)
    assert view is not None
    assert set(view.instances) == {original}, "the other side loaded a copy"

    cluster.heal()
    cluster.wait_until_consistent(timeout=60)
    cluster.wait_for(
        lambda: serving(cluster, via=0) == [frozenset({model_node})]
        and serving(cluster, via=1) == [frozenset({model_node})],
        timeout=60,
        what="the model to run once, on its node, in both views",
    )
    assert deployments(cluster, 0) == deployments(cluster, 1) == launched
    _ok(cluster, "after the heal", via=0)


def test_a_model_stopped_while_it_is_kept_does_not_start_when_its_node_returns(
    make_cluster: MakeCluster,
) -> None:
    """Two nodes; the model's node dies. The other keeps the model for it (half of the
    cluster: the split hold), and the user stops it there. When a node with its files comes
    back, nothing starts."""
    cluster = make_cluster(2)
    instance = cluster.launch(on=(1,))
    launched = deployment_of(cluster, instance, via=0)

    cluster.kill(1)
    time.sleep(5)
    assert instance in (cluster.state(0) or State()).instances
    assert launched in deployments(cluster, 0)

    response = httpx.delete(f"{cluster.nodes[0].url}/instance/{launched}", timeout=5)
    assert response.status_code == 200, response.text
    cluster.wait_for(
        lambda: launched not in deployments(cluster, 0),
        timeout=10,
        what="the stop to apply",
    )

    cluster.restart(1)
    cluster.wait_until_consistent(timeout=60)
    time.sleep(5)
    assert serving(cluster, via=0) == []
    assert deployments(cluster, 0) == set()


def test_a_model_launched_again_while_it_is_kept_runs_once(
    make_cluster: MakeCluster,
) -> None:
    """Two nodes; the model's node dies, and the other keeps the model for it. Its requests fail,
    so the user launches it again on the node left. That replaces the kept copy at once, so
    the model runs once, here, and nothing else is placed when the split hold ends."""
    cluster = make_cluster(2)
    kept = cluster.launch(on=(1,))
    launched = deployment_of(cluster, kept, via=0)

    cluster.kill(1)
    time.sleep(5)
    assert kept in (cluster.state(0) or State()).instances

    again = cluster.launch(on=(0,))
    assert set((cluster.state(0) or State()).instances) == {again}
    assert launched not in deployments(cluster, 0)
    _ok(cluster, "launched again", via=0)


def test_the_same_placement_posted_twice_runs_once(make_cluster: MakeCluster) -> None:
    """POST /instance twice with one preview (a double click, a client's retry). The second post
    is ignored at once; checked 10 s later."""
    cluster = make_cluster(3)
    master = cluster.wait_for_master(timeout=60)
    preview = cluster.preview((1,), via=master)
    body = CreateInstanceParams(instance=preview).model_dump_json()
    for _ in range(2):
        response = httpx.post(
            cluster.nodes[master].url + "/instance",
            content=body,
            headers={"content-type": "application/json"},
            timeout=10,
        )
        assert response.status_code == 200, response.text
        time.sleep(2)
    time.sleep(10)
    state = cluster.state(master)
    assert state is not None
    assert len(state.instances) == 1


def test_a_model_on_two_nodes_comes_back_on_both_after_one_sleeps(
    make_cluster: MakeCluster,
) -> None:
    """Two nodes, the model on both; the follower sleeps 25 s (Suspect to the master, within
    the grace) and wakes. Nothing is removed or placed elsewhere meanwhile: the model is never
    placed on the master's node alone, and serves on both again. Measured 18 s after the wake;
    bound 60 s."""
    cluster = make_cluster(2)
    master = cluster.wait_for_master(timeout=60)
    follower = 1 - master
    both = frozenset(node.node_id for node in cluster.nodes)
    cluster.launch(on=(0, 1), via=master)

    cluster.pause(follower)
    time.sleep(25)
    cluster.resume(follower)
    seen = placements_while(
        cluster,
        master,
        lambda: serving(cluster, via=master) == [both],
        timeout=60,
        what="the model to run on both nodes again",
    )
    # Right after the wake its runners' statuses are those from before the sleep, and the
    # master's node may not reach it yet: a chat ends at once (device_unreachable) until it does
    cluster.wait_serving(via=master, timeout=60)
    assert all(nodes == both for nodes in seen), f"placed on fewer nodes: {seen}"
    assert serving(cluster, via=master) == [both]
    _ok(cluster, "after the sleep", via=master)


def test_a_pair_whose_model_node_is_killed_and_restarted_serves_it_before_the_split_hold(
    make_cluster: MakeCluster,
) -> None:
    """kill -9 of the node running the model in a pair, restarted at once under a new id. The
    other node keeps the old id, with the model, while it reads half of the cluster. Once the
    new id counts it reads a majority, removes the old id when it is Gone (the lease or probe,
    then the 30 s grace), and places the model again, well before the 5-minute split hold.
    Measured 31 s after the kill; bound 90 s."""
    cluster = make_cluster(2)
    master = cluster.wait_for_master(timeout=60)
    follower = 1 - master
    original = cluster.launch(on=(follower,), via=master)
    old_id = cluster.nodes[follower].node_id
    cluster.kill(follower)
    cluster.restart(follower)

    def replaced() -> bool:
        state = cluster.state(master)
        return (
            state is not None
            and original not in state.instances
            and not state.topology.contains_node(old_id)
            and serving(cluster, via=master) != []
        )

    cluster.wait_for(replaced, timeout=90, what="the model placed again on the pair")
    cluster.wait_serving(via=master, timeout=30)


# --- a change of master, and a full restart ----------------------------------------------


def _launch_while_the_master_sleeps(cluster: Cluster) -> tuple[int, int, DeploymentId]:
    """Two nodes: the master sleeps (SIGSTOP), the other node takes over, and the user launches
    a model there. Returns the sleeping master, the other node and the deployment."""
    master = cluster.wait_for_master(timeout=60)
    other = 1 - master
    cluster.pause(master)
    cluster.wait_for_master(timeout=60, other_than=master)
    instance = cluster.launch(on=(other,), via=other)
    return master, other, deployment_of(cluster, instance, via=other)


def _runs_in_both_views(cluster: Cluster, launched: DeploymentId) -> None:
    cluster.wait_until_consistent(timeout=90)
    cluster.wait_for(
        lambda: launched in deployments(cluster, 0)
        and serving(cluster, via=0) != []
        and serving(cluster, via=1) != [],
        timeout=90,
        what="the launch to run in both views after the wake",
    )


def test_a_launch_made_while_the_master_slept_survives_its_return(
    make_cluster: MakeCluster,
) -> None:
    """The master wakes at once. If its older state wins the election the other node adopts it,
    and the launch is brought back: the API sends it again, and the node that has it tells the
    master that won. It comes back on the node it was launched on, even when the woken master
    won: the node says where the launch ran (on the other Mac in 1 of 2 runs without that).
    Bound 90 s after the wake."""
    cluster = make_cluster(2)
    master, other, launched = _launch_while_the_master_sleeps(cluster)

    cluster.resume(master)
    _runs_in_both_views(cluster, launched)
    launched_on = frozenset({cluster.nodes[other].node_id})
    assert serving(cluster, via=0) == serving(cluster, via=1) == [launched_on]
    _ok(cluster, "after the wake", via=0)


def test_a_launch_older_than_the_apis_resend_window_survives_a_master_that_slept(
    make_cluster: MakeCluster,
) -> None:
    """As above, but the master sleeps past the time the API keeps a launch to send it again
    (CLUSTER_COMMAND_TIMEOUT), so if it wins the election only the node's announcement can bring
    the launch back. Bound 90 s after the wake."""
    cluster = make_cluster(2)
    master, _, launched = _launch_while_the_master_sleeps(cluster)
    time.sleep(CLUSTER_COMMAND_TIMEOUT + 5)

    cluster.resume(master)
    _runs_in_both_views(cluster, launched)
    cluster.wait_serving(via=0, timeout=30)


def test_after_a_full_restart_a_model_waits_for_the_user(
    make_cluster: MakeCluster,
) -> None:
    """Every node stops and starts again, under new ids. Every node kept the model on disk: it
    comes back held, so nothing loads at start-up, and loads when the user resumes it."""
    cluster = make_cluster(2)
    instance = cluster.launch(on=(1,))
    launched = deployment_of(cluster, instance, via=0)
    # Each node saves what it knows within its keeper's SYNC_INTERVAL (1 s); a launch and a
    # power cut within that second lose the launch
    time.sleep(3)

    cluster.kill(0)
    cluster.kill(1)
    cluster.restart(0)
    cluster.restart(1)
    cluster.wait_until_consistent(timeout=90)

    def held() -> bool:
        state = cluster.state(0)
        deployment = None if state is None else state.deployments.get(launched)
        return (
            deployment is not None
            and deployment.hold is not None
            and deployment.hold.kind == "Restored"
        )

    cluster.wait_for(held, timeout=60, what="the model to come back held")
    time.sleep(3)
    assert serving(cluster, via=0) == [], "a model loaded by itself at start-up"

    response = httpx.post(
        f"{cluster.nodes[0].url}/deployments/{launched}/resume", timeout=5
    )
    assert response.status_code == 200, response.text
    cluster.wait_for(
        lambda: serving(cluster, via=0) != [],
        timeout=60,
        what="the model to load once resumed",
    )
    _ok(cluster, "after the restart", via=0)
