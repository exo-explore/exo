"""Cluster scenarios on real exo nodes (exo.testkit), with main's behaviour as the expectation.

Tests marked xfail are failures exo has today. Each says what goes wrong and what was
measured, in this harness and in chaos runs on a cluster of real Macs. The change that fixes
one removes its marker; strict means an unexpected pass fails the run, so a fix can't go
unnoticed.

No scenario assumes which node is the master: each asks the cluster (`_roles`), and one that
stops the master checks first that it is stopping the master.

Timing bounds are what these scenarios measured on main, with a margin, and say so.
The fast ones run in CI on every push (`pytest -m "cluster and not slow"`); `slow` ones run
nightly (`pytest src/exo/testkit -m ""` runs them all).
"""

import contextlib
import threading
import time
from collections.abc import Iterator
from dataclasses import dataclass

import pytest

from exo.shared.types.worker.instances import InstanceId
from exo.testkit.cluster import (
    EXPECTED_FAILURES,
    ChatHandle,
    ChatResult,
    Cluster,
    HarnessError,
    HarnessTimeoutError,
    NoPlacementError,
)
from exo.testkit.stub.config import StubConfig
from exo.testkit.tests.conftest import MakeCluster

pytestmark = pytest.mark.cluster

# A reply long enough to still be streaming whatever happens to the cluster (60 s)
LONG = "[stub tokens=3000 rate=50]"
# Seconds a running chat streams before the scenario's fault
STREAM_FIRST = 3.0


class ScenarioSetupError(Exception):
    """The scenario didn't get the setup it is about (not an expected failure of exo's)."""


@contextlib.contextmanager
def _setup(what: str) -> Iterator[None]:
    """The steps before a scenario's fault. Whatever goes wrong here, a harness timeout or
    a broken expectation included, is the setup not holding: never the failure an xfail
    expects."""
    try:
        yield
    except (HarnessError, AssertionError) as error:
        raise ScenarioSetupError(f"setup ({what}) failed: {error}") from error


class BeyondTheKnownFailureError(Exception):
    """A failure past the point a scenario's xfail is about (not the failure it expects)."""


@contextlib.contextmanager
def _beyond_the_known_failure(what: str) -> Iterator[None]:
    """Steps an xfail's known failure would have stopped before: once that failure is fixed,
    anything that goes wrong here is a new failure, never the expected one."""
    try:
        yield
    except (HarnessError, AssertionError) as error:
        raise BeyondTheKnownFailureError(f"{what} failed: {error}") from error


@dataclass(frozen=True)
class _Roles:
    master: int
    followers: tuple[int, ...]


def _roles(cluster: Cluster) -> _Roles:
    """The master every node follows, and the other nodes."""
    master = cluster.wait_for_master(timeout=30)
    return _Roles(
        master, tuple(index for index in range(len(cluster.nodes)) if index != master)
    )


def _is_the_master(cluster: Cluster, index: int) -> None:
    """Guard for a scenario about losing the master: it must be the master it stops."""
    if cluster.master() != index:
        raise ScenarioSetupError(
            f"n{index} isn't the master the scenario is about: {cluster.masters()}"
        )


def _ok(result: ChatResult) -> None:
    assert result.ok, result.summary()
    assert not result.problems(), result.problems()


def _clean_error(result: ChatResult) -> None:
    """The chat ended with an error, and every token it got before was right."""
    assert not result.ok, f"expected an error: {result.summary()}"
    assert result.error, result.summary()
    assert not result.problems(), result.problems()


def _streaming(cluster: Cluster, prompt: str, via: int) -> ChatHandle:
    handle = cluster.chat_in_background(prompt, via=via, timeout=150)
    time.sleep(STREAM_FIRST)
    if handle.done():
        raise ScenarioSetupError(
            f"the chat ended before the fault: {handle.result(1).summary()}"
        )
    return handle


def _instance_gone(cluster: Cluster, instance_id: InstanceId, via: int) -> bool:
    state = cluster.state(via)
    return state is not None and instance_id not in state.instances


def _instance_present(cluster: Cluster, instance_id: InstanceId, via: int) -> bool:
    state = cluster.state(via)
    return state is not None and instance_id in state.instances


def _runner_statuses(cluster: Cluster, instance: InstanceId, via: int) -> list[str]:
    state = cluster.state(via)
    if state is None or instance not in state.instances:
        return []
    runners = state.instances[instance].shard_assignments.runner_to_shard
    return [type(state.runners.get(runner)).__name__ for runner in runners]


# --- serving ---------------------------------------------------------------------------


def test_every_node_serves_every_token(make_cluster: MakeCluster) -> None:
    """A single-node and a two-node (pipeline, ring) instance serve concurrent chats through
    every node's API, streamed and not, and every token arrives, in order."""
    cluster = make_cluster(3)
    master = cluster.wait_for_master(timeout=10)
    assert cluster.master() == master
    assert set(cluster.masters().values()) == {cluster.nodes[master].node_id}
    cluster.launch(on=(1,))
    cluster.launch(on=(0, 2))
    handles = [
        cluster.chat_in_background(
            f"serve {via} {stream} [stub tokens=60 rate=200]", via=via, stream=stream
        )
        for via in range(3)
        for stream in (True, False)
    ]
    for handle in handles:
        _ok(handle.result(60))
    assert cluster.violations() == []


# --- the harness itself ----------------------------------------------------------------


def test_killing_a_node_leaves_its_semaphores_to_the_resource_tracker(
    make_cluster: MakeCluster,
) -> None:
    """The harness's kill -9 spares multiprocessing's resource tracker, which then unlinks
    the node's named semaphores. Killed with the node, it would leak them until the
    machine restarts; macOS allows 10,000, and a shared test machine ran out."""
    cluster = make_cluster(2)
    follower = _roles(cluster).followers[0]
    cluster.launch(on=(follower,))
    node = cluster.nodes[follower]
    cluster.kill(follower)
    cluster.wait_for(
        lambda: "leaked semaphore objects to clean up" in node.output_path.read_text(),
        timeout=15,
        what="the node's resource tracker to unlink its semaphores",
    )


@pytest.mark.slow
def test_two_clusters_on_one_machine_never_meet(make_cluster: MakeCluster) -> None:
    """Two clusters at once, each serving a two-node instance: their stub runners never
    share a ring port (a collision once paired runners of different clusters)."""
    clusters = [make_cluster(2), make_cluster(2)]
    for cluster in clusters:
        cluster.launch(on=(0, 1))
    handles = [
        cluster.chat_in_background(
            f"cluster {index} [stub tokens=100 rate=100]", via=index % 2
        )
        for index, cluster in enumerate(clusters)
    ]
    for handle in handles:
        _ok(handle.result(60))
    for cluster in clusters:
        assert cluster.violations() == []


# --- a follower dies -------------------------------------------------------------------


def test_killing_a_follower_ends_its_requests_with_a_clean_error(
    make_cluster: MakeCluster,
) -> None:
    """kill -9 of a follower holding half of a two-node instance. Main: the in-flight chat
    ends with an error (measured 3.6 s; bound 15 s: the supervisor checks every 5 s); the
    master removes the node and deletes the instance (measured 16 s; bound 35 s) and nothing
    places it again; a new chat for the model fails at once. On real Macs: removed after
    15-17 s, with clean errors."""
    cluster = make_cluster(3)
    roles = _roles(cluster)
    a, b = roles.followers
    instance = cluster.launch(on=(a, b))
    chat = _streaming(cluster, "follower dies " + LONG, via=roles.master)
    killed_at = time.monotonic()
    cluster.kill(b)

    result = chat.result(60)
    _clean_error(result)
    assert result.ended - killed_at < 15, result.summary()

    cluster.wait_until_consistent(timeout=35)
    assert _instance_gone(cluster, instance, via=roles.master)
    after = cluster.chat("after the loss", via=roles.master, timeout=10)
    assert after.status_code == 404, after.summary()


@pytest.mark.slow
@pytest.mark.xfail(
    strict=True,
    raises=EXPECTED_FAILURES,
    reason="A model that lost a node is never placed again (2 h of chaos on real Macs took "
    "48 manual relaunches)",
)
def test_a_model_that_lost_a_node_is_placed_again(make_cluster: MakeCluster) -> None:
    with _setup("a two-node instance on the followers"):
        cluster = make_cluster(3)
        roles = _roles(cluster)
        a, b = roles.followers
        cluster.launch(on=(a, b))
    cluster.kill(b)
    cluster.wait_for(
        lambda: cluster.chat("anyone serving?", via=roles.master, timeout=10).ok,
        timeout=90,
        what="the model to serve again on the nodes that are left",
    )


# --- the master dies -------------------------------------------------------------------


# A crashed master's replacement: its followers' links to it close at once, they ask whether
# it is still there for up to the membership's 2 s PROBE_TIMEOUT before it is Suspect, then
# elect a new master in the election's 3 s DEFAULT_ELECTION_TIMEOUT. About 5 s; twice that
# for a busy machine
CRASHED_MASTER_REPLACED_WITHIN = 10.0


@pytest.mark.slow
def test_killing_the_master_elects_a_new_one_and_fails_in_flight_chats(
    make_cluster: MakeCluster,
) -> None:
    """kill -9 of the master while a chat streams from a healthy instance on a follower.
    Main: the followers elect a new master (measured 5.5 s; bound 10 s; before #2586 usually
    only after the 30 s master-silence timeout, as in 2 of 3 master kills on real Macs), the
    chat fails when they do, and the dead master leaves every view when the new master
    times it out (measured 20.5 s after the kill; bound 45 s after the new master)."""
    cluster = make_cluster(3)
    roles = _roles(cluster)
    serving, asking = roles.followers
    cluster.launch(on=(serving,))
    chat = _streaming(cluster, "master dies " + LONG, via=asking)
    _is_the_master(cluster, roles.master)
    cluster.kill(roles.master)

    new_master = cluster.wait_for_master(
        timeout=CRASHED_MASTER_REPLACED_WITHIN, other_than=roles.master
    )
    assert new_master in roles.followers
    _clean_error(chat.result(60))
    cluster.wait_until_consistent(timeout=45)
    _ok(cluster.chat("after the new master", via=asking, timeout=30))


@pytest.mark.slow
def test_a_crashed_master_is_replaced_quickly(make_cluster: MakeCluster) -> None:
    """Three crashes of the master in a row, each replaced within 10 s. Before #2586 the
    followers usually (about 9 crashes in 10) waited out the 30 s master-silence timeout,
    as zenoh can keep a crashed router's liveliness token in a mesh of three or more; three
    crashes in a row make a pass by chance unlikely if that comes back."""
    with _setup("a cluster"):
        cluster = make_cluster(3)
    for _ in range(3):
        with _setup("finding the master"):
            master = _roles(cluster).master
        _is_the_master(cluster, master)
        cluster.kill(master)
        cluster.wait_for_master(
            timeout=CRASHED_MASTER_REPLACED_WITHIN, other_than=master
        )
        with _setup("the crashed node rejoining as a new one"):
            cluster.restart(master)
            cluster.wait_for_cluster(timeout=60)


@pytest.mark.xfail(
    strict=True,
    raises=EXPECTED_FAILURES,
    reason="Every change of master ends every in-flight request, a graceful quit "
    "included (6 of 6 failed on real Macs, where 16-22% of failed requests in a 2 h soak "
    "were on instances that lost no node)",
)
def test_a_master_change_keeps_chats_on_healthy_instances(
    make_cluster: MakeCluster,
) -> None:
    """The master quits while a chat streams from an instance on a follower, asked through
    the other follower."""
    with _setup("an instance on a follower, streaming"):
        cluster = make_cluster(3)
        roles = _roles(cluster)
        serving, asking = roles.followers
        cluster.launch(on=(serving,))
        chat = _streaming(cluster, "master quits [stub tokens=500 rate=50]", via=asking)
    _is_the_master(cluster, roles.master)
    cluster.quit(roles.master)
    _ok(chat.result(60))


@pytest.mark.slow
def test_quitting_the_master_elects_a_new_one_quickly(
    make_cluster: MakeCluster,
) -> None:
    """Main: a master that quits is replaced within seconds (measured 4.4 s; bound 15 s),
    and since #2582 it leaves every view at once; before, it stayed listed until it timed
    out (measured 19.7 s). Bound 40 s for the views to agree."""
    cluster = make_cluster(3)
    master = _roles(cluster).master
    _is_the_master(cluster, master)
    quit_at = time.monotonic()
    cluster.quit(master)
    cluster.wait_for_master(timeout=15, other_than=master)
    cluster.wait_until_consistent(timeout=40)
    assert time.monotonic() - quit_at < 60


@pytest.mark.slow
def test_a_launch_sent_while_the_master_is_unreachable_is_placed(
    make_cluster: MakeCluster,
) -> None:
    """The master is cut off (blackhole) and a follower's API takes a launch at once. The
    followers notice the master is gone only when zenoh's 10 s lease runs out, so the
    launch first goes to the cut-off master alone. Before #2581 it was lost: commands were
    sent once and answered "Command received." at once (2 of 2 such launches never appeared
    on real Macs). Since #2581 the API sends it again, also to the next
    master, and answers only once it is placed (within its 46 s CLUSTER_COMMAND_TIMEOUT)."""
    cluster = make_cluster(3)
    roles = _roles(cluster)
    follower = roles.followers[0]
    instance = cluster.preview((follower,), via=follower)
    _is_the_master(cluster, roles.master)
    cluster.isolate(roles.master)
    answered: list[BaseException | None] = []

    def launch() -> None:
        try:
            cluster.create(instance, via=follower)
            answered.append(None)
        except BaseException as error:
            answered.append(error)

    sender = threading.Thread(target=launch, daemon=True)
    sender.start()
    cluster.wait_for(
        lambda: _instance_present(cluster, instance.instance_id, via=follower),
        timeout=90,
        what="the launch to be placed by the next master",
    )
    sender.join(70)
    assert answered == [None], f"the launch was answered with an error: {answered}"


# --- a follower quits ------------------------------------------------------------------


@pytest.mark.slow
def test_quitting_a_follower_ends_its_requests_and_removes_it(
    make_cluster: MakeCluster,
) -> None:
    """A follower holding half of a two-node instance quits (SIGINT). Main: it exits in a few
    seconds; the in-flight chat and a chat sent after it left both end with an error
    (measured 8.6 s; bound 20 s); the node leaves every view (at once since #2582, 14.5 s
    before; bound 35 s)."""
    cluster = make_cluster(3)
    roles = _roles(cluster)
    a, b = roles.followers
    cluster.launch(on=(a, b))
    chat = _streaming(cluster, "follower quits " + LONG, via=roles.master)
    quit_at = time.monotonic()
    assert cluster.quit(b) < 15
    late = cluster.chat_in_background(
        "sent after the quit " + LONG, via=roles.master, timeout=60
    )
    for result in (chat.result(60), late.result(60)):
        _clean_error(result)
        assert result.ended - quit_at < 20, result.summary()
    cluster.wait_until_consistent(timeout=35)


@pytest.mark.slow
def test_a_node_that_quits_with_an_instance_leaves_no_task_behind(
    make_cluster: MakeCluster,
) -> None:
    """A follower quits while its runner is generating, and a chat arrives just after. The
    master removes the quitting node, and deletes its instances, while the node is still
    shutting down; its worker then plans bookkeeping tasks for its runner (a Shutdown, a
    CancelTask for the request it was serving). Between #2582 and #2587 these could reach
    the state and stay Pending in every node's state for good (1 run in 5); since #2587 a
    task for an instance that is gone is never added."""
    with _setup("a two-node instance on the followers, streaming"):
        cluster = make_cluster(3)
        roles = _roles(cluster)
        a, b = roles.followers
        cluster.launch(on=(a, b))
        chat = _streaming(cluster, "quits while serving " + LONG, via=roles.master)
    cluster.quit(b)
    late = cluster.chat_in_background(
        "sent after the quit " + LONG, via=roles.master, timeout=60
    )
    chat.result(60)
    late.result(60)
    cluster.wait_until_consistent(timeout=20)


@pytest.mark.slow
def test_a_node_that_quits_leaves_every_view_at_once(make_cluster: MakeCluster) -> None:
    """A follower quits (SIGINT). Since #2582 a quitting node says it is leaving, and every
    other node drops it at once (bound 5 s after it exited). Before, nothing said so: it
    stayed listed, with its models, for the 15 s silence timeout."""
    cluster = make_cluster(3)
    cluster.quit(_roles(cluster).followers[0])
    cluster.wait_until_consistent(timeout=5)


# --- a follower sleeps -----------------------------------------------------------------


def _pause_holding_half_an_instance(
    cluster: Cluster, seconds: float
) -> tuple[InstanceId, ChatResult, int]:
    """A follower holding half of a two-node instance sleeps (SIGSTOP) for `seconds` while a
    chat streams through the master. Returns the instance, the chat and the master."""
    with _setup("a two-node instance on the followers, streaming"):
        roles = _roles(cluster)
        a, b = roles.followers
        instance = cluster.launch(on=(a, b))
        chat = _streaming(cluster, f"sleeps {seconds:g}s " + LONG, via=roles.master)
    cluster.pause(b)
    time.sleep(seconds)
    cluster.resume(b)
    return instance, chat.result(120), roles.master


@pytest.mark.xfail(
    strict=True,
    raises=EXPECTED_FAILURES,
    reason="A pause longer than zenoh's 10 s lease removes the node and deletes its models "
    "(after a 12 s pause, removed at +16 s here; on real Macs a 10 s sleep failed 13 of 19 "
    "requests)",
)
def test_a_12s_pause_keeps_the_model(make_cluster: MakeCluster) -> None:
    """A short sleep: 12 s, clear of zenoh's 10 s lease (at exactly 10 s the outcome depends
    on which timer fires first)."""
    with _setup("a cluster"):
        cluster = make_cluster(3)
    instance, _, master = _pause_holding_half_an_instance(cluster, 12)
    time.sleep(20)
    assert _instance_present(cluster, instance, via=master), "the model was deleted"
    _ok(cluster.chat("after the pause", via=master, timeout=30))


@pytest.mark.slow
def test_a_40s_pause_removes_the_node_and_it_rejoins(make_cluster: MakeCluster) -> None:
    """SIGSTOP of a follower holding half of an instance for 40 s. Main: the in-flight chat
    fails, the node is removed and its instance deleted (~16 s); on waking it briefly runs
    as its own master, then merges back (measured +56 s; bound 60 s after waking) with no
    ghosts. On real Macs, 60 s and 300 s sleeps: removed at 17-18 s, merged back in 17-29 s,
    once after 72 s as its own master."""
    cluster = make_cluster(3)
    instance, result, master = _pause_holding_half_an_instance(cluster, 40)
    _clean_error(result)
    cluster.wait_until_consistent(timeout=60)
    assert _instance_gone(cluster, instance, via=master)


# --- runner faults ---------------------------------------------------------------------


@pytest.mark.parametrize("rank", [0, 1])
def test_a_runner_that_crashes_mid_reply_ends_it_and_its_instance_recovers(
    make_cluster: MakeCluster, rank: int
) -> None:
    """One rank of a two-node instance dies (SIGKILL, as a segfault or the OS would) before
    token 10. Main: the chat ends with an error (measured 0.4-0.5 s after it started; bound
    15 s) and the workers rebuild the instance's runners, which serve again (measured
    4.4-5.5 s after the error; bound 30 s). Right after the crash the nodes' state can still
    say the old runners are ready, so the test waits for a chat that succeeds rather than
    for ready runners."""
    with _setup("a two-node instance"):
        cluster = make_cluster(3)
        cluster.launch(on=(1, 2))
    started = time.monotonic()
    result = cluster.chat(
        f"crash {rank} [stub tokens=200 rate=50 fault=crash at=10 rank={rank}]",
        via=0,
        timeout=60,
    )
    _clean_error(result)
    assert result.ended - started < 15, result.summary()
    _ok(cluster.wait_serving(via=0, timeout=30))


@pytest.mark.slow
def test_a_rank_crash_with_slow_runner_starts_leaves_its_instance_serving(
    make_cluster: MakeCluster,
) -> None:
    """Rank 0 crashes mid-reply, and runners take 8 s to start (longer than the 3 s shutdown
    grace). Before #2594, each node could stop its fresh replacement while the other node's
    runner still showed Failed, and stopping it showed it Failed in turn: the nodes
    ping-ponged until the worker gave up and deleted the instance (3 of 5 runs under heavy
    load). Now a runner that hasn't joined anyone is left to start: serving again 12.6-12.8 s
    after the error; bound 60 s."""
    with _setup("a two-node instance with slow runner starts"):
        cluster = make_cluster(3)
        instance = cluster.launch(on=(1, 2))
        # Above worker.main.RUNNER_SHUTDOWN_GRACE (3 s): a replacement still starting can't
        # stop cleanly when it is told to, and is killed (at exactly 3 s it is a race)
        cluster.configure_stub(StubConfig(start_seconds=8))
    result = cluster.chat(
        "slow starts [stub tokens=200 rate=50 fault=crash at=10 rank=0]",
        via=0,
        timeout=60,
    )
    _clean_error(result)
    try:
        _ok(cluster.wait_serving(via=0, timeout=60))
    except HarnessTimeoutError as error:
        gone = _instance_gone(cluster, instance, via=0)
        raise AssertionError(f"{error}; instance deleted: {gone}") from error


@pytest.mark.slow
def test_a_runner_that_raises_mid_reply_ends_it_and_its_instance_recovers(
    make_cluster: MakeCluster,
) -> None:
    """Rank 0 of a two-node instance raises mid-reply; rank 1 then loses its ring and raises
    too, as MLX does. Main: the chat ends with an error and the workers rebuild the
    instance, which serves again (measured 2.6-3.2 s after the error; bound 30 s; about 6 s
    with MLX on real hardware, where each node starts its replacement twice)."""
    cluster = make_cluster(3)
    cluster.launch(on=(1, 2))
    result = cluster.chat(
        "raise [stub tokens=200 rate=50 fault=raise at=10 rank=0]", via=0, timeout=60
    )
    _clean_error(result)
    _ok(cluster.wait_serving(via=0, timeout=30))
    cluster.wait_until_consistent(timeout=30)


@pytest.mark.slow
def test_a_runner_that_fails_to_connect_once_leaves_its_instance_serving(
    make_cluster: MakeCluster,
) -> None:
    """Rank 1 raises while it connects and rank 0 waits in ring init (seen with MLX 0.32.4 on
    real hardware). Before #2594, rank 1's replacement paired with rank 0's old process,
    which was shutting down: rank 0 ended Idle and rank 1 Connected for good, chats got 503
    and a ConnectToGroup task stayed Running. Now a replacement starts only once the runners
    that had joined the old one are going: measured ready 7.8-8.0 s after the launch on a
    quiet machine, 10-30 s on a loaded one (12-28 s on real hardware); bound 90 s."""
    with _setup("a placement whose second rank fails to connect once"):
        cluster = make_cluster(3)
        cluster.configure_stub(
            StubConfig(
                fault_phase="connect", fault="raise", fault_rank=1, fault_limit=1
            )
        )
        instance = cluster.place(on=(1, 2))
    try:
        cluster.wait_ready(instance, timeout=90)
    except HarnessTimeoutError as error:
        raise AssertionError(
            f"{error}; runners {_runner_statuses(cluster, instance, 0)}"
        ) from error
    _ok(cluster.chat("after the failed connect", via=0, timeout=30))
    cluster.wait_until_consistent(timeout=30)


# --- restarts --------------------------------------------------------------------------


def test_a_restarted_node_leaves_no_ghost_and_no_instance(
    make_cluster: MakeCluster,
) -> None:
    """kill -9 of a follower holding an instance, restarted at once: exo gives it a new id,
    so it comes back as a new device. Main: the old id lingers as a ghost (measured 15.6 s)
    and its instance is deleted when it times out (16.1 s); bound 45 s. On real Macs the
    ghost lasted 7-11 s."""
    cluster = make_cluster(3)
    roles = _roles(cluster)
    node = roles.followers[-1]
    instance = cluster.launch(on=(node,))
    old_id = cluster.nodes[node].node_id
    cluster.kill(node)
    cluster.restart(node)
    assert cluster.nodes[node].node_id != old_id
    cluster.wait_until_consistent(timeout=45)
    assert _instance_gone(cluster, instance, via=roles.master)


# --- races -----------------------------------------------------------------------------


def test_a_chat_sent_as_its_instance_is_deleted_never_hangs(
    make_cluster: MakeCluster,
) -> None:
    """DELETE /instance and, 0-12 ms later, a chat for the same model, through a follower whose
    view still holds the instance. Before #2560 some of these hung forever (4 of 12 trials
    on real Macs); each must now end within 15 s."""
    cluster = make_cluster(2)
    follower = _roles(cluster).followers[0]
    for trial in range(12):
        instance = cluster.launch(on=(follower,), via=follower)
        results: list[ChatResult] = []

        def chat(trial: int, results: list[ChatResult]) -> None:
            results.append(cluster.chat(f"race {trial}", via=follower, timeout=15))

        sender = threading.Timer(trial / 1000, chat, args=(trial, results))
        sender.start()
        assert cluster.delete(instance, via=follower).status_code == 200
        sender.join(20)
        assert results, f"trial {trial}: the chat didn't return"
        (result,) = results
        assert not result.timed_out, f"trial {trial}: {result.summary()}"
        assert not result.problems(), result.problems()
        cluster.wait_for(
            lambda deleted=instance: _instance_gone(cluster, deleted, via=follower),
            timeout=10,
            what="the deletion",
        )


# --- network cuts ----------------------------------------------------------------------


@pytest.mark.slow
def test_a_cut_node_crowns_itself_and_rejoins_after_the_heal(
    make_cluster: MakeCluster,
) -> None:
    """A follower holding half of an instance is cut off from both others (blackhole) for
    20 s. Main: the cut node makes itself master of a one-node cluster (measured +13.4 s;
    +13.6-13.9 s with real network cuts between Macs), the majority removes it and deletes
    the instance (~16 s); after the heal it merges back with one master and no ghosts
    (measured 3.5 s; bound 30 s)."""
    cluster = make_cluster(3)
    roles = _roles(cluster)
    a, b = roles.followers
    instance = cluster.launch(on=(a, b))
    chat = _streaming(cluster, "cut off " + LONG, via=roles.master)
    cluster.isolate(b)
    cluster.wait_for(
        lambda: cluster.nodes[b].master() == cluster.nodes[b].node_id,
        timeout=25,
        what="the cut node to make itself master",
    )
    _clean_error(chat.result(60))
    time.sleep(max(0.0, 20 - STREAM_FIRST))
    cluster.heal()
    cluster.wait_until_consistent(timeout=30)
    assert _instance_gone(cluster, instance, via=roles.master)


@pytest.mark.slow
@pytest.mark.xfail(
    strict=True,
    raises=EXPECTED_FAILURES,
    reason="A node cut off for longer than about 11 s loses its models, and the cut node "
    "deletes every model in its own view (a 20 s cut deleted them at +16 s here and with "
    "real network cuts between Macs)",
)
def test_a_20s_cut_keeps_the_model(make_cluster: MakeCluster) -> None:
    with _setup("a model on the two followers"):
        cluster = make_cluster(3)
        roles = _roles(cluster)
        a, b = roles.followers
        instance = cluster.launch(on=(a, b))
    cluster.isolate(b)
    time.sleep(20)
    cluster.heal()
    cluster.wait_until_consistent(timeout=30)
    assert _instance_present(cluster, instance, via=roles.master), (
        "the model was deleted"
    )


# --- configuration ---------------------------------------------------------------------


@pytest.mark.slow
@pytest.mark.xfail(
    strict=True,
    # Only the refused placement: a placement that then fails to serve is another failure
    # (raised as BeyondTheKnownFailureError)
    raises=AssertionError,
    reason="Found by this harness: each node probes its peers' /node_id at its own "
    "--api-port, so nodes started with different API ports never get a topology edge, and "
    "every placement across them is refused",
)
def test_nodes_with_different_api_ports_can_run_a_model_together(
    make_cluster: MakeCluster,
) -> None:
    with _setup("two nodes, each with its own API port"):
        cluster = make_cluster(2, probe_own_port=True, require_edges=False)
    try:
        cluster.preview(on=(0, 1))
    except NoPlacementError as error:
        # No edges between the two, so no ring placement across them
        raise AssertionError(str(error)) from error
    with _beyond_the_known_failure("serving across the two once placed"):
        cluster.launch(on=(0, 1), timeout=30)
        _ok(cluster.chat("across two API ports", via=0, timeout=30))
