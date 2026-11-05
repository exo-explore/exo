# pyright: reportPrivateUsage=false
"""Cluster scenarios about which nodes are in the cluster: a node is removed only once its
membership says it is Gone (Suspect for the grace) and the quorum read allows it, and a request
or a launch that needs a device this node can't reach ends at once.

Bounds come from the membership's timers (rust/networking/src/membership.rs).
"""

import threading
import time

import httpx
import pytest
from pydantic import ValidationError

from exo.api.types import CreateInstanceParams
from exo.routing import membership
from exo.routing.membership import MembershipView, MemberState
from exo.shared.request_errors import DEVICE_UNREACHABLE
from exo.shared.types.common import NodeId
from exo.testkit.cluster import Cluster
from exo.testkit.tests.conftest import MakeCluster
from exo.testkit.tests.test_scenarios import (
    LONG,
    ScenarioSetupError,
    _instance_gone,
    _is_the_master,
    _ok,
    _roles,
    _streaming,
)

pytestmark = pytest.mark.cluster

# zenoh's lease, the probe of a node whose connection closed, how often a node reached only
# through another one is asked again, and the grace from Suspect to Gone
LEASE = 10.0
PROBE_TIMEOUT = 2.0
REPROBE_INTERVAL = LEASE
GRACE = membership.GRACE.total_seconds()
# What a request is told when a device serving its model can't be reached
UNREACHABLE = DEVICE_UNREACHABLE


def _member_state(cluster: Cluster, via: int, of: int) -> MemberState | None:
    """The state of node `of` in node `via`'s membership view (GET /membership)"""
    try:
        response = cluster.get(via, "/membership")
        view = MembershipView.model_validate_json(response.content)
    except (httpx.HTTPError, ValidationError):
        return None
    node_id = cluster.nodes[of].node_id
    return next(
        (member.state for member in view.members if member.node_id == node_id), None
    )


def _lists(cluster: Cluster, via: int, of: int) -> bool:
    state = cluster.state(via)
    return (
        state is not None and cluster.nodes[of].node_id in state.topology.list_nodes()
    )


def _quorum_read(cluster: Cluster, via: int) -> tuple[str, int] | None:
    """Node `via`'s quorum read and how many nodes it counts (GET /membership)"""
    try:
        view = MembershipView.model_validate_json(
            cluster.get(via, "/membership").content
        )
    except (httpx.HTTPError, ValidationError):
        return None
    return view.quorum, view.last_known


# --- a cut, a crash or a sleep -------------------------------------------------------------


@pytest.mark.slow
def test_a_follower_cut_off_for_40s_keeps_its_view_and_is_removed_after_the_grace(
    make_cluster: MakeCluster,
) -> None:
    """A follower holding half of an instance is cut off from both others (blackhole) for
    40 s.

    - The cut node sees both peers Suspect within the lease and the probe, and keeps its
      view: it reaches 1 of the 3 nodes it knows (a minority), so even as its own master it
      removes no node and deletes no model.
    - Its requests for the model fail at once, saying a device can't be reached; so do the
      master's, once the cut node is Suspect there.
    - The master removes the cut node, and deletes the instance, once it is Gone: GRACE
      after it was Suspect (bound: LEASE - 1 s + GRACE at the earliest, LEASE +
      PROBE_TIMEOUT + GRACE + 5 s at the latest, after the cut).
    - After the heal it rejoins, with no ghosts."""
    cluster = make_cluster(3)
    roles = _roles(cluster)
    a, b = roles.followers
    instance = cluster.launch(on=(a, b))
    cut_at = time.monotonic()
    cluster.isolate(b)
    cluster.wait_for(
        lambda: all(
            _member_state(cluster, b, peer) == "Suspect" for peer in (roles.master, a)
        ),
        timeout=LEASE + PROBE_TIMEOUT + 5,
        what="the cut node to see both peers Suspect",
    )
    cluster.note(
        f"the cut node sees both peers Suspect at +{time.monotonic() - cut_at:.1f}s"
    )
    result = cluster.chat("through the cut node", via=b, timeout=30)
    assert result.error == UNREACHABLE, result.summary()
    assert result.duration < 2, result.summary()

    cluster.wait_for(
        lambda: _member_state(cluster, roles.master, b) == "Suspect",
        timeout=LEASE + PROBE_TIMEOUT + 5,
        what="the master to see the cut node Suspect",
    )
    result = cluster.chat("through the master", via=roles.master, timeout=30)
    assert result.error == UNREACHABLE, result.summary()
    assert result.duration < 2, result.summary()

    cluster.wait_for(
        lambda: not _lists(cluster, roles.master, b),
        timeout=LEASE + PROBE_TIMEOUT + GRACE + 5,
        what="the master to remove the cut node",
    )
    removed_after = time.monotonic() - cut_at
    cluster.note(f"the master removed the cut node at +{removed_after:.1f}s")
    assert removed_after >= LEASE - 1 + GRACE, f"removed at +{removed_after:.1f}s"
    cluster.wait_for(
        lambda: _instance_gone(cluster, instance, via=roles.master),
        timeout=5,
        what="the master to delete the instance",
    )

    time.sleep(max(0.0, 40 - (time.monotonic() - cut_at)))
    cut_view = cluster.state(b)
    assert cut_view is not None
    assert {node.node_id for node in cluster.nodes} <= set(
        cut_view.topology.list_nodes()
    )
    assert instance in cut_view.instances, "the cut node deleted the model"
    cluster.heal()
    cluster.wait_until_consistent(timeout=60)


@pytest.mark.slow
def test_two_nodes_cut_for_20s_keep_their_shared_model(
    make_cluster: MakeCluster,
) -> None:
    """Two nodes sharing an instance lose each other for 20 s (the common home case).
    Each reaches exactly half of the cluster, and each makes itself master (elections are
    unchanged). Requests for the shared model fail at once on both sides, saying a device
    can't be reached; neither side removes the other or the model within the grace, and
    after the heal they merge back with the model kept. (Main removed the other node and
    deleted the model on both sides at about +16 s.)"""
    cluster = make_cluster(2)
    instance = cluster.launch(on=(0, 1))
    cut_at = time.monotonic()
    cluster.cut(0, 1)
    for via, other in ((0, 1), (1, 0)):
        cluster.wait_for(
            lambda via=via, other=other: _member_state(cluster, via, other)
            == "Suspect",
            timeout=LEASE + PROBE_TIMEOUT + 5,
            what=f"n{via} to see n{other} Suspect",
        )
        result = cluster.chat(f"through n{via} during the cut", via=via, timeout=30)
        assert result.error == UNREACHABLE, result.summary()
        assert result.duration < 2, result.summary()
    time.sleep(max(0.0, 20 - (time.monotonic() - cut_at)))
    for via in (0, 1):
        assert not _instance_gone(cluster, instance, via=via), (
            f"n{via} deleted the model"
        )
    cluster.heal()
    cluster.wait_until_consistent(timeout=30)
    assert not _instance_gone(cluster, instance, via=0), "the model was deleted"
    cluster.wait_ready(instance, timeout=60)
    _ok(cluster.chat("after the heal", via=1, timeout=30))


@pytest.mark.slow
def test_a_crashed_node_reached_through_a_third_is_removed(
    make_cluster: MakeCluster,
) -> None:
    """The direct link between the master and a follower is torn down (reset),
    so the master reaches the follower only through the third node; then the follower is
    killed. The third node, directly connected, sees it Suspect at once and Gone after the
    grace. The master must see it Gone too, and remove it: with removal driven by Gone alone
    (no silence timer behind it), a node it wrongly keeps Alive is never removed. It is
    asked again every REPROBE_INTERVAL, so it is Suspect within REPROBE_INTERVAL +
    PROBE_TIMEOUT of the kill. Bound: that plus GRACE plus 5 s."""
    cluster = make_cluster(3)
    roles = _roles(cluster)
    relay, victim = roles.followers
    cluster.cut(roles.master, victim, mode="reset")
    cluster.wait_for(
        lambda: _member_state(cluster, roles.master, victim) == "Alive",
        timeout=LEASE + PROBE_TIMEOUT + 5,
        what="the master to still reach the follower (through the third node) after the reset",
    )
    time.sleep(PROBE_TIMEOUT + 1)
    assert _member_state(cluster, roles.master, victim) == "Alive", (
        "the master lost the follower although the third node relays"
    )
    killed_at = time.monotonic()
    cluster.kill(victim)
    cluster.wait_for(
        lambda: _member_state(cluster, relay, victim) == "Gone",
        timeout=REPROBE_INTERVAL + PROBE_TIMEOUT + GRACE + 5,
        what="the third node (directly connected) to see the killed node Gone",
    )
    cluster.note(
        f"the third node sees it Gone at +{time.monotonic() - killed_at:.1f}s; the master "
        f"sees it {_member_state(cluster, roles.master, victim)}"
    )
    cluster.wait_for(
        lambda: not _lists(cluster, roles.master, victim),
        timeout=max(
            1.0,
            REPROBE_INTERVAL
            + PROBE_TIMEOUT
            + GRACE
            + 5
            - (time.monotonic() - killed_at),
        ),
        what="the master to remove the killed node",
    )


@pytest.mark.slow
def test_two_nodes_a_40s_sleep_keeps_the_shared_model(
    make_cluster: MakeCluster,
) -> None:
    """Two nodes share an instance, and the follower sleeps (SIGSTOP) for 40 s: past the lease
    and the grace, so the master sees it Gone, but far short of the split hold. In a pair
    the master keeps the node and the instance until the split hold, and its chats fail
    fast meanwhile; once the sleeper is back the model serves again. On real Macs the
    instance used to end at Gone, about 5 s before a follower asleep for 35 s was back
    (discovery holds off for 12 s after a stall)."""
    cluster = make_cluster(2)
    roles = _roles(cluster)
    (follower,) = roles.followers
    instance = cluster.launch(on=(roles.master, follower))
    paused_at = time.monotonic()
    cluster.pause(follower)
    cluster.wait_for(
        lambda: _member_state(cluster, roles.master, follower) == "Gone",
        timeout=LEASE + PROBE_TIMEOUT + GRACE + 10,
        what="the master to see the sleeper Gone",
    )
    cluster.note(f"the sleeper is Gone at +{time.monotonic() - paused_at:.1f}s")
    result = cluster.chat("during the sleep", via=roles.master, timeout=30)
    assert result.error == UNREACHABLE, result.summary()
    time.sleep(max(0.0, 45 - (time.monotonic() - paused_at)))
    assert not _instance_gone(cluster, instance, via=roles.master), (
        "the model was deleted"
    )
    assert _lists(cluster, roles.master, follower), "the sleeper was removed"
    cluster.resume(follower)
    cluster.wait_for(
        lambda: _member_state(cluster, roles.master, follower) == "Alive",
        timeout=40,
        what="the sleeper to be back",
    )
    cluster.wait_ready(instance, via=roles.master, timeout=60)
    _ok(cluster.chat("after the sleep", via=roles.master, timeout=30))
    assert not _instance_gone(cluster, instance, via=roles.master)


@pytest.mark.slow
def test_a_chat_through_a_cut_off_node_to_the_far_side_ends_at_suspect(
    make_cluster: MakeCluster,
) -> None:
    """Two nodes; the model runs on the master's side only, and a chat to it streams through
    the follower's API. The follower is cut off (blackhole, both ways): the chat's output can
    no longer reach its API. It must end when the follower sees the master's side Suspect
    (the lease, then the probe), saying a device can't be reached."""
    cluster = make_cluster(2)
    roles = _roles(cluster)
    (follower,) = roles.followers
    cluster.launch(on=(roles.master,))
    chat = _streaming(cluster, "through the cut-off node " + LONG, via=follower)
    cut_at = time.monotonic()
    cluster.isolate(follower)
    result = chat.result(60)
    cluster.note(f"the chat ended at +{result.ended - cut_at:.1f}s: {result.error}")
    assert result.error == UNREACHABLE, result.summary()
    assert result.ended - cut_at < LEASE + PROBE_TIMEOUT + 5, result.summary()
    cluster.heal()


# --- quick restarts, and losing the master --------------------------------------------


@pytest.mark.slow
def test_a_pair_whose_other_node_restarts_quickly_three_times_stays_a_majority(
    make_cluster: MakeCluster,
) -> None:
    """In a pair, the follower quits and comes back under a new id three times, each time
    quitting again about a second after the master sees it, before it ever reports; then it
    comes back and stays. A node that never reported isn't in the cluster's state and never
    counts, so the master never reads a minority (sampled every 0.25 s). (Counting every node
    seen, it read 2 of 6, a minority, for 5 min.) A quit is announced as a removal, which
    applies once the node is lost and this node reads a majority: alone with the lost one,
    the master reads Half (1 of 2) from the first quit until the last restart is listed, then
    a majority of 3 for at most a tick, removes the old id, and reads a majority of 2.
    Bound: GRACE + 20 s after the last restart."""
    cluster = make_cluster(2)
    roles = _roles(cluster)
    (follower,) = roles.followers
    reads: list[tuple[float, str, int]] = []
    sampling = threading.Event()

    def sample() -> None:
        started = time.monotonic()
        while not sampling.is_set():
            read = _quorum_read(cluster, roles.master)
            if read is not None:
                reads.append((time.monotonic() - started, *read))
            time.sleep(0.25)

    sampler = threading.Thread(target=sample, daemon=True)
    sampler.start()
    try:
        cluster.quit(follower)
        for _ in range(3):
            cluster.restart(follower)
            cluster.wait_for(
                lambda: _member_state(cluster, roles.master, follower) == "Alive",
                timeout=30,
                what="the master to see the restarted node",
            )
            time.sleep(1)
            if _lists(cluster, roles.master, follower):
                raise ScenarioSetupError("the restarted node reported before it quit")
            cluster.quit(follower)
        cluster.restart(follower)
        cluster.wait_for(
            lambda: _quorum_read(cluster, roles.master) == ("Majority", 2),
            timeout=GRACE + 20,
            what="the master to read a majority of 2",
        )
    finally:
        sampling.set()
        sampler.join(5)
    by_quorum = {
        quorum: sum(1 for read in reads if read[1] == quorum)
        for quorum in ("Majority", "Half", "Minority")
    }
    most_counted = max((read[2] for read in reads), default=0)
    cluster.note(
        f"{len(reads)} quorum reads sampled: {by_quorum}; at most {most_counted} counted"
    )
    assert reads, "no quorum read sampled"
    assert by_quorum["Minority"] == 0, [
        read for read in reads if read[1] == "Minority"
    ][:5]
    over_two = [read for read in reads if read[2] > 2]
    assert most_counted <= 3 and all(read[1] == "Majority" for read in over_two), (
        over_two[:5]
    )


@pytest.mark.slow
def test_a_cut_off_master_is_replaced_in_time_for_a_launch_sent_at_the_cut(
    make_cluster: MakeCluster,
) -> None:
    """The master is cut off from both followers (not crashed: it runs on and still calls
    itself master), and a launch goes in through a follower at the cut. The follower's API
    sends a cluster change again to a new master, and answers 503 (and calls the launch off)
    if the change doesn't show within CLUSTER_COMMAND_TIMEOUT (46 s). So the followers must
    elect a new master, and it must place the launch, well within that. Notes when both
    followers follow one of them, and the launch's answer and when it came. Bound: the
    launch is answered 200."""
    cluster = make_cluster(3)
    roles = _roles(cluster)
    on, via = roles.followers
    followers = {cluster.nodes[index].node_id for index in roles.followers}
    instance = cluster.preview((on,), via=via)
    body = CreateInstanceParams(instance=instance).model_dump_json()
    answers: list[tuple[float, int, str]] = []

    def launch(cut_at: float) -> None:
        try:
            response = httpx.post(
                cluster.nodes[via].url + "/instance",
                content=body,
                headers={"content-type": "application/json"},
                timeout=90,
            )
            answers.append(
                (time.monotonic() - cut_at, response.status_code, response.text[:300])
            )
        except httpx.HTTPError as error:
            answers.append((time.monotonic() - cut_at, 0, repr(error)))

    def new_master() -> NodeId | None:
        followed = {cluster.nodes[index].master() for index in roles.followers}
        if len(followed) != 1:
            return None
        (master,) = followed
        return master if master in followers else None

    _is_the_master(cluster, roles.master)
    cut_at = time.monotonic()
    cluster.isolate(roles.master)
    sender = threading.Thread(target=launch, args=(cut_at,), daemon=True)
    sender.start()
    cluster.wait_for_value(
        new_master, timeout=90, what="both followers to follow a new master"
    )
    cluster.note(
        f"both followers follow a new master at +{time.monotonic() - cut_at:.1f}s"
    )
    sender.join(100)
    assert answers, "the launch got no answer within 100 s"
    took, status, text = answers[0]
    cluster.note(
        f"the launch through a follower was answered {status} at +{took:.1f}s: {text}"
    )
    assert status == 200, (status, took, text)
