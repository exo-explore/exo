"""Serving without the master, on real exo nodes (exo.testkit).

A chat's request goes from its API node to its instance's output node over HTTP and its tokens
come back on the same response, so nothing that happens to the master reaches it; killing a node
of the instance ends it with a clean error at once or within the supervisor's check; a request
that some rank didn't take leaves nothing behind on any rank; concurrent chats and cancels work
while the master is gone; a pause of the API node doesn't end its chat; and a cut between
the API node and the output node ends the chat cleanly. Every token of every reply is checked (ChatResult.problems()). Bounds are measured
values plus a stated margin.
"""

import time
from typing import cast

import httpx
import pytest

from exo.shared.types.common import NodeId
from exo.shared.types.worker.instances import InstanceId
from exo.testkit.cluster import STUB_MODEL, ChatHandle, ChatResult, Cluster
from exo.testkit.node import REFUSE_DELIVERIES_ENV, WEDGE_SECONDS_ENV
from exo.testkit.tests.conftest import MakeCluster

pytestmark = pytest.mark.cluster


# A reply still streaming after each scenario's fault: 1200 tokens at 50/s is 24 s. The stub's
# rate depends on sleep accuracy, which macOS stretches under load (sleep(20 ms) measured
# 127 ms on the test machine, so 13 tok/s), so timeouts allow for 4x that and bounds below are
# on gaps between tokens, which the slack doesn't stretch past 0.2 s.
LONG = "[stub tokens=1200 rate=50]"
STREAM_FIRST = 3.0
REPLY_TIMEOUT = 300


def _ok(result: ChatResult) -> None:
    assert result.ok, result.summary()
    assert not result.problems(), result.problems()


def _clean_error(result: ChatResult) -> None:
    assert not result.ok, f"expected an error: {result.summary()}"
    assert result.error, result.summary()
    assert not result.problems(), result.problems()


def _streaming(cluster: Cluster, prompt: str, via: int) -> ChatHandle:
    handle = cluster.chat_in_background(prompt, via=via, timeout=REPLY_TIMEOUT)
    time.sleep(STREAM_FIRST)
    assert not handle.done(), "the chat ended before the fault"
    return handle


def _roles(cluster: Cluster) -> tuple[int, int, int]:
    """The master and the two other nodes, as the cluster elected them: never assume which
    node is the master (a scenario that faults the wrong node passes on any build)"""
    master = cluster.wait_for_master(timeout=60)
    first, second = (index for index in range(3) if index != master)
    return master, first, second


def _is_the_master(cluster: Cluster, master: int) -> None:
    assert cluster.wait_for_master(timeout=30) == master, (
        "the master changed before the fault"
    )


def _node_index(cluster: Cluster, node_id: NodeId) -> int:
    return next(i for i, node in enumerate(cluster.nodes) if node.node_id == node_id)


def _ranks(cluster: Cluster, instance_id: InstanceId, via: int) -> list[int]:
    """The instance's nodes by device rank"""
    state = cluster.state(via)
    assert state is not None
    assignments = state.instances[instance_id].shard_assignments
    by_rank = {
        assignments.runner_to_shard[runner].device_rank: node
        for node, runner in assignments.node_to_runner.items()
    }
    return [_node_index(cluster, by_rank[rank]) for rank in sorted(by_rank)]


def _through_the_network(cluster: Cluster, source: int, sink: int) -> None:
    """A short chat from `source` to an instance on `sink`, which must have gone through the
    pair's proxy in the simulated network: otherwise a cut between them tests nothing"""
    pair = (cluster.nodes[source].name, cluster.nodes[sink].name)
    before = cluster.network.api_connections.get(pair, 0)
    result = cluster.chat("routed [stub tokens=5]", via=source, timeout=30)
    assert result.ok and cluster.network.api_connections.get(pair, 0) > before, (
        f"{pair[0]}'s request to {pair[1]} didn't go through the simulated network "
        f"({result.summary()}), so a cut between them wouldn't reach it: exo.testkit.node "
        "routes exo.shared.serving.node_urls through the pair's proxy"
    )


def _held_requests(cluster: Cluster, index: int) -> list[str]:
    """The requests the node's runners hold (exo.worker.serving)"""
    served = cast(dict[str, object], cluster.get(index, "/_exo/serving").json())
    runners = cast(dict[str, list[str]], served["runners"])
    return [task for tasks in runners.values() for task in tasks]


# --- (a) the master is not on a chat's path ---------------------------------------------


def test_a_chat_streams_through_a_master_kill(make_cluster: MakeCluster) -> None:
    """kill -9 of the master mid-reply. Main: the chat fails (the new master starts without its
    task; CLUSTER #2). Branch: every token arrives exactly once, in order, with no gap longer
    than 2 s (a token is due every 20 ms)."""
    cluster = make_cluster(3)
    master, serving, asking = _roles(cluster)
    cluster.launch(on=(serving,), via=asking)
    chat = _streaming(cluster, "master killed " + LONG, via=asking)
    _is_the_master(cluster, master)
    cluster.kill(master)
    result = chat.result(REPLY_TIMEOUT)
    _ok(result)
    assert result.longest_gap < 2.0, result.summary()


def test_a_chat_streams_through_a_master_freeze_and_the_election(
    make_cluster: MakeCluster,
) -> None:
    """SIGSTOP of the master for 20 s (past the 10 s lease, so the others elect a new master),
    then SIGCONT: the old master is demoted. Main: tokens stop with the master, and the change
    of master ends the chat. Branch: it streams on, every token, gaps under 2 s."""
    cluster = make_cluster(3)
    master, serving, asking = _roles(cluster)
    cluster.launch(on=(serving,), via=asking)
    chat = _streaming(cluster, "master frozen " + LONG, via=asking)
    _is_the_master(cluster, master)
    cluster.pause(master)
    time.sleep(20)
    cluster.resume(master)
    result = chat.result(REPLY_TIMEOUT)
    _ok(result)
    assert result.longest_gap < 2.0, result.summary()


def test_a_short_master_freeze_doesnt_pause_a_chat(make_cluster: MakeCluster) -> None:
    """SIGSTOP of the master for 8 s, below the lease (no election). Main: every token waits
    for the master, so the reply pauses for the whole freeze (8 s; B1's real-host run: 9.02 s).
    Branch: no gap over 1 s."""
    cluster = make_cluster(3)
    master, serving, asking = _roles(cluster)
    cluster.launch(on=(serving,), via=asking)
    chat = _streaming(cluster, "master paused " + LONG, via=asking)
    _is_the_master(cluster, master)
    cluster.pause(master)
    time.sleep(8)
    cluster.resume(master)
    result = chat.result(REPLY_TIMEOUT)
    _ok(result)
    assert result.longest_gap < 1.0, result.summary()


# --- (b) a rank dies ------------------------------------------------------------------------


def test_killing_the_output_rank_ends_its_chat_at_once(
    make_cluster: MakeCluster,
) -> None:
    """kill -9 of the node holding rank 0 of a two-node instance, while a chat streams through
    the third node's API. Branch: the stream's connection breaks, so the chat ends with a clean
    error within 2 s (measured well under 1 s). Main: it ends when the surviving rank's
    supervisor notices the ring broke (checks every 5 s) or the master removes the node (15 s)."""
    cluster = make_cluster(3)
    master, first, second = _roles(cluster)
    instance = cluster.launch(on=(first, second), via=master)
    rank0, _ = _ranks(cluster, instance, via=master)
    chat = _streaming(cluster, "rank 0 killed " + LONG, via=master)
    killed_at = time.monotonic()
    cluster.kill(rank0)
    result = chat.result(REPLY_TIMEOUT)
    _clean_error(result)
    assert result.ended - killed_at < 2.0, result.summary()


def test_killing_the_other_rank_ends_its_chat_quickly(
    make_cluster: MakeCluster,
) -> None:
    """kill -9 of the node holding rank 1. The output node's runner loses its ring; its
    supervisor (checking every 5 s) fails it, which ends the chat. Bound 10 s: the 5 s check
    plus margin. Main measured 3.6 s for the same fault (the existing scenario's bound: 15 s)."""
    cluster = make_cluster(3)
    master, first, second = _roles(cluster)
    instance = cluster.launch(on=(first, second), via=master)
    _, rank1 = _ranks(cluster, instance, via=master)
    chat = _streaming(cluster, "rank 1 killed " + LONG, via=master)
    killed_at = time.monotonic()
    cluster.kill(rank1)
    result = chat.result(REPLY_TIMEOUT)
    _clean_error(result)
    assert result.ended - killed_at < 10.0, result.summary()


# --- (c) a request some rank didn't take ----------------------------------------------------


def test_a_request_a_frozen_rank_didnt_take_leaves_no_task_on_any_rank(
    make_cluster: MakeCluster, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Fault injection (no counterpart on main, where the log delivers every task to every
    rank): with the wedge bound shortened to 4 s, a rank of a three-node instance is frozen as
    a chat arrives, for 9 s (below the master's 15 s removal and the 10 s lease). The other
    ranks take the request, the frozen one doesn't in time, so it is refused (503). When the
    frozen rank wakes, it takes the task and its cancellation, and every rank drops it: no rank
    holds a task, and the instance serves the next chat."""
    monkeypatch.setenv(WEDGE_SECONDS_ENV, "4")
    cluster = make_cluster(3)
    master, _, _ = _roles(cluster)
    instance = cluster.launch(on=(0, 1, 2), via=master)
    output, *others = _ranks(cluster, instance, via=master)
    # Not the master, if it can be helped: freezing the master tests something else too
    frozen = next((index for index in others if index != master), others[-1])
    _ok(cluster.chat("warm [stub tokens=20]", via=output, timeout=30))

    cluster.pause(frozen)
    started = time.monotonic()
    refused = cluster.chat("while frozen [stub tokens=50]", via=output, timeout=30)
    assert refused.status_code == 503, refused.summary()
    # The wedge bound (4 s) plus the slack of a loaded machine's timers
    assert time.monotonic() - started < 8, refused.summary()
    held_meanwhile = _held_requests(cluster, output)
    time.sleep(max(0.0, 9 - (time.monotonic() - started)))
    cluster.resume(frozen)

    assert held_meanwhile, "the ranks that took it held it while the frozen rank slept"
    cluster.wait_for(
        lambda: all(not _held_requests(cluster, index) for index in (0, 1, 2)),
        timeout=20,
        what="every rank to drop the request",
    )
    _ok(cluster.chat("after the wake [stub tokens=40]", via=output, timeout=30))


def test_a_rank_that_refuses_every_delivery_gets_its_instance_restarted(
    make_cluster: MakeCluster, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Fault injection (no counterpart on main): the ranks answer every delivery with 403, as
    a member check that doesn't know the sender would, and a client sends a chat every
    second. Each chat is refused at once (503), the output node's runner holds at most the
    first request (the instance refuses new requests while that one's cancellation is
    outstanding, without delivering them), and after CANCEL_REFUSED_LIMIT (30 s) the output
    node fails its runner, so the instance's runners start again. Before: each chat waited 30 s
    for a 503 saying the model was being started again, and the runner piled up a request
    per 0.2 s resend; then, one held request and one retry loop per chat until the stall
    watchdog's 90 s."""
    monkeypatch.setenv(REFUSE_DELIVERIES_ENV, "1")
    cluster = make_cluster(3)
    master, first, second = _roles(cluster)
    instance = cluster.launch(on=(first, second), via=master)
    output, _ = _ranks(cluster, instance, via=master)
    started = time.monotonic()
    held_most = 0
    for index in range(8):
        result = cluster.chat(
            f"refused {index} [stub tokens=20]", via=output, timeout=30
        )
        assert result.status_code == 503, result.summary()
        # A loaded machine's slack; the refusal itself takes milliseconds
        assert result.duration < 2.0, result.summary()
        held_most = max(held_most, len(_held_requests(cluster, output)))
        time.sleep(1.0)
    assert held_most <= 1, f"the output runner held {held_most} requests"
    cluster.wait_for(
        lambda: "Failing this device's runner" in cluster.nodes[output].log(),
        timeout=max(1.0, 45 - (time.monotonic() - started)),
        what="the output node to fail its runner",
    )
    # CANCEL_REFUSED_LIMIT (30 s) from the first chat, plus a retry's spacing and the slack
    # of a loaded machine
    assert time.monotonic() - started < 45


# --- (d) concurrency and cancel -------------------------------------------------------------


def test_concurrent_chats_and_a_cancel_while_the_master_is_dead(
    make_cluster: MakeCluster,
) -> None:
    """Two instances, eight concurrent chats through two API nodes, and the master killed while
    they stream. Main: every one of them fails with the change of master. Branch: every reply
    complete, each token exactly once and in order. Then a chat cancelled by its client (the
    connection closed) stops on every rank (bound 10 s: see below)."""
    cluster = make_cluster(3)
    master, first, second = _roles(cluster)
    single = cluster.launch(on=(first,), via=first)
    pair = cluster.launch(on=(first, second), via=first)
    assert single != pair
    chats = [
        cluster.chat_in_background(
            f"concurrent {index} [stub tokens=400 rate=50]",
            via=(first, second)[index % 2],
            timeout=150,
        )
        for index in range(8)
    ]
    time.sleep(STREAM_FIRST)
    _is_the_master(cluster, master)
    cluster.kill(master)
    for chat in chats:
        _ok(chat.result(REPLY_TIMEOUT))

    node = cluster.nodes[first]
    with (
        httpx.Client(timeout=30) as client,
        client.stream(
            "POST",
            f"{node.url}/v1/chat/completions",
            json={
                "model": STUB_MODEL,
                "messages": [{"role": "user", "content": "to cancel " + LONG}],
                "stream": True,
            },
        ) as response,
    ):
        assert response.status_code == 200
        lines = 0
        for line in response.iter_lines():
            lines += line.startswith("data:")
            if lines >= 20:
                break
    closed_at = time.monotonic()
    cluster.wait_for(
        lambda: all(not _held_requests(cluster, index) for index in (first, second)),
        timeout=15,
        what="every rank to drop the cancelled chat",
    )
    # Most of this is the API's HTTP server noticing that the client closed its connection (the
    # same on main); the cancel itself then reaches every rank at once. Measured 0.5-3 s, once
    # 6.1 s on a loaded machine; bound 10 s.
    assert time.monotonic() - closed_at < 10.0


# --- (e) an instance on a device the API node can't reach --------------------------------


def test_chats_skip_an_instance_whose_device_cant_be_reached(
    make_cluster: MakeCluster,
) -> None:
    """Two one-node instances of the model, on the two followers; one follower is frozen. Once
    the API node (the master) sees it Suspect, six chats at once all go to the other
    instance. A frozen device accepts the connection and never answers, so a chat sent to it
    waited for the 30 s bound before moving on (about half of them did). Measured 0.04-0.05 s
    each; bound 5 s."""
    cluster = make_cluster(3)
    master, serving, frozen = _roles(cluster)
    kept = cluster.launch(on=(serving,), via=master)
    lost = cluster.launch(on=(frozen,), via=master)
    assert kept != lost, "the two launches gave one instance"
    _ok(cluster.chat("warm [stub tokens=5]", via=master, timeout=60))
    cluster.pause(frozen)
    # Past the lease and the probe: Suspect, and not yet Gone (about 40 s)
    time.sleep(16)
    sent = time.monotonic()
    chats = [
        cluster.chat_in_background(
            f"chat {index} [stub tokens=5]", via=master, timeout=60
        )
        for index in range(6)
    ]
    results = [chat.result(90) for chat in chats]
    cluster.resume(frozen)
    for result in results:
        _ok(result)
        assert result.ended - sent < 5.0, result.summary()


# --- the API node pauses -------------------------------------------------------------------


@pytest.mark.parametrize("seconds", [15, 30])
def test_a_chat_carries_on_through_a_pause_of_its_api_node(
    make_cluster: MakeCluster, seconds: int
) -> None:
    """SIGSTOP of the API node (a Mac asleep) for longer than the lease, while its chat streams
    from another node. The output node isn't told (it doesn't ask the membership about the API
    node: only the stream closing cancels), and what it sends meanwhile waits in the connection.
    On waking, the API node reads it and the chat completes, every token in order. The API node
    counts a stream's silence on its own running time: a timer on each read fired as the node
    woke from a pause of about the wedge bound (30 s) and ended the chat. Main completes too
    (its tokens wait in the master's log)."""
    cluster = make_cluster(3)
    _, serving, asking = _roles(cluster)
    cluster.launch(on=(serving,), via=asking)
    chat = _streaming(
        cluster, f"api paused {seconds}s [stub tokens=2500 rate=50]", via=asking
    )
    cluster.pause(asking)
    time.sleep(seconds)
    cluster.resume(asking)
    result = chat.result(REPLY_TIMEOUT)
    _ok(result)


# --- the price of one path ----------------------------------------------------------------


@pytest.mark.parametrize("mode", ["reset", "blackhole"])
def test_a_cut_between_the_api_node_and_the_output_node_ends_its_chat_cleanly(
    make_cluster: MakeCluster, monkeypatch: pytest.MonkeyPatch, mode: str
) -> None:
    """The API node and the output node lose each other while both still reach the master.
    Branch: the chat's stream is its only path, so it ends with a clean error: at once for a
    reset; for a blackhole when the stream has been silent for the wedge bound (shortened to
    4 s here; 30 s in exo, or the 10 s lease once membership marks the node unreachable).
    Main: the tokens go round through the master, so the chat carries on. This is where the
    single path costs something; it is reported, not hidden."""
    monkeypatch.setenv(WEDGE_SECONDS_ENV, "4")
    cluster = make_cluster(3)
    _, serving, asking = _roles(cluster)
    cluster.launch(on=(serving,), via=asking)
    _through_the_network(cluster, asking, serving)
    chat = _streaming(cluster, f"cut {mode} " + LONG, via=asking)
    cut_at = time.monotonic()
    cluster.cut(serving, asking, mode="reset" if mode == "reset" else "blackhole")
    result = chat.result(REPLY_TIMEOUT)
    _clean_error(result)
    bound = 2.0 if mode == "reset" else 4.0 + 4.0
    assert result.ended - cut_at < bound, result.summary()
