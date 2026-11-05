# pyright: reportPrivateUsage=false
"""Serving a request without the master, at the output node: every rank of the instance takes
the request before it is accepted, its output streams back in order, and a request that doesn't
run to its end leaves no task on any rank, however its delivery failed."""

import contextlib
import time

import anyio
import pytest

import exo.worker.serving as worker_serving
from exo.shared.request_errors import (
    DEVICE_DISCONNECTED,
    INSTANCE_GONE_BEFORE_START,
    MODEL_STOPPED,
    NOT_TAKEN,
)
from exo.shared.types.chunks import Chunk, ErrorChunk, TokenChunk
from exo.shared.types.common import NodeId
from exo.shared.types.serving import (
    Accepted,
    Delivery,
    Refused,
    ServeFrame,
    ServeRequest,
)
from exo.shared.types.worker.instances import InstanceId
from exo.shared.types.worker.runners import RunnerFailed, RunnerLoading
from exo.worker.serving import DeliveryOutcome, RefusalError
from exo.worker.tests.serving_cluster import ServingCluster, serving_cluster

API = NodeId("api")


def request_for(cluster: ServingCluster, prompt: str = "hello") -> ServeRequest:
    return ServeRequest(target=cluster.rank_nodes[0], task=cluster.task(prompt))


async def frames_of(
    cluster: ServingCluster, request: ServeRequest, limit: int | None = None
) -> list[ServeFrame]:
    """The request's frames (heartbeats left out), stopping after `limit` chunks"""
    received: list[ServeFrame] = []
    chunks = 0
    async with contextlib.aclosing(
        cluster.hosts[request.target].serve(request)
    ) as frames:
        async for frame in frames:
            if frame is None:
                continue
            received.append(frame)
            if not isinstance(frame, (Accepted, Refused)):
                chunks += 1
                if limit is not None and chunks >= limit:
                    break
    return received


def texts(frames: list[ServeFrame]) -> list[str]:
    return [frame.text for frame in frames if isinstance(frame, TokenChunk)]


def nothing_left(cluster: ServingCluster) -> bool:
    return all(not tasks for tasks in cluster.in_progress().values())


async def test_every_rank_holds_a_request_before_it_is_accepted_and_its_output_streams_in_order() -> (
    None
):
    async with serving_cluster(3, tokens=20) as cluster:
        request = request_for(cluster)
        frames = await frames_of(cluster, request)

        assert isinstance(frames[0], Accepted)
        assert texts(frames) == [f"t{index} " for index in range(20)]
        assert all(request.task.task_id in rank.received for rank in cluster.ranks), (
            "every rank was given the task"
        )
        await cluster.settled(lambda: nothing_left(cluster), what="ranks forgot it")


async def test_a_rank_that_cant_take_a_request_refuses_it_and_it_runs_nowhere() -> None:
    async with serving_cluster(3) as cluster:
        # rank 2's runner is restarting
        cluster.ranks[2].supervisor.status = RunnerLoading()
        request = request_for(cluster)
        frames = await frames_of(cluster, request)

        # Said as a model that isn't ready, which the API node waits for
        assert frames == [
            Refused(
                message="The model is not ready yet: it is being started again. "
                "Try again in 5 s."
            )
        ]
        # The ranks that took it are told to drop it
        task_id = request.task.task_id
        await cluster.settled(
            lambda: all(task_id in rank.cancels_received for rank in cluster.ranks[:2]),
            what="cancelled where it was taken",
        )
        assert task_id not in cluster.ranks[2].received
        assert cluster.agreement is not None
        assert task_id not in cluster.agreement.started


async def test_a_rank_frozen_while_a_request_is_delivered_leaves_no_task_anywhere(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """(c) The partial-agreement case: rank 2 is frozen as the request arrives, so it doesn't
    take it in time and the request is refused. Ranks 0 and 1 hold the task, which on its own
    would stall the instance (they wait for rank 2 in the agreement). Once rank 2 wakes it
    takes the task and its cancellation, every rank drops it, and the instance serves the next
    request."""
    monkeypatch.setattr(worker_serving, "WEDGE_SECONDS", 0.3)
    async with serving_cluster(3, tokens=5) as cluster:
        cluster.freeze(cluster.rank_nodes[2])
        request = request_for(cluster)
        with anyio.fail_after(5):
            frames = await frames_of(cluster, request)
        assert frames == [Refused(message=NOT_TAKEN)]
        assert cluster.in_progress()[cluster.rank_nodes[0]] == [request.task.task_id]

        # On waking, rank 2 takes the delivery it was sent while frozen, so for a moment every
        # rank holds the task and the ranks may start it; the cancellation, sent again until
        # rank 2 takes it, then ends it on every rank
        cluster.wake(cluster.rank_nodes[2])
        await cluster.settled(lambda: nothing_left(cluster), what="dropped everywhere")
        assert cluster.agreement is not None
        assert request.task.task_id in cluster.agreement.ended

        follow_up = await frames_of(cluster, request_for(cluster, "again"))
        assert isinstance(follow_up[0], Accepted)
        assert len(texts(follow_up)) == 5


async def test_a_task_and_its_cancellation_reach_a_rank_that_never_got_the_task() -> (
    None
):
    async with serving_cluster(2) as cluster:
        task = cluster.task()
        peer = cluster.hosts[cluster.rank_nodes[1]]
        await peer.deliver(Delivery(task=task, cancelled=True))
        rank = cluster.ranks[1]
        await cluster.settled(lambda: task.task_id in rank.cancels_received)
        assert rank.received == [task.task_id]


async def test_the_same_delivery_twice_starts_the_task_once() -> None:
    async with serving_cluster(2) as cluster:
        task = cluster.task()
        peer = cluster.hosts[cluster.rank_nodes[1]]
        await peer.deliver(Delivery(task=task))
        await peer.deliver(Delivery(task=task))
        await anyio.sleep(0.1)
        assert cluster.ranks[1].received == [task.task_id]


async def test_a_rank_whose_runner_isnt_ready_refuses_a_delivery() -> None:
    async with serving_cluster(2) as cluster:
        cluster.ranks[1].supervisor.status = RunnerLoading()
        with pytest.raises(RefusalError):
            await cluster.hosts[cluster.rank_nodes[1]].deliver(
                Delivery(task=cluster.task())
            )
        assert cluster.ranks[1].received == []


async def test_closing_a_stream_cancels_the_request_on_every_rank() -> None:
    async with serving_cluster(3, tokens=1000, token_interval=0.01) as cluster:
        request = request_for(cluster)
        frames = await frames_of(cluster, request, limit=5)
        assert len(texts(frames)) == 5
        await cluster.settled(
            lambda: nothing_left(cluster), what="cancelled everywhere"
        )
        assert cluster.agreement is not None
        assert request.task.task_id in cluster.agreement.ended


async def test_a_request_for_a_node_that_isnt_the_output_node_is_refused() -> None:
    async with serving_cluster(2) as cluster:
        request = ServeRequest(target=cluster.rank_nodes[1], task=cluster.task())
        assert await frames_of(cluster, request) == [
            Refused(message=DEVICE_DISCONNECTED)
        ]
        assert all(not rank.received for rank in cluster.ranks)


async def test_a_request_for_an_unknown_instance_is_refused() -> None:
    async with serving_cluster(2) as cluster:
        task = cluster.task().model_copy(update={"instance_id": InstanceId("gone")})
        request = ServeRequest(target=cluster.rank_nodes[0], task=task)
        assert await frames_of(cluster, request) == [
            Refused(message=INSTANCE_GONE_BEFORE_START)
        ]


async def test_a_request_on_an_instance_with_an_unreachable_node_is_refused() -> None:
    async with serving_cluster(3) as cluster:
        cluster.reachability.lose(cluster.rank_nodes[2])
        assert await frames_of(cluster, request_for(cluster)) == [
            Refused(message=DEVICE_DISCONNECTED)
        ]
        assert all(not rank.received for rank in cluster.ranks)


async def _stream_until(
    cluster: ServingCluster, request: ServeRequest, act: object
) -> list[Chunk]:
    """Stream a long request, call `act` after a few tokens, and return its chunks"""
    chunks: list[Chunk] = []
    async with contextlib.aclosing(
        cluster.hosts[request.target].serve(request)
    ) as frames:
        async for frame in frames:
            if frame is None or isinstance(frame, (Accepted, Refused)):
                continue
            chunks.append(frame)
            if len(chunks) == 3:
                assert callable(act)
                act()
    return chunks


async def test_a_request_on_a_deleted_instance_ends_saying_why() -> None:
    async with serving_cluster(2, tokens=1000) as cluster:
        request = request_for(cluster)
        chunks = await _stream_until(
            cluster,
            request,
            lambda: cluster.hosts[cluster.rank_nodes[0]].end_instance(
                cluster.instance.instance_id, MODEL_STOPPED
            ),
        )
        assert isinstance(chunks[-1], ErrorChunk)
        assert chunks[-1].error_message == MODEL_STOPPED


async def test_a_request_ends_when_a_rank_becomes_unreachable() -> None:
    async with serving_cluster(3, tokens=1000) as cluster:
        chunks = await _stream_until(
            cluster,
            request_for(cluster),
            lambda: cluster.reachability.lose(cluster.rank_nodes[2]),
        )
        assert isinstance(chunks[-1], ErrorChunk)
        assert chunks[-1].error_message == DEVICE_DISCONNECTED


async def test_a_runner_that_stops_ends_its_requests() -> None:
    async with serving_cluster(2, tokens=1000) as cluster:
        chunks = await _stream_until(
            cluster, request_for(cluster), cluster.ranks[0].supervisor.shutdown
        )
        assert isinstance(chunks[-1], ErrorChunk)
        assert chunks[-1].error_message == MODEL_STOPPED


async def test_a_rank_lost_while_a_request_is_delivered_refuses_it_at_once() -> None:
    async with serving_cluster(3) as cluster:
        cluster.freeze(cluster.rank_nodes[2])
        request = request_for(cluster)

        async def lose_rank_2() -> None:
            await anyio.sleep(0.2)
            cluster.reachability.lose(cluster.rank_nodes[2])

        frames: list[ServeFrame] = []
        with anyio.fail_after(2):
            async with anyio.create_task_group() as tg:
                tg.start_soon(lose_rank_2)
                frames = await frames_of(cluster, request)
        assert frames == [Refused(message=DEVICE_DISCONNECTED)]
        cluster.wake(cluster.rank_nodes[2])


async def test_a_rank_refusing_a_delivery_for_another_reason_is_given_its_cancellation_until_it_takes_it(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Only a rank's answer that it has no ready runner means its instance is being started
    again. Any other answer (here, refused by its member check) leaves the ranks that took the
    request holding it, so they would stall the instance: the request is refused as not taken
    (not as "being started again"), and every rank is given its cancellation until each takes
    it. Before, any answer but 200 ended those attempts at once."""
    monkeypatch.setattr(worker_serving, "STREAM_HEARTBEAT_INTERVAL", 0.05)
    async with serving_cluster(3, tokens=5) as cluster:
        refusing = cluster.rank_nodes[2]
        cluster.refuse(refusing, "answered 403: not a cluster member")
        request = request_for(cluster)
        frames = await frames_of(cluster, request)
        assert frames == [Refused(message=NOT_TAKEN)]

        task_id = request.task.task_id
        await anyio.sleep(0.3)
        # Still trying: ranks 0 and 1 hold the task and its cancellation, rank 2 neither
        assert task_id in cluster.in_progress()[cluster.rank_nodes[0]]
        assert task_id not in cluster.ranks[2].received

        del cluster.refusing[refusing]
        await cluster.settled(lambda: nothing_left(cluster), what="dropped everywhere")
        assert cluster.ranks[2].arrivals == [("task", task_id), ("cancel", task_id)]
        follow_up = await frames_of(cluster, request_for(cluster, "again"))
        assert isinstance(follow_up[0], Accepted)


async def test_a_stream_closed_while_its_request_is_delivered_is_cancelled_on_every_rank(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The API node goes away while a rank is slow to take its request: every rank it reached
    is given its cancellation, and it ends everywhere."""
    monkeypatch.setattr(worker_serving, "STREAM_HEARTBEAT_INTERVAL", 0.05)
    async with serving_cluster(2, tokens=100, token_interval=0.02) as cluster:
        peer = cluster.rank_nodes[1]
        cluster.freeze(peer)
        request = request_for(cluster)
        frames = cluster.hosts[request.target].serve(request)
        assert await anext(frames) is None  # a heartbeat: still being delivered
        await frames.aclose()

        cluster.wake(peer)
        task_id = request.task.task_id
        await cluster.settled(lambda: nothing_left(cluster), what="dropped everywhere")
        for rank in cluster.ranks:
            assert ("cancel", task_id) in rank.arrivals, rank.node_id
        assert cluster.agreement is not None
        assert task_id in cluster.agreement.ended


async def test_an_image_request_whose_stream_closed_while_it_waited_its_turn_reaches_no_rank(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An image request waits for the one before it to reach every rank. If its API node goes
    away meanwhile, it is never delivered. Before, the output node's runner took it once its
    turn came, with no cancellation, and the ranks ran it for nobody."""
    monkeypatch.setattr(worker_serving, "STREAM_HEARTBEAT_INTERVAL", 0.05)
    async with serving_cluster(2, tokens=3) as cluster:
        output = cluster.rank_nodes[-1]
        frozen = cluster.rank_nodes[0]
        first = ServeRequest(target=output, task=cluster.image_task("one"))
        second = ServeRequest(target=output, task=cluster.image_task("two"))
        cluster.freeze(frozen)
        first_frames: list[ServeFrame] = []

        async def send_first() -> None:
            first_frames.extend(await frames_of(cluster, first))

        async with anyio.create_task_group() as tg:
            tg.start_soon(send_first)
            await anyio.sleep(0.1)
            frames = cluster.hosts[output].serve(second)
            assert await anext(frames) is None  # waiting for the first to be delivered
            await frames.aclose()
            await anyio.sleep(0.1)
            cluster.wake(frozen)
        assert isinstance(first_frames[0], Accepted)
        await anyio.sleep(0.2)
        for rank in cluster.ranks:
            assert ("task", second.task.task_id) not in rank.arrivals, rank.node_id


async def test_an_image_request_not_every_rank_took_reaches_every_rank_before_the_next(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Image ranks run requests in the order they arrive, without agreeing on it. A request a
    rank refused at first reaches that rank (with its cancellation) before any later request
    does: one sent meanwhile is refused without reaching any rank, so every rank holds the
    requests in the same order and the stages never pair different requests. Before, the next
    request could be delivered while the first's cancellation was still being retried."""
    monkeypatch.setattr(worker_serving, "STREAM_HEARTBEAT_INTERVAL", 0.05)
    async with serving_cluster(2, tokens=3) as cluster:
        output = cluster.rank_nodes[-1]
        other = cluster.rank_nodes[0]

        def image_request(prompt: str) -> ServeRequest:
            return ServeRequest(target=output, task=cluster.image_task(prompt))

        first, meanwhile, after = (image_request(p) for p in ("one", "two", "three"))
        # The first request's delivery and the first attempt at its cancellation
        cluster.refuse(other, "answered 500", times=2)
        assert await frames_of(cluster, first) == [Refused(message=NOT_TAKEN)]
        assert await frames_of(cluster, meanwhile) == [Refused(message=NOT_TAKEN)]

        one = first.task.task_id
        await cluster.settled(
            lambda: all(("cancel", one) in rank.arrivals for rank in cluster.ranks),
            what="the first cancelled everywhere",
        )
        after_frames = await frames_of(cluster, after)
        assert isinstance(after_frames[0], Accepted)
        for rank in cluster.ranks:
            tasks = [task_id for kind, task_id in rank.arrivals if kind == "task"]
            assert tasks == [one, after.task.task_id], rank.node_id


async def test_a_task_the_output_runner_already_had_is_refused_not_streamed_forever() -> (
    None
):
    """A request's task is served once. Sent again after it ended, it is refused at once,
    where it was accepted and then streamed heartbeats forever (its runner skips a task it
    has finished)."""
    async with serving_cluster(2, tokens=3) as cluster:
        request = request_for(cluster)
        first = await frames_of(cluster, request)
        assert isinstance(first[0], Accepted)
        with anyio.fail_after(2):
            again = await frames_of(cluster, request)
        assert again == [Refused(message=NOT_TAKEN)]


def _counting_cancellations(cluster: ServingCluster, node: NodeId) -> list[float]:
    """When each cancellation was sent to the node"""
    sent: list[float] = []
    deliver = cluster._deliver

    async def counting(peer: NodeId, delivery: Delivery) -> DeliveryOutcome:
        if peer == node and delivery.cancelled:
            sent.append(time.monotonic())
        return await deliver(peer, delivery)

    for host in cluster.hosts.values():
        host._deliver_to = counting
    return sent


async def test_requests_meanwhile_are_refused_without_being_delivered(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """While a rank hasn't taken a request's cancellation, the instance can't serve: later
    requests are refused at once and never reach a runner. Before, each was delivered, held by
    the ranks that took it, and given a retry loop of its own: one more per client request."""
    monkeypatch.setattr(worker_serving, "STREAM_HEARTBEAT_INTERVAL", 0.05)
    async with serving_cluster(2) as cluster:
        cluster.refuse(cluster.rank_nodes[1], "answered 403: not a cluster member")
        first = request_for(cluster)
        assert await frames_of(cluster, first) == [Refused(message=NOT_TAKEN)]
        later = [request_for(cluster, f"again {index}") for index in range(5)]
        for request in later:
            with anyio.fail_after(1):
                assert await frames_of(cluster, request) == [Refused(message=NOT_TAKEN)]
        assert cluster.ranks[0].received == [first.task.task_id]


async def test_a_rank_that_keeps_refusing_a_cancellation_gets_the_instance_restarted(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An image instance whose first stage refuses deliveries for any reason but having no
    ready runner: the request's cancellation is retried, and the image lock held, until
    CANCEL_REFUSED_LIMIT; then the output node fails its runner (as the stall watchdog does for
    text), so the instance's runners start again. Before, image instances (which have no
    watchdog) retried for ever, and every later image request waited WEDGE_SECONDS for its turn."""
    monkeypatch.setattr(worker_serving, "STREAM_HEARTBEAT_INTERVAL", 0.05)
    monkeypatch.setattr(worker_serving, "CANCEL_REFUSED_LIMIT", 0.5)
    async with serving_cluster(2, tokens=3) as cluster:
        output = cluster.rank_nodes[-1]
        first_stage = cluster.rank_nodes[0]
        attempts = _counting_cancellations(cluster, first_stage)
        cluster.refuse(first_stage, "answered 403: not a cluster member")
        first = ServeRequest(target=output, task=cluster.image_task("one"))
        assert await frames_of(cluster, first) == [Refused(message=NOT_TAKEN)]
        second = ServeRequest(target=output, task=cluster.image_task("two"))
        with anyio.fail_after(0.3):
            assert await frames_of(cluster, second) == [Refused(message=NOT_TAKEN)]

        output_runner = cluster.ranks[-1].supervisor
        await cluster.settled(
            lambda: isinstance(output_runner.status, RunnerFailed),
            what="the output node failed its runner",
        )
        tried = len(attempts)
        await anyio.sleep(0.3)
        assert len(attempts) == tried, "the cancellation stopped being retried"
        assert (
            not cluster.hosts[output]
            ._image_locks[cluster.instance.instance_id]
            .locked()
        )


async def test_a_rank_suspect_for_a_moment_still_gets_the_cancellation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A rank refuses deliveries, so a request's cancellation is being retried, and it goes
    Suspect for a moment (its link closed, the probe went unanswered), then is Alive again and
    takes deliveries. The retries carry on through the moment, so every rank drops the request.
    Before, unreachable for a moment ended them: the output rank held the request with nothing
    to drop it, and the instance took new requests again."""
    monkeypatch.setattr(worker_serving, "STREAM_HEARTBEAT_INTERVAL", 0.1)
    monkeypatch.setattr(worker_serving, "CANCEL_REFUSED_LIMIT", 2.0)
    async with serving_cluster(2, tokens=5) as cluster:
        output, peer = cluster.rank_nodes
        cluster.refuse(peer, "answered 403: not a cluster member")
        request = request_for(cluster)
        assert await frames_of(cluster, request) == [Refused(message=NOT_TAKEN)]
        await anyio.sleep(0.25)
        cluster.reachability.lose(peer)
        await anyio.sleep(0.3)
        cluster.reachability.reachable.add(peer)
        del cluster.refusing[peer]

        task_id = request.task.task_id
        await cluster.settled(lambda: nothing_left(cluster), what="dropped everywhere")
        assert cluster.ranks[1].arrivals == [("task", task_id), ("cancel", task_id)]
        assert not cluster.hosts[output]._unsettled[cluster.instance.instance_id]


async def test_a_rank_that_never_answers_gets_the_instance_restarted_within_the_limit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Each attempt to give a rank a cancellation is cut short to fit CANCEL_REFUSED_LIMIT, so
    the limit holds against a rank that doesn't answer at all. Before, it was checked only
    after an attempt that could itself take the wedge bound: about twice the limit."""
    monkeypatch.setattr(worker_serving, "STREAM_HEARTBEAT_INTERVAL", 0.05)
    monkeypatch.setattr(worker_serving, "CANCEL_REFUSED_LIMIT", 0.5)
    async with serving_cluster(2, tokens=5) as cluster:
        peer = cluster.rank_nodes[1]
        # Refused at once (the rank isn't heard from), then frozen: it never answers again
        cluster.refuse(peer, "answered 500", times=1)
        cluster.freeze(peer)
        request = request_for(cluster)
        assert await frames_of(cluster, request) == [Refused(message=NOT_TAKEN)]
        started = time.monotonic()
        output_runner = cluster.ranks[0].supervisor
        await cluster.settled(
            lambda: isinstance(output_runner.status, RunnerFailed),
            what="the output node failed its runner",
        )
        assert time.monotonic() - started < 1.5
        cluster.wake(peer)
