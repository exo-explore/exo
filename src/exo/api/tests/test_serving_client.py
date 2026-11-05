"""Serving a request without the master, at the API node: the request's output streams from the
instance's output node in order and exactly once, and the stream ends with a clear error as soon
as the output node goes away, goes silent, or a node of the instance becomes unreachable."""

import contextlib
import time
from datetime import timedelta

import anyio
import pytest
from fastapi import HTTPException

import exo.api.main as api_main
import exo.api.serving_client as serving_client
import exo.worker.serving as worker_serving
from exo.api.main import API as API_
from exo.api.recent_events import RecentEvents
from exo.api.serving_client import ServeStream, UnroutableError
from exo.shared.request_errors import (
    DEVICE_DISCONNECTED,
    DEVICE_NOT_RESPONDING,
)
from exo.shared.types.chunks import Chunk, ErrorChunk, TokenChunk
from exo.shared.types.common import NodeId, SessionId, SystemId
from exo.shared.types.events import IndexedEvent, StateSnapshot
from exo.shared.types.worker.runners import RunnerLoading, RunnerReady
from exo.utils.channels import channel
from exo.worker.serving import RefusalError
from exo.worker.tests.serving_cluster import ServingCluster, serving_cluster

API = NodeId("api")


async def open_stream(cluster: ServingCluster, via: NodeId = API) -> ServeStream:
    return await cluster.clients[via].open(cluster.task(), cluster.instance)


async def read(
    stream: ServeStream, on_chunk: object = None, after: int = 0
) -> list[Chunk]:
    received: list[Chunk] = []
    async with contextlib.aclosing(stream.chunks()) as chunks:
        async for chunk in chunks:
            received.append(chunk)
            if on_chunk is not None and len(received) == after:
                assert callable(on_chunk)
                on_chunk()
    return received


def texts(chunks: list[Chunk]) -> list[str]:
    return [chunk.text for chunk in chunks if isinstance(chunk, TokenChunk)]


async def test_a_request_streams_from_another_node_in_order_and_exactly_once() -> None:
    async with serving_cluster(2, tokens=50) as cluster:
        chunks = await read(await open_stream(cluster))
        assert texts(chunks) == [f"t{index} " for index in range(50)]
        assert not any(isinstance(chunk, ErrorChunk) for chunk in chunks)


async def test_a_request_streams_in_process_when_this_node_is_the_output_node() -> None:
    async with serving_cluster(2, api_nodes=(NodeId("rank0"),), tokens=10) as cluster:
        chunks = await read(await open_stream(cluster, via=NodeId("rank0")))
        assert texts(chunks) == [f"t{index} " for index in range(10)]


async def test_a_request_the_output_node_refuses_is_reported_before_any_output() -> (
    None
):
    async with serving_cluster(2) as cluster:
        cluster.ranks[1].supervisor.status = RunnerLoading()
        with pytest.raises(RefusalError) as refused:
            await open_stream(cluster)
        assert refused.value.message.startswith("The model is not ready yet: ")
        assert cluster.clients[API].streams() == []


async def test_a_request_to_an_output_node_that_cant_be_reached_reaches_no_one() -> (
    None
):
    async with serving_cluster(2) as cluster:
        cluster.kill(cluster.rank_nodes[0])
        with pytest.raises(UnroutableError):
            await open_stream(cluster)


async def test_killing_the_output_node_ends_the_stream_at_once() -> None:
    """(b) at the API node: the connection breaks, so the stream ends with a clear error"""
    async with serving_cluster(2, tokens=1000, token_interval=0.01) as cluster:
        stream = await open_stream(cluster)
        start = anyio.current_time()
        chunks = await read(
            stream, lambda: cluster.kill(cluster.rank_nodes[0]), after=5
        )
        assert anyio.current_time() - start < 1.0
        assert isinstance(chunks[-1], ErrorChunk)
        assert chunks[-1].error_message == DEVICE_DISCONNECTED


async def test_a_silent_output_node_ends_the_stream_after_the_wedge_bound(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(serving_client, "WEDGE_SECONDS", 0.3)
    monkeypatch.setattr(serving_client, "STREAM_HEARTBEAT_INTERVAL", 0.03)
    async with serving_cluster(2, tokens=1000, token_interval=0.01) as cluster:
        stream = await open_stream(cluster)
        with anyio.fail_after(5):
            chunks = await read(
                stream, lambda: cluster.freeze(cluster.rank_nodes[0]), after=5
            )
        assert isinstance(chunks[-1], ErrorChunk)
        assert chunks[-1].error_message == DEVICE_NOT_RESPONDING
        cluster.wake(cluster.rank_nodes[0])


async def test_a_short_freeze_of_the_output_node_loses_nothing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(serving_client, "WEDGE_SECONDS", 1.0)
    monkeypatch.setattr(serving_client, "STREAM_HEARTBEAT_INTERVAL", 0.1)
    async with serving_cluster(2, tokens=40, token_interval=0.01) as cluster:
        stream = await open_stream(cluster)

        def pause() -> None:
            cluster.freeze(cluster.rank_nodes[0])

            async def wake_later() -> None:
                await anyio.sleep(0.4)
                cluster.wake(cluster.rank_nodes[0])

            assert cluster.clients[API]._tg.is_running()  # pyright: ignore[reportPrivateUsage]
            cluster.clients[API]._tg.start_soon(wake_later)  # pyright: ignore[reportPrivateUsage]

        chunks = await read(stream, pause, after=10)
        assert texts(chunks) == [f"t{index} " for index in range(40)]


async def test_a_pause_of_this_node_isnt_its_output_nodes_silence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """This node is stopped (here: its event loop blocked) for longer than the wedge bound. On
    waking it reads what is waiting, and the stream carries on. A timer on each read fired as
    the node woke, ending a chat its output node was still streaming."""
    monkeypatch.setattr(serving_client, "WEDGE_SECONDS", 0.3)
    monkeypatch.setattr(serving_client, "STREAM_HEARTBEAT_INTERVAL", 0.03)
    monkeypatch.setattr(serving_client, "STALL_THRESHOLD", timedelta(seconds=0.1))
    async with serving_cluster(2, tokens=40, token_interval=0.01) as cluster:
        stream = await open_stream(cluster)

        def stop_this_node() -> None:
            time.sleep(0.5)

        with anyio.fail_after(5):
            chunks = await read(stream, stop_this_node, after=5)
        assert texts(chunks) == [f"t{index} " for index in range(40)]


async def test_a_stream_ends_when_a_node_of_its_instance_becomes_unreachable() -> None:
    async with serving_cluster(3, tokens=1000) as cluster:
        stream = await open_stream(cluster)
        chunks = await read(
            stream, lambda: cluster.reachability.lose(cluster.rank_nodes[2]), after=3
        )
        assert isinstance(chunks[-1], ErrorChunk)
        assert chunks[-1].error_message == DEVICE_DISCONNECTED


async def test_cancelling_a_stream_ends_it_quietly_and_every_rank_drops_the_request() -> (
    None
):
    """(d) cancel"""
    async with serving_cluster(3, tokens=10_000, token_interval=0.005) as cluster:
        stream = await open_stream(cluster)
        chunks = await read(stream, stream.cancel, after=5)
        assert stream.cancelled
        assert not any(isinstance(chunk, ErrorChunk) for chunk in chunks)
        await cluster.settled(
            lambda: all(not tasks for tasks in cluster.in_progress().values()),
            what="dropped on every rank",
        )
        assert cluster.clients[API].streams() == []


async def test_concurrent_requests_from_two_api_nodes_each_get_their_own_output() -> (
    None
):
    """(d) concurrency: every request's tokens, exactly once, in order"""
    api_nodes = (NodeId("api1"), NodeId("api2"))
    async with serving_cluster(2, api_nodes=api_nodes, tokens=30) as cluster:
        results: dict[int, list[str]] = {}

        async def one(index: int) -> None:
            stream = await open_stream(cluster, via=api_nodes[index % 2])
            results[index] = texts(await read(stream))

        with anyio.fail_after(10):
            async with anyio.create_task_group() as tg:
                for index in range(12):
                    tg.start_soon(one, index)
        assert all(
            got == [f"t{token} " for token in range(30)] for got in results.values()
        )
        assert len(results) == 12


async def test_a_change_of_master_and_its_snapshot_leave_streams_alone() -> None:
    """(a) at the API node: following a new master, and catching up from its snapshot, used to
    end every request in flight. Neither touches a request's stream now."""
    async with serving_cluster(2, tokens=60, token_interval=0.01) as cluster:
        api = object.__new__(API_)
        api.state = cluster.state
        api._recent_events = RecentEvents()  # pyright: ignore[reportPrivateUsage]
        api._system_id = SystemId()  # pyright: ignore[reportPrivateUsage]
        api.paused = False
        api.paused_ev = anyio.Event()
        events, api.event_receiver = channel[IndexedEvent | StateSnapshot]()
        stream = await open_stream(cluster)

        async def change_master() -> None:
            await anyio.sleep(0.1)
            api.follow_new_master(result_clock=1)
            await events.send(
                StateSnapshot(
                    session=SessionId(master_node_id=NodeId("new"), election_clock=1),
                    requester=SystemId(),
                    state=cluster.state,
                )
            )
            events.close()

        chunks: list[Chunk] = []
        async with anyio.create_task_group() as tg:
            tg.start_soon(api._apply_state)  # pyright: ignore[reportPrivateUsage]
            tg.start_soon(change_master)
            chunks = await read(stream)
        assert texts(chunks) == [f"t{index} " for index in range(60)]


def api_node(cluster: ServingCluster) -> API_:
    api = object.__new__(API_)
    api.state = cluster.state
    api._serving = cluster.clients[API]  # pyright: ignore[reportPrivateUsage]
    api._trace_ranks = {}  # pyright: ignore[reportPrivateUsage]
    return api


async def test_a_request_waits_while_its_instance_restarts_and_then_runs() -> None:
    """This node's view lags: a runner of the instance just failed and is being started again.
    The output node refuses the request as not ready, and the API node waits for it, as for a
    model that isn't ready, instead of answering 503 at once."""
    async with serving_cluster(2, tokens=10) as cluster:
        cluster.ranks[1].supervisor.status = RunnerLoading()
        api = api_node(cluster)

        async def recover() -> None:
            await anyio.sleep(0.5)
            cluster.ranks[1].supervisor.status = RunnerReady()

        stream: ServeStream | None = None
        with anyio.fail_after(10):
            async with anyio.create_task_group() as tg:
                tg.start_soon(recover)
                stream = await api._start(cluster.task().task_params)  # pyright: ignore[reportPrivateUsage]
        assert stream is not None
        assert len(texts(await read(stream))) == 10


async def test_a_request_whose_instance_never_gets_ready_gets_a_503(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(api_main, "MODEL_READY_WAIT_SECONDS", 0.5)
    async with serving_cluster(2) as cluster:
        cluster.ranks[1].supervisor.status = RunnerLoading()
        with pytest.raises(HTTPException) as raised:
            await api_node(cluster)._start(cluster.task().task_params)  # pyright: ignore[reportPrivateUsage]
        assert raised.value.status_code == 503
        assert str(raised.value.detail).startswith("The model is not ready yet: ")


async def test_a_request_whose_client_leaves_while_it_is_delivered_is_dropped_everywhere(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Opening a stream is cancelled (its client left) while a rank is slow to take the
    request: the stream closes, and once the rank wakes every rank drops the request."""
    monkeypatch.setattr(worker_serving, "STREAM_HEARTBEAT_INTERVAL", 0.05)
    async with serving_cluster(2, tokens=100_000, token_interval=0.002) as cluster:
        cluster.freeze(cluster.rank_nodes[1])
        with anyio.move_on_after(0.3):
            await open_stream(cluster)
        assert cluster.clients[API].streams() == []
        cluster.wake(cluster.rank_nodes[1])
        await cluster.settled(
            lambda: all(not tasks for tasks in cluster.in_progress().values()),
            what="dropped everywhere",
        )
