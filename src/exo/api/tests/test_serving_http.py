# pyright: reportPrivateUsage=false
"""The internal serving routes over real HTTP: a request's output streams as NDJSON lines,
closing the connection cancels the request on every rank, and only nodes of the cluster may
call the routes (they hand tasks straight to runners)."""

import contextlib
import socket
from collections.abc import AsyncGenerator, AsyncIterator, Callable
from typing import cast

import anyio
import httpx
import pytest
from anyio.abc import TaskGroup
from fastapi import FastAPI, Request
from fastapi.responses import Response, StreamingResponse
from fastapi.testclient import TestClient
from hypercorn.asyncio import serve  # pyright: ignore[reportUnknownVariableType]
from hypercorn.config import Config
from hypercorn.typing import ASGIFramework

import exo.shared.serving as shared_serving
from exo.api.closes_streams import ClosesStreams, close_with_exchange
from exo.api.cluster_routes import ClusterRoutes, ClusterRoutesServer
from exo.api.listening_socket import listen_on_api_port
from exo.api.serving_client import ServeStream, ServingClient
from exo.shared.models.model_cards import ModelId
from exo.shared.request_errors import NOT_A_MEMBER
from exo.shared.types.chunks import Chunk, ErrorChunk, TokenChunk
from exo.shared.types.common import CommandId, NodeId
from exo.shared.types.profiling import NetworkInterfaceInfo, NodeNetworkInfo
from exo.shared.types.serving import Accepted, Delivery, serve_frames
from exo.shared.types.state import State
from exo.shared.types.tasks import TextGeneration
from exo.shared.types.text_generation import (
    InputMessage,
    InputMessageContent,
    TextGenerationTaskParams,
)
from exo.shared.types.worker.instances import InstanceId
from exo.shared.types.worker.runners import RunnerLoading
from exo.worker.serving import (
    DECLINED_HEADER,
    DECLINED_HEADER_VALUE,
    DECLINED_STATUS,
    DELIVER_PATH,
    SERVE_PATH,
    Declined,
    RefusalError,
    ServingHost,
    Undelivered,
)
from exo.worker.tests.serving_cluster import (
    FakeReachability,
    ServingCluster,
    serving_cluster,
)

API_NODE = NodeId("api")


def task_for_test() -> TextGeneration:
    return TextGeneration(
        command_id=CommandId(),
        instance_id=InstanceId("instance"),
        task_params=TextGenerationTaskParams(
            model=ModelId("test-model"),
            input=[InputMessage(role="user", content=InputMessageContent("hi"))],
        ),
    )


def app_for(cluster: ServingCluster, node: NodeId) -> FastAPI:
    """The serving routes of one node's API, as exo registers them"""
    app = FastAPI()
    ClusterRoutes(state=lambda: cluster.state, host=cluster.hosts[node]).add_to(app)
    return app


def free_port() -> int:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return cast(int, probe.getsockname()[1])


@contextlib.asynccontextmanager
async def http_server(app: FastAPI) -> AsyncIterator[str]:
    port = free_port()
    config = Config()
    config.bind = [f"127.0.0.1:{port}"]
    config.accesslog = None
    stop = anyio.Event()
    async with anyio.create_task_group() as tg:
        tg.start_soon(
            lambda: serve(cast(ASGIFramework, app), config, shutdown_trigger=stop.wait)
        )
        with anyio.fail_after(5):
            while True:
                with (
                    contextlib.suppress(OSError),
                    socket.create_connection(("127.0.0.1", port), timeout=0.2),
                ):
                    break
                await anyio.sleep(0.02)
        try:
            yield f"http://127.0.0.1:{port}"
        finally:
            stop.set()


def through(url: str) -> Callable[[State, NodeId, NodeId, int], list[str]]:
    def node_urls(
        _state: State, _source: NodeId, _sink: NodeId, _port: int
    ) -> list[str]:
        return [url]

    return node_urls


async def http_client(cluster: ServingCluster, tg: TaskGroup) -> ServingClient:
    client = ServingClient(
        API_NODE,
        api_port=52415,
        host=None,
        state=lambda: cluster.state,
        reachability=cluster.reachability,
    )
    tg.start_soon(client.run)
    await anyio.sleep(0.05)
    return client


async def read(stream: ServeStream, stop_after: int | None = None) -> list[Chunk]:
    received: list[Chunk] = []
    async with contextlib.aclosing(stream.chunks()) as chunks:
        async for chunk in chunks:
            received.append(chunk)
            if stop_after is not None and len(received) >= stop_after:
                stream.cancel()
    return received


async def test_a_request_streams_over_http_in_order_and_exactly_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async with (
        serving_cluster(2, tokens=200, token_interval=0.001) as cluster,
        http_server(app_for(cluster, cluster.rank_nodes[0])) as url,
        anyio.create_task_group() as tg,
    ):
        monkeypatch.setattr(shared_serving, "node_urls", through(url))
        client = await http_client(cluster, tg)
        stream = await client.open(cluster.task(), cluster.instance)
        chunks = await read(stream)
        tg.cancel_scope.cancel()
        assert [c.text for c in chunks if isinstance(c, TokenChunk)] == [
            f"t{index} " for index in range(200)
        ]
        assert not any(isinstance(chunk, ErrorChunk) for chunk in chunks)


async def test_closing_the_connection_cancels_the_request_on_every_rank(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async with (
        serving_cluster(3, tokens=100_000, token_interval=0.002) as cluster,
        http_server(app_for(cluster, cluster.rank_nodes[0])) as url,
        anyio.create_task_group() as tg,
    ):
        monkeypatch.setattr(shared_serving, "node_urls", through(url))
        client = await http_client(cluster, tg)
        stream = await client.open(cluster.task(), cluster.instance)
        chunks = await read(stream, stop_after=10)
        assert len(chunks) >= 10
        # The output node notices the closed connection when it next writes: within a token,
        # or a heartbeat at worst
        await cluster.settled(
            lambda: all(not tasks for tasks in cluster.in_progress().values()),
            timeout=10,
            what="dropped on every rank",
        )
        tg.cancel_scope.cancel()


async def test_a_device_outside_the_cluster_is_refused() -> None:
    async with serving_cluster(2) as cluster:
        app = app_for(cluster, cluster.rank_nodes[1])
        body = Delivery(task=cluster.task()).model_dump_json()
        stranger = TestClient(app, client=("203.0.113.9", 50000))
        assert stranger.post(DELIVER_PATH, content=body).status_code == 403
        assert stranger.post(SERVE_PATH, content=body).status_code == 403
        await anyio.sleep(0.1)
        assert cluster.ranks[1].received == []


async def test_a_node_of_the_cluster_may_deliver(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async with serving_cluster(2) as cluster:
        task = cluster.task()
        async with (
            http_server(app_for(cluster, cluster.rank_nodes[1])) as url,
            httpx.AsyncClient() as http,
        ):
            response = await http.post(
                f"{url}{DELIVER_PATH}",
                content=Delivery(task=task).model_dump_json(),
            )
        # Loopback is a member's address (every node on this machine)
        assert response.status_code == 200
        await cluster.settled(lambda: cluster.ranks[1].received == [task.task_id])


async def test_a_delivery_is_sent_again_when_the_node_drops_the_connection_unanswered() -> (
    None
):
    """Found on real hosts: a node frozen past its server's keep-alive timeout closes the idle
    connection the delivery was sent on as it wakes, without answering. Taking a delivery twice
    is harmless, so it is sent once more, on a fresh connection."""
    peer = NodeId("peer")
    calls: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(str(request.url))
        if len(calls) == 1:
            raise httpx.RemoteProtocolError(
                "Server disconnected without sending a response."
            )
        return httpx.Response(200, json={"detail": "taken"})

    state = State(
        node_network={
            peer: NodeNetworkInfo(
                interfaces=[NetworkInterfaceInfo(name="en0", ip_address="10.0.0.9")]
            )
        }
    )
    host = ServingHost(
        NodeId("me"),
        state=lambda: state,
        runners=dict,
        reachability=FakeReachability({peer}),
        api_port=52415,
    )
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        host._client = client
        outcome = await host._deliver_over_http(peer, Delivery(task=task_for_test()))
    assert outcome is None
    assert calls == [f"http://10.0.0.9:52415{DELIVER_PATH}"] * 2


def accepting_output_node() -> FastAPI:
    """An output node that accepts every request at once, then sends heartbeats, as a long reply
    does"""
    app = FastAPI()

    @app.post(SERVE_PATH)
    async def _serve(request: Request) -> StreamingResponse:  # pyright: ignore[reportUnusedFunction]
        await request.body()

        async def lines() -> AsyncGenerator[bytes, None]:
            yield serve_frames.dump_json(Accepted()) + b"\n"
            while True:
                await anyio.sleep(0.5)
                yield b"\n"

        return StreamingResponse(lines(), media_type="application/x-ndjson")

    return app


async def test_one_api_node_holds_more_than_a_hundred_streams_at_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """httpx's default pool holds 100 connections, and a request past them waited for one of
    the others to end, with no answer at all: the 101st chat from one API node hung."""
    in_flight = 110
    async with (
        serving_cluster(1) as cluster,
        http_server(accepting_output_node()) as url,
        anyio.create_task_group() as tg,
    ):
        monkeypatch.setattr(shared_serving, "node_urls", through(url))
        client = await http_client(cluster, tg)
        streams: list[ServeStream] = []

        async def one() -> None:
            streams.append(await client.open(cluster.task(), cluster.instance))

        with anyio.fail_after(10):
            async with anyio.create_task_group() as opening:
                for _ in range(in_flight):
                    opening.start_soon(one)
        assert len(streams) == in_flight
        for stream in streams:
            stream.cancel()
        tg.cancel_scope.cancel()


def single_node_output(peer: NodeId, address: str) -> State:
    return State(
        node_network={
            peer: NodeNetworkInfo(
                interfaces=[NetworkInterfaceInfo(name="en0", ip_address=address)]
            )
        }
    )


async def test_a_request_is_sent_again_when_the_output_node_drops_the_connection_unanswered(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A node that was frozen or asleep past its server's keep-alive timeout closes the idle
    connection a request went on as it wakes, without answering. The request is sent once
    more on a fresh connection, where it got a 503 ("disconnected")."""
    calls: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(str(request.url))
        if len(calls) == 1:
            raise httpx.RemoteProtocolError(
                "Server disconnected without sending a response."
            )
        return httpx.Response(200, content=serve_frames.dump_json(Accepted()) + b"\n")

    async with serving_cluster(1) as cluster:
        client = ServingClient(
            API_NODE,
            api_port=52415,
            host=None,
            state=lambda: cluster.state,
            reachability=cluster.reachability,
        )
        monkeypatch.setattr(
            shared_serving, "node_urls", through("http://10.0.0.9:52415")
        )
        async with (
            httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http,
            client._tg,
        ):
            client._client = http
            stream = await client.open(cluster.task(), cluster.instance)
            assert stream.task.instance_id == cluster.instance.instance_id
            client._tg.cancel_tasks()
    assert calls == [f"http://10.0.0.9:52415{SERVE_PATH}"] * 2


async def test_an_output_node_that_refuses_this_node_as_a_stranger_says_so(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A 403 from the output node's member check is a refusal ("doesn't recognise this
    device"), not "can't reach the device running this model"."""

    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(403, json={"detail": "Only devices of the cluster"})

    async with serving_cluster(1) as cluster:
        client = ServingClient(
            API_NODE,
            api_port=52415,
            host=None,
            state=lambda: cluster.state,
            reachability=cluster.reachability,
        )
        monkeypatch.setattr(
            shared_serving, "node_urls", through("http://10.0.0.9:52415")
        )
        async with (
            httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http,
            client._tg,
        ):
            client._client = http
            with pytest.raises(RefusalError) as refused:
                await client.open(cluster.task(), cluster.instance)
            assert refused.value.message == NOT_A_MEMBER
            client._tg.cancel_tasks()


@pytest.mark.parametrize(
    ("status", "headers", "outcome"),
    [
        (200, {}, None),
        (DECLINED_STATUS, {DECLINED_HEADER: DECLINED_HEADER_VALUE}, "declined"),
        (DECLINED_STATUS, {}, "undelivered"),
        (403, {}, "undelivered"),
        (404, {}, "undelivered"),
        (500, {}, "undelivered"),
    ],
    ids=["taken", "no ready runner", "409 without the reason", "403", "404", "500"],
)
async def test_only_a_ranks_no_ready_runner_answer_is_a_decline(
    status: int, headers: dict[str, str], outcome: object
) -> None:
    """A decline ends the attempts to give a rank a request's cancellation (its instance is
    being started again). Any other answer means nothing about its runner, so it is tried
    again: before, a 403 or a 500 was taken for a decline, and the ranks that took the request
    held it until a watchdog restarted them."""
    peer = NodeId("peer")
    host = ServingHost(
        NodeId("me"),
        state=lambda: single_node_output(peer, "10.0.0.9"),
        runners=dict,
        reachability=FakeReachability({peer}),
        api_port=52415,
    )

    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(status, json={"detail": "an answer"}, headers=headers)

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        host._client = client
        answer = await host._deliver_over_http(peer, Delivery(task=task_for_test()))
    if outcome == "undelivered":
        assert isinstance(answer, Undelivered)
    elif outcome == "declined":
        assert isinstance(answer, Declined)
    else:
        assert answer == outcome


async def test_a_node_of_the_cluster_at_its_own_address_may_deliver() -> None:
    """The member check accepts a node's own address, not only loopback"""
    async with serving_cluster(2) as cluster:
        app = app_for(cluster, cluster.rank_nodes[1])
        member_address = cluster.state.node_network[cluster.rank_nodes[0]].interfaces[0]
        member = TestClient(app, client=(member_address.ip_address, 50000))
        task = cluster.task()
        response = member.post(
            DELIVER_PATH, content=Delivery(task=task).model_dump_json()
        )
        assert response.status_code == 200
        await cluster.settled(lambda: cluster.ranks[1].received == [task.task_id])


async def test_a_rank_with_no_ready_runner_answers_so() -> None:
    async with serving_cluster(2) as cluster:
        cluster.ranks[1].supervisor.status = RunnerLoading()
        app = app_for(cluster, cluster.rank_nodes[1])
        response = TestClient(app, client=("127.0.0.1", 50000)).post(
            DELIVER_PATH, content=Delivery(task=cluster.task()).model_dump_json()
        )
        assert response.status_code == DECLINED_STATUS
        assert DECLINED_HEADER in response.headers


async def test_a_request_stream_closes_when_its_http_exchange_ends_though_no_body_read_it() -> (
    None
):
    """A response's body closes its request's stream when it ends. But a client that gives up
    before its response starts (the response waits until every rank holds the request, which a
    slow rank delays) can leave the body never started: Starlette cancels the response as soon
    as it sees the disconnect. The request then ran to its end for nobody. Now a stream closes
    when the HTTP exchange that opened it ends, however it ends."""
    async with serving_cluster(2, tokens=100_000, token_interval=0.002) as cluster:
        app = FastAPI()
        app.add_middleware(ClosesStreams)

        @app.post("/chat")
        async def _chat() -> Response:  # pyright: ignore[reportUnusedFunction]
            stream = await cluster.clients[API_NODE].open(
                cluster.task(), cluster.instance
            )
            close_with_exchange(stream)
            # The body that would read the stream never starts
            return Response(status_code=204)

        async with http_server(app) as url, httpx.AsyncClient() as http:
            assert (await http.post(f"{url}/chat")).status_code == 204
        await cluster.settled(
            lambda: all(not tasks for tasks in cluster.in_progress().values()),
            timeout=10,
            what="dropped on every rank",
        )


async def test_a_node_without_the_api_still_takes_the_clusters_requests_for_its_models() -> (
    None
):
    """A node started with --no-api serves the cluster's routes alone on the API port (and
    /node_id, which other nodes probe), so the models it runs can still be used: before, no
    other node could reach them. The public API, the dashboard and the API docs stay off."""
    async with serving_cluster(2) as cluster:
        port = free_port()
        node = cluster.rank_nodes[1]
        server = ClusterRoutesServer(
            node,
            listening_socket=listen_on_api_port(port),
            state=lambda: cluster.state,
            host=cluster.hosts[node],
        )
        task = cluster.task()
        async with anyio.create_task_group() as tg, httpx.AsyncClient() as http:
            tg.start_soon(server.run)
            url = f"http://127.0.0.1:{port}"
            with anyio.fail_after(5):
                while True:
                    with contextlib.suppress(httpx.TransportError):
                        node_id = cast(str, (await http.get(f"{url}/node_id")).json())
                        break
                    await anyio.sleep(0.02)
            assert node_id == node
            delivered = await http.post(
                f"{url}{DELIVER_PATH}", content=Delivery(task=task).model_dump_json()
            )
            assert delivered.status_code == 200
            for public in (
                "/v1/models",
                "/state",
                "/",
                "/docs",
                "/docs/oauth2-redirect",
                "/redoc",
                "/openapi.json",
            ):
                assert (await http.get(f"{url}{public}")).status_code == 404, public
            tg.cancel_scope.cancel()
        await cluster.settled(lambda: cluster.ranks[1].received == [task.task_id])


async def test_a_neighbour_on_the_clusters_network_that_isnt_a_device_of_it_is_refused() -> (
    None
):
    """Not loopback, on the same network as the cluster's devices, but no device's address: 403,
    and nothing reaches the runner"""
    async with serving_cluster(2) as cluster:
        app = app_for(cluster, cluster.rank_nodes[1])
        addresses = {
            interface.ip_address
            for info in cluster.state.node_network.values()
            for interface in info.interfaces
        }
        neighbour = "10.0.0.250"
        assert neighbour not in addresses
        stranger = TestClient(app, client=(neighbour, 50000))
        body = Delivery(task=cluster.task()).model_dump_json()
        assert stranger.post(DELIVER_PATH, content=body).status_code == 403
        await anyio.sleep(0.1)
        assert cluster.ranks[1].received == []
