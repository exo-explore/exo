"""The routes the nodes of a cluster call on each other to serve requests without the master
(exo.worker.serving), on the API's port. Only nodes of the cluster may call them: they hand tasks
straight to runners.

A node started with --no-api still serves these (ClusterRoutesServer), with /node_id that other
nodes probe to verify its addresses, but not the public API or the dashboard: requests for the
models it runs reach it here.
"""

import contextlib
import socket
from collections.abc import AsyncGenerator, Callable
from typing import Final, final

import anyio
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse
from loguru import logger

from exo.api.listening_socket import serve_on_socket
from exo.shared.serving import is_member_address
from exo.shared.types.common import CommandId, NodeId
from exo.shared.types.serving import Delivery, ServeRequest, serve_frames
from exo.shared.types.state import State
from exo.worker.serving import (
    DECLINED_HEADER,
    DECLINED_HEADER_VALUE,
    DECLINED_STATUS,
    DELIVER_PATH,
    SERVE_PATH,
    RefusalError,
    ServingHost,
)

SERVING_PATH: Final = "/_exo/serving"


@final
class ClusterRoutes:
    def __init__(
        self,
        *,
        state: Callable[[], State],
        host: ServingHost | None,
        # The requests this node's API streams to its clients
        streaming: Callable[[], list[CommandId]] = list,
    ):
        self._state = state
        self._host = host
        self._streaming = streaming

    def add_to(self, app: FastAPI) -> None:
        app.post(SERVE_PATH, response_model=None)(self.serve)
        app.post(DELIVER_PATH, response_model=None)(self.deliver)
        app.get(SERVING_PATH, response_model=None)(self.serving)

    def _member_only(self, request: Request) -> JSONResponse | None:
        """403 unless the caller is a node of the cluster: these routes hand tasks straight to
        runners, so anyone else on the network could hang or use a model"""
        client = request.client.host if request.client is not None else None
        if is_member_address(self._state(), client):
            return None
        logger.warning(
            f"Refusing {request.url.path} from {client}: not a cluster member"
        )
        return JSONResponse(
            status_code=403,
            content={"detail": "Only devices of the cluster may call this"},
        )

    async def serve(self, request: Request) -> Response:
        if (refusal := self._member_only(request)) is not None:
            return refusal
        if self._host is None:
            return JSONResponse(
                status_code=404, content={"detail": "This device runs no models"}
            )
        frames = self._host.serve(
            ServeRequest.model_validate_json(await request.body())
        )

        async def lines() -> AsyncGenerator[bytes, None]:
            async with contextlib.aclosing(frames) as each:
                async for frame in each:
                    yield (
                        b"\n"
                        if frame is None
                        else serve_frames.dump_json(frame) + b"\n"
                    )

        return StreamingResponse(lines(), media_type="application/x-ndjson")

    async def deliver(self, request: Request) -> Response:
        if (refusal := self._member_only(request)) is not None:
            return refusal
        if self._host is None:
            return JSONResponse(
                status_code=404, content={"detail": "This device runs no models"}
            )
        delivery = Delivery.model_validate_json(await request.body())
        try:
            await self._host.deliver(delivery)
        except RefusalError as refusal:
            return JSONResponse(
                status_code=DECLINED_STATUS,
                content={"detail": refusal.message},
                headers={DECLINED_HEADER: DECLINED_HEADER_VALUE},
            )
        return JSONResponse(status_code=200, content={"detail": "taken"})

    async def serving(self, request: Request) -> Response:
        """What this node is serving, for diagnosing a request that seems stuck: the requests
        it streams to its clients, those it streams from its runners, and the requests each
        of its runners holds"""
        if (refusal := self._member_only(request)) is not None:
            return refusal
        host = self._host
        return JSONResponse(
            content={
                "streaming": [str(command_id) for command_id in self._streaming()],
                "serving": [
                    str(task.command_id)
                    for task in (host.open_requests() if host is not None else [])
                ],
                "runners": host.runner_tasks() if host is not None else {},
            }
        )


@final
class ClusterRoutesServer:
    """The cluster's routes alone, on the API's port, for a node started with --no-api: no
    public API, dashboard or API docs"""

    def __init__(
        self,
        node_id: NodeId,
        *,
        listening_socket: socket.socket,
        state: Callable[[], State],
        host: ServingHost,
    ):
        self._listening_socket = listening_socket
        self.app = FastAPI(docs_url=None, redoc_url=None, openapi_url=None)
        self.app.get("/node_id")(lambda: node_id)
        ClusterRoutes(state=state, host=host).add_to(self.app)

    async def run(self) -> None:
        logger.info("Serving only the cluster's routes on the API port (--no-api)")
        shutdown = anyio.Event()
        async with anyio.create_task_group() as tg:
            tg.start_soon(serve_on_socket, self.app, self._listening_socket, shutdown)
            try:
                await anyio.sleep_forever()
            finally:
                shutdown.set()
