"""The API node's side of serving a request, without the master.

The API node sends the request to the output node of an instance it chose (exo.shared.serving),
over HTTP, or in-process when the output node is this node. It waits for Accepted (every rank of
the instance holds the task) before answering its client, then reads the output from the same
response: TCP keeps it in order and complete. Closing the stream cancels the request.

A stream ends with an error when the output node goes away (the connection breaks), says
nothing for WEDGE_SECONDS (it is wedged, or the network between dropped it), or when a node of the
instance becomes unreachable.
"""

import contextlib
from collections.abc import AsyncGenerator, AsyncIterator, Callable, Mapping
from contextlib import AbstractAsyncContextManager
from dataclasses import dataclass, field
from typing import final

import anyio
import httpx
from anyio import BrokenResourceError, ClosedResourceError
from loguru import logger

from exo.routing.membership import STALL_THRESHOLD, Reachability
from exo.shared import serving
from exo.shared.models.model_cards import ModelId
from exo.shared.request_errors import (
    DEVICE_DISCONNECTED,
    DEVICE_NOT_RESPONDING,
    DEVICE_UNREACHABLE,
    INTERRUPTED,
    NOT_A_MEMBER,
)
from exo.shared.serving import STREAM_HEARTBEAT_INTERVAL, WEDGE_SECONDS
from exo.shared.types.chunks import Chunk, ErrorChunk
from exo.shared.types.common import CommandId, NodeId
from exo.shared.types.serving import (
    Accepted,
    Refused,
    ServeFrame,
    ServeRequest,
    serve_frames,
)
from exo.shared.types.state import State
from exo.shared.types.tasks import GenerationTask
from exo.shared.types.worker.instances import Instance, InstanceId
from exo.utils.channels import Receiver, Sender, channel
from exo.utils.task_group import TaskGroup
from exo.worker.serving import (
    CONNECT_TIMEOUT,
    CONNECTION_LIMITS,
    SERVE_PATH,
    RefusalError,
    ServingHost,
)


class UnroutableError(Exception):
    """No address of the output node answered: the request reached no one"""


@final
@dataclass(eq=False)
class ServeStream:
    """A request's output, as it streams from its output node"""

    task: GenerationTask
    nodes: frozenset[NodeId]
    _send: Sender[Chunk]
    _receive: Receiver[Chunk]
    _scope: anyio.CancelScope = field(default_factory=anyio.CancelScope)
    cancelled: bool = False
    # Why it was ended from outside, if it was
    ended_because: str | None = None
    # When its output node last said anything (a heartbeat counts)
    heard_at: float = field(default_factory=anyio.current_time)

    @property
    def command_id(self) -> CommandId:
        return self.task.command_id

    async def chunks(self) -> AsyncGenerator[Chunk, None]:
        """The request's chunks, until its output ends (or the stream does)"""
        try:
            with self._receive as chunks:
                async for chunk in chunks:
                    yield chunk
        finally:
            self.close()

    def cancel(self) -> None:
        """The client asked to cancel: end the stream quietly"""
        self.cancelled = True
        self.close()

    def end(self, error_message: str) -> None:
        """End the stream with an error for the client"""
        self.ended_because = error_message
        with contextlib.suppress(ClosedResourceError, BrokenResourceError):
            self._send.send_nowait(
                ErrorChunk(
                    model=ModelId(self.task.task_params.model),
                    error_message=error_message,
                )
            )
        self.close()

    def close(self) -> None:
        """Stop reading the request's output, which cancels it at its output node"""
        self._scope.cancel()
        self._send.close()


@final
class ServingClient:
    def __init__(
        self,
        node_id: NodeId,
        *,
        api_port: int,
        host: ServingHost | None,
        state: Callable[[], State],
        reachability: Reachability,
        # A request's frames from another node, raising UnroutableError if it reached no one:
        # over HTTP, unless a test wires nodes together in-process
        frames_from: Callable[
            [ServeRequest],
            AbstractAsyncContextManager[AsyncIterator[ServeFrame | None]],
        ]
        | None = None,
    ):
        self.node_id = node_id
        self._frames_from = frames_from or self._frames_over_http
        self._api_port = api_port
        self._host = host
        self._state = state
        self._reachability = reachability
        self._tg = TaskGroup()
        self._client: httpx.AsyncClient | None = None
        self._streams: dict[CommandId, ServeStream] = {}

    async def run(self) -> None:
        # A stream has no timeout of its own (see _end_silent_streams); connecting,
        # CONNECT_TIMEOUT. One connection per request in flight, with no cap (CONNECTION_LIMITS).
        async with httpx.AsyncClient(
            timeout=httpx.Timeout(None, connect=CONNECT_TIMEOUT),
            limits=CONNECTION_LIMITS,
            trust_env=False,
        ) as client:
            self._client = client
            async with self._tg as tg:
                tg.start_soon(self._end_streams_of_unreachable_nodes)
                tg.start_soon(self._end_silent_streams)
                await anyio.sleep_forever()

    def stream(self, command_id: CommandId) -> ServeStream | None:
        return self._streams.get(command_id)

    def streams(self) -> list[CommandId]:
        return list(self._streams)

    def in_flight(self) -> Mapping[InstanceId, int]:
        counts: dict[InstanceId, int] = {}
        for stream in self._streams.values():
            counts[stream.task.instance_id] = counts.get(stream.task.instance_id, 0) + 1
        return counts

    def routable(self, instance: Instance, *, images: bool) -> bool:
        """Whether this node can try to reach the instance's output node (it has an address
        of it, or is it). Whether its nodes are reachable is the API's check, on its membership
        view (exo.api.main: _can_serve, _reachable_model_readiness)."""
        target = serving.output_node(instance, images=images)
        if target == self.node_id and self._host is not None:
            return True
        return bool(
            serving.node_urls(self._state(), self.node_id, target, self._api_port)
        )

    async def open(self, task: GenerationTask, instance: Instance) -> ServeStream:
        """Send the request to its instance's output node, and return its stream once every
        rank holds it. Raises UnroutableError if the request reached no one, RefusalError if
        it was refused (nothing of it runs)."""
        target = serving.output_node(instance, images=serving.is_image_task(task))
        request = ServeRequest(target=target, task=task)
        send, receive = channel[Chunk]()
        stream = ServeStream(
            task=task,
            nodes=frozenset(instance.shard_assignments.node_to_runner),
            _send=send,
            _receive=receive,
        )
        accepted = anyio.Event()
        outcome: list[BaseException | None] = []
        self._streams[task.command_id] = stream
        self._tg.start_soon(self._pump, request, stream, accepted, outcome)
        try:
            await accepted.wait()
        except BaseException:
            # Its client went away while it was being delivered: cancel it on every rank
            stream.close()
            self._streams.pop(task.command_id, None)
            raise
        if outcome and outcome[0] is not None:
            self._streams.pop(task.command_id, None)
            raise outcome[0]
        return stream

    async def _pump(
        self,
        request: ServeRequest,
        stream: ServeStream,
        accepted: anyio.Event,
        outcome: list[BaseException | None],
    ) -> None:
        def settle(error: BaseException | None) -> None:
            if not accepted.is_set():
                outcome.append(error)
                accepted.set()

        try:
            with stream._scope:  # pyright: ignore[reportPrivateUsage]
                async with self._frames(request) as frames:
                    while True:
                        frame = await anext(frames, _END)
                        stream.heard_at = anyio.current_time()
                        match frame:
                            case _End():
                                settle(RefusalError(DEVICE_DISCONNECTED))
                                return
                            case None:
                                continue
                            case Accepted():
                                settle(None)
                            case Refused(message=message):
                                settle(RefusalError(message))
                                return
                            case chunk:
                                with contextlib.suppress(
                                    ClosedResourceError, BrokenResourceError
                                ):
                                    await stream._send.send(chunk)  # pyright: ignore[reportPrivateUsage]
        except (UnroutableError, RefusalError) as error:
            # Before any output: the request reached no one, or was refused
            settle(error)
        except (httpx.HTTPError, ValueError) as error:
            logger.warning(
                f"Request {request.task.command_id}: stream from {request.target} "
                f"broke ({type(error).__name__}: {error})"
            )
            settle(RefusalError(DEVICE_DISCONNECTED))
            stream.end(DEVICE_DISCONNECTED)
        except Exception as error:
            # One request's failure must not take the API's other requests with it
            logger.opt(exception=error).error(
                f"Request {request.task.command_id} failed"
            )
            settle(RefusalError(INTERRUPTED))
            stream.end(INTERRUPTED)
        finally:
            settle(RefusalError(stream.ended_because or DEVICE_DISCONNECTED))
            stream.close()
            self._streams.pop(request.task.command_id, None)

    @contextlib.asynccontextmanager
    async def _frames(
        self, request: ServeRequest
    ) -> AsyncIterator[AsyncIterator[ServeFrame | None]]:
        """The request's frames from its output node (None: a heartbeat)"""
        if request.target == self.node_id and self._host is not None:
            async with contextlib.aclosing(self._host.serve(request)) as frames:
                yield frames
            return
        async with self._frames_from(request) as frames:
            yield frames

    @contextlib.asynccontextmanager
    async def _frames_over_http(
        self, request: ServeRequest
    ) -> AsyncIterator[AsyncIterator[ServeFrame | None]]:
        assert self._client is not None
        body = request.model_dump_json()
        refusal: RefusalError | None = None
        for url in serving.node_urls(
            self._state(), self.node_id, request.target, self._api_port
        ):
            response = await self._send(request, f"{url}{SERVE_PATH}", body)
            if response is None:
                continue
            try:
                if response.status_code == 200:
                    yield _lines(response)
                    return
                # Not a node that serves this: refused as a stranger, running no models, or
                # another program on that address. Another address may reach it.
                detail = (await response.aread())[:200].decode(errors="replace")
                logger.warning(
                    f"Request {request.task.command_id}: {url} answered "
                    f"{response.status_code}: {detail}"
                )
                if response.status_code == 403:
                    refusal = RefusalError(NOT_A_MEMBER)
            finally:
                await response.aclose()
        raise refusal or UnroutableError(DEVICE_UNREACHABLE)

    async def _send(
        self, request: ServeRequest, url: str, body: str
    ) -> httpx.Response | None:
        """The output node's response, its body still to read; None if this address didn't
        connect. Sent once more on a fresh connection if the node closed the one it went on
        without answering: a node that was frozen or asleep past its server's keep-alive
        timeout closes idle connections as it wakes. Safe: it never read the request, and a
        node refuses a request it already has."""
        assert self._client is not None
        outgoing = self._client.build_request(
            "POST", url, content=body, headers={"content-type": "application/json"}
        )
        try:
            try:
                return await self._client.send(outgoing, stream=True)
            except httpx.RemoteProtocolError:
                logger.info(
                    f"Request {request.task.command_id}: {url} closed the connection "
                    "without answering; sending it again"
                )
            return await self._client.send(outgoing, stream=True)
        except (httpx.ConnectError, httpx.ConnectTimeout):
            logger.info(f"Request {request.task.command_id}: {url} didn't connect")
            return None

    async def _end_silent_streams(self) -> None:
        """End a stream whose output node has said nothing for WEDGE_SECONDS: it sends a
        heartbeat while it lives. Only this node's running time counts: a check late by more than
        STALL_THRESHOLD (the membership's rule) means this node itself was stopped, while what
        the output node sent waited in the socket, so the silence counts from then on. A timer
        on each read would fire as the node wakes, before it has read what is waiting."""
        checked_at = anyio.current_time()
        while True:
            await anyio.sleep(STREAM_HEARTBEAT_INTERVAL)
            now = anyio.current_time()
            this_node_was_stopped = (
                now - checked_at
                > STREAM_HEARTBEAT_INTERVAL + STALL_THRESHOLD.total_seconds()
            )
            checked_at = now
            for stream in list(self._streams.values()):
                if this_node_was_stopped:
                    stream.heard_at = now
                elif now - stream.heard_at >= WEDGE_SECONDS:
                    logger.warning(
                        f"Request {stream.command_id}: nothing from its output node for "
                        f"{WEDGE_SECONDS:.0f}s"
                    )
                    stream.end(DEVICE_NOT_RESPONDING)

    async def _end_streams_of_unreachable_nodes(self) -> None:
        with self._reachability.watch() as unreachable:
            async for node_id in unreachable:
                for stream in list(self._streams.values()):
                    if node_id in stream.nodes:
                        logger.warning(
                            f"Ending request {stream.command_id}: {node_id} can't be reached"
                        )
                        stream.end(DEVICE_UNREACHABLE)


@final
class _End:
    """The output node closed the stream"""


_END = _End()


async def _lines(response: httpx.Response) -> AsyncGenerator[ServeFrame | None, None]:
    async for line in response.aiter_lines():
        yield serve_frames.validate_json(line) if line.strip() else None
