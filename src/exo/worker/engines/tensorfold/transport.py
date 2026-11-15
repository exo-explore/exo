import asyncio
import concurrent.futures
import queue
import threading
from collections.abc import Coroutine
from pathlib import Path
from typing import TypeVar

import httpx
from pydantic import ValidationError

from exo.shared.types.chunks import Chunk, EngineHeartbeatChunk, ErrorChunk
from exo.shared.types.common import ModelId
from exo.shared.types.tasks import TaskId
from exo.shared.types.worker.runner_response import FinishedResponse
from exo.worker.engines.tensorfold.capabilities import Health
from exo.worker.engines.tensorfold.protocol import StreamProtocol
from exo.worker.engines.tensorfold.requests import invalid_request
from exo.worker.engines.tensorfold.sse import EventDecoder, Heartbeat, ProtocolError
from exo.worker.engines.tensorfold.wire import ErrorFrame

Result = tuple[TaskId, Chunk | FinishedResponse]
_Result = TypeVar("_Result")


class StreamTransport:
    def __init__(self, socket_path: Path, request_timeout: float):
        self.events: queue.Queue[Result] = queue.Queue(maxsize=256)
        self._socket_path = socket_path
        self._request_timeout = request_timeout
        self._ready = threading.Event()
        self._loop: asyncio.AbstractEventLoop | None = None
        self._stop: asyncio.Event | None = None
        self._client: httpx.AsyncClient | None = None
        self._failure: BaseException | None = None
        self._thread = threading.Thread(
            target=self._run, name="tensorfold-streams", daemon=True
        )
        self._thread.start()
        if not self._ready.wait(timeout=10.0):
            raise RuntimeError("TensorFold stream transport did not start")
        self.check_alive()

    def _run(self) -> None:
        try:
            asyncio.run(self._serve())
        except BaseException as error:
            self._failure = error
            self._ready.set()

    async def _serve(self) -> None:
        self._loop = asyncio.get_running_loop()
        self._stop = asyncio.Event()
        transport = httpx.AsyncHTTPTransport(uds=str(self._socket_path))
        async with httpx.AsyncClient(
            transport=transport,
            base_url="http://localhost",
            timeout=httpx.Timeout(self._request_timeout, connect=5.0),
            trust_env=False,
        ) as client:
            self._client = client
            self._ready.set()
            await self._stop.wait()
            tasks = [
                task
                for task in asyncio.all_tasks()
                if task is not asyncio.current_task()
            ]
            for task in tasks:
                task.cancel()
            if tasks:
                await asyncio.gather(*tasks, return_exceptions=True)
        self._client = None

    def check_alive(self) -> None:
        if self._failure is not None:
            raise RuntimeError("TensorFold stream transport failed") from self._failure
        if not self._thread.is_alive():
            raise RuntimeError("TensorFold stream transport stopped")

    def schedule(
        self, coroutine: Coroutine[object, object, _Result]
    ) -> concurrent.futures.Future[_Result]:
        self.check_alive()
        assert self._loop is not None
        return asyncio.run_coroutine_threadsafe(coroutine, self._loop)

    async def _publish(self, result: Result) -> None:
        while True:
            try:
                self.events.put_nowait(result)
                return
            except queue.Full:
                await asyncio.sleep(0.005)

    async def generate(self, task_id: TaskId, model_id: ModelId, body: bytes) -> None:
        try:
            await self._stream(task_id, model_id, body, publish=True)
        except TimeoutError:
            await self._publish(
                (
                    task_id,
                    ErrorChunk(
                        model=model_id,
                        error_message=(
                            "TensorFold produced no output within its "
                            f"{self._request_timeout:g} s idle deadline"
                        ),
                    ),
                )
            )
        except (httpx.HTTPError, ProtocolError, ValidationError, UnicodeError) as error:
            await self._publish(
                (task_id, ErrorChunk(model=model_id, error_message=str(error)))
            )
        await self._publish((task_id, FinishedResponse()))

    async def warmup(self, model_id: ModelId, body: bytes) -> None:
        await self._stream(TaskId("tensorfold-warmup"), model_id, body, publish=False)

    async def _stream(
        self, task_id: TaskId, model_id: ModelId, body: bytes, *, publish: bool
    ) -> None:
        client = self._client
        assert client is not None
        decoder = EventDecoder()
        protocol = StreamProtocol(model_id)
        # The deadline restarts with each output event, so a long reply that keeps
        # producing finishes; heartbeats alone never extend it.
        loop = asyncio.get_running_loop()
        async with (
            asyncio.timeout(self._request_timeout) as idle,
            client.stream(
                "POST",
                "/v1/chat/completions",
                content=body,
                headers={
                    "Content-Type": "application/json",
                    "Accept": "text/event-stream",
                },
            ) as response,
        ):
            if response.status_code != 200:
                content = bytearray()
                async for part in response.aiter_bytes():
                    content.extend(part)
                    if len(content) > 65536:
                        raise ProtocolError("TensorFold error response exceeds 64 KiB")
                try:
                    failure = ErrorFrame.model_validate_json(bytes(content))
                except ValidationError:
                    raise ProtocolError(
                        f"TensorFold returned HTTP {response.status_code}"
                    ) from None
                if response.status_code == 400:
                    raise ProtocolError(invalid_request(failure.error.message))
                raise ProtocolError(failure.error.message)
            if "text/event-stream" not in response.headers.get("content-type", ""):
                raise ProtocolError("TensorFold did not return an event stream")
            async for part in response.aiter_bytes():
                for event in decoder.feed(part):
                    if isinstance(event, Heartbeat):
                        if publish:
                            await self._publish(
                                (task_id, EngineHeartbeatChunk(model=model_id))
                            )
                    else:
                        idle.reschedule(loop.time() + self._request_timeout)
                        for chunk in protocol.feed(event):
                            if publish:
                                await self._publish((task_id, chunk))
                if protocol.done:
                    break
        decoder.finish()
        if not protocol.done:
            raise ProtocolError("TensorFold disconnected before completing the request")
        health_response = await client.get("/health")
        health_response.raise_for_status()
        health = Health.model_validate_json(health_response.content)
        terminal = protocol.finish(health.memory.peak)
        if publish:
            await self._publish((task_id, terminal))

    def close(self) -> None:
        if (
            self._loop is not None
            and self._stop is not None
            and self._thread.is_alive()
        ):
            self._loop.call_soon_threadsafe(self._stop.set)
            self._thread.join(timeout=5.0)
            if self._thread.is_alive():
                raise RuntimeError("TensorFold stream transport could not be stopped")
