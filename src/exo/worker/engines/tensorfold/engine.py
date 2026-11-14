import concurrent.futures
import queue
from collections import deque
from collections.abc import Iterable
from typing import BinaryIO, override

from anyio import ClosedResourceError, EndOfStream

from exo.shared.types.chunks import Chunk, ErrorChunk
from exo.shared.types.common import ModelId
from exo.shared.types.tasks import (
    CANCEL_ALL_TASKS,
    GenerationTask,
    TaskId,
    TextGeneration,
)
from exo.shared.types.text_generation import (
    InputMessage,
    InputMessageContent,
    TextGenerationTaskParams,
)
from exo.shared.types.worker.runner_response import CancelledResponse, FinishedResponse
from exo.utils.channels import MpReceiver
from exo.worker.disaggregated.server import PrefillRequest
from exo.worker.engines.base import Engine
from exo.worker.engines.tensorfold.capabilities import LoadedModel
from exo.worker.engines.tensorfold.process import NativeProcess
from exo.worker.engines.tensorfold.requests import (
    UnsupportedRequestError,
    invalid_request,
    request_body,
)
from exo.worker.engines.tensorfold.transport import StreamTransport

Result = tuple[TaskId, Chunk | CancelledResponse | FinishedResponse]


class TensorFoldEngine(Engine):
    supports_prefill = False

    def __init__(
        self,
        process: NativeProcess,
        loaded: LoadedModel,
        cancel_receiver: MpReceiver[TaskId],
    ):
        self._process = process
        self._loaded = loaded
        self._model_id = ModelId(loaded.id)
        self._cancel_receiver = cancel_receiver
        self._jobs: dict[TaskId, concurrent.futures.Future[None]] = {}
        self._immediate: deque[Result] = deque()
        self._closed = False
        self._transport = StreamTransport(
            process.socket_path, process.settings.request_timeout
        )

    @override
    def warmup(self) -> None:
        params = TextGenerationTaskParams(
            model=self._model_id,
            input=[InputMessage(role="user", content=InputMessageContent("Hi"))],
            max_output_tokens=1,
            temperature=0.0,
        )
        body = request_body(params, self._loaded)
        future = self._transport.schedule(self._transport.warmup(self._model_id, body))
        try:
            future.result(timeout=self._process.settings.request_timeout + 5.0)
        except BaseException:
            future.cancel()
            self.close()
            raise

    @override
    def submit(self, task: GenerationTask) -> None:
        if self._closed:
            raise RuntimeError("TensorFold engine is closed")
        if task.task_id in self._jobs:
            raise ValueError("TensorFold received a duplicate task")
        try:
            if not isinstance(task, TextGeneration):
                raise ValueError(
                    "TensorFold's EXO backend accepts text generation tasks only"
                )
            if len(self._jobs) >= 128:
                raise ValueError("TensorFold already has 128 outstanding requests")
            body = request_body(task.task_params, self._loaded)
        except ValueError as error:
            # A refusal the client can fix carries EXO's invalid-request sentence (HTTP 400)
            message = (
                invalid_request(str(error))
                if isinstance(error, UnsupportedRequestError)
                else str(error)
            )
            self._immediate.extend(
                [
                    (
                        task.task_id,
                        ErrorChunk(model=self._model_id, error_message=message),
                    ),
                    (task.task_id, FinishedResponse()),
                ]
            )
            return
        self._jobs[task.task_id] = self._transport.schedule(
            self._transport.generate(task.task_id, self._model_id, body)
        )

    def _cancel(self) -> list[Result]:
        try:
            requested = self._cancel_receiver.collect()
        except (ClosedResourceError, EndOfStream):
            requested = [CANCEL_ALL_TASKS]
        targets = set(self._jobs) if CANCEL_ALL_TASKS in requested else set(requested)
        output: list[Result] = []
        for task_id in targets:
            if (future := self._jobs.pop(task_id, None)) is not None:
                future.cancel()
                output.append((task_id, CancelledResponse()))
        return output

    @override
    def check_health(self) -> None:
        self._process.check_alive()
        self._transport.check_alive()

    @override
    def step(self) -> Iterable[Result]:
        self.check_health()
        output = self._cancel()
        output.extend(self._immediate)
        self._immediate.clear()
        for index in range(64):
            try:
                task_id, result = self._transport.events.get(
                    timeout=0.02 if index == 0 and not output else 0
                )
            except queue.Empty:
                break
            if task_id not in self._jobs:
                continue
            output.append((task_id, result))
            if isinstance(result, FinishedResponse):
                self._jobs.pop(task_id)
        for future in self._jobs.values():
            if future.done() and not future.cancelled():
                future.result()
        return output

    @override
    def serve_prefill(self, request: PrefillRequest, wfile: BinaryIO) -> None:
        raise ValueError("TensorFold does not export EXO disaggregated prefill state")

    @override
    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        for future in self._jobs.values():
            future.cancel()
        self._jobs.clear()
        try:
            self._transport.close()
        finally:
            self._process.close()
