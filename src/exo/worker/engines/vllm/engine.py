import itertools
import time
from collections import deque
from collections.abc import Generator, Iterator
from dataclasses import dataclass, field
from typing import BinaryIO

from mlx_lm.tokenizer_utils import TokenizerWrapper

from exo.shared.constants import EXO_MAX_CONCURRENT_REQUESTS
from exo.shared.types.chunks import ErrorChunk, GenerationChunk, PrefillProgressChunk
from exo.shared.types.common import ModelId
from exo.shared.types.events import ChunkGenerated, Event
from exo.shared.types.tasks import (
    CANCEL_ALL_TASKS,
    GenerationTask,
    TaskId,
    TextGeneration,
)
from exo.shared.types.text_generation import TextGenerationTaskParams
from exo.shared.types.worker.runner_response import (
    CancelledResponse,
    FinishedResponse,
    GenerationResponse,
)
from exo.utils.channels import MpReceiver, MpSender
from exo.worker.disaggregated.server import PrefillJob
from exo.worker.engines.base import Engine
from exo.worker.engines.vllm.prompt_format import format_vllm_prompt
from exo.worker.engines.vllm.vllm_generator import VllmBatchEngine
from exo.worker.runner.bootstrap import logger
from exo.worker.runner.llm_inference.model_output_parsers import (
    apply_all_parsers,
    map_responses_to_chunks,
)
from exo.worker.runner.llm_inference.tool_parsers import ToolParser


class GeneratorQueue[T]:
    def __init__(self) -> None:
        self._q = deque[T]()

    def push(self, t: T) -> None:
        self._q.append(t)

    def gen(self) -> Generator[T | None]:
        while True:
            if len(self._q) == 0:
                yield None
            else:
                yield self._q.popleft()


EXO_RUNNER_MUST_FAIL = "EXO RUNNER MUST FAIL"
EXO_RUNNER_MUST_TIMEOUT = "EXO RUNNER MUST TIMEOUT"


def _check_for_debug_prompts(task_params: TextGenerationTaskParams) -> None:
    """Keep the cheap debug prompt hooks without importing the MLX engine."""
    if len(task_params.input) == 0:
        return
    prompt = task_params.input[0].content
    if not prompt:
        return
    if EXO_RUNNER_MUST_FAIL in prompt:
        raise Exception("Artificial runner exception - for testing purposes only.")
    if EXO_RUNNER_MUST_TIMEOUT in prompt:
        time.sleep(100)


@dataclass(eq=False)
class VllmEngine(Engine):
    """Single-node vLLM implementation of the exo Engine interface.

    This intentionally duplicates the local orchestration from the MLX
    BatchGenerator instead of trying to share a batch abstraction too early.
    The vLLM-specific tokenization/sampling/stepping remains inside
    VllmBatchEngine.
    """

    tool_parser: ToolParser | None
    model_id: ModelId
    cancel_receiver: MpReceiver[TaskId]
    event_sender: MpSender[Event]
    _gen: VllmBatchEngine
    max_concurrent_requests: int = EXO_MAX_CONCURRENT_REQUESTS
    check_for_cancel_every: int = 50

    _cancelled_tasks: set[TaskId] = field(default_factory=set, init=False)
    _all_tasks: dict[TaskId, TextGeneration] = field(default_factory=dict, init=False)
    _queue: deque[TextGeneration] = field(default_factory=deque, init=False)
    _active_tasks: dict[
        TaskId,
        tuple[
            TextGeneration,
            GeneratorQueue[GenerationResponse],
            Iterator[GenerationChunk | None],
        ],
    ] = field(default_factory=dict, init=False)

    def warmup(self) -> None:
        self.check_for_cancel_every = self._gen.warmup()

    def submit(self, task: GenerationTask) -> None:
        assert isinstance(task, TextGeneration)
        self._cancelled_tasks.discard(CANCEL_ALL_TASKS)
        self._all_tasks[task.task_id] = task
        self._queue.append(task)

    def step(
        self,
    ) -> Iterator[
        tuple[TaskId, GenerationChunk | CancelledResponse | FinishedResponse]
    ]:
        self._collect_cancellations()
        output: list[
            tuple[TaskId, GenerationChunk | CancelledResponse | FinishedResponse]
        ] = list(self._apply_cancellations())

        while self._queue and len(self._active_tasks) < self.max_concurrent_requests:
            task = self._queue.popleft()
            if self.should_cancel(task.task_id):
                output.append((task.task_id, CancelledResponse()))
                self._all_tasks.pop(task.task_id, None)
                continue

            try:
                task_id, queue, output_generator = self._start_task(task)
            except Exception as e:
                self._send_error(task, e)
                self._all_tasks.pop(task.task_id, None)
                raise

            self._active_tasks[task_id] = (task, queue, output_generator)

        if not self._gen.has_work:
            return iter(output)

        results = self._gen.step()
        for task_id, response in results:
            if task_id not in self._active_tasks:
                logger.warning(f"{task_id=} not found in active vLLM tasks")
                continue

            task, queue, output_generator = self._active_tasks[task_id]
            queue.push(response)
            while (parsed := next(output_generator, None)) is not None:
                output.append((task.task_id, parsed))

            if response.finish_reason is not None:
                output.append((task.task_id, FinishedResponse()))
                del self._active_tasks[task_id]
                self._all_tasks.pop(task.task_id, None)

        return itertools.chain(output, self._apply_cancellations())

    def close(self) -> None:
        self._gen.close()

    def serve_prefill(self, job: PrefillJob, wfile: BinaryIO) -> None:
        raise NotImplementedError("vLLM serve_prefill is not supported yet")

    def _start_task(
        self, task: TextGeneration
    ) -> tuple[
        TaskId,
        GeneratorQueue[GenerationResponse],
        Iterator[GenerationChunk | None],
    ]:
        _check_for_debug_prompts(task.task_params)

        token_ids, prompt_text, _ = format_vllm_prompt(
            self._gen.engine, task.task_params
        )

        queue = GeneratorQueue[GenerationResponse]()
        if task.task_params.bench:
            output_generator: Iterator[GenerationChunk | None] = map(
                lambda r: map_responses_to_chunks(r, self.model_id), queue.gen()
            )
        else:
            output_generator = apply_all_parsers(
                queue.gen(),
                prompt_text,
                self.tool_parser,
                TokenizerWrapper(self._gen.engine.get_tokenizer()),
                self.model_id,
                task.task_params.tools,
            )

        check_for_cancel_every = max(self.check_for_cancel_every, 1)
        tokens_since_cancel_check = check_for_cancel_every

        def on_prefill_progress(processed: int, total: int) -> None:
            self._collect_cancellations()
            if self.should_cancel(task.task_id):
                self._cancelled_tasks.add(task.task_id)
            self.event_sender.send(
                ChunkGenerated(
                    command_id=task.command_id,
                    chunk=PrefillProgressChunk(
                        model=self.model_id,
                        processed_tokens=processed,
                        total_tokens=total,
                    ),
                )
            )

        def on_generation_token() -> None:
            nonlocal tokens_since_cancel_check
            tokens_since_cancel_check += 1
            if tokens_since_cancel_check >= check_for_cancel_every:
                tokens_since_cancel_check = 0
                self._collect_cancellations()
                if self.should_cancel(task.task_id):
                    self._cancelled_tasks.add(task.task_id)

        task_id = self._gen.submit(
            task_id=task.task_id,
            task_params=task.task_params,
            on_prefill_progress=on_prefill_progress,
            on_generation_token=on_generation_token,
            token_ids=token_ids,
        )
        return task_id, queue, output_generator

    def _collect_cancellations(self) -> None:
        for task_id in self.cancel_receiver.collect():
            if task_id == CANCEL_ALL_TASKS:
                self._cancelled_tasks.add(CANCEL_ALL_TASKS)
            elif task_id in self._all_tasks:
                self._cancelled_tasks.add(task_id)

    def _apply_cancellations(self) -> Iterator[tuple[TaskId, CancelledResponse]]:
        if not self._cancelled_tasks:
            return iter([])

        cancel_all = CANCEL_ALL_TASKS in self._cancelled_tasks
        results: list[tuple[TaskId, CancelledResponse]] = []
        task_ids_to_abort: list[TaskId] = []

        for task_id, (task, _, _) in list(self._active_tasks.items()):
            if cancel_all or task.task_id in self._cancelled_tasks:
                task_ids_to_abort.append(task_id)
                results.append((task.task_id, CancelledResponse()))
                del self._active_tasks[task_id]
                self._all_tasks.pop(task.task_id, None)

        if self._queue:
            kept_queue: deque[TextGeneration] = deque()
            for task in self._queue:
                if cancel_all or task.task_id in self._cancelled_tasks:
                    results.append((task.task_id, CancelledResponse()))
                    self._all_tasks.pop(task.task_id, None)
                else:
                    kept_queue.append(task)
            self._queue = kept_queue

        if task_ids_to_abort:
            self._gen.cancel(task_ids_to_abort)

        already_cancelled = {task_id for task_id, _ in results}
        for task_id in self._cancelled_tasks:
            if (
                task_id != CANCEL_ALL_TASKS
                and task_id in self._all_tasks
                and task_id not in already_cancelled
            ):
                results.append((task_id, CancelledResponse()))
                self._all_tasks.pop(task_id, None)

        self._cancelled_tasks.clear()
        return iter(results)

    def _send_error(self, task: TextGeneration, e: Exception) -> None:
        self.event_sender.send(
            ChunkGenerated(
                command_id=task.command_id,
                chunk=ErrorChunk(
                    model=self.model_id,
                    finish_reason="error",
                    error_message=str(e),
                ),
            )
        )
