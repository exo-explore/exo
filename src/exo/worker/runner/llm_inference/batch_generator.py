from collections import deque
from collections.abc import Generator
from dataclasses import dataclass, field

import mlx.core as mx
from mlx_lm.tokenizer_utils import TokenizerWrapper

from exo.shared.types.chunks import ErrorChunk, PrefillProgressChunk
from exo.shared.types.common import ModelId
from exo.shared.types.events import ChunkGenerated, Event
from exo.shared.types.mlx import Model
from exo.shared.types.tasks import TaskId, TextGeneration
from exo.shared.types.text_generation import TextGenerationTaskParams
from exo.shared.types.worker.runner_response import GenerationResponse
from exo.utils.channels import MpReceiver, MpSender
from exo.worker.engines.mlx.cache import KVPrefixCache
from exo.worker.engines.mlx.generator.generate import PrefillCancelled, mlx_generate
from exo.worker.engines.mlx.utils_mlx import (
    apply_chat_template,
    mx_any,
)

EXO_RUNNER_MUST_FAIL = "EXO RUNNER MUST FAIL"
EXO_RUNNER_MUST_OOM = "EXO RUNNER MUST OOM"
EXO_RUNNER_MUST_TIMEOUT = "EXO RUNNER MUST TIMEOUT"


def _check_for_debug_prompts(task_params: TextGenerationTaskParams) -> None:
    """Check for debug prompt triggers in the input."""
    import time

    from exo.worker.engines.mlx.utils_mlx import mlx_force_oom

    if len(task_params.input) == 0:
        return
    prompt = task_params.input[0].content
    if not prompt:
        return
    if EXO_RUNNER_MUST_FAIL in prompt:
        raise Exception("Artificial runner exception - for testing purposes only.")
    if EXO_RUNNER_MUST_OOM in prompt:
        mlx_force_oom()
    if EXO_RUNNER_MUST_TIMEOUT in prompt:
        time.sleep(100)


@dataclass(eq=False)
class BatchGenerator:
    model: Model
    tokenizer: TokenizerWrapper
    group: mx.distributed.Group | None
    kv_prefix_cache: KVPrefixCache | None
    model_id: ModelId
    device_rank: int
    cancel_receiver: MpReceiver[TaskId]
    cancelled_tasks: set[TaskId]
    event_sender: MpSender[Event]
    check_for_cancel_every: int

    _queue: deque[tuple[TextGeneration, MpSender[GenerationResponse]]] = field(
        default_factory=deque, init=False
    )
    _active: (
        tuple[
            TextGeneration,
            MpSender[GenerationResponse],
            Generator[GenerationResponse],
        ]
        | None
    ) = field(default=None, init=False)
    _pending_close: MpSender[GenerationResponse] | None = field(
        default=None, init=False
    )

    def submit(
        self,
        task: TextGeneration,
        sender: MpSender[GenerationResponse],
    ) -> None:
        self._queue.append((task, sender))
        if self._active is None:
            self._start_next()

    def step(self) -> None:
        if self._pending_close is not None:
            self._pending_close.close()
            self._pending_close = None

        if self._active is None:
            if self._queue:
                self._start_next()
            else:
                return

        if self._active is None:
            return

        task, sender, gen = self._active
        try:
            response = next(gen)
            sender.send(response)
        except (StopIteration, PrefillCancelled):
            self._pending_close = sender
            self._active = None
            if self._queue:
                self._start_next()
        except Exception as e:
            self._send_error(task, e)
            sender.close()
            self._active = None
            raise

    def _start_next(self) -> None:
        task, sender = self._queue.popleft()
        try:
            gen = self._build_generator(task)
        except Exception as e:
            self._send_error(task, e)
            sender.close()
            raise
        self._active = (task, sender, gen)

    def _send_error(self, task: TextGeneration, e: Exception) -> None:
        if self.device_rank == 0:
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

    def _build_generator(self, task: TextGeneration) -> Generator[GenerationResponse]:
        _check_for_debug_prompts(task.task_params)
        prompt = apply_chat_template(self.tokenizer, task.task_params)

        def on_prefill_progress(processed: int, total: int) -> None:
            if self.device_rank == 0:
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

        def distributed_prompt_progress_callback() -> None:
            self.cancelled_tasks.update(self.cancel_receiver.collect())
            want_to_cancel = (task.task_id in self.cancelled_tasks) or (
                TaskId("CANCEL_CURRENT_TASK") in self.cancelled_tasks
            )
            if mx_any(want_to_cancel, self.group):
                raise PrefillCancelled()

        tokens_since_cancel_check = self.check_for_cancel_every

        def on_generation_token() -> None:
            nonlocal tokens_since_cancel_check
            tokens_since_cancel_check += 1
            if tokens_since_cancel_check >= self.check_for_cancel_every:
                tokens_since_cancel_check = 0
                self.cancelled_tasks.update(self.cancel_receiver.collect())
                want_to_cancel = (task.task_id in self.cancelled_tasks) or (
                    TaskId("CANCEL_CURRENT_TASK") in self.cancelled_tasks
                )
                if mx_any(want_to_cancel, self.group):
                    raise PrefillCancelled()

        return mlx_generate(
            model=self.model,
            tokenizer=self.tokenizer,
            task=task.task_params,
            prompt=prompt,
            kv_prefix_cache=self.kv_prefix_cache,
            on_prefill_progress=on_prefill_progress,
            distributed_prompt_progress_callback=distributed_prompt_progress_callback,
            on_generation_token=on_generation_token,
            group=self.group,
        )
