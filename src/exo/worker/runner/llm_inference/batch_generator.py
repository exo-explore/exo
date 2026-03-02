from abc import ABC, abstractmethod
from collections import deque
from collections.abc import Generator
from dataclasses import dataclass, field
from typing import cast

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


class InferenceGenerator(ABC):
    @abstractmethod
    def submit(
        self,
        task: TextGeneration,
        sender: MpSender[GenerationResponse],
    ) -> None: ...

    @abstractmethod
    def step(self) -> set[TaskId]: ...

    @abstractmethod
    def close(self) -> None: ...


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
class SequentialGenerator(InferenceGenerator):
    model: Model
    tokenizer: TokenizerWrapper
    group: mx.distributed.Group | None
    kv_prefix_cache: KVPrefixCache | None
    model_id: ModelId
    device_rank: int
    cancel_receiver: MpReceiver[TaskId]
    event_sender: MpSender[Event]
    check_for_cancel_every: int

    _cancelled_tasks: set[TaskId] = field(default_factory=set, init=False)
    _maybe_queue: list[tuple[TextGeneration, MpSender[GenerationResponse]]] = field(
        default_factory=list, init=False
    )
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
        self._cancelled_tasks.discard(TaskId("CANCEL_CURRENT_TASK"))
        self._maybe_queue.append((task, sender))

    def agree_on_tasks(self) -> None:
        """Agree between all ranks about the task ordering (some may have received in different order or not at all)."""
        agreed, different = mx_all_gather_tasks(
            [task for task, _ in self._maybe_queue], self.group
        )
        self._queue.extend(
            (task, sender) for task, sender in self._maybe_queue if task in agreed
        )
        self._maybe_queue = [
            (task, sender) for task, sender in self._maybe_queue if task in different
        ]

    def step(self) -> set[TaskId]:
        if self._pending_close is not None:
            self._pending_close.close()
            self._pending_close = None

        if self._active is None:
            self.agree_on_tasks()

            if self._queue:
                self._start_next()
            else:
                return self._cancelled_tasks

        if self._active is None:
            return self._cancelled_tasks

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
        return self._cancelled_tasks

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
            self._cancelled_tasks.update(self.cancel_receiver.collect())
            want_to_cancel = (task.task_id in self._cancelled_tasks) or (
                TaskId("CANCEL_CURRENT_TASK") in self._cancelled_tasks
            )
            if mx_any(want_to_cancel, self.group):
                raise PrefillCancelled()

            self.agree_on_tasks()

        tokens_since_cancel_check = self.check_for_cancel_every

        def on_generation_token() -> None:
            nonlocal tokens_since_cancel_check
            tokens_since_cancel_check += 1
            if tokens_since_cancel_check >= self.check_for_cancel_every:
                tokens_since_cancel_check = 0
                self._cancelled_tasks.update(self.cancel_receiver.collect())
                want_to_cancel = (task.task_id in self._cancelled_tasks) or (
                    TaskId("CANCEL_CURRENT_TASK") in self._cancelled_tasks
                )
                if mx_any(want_to_cancel, self.group):
                    raise PrefillCancelled()

                self.agree_on_tasks()

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

    def close(self) -> None:
        del self.model, self.tokenizer, self.group


def mx_all_gather_tasks(
    tasks: list[TextGeneration],
    group: mx.distributed.Group | None,
) -> tuple[list[TextGeneration], list[TextGeneration]]:
    def encode_task_id(task_id: TaskId) -> list[int]:
        utf8_task_id = task_id.encode()
        return [
            int.from_bytes(utf8_task_id[i : i + 1]) for i in range(len(utf8_task_id))
        ]

    def decode_task_id(encoded_task_id: list[int]) -> TaskId:
        return TaskId(
            bytes.decode(b"".join((x).to_bytes(length=1) for x in encoded_task_id))
        )

    uuid_byte_length = 36

    n_tasks = len(tasks)
    all_counts = cast(
        list[int],
        mx.distributed.all_gather(mx.array([n_tasks]), group=group).tolist(),
    )
    max_tasks = max(all_counts)
    world_size: int = 1 if group is None else group.size()

    if max_tasks == 0:
        return [], []

    padded = [encode_task_id(task.task_id) for task in tasks] + [
        [0] * uuid_byte_length
    ] * (max_tasks - n_tasks)
    gathered = cast(
        list[list[list[int]]],
        mx.distributed.all_gather(mx.array(padded), group=group)
        .reshape(world_size, max_tasks, -1)
        .tolist(),
    )
    all_task_ids: list[list[TaskId]] = [
        [decode_task_id(encoded_task_id) for encoded_task_id in rank_tasks[:count]]
        for rank_tasks, count in zip(gathered, all_counts, strict=True)
    ]

    agreed_ids: set[TaskId] = set(all_task_ids[0])
    for rank_tasks in all_task_ids[1:]:
        agreed_ids &= set(rank_tasks)

    local_tasks = {task.task_id: task for task in tasks}
    agreed = [local_tasks[tid] for tid in sorted(agreed_ids)]
    different = [task for task in tasks if task.task_id not in agreed_ids]
    return agreed, different
