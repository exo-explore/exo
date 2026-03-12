import gc
import os
import time
from collections import deque
from collections.abc import Iterable
from dataclasses import dataclass, field

os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"

import torch
from vllm.engine.arg_utils import EngineArgs
from vllm.v1.engine.llm_engine import LLMEngine

from exo.shared.types.api import (
    CompletionTokensDetails,
    GenerationStats,
    PromptTokensDetails,
    Usage,
)
from exo.shared.types.common import ModelId
from exo.shared.types.memory import Memory
from exo.shared.types.tasks import CANCEL_ALL_TASKS, TaskId, TextGeneration
from exo.shared.types.worker.runner_response import GenerationResponse
from exo.utils.channels import MpReceiver
from exo.worker.engines.vllm.growable_cache import patch_vllm
from exo.worker.engines.vllm.prompt_format import (
    format_vllm_prompt,
    make_vllm_sampling_params,
)
from exo.worker.runner.bootstrap import logger
from exo.worker.runner.llm_inference.batch_generator import (
    Cancelled,
    Finished,
    InferenceGenerator,
)
from exo.worker.runner.llm_inference.tool_parsers import ToolParser, infer_tool_parser


def _make_usage(prompt_tokens: int, completion_tokens: int) -> Usage:
    return Usage(
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=prompt_tokens + completion_tokens,
        prompt_tokens_details=PromptTokensDetails(),
        completion_tokens_details=CompletionTokensDetails(),
    )


@dataclass(eq=False)
class VllmGenerator(InferenceGenerator):
    engine: LLMEngine
    model_id: ModelId
    tool_parser: ToolParser | None
    cancel_receiver: MpReceiver[TaskId]

    _cancelled_tasks: set[TaskId] = field(default_factory=set, init=False)
    _queue: deque[TextGeneration] = field(default_factory=deque, init=False)
    _active: tuple[TextGeneration, str, int, str, int, float, float | None] | None = (
        field(default=None, init=False)
    )

    def warmup(self) -> None:
        pass

    def submit(self, task: TextGeneration) -> None:
        self._cancelled_tasks.discard(CANCEL_ALL_TASKS)
        self._queue.append(task)

    def _check_cancellations(self) -> None:
        for task_id in self.cancel_receiver.collect():
            if task_id == CANCEL_ALL_TASKS:
                self._cancelled_tasks.add(CANCEL_ALL_TASKS)
            else:
                self._cancelled_tasks.add(task_id)

    def step(
        self,
    ) -> Iterable[tuple[TaskId, GenerationResponse | Cancelled | Finished]]:
        self._check_cancellations()

        if self._active is None and not self._queue:
            return []

        if self._active is None:
            task = self._queue.popleft()
            if self.should_cancel(task.task_id):
                self._cancelled_tasks.discard(task.task_id)
                return [(task.task_id, Cancelled())]
            prompt, prompt_token_count = format_vllm_prompt(
                self.engine, task.task_params
            )
            request_id = str(task.task_id)
            sampling_params = make_vllm_sampling_params(self.engine, task.task_params)
            self.engine.add_request(request_id, prompt, sampling_params)
            self._active = (
                task,
                request_id,
                prompt_token_count,
                "",
                0,
                time.perf_counter(),
                None,
            )

        (
            task,
            request_id,
            prompt_token_count,
            prev_text,
            prev_token_count,
            start_time,
            first_token_time,
        ) = self._active

        if self.should_cancel(task.task_id):
            self._active = None
            self._cancelled_tasks.discard(task.task_id)
            return [(task.task_id, Cancelled())]

        if not self.engine.has_unfinished_requests():
            self._active = None
            return [(task.task_id, Finished())]

        outputs = self.engine.step()
        results: list[tuple[TaskId, GenerationResponse | Cancelled | Finished]] = []

        for output in outputs:
            if output.request_id != request_id:
                continue
            completion = output.outputs[0]
            new_text = completion.text[len(prev_text) :]
            new_token_count = len(completion.token_ids)
            new_tokens = completion.token_ids[prev_token_count:]
            finish_reason = completion.finish_reason

            prev_text = completion.text
            prev_token_count = new_token_count
            if first_token_time is None and new_text:
                first_token_time = time.perf_counter()
            self._active = (
                task,
                request_id,
                prompt_token_count,
                prev_text,
                prev_token_count,
                start_time,
                first_token_time,
            )

            if new_text:
                token_id = new_tokens[-1] if new_tokens else 0
                results.append(
                    (
                        task.task_id,
                        GenerationResponse(
                            text=new_text,
                            token=token_id,
                            finish_reason=None,
                            usage=None,
                        ),
                    )
                )

            if finish_reason:
                now = time.perf_counter()
                prefill_elapsed = (first_token_time or now) - start_time
                decode_elapsed = now - (first_token_time or now)
                usage = _make_usage(prompt_token_count, new_token_count)
                stats = GenerationStats(
                    prompt_tps=prompt_token_count / prefill_elapsed
                    if prefill_elapsed > 0
                    else 0.0,
                    generation_tps=new_token_count / decode_elapsed
                    if decode_elapsed > 0
                    else 0.0,
                    prompt_tokens=prompt_token_count,
                    generation_tokens=new_token_count,
                    peak_memory_usage=Memory.from_bytes(
                        torch.cuda.max_memory_allocated()  # pyright: ignore[reportUnknownMemberType, reportUnknownArgumentType, reportAttributeAccessIssue]
                    ),
                )
                results.append(
                    (
                        task.task_id,
                        GenerationResponse(
                            text="",
                            token=0,
                            finish_reason=finish_reason
                            if finish_reason in ("stop", "length", "content_filter")
                            else "stop",
                            usage=usage,
                            stats=stats,
                        ),
                    )
                )
                results.append((task.task_id, Finished()))
                self._active = None

        return results

    def close(self) -> None:
        del self.engine
        gc.collect()
        torch.cuda.empty_cache()
        if torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()


def load_vllm_engine(
    model_path: str,
    model_id: ModelId,
    trust_remote_code: bool,
) -> tuple[LLMEngine, ToolParser | None]:
    patch_vllm()

    engine_args = EngineArgs(
        model=model_path,
        served_model_name=str(model_id),
        gpu_memory_utilization=0.05,
        trust_remote_code=trust_remote_code,
    )

    engine = LLMEngine.from_engine_args(engine_args)

    tool_parser: ToolParser | None = None
    tokenizer = engine.get_tokenizer()
    chat_template = getattr(tokenizer, "chat_template", None)
    if isinstance(chat_template, str):
        tool_parser = infer_tool_parser(chat_template)
        if tool_parser:
            logger.info(
                f"inferred tool parser: {tool_parser.start_parsing} / {tool_parser.end_parsing}"
            )

    logger.info(f"vLLM engine loaded for {model_id}")

    return engine, tool_parser
