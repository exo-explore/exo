import gc
import json
import os
import re
import sys
import time
from collections import deque
from collections.abc import Callable, Generator, Iterable
from dataclasses import dataclass, field
from pathlib import Path

os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"

import torch
from vllm.engine.arg_utils import EngineArgs
from vllm.sampling_params import SamplingParams
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
from exo.shared.types.worker.runner_response import GenerationResponse, ToolCallResponse
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
    GeneratorQueue,
    InferenceGenerator,
)
from exo.worker.runner.llm_inference.model_output_parsers import apply_vllm_parsers
from exo.worker.runner.llm_inference.tool_parsers import ToolParser, infer_tool_parser

@dataclass
class _ActiveRequest:
    task: TextGeneration
    request_id: str
    prompt_token_count: int
    queue: GeneratorQueue[GenerationResponse]
    parsed_gen: Generator[GenerationResponse | ToolCallResponse | None]
    prev_text: str = ""
    prev_token_count: int = 0
    start_time: float = field(default_factory=time.perf_counter)
    first_token_time: float | None = None


@dataclass(eq=False)
class VllmGenerator(InferenceGenerator):
    engine: LLMEngine
    model_id: ModelId
    tool_parser: ToolParser | None
    cancel_receiver: MpReceiver[TaskId]

    _cancelled_tasks: set[TaskId] = field(default_factory=set, init=False)
    _pending: deque[TextGeneration] = field(default_factory=deque, init=False)
    _active: _ActiveRequest | None = field(default=None, init=False)

    def warmup(self) -> None:
        tokenizer = self.engine.get_tokenizer()
        messages = [{"role": "user", "content": "Prompt to warm up the inference engine. Repeat this."}]
        prompt_text: str = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)  # type: ignore
        token_ids: list[int] = tokenizer.encode(prompt_text, add_special_tokens=False)  # type: ignore
        params = SamplingParams(max_tokens=50, detokenize=False)
        self.engine.add_request("warmup", {"prompt_token_ids": token_ids}, params)
        tokens_generated = 0
        while self.engine.has_unfinished_requests():
            self.engine.step()
            tokens_generated += 1
        logger.info(f"vLLM warmup complete, generated {tokens_generated} tokens")

    def submit(self, task: TextGeneration) -> None:
        self._cancelled_tasks.discard(CANCEL_ALL_TASKS)
        self._pending.append(task)

    def _check_cancellations(self) -> None:
        for task_id in self.cancel_receiver.collect():
            if task_id == CANCEL_ALL_TASKS:
                self._cancelled_tasks.add(CANCEL_ALL_TASKS)
            else:
                self._cancelled_tasks.add(task_id)

    def step(
        self,
    ) -> Iterable[
        tuple[TaskId, GenerationResponse | ToolCallResponse | Cancelled | Finished]
    ]:
        self._check_cancellations()

        if self._active is None and not self._pending:
            return []

        tokenizer = self.engine.get_tokenizer()
        think_start: str | None = getattr(tokenizer, "think_start", None)
        think_end: str | None = getattr(tokenizer, "think_end", None)

        if self._active is None:
            task = self._pending.popleft()
            if self.should_cancel(task.task_id):
                self._cancelled_tasks.discard(task.task_id)
                return [(task.task_id, Cancelled())]
            token_ids, prompt_text, prompt_token_count = format_vllm_prompt(
                self.engine, task.task_params
            )
            logger.info(prompt_text)
            request_id = str(task.task_id)
            sampling_params = make_vllm_sampling_params(self.engine, task.task_params)
            self.engine.add_request(
                request_id,
                {"prompt_token_ids": token_ids},
                sampling_params,
            )

            queue: GeneratorQueue[GenerationResponse] = GeneratorQueue()
            parsed_gen = apply_vllm_parsers(
                queue.gen(),
                self.model_id,
                prompt_text,
                self.tool_parser,
                task.task_params.tools,
                think_start=think_start,
                think_end=think_end,
            )

            self._active = _ActiveRequest(
                task=task,
                request_id=request_id,
                prompt_token_count=prompt_token_count,
                queue=queue,
                parsed_gen=parsed_gen,
            )

        active = self._active

        if self.should_cancel(active.task.task_id):
            self._active = None
            self._cancelled_tasks.discard(active.task.task_id)
            return [(active.task.task_id, Cancelled())]

        if not self.engine.has_unfinished_requests():
            self._active = None
            return [(active.task.task_id, Finished())]

        outputs = self.engine.step()
        finished = False

        for output in outputs:
            if output.request_id != active.request_id:
                continue
            completion = output.outputs[0]
            new_token_count = len(completion.token_ids)
            new_tokens = completion.token_ids[active.prev_token_count :]

            new_text = completion.text[len(active.prev_text) :]
            finish_reason = completion.finish_reason

            active.prev_text = completion.text
            active.prev_token_count = new_token_count
            if active.first_token_time is None and new_text:
                active.first_token_time = time.perf_counter()

            finish_usage: Usage | None = None
            finish_stats: GenerationStats | None = None
            mapped_finish_reason: str | None = None
            if finish_reason:
                now = time.perf_counter()
                prefill_elapsed = (active.first_token_time or now) - active.start_time
                decode_elapsed = now - (active.first_token_time or now)
                finish_usage = Usage(
                    prompt_tokens=active.prompt_token_count,
                    completion_tokens=new_token_count,
                    total_tokens=active.prompt_token_count + new_token_count,
                    prompt_tokens_details=PromptTokensDetails(),
                    completion_tokens_details=CompletionTokensDetails(),
                )
                finish_stats = GenerationStats(
                    prompt_tps=active.prompt_token_count / prefill_elapsed
                    if prefill_elapsed > 0
                    else 0.0,
                    generation_tps=new_token_count / decode_elapsed
                    if decode_elapsed > 0
                    else 0.0,
                    prompt_tokens=active.prompt_token_count,
                    generation_tokens=new_token_count,
                    peak_memory_usage=Memory.from_bytes(
                        torch.cuda.max_memory_allocated()  # pyright: ignore[reportUnknownMemberType, reportUnknownArgumentType, reportAttributeAccessIssue]
                    ),
                )
                mapped_finish_reason = (
                    finish_reason
                    if finish_reason in ("stop", "length", "content_filter")
                    else "stop"
                )
                finished = True

            for i, token_id in enumerate(new_tokens):
                is_last = i == len(new_tokens) - 1
                active.queue.push(
                    GenerationResponse(
                        text=new_text if is_last else "",
                        token=token_id,
                        finish_reason=mapped_finish_reason if is_last and finished else None,
                        usage=finish_usage if is_last and finished else None,
                        stats=finish_stats if is_last and finished else None,
                    )
                )

        results: list[
            tuple[
                TaskId,
                GenerationResponse | ToolCallResponse | Cancelled | Finished,
            ]
        ] = []
        parser_alive = False
        for parsed in active.parsed_gen:
            parser_alive = True
            if parsed is None:
                break
            results.append((active.task.task_id, parsed))

        if not parser_alive:
            self.engine.abort_request([active.request_id])
            results.append((active.task.task_id, Finished()))
            self._active = None
            return results

        if finished:
            results.append((active.task.task_id, Finished()))
            self._active = None

        return results

    def close(self) -> None:
        del self.engine
        gc.collect()
        torch.cuda.empty_cache()
        if torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()


_weight_loading_callback: Callable[[int, int], None] | None = None
_weight_loading_patched = False

_LAYER_INDEX_PATTERN = re.compile(r"\.layers\.(\d+)\.")


def _get_total_layers(model_dir: Path) -> int:
    config_file = model_dir / "config.json"
    if config_file.exists():
        with open(config_file) as f:
            config: dict[str, object] = json.load(f)
        num = config.get("num_hidden_layers")
        if isinstance(num, int) and num > 0:
            return num
    return 1


def _wrap_weights_iterator(original: Callable[..., Generator[tuple[str, "torch.Tensor"], None, None]]) -> Callable[..., Generator[tuple[str, "torch.Tensor"], None, None]]:  # pyright: ignore[reportUnknownParameterType]
    def patched(hf_weights_files: list[str], *args: object, **kwargs: object) -> Generator[tuple[str, "torch.Tensor"], None, None]:  # pyright: ignore[reportUnknownParameterType]
        callback = _weight_loading_callback
        if callback is not None and hf_weights_files:
            model_dir = Path(hf_weights_files[0]).parent
            total_layers = _get_total_layers(model_dir)
            seen_layers: set[int] = set()
            last_reported = 0
            for name, tensor in original(hf_weights_files, *args, **kwargs):  # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType]
                yield name, tensor  # pyright: ignore[reportUnknownArgumentType]
                match = _LAYER_INDEX_PATTERN.search(name)
                if match:
                    seen_layers.add(int(match.group(1)))
                current = len(seen_layers)
                if current > last_reported:
                    callback(current, total_layers)
                    last_reported = current
            callback(total_layers, total_layers)
        else:
            yield from original(hf_weights_files, *args, **kwargs)  # pyright: ignore[reportUnknownMemberType]
    return patched


def _monkey_patch_iterator(weight_utils: object, attr_name: str) -> None:  # pyright: ignore[reportUnknownParameterType]
    original = getattr(weight_utils, attr_name, None)
    if original is None:
        return
    patched = _wrap_weights_iterator(original)  # pyright: ignore[reportUnknownArgumentType]
    setattr(weight_utils, attr_name, patched)
    for mod in list(sys.modules.values()):
        if mod is None or mod is weight_utils:
            continue
        for name in list(vars(mod)):
            if vars(mod)[name] is original:
                setattr(mod, name, patched)


def _patch_weight_loading_progress() -> None:
    global _weight_loading_patched
    if _weight_loading_patched:
        return
    _weight_loading_patched = True

    from vllm.model_executor.model_loader import weight_utils  # pyright: ignore[reportMissingImports]

    _monkey_patch_iterator(weight_utils, "safetensors_weights_iterator")
    _monkey_patch_iterator(weight_utils, "fastsafetensors_weights_iterator")


def load_vllm_engine(
    model_path: str,
    model_id: ModelId,
    trust_remote_code: bool,
    on_layer_loaded: Callable[[int, int], None] | None = None,
) -> tuple[LLMEngine, ToolParser | None]:
    global _weight_loading_callback
    patch_vllm()
    _patch_weight_loading_progress()

    engine_args = EngineArgs(
        model=model_path,
        served_model_name=str(model_id),
        gpu_memory_utilization=0.05,
        trust_remote_code=trust_remote_code,
        load_format="fastsafetensors",
    )

    _weight_loading_callback = on_layer_loaded
    try:
        engine = LLMEngine.from_engine_args(engine_args)
    finally:
        _weight_loading_callback = None

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
