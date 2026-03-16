import gc
import json
import re
import sys
import time
from collections import deque
from collections.abc import Callable, Generator, Iterable
from dataclasses import dataclass, field
from pathlib import Path

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
from exo.worker.engines.kv_cache import TorchKVCache
from exo.worker.engines.mlx.cache import KVPrefixCache
from exo.worker.engines.vllm.growable_cache import (
    _exo_prefix_cache_ref,
    _growable_model_runner_ref,
    patch_vllm,
)
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


def _build_layer_groups(kv_cache_config: object) -> list[int]:
    group_lookup: dict[str, int] = {}
    for group_idx, group_spec in enumerate(kv_cache_config.kv_cache_groups):  # type: ignore
        for layer_name in group_spec.layer_names:  # type: ignore
            group_lookup[layer_name] = group_idx  # type: ignore

    layer_to_group: list[int] = []
    for tensor_spec in kv_cache_config.kv_cache_tensors:  # type: ignore
        for name in tensor_spec.shared_by:  # type: ignore
            layer_to_group.append(group_lookup[name])  # type: ignore
    return layer_to_group


@dataclass
class _ActiveRequest:
    task: TextGeneration
    request_id: str
    prompt_token_count: int
    prompt_token_ids: list[int]
    queue: GeneratorQueue[GenerationResponse]
    parsed_gen: Generator[GenerationResponse | ToolCallResponse | None]
    prefill_done: bool = False
    prev_text: str = ""
    prev_token_count: int = 0
    start_time: float = field(default_factory=time.perf_counter)
    first_token_time: float | None = None


@dataclass(eq=False)
class VllmSequentialGenerator(InferenceGenerator):
    engine: LLMEngine
    model_id: ModelId
    tool_parser: ToolParser | None
    cancel_receiver: MpReceiver[TaskId]
    prefix_cache: KVPrefixCache

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
            sampling_params = make_vllm_sampling_params(self.engine, task.task_params, self.model_id)
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
                prompt_token_ids=token_ids,
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

        # --- Prefill phase: step engine until first token arrives ---
        if not active.prefill_done:
            while self.engine.has_unfinished_requests():
                outputs = self.engine.step()
                for output in outputs:
                    if output.request_id != active.request_id:
                        continue
                    if len(output.outputs[0].token_ids) > 0:
                        active.first_token_time = time.perf_counter()
                        active.prefill_done = True
                        self._save_prefix_cache(active)
                        break
                if active.prefill_done:
                    break
            if not active.prefill_done:
                self._active = None
                return [(active.task.task_id, Finished())]
            # Fall through to process the outputs from the final prefill step
        else:
            outputs = self.engine.step()
        finished = False
        results: list[
            tuple[
                TaskId,
                GenerationResponse | ToolCallResponse | Cancelled | Finished,
            ]
        ] = []

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

            tokenizer = self.engine.get_tokenizer()
            for i, token_id in enumerate(new_tokens):
                is_last = i == len(new_tokens) - 1
                token_text: str = tokenizer.decode([token_id])  # type: ignore[reportUnknownMemberType]
                active.queue.push(
                    GenerationResponse(
                        text=token_text,
                        token=token_id,
                        finish_reason=mapped_finish_reason if is_last and finished else None,
                        usage=finish_usage if is_last and finished else None,
                        stats=finish_stats if is_last and finished else None,
                    )
                )
                try:
                    parsed = next(active.parsed_gen)
                except StopIteration:
                    self.engine.abort_request([active.request_id])
                    results.append((active.task.task_id, Finished()))
                    self._active = None
                    return results
                if parsed is not None:
                    results.append((active.task.task_id, parsed))

        if finished:
            logger.info(f"vLLM generation done for request {active.request_id}")
            results.append((active.task.task_id, Finished()))
            self._active = None

        return results

    def _get_coordinator(self) -> object | None:
        if not hasattr(self, "_coordinator_cached"):
            try:
                engine_core = self.engine.engine_core.engine_core  # type: ignore
                self._coordinator_cached: object | None = engine_core.scheduler.kv_cache_manager.coordinator  # type: ignore
            except Exception:
                self._coordinator_cached = None
        return self._coordinator_cached

    def _get_kv_cache_config(self) -> object | None:
        if not hasattr(self, "_kv_cache_config_cached"):
            try:
                engine_core = self.engine.engine_core.engine_core  # type: ignore
                self._kv_cache_config_cached: object | None = engine_core.scheduler.kv_cache_manager.kv_cache_config  # type: ignore
            except Exception:
                self._kv_cache_config_cached = None
        return self._kv_cache_config_cached

    def _save_prefix_cache(self, active: _ActiveRequest) -> None:
        try:
            coordinator = self._get_coordinator()
            model_runner = _growable_model_runner_ref[0]
            kv_cache_config = self._get_kv_cache_config()
            if coordinator is None or model_runner is None or kv_cache_config is None:
                return

            internal_id: str | None = None
            for mgr in coordinator.single_type_managers:  # type: ignore
                for key in mgr.req_to_blocks:  # type: ignore
                    if str(key).startswith(active.request_id):  # type: ignore
                        internal_id = str(key)  # type: ignore
                        break
                if internal_id:
                    break
            if internal_id is None:
                return

            null_block = coordinator.block_pool.null_block  # type: ignore

            block_ids_per_group: list[list[int]] = []
            token_offset_per_group: list[int] = []
            for mgr in coordinator.single_type_managers:  # type: ignore
                blocks = mgr.req_to_blocks.get(internal_id)  # type: ignore
                if not blocks:
                    block_ids_per_group.append([])
                    token_offset_per_group.append(0)
                    continue
                block_size: int = mgr.block_size  # type: ignore
                num_leading_nulls = 0
                for b in blocks:  # type: ignore
                    if b is null_block or b.is_null:  # type: ignore
                        num_leading_nulls += 1
                    else:
                        break
                real_blocks = [b for b in blocks if b is not null_block and not b.is_null]  # type: ignore
                block_ids_per_group.append([b.block_id for b in real_blocks])  # type: ignore
                token_offset_per_group.append(num_leading_nulls * block_size)

            layer_to_group = _build_layer_groups(kv_cache_config)
            torch_cache = TorchKVCache.from_vllm_cache(
                model_runner.kv_caches,  # type: ignore
                block_ids_per_group,
                layer_to_group,
                active.prompt_token_count,
                token_offset_per_group,
            )
            self.prefix_cache.add_from_torch(active.prompt_token_ids, torch_cache)
        except Exception:
            logger.opt(exception=True).warning("Failed to save prefix cache")

    def close(self) -> None:
        del self.engine
        gc.collect()
        torch.cuda.empty_cache()
        if torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()


_EXO_MAX_CONCURRENT_VLLM_REQUESTS = 8


@dataclass(eq=False)
class VllmBatchGenerator(InferenceGenerator):
    engine: LLMEngine
    model_id: ModelId
    tool_parser: ToolParser | None
    cancel_receiver: MpReceiver[TaskId]
    prefix_cache: KVPrefixCache

    _cancelled_tasks: set[TaskId] = field(default_factory=set, init=False)
    _pending: deque[TextGeneration] = field(default_factory=deque, init=False)
    _active: dict[str, _ActiveRequest] = field(default_factory=dict, init=False)

    def warmup(self) -> None:
        tokenizer = self.engine.get_tokenizer()
        messages = [{"role": "user", "content": "Prompt to warm up the inference engine. Repeat this."}]
        prompt_text: str = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)  # type: ignore
        token_ids: list[int] = tokenizer.encode(prompt_text, add_special_tokens=False)  # type: ignore
        params = SamplingParams(max_tokens=50, detokenize=False)
        self.engine.add_request("warmup", {"prompt_token_ids": token_ids}, params)
        while self.engine.has_unfinished_requests():
            self.engine.step()
        logger.info("vLLM batch warmup complete")

    def submit(self, task: TextGeneration) -> None:
        self._cancelled_tasks.discard(CANCEL_ALL_TASKS)
        self._pending.append(task)

    def _check_cancellations(self) -> None:
        for task_id in self.cancel_receiver.collect():
            if task_id == CANCEL_ALL_TASKS:
                self._cancelled_tasks.add(CANCEL_ALL_TASKS)
            else:
                self._cancelled_tasks.add(task_id)

    def _start_request(self, task: TextGeneration) -> _ActiveRequest:
        token_ids, prompt_text, prompt_token_count = format_vllm_prompt(
            self.engine, task.task_params
        )
        logger.info(prompt_text)
        request_id = str(task.task_id)
        sampling_params = make_vllm_sampling_params(self.engine, task.task_params, self.model_id)
        self.engine.add_request(
            request_id,
            {"prompt_token_ids": token_ids},
            sampling_params,
        )
        tokenizer = self.engine.get_tokenizer()
        think_start: str | None = getattr(tokenizer, "think_start", None)
        think_end: str | None = getattr(tokenizer, "think_end", None)
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
        return _ActiveRequest(
            task=task,
            request_id=request_id,
            prompt_token_count=prompt_token_count,
            prompt_token_ids=token_ids,
            queue=queue,
            parsed_gen=parsed_gen,
        )

    def step(
        self,
    ) -> Iterable[
        tuple[TaskId, GenerationResponse | ToolCallResponse | Cancelled | Finished]
    ]:
        self._check_cancellations()

        results: list[
            tuple[TaskId, GenerationResponse | ToolCallResponse | Cancelled | Finished]
        ] = []

        for task_id in list(self._cancelled_tasks):
            rid = str(task_id)
            if rid in self._active:
                self.engine.abort_request([rid])
                del self._active[rid]
                self._cancelled_tasks.discard(task_id)
                results.append((task_id, Cancelled()))
            elif CANCEL_ALL_TASKS in self._cancelled_tasks:
                for rid, active in list(self._active.items()):
                    self.engine.abort_request([rid])
                    results.append((active.task.task_id, Cancelled()))
                self._active.clear()
                self._cancelled_tasks.discard(CANCEL_ALL_TASKS)
                break

        while self._pending and len(self._active) < _EXO_MAX_CONCURRENT_VLLM_REQUESTS:
            task = self._pending.popleft()
            if self.should_cancel(task.task_id):
                self._cancelled_tasks.discard(task.task_id)
                results.append((task.task_id, Cancelled()))
                continue
            active = self._start_request(task)
            self._active[active.request_id] = active

        if not self._active:
            return results

        outputs = self.engine.step()
        tokenizer = self.engine.get_tokenizer()

        for output in outputs:
            rid = output.request_id
            if rid not in self._active:
                continue
            active = self._active[rid]
            completion = output.outputs[0]
            new_token_count = len(completion.token_ids)
            new_tokens = completion.token_ids[active.prev_token_count:]
            finish_reason = completion.finish_reason

            active.prev_text = completion.text
            active.prev_token_count = new_token_count
            if active.first_token_time is None and new_tokens:
                active.first_token_time = time.perf_counter()
                if not active.prefill_done:
                    active.prefill_done = True
                    self._save_prefix_cache(active)

            finish_usage: Usage | None = None
            finish_stats: GenerationStats | None = None
            mapped_finish_reason: str | None = None
            finished = False
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
                    prompt_tps=active.prompt_token_count / prefill_elapsed if prefill_elapsed > 0 else 0.0,
                    generation_tps=new_token_count / decode_elapsed if decode_elapsed > 0 else 0.0,
                    prompt_tokens=active.prompt_token_count,
                    generation_tokens=new_token_count,
                    peak_memory_usage=Memory.from_bytes(
                        torch.cuda.max_memory_allocated()  # pyright: ignore[reportUnknownMemberType, reportUnknownArgumentType, reportAttributeAccessIssue]
                    ),
                )
                mapped_finish_reason = (
                    finish_reason if finish_reason in ("stop", "length", "content_filter") else "stop"
                )
                finished = True

            for i, token_id in enumerate(new_tokens):
                is_last = i == len(new_tokens) - 1
                token_text: str = tokenizer.decode([token_id])  # type: ignore[reportUnknownMemberType]
                active.queue.push(
                    GenerationResponse(
                        text=token_text,
                        token=token_id,
                        finish_reason=mapped_finish_reason if is_last and finished else None,
                        usage=finish_usage if is_last and finished else None,
                        stats=finish_stats if is_last and finished else None,
                    )
                )
                try:
                    parsed = next(active.parsed_gen)
                except StopIteration:
                    self.engine.abort_request([rid])
                    results.append((active.task.task_id, Finished()))
                    del self._active[rid]
                    break
                if parsed is not None:
                    results.append((active.task.task_id, parsed))
            else:
                if finished:
                    logger.info(f"vLLM generation done for request {rid}")
                    results.append((active.task.task_id, Finished()))
                    del self._active[rid]

        return results

    def _get_coordinator(self) -> object | None:
        if not hasattr(self, "_coordinator_cached"):
            try:
                engine_core = self.engine.engine_core.engine_core  # type: ignore
                self._coordinator_cached: object | None = engine_core.scheduler.kv_cache_manager.coordinator  # type: ignore
            except Exception:
                self._coordinator_cached = None
        return self._coordinator_cached

    def _get_kv_cache_config(self) -> object | None:
        if not hasattr(self, "_kv_cache_config_cached"):
            try:
                engine_core = self.engine.engine_core.engine_core  # type: ignore
                self._kv_cache_config_cached: object | None = engine_core.scheduler.kv_cache_manager.kv_cache_config  # type: ignore
            except Exception:
                self._kv_cache_config_cached = None
        return self._kv_cache_config_cached

    def _save_prefix_cache(self, active: _ActiveRequest) -> None:
        try:
            coordinator = self._get_coordinator()
            model_runner = _growable_model_runner_ref[0]
            kv_cache_config = self._get_kv_cache_config()
            if coordinator is None or model_runner is None or kv_cache_config is None:
                return
            internal_id: str | None = None
            for mgr in coordinator.single_type_managers:  # type: ignore
                for key in mgr.req_to_blocks:  # type: ignore
                    if str(key).startswith(active.request_id):  # type: ignore
                        internal_id = str(key)  # type: ignore
                        break
                if internal_id:
                    break
            if internal_id is None:
                return
            null_block = coordinator.block_pool.null_block  # type: ignore
            block_ids_per_group: list[list[int]] = []
            token_offset_per_group: list[int] = []
            for mgr in coordinator.single_type_managers:  # type: ignore
                blocks = mgr.req_to_blocks.get(internal_id)  # type: ignore
                if not blocks:
                    block_ids_per_group.append([])
                    token_offset_per_group.append(0)
                    continue
                block_size: int = mgr.block_size  # type: ignore
                num_leading_nulls = 0
                for b in blocks:  # type: ignore
                    if b is null_block or b.is_null:  # type: ignore
                        num_leading_nulls += 1
                    else:
                        break
                real_blocks = [b for b in blocks if b is not null_block and not b.is_null]  # type: ignore
                block_ids_per_group.append([b.block_id for b in real_blocks])  # type: ignore
                token_offset_per_group.append(num_leading_nulls * block_size)
            layer_to_group = _build_layer_groups(kv_cache_config)
            torch_cache = TorchKVCache.from_vllm_cache(
                model_runner.kv_caches,  # type: ignore
                block_ids_per_group,
                layer_to_group,
                active.prompt_token_count,
                token_offset_per_group,
            )
            self.prefix_cache.add_from_torch(active.prompt_token_ids, torch_cache)
        except Exception:
            logger.opt(exception=True).warning("Failed to save prefix cache")

    def close(self) -> None:
        for rid in list(self._active):
            self.engine.abort_request([rid])
        self._active.clear()
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

    from vllm.model_executor.model_loader import (
        weight_utils,  # pyright: ignore[reportMissingImports]
    )

    _monkey_patch_iterator(weight_utils, "safetensors_weights_iterator")
    _monkey_patch_iterator(weight_utils, "fastsafetensors_weights_iterator")

    import huggingface_hub  # pyright: ignore[reportMissingImports]
    _noop_metadata = lambda *_a, **_kw: None  # pyright: ignore[reportUnknownLambdaType]
    original_metadata = huggingface_hub.get_safetensors_metadata  # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType]
    huggingface_hub.get_safetensors_metadata = _noop_metadata  # pyright: ignore[reportAttributeAccessIssue]
    for mod in list(sys.modules.values()):
        if mod is None or mod is huggingface_hub:
            continue
        for attr in list(vars(mod)):
            if vars(mod)[attr] is original_metadata:
                setattr(mod, attr, _noop_metadata)


def load_vllm_engine(
    model_path: str,
    model_id: ModelId,
    trust_remote_code: bool,
    on_layer_loaded: Callable[[int, int], None] | None = None,
) -> tuple[LLMEngine, ToolParser | None, KVPrefixCache]:
    global _weight_loading_callback
    patch_vllm()
    _patch_weight_loading_progress()

    prefix_cache = KVPrefixCache(group=None)
    _exo_prefix_cache_ref[0] = prefix_cache

    engine_args = EngineArgs(
        model=model_path,
        served_model_name=str(model_id),
        gpu_memory_utilization=0.05,
        trust_remote_code=trust_remote_code,
        load_format="fastsafetensors",
        enable_prefix_caching=False,
        attention_backend="TRITON_ATTN",
        # torch.compile's piecewise compilation produces kernels that assume
        # KV cache blocks are written by previous compiled forward passes.
        # Our prefix cache writes KV externally via tensor indexing and skips
        # the forward pass, which leaves the compiled kernel's internal state
        # inconsistent with the actual block data, producing garbage output.
        enforce_eager=True,
        disable_log_stats=True,
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

    return engine, tool_parser, prefix_cache
