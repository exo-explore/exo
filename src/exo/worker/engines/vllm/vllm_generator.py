import gc
import json
import re
import sys
import time
from collections.abc import Callable, Generator
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
from exo.shared.types.text_generation import TextGenerationTaskParams
from exo.shared.types.worker.runner_response import GenerationResponse
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
class _EngineRequest:
    uid: int
    request_id: str
    prompt_token_count: int
    prompt_token_ids: list[int]
    prefill_done: bool = False
    prefill_steps: int = 0
    prev_text: str = ""
    prev_token_count: int = 0
    start_time: float = field(default_factory=time.perf_counter)
    first_token_time: float | None = None
    on_generation_token: Callable[[], None] | None = None
    on_prefill_progress: Callable[[int, int], None] | None = None


def _save_prefix_cache(
    engine: LLMEngine,
    prefix_cache: KVPrefixCache,
    request_id: str,
    prompt_token_ids: list[int],
    prompt_token_count: int,
) -> None:
    try:
        coordinator = None
        model_runner = _growable_model_runner_ref[0]
        kv_cache_config = None
        try:
            engine_core = engine.engine_core.engine_core  # type: ignore
            coordinator = engine_core.scheduler.kv_cache_manager.coordinator  # type: ignore
            kv_cache_config = engine_core.scheduler.kv_cache_manager.kv_cache_config  # type: ignore
        except Exception:
            pass
        if coordinator is None or model_runner is None or kv_cache_config is None:
            return

        internal_id: str | None = None
        for mgr in coordinator.single_type_managers:  # type: ignore
            for key in mgr.req_to_blocks:  # type: ignore
                if str(key).startswith(request_id):  # type: ignore
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
            prompt_token_count,
            token_offset_per_group,
        )
        prefix_cache.add_from_torch(prompt_token_ids, torch_cache)
    except Exception:
        logger.opt(exception=True).warning("Failed to save prefix cache")


def _build_generation_response(
    tokenizer: object,
    token_id: int,
    finish_reason: str | None,
    prompt_token_count: int,
    completion_tokens: int,
    start_time: float,
    first_token_time: float | None,
) -> GenerationResponse:
    token_text: str = tokenizer.decode([token_id])  # type: ignore[reportUnknownMemberType]
    finish_usage: Usage | None = None
    finish_stats: GenerationStats | None = None
    mapped_finish_reason: str | None = None
    if finish_reason:
        now = time.perf_counter()
        prefill_elapsed = (first_token_time or now) - start_time
        decode_elapsed = now - (first_token_time or now)
        finish_usage = Usage(
            prompt_tokens=prompt_token_count,
            completion_tokens=completion_tokens,
            total_tokens=prompt_token_count + completion_tokens,
            prompt_tokens_details=PromptTokensDetails(),
            completion_tokens_details=CompletionTokensDetails(),
        )
        finish_stats = GenerationStats(
            prompt_tps=prompt_token_count / prefill_elapsed if prefill_elapsed > 0 else 0.0,
            generation_tps=completion_tokens / decode_elapsed if decode_elapsed > 0 else 0.0,
            prompt_tokens=prompt_token_count,
            generation_tokens=completion_tokens,
            peak_memory_usage=Memory.from_bytes(
                torch.cuda.max_memory_allocated()  # pyright: ignore[reportUnknownMemberType, reportUnknownArgumentType, reportAttributeAccessIssue]
            ),
        )
        mapped_finish_reason = finish_reason if finish_reason in ("stop", "length", "content_filter") else "stop"
    return GenerationResponse(
        text=token_text,
        token=token_id,
        finish_reason=mapped_finish_reason,
        usage=finish_usage,
        stats=finish_stats,
    )


def vllm_generate(
    engine: LLMEngine,
    model_id: ModelId,
    task: TextGenerationTaskParams,
    prompt: str,
    prefix_cache: KVPrefixCache,
    on_prefill_progress: Callable[[int, int], None] | None = None,
    distributed_prompt_progress_callback: Callable[[], None] | None = None,
    on_generation_token: Callable[[], None] | None = None,
) -> Generator[GenerationResponse, None, None]:
    token_ids, prompt_text, prompt_token_count = format_vllm_prompt(engine, task)
    logger.info(prompt_text)
    request_id = f"vllm-seq-{time.monotonic_ns()}"
    sampling_params = make_vllm_sampling_params(engine, task, model_id)
    engine.add_request(request_id, {"prompt_token_ids": token_ids}, sampling_params)

    tokenizer = engine.get_tokenizer()
    max_batch_tokens: int = getattr(engine.model_config, "max_num_batched_tokens", 2048) or 2048  # type: ignore[reportUnknownMemberType]
    start_time = time.perf_counter()
    first_token_time: float | None = None
    prev_token_count = 0
    prefill_done = False
    prefill_steps = 0

    while engine.has_unfinished_requests():
        if distributed_prompt_progress_callback and not prefill_done:
            distributed_prompt_progress_callback()
        outputs = engine.step()

        for output in outputs:
            if output.request_id != request_id:
                continue
            completion = output.outputs[0]
            new_token_count = len(completion.token_ids)
            new_tokens = completion.token_ids[prev_token_count:]
            finish_reason = completion.finish_reason
            prev_token_count = new_token_count

            if not prefill_done and not new_tokens:
                prefill_steps += 1
                if on_prefill_progress:
                    on_prefill_progress(
                        min(prefill_steps * max_batch_tokens, prompt_token_count),
                        prompt_token_count,
                    )
                continue

            if not prefill_done and new_tokens:
                first_token_time = time.perf_counter()
                prefill_done = True
                _save_prefix_cache(engine, prefix_cache, request_id, token_ids, prompt_token_count)

            for i, token_id in enumerate(new_tokens):
                is_last = i == len(new_tokens) - 1
                if on_generation_token:
                    on_generation_token()
                yield _build_generation_response(
                    tokenizer, token_id,
                    finish_reason if is_last and finish_reason else None,
                    prompt_token_count, new_token_count,
                    start_time, first_token_time,
                )


def warmup_vllm_engine(engine: LLMEngine) -> None:
    tokenizer = engine.get_tokenizer()
    messages = [{"role": "user", "content": "Prompt to warm up the inference engine. Repeat this."}]
    prompt_text: str = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)  # type: ignore
    token_ids: list[int] = tokenizer.encode(prompt_text, add_special_tokens=False)  # type: ignore
    params = SamplingParams(max_tokens=50, detokenize=False)
    engine.add_request("warmup", {"prompt_token_ids": token_ids}, params)
    while engine.has_unfinished_requests():
        engine.step()
    logger.info("vLLM warmup complete")


@dataclass(eq=False)
class VllmBatchEngine:
    engine: LLMEngine
    model_id: ModelId
    prefix_cache: KVPrefixCache

    _active: dict[int, _EngineRequest] = field(default_factory=dict, init=False)
    _next_uid: int = field(default=0, init=False)

    @property
    def has_work(self) -> bool:
        return bool(self._active) or self.engine.has_unfinished_requests()

    def submit(
        self,
        task_params: TextGenerationTaskParams,
        prompt: str,
        on_prefill_progress: Callable[[int, int], None] | None = None,
        distributed_prompt_progress_callback: Callable[[], None] | None = None,
        on_generation_token: Callable[[], None] | None = None,
    ) -> int:
        token_ids, prompt_text, prompt_token_count = format_vllm_prompt(self.engine, task_params)
        logger.info(prompt_text)
        uid = self._next_uid
        self._next_uid += 1
        request_id = f"vllm-batch-{uid}"
        sampling_params = make_vllm_sampling_params(self.engine, task_params, self.model_id)
        self.engine.add_request(request_id, {"prompt_token_ids": token_ids}, sampling_params)
        self._active[uid] = _EngineRequest(
            uid=uid,
            request_id=request_id,
            prompt_token_count=prompt_token_count,
            prompt_token_ids=token_ids,
            on_generation_token=on_generation_token,
            on_prefill_progress=on_prefill_progress,
        )
        return uid

    def step(self) -> list[tuple[int, GenerationResponse]]:
        if not self.has_work:
            return []

        outputs = self.engine.step()
        tokenizer = self.engine.get_tokenizer()
        max_batch_tokens: int = getattr(self.engine.model_config, "max_num_batched_tokens", 2048) or 2048  # type: ignore[reportUnknownMemberType]
        results: list[tuple[int, GenerationResponse]] = []

        rid_to_uid = {req.request_id: uid for uid, req in self._active.items()}

        for output in outputs:
            uid = rid_to_uid.get(output.request_id)
            if uid is None:
                continue
            req = self._active[uid]
            completion = output.outputs[0]
            new_token_count = len(completion.token_ids)
            new_tokens = completion.token_ids[req.prev_token_count:]
            finish_reason = completion.finish_reason
            req.prev_token_count = new_token_count

            if not req.prefill_done and not new_tokens:
                req.prefill_steps += 1
                if req.on_prefill_progress:
                    req.on_prefill_progress(
                        min(req.prefill_steps * max_batch_tokens, req.prompt_token_count),
                        req.prompt_token_count,
                    )
                continue

            if not req.prefill_done and new_tokens:
                req.first_token_time = time.perf_counter()
                req.prefill_done = True
                _save_prefix_cache(
                    self.engine, self.prefix_cache,
                    req.request_id, req.prompt_token_ids, req.prompt_token_count,
                )

            for i, token_id in enumerate(new_tokens):
                is_last = i == len(new_tokens) - 1
                if req.on_generation_token:
                    req.on_generation_token()
                results.append((uid, _build_generation_response(
                    tokenizer, token_id,
                    finish_reason if is_last and finish_reason else None,
                    req.prompt_token_count, new_token_count,
                    req.start_time, req.first_token_time,
                )))

            if finish_reason:
                del self._active[uid]

        for req in self._active.values():
            if not req.prefill_done:
                req.prefill_steps += 1
                if req.on_prefill_progress:
                    req.on_prefill_progress(
                        min(req.prefill_steps * max_batch_tokens, req.prompt_token_count),
                        req.prompt_token_count,
                    )

        return results

    def cancel(self, uids: list[int]) -> None:
        rids = [self._active[uid].request_id for uid in uids if uid in self._active]
        if rids:
            self.engine.abort_request(rids)
        for uid in uids:
            self._active.pop(uid, None)

    def close(self) -> None:
        rids = [req.request_id for req in self._active.values()]
        if rids:
            self.engine.abort_request(rids)
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
    def _noop_metadata(*_a: object, **_kw: object) -> None: pass  # pyright: ignore[reportUnknownParameterType]
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
