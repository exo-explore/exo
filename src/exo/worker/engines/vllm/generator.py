import gc
import json
import math
import re
import sys
import time
from collections.abc import Callable, Generator
from dataclasses import dataclass, field
from typing import cast

import torch
from vllm.config import CompilationConfig
from vllm.config.compilation import CompilationMode, CUDAGraphMode
from vllm.engine.arg_utils import EngineArgs
from vllm.entrypoints.chat_utils import (
    ChatCompletionMessageParam,
    CustomChatCompletionMessageParam,
)
from vllm.outputs import RequestOutput
from vllm.sampling_params import SamplingParams
from vllm.tokenizers import TokenizerLike
from vllm.v1.attention.backends.registry import AttentionBackendEnum
from vllm.v1.engine.llm_engine import LLMEngine
from vllm.v1.kv_cache_interface import KVCacheConfig

from exo.api.types import (
    CompletionTokensDetails,
    GenerationStats,
    PromptTokensDetails,
    Usage,
)
from exo.download.download_utils import build_model_path
from exo.shared.types.common import ModelId
from exo.shared.types.memory import Memory
from exo.shared.types.tasks import TaskId
from exo.shared.types.text_generation import TextGenerationTaskParams
from exo.shared.types.worker.runner_response import GenerationResponse
from exo.worker.runner.bootstrap import logger
from exo.worker.runner.llm_inference.tool_parsers import ToolParser, infer_tool_parser


@dataclass
class _EngineRequest:
    request_id: str
    prompt_token_count: int
    prefill_done: bool = False
    prefill_steps: int = 0
    prev_text: str = ""
    prev_token_count: int = 0
    start_time: float = field(default_factory=time.perf_counter)
    first_token_time: float | None = None
    on_generation_token: Callable[[], None] | None = None
    on_prefill_progress: Callable[[int, int], None] | None = None


def _stop_token_ids(tokenizer: TokenizerLike, model_id: ModelId) -> set[int]:
    from exo.worker.engines.mlx.utils_mlx import get_eos_token_ids_for_model

    ids: set[int] = set()
    eos_id = getattr(tokenizer, "eos_token_id", None)
    if eos_id is not None:
        ids.add(eos_id)  # pyright: ignore[reportAny]
    extra = get_eos_token_ids_for_model(model_id)
    if extra:
        ids.update(extra)
    return ids


def _build_generation_response(
    tokenizer: TokenizerLike,
    token_id: int,
    finish_reason: str | None,
    prompt_token_count: int,
    completion_tokens: int,
    start_time: float,
    first_token_time: float | None,
    suppress_text: bool = False,
) -> GenerationResponse:
    token_text: str = "" if suppress_text else tokenizer.decode([token_id])
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
            prompt_tps=prompt_token_count / prefill_elapsed
            if prefill_elapsed > 0
            else 0.0,
            generation_tps=completion_tokens / decode_elapsed
            if decode_elapsed > 0
            else 0.0,
            prompt_tokens=prompt_token_count,
            generation_tokens=completion_tokens,
            peak_memory_usage=Memory.from_bytes(torch.cuda.max_memory_allocated()),
        )
        mapped_finish_reason = (
            finish_reason
            if finish_reason in ("stop", "length", "content_filter")
            else "stop"
        )
    return GenerationResponse(
        text=token_text,
        token=token_id,
        finish_reason=mapped_finish_reason,
        usage=finish_usage,
        stats=finish_stats,
    )


def warmup_vllm_engine(engine: LLMEngine) -> int:
    tokenizer = engine.get_tokenizer()
    messages = [
        cast(
            ChatCompletionMessageParam,
            CustomChatCompletionMessageParam(
                role="user",
                content="Prompt to warm up the inference engine. Repeat this.",
            ),
        )
    ]
    prompt_text: str | list[int] = tokenizer.apply_chat_template(  # pyright: ignore[reportUnknownMemberType]
        messages, tokenize=False, add_generation_prompt=True
    )
    if isinstance(prompt_text, list):
        token_ids = prompt_text
    else:
        token_ids: list[int] = tokenizer.encode(prompt_text, add_special_tokens=False)

    params = SamplingParams(max_tokens=50, detokenize=False)
    engine.add_request("warmup", {"prompt_token_ids": token_ids}, params)
    t = time.monotonic()
    tokens_generated = 0
    while engine.has_unfinished_requests():
        engine.step()
        tokens_generated += 1
    elapsed = max(time.monotonic() - t, 0.001)
    check_for_cancel_every = min(math.ceil(tokens_generated / elapsed), 100)
    logger.info(
        f"vLLM warmup complete, check_for_cancel_every={check_for_cancel_every}"
    )
    return check_for_cancel_every


@dataclass(eq=False)
class VllmBatchEngine:
    engine: LLMEngine
    model_id: ModelId

    _active: dict[TaskId, _EngineRequest] = field(default_factory=dict, init=False)

    def warmup(self) -> int:
        return warmup_vllm_engine(self.engine)

    @property
    def has_work(self) -> bool:
        return bool(self._active) or self.engine.has_unfinished_requests()

    def submit(
        self,
        task_id: TaskId,
        task_params: TextGenerationTaskParams,
        token_ids: list[int],
        on_prefill_progress: Callable[[int, int], None] | None = None,
        on_generation_token: Callable[[], None] | None = None,
    ) -> TaskId:
        from exo.worker.engines.vllm.prompt_format import make_vllm_sampling_params

        sampling_params = make_vllm_sampling_params(
            self.engine, task_params, self.model_id
        )
        self.engine.add_request(
            task_id, {"prompt_token_ids": token_ids}, sampling_params
        )
        self._active[task_id] = _EngineRequest(
            request_id=task_id,
            prompt_token_count=len(token_ids),
            on_generation_token=on_generation_token,
            on_prefill_progress=on_prefill_progress,
        )
        return task_id

    def step(self) -> list[tuple[TaskId, GenerationResponse]]:
        if not self.has_work:
            return []

        outputs = self.engine.step()
        tokenizer = self.engine.get_tokenizer()
        stop_ids = _stop_token_ids(tokenizer, self.model_id)
        max_batch_tokens: int = (
            getattr(self.engine.model_config, "max_num_batched_tokens", 2048) or 2048
        )
        results: list[tuple[TaskId, GenerationResponse]] = []

        for output in outputs:
            # todo: PoolingRequestOutputs
            assert isinstance(output, RequestOutput)
            task_id = TaskId(output.request_id)
            if task_id not in self._active:
                continue
            req = self._active[task_id]
            completion = output.outputs[0]
            new_token_count = len(completion.token_ids)
            new_tokens = completion.token_ids[req.prev_token_count :]
            finish_reason = completion.finish_reason
            req.prev_token_count = new_token_count

            if not req.prefill_done and not new_tokens:
                req.prefill_steps += 1
                if req.on_prefill_progress:
                    req.on_prefill_progress(
                        min(
                            req.prefill_steps * max_batch_tokens, req.prompt_token_count
                        ),
                        req.prompt_token_count,
                    )
                continue

            if not req.prefill_done and new_tokens:
                req.first_token_time = time.perf_counter()
                req.prefill_done = True

            for i, token_id in enumerate(new_tokens):
                is_last = i == len(new_tokens) - 1
                is_final_stop = is_last and finish_reason and token_id in stop_ids
                if req.on_generation_token:
                    req.on_generation_token()
                results.append(
                    (
                        task_id,
                        _build_generation_response(
                            tokenizer,
                            token_id,
                            finish_reason if is_last and finish_reason else None,
                            req.prompt_token_count,
                            new_token_count,
                            req.start_time,
                            req.first_token_time,
                            suppress_text=bool(is_final_stop),
                        ),
                    )
                )

            if finish_reason:
                del self._active[task_id]

        for req in self._active.values():
            if not req.prefill_done:
                req.prefill_steps += 1
                if req.on_prefill_progress:
                    req.on_prefill_progress(
                        min(
                            req.prefill_steps * max_batch_tokens, req.prompt_token_count
                        ),
                        req.prompt_token_count,
                    )

        return results

    def cancel(self, task_ids: list[TaskId]) -> None:
        to_abort = [str(tid) for tid in task_ids if tid in self._active]
        if to_abort:
            self.engine.abort_request(to_abort)
        for tid in task_ids:
            self._active.pop(tid, None)

    def close(self) -> None:
        if not hasattr(self, "engine"):
            return
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


def get_weight_loading_callback() -> Callable[[int, int], None] | None:
    return _weight_loading_callback


def set_weight_loading_callback(cb: Callable[[int, int], None] | None) -> None:
    global _weight_loading_callback
    _weight_loading_callback = cb


_LAYER_INDEX_PATTERN = re.compile(r"\.layers\.(\d+)\.")
_n_layers: int = 1


def get_n_layers() -> int:
    return _n_layers


def set_n_layers(n: int) -> None:
    global _n_layers
    _n_layers = n


def _wrap_weights_iterator(
    original: Callable[..., Generator[tuple[str, "torch.Tensor"], None, None]],
) -> Callable[..., Generator[tuple[str, "torch.Tensor"], None, None]]:
    def patched(
        hf_weights_files: list[str], *args: object, **kwargs: object
    ) -> Generator[tuple[str, "torch.Tensor"], None, None]:
        callback = get_weight_loading_callback()
        if callback is not None and hf_weights_files:
            total_layers = get_n_layers()
            seen_layers: set[int] = set()
            last_reported = 0
            for name, tensor in original(hf_weights_files, *args, **kwargs):
                yield name, tensor
                match = _LAYER_INDEX_PATTERN.search(name)
                if match:
                    seen_layers.add(int(match.group(1)))
                current = len(seen_layers)
                if current > last_reported:
                    callback(current, total_layers)
                    last_reported = current
            callback(total_layers, total_layers)
        else:
            yield from original(hf_weights_files, *args, **kwargs)

    return patched


def _monkey_patch_iterator(weight_utils: object, attr_name: str) -> None:
    original = getattr(weight_utils, attr_name, None)
    if original is None:
        return
    patched = _wrap_weights_iterator(original)  # pyright: ignore[reportAny]
    setattr(weight_utils, attr_name, patched)
    for mod in list(sys.modules.values()):
        if mod is weight_utils:
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
        weight_utils,
    )

    _monkey_patch_iterator(weight_utils, "safetensors_weights_iterator")
    _monkey_patch_iterator(weight_utils, "fastsafetensors_weights_iterator")

    import huggingface_hub

    def _noop_metadata(*_a: object, **_kw: object) -> None:
        pass

    original_metadata = huggingface_hub.get_safetensors_metadata
    huggingface_hub.get_safetensors_metadata = _noop_metadata
    for mod in list(sys.modules.values()):
        if mod is huggingface_hub:
            continue
        for attr in list(vars(mod)):
            if vars(mod)[attr] is original_metadata:
                setattr(mod, attr, _noop_metadata)


def build_layer_groups(kv_cache_config: KVCacheConfig) -> list[int]:
    group_lookup: dict[str, int] = {}
    for group_idx, group_spec in enumerate(kv_cache_config.kv_cache_groups):
        for layer_name in group_spec.layer_names:
            group_lookup[layer_name] = group_idx

    layer_to_group: list[int] = []
    for tensor_spec in kv_cache_config.kv_cache_tensors:
        for name in tensor_spec.shared_by:
            layer_to_group.append(group_lookup[name])
    return layer_to_group


def load_vllm_engine(
    model_id: ModelId,
    trust_remote_code: bool,
    n_layers: int = 1,
    on_layer_loaded: Callable[[int, int], None] | None = None,
    kv_connector_cls: type[object] | None = None,
) -> tuple[LLMEngine, ToolParser | None]:
    model_path = build_model_path(model_id)
    _patch_weight_loading_progress()

    set_n_layers(n_layers)

    # Use the dict-with-colon form the original branch used. The typed
    # `KVTransferConfig` object goes through a different vLLM code path
    # and (with `kv_load_failure_policy="recompute"`) trips the
    # APC/hybrid/chunked-prefill kv-cache nesting bug.
    kv_transfer_config: dict[str, str] | None = None
    if kv_connector_cls is not None:
        kv_transfer_config = {
            "kv_connector": (
                f"{kv_connector_cls.__module__}:{kv_connector_cls.__name__}"
            ),
            "kv_role": "kv_both",
        }

    has_mamba = False
    try:
        with open(model_path / "config.json") as f:
            model_config = json.load(f)  # pyright: ignore[reportAny]
        text_config = model_config.get("text_config", model_config)  # pyright: ignore[reportAny]
        has_mamba = "mamba_ssm_dtype" in text_config or "linear_attention" in (
            text_config.get("layer_types") or []  # pyright: ignore[reportAny]
        )
    except Exception:
        pass
    if has_mamba:
        backends = [AttentionBackendEnum.FLASH_ATTN, AttentionBackendEnum.TRITON_ATTN]
    else:
        backends = [
            AttentionBackendEnum.FLASHINFER,
            AttentionBackendEnum.FLASH_ATTN,
            AttentionBackendEnum.TRITON_ATTN,
        ]

    engine: LLMEngine | None = None
    for backend in backends:
        try:
            engine_args = EngineArgs(
                model=str(model_path.expanduser().resolve()),
                served_model_name=str(model_id),
                gpu_memory_utilization=0.05,
                trust_remote_code=trust_remote_code,
                load_format="fastsafetensors",
                enable_prefix_caching=True,
                attention_backend=backend,
                compilation_config=CompilationConfig(
                    mode=CompilationMode.NONE,
                    cudagraph_mode=CUDAGraphMode.NONE,
                ),
                disable_log_stats=True,
                max_num_batched_tokens=4096,
                kv_transfer_config=kv_transfer_config, # pyright: ignore[reportArgumentType]
                disable_hybrid_kv_cache_manager=False,
                kv_cache_dtype="auto",
            )

            set_weight_loading_callback(on_layer_loaded)
            engine = LLMEngine.from_engine_args(engine_args)
            logger.info(f"vLLM engine using attention backend: {backend}")
            break
        except (ValueError, RuntimeError, NotImplementedError) as e:
            logger.warning(f"Attention backend {backend} failed: {e}, trying next")
            engine = None

            gc.collect()
            torch.cuda.empty_cache()
            continue

    if engine is None:
        raise RuntimeError(f"No attention backend worked for {model_id}")

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
