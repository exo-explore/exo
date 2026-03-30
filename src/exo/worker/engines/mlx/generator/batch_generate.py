import os
import time
from dataclasses import dataclass, field
from typing import Callable, cast

import mlx.core as mx
from mlx_lm.generate import (
    BatchGenerator as MlxBatchGenerator,
)
from mlx_lm.models.cache import RotatingKVCache
from mlx_lm.sample_utils import make_logits_processors, make_sampler
from mlx_lm.tokenizer_utils import StreamingDetokenizer, TokenizerWrapper

from exo.shared.types.api import (
    CompletionTokensDetails,
    FinishReason,
    GenerationStats,
    PromptTokensDetails,
    TopLogprobItem,
    Usage,
)
from exo.shared.types.memory import Memory
from exo.shared.types.mlx import KVCacheType, Model
from exo.shared.types.text_generation import TextGenerationTaskParams
from exo.shared.types.worker.runner_response import GenerationResponse
from exo.worker.engines.mlx.cache import (
    CacheSnapshot,
    KVPrefixCache,
    encode_prompt,
    make_kv_cache,
)
from exo.worker.engines.mlx.constants import DEFAULT_TOP_LOGPROBS, MAX_TOKENS
from exo.worker.engines.mlx.generator.generate import (
    ban_token_ids,
    eos_ids_from_tokenizer,
    extract_top_logprobs,
    prefill,
)
from exo.worker.engines.mlx.utils_mlx import fix_unmatched_think_end_tokens
from exo.worker.runner.bootstrap import logger

_MIN_PREFIX_HIT_RATIO_TO_UPDATE = 0.5


def _stop_sequences(task_params: TextGenerationTaskParams) -> list[str]:
    if task_params.stop is None:
        return []
    if isinstance(task_params.stop, str):
        return [task_params.stop]
    return task_params.stop


@dataclass
class _EngineTask:
    uid: int
    task_params: TextGenerationTaskParams
    all_prompt_tokens: mx.array
    prefix_hit_length: int
    matched_index: int | None
    cache_snapshots: list[CacheSnapshot] | None
    detokenizer: StreamingDetokenizer
    on_generation_token: Callable[[], None] | None = None
    generated_text_parts: list[str] = field(default_factory=list)
    potential_stop_sequence_text: str = ""
    completion_tokens: int = 0
    generation_start_time: float = 0.0
    in_thinking: bool = False
    reasoning_tokens: int = 0
    prefill_tps: float = 0.0


@dataclass(eq=False)
class ExoBatchGenerator:
    model: Model
    tokenizer: TokenizerWrapper
    group: mx.distributed.Group | None
    kv_prefix_cache: KVPrefixCache | None

    _exo_gen: MlxBatchGenerator = field(init=False)
    _active_tasks: dict[int, _EngineTask] = field(default_factory=dict, init=False)

    def __post_init__(self) -> None:
        use_speculative = os.environ.get("EXO_SPECULATIVE", "0") == "1"
        stop_tokens = set(eos_ids_from_tokenizer(self.tokenizer))

        if use_speculative:
            try:
                from exo.worker.engines.mlx.speculative.mtp_module import MTPPredictor
                from exo.worker.engines.mlx.speculative.mtp_batch_generator import MTPBatchGenerator

                mtp_weights = self._resolve_mtp_weights()
                gamma = int(os.environ.get("EXO_SPECULATIVE_GAMMA", "2"))

                if mtp_weights:
                    mtp = MTPPredictor(self.model, mtp_weights, quantize=False)
                    temp = float(os.environ.get("EXO_SPECULATIVE_TEMP", "0.7"))
                    alpha = float(os.environ.get("EXO_SPECULATIVE_ALPHA", "1.0"))
                    self._exo_gen = MTPBatchGenerator(
                        model=self.model,
                        mtp_predictor=mtp,
                        gamma=gamma,
                        temp=temp,
                        alpha=alpha,
                        stop_tokens=stop_tokens,
                        prefill_step_size=4096,
                    )
                    logger.info(f"MTP speculative decoding enabled (γ={gamma}, T={temp})")
                else:
                    logger.warning("EXO_SPECULATIVE=1 but could not find MTP weights. Falling back to standard generation.")
                    self._exo_gen = MlxBatchGenerator(
                        model=self.model,
                        stop_tokens=stop_tokens,
                        prefill_step_size=4096,
                    )
            except Exception as e:
                logger.warning(f"Failed to initialize MTP speculative decoding: {e}. Falling back to standard generation.")
                self._exo_gen = MlxBatchGenerator(
                    model=self.model,
                    stop_tokens=stop_tokens,
                    prefill_step_size=4096,
                )
        else:
            self._exo_gen = MlxBatchGenerator(
                model=self.model,
                stop_tokens=stop_tokens,
                prefill_step_size=4096,
            )

    def _resolve_mtp_weights(self) -> str | None:
        """Find MTP weights: explicit path, explicit HF model, or auto-extract."""
        # 1. Explicit path
        explicit_path = os.environ.get("EXO_MTP_WEIGHTS", "")
        if explicit_path and os.path.exists(explicit_path):
            return explicit_path

        # 2. Explicit HF model repo containing MTP weights
        mtp_model = os.environ.get("EXO_MTP_MODEL", "")

        # 3. Auto-detect: if no EXO_MTP_MODEL set, try to infer from model config
        if not mtp_model:
            try:
                inner = getattr(self.model, 'model', None) or self.model.language_model.model
                args = getattr(inner, 'args', None)
                if args and getattr(args, 'mtp_num_hidden_layers', 0) > 0:
                    model_type = getattr(args, 'model_type', '')
                    if 'qwen3_5' in model_type or 'qwen3.5' in str(type(self.model).__module__):
                        # Default pairing for Qwen3.5-27B
                        mtp_model = "Qwen/Qwen3.5-27B"
                        logger.info(f"Auto-detected MTP model: {mtp_model}")
            except Exception:
                pass

        if not mtp_model:
            return None

        # Download and extract MTP weights from HF repo
        try:
            return self._extract_mtp_from_hf(mtp_model)
        except Exception as e:
            logger.warning(f"Failed to extract MTP weights from {mtp_model}: {e}")
            return None

    def _extract_mtp_from_hf(self, repo_id: str) -> str:
        """Download MTP tensors from HF repo and cache as a single safetensors file."""
        import hashlib
        from pathlib import Path
        from huggingface_hub import snapshot_download
        from safetensors.torch import load_file, save_file

        cache_dir = Path.home() / ".cache" / "exo" / "mtp_weights"
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache_key = hashlib.md5(repo_id.encode()).hexdigest()[:12]
        cached_path = cache_dir / f"mtp_{cache_key}.safetensors"

        if cached_path.exists():
            logger.info(f"Using cached MTP weights: {cached_path}")
            return str(cached_path)

        logger.info(f"Downloading MTP weights from {repo_id}...")
        model_dir = snapshot_download(
            repo_id,
            allow_patterns=["*.safetensors", "*.json"],
        )

        # Extract MTP tensors from all safetensors files
        mtp_tensors = {}
        model_path = Path(model_dir)
        for sf_file in sorted(model_path.glob("*.safetensors")):
            tensors = load_file(str(sf_file))
            for k, v in tensors.items():
                if k.startswith("model.mtp."):
                    # Strip "model." prefix to match our MTPPredictor format
                    clean_key = k[len("model."):]
                    mtp_tensors[clean_key] = v

        if not mtp_tensors:
            raise ValueError(f"No MTP tensors found in {repo_id}")

        save_file(mtp_tensors, str(cached_path))
        logger.info(f"Extracted {len(mtp_tensors)} MTP tensors → {cached_path} ({cached_path.stat().st_size / 1e6:.0f}MB)")
        return str(cached_path)

    @property
    def has_work(self) -> bool:
        return (
            bool(self._active_tasks)
            or bool(self._exo_gen.unprocessed_prompts)
            or self._exo_gen.active_batch is not None
        )

    def submit(
        self,
        task_params: TextGenerationTaskParams,
        prompt: str,
        on_prefill_progress: Callable[[int, int], None] | None = None,
        distributed_prompt_progress_callback: Callable[[], None] | None = None,
        on_generation_token: Callable[[], None] | None = None,
    ) -> int:
        all_prompt_tokens = encode_prompt(self.tokenizer, prompt)
        all_prompt_tokens = fix_unmatched_think_end_tokens(
            all_prompt_tokens, self.tokenizer
        )

        is_bench = task_params.bench

        prefix_hit_length = 0
        matched_index: int | None = None
        prompt_tokens = all_prompt_tokens

        if self.kv_prefix_cache is not None and not is_bench:
            cache, remaining_tokens, matched_index = self.kv_prefix_cache.get_kv_cache(
                self.model, all_prompt_tokens
            )
            prefix_hit_length = len(all_prompt_tokens) - len(remaining_tokens)
            if prefix_hit_length > 0:
                logger.info(
                    f"KV cache hit: {prefix_hit_length}/{len(all_prompt_tokens)} tokens "
                    f"cached ({100 * prefix_hit_length / len(all_prompt_tokens):.1f}%)"
                )
                prompt_tokens = remaining_tokens
            else:
                cache = make_kv_cache(self.model)
        else:
            cache = make_kv_cache(self.model)

        seed = task_params.seed if task_params.seed is not None else 42
        mx.random.seed(seed)

        sampler = make_sampler(
            temp=task_params.temperature
            if task_params.temperature is not None
            else 0.7,
            top_p=task_params.top_p if task_params.top_p is not None else 1.0,
            min_p=task_params.min_p if task_params.min_p is not None else 0.05,
            top_k=task_params.top_k if task_params.top_k is not None else 0,
        )

        _prefill_tps, _prefill_tokens, cache_snapshots = prefill(
            self.model,
            self.tokenizer,
            sampler,
            prompt_tokens[:-1],
            cache,
            self.group,
            on_prefill_progress,
            distributed_prompt_progress_callback,
        )

        # MTP prefill: build MTP KV cache from prompt hidden states
        # Pair position i with token i+1 (MTP predicts token t+2 from hidden[t] + embed[t+1])
        if hasattr(self._exo_gen, 'mtp'):
            prompt_pre_norm = self._exo_gen._captured.get('prompt_pre_norm')
            if prompt_pre_norm is not None:
                mx.eval(prompt_pre_norm)
                self._exo_gen.mtp.reset_cache()
                S_pre = prompt_pre_norm.shape[1]
                if S_pre > 0 and len(all_prompt_tokens) > S_pre:
                    mtp_toks = all_prompt_tokens[1:S_pre + 1].tolist()
                    _ = self._exo_gen.mtp.predict(
                        prompt_pre_norm,
                        mx.array([mtp_toks])
                    )
                    mx.eval(_)
                logger.info(f"MTP cache prefilled ({S_pre} positions)")

        # We need to clamp rotating kv caches to max size so that mlx lm's _merge_caches behaves
        for c in cache:
            if (
                isinstance(c, RotatingKVCache)
                and c.keys is not None
                and c.values is not None
                and c.keys.shape[2] > c.max_size
            ):
                trim_size = c.keys.shape[2] - c.max_size
                c.keys = c._trim(trim_size, c.keys)
                c.values = c._trim(trim_size, c.values)
                c._idx = c.max_size

        if not is_bench:
            self._save_prefix_cache(
                all_prompt_tokens,
                list(cache),
                cache_snapshots,
                prefix_hit_length,
                matched_index,
            )

        last_tokens = prompt_tokens[-2:]

        logits_processors: list[Callable[[mx.array, mx.array], mx.array]] = (
            make_logits_processors(
                repetition_penalty=task_params.repetition_penalty,
                repetition_context_size=task_params.repetition_context_size,
            )
        )
        if is_bench:
            # Only sample length eos tokens
            eos_ids = eos_ids_from_tokenizer(self.tokenizer)
            logits_processors = [ban_token_ids(eos_ids)] + logits_processors

        max_tokens = task_params.max_output_tokens or MAX_TOKENS

        uids = self._exo_gen.insert(
            prompts=[last_tokens.tolist()],
            max_tokens=[max_tokens],
            caches=[list(cache)],
            samplers=[sampler],
            logits_processors=[logits_processors],
        )

        assert len(uids) == 1

        uid = uids[0]

        self._active_tasks[uid] = _EngineTask(
            uid=uid,
            task_params=task_params,
            all_prompt_tokens=all_prompt_tokens,
            prefix_hit_length=prefix_hit_length,
            matched_index=matched_index,
            cache_snapshots=cache_snapshots or None,
            detokenizer=self.tokenizer.detokenizer,
            on_generation_token=on_generation_token,
            generation_start_time=time.perf_counter(),
            prefill_tps=_prefill_tps,
        )

        return uid

    def step(self) -> list[tuple[int, GenerationResponse]]:
        if not self.has_work:
            return []

        responses = self._exo_gen.next()

        results: list[tuple[int, GenerationResponse]] = []

        for response in responses:
            if response.uid not in self._active_tasks:
                logger.warning(
                    f"response uid {response.uid} was not found - should be active"
                )
                continue

            state = self._active_tasks[response.uid]
            if state.on_generation_token is not None:
                state.on_generation_token()
            if response.finish_reason != "stop":
                state.detokenizer.add_token(response.token)
            if response.finish_reason is not None:
                state.detokenizer.finalize()
            text = state.detokenizer.last_segment
            state.completion_tokens += 1
            state.generated_text_parts.append(text)
            state.potential_stop_sequence_text += text

            think_start = self.tokenizer.think_start
            think_end = self.tokenizer.think_end
            if think_start is not None and text == think_start:
                state.in_thinking = True
            elif think_end is not None and text == think_end:
                state.in_thinking = False
            if state.in_thinking:
                state.reasoning_tokens += 1

            finish_reason: FinishReason | None = cast(
                FinishReason | None, response.finish_reason
            )
            task_params = state.task_params
            stop_sequences = _stop_sequences(task_params)
            max_stop_len = max((len(s) for s in stop_sequences), default=0)

            if stop_sequences:
                for stop_seq in stop_sequences:
                    if stop_seq in state.potential_stop_sequence_text:
                        stop_index = state.potential_stop_sequence_text.find(stop_seq)
                        text_before_stop = state.potential_stop_sequence_text[
                            :stop_index
                        ]
                        chunk_start = len(state.potential_stop_sequence_text) - len(
                            text
                        )
                        text = text_before_stop[chunk_start:]
                        finish_reason = "stop"
                        break

            is_done = finish_reason is not None

            logprob: float | None = None
            top_logprobs: list[TopLogprobItem] | None = None
            if task_params.logprobs:
                logprob, top_logprobs = extract_top_logprobs(
                    logprobs=response.logprobs,
                    tokenizer=self.tokenizer,
                    top_logprobs=task_params.top_logprobs or DEFAULT_TOP_LOGPROBS,
                    selected_token=response.token,
                )

            stats: GenerationStats | None = None
            usage: Usage | None = None
            if is_done:
                try:
                    mlx_stats = self._exo_gen.stats()
                    generation_tps = mlx_stats.generation_tps
                except ZeroDivisionError:
                    generation_elapsed = (
                        time.perf_counter() - state.generation_start_time
                    )
                    generation_tps = (
                        state.completion_tokens / generation_elapsed
                        if generation_elapsed > 0
                        else 0.0
                    )

                stats = GenerationStats(
                    prompt_tps=state.prefill_tps,
                    generation_tps=generation_tps,
                    prompt_tokens=len(state.all_prompt_tokens),
                    generation_tokens=state.completion_tokens,
                    peak_memory_usage=Memory.from_gb(mx.get_peak_memory() / 1e9),
                )
                total_prompt_tokens = len(state.all_prompt_tokens)
                usage = Usage(
                    prompt_tokens=total_prompt_tokens,
                    completion_tokens=state.completion_tokens,
                    total_tokens=total_prompt_tokens + state.completion_tokens,
                    prompt_tokens_details=PromptTokensDetails(
                        cached_tokens=state.prefix_hit_length
                    ),
                    completion_tokens_details=CompletionTokensDetails(
                        reasoning_tokens=state.reasoning_tokens
                    ),
                )

            results.append(
                (
                    response.uid,
                    GenerationResponse(
                        text=text,
                        token=response.token,
                        logprob=logprob,
                        top_logprobs=top_logprobs,
                        finish_reason=finish_reason,
                        stats=stats,
                        usage=usage,
                    ),
                )
            )

            if is_done:
                del self._active_tasks[response.uid]
            elif (
                max_stop_len > 0
                and len(state.potential_stop_sequence_text) > max_stop_len
            ):
                state.potential_stop_sequence_text = state.potential_stop_sequence_text[
                    -max_stop_len:
                ]

        return results

    def cancel(self, uids: list[int]) -> None:
        self._exo_gen.remove(uids)
        for uid in uids:
            self._active_tasks.pop(uid, None)

    def close(self) -> None:
        self._exo_gen.close()

    def _save_prefix_cache(
        self,
        all_prompt_tokens: mx.array,
        cache: KVCacheType,
        cache_snapshots: list[CacheSnapshot] | None,
        prefix_hit_length: int,
        matched_index: int | None,
    ) -> None:
        if self.kv_prefix_cache is None:
            return

        try:
            hit_ratio = (
                prefix_hit_length / len(all_prompt_tokens)
                if len(all_prompt_tokens) > 0
                else 0.0
            )
            if (
                matched_index is not None
                and hit_ratio >= _MIN_PREFIX_HIT_RATIO_TO_UPDATE
            ):
                self.kv_prefix_cache.update_kv_cache(
                    matched_index,
                    all_prompt_tokens,
                    cache,
                    cache_snapshots,
                    restore_pos=prefix_hit_length,
                )
            else:
                self.kv_prefix_cache.add_kv_cache(
                    all_prompt_tokens, cache, cache_snapshots
                )
        except Exception:
            logger.warning("Failed to save prefix cache", exc_info=True)
