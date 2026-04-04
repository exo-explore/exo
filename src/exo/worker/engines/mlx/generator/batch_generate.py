import contextlib
import os
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Generator, cast

import mlx.core as mx
from mlx_lm.generate import (
    BatchGenerator as MlxBatchGenerator,
)
from mlx_lm.generate import (
    generation_stream,
    stream_generate,
)
from mlx_lm.models.cache import RotatingKVCache
from mlx_lm.sample_utils import make_logits_processors, make_sampler
from mlx_lm.tokenizer_utils import StreamingDetokenizer, TokenizerWrapper

from exo.api.types import (
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
    has_non_kv_caches,
    make_kv_cache,
)
from exo.worker.engines.mlx.constants import DEFAULT_TOP_LOGPROBS, KV_BITS, KV_GROUP_SIZE, MAX_TOKENS
from exo.worker.engines.mlx.generator.generate import (
    ban_token_ids,
    eos_ids_from_tokenizer,
    extract_top_logprobs,
    patch_embed_tokens,
    prefill,
)
from exo.worker.engines.mlx.utils_mlx import (
    fix_unmatched_think_end_tokens,
    system_prompt_token_count,
)
from exo.worker.engines.mlx.vision import (
    MediaRegion,
    VisionProcessor,
    VisionResult,
    prepare_vision,
)
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
    generation_time_at_start: float = 0.0
    in_thinking: bool = False
    reasoning_tokens: int = 0
    prefill_tps: float = 0.0
    media_regions: list[MediaRegion] = field(default_factory=list)


@dataclass(eq=False)
class ExoBatchGenerator:
    model: Model
    tokenizer: TokenizerWrapper
    group: mx.distributed.Group | None
    kv_prefix_cache: KVPrefixCache | None
    vision_processor: VisionProcessor | None = None
    model_id: str = ""

    _mlx_gen: MlxBatchGenerator = field(init=False)
    _active_tasks: dict[int, _EngineTask] = field(default_factory=dict, init=False)
    _pp_spec_active: bool = field(init=False, default=False)
    _pp_spec_gen: Generator[tuple[int, mx.array], None, None] | None = field(init=False, default=None)
    _pp_spec_uid: int | None = field(init=False, default=None)
    _pp_spec_eos: set[int] = field(init=False, default_factory=set)
    _uid_counter: int = field(init=False, default=0)

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
                    self._mlx_gen = MTPBatchGenerator(
                        model=self.model,
                        mtp_predictor=mtp,
                        gamma=gamma,
                        temp=temp,
                        alpha=alpha,
                        stop_tokens=stop_tokens,
                        prefill_step_size=4096,
                    )
                    logger.info(f"MTP speculative decoding enabled (γ={gamma}, T={temp})")
                    # Skip warmup — OOMs on 397B (Metal abort, uncatchable)
                else:
                    logger.warning("EXO_SPECULATIVE=1 but could not find MTP weights. Falling back.")
                    self._mlx_gen = MlxBatchGenerator(model=self.model, stop_tokens=stop_tokens, prefill_step_size=4096)
            except Exception as e:
                logger.warning(f"Failed to init MTP speculative: {e}. Falling back.")
                self._mlx_gen = MlxBatchGenerator(model=self.model, stop_tokens=stop_tokens, prefill_step_size=4096)
        else:
            self._mlx_gen = MlxBatchGenerator(model=self.model, stop_tokens=stop_tokens, prefill_step_size=4096)

        self._mlx_gen._needs_topk = False  # pyright: ignore[reportAttributeAccessIssue]
        self._pp_spec_eos = set(eos_ids_from_tokenizer(self.tokenizer))

        # Enable PP speculation if draft model is configured and we're in PP mode
        draft_path = os.environ.get("EXO_PP_DRAFT_MODEL", "")
        if draft_path and self.group is not None and self.group.size() > 1:
            try:
                from ..pp_speculation import get_pipeline_info
                if get_pipeline_info(self.model) is not None:
                    self._pp_spec_active = True
                    logger.info("PP speculation enabled in BatchGenerator")
                    # Load MTP for PP speculation (prefer pre-quantized 4-bit, 3.7GB)
                    if use_speculative:
                        mtp_weights = self._resolve_mtp_weights()
                        if mtp_weights:
                            try:
                                self._pp_mtp = MTPPredictor(
                                    self.model, mtp_weights, quantize=False,
                                )
                                logger.info(f"PP MTP loaded from {mtp_weights}")
                            except Exception as e:
                                import traceback
                                logger.warning(f"PP MTP load failed: {e}\n{traceback.format_exc()}")
                                self._pp_mtp = None
                        else:
                            self._pp_mtp = None
                    else:
                        self._pp_mtp = None
            except Exception:
                pass

    def _resolve_mtp_weights(self) -> str | None:
        """Find MTP weights: explicit path, local model dir, or HF repo extraction.

        Detection order:
        1. EXO_MTP_WEIGHTS env var (explicit path)
        2. Pre-quantized cache (~/.cache/exo/mtp_weights/mtp_*_q4.safetensors)
        3. Bf16 cache (~/.cache/exo/mtp_weights/mtp_*.safetensors)
        4. Local model directory (check weight index for mtp.* keys)
        5. HF repo download (selective shard download)
        """
        import hashlib
        from pathlib import Path

        # 1. Explicit path
        explicit_path = os.environ.get("EXO_MTP_WEIGHTS", "")
        if explicit_path and os.path.exists(explicit_path):
            return explicit_path

        # Determine source HF repo for MTP weights
        mtp_repo = os.environ.get("EXO_MTP_MODEL", "")
        if not mtp_repo:
            mtp_repo = self._detect_mtp_repo()
        if not mtp_repo:
            return None

        # 2-3. Check cache
        cache_dir = Path.home() / ".cache" / "exo" / "mtp_weights"
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache_key = hashlib.md5(mtp_repo.encode()).hexdigest()[:12]

        q4_path = cache_dir / f"mtp_{cache_key}_q4.safetensors"
        if q4_path.exists():
            logger.info(f"Using pre-quantized MTP weights: {q4_path}")
            return str(q4_path)

        bf16_path = cache_dir / f"mtp_{cache_key}.safetensors"
        if bf16_path.exists():
            # Auto-quantize bf16 → q4 on first load
            q4 = self._auto_quantize_mtp(bf16_path, q4_path)
            if q4:
                return q4
            return str(bf16_path)

        # 4. Check local model directory for MTP weights
        if self.model_id:
            from exo.download.download_utils import build_model_path
            from exo.shared.types.common import ModelId
            local_path = self._extract_mtp_from_local(
                build_model_path(ModelId(self.model_id)), cache_dir, cache_key
            )
            if local_path:
                q4 = self._auto_quantize_mtp(Path(local_path), q4_path)
                return q4 or local_path

        # 5. Download from HF repo
        try:
            dl_path = self._extract_mtp_from_hf(mtp_repo)
            if dl_path:
                q4 = self._auto_quantize_mtp(Path(dl_path), q4_path)
                return q4 or dl_path
        except Exception as e:
            logger.warning(f"Failed to extract MTP weights from {mtp_repo}: {e}")
        return None

    def _detect_mtp_repo(self) -> str:
        """Detect the HF repo containing MTP weights for this model.

        Checks model args for mtp_num_hidden_layers and model_type to determine
        which HF repo has the MTP weights. Returns '' if MTP not supported.
        """
        try:
            inner = getattr(self.model, 'model', None) or self.model.language_model.model
            args = (getattr(self.model, 'args', None)
                    or getattr(inner, 'args', None)
                    or getattr(getattr(inner, 'model', None), 'args', None))
            model_type = getattr(args, 'model_type', '') if args else ''
            if not model_type and args and hasattr(args, 'text_config'):
                model_type = args.text_config.get('model_type', '')

            # Check if model has MTP layers configured
            has_mtp = args and getattr(args, 'mtp_num_hidden_layers', 0) > 0

            if has_mtp or 'qwen3_5' in model_type:
                # Map model_id to original HF repo (strip mlx-community prefix + quant suffix)
                repo = self._model_id_to_hf_repo(self.model_id) if self.model_id else ''
                if repo:
                    logger.info(f"Auto-detected MTP repo: {repo} (model_type={model_type})")
                    return repo

                # Fallback: known model type mappings
                if model_type == 'qwen3_5_moe':
                    return "Qwen/Qwen3.5-397B-A17B"
                elif 'qwen3_5' in model_type:
                    return "Qwen/Qwen3.5-27B"
        except Exception as e:
            logger.warning(f"MTP detection failed: {e}")
        return ''

    @staticmethod
    def _model_id_to_hf_repo(model_id: str) -> str:
        """Map an MLX model ID to the original HF repo containing MTP weights.

        e.g. 'mlx-community/Qwen3.5-397B-A17B-4bit' → 'Qwen/Qwen3.5-397B-A17B'
        """
        # Strip common MLX community prefixes
        name = model_id
        if name.startswith('mlx-community/'):
            name = name[len('mlx-community/'):]

        # Strip quantization suffixes
        for suffix in ['-4bit', '-8bit', '-bf16', '-fp16', '-MLX-4bit', '-MLX-8bit', '-MLX']:
            if name.endswith(suffix):
                name = name[:-len(suffix)]
                break

        # Map to original Qwen repo
        if name.startswith('Qwen3'):
            return f"Qwen/{name}"

        return ''

    def _extract_mtp_from_local(self, model_dir, cache_dir, cache_key) -> str | None:
        """Check local model directory for MTP weights and extract if found."""
        import json
        from pathlib import Path

        model_dir = Path(model_dir)
        idx_path = model_dir / "model.safetensors.index.json"
        if not idx_path.exists():
            return None

        with open(idx_path) as f:
            idx = json.load(f)

        mtp_keys = [k for k in idx["weight_map"] if k.startswith("mtp.")]
        if not mtp_keys:
            return None

        mtp_shards = sorted({idx["weight_map"][k] for k in mtp_keys})
        logger.info(f"Found {len(mtp_keys)} MTP weights in local model ({len(mtp_shards)} shards)")

        # Extract MTP tensors from local shards
        from safetensors.torch import load_file, save_file
        mtp_tensors = {}
        for shard_name in mtp_shards:
            shard_path = model_dir / shard_name
            if not shard_path.exists():
                return None  # shard missing, can't extract locally
            tensors = load_file(str(shard_path))
            for k, v in tensors.items():
                if k.startswith("mtp."):
                    mtp_tensors[k] = v

        if not mtp_tensors:
            return None

        cached_path = cache_dir / f"mtp_{cache_key}.safetensors"
        save_file(mtp_tensors, str(cached_path))
        logger.info(f"Extracted {len(mtp_tensors)} MTP tensors from local model → {cached_path}")
        return str(cached_path)

    @staticmethod
    def _auto_quantize_mtp(bf16_path, q4_path) -> str | None:
        """Auto-quantize bf16 MTP weights to 4-bit. Returns q4 path or None on failure."""
        try:
            import mlx.core as mx
            import mlx.nn as nn

            logger.info(f"Auto-quantizing MTP weights → {q4_path}")
            weights = mx.load(str(bf16_path))
            q_weights = {}
            for k, v in weights.items():
                if v.ndim == 2 and min(v.shape) >= 64:
                    lin = nn.Linear(v.shape[1], v.shape[0], bias=False)
                    lin.weight = v
                    ql = nn.QuantizedLinear.from_linear(lin, group_size=64, bits=4)
                    q_weights[k] = ql.weight
                    q_weights[k.replace('.weight', '.scales')] = ql.scales
                    q_weights[k.replace('.weight', '.biases')] = ql.biases
                    mx.eval(ql.weight, ql.scales, ql.biases)
                    del lin, ql
                else:
                    q_weights[k] = v
            mx.save_safetensors(str(q4_path), q_weights)
            logger.info(f"Auto-quantized MTP: {len(q_weights)} tensors → {q4_path}")
            return str(q4_path)
        except Exception as e:
            logger.warning(f"Auto-quantize MTP failed: {e}")
            return None

    def _extract_mtp_from_hf(self, repo_id: str) -> str:
        """Download MTP tensors from HF repo and cache as a single safetensors file.

        Uses the weight index to only download shards containing MTP weights,
        avoiding a full model download for large models (e.g. 397B = 220GB).
        """
        import hashlib
        import json
        from pathlib import Path
        from huggingface_hub import hf_hub_download
        from safetensors.torch import load_file, save_file

        cache_dir = Path.home() / ".cache" / "exo" / "mtp_weights"
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache_key = hashlib.md5(repo_id.encode()).hexdigest()[:12]
        cached_path = cache_dir / f"mtp_{cache_key}.safetensors"

        logger.info(f"Downloading MTP weights from {repo_id}...")

        # Use weight index to find only the shards containing MTP weights
        try:
            idx_path = hf_hub_download(repo_id, "model.safetensors.index.json")
            with open(idx_path) as f:
                idx = json.load(f)
            mtp_shards = {
                shard for key, shard in idx["weight_map"].items()
                if key.startswith("model.mtp.") or key.startswith("mtp.")
            }
            logger.info(f"MTP weights span {len(mtp_shards)} of {len(set(idx['weight_map'].values()))} shards")
        except Exception:
            mtp_shards = None

        mtp_tensors = {}

        if mtp_shards:
            # Download only the shards containing MTP weights
            for shard_name in sorted(mtp_shards):
                shard_path = hf_hub_download(repo_id, shard_name)
                tensors = load_file(shard_path)
                for k, v in tensors.items():
                    if k.startswith("model.mtp."):
                        mtp_tensors[k.replace("model.mtp.", "")] = v
                    elif k.startswith("mtp."):
                        mtp_tensors[k] = v
        else:
            # Fallback: download all safetensors (small models)
            from huggingface_hub import snapshot_download
            model_dir = snapshot_download(repo_id, allow_patterns=["*.safetensors", "*.json"])
            model_path = Path(model_dir)
            for sf_file in sorted(model_path.glob("*.safetensors")):
                tensors = load_file(str(sf_file))
                for k, v in tensors.items():
                    if k.startswith("model.mtp."):
                        mtp_tensors[k.replace("model.mtp.", "")] = v

        if not mtp_tensors:
            raise ValueError(f"No MTP tensors found in {repo_id}")

        save_file(mtp_tensors, str(cached_path))
        logger.info(f"Cached {len(mtp_tensors)} MTP tensors to {cached_path}")
        return str(cached_path)

    def warmup_speculative(self, model, tokenizer) -> None:
        """Warm up the speculative decoding path (MTP draft + verify kernels)."""
        if not hasattr(self._mlx_gen, 'mtp'):
            return

        from mlx_lm.models import cache as cache_mod
        from exo.worker.engines.mlx.speculative.mtp_module import speculative_forward, draft_tokens

        logger.info("Warming up speculative decoding kernels...")
        mtp = self._mlx_gen.mtp
        gamma = self._mlx_gen.gamma

        warmup_prompt = tokenizer.encode("Warm up speculative decoding.")
        cache = cache_mod.make_prompt_cache(model)
        mtp.reset_cache()

        pre_norm, logits = speculative_forward(model, mx.array([warmup_prompt]), cache)
        mx.eval(pre_norm, logits)
        next_token = mx.argmax(logits[0, -1], axis=-1).item()

        if pre_norm.shape[1] > 1:
            _ = mtp.predict(pre_norm[:, :-1, :], mx.array([warmup_prompt[1:]]))
            mx.eval(_)

        last_pn = pre_norm[:, -1:, :]
        next_arr = mx.array([[next_token]])
        for _ in range(3):
            draft_ids, _ = draft_tokens(mtp, last_pn, next_arr, gamma, 0.0)
            draft_concat = mx.concatenate([d.reshape(1, 1) for d in draft_ids], axis=1)
            verify_input = mx.concatenate([next_arr, draft_concat], axis=1)
            vpn, vl = speculative_forward(model, verify_input, cache, speculative=True)
            all_next = mx.argmax(vl[0], axis=-1)
            mx.eval(vpn, all_next)
            next_arr = all_next[0].reshape(1, 1)
            last_pn = vpn[:, 0:1, :]
            for i, c in enumerate(cache):
                if hasattr(c, 'base'):
                    cache[i] = c.base

        logger.info("Speculative warmup complete")

    @property
    def has_work(self) -> bool:
        return (
            bool(self._active_tasks)
            or bool(self._mlx_gen.unprocessed_prompts)
            or self._mlx_gen.active_batch is not None
            or self._pp_spec_gen is not None
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

        vision: VisionResult | None = None
        media_regions: list[MediaRegion] = []

        if self.vision_processor is not None:
            try:
                vision = prepare_vision(
                    images=task_params.images,
                    chat_template_messages=task_params.chat_template_messages,
                    vision_processor=self.vision_processor,
                    tokenizer=self.tokenizer,
                    model=self.model,
                    model_id=task_params.model,
                    task_params=task_params,
                )
            except Exception:
                logger.opt(exception=True).warning(
                    "Vision processing failed, falling back to text-only"
                )

        if vision is not None:
            all_prompt_tokens = vision.prompt_tokens
            media_regions = vision.media_regions

        is_bench = task_params.bench

        prefix_hit_length = 0
        matched_index: int | None = None
        prompt_tokens = all_prompt_tokens

        if self.kv_prefix_cache is not None and not is_bench:
            cache, remaining_tokens, matched_index = self.kv_prefix_cache.get_kv_cache(
                self.model, all_prompt_tokens, media_regions=media_regions
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

        vision_ctx = (
            patch_embed_tokens(
                self.model, vision.embeddings, prefix_hit_length, len(prompt_tokens) - 1
            )
            if vision is not None
            else contextlib.nullcontext()
        )
        with vision_ctx:
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
            min_prefix_hit_length = max(
                1000, system_prompt_token_count(task_params, self.tokenizer)
            )
            self._save_prefix_cache(
                all_prompt_tokens,
                list(cache),
                cache_snapshots,
                prefix_hit_length,
                matched_index,
                min_prefix_hit_length,
                media_regions,
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

        if self._pp_spec_active:
            return self._submit_pp_spec(
                task_params, all_prompt_tokens, prefix_hit_length, matched_index,
                cache_snapshots, cache, last_tokens, sampler, logits_processors,
                max_tokens, on_generation_token, _prefill_tps,
            )

        uids = self._mlx_gen.insert(
            prompts=[last_tokens.tolist()],
            max_tokens=[max_tokens],
            caches=[list(cache)],
            samplers=[sampler],
            logits_processors=[logits_processors],
        )

        assert len(uids) == 1

        uid = uids[0]

        # MTP prefill: build MTP cache from prompt hidden states
        if hasattr(self._mlx_gen, 'mtp'):
            prompt_pre_norm = self._mlx_gen._captured.get('prompt_pre_norm')
            if prompt_pre_norm is not None:
                mx.eval(prompt_pre_norm)
                self._mlx_gen.mtp.reset_cache()
                S_pre = prompt_pre_norm.shape[1]
                if S_pre > 1:
                    toks_list = all_prompt_tokens.tolist() if hasattr(all_prompt_tokens, 'tolist') else list(all_prompt_tokens)
                    mtp_tokens = toks_list[1:S_pre]
                    _ = self._mlx_gen.mtp.predict(
                        prompt_pre_norm[:, :-1, :],
                        mx.array([mtp_tokens])
                    )
                    mx.eval(_)
                    logger.info(f"MTP cache prefilled ({S_pre} positions)")

        # Set per-request temperature for speculative
        if hasattr(self._mlx_gen, '_request_temp'):
            env_temp = os.environ.get("EXO_SPECULATIVE_TEMP")
            if env_temp is not None:
                self._mlx_gen._request_temp[uid] = float(env_temp)
            elif task_params.temperature is not None:
                self._mlx_gen._request_temp[uid] = task_params.temperature

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
            generation_time_at_start=self._mlx_gen._stats.generation_time,
            media_regions=media_regions,
        )

        return uid

    def _submit_pp_spec(
        self,
        task_params: TextGenerationTaskParams,
        all_prompt_tokens: mx.array,
        prefix_hit_length: int,
        matched_index: int | None,
        cache_snapshots: list[CacheSnapshot] | None,
        cache: list[Any],
        last_tokens: mx.array,
        sampler: Callable,
        logits_processors: list[Callable],
        max_tokens: int,
        on_generation_token: Callable[[], None] | None,
        prefill_tps: float,
    ) -> int:
        """Set up PP speculative decode for this task."""
        from ..pp_speculation import (
            get_pipeline_info,
            pp_speculative_decode_loop,
            _install_spec_layers,
        )

        pp_info = get_pipeline_info(self.model)
        assert pp_info is not None
        pp_rank, pp_world_size, pp_group = pp_info

        inner = getattr(self.model, "language_model", self.model)
        _install_spec_layers(inner)

        _pp_draft = getattr(self.model, "_pp_draft_model", None)
        _pp_draft_cache = getattr(self.model, "_pp_draft_cache", None)

        # Prefill draft cache with tail of prompt (rank 0 only)
        # The draft model uses a RotatingKVCache, so only recent tokens matter.
        if pp_rank == 0 and _pp_draft is not None:
            _draft_kv_window = int(os.environ.get("EXO_DRAFT_KV_WINDOW", "4096"))
            _draft_tokens = all_prompt_tokens[-_draft_kv_window:]
            _draft_chunk = 512
            for i in range(0, len(_draft_tokens), _draft_chunk):
                _pp_draft(_draft_tokens[i:i + _draft_chunk][None], cache=_pp_draft_cache)
                mx.eval([c.state if hasattr(c, 'state') else c for c in _pp_draft_cache])
            mx.clear_cache()
            logger.info(f"Draft model prefilled with {len(_draft_tokens)} tokens (of {len(all_prompt_tokens)} total)")

        # First token via standard PP
        _first_gen = stream_generate(
            model=self.model, tokenizer=self.tokenizer, prompt=last_tokens,
            max_tokens=1, sampler=sampler, logits_processors=logits_processors,
            prompt_cache=cache, prefill_step_size=1,
            kv_group_size=KV_GROUP_SIZE, kv_bits=KV_BITS,
        )
        _first_out = next(_first_gen)
        first_y = mx.array([_first_out.token])
        mx.eval(first_y)

        logger.info(f"PP speculation active: rank={pp_rank}")

        # Get PP MTP predictor (lightweight, skip_mlp=True)
        _pp_mtp = getattr(self, '_pp_mtp', None)
        if _pp_mtp is not None:
            logger.info("PP speculation using MTP for drafting")

        # Create the spec decode generator
        self._pp_spec_gen = pp_speculative_decode_loop(
            model=self.model, draft_model=_pp_draft,
            prompt_cache=cache, draft_cache=_pp_draft_cache,
            sampler=sampler, logits_processors=logits_processors,
            first_y=first_y, first_logprobs=mx.zeros(1),
            max_tokens=max_tokens - 1,
            pp_rank=pp_rank, pp_world_size=pp_world_size,
            pp_group=pp_group,
            mtp_predictor=_pp_mtp,
        )

        self._uid_counter += 1
        uid = self._uid_counter
        self._pp_spec_uid = uid

        # Store first token to yield on first step()
        self._pp_first_token = _first_out.token

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
            prefill_tps=prefill_tps,
        )

        return uid

    def _step_pp_spec(self) -> list[MlxBatchGenerator.Response]:
        """Get next token from PP speculative decode loop."""
        uid = self._pp_spec_uid
        assert uid is not None

        # Yield the first token if we haven't yet
        if hasattr(self, '_pp_first_token'):
            tok = self._pp_first_token
            del self._pp_first_token
            # Don't check EOS here — outer step() handles stop detection
            # (tokenizer EOS includes <think> which is not generation-ending)
            return [MlxBatchGenerator.Response(
                uid=uid, token=tok, logprobs=mx.zeros(1),
                finish_reason=None, prompt_cache=lambda: [],
            )]

        assert self._pp_spec_gen is not None
        try:
            tok_id, lp = next(self._pp_spec_gen)
            return [MlxBatchGenerator.Response(
                uid=uid, token=tok_id, logprobs=lp,
                finish_reason=None, prompt_cache=lambda: [],
            )]
        except StopIteration:
            # max_tokens reached
            self._pp_spec_gen = None
            self._pp_spec_uid = None
            return [MlxBatchGenerator.Response(
                uid=uid, token=0, logprobs=mx.zeros(1),
                finish_reason="length", prompt_cache=lambda: [],
            )]

    def step(self) -> list[tuple[int, GenerationResponse]]:
        if not self.has_work:
            return []

        # Use PP speculation decode if active
        if self._pp_spec_gen is not None:
            _step_tic = time.perf_counter()
            responses = self._step_pp_spec()
            _next_elapsed = time.perf_counter() - _step_tic
        else:
            self._mlx_gen._needs_topk = any(  # pyright: ignore[reportAttributeAccessIssue]
                t.task_params.logprobs for t in self._active_tasks.values()
            )
            _step_tic = time.perf_counter()
            responses = self._mlx_gen.next()
            _next_elapsed = time.perf_counter() - _step_tic

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
            if task_params.logprobs and os.environ.get("EXO_DISABLE_LOGPROBS") != "1":
                with mx.stream(generation_stream):
                    logprob, top_logprobs = extract_top_logprobs(
                        logprobs=response.logprobs,
                        tokenizer=self.tokenizer,
                        top_logprobs=task_params.top_logprobs or DEFAULT_TOP_LOGPROBS,
                        selected_token=response.token,
                        precomputed_indices=getattr(response, "_topk_indices", None),
                        precomputed_values=getattr(response, "_topk_values", None),
                        precomputed_selected=getattr(
                            response, "_selected_logprob", None
                        ),
                    )

            stats: GenerationStats | None = None
            usage: Usage | None = None
            if is_done:
                if self._pp_spec_gen is not None or self._pp_spec_uid is not None:
                    gen_elapsed = time.perf_counter() - state.generation_start_time
                    generation_tps = (
                        state.completion_tokens / gen_elapsed
                        if gen_elapsed > 0
                        else 0.0
                    )
                    # Clean up spec state
                    self._pp_spec_gen = None
                    self._pp_spec_uid = None
                else:
                    gen_time_delta = (
                        self._mlx_gen._stats.generation_time
                        - state.generation_time_at_start
                    )
                    generation_tps = (
                        state.completion_tokens / gen_time_delta
                        if gen_time_delta > 0
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

        _step_elapsed = time.perf_counter() - _step_tic
        _overhead = _step_elapsed - _next_elapsed
        if self._mlx_gen._next_count % 64 == 0 and responses:
            logger.debug(
                f"step overhead: {_overhead * 1000:.2f}ms (next={_next_elapsed * 1000:.2f}ms total={_step_elapsed * 1000:.2f}ms)"
            )

        return results

    def cancel(self, uids: list[int]) -> None:
        self._mlx_gen.remove(uids)
        for uid in uids:
            self._active_tasks.pop(uid, None)

    def close(self) -> None:
        self._mlx_gen.close()

    def _save_prefix_cache(
        self,
        all_prompt_tokens: mx.array,
        cache: KVCacheType,
        cache_snapshots: list[CacheSnapshot] | None,
        prefix_hit_length: int,
        matched_index: int | None,
        min_prefix_hit_length: int = 1000,
        media_regions: list[MediaRegion] | None = None,
    ) -> None:
        if self.kv_prefix_cache is None:
            return

        try:
            hit_ratio = (
                prefix_hit_length / len(all_prompt_tokens)
                if len(all_prompt_tokens) > 0
                else 0.0
            )
            if matched_index is not None and (
                prefix_hit_length >= min_prefix_hit_length
                and hit_ratio >= _MIN_PREFIX_HIT_RATIO_TO_UPDATE
            ):
                self.kv_prefix_cache.update_kv_cache(
                    matched_index,
                    all_prompt_tokens,
                    cache,
                    cache_snapshots,
                    restore_pos=prefix_hit_length,
                    media_regions=media_regions,
                )
            else:
                self.kv_prefix_cache.add_kv_cache(
                    all_prompt_tokens,
                    cache,
                    cache_snapshots,
                    media_regions=media_regions,
                )
        except Exception:
            logger.warning("Failed to save prefix cache", exc_info=True)
