import time

import mlx.core as mx
from mlx_lm.sample_utils import make_sampler
from mlx_lm.tokenizer_utils import TokenizerWrapper

from exo.worker.disaggregated.server import PrefillRequest
from exo.worker.engines.mlx.cache import (
    KVPrefixCache,
    cache_length,
    make_kv_cache,
    snapshot_ssm_states,
)
from exo.worker.engines.mlx.generator.generate import prefill as mlx_prefill
from exo.worker.engines.mlx.types import KVCacheType, Model
from exo.worker.engines.mlx.utils_mlx import fix_unmatched_think_end_tokens
from exo.worker.runner.bootstrap import logger


def run_prefill_for_request(
    *,
    model: Model,
    tokenizer: TokenizerWrapper,
    group: mx.distributed.Group | None,
    kv_prefix_cache: KVPrefixCache | None,
    request: PrefillRequest,
) -> KVCacheType:
    prompt_tokens = mx.array(request.token_ids)
    prompt_tokens = fix_unmatched_think_end_tokens(prompt_tokens, tokenizer)
    n_tokens = int(prompt_tokens.shape[0])
    t0 = time.perf_counter()

    matched_index: int | None = None
    prefix_hit_length = 0
    if kv_prefix_cache is not None:
        cache, remaining, matched_index, _ = kv_prefix_cache.get_kv_cache(
            model, prompt_tokens
        )
        prefix_hit_length = n_tokens - int(remaining.shape[0])
    else:
        cache = make_kv_cache(model)
        remaining = prompt_tokens

    target_offset = max(0, n_tokens - 2)
    new_tokens = max(0, target_offset - prefix_hit_length)
    prefill_input = remaining[:new_tokens]
    if int(prefill_input.shape[0]) > 0:
        sampler = make_sampler(temp=1.0)
        _ = mlx_prefill(
            model=model,
            tokenizer=tokenizer,
            sampler=sampler,
            prompt_tokens=prefill_input,
            cache=cache,
            group=group,
            on_prefill_progress=None,
            distributed_prompt_progress_callback=None,
        )

    if kv_prefix_cache is not None:
        try:
            cache_snapshots = [snapshot_ssm_states(cache)]
            hit_ratio = prefix_hit_length / n_tokens if n_tokens > 0 else 0.0
            if matched_index is not None and hit_ratio >= 0.5:
                kv_prefix_cache.update_kv_cache(
                    matched_index,
                    prompt_tokens,
                    cache,
                    cache_snapshots,
                    restore_pos=prefix_hit_length,
                )
            else:
                kv_prefix_cache.add_kv_cache(prompt_tokens, cache, cache_snapshots)
        except Exception:
            logger.opt(exception=True).warning(
                "Failed to save prefix cache on prefill server"
            )

    elapsed = time.perf_counter() - t0
    final_offset = cache_length(cache)
    logger.info(
        f"Prefill: request_id={request.request_id} "
        f"{n_tokens} tokens (prefix_hit={prefix_hit_length}, "
        f"final_offset={final_offset}) in {elapsed * 1000:.0f}ms"
    )
    return cache
