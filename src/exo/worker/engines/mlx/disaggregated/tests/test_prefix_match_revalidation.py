"""A background prefill must not save through a prefix-cache index that has moved.

submit() records the index of the prefix-cache entry it matched; the background fetch
finishes later, after other requests may have saved or evicted entries. Eviction removes
entries by position, so the recorded index can name a different prompt by then.
"""

import mlx.core as mx

from exo.shared.types.common import ModelId
from exo.shared.types.text_generation import (
    InputMessage,
    InputMessageContent,
    TextGenerationTaskParams,
)
from exo.worker.engines.mlx.cache import KVPrefixCache
from exo.worker.engines.mlx.generator.batch_generate import (
    ExoBatchGenerator,
    _PreparedSubmission,  # pyright: ignore[reportPrivateUsage]
)

_SHARED = [5, 6, 7, 8, 9, 10]


def _generator(prompts: list[list[int]]) -> ExoBatchGenerator:
    gen = object.__new__(ExoBatchGenerator)
    cache = KVPrefixCache(None)
    cache.prompts = [mx.array(p) for p in prompts]
    gen.kv_prefix_cache = cache
    return gen


def _prepared(matched_index: int | None) -> _PreparedSubmission:
    return _PreparedSubmission(
        task_params=TextGenerationTaskParams(
            model=ModelId("m"),
            input=[InputMessage(role="user", content=InputMessageContent("x"))],
        ),
        cache=[],
        sampler=lambda logits: logits,
        prompt_tokens=mx.array([11, 12]),
        all_prompt_tokens=mx.array([*_SHARED, 11, 12]),
        prefix_hit_length=len(_SHARED),
        matched_index=matched_index,
        is_exact_hit=False,
        media_regions=[],
        is_bench=False,
        on_generation_token=None,
    )


def test_unmoved_entry_is_kept() -> None:
    gen = _generator([[1, 2, 3], [*_SHARED, 99]])
    revalidated = gen._revalidate_prefix_match(_prepared(1))  # pyright: ignore[reportPrivateUsage]
    assert revalidated.matched_index == 1


def test_moved_entry_is_found_again() -> None:
    # Matched at index 1; entry 0 was evicted since, so the prefix now sits at index 0
    # and index 1 holds an unrelated prompt.
    gen = _generator([[*_SHARED, 99], [42, 43, 44]])
    revalidated = gen._revalidate_prefix_match(_prepared(1))  # pyright: ignore[reportPrivateUsage]
    assert revalidated.matched_index == 0


def test_evicted_entry_saves_as_new() -> None:
    gen = _generator([[42, 43, 44]])
    revalidated = gen._revalidate_prefix_match(_prepared(1))  # pyright: ignore[reportPrivateUsage]
    assert revalidated.matched_index is None
