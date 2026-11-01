"""Local prefill inside the running batch (EXO_NONBLOCKING_PREFILL).

With the flag on, submit() leaves a qualifying prompt to mlx_lm's BatchGenerator, which
prefills it a chunk per step between decode steps, instead of prefilling it before it
returns. These tests drive the real ExoBatchGenerator with a small random-weight Llama and
a word-level tokenizer built in memory, so they need no download and run on MLX CPU.
"""

import contextlib
import functools
import sys
from collections.abc import Callable, Iterator
from typing import cast

import mlx.core as mx
import mlx.nn as nn
import pytest
from mlx_lm.generate import stream_generate
from mlx_lm.models import (
    llama,  # pyright: ignore[reportAttributeAccessIssue, reportUnknownVariableType]
)
from mlx_lm.models.cache import (
    ArraysCache,
    BatchKVCache,
    KVCache,
    QuantizedKVCache,
    RotatingKVCache,
)
from mlx_lm.tokenizer_utils import TokenizerWrapper
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import WhitespaceSplit
from transformers import PreTrainedTokenizerFast

from exo.shared.types.common import ModelId
from exo.shared.types.text_generation import (
    InputMessage,
    InputMessageContent,
    TextGenerationTaskParams,
)
from exo.shared.types.worker.runner_response import GenerationResponse
from exo.worker.engines.mlx import cache as cache_module
from exo.worker.engines.mlx.cache import KVPrefixCache, cache_length
from exo.worker.engines.mlx.generator import batch_generate, generate
from exo.worker.engines.mlx.generator.batch_generate import (
    ExoBatchGenerator,
    can_prefill_in_batch,
    matched_index_if_unchanged,
    prefill_tokens_per_prompt,
)
from exo.worker.engines.mlx.generator.generate import PrefillCancelled
from exo.worker.engines.mlx.patches import opt_batch_gen
from exo.worker.engines.mlx.patches.opt_batch_gen import apply_batch_gen_patch
from exo.worker.engines.mlx.types import KVCacheType, Model
from exo.worker.engines.mlx.vision import VisionResult

_VOCABULARY_SIZE = 128
# Tokens 0 and 1 are reserved, so prompts use 2..127
_FIRST_WORD = 2
_STEP_TOKENS = 64

PrefillResult = tuple[float, int, list[cache_module.CacheSnapshot]]


def _word(token: int) -> str:
    return f"t{token:03d}"


def _tokenizer() -> TokenizerWrapper:
    vocabulary = {"<unk>": 0, "<pad>": 1} | {
        _word(token): token for token in range(_FIRST_WORD, _VOCABULARY_SIZE)
    }
    tokenizer = Tokenizer(WordLevel(vocabulary, unk_token="<unk>"))
    tokenizer.pre_tokenizer = WhitespaceSplit()
    # No stop tokens: every request runs to its token limit, whatever the random model says
    return TokenizerWrapper(
        PreTrainedTokenizerFast(tokenizer_object=tokenizer, unk_token="<unk>"),
        eos_token_ids=[],
    )


def _model(hidden_size: int = 64, layers: int = 2, seed: int = 0) -> Model:
    """A small Llama with random weights: plain KVCache, like the models this targets.

    EXO ships no type stub for mlx_lm's llama module, hence the ignores.
    """
    args = llama.ModelArgs(  # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType]
        model_type="llama",
        hidden_size=hidden_size,
        num_hidden_layers=layers,
        intermediate_size=hidden_size * 2,
        num_attention_heads=4,
        num_key_value_heads=2,
        rms_norm_eps=1e-5,
        vocab_size=_VOCABULARY_SIZE,
    )
    mx.random.seed(seed)
    model = llama.Model(args)  # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType]
    mx.eval(model.parameters())  # pyright: ignore[reportUnknownMemberType, reportUnknownArgumentType]
    return cast(Model, cast(object, model))


def _tokens(count: int, salt: int = 0) -> list[int]:
    span = _VOCABULARY_SIZE - _FIRST_WORD
    return [
        _FIRST_WORD + (index * 7 + salt * 13 + index // 5) % span
        for index in range(count)
    ]


def _prompt(tokens: list[int]) -> str:
    return " ".join(_word(token) for token in tokens)


def _params(max_tokens: int) -> TextGenerationTaskParams:
    return TextGenerationTaskParams(
        model=ModelId("tiny-llama"),
        input=[InputMessage(role="user", content=InputMessageContent("x"))],
        max_output_tokens=max_tokens,
        temperature=0.0,
        logprobs=True,
    )


@pytest.fixture(scope="module")
def tokenizer() -> TokenizerWrapper:
    return _tokenizer()


@pytest.fixture(scope="module")
def model() -> Model:
    return _model()


@pytest.fixture(autouse=True)
def runner_setup(monkeypatch: pytest.MonkeyPatch) -> None:
    # The runner installs EXO's mlx_lm batch patches; memory pressure on the test machine
    # must not evict prefix-cache entries behind a test's back.
    apply_batch_gen_patch()
    monkeypatch.setattr(cache_module, "get_memory_used_percentage", lambda: 0.0)


@pytest.fixture
def nonblocking(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(batch_generate, "EXO_NONBLOCKING_PREFILL", True)
    monkeypatch.setattr(batch_generate, "EXO_PREFILL_STEP_TOKENS", _STEP_TOKENS)


@pytest.fixture
def cpu_device(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Runs the engine on the CPU device. Setting the default device is not enough:
    mlx_lm made generation_stream on the default device when it was imported, the engine
    enters it explicitly, and stream_generate takes it as a default argument."""
    cpu = mx.default_stream(mx.cpu)
    # The package's mlx_lm.generate attribute is the function, not the module
    mlx_generate = sys.modules["mlx_lm.generate"]
    for module in (mlx_generate, generate, batch_generate, opt_batch_gen):
        monkeypatch.setattr(module, "generation_stream", cpu)
    monkeypatch.setattr(
        generate, "stream_generate", functools.partial(stream_generate, stream=cpu)
    )
    device = mx.default_device()
    mx.set_default_device(mx.cpu)
    try:
        yield
    finally:
        mx.set_default_device(device)


def _no_synchronous_prefill(monkeypatch: pytest.MonkeyPatch) -> None:
    def refuse(*_: object) -> PrefillResult:
        raise AssertionError(
            "prefill() ran: the prompt should be prefilled in the batch"
        )

    monkeypatch.setattr(batch_generate, "prefill", refuse)


def _record_synchronous_prefill(
    monkeypatch: pytest.MonkeyPatch,
) -> list[int]:
    """Replace prefill() with a stub that records the prompt length and fills nothing."""
    calls: list[int] = []

    def record(
        _model: Model,
        _tokenizer: TokenizerWrapper,
        _sampler: Callable[[mx.array], mx.array],
        prompt_tokens: mx.array,
        *_: object,
    ) -> PrefillResult:
        calls.append(len(prompt_tokens))
        return 0.0, 0, []

    monkeypatch.setattr(batch_generate, "prefill", record)
    return calls


def _drain(
    generator: ExoBatchGenerator,
) -> dict[int, list[GenerationResponse]]:
    responses: dict[int, list[GenerationResponse]] = {}
    while generator.has_work:
        for uid, response in generator.step():
            responses.setdefault(uid, []).append(response)
    return responses


def _prefilling(generator: ExoBatchGenerator) -> dict[int, object]:
    return cast(dict[int, object], generator._prefills_in_batch)  # pyright: ignore[reportPrivateUsage]


def _mlx_prefill_step_size(generator: ExoBatchGenerator) -> int:
    return generator._mlx_gen.prefill_step_size  # pyright: ignore[reportPrivateUsage]


def _uids_in_prompt_batch(generator: ExoBatchGenerator) -> list[int]:
    prompt_batch = generator._mlx_gen._prompt_batch  # pyright: ignore[reportPrivateUsage]
    # The stub does not declare uids on PromptProcessingBatch
    return cast(list[int], getattr(prompt_batch, "uids"))  # noqa: B009


# Which requests qualify


def _plain_cache() -> KVCacheType:
    return [KVCache(), KVCache()]


@pytest.mark.parametrize(
    (
        "cache",
        "uncached_tokens",
        "uses_remote_prefill",
        "has_vision",
        "group_size",
        "expected",
    ),
    [
        (_plain_cache(), 500, False, False, 1, True),
        (_plain_cache(), 3, False, False, 1, True),
        (_plain_cache(), 2, False, False, 1, False),
        (_plain_cache(), 500, True, False, 1, False),
        (_plain_cache(), 500, False, True, 1, False),
        (_plain_cache(), 500, False, False, 2, False),
        ([KVCache(), RotatingKVCache(max_size=128)], 500, False, False, 1, False),
        ([KVCache(), ArraysCache(size=2)], 500, False, False, 1, False),
        ([QuantizedKVCache(group_size=32, bits=4)], 500, False, False, 1, False),
    ],
    ids=[
        "plain-kv",
        "three-tokens",
        "two-tokens",
        "remote",
        "vision",
        "multi-node",
        "rotating",
        "ssm",
        "quantised",
    ],
)
def test_which_requests_are_prefilled_in_the_batch(
    cache: KVCacheType,
    uncached_tokens: int,
    uses_remote_prefill: bool,
    has_vision: bool,
    group_size: int,
    expected: bool,
) -> None:
    assert (
        can_prefill_in_batch(
            cache,
            uncached_tokens=uncached_tokens,
            uses_remote_prefill=uses_remote_prefill,
            has_vision=has_vision,
            group_size=group_size,
        )
        is expected
    )


def test_the_step_budget_is_shared_between_prompts() -> None:
    assert prefill_tokens_per_prompt(512, 0) == 512
    assert prefill_tokens_per_prompt(512, 1) == 512
    assert prefill_tokens_per_prompt(512, 2) == 256
    # Each prompt still moves when many share a small budget
    assert prefill_tokens_per_prompt(512, 32) == 32
    assert prefill_tokens_per_prompt(100, 8) == 32


# The prefix-cache entry matched at submit, checked again at save

_SHARED = [5, 6, 7, 8, 9, 10]
_PROMPT = mx.array([*_SHARED, 11, 12])


def test_an_unmoved_entry_is_updated() -> None:
    cached = [mx.array([1, 2, 3]), mx.array([*_SHARED, 99])]
    assert matched_index_if_unchanged(cached, _PROMPT, len(_SHARED), 1) == 1


def test_a_moved_entry_is_not_updated_through_its_old_index() -> None:
    # Entry 0 was evicted: the matched prefix is now at index 0, and index 1 holds an
    # unrelated prompt that an update would overwrite.
    cached = [mx.array([*_SHARED, 99]), mx.array([42, 43, 44])]
    assert matched_index_if_unchanged(cached, _PROMPT, len(_SHARED), 1) is None


def test_an_evicted_entry_is_saved_as_new() -> None:
    cached = [mx.array([42, 43, 44])]
    assert matched_index_if_unchanged(cached, _PROMPT, len(_SHARED), 1) is None
    assert matched_index_if_unchanged(cached, _PROMPT, 0, None) is None


# submit() and step() with a real model


@pytest.mark.usefixtures("nonblocking")
def test_submit_returns_before_prefill_and_streams_keep_decoding(
    model: Model, tokenizer: TokenizerWrapper, monkeypatch: pytest.MonkeyPatch
) -> None:
    _no_synchronous_prefill(monkeypatch)
    generator = ExoBatchGenerator(
        model=model,
        tokenizer=tokenizer,
        group=None,
        kv_prefix_cache=KVPrefixCache(None),
    )
    stream = generator.submit(_params(max_tokens=200), _prompt(_tokens(8, salt=1)))
    while not any(uid == stream for uid, _ in generator.step()):
        pass

    long_tokens = _tokens(600)
    progress: list[tuple[int, int]] = []
    long_prompt = generator.submit(
        _params(max_tokens=4),
        _prompt(long_tokens),
        on_prefill_progress=lambda processed, total: progress.append(
            (processed, total)
        ),
    )
    assert long_prompt in _prefilling(generator)

    steps_before_first_token = 0
    while True:
        step_uids = [uid for uid, _ in generator.step()]
        assert step_uids.count(stream) == 1, "the stream missed a step"
        if long_prompt in step_uids:
            break
        steps_before_first_token += 1
        if long_prompt in _prefilling(generator):
            assert long_prompt in _uids_in_prompt_batch(generator)

    # 598 tokens at 64 a step, then the two tail tokens
    assert steps_before_first_token >= 598 // _STEP_TOKENS
    assert long_prompt not in _prefilling(generator)
    assert progress[0] == (0, 599)
    assert progress[-1] == (599, 599)
    assert [processed for processed, _ in progress] == sorted(
        processed for processed, _ in progress
    )

    prefix_cache = generator.kv_prefix_cache
    assert prefix_cache is not None
    saved = [
        index
        for index, prompt in enumerate(prefix_cache.prompts)
        if prompt.tolist() == long_tokens
    ]
    assert len(saved) == 1
    assert cache_length(prefix_cache.caches[saved[0]]) == len(long_tokens) - 2
    _drain(generator)


def test_with_the_flag_off_prefill_runs_in_submit(
    model: Model, tokenizer: TokenizerWrapper, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls = _record_synchronous_prefill(monkeypatch)
    generator = ExoBatchGenerator(
        model=model, tokenizer=tokenizer, group=None, kv_prefix_cache=None
    )
    uid = generator.submit(_params(max_tokens=2), _prompt(_tokens(300)))

    assert calls == [299]
    assert uid not in _prefilling(generator)
    generator.step()
    assert _mlx_prefill_step_size(generator) == 4096


class _TwoRankGroup:
    def size(self) -> int:
        return 2


def _rotating_cache(model: Model, *_: object) -> KVCacheType:
    return [RotatingKVCache(max_size=4096) for _ in model.layers]


def _text_as_vision(**_: object) -> VisionResult:
    return VisionResult(
        prompt="",
        prompt_tokens=mx.array(_tokens(300)),
        embeddings=mx.zeros((1, 8)),
        media_regions=[],
        image_token_id=0,
    )


def _no_embedding_patch(*_: object, **__: object) -> contextlib.nullcontext[None]:
    return contextlib.nullcontext()


def _remote_prefill_fails(*_: object, **__: object) -> PrefillResult:
    raise ConnectionError("prefill server unreachable")


@pytest.mark.usefixtures("nonblocking")
@pytest.mark.parametrize(
    "excluded_by", ["rotating-cache", "multi-node", "vision", "remote"]
)
def test_excluded_requests_prefill_in_submit(
    excluded_by: str,
    model: Model,
    tokenizer: TokenizerWrapper,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = _record_synchronous_prefill(monkeypatch)
    group: mx.distributed.Group | None = None
    vision_processor = None
    params = _params(max_tokens=2)
    if excluded_by == "rotating-cache":
        monkeypatch.setattr(batch_generate, "make_kv_cache", _rotating_cache)
    elif excluded_by == "multi-node":
        group = cast(mx.distributed.Group, cast(object, _TwoRankGroup()))
    elif excluded_by == "vision":
        vision_processor = cast(batch_generate.VisionProcessor, object())
        monkeypatch.setattr(batch_generate, "prepare_vision", _text_as_vision)
        monkeypatch.setattr(
            batch_generate,
            "patch_embed_tokens",
            _no_embedding_patch,
        )
    else:
        # The local fallback after a failed remote prefill stays synchronous too
        monkeypatch.setattr(batch_generate, "REMOTE_PREFILL_MIN_TOKENS", 100)
        monkeypatch.setattr(batch_generate, "remote_prefill", _remote_prefill_fails)
        params = params.model_copy(update={"prefill_endpoint": "127.0.0.1:1"})

    generator = ExoBatchGenerator(
        model=model,
        tokenizer=tokenizer,
        group=group,
        kv_prefix_cache=None,
        vision_processor=vision_processor,
    )
    uid = generator.submit(params, _prompt(_tokens(300)))

    assert calls == [299]
    assert uid not in _prefilling(generator)


# The prefix cache


def _run_alone(
    generator: ExoBatchGenerator, tokens: list[int], max_tokens: int = 2
) -> list[GenerationResponse]:
    uid = generator.submit(_params(max_tokens=max_tokens), _prompt(tokens))
    return _drain(generator)[uid]


def _entry_holding(prefix_cache: KVPrefixCache, tokens: list[int]) -> int:
    matches = [
        index
        for index, prompt in enumerate(prefix_cache.prompts)
        if prompt.tolist() == tokens
    ]
    assert len(matches) == 1
    return matches[0]


def test_the_saved_prefix_matches_the_synchronous_prefill(
    model: Model, tokenizer: TokenizerWrapper, monkeypatch: pytest.MonkeyPatch
) -> None:
    tokens = _tokens(1200)
    saved: list[KVCacheType] = []
    for enabled in (False, True):
        monkeypatch.setattr(batch_generate, "EXO_NONBLOCKING_PREFILL", enabled)
        monkeypatch.setattr(batch_generate, "EXO_PREFILL_STEP_TOKENS", _STEP_TOKENS)
        prefix_cache = KVPrefixCache(None)
        generator = ExoBatchGenerator(
            model=model, tokenizer=tokenizer, group=None, kv_prefix_cache=prefix_cache
        )
        _run_alone(generator, tokens)
        saved.append(prefix_cache.caches[_entry_holding(prefix_cache, tokens)])

    synchronous, in_batch = saved
    assert cache_length(in_batch) == cache_length(synchronous) == len(tokens) - 2
    for expected, actual in zip(
        cast(list[KVCache], synchronous), cast(list[KVCache], in_batch), strict=True
    ):
        assert expected.offset == actual.offset
        for expected_array, actual_array in (
            (expected.keys, actual.keys),
            (expected.values, actual.values),
        ):
            assert expected_array is not None and actual_array is not None
            # The buffers grow in steps, so only the first offset positions hold tokens
            assert mx.allclose(
                expected_array[..., : expected.offset, :],
                actual_array[..., : actual.offset, :],
                atol=1e-4,
            ).item()


def _evict(
    prefix_cache: KVPrefixCache, index: int, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Evict one entry the way memory pressure would: least recently used first."""
    remaining = len(prefix_cache.caches) - 1
    last_used = prefix_cache._last_used  # pyright: ignore[reportPrivateUsage]
    last_used[index] = min(last_used) - 1

    def over_limit(_incoming_bytes: int) -> bool:
        return len(prefix_cache.caches) > remaining

    with monkeypatch.context() as patch:
        patch.setattr(prefix_cache, "_over_limit", over_limit)
        prefix_cache._evict_if_needed()  # pyright: ignore[reportPrivateUsage]


@pytest.mark.usefixtures("nonblocking")
def test_an_entry_evicted_during_prefill_does_not_overwrite_its_neighbour(
    model: Model, tokenizer: TokenizerWrapper, monkeypatch: pytest.MonkeyPatch
) -> None:
    prefix_cache = KVPrefixCache(None)
    generator = ExoBatchGenerator(
        model=model, tokenizer=tokenizer, group=None, kv_prefix_cache=prefix_cache
    )
    document = _tokens(1200)
    filler = _tokens(40, salt=3)
    _run_alone(generator, document)
    _run_alone(generator, filler)
    assert [prompt.tolist() for prompt in prefix_cache.prompts] == [document, filler]

    # A partial hit on entry 0 long enough that its save would update entry 0 in place
    follow_up = document + _tokens(600, salt=5)
    uid = generator.submit(_params(max_tokens=2), _prompt(follow_up))
    generator.step()
    generator.step()
    assert uid in _prefilling(generator)
    _evict(prefix_cache, 0, monkeypatch)
    assert [prompt.tolist() for prompt in prefix_cache.prompts] == [filler]
    _drain(generator)

    assert [prompt.tolist() for prompt in prefix_cache.prompts] == [filler, follow_up]
    assert cache_length(prefix_cache.caches[0]) == len(filler) - 2
    assert cache_length(prefix_cache.caches[1]) == len(follow_up) - 2


@pytest.mark.usefixtures("nonblocking")
def test_an_unmoved_entry_is_updated_in_place(
    model: Model, tokenizer: TokenizerWrapper
) -> None:
    prefix_cache = KVPrefixCache(None)
    generator = ExoBatchGenerator(
        model=model, tokenizer=tokenizer, group=None, kv_prefix_cache=prefix_cache
    )
    document = _tokens(1200)
    _run_alone(generator, document)
    follow_up = document + _tokens(600, salt=5)
    _run_alone(generator, follow_up)

    assert [prompt.tolist() for prompt in prefix_cache.prompts] == [follow_up]


# Cancellation


@pytest.mark.usefixtures("nonblocking")
def test_a_prompt_cancelled_while_prefilling_leaves_the_batch(
    model: Model, tokenizer: TokenizerWrapper
) -> None:
    prefix_cache = KVPrefixCache(None)
    generator = ExoBatchGenerator(
        model=model, tokenizer=tokenizer, group=None, kv_prefix_cache=prefix_cache
    )
    stream = generator.submit(_params(max_tokens=30), _prompt(_tokens(8, salt=1)))
    long_tokens = _tokens(600)
    long_prompt = generator.submit(_params(max_tokens=4), _prompt(long_tokens))
    generator.step()
    generator.step()
    assert long_prompt in _uids_in_prompt_batch(generator)

    generator.cancel([long_prompt])
    responses = _drain(generator)

    assert long_prompt not in responses
    assert len(responses[stream]) == 30
    assert not _prefilling(generator)
    assert not generator.has_work
    assert all(prompt.tolist() != long_tokens for prompt in prefix_cache.prompts)


@pytest.mark.usefixtures("nonblocking")
def test_the_cancel_hook_does_not_abort_the_step(
    model: Model, tokenizer: TokenizerWrapper
) -> None:
    generator = ExoBatchGenerator(
        model=model, tokenizer=tokenizer, group=None, kv_prefix_cache=None
    )
    stream = generator.submit(_params(max_tokens=30), _prompt(_tokens(8, salt=1)))
    hook_calls = 0

    def cancel_on_second_chunk() -> None:
        # The runner's hook records the cancellation, then raises
        nonlocal hook_calls
        hook_calls += 1
        if hook_calls == 2:
            raise PrefillCancelled()

    long_prompt = generator.submit(
        _params(max_tokens=4),
        _prompt(_tokens(600)),
        distributed_prompt_progress_callback=cancel_on_second_chunk,
    )
    while hook_calls < 2:
        generator.step()
    # As the runner does at the end of its step
    generator.cancel([long_prompt])
    responses = _drain(generator)

    assert hook_calls == 2
    assert long_prompt not in responses
    assert len(responses[stream]) == 30
    assert not generator.has_work


# The batch cache's dtype


def _half_precision_model() -> Model:
    model = _model()
    # nn.Module.set_dtype returns None, which EXO's mlx stub leaves unannotated
    cast(nn.Module, cast(object, model)).set_dtype(mx.float16)  # pyright: ignore[reportUnknownMemberType]
    return model


def _prompt_batch_dtypes(generator: ExoBatchGenerator) -> set[mx.Dtype]:
    prompt_batch = generator._mlx_gen._prompt_batch  # pyright: ignore[reportPrivateUsage]
    # The stub does not declare prompt_cache on PromptProcessingBatch
    prompt_cache = cast(list[BatchKVCache], getattr(prompt_batch, "prompt_cache"))  # noqa: B009
    return {
        array.dtype
        for entry in prompt_cache
        for array in (entry.keys, entry.values)
        if array is not None
    }


@pytest.mark.usefixtures("nonblocking")
def test_a_fresh_prompt_joining_keeps_the_batch_cache_in_float16(
    tokenizer: TokenizerWrapper,
) -> None:
    """A fresh prompt joins a prompt that has already been prefilled in part, which the
    synchronous path never does. Before mlx-lm #1491, BatchKVCache.extend turned the prompt
    batch's cache to float32 here, and those sequences then decoded in float32."""
    generator = ExoBatchGenerator(
        model=_half_precision_model(),
        tokenizer=tokenizer,
        group=None,
        kv_prefix_cache=None,
    )
    first = generator.submit(_params(max_tokens=2), _prompt(_tokens(600)))
    generator.step()
    assert first in _uids_in_prompt_batch(generator)
    assert _prompt_batch_dtypes(generator) == {mx.float16}

    second = generator.submit(_params(max_tokens=2), _prompt(_tokens(300, salt=1)))
    assert second in _prefilling(generator)
    generator.step()
    assert {first, second} <= set(_uids_in_prompt_batch(generator))

    assert _prompt_batch_dtypes(generator) == {mx.float16}
    responses = _drain(generator)
    assert len(responses[first]) == len(responses[second]) == 2


# Penalties


def _recording_penalty(
    contexts: list[list[int]],
) -> Callable[..., list[Callable[[mx.array, mx.array], mx.array]]]:
    def record(tokens: mx.array, logits: mx.array) -> mx.array:
        contexts.append(cast(list[int], tokens.tolist()))
        return logits

    def make(**_: object) -> list[Callable[[mx.array, mx.array], mx.array]]:
        return [record]

    return make


def test_penalties_see_the_same_tokens_as_on_the_synchronous_path(
    model: Model,
    tokenizer: TokenizerWrapper,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tokens = _tokens(600)
    seen: list[list[list[int]]] = []
    for enabled in (False, True):
        monkeypatch.setattr(batch_generate, "EXO_NONBLOCKING_PREFILL", enabled)
        monkeypatch.setattr(batch_generate, "EXO_PREFILL_STEP_TOKENS", _STEP_TOKENS)
        contexts: list[list[int]] = []
        monkeypatch.setattr(
            batch_generate, "make_logits_processors", _recording_penalty(contexts)
        )
        generator = ExoBatchGenerator(
            model=model, tokenizer=tokenizer, group=None, kv_prefix_cache=None
        )
        _run_alone(generator, tokens, max_tokens=4)
        seen.append(contexts)

    synchronous, in_batch = seen
    # The prompt's second-to-last token, then one more token each step
    assert synchronous[0] == tokens[-2:-1]
    assert in_batch == synchronous


# Metal float32 rounding exceeds the 1e-3 bound on the unchanged synchronous path too
@pytest.mark.usefixtures("cpu_device")
@pytest.mark.parametrize(
    "penalty",
    [{"presence_penalty": 1.5}, {"repetition_penalty": 1.3}],
    ids=["presence", "repetition"],
)
def test_penalties_give_the_synchronous_logprobs(
    penalty: dict[str, float],
    model: Model,
    tokenizer: TokenizerWrapper,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Both penalties look at the last 20 tokens of the context, so had the prompt been
    counted, the first replies' log-probabilities would differ."""
    tokens = _tokens(600)
    params = _params(max_tokens=16).model_copy(update=penalty)
    runs: list[list[tuple[int, float]]] = []
    for enabled in (False, True):
        monkeypatch.setattr(batch_generate, "EXO_NONBLOCKING_PREFILL", enabled)
        monkeypatch.setattr(batch_generate, "EXO_PREFILL_STEP_TOKENS", _STEP_TOKENS)
        generator = ExoBatchGenerator(
            model=model, tokenizer=tokenizer, group=None, kv_prefix_cache=None
        )
        uid = generator.submit(params, _prompt(tokens))
        assert (uid in _prefilling(generator)) is enabled
        runs.append(
            [
                (response.token, cast(float, response.logprob))
                for response in _drain(generator)[uid]
            ]
        )

    synchronous, in_batch = runs
    assert [token for token, _ in in_batch] == [token for token, _ in synchronous]
    assert (
        max(abs(a - e) for (_, a), (_, e) in zip(in_batch, synchronous, strict=True))
        < 1e-3
    )


# Output is unchanged


def _logprobs_by_prompt(
    model: Model,
    tokenizer: TokenizerWrapper,
    prompts: list[list[int]],
    concurrent: bool,
) -> list[list[tuple[int, float]]]:
    generator = ExoBatchGenerator(
        model=model,
        tokenizer=tokenizer,
        group=None,
        kv_prefix_cache=KVPrefixCache(None),
    )
    if concurrent:
        uids = [generator.submit(_params(max_tokens=24), _prompt(p)) for p in prompts]
        responses = _drain(generator)
        by_uid = [responses[uid] for uid in uids]
    else:
        by_uid = [_run_alone(generator, p, max_tokens=24) for p in prompts]
    return [
        [(response.token, cast(float, response.logprob)) for response in responses]
        for responses in by_uid
    ]


@pytest.mark.slow
# Metal float32 rounding exceeds the 1e-3 bound on the unchanged synchronous path too
@pytest.mark.usefixtures("cpu_device")
@pytest.mark.parametrize("concurrent", [False, True], ids=["alone", "concurrent"])
def test_prefill_in_batch_matches_synchronous_logprobs(
    tokenizer: TokenizerWrapper, monkeypatch: pytest.MonkeyPatch, concurrent: bool
) -> None:
    model = _model(hidden_size=256, layers=4, seed=1)
    document = _tokens(3000, salt=2)
    # Fresh, then an exact and a partial prefix hit on the first
    prompts = [document, document, document + _tokens(700, salt=4), _tokens(50, salt=6)]

    runs: list[list[list[tuple[int, float]]]] = []
    for enabled in (False, True):
        monkeypatch.setattr(batch_generate, "EXO_NONBLOCKING_PREFILL", enabled)
        monkeypatch.setattr(batch_generate, "EXO_PREFILL_STEP_TOKENS", 256)
        runs.append(_logprobs_by_prompt(model, tokenizer, prompts, concurrent))

    synchronous, in_batch = runs
    for expected, actual in zip(synchronous, in_batch, strict=True):
        assert [token for token, _ in actual] == [token for token, _ in expected]
        assert (
            max(abs(a - e) for (_, a), (_, e) in zip(actual, expected, strict=True))
            < 1e-3
        )
