# pyright: reportAny=false, reportUnknownVariableType=false
# pyright: reportUnknownMemberType=false, reportUnknownArgumentType=false
# pyright: reportUnknownLambdaType=false, reportPrivateUsage=false
# pyright: reportInvalidCast=false
"""Speculative decoding with a draft model (speculative.py) through mlx_generate.

Tiny random-weight Llama models and an in-memory word-level tokenizer: no downloads.
The target decides every emitted token, so identical greedy output alone would also pass
with a draft that is ignored or misaligned. The tests therefore also check acceptance:
a draft with the target's own weights must have every draft token accepted (only true
when the draft's cache is at the target's position), and a perturbed draft must have some
but not all accepted."""

import functools
import json
import math
import sys
from collections.abc import Iterator
from pathlib import Path
from typing import Any, cast

import mlx.core as mx
import pytest
from mlx.utils import tree_flatten, tree_map
from mlx_lm.generate import stream_generate
from mlx_lm.models.cache import KVCache, RotatingKVCache
from mlx_lm.models.llama import Model as LlamaModel
from mlx_lm.models.llama import ModelArgs
from mlx_lm.tokenizer_utils import TokenizerWrapper
from pydantic import ValidationError
from tokenizers import Tokenizer, decoders, models, pre_tokenizers
from transformers import PreTrainedTokenizerFast

from exo.api.types import BenchChatCompletionRequest, GenerationStats
from exo.download import download_utils
from exo.shared.types.common import ModelId
from exo.shared.types.text_generation import (
    InputMessage,
    InputMessageContent,
    TextGenerationTaskParams,
)
from exo.worker.engines.mlx import builder, speculative
from exo.worker.engines.mlx.cache import KVPrefixCache
from exo.worker.engines.mlx.generator import generate
from exo.worker.engines.mlx.generator.generate import mlx_generate
from exo.worker.engines.mlx.speculative import (
    DraftConfig,
    DraftConfigError,
    DraftModel,
    SpeculativeStats,
    draft_incompatibility,
    load_draft,
    resolve_draft_config,
    resolve_draft_path,
    speculation_blocker,
)
from exo.worker.engines.mlx.types import Model
from exo.worker.runner.bootstrap import logger

VOCAB = 64
EOS = "<eos>"


@pytest.fixture
def cpu_device(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Runs the engine on the CPU device, so the greedy comparisons do not depend on a
    GPU's multi-token kernels (on Metal, a k+1 token verify can settle a bf16 near-tie
    differently from a one-token decode). mlx-lm made generation_stream on the default
    device when it was imported, and stream_generate takes it as a default argument."""
    cpu = mx.default_stream(mx.cpu)
    # The package's mlx_lm.generate attribute is the function, not the module
    mlx_lm_generate = sys.modules["mlx_lm.generate"]
    for module in (mlx_lm_generate, generate, speculative):
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


@pytest.fixture
def log_messages() -> Iterator[list[str]]:
    messages: list[str] = []
    handler = logger.add(lambda m: messages.append(str(m)), format="{message}")
    try:
        yield messages
    finally:
        logger.remove(handler)


def _word_tokenizer(vocab: int = VOCAB, swap: bool = False) -> TokenizerWrapper:
    """Words w0..w{vocab-1} with ids 0..vocab-1, and EOS at id `vocab`, which the models
    (logits width `vocab`) never produce. `swap` exchanges two ids: a different tokenizer."""
    ids = {f"w{i}": i for i in range(vocab)}
    if swap:
        ids["w2"], ids["w3"] = 3, 2
    ids[EOS] = vocab
    tokenizer = Tokenizer(models.WordLevel(vocab=ids, unk_token="w0"))
    tokenizer.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
    tokenizer.decoder = decoders.WordPiece(prefix="##")
    return TokenizerWrapper(
        PreTrainedTokenizerFast(tokenizer_object=tokenizer, eos_token=EOS)
    )


def _llama_args(vocab: int, hidden: int, layers: int) -> ModelArgs:
    return ModelArgs(
        model_type="llama",
        hidden_size=hidden,
        num_hidden_layers=layers,
        intermediate_size=2 * hidden,
        num_attention_heads=4,
        num_key_value_heads=2,
        rms_norm_eps=1e-6,
        vocab_size=vocab,
        rope_theta=10000.0,
        tie_word_embeddings=True,
    )


def _llama(
    seed: int, vocab: int = VOCAB, hidden: int = 64, layers: int = 2, scale: float = 1.0
) -> Model:
    mx.random.seed(seed)
    model = LlamaModel(_llama_args(vocab, hidden, layers))
    model.update(
        tree_map(
            lambda p: scale * mx.random.normal(shape=p.shape, dtype=mx.float32),
            model.parameters(),
        )
    )
    mx.eval(model.parameters())
    return cast(Model, model)


def _perturbed(target: Model, seed: int, noise: float) -> Model:
    """The target's architecture and weights plus Gaussian noise: a draft that agrees
    with the target often but not always."""
    args = cast(ModelArgs, target.args)
    mx.random.seed(seed)
    model = LlamaModel(args)
    params = cast(dict[str, Any], target.parameters())
    model.update(
        tree_map(
            lambda p: p + noise * mx.random.normal(shape=p.shape, dtype=p.dtype),
            params,
        )
    )
    mx.eval(model.parameters())
    return cast(Model, model)


def _draft(model: Model, k: int = 3) -> DraftModel:
    return DraftModel(model=model, name="test/draft", num_draft_tokens=k)


def _task(**kwargs: Any) -> TextGenerationTaskParams:
    params: dict[str, Any] = {
        "model": ModelId("test/target"),
        "input": [InputMessage(role="user", content=InputMessageContent("hello"))],
        "max_output_tokens": 24,
        "temperature": 0.0,
    }
    params.update(kwargs)
    return TextGenerationTaskParams(**params)


def _prompt(n: int, offset: int = 0) -> str:
    return " ".join(f"w{(7 * i + 3 + offset) % VOCAB}" for i in range(n))


def _run(
    model: Model,
    tokenizer: TokenizerWrapper,
    prompt: str,
    task: TextGenerationTaskParams,
    draft: DraftModel | None = None,
    kv_prefix_cache: KVPrefixCache | None = None,
) -> tuple[list[int], GenerationStats]:
    tokens: list[int] = []
    stats: GenerationStats | None = None
    for response in mlx_generate(
        model=model,
        tokenizer=tokenizer,
        task=task,
        prompt=prompt,
        kv_prefix_cache=kv_prefix_cache,
        group=None,
        draft=draft,
    ):
        tokens.append(response.token)
        if response.stats is not None:
            stats = response.stats
    assert stats is not None
    return tokens, stats


def _acceptance(stats: GenerationStats) -> float:
    assert stats.draft_tokens_proposed is not None
    assert stats.draft_tokens_accepted is not None
    assert stats.draft_tokens_proposed > 0
    return stats.draft_tokens_accepted / stats.draft_tokens_proposed


# Configuration


def test_draft_config_is_off_by_default() -> None:
    assert resolve_draft_config(ModelId("a/target"), {}) is None
    assert resolve_draft_config(ModelId("a/target"), {"EXO_DRAFT_MODEL": " "}) is None


def test_draft_config_one_draft_for_every_instance() -> None:
    config = resolve_draft_config(
        ModelId("a/target"), {"EXO_DRAFT_MODEL": "a/draft-1b"}
    )
    assert config == DraftConfig(source="a/draft-1b", num_draft_tokens=3)


def test_draft_config_per_target_pairs() -> None:
    env = {
        "EXO_DRAFT_MODEL": "a/llama-70b=a/llama-1b, a/qwen-32b=/models/qwen-0.5b",
        "EXO_NUM_DRAFT_TOKENS": "2",
    }
    assert resolve_draft_config(ModelId("a/llama-70b"), env) == DraftConfig(
        source="a/llama-1b", num_draft_tokens=2
    )
    assert resolve_draft_config(ModelId("a/qwen-32b"), env) == DraftConfig(
        source="/models/qwen-0.5b", num_draft_tokens=2
    )
    assert resolve_draft_config(ModelId("a/other"), env) is None


@pytest.mark.parametrize(
    "env",
    [
        {"EXO_DRAFT_MODEL": "a/llama-70b=a/llama-1b,a/qwen"},
        {"EXO_DRAFT_MODEL": "a/llama-70b="},
        {"EXO_DRAFT_MODEL": "=a/llama-1b"},
        {"EXO_DRAFT_MODEL": "a/x=a/y,a/x=a/z"},
        {"EXO_DRAFT_MODEL": "a/draft", "EXO_NUM_DRAFT_TOKENS": "three"},
        {"EXO_DRAFT_MODEL": "a/draft", "EXO_NUM_DRAFT_TOKENS": "0"},
        {"EXO_DRAFT_MODEL": "a/draft", "EXO_NUM_DRAFT_TOKENS": "17"},
        # A malformed setting is reported even where no draft would apply
        {"EXO_NUM_DRAFT_TOKENS": "-1"},
    ],
)
def test_draft_config_rejects_malformed_settings(env: dict[str, str]) -> None:
    with pytest.raises(DraftConfigError):
        resolve_draft_config(ModelId("a/other-target"), env)


def test_bench_request_num_draft_tokens_is_bounded() -> None:
    def request(n: int) -> BenchChatCompletionRequest:
        return BenchChatCompletionRequest.model_validate(
            {"model": "a/b", "messages": [], "num_draft_tokens": n}
        )

    assert request(0).num_draft_tokens == 0
    assert request(16).num_draft_tokens == 16
    with pytest.raises(ValidationError):
        request(17)
    with pytest.raises(ValidationError):
        request(-1)


def test_stats_count_rounds_and_the_last_round_cut_by_max_tokens() -> None:
    stats = SpeculativeStats(num_draft_tokens=3, max_tokens=8)
    # Round 1: 3 accepted + bonus; round 2: 1 accepted + correction; round 3: k=min(3, 2)
    for from_draft in [True, True, True, False, True, False, True, True]:
        stats.observe(from_draft)
    assert (stats.rounds, stats.proposed, stats.accepted) == (3, 8, 6)
    assert stats.emitted == 8
    assert math.isclose(stats.acceptance, 0.75)


# Compatibility and loading


def test_vocabularies_must_match() -> None:
    target, draft = _llama(1), _llama(2, hidden=32, layers=1)
    assert (
        draft_incompatibility(target, _word_tokenizer(), draft, _word_tokenizer())
        is None
    )
    reason = draft_incompatibility(
        target, _word_tokenizer(), draft, _word_tokenizer(swap=True)
    )
    assert reason is not None and "vocabularies differ" in reason


def test_draft_logits_may_not_be_wider_than_the_target() -> None:
    target = _llama(1, vocab=VOCAB)
    narrower, wider = _llama(2, vocab=VOCAB - 8), _llama(3, vocab=VOCAB + 8)
    tokenizer = _word_tokenizer()
    assert draft_incompatibility(target, tokenizer, narrower, tokenizer) is None
    reason = draft_incompatibility(target, tokenizer, wider, tokenizer)
    assert reason is not None and "wider" in reason


def _save(model: Model, tokenizer: TokenizerWrapper, path: Path) -> Path:
    path.mkdir(parents=True)
    args = cast(ModelArgs, model.args)
    config = {
        key: value
        for key, value in vars(args).items()
        if isinstance(value, (int, float, str, bool))
    }
    (path / "config.json").write_text(json.dumps(config))
    weights = dict(cast(list[tuple[str, mx.array]], tree_flatten(model.parameters())))
    mx.save_safetensors(str(path / "model.safetensors"), weights)
    tokenizer._tokenizer.save_pretrained(str(path))
    return path


@pytest.mark.usefixtures("cpu_device")
def test_draft_loads_from_a_directory_and_drafts(tmp_path: Path) -> None:
    tokenizer = _word_tokenizer()
    target = _llama(1)
    path = _save(_perturbed(target, 5, 0.05), tokenizer, tmp_path / "draft")
    draft = load_draft(DraftConfig(str(path), 2), target, tokenizer)
    assert draft is not None and draft.num_draft_tokens == 2

    plain, _ = _run(target, tokenizer, _prompt(12), _task())
    tokens, stats = _run(target, tokenizer, _prompt(12), _task(), draft=draft)
    assert tokens == plain
    assert _acceptance(stats) > 0


@pytest.mark.usefixtures("cpu_device")
def test_draft_loads_by_model_id_from_the_exo_model_dirs(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    tokenizer = _word_tokenizer()
    target = _llama(1)
    models_dir = tmp_path / "models"
    path = _save(_perturbed(target, 5, 0.05), tokenizer, models_dir / "test--draft")
    monkeypatch.setattr(download_utils, "EXO_MODELS_DIRS", (models_dir,))
    monkeypatch.setattr(download_utils, "EXO_MODELS_READ_ONLY_DIRS", ())
    # EXO counts a model directory as complete only with a safetensors index
    assert resolve_draft_path("test/draft") is None
    weight_map = {
        name: "model.safetensors" for name, _ in tree_flatten(target.parameters())
    }
    (path / "model.safetensors.index.json").write_text(
        json.dumps({"metadata": {}, "weight_map": weight_map})
    )
    assert resolve_draft_path("test/draft") == path
    draft = load_draft(DraftConfig("test/draft", 3), target, tokenizer)
    assert draft is not None
    _, stats = _run(target, tokenizer, _prompt(12), _task(), draft=draft)
    assert _acceptance(stats) > 0


def test_missing_draft_is_not_loaded(tmp_path: Path, log_messages: list[str]) -> None:
    tokenizer = _word_tokenizer()
    assert resolve_draft_path(str(tmp_path / "absent")) is None
    assert resolve_draft_path("nobody/no-such-draft-model") is None
    assert (
        load_draft(DraftConfig(str(tmp_path / "absent"), 3), _llama(1), tokenizer)
        is None
    )
    assert any("is not on this node" in m for m in log_messages)


def test_incompatible_draft_is_not_loaded(
    tmp_path: Path, log_messages: list[str]
) -> None:
    path = _save(_llama(2), _word_tokenizer(swap=True), tmp_path / "draft")
    assert load_draft(DraftConfig(str(path), 3), _llama(1), _word_tokenizer()) is None
    assert any("vocabularies differ" in m for m in log_messages)


def test_unloadable_draft_is_not_loaded(
    tmp_path: Path, log_messages: list[str]
) -> None:
    path = tmp_path / "draft"
    path.mkdir()
    (path / "config.json").write_text(json.dumps({"model_type": "llama"}))
    assert load_draft(DraftConfig(str(path), 3), _llama(1), _word_tokenizer()) is None
    assert any("failed to load" in m for m in log_messages)


@pytest.mark.parametrize(
    ("no_batch", "group", "loaded"),
    [(True, None, True), (False, None, False), (True, "group", False)],
)
def test_builder_loads_the_draft_only_for_a_one_request_single_node_engine(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    log_messages: list[str],
    no_batch: bool,
    group: str | None,
    loaded: bool,
) -> None:
    tokenizer = _word_tokenizer()
    path = _save(_llama(2, hidden=32, layers=1), tokenizer, tmp_path / "draft")
    monkeypatch.setattr(builder, "use_generation_stream_by_default", lambda: None)
    monkeypatch.setenv("EXO_DRAFT_MODEL", str(path))
    if no_batch:
        monkeypatch.setenv("EXO_NO_BATCH", "1")
    else:
        monkeypatch.delenv("EXO_NO_BATCH", raising=False)
    mlx_builder = builder.MlxBuilder(
        model_id=ModelId("test/target"),
        event_sender=cast(Any, None),
        cancel_receiver=cast(Any, None),
        inference_model=_llama(1),
        tokenizer=tokenizer,
        group=cast(mx.distributed.Group, object()) if group else None,
    )
    draft = mlx_builder._load_draft()
    assert (draft is not None) == loaded
    if not no_batch:
        assert any("this instance batches requests" in m for m in log_messages)
    if group:
        assert any("single-node instances only" in m for m in log_messages)


def test_builder_ignores_a_malformed_setting(
    monkeypatch: pytest.MonkeyPatch, log_messages: list[str]
) -> None:
    monkeypatch.setattr(builder, "use_generation_stream_by_default", lambda: None)
    monkeypatch.setenv("EXO_DRAFT_MODEL", "a=b,c")
    mlx_builder = builder.MlxBuilder(
        model_id=ModelId("test/target"),
        event_sender=cast(Any, None),
        cancel_receiver=cast(Any, None),
        inference_model=_llama(1),
        tokenizer=_word_tokenizer(),
    )
    assert mlx_builder._load_draft() is None
    assert any("is not 'target=draft'" in m for m in log_messages)


# Greedy decoding: identical output, and acceptance that proves the draft ran aligned


# The prefix cache restores the target to different positions; the draft has no prefix
# cache and must still start where the target does. (earlier prompt, expected hit log)
_PREFIX_CASES = {
    "none": (None, None),
    # Shares 12 tokens, stored as 10: the target prefills the rest, decode starts from 2
    "partial": (12, "KV cache hit: 10/20"),
    # The same prompt again: an exact hit trimmed to the stored 18 tokens
    "repeat": (20, "KV cache hit: 18/20"),
    # A longer earlier prompt: an exact hit at 19 tokens, nothing left to prefill and
    # decode starts from 1 token
    "exact": (24, "KV cache hit: 19/20"),
}


@pytest.mark.usefixtures("cpu_device")
@pytest.mark.parametrize("prefix", list(_PREFIX_CASES))
def test_draft_with_the_targets_weights_accepts_everything(
    prefix: str, log_messages: list[str]
) -> None:
    tokenizer = _word_tokenizer()
    target = _llama(1)
    draft = _draft(_perturbed(target, 9, 0.0), k=3)
    prompt = _prompt(20)
    earlier, hit_log = _PREFIX_CASES[prefix]

    def run(d: DraftModel | None) -> tuple[list[int], GenerationStats]:
        kv_prefix_cache = None
        if earlier is not None:
            kv_prefix_cache = KVPrefixCache(None)
            _run(
                target,
                tokenizer,
                _prompt(earlier),
                _task(max_output_tokens=4),
                None,
                kv_prefix_cache,
            )
        log_messages.clear()
        return _run(target, tokenizer, prompt, _task(), d, kv_prefix_cache)

    plain, plain_stats = run(None)
    tokens, stats = run(draft)
    assert tokens == plain
    assert plain_stats.draft_tokens_proposed is None
    assert _acceptance(stats) == 1.0
    if hit_log is not None:
        assert any(hit_log in m for m in log_messages)


@pytest.mark.usefixtures("cpu_device")
@pytest.mark.parametrize("k", [1, 2, 3, 5])
def test_perturbed_draft_gives_identical_greedy_output(k: int) -> None:
    tokenizer = _word_tokenizer()
    target = _llama(1)
    draft = _draft(_perturbed(target, 5, 0.05), k=k)
    for offset in range(3):
        prompt = _prompt(10 + 3 * offset, offset)
        plain, _ = _run(target, tokenizer, prompt, _task(max_output_tokens=40))
        tokens, stats = _run(
            target, tokenizer, prompt, _task(max_output_tokens=40), draft
        )
        assert tokens == plain
        assert len(tokens) == 40
        assert 0.0 < _acceptance(stats) < 1.0


@pytest.mark.usefixtures("cpu_device")
def test_smaller_draft_gives_identical_greedy_output() -> None:
    tokenizer = _word_tokenizer()
    target = _llama(1)
    draft = _draft(_llama(4, hidden=32, layers=1), k=4)
    prompt = _prompt(16)
    plain, _ = _run(target, tokenizer, prompt, _task(max_output_tokens=32))
    tokens, stats = _run(target, tokenizer, prompt, _task(max_output_tokens=32), draft)
    assert tokens == plain
    assert stats.draft_tokens_proposed is not None and stats.draft_tokens_proposed > 0


@pytest.mark.usefixtures("cpu_device")
@pytest.mark.parametrize(
    "task_kwargs",
    [
        {"repetition_penalty": 1.5, "repetition_context_size": 8},
        {"presence_penalty": 0.8, "frequency_penalty": 0.4},
        # Bench requests ban EOS with a logits processor
        {"bench": True},
    ],
)
def test_identical_greedy_output_with_logits_processors(
    task_kwargs: dict[str, Any],
) -> None:
    tokenizer = _word_tokenizer()
    target = _llama(1)
    draft = _draft(_perturbed(target, 5, 0.05), k=3)
    prompt = _prompt(14)
    plain, _ = _run(target, tokenizer, prompt, _task(**task_kwargs))
    tokens, stats = _run(target, tokenizer, prompt, _task(**task_kwargs), draft)
    assert tokens == plain
    assert 0.0 < _acceptance(stats) < 1.0


@pytest.mark.usefixtures("cpu_device")
def test_stop_sequence_and_short_max_tokens() -> None:
    tokenizer = _word_tokenizer()
    target = _llama(1)
    draft = _draft(_perturbed(target, 9, 0.0), k=5)
    prompt = _prompt(10)
    plain, _ = _run(target, tokenizer, prompt, _task(max_output_tokens=24))
    stop = tokenizer.decode(plain[7:9]).strip()
    for task in (_task(max_output_tokens=2), _task(max_output_tokens=24, stop=stop)):
        expected, _ = _run(target, tokenizer, prompt, task)
        tokens, stats = _run(target, tokenizer, prompt, task, draft)
        assert tokens == expected
        assert len(expected) < len(plain)
        assert stats.draft_tokens_proposed is not None


# Fallbacks: decode as without a draft, and say why


@pytest.mark.usefixtures("cpu_device")
def test_request_can_turn_the_draft_off(log_messages: list[str]) -> None:
    tokenizer = _word_tokenizer()
    target = _llama(1)
    draft = _draft(_perturbed(target, 9, 0.0))
    plain, _ = _run(target, tokenizer, _prompt(10), _task())
    tokens, stats = _run(
        target, tokenizer, _prompt(10), _task(num_draft_tokens=0), draft
    )
    assert tokens == plain
    assert stats.draft_tokens_proposed is None
    assert any("num_draft_tokens to 0" in m for m in log_messages)


@pytest.mark.usefixtures("cpu_device")
def test_request_can_choose_num_draft_tokens() -> None:
    tokenizer = _word_tokenizer()
    target = _llama(1)
    draft = _draft(_perturbed(target, 9, 0.0), k=3)
    _, stats = _run(
        target,
        tokenizer,
        _prompt(10),
        _task(num_draft_tokens=5, max_output_tokens=24),
        draft,
    )
    # All accepted: rounds of 5 drafts + 1 bonus token; the last round is cut at 24
    assert (stats.draft_tokens_proposed, stats.draft_tokens_accepted) == (20, 20)


def _with_rotating_cache(model: Model) -> Model:
    layers = len(model.layers)
    model.make_cache = lambda: [RotatingKVCache(max_size=512) for _ in range(layers)]
    return model


@pytest.mark.usefixtures("cpu_device")
@pytest.mark.parametrize("which", ["target", "draft"])
def test_cache_that_cannot_rewind_falls_back(
    which: str, log_messages: list[str]
) -> None:
    tokenizer = _word_tokenizer()
    target = _llama(1)
    draft_model = _perturbed(target, 9, 0.0)
    if which == "target":
        target = _with_rotating_cache(target)
    else:
        draft_model = _with_rotating_cache(draft_model)
    plain, _ = _run(target, tokenizer, _prompt(10), _task())
    tokens, stats = _run(target, tokenizer, _prompt(10), _task(), _draft(draft_model))
    assert tokens == plain
    assert stats.draft_tokens_proposed is None
    assert any(
        f"the {which} cache (RotatingKVCache) cannot rewind" in m for m in log_messages
    )


def test_blocker_reasons() -> None:
    target = _llama(1)
    draft = _draft(_llama(2, hidden=32, layers=1))
    caches: list[KVCache] = [KVCache() for _ in target.layers]
    draft_caches: list[KVCache] = [KVCache() for _ in draft.model.layers]

    def blocker(**kwargs: Any) -> str | None:
        params: dict[str, Any] = {
            "model": target,
            "caches": caches,
            "draft": draft,
            "draft_caches": draft_caches,
            "group": None,
            "has_vision": False,
            "num_draft_tokens": 3,
        }
        params.update(kwargs)
        return speculation_blocker(**params)

    assert blocker() is None
    assert blocker(group=object()) == "the instance is distributed"
    assert blocker(has_vision=True) == "the request has images"
    assert blocker(num_draft_tokens=0) == "the request set num_draft_tokens to 0"
    assert "2 layers" in cast(str, blocker(caches=caches[:1]))


# Sampling: the emitted tokens follow the target's distribution


def _softmax(logits: mx.array, temperature: float) -> list[float]:
    return cast(list[float], mx.softmax(logits / temperature, axis=-1).tolist())


def _exact_marginals(
    model: Model, prompt_ids: list[int], temperature: float, vocab: int
) -> tuple[list[float], list[float], list[float]]:
    """The target's exact distributions of the first three emitted tokens."""

    def next_probs(ids: list[int]) -> list[float]:
        logits: mx.array = model(mx.array(ids)[None], cache=None)[0, -1]
        logits = logits.astype(mx.float32)
        return _softmax(logits, temperature)

    p0 = next_probs(prompt_ids)
    p1 = [0.0] * vocab
    p2 = [0.0] * vocab
    for t0 in range(vocab):
        p1_given = next_probs([*prompt_ids, t0])
        for t1 in range(vocab):
            p1[t1] += p0[t0] * p1_given[t1]
            p2_given = next_probs([*prompt_ids, t0, t1])
            weight = p0[t0] * p1_given[t1]
            for t2 in range(vocab):
                p2[t2] += weight * p2_given[t2]
    return p0, p1, p2


def _total_variation(samples: list[int], probs: list[float]) -> float:
    counts = [0] * len(probs)
    for s in samples:
        counts[s] += 1
    return 0.5 * sum(
        abs(c / len(samples) - p) for c, p in zip(counts, probs, strict=True)
    )


@pytest.mark.slow
@pytest.mark.usefixtures("cpu_device")
def test_sampled_tokens_follow_the_targets_distribution() -> None:
    vocab, temperature, runs = 8, 4.0, 600
    tokenizer = _word_tokenizer(vocab)
    target = _llama(1, vocab=vocab, hidden=32, layers=1)
    draft = _draft(_perturbed(target, 5, 0.1), k=3)
    prompt = "w1 w5 w2 w7 w3"
    exact = _exact_marginals(target, [1, 5, 2, 7, 3], temperature, vocab)
    # A distribution worth testing: no token takes most of the mass
    assert all(max(p) < 0.7 for p in exact)

    def sample(d: DraftModel | None) -> tuple[list[list[int]], float]:
        positions: list[list[int]] = [[], [], []]
        accepted = proposed = 0
        for seed in range(1, runs + 1):
            task = _task(
                max_output_tokens=3,
                temperature=temperature,
                top_p=1.0,
                min_p=0.0,
                top_k=0,
                seed=seed,
            )
            tokens, stats = _run(target, tokenizer, prompt, task, d)
            for i in range(3):
                positions[i].append(tokens[i])
            accepted += stats.draft_tokens_accepted or 0
            proposed += stats.draft_tokens_proposed or 0
        return positions, accepted / proposed if proposed else 0.0

    plain, _ = sample(None)
    speculative_samples, acceptance = sample(draft)
    # Independent draft and target samples agree about as often as sum(p * q): both the
    # accepted and the corrected tokens are exercised
    assert 0.05 < acceptance < 0.9
    for position in range(3):
        # Sampling noise at 600 runs over 8 tokens is about 0.05 in total variation
        plain_tv = _total_variation(plain[position], exact[position])
        speculative_tv = _total_variation(
            speculative_samples[position], exact[position]
        )
        print(
            f"position {position}: TV plain {plain_tv:.3f}, speculative {speculative_tv:.3f}"
        )
        assert plain_tv < 0.1
        assert speculative_tv < 0.1
