"""DeepSeek V4's MLX model on a tiny random configuration whose indexer top-k binds after 32
tokens: one-shot prefill, exo's chunked prefill and one-token decode give the same logits
(which needs the indexer's RoPE along positions and its causal top-k), the split attention
path equals the unsplit one, and the reference's rounding helpers. Real-weight comparisons
with DeepSeek's reference implementation are in the PR's report."""

from collections.abc import Iterator

import mlx.core as mx
import numpy as np
import pytest

from exo.worker.engines.mlx.models.deepseek_v4 import model as dsv4
from exo.worker.engines.mlx.models.deepseek_v4.model import Model, ModelArgs

PROMPT_TOKENS = 160


def _tiny_model() -> Model:
    mx.random.seed(0)
    args = ModelArgs(
        vocab_size=256,
        hidden_size=128,
        num_hidden_layers=4,
        num_attention_heads=4,
        q_lora_rank=64,
        o_lora_rank=32,
        o_groups=2,
        head_dim=128,
        qk_rope_head_dim=64,
        sliding_window=16,
        # window-only, compressed with the indexer (ratio 4), heavily compressed, indexer
        compress_ratios=[0, 4, 128, 4],
        index_n_heads=4,
        index_head_dim=64,
        index_topk=8,
        moe_intermediate_size=64,
        n_routed_experts=4,
        num_experts_per_tok=2,
        num_hash_layers=1,
        num_nextn_predict_layers=0,
        max_position_embeddings=4096,
    )
    model = Model(args)
    mx.eval(model.parameters())
    return model


def _prompt() -> mx.array:
    rng = np.random.default_rng(0)
    return mx.array(rng.integers(0, 256, size=(1, PROMPT_TOKENS)).astype(np.int32))


def _logits(model: Model, tokens: mx.array, step: int) -> np.ndarray:
    """Next-token log-probs at every position, feeding the prompt `step` tokens at a time."""
    cache = model.make_cache()
    out: list[mx.array] = []
    for start in range(0, tokens.shape[1], step):
        logits = model(tokens[:, start : start + step], cache=cache).astype(mx.float32)
        out.append(logits - mx.logsumexp(logits, axis=-1, keepdims=True))
        mx.eval(out[-1])
    return np.array(mx.concatenate(out, axis=1)[0])


def _no_kv_rounding(kv: mx.array, _rope_dims: int) -> mx.array:
    return kv


def _no_indexer_rounding(x: mx.array) -> mx.array:
    return x


@pytest.fixture(scope="module")
def tiny_model() -> Iterator[Model]:
    """The model with the reference's FP8 / FP4 roundings off: they are discontinuous, so
    the 1e-7 differences between the decode and prefill kernels can flip a rounding and the
    comparisons below would measure that noise instead of the cache and indexer logic."""
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(dsv4, "_fp8_kv_sim", _no_kv_rounding)
        patch.setattr(dsv4, "_fp4_indexer_sim", _no_indexer_rounding)
        yield _tiny_model()


@pytest.fixture(scope="module")
def one_shot(tiny_model: Model) -> np.ndarray:
    return _logits(tiny_model, _prompt(), PROMPT_TOKENS)


@pytest.mark.parametrize("step", [1, 48])
def test_chunked_prefill_and_decode_match_one_shot(
    tiny_model: Model, one_shot: np.ndarray, step: int
) -> None:
    """Past 4 * index_topk = 32 tokens the indexer selects rows, so this only holds when its
    query RoPE follows positions (the same in every step) and a multi-token step can't pick
    rows that complete after the query. Without those fixes the log-probs differ by up to
    1.3 nats from position 3 on."""
    stepped = _logits(tiny_model, _prompt(), step)
    np.testing.assert_allclose(stepped, one_shot, atol=1e-4)


def test_split_attention_matches_unsplit(
    tiny_model: Model, one_shot: np.ndarray, monkeypatch: pytest.MonkeyPatch
) -> None:
    # 4 heads x up to 160 queries x ~60 keys: a limit of 2**10 scores forces small slices.
    monkeypatch.setattr(dsv4, "_SDPA_MAX_SCORES", 2**10)
    split = _logits(tiny_model, _prompt(), PROMPT_TOKENS)
    np.testing.assert_allclose(split, one_shot, atol=1e-4)


def test_pow2_ceil_is_exact() -> None:
    x = mx.array([1.0, 1.5, 3.0, 0.75, 2.0**-20, 448.0], dtype=mx.float32)
    expected = [1.0, 2.0, 4.0, 1.0, 2.0**-20, 512.0]
    assert np.array(dsv4._pow2_ceil(x)).tolist() == expected  # pyright: ignore[reportPrivateUsage]


def test_e2m1_rounds_to_nearest_with_ties_to_even() -> None:
    values = [0.25, 0.3, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0, 5.5, -0.75, -6.0]
    expected = [0.0, 0.5, 1.0, 1.0, 2.0, 2.0, 4.0, 4.0, 6.0, -1.0, -6.0]
    rounded = dsv4._round_e2m1(mx.array(values, dtype=mx.float32))  # pyright: ignore[reportPrivateUsage]
    assert np.array(rounded).tolist() == expected


def test_fp8_kv_rounding_keeps_the_rope_dims() -> None:
    mx.random.seed(1)
    kv = mx.random.normal((2, 3, 192))
    rounded = dsv4._fp8_kv_sim(kv, 64)  # pyright: ignore[reportPrivateUsage]
    assert mx.array_equal(rounded[..., -64:], kv[..., -64:])
    rel = mx.abs(rounded[..., :-64] - kv[..., :-64]) / mx.abs(kv[..., :-64])
    # e4m3 keeps 3 mantissa bits: at most 2**-4 relative error away from tiny values
    assert float(mx.max(mx.where(mx.abs(kv[..., :-64]) > 0.1, rel, 0.0))) <= 2**-4
