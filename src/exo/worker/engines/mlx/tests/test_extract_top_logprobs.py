# type: ignore
import math
from unittest.mock import MagicMock

import mlx.core as mx
import pytest

from exo.worker.engines.mlx.generator.generate import extract_top_logprobs


def _mock_tokenizer() -> MagicMock:
    tok = MagicMock()
    tok.decode = lambda ids: f"tok_{ids[0]}"
    return tok


def _make_logprobs(values: list[float]) -> mx.array:
    arr = mx.array(values, dtype=mx.float32)
    mx.eval(arr)
    return arr


class TestExtractTopLogprobsFallback:
    def test_returns_correct_selected_logprob(self) -> None:
        lp = _make_logprobs([-1.0, -2.0, -0.5, -3.0, -4.0])
        selected, _ = extract_top_logprobs(lp, _mock_tokenizer(), top_logprobs=3, selected_token=2)
        assert selected == pytest.approx(-0.5)

    def test_returns_top_k_sorted_descending(self) -> None:
        lp = _make_logprobs([-1.0, -2.0, -0.5, -3.0, -4.0])
        _, items = extract_top_logprobs(lp, _mock_tokenizer(), top_logprobs=3, selected_token=0)
        logprob_values = [item.logprob for item in items]
        assert logprob_values == sorted(logprob_values, reverse=True)
        assert len(items) == 3

    def test_top_tokens_are_most_probable(self) -> None:
        lp = _make_logprobs([-5.0, -1.0, -3.0, -0.1, -2.0])
        _, items = extract_top_logprobs(lp, _mock_tokenizer(), top_logprobs=2, selected_token=0)
        token_ids = [int(item.token.split("_")[1]) for item in items]
        assert 3 in token_ids
        assert 1 in token_ids

    def test_top_logprobs_clamped_to_vocab_size(self) -> None:
        lp = _make_logprobs([-1.0, -2.0, -3.0, -4.0, -5.0])
        _, items = extract_top_logprobs(lp, _mock_tokenizer(), top_logprobs=10, selected_token=0)
        assert len(items) == 4

    def test_nan_logprobs_filtered(self) -> None:
        lp = _make_logprobs([-1.0, float("nan"), -0.5])
        _, items = extract_top_logprobs(lp, _mock_tokenizer(), top_logprobs=3, selected_token=0)
        for item in items:
            assert not math.isnan(item.logprob)

    def test_token_bytes_correct(self) -> None:
        tok = MagicMock()
        tok.decode = lambda ids: "hello"
        lp = _make_logprobs([-1.0, -2.0])
        _, items = extract_top_logprobs(lp, tok, top_logprobs=2, selected_token=0)
        assert items[0].bytes == list("hello".encode("utf-8"))


class TestExtractTopLogprobsPrecomputed:
    def test_uses_precomputed_data(self) -> None:
        lp = _make_logprobs([-99.0])
        selected, items = extract_top_logprobs(
            lp,
            _mock_tokenizer(),
            top_logprobs=2,
            selected_token=0,
            precomputed_indices=[3, 1, 0],
            precomputed_values=[-0.1, -1.0, -5.0],
            precomputed_selected=-0.1,
        )
        assert selected == pytest.approx(-0.1)
        assert len(items) == 2
        assert items[0].token == "tok_3"
        assert items[0].logprob == pytest.approx(-0.1)
        assert items[1].token == "tok_1"
        assert items[1].logprob == pytest.approx(-1.0)

    def test_slices_precomputed_to_requested_k(self) -> None:
        lp = _make_logprobs([-99.0])
        _, items = extract_top_logprobs(
            lp,
            _mock_tokenizer(),
            top_logprobs=1,
            selected_token=0,
            precomputed_indices=[3, 1, 0, 2, 4],
            precomputed_values=[-0.1, -1.0, -2.0, -3.0, -4.0],
            precomputed_selected=-0.1,
        )
        assert len(items) == 1
        assert items[0].token == "tok_3"

    def test_falls_back_when_precomputed_partial(self) -> None:
        lp = _make_logprobs([-1.0, -2.0, -0.5])
        selected, items = extract_top_logprobs(
            lp,
            _mock_tokenizer(),
            top_logprobs=2,
            selected_token=2,
            precomputed_indices=[0, 2],
            precomputed_values=None,
            precomputed_selected=None,
        )
        assert selected == pytest.approx(-0.5)
        assert len(items) == 2

    def test_precomputed_matches_fallback(self) -> None:
        lp = _make_logprobs([-1.0, -0.3, -2.5, -0.1, -4.0, -0.8, -3.0, -1.5])
        tok = _mock_tokenizer()

        selected_fb, items_fb = extract_top_logprobs(lp, tok, top_logprobs=5, selected_token=1)

        pre_indices = [item.token.split("_")[1] for item in items_fb]
        pre_indices_int = [int(x) for x in pre_indices]
        pre_values = [item.logprob for item in items_fb]

        selected_pc, items_pc = extract_top_logprobs(
            lp,
            tok,
            top_logprobs=5,
            selected_token=1,
            precomputed_indices=pre_indices_int,
            precomputed_values=pre_values,
            precomputed_selected=selected_fb,
        )

        assert selected_pc == pytest.approx(selected_fb)
        assert len(items_pc) == len(items_fb)
        for a, b in zip(items_pc, items_fb):
            assert a.token == b.token
            assert a.logprob == pytest.approx(b.logprob)
