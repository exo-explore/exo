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
        selected, _ = extract_top_logprobs(
            lp, _mock_tokenizer(), top_logprobs=3, selected_token=2
        )
        assert selected == pytest.approx(-0.5)

    def test_returns_top_k_sorted_descending(self) -> None:
        lp = _make_logprobs([-1.0, -2.0, -0.5, -3.0, -4.0])
        _, items = extract_top_logprobs(
            lp, _mock_tokenizer(), top_logprobs=3, selected_token=0
        )
        logprob_values = [item.logprob for item in items]
        assert logprob_values == sorted(logprob_values, reverse=True)
        assert len(items) == 3

    def test_top_tokens_are_most_probable(self) -> None:
        lp = _make_logprobs([-5.0, -1.0, -3.0, -0.1, -2.0])
        _, items = extract_top_logprobs(
            lp, _mock_tokenizer(), top_logprobs=2, selected_token=0
        )
        token_ids = [int(item.token.split("_")[1]) for item in items]
        assert 3 in token_ids
        assert 1 in token_ids

    def test_top_logprobs_clamped_to_vocab_size(self) -> None:
        lp = _make_logprobs([-1.0, -2.0, -3.0, -4.0, -5.0])
        _, items = extract_top_logprobs(
            lp, _mock_tokenizer(), top_logprobs=10, selected_token=0
        )
        assert len(items) == 4

    def test_nan_logprobs_filtered(self) -> None:
        lp = _make_logprobs([-1.0, float("nan"), -0.5])
        _, items = extract_top_logprobs(
            lp, _mock_tokenizer(), top_logprobs=3, selected_token=0
        )
        for item in items:
            assert not math.isnan(item.logprob)

    def test_token_bytes_correct(self) -> None:
        tok = MagicMock()
        tok.decode = lambda ids: "hello"
        lp = _make_logprobs([-1.0, -2.0])
        _, items = extract_top_logprobs(lp, tok, top_logprobs=2, selected_token=0)
        assert items[0].bytes == list("hello".encode("utf-8"))
