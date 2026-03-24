# type: ignore
import math
from unittest.mock import MagicMock

import mlx.core as mx
import mlx.nn as nn
import pytest
from mlx_lm.generate import BatchGenerator

from exo.worker.engines.mlx.generator.generate import extract_top_logprobs
from exo.worker.engines.mlx.patches.opt_batch_gen import (
    _PRECOMPUTE_TOP_K,
    apply_batch_gen_patch,
)


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

        selected_fb, items_fb = extract_top_logprobs(
            lp, tok, top_logprobs=5, selected_token=1
        )

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
        for a, b in zip(items_pc, items_fb, strict=True):
            assert a.token == b.token
            assert a.logprob == pytest.approx(b.logprob)


def _tiny_model() -> nn.Module:
    from mlx_lm.models.llama import Model, ModelArgs

    mx.random.seed(42)
    args = ModelArgs(
        model_type="llama",
        hidden_size=64,
        num_hidden_layers=2,
        intermediate_size=128,
        num_attention_heads=2,
        num_key_value_heads=1,
        rms_norm_eps=1e-6,
        vocab_size=256,
        rope_theta=10000.0,
        tie_word_embeddings=True,
    )
    model = Model(args)
    mx.eval(model.parameters())
    return model


@pytest.mark.slow
class TestBatchedTopKPrecompute:
    @pytest.fixture(autouse=True)
    def _reset_globals(self) -> None:
        import exo.worker.engines.mlx.patches.opt_batch_gen as _mod

        _mod._pending_topk_idx = None
        _mod._pending_topk_val = None
        _mod._pending_selected_lps = None

    def _run_generator(
        self, model: nn.Module, prompts: list[list[int]], steps: int, needs_topk: bool
    ) -> list[list[BatchGenerator.Response]]:
        apply_batch_gen_patch()
        gen = BatchGenerator(model=model, stop_tokens=set(), prefill_step_size=512)
        gen._needs_topk = needs_topk
        gen.insert(prompts)
        all_responses: list[list[BatchGenerator.Response]] = []
        for _ in range(steps + len(prompts)):
            responses = gen.next()
            if responses:
                all_responses.append(responses)
            if gen.active_batch is None and not gen.unprocessed_prompts:
                break
        gen.close()
        return all_responses

    def test_precomputed_topk_attached_to_responses(self) -> None:
        model = _tiny_model()
        steps = self._run_generator(model, [[1, 2, 3]], 5, needs_topk=True)
        found_precomputed = False
        for step_responses in steps:
            for resp in step_responses:
                if hasattr(resp, "_topk_indices"):
                    found_precomputed = True
                    assert hasattr(resp, "_topk_values"), (
                        "Response missing _topk_values"
                    )
                    assert hasattr(resp, "_selected_logprob"), (
                        "Response missing _selected_logprob"
                    )
                    assert len(resp._topk_indices) == _PRECOMPUTE_TOP_K
                    assert len(resp._topk_values) == _PRECOMPUTE_TOP_K
        assert found_precomputed, "No responses had precomputed topk"

    def test_no_topk_when_not_needed(self) -> None:
        model = _tiny_model()
        steps = self._run_generator(model, [[1, 2, 3]], 5, needs_topk=False)
        for step_responses in steps:
            for resp in step_responses:
                assert not hasattr(resp, "_topk_indices")

    def test_precomputed_matches_fallback_in_batch(self) -> None:
        model = _tiny_model()
        tok = _mock_tokenizer()
        steps = self._run_generator(model, [[1, 2, 3]], 10, needs_topk=True)
        for step_responses in steps[1:]:
            for resp in step_responses:
                if not hasattr(resp, "_topk_indices"):
                    continue
                selected_fb, items_fb = extract_top_logprobs(
                    resp.logprobs, tok, top_logprobs=5, selected_token=resp.token
                )
                selected_pc, items_pc = extract_top_logprobs(
                    resp.logprobs,
                    tok,
                    top_logprobs=5,
                    selected_token=resp.token,
                    precomputed_indices=resp._topk_indices,
                    precomputed_values=resp._topk_values,
                    precomputed_selected=resp._selected_logprob,
                )
                assert selected_pc == pytest.approx(selected_fb, abs=1e-5)
                for a, b in zip(items_pc, items_fb, strict=True):
                    assert a.token == b.token
                    assert a.logprob == pytest.approx(b.logprob, abs=1e-5)

    def test_topk_correct_after_batch_shrink(self) -> None:
        model = _tiny_model()
        tok = _mock_tokenizer()
        apply_batch_gen_patch()
        gen = BatchGenerator(
            model=model, stop_tokens={0}, prefill_step_size=512, max_tokens=3
        )
        gen._needs_topk = True
        gen.insert([[1, 2, 3], [4, 5, 6]], max_tokens=[3, 20])

        seen_shrink = False
        for _ in range(30):
            responses = gen.next()
            for resp in responses:
                if resp.finish_reason is not None:
                    seen_shrink = True
                    continue
                if not hasattr(resp, "_topk_indices"):
                    continue
                selected_fb, items_fb = extract_top_logprobs(
                    resp.logprobs, tok, top_logprobs=5, selected_token=resp.token
                )
                selected_pc, _ = extract_top_logprobs(
                    resp.logprobs,
                    tok,
                    top_logprobs=5,
                    selected_token=resp.token,
                    precomputed_indices=resp._topk_indices,
                    precomputed_values=resp._topk_values,
                    precomputed_selected=resp._selected_logprob,
                )
                assert selected_pc == pytest.approx(selected_fb, abs=1e-5), (
                    f"Mismatch after batch shrink: precomputed={selected_pc}, fallback={selected_fb}"
                )
            if gen.active_batch is None and not gen.unprocessed_prompts:
                break

        gen.close()
        assert seen_shrink, "Expected at least one request to finish (batch shrink)"
