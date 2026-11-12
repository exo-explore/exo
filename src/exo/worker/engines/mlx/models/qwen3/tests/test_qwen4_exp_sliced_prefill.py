# pyright: reportUnusedFunction=false, reportPrivateUsage=false
"""Prefill attends in slices of query rows (qwen4_exp.attention_rows, EXO_ATTN_SLICE_BYTES):
the same output as attending with every row at once, with and without a cache, before and
after the indexer starts selecting keys. On the CPU, where only rounding separates the two."""

from collections.abc import Generator
from pathlib import Path

import mlx.core as mx
import numpy as np
import pytest

from exo.worker.engines.mlx.models.qwen3 import qwen4_exp
from exo.worker.engines.mlx.models.qwen3.tests.test_qwen4_exp import (
    EOS,
    _attached_tiny_model,
)


@pytest.fixture(autouse=True)
def _on_the_cpu() -> Generator[None]:
    device = mx.default_device()
    mx.set_default_device(mx.cpu)
    try:
        yield
    finally:
        mx.set_default_device(device)


def test_slices_are_sized_to_the_budget() -> None:
    budget = qwen4_exp._ATTN_SLICE_BYTES
    per_pair = qwen4_exp._BYTES_PER_QUERY_KEY
    assert qwen4_exp.attention_rows(4096, budget // (per_pair * 1000)) == 1000
    assert qwen4_exp.attention_rows(100, 1000) == 100  # a short prompt is one slice
    assert qwen4_exp.attention_rows(4096, 10**9) == 16  # at least 16 rows


@pytest.mark.parametrize("chunks", [[70], [9, 30, 1, 30], [41, 29]])
def test_sliced_prefill_matches_one_slice(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, chunks: list[int]
) -> None:
    model = _attached_tiny_model(tmp_path)
    tokens = np.random.default_rng(1).integers(0, 240, size=(1, 70))
    tokens[0, 23] = EOS
    outputs: dict[int, mx.array] = {}
    for budget in (1 << 40, 1):  # every row at once; then 16-row slices
        monkeypatch.setattr(qwen4_exp, "_ATTN_SLICE_BYTES", budget)
        cache = model.make_cache()
        parts: list[mx.array] = []
        start = 0
        for length in chunks:
            parts.append(model(mx.array(tokens[:, start : start + length]), cache))
            start += length
        outputs[budget] = mx.concatenate(parts, axis=1)
    whole, sliced = outputs[1 << 40], outputs[1]
    assert float(mx.abs(sliced - whole).max().item()) < 1e-5, chunks
