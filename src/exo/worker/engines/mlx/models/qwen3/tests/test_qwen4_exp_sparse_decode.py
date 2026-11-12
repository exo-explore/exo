# pyright: reportAny=false, reportUnknownMemberType=false, reportUnknownArgumentType=false, reportPrivateUsage=false, reportUnusedFunction=false
"""Sparse decode attention in qwen4_exp (EXO_QSA_SPARSE): decode and verify steps attend to
the indexer's selected keys gathered, and the indexer reuses the pooled keys of complete
blocks. Checked against the dense path (a mask over every key) on the tiny model of
test_qwen4_exp.py, on the CPU so that only rounding separates the two."""

from collections.abc import Generator
from copy import deepcopy
from pathlib import Path

import mlx.core as mx
import numpy as np
import pytest

from exo.worker.engines.mlx.models.qwen3 import qwen4_exp
from exo.worker.engines.mlx.models.qwen3.qwen4_exp import (
    QSAIndexer,
    QSAKVCache,
    TextArgs,
)
from exo.worker.engines.mlx.models.qwen3.tests.test_qwen4_exp import (
    TINY,
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


def _close(a: mx.array, b: mx.array, rel: float = 1e-4) -> bool:
    scale = max(float(mx.abs(b).max().item()), 1e-6)
    return float(mx.abs(a - b).max().item()) <= rel * scale


@pytest.mark.parametrize("rows", [1, 2, 3])
def test_selected_keys_are_the_ones_the_mask_marks(rows: int) -> None:
    indexer = QSAIndexer(TextArgs.from_dict(TINY), rotary_dims=8)
    x = mx.random.normal((1, 47, 64))
    mask = indexer.select(x, None)
    assert mask is not None
    keep = np.array(mask[0, 0])
    # rows before, at and after the selection starts (16 keys), and across block edges
    for start in (14, 17, 39, 40, 41, 44):
        cache = QSAKVCache()
        indexer.select(x[:, :start], cache)
        cache.offset = start  # the attention layer advances it after the indexer
        selected = indexer.select_keys(x[:, start : start + rows], cache)
        assert len(selected) == rows
        for i, positions in enumerate(selected):
            want = np.nonzero(keep[start + i, : start + i + 1])[0]
            if positions is None:  # dense: every key up to the row
                assert len(want) == start + i + 1, (start, i)
            else:
                np.testing.assert_array_equal(np.array(positions), want)


def test_pooled_blocks_are_reused_and_rebuilt_when_the_keys_change() -> None:
    indexer = QSAIndexer(TextArgs.from_dict(TINY), rotary_dims=8)
    x = mx.random.normal((1, 300, 64))
    cache = QSAKVCache()
    for start, stop in ((0, 37), (37, 38), (38, 100), (100, 261)):  # 261 grows past 256
        indexer.select(x[:, start:stop], cache)
        cache.offset = stop
        assert cache.pooled is not None and cache.index_keys is not None
        want = indexer._pool(cache.index_keys[:, :stop], 0, stop // indexer.ratio)
        assert _close(cache.pooled, want, 1e-6), stop
    # a trim (a rejected draft, a prefix-cache restore) re-pools the blocks after it
    cache.trim(30)
    indexer.select(x[:, 231:232], cache)
    cache.offset = 232
    assert cache.index_keys is not None and cache.pooled is not None
    assert _close(cache.pooled, indexer._pool(cache.index_keys[:, :232], 0, 58), 1e-6)
    # a copy (the prefix cache's) keeps them in step with its own keys
    copy = deepcopy(cache)
    assert copy.pooled_from is copy.index_keys
    indexer.select(x[:, 232:233], copy)


@pytest.mark.parametrize("prefilled", [10, 40], ids=["dense", "sparse selection"])
def test_sparse_decode_matches_the_dense_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, prefilled: int
) -> None:
    model = _attached_tiny_model(tmp_path)
    tokens = [int(t) for t in np.random.default_rng(3).integers(0, 240, size=80)]
    outputs: dict[bool, list[mx.array]] = {}
    for sparse in (False, True):
        monkeypatch.setattr(qwen4_exp, "_QSA_SPARSE", sparse)
        cache = model.make_cache()
        model(mx.array([tokens[:prefilled]]), cache)
        steps = [
            model(mx.array([[t]]), cache) for t in tokens[prefilled : prefilled + 30]
        ]
        # and a 3-row call (a verify step's shape) after them
        steps.append(model(mx.array([tokens[prefilled + 30 : prefilled + 33]]), cache))
        outputs[sparse] = steps
    for i, (dense, sparse) in enumerate(
        zip(outputs[False], outputs[True], strict=True)
    ):
        assert _close(sparse, dense), i
