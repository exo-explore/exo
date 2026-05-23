"""Opt-in numerical parity test against a real Qwen MTP artifact.

Gated on ``MLX_LM_RUN_NETWORK_TESTS=1`` because it requires the model
checkpoint on disk (~30 GB) and ~10 s of compute per run.

Pass criterion: top-1 agreement between MTP's next-next-token
prediction and the main lm_head's prediction over a fixed
post-norm hidden context is >= 60%. This is the regression that
prevents PR-#1226-style breakage.

The probe walks the model incrementally rather than via a batched
prefill -- the Qwen3.5 GatedDeltaNet implementation in mlx-lm is not
strictly per-position causal during batched prefill, which would
distort the per-position parity numbers (see
scripts/mtp_parity_probe.py for the falsification probe that
identified this).
"""

# pyright: reportAny=false, reportUnknownMemberType=false, reportUnknownArgumentType=false, reportUnknownVariableType=false, reportUnknownParameterType=false, reportMissingParameterType=false, reportUnnecessaryTypeIgnoreComment=false

from __future__ import annotations

import os
from pathlib import Path

import mlx.core as mx
import pytest

from exo.worker.engines.mlx.vendor.qwen3_5_mtp_loader import load_mtp_model

_NETWORK_TESTS_FLAG = "MLX_LM_RUN_NETWORK_TESTS"
_MODEL_PATH_ENV = "EXO_NATIVE_MTP_PARITY_MODEL_PATH"
_MODEL_DIR_RAW = os.environ.get(_MODEL_PATH_ENV)
_MODEL_DIR = Path(_MODEL_DIR_RAW) if _MODEL_DIR_RAW else None

# 48 tokens of natural English; incremental forward over this is ~5 s on M5 Max.
_PROMPT = (
    "The quick brown fox jumps over the lazy dog. In a small village by the "
    "river there lived an old clockmaker whose hands could repair any "
    "broken timepiece. Each morning before the sun rose he would walk "
    "through the cobbled streets carrying a leather satchel full of tiny "
    "tools, and the children would wave from their windows as he passed."
)

_TOP1_FLOOR = 0.60


@pytest.mark.slow
@pytest.mark.skipif(
    os.environ.get(_NETWORK_TESTS_FLAG) != "1",
    reason=f"set {_NETWORK_TESTS_FLAG}=1 to run the MTP parity regression",
)
@pytest.mark.skipif(
    _MODEL_DIR is None,
    reason=f"set {_MODEL_PATH_ENV} to a local Qwen MTP model directory",
)
@pytest.mark.skipif(
    _MODEL_DIR is not None and not _MODEL_DIR.exists(),
    reason=f"configured Qwen MTP model directory does not exist: {_MODEL_DIR}",
)
def test_mtp_top1_agreement_against_lm_head() -> None:
    """Top-1 agreement with a primed MTP cache must stay above the floor."""
    try:
        from transformers import AutoTokenizer  # type: ignore[import-untyped]
    except ImportError:
        pytest.skip("transformers not installed")

    assert _MODEL_DIR is not None
    model, _cfg = load_mtp_model(_MODEL_DIR, lazy=False, strict=True)
    tok = AutoTokenizer.from_pretrained(str(_MODEL_DIR))

    token_ids = tok(_PROMPT, return_tensors="np")["input_ids"][0].tolist()
    token_ids = token_ids[: min(len(token_ids), 48)]
    assert len(token_ids) >= 16, f"too few tokens to test: {len(token_ids)}"
    tokens = mx.array([token_ids], dtype=mx.int32)
    seq_len = tokens.shape[1]

    # Incremental main forward to get truly causal per-position post-norm
    # hidden + lm_head logits.
    text_model = model.language_model
    inner = text_model.model
    cache = text_model.make_cache()
    post_per_pos: list[mx.array] = []
    logits_per_pos: list[mx.array] = []
    from mlx_lm.models.base import create_attention_mask, create_ssm_mask

    for t in range(seq_len):
        one = tokens[:, t : t + 1]
        h = inner.embed_tokens(one)
        fa_mask = create_attention_mask(h, cache[inner.fa_idx])
        ssm_mask = create_ssm_mask(h, cache[inner.ssm_idx])
        for layer, layer_cache in zip(inner.layers, cache, strict=True):
            mask = ssm_mask if layer.is_linear else fa_mask
            h = layer(h, mask=mask, cache=layer_cache)
        post = inner.norm(h)
        if text_model.args.tie_word_embeddings:
            lg = inner.embed_tokens.as_linear(post)
        else:
            lg = text_model.lm_head(post)
        mx.eval(post, lg)
        post_per_pos.append(post)
        logits_per_pos.append(lg)
    post_norm_hidden = mx.concatenate(post_per_pos, axis=1)
    main_logits = mx.concatenate(logits_per_pos, axis=1)
    mx.eval(post_norm_hidden, main_logits)

    # Primed-cache MTP walk
    mtp_cache = model.make_mtp_cache()
    matches = 0
    total = 0
    for t in range(seq_len - 1):
        h_t = post_norm_hidden[:, t : t + 1, :]
        next_tok = tokens[:, t + 1 : t + 2]
        mtp_logits = model.mtp_forward(h_t, next_tok, mtp_cache=mtp_cache)
        assert isinstance(mtp_logits, mx.array)
        mx.eval(mtp_logits)
        mtp_pred = int(mx.argmax(mtp_logits[0, -1]).item())
        main_pred = int(mx.argmax(main_logits[0, t + 1]).item())
        if mtp_pred == main_pred:
            matches += 1
        total += 1
    top1 = matches / total
    assert top1 >= _TOP1_FLOOR, (
        f"MTP top-1 agreement {top1:.4f} < floor {_TOP1_FLOOR:.2f} "
        f"(matches={matches}/{total}) -- this is the PR #1226 regression"
    )
