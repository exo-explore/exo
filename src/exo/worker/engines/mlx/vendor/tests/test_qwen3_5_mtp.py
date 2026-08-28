"""Synthetic unit tests for the native Qwen3.5/3.6 MTP model class.

These tests build tiny configurations and synthetic weight dicts so the
whole suite runs in seconds without requiring any model download.
The opt-in parity test against the real MTPLX artifact lives in a
separate file (test_qwen3_5_mtp_parity.py).
"""

# pyright: reportAny=false, reportUnknownMemberType=false, reportUnknownArgumentType=false, reportUnknownVariableType=false, reportUnknownParameterType=false, reportMissingParameterType=false, reportPrivateUsage=false, reportArgumentType=false, reportOperatorIssue=false

from __future__ import annotations

from typing import Any, Dict

import mlx.core as mx
import mlx.nn as nn
import pytest

from exo.worker.engines.mlx.vendor.qwen3_5_mtp import (
    Model,
    ModelArgs,
    MTPModule,
    MTPWeightsNotFound,
    TextModelArgs,
    _classify_mtp_key_set,
    _quantize_mtp_module,
)

# ---------------------------------------------------------------------------
# Test fixtures: minimal "tiny" Qwen3.6-shaped config.
# ---------------------------------------------------------------------------


def _tiny_text_config(*, with_mtp: bool = True) -> Dict[str, Any]:
    """Tiny config (32 hidden, 1 layer per group, 1 MTP layer) for fast tests.

    The shapes are chosen to be self-consistent with the Qwen3.6 architecture
    constraints (head dims, GQA, full_attention_interval) while staying
    extremely small.
    """
    cfg: Dict[str, Any] = {
        "model_type": "qwen3_5_text",
        "hidden_size": 64,
        "intermediate_size": 128,
        "num_hidden_layers": 4,
        "num_attention_heads": 4,
        "rms_norm_eps": 1e-6,
        "vocab_size": 256,
        "num_key_value_heads": 2,
        "max_position_embeddings": 256,
        "linear_num_value_heads": 4,
        "linear_num_key_heads": 2,
        "linear_key_head_dim": 16,
        "linear_value_head_dim": 16,
        "linear_conv_kernel_dim": 4,
        "tie_word_embeddings": False,
        "attention_bias": False,
        "head_dim": 16,
        "full_attention_interval": 2,
        "num_experts": 0,
        "rope_parameters": {
            "type": "default",
            "rope_theta": 10000.0,
            "partial_rotary_factor": 1.0,
        },
    }
    if with_mtp:
        cfg["mtp_num_hidden_layers"] = 1
    return cfg


def _tiny_model_args(*, with_mtp: bool = True) -> ModelArgs:
    return ModelArgs(
        model_type="qwen3_5",
        text_config=_tiny_text_config(with_mtp=with_mtp),
    )


# ---------------------------------------------------------------------------
# Shape tests
# ---------------------------------------------------------------------------


def test_forward_shapes() -> None:
    """``Model(...)(tokens)`` returns logits of shape (B, T, V)."""
    args = _tiny_model_args(with_mtp=True)
    model = Model(args)
    mx.eval(model.parameters())
    tokens = mx.array([[1, 2, 3, 4, 5]], dtype=mx.int32)
    logits = model(tokens)
    assert isinstance(logits, mx.array)
    mx.eval(logits)
    assert logits.shape == (1, 5, 256)


def test_forward_return_hidden_shapes() -> None:
    """``return_hidden=True`` returns (logits, hidden) with hidden = post-norm."""
    args = _tiny_model_args(with_mtp=True)
    model = Model(args)
    mx.eval(model.parameters())
    tokens = mx.array([[1, 2, 3, 4, 5]], dtype=mx.int32)
    result = model(tokens, return_hidden=True)
    assert isinstance(result, tuple)
    logits, hidden = result
    mx.eval(logits, hidden)
    assert logits.shape == (1, 5, 256)
    assert hidden.shape == (1, 5, 64)


def test_mtp_forward_shape() -> None:
    """``mtp_forward`` returns logits of shape (B, T, V) from hidden + token."""
    args = _tiny_model_args(with_mtp=True)
    model = Model(args)
    mx.eval(model.parameters())
    hidden = mx.random.uniform(shape=(1, 1, 64))
    next_token = mx.array([[42]], dtype=mx.int32)
    cache = model.make_mtp_cache()
    logits = model.mtp_forward(hidden, next_token, mtp_cache=cache)
    assert isinstance(logits, mx.array)
    mx.eval(logits)
    assert logits.shape == (1, 1, 256)


def test_make_mtp_cache_length_matches_mtp_layers() -> None:
    args = _tiny_model_args(with_mtp=True)
    model = Model(args)
    cache = model.make_mtp_cache()
    assert len(cache) == args.text_config["mtp_num_hidden_layers"]


# ---------------------------------------------------------------------------
# Norm-shift gate idempotence
# ---------------------------------------------------------------------------


def _build_synthetic_main_weights(args: TextModelArgs) -> Dict[str, mx.array]:
    """Build a synthetic weight dict that matches the main model shape.

    Norms start with mean ~0 (the unshifted form) so we can observe the
    +1.0 shift kicking in.
    """
    hidden = args.hidden_size
    vocab = args.vocab_size
    inter = args.intermediate_size
    head_dim = args.head_dim
    n_heads = args.num_attention_heads
    n_kv = args.num_key_value_heads
    # Use float32 with mean 0 so the shift gate fires deterministically.
    rng = mx.random.uniform
    weights: Dict[str, mx.array] = {
        "model.embed_tokens.weight": rng(shape=(vocab, hidden)) - 0.5,
        "model.norm.weight": mx.zeros((hidden,), dtype=mx.float32),
        "lm_head.weight": rng(shape=(vocab, hidden)) - 0.5,
    }
    for li in range(args.num_hidden_layers):
        is_linear = (li + 1) % args.full_attention_interval != 0
        prefix = f"model.layers.{li}"
        weights[f"{prefix}.input_layernorm.weight"] = mx.zeros(
            (hidden,), dtype=mx.float32
        )
        weights[f"{prefix}.post_attention_layernorm.weight"] = mx.zeros(
            (hidden,), dtype=mx.float32
        )
        weights[f"{prefix}.mlp.gate_proj.weight"] = rng(shape=(inter, hidden)) - 0.5
        weights[f"{prefix}.mlp.down_proj.weight"] = rng(shape=(hidden, inter)) - 0.5
        weights[f"{prefix}.mlp.up_proj.weight"] = rng(shape=(inter, hidden)) - 0.5
        if is_linear:
            # GatedDeltaNet weights -- shapes mirror what stock code builds.
            key_dim = args.linear_num_key_heads * args.linear_key_head_dim
            value_dim = args.linear_num_value_heads * args.linear_value_head_dim
            conv_dim = key_dim * 2 + value_dim
            weights[f"{prefix}.linear_attn.conv1d.weight"] = (
                rng(shape=(conv_dim, 1, args.linear_conv_kernel_dim)) - 0.5
            )
            weights[f"{prefix}.linear_attn.in_proj_qkv.weight"] = (
                rng(shape=(conv_dim, hidden)) - 0.5
            )
            weights[f"{prefix}.linear_attn.in_proj_z.weight"] = (
                rng(shape=(value_dim, hidden)) - 0.5
            )
            weights[f"{prefix}.linear_attn.in_proj_b.weight"] = (
                rng(shape=(args.linear_num_value_heads, hidden)) - 0.5
            )
            weights[f"{prefix}.linear_attn.in_proj_a.weight"] = (
                rng(shape=(args.linear_num_value_heads, hidden)) - 0.5
            )
            weights[f"{prefix}.linear_attn.out_proj.weight"] = (
                rng(shape=(hidden, value_dim)) - 0.5
            )
            weights[f"{prefix}.linear_attn.dt_bias"] = mx.ones(
                (args.linear_num_value_heads,)
            )
            weights[f"{prefix}.linear_attn.A_log"] = mx.zeros(
                (args.linear_num_value_heads,)
            )
            weights[f"{prefix}.linear_attn.norm.weight"] = mx.ones(
                (args.linear_value_head_dim,)
            )
        else:
            weights[f"{prefix}.self_attn.q_proj.weight"] = (
                rng(shape=(n_heads * head_dim * 2, hidden)) - 0.5
            )
            weights[f"{prefix}.self_attn.k_proj.weight"] = (
                rng(shape=(n_kv * head_dim, hidden)) - 0.5
            )
            weights[f"{prefix}.self_attn.v_proj.weight"] = (
                rng(shape=(n_kv * head_dim, hidden)) - 0.5
            )
            weights[f"{prefix}.self_attn.o_proj.weight"] = (
                rng(shape=(hidden, n_heads * head_dim)) - 0.5
            )
            weights[f"{prefix}.self_attn.q_norm.weight"] = mx.zeros(
                (head_dim,), dtype=mx.float32
            )
            weights[f"{prefix}.self_attn.k_norm.weight"] = mx.zeros(
                (head_dim,), dtype=mx.float32
            )
    return weights


def _build_synthetic_mtp_weights(args: TextModelArgs) -> Dict[str, mx.array]:
    """Build a synthetic MTP weight dict with unshifted norms."""
    hidden = args.hidden_size
    inter = args.intermediate_size
    head_dim = args.head_dim
    n_heads = args.num_attention_heads
    n_kv = args.num_key_value_heads
    rng = mx.random.uniform
    weights: Dict[str, mx.array] = {
        "mtp.fc.weight": rng(shape=(hidden, 2 * hidden)) - 0.5,
        "mtp.pre_fc_norm_hidden.weight": mx.zeros((hidden,), dtype=mx.float32),
        "mtp.pre_fc_norm_embedding.weight": mx.zeros((hidden,), dtype=mx.float32),
        "mtp.norm.weight": mx.zeros((hidden,), dtype=mx.float32),
        "mtp.layers.0.input_layernorm.weight": mx.zeros((hidden,), dtype=mx.float32),
        "mtp.layers.0.post_attention_layernorm.weight": mx.zeros(
            (hidden,), dtype=mx.float32
        ),
        "mtp.layers.0.self_attn.q_proj.weight": rng(
            shape=(n_heads * head_dim * 2, hidden)
        )
        - 0.5,
        "mtp.layers.0.self_attn.k_proj.weight": rng(shape=(n_kv * head_dim, hidden))
        - 0.5,
        "mtp.layers.0.self_attn.v_proj.weight": rng(shape=(n_kv * head_dim, hidden))
        - 0.5,
        "mtp.layers.0.self_attn.o_proj.weight": rng(shape=(hidden, n_heads * head_dim))
        - 0.5,
        "mtp.layers.0.self_attn.q_norm.weight": mx.zeros((head_dim,), dtype=mx.float32),
        "mtp.layers.0.self_attn.k_norm.weight": mx.zeros((head_dim,), dtype=mx.float32),
        "mtp.layers.0.mlp.gate_proj.weight": rng(shape=(inter, hidden)) - 0.5,
        "mtp.layers.0.mlp.down_proj.weight": rng(shape=(hidden, inter)) - 0.5,
        "mtp.layers.0.mlp.up_proj.weight": rng(shape=(inter, hidden)) - 0.5,
    }
    return weights


def test_norm_shift_idempotent() -> None:
    """Sanitize shifts mean=0 norms to mean=1; a second sanitize does NOT shift again."""
    args = _tiny_model_args(with_mtp=True)
    model = Model(args)
    main = _build_synthetic_main_weights(model.language_model.args)
    mtp = _build_synthetic_mtp_weights(model.language_model.args)
    raw: Dict[str, mx.array] = {}
    for k, v in main.items():
        raw["language_model." + k] = v
    for k, v in mtp.items():
        raw["language_model." + k] = v

    sanitized = model.sanitize(raw)
    # After first sanitize: all norm.weights should have mean ~1.
    norm_keys = [
        k
        for k in sanitized
        if k.endswith(("input_layernorm.weight", "norm.weight", "q_norm.weight"))
        and sanitized[k].ndim == 1
    ]
    assert len(norm_keys) > 0
    for k in norm_keys:
        m = float(sanitized[k].astype(mx.float32).mean().item())
        assert abs(m - 1.0) < 0.05, f"first sanitize on {k}: mean={m}"

    # Second sanitize on the already-shifted dict should NOT shift again.
    # The sanitized output is already keyed in the canonical
    # ``language_model.model.X`` form, so we can feed it straight back
    # to ``model.sanitize`` (the outer Model.sanitize tolerates the
    # ``language_model.*`` prefix as a pass-through).
    sanitized2 = model.sanitize(sanitized)
    for k in norm_keys:
        m = float(sanitized2[k].astype(mx.float32).mean().item())
        assert abs(m - 1.0) < 0.05, (
            f"second sanitize on {k}: mean={m} (double-shifted!)"
        )


# ---------------------------------------------------------------------------
# Missing-weights diagnostics
# ---------------------------------------------------------------------------


def test_missing_weights_raises() -> None:
    """If MTP is declared but no MTP weights and no loader: ``MTPWeightsNotFound``."""
    args = _tiny_model_args(with_mtp=True)
    model = Model(args)
    main = _build_synthetic_main_weights(model.language_model.args)
    raw: Dict[str, mx.array] = {"language_model." + k: v for k, v in main.items()}
    # Crucially: do NOT install a sidecar loader, do NOT include mtp.* keys.
    with pytest.raises(MTPWeightsNotFound) as excinfo:
        model.sanitize(raw)
    assert "mtp.safetensors" in excinfo.value.candidates


def test_no_mtp_declared_does_not_raise() -> None:
    """If config has ``mtp_num_hidden_layers=0``, sanitize is happy without MTP keys."""
    args = _tiny_model_args(with_mtp=False)
    model = Model(args)
    main = _build_synthetic_main_weights(model.language_model.args)
    raw: Dict[str, mx.array] = {"language_model." + k: v for k, v in main.items()}
    sanitized = model.sanitize(raw)
    assert not any(k.startswith("mtp.") for k in sanitized)


# ---------------------------------------------------------------------------
# Cache-share differential (proves cache seeding matters)
# ---------------------------------------------------------------------------


def test_cache_share_differential() -> None:
    """Primed-cache MTP differs from fresh-cache MTP after multiple steps."""
    args = _tiny_model_args(with_mtp=True)
    model = Model(args)
    mx.eval(model.parameters())

    # Walk 4 hidden+token pairs through the MTP head with two cache strategies:
    rng_h1 = mx.random.uniform(shape=(1, 1, 64), key=mx.random.key(7))
    rng_h2 = mx.random.uniform(shape=(1, 1, 64), key=mx.random.key(11))
    rng_h3 = mx.random.uniform(shape=(1, 1, 64), key=mx.random.key(13))
    rng_h4 = mx.random.uniform(shape=(1, 1, 64), key=mx.random.key(17))
    hiddens = [rng_h1, rng_h2, rng_h3, rng_h4]
    tokens = [mx.array([[i + 5]], dtype=mx.int32) for i in range(4)]

    # Primed: same cache for all steps
    primed = model.make_mtp_cache()
    primed_outputs = []
    for h, t in zip(hiddens, tokens, strict=True):
        out = model.mtp_forward(h, t, mtp_cache=primed)
        mx.eval(out)
        primed_outputs.append(out)

    # Fresh: new cache each step
    fresh_outputs = []
    for h, t in zip(hiddens, tokens, strict=True):
        fresh = model.make_mtp_cache()
        out = model.mtp_forward(h, t, mtp_cache=fresh)
        mx.eval(out)
        fresh_outputs.append(out)

    # The very first step must agree (both have empty cache).
    first_diff = float(mx.max(mx.abs(primed_outputs[0] - fresh_outputs[0])).item())
    assert first_diff < 1e-4, f"first-step primed/fresh diverge: {first_diff}"

    # Later steps must diverge -- proves cache priming is load-bearing.
    last_diff = float(mx.max(mx.abs(primed_outputs[-1] - fresh_outputs[-1])).item())
    assert last_diff > 1e-3, (
        f"last-step primed/fresh agree (diff={last_diff}); cache priming "
        "isn't actually changing MTP outputs"
    )


# ---------------------------------------------------------------------------
# Hidden-variant correctness (post-norm vs pre-norm)
# ---------------------------------------------------------------------------


def test_post_vs_pre_norm() -> None:
    """The two hidden variants give different logits and hidden."""
    args = _tiny_model_args(with_mtp=True)
    model = Model(args)
    mx.eval(model.parameters())
    hidden = mx.random.uniform(shape=(1, 1, 64))
    token = mx.array([[42]], dtype=mx.int32)
    cache_post = model.make_mtp_cache()
    cache_pre = model.make_mtp_cache()
    out_post = model.mtp_forward(
        hidden,
        token,
        mtp_cache=cache_post,
        mtp_hidden_variant="post_norm",
        return_hidden=True,
    )
    out_pre = model.mtp_forward(
        hidden,
        token,
        mtp_cache=cache_pre,
        mtp_hidden_variant="pre_norm",
        return_hidden=True,
    )
    assert isinstance(out_post, tuple) and isinstance(out_pre, tuple)
    # Logits are computed from POST-norm in both cases (the lm_head only
    # ever sees post-norm). What changes is the *returned* hidden which
    # downstream draft-loops use as the input for the next MTP step.
    h_post = out_post[1]
    h_pre = out_pre[1]
    diff = float(mx.max(mx.abs(h_post - h_pre)).item())
    assert diff > 1e-3, (
        f"post-norm and pre-norm hidden variants are identical (diff={diff}); "
        "variant switch is a no-op"
    )


# ---------------------------------------------------------------------------
# Quant-policy classification
# ---------------------------------------------------------------------------


def test_classify_unquantized() -> None:
    keys = (
        "mtp.fc.weight",
        "mtp.layers.0.self_attn.q_proj.weight",
        "mtp.layers.0.self_attn.k_proj.weight",
        "mtp.norm.weight",
    )
    assert _classify_mtp_key_set(keys) == "unquantized"


def test_classify_cyankiwi() -> None:
    keys = (
        "mtp.fc.weight",  # fc unquantized
        "mtp.layers.0.self_attn.q_proj.weight",
        "mtp.layers.0.self_attn.q_proj.scales",  # presence of attn scales -> cyankiwi
        "mtp.layers.0.self_attn.q_proj.biases",
        "mtp.norm.weight",
    )
    assert _classify_mtp_key_set(keys) == "cyankiwi"


def test_classify_all_quantized() -> None:
    keys = (
        "mtp.fc.weight",
        "mtp.fc.scales",  # presence of fc scales -> "all"
        "mtp.fc.biases",
        "mtp.layers.0.self_attn.q_proj.weight",
        "mtp.layers.0.self_attn.q_proj.scales",
        "mtp.layers.0.self_attn.q_proj.biases",
    )
    assert _classify_mtp_key_set(keys) == "all"


def test_quantize_mtp_module_cyankiwi_leaves_fc_unquantized() -> None:
    args = TextModelArgs.from_dict(_tiny_text_config(with_mtp=True))
    mtp = MTPModule(args, 1)
    _quantize_mtp_module(mtp, policy="cyankiwi", bits=8, group_size=32)
    # fc should still be a plain Linear (NOT QuantizedLinear)
    assert isinstance(mtp.fc, nn.Linear)
    assert not isinstance(mtp.fc, nn.QuantizedLinear)
    # attention q_proj should be QuantizedLinear
    qproj = mtp.layers[0].self_attn.q_proj
    assert isinstance(qproj, nn.QuantizedLinear)


def test_quantize_mtp_module_all_quantizes_fc() -> None:
    args = TextModelArgs.from_dict(_tiny_text_config(with_mtp=True))
    mtp = MTPModule(args, 1)
    _quantize_mtp_module(mtp, policy="all", bits=8, group_size=32)
    assert isinstance(mtp.fc, nn.QuantizedLinear)


def test_quantize_mtp_module_unquantized_is_noop() -> None:
    args = TextModelArgs.from_dict(_tiny_text_config(with_mtp=True))
    mtp = MTPModule(args, 1)
    _quantize_mtp_module(mtp, policy="unquantized", bits=8, group_size=32)
    assert isinstance(mtp.fc, nn.Linear) and not isinstance(mtp.fc, nn.QuantizedLinear)
    assert not isinstance(mtp.layers[0].self_attn.q_proj, nn.QuantizedLinear)
