"""Native Multi-Token Prediction drafter for Qwen3.5/3.6.

Native MTP drafts come from the target model's own MTP head: a single
transformer layer that consumes the target's post-norm hidden state +
the previously-sampled token and predicts the next token via the shared
``lm_head``. No sibling drafter LM is loaded and no separate model KV
cache exists -- the MTP layer has its own KV cache that's populated from
prompt prefill and extended on every accepted draft.

**Cache rollback (Qwen3.6 GatedDeltaNet)**

``mlx_trim_prompt_cache`` decrements KV offsets but does NOT roll back
the GatedDeltaNet linear-attention recurrent state on Qwen3.5/3.6
targets, so a bare trim after the batched verify leaves the GDN cache
on the SPECULATIVE trajectory (the rejected drafts have been folded
into the recurrent state and can't be undone via offset
arithmetic). The ``trim_after_batched_probe`` measurement showed
54.2% logit mismatch (Logit L2 distances 300-1000) without the fix.

The loop snapshots GDN state before the batched verify; on partial accept it
trims the main cache fully, restores GDN from the snapshot, and re-forwards
``[current_token, *drafts[:n_accepted]]`` so the cache is advanced through
retained tokens only. On full accept the cache is already on-trajectory, so
rollback is skipped.

The MTP cache is full-attention (``vendor/qwen3_5_mtp.py``
``fa_layer_idx``), so its trim path uses plain
:func:`mlx_trim_prompt_cache` -- no rollback needed.

The loop:

1. **Prime** the MTP KV cache by walking prompt positions through the
   main model on a fresh cache to capture post-norm hiddens, then
   feeding those into ``model.mtp_update_cache`` so the MTP attention
   sees prompt context before the first draft fires.
2. **Draft** K tokens by chaining ``model.mtp_forward`` -- each step
   consumes the previous step's post-norm hidden so the MTP layer
   builds a self-referential chain. K is bounded by
   ``card.native_mtp.max_k`` upstream (default 3).
3. **Verify** all K drafts in a single ``(K+1)``-token target forward.
   Walk the target's preferred tokens; accept while they match drafts,
   emit one bonus from the target's choice at the first mismatch (or
   after a full-accept). Trim both caches by ``K - num_accepted``.

Native MTP is single-node-only by design (the loader gate
:func:`exo.worker.engines.mlx.utils_mlx._native_mtp_loader_eligible`
only fires on single-node placements): a TP-sharded verify forward
amortises ``K+1`` tokens over ``K+1`` compute units, eating the MTP
speedup.
"""

# pyright: reportUnknownMemberType=false, reportUnknownArgumentType=false, reportUnknownVariableType=false, reportAttributeAccessIssue=false, reportCallIssue=false, reportArgumentType=false, reportAny=false
# This module composes over untyped mlx-lm tokenizer / vendor MTP
# attributes; the type-checker can't see through nn.Module subclassing
# in the way mlx-lm uses it. Mirrors the same pragma block as
# ``vendor/qwen3_5_mtp.py``.

from __future__ import annotations

import os
import time
from collections.abc import Callable, Generator, Iterable, Sequence
from typing import Any, Literal, cast, final

import mlx.core as mx
import mlx.nn as nn
from mlx.nn.layers.distributed import sum_gradients
from mlx_lm.generate import GenerationResponse
from mlx_lm.models.base import create_attention_mask, create_ssm_mask
from mlx_lm.models.cache import trim_prompt_cache as mlx_trim_prompt_cache
from mlx_lm.sample_utils import apply_min_p, apply_top_k, apply_top_p
from mlx_lm.tokenizer_utils import TokenizerWrapper

from exo.worker.engines.mlx.spec_cache import (
    restore_cache,
    rollback_after_verify,
    snapshot_untrimmable_cache_lazy,
)
from exo.worker.engines.mlx.types import KVCacheType, Model
from exo.worker.runner.bootstrap import logger

_MoeVerifierPolicy = Literal[
    "safe",
    "hybrid_full_accept",
    "batched",
    "row_moe",
    "route_locked",
]
_DENSE_GDN_STATE_HISTORY_MODEL_TYPES = frozenset({"qwen3_5", "qwen3_5_text"})
_MOE_GDN_STATE_HISTORY_MODEL_TYPES = frozenset({"qwen3_5_moe", "qwen3_5_moe_text"})


def is_native_mtp_dispatchable(model: object) -> bool:
    """Return ``True`` when ``model`` is a vendored native-MTP Model instance.

    The loader (:mod:`exo.worker.engines.mlx.vendor.qwen3_5_mtp_loader`)
    swaps in :class:`vendor.qwen3_5_mtp.Model` (or its
    ``_PatchedMtpModel`` subclass) when the card declares ``native_mtp``
    and the on-disk weights are recoverable. We ``isinstance``-check
    against that concrete class rather than probe for the
    ``mtp_forward`` / ``make_mtp_cache`` / ``mtp_update_cache`` methods
    structurally because ``unittest.mock.MagicMock()`` auto-creates any
    attribute on access -- a structural check would falsely engage the
    NativeMTPDrafter for any mocked target model and break unrelated
    routing tests. The import is local to keep this module's import
    surface small for workers that never load Qwen3.5/3.6.

    This is the dispatch-side gate. The builder-side gate
    (:func:`exo.worker.engines.mlx.utils_mlx.is_native_mtp_runnable`)
    consults the card + placement before the model is loaded.
    """
    try:
        from exo.worker.engines.mlx.vendor.qwen3_5_mtp import Model as _VendorMtpModel
    except ImportError:
        return False
    return isinstance(model, _VendorMtpModel)


def _eos_ids_from_tokenizer(tokenizer: TokenizerWrapper) -> list[int]:
    """Tokenizer-agnostic EOS lookup mirroring drafter.py / coupled_drafter.py."""
    eos = getattr(tokenizer, "eos_token_ids", None)
    if eos is None:
        eos = getattr(tokenizer, "eos_token_id", None)
    if eos is None:
        return []
    if isinstance(eos, int):
        return [eos]
    if isinstance(eos, Iterable) and not isinstance(eos, (str, bytes)):
        return [int(x) for x in eos]
    try:
        return [int(eos)]
    except (TypeError, ValueError):
        return []


def _token_array_to_int(token: mx.array) -> int:
    """Read a single-token ``mx.array`` as a Python ``int``."""
    return int(token.reshape(()).item())


@final
class _SpecSampler:
    """Reconstructs the request's sampling distribution for lossless MTP.

    The drafter only receives a ``sampler`` (``logits -> token``), which is
    not enough for probability-ratio (Leviathan/Chen) acceptance: that needs
    the *distribution* the request samples from. We rebuild it by replicating
    mlx-lm's ``make_sampler`` chain (``top_p`` / ``min_p`` / ``top_k`` masking,
    then ``/temperature`` and a softmax) so the draft distribution ``q`` and
    the target distribution ``p`` are computed with identical transforms.

    Mirrors ``mtp_generate_step`` in mlx-lm PR #990 (the PR this drafter
    derives from): ``q``/``p`` are the temperature-adjusted, filtered
    distributions; draft tokens are sampled from ``q``; a draft token ``x`` is
    accepted with probability ``min(1, p(x)/q(x))``; on rejection the
    replacement is drawn from the normalized residual ``(p - q)+``; after a
    full accept the bonus is drawn from ``p``. With ``temperature == 0`` this
    reduces to "accept iff ``x == argmax(p)``", recovering exact greedy.

    Note: ``logits_processors`` (repetition/presence/frequency penalties,
    logit bias) are intentionally NOT applied here. They are stateful over the
    emitted-token history and cannot be replayed consistently across the K
    chained MTP draft steps and the batched K+1 verify, so applying them
    inconsistently would BREAK the lossless guarantee rather than help it.
    Penalties stay unhandled on the native-MTP path (as before); only the
    request's temperature/top_p/top_k/min_p now govern the distribution.
    """

    def __init__(
        self,
        *,
        temperature: float,
        top_p: float,
        top_k: int,
        min_p: float,
        min_tokens_to_keep: int,
    ) -> None:
        self._temperature: float = temperature
        self._is_greedy: bool = temperature <= 0.0
        self._top_p: float = top_p
        self._top_k: int = top_k
        self._min_p: float = min_p
        self._min_tokens_to_keep: int = max(1, min_tokens_to_keep)

    @property
    def is_greedy(self) -> bool:
        return self._is_greedy

    def _filter(self, logprobs: mx.array) -> mx.array:
        """Apply the request's masking filters (top_p / min_p / top_k)."""
        masked = logprobs
        if 0.0 < self._top_p < 1.0:
            masked = apply_top_p(masked, self._top_p)
        if self._min_p != 0.0:
            masked = apply_min_p(masked, self._min_p, self._min_tokens_to_keep)
        if self._top_k > 0:
            masked = apply_top_k(masked, self._top_k)
        return masked

    def distribution(self, last_logits: mx.array) -> mx.array:
        """Return the request's sampling distribution ``p`` for ``(1, V)`` logits.

        ``last_logits`` is the final-position logit row, shape ``(1, V)``. The
        returned array is normalized probabilities over the vocab with masked
        entries at exactly ``0`` (temperature-adjusted, filtered).
        """
        logprobs = last_logits - mx.logsumexp(last_logits, axis=-1, keepdims=True)
        if self._is_greedy:
            # Degenerate distribution at the argmax; the ratio test below
            # reduces to "x == argmax(p)" and the residual to argmax(p).
            argmax = mx.argmax(logprobs, axis=-1)
            return mx.zeros_like(logprobs).at[0, argmax].add(1.0)
        masked = self._filter(logprobs)
        scaled = masked / self._temperature
        accept_logprobs = scaled - mx.logsumexp(scaled, axis=-1, keepdims=True)
        return mx.exp(accept_logprobs)

    def sample_from_distribution(self, distribution: mx.array) -> mx.array:
        """Sample one token id ``(1, 1)`` int32 from a probability row ``(1, V)``."""
        if self._is_greedy:
            return mx.argmax(distribution, axis=-1).reshape(1, 1).astype(mx.int32)
        # ``mx.log(0) -> -inf`` which categorical treats as probability 0.
        return (
            mx.random.categorical(mx.log(distribution)).reshape(1, 1).astype(mx.int32)
        )

    def residual_token(self, target_dist: mx.array, draft_dist: mx.array) -> mx.array:
        """Sample the replacement token from the residual ``(p - q)+`` (or ``p``)."""
        if self._is_greedy:
            return mx.argmax(target_dist, axis=-1).reshape(1, 1).astype(mx.int32)
        residual = mx.maximum(target_dist - draft_dist, 0.0)
        normalizer = residual.sum(axis=-1, keepdims=True)
        # When the residual mass is ~0 (draft already covered all target mass),
        # fall back to sampling from the target distribution directly.
        distribution = mx.where(normalizer > 0, residual, target_dist)
        return self.sample_from_distribution(distribution)


def _accept_draft_token(
    *,
    is_greedy: bool,
    draft_token: int,
    target_argmax: int,
    target_prob: float,
    draft_prob: float,
    uniform: float,
) -> bool:
    """Probability-ratio acceptance for one drafted token.

    Greedy: accept iff the drafted token equals the target's argmax.
    Stochastic: accept with probability ``min(1, p(x)/q(x))`` against a
    pre-drawn ``uniform`` in ``[0, 1)`` (Leviathan/Chen). ``draft_prob`` is
    the draft head's probability ``q(x)`` for the drafted token; a draft that
    the target assigns zero mass (``p(x) == 0``, e.g. masked out by top_k) is
    always rejected.
    """
    if is_greedy:
        return draft_token == target_argmax
    if draft_prob <= 0.0:
        # The draft token was not in the draft's own support; treat as a hard
        # reject so the residual replacement (which equals p here) is emitted.
        return False
    return uniform < (target_prob / draft_prob)


def _spec_accept_walk(
    *,
    spec_sampler: _SpecSampler,
    verify_logits: mx.array,
    draft_tokens: Sequence[int],
    draft_distributions: Sequence[mx.array],
    accept_uniforms: Sequence[float],
    k: int,
) -> tuple[int, mx.array]:
    """Probability-ratio accept walk over a batched ``(1, K+1, V)`` verify.

    Returns ``(num_accepted, replacement_token_arr)`` where
    ``replacement_token_arr`` is the ``(1, 1)`` int32 token emitted after the
    accepted prefix -- the residual sample at the first rejected position, or
    the target bonus sample at position ``num_accepted == K`` (full accept).
    The caller's cache machinery already handles trimming/rollback to land the
    cache on ``[current_token, *drafts[:num_accepted]]``.
    """
    # Per-position target distributions p_0..p_K.
    target_dists = [
        spec_sampler.distribution(verify_logits[:, i, :]) for i in range(k + 1)
    ]
    # Consolidated readback: single sync for all K+1 argmax predictions.
    target_argmax_arr = mx.argmax(verify_logits, axis=-1).astype(mx.int32)
    mx.eval(target_argmax_arr)
    target_argmaxes: list[int] = [
        int(t) for t in list(target_argmax_arr.squeeze(0).tolist())
    ]
    num_accepted = 0
    for i in range(k):
        draft_token = draft_tokens[i]
        target_prob = float(target_dists[i][0, draft_token].item())
        draft_prob = float(draft_distributions[i][0, draft_token].item())
        if _accept_draft_token(
            is_greedy=spec_sampler.is_greedy,
            draft_token=draft_token,
            target_argmax=target_argmaxes[i],
            target_prob=target_prob,
            draft_prob=draft_prob,
            uniform=accept_uniforms[i],
        ):
            num_accepted += 1
            continue
        # Rejected at position i: replacement from residual (p_i - q_i)+.
        replacement = spec_sampler.residual_token(
            target_dists[i], draft_distributions[i]
        )
        return num_accepted, replacement
    # Full accept: bonus sampled from the target distribution at position K.
    bonus = spec_sampler.sample_from_distribution(target_dists[k])
    return num_accepted, bonus


def _native_mtp_model_type(model: Model) -> str:
    """Best-effort model_type lookup for vendored Qwen3.5/3.6 models."""
    outer_args = getattr(model, "args", None)
    outer_model_type = getattr(outer_args, "model_type", None)
    if isinstance(outer_model_type, str):
        return outer_model_type
    language_model = getattr(model, "language_model", None)
    args = getattr(language_model, "args", None)
    model_type = getattr(args, "model_type", None)
    return model_type if isinstance(model_type, str) else ""


def _requires_sequential_gdn_path(model: Model) -> bool:
    """Return True for 35B-A3B/MoE native MTP, where batched GDN state drifts.

    The dense 27B path has benchmarked well with the batched verifier. The
    qwen3_5_moe 35B path does not: batched prefill changes the recurrent
    GatedDeltaNet state enough that MTP drafts the wrong first token, and a
    batched verifier can produce wrong later-position bonus tokens. Keep this
    slow path scoped to MoE until we have a fused GDN scan that is numerically
    equivalent to token-by-token recurrent updates.
    """
    return _native_mtp_model_type(model) in {"qwen3_5_moe", "qwen3_5_moe_text"}


def _gdn_state_history_commit_default(
    *,
    model_type: str,
    moe_verifier_policy: _MoeVerifierPolicy,
) -> bool:
    """Return whether GDN prefix-state commit is default-on for this model."""
    if model_type in _DENSE_GDN_STATE_HISTORY_MODEL_TYPES:
        return True
    return (
        model_type in _MOE_GDN_STATE_HISTORY_MODEL_TYPES
        and moe_verifier_policy == "route_locked"
    )


def _gdn_state_history_kernel_supported(model: Model) -> bool:
    """Return True when the loaded GDN layers fit the state-history kernel."""
    language_model = getattr(model, "language_model", None)
    inner = getattr(language_model, "model", None)
    layers = getattr(inner, "layers", None)
    if not isinstance(layers, list):
        return False
    for layer in layers:
        if not bool(getattr(layer, "is_linear", False)):
            continue
        gdn = getattr(layer, "linear_attn", None)
        head_k_dim = getattr(gdn, "head_k_dim", None)
        head_v_dim = getattr(gdn, "head_v_dim", None)
        if not isinstance(head_k_dim, int) or not isinstance(head_v_dim, int):
            return False
        return head_k_dim >= 32 and head_k_dim % 32 == 0 and head_v_dim > 0
    return False


def _moe_verifier_policy() -> _MoeVerifierPolicy:
    """Return the qwen3_5_moe verifier policy for native-MTP experiments.

    ``route_locked`` is the optimized production default for Qwen3.6 MoE.
    ``safe`` is the coherent reference path: token-by-token verification.
    ``hybrid_full_accept`` first tries a batched verifier and commits it only
    when every draft token matches; partial/miss rounds fall back to ``safe``.
    ``batched`` is the research fast path for falsifying whether the batched
    verifier can be made coherent under the current cache repair rules.
    ``row_moe`` keeps attention/GatedDeltaNet batched but evaluates sparse
    MoE blocks one token row at a time, matching the token-by-token verifier
    while avoiding full sequential layer execution. ``route_locked`` row-steps
    only the router gate, then runs the fixed-index expert and shared-expert
    work batched.
    """
    raw = os.environ.get("EXO_NATIVE_MTP_MOE_VERIFY", "route_locked")
    normalized = raw.strip().lower().replace("-", "_")
    if normalized in {
        "safe",
        "hybrid_full_accept",
        "batched",
        "row_moe",
        "route_locked",
    }:
        return cast(_MoeVerifierPolicy, normalized)
    logger.warning(
        f"[native MTP] unknown EXO_NATIVE_MTP_MOE_VERIFY={raw!r}; using 'safe'"
    )
    return "safe"


def _env_flag(name: str, *, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _env_int(name: str, *, default: int, minimum: int) -> int:
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        value = int(raw)
    except ValueError:
        logger.warning(f"[native MTP] invalid {name}={raw!r}; using {default}")
        return default
    return max(minimum, value)


def _env_float(name: str, *, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        return float(raw)
    except ValueError:
        logger.warning(f"[native MTP] invalid {name}={raw!r}; using {default}")
        return default


_ASYNC_DRAFT_EVAL_ENABLED = _env_flag("EXO_NATIVE_MTP_ASYNC_DRAFT_EVAL", default=True)


def _cache_state_arrays(cache: Sequence[Any]) -> list[mx.array]:
    """Return materialisable cache state arrays without forcing evaluation."""
    state_arrays: list[mx.array] = []
    for entry in cache:
        state_obj: object = getattr(entry, "state", None)
        if isinstance(state_obj, mx.array):
            state_arrays.append(state_obj)
        elif isinstance(state_obj, (list, tuple)):
            for inner in cast(Sequence[object], state_obj):
                if isinstance(inner, mx.array):
                    state_arrays.append(inner)
    return state_arrays


def _materialize_cache_state(cache: Sequence[Any]) -> None:
    """Force ``mx.eval`` on every materialisable state in ``cache``.

    KVCache exposes ``.state`` as ``mx.array``; some custom caches use a
    tuple/list of arrays. We walk both shapes so the cache writes
    complete before the next forward consumes them.
    """
    state_arrays = _cache_state_arrays(cache)
    if state_arrays:
        mx.eval(state_arrays)


def _async_materialize_cache_state(
    cache: Sequence[Any],
    *arrays: mx.array,
) -> None:
    """Schedule draft/cache evaluation without waiting immediately.

    MLX evaluation is lazy. K-chain draft work can overlap with Python-side
    verifier graph construction as long as we schedule the draft tokens and
    MTP cache writes before building the target verify forward. The first
    required readback still synchronizes, but it has less work left to wait on.
    """
    state_arrays = _cache_state_arrays(cache)
    eval_arrays = [*arrays, *state_arrays]
    if eval_arrays:
        mx.async_eval(eval_arrays)


def _is_sparse_moe_block(mlp: object) -> bool:
    """Return True for Qwen3.5/Next sparse MoE blocks."""
    return (
        callable(getattr(mlp, "gate", None))
        and callable(getattr(mlp, "switch_mlp", None))
        and isinstance(getattr(mlp, "top_k", None), int)
    )


def _row_stepped_sparse_moe(mlp: object, mlp_input: mx.array) -> mx.array:
    """Evaluate a Qwen sparse MoE block one verify token at a time."""
    if not callable(mlp):
        raise TypeError(f"mlp object is not callable: {type(mlp).__name__}")
    return mx.concatenate(
        [mlp(mlp_input[:, i : i + 1, :]) for i in range(int(mlp_input.shape[1]))],
        axis=1,
    )


def _route_locked_sparse_moe(mlp: object, mlp_input: mx.array) -> mx.array:
    """Evaluate sparse MoE with exact row-stepped routing and batched experts."""
    gate = getattr(mlp, "gate", None)
    switch_mlp = getattr(mlp, "switch_mlp", None)
    shared_expert = getattr(mlp, "shared_expert", None)
    shared_expert_gate = getattr(mlp, "shared_expert_gate", None)
    top_k = getattr(mlp, "top_k", None)
    if (
        not callable(gate)
        or not callable(switch_mlp)
        or not callable(shared_expert)
        or not callable(shared_expert_gate)
        or not isinstance(top_k, int)
        or top_k <= 0
    ):
        raise TypeError(
            f"mlp object is not a supported sparse MoE: {type(mlp).__name__}"
        )

    sharding_group = getattr(mlp, "sharding_group", None)
    x = (
        sum_gradients(sharding_group)(mlp_input)
        if sharding_group is not None
        else mlp_input
    )

    # Qwen3.6-35B's BF16 router gate takes a different MLX kernel at M>1 than
    # at M=1. Route rows one-by-one to match token verification exactly, then
    # keep the expensive expert matmuls batched with fixed indices.
    gates = mx.concatenate(
        [gate(x[:, i : i + 1, :]) for i in range(int(x.shape[1]))],
        axis=1,
    )
    gates = mx.softmax(gates, axis=-1, precise=True)
    indices = mx.argpartition(gates, kth=-top_k, axis=-1)[..., -top_k:]
    scores = mx.take_along_axis(gates, indices, axis=-1)
    if bool(getattr(mlp, "norm_topk_prob", False)):
        scores = scores / scores.sum(axis=-1, keepdims=True)

    expert_output = cast(mx.array, switch_mlp(x, indices))
    output = (expert_output * scores[..., None]).sum(axis=-2)
    shared_gate_output = cast(mx.array, shared_expert_gate(x))
    shared_expert_output = cast(mx.array, shared_expert(x))
    shared_output = mx.sigmoid(shared_gate_output) * shared_expert_output
    output = output + shared_output

    if sharding_group is not None:
        output = mx.distributed.all_sum(output, group=sharding_group)
    return output


def _target_forward_row_moe(
    *,
    model: Model,
    inputs: mx.array,
    cache: KVCacheType,
) -> tuple[mx.array, mx.array]:
    """Target forward with batched non-MoE ops and row-stepped sparse MoE.

    ``scripts/native_mtp_verify_parity_probe.py`` shows qwen3_5_moe's first
    batched verifier mismatch occurs inside sparse MoE gate/evaluation, while
    GatedDeltaNet internals and full-attention cache updates stay exact for the
    K+1 verify window. This forward preserves those batched exact operations
    and only splits each sparse MoE call across token rows.
    """
    text_model = model.language_model
    inner = text_model.model
    hidden_states = inner.embed_tokens(inputs)
    cache_list = cast("list[Any]", cache)
    fa_mask = create_attention_mask(hidden_states, cache_list[inner.fa_idx])
    ssm_mask = create_ssm_mask(hidden_states, cache_list[inner.ssm_idx])

    for layer, layer_cache in zip(inner.layers, cache_list, strict=True):
        selected_mask = ssm_mask if bool(layer.is_linear) else fa_mask
        normalized = layer.input_layernorm(hidden_states)
        if bool(layer.is_linear):
            residual = layer.linear_attn(
                normalized,
                mask=selected_mask,
                cache=layer_cache,
            )
        else:
            residual = layer.self_attn(
                normalized,
                mask=selected_mask,
                cache=layer_cache,
            )
        mid = hidden_states + residual
        mlp_input = layer.post_attention_layernorm(mid)
        mlp_output = (
            _row_stepped_sparse_moe(layer.mlp, mlp_input)
            if _is_sparse_moe_block(layer.mlp)
            else layer.mlp(mlp_input)
        )
        hidden_states = mid + mlp_output

    post = inner.norm(hidden_states)
    if text_model.args.tie_word_embeddings:
        logits = inner.embed_tokens.as_linear(post)
    else:
        logits = text_model.lm_head(post)
    return logits, post


def _target_forward_route_locked_moe(
    *,
    model: Model,
    inputs: mx.array,
    cache: KVCacheType,
) -> tuple[mx.array, mx.array]:
    """Target forward with row-locked MoE routing and batched expert work."""
    text_model = model.language_model
    inner = text_model.model
    hidden_states = inner.embed_tokens(inputs)
    cache_list = cast("list[Any]", cache)
    fa_mask = create_attention_mask(hidden_states, cache_list[inner.fa_idx])
    ssm_mask = create_ssm_mask(hidden_states, cache_list[inner.ssm_idx])

    for layer, layer_cache in zip(inner.layers, cache_list, strict=True):
        selected_mask = ssm_mask if bool(layer.is_linear) else fa_mask
        normalized = layer.input_layernorm(hidden_states)
        if bool(layer.is_linear):
            residual = layer.linear_attn(
                normalized,
                mask=selected_mask,
                cache=layer_cache,
            )
        else:
            residual = layer.self_attn(
                normalized,
                mask=selected_mask,
                cache=layer_cache,
            )
        mid = hidden_states + residual
        mlp_input = layer.post_attention_layernorm(mid)
        mlp_output = (
            _route_locked_sparse_moe(layer.mlp, mlp_input)
            if _is_sparse_moe_block(layer.mlp)
            else layer.mlp(mlp_input)
        )
        hidden_states = mid + mlp_output

    post = inner.norm(hidden_states)
    if text_model.args.tie_word_embeddings:
        logits = inner.embed_tokens.as_linear(post)
    else:
        logits = text_model.lm_head(post)
    return logits, post


GdnPrefixHistory = dict[int, list[Any]]


def _lm_head_from_hidden(model: Model, hidden: mx.array) -> mx.array:
    text_model = model.language_model
    inner = text_model.model
    if text_model.args.tie_word_embeddings:
        return inner.embed_tokens.as_linear(hidden)
    return text_model.lm_head(hidden)


def _target_post_norm_hidden(
    *,
    model: Model,
    inputs: mx.array,
    cache: KVCacheType,
) -> mx.array:
    """Run the target body and return post-norm hidden without ``lm_head``.

    Prompt-cache repair and MTP-cache priming need exact post-norm target
    hidden states, not vocabulary logits. The vendored Qwen inner model exposes
    that post-norm body output directly, so these internal paths can avoid a
    vocabulary projection whose output would be discarded.
    """
    text_model = getattr(model, "language_model", None)
    inner = getattr(text_model, "model", None)
    if callable(inner):
        result = inner(
            inputs,
            cache=cast("list[Any]", cache),
            return_hidden=True,
        )
        if isinstance(result, tuple):
            post_hidden = result[0]
            if isinstance(post_hidden, mx.array):
                return post_hidden
        if isinstance(result, mx.array):
            return result
    _logits, hidden = cast(
        tuple[mx.array, mx.array],
        model(inputs, cache=cache, return_hidden=True),
    )
    return hidden


def _entry_is_trimmable(entry: Any) -> bool:
    method = getattr(entry, "is_trimmable", None)
    if not callable(method):
        return False
    try:
        return bool(method())
    except Exception:
        return False


def _trim_trimmable_cache_entries(cache: Sequence[Any], n_tokens: int) -> None:
    if n_tokens <= 0:
        return
    for entry in cache:
        trim = getattr(entry, "trim", None)
        if _entry_is_trimmable(entry) and callable(trim):
            trim(n_tokens)


def _set_cache_entry_state_lazy(entry: Any, state: Any) -> None:
    replace_state = getattr(entry, "replace_state", None)
    if callable(replace_state):
        replace_state(state)
        return
    current = getattr(entry, "state", None)
    if (
        isinstance(current, list)
        and isinstance(state, list)
        and len(current) == len(state)
    ):
        current[:] = state
        return
    entry.state = state


def _make_gated_delta_state_history_kernel() -> Any:
    source = """
        auto n = thread_position_in_grid.z;
        auto b_idx = n / Hv;
        auto hv_idx = n % Hv;
        auto hk_idx = hv_idx / (Hv / Hk);
        constexpr int n_per_t = Dk / 32;

        auto q_ = q + b_idx * T * Hk * Dk + hk_idx * Dk;
        auto k_ = k + b_idx * T * Hk * Dk + hk_idx * Dk;
        auto v_ = v + b_idx * T * Hv * Dv + hv_idx * Dv;
        y += b_idx * T * Hv * Dv + hv_idx * Dv;

        auto dk_idx = thread_position_in_threadgroup.x;
        auto dv_idx = thread_position_in_grid.y;

        auto i_state = state_in + (n * Dv + dv_idx) * Dk;
        auto o_state = state_out + (n * Dv + dv_idx) * Dk;

        float state[n_per_t];
        for (int i = 0; i < n_per_t; ++i) {
          auto s_idx = n_per_t * dk_idx + i;
          state[i] = static_cast<float>(i_state[s_idx]);
        }

        auto a_ = a + b_idx * T * Hv;
        auto b_ = b + b_idx * T * Hv;

        for (int t = 0; t < T; ++t) {
          float a_val = static_cast<float>(a_[hv_idx]);
          float dt_val = static_cast<float>(dt_bias[hv_idx]);
          float x_g = a_val + dt_val;
          float sp = (x_g > 20.0f) ? x_g : log(1.0f + exp(x_g));
          float g_val = exp(-exp(static_cast<float>(A_log[hv_idx])) * sp);
          float beta_val = 1.0f / (1.0f + exp(-static_cast<float>(b_[hv_idx])));

          float kv_mem = 0.0f;
          for (int i = 0; i < n_per_t; ++i) {
            auto s_idx = n_per_t * dk_idx + i;
            state[i] = state[i] * g_val;
            kv_mem += state[i] * k_[s_idx];
          }
          kv_mem = simd_sum(kv_mem);

          auto delta = (v_[dv_idx] - kv_mem) * beta_val;

          float out = 0.0f;
          for (int i = 0; i < n_per_t; ++i) {
            auto s_idx = n_per_t * dk_idx + i;
            state[i] = state[i] + k_[s_idx] * delta;
            out += state[i] * q_[s_idx];
          }
          out = simd_sum(out);
          if (thread_index_in_simdgroup == 0) {
            y[dv_idx] = static_cast<InT>(out);
          }

          auto hist = state_history
            + (((b_idx * T + t) * Hv + hv_idx) * Dv + dv_idx) * Dk;
          for (int i = 0; i < n_per_t; ++i) {
            auto s_idx = n_per_t * dk_idx + i;
            hist[s_idx] = static_cast<StT>(state[i]);
          }

          q_ += Hk * Dk;
          k_ += Hk * Dk;
          v_ += Hv * Dv;
          y += Hv * Dv;
          a_ += Hv;
          b_ += Hv;
        }
        for (int i = 0; i < n_per_t; ++i) {
          auto s_idx = n_per_t * dk_idx + i;
          o_state[s_idx] = static_cast<StT>(state[i]);
        }
    """
    return mx.fast.metal_kernel(
        name="gated_delta_step_state_history",
        input_names=["q", "k", "v", "a", "b", "A_log", "dt_bias", "state_in", "T"],
        output_names=["y", "state_out", "state_history"],
        source=source,
    )


_gated_delta_state_history_kernel: Any | None = None


def _gated_delta_update_with_state_history(
    *,
    q: mx.array,
    k: mx.array,
    v: mx.array,
    a: mx.array,
    b: mx.array,
    a_log: mx.array,
    dt_bias: mx.array,
    state: mx.array,
) -> tuple[mx.array, mx.array, mx.array]:
    global _gated_delta_state_history_kernel
    kernel = _gated_delta_state_history_kernel
    if kernel is None:
        kernel = _make_gated_delta_state_history_kernel()
        _gated_delta_state_history_kernel = kernel
    typed_kernel = cast(
        Callable[..., tuple[mx.array, mx.array, mx.array]],
        kernel,
    )
    batch_size, verify_width, num_key_heads, key_dim = k.shape
    num_value_heads, value_dim = v.shape[2:]
    return typed_kernel(
        inputs=[q, k, v, a, b, a_log, dt_bias, state, verify_width],
        template=[
            ("InT", q.dtype),
            ("StT", state.dtype),
            ("Dk", key_dim),
            ("Dv", value_dim),
            ("Hk", num_key_heads),
            ("Hv", num_value_heads),
        ],
        grid=(32, value_dim, batch_size * num_value_heads),
        threadgroup=(32, 4, 1),
        output_shapes=[
            (batch_size, verify_width, num_value_heads, value_dim),
            state.shape,
            (batch_size, verify_width, num_value_heads, value_dim, key_dim),
        ],
        output_dtypes=[q.dtype, state.dtype, state.dtype],
    )


def _gdn_linear_attn_with_state_history(
    gdn: Any,
    inputs: mx.array,
    *,
    cache: Any,
) -> tuple[mx.array, list[Any]]:
    """Run one GatedDeltaNet layer and return cache states after each prefix."""
    batch_size, verify_width, _ = inputs.shape
    qkv = gdn.in_proj_qkv(inputs)
    z = gdn.in_proj_z(inputs).reshape(
        batch_size, verify_width, gdn.num_v_heads, gdn.head_v_dim
    )
    b = gdn.in_proj_b(inputs)
    a = gdn.in_proj_a(inputs)

    if cache is not None and cache[0] is not None:
        conv_state = cache[0]
    else:
        conv_state = mx.zeros(
            (batch_size, gdn.conv_kernel_size - 1, gdn.conv_dim),
            dtype=inputs.dtype,
        )

    conv_input = mx.concatenate([conv_state, qkv], axis=1)
    n_keep = gdn.conv_kernel_size - 1
    if cache is not None:
        cache[0] = mx.contiguous(conv_input[:, -n_keep:, :])
    conv_prefix_states = [
        mx.contiguous(conv_input[:, prefix_index + 1 : prefix_index + 1 + n_keep, :])
        for prefix_index in range(verify_width)
    ]

    conv_out = nn.silu(gdn.conv1d(conv_input))
    q, k_arr, v = [
        t.reshape(batch_size, verify_width, h, d)
        for t, h, d in zip(
            mx.split(conv_out, [gdn.key_dim, 2 * gdn.key_dim], -1),
            [gdn.num_k_heads, gdn.num_k_heads, gdn.num_v_heads],
            [gdn.head_k_dim, gdn.head_k_dim, gdn.head_v_dim],
            strict=True,
        )
    ]

    state = cache[1] if cache else None
    if state is None:
        state = mx.zeros(
            (batch_size, gdn.num_v_heads, gdn.head_v_dim, gdn.head_k_dim),
            dtype=mx.float32,
        )
    inv_scale = k_arr.shape[-1] ** -0.5
    q = inv_scale * q * mx.rsqrt((q * q).sum(axis=-1, keepdims=True) + 1e-6)
    k_arr = k_arr * mx.rsqrt((k_arr * k_arr).sum(axis=-1, keepdims=True) + 1e-6)

    out, state, state_history = _gated_delta_update_with_state_history(
        q=q,
        k=k_arr,
        v=v,
        a=a,
        b=b,
        a_log=gdn.A_log,
        dt_bias=gdn.dt_bias,
        state=state,
    )

    if cache is not None:
        cache[1] = state
        cache.advance(verify_width)

    out = gdn.norm(out, z)
    out = gdn.out_proj(out.reshape(batch_size, verify_width, -1))
    prefix_states = [
        [conv_prefix_states[prefix_index], state_history[:, prefix_index, ...]]
        for prefix_index in range(verify_width)
    ]
    return out, prefix_states


def _target_forward_with_gdn_state_history(
    *,
    model: Model,
    inputs: mx.array,
    cache: KVCacheType,
    policy: _MoeVerifierPolicy,
) -> tuple[mx.array, mx.array, GdnPrefixHistory]:
    """Target forward with normal-width GDN state-history outputs."""
    text_model = model.language_model
    inner = text_model.model
    hidden_states = inner.embed_tokens(inputs)
    cache_list = cast("list[Any]", cache)
    fa_mask = create_attention_mask(hidden_states, cache_list[inner.fa_idx])
    ssm_mask = create_ssm_mask(hidden_states, cache_list[inner.ssm_idx])
    if isinstance(ssm_mask, mx.array):
        raise ValueError("state-history GDN verifier does not support SSM masks")
    prefix_history: GdnPrefixHistory = {}

    for layer_index, (layer, layer_cache) in enumerate(
        zip(inner.layers, cache_list, strict=True)
    ):
        selected_mask = ssm_mask if bool(layer.is_linear) else fa_mask
        normalized = layer.input_layernorm(hidden_states)
        if bool(layer.is_linear):
            if selected_mask is not None:
                raise ValueError(
                    "state-history GDN verifier requires unmasked decode cache"
                )
            residual, layer_history = _gdn_linear_attn_with_state_history(
                layer.linear_attn,
                normalized,
                cache=layer_cache,
            )
            prefix_history[layer_index] = layer_history
        else:
            residual = layer.self_attn(
                normalized,
                mask=selected_mask,
                cache=layer_cache,
            )
        mid = hidden_states + residual
        mlp_input = layer.post_attention_layernorm(mid)
        if _is_sparse_moe_block(layer.mlp):
            if policy == "route_locked":
                mlp_output = _route_locked_sparse_moe(layer.mlp, mlp_input)
            elif policy == "row_moe":
                mlp_output = _row_stepped_sparse_moe(layer.mlp, mlp_input)
            else:
                raise ValueError(
                    f"unsupported state-history MoE verifier policy: {policy}"
                )
        else:
            mlp_output = layer.mlp(mlp_input)
        hidden_states = mid + mlp_output

    post = inner.norm(hidden_states)
    logits = _lm_head_from_hidden(model, post)
    return logits, post, prefix_history


def _commit_gdn_prefix_state(
    *,
    cache: KVCacheType,
    prefix_history: GdnPrefixHistory,
    accept_count: int,
    k: int,
) -> None:
    """Commit verifier cache to ``[current, drafts[:accept_count]]``."""
    cache_list = cast("list[Any]", cache)
    _trim_trimmable_cache_entries(cache_list, k - accept_count)
    for cache_index, states_by_prefix in prefix_history.items():
        _set_cache_entry_state_lazy(
            cache_list[cache_index],
            states_by_prefix[accept_count],
        )


def prime_mtp_cache_from_prompt(
    *,
    model: Model,
    full_prompt_tokens: Sequence[int],
    mtp_cache: list[Any],
) -> int:
    """Walk prefill positions through the main model to prime the MTP cache.

    Forwards ``prompt[:-1]`` through a fresh main-model cache, captures
    post-norm hidden states for every prefill position, then feeds
    them through ``model.mtp_update_cache`` in a single batched call.
    After priming, the MTP attention cache holds K/V for positions
    ``0..N-2``; the first ``mtp_forward`` in the K-chain draft loop
    adds position ``N-1`` naturally.

    Returns the number of positions primed (``N-1``), or ``0`` when
    the prompt is shorter than 2 tokens.

    Known cost: this re-walks the prompt through a fresh main-model cache
    to prime the separate MTP cache, so the prompt is prefilled twice (once
    by ``exo.prefill`` for the main KV cache, once here for the MTP cache).
    Folding the two into a single capture-prefill is a future optimisation
    at the :func:`exo.worker.engines.mlx.prefill` seam, not in this drafter.
    """
    n = len(full_prompt_tokens)
    if n < 2:
        return 0
    prefill_arr = mx.array(list(full_prompt_tokens[:-1]), dtype=mx.int32)
    next_arr = mx.array(list(full_prompt_tokens[1:]), dtype=mx.int32)
    fresh_cache: list[Any] = cast("list[Any]", model.make_cache())
    prefill_hidden = _target_post_norm_hidden(
        model=model,
        inputs=prefill_arr[None],
        cache=fresh_cache,
    )
    mx.eval(prefill_hidden)
    model.mtp_update_cache(prefill_hidden, next_arr[None], mtp_cache=mtp_cache)
    _materialize_cache_state(mtp_cache)
    del fresh_cache
    return n - 1


def prime_mtp_cache_from_prompt_incremental(
    *,
    model: Model,
    full_prompt_tokens: Sequence[int],
    mtp_cache: list[Any],
) -> int:
    """Prime MTP cache by stepping target hiddens token by token.

    This is the correctness path for qwen3_5_moe. Its recurrent
    GatedDeltaNet state is sensitive to batched prefill; the batched
    :func:`prime_mtp_cache_from_prompt` can produce hiddens that are close
    enough for ordinary logits but wrong enough for MTP acceptance.
    """
    n = len(full_prompt_tokens)
    if n < 2:
        return 0
    fresh_cache: list[Any] = cast("list[Any]", model.make_cache())
    for i in range(n - 1):
        token_arr = mx.array([[int(full_prompt_tokens[i])]], dtype=mx.int32)
        next_arr = mx.array([[int(full_prompt_tokens[i + 1])]], dtype=mx.int32)
        hidden = _target_post_norm_hidden(
            model=model,
            inputs=token_arr,
            cache=fresh_cache,
        )
        mtp_hidden = model.mtp_update_cache(
            hidden[:, -1:, :],
            next_arr,
            mtp_cache=mtp_cache,
        )
        mx.eval(hidden, mtp_hidden)
    _materialize_cache_state(mtp_cache)
    del fresh_cache
    return n - 1


def rebuild_prompt_cache_and_prime_mtp_cache_incremental(
    *,
    model: Model,
    full_prompt_tokens: Sequence[int],
    prompt_cache: KVCacheType,
    mtp_cache: list[Any],
) -> tuple[int, int]:
    """Rebuild target prompt cache while priming MTP cache in one pass."""
    n = len(full_prompt_tokens)
    cache_list = cast("list[Any]", prompt_cache)
    if n < 2:
        cache_list[:] = cast("list[Any]", model.make_cache())
        return 0, 0

    tokens_to_prefill = max(0, n - 2)
    fresh_cache: list[Any] = cast("list[Any]", model.make_cache())
    prompt_boundary_snapshot: Any | None = None
    for i in range(n - 1):
        token_arr = mx.array([[int(full_prompt_tokens[i])]], dtype=mx.int32)
        next_arr = mx.array([[int(full_prompt_tokens[i + 1])]], dtype=mx.int32)
        hidden = _target_post_norm_hidden(
            model=model,
            inputs=token_arr,
            cache=fresh_cache,
        )
        mtp_hidden = model.mtp_update_cache(
            hidden[:, -1:, :],
            next_arr,
            mtp_cache=mtp_cache,
        )
        mx.eval(hidden, mtp_hidden)
        if i + 1 == tokens_to_prefill:
            prompt_boundary_snapshot = snapshot_untrimmable_cache_lazy(fresh_cache)

    if tokens_to_prefill == 0:
        cache_list[:] = cast("list[Any]", model.make_cache())
    else:
        _trim_trimmable_cache_entries(fresh_cache, (n - 1) - tokens_to_prefill)
        if prompt_boundary_snapshot is None:
            raise RuntimeError("missing prompt boundary snapshot")
        restore_cache(fresh_cache, prompt_boundary_snapshot)
        cache_list[:] = fresh_cache
    _materialize_cache_state(mtp_cache)
    return tokens_to_prefill, n - 1


def rebuild_prompt_cache_incremental(
    *,
    model: Model,
    full_prompt_tokens: Sequence[int],
    prompt_cache: KVCacheType,
) -> int:
    """Replace ``prompt_cache`` with token-by-token state through prompt[:-2].

    Exo's standard prefill uses mlx-lm's chunked prefill and then trims the
    decode tail. That is fast, but qwen3_5_moe's GatedDeltaNet recurrent
    state is not equivalent to a token-by-token cache under that path. Native
    MTP needs the exact recurrent hidden/cache state, so the 35B MoE safe path
    rebuilds it incrementally before the two-token decode tail is consumed.
    """
    fresh_cache: list[Any] = cast("list[Any]", model.make_cache())
    tokens_to_prefill = max(0, len(full_prompt_tokens) - 2)
    for token in full_prompt_tokens[:tokens_to_prefill]:
        token_arr = mx.array([[int(token)]], dtype=mx.int32)
        hidden = _target_post_norm_hidden(
            model=model,
            inputs=token_arr,
            cache=fresh_cache,
        )
        mx.eval(hidden)
    cache_list = cast("list[Any]", prompt_cache)
    cache_list[:] = fresh_cache
    return tokens_to_prefill


@final
class NativeMTPDrafter:
    """Drafter-protocol shim for native MTP (vendored Qwen3.5/3.6).

    Constructor takes ``k`` explicitly; the builder sources it from
    :attr:`exo.shared.models.model_cards.NativeMTPConfig.default_k`
    (overridable per-request via ``TaskParams.num_draft_tokens``).

    The drafter reports ``mode="model"`` so existing telemetry counts
    it as drafter-active. The architecture-level distinction
    (``native_mtp`` vs sibling-LM ``model``) is surfaced via the
    builder log line at dispatch time; a future :data:`DraftMode`
    extension can expose it via ``GenerationStats`` if needed.
    """

    def __init__(self, *, k: int) -> None:
        if k < 1:
            raise ValueError(f"k must be >= 1, got {k}")
        self._k: int = k
        self._metrics: dict[str, int] = {
            "proposed_draft_tokens": 0,
            "accepted_draft_tokens": 0,
            "spec_decode_rounds": 0,
        }

    @property
    def mode(self) -> str:
        return "model"

    @property
    def num_draft_tokens(self) -> int:
        return self._k

    def metrics(self) -> dict[str, int]:
        return dict(self._metrics)

    def stream(
        self,
        *,
        model: Model,
        tokenizer: TokenizerWrapper,
        prompt: mx.array,
        context_tokens: Sequence[int],
        prompt_cache: KVCacheType,
        max_tokens: int,
        sampler: Callable[[mx.array], mx.array],
        logits_processors: Sequence[Callable[[mx.array, mx.array], mx.array]],
        prefill_step_size: int = 1,
        temperature: float = 0.0,
        top_p: float = 0.0,
        top_k: int = 0,
        min_p: float = 0.0,
        min_tokens_to_keep: int = 1,
    ) -> Generator[GenerationResponse, None, None]:
        # LOSSLESS / distribution-preserving native MTP. The K-chain drafts
        # are SAMPLED from the MTP head's per-step distribution ``q`` (after
        # the request's temperature/top_p/top_k/min_p transforms) and the
        # batched/sequential verify accepts each draft via probability-ratio
        # (Leviathan/Chen) rejection sampling against the target distribution
        # ``p`` -- so the emitted marginal matches the request's sampler at any
        # temperature. With ``temperature == 0`` the scheme collapses to exact
        # greedy (accept iff draft == argmax(p)), token-identical to plain
        # greedy decode. See :class:`_SpecSampler` for the reconstruction of
        # ``p``/``q`` from the raw sampling params (the request only hands us a
        # ``sampler`` callable, which is insufficient -- we need the
        # distribution, so the params are threaded in from ``mlx_generate``).
        #
        # ``logits_processors`` (repetition/presence/frequency penalties,
        # logit bias) remain UNHANDLED on this path: they are stateful over
        # emitted-token history and cannot be replayed consistently across the
        # chained draft steps and the batched verify without breaking the
        # lossless guarantee; applying them inconsistently is worse than not at
        # all. ``sampler`` is likewise unused -- the params drive the
        # reconstructed distribution directly. ``prefill_step_size`` is unused
        # (MTP-cache priming is batched).
        del sampler, logits_processors, prefill_step_size
        spec_sampler = _SpecSampler(
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            min_p=min_p,
            min_tokens_to_keep=min_tokens_to_keep,
        )
        if not is_native_mtp_dispatchable(model):
            raise RuntimeError(
                "NativeMTPDrafter.stream called on a model without "
                "mtp_forward / make_mtp_cache / mtp_update_cache. The "
                "dispatch-site gate (is_native_mtp_dispatchable) should "
                "have caught this."
            )

        detokenizer = tokenizer.detokenizer
        detokenizer.reset()
        eos_ids = _eos_ids_from_tokenizer(tokenizer)

        prompt_tail_size = int(prompt.size)

        mtp_cache: list[Any] = cast("list[Any]", model.make_mtp_cache())
        if not mtp_cache:
            raise RuntimeError(
                "model.make_mtp_cache() returned empty list -- model "
                "has no MTP head. Loader/dispatch drift."
            )

        model_type = _native_mtp_model_type(model)
        sequential_gdn_path = _requires_sequential_gdn_path(model)
        moe_verifier_policy = _moe_verifier_policy() if sequential_gdn_path else "safe"
        if sequential_gdn_path and moe_verifier_policy != "safe":
            logger.info(
                f"[native MTP] qwen3_5_moe verifier policy={moe_verifier_policy!r}"
            )
        gdn_state_history_default = _gdn_state_history_commit_default(
            model_type=model_type,
            moe_verifier_policy=moe_verifier_policy,
        )
        gdn_state_history_supported = (
            _gdn_state_history_kernel_supported(model)
            if gdn_state_history_default
            else False
        )
        gdn_state_history_commit = (
            gdn_state_history_default
            and gdn_state_history_supported
            # Default-on for dense qwen3_5 and for qwen3_5_moe route-locked
            # native MTP after broad production stream parity. Set the env var
            # to 0 as a kill switch.
            and _env_flag("EXO_NATIVE_MTP_GDN_STATE_HISTORY_COMMIT", default=True)
        )
        if gdn_state_history_default and not gdn_state_history_supported:
            logger.warning(
                f"[native MTP] {model_type or 'unknown model'} GDN "
                "state-history prefix commit disabled: unsupported GDN head shape"
            )
        if gdn_state_history_commit:
            logger.info(
                f"[native MTP] {model_type or 'unknown model'} GDN "
                "state-history prefix commit enabled"
            )
        row_moe_adaptive = (
            sequential_gdn_path
            and moe_verifier_policy == "row_moe"
            and _env_flag("EXO_NATIVE_MTP_ROW_MOE_ADAPTIVE", default=True)
        )
        row_moe_adaptive_min_rounds = _env_int(
            "EXO_NATIVE_MTP_ROW_MOE_ADAPTIVE_MIN_ROUNDS",
            default=8,
            minimum=1,
        )
        row_moe_adaptive_min_acceptance = _env_float(
            "EXO_NATIVE_MTP_ROW_MOE_ADAPTIVE_MIN_ACCEPTANCE",
            default=0.55,
        )
        row_moe_adaptive_switched = False
        primed_positions = 0
        if context_tokens:
            prime_start = time.perf_counter()
            if sequential_gdn_path:
                repaired_positions, primed_positions = (
                    rebuild_prompt_cache_and_prime_mtp_cache_incremental(
                        model=model,
                        full_prompt_tokens=context_tokens,
                        prompt_cache=prompt_cache,
                        mtp_cache=mtp_cache,
                    )
                )
                prime_elapsed_ms = (time.perf_counter() - prime_start) * 1000
                logger.info(
                    "[native MTP] qwen3_5_moe safe path rebuilt target cache "
                    f"from {repaired_positions} prompt positions and primed "
                    f"MTP cache from {primed_positions} prompt positions in "
                    f"{prime_elapsed_ms:.1f}ms"
                )
            else:
                primed_positions = prime_mtp_cache_from_prompt(
                    model=model,
                    full_prompt_tokens=context_tokens,
                    mtp_cache=mtp_cache,
                )
                prime_elapsed_ms = (time.perf_counter() - prime_start) * 1000
                logger.info(
                    f"[native MTP] primed MTP cache from {primed_positions} "
                    f"prompt positions in {prime_elapsed_ms:.1f}ms"
                )

        # First main-model forward: decode_in covers the 2-token prefill
        # tail. The caller has aligned ``prompt_cache`` to
        # ``full_prompt[:-2]`` via exo.prefill + trim(2); this brings
        # the cache to ``len-1`` via position N-2 and returns logits
        # at position N-1 (which we sample for the first emitted token).
        decode_in = prompt.astype(mx.int32)[None]
        if int(decode_in.shape[1]) > 1:
            # Qwen3.5/3.6 GatedDeltaNet can produce different last-position
            # logits for a small batched tail than for the same tokens stepped
            # through the recurrent state. Mirror mlx-lm's prefill/decode split:
            # advance every tail token except the last, then sample from the last.
            tail_prefix_hidden = _target_post_norm_hidden(
                model=model,
                inputs=decode_in[:, :-1],
                cache=prompt_cache,
            )
            first_logits, first_hidden = cast(
                tuple[mx.array, mx.array],
                model(decode_in[:, -1:], cache=prompt_cache, return_hidden=True),
            )
            mx.eval(tail_prefix_hidden, first_logits, first_hidden)
        else:
            first_logits, first_hidden = cast(
                tuple[mx.array, mx.array],
                model(decode_in, cache=prompt_cache, return_hidden=True),
            )
            mx.eval(first_logits, first_hidden)
        # Sample the first emitted token from the request's distribution
        # (argmax when greedy) instead of a hard argmax.
        current_token_arr = spec_sampler.sample_from_distribution(
            spec_sampler.distribution(first_logits[:, -1, :])
        )
        mx.eval(current_token_arr)
        current_token = _token_array_to_int(current_token_arr)
        current_hidden = first_hidden[:, -1:, :]

        proposed = 0
        accepted = 0
        rounds = 0
        ntoks = 0
        finish_reason: str | None = None
        prompt_tps_local = 0.0

        tic = time.perf_counter()
        if current_token in eos_ids:
            detokenizer.add_token(current_token)
            detokenizer.finalize()
            yield GenerationResponse(
                text=detokenizer.last_segment,
                token=current_token,
                logprobs=mx.zeros((1,)),
                from_draft=False,
                prompt_tokens=prompt_tail_size,
                prompt_tps=0.0,
                generation_tokens=1,
                generation_tps=0.0,
                peak_memory=mx.get_peak_memory() / 1e9,
                finish_reason="stop",
            )
            return

        prompt_time = time.perf_counter() - tic
        prompt_tps_local = prompt_tail_size / prompt_time if prompt_time > 0 else 0.0
        tic = time.perf_counter()
        ntoks = 1
        detokenizer.add_token(current_token)
        elapsed = time.perf_counter() - tic
        yield GenerationResponse(
            text=detokenizer.last_segment,
            token=current_token,
            logprobs=mx.zeros((1,)),
            from_draft=False,
            prompt_tokens=prompt_tail_size,
            prompt_tps=prompt_tps_local,
            generation_tokens=ntoks,
            generation_tps=ntoks / elapsed if elapsed > 0 else 0.0,
            peak_memory=mx.get_peak_memory() / 1e9,
            finish_reason=None,
        )

        # Main loop: cache offset = N, current_{token, hidden} from
        # most recent emission.
        while ntoks < max_tokens:
            # ---- Draft K via chained mtp_forward ----
            # Each draft token is SAMPLED from its MTP-step distribution ``q``
            # (argmax when greedy); ``q`` is retained per step so the verify
            # step can run the probability-ratio acceptance test.
            draft_token_arrays: list[mx.array] = []
            draft_distributions: list[mx.array] = []
            chain_hidden = current_hidden
            chain_tok_arr = current_token_arr
            for _ in range(self._k):
                mtp_logits, mtp_hidden = cast(
                    tuple[mx.array, mx.array],
                    model.mtp_forward(
                        chain_hidden,
                        chain_tok_arr,
                        mtp_cache=mtp_cache,
                        return_hidden=True,
                    ),
                )
                draft_dist = spec_sampler.distribution(mtp_logits[:, -1, :])
                next_draft_arr = spec_sampler.sample_from_distribution(draft_dist)
                draft_token_arrays.append(next_draft_arr)
                draft_distributions.append(draft_dist)
                chain_hidden = mtp_hidden[:, -1:, :]
                chain_tok_arr = next_draft_arr
            # Consolidated eval: single GPU dispatch for all draft arrays
            all_drafts = (
                draft_token_arrays[0]
                if self._k == 1
                else mx.concatenate(draft_token_arrays, axis=1)
            )
            if _ASYNC_DRAFT_EVAL_ENABLED:
                _async_materialize_cache_state(
                    mtp_cache,
                    all_drafts,
                    *draft_distributions,
                )
            else:
                mx.eval(all_drafts, *draft_distributions)
                _materialize_cache_state(mtp_cache)
            proposed += self._k
            rounds += 1

            # Pre-draw the per-position uniforms for the ratio test up-front so
            # the acceptance walk needs no extra GPU sync per token.
            accept_uniforms: list[float] = (
                [0.0] * self._k
                if spec_sampler.is_greedy
                else cast(
                    "list[float]",
                    mx.random.uniform(shape=(self._k,)).tolist(),
                )
            )

            # ---- Verify: target forward on [current_token, *drafts] ----
            # Consolidated readback: single CPU-GPU sync for all K draft tokens
            # instead of K sequential .item() calls (~50-150us each).
            draft_tokens: list[int] = [
                int(t) for t in list(all_drafts.reshape(-1).tolist())
            ]
            num_accepted_this_round = 0
            bonus_tok = current_token
            next_current_hidden = current_hidden
            next_current_token_arr = current_token_arr
            pre_verify_snapshot: Any | None = None
            prefix_history: GdnPrefixHistory | None = None
            verify_hidden: mx.array | None = None

            if sequential_gdn_path:
                used_batched_moe_verify = False
                if moe_verifier_policy != "safe":
                    verify_in = mx.concatenate(
                        [current_token_arr, *draft_token_arrays], axis=1
                    )
                    if gdn_state_history_commit:
                        (
                            verify_logits,
                            verify_hidden,
                            prefix_history,
                        ) = _target_forward_with_gdn_state_history(
                            model=model,
                            inputs=verify_in,
                            cache=prompt_cache,
                            policy=moe_verifier_policy,
                        )
                    else:
                        pre_verify_snapshot = snapshot_untrimmable_cache_lazy(
                            cast("list[Any]", prompt_cache)
                        )
                        verify_logits, verify_hidden = (
                            _target_forward_row_moe(
                                model=model,
                                inputs=verify_in,
                                cache=prompt_cache,
                            )
                            if moe_verifier_policy == "row_moe"
                            else _target_forward_route_locked_moe(
                                model=model,
                                inputs=verify_in,
                                cache=prompt_cache,
                            )
                            if moe_verifier_policy == "route_locked"
                            else cast(
                                tuple[mx.array, mx.array],
                                model(
                                    verify_in,
                                    cache=prompt_cache,
                                    return_hidden=True,
                                ),
                            )
                        )
                    mx.eval(verify_logits, verify_hidden)
                    batched_accepted, batched_replacement_arr = _spec_accept_walk(
                        spec_sampler=spec_sampler,
                        verify_logits=verify_logits,
                        draft_tokens=draft_tokens,
                        draft_distributions=draft_distributions,
                        accept_uniforms=accept_uniforms,
                        k=self._k,
                    )
                    mx.eval(batched_replacement_arr)

                    can_commit_batched = (
                        moe_verifier_policy in {"batched", "row_moe", "route_locked"}
                        or batched_accepted == self._k
                    )
                    if can_commit_batched:
                        num_accepted_this_round = batched_accepted
                        bonus_tok = _token_array_to_int(batched_replacement_arr)
                        next_current_token_arr = batched_replacement_arr
                        next_current_hidden = verify_hidden[
                            :,
                            num_accepted_this_round : num_accepted_this_round + 1,
                            :,
                        ]
                        used_batched_moe_verify = True
                    else:
                        assert pre_verify_snapshot is not None
                        rollback_after_verify(
                            cast("list[Any]", prompt_cache),
                            pre_verify_snapshot,
                            verified_tokens=self._k + 1,
                        )

                if not used_batched_moe_verify:
                    # qwen3_5_moe safe verifier: feed only the retained prefix
                    # token-by-token. Batched verification corrupts later-
                    # position GatedDeltaNet state/logits on 35B-A3B, producing
                    # coherent-looking but wrong bonuses such as "if items"
                    # instead of "if len". Each position runs the same
                    # probability-ratio acceptance test as the batched paths.
                    for i in range(self._k + 1):
                        verify_token_arr = (
                            current_token_arr if i == 0 else draft_token_arrays[i - 1]
                        )
                        verify_logits_i, verify_hidden_i = cast(
                            tuple[mx.array, mx.array],
                            model(
                                verify_token_arr,
                                cache=prompt_cache,
                                return_hidden=True,
                            ),
                        )
                        target_dist_i = spec_sampler.distribution(
                            verify_logits_i[:, -1, :]
                        )
                        mx.eval(target_dist_i, verify_hidden_i)
                        next_current_hidden = verify_hidden_i[:, -1:, :]
                        if i < self._k:
                            draft_token = draft_tokens[i]
                            target_argmax = int(
                                mx.argmax(verify_logits_i[:, -1, :], axis=-1).item()
                            )
                            target_prob = float(target_dist_i[0, draft_token].item())
                            draft_prob = float(
                                draft_distributions[i][0, draft_token].item()
                            )
                            if _accept_draft_token(
                                is_greedy=spec_sampler.is_greedy,
                                draft_token=draft_token,
                                target_argmax=target_argmax,
                                target_prob=target_prob,
                                draft_prob=draft_prob,
                                uniform=accept_uniforms[i],
                            ):
                                num_accepted_this_round += 1
                                continue
                            # Rejected: replacement from residual (p_i - q_i)+.
                            replacement_arr = spec_sampler.residual_token(
                                target_dist_i, draft_distributions[i]
                            )
                        else:
                            # Full accept: bonus sampled from the target dist.
                            replacement_arr = spec_sampler.sample_from_distribution(
                                target_dist_i
                            )
                        mx.eval(replacement_arr)
                        bonus_tok = _token_array_to_int(replacement_arr)
                        next_current_token_arr = replacement_arr
                        break
                elif num_accepted_this_round < self._k:
                    if gdn_state_history_commit:
                        assert prefix_history is not None
                        _commit_gdn_prefix_state(
                            cache=prompt_cache,
                            prefix_history=prefix_history,
                            accept_count=num_accepted_this_round,
                            k=self._k,
                        )
                    else:
                        assert pre_verify_snapshot is not None
                        rollback_after_verify(
                            cast("list[Any]", prompt_cache),
                            pre_verify_snapshot,
                            verified_tokens=self._k + 1,
                        )
                        repair_in = mx.concatenate(
                            [
                                current_token_arr,
                                *draft_token_arrays[:num_accepted_this_round],
                            ],
                            axis=1,
                        )
                        _repair_logits, _repair_hidden = (
                            _target_forward_row_moe(
                                model=model,
                                inputs=repair_in,
                                cache=prompt_cache,
                            )
                            if moe_verifier_policy == "row_moe"
                            else _target_forward_route_locked_moe(
                                model=model,
                                inputs=repair_in,
                                cache=prompt_cache,
                            )
                            if moe_verifier_policy == "route_locked"
                            else cast(
                                tuple[mx.array, mx.array],
                                model(
                                    repair_in,
                                    cache=prompt_cache,
                                    return_hidden=True,
                                ),
                            )
                        )
                        mx.eval(_repair_logits, _repair_hidden)
                        next_current_hidden = _repair_hidden[:, -1:, :]
                accepted += num_accepted_this_round
                if (
                    row_moe_adaptive
                    and not row_moe_adaptive_switched
                    and rounds >= row_moe_adaptive_min_rounds
                    and proposed > 0
                    and (accepted / proposed) < row_moe_adaptive_min_acceptance
                ):
                    logger.info(
                        "[native MTP] row_moe acceptance "
                        f"{accepted}/{proposed}={accepted / proposed:.1%} after "
                        f"{rounds} rounds; switching verifier to safe for this "
                        "generation"
                    )
                    moe_verifier_policy = "safe"
                    row_moe_adaptive_switched = True
            else:
                verify_in = mx.concatenate(
                    [current_token_arr, *draft_token_arrays], axis=1
                )
                if gdn_state_history_commit:
                    (
                        verify_logits,
                        verify_hidden,
                        prefix_history,
                    ) = _target_forward_with_gdn_state_history(
                        model=model,
                        inputs=verify_in,
                        cache=prompt_cache,
                        policy=moe_verifier_policy,
                    )
                else:
                    # Snapshot GDN state BEFORE verify advances cache by K+1 so
                    # we can roll back on partial accept; see module docstring.
                    pre_verify_snapshot = snapshot_untrimmable_cache_lazy(
                        cast("list[Any]", prompt_cache)
                    )
                    verify_logits, verify_hidden = cast(
                        tuple[mx.array, mx.array],
                        model(verify_in, cache=prompt_cache, return_hidden=True),
                    )
                mx.eval(verify_logits, verify_hidden)

                # Probability-ratio accept-walk (lossless speculative sampling).
                num_accepted_this_round, replacement_arr = _spec_accept_walk(
                    spec_sampler=spec_sampler,
                    verify_logits=verify_logits,
                    draft_tokens=draft_tokens,
                    draft_distributions=draft_distributions,
                    accept_uniforms=accept_uniforms,
                    k=self._k,
                )
                mx.eval(replacement_arr)
                accepted += num_accepted_this_round
                bonus_tok = _token_array_to_int(replacement_arr)
                next_current_token_arr = replacement_arr
                next_current_hidden = verify_hidden[
                    :, num_accepted_this_round : num_accepted_this_round + 1, :
                ]

            # Emit accepted drafts.
            for i in range(num_accepted_this_round):
                tok = draft_tokens[i]
                ntoks += 1
                detokenizer.add_token(tok)
                elapsed = time.perf_counter() - tic
                yield GenerationResponse(
                    text=detokenizer.last_segment,
                    token=tok,
                    logprobs=mx.zeros((1,)),
                    from_draft=True,
                    prompt_tokens=prompt_tail_size,
                    prompt_tps=prompt_tps_local,
                    generation_tokens=ntoks,
                    generation_tps=ntoks / elapsed if elapsed > 0 else 0.0,
                    peak_memory=mx.get_peak_memory() / 1e9,
                    finish_reason=None,
                )
                if tok in eos_ids:
                    finish_reason = "stop"
                    break
                if ntoks >= max_tokens:
                    finish_reason = "length"
                    break

            # Emit bonus.
            if finish_reason is None:
                ntoks += 1
                detokenizer.add_token(bonus_tok)
                elapsed = time.perf_counter() - tic
                yield GenerationResponse(
                    text=detokenizer.last_segment,
                    token=bonus_tok,
                    logprobs=mx.zeros((1,)),
                    from_draft=False,
                    prompt_tokens=prompt_tail_size,
                    prompt_tps=prompt_tps_local,
                    generation_tokens=ntoks,
                    generation_tps=ntoks / elapsed if elapsed > 0 else 0.0,
                    peak_memory=mx.get_peak_memory() / 1e9,
                    finish_reason=None,
                )
                if bonus_tok in eos_ids:
                    finish_reason = "stop"

            # ---- Cache trims ----
            # Main cache: on partial accept (n < K) we trim back to the
            # pre-verify offset and restore GDN from the snapshot, then
            # re-forward [current_token, *drafts[:n]] to land cache at
            # N + n + 1 with the GDN recurrent state advanced through
            # the retained tokens only. On full accept (n == K) the
            # batched verify's cache state at N+K+1 is already on-
            # trajectory (verified by batched_vs_solo_probe in the
            # research lane), so no rollback is needed.
            if not sequential_gdn_path and num_accepted_this_round < self._k:
                try:
                    if gdn_state_history_commit:
                        assert prefix_history is not None
                        _commit_gdn_prefix_state(
                            cache=prompt_cache,
                            prefix_history=prefix_history,
                            accept_count=num_accepted_this_round,
                            k=self._k,
                        )
                    else:
                        assert pre_verify_snapshot is not None
                        rollback_after_verify(
                            cast("list[Any]", prompt_cache),
                            pre_verify_snapshot,
                            verified_tokens=self._k + 1,
                        )
                        repair_in = mx.concatenate(
                            [
                                current_token_arr,
                                *draft_token_arrays[:num_accepted_this_round],
                            ],
                            axis=1,
                        )
                        _repair_logits, _repair_hidden = cast(
                            tuple[mx.array, mx.array],
                            model(repair_in, cache=prompt_cache, return_hidden=True),
                        )
                        mx.eval(_repair_logits, _repair_hidden)
                        next_current_hidden = _repair_hidden[:, -1:, :]
                except Exception as exc:
                    logger.warning(
                        f"[native MTP] cache repair after partial accept "
                        f"(n={num_accepted_this_round}/K={self._k}) raised "
                        f"{type(exc).__name__}: {exc}; continuing with stale cache"
                    )
            # MTP cache: chain drafting added K positions; only the
            # accepted prefix's K/V corresponds to tokens that materialised
            # in the main stream. The MTP attention layer is full-attention
            # (qwen3_5_mtp.py uses ``fa_layer_idx``) so the stock trim is
            # correct here -- no rollback asymmetry to worry about.
            mtp_retained = min(self._k, num_accepted_this_round + 1)
            mtp_trim = self._k - mtp_retained
            if mtp_trim > 0:
                try:
                    mlx_trim_prompt_cache(cast(list[object], mtp_cache), mtp_trim)
                except Exception as exc:
                    logger.warning(
                        f"[native MTP] mtp cache trim({mtp_trim}) raised "
                        f"{type(exc).__name__}: {exc}; continuing with stale cache"
                    )

            if finish_reason is not None or ntoks >= max_tokens:
                if finish_reason is None and ntoks >= max_tokens:
                    finish_reason = "length"
                break

            current_token = bonus_tok
            current_token_arr = next_current_token_arr
            current_hidden = next_current_hidden

        detokenizer.finalize()
        if finish_reason is None:
            finish_reason = "length" if ntoks >= max_tokens else "stop"
        elapsed = time.perf_counter() - tic
        acceptance = (accepted / proposed) if proposed > 0 else 0.0
        self._metrics["proposed_draft_tokens"] = proposed
        self._metrics["accepted_draft_tokens"] = accepted
        self._metrics["spec_decode_rounds"] = rounds
        logger.info(
            f"[native MTP] K={self._k} stream done: tokens={ntoks}, "
            f"proposed={proposed}, accepted={accepted}, "
            f"acceptance={acceptance:.1%}, rounds={rounds}, "
            f"finish={finish_reason}, primed_positions={primed_positions}"
        )
        yield GenerationResponse(
            text=detokenizer.last_segment,
            token=current_token,
            logprobs=mx.zeros((1,)),
            from_draft=False,
            prompt_tokens=prompt_tail_size,
            prompt_tps=prompt_tps_local,
            generation_tokens=ntoks,
            generation_tps=ntoks / elapsed if elapsed > 0 else 0.0,
            peak_memory=mx.get_peak_memory() / 1e9,
            finish_reason=finish_reason,
        )


__all__ = [
    "NativeMTPDrafter",
    "is_native_mtp_dispatchable",
    "prime_mtp_cache_from_prompt",
]
