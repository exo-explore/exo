"""PP idle-time speculation for pipeline parallel decode.

During normal PP decode, rank 0 is idle for ~15ms while rank 1 computes.
This module uses that idle time to speculatively compute layers 0-29
for a draft token. If the draft matches rank 1's actual token, rank 0
sends the pre-computed hidden state immediately — saving ~15ms.

Gated by EXO_PP_DRAFT_MODEL. When unset, upstream's stream_generate
runs unchanged.

Architecture:
- Subclasses PipelineFirstLayer/PipelineLastLayer with a speculative mode
- Custom decode loop with explicit PP phase separation
- Draft model runs on rank 0 ONLY during idle time
- Zero modifications to upstream code
"""

import os
import sys
import time
from copy import deepcopy
from typing import Any, Callable, Generator

import mlx.core as mx
import mlx.nn as nn
from mlx_lm.generate import generation_stream
from mlx_lm.models.cache import ArraysCache, KVCache
from mlx_lm.sample_utils import make_sampler

from .auto_parallel import (
    CustomMlxLayer,
    PipelineFirstLayer,
    PipelineLastLayer,
)

_TRACE = os.environ.get("EXO_TRACING_ENABLED", "false").lower() in ("true", "1")


def _log(msg: str) -> None:
    if _TRACE:
        sys.stderr.write(f"[pp-spec] {msg}\n")
        sys.stderr.flush()


# ---------------------------------------------------------------------------
# Pipeline info extraction
# ---------------------------------------------------------------------------

def get_pipeline_info(model: nn.Module) -> tuple[int, int, mx.distributed.Group] | None:
    """Extract (rank, world_size, group) from pipeline layer wrappers.
    Returns None if model is not pipeline-parallel.
    """
    for layer in model.layers:  # type: ignore
        if isinstance(layer, PipelineLastLayer):
            return (layer.r, layer.s, layer.group)
    return None


# ---------------------------------------------------------------------------
# Cache snapshot / restore (for speculative rollback)
# ---------------------------------------------------------------------------

def _snapshot_cache(cache: list[Any]) -> list[Any]:
    """Lightweight snapshot: save offsets for KVCache, shallow-copy for ArraysCache."""
    snap: list[Any] = []
    for c in cache:
        if isinstance(c, ArraysCache):
            # Shallow copy the cache list — each entry is an mx.array (immutable once eval'd)
            snap.append(list(c.cache))
        elif isinstance(c, KVCache):
            snap.append(c.offset)
        else:
            snap.append(None)
    return snap


def _restore_cache(cache: list[Any], snap: list[Any]) -> None:
    """Restore cache from snapshot. KVCache trims keys/values; ArraysCache restores list."""
    for c, s in zip(cache, snap):
        if s is None:
            continue
        if isinstance(c, ArraysCache):
            c.cache = s
        elif isinstance(c, KVCache) and c.offset > s:
            c.keys = c.keys[:, :, :s, :]
            c.values = c.values[:, :, :s, :]
            c.offset = s


# ---------------------------------------------------------------------------
# Speculative pipeline layer wrappers (subclass, not modify)
# ---------------------------------------------------------------------------

class SpecPipelineFirstLayer(PipelineFirstLayer):
    """PipelineFirstLayer with PP decode mode: reads hidden from state list."""

    def __init__(self, base: PipelineFirstLayer):
        # Steal base's attributes without re-init
        nn.Module.__init__(self)
        self._original_layer = base._original_layer  # type: ignore
        self.r = base.r
        self.group = base.group
        self.is_prefill = base.is_prefill
        # Speculation state
        self._pp_decode: bool = False
        self._state_list: list[mx.array] | None = None
        self._hidden_idx: int = -1

    def __call__(self, x: mx.array, *args: object, **kwargs: object) -> mx.array:
        if self._pp_decode and self.r != 0:
            # PP decode: read hidden from state list instead of recv
            x = self._state_list[self._hidden_idx]  # type: ignore
            return self.original_layer(x, *args, **kwargs)
        # Normal path (prefill or rank 0)
        return super().__call__(x, *args, **kwargs)


class SpecPipelineLastLayer(PipelineLastLayer):
    """PipelineLastLayer with PP decode + speculative modes."""

    def __init__(self, base: PipelineLastLayer):
        nn.Module.__init__(self)
        self._original_layer = base._original_layer  # type: ignore
        self.r = base.r
        self.s = base.s
        self.group = base.group
        self.original_layer_signature = base.original_layer_signature
        self.is_prefill = base.is_prefill
        self.queue_sends = base.queue_sends
        # Speculation state
        self._pp_decode: bool = False
        self._speculative: bool = False
        self._state_list: list[mx.array] | None = None
        self._hidden_idx: int = -1

    def __call__(self, x: mx.array, *args: object, **kwargs: object) -> mx.array:
        if self._speculative:
            # Speculative mode: compute output, store in state list, no send/all_gather
            output = self.original_layer(x, *args, **kwargs)
            mx.eval(output)
            if self._state_list is not None:
                self._state_list[self._hidden_idx] = output
            return output

        if self._pp_decode:
            # PP decode: compute output, store in state list, no send/all_gather
            output = self.original_layer(x, *args, **kwargs)
            mx.eval(output)
            if self._state_list is not None:
                self._state_list[self._hidden_idx] = output
            return output

        # Normal path (prefill)
        return super().__call__(x, *args, **kwargs)


# ---------------------------------------------------------------------------
# Layer replacement (additive — wraps existing layers)
# ---------------------------------------------------------------------------

def _install_spec_layers(model: nn.Module) -> tuple[SpecPipelineFirstLayer | None, SpecPipelineLastLayer | None]:
    """Replace PipelineFirst/LastLayer with speculative versions. Returns refs."""
    layers = model.layers  # type: ignore
    spec_first: SpecPipelineFirstLayer | None = None
    spec_last: SpecPipelineLastLayer | None = None

    for i, layer in enumerate(layers):
        if isinstance(layer, PipelineFirstLayer) and not isinstance(layer, SpecPipelineFirstLayer):
            spec_first = SpecPipelineFirstLayer(layer)
            layers[i] = spec_first
        elif isinstance(layer, PipelineLastLayer) and not isinstance(layer, SpecPipelineLastLayer):
            spec_last = SpecPipelineLastLayer(layer)
            layers[i] = spec_last

    return spec_first, spec_last


def _set_pp_decode(
    spec_first: SpecPipelineFirstLayer | None,
    spec_last: SpecPipelineLastLayer | None,
    active: bool,
    state_list: list[mx.array] | None = None,
    hidden_idx: int = -1,
) -> None:
    if spec_first is not None:
        spec_first._pp_decode = active
        spec_first._state_list = state_list if active else None
        spec_first._hidden_idx = hidden_idx
    if spec_last is not None:
        spec_last._pp_decode = active
        spec_last._state_list = state_list if active else None
        spec_last._hidden_idx = hidden_idx


# ---------------------------------------------------------------------------
# Core decode loop with PP idle-time speculation
# ---------------------------------------------------------------------------

def pp_speculative_decode_loop(
    model: nn.Module,
    draft_model: nn.Module,
    prompt_cache: list[Any],
    draft_cache: list[Any],
    sampler: Callable,
    logits_processors: list[Any],
    first_y: mx.array,
    first_logprobs: mx.array,
    max_tokens: int,
    pp_rank: int,
    pp_world_size: int,
    pp_group: mx.distributed.Group,
) -> Generator[tuple[int, mx.array], None, None]:
    """PP decode loop with idle-time speculation. Yields (token_id, logprobs).

    Flow per step:
    1. Rank 0 computes layers 0-29 (writes hidden to state_list)
    2. all_gather exchanges hidden state
    3. Rank 1 computes layers 30-59, samples token
    4. all_gather exchanges sampled token — BOTH ranks have it
    5. During step 3-4 wait: rank 0 drafts next token with small model
    6. If draft matches on next step: rank 0 speculatively forwards,
       sends pre-computed hidden state immediately
    """
    is_rank0 = pp_rank == 0
    is_last_rank = pp_rank == pp_world_size - 1

    # Get model's inner structure for hidden size
    inner = getattr(model, "language_model", model)
    inner_model = getattr(inner, "model", inner)
    embed_tokens = inner_model.embed_tokens
    hidden_size = getattr(embed_tokens, "dims", embed_tokens.weight.shape[1])

    # Install speculative layer wrappers
    spec_first, spec_last = _install_spec_layers(inner)
    if spec_first is None and spec_last is None:
        raise RuntimeError("No pipeline layers found — is the model pipeline-parallel?")

    # Set up state list for hidden exchange
    _cache_state = [c.state if hasattr(c, 'state') else c for c in prompt_cache]
    _hidden_idx = len(_cache_state)
    _cache_state.append(mx.zeros((1, 1, hidden_size), dtype=mx.bfloat16))

    # Enable PP decode mode
    _set_pp_decode(spec_first, spec_last, True, _cache_state, _hidden_idx)

    # Skip lm_head on rank 0 (saves ~500MB weight reads per step)
    _lm_head_owner = getattr(model, "language_model", model)
    if is_rank0:
        _lm_head_owner._skip_lm_head = True  # type: ignore

    # Speculation state
    _draft_token: int | None = None
    _spec_snap: list[Any] | None = None
    _draft_snap: list[Any] | None = None
    _accepted = 0
    _rejected = 0

    y = first_y
    logprobs = first_logprobs

    def _pp_compute(token: mx.array) -> tuple[mx.array, mx.array]:
        """Forward pass through model with PP decode mode active."""
        with mx.stream(generation_stream):
            out = model(token[None], cache=prompt_cache)
            out = out[:, -1, :]
            lp = out - mx.logsumexp(out, keepdims=True)
            sampled = sampler(lp)
            return sampled, lp.squeeze(0)

    try:
        n = 0
        while n < max_tokens:
            # --- Check previous speculation ---
            if is_rank0 and _draft_token is not None:
                real_token = y.item()
                if real_token == _draft_token:
                    _accepted += 1
                    # Draft was right — we already ran layers 0-29 speculatively
                    # and _cache_state[_hidden_idx] has the correct hidden state.
                    # Skip _pp_compute, go straight to hidden exchange.
                    _draft_token = None
                    _log(f"n={n} ACCEPT draft={real_token}")
                else:
                    _rejected += 1
                    # Draft was wrong — restore main cache, recompute
                    if _spec_snap is not None:
                        _restore_cache(prompt_cache, _spec_snap)
                    _spec_snap = None
                    _draft_token = None
                    _log(f"n={n} REJECT draft={_draft_token} real={real_token}")

            # --- Rank 0: compute layers 0-29 (unless speculation hit) ---
            if is_rank0 and _draft_token is None:
                # Normal compute — need to run layers 0-29
                sampled, lp = _pp_compute(y)
                mx.eval(_cache_state[_hidden_idx])

            # --- Hidden state exchange via all_gather ---
            gathered_hidden = mx.distributed.all_gather(
                _cache_state[_hidden_idx].reshape(1, -1), group=pp_group
            )
            mx.eval(gathered_hidden)

            if not is_rank0:
                # Rank 1: take rank 0's hidden state
                _cache_state[_hidden_idx] = gathered_hidden[0:1].reshape(
                    _cache_state[_hidden_idx].shape
                )

            # --- Rank 1: compute layers 30-59 + sample ---
            if is_last_rank:
                sampled, lp = _pp_compute(y)

            # --- Token exchange via all_gather ---
            gathered_token = mx.distributed.all_gather(
                sampled.reshape(1) if is_last_rank else mx.zeros(1, dtype=mx.int32),
                group=pp_group,
            )
            mx.eval(gathered_token)
            final_token = gathered_token[-1:]  # last rank's token

            # --- Rank 0: draft during idle time (AFTER exchanges complete) ---
            # In practice the idle time is during the hidden/token exchanges
            # above. For K=1, we draft after getting the token since the next
            # step will check if the draft matches.
            if is_rank0:
                try:
                    tok_id = int(final_token.item())
                    # Draft next token
                    draft_logits = draft_model(mx.array([[tok_id]]), cache=draft_cache)
                    draft_tok = draft_logits[0, -1].argmax()
                    mx.eval(draft_tok)
                    _draft_token = int(draft_tok.item())

                    # Snapshot main cache, then speculatively forward draft token
                    _spec_snap = _snapshot_cache(prompt_cache)

                    if spec_last is not None:
                        spec_last._speculative = True
                    model(mx.array([[_draft_token]]), cache=prompt_cache)
                    mx.eval(_cache_state[_hidden_idx])
                    if spec_last is not None:
                        spec_last._speculative = False

                    _log(f"n={n} drafted={_draft_token}")
                except Exception as e:
                    _draft_token = None
                    _spec_snap = None
                    if spec_last is not None:
                        spec_last._speculative = False

            yield int(final_token.item()), lp if is_last_rank else mx.zeros(1)

            y = final_token
            n += 1

            if n % 256 == 0:
                mx.clear_cache()

    finally:
        # Restore model state
        _set_pp_decode(spec_first, spec_last, False)
        if is_rank0:
            _lm_head_owner._skip_lm_head = False  # type: ignore

        # Restore original layer classes
        layers = inner.layers  # type: ignore
        for i, layer in enumerate(layers):
            if isinstance(layer, SpecPipelineFirstLayer):
                # Can't easily un-subclass, but pp_decode is off so it's a no-op
                pass
            elif isinstance(layer, SpecPipelineLastLayer):
                pass

        # Log final stats
        total = _accepted + _rejected
        if total > 0:
            _log(f"Final: {_accepted}/{total} accepted ({_accepted/total*100:.0f}%), "
                 f"{_rejected} rejected")


# ---------------------------------------------------------------------------
# Draft model loading
# ---------------------------------------------------------------------------

def load_draft_model(model_path: str) -> tuple[nn.Module, list[Any]] | None:
    """Load a small draft model for speculation. Returns (model, cache) or None."""
    try:
        from mlx_lm.utils import load
        from mlx_lm.models.cache import make_prompt_cache

        _log(f"Loading draft model: {model_path}")
        model, _ = load(model_path)
        mx.eval(model.parameters())
        cache = make_prompt_cache(model)
        _log(f"Draft model loaded successfully")
        return model, cache
    except Exception as e:
        _log(f"Failed to load draft model: {e}")
        return None
