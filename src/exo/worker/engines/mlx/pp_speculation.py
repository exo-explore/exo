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

Overlap strategy:
- Hidden exchange uses send/recv (not all_gather) so rank 0 can proceed
  immediately after sending, drafting DURING rank 1's compute time.
- Token exchange uses all_gather (both ranks need the sampled token).
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
    """PipelineFirstLayer with PP recv mode for overlapped hidden exchange."""

    def __init__(self, base: PipelineFirstLayer):
        super().__init__(base.original_layer, base.r, base.group)
        self.is_prefill = base.is_prefill
        self._pp_recv: bool = False

    def __call__(self, x: mx.array, *args: object, **kwargs: object) -> mx.array:
        if self._pp_recv and self.r != 0:
            # Recv hidden from previous rank (blocks until rank 0 sends)
            mx.eval(x)
            x = mx.distributed.recv_like(x, (self.r - 1), group=self.group)
            mx.eval(x)
            return self.original_layer(x, *args, **kwargs)
        # Normal path (prefill or rank 0)
        return super().__call__(x, *args, **kwargs)


class SpecPipelineLastLayer(PipelineLastLayer):
    """PipelineLastLayer with PP send + speculative modes."""

    def __init__(self, base: PipelineLastLayer):
        super().__init__(base.original_layer, base.r, base.s, base.group)
        self.is_prefill = base.is_prefill
        self.queue_sends = base.queue_sends
        self._pp_send: bool = False
        self._pp_decode: bool = False
        self._speculative: bool = False
        self._state_list: list[mx.array] | None = None
        self._hidden_idx: int = -1

    def __call__(self, x: mx.array, *args: object, **kwargs: object) -> mx.array:
        if self._speculative:
            # Speculative mode: compute, store, NO send (don't leak speculation to rank 1)
            output = self.original_layer(x, *args, **kwargs)
            mx.eval(output)
            if self._state_list is not None:
                self._state_list[self._hidden_idx] = output
            return output

        if self._pp_send:
            # Send mode (rank 0): compute, send to rank 1, store locally
            output = self.original_layer(x, *args, **kwargs)
            mx.eval(output)
            if self.r != self.s - 1:
                sent = mx.distributed.send(output, (self.r + 1) % self.s, group=self.group)
                mx.eval(sent)
            if self._state_list is not None:
                self._state_list[self._hidden_idx] = output
            return output

        if self._pp_decode:
            # Decode mode (rank 1): compute, store, no comms
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


def _configure_layers(
    spec_first: SpecPipelineFirstLayer | None,
    spec_last: SpecPipelineLastLayer | None,
    *,
    pp_send: bool = False,
    pp_recv: bool = False,
    pp_decode: bool = False,
    speculative: bool = False,
    state_list: list[mx.array] | None = None,
    hidden_idx: int = -1,
) -> None:
    """Configure spec layer modes."""
    if spec_first is not None:
        spec_first._pp_recv = pp_recv
    if spec_last is not None:
        spec_last._pp_send = pp_send
        spec_last._pp_decode = pp_decode
        spec_last._speculative = speculative
        spec_last._state_list = state_list
        spec_last._hidden_idx = hidden_idx


# ---------------------------------------------------------------------------
# Core decode loop with PP idle-time speculation (overlapped)
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

    Overlapped flow per step:
    1. Rank 0: compute layers 0-29, SEND hidden to rank 1
    2. PARALLEL:
       - Rank 0: draft + speculative forward (during rank 1's compute)
       - Rank 1: RECV hidden, compute layers 30-59, sample token
    3. all_gather: exchange sampled token (both ranks get it)
    4. Verify: if draft matches, skip rank 0's compute next step
    """
    is_rank0 = pp_rank == 0
    is_last_rank = pp_rank == pp_world_size - 1

    # Get model's inner structure for hidden size
    inner = getattr(model, "language_model", model)
    inner_model = getattr(inner, "model", inner)
    embed_tokens = inner_model.embed_tokens
    hidden_size = getattr(embed_tokens, "dims", embed_tokens.weight.shape[1])

    # Find speculative layer wrappers (already installed by caller)
    spec_first, spec_last = None, None
    for layer in inner.layers:  # type: ignore
        if isinstance(layer, SpecPipelineFirstLayer):
            spec_first = layer
        elif isinstance(layer, SpecPipelineLastLayer):
            spec_last = layer
    if spec_first is None and spec_last is None:
        spec_first, spec_last = _install_spec_layers(inner)

    # State list for hidden exchange
    _cache_state = [c.state if hasattr(c, 'state') else c for c in prompt_cache]
    _hidden_idx = len(_cache_state)
    _cache_state.append(mx.zeros((1, 1, hidden_size), dtype=mx.bfloat16))

    # Skip lm_head on rank 0 (saves ~500MB weight reads per step)
    _lm_head_owner = getattr(model, "language_model", model)
    if is_rank0:
        _lm_head_owner._skip_lm_head = True  # type: ignore

    # Speculation state
    _draft_token: int | None = None
    _spec_snap: list[Any] | None = None
    _accepted = 0
    _rejected = 0

    y = first_y
    logprobs = first_logprobs

    def _rank0_compute(token: mx.array) -> None:
        """Rank 0: forward layers 0-29 in pp_send mode (sends hidden to rank 1)."""
        _configure_layers(spec_first, spec_last,
                          pp_send=True, state_list=_cache_state, hidden_idx=_hidden_idx)
        with mx.stream(generation_stream):
            model(token[None], cache=prompt_cache)

    def _rank0_speculative_fwd(token_id: int) -> None:
        """Rank 0: speculatively forward draft token (no send)."""
        if spec_last is not None:
            spec_last._speculative = True
            spec_last._pp_send = False
        with mx.stream(generation_stream):
            model(mx.array([[token_id]]), cache=prompt_cache)
            mx.eval(_cache_state[_hidden_idx])
        if spec_last is not None:
            spec_last._speculative = False

    def _rank1_compute(token: mx.array) -> tuple[mx.array, mx.array]:
        """Rank 1: recv hidden, forward layers 30-59, sample."""
        _configure_layers(spec_first, spec_last,
                          pp_recv=True, pp_decode=True,
                          state_list=_cache_state, hidden_idx=_hidden_idx)
        with mx.stream(generation_stream):
            out = model(token[None], cache=prompt_cache)
            out = out[:, -1, :]
            lp = out - mx.logsumexp(out, keepdims=True)
            sampled = sampler(lp)
            return sampled, lp.squeeze(0)

    try:
        n = 0
        while n < max_tokens:
            # ==== RANK 0: compute + send hidden, then draft during idle time ====
            if is_rank0:
                if _draft_token is None:
                    # Normal: compute layers 0-29, send hidden to rank 1
                    _rank0_compute(y)
                else:
                    # Previous draft accepted: hidden already precomputed.
                    # Send it directly to rank 1.
                    mx.eval(_cache_state[_hidden_idx])
                    sent = mx.distributed.send(
                        _cache_state[_hidden_idx],
                        (pp_rank + 1) % pp_world_size, group=pp_group
                    )
                    mx.eval(sent)

                # -- Draft DURING rank 1's compute (the ~14ms idle window) --
                try:
                    draft_logits = draft_model(mx.array([[y.item()]]), cache=draft_cache)
                    draft_tok = draft_logits[0, -1].argmax()
                    mx.eval(draft_tok)
                    _draft_token = int(draft_tok.item())

                    _spec_snap = _snapshot_cache(prompt_cache)
                    _rank0_speculative_fwd(_draft_token)
                    _log(f"n={n} drafted={_draft_token}")
                except Exception:
                    _draft_token = None
                    _spec_snap = None
                    if spec_last is not None:
                        spec_last._speculative = False

            # ==== RANK 1: recv hidden + compute + sample (parallel with rank 0's draft) ====
            if is_last_rank:
                sampled, lp = _rank1_compute(y)

            # ==== TOKEN EXCHANGE (both ranks sync here) ====
            gathered_token = mx.distributed.all_gather(
                sampled.reshape(1) if is_last_rank else mx.zeros(1, dtype=mx.int32),
                group=pp_group,
            )
            mx.eval(gathered_token)
            final_token = gathered_token[-1:]

            # ==== VERIFY draft ====
            if is_rank0 and _draft_token is not None:
                real_token = int(final_token.item())
                if real_token == _draft_token:
                    _accepted += 1
                    _log(f"n={n} ACCEPT draft={_draft_token}")
                    # Advance draft cache with accepted token
                    draft_model(mx.array([[real_token]]), cache=draft_cache)
                else:
                    _rejected += 1
                    _log(f"n={n} REJECT draft={_draft_token} real={real_token}")
                    if _spec_snap is not None:
                        _restore_cache(prompt_cache, _spec_snap)
                    _spec_snap = None
                    _draft_token = None
                    # Correct draft model's cache
                    draft_model(mx.array([[real_token]]), cache=draft_cache)
            elif is_rank0:
                # First step or error: advance draft cache
                draft_model(mx.array([[int(final_token.item())]]), cache=draft_cache)

            yield int(final_token.item()), lp if is_last_rank else mx.zeros(1)

            y = final_token
            n += 1

            if n % 256 == 0:
                mx.clear_cache()

    finally:
        # Restore model state
        _configure_layers(spec_first, spec_last)  # all modes off
        if is_rank0:
            _lm_head_owner._skip_lm_head = False  # type: ignore

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
