"""Pipelined speculative-decoding spec loop.

Implements :class:`PipelinedModelDrafter` -- a custom spec loop that
talks to the drafter through a :class:`DrafterTransport` (in-process,
remote, ...). The win over :class:`ModelDrafter` (which delegates to
``mlx_lm.speculative_generate_step``) is **cross-round speculation**:
while the target rank verifies round ``t``'s drafts, the drafter
speculatively starts round ``t + 1`` by predicting the would-be bonus
token and continuing for ``K`` more forwards. If the target's actual
bonus matches the drafter's predicted bonus, round ``t + 1``'s drafts
are already in hand by the time round ``t``'s verify finishes; if not,
the speculative work is rolled back and the standard non-speculative
path runs.

Apple-Silicon caveat: MLX serialises Metal command queues per device,
so the in-process overlap factor between drafter and target forwards
is ~0.1-0.3 (parallelism is bounded by memory-bandwidth contention,
not GPU saturation). The architecture's payoff scales with topology:
on a multi-machine deployment where target verify includes a network
round-trip, the speculative drafter forward fully overlaps the
network latency and the gain unlocks. :class:`RemoteTransport` ships
exactly that case.

Multi-target asymmetric placement (V2 ``target_subgroup_size > 1``)
--------------------------------------------------------------------
The target group is tensor-parallel across N nodes; the drafter lives
on a different node and talks to target rank 0 over a TCP socket. Per-
round flow:

  1. **Drafter -> target rank 0 (socket).** Rank 0 issues an
     ``OP_FORWARD`` over the wire, gets back ``k_this`` drafts.
  2. **Rank 0 -> all target ranks (collective).** Rank 0 broadcasts
     the drafts on the target subgroup via :func:`_broadcast_drafts`.
     Non-root ranks receive into the same buffer shape; the broadcast
     uses :func:`mx_broadcast_int_list` (a length-prefixed
     ``all_sum``). Drafter-rank does NOT participate -- it isn't a
     member of the target subgroup.
  3. **All target ranks (collective).** Run the verify forward
     ``model([seed, *drafts])`` -- a TP all-reduce inside the model
     makes logits byte-identical on every target rank.
  4. **Rank 0 samples + broadcasts target tokens.** The sampler is
     non-deterministic (temperature > 0 uses MLX's per-rank PRNG) so
     each rank would otherwise produce divergent ``target_tokens``,
     diverge on ``num_accepted``, trim the prompt cache by different
     amounts, and desync at the next TP forward. Rank 0 samples
     locally and broadcasts the chosen tokens via
     :func:`_broadcast_target_tokens`; non-root ranks consume the
     broadcast and skip the sampler entirely. Determinism then falls
     out of the broadcast contract rather than relying on RNG state
     coordination.
  5. **All target ranks compute identical accept/reject.** Both ranks
     compare ``target_tokens`` (now identical from broadcast) against
     ``drafts`` (also identical from step 2), reach the same
     ``num_accepted``, and trim the prompt cache by the same amount.
  6. **Drafter cache reconciliation on rank 0 only.** Rank 0 issues
     any required ``OP_TRIM_CACHE`` / next-round ``OP_FORWARD`` over
     the socket; non-root just waits for the next draft broadcast
     round at step 2.

The collective overhead per round is two small ``all_sum`` calls
(drafts ``k+1`` ints, target tokens ``k+1`` ints) -- microsecond-
range on Thunderbolt RDMA, negligible against the verify forward.

Recovery: drafter-rank death mid-generation
-------------------------------------------
If the drafter rank crashes between rounds, root's
``transport.forward`` raises :class:`OSError` (subclassed as
``ConnectionError`` / ``BrokenPipeError`` depending on which side
closed). The recovery is layered:

  1. **Within-request abort** (this module). Before re-raising, the
     :func:`_pipelined_speculative_step` wrapper broadcasts
     :data:`DRAFT_ABORT_SENTINEL` on the draft channel. Non-root
     ranks decode the sentinel inside :func:`_broadcast_drafts` and
     raise :class:`DrafterAbortedError`, exiting their spec loop in
     lockstep with root rather than blocking on a next-round
     broadcast that will never arrive. The
     :class:`exo.worker.engines.mlx.generator.remote_drafter.RemoteTransport`
     also flips a sticky ``is_failed`` flag so subsequent
     :meth:`open_session` calls fail fast instead of allocating a
     new spec session on a dead wire.

  2. **Cross-request teardown** (control plane). The runner
     subprocess that owned the failed transport surfaces the
     exception out of ``mlx_generate``, the runner crashes, the
     supervisor reports :class:`RunnerFailed`, and the master's
     worker-plan ``_kill_runner`` rule shuts every peer rank down
     in the same instance. A fresh placement is re-issued on the
     next planning tick.

  3. **Drafter-node disconnect** (control plane). When the drafter
     *node* goes offline (rather than the drafter *process*), the
     master's instance-deletion loop iterates
     ``instance.all_node_to_runner`` (target + drafter) and emits
     :class:`InstanceDeleted` once the drafter node leaves
     ``connected_node_ids``. Workers pick up the deletion in the
     usual plan tick. Total time-to-recovery is bounded by the
     master's ``node_inactivity_timeout`` (5 s) plus the
     supervisor's SIGTERM/SIGKILL escalation budget (worst case
     ~25 s), the same envelope as a target-rank crash.

Target-rank death (a peer target rank in the TP subgroup) takes
the same path as case 3 above: the master's instance-deletion
loop already covered ``shard_assignments.node_to_runner``; the
worker plan's ``_kill_runner`` rule gossips ``RunnerFailed``
across the surviving ranks and the supervisor SIGKILL chain
unblocks any in-flight TP collectives.

Cache accounting (drafter side) -- this is the only complex bit, so
spelled out here once and referenced from the code:

  Notation: ``O`` = drafter cache offset before round ``t``'s propose.
  ``K`` = ``num_draft_tokens``.

  Round ``t`` propose, length-1 seed (partial-accept-from-prev case):
    ``forward([seed_t], K)`` -> K outputs. K forwards, each adds 1
    position. Cache offset O+K. Cache content extends with
    ``[seed_t, d_0..d_{K-2}]`` (the K-th draft d_{K-1} is the K-th
    output, *not* fed back as input).

  Round ``t`` propose, length-2 seed (full-accept-from-prev case):
    ``forward([drafts_{t-1}[-1], seed_t], K)`` -> K outputs. K forwards;
    the first has length-2 input, so cache extends by K+1.

  Speculative round ``t + 1`` (cross-round speculation):
    ``forward([drafts_t[-1]], K + 1)`` -> K+1 outputs. K+1 forwards,
    cache extends by K+1. Outputs are
    ``[d^pred_K, d^spec_0, ..., d^spec_{K-1}]``: the first is the
    drafter's prediction of bonus_t (compared against actual bonus_t
    to detect speculation hit); the rest are round t+1's drafts.
    Cache offset after speculation: O+2K+1.

  Round ``t`` accept outcomes:

    * Partial accept (``num_accepted < K_this``): drafter cache trim
      by ``max(K_this - num_accepted - 1, 0)``. If speculation was
      active, also rollback ``K + 1``. Round ``t + 1``'s propose is a
      length-1-seed call.
    * Full accept, speculation MISS (``bonus_t != d^pred_K``): rollback
      ``K + 1``. Round ``t + 1``'s propose is a length-2-seed call.
    * Full accept, speculation HIT: no rollback. Drafter cache
      offset O+2K+1, content matches what mlx_lm's ``_draft_generate``
      would produce after a length-2 first forward + K-1 length-1
      forwards in round t+1. Round ``t + 1``'s drafts come from the
      speculative outputs; round ``t + 1`` skips its own propose call.
    * Truncated last round (``K_this < K``): speculation is disabled
      because there's no round t+1 to feed.

The matching :func:`_pipelined_speculative_step` enforces this
accounting; any divergence between the comments above and the code is
a bug, please flag it.
"""

from __future__ import annotations

import contextlib
import os as _diag_os
import sys as _diag_sys
import time
from typing import Callable, Final, Generator, Sequence, Sized, cast, final

import mlx.core as mx
from mlx_lm.generate import GenerationResponse
from mlx_lm.models.cache import trim_prompt_cache as mlx_trim_prompt_cache
from mlx_lm.tokenizer_utils import TokenizerWrapper

from exo.worker.engines.mlx.generator.drafter import DraftMode
from exo.worker.engines.mlx.generator.drafter_transport import (
    DrafterTransport,
    DraftFuture,
)
from exo.worker.engines.mlx.types import KVCacheType, Model
from exo.worker.engines.mlx.utils_mlx import (
    TargetPeerFanout,
    mx_broadcast_int_list,
    target_peer_broadcast_int_list,
)
from exo.worker.runner.bootstrap import logger as _diag_logger

# Per-round spec-decode diagnostics. Off by default; set
# ``EXO_SPEC_DIAG=1`` to enable. When enabled, each call writes both
# to loguru and to ``/tmp/spec_diag_<pid>.log`` so diagnostics survive
# whatever's swallowing the runner subprocess's stdout (loguru
# forwarding has been observed to drop on some nodes in our cluster).
#
# Added during gemma-4 asymmetric-drafter bring-up to localize a
# TP-collective deadlock; the hooks are kept (gated) so future
# correctness regressions can be isolated quickly without redeploying
# with new logging.
_SPEC_DIAG_ENABLED: Final[bool] = _diag_os.environ.get("EXO_SPEC_DIAG", "") in (
    "1",
    "true",
    "yes",
)


def _spec_diag(message: str) -> None:
    """Emit a spec-decode diagnostic line. No-op unless ``EXO_SPEC_DIAG``."""
    if not _SPEC_DIAG_ENABLED:
        return
    _diag_logger.info(message)
    try:
        path = f"/tmp/spec_diag_{_diag_os.getpid()}.log"
        with open(path, "a", encoding="utf-8") as fh:
            _ = fh.write(f"{time.time():.6f} {message}\n")
    except OSError:
        try:
            _ = _diag_sys.stderr.write(f"[spec-diag fallback] {message}\n")
            _diag_sys.stderr.flush()
        except OSError:
            pass


# Length-prefix slot value reserved for the "drafter aborted" signal.
# Picked from the int32 positive range so it survives
# ``_validate_broadcast_values`` (well above any legitimate ``K``,
# below ``_MX_BROADCAST_MAX_VALUE`` so the validator accepts it).
DRAFT_ABORT_SENTINEL: Final[int] = (1 << 31) - 2


@final
class DrafterAbortedError(RuntimeError):
    """Raised by non-root target ranks when root signals draft abort.

    Root encodes :data:`DRAFT_ABORT_SENTINEL` in the broadcast
    length-prefix slot when its ``transport.forward()`` raises
    (drafter rank crashed, socket dropped, etc). Non-root ranks
    decode the sentinel inside :func:`_broadcast_drafts` and raise
    this exception so the spec loop on every rank exits in lockstep,
    rather than non-root hanging forever on the next-round draft
    broadcast that root will never send.
    """


def _get_eos_ids(tokenizer: TokenizerWrapper) -> list[int]:
    eos: list[int] | None = getattr(tokenizer, "eos_token_ids", None)
    if eos is None:
        return []
    return eos


def _get_tokenizer_vocab_size(tokenizer: TokenizerWrapper) -> int | None:
    """Return the *full* tokenizer vocabulary size including added tokens.

    Used by the spec-decode loop as an early sanity check on emitted
    token ids: anything outside ``[0, vocab_size)`` cannot have come
    from a clean broadcast (the sampler and drafter both produce ids
    in that range), so it always points at a wire-level corruption
    upstream. Returns ``None`` when the tokenizer doesn't expose a
    vocab size (extremely defensive; mlx_lm tokenizers do).

    Important: HuggingFace fast tokenizers expose ``vocab_size`` as the
    *base* vocabulary, which excludes added tokens (chat templates,
    EOS, control tokens). Using ``vocab_size`` alone made this guard
    misclassify legitimate added tokens as wire corruption and
    crash the runner. We therefore prefer:

    1. ``len(tokenizer)`` -- the canonical HF API for the full
       vocabulary including added tokens.
    2. ``vocab_size + len(get_added_vocab())`` -- explicit added-token
       sum when the tokenizer-wrapper hides ``__len__``.
    3. ``max(vocab.values()) + 1`` -- last-resort scan over the
       internal vocab map (slow path, but used only on tokenizers
       that don't expose either of the above).
    4. ``vocab_size`` -- only when nothing else is available; this
       falls back to the original behaviour and may incorrectly
       flag added tokens, but is still better than disabling the
       guard entirely.
    """
    inner: object = getattr(tokenizer, "_tokenizer", None)
    if inner is None:
        return None
    try:
        full_len = len(cast("Sized", inner))
    except TypeError:
        full_len = None
    if isinstance(full_len, int) and full_len > 0:
        return full_len
    base: int | None = None
    raw_base: object = getattr(inner, "vocab_size", None)
    if isinstance(raw_base, int) and raw_base > 0:
        base = raw_base
    added: int = 0
    get_added = getattr(inner, "get_added_vocab", None)
    if callable(get_added):
        try:
            added_vocab = cast("dict[object, object]", get_added())
            added = len(added_vocab)
        except Exception:
            added = 0
    if base is not None:
        return base + added
    vocab: object = getattr(inner, "vocab", None)
    if isinstance(vocab, dict) and vocab:
        return max(cast("dict[object, int]", vocab).values()) + 1
    return None


def _process_logits_for_position(
    raw_logits: mx.array,
    prev_tokens: mx.array,
    logits_processors: list[Callable[[mx.array, mx.array], mx.array]],
) -> mx.array:
    """Apply logits processors and convert to logprobs (single position)."""
    out = raw_logits
    for proc in logits_processors:
        out = proc(prev_tokens, out)
    return out - mx.logsumexp(out, axis=-1, keepdims=True)


@final
class PipelinedModelDrafter:
    """Speculative decoding via a drafter accessed through :class:`DrafterTransport`.

    Owns its own spec loop so the drafter can be remote (different MLX
    rank) without the target rank loading the drafter model. The
    transport-agnostic propose/trim primitives mean swapping
    in-process for remote drafter placement is a one-line construction
    change at :func:`make_drafter`; the spec loop is unaffected.

    Multi-target asymmetric placement (``target_subgroup_size > 1``):
    the target root rank holds the drafter socket (``transport`` is
    set, ``is_target_root=True``) and broadcasts each round's drafts on
    ``target_group`` so non-root target ranks receive them in lockstep.
    Non-root ranks construct with ``transport=None`` and consume the
    broadcast each round; both ranks then run the same verify forward
    (which is a TP collective on the model itself) and reach identical
    accept/reject decisions deterministically because TP all-reduces
    the final logits to be byte-identical on every rank.
    """

    def __init__(
        self,
        *,
        transport: DrafterTransport | None,
        num_draft_tokens: int,
        target_group: mx.distributed.Group | None = None,
        target_peer_fanout: TargetPeerFanout | None = None,
        is_target_root: bool = True,
    ) -> None:
        if num_draft_tokens < 1:
            raise ValueError(f"num_draft_tokens must be >= 1, got {num_draft_tokens}")
        if transport is None:
            # Multi-target consumer rank: no socket, drafts arrive via
            # broadcast on ``target_group``.
            if is_target_root:
                raise ValueError(
                    "transport=None requires is_target_root=False (the "
                    "consumer rank does not own the drafter socket)"
                )
            if target_group is None:
                raise ValueError(
                    "transport=None requires a target_group to receive "
                    "draft broadcasts on"
                )
        else:
            if num_draft_tokens > transport.num_draft_tokens:
                raise ValueError(
                    f"num_draft_tokens ({num_draft_tokens}) exceeds transport's "
                    f"max ({transport.num_draft_tokens})"
                )
            if not is_target_root:
                raise ValueError(
                    "is_target_root=False on a transport-owning rank is a "
                    "configuration error: the rank that holds the drafter "
                    "socket is the broadcast root by definition"
                )
        self._transport = transport
        self._num_draft_tokens = num_draft_tokens
        self._target_group = target_group
        self._target_peer_fanout = target_peer_fanout
        self._is_target_root = is_target_root
        # Per-request spec-decode telemetry. Mutated in place by the
        # spec body each round; read by ``mlx_generate`` after streaming
        # completes to populate ``GenerationStats``. Single-request
        # lifecycle (a fresh drafter is built per request in
        # ``mlx_generate``), so no thread-safety concerns.
        self._metrics: dict[str, int] = {
            "proposed_draft_tokens": 0,
            "accepted_draft_tokens": 0,
            "spec_decode_rounds": 0,
        }

    @property
    def mode(self) -> DraftMode:
        return "pipelined"

    @property
    def num_draft_tokens(self) -> int:
        return self._num_draft_tokens

    def metrics(self) -> dict[str, int]:
        """Snapshot of accumulated spec-decode metrics for this request.

        Keys: ``proposed_draft_tokens`` (total drafts proposed across all
        rounds), ``accepted_draft_tokens`` (drafts the target accepted),
        ``spec_decode_rounds`` (rounds executed). Acceptance rate is
        ``accepted / proposed`` when ``proposed > 0``. Counters reset on
        each new request via the per-request drafter construction in
        ``mlx_generate``; mutate in lockstep with the spec loop.
        """
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
    ) -> Generator[GenerationResponse, None, None]:
        yield from _pipelined_stream_generate(
            model=model,
            tokenizer=tokenizer,
            prompt=prompt,
            context_tokens=list(context_tokens),
            prompt_cache=prompt_cache,
            max_tokens=max_tokens,
            sampler=sampler,
            logits_processors=list(logits_processors),
            transport=self._transport,
            num_draft_tokens=self._num_draft_tokens,
            prefill_step_size=prefill_step_size,
            target_group=self._target_group,
            target_peer_fanout=self._target_peer_fanout,
            is_target_root=self._is_target_root,
            metrics=self._metrics,
        )

    def shutdown(self) -> None:
        """Release transport resources."""
        if self._transport is not None:
            self._transport.shutdown()


def _pipelined_stream_generate(
    *,
    model: Model,
    tokenizer: TokenizerWrapper,
    prompt: mx.array,
    context_tokens: list[int],
    prompt_cache: KVCacheType,
    max_tokens: int,
    sampler: Callable[[mx.array], mx.array],
    logits_processors: list[Callable[[mx.array, mx.array], mx.array]],
    transport: DrafterTransport | None,
    num_draft_tokens: int,
    prefill_step_size: int,
    target_group: mx.distributed.Group | None = None,
    target_peer_fanout: TargetPeerFanout | None = None,
    is_target_root: bool = True,
    metrics: dict[str, int] | None = None,
) -> Generator[GenerationResponse, None, None]:
    """Mirror of ``mlx_lm.stream_generate`` framing for the pipelined drafter.

    The framing (detokenisation, tps tracking, finish reasons) matches
    :func:`exo.worker.engines.mlx.generator.drafter._ngram_stream_generate`
    so the call site in ``mlx_generate`` doesn't branch on drafter type.
    """
    detokenizer = tokenizer.detokenizer
    detokenizer.reset()  # type: ignore[reportUnknownMemberType]
    eos_ids = _get_eos_ids(tokenizer)
    # Vocab bound for early surfacing of broadcast corruption.
    # ``add_token`` would otherwise blow up deep inside the SPM
    # detokenizer with ``IndexError: list index out of range`` and
    # the operator has to dig through the mlx_lm internals to learn
    # which token id was bogus.
    vocab_size = _get_tokenizer_vocab_size(tokenizer)

    token_iter = _pipelined_speculative_step(
        prompt=prompt,
        model=model,
        transport=transport,
        prompt_cache=prompt_cache,
        max_tokens=max_tokens,
        sampler=sampler,
        logits_processors=logits_processors,
        num_draft_tokens=num_draft_tokens,
        prefill_step_size=prefill_step_size,
        prompt_token_count=len(context_tokens),
        target_group=target_group,
        target_peer_fanout=target_peer_fanout,
        is_target_root=is_target_root,
        metrics=metrics,
    )

    prompt_size = len(context_tokens)
    tic = time.perf_counter()
    prompt_tps = 0.0
    n = -1
    token = 0
    logprobs = mx.zeros((1,))
    from_draft = False
    finish_reason: str | None = None
    for n, (token, logprobs, from_draft) in enumerate(token_iter):
        if n == 0:
            prompt_time = time.perf_counter() - tic
            prompt_tps = prompt_size / prompt_time if prompt_time > 0 else 0.0
            tic = time.perf_counter()
        if token in eos_ids:
            finish_reason = "stop"
            break
        if vocab_size is not None and not 0 <= token < vocab_size:
            raise RuntimeError(
                f"pipelined drafter emitted token id {token} outside "
                f"tokenizer vocab [0, {vocab_size}); "
                "this is a wire-protocol bug in the spec-decode "
                "broadcast path (cross-stream JACCL collision or "
                "rank divergence). The runner will crash and the "
                "supervisor will rebuild the instance."
            )
        detokenizer.add_token(token)  # type: ignore[reportUnknownMemberType]
        if (n + 1) == max_tokens:
            finish_reason = "length"
            break
        elapsed = time.perf_counter() - tic
        yield GenerationResponse(
            text=detokenizer.last_segment,
            token=token,
            logprobs=logprobs,
            from_draft=from_draft,
            prompt_tokens=prompt_size,
            prompt_tps=prompt_tps,
            generation_tokens=n + 1,
            generation_tps=(n + 1) / elapsed if elapsed > 0 else 0.0,
            peak_memory=mx.get_peak_memory() / 1e9,
            finish_reason=None,
        )

    detokenizer.finalize()  # type: ignore[reportUnknownMemberType]
    elapsed = time.perf_counter() - tic
    yield GenerationResponse(
        text=detokenizer.last_segment,
        token=token,
        logprobs=logprobs,
        from_draft=from_draft,
        prompt_tokens=prompt_size,
        prompt_tps=prompt_tps,
        generation_tokens=n + 1 if n >= 0 else 0,
        generation_tps=(n + 1) / elapsed if elapsed > 0 and n >= 0 else 0.0,
        peak_memory=mx.get_peak_memory() / 1e9,
        finish_reason=finish_reason or ("stop" if token in eos_ids else "length"),
    )


def _broadcast_int_list(
    payload: list[int] | None,
    *,
    length: int,
    target_group: mx.distributed.Group | None,
    target_peer_fanout: TargetPeerFanout | None,
    is_root: bool,
) -> list[int]:
    """Pick the correct fixed-length int broadcast for the active wiring.

    Multi-target asymmetric placements ride
    :func:`target_peer_broadcast_int_list` (TCP fanout, immune to
    JACCL int/float wire conflation). Every other path -- single-rank
    targets, symmetric multi-rank without a drafter, test fakes that
    bring up a ``mx.distributed.Group`` without populating a fanout
    -- falls through to :func:`mx_broadcast_int_list`. The fallback
    is correct in those cases because the JACCL bug only manifests
    when the spec-decode int broadcasts interleave with the model's
    TP ``all_sum`` collectives on the same group; without spec
    decode (no drafter) or without a multi-rank target (no TP
    collectives) the interleaving cannot happen.
    """
    if target_peer_fanout is not None:
        return target_peer_broadcast_int_list(
            payload, length, target_peer_fanout, is_root=is_root
        )
    return mx_broadcast_int_list(payload, length, target_group, is_root=is_root)


def _broadcast_drafts(
    drafts: list[int] | None,
    *,
    k: int,
    target_group: mx.distributed.Group | None,
    target_peer_fanout: TargetPeerFanout | None,
    is_root: bool,
) -> list[int]:
    """Rank-0 broadcast of a draft list, padded to ``k`` slots + length prefix.

    Wire format: ``[len(drafts), drafts[0], ..., drafts[len-1], 0, 0, ...]``
    of fixed length ``k + 1``. Encoding the length up front lets us use
    a single fixed-size ``all_sum`` collective per round (vs. a
    count-then-payload two-collective handshake) on the spec-decode hot
    path -- the cost is a few unused int32 slots when the drafter
    returns fewer than ``k`` drafts.

    Single-rank short-circuit (``target_group is None``): returns
    ``drafts`` on the root and is a programming error elsewhere (the
    consumer rank must always have a group to receive on).
    """
    if target_group is None and target_peer_fanout is None:
        if not is_root or drafts is None:
            raise RuntimeError("non-root broadcast consumer requires target_group")
        return list(drafts)
    if is_root:
        if drafts is None:
            raise RuntimeError("root broadcaster requires drafts")
        if len(drafts) > k:
            raise RuntimeError(
                f"drafts length ({len(drafts)}) exceeds k ({k}); "
                "transport must clamp before broadcasting"
            )
        payload = [len(drafts)] + list(drafts) + [0] * (k - len(drafts))
        broadcast = _broadcast_int_list(
            payload,
            length=k + 1,
            target_group=target_group,
            target_peer_fanout=target_peer_fanout,
            is_root=True,
        )
    else:
        broadcast = _broadcast_int_list(
            None,
            length=k + 1,
            target_group=target_group,
            target_peer_fanout=target_peer_fanout,
            is_root=False,
        )
    actual_len = broadcast[0]
    if actual_len == DRAFT_ABORT_SENTINEL:
        # Root has flagged a drafter-side failure (see
        # :func:`_broadcast_abort`). Surface a typed exception so the
        # spec loop on this rank exits in lockstep with root rather
        # than waiting on the next-round broadcast that won't arrive.
        raise DrafterAbortedError(
            "drafter aborted; root signalled abort via length-prefix "
            "sentinel after a transport-side failure"
        )
    if actual_len < 0 or actual_len > k:
        raise RuntimeError(
            f"draft broadcast decoded invalid length {actual_len} (buffer {broadcast})"
        )
    return broadcast[1 : 1 + actual_len]


def _broadcast_abort(
    *,
    k: int,
    target_group: mx.distributed.Group | None,
    target_peer_fanout: TargetPeerFanout | None,
) -> None:
    """Root-only: broadcast the abort sentinel on the draft channel.

    Encodes :data:`DRAFT_ABORT_SENTINEL` as the length-prefix of an
    otherwise-zero ``k + 1`` int payload, matching the wire shape of
    a normal :func:`_broadcast_drafts` round so non-root ranks
    decode it on the same fixed-size collective they were already
    waiting on. Non-root surfaces it as :class:`DrafterAbortedError`.

    Single-rank short-circuit (``target_group is None``): no peers
    to notify, so this is a no-op. The local rank still re-raises
    the underlying transport exception that triggered the abort.
    """
    if target_group is None and target_peer_fanout is None:
        return
    payload = [DRAFT_ABORT_SENTINEL] + [0] * k
    _ = _broadcast_int_list(
        payload,
        length=k + 1,
        target_group=target_group,
        target_peer_fanout=target_peer_fanout,
        is_root=True,
    )


def _broadcast_target_tokens(
    target_tokens: list[int] | None,
    *,
    k: int,
    k_this: int,
    target_group: mx.distributed.Group | None,
    target_peer_fanout: TargetPeerFanout | None,
    is_root: bool,
) -> list[int]:
    """Rank-0 broadcast of post-verify sampled tokens, slot count ``k + 1``.

    Why a separate broadcast from the drafts: the sampler is the only
    non-deterministic step in the verify path. With temperature > 0
    each target rank's MLX PRNG advances independently, so identical
    logits produce divergent ``target_tokens`` and the ranks desync on
    the next TP forward. Broadcasting the chosen tokens from rank 0
    makes the sampler effectively a rank-0 operation; non-root ranks
    skip the sampler entirely.

    Wire format: fixed-size ``k + 1`` int buffer (the verify forward
    always produces exactly ``k_this + 1`` tokens; trailing slots are
    zero-padded so the buffer shape doesn't change with ``k_this``).
    Both ranks know ``k_this`` from the prior draft broadcast, so we
    skip the length prefix and slice on receive.

    Single-rank short-circuit (``target_group is None``): identity on
    root; programming error on consumer (no broadcast peer).
    """
    if target_group is None and target_peer_fanout is None:
        if not is_root or target_tokens is None:
            raise RuntimeError("non-root broadcast consumer requires target_group")
        if len(target_tokens) != k_this + 1:
            raise RuntimeError(
                f"target_tokens length ({len(target_tokens)}) must "
                f"equal k_this + 1 ({k_this + 1}); the verifier always "
                "emits exactly that many tokens per round"
            )
        return list(target_tokens)
    if is_root:
        if target_tokens is None:
            raise RuntimeError("root broadcaster requires target_tokens")
        if len(target_tokens) != k_this + 1:
            raise RuntimeError(
                f"target_tokens length ({len(target_tokens)}) must "
                f"equal k_this + 1 ({k_this + 1}); the verifier always "
                "emits exactly that many tokens per round"
            )
        payload = list(target_tokens) + [0] * (k - k_this)
        broadcast = _broadcast_int_list(
            payload,
            length=k + 1,
            target_group=target_group,
            target_peer_fanout=target_peer_fanout,
            is_root=True,
        )
    else:
        broadcast = _broadcast_int_list(
            None,
            length=k + 1,
            target_group=target_group,
            target_peer_fanout=target_peer_fanout,
            is_root=False,
        )
    return broadcast[: k_this + 1]


def _pipelined_speculative_step(
    *,
    prompt: mx.array,
    model: Model,
    transport: DrafterTransport | None,
    prompt_cache: KVCacheType,
    max_tokens: int,
    sampler: Callable[[mx.array], mx.array],
    logits_processors: list[Callable[[mx.array, mx.array], mx.array]],
    num_draft_tokens: int,
    prefill_step_size: int,
    prompt_token_count: int,
    target_group: mx.distributed.Group | None = None,
    target_peer_fanout: TargetPeerFanout | None = None,
    is_target_root: bool = True,
    metrics: dict[str, int] | None = None,
) -> Generator[tuple[int, mx.array, bool], None, None]:
    """Public spec-step generator with drafter-failure recovery.

    Wraps :func:`_pipelined_speculative_step_body` so that any
    :class:`OSError` originating from the drafter wire on the root
    rank (socket close, broken pipe, peer reset, etc.) also
    broadcasts :data:`DRAFT_ABORT_SENTINEL` to non-root target
    ranks. Non-root decodes it inside :func:`_broadcast_drafts`
    and raises :class:`DrafterAbortedError`, exiting the spec loop
    in lockstep with root. Without this wrap, root would re-raise
    cleanly while non-root sat indefinitely on the next-round
    draft broadcast that root will never send.

    Non-root and single-rank placements pass through unchanged:
    non-root never touches the transport (so there is nothing to
    abort from); :func:`_broadcast_abort` short-circuits when
    ``target_group is None`` (no peers to notify). The local rank
    re-raises the underlying exception in both cases.
    """
    inner = _pipelined_speculative_step_body(
        prompt=prompt,
        model=model,
        transport=transport,
        prompt_cache=prompt_cache,
        max_tokens=max_tokens,
        sampler=sampler,
        logits_processors=logits_processors,
        num_draft_tokens=num_draft_tokens,
        prefill_step_size=prefill_step_size,
        prompt_token_count=prompt_token_count,
        target_group=target_group,
        target_peer_fanout=target_peer_fanout,
        is_target_root=is_target_root,
        metrics=metrics,
    )
    try:
        yield from inner
    except OSError:
        if is_target_root:
            # Recovery best-effort: if the abort broadcast itself
            # fails (e.g. ``target_group`` is also dead), the
            # supervisor SIGKILL chain still tears non-root
            # runners down via the master's instance-deletion
            # path. Suppression keeps the original ``OSError``
            # intact for the caller's traceback.
            with contextlib.suppress(Exception):
                _broadcast_abort(
                    k=num_draft_tokens,
                    target_group=target_group,
                    target_peer_fanout=target_peer_fanout,
                )
        raise


def _pipelined_speculative_step_body(
    *,
    prompt: mx.array,
    model: Model,
    transport: DrafterTransport | None,
    prompt_cache: KVCacheType,
    max_tokens: int,
    sampler: Callable[[mx.array], mx.array],
    logits_processors: list[Callable[[mx.array, mx.array], mx.array]],
    num_draft_tokens: int,
    prefill_step_size: int,
    prompt_token_count: int,
    target_group: mx.distributed.Group | None = None,
    target_peer_fanout: TargetPeerFanout | None = None,
    is_target_root: bool = True,
    metrics: dict[str, int] | None = None,
) -> Generator[tuple[int, mx.array, bool], None, None]:
    """Cross-round speculative decoding loop using ``transport``.

    See module docstring for the cache-accounting derivation. This
    function maintains:

      * ``drafts``: list[int] of length K_this -- this round's drafts.
      * ``seed``: int -- the seed token for this round (target verify
        consumes ``[seed, *drafts]``).
      * ``next_round_inputs``: list[int] -- input shape for next round's
        propose call (length 1 for partial-accept-from-this, length 2
        for full-accept-from-this).
      * ``speculative_future``: optional Future from a speculative
        forward issued in parallel with target verify. ``None`` when
        speculation is not in flight.

    ``prompt_token_count`` is captured so logits processors that need
    the running token count (rare, e.g. positional repetition penalty
    that scales with absolute position) get accurate values.

    Multi-target asymmetric (``target_group is not None``): only the
    target root rank holds the drafter ``transport``; non-root target
    ranks pass ``transport=None`` and receive each round's drafts via
    a rank-0 broadcast on ``target_group``. Both ranks then run the
    verify forward in TP lockstep -- the model's final all-reduce
    makes logits byte-identical across target ranks, so accept/reject
    decisions and emitted token sequences match deterministically
    without any further coordination.
    """
    if (transport is None) and is_target_root:
        raise RuntimeError(
            "_pipelined_speculative_step: target root requires transport"
        )
    if (transport is None) and target_group is None:
        raise RuntimeError(
            "_pipelined_speculative_step: non-root target rank requires "
            "target_group to receive draft broadcasts"
        )

    k = num_draft_tokens
    y = prompt.astype(mx.uint32)

    _diag_rank = (
        target_group.rank()
        if target_group is not None
        else (0 if is_target_root else -1)
    )
    _spec_diag(
        f"rank {_diag_rank}: spec body entered "
        f"(prompt size={int(prompt.size)}, k={k}, root={is_target_root})"
    )

    # Mirror mlx_lm._prefill: caller has aligned ``prompt_cache`` to
    # ``context_tokens[:-2]`` via ``exo.prefill`` + ``trim(2)``; this loop
    # advances the cache by one more token, leaving ``y`` (length 1) as
    # the seed for the spec loop.
    _diag_prefill_iters = 0
    while y.size > 1:
        _diag_prefill_t0 = time.perf_counter()
        n_to_process = min(prefill_step_size, y.size - 1)
        model(y[:n_to_process][None], cache=prompt_cache)
        mx.eval([c.state for c in prompt_cache])  # type: ignore[reportArgumentType]
        y = y[n_to_process:]
        mx.clear_cache()
        _spec_diag(
            f"rank {_diag_rank}: spec-body prefill iter "
            f"{_diag_prefill_iters} done in "
            f"{(time.perf_counter() - _diag_prefill_t0) * 1000:.1f}ms "
            f"(remaining y.size={int(y.size)})"
        )
        _diag_prefill_iters += 1

    _diag_seed_t0 = time.perf_counter()
    seed = int(y.item())
    _spec_diag(
        f"rank {_diag_rank}: seed materialized in "
        f"{(time.perf_counter() - _diag_seed_t0) * 1000:.1f}ms (seed={seed})"
    )
    # ``prev_tokens`` carries the running token sequence (prompt +
    # emitted) so logits processors with state see consistent context.
    # Mirror :func:`drafter._ngram_speculative_step`: start from prompt.
    prev_tokens = mx.array([seed], dtype=mx.uint32)
    del prompt_token_count  # currently unused; kept for forward-compat

    # Round 0 propose: synchronous, no speculation possible yet because
    # we don't have prior drafts to chain off of. On the root the
    # drafter forward issues a socket round-trip; on non-root target
    # ranks we skip that and just receive the broadcast.
    if transport is not None:
        _diag_fwd_t0 = time.perf_counter()
        _spec_diag(
            f"rank {_diag_rank}: round 0 about to call transport.forward([seed], k={k})"
        )
        drafts_future = transport.forward([seed], k)
        drafts_local: list[int] | None = drafts_future.result()
        _spec_diag(
            f"rank {_diag_rank}: round 0 transport.forward "
            f"returned in {(time.perf_counter() - _diag_fwd_t0) * 1000:.1f}ms "
            f"(drafts_local len={len(drafts_local) if drafts_local else 0})"
        )
    else:
        drafts_local = None
    _diag_bcast_t0 = time.perf_counter()
    _spec_diag(
        f"rank {_diag_rank}: round 0 about to call "
        f"_broadcast_drafts (root={is_target_root})"
    )
    drafts = _broadcast_drafts(
        drafts_local,
        k=k,
        target_group=target_group,
        target_peer_fanout=target_peer_fanout,
        is_root=is_target_root,
    )
    _spec_diag(
        f"rank {_diag_rank}: round 0 _broadcast_drafts done "
        f"in {(time.perf_counter() - _diag_bcast_t0) * 1000:.1f}ms "
        f"(drafts len={len(drafts)})"
    )

    speculative_future: DraftFuture | None = None
    ntoks = 0
    _diag_round = 0

    while ntoks < max_tokens:
        budget = max_tokens - ntoks
        k_this = min(k, len(drafts), budget)
        if k_this < 1:
            break
        drafts = drafts[:k_this]
        _diag_round += 1
        _spec_diag(
            f"rank {_diag_rank}: round {_diag_round} top "
            f"(ntoks={ntoks}, k_this={k_this})"
        )

        # ----- Cross-round speculation: dispatch in parallel with verify -----
        # Speculate only when:
        #   * full k_this drafts (truncated last rounds have no t+1 to feed),
        #   * budget remains for an entire next round's verify after this one.
        #
        # The speculative forward consumes ``drafts[-1]`` (= drafter's last
        # draft this round) as its first input, doing k+1 forwards. The
        # first output is the drafter's prediction of bonus_t (used to
        # detect speculation hit); the remaining k outputs are round
        # t+1's drafts if speculation hits.
        #
        # Speculation only fires on the rank that owns the transport.
        # Non-root target ranks have no socket and would have nothing
        # to dispatch; they catch up via the next-round broadcast.
        speculation_active = (
            transport is not None
            and k_this == k
            and ntoks + (k_this + 1) + k + 1 <= max_tokens
            and speculative_future is None
        )
        if speculation_active:
            assert transport is not None  # narrowed by speculation_active
            speculative_future = transport.forward([drafts[-1]], k + 1)

        # ----- Target verify -----
        seed_arr = mx.array([seed], dtype=mx.uint32)
        draft_arr = mx.array(drafts, dtype=mx.uint32)
        verify_input = mx.concatenate([seed_arr, draft_arr])
        _diag_verify_t0 = time.perf_counter()
        logits = model(verify_input[None], cache=prompt_cache)
        # CRITICAL: force eval of ``logits`` on every target rank so the
        # TP all-reduce kernels embedded in ``model()`` actually launch
        # before any rank proceeds to its next blocking step. Without
        # this, non-root ranks dispatch the verify forward (lazy graph
        # only) and then enter the TCP recv in ``_broadcast_target_tokens``,
        # leaving the all-reduce un-launched on their side. The root
        # rank's ``mx.eval(sampled_batch)`` then deadlocks waiting for
        # the matching all-reduce on every peer. This mirrors the
        # prefill loop's ``mx.eval([c.state for c in prompt_cache])``,
        # which is what made the round-0 prefill collectives pair up
        # correctly on both ranks. Cost: one synchronization per round
        # (~the verify forward time, which we'd block on at the sampler
        # step anyway on root). Benefit: guaranteed pairing of TP
        # collectives across all target ranks under JACCL or ring.
        mx.eval(logits)
        _spec_diag(
            f"rank {_diag_rank}: round {_diag_round} model(verify) + eval "
            f"completed in {(time.perf_counter() - _diag_verify_t0) * 1000:.1f}ms "
            f"(verify_len={k_this + 1})"
        )
        # logits shape: (1, k_this + 1, vocab)

        target_logprobs: list[mx.array]
        target_tokens: list[int]
        # Fast path: every processor advertises position independence
        # (or there are none). Apply them once to the batched
        # ``(K+1, vocab)`` logits, sample all positions in one call,
        # and pay a single host-device sync per round instead of K+1.
        # On a target with ~10ms step time this saves ~10-15ms per
        # round -- typically the difference between net-win and net-loss
        # for spec-decode on fast quantised targets.
        #
        # Multi-target determinism: ``logits`` is byte-identical across
        # target ranks because the model's final layer all-reduces it
        # via TP. Logits processors are pure functions of ``logits`` and
        # ``prev_tokens`` (also identical across ranks), so logprobs are
        # identical too. The sampler is the only non-deterministic step
        # (``mx.random.categorical`` uses MLX's per-rank PRNG). Rank 0
        # samples; non-root ranks skip the sampler and pick up tokens
        # from the broadcast below. Logprobs are still computed locally
        # on every rank because they're cheap and the yield contract
        # passes them upward (the user only ever sees rank 0's, but
        # keeping the local view matches the single-rank path).
        position_independent = all(
            getattr(p, "position_independent", False) for p in logits_processors
        )
        if position_independent:
            batched_logits = logits.squeeze(0)
            for proc in logits_processors:
                batched_logits = proc(prev_tokens, batched_logits)
            batched_logprobs = batched_logits - mx.logsumexp(
                batched_logits, axis=-1, keepdims=True
            )
            target_logprobs = [batched_logprobs[i] for i in range(k_this + 1)]
            if is_target_root:
                _diag_sample_t0 = time.perf_counter()
                _spec_diag(
                    f"rank {_diag_rank}: round {_diag_round} root: about to "
                    f"call sampler(batched_logprobs)"
                )
                sampled_batch = sampler(batched_logprobs)
                _spec_diag(
                    f"rank {_diag_rank}: round {_diag_round} root: about to "
                    f"mx.eval(sampled_batch) (this forces verify forward + "
                    f"all_sum to actually run)"
                )
                mx.eval(sampled_batch)
                _spec_diag(
                    f"rank {_diag_rank}: round {_diag_round} root: "
                    f"mx.eval(sampled_batch) done in "
                    f"{(time.perf_counter() - _diag_sample_t0) * 1000:.1f}ms"
                )
                target_tokens = [int(t) for t in sampled_batch.tolist()]  # type: ignore[reportUnknownArgumentType]
            else:
                # Filled by broadcast below; skip the sampler entirely.
                target_tokens = []
                _spec_diag(
                    f"rank {_diag_rank}: round {_diag_round} non-root: "
                    f"skipped sampler, awaiting broadcast"
                )
        else:
            # Stateful path: logits processors (e.g. repetition penalty)
            # depend on ``running_prev`` which only resolves between
            # positions, so we can't batch. Per-position sync is the
            # cost of correctness here.
            #
            # Cross-rank determinism subtlety: the loop's ``running_prev``
            # advances by the sampled token at each position. On rank 0
            # we sample to advance it; on non-root ranks we don't have
            # the token yet (the broadcast happens after the loop), so
            # we'd advance with the wrong tokens. To keep the per-rank
            # codepath identical we sample on every rank and broadcast
            # after; the broadcast then overwrites ``target_tokens`` so
            # downstream accept/reject is identical. Per-rank sampler
            # divergence inside this loop is harmless because nothing
            # consumes ``target_tokens`` between sampler call and
            # broadcast; it gets clobbered before use.
            target_logprobs = []
            target_tokens = []
            running_prev = prev_tokens
            for i in range(k_this + 1):
                position_logits = logits[:, i, :].squeeze(0)
                position_logprobs = _process_logits_for_position(
                    position_logits, running_prev, logits_processors
                )
                sampled = sampler(position_logprobs)
                mx.eval(sampled)
                sampled_token = int(sampled.item())
                target_logprobs.append(position_logprobs)
                target_tokens.append(sampled_token)
                running_prev = mx.concatenate(
                    [running_prev, mx.array([sampled_token], dtype=mx.uint32)]
                )

        # Broadcast rank-0's chosen tokens to every target rank so
        # accept/reject decisions are bit-identical. Single-rank
        # placements (``target_group is None``) short-circuit to
        # identity, so this is free for the non-multi-target paths.
        _diag_tbcast_t0 = time.perf_counter()
        _spec_diag(
            f"rank {_diag_rank}: round {_diag_round} about to call "
            f"_broadcast_target_tokens (root={is_target_root})"
        )
        target_tokens = _broadcast_target_tokens(
            target_tokens if is_target_root else None,
            k=k,
            k_this=k_this,
            target_group=target_group,
            target_peer_fanout=target_peer_fanout,
            is_root=is_target_root,
        )
        _spec_diag(
            f"rank {_diag_rank}: round {_diag_round} _broadcast_target_tokens "
            f"done in {(time.perf_counter() - _diag_tbcast_t0) * 1000:.1f}ms "
            f"(target_tokens len={len(target_tokens)})"
        )

        # ----- Greedy accept loop -----
        num_accepted = 0
        for i in range(k_this):
            if target_tokens[i] == drafts[i]:
                num_accepted += 1
            else:
                break

        # Per-round telemetry: ``k_this`` drafts proposed,
        # ``num_accepted`` accepted by the greedy verifier. The bonus
        # token (target's correction or full-accept tail) is *not* a
        # draft, so it doesn't count against acceptance rate. Mutates
        # the caller's dict in place; ``metrics is None`` for the
        # synthetic single-rank tests that bypass the drafter wrapper.
        if metrics is not None:
            metrics["proposed_draft_tokens"] += k_this
            metrics["accepted_draft_tokens"] += num_accepted
            metrics["spec_decode_rounds"] += 1

        # ----- Emit accepted drafts + correction/bonus -----
        emit_count = num_accepted + 1
        for j in range(emit_count):
            tok = drafts[j] if j < num_accepted else target_tokens[j]
            from_draft = j < num_accepted
            yield tok, target_logprobs[j], from_draft
            prev_tokens = mx.concatenate(
                [prev_tokens, mx.array([tok], dtype=mx.uint32)]
            )
            ntoks += 1
            if ntoks >= max_tokens:
                break

        # ----- Target cache trim (rejected draft positions) -----
        # Verify forward extended target cache by k_this + 1; we keep
        # ``num_accepted + 1`` of those (= emit_count) so trim
        # ``k_this - num_accepted``.
        target_trim = k_this - num_accepted
        if target_trim > 0:
            mlx_trim_prompt_cache(cast(list[object], prompt_cache), target_trim)  # type: ignore[reportArgumentType]

        if ntoks >= max_tokens:
            # Discard any in-flight speculation; we're done. Rolling back
            # the drafter cache isn't strictly necessary (the loop is
            # exiting), but keeps the cache in a consistent state for
            # any subsequent runs that might reuse the transport.
            if speculative_future is not None:
                _drain_future(speculative_future)
                assert transport is not None  # speculative_future is set only on root
                transport.trim_cache(k + 1)
                speculative_future = None
            break

        # ``next_seed`` is the target's chosen token at the rejection
        # point (partial accept) or the bonus position (full accept).
        next_seed: int
        if num_accepted == k_this:
            next_seed = target_tokens[k_this]
        else:
            next_seed = target_tokens[num_accepted]

        # ----- Drafter cache reconciliation + next-round setup -----
        # Only the root rank touches ``transport``; non-root target
        # ranks compute ``next_drafts_local = None`` and pick up the
        # actual drafts from the rank-0 broadcast below.
        next_drafts_local: list[int] | None
        if transport is not None:
            if num_accepted < k_this:
                # Partial accept (regardless of speculation state).
                drafter_trim_partial = max(k_this - num_accepted - 1, 0)
                if speculative_future is not None:
                    # Speculative work is bound to a different (assumed-
                    # full-accept) future; discard it and trim its k+1
                    # positions plus the partial-accept trim.
                    _drain_future(speculative_future)
                    transport.trim_cache(k + 1 + drafter_trim_partial)
                    speculative_future = None
                elif drafter_trim_partial > 0:
                    transport.trim_cache(drafter_trim_partial)
                next_drafts_local = transport.forward([next_seed], k).result()
            else:
                # Full accept at this round.
                if speculative_future is not None:
                    spec_outputs = speculative_future.result()
                    speculative_future = None
                    bonus_predicted = spec_outputs[0]
                    if bonus_predicted == next_seed:
                        # SPECULATION HIT. Round t+1's drafts come for free.
                        # Drafter cache state is correct (offset O+2k+1
                        # matches what a length-2-seed propose for round
                        # t+1 would produce).
                        next_drafts_local = spec_outputs[1 : k + 1]
                    else:
                        # SPECULATION MISS. Rollback the k+1 speculative
                        # positions and run a standard length-2-seed
                        # propose for round t+1.
                        transport.trim_cache(k + 1)
                        next_drafts_local = transport.forward(
                            [drafts[-1], next_seed], k
                        ).result()
                else:
                    # Full accept, speculation was inactive. Standard
                    # length-2-seed propose for round t+1.
                    next_drafts_local = transport.forward(
                        [drafts[-1], next_seed], k
                    ).result()
        else:
            next_drafts_local = None

        _diag_nbcast_t0 = time.perf_counter()
        _spec_diag(
            f"rank {_diag_rank}: round {_diag_round} about to call "
            f"_broadcast_drafts (next, accepted={num_accepted}/{k_this})"
        )
        next_drafts = _broadcast_drafts(
            next_drafts_local,
            k=k,
            target_group=target_group,
            target_peer_fanout=target_peer_fanout,
            is_root=is_target_root,
        )
        _spec_diag(
            f"rank {_diag_rank}: round {_diag_round} _broadcast_drafts (next) "
            f"done in {(time.perf_counter() - _diag_nbcast_t0) * 1000:.1f}ms "
            f"(next_drafts len={len(next_drafts)})"
        )

        seed = next_seed
        drafts = next_drafts


def _drain_future(future: DraftFuture) -> None:
    """Block on ``future`` and discard its result.

    Used when speculation misses or the loop exits early: the drafter
    forwards have already executed; we just need to ensure the future
    is resolved before issuing dependent transport operations
    (``trim_cache``, ``shutdown``). Exceptions from the forwards
    surface elsewhere (transport's own error path); we suppress them
    here to avoid double-reporting.
    """
    import contextlib

    with contextlib.suppress(Exception):
        future.result()


__all__ = ["PipelinedModelDrafter"]
