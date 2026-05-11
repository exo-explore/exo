import contextlib
import os
import socket
from collections.abc import Generator
from dataclasses import dataclass
from typing import cast

import mlx.core as mx
from mlx_lm.tokenizer_utils import TokenizerWrapper

from exo.shared.constants import EXO_MAX_CONCURRENT_REQUESTS
from exo.shared.types.common import ModelId
from exo.shared.types.events import Event
from exo.shared.types.tasks import TaskId
from exo.shared.types.worker.instances import BoundInstance
from exo.shared.types.worker.runner_response import ModelLoadingResponse
from exo.utils.channels import MpReceiver, MpSender
from exo.worker.engines.base import Builder, Engine
from exo.worker.runner.bootstrap import logger
from exo.worker.runner.llm_inference.batch_generator import (
    DEFAULT_DRAFTER_MIN_OUTPUT_TOKENS,
    DEFAULT_NUM_DRAFT_TOKENS,
    EXO_ADAPTIVE_DRAFT_TOKENS,
    EXO_DRAFTER_MIN_OUTPUT_TOKENS,
    EXO_NUM_DRAFT_TOKENS,
    BatchGenerator,
    SequentialGenerator,
    parse_env_int,
)
from exo.worker.runner.llm_inference.tool_parsers import make_mlx_parser

from .cache import KVPrefixCache
from .generator.coupled_drafter import is_coupled_drafter_dispatchable
from .generator.drafter import EXO_DRAFT_MODE_ENV, parse_draft_mode
from .types import Model
from .utils_mlx import (
    CoupledDrafter,
    initialize_mlx,
    load_mlx_items,
)
from .vision import VisionProcessor


@dataclass
class MlxBuilder(Builder):
    model_id: ModelId
    event_sender: MpSender[Event]
    cancel_receiver: MpReceiver[TaskId]
    inference_model: Model | None = None
    tokenizer: TokenizerWrapper | None = None
    # ``group`` is the target ranks' ``mx.distributed.Group``: pipeline
    # / tensor / batch collectives all run on it. Under the v3+ wire
    # the drafter is NOT a member of this group (asymmetric drafters
    # talk to target rank 0 over a TCP socket; see ``drafter_socket``
    # below).
    group: mx.distributed.Group | None = None
    # Connected TCP socket from target rank 0 to the drafter rank.
    # Set ONLY on target rank 0 of an asymmetric placement; ``None``
    # everywhere else (other target ranks don't drive drafter IPC, and
    # single-device / symmetric multi-rank builds have no drafter
    # wire at all).
    drafter_socket: socket.socket | None = None
    drafter_rank_in_parent: int | None = None
    # Inter-target-rank TCP fanout for the spec-decode int-broadcast
    # wire. Allocated by :func:`initialize_mlx` on multi-target
    # asymmetric placements; ``None`` for single-target / symmetric
    # builds. See :class:`TargetPeerFanout`.
    target_peer_fanout: object | None = None
    vision_processor: VisionProcessor | None = None
    draft_model: Model | None = None
    draft_model_id: ModelId | None = None
    # Coupled (mtp/dflash) drafter loaded via mlx-vlm. Mutually exclusive
    # with ``draft_model`` at the loader level: ``load_mlx_items`` tries
    # the coupled path first when the card declares ``coupled_drafter``
    # and falls back to the standard external drafter only on coupled
    # load failure (or when the card declares only the legacy list).
    #
    # Phase 2a foundation: this field is populated by the loader and
    # forwarded into ``SequentialGenerator``, but neither the builder
    # gate (BatchGenerator vs SequentialGenerator) nor ``mlx_generate``
    # itself yet reads it -- they see it as if it were ``None``. The
    # follow-up adds the round loop on top of vendored
    # ``rollback_speculative_cache`` + extended forward kwargs in the
    # mlx-lm fork's gemma4_text.py.
    coupled_drafter: CoupledDrafter | None = None

    def connect(self, bound_instance: BoundInstance) -> None:
        split = initialize_mlx(bound_instance)
        self.group = split.target_subgroup
        # Only target rank 0 in an asymmetric placement holds a drafter
        # socket; every other rank sees ``None`` here. ``MlxGroupSplit``
        # types it as ``object | None`` to keep the dataclass importable
        # without ``socket``; cast back to the concrete type for
        # consumers.
        if split.drafter_socket is not None:
            self.drafter_socket = cast(socket.socket, split.drafter_socket)
        else:
            self.drafter_socket = None
        self.drafter_rank_in_parent = split.drafter_rank_in_parent
        self.target_peer_fanout = split.target_peer_fanout

    def load(self, bound_instance: BoundInstance) -> Generator[ModelLoadingResponse]:
        (
            self.inference_model,
            self.tokenizer,
            self.vision_processor,
            self.draft_model,
            self.draft_model_id,
            self.coupled_drafter,
        ) = yield from load_mlx_items(bound_instance, self.group)

    def close(self) -> None:
        # Drop drafters BEFORE the target / tokenizer / group: coupled
        # drafters bind to the target's input embeddings via mlx-vlm's
        # ``bind`` so they hold a strong reference into the target;
        # standard drafters can hold a weak reference into the target's
        # mx.distributed.Group on multi-rank builds. Reordering this
        # after ``del self.inference_model`` triggered an
        # ``AttributeError`` chain in PR #20 round-(N+10) -- preserve
        # that invariant here even though Phase 2a doesn't yet exercise
        # the coupled path through the generator.
        with contextlib.suppress(NameError, AttributeError):
            del self.coupled_drafter
        with contextlib.suppress(NameError, AttributeError):
            del self.draft_model
        with contextlib.suppress(NameError, AttributeError):
            del self.inference_model
        with contextlib.suppress(NameError, AttributeError):
            del self.tokenizer
        with contextlib.suppress(NameError, AttributeError):
            del self.group
        if self.drafter_socket is not None:
            with contextlib.suppress(OSError):
                self.drafter_socket.close()
            self.drafter_socket = None

    def build(
        self,
    ) -> Engine:
        assert self.inference_model
        assert self.tokenizer

        vision_processor = self.vision_processor

        tool_parser = None
        logger.info(
            f"model has_tool_calling={self.tokenizer.has_tool_calling} using tokens {self.tokenizer.tool_call_start}, {self.tokenizer.tool_call_end}"
        )
        if (
            self.tokenizer.tool_call_start
            and self.tokenizer.tool_call_end
            and self.tokenizer.tool_parser  # type: ignore
        ):
            tool_parser = make_mlx_parser(
                self.tokenizer.tool_call_start,
                self.tokenizer.tool_call_end,
                self.tokenizer.tool_parser,  # type: ignore
            )

        kv_prefix_cache = KVPrefixCache(self.group)
        # Item 6: dedicated KVPrefixCache for the drafter so multi-turn
        # workloads don't repeatedly prefill the drafter on the same prefix.
        # Allocated only when a drafter is actually loaded; None means
        # mlx_generate falls back to the per-request drafter prefill.
        #
        # Coupled drafters (mtp/dflash) have no independent KV cache --
        # ``mtp`` reads the target's KV via ``set_shared_kv`` and ``dflash``
        # owns a tiny per-step cache that's reset every round -- so they
        # need no KVPrefixCache. The generator-side dispatch handles that
        # branch separately and never reads ``drafter_kv_prefix_cache``
        # for coupled drafters.
        drafter_kv_prefix_cache: KVPrefixCache | None = (
            KVPrefixCache(self.group) if self.draft_model is not None else None
        )

        device_rank = 0 if self.group is None else self.group.rank()

        # Speculative decoding (model or n-gram) currently flows only through
        # SequentialGenerator -> mlx_generate. Upstream BatchGenerator does
        # not accept a draft model and has no hook for n-gram drafting, so
        # force the sequential path whenever speculative decoding could
        # plausibly run for any request: a drafter model is loaded *or*
        # ``EXO_DRAFT_MODE=ngram`` is set process-wide *or* the operator
        # opted into request-level draft overrides via
        # ``EXO_ALLOW_REQUEST_DRAFTING``. Per-request overrides
        # (``TaskParams.draft_mode``) only apply within the surface that
        # the chosen generator exposes.
        #
        # Codex P2 (PR #19 round 2): without ``EXO_ALLOW_REQUEST_DRAFTING``
        # a node started in normal batch mode silently dropped
        # ``draft_mode="ngram"`` request overrides because BatchGenerator
        # has no spec-decoding hook. This broke the newly added
        # API-level override path for A/B tests and mixed traffic. The
        # opt-in trades batching for per-request spec-decoding control;
        # operators who don't need request-level spec stay on
        # BatchGenerator with the default settings.
        #
        # Codex P1 (PR #19 round-(N+3), builder.py:136): on multi-device
        # runners, ``mlx_generate`` unconditionally demotes
        # ``draft_mode`` to ``"none"`` (see ``generate.py``: ``if group
        # is not None: draft_mode = "none"``), so swapping to
        # ``SequentialGenerator`` for drafting buys nothing and only
        # loses batching. PR #20 reintroduces speculative decoding for
        # asymmetric placements, but PR #19 stand-alone has no
        # multi-device drafter path. Gate the sequential fallback on
        # single-device runners; multi-device nodes keep
        # ``BatchGenerator`` regardless of ``EXO_DRAFT_MODE`` /
        # ``EXO_ALLOW_REQUEST_DRAFTING`` so concurrent traffic doesn't
        # silently lose throughput.
        #
        # Codex P1 (PR #19 round-(N+6), builder.py:151): drop
        # ``configured_draft_mode == "ngram"`` from the
        # force-sequential trigger. ``mlx_generate`` now demotes
        # ``draft_mode="ngram"`` to ``"none"`` for any non-greedy
        # request (see :func:`_request_is_greedy_sampling`), and the
        # default sampler path uses ``temperature=0.7`` when the
        # request omits temperature. So a worker booted with
        # ``EXO_DRAFT_MODE=ngram`` against mixed traffic would
        # disable batching for the entire worker yet only run
        # speculation for the (rare) greedy subset -- a strict
        # throughput regression for the common case. n-gram remains
        # opt-in via ``EXO_NO_BATCH=1`` (operators who explicitly
        # want greedy-only n-gram acceleration) or
        # ``EXO_ALLOW_REQUEST_DRAFTING=1`` (per-request override
        # path); without either, ngram requests fall back to plain
        # decode under BatchGenerator and the worker keeps full
        # batching throughput. Emit a warning when this condition
        # holds so operators know n-gram won't actually run.
        # Phase 2c re-enables the coupled-drafter influence on builder-
        # side gates: now that ``mlx_generate`` dispatches coupled
        # (mtp/dflash) drafters through :class:`CoupledModelDrafter`,
        # treating a loaded coupled drafter as "drafter loaded" both
        # (a) flips the implicit ``draft_mode`` default to ``"model"``
        # so single-node Gemma 4 deployments pick up the speedup
        # automatically and (b) forces :class:`SequentialGenerator`
        # over :class:`BatchGenerator` since the latter has no
        # spec-decode hook. The coupled-drafter check is OR'd with
        # ``draft_model`` everywhere downstream so the existing standard-
        # drafter-only deployments are unaffected.
        #
        # Codex P2 (PR #25 round-(N+3), builder.py:241): gate the
        # coupled signal on the drafter being DISPATCHABLE, not just
        # loaded. The loader accepts both ``"mtp"`` and ``"dflash"``
        # but the generator dispatch only drives ``"mtp"`` today --
        # ``"dflash"`` falls back to ``make_drafter(mode="none")``
        # inside :func:`mlx_generate`. Without this gate, a
        # dflash-only setup would force :class:`SequentialGenerator`
        # (losing batch throughput) while requests actually run plain
        # decoding. ``is_coupled_drafter_dispatchable`` mirrors the
        # generator's own dispatch check.
        coupled_drafter_dispatchable = (
            self.coupled_drafter is not None
            and is_coupled_drafter_dispatchable(self.coupled_drafter.kind)
        )
        any_drafter_loaded = (
            self.draft_model is not None or coupled_drafter_dispatchable
        )
        configured_draft_mode = parse_draft_mode(
            os.environ.get(EXO_DRAFT_MODE_ENV),
            default="model" if any_drafter_loaded else "none",
        )
        allow_request_drafting = os.environ.get(
            "EXO_ALLOW_REQUEST_DRAFTING", ""
        ).lower() in {"1", "true", "yes"}
        is_single_device = self.group is None or self.group.size() == 1

        # Asymmetric placement: drafter lives on a separate node; only
        # target rank 0 owns the drafter wire (``drafter_socket``).
        # Force the SequentialGenerator path (BatchGenerator has no
        # spec-decoding hook) and build a long-lived RemoteTransport
        # that the spec loop reuses across requests.
        #
        # Other target ranks in an asymmetric placement (rank >= 1) see
        # ``drafter_socket is None`` and treat their build the same as
        # symmetric multi-rank: they participate in target collectives
        # but never call drafter ops directly. The spec loop's
        # rank-0-only sampling decision keeps that invariant.
        is_asymmetric_target_rank_zero = self.drafter_socket is not None
        is_asymmetric = (
            is_asymmetric_target_rank_zero or self.drafter_rank_in_parent is not None
        )

        # Conflict-merge note (PR #20 round-(N+12)): combines two
        # gates on the path that forces ``SequentialGenerator`` over
        # ``BatchGenerator``:
        #
        #   * PR #19's single-device-only sequential gate: in-process
        #     standard / n-gram drafting can only run on single-device
        #     runners because ``mlx_generate`` demotes ``draft_mode``
        #     to ``"none"`` when no coupled drafter is loaded on the
        #     multi-device branch. The gate honours
        #     ``EXO_DRAFT_MODE=none`` to avoid losing batching with
        #     zero speculative-decode benefit.
        #   * PR #20's asymmetric-pipelined gate: when the runner is
        #     a target rank in an asymmetric placement, batching is
        #     incompatible with the drafter wire, so the sequential
        #     path is mandatory regardless of ``draft_model`` /
        #     ``EXO_DRAFT_MODE``.
        #   * Coupled-drafter tensor-parallel gate: a coupled drafter
        #     (MTP / DFlash) replicates per rank and consumes the
        #     post-all-reduce hidden state in-process. ``mlx_generate``
        #     accepts this for ``group is not None`` placements (see
        #     ``coupled_drafter_eligible`` there), so we must force
        #     ``SequentialGenerator`` on TP runners that load a coupled
        #     drafter -- ``BatchGenerator`` has no spec-decoding hook.
        drafting_can_run_here = is_single_device or coupled_drafter_dispatchable
        drafter_loaded_will_run = any_drafter_loaded and configured_draft_mode != "none"
        force_sequential_for_drafter = drafting_can_run_here and (
            drafter_loaded_will_run
            or allow_request_drafting
            or configured_draft_mode == "pipelined"
        )
        ngram_configured_without_force_sequential = (
            drafting_can_run_here
            and configured_draft_mode == "ngram"
            and not force_sequential_for_drafter
        )
        drafter_loaded_but_explicitly_disabled = (
            drafting_can_run_here
            and any_drafter_loaded
            and configured_draft_mode == "none"
            and not allow_request_drafting
        )

        # Long-lived ``RemoteTransport`` (NOT a per-task DrafterTransport).
        # Each in-flight request opens its own session via
        # :meth:`RemoteTransport.open_session`; the session handle is the
        # actual DrafterTransport consumed by the spec loop. See
        # ``remote_drafter.py`` module docstring for the wire-protocol
        # session multiplexing rationale.
        from exo.worker.engines.mlx.generator.remote_drafter import RemoteTransport

        remote_drafter_transport: RemoteTransport | None = None
        if is_asymmetric_target_rank_zero:
            assert self.drafter_socket is not None
            from exo.worker.engines.mlx.generator.remote_drafter import (
                make_remote_transport,
            )

            num_draft_tokens_remote = parse_env_int(
                EXO_NUM_DRAFT_TOKENS, DEFAULT_NUM_DRAFT_TOKENS
            )
            target_world_size = self.group.size() if self.group is not None else 1
            logger.info(
                "Allocating long-lived RemoteTransport: "
                f"target_world_size={target_world_size} "
                f"drafter_rank={self.drafter_rank_in_parent} "
                f"K={num_draft_tokens_remote} "
                f"transport=tcp_socket"
            )
            remote_drafter_transport = make_remote_transport(
                draft_model=None,
                draft_cache=None,
                num_draft_tokens=num_draft_tokens_remote,
                sock=self.drafter_socket,
            )

        if (
            os.environ.get("EXO_NO_BATCH")
            or force_sequential_for_drafter
            or is_asymmetric
        ):
            if is_asymmetric:
                logger.info(
                    "using SequentialGenerator (asymmetric placement: "
                    "drafter lives on a separate MLX rank, pipelined+remote spec)"
                )
            elif force_sequential_for_drafter:
                if allow_request_drafting and not any_drafter_loaded:
                    logger.info(
                        "using SequentialGenerator (EXO_ALLOW_REQUEST_DRAFTING set; "
                        "BatchGenerator has no spec-decoding hook for request "
                        "overrides)"
                    )
                elif coupled_drafter_dispatchable:
                    assert self.coupled_drafter is not None  # narrowed by gate
                    logger.info(
                        f"using SequentialGenerator (coupled drafter loaded: "
                        f"{self.coupled_drafter.model_id} kind={self.coupled_drafter.kind!r}; "
                        f"draft_mode={configured_draft_mode!r}; BatchGenerator "
                        f"has no spec-decoding hook for coupled MTP/DFlash)"
                    )
                else:
                    logger.info(
                        f"using SequentialGenerator (draft_mode={configured_draft_mode!r}; "
                        f"BatchGenerator has no spec-decoding hook)"
                    )
            else:
                logger.info("using SequentialGenerator (batching disabled)")

            num_draft_tokens = parse_env_int(
                EXO_NUM_DRAFT_TOKENS, DEFAULT_NUM_DRAFT_TOKENS
            )
            drafter_min_output_tokens = parse_env_int(
                EXO_DRAFTER_MIN_OUTPUT_TOKENS,
                DEFAULT_DRAFTER_MIN_OUTPUT_TOKENS,
                minimum=0,
            )
            adaptive_draft_tokens = os.environ.get(
                EXO_ADAPTIVE_DRAFT_TOKENS, ""
            ).lower() in {"1", "true", "yes"}
            if force_sequential_for_drafter or is_asymmetric:
                logger.info(
                    f"speculative decoding: mode={'pipelined+remote' if is_asymmetric else configured_draft_mode}, "
                    f"K={num_draft_tokens} (adaptive={adaptive_draft_tokens}), "
                    f"skip_drafter_when_max_tokens<={drafter_min_output_tokens}"
                )

            # Concurrent in-flight tasks. Asymmetric pipelined+remote
            # rides the same ``EXO_MAX_CONCURRENT_REQUESTS`` cap as every
            # other config now that the wire protocol carries a
            # ``session_id`` slot: each in-flight target request opens
            # its own ``_SessionHandle`` via
            # ``RemoteTransport.open_session()`` and the drafter rank
            # multiplexes per-session KV caches. The wire stays serial
            # (single ``ThreadPoolExecutor`` on the target, single recv
            # loop on the drafter) so ``mx.distributed.send/recv``
            # ordering is preserved; concurrency comes from interleaving
            # forward / verify rounds across sessions, which is the
            # whole point of asymmetric placement -- keep the drafter
            # rank busy serving session A while the target verifies
            # session B's drafts.
            max_concurrent_tasks = EXO_MAX_CONCURRENT_REQUESTS
            if max_concurrent_tasks > 1:
                logger.info(
                    f"SequentialGenerator round-robin concurrency: "
                    f"max_concurrent_tasks={max_concurrent_tasks} "
                    f"(EXO_MAX_CONCURRENT_REQUESTS)"
                )

            return SequentialGenerator(
                model=self.inference_model,
                tokenizer=self.tokenizer,
                group=self.group,
                tool_parser=tool_parser,
                kv_prefix_cache=kv_prefix_cache,
                model_id=self.model_id,
                device_rank=device_rank,
                cancel_receiver=self.cancel_receiver,
                event_sender=self.event_sender,
                vision_processor=vision_processor,
                draft_model=self.draft_model,
                draft_model_id=self.draft_model_id,
                coupled_drafter=self.coupled_drafter,
                drafter_kv_prefix_cache=drafter_kv_prefix_cache,
                num_draft_tokens=num_draft_tokens,
                drafter_min_output_tokens=drafter_min_output_tokens,
                adaptive_draft_tokens=adaptive_draft_tokens,
                drafter_rank_in_parent=self.drafter_rank_in_parent,
                remote_drafter_transport=remote_drafter_transport,
                target_peer_fanout=self.target_peer_fanout,
                max_concurrent_tasks=max_concurrent_tasks,
            )
        else:
            # Codex P1 (PR #19 round-(N+3), builder.py:136): make the
            # multi-device drafting-disabled path explicit so operators
            # don't silently observe missing speculative decoding.
            drafting_was_requested = (
                any_drafter_loaded
                or configured_draft_mode == "ngram"
                or allow_request_drafting
            )
            if not drafting_can_run_here and drafting_was_requested:
                logger.info(
                    f"using BatchGenerator (drafting unavailable on multi-device "
                    f"runner: group.size={self.group.size() if self.group is not None else 1}; "
                    f"mlx_generate would demote draft_mode='none' anyway, keeping "
                    f"batching for throughput)"
                )
            elif drafter_loaded_but_explicitly_disabled:
                # Codex P1 (PR #19 round-(N+8), builder.py:169): a
                # drafter model is loaded but the operator set
                # ``EXO_DRAFT_MODE=none``, so every request resolves
                # to ``draft_mode="none"`` in ``mlx_generate``.
                # SequentialGenerator would lose batching for
                # nothing in this configuration. Keep
                # BatchGenerator and surface the choice loudly so
                # operators see why their loaded drafter weights
                # appear inactive.
                loaded_drafter_id: object = (
                    self.coupled_drafter.model_id
                    if self.coupled_drafter is not None
                    else self.draft_model_id
                )
                logger.info(
                    f"using BatchGenerator (drafter weights loaded "
                    f"({loaded_drafter_id}) but EXO_DRAFT_MODE='none' "
                    f"explicitly disables speculation; keeping batching "
                    f"for throughput. Set EXO_DRAFT_MODE='model' or "
                    f"clear the env var to re-enable spec decode)"
                )
            elif ngram_configured_without_force_sequential:
                # Codex P1 (PR #19 round-(N+6), builder.py:151): make
                # the n-gram-on-BatchGenerator no-op path explicit so
                # operators see that ``EXO_DRAFT_MODE=ngram`` alone
                # has no runtime effect. To actually run n-gram set
                # ``EXO_NO_BATCH=1`` (greedy-only deployments) or
                # ``EXO_ALLOW_REQUEST_DRAFTING=1`` (per-request
                # override path).
                logger.warning(
                    "using BatchGenerator with EXO_DRAFT_MODE='ngram' set: "
                    "BatchGenerator has no spec-decoding hook so n-gram "
                    "drafting will be a no-op for every request. To run "
                    "n-gram set EXO_NO_BATCH=1 (forces SequentialGenerator) "
                    "or EXO_ALLOW_REQUEST_DRAFTING=1 (per-request override "
                    "path); batching is preserved here because the prior "
                    "behaviour disabled batching worker-wide for non-greedy "
                    "traffic that mlx_generate now demotes to 'none' anyway."
                )
            else:
                logger.info("using BatchGenerator")
            return BatchGenerator(
                model=self.inference_model,
                tokenizer=self.tokenizer,
                group=self.group,
                tool_parser=tool_parser,
                kv_prefix_cache=kv_prefix_cache,
                model_id=self.model_id,
                device_rank=device_rank,
                cancel_receiver=self.cancel_receiver,
                event_sender=self.event_sender,
                vision_processor=vision_processor,
            )
