"""Runner for an asymmetric drafter rank.

The asymmetric placement layer (``master.placement``) selects a
drafter-eligible node whenever a model card lists
:attr:`ModelCard.drafter_eligible_nodes` and at least one eligible host
is socket-reachable from target rank 0. The drafter loads its own
(smaller) drafter model on that node and runs :func:`drafter_serve_loop`
to field forwards from target rank 0 over a direct TCP socket.

Under the v3+ wire the drafter rank is NOT a member of the target
ranks' ``mx.distributed.Group``. It does not call
``mx.distributed.init`` at all -- it dials
``DrafterPlacement.drafter_socket_host:drafter_socket_port`` and runs
the serve loop over the resulting socket. Decoupling drafter IPC from
``mx.distributed`` lets target ranks of any size run TP/PP collectives
without requiring ``Group.split`` (which jaccl/ring backends do not
implement on Apple Silicon).

This module follows the same lifecycle as :class:`exo.worker.runner.runner.Runner`
(``Idle -> Connecting -> Connected -> Loading -> Loaded -> WarmingUp ->
Ready -> Running``) so the worker plan's readiness checks (which iterate
``Instance.all_runner_ids``) treat the drafter rank like any other rank.
The internals differ:

  * No target shard, no tokenizer, no chat-completion handling. The
    drafter has its own ``ModelCard`` and only loads the drafter
    weights.
  * No ``Engine`` wrapper. ``StartWarmup`` does a single forward to
    JIT-compile Metal kernels, then the drafter steps directly into
    :func:`drafter_serve_loop`, which blocks on socket recv until the
    target rank sends ``OP_SHUTDOWN``.
  * ``Shutdown`` arrives via the worker plan after target ranks have
    already sent ``OP_SHUTDOWN``; on the drafter side we just clean up
    state.

The module is import-cheap: it does not pull in any target-side
generator code (``generate.py``, ``batch_generator.py``, etc.). The
drafter runs in its own process with its own model, so memory and
import time stay tight.
"""

from __future__ import annotations

import contextlib
import socket
import time
from typing import TYPE_CHECKING, cast, final

import mlx.core as mx
from loguru import logger as loguru_logger
from mlx_lm.utils import load_model

from exo.download.download_utils import build_model_path, resolve_existing_model
from exo.shared.types.events import (
    Event,
    RunnerStatusUpdated,
    TaskAcknowledged,
    TaskStatusUpdated,
)
from exo.shared.types.tasks import (
    ConnectToGroup,
    LoadModel,
    Shutdown,
    StartWarmup,
    Task,
    TaskId,
    TaskStatus,
)
from exo.shared.types.worker.instances import BoundInstance, DrafterPlacement
from exo.shared.types.worker.runners import (
    RunnerConnected,
    RunnerConnecting,
    RunnerIdle,
    RunnerLoaded,
    RunnerLoading,
    RunnerReady,
    RunnerRunning,
    RunnerShutdown,
    RunnerShuttingDown,
    RunnerStatus,
    RunnerWarmingUp,
)
from exo.utils.channels import ClosedResourceError, EndOfStream, MpReceiver, MpSender

if TYPE_CHECKING:
    from exo.worker.engines.mlx.types import KVCacheType, Model


@final
class DrafterRunner:
    """Lifecycle manager for the drafter rank in an asymmetric instance.

    Same task-driven state machine as the target runner -- the worker
    plan dispatches ``ConnectToGroup``, ``LoadModel``, ``StartWarmup``,
    and ``Shutdown`` in order; readiness gates iterate
    ``Instance.all_runner_ids`` so the drafter participates in
    barriers exactly like a target rank.
    """

    def __init__(
        self,
        bound_instance: BoundInstance,
        event_sender: MpSender[Event],
        task_receiver: MpReceiver[Task],
    ) -> None:
        assert bound_instance.is_drafter_rank, (
            "DrafterRunner can only be constructed for an asymmetric drafter "
            "rank; check `bound_instance.is_drafter_rank` before instantiation."
        )
        placement = bound_instance.instance.drafter_placement
        assert placement is not None
        self._placement: DrafterPlacement = placement

        self.bound_instance = bound_instance
        self.runner_id = bound_instance.bound_runner_id
        self.event_sender = event_sender
        self.task_receiver = task_receiver

        self.drafter_socket: socket.socket | None = None
        self.draft_model: Model | None = None

        self._setup_start = time.perf_counter()
        self._update_status(RunnerIdle())
        loguru_logger.info(
            f"DrafterRunner created (runner_id={self.runner_id} "
            f"node={bound_instance.bound_node_id} "
            f"drafter_model_id={self._placement.drafter_model_id} "
            f"drafter_rank={self._placement.drafter_rank})"
        )

    def main(self) -> None:
        try:
            with self.task_receiver:
                for task in self.task_receiver:
                    if not self._dispatch(task):
                        return
        except (EndOfStream, ClosedResourceError):
            loguru_logger.warning("DrafterRunner task stream closed")

    def _dispatch(self, task: Task) -> bool:
        """Process one task; return ``False`` to exit the main loop."""
        self._send_task_status(task.task_id, TaskStatus.Running)
        match task:
            case ConnectToGroup() if isinstance(self.current_status, RunnerIdle):
                self._handle_connect(task)
            case LoadModel() if isinstance(self.current_status, RunnerConnected):
                self._handle_load(task)
            case StartWarmup() if isinstance(self.current_status, RunnerLoaded):
                self._handle_start_warmup(task)
            case Shutdown():
                self._handle_shutdown(task)
                return False
            case _:
                raise ValueError(
                    f"DrafterRunner received {task.__class__.__name__} outside "
                    f"of state machine in {self.current_status=}"
                )
        return True

    def _handle_connect(self, task: Task) -> None:
        """Dial target rank 0's drafter listener; no mx.distributed init.

        Under the v3+ wire the drafter is outside the target's
        ``mx.distributed.Group``. ``ConnectToGroup`` is the natural
        place to establish the drafter wire (the lifecycle stage runs
        in parallel with target ranks initialising mx.distributed,
        which gives target rank 0 time to bind before we dial).
        :func:`dial_target` retries with backoff up to two minutes,
        comfortably covering target rank 0's bind delay.
        """
        from exo.worker.engines.mlx.generator.drafter_socket import dial_target

        self._update_status(RunnerConnecting())
        self._acknowledge(task)
        host = self._placement.drafter_socket_host
        port = self._placement.drafter_socket_port
        loguru_logger.info(
            f"DrafterRunner dialing target rank 0 at {host}:{port} "
            f"(drafter_model_id={self._placement.drafter_model_id})"
        )
        self.drafter_socket = dial_target(host, port)
        loguru_logger.info(
            f"DrafterRunner connected over socket "
            f"(drafter_rank={self._placement.drafter_rank})"
        )
        self._send_task_status(task.task_id, TaskStatus.Complete)
        self._update_status(RunnerConnected())

    def _handle_load(self, task: Task) -> None:
        drafter_id = self._placement.drafter_model_id
        drafter_path = resolve_existing_model(drafter_id)
        if drafter_path is None:
            # Build a fallback path so the error message points at where
            # the operator should drop the weights.
            expected_path = build_model_path(drafter_id)
            raise FileNotFoundError(
                f"Drafter weights for {drafter_id} not found on this node "
                f"(expected at {expected_path}). Asymmetric drafter "
                "placement requires pre-downloading the drafter model "
                "on every drafter-eligible node; auto-download is not "
                "yet implemented for the drafter rank."
            )

        self._update_status(RunnerLoading(layers_loaded=0, total_layers=0))
        self._acknowledge(task)

        load_start = time.perf_counter()
        loguru_logger.info(f"DrafterRunner loading {drafter_id} from {drafter_path}")
        model, _ = load_model(drafter_path, lazy=True, strict=False)
        mx.eval(model)
        self.draft_model = cast("Model", model)
        # ``draft_cache`` is no longer pre-allocated -- the serve loop
        # multiplexes per-session caches keyed on ``session_id`` (target
        # rank's :meth:`RemoteTransport.open_session` allocation) and
        # builds each one lazily via ``make_kv_cache(model=...)`` on
        # the matching ``OP_PREFILL``. Holding only the model means
        # cluster-idle memory stays small (~drafter weights, no KV
        # cache); active memory scales linearly with concurrent target
        # requests, capped by the runner's ``EXO_MAX_CONCURRENT_REQUESTS``.
        loguru_logger.info(
            f"DrafterRunner loaded {drafter_id} in "
            f"{(time.perf_counter() - load_start):.2f}s"
        )

        self._send_task_status(task.task_id, TaskStatus.Complete)
        self._update_status(RunnerLoaded())

    def _handle_start_warmup(self, task: Task) -> None:
        from exo.worker.engines.mlx.cache import make_kv_cache

        assert self.drafter_socket is not None
        assert self.draft_model is not None

        self._update_status(RunnerWarmingUp())
        self._acknowledge(task)

        # JIT-compile drafter Metal kernels with a single forward
        # against a throwaway cache so the first real spec-decode round
        # on the target rank doesn't eat the compile latency. The
        # warmup cache is GC'd at the end of this method; per-session
        # caches are allocated lazily inside :func:`drafter_serve_loop`
        # on each ``OP_PREFILL``.
        warmup_start = time.perf_counter()
        warmup_cache = make_kv_cache(model=self.draft_model)
        seed = mx.array([[0]], dtype=mx.uint32)
        _ = self.draft_model(seed, cache=warmup_cache)
        mx.eval([c.state for c in warmup_cache])  # type: ignore[reportArgumentType]
        del warmup_cache
        loguru_logger.info(
            f"DrafterRunner warmup complete in "
            f"{(time.perf_counter() - warmup_start):.2f}s; "
            f"setup_total={(time.perf_counter() - self._setup_start):.2f}s"
        )

        self._send_task_status(task.task_id, TaskStatus.Complete)
        # The drafter has no prefill server, so prefill_server_port is None.
        self._update_status(RunnerReady(prefill_server_port=None))
        self._update_status(RunnerRunning())

        # Enter the drafter serve loop. This blocks until the target
        # rank sends OP_SHUTDOWN. The serve loop's send/recv use the
        # parent group; target rank 0 is conventionally the only target
        # rank that drives drafter IPC.
        self._serve_loop()

        # OP_SHUTDOWN arrived; transition back to Ready so the worker
        # plan's Shutdown task can drive us to RunnerShutdown.
        self._update_status(RunnerReady(prefill_server_port=None))

    def _serve_loop(self) -> None:
        from exo.worker.engines.mlx.cache import make_kv_cache
        from exo.worker.engines.mlx.generator.remote_drafter import drafter_serve_loop

        assert self.drafter_socket is not None
        assert self.draft_model is not None

        # ``num_draft_tokens`` here only sizes the response buffer; the
        # spec loop on the target side may issue forwards with
        # ``num_forwards`` up to K+1, so we mirror exactly its config.
        num_draft_tokens = self._num_draft_tokens()
        loguru_logger.info(
            f"DrafterRunner entering serve_loop "
            f"(K={num_draft_tokens}, transport=tcp_socket)"
        )
        # Capture ``draft_model`` in the closure so the serve loop can
        # allocate per-session caches lazily without re-entering
        # ``DrafterRunner`` state. Dummy assertion here to satisfy the
        # type checker (``self.draft_model`` is ``Model | None`` at the
        # field level but we asserted not None above).
        draft_model = self.draft_model

        def _make_session_cache() -> "KVCacheType":
            return make_kv_cache(model=draft_model)

        drafter_serve_loop(
            draft_model=draft_model,
            make_draft_cache=_make_session_cache,
            num_draft_tokens=num_draft_tokens,
            sock=self.drafter_socket,
        )
        loguru_logger.info("DrafterRunner serve_loop exited via OP_SHUTDOWN")

    @staticmethod
    def _num_draft_tokens() -> int:
        # Same default the target-side builder uses; reading the env
        # var keeps drafter and target in lock-step without an explicit
        # IPC message at warmup time.
        from exo.worker.runner.llm_inference.batch_generator import (
            DEFAULT_NUM_DRAFT_TOKENS,
            EXO_NUM_DRAFT_TOKENS,
            parse_env_int,
        )

        return parse_env_int(EXO_NUM_DRAFT_TOKENS, default=DEFAULT_NUM_DRAFT_TOKENS)

    def _handle_shutdown(self, task: Task) -> None:
        loguru_logger.info("DrafterRunner shutting down")
        self._update_status(RunnerShuttingDown())
        self._acknowledge(task)
        # Release the model so the drafter rank's process frees its
        # drafter weights before exiting. Per-session caches were owned
        # by :func:`drafter_serve_loop`; they were dropped when the
        # loop returned via ``OP_SHUTDOWN``.
        self.draft_model = None
        if self.drafter_socket is not None:
            with contextlib.suppress(OSError):
                self.drafter_socket.close()
            self.drafter_socket = None
        import gc

        gc.collect()
        self._send_task_status(task.task_id, TaskStatus.Complete)
        self._update_status(RunnerShutdown())

    # -- helpers ---------------------------------------------------------

    def _update_status(self, status: RunnerStatus) -> None:
        self.current_status: RunnerStatus = status
        self.event_sender.send(
            RunnerStatusUpdated(runner_id=self.runner_id, runner_status=status)
        )

    def _send_task_status(self, task_id: TaskId, status: TaskStatus) -> None:
        self.event_sender.send(TaskStatusUpdated(task_id=task_id, task_status=status))

    def _acknowledge(self, task: Task) -> None:
        self.event_sender.send(TaskAcknowledged(task_id=task.task_id))


__all__ = ["DrafterRunner"]
