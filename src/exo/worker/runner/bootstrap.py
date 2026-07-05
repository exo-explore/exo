import os
import resource
import threading
import time
import traceback
from dataclasses import dataclass
from typing import Self, cast

import loguru

from exo.shared.types.events import Event
from exo.shared.types.tasks import Task, TaskId
from exo.shared.types.worker.instances import BoundInstance
from exo.utils.channels import ClosedResourceError, MpReceiver, MpSender
from exo.worker.engines.base import Builder

logger: "loguru.Logger" = loguru.logger


@dataclass(frozen=True)
class RunnerTerminationError:
    exception_type: str
    exception_message: str
    exception_repr: str
    traceback: str

    @classmethod
    def from_exception(cls, e: Exception) -> Self:
        return cls(
            exception_type=type(e).__qualname__,
            exception_message=str(e),
            exception_repr=repr(e),
            traceback="".join(
                traceback.TracebackException.from_exception(e).format(chain=True)
            ),
        )

    def __str__(self) -> str:
        return f"{self.exception_type}: {self.exception_message}\n{self.traceback}"


def _start_orphan_watchdog() -> None:
    """Exit the runner if its parent worker process dies.

    The runner is a multiprocessing child holding large Metal/unified-memory
    allocations and, for multi-node instances, RDMA queue pairs. If the worker
    is killed hard (SIGKILL, crash), the runner is silently reparented (pid 1 /
    launchd) and idles forever: RSS looks small while the real physical
    footprint can be hundreds of GB, so later placements fail with "does not
    fit" and replacement runners can crash-loop on the still-held queue pairs.
    There is nothing to clean up gracefully once the parent is gone — poll the
    parent pid and hard-exit when it changes.

    Disable with EXO_DISABLE_ORPHAN_WATCHDOG=1 (e.g. when intentionally
    debugging a runner that outlives its worker).
    """
    if os.environ.get("EXO_DISABLE_ORPHAN_WATCHDOG") == "1":
        return
    parent_process_id = os.getppid()

    def watch_parent() -> None:
        while True:
            time.sleep(5.0)
            if os.getppid() != parent_process_id:
                logger.warning(
                    f"runner parent (pid {parent_process_id}) died; "
                    "exiting to release model memory and interconnect resources"
                )
                os._exit(1)

    threading.Thread(target=watch_parent, name="orphan-watchdog", daemon=True).start()


def entrypoint(
    bound_instance: BoundInstance,
    event_sender: MpSender[Event | RunnerTerminationError],
    task_receiver: MpReceiver[Task],
    cancel_receiver: MpReceiver[TaskId],
    _logger: "loguru.Logger",
) -> None:
    global logger
    logger = _logger

    _start_orphan_watchdog()

    soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    resource.setrlimit(resource.RLIMIT_NOFILE, (min(max(soft, 2048), hard), hard))

    fast_synch_override = os.environ.get("EXO_FAST_SYNCH")
    if fast_synch_override == "false":
        os.environ["MLX_METAL_FAST_SYNCH"] = "0"
    else:
        os.environ["MLX_METAL_FAST_SYNCH"] = "1"

    logger.info(f"Fast synch flag: {os.environ['MLX_METAL_FAST_SYNCH']}")

    # Import main after setting global logger - this lets us just import logger from this module
    try:
        event_sender_downcast: MpSender[Event] = cast(MpSender[Event], event_sender)

        from exo.worker.runner.runner import Runner

        builder: Builder
        if bound_instance.is_image_model:
            from exo.worker.engines.image.builder import MfluxBuilder

            builder = MfluxBuilder(
                event_sender_downcast, cancel_receiver, bound_instance.bound_shard
            )
        else:
            from exo.worker.engines.mlx.patches import apply_mlx_patches

            apply_mlx_patches()

            from exo.worker.engines.mlx.builder import MlxBuilder

            # evil sharing of the event sender
            builder = MlxBuilder(
                model_id=bound_instance.bound_shard.model_card.model_id,
                event_sender=event_sender_downcast,
                cancel_receiver=cancel_receiver,
            )

        runner = Runner(bound_instance, builder, event_sender_downcast, task_receiver)
        runner.main()
    except ClosedResourceError:
        logger.warning("Runner communication closed unexpectedly")
    except Exception as e:
        logger.opt(exception=e).warning(
            f"Runner {bound_instance.bound_runner_id} crashed with critical exception {e}"
        )
        event_sender.send(RunnerTerminationError.from_exception(e))
        raise SystemExit(1) from e
    finally:
        try:
            event_sender.close()
            task_receiver.close()
        finally:
            event_sender.join()
            task_receiver.join()
            logger.info("bye from the runner")
