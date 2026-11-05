import multiprocessing
import os
import resource
import threading
import traceback
from dataclasses import dataclass
from typing import Self, cast

import loguru

from exo.shared.types.events import Event, RunnerEvent
from exo.shared.types.tasks import Task, TaskId
from exo.shared.types.worker.instances import (
    BoundInstance,
    Instance,
    MlxJacclInstance,
)
from exo.utils.channels import ClosedResourceError, MpReceiver, MpSender
from exo.worker.engines.base import Builder
from exo.worker.engines.mlx.models.spec_registry import fast_synch_allowed

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


def use_fast_synch(instance: Instance, override: str | None) -> bool:
    """Whether MLX should synchronise its CPU and GPU work by spinning on shared memory.

    It saves latency on every hand-over between the GPU and the CPU, which an RDMA
    (JACCL) instance makes for each collective. Over the TCP ring it makes no measurable
    difference to a model's speed, and it breaks down when two runners share a Mac's GPU:
    both models slow to a crawl, the ring fails with EFAULT, or the GPU deadlocks.
    A model family can turn it off for its models (FamilySpec.fast_synch).
    `--fast-synch` / `--no-fast-synch` (EXO_FAST_SYNCH) override the choice.
    """
    if override == "true":
        return True
    if override == "false":
        return False
    return isinstance(instance, MlxJacclInstance) and all(
        fast_synch_allowed(shard.model_card.model_type)
        for shard in instance.shard_assignments.runner_to_shard.values()
    )


def exit_with_parent() -> None:
    """End this process as soon as the process that started it (the node) is gone.

    A runner waits on pipes to its node, and holds both ends of them, so it never sees an end
    of file when the node dies: killed with SIGKILL (a crash, or the OS ending it), exo leaves
    its runners running for good, each holding its model's memory, and a restarted exo can't
    load the model. multiprocessing gives a spawned process the read end of a pipe whose write
    end only its parent holds, while the parent's Process object for it is open (it can't be
    closed, nor collected, while the child runs): `parent_process().join()` returns when the
    parent dies, and not before, so this can't fire while the node lives.

    The thread needs the GIL to exit. A runner inside native code that holds it exits only once
    that call returns: a lower rank orphaned in MLX's ring set-up, which waits in accept with
    the GIL held (JACCL's set-up may too), keeps its process and the instance's port, though no
    model memory. A watcher that doesn't need the GIL would close that gap."""
    parent = multiprocessing.parent_process()
    if parent is None:
        return

    def watch() -> None:
        parent.join()
        # No one is left to tell or to clean up for: the node's memory and GPU state go with
        # this process, and its resource tracker releases its semaphores once it is gone
        os._exit(1)

    threading.Thread(target=watch, name="exit-with-parent", daemon=True).start()


def entrypoint(
    bound_instance: BoundInstance,
    event_sender: MpSender[Event | RunnerTerminationError],
    task_receiver: MpReceiver[Task],
    cancel_receiver: MpReceiver[TaskId],
    _logger: "loguru.Logger",
) -> None:
    global logger
    logger = _logger

    exit_with_parent()

    soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    resource.setrlimit(resource.RLIMIT_NOFILE, (min(max(soft, 2048), hard), hard))

    fast_synch = use_fast_synch(
        bound_instance.instance, os.environ.get("EXO_FAST_SYNCH")
    )
    os.environ["MLX_METAL_FAST_SYNCH"] = "1" if fast_synch else "0"

    logger.info(f"Fast synch flag: {os.environ['MLX_METAL_FAST_SYNCH']}")

    # Import main after setting global logger - this lets us just import logger from this module
    try:
        event_sender_downcast: MpSender[RunnerEvent] = cast(
            MpSender[RunnerEvent], event_sender
        )

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
