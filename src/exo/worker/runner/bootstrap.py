from __future__ import annotations

import os
import threading
from multiprocessing.sharedctypes import Synchronized

import loguru

from exo.shared.types.events import Event, RunnerStatusUpdated
from exo.shared.types.tasks import Task, TaskId
from exo.shared.types.worker.instances import BoundInstance, MlxJacclInstance
from exo.shared.types.worker.runners import RunnerFailed
from exo.utils.channels import ClosedResourceError, MpReceiver, MpSender

logger: "loguru.Logger" = loguru.logger

HEARTBEAT_INTERVAL_SECONDS = 0.5


def _heartbeat_loop(heartbeat: Synchronized[int], stop: threading.Event) -> None:
    """Daemon thread that periodically increments the heartbeat counter."""
    while not stop.is_set():
        heartbeat.value += 1
        stop.wait(HEARTBEAT_INTERVAL_SECONDS)


def entrypoint(
    bound_instance: BoundInstance,
    event_sender: MpSender[Event],
    task_receiver: MpReceiver[Task],
    cancel_receiver: MpReceiver[TaskId],
    _logger: "loguru.Logger",
    heartbeat: Synchronized[int] | None = None,
) -> None:
    fast_synch_override = os.environ.get("EXO_FAST_SYNCH")
    if fast_synch_override == "on" or (
        fast_synch_override != "off"
        and (
            isinstance(bound_instance.instance, MlxJacclInstance)
            and len(bound_instance.instance.jaccl_devices) >= 2
        )
    ):
        os.environ["MLX_METAL_FAST_SYNCH"] = "1"
    else:
        os.environ["MLX_METAL_FAST_SYNCH"] = "0"

    global logger
    logger = _logger

    logger.info(f"Fast synch flag: {os.environ['MLX_METAL_FAST_SYNCH']}")

    # Start heartbeat thread so the supervisor can detect if we freeze.
    stop_heartbeat = threading.Event()
    heartbeat_thread: threading.Thread | None = None
    if heartbeat is not None:
        heartbeat_thread = threading.Thread(
            target=_heartbeat_loop,
            args=(heartbeat, stop_heartbeat),
            daemon=True,
        )
        heartbeat_thread.start()

    # Import main after setting global logger - this lets us just import logger from this module
    try:
        from exo.worker.runner.runner import main

        main(bound_instance, event_sender, task_receiver, cancel_receiver)
    except ClosedResourceError:
        logger.warning("Runner communication closed unexpectedly")
    except Exception as e:
        logger.opt(exception=e).warning(
            f"Runner {bound_instance.bound_runner_id} crashed with critical exception {e}"
        )
        event_sender.send(
            RunnerStatusUpdated(
                runner_id=bound_instance.bound_runner_id,
                runner_status=RunnerFailed(error_message=str(e)),
            )
        )
    finally:
        stop_heartbeat.set()
        if heartbeat_thread is not None:
            heartbeat_thread.join(timeout=1)
        try:
            event_sender.close()
            task_receiver.close()
        finally:
            event_sender.join()
            task_receiver.join()
            logger.info("bye from the runner")
