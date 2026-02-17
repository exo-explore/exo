import os

import loguru

from exo.shared.types.events import Event, RunnerStatusUpdated
from exo.shared.types.tasks import Task, TaskId
from exo.shared.types.worker.instances import BoundInstance, MlxJacclInstance
from exo.shared.types.worker.runners import RunnerFailed
from exo.utils.channels import ClosedResourceError, MpReceiver, MpSender

logger: "loguru.Logger" = loguru.logger


def entrypoint(
    bound_instance: BoundInstance,
    event_sender: MpSender[Event],
    task_receiver: MpReceiver[Task],
    cancel_receiver: MpReceiver[TaskId],
    _logger: "loguru.Logger",
    pipe_fifo_paths: tuple[str, str] | None = None,
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

    # Open JACCL FIFOs by path and set env vars for C++ SideChannel.
    # Named pipes (FIFOs) work across multiprocessing spawn (macOS default).
    if pipe_fifo_paths is not None:
        fifo_c2p, fifo_p2c = pipe_fifo_paths
        # C++ reads gathered data from p2c (PIPE_IN), writes local data to c2p (PIPE_OUT)
        pipe_in_fd = os.open(fifo_p2c, os.O_RDONLY)
        pipe_out_fd = os.open(fifo_c2p, os.O_WRONLY)
        os.environ["MLX_JACCL_PIPE_IN"] = str(pipe_in_fd)
        os.environ["MLX_JACCL_PIPE_OUT"] = str(pipe_out_fd)

    global logger
    logger = _logger

    logger.info(f"Fast synch flag: {os.environ['MLX_METAL_FAST_SYNCH']}")

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
        try:
            event_sender.close()
            task_receiver.close()
            cancel_receiver.close()
        finally:
            event_sender.join()
            task_receiver.join()
            cancel_receiver.join()
            logger.info("bye from the runner")
