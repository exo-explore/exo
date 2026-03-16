import os
import resource

import loguru

from exo.shared.types.events import Event, RunnerStatusUpdated
from exo.shared.types.tasks import Task, TaskId
from exo.shared.types.worker.instances import BoundInstance
from exo.shared.types.worker.runners import RunnerFailed
from exo.utils.channels import ClosedResourceError, MpReceiver, MpSender
from exo.worker.runner.failure_store import persist_runner_failure

logger: "loguru.Logger" = loguru.logger


def entrypoint(
    bound_instance: BoundInstance,
    event_sender: MpSender[Event],
    task_receiver: MpReceiver[Task],
    cancel_receiver: MpReceiver[TaskId],
    _logger: "loguru.Logger",
) -> None:
    global logger
    logger = _logger

    soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    resource.setrlimit(resource.RLIMIT_NOFILE, (min(max(soft, 2048), hard), hard))

    fast_synch_override = os.environ.get("EXO_FAST_SYNCH")
    if fast_synch_override != "off":
        os.environ["MLX_METAL_FAST_SYNCH"] = "1"
    else:
        os.environ["MLX_METAL_FAST_SYNCH"] = "0"

    logger.info(f"Fast synch flag: {os.environ['MLX_METAL_FAST_SYNCH']}")

    # Import main after setting global logger - this lets us just import logger from this module
    try:
        if bound_instance.is_image_model:
            from exo.worker.runner.image_models.runner import Runner as ImageRunner

            runner = ImageRunner(
                bound_instance, event_sender, task_receiver, cancel_receiver
            )
            runner.main()
        else:
            from exo.worker.runner.llm_inference.runner import Runner

            runner = Runner(
                bound_instance, event_sender, task_receiver, cancel_receiver
            )
            runner.main()

    except ClosedResourceError:
        logger.warning("Runner communication closed unexpectedly")
    except Exception as e:
        logger.opt(exception=e).warning(
            f"Runner {bound_instance.bound_runner_id} crashed with critical exception {e}"
        )
        try:
            event_sender.send(
                RunnerStatusUpdated(
                    runner_id=bound_instance.bound_runner_id,
                    runner_status=RunnerFailed(error_message=str(e)),
                )
            )
        except (BrokenPipeError, OSError, ClosedResourceError):
            persist_runner_failure(
                bound_instance,
                error_message=str(e),
                source="runner_bootstrap_exception",
            )
    finally:
        try:
            event_sender.close()
            task_receiver.close()
        finally:
            event_sender.join()
            task_receiver.join()
            logger.info("bye from the runner")
