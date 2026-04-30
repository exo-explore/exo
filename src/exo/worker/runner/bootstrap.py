import os
import resource

import loguru

from exo.shared.types.events import Event, RunnerStatusUpdated
from exo.shared.types.tasks import Task, TaskId
from exo.shared.types.worker.instances import BoundInstance
from exo.shared.types.worker.runners import RunnerFailed
from exo.utils.channels import ClosedResourceError, MpReceiver, MpSender
from exo.worker.engines.base import Builder

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
    if fast_synch_override == "false":
        os.environ["MLX_METAL_FAST_SYNCH"] = "0"
    else:
        os.environ["MLX_METAL_FAST_SYNCH"] = "1"

    runner_context = (
        f"instance_id={bound_instance.instance.instance_id} "
        f"runner_id={bound_instance.bound_runner_id} "
        f"node_id={bound_instance.bound_node_id} "
        f"model_id={bound_instance.bound_shard.model_card.model_id}"
    )
    logger.info(
        f"Runner bootstrap starting {runner_context} "
        f"fast_synch={os.environ['MLX_METAL_FAST_SYNCH']}"
    )

    # Import main after setting global logger - this lets us just import logger from this module
    try:
        from exo.worker.runner.runner import Runner

        builder: Builder

        if bound_instance.is_image_model:
            from exo.worker.engines.image.builder import MfluxBuilder

            builder = MfluxBuilder(
                event_sender, cancel_receiver, bound_instance.bound_shard
            )
        else:
            from exo.worker.engines.mlx.patches import apply_mlx_patches

            apply_mlx_patches()

            from exo.worker.engines.mlx.builder import MlxBuilder

            # evil sharing of the event sender
            builder = MlxBuilder(
                model_id=bound_instance.bound_shard.model_card.model_id,
                event_sender=event_sender,
                cancel_receiver=cancel_receiver,
            )

        runner = Runner(bound_instance, builder, event_sender, task_receiver)
        runner_kind = "image" if bound_instance.is_image_model else "text"
        logger.info(f"Starting {runner_kind} runner main loop {runner_context}")
        runner.main()

    except ClosedResourceError:
        logger.warning("Runner communication closed unexpectedly")
    except Exception as e:
        logger.opt(exception=e).warning(
            f"Runner {bound_instance.bound_runner_id} crashed with critical exception {e} "
            f"{runner_context}"
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
        finally:
            event_sender.join()
            task_receiver.join()
            logger.info(f"bye from the runner {runner_context}")
