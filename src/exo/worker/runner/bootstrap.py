import os

import loguru

from exo.shared.types.events import Event, RunnerStatusUpdated
from exo.shared.types.tasks import Task
from exo.shared.types.worker.instances import (
    BoundInstance,
    MlxJacclInstance,
)
from exo.shared.types.worker.runners import RunnerFailed
from exo.utils.channels import ClosedResourceError, MpReceiver, MpSender

logger: "loguru.Logger" = loguru.logger


def entrypoint(
    bound_instance: BoundInstance,
    event_sender: MpSender[Event],
    task_receiver: MpReceiver[Task],
    _logger: "loguru.Logger",
) -> None:
    global logger
    logger = _logger

    # Route based on instance type
    try:
        from exo.plugins.registry import PluginRegistry, discover_plugins

        # Discover plugins in subprocess (they aren't inherited from main process)
        discover_plugins()

        registry = PluginRegistry.get()
        instance = bound_instance.instance

        # Check if a plugin handles this instance type
        plugin = registry.get_plugin_for_instance(instance)
        if plugin is not None:
            # Delegate to plugin runner
            plugin.create_runner(bound_instance, event_sender, task_receiver)
        else:
            # MLX runner (default)
            if (
                isinstance(bound_instance.instance, MlxJacclInstance)
                and len(bound_instance.instance.ibv_devices) >= 2
            ):
                os.environ["MLX_METAL_FAST_SYNCH"] = "1"

            from exo.worker.runner.runner import main

            main(bound_instance, event_sender, task_receiver)
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
        finally:
            event_sender.join()
            task_receiver.join()
            logger.info("bye from the runner")
