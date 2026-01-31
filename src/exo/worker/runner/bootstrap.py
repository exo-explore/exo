import os

import loguru

from exo.shared.types.events import Event, RunnerStatusUpdated
from exo.shared.types.tasks import Task
from exo.shared.types.worker.instances import (
    BoundInstance,
    MlxJacclInstance,
    MlxRingInstance,
    PytorchInstance,
)
from exo.shared.types.worker.runners import RunnerFailed
from exo.utils.channels import ClosedResourceError, MpReceiver, MpSender
from exo.worker.engines.base_engine import Engine

logger: "loguru.Logger" = loguru.logger


def _create_engine(bound_instance: BoundInstance) -> Engine:
    """Create the appropriate engine based on instance type."""
    instance = bound_instance.instance
    
    # Check for feature flag to disable PyTorch
    if isinstance(instance, PytorchInstance) and os.getenv("EXO_DISABLE_PYTORCH"):
        raise ValueError(
            "PyTorch backend is disabled via EXO_DISABLE_PYTORCH environment variable"
        )
    
    # Select engine based on instance type - import only the needed engine
    if isinstance(instance, PytorchInstance):
        from exo.worker.engines.pytorch import PytorchEngine
        return PytorchEngine(bound_instance)
    elif isinstance(instance, (MlxRingInstance, MlxJacclInstance)):
        from exo.worker.engines.mlx import MlxEngine
        return MlxEngine(bound_instance)
    else:
        raise ValueError(
            f"Unknown instance type: {type(instance).__name__}. "
            f"Supported: PytorchInstance, MlxRingInstance, MlxJacclInstance"
        )


import anyio

async def entrypoint(
    bound_instance: BoundInstance,
    event_sender: MpSender[Event],
    task_receiver: MpReceiver[Task],
    _logger: "loguru.Logger",
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

    # Select and instantiate the engine using registry
    engine = _create_engine(bound_instance)

    # Import main after setting global logger - this lets us just import logger from this module
    try:
        from exo.worker.runner.runner import main

        # main() is async - await it
        await main(bound_instance, event_sender, task_receiver, engine)
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
