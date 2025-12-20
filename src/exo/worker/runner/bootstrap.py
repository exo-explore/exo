import os

import loguru

from exo.shared.types.events import Event
from exo.shared.types.tasks import Task
from exo.shared.types.worker.instances import BoundInstance, LlamaCppInstance, MlxJacclInstance
from exo.utils.channels import MpReceiver, MpSender

logger: "loguru.Logger"


if os.getenv("EXO_TESTS") == "1":
    logger = loguru.logger


def entrypoint(
    bound_instance: BoundInstance,
    event_sender: MpSender[Event],
    task_receiver: MpReceiver[Task],
    _logger: "loguru.Logger",
) -> None:
    # Set MLX-specific environment variables for Apple Silicon
    if (
        isinstance(bound_instance.instance, MlxJacclInstance)
        and len(bound_instance.instance.ibv_devices) >= 2
    ):
        os.environ["MLX_METAL_FAST_SYNCH"] = "1"

    # For llama.cpp instances, we might set specific environment variables
    if isinstance(bound_instance.instance, LlamaCppInstance):
        # Set thread count based on available CPUs if not already set
        if "LLAMA_N_THREADS" not in os.environ:
            cpu_count = os.cpu_count()
            if cpu_count:
                os.environ["LLAMA_N_THREADS"] = str(min(cpu_count, 8))

    global logger
    logger = _logger

    # Import main after setting global logger - this lets us just import logger from this module
    from exo.worker.runner.runner import main

    main(bound_instance, event_sender, task_receiver)
