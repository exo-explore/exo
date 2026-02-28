import contextlib
import os
import sys

import loguru
from pydantic import TypeAdapter

from exo.shared.types.events import Event, RunnerStatusUpdated
from exo.shared.types.tasks import Task, TaskId
from exo.shared.types.worker.runner_args import RunnerCliArgs
from exo.shared.types.worker.runners import RunnerFailed
from exo.utils.fd_channels import FdReceiver, FdSender

logger: "loguru.Logger" = loguru.logger

# Type adapters for union types
event_adapter: TypeAdapter[Event] = TypeAdapter(Event)
task_adapter: TypeAdapter[Task] = TypeAdapter(Task)
task_id_adapter: TypeAdapter[TaskId] = TypeAdapter(TaskId)


def entrypoint() -> None:
    """
    Entrypoint for the runner process.

    Expects a single command-line argument containing JSON-serialized RunnerCliArgs.
    File descriptors for communication are inherited via pass_fds.
    """
    if len(sys.argv) < 2:
        print("Usage: runner <json_args>", file=sys.stderr)
        sys.exit(1)

    # Parse CLI arguments
    args_json = sys.argv[1]
    args = RunnerCliArgs.model_validate_json(args_json)

    # Configure logging
    logger.remove()
    logger.add(sys.stderr, level=args.log_level)

    # Set up environment
    fast_synch_override = os.environ.get("EXO_FAST_SYNCH")
    if fast_synch_override != "off":
        os.environ["MLX_METAL_FAST_SYNCH"] = "1"
    else:
        os.environ["MLX_METAL_FAST_SYNCH"] = "0"

    logger.info(f"Fast synch flag: {os.environ['MLX_METAL_FAST_SYNCH']}")

    # Create FD-based channels with TypeAdapters
    # Child uses: event_fd for writing events, task_fd for reading tasks, cancel_fd for reading cancels
    event_sender = FdSender(args.event_fd, event_adapter)
    task_receiver = FdReceiver(args.task_fd, task_adapter)
    cancel_receiver = FdReceiver(args.cancel_fd, task_id_adapter)

    bound_instance = args.bound_instance

    try:
        # Import main after setting global logger - this lets us just import logger from this module
        if bound_instance.is_image_model:
            from exo.worker.runner.image_models.runner import main
        else:
            from exo.worker.runner.llm_inference.runner import main

        main(bound_instance, event_sender, task_receiver, cancel_receiver)

    except Exception as e:
        logger.opt(exception=e).warning(
            f"Runner {bound_instance.bound_runner_id} crashed with critical exception {e}"
        )
        with contextlib.suppress(Exception):
            event_sender.send(
                RunnerStatusUpdated(
                    runner_id=bound_instance.bound_runner_id,
                    runner_status=RunnerFailed(error_message=str(e)),
                )
            )
    finally:
        # Clean up channels
        with contextlib.suppress(Exception):
            event_sender.close()
        with contextlib.suppress(Exception):
            task_receiver.close()
        with contextlib.suppress(Exception):
            cancel_receiver.close()

        logger.info("bye from the runner")


if __name__ == "__main__":
    entrypoint()
