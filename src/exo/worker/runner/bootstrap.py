"""--- not doing this anymore
import faulthandler
import os
import sys
"""

import loguru

from exo.shared.types.events import Event
from exo.shared.types.tasks import Task
from exo.shared.types.worker.instances import BoundInstance
from exo.utils.channels import MpReceiver, MpSender

""" -- not doing this anymore
def _redirect_stderr_to_file(path: str) -> None:
    # Replace fd 2 (stderr) with a file descriptor pointing to `path`
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o644)
    os.dup2(fd, 2)
    os.close(fd)
    # Rebind sys.stderr so Python's own writes go to the new fd as well (line-buffered)
    sys.stderr = os.fdopen(2, "w", buffering=1, closefd=False)
"""


def entrypoint(
    bound_instance: BoundInstance,
    event_sender: MpSender[Event],
    task_receiver: MpReceiver[Task],
    # err_path: str,
    _logger: "loguru.Logger",
) -> None:
    """
    Minimal entrypoint for the spawned child process.

    It redirects fd=2 (stderr) to a pipe provided by the parent, *then* imports
    the heavy runner module so that any C/C++ or MLX logs/crashes land in that pipe.
    """
    """ --- not doing this anymore
    _redirect_stderr_to_file(err_path)
    faulthandler.enable(file=sys.stderr, all_threads=True)
    """
    import os

    os.environ["MLX_METAL_FAST_SYNCH"] = "1"

    global logger
    logger = _logger

    # Import the heavy runner only after stderr is redirected
    from exo.worker.runner.runner import main

    main(bound_instance, event_sender, task_receiver)


logger: "loguru.Logger"
