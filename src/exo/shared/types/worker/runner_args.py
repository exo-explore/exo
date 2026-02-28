"""CLI arguments for runner processes passed as JSON."""

from exo.shared.types.worker.instances import BoundInstance
from exo.utils.pydantic_ext import FrozenModel


class RunnerCliArgs(FrozenModel):
    """
    Arguments passed to the runner process via JSON on the command line.

    File descriptors for communication are passed separately via pass_fds.
    Each logical channel uses a single pipe:
    - Events: child -> parent (child writes, parent reads)
    - Tasks: parent -> child (parent writes, child reads)
    - Cancels: parent -> child (parent writes, child reads)
    """

    bound_instance: BoundInstance
    event_fd: int  # Child writes events here
    task_fd: int  # Child reads tasks from here
    cancel_fd: int  # Child reads cancels from here
    log_level: str = "INFO"
