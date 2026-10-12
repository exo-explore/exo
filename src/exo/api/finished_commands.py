"""A request's task stays in state until a master gets the request's TaskFinished. That command can be
lost: no master may be reachable when it is sent, or the master that gets it may be replaced by one
whose state still has the task. So the API sends it again while state still has the task."""

from collections.abc import Mapping
from typing import Final

from exo.shared.types.common import CommandId
from exo.shared.types.state import State
from exo.shared.types.tasks import ImageEdits, ImageGeneration, TextGeneration

RESEND_INTERVAL_SECONDS: Final = 10.0
# Long enough for a master that was replaced to come back with its older state
FORGET_AFTER_SECONDS: Final = 300.0


def finishes_to_resend(
    finished_at: Mapping[CommandId, float], state: State, now: float
) -> tuple[list[CommandId], list[CommandId]]:
    """Of the commands this API finished (and when), those to send TaskFinished for again, because
    state still has their task, and those to forget: their task is gone and they finished long ago."""
    with_tasks = {
        task.command_id
        for task in state.tasks.values()
        if isinstance(task, (TextGeneration, ImageGeneration, ImageEdits))
    }
    resend = [command_id for command_id in finished_at if command_id in with_tasks]
    forget = [
        command_id
        for command_id, finished in finished_at.items()
        if command_id not in with_tasks and now - finished > FORGET_AFTER_SECONDS
    ]
    return resend, forget
