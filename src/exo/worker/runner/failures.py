"""What a request is told when the runner serving it fails.

A runner's error is often a traceback, or MLX's own words for a low-level fault. The node's log
keeps those; the request gets one short sentence the user can act on (exo.shared.request_errors,
which also gives the stable code API clients see).
"""

import re
from collections.abc import Sequence
from typing import final

from exo.shared.request_errors import (
    DEVICE_DISCONNECTED,
    GPU_TIMEOUT,
    MODEL_FAILED,
    MODEL_FAILED_BECAUSE,
    MODEL_STALLED,
    OUT_OF_MEMORY,
    user_facing_detail,
)
from exo.worker.runner.bootstrap import RunnerTerminationError
from exo.worker.runner.diagnostics import (
    KnownRunnerDiagnostic,
    RunnerMetalGpuTimeout,
    RunnerMetalOutOfMemory,
    RunnerRingSocketReceivingError,
    RunnerRingTransportError,
    metal_out_of_memory,
)


class RunnerStoppedError(Exception):
    """Why the supervisor stopped a runner that was still running"""


@final
class RunnerLostPeersError(RunnerStoppedError):
    """MLX gave up on the connection to the instance's other runners"""


@final
class RunnerStalledError(RunnerStoppedError):
    """The runner made no progress while generating"""


# What a runner's error looks like when it lost the connection to another runner of its
# instance: MLX's distributed backends report it in these words
_PEER_LOST_MARKERS = (
    "[ring]",
    "[jaccl]",
    "connection to a peer was lost",
    "connection reset by peer",
    "econnreset",
)
_METAL_GPU_TIMEOUT = re.compile(r"\[METAL\].*GPU\s+Timeout", re.IGNORECASE)


def describe_failure(
    error: RunnerTerminationError | Exception | None,
    diagnostics: Sequence[KnownRunnerDiagnostic] = (),
) -> str:
    """Why the requests of a failed runner ended, in one short sentence.

    error is what the runner raised (as reported by its process, or caught in the engine), or
    the supervisor's own reason for stopping it; diagnostics are what was recognised in the
    runner's output.
    """
    message = (
        error.exception_message
        if isinstance(error, RunnerTerminationError)
        else str(error)
        if error is not None
        else ""
    )
    if any(isinstance(d, RunnerMetalOutOfMemory) for d in diagnostics) or (
        metal_out_of_memory(message, ()) is not None
    ):
        return OUT_OF_MEMORY
    if any(isinstance(d, RunnerMetalGpuTimeout) for d in diagnostics) or (
        _METAL_GPU_TIMEOUT.search(message) is not None
    ):
        return GPU_TIMEOUT
    if (
        isinstance(error, RunnerLostPeersError)
        or any(
            isinstance(d, (RunnerRingTransportError, RunnerRingSocketReceivingError))
            for d in diagnostics
        )
        or any(marker in message.lower() for marker in _PEER_LOST_MARKERS)
    ):
        return DEVICE_DISCONNECTED
    if isinstance(error, RunnerStalledError):
        return MODEL_STALLED
    # The runner's own error, unlike the supervisor's reasons, can tell the user something
    detail = (
        "" if isinstance(error, RunnerStoppedError) else user_facing_detail(message)
    )
    if not detail:
        return MODEL_FAILED
    return f"{MODEL_FAILED_BECAUSE}{detail.rstrip('.')}."
