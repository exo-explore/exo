"""What a request is told when it ends with an error, and the stable code that says why.

A failed request ends with one short sentence for the user; the details (a runner's traceback,
say) stay in the node's log. API clients get a stable code with it, which they can act on
without parsing the sentence. The code isn't sent between nodes: it is looked up from the
sentence where the API answers the request. An ErrorChunk with a field older builds don't
know would be rejected by them, and a node that rejects an event stalls until it next catches
up from a snapshot of the cluster's state.
"""

import re
from typing import Literal

from exo.shared.types.worker.instances import InstanceRemovalReason

# Why a request ended with an error. Stable: clients match them.
GenerationErrorCode = Literal[
    # A device serving the model disconnected or left the cluster
    "device_left",
    # A device serving the model can't be reached now, as the node that answers sees it
    "device_unreachable",
    # The model was stopped (its instance deleted) while the request was in flight
    "model_stopped",
    # A device serving the model ran out of GPU memory
    "out_of_memory",
    # The GPU of a device serving the model timed out
    "gpu_timeout",
    # The model stopped making progress and is being restarted
    "model_stalled",
    # The model's process failed in some other way
    "model_failed",
    # The request's stream ended for another reason (such as a change of master)
    "interrupted",
    # The model isn't ready to serve yet (downloading, loading, or failed to start)
    "model_not_ready",
    # The prompt and max_tokens don't fit in the model's context window (OpenAI's code)
    "context_length_exceeded",
    # The request can't be run as it is (its chat template can't render it, say)
    "invalid_request",
]

DEVICE_DISCONNECTED = (
    "A device serving this model disconnected before the reply finished."
)
DEVICE_LEFT = "A device serving this model left the cluster before the reply finished."
# A request whose model's instances all have a device this node can't reach now (Suspect or Gone
# in its membership view), or that was running on one when the device became unreachable. It
# ends at once rather than wait for output that can't arrive, or for the device to be removed.
# The device may be back in a moment: it is unreachable from when it stops answering, well before
# it is known to be gone.
DEVICE_UNREACHABLE = (
    "A device serving this model can't be reached right now. Try again in a moment."
)
MODEL_STOPPED = "The model was stopped before the reply finished."
# A request sent to an instance that was deleted (the model was stopped, or a node was lost)
# just before the request could start
INSTANCE_GONE_BEFORE_START = (
    "The model instance was shut down before this request could start."
)
OUT_OF_MEMORY = (
    "A device serving this model ran out of GPU memory before the reply finished."
)
GPU_TIMEOUT = (
    "The GPU of a device serving this model timed out before the reply finished."
)
MODEL_STALLED = (
    "The model stopped responding before the reply finished and is being restarted."
)
MODEL_FAILED = "The model stopped unexpectedly before the reply finished."
# Followed by what the model's own error says
MODEL_FAILED_BECAUSE = "The model stopped before the reply finished: "
INTERRUPTED = "The model instance serving this request stopped before it finished"
# Followed by what the model is doing: see exo.shared.model_readiness
MODEL_NOT_READY = "The model is not ready yet: "
# Followed by the numbers: OpenAI's words, which clients and tools recognise
CONTEXT_LENGTH_EXCEEDED = "This model's maximum context length is "
# Followed by what was wrong with the request
INVALID_REQUEST = "This request can't be run on this model: "
# Followed by the model's name: the master had no instance of the model for a request that the
# API had sent because it saw one, so it had just been deleted
NO_INSTANCE_FOR_MODEL = "No instance found for model "

_CODES: dict[str, GenerationErrorCode] = {
    DEVICE_DISCONNECTED: "device_left",
    DEVICE_LEFT: "device_left",
    DEVICE_UNREACHABLE: "device_unreachable",
    MODEL_STOPPED: "model_stopped",
    INSTANCE_GONE_BEFORE_START: "model_stopped",
    OUT_OF_MEMORY: "out_of_memory",
    GPU_TIMEOUT: "gpu_timeout",
    MODEL_STALLED: "model_stalled",
    MODEL_FAILED: "model_failed",
    INTERRUPTED: "interrupted",
}
_PREFIX_CODES: tuple[tuple[str, GenerationErrorCode], ...] = (
    (MODEL_FAILED_BECAUSE, "model_failed"),
    (NO_INSTANCE_FOR_MODEL, "model_stopped"),
    (MODEL_NOT_READY, "model_not_ready"),
    (CONTEXT_LENGTH_EXCEEDED, "context_length_exceeded"),
    (INVALID_REQUEST, "invalid_request"),
)


# What a request in flight is told when its instance is removed, by the removal's reason
_REMOVAL_MESSAGES: dict[InstanceRemovalReason, str] = {
    "NodeLost": DEVICE_LEFT,
    "UserStopped": MODEL_STOPPED,
    "Duplicate": INTERRUPTED,
    "StartFailed": MODEL_FAILED,
}


def removal_message(reason: InstanceRemovalReason) -> str:
    """What a request in flight is told when its instance is removed for this reason."""
    return _REMOVAL_MESSAGES[reason]


def generation_error_code(message: str) -> GenerationErrorCode | None:
    """The code of a request's error message, None for a message exo doesn't know"""
    if (code := _CODES.get(message)) is not None:
        return code
    return next(
        (code for prefix, code in _PREFIX_CODES if message.startswith(prefix)), None
    )


# A file path: its folders (which can name the user) are left out of what the user is told
_PATH = re.compile(r"(?<![\w:/.~])~?(?:/[^\s/'\"`:,;()]+)+/?")
# The longest error detail passed on to the user
_MAX_DETAIL = 160


def user_facing_detail(error: str) -> str:
    """An error's first line, fit to tell a user: at most 160 characters, and with only the
    file name of any path in it."""
    line = next((line.strip() for line in error.splitlines() if line.strip()), "")
    line = _PATH.sub(_file_name, line)
    if len(line) > _MAX_DETAIL:
        line = line[: _MAX_DETAIL - 1].rstrip() + "…"
    return line


def _file_name(path: re.Match[str]) -> str:
    parts = [part for part in path.group(0).split("/") if part]
    # A single part ("/v1", say) names no folder; a longer path is cut to its file name
    return path.group(0) if len(parts) <= 1 else parts[-1]
