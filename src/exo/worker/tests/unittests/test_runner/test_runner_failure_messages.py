"""A request whose runner fails ends with one short sentence that has a stable code, never a
traceback. The full error is logged on the node."""

import pytest

from exo.shared.request_errors import GenerationErrorCode, generation_error_code
from exo.worker.runner.bootstrap import RunnerTerminationError
from exo.worker.runner.diagnostics import (
    KnownRunnerDiagnostic,
    RunnerMetalGpuTimeout,
    RunnerMetalOutOfMemory,
    RunnerRingSocketReceivingError,
    RunnerRingTransportError,
)
from exo.worker.runner.failures import (
    RunnerLostPeersError,
    RunnerStalledError,
    RunnerStoppedError,
    describe_failure,
)

# What a runner raised when another node of its instance quit mid-generation
_PEER_LOST_TRACEBACK = """Traceback (most recent call last):
  File "exo/worker/runner/bootstrap.py", line 88, in entrypoint
    runner.main()
  File "exo/worker/runner/runner.py", line 221, in main
    self.handle_first_task(item)
  File "exo/worker/engines/mlx/generator/generate.py", line 640, in mlx_generate
    mx.eval(tokens)
RuntimeError: [ring] connection to a peer was lost
"""


def _raised(
    exception_type: str, message: str, traceback: str = ""
) -> RunnerTerminationError:
    return RunnerTerminationError(
        exception_type=exception_type,
        exception_message=message,
        exception_repr=f"{exception_type}({message!r})",
        traceback=traceback,
    )


@pytest.mark.parametrize(
    ("error", "diagnostics", "code", "message"),
    [
        (
            _raised(
                "RuntimeError",
                "[ring] connection to a peer was lost",
                _PEER_LOST_TRACEBACK,
            ),
            [],
            "device_left",
            "A device serving this model disconnected before the reply finished.",
        ),
        (
            RunnerLostPeersError("Lost the connection to the instance's other runners"),
            [],
            "device_left",
            "A device serving this model disconnected before the reply finished.",
        ),
        (
            RunnerStoppedError("Runner found to be dead"),
            [RunnerRingTransportError(message="Ring transport aborted")],
            "device_left",
            "A device serving this model disconnected before the reply finished.",
        ),
        (
            None,
            [
                RunnerRingSocketReceivingError(
                    message="Ring socket receive failed: errno 54",
                    error_number=54,
                    error_name="ECONNRESET",
                    error_description="Connection reset by peer",
                )
            ],
            "device_left",
            "A device serving this model disconnected before the reply finished.",
        ),
        (
            _raised("RuntimeError", "[METAL] Command buffer execution failed"),
            [RunnerMetalOutOfMemory(message="Out of GPU memory")],
            "out_of_memory",
            "A device serving this model ran out of GPU memory before the reply finished.",
        ),
        (
            RunnerStoppedError("Runner found to be dead"),
            [RunnerMetalGpuTimeout(message="Metal GPU timeout")],
            "gpu_timeout",
            "The GPU of a device serving this model timed out before the reply finished.",
        ),
        (
            RunnerStalledError("Runner made no progress for 90s while generating"),
            [],
            "model_stalled",
            "The model stopped responding before the reply finished and is being restarted.",
        ),
        (
            _raised(
                "ValueError",
                "Unsupported image format.\nSecond line",
                "Traceback (most recent call last):\n  ...\nValueError: Unsupported image format.",
            ),
            [],
            "model_failed",
            "The model stopped before the reply finished: Unsupported image format.",
        ),
        (
            RunnerStoppedError("Runner found to be dead"),
            [],
            "model_failed",
            "The model stopped unexpectedly before the reply finished.",
        ),
        (
            RunnerStoppedError("Runner took more than 600s to load the model"),
            [],
            "model_failed",
            "The model stopped unexpectedly before the reply finished.",
        ),
        (
            None,
            [],
            "model_failed",
            "The model stopped unexpectedly before the reply finished.",
        ),
        # Raised in the engine, as it generates
        (
            RuntimeError("[ring] connection to a peer was lost"),
            [],
            "device_left",
            "A device serving this model disconnected before the reply finished.",
        ),
        (
            RuntimeError(
                "[METAL] Command buffer execution failed: Insufficient Memory "
                "(00000008:kIOGPUCommandBufferCallbackErrorOutOfMemory)"
            ),
            [],
            "out_of_memory",
            "A device serving this model ran out of GPU memory before the reply finished.",
        ),
        (
            RuntimeError(
                "[METAL] Command buffer execution failed: Caused GPU Timeout Error "
                "(00000002:kIOGPUCommandBufferCallbackErrorTimeout)"
            ),
            [],
            "gpu_timeout",
            "The GPU of a device serving this model timed out before the reply finished.",
        ),
        (
            ValueError("Chat template failed to render"),
            [],
            "model_failed",
            "The model stopped before the reply finished: Chat template failed to render.",
        ),
    ],
)
def test_each_failure_gets_a_short_message_and_a_stable_code(
    error: RunnerTerminationError | Exception | None,
    diagnostics: list[KnownRunnerDiagnostic],
    code: GenerationErrorCode,
    message: str,
) -> None:
    assert describe_failure(error, diagnostics) == message
    assert generation_error_code(message) == code


def test_a_long_runner_error_is_cut_to_one_short_line() -> None:
    message = describe_failure(_raised("RuntimeError", "x" * 1000), [])

    assert generation_error_code(message) == "model_failed"
    assert "\n" not in message
    assert len(message) < 220
    assert message.endswith("….")


@pytest.mark.parametrize(
    ("error", "told"),
    [
        (
            "[Errno 2] No such file or directory: '/opt/exo/models/m/model.safetensors'",
            "[Errno 2] No such file or directory: 'model.safetensors'",
        ),
        ("bad value in ~/exo/config.toml", "bad value in config.toml"),
        (
            "can't reach http://127.0.0.1:8080/v1/x",
            "can't reach http://127.0.0.1:8080/v1/x",
        ),
    ],
)
def test_a_runner_error_never_names_the_folders_of_a_path(
    error: str, told: str
) -> None:
    assert describe_failure(ValueError(error)) == (
        f"The model stopped before the reply finished: {told}."
    )
