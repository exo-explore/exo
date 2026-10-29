"""A runner whose GPU ran out of memory says so: otherwise only its peers' lost-connection
errors name a cause."""

from exo.worker.runner.diagnostics import (
    RunnerDiagnosticCollector,
    RunnerMetalGpuTimeout,
    RunnerMetalOutOfMemory,
    RunnerUnknown,
)

OUT_OF_MEMORY = (
    "[METAL] Command buffer execution failed: Insufficient Memory "
    "(00000008:kIOGPUCommandBufferCallbackErrorOutOfMemory)"
)


def test_an_aborted_runner_out_of_gpu_memory_is_recognised() -> None:
    collector = RunnerDiagnosticCollector()

    collector.record_line(
        "libc++abi: terminating due to uncaught exception of type std::runtime_error: "
        + OUT_OF_MEMORY
    )

    (diagnostic,) = collector.diagnostics()
    assert isinstance(diagnostic, RunnerMetalOutOfMemory)
    assert diagnostic.message.startswith("Out of GPU memory")


def test_other_metal_errors_are_not_taken_for_running_out_of_memory() -> None:
    collector = RunnerDiagnosticCollector()

    collector.record_line(
        "[METAL] Command buffer execution failed: Caused GPU Timeout Error "
        "(00000002:kIOGPUCommandBufferCallbackErrorTimeout)"
    )
    collector.record_line("[ring] Socket 54 was closed by the peer")

    assert [type(d) for d in collector.diagnostics()] == [
        RunnerMetalGpuTimeout,
        RunnerUnknown,
    ]
