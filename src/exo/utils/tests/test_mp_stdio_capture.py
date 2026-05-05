import os
import sys
import time

import mlx.core as mx
import pytest
from _pytest.capture import CaptureFixture
from anyio import EndOfStream, create_task_group, fail_after
from anyio.abc import ByteReceiveStream

from exo.utils.mp_stdio_capture import (
    CapturedMpProcess,
    open_process,
)


def _write_to_stdio(prefix: str, *, stderr_suffix: str) -> None:
    print(f"{prefix}: python stdout")
    print(f"{prefix}: python stderr {stderr_suffix}", file=sys.stderr)
    os.write(1, f"{prefix}: fd stdout\n".encode())
    os.write(2, f"{prefix}: fd stderr {stderr_suffix}\n".encode())


def _write_large_output() -> None:
    os.write(1, b"stdout-0123456789")
    os.write(2, b"stderr-0123456789")


def _raise_after_stderr_write() -> None:
    os.write(2, b"stderr before exception\n")
    raise RuntimeError("child boom")


def _sleep_without_output() -> None:
    time.sleep(0.1)


def _mlx_force_oom(size: int = 40_000) -> None:
    """
    Force an Out-Of-Memory (OOM) error in MLX by performing large tensor operations.
    """
    print("CHILD: start")

    mx.set_default_device(mx.gpu)
    a = mx.random.uniform(shape=(size, size), dtype=mx.float32)
    b = mx.random.uniform(shape=(size, size), dtype=mx.float32)
    mx.eval(a, b)
    c = mx.matmul(a, b)
    d = mx.matmul(a, c)
    e = mx.matmul(b, c)
    f = mx.sigmoid(d + e)
    mx.eval(f)

    print("CHILD: end")


async def _collect_stream(
    stream: ByteReceiveStream,
    output: bytearray,
) -> None:
    while True:
        try:
            output.extend(await stream.receive())
        except EndOfStream:
            return


async def _collect_process_output(
    process: CapturedMpProcess,
) -> tuple[int, bytes, bytes]:
    stdout = bytearray()
    stderr = bytearray()

    async with create_task_group() as task_group:
        task_group.start_soon(_collect_stream, process.stdout, stdout)
        task_group.start_soon(_collect_stream, process.stderr, stderr)
        await process.wait()

    if process.returncode is None:
        raise RuntimeError("process exited without a return code")
    exitcode = process.returncode
    return exitcode, bytes(stdout), bytes(stderr)


@pytest.mark.asyncio
async def test_spawn_process_captures_stdout_and_stderr_separately(
    capfd: CaptureFixture[str],
) -> None:
    process = await open_process(
        _write_to_stdio,
        args=("child",),
        kwargs={"stderr_suffix": "error"},
    )

    async with process:
        exitcode, stdout_bytes, stderr_bytes = await _collect_process_output(process)

    parent_output = capfd.readouterr()
    stdout = stdout_bytes.decode("utf-8", errors="replace")
    stderr = stderr_bytes.decode("utf-8", errors="replace")

    assert exitcode == 0
    assert "child: python stdout" in stdout
    assert "child: fd stdout" in stdout
    assert "child: python stderr error" in stderr
    assert "child: fd stderr error" in stderr
    assert "child:" not in parent_output.out
    assert "child:" not in parent_output.err


@pytest.mark.asyncio
async def test_default_open_process_uses_spawn_backend() -> None:
    process = await open_process(
        _write_to_stdio,
        args=("default",),
        kwargs={"stderr_suffix": "error"},
    )
    async with process:
        exitcode, stdout, stderr = await _collect_process_output(process)

    assert exitcode == 0
    assert b"default: python stdout" in stdout
    assert b"default: python stderr error" in stderr


@pytest.mark.asyncio
async def test_stdout_stream_honors_receive_size() -> None:
    process = await open_process(_write_large_output)

    async with process:
        first_stdout = await process.stdout.receive(6)
        remaining_stdout = bytearray()
        stderr = bytearray()

        async with create_task_group() as task_group:
            task_group.start_soon(_collect_stream, process.stdout, remaining_stdout)
            task_group.start_soon(_collect_stream, process.stderr, stderr)
            await process.wait()

    if process.returncode is None:
        raise RuntimeError("process exited without a return code")
    exitcode = process.returncode
    assert exitcode == 0
    assert first_stdout == b"stdout"
    assert bytes(remaining_stdout) == b"-0123456789"
    assert bytes(stderr) == b"stderr-0123456789"


@pytest.mark.asyncio
async def test_child_exception_traceback_is_captured_from_stderr() -> None:
    process = await open_process(_raise_after_stderr_write)

    async with process:
        exitcode, _, stderr_bytes = await _collect_process_output(process)

    assert exitcode == 1
    stderr = stderr_bytes.decode("utf-8", errors="replace")
    assert "stderr before exception" in stderr
    assert "RuntimeError: child boom" in stderr


@pytest.mark.asyncio
async def test_aclose_can_cancel_idle_drainers_before_child_exits() -> None:
    process = await open_process(_sleep_without_output)

    with fail_after(2):
        await process.aclose()

    assert process.returncode == 0


@pytest.mark.asyncio
async def test_death(capsys: CaptureFixture[str]) -> None:
    with capsys.disabled():
        process = await open_process(_mlx_force_oom)
        async with process:
            _, stdout, stderr = await _collect_process_output(process)

        print("PARENT: done")

        print("CHILD out:", stdout.decode("utf-8", errors="replace"))
        print("CHILD err:", stderr.decode("utf-8", errors="replace"), "hello :)")
