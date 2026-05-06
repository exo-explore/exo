import contextlib
import os
import signal
import sys
import time
from collections.abc import Callable

import mlx.core as mx
import pytest
from _pytest.capture import CaptureFixture
from anyio import EndOfStream, create_task_group, fail_after
from anyio.abc import ByteReceiveStream

from exo.utils.async_process import (
    AsyncSpawnProcess,
)
from exo.utils.channels import MpSender, mp_channel


def _write_to_stdio(prefix: str, *, stderr_suffix: str) -> None:
    print(f"{prefix}: python stdout")
    print(f"{prefix}: python stderr {stderr_suffix}", file=sys.stderr)
    os.write(1, f"{prefix}: fd stdout\n".encode())
    os.write(2, f"{prefix}: fd stderr {stderr_suffix}\n".encode())


def _write_large_output() -> None:
    os.write(1, b"stdout-0123456789")
    os.write(2, b"stderr-0123456789")


def _write_all(fd: int, data: bytes) -> None:
    remaining = memoryview(data)
    while remaining:
        written = os.write(fd, remaining)
        remaining = remaining[written:]


def _write_large_exact_output(size: int) -> None:
    _write_all(1, b"stdout:" + (b"x" * size))
    _write_all(2, b"stderr:" + (b"y" * size))


def _raise_after_stderr_write() -> None:
    os.write(2, b"stderr before exception\n")
    raise RuntimeError("child boom")


def _exit_after_stdio_write(prefix: str, exitcode: int) -> None:
    os.write(1, f"{prefix}: stdout before _exit\n".encode())
    os.write(2, f"{prefix}: stderr before _exit\n".encode())
    os._exit(exitcode)


def _abort_after_stdio_write(prefix: str) -> None:
    os.write(1, f"{prefix}: stdout before abort\n".encode())
    os.write(2, f"{prefix}: stderr before abort\n".encode())
    os.abort()


def _close_stdio_and_exit() -> None:
    os.close(1)
    os.close(2)
    os._exit(0)


def _sleep_without_output() -> None:
    time.sleep(0.1)


def _send_over_mp_channel(send: MpSender[str]) -> None:
    send.send("hello from child")
    send.close()


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
        process: AsyncSpawnProcess,
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


def _fd_identity(fd: int) -> tuple[int, int]:
    fd_stat = os.fstat(fd)
    return fd_stat.st_dev, fd_stat.st_ino


def _fd_count() -> int | None:
    for fd_dir in ("/proc/self/fd", "/dev/fd"):
        with contextlib.suppress(OSError):
            return len(os.listdir(fd_dir))
    return None


async def _run_and_collect(
        target: Callable[..., object] | None,
        *,
        args: tuple[object, ...] = (),
        kwargs: dict[str, object] | None = None,
        stream_buffer_size: int = 16,
) -> tuple[int, bytes, bytes]:
    process = AsyncSpawnProcess(
        target,
        args=args,
        kwargs=kwargs,
        stream_buffer_size=stream_buffer_size,
    )
    await process.start()
    async with process:
        return await _collect_process_output(process)


@pytest.mark.asyncio
async def test_spawn_process_captures_stdout_and_stderr_separately(
        capfd: CaptureFixture[str],
) -> None:
    process = AsyncSpawnProcess(
        _write_to_stdio,
        args=("child",),
        kwargs={"stderr_suffix": "error"},
    )
    await process.start()

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
async def test_process_with_no_target_exits_successfully() -> None:
    exitcode, stdout, stderr = await _run_and_collect(None)

    assert exitcode == 0
    assert stdout == b""
    assert stderr == b""


@pytest.mark.asyncio
async def test_stdout_stream_honors_receive_size() -> None:
    process = AsyncSpawnProcess(_write_large_output)
    await process.start()

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
async def test_large_stdout_and_stderr_are_not_lost_with_bounded_buffers() -> None:
    size = 1024 * 1024
    exitcode, stdout, stderr = await _run_and_collect(
        _write_large_exact_output,
        args=(size,),
        stream_buffer_size=1,
    )

    assert exitcode == 0
    assert stdout == b"stdout:" + (b"x" * size)
    assert stderr == b"stderr:" + (b"y" * size)


@pytest.mark.asyncio
async def test_child_exception_traceback_is_captured_from_stderr() -> None:
    process = AsyncSpawnProcess(_raise_after_stderr_write)
    await process.start()

    async with process:
        exitcode, _, stderr_bytes = await _collect_process_output(process)

    assert exitcode == 1
    stderr = stderr_bytes.decode("utf-8", errors="replace")
    assert "stderr before exception" in stderr
    assert "RuntimeError: child boom" in stderr


@pytest.mark.asyncio
async def test_repeated_bad_children_do_not_pollute_or_replace_parent_stdio(
        capfd: CaptureFixture[str],
) -> None:
    stdout_object = sys.stdout
    stderr_object = sys.stderr
    stdout_identity = _fd_identity(1)
    stderr_identity = _fd_identity(2)

    cases: tuple[tuple[Callable[..., object], tuple[object, ...]], ...] = (
        (_raise_after_stderr_write, ()),
        (_exit_after_stdio_write, ("exit-child", 17)),
        (_abort_after_stdio_write, ("abort-child",)),
    )

    for iteration in range(3):
        for target, args in cases:
            exitcode, stdout, stderr = await _run_and_collect(
                target,
                args=args,
                stream_buffer_size=1,
            )

            assert exitcode != 0
            if target is _exit_after_stdio_write:
                assert stdout == b"exit-child: stdout before _exit\n"
                assert stderr == b"exit-child: stderr before _exit\n"
            elif target is _abort_after_stdio_write:
                assert b"abort-child: stdout before abort\n" in stdout
                assert b"abort-child: stderr before abort\n" in stderr
                assert exitcode == -signal.SIGABRT
            else:
                assert stdout == b""
                assert b"stderr before exception\n" in stderr
                assert b"RuntimeError: child boom" in stderr

        print(f"parent stdout still works {iteration}")
        print(f"parent stderr still works {iteration}", file=sys.stderr)

    parent_output = capfd.readouterr()

    assert sys.stdout is stdout_object
    assert sys.stderr is stderr_object
    assert _fd_identity(1) == stdout_identity
    assert _fd_identity(2) == stderr_identity
    assert "parent stdout still works 0" in parent_output.out
    assert "parent stdout still works 2" in parent_output.out
    assert "parent stderr still works 0" in parent_output.err
    assert "parent stderr still works 2" in parent_output.err
    assert "exit-child:" not in parent_output.out
    assert "exit-child:" not in parent_output.err
    assert "abort-child:" not in parent_output.out
    assert "abort-child:" not in parent_output.err
    assert "child boom" not in parent_output.err


@pytest.mark.asyncio
async def test_child_can_close_stdio_without_corrupting_parent_stdio(
        capfd: CaptureFixture[str],
) -> None:
    stdout_identity = _fd_identity(1)
    stderr_identity = _fd_identity(2)

    exitcode, stdout, stderr = await _run_and_collect(_close_stdio_and_exit)
    os.write(1, b"parent stdout after child closed stdio\n")
    os.write(2, b"parent stderr after child closed stdio\n")
    parent_output = capfd.readouterr()

    assert exitcode == 0
    assert stdout == b""
    assert stderr == b""
    assert _fd_identity(1) == stdout_identity
    assert _fd_identity(2) == stderr_identity
    assert "parent stdout after child closed stdio" in parent_output.out
    assert "parent stderr after child closed stdio" in parent_output.err


@pytest.mark.asyncio
async def test_repeated_crashing_children_do_not_grow_parent_fd_table() -> None:
    await _run_and_collect(_exit_after_stdio_write, args=("warmup", 23))
    before = _fd_count()
    if before is None:
        pytest.skip("fd table count is not available on this platform")

    for iteration in range(20):
        exitcode, stdout, stderr = await _run_and_collect(
            _exit_after_stdio_write,
            args=(f"fd-child-{iteration}", 31),
            stream_buffer_size=1,
        )

        assert exitcode == 31
        assert stdout == f"fd-child-{iteration}: stdout before _exit\n".encode()
        assert stderr == f"fd-child-{iteration}: stderr before _exit\n".encode()

    after = _fd_count()
    assert after is not None
    assert after <= before + 2


@pytest.mark.asyncio
async def test_aclose_can_cancel_idle_drainers_before_child_exits() -> None:
    process = AsyncSpawnProcess(_sleep_without_output)
    await process.start()

    with fail_after(2):
        await process.aclose()

    assert process.returncode == 0


@pytest.mark.asyncio
async def test_spawn_process_can_use_spawn_context_mp_channel() -> None:
    send, recv = mp_channel[str](context=AsyncSpawnProcess.context())
    process = AsyncSpawnProcess(_send_over_mp_channel, args=(send,))
    await process.start()

    async with process:
        with fail_after(2):
            assert await recv.receive_async() == "hello from child"
            assert await process.wait() == 0

    with contextlib.suppress(Exception):
        recv.close()


@pytest.mark.asyncio
@pytest.mark.skip(reason="manual MLX OOM isolation check")
async def test_death(capsys: CaptureFixture[str]) -> None:
    with capsys.disabled():
        process = AsyncSpawnProcess(_mlx_force_oom)
        await process.start()
        async with process:
            _, stdout, stderr = await _collect_process_output(process)

        print("PARENT: done")

        print("CHILD out:", stdout.decode("utf-8", errors="replace"))
        print("CHILD err:", stderr.decode("utf-8", errors="replace"), "hello :)")
