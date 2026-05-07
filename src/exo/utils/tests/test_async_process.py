import contextlib
import os
import signal
import sys
import time
from collections.abc import AsyncIterator, Callable
from types import FrameType

import mlx.core as mx
import pytest
from _pytest.capture import CaptureFixture
from anyio import EndOfStream, create_task_group, fail_after
from pytest import MonkeyPatch

import exo.utils.async_process as async_process
from exo.utils.async_process import (
    AsyncProcess,
)
from exo.utils.channels import MpSender, Receiver, mp_channel


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


def _exit_on_sigterm(exitcode: int) -> None:
    def handle_sigterm(_signum: int, _frame: FrameType | None) -> None:
        os._exit(exitcode)

    signal.signal(signal.SIGTERM, handle_sigterm)
    os.write(1, b"sigterm-ready\n")
    while True:
        time.sleep(0.1)


def _exit_after_repeated_sigterm(required_count: int, exitcode: int) -> None:
    sigterm_count = 0

    def handle_sigterm(_signum: int, _frame: FrameType | None) -> None:
        nonlocal sigterm_count
        sigterm_count += 1
        if sigterm_count >= required_count:
            os._exit(exitcode)

    signal.signal(signal.SIGTERM, handle_sigterm)
    os.write(1, b"sigterm-ready\n")
    while True:
        time.sleep(0.1)


def _ignore_sigterm_forever() -> None:
    signal.signal(signal.SIGTERM, signal.SIG_IGN)
    os.write(1, b"sigterm-ready\n")
    while True:
        time.sleep(0.1)


def _sleep_forever() -> None:
    while True:
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
    stream: Receiver[bytes],
    output: bytearray,
) -> None:
    while True:
        try:
            output.extend(await stream.receive())
        except EndOfStream:
            return


async def _collect_process_output(
    process: AsyncProcess,
) -> tuple[int, bytes, bytes]:
    stdout = bytearray()
    stderr = bytearray()
    exitcodes: list[int] = []

    async with create_task_group() as task_group:
        task_group.start_soon(_collect_stream, process.stdout, stdout)
        task_group.start_soon(_collect_stream, process.stderr, stderr)
        exitcodes.append(await process.wait())

    if not exitcodes:
        raise RuntimeError("process exited without a return code")
    return exitcodes[0], bytes(stdout), bytes(stderr)


def _fd_identity(fd: int) -> tuple[int, int]:
    fd_stat = os.fstat(fd)
    return fd_stat.st_dev, fd_stat.st_ino


def _fd_count() -> int | None:
    for fd_dir in ("/proc/self/fd", "/dev/fd"):
        with contextlib.suppress(OSError):
            return len(os.listdir(fd_dir))
    return None


@contextlib.asynccontextmanager
async def _started_process(process: AsyncProcess) -> AsyncIterator[None]:
    async with create_task_group() as task_group:
        await task_group.start(process.run)
        try:
            yield
        finally:
            await process.stop()


async def _run_and_collect(
    target: Callable[..., object] | None,
    *,
    args: tuple[object, ...] = (),
    kwargs: dict[str, object] | None = None,
) -> tuple[int, bytes, bytes]:
    process = AsyncProcess(
        target,
        args=args,
        kwargs=kwargs,
    )
    async with _started_process(process):
        return await _collect_process_output(process)


@pytest.mark.anyio
async def test_spawn_process_captures_stdout_and_stderr_separately(
    capfd: CaptureFixture[str],
) -> None:
    process = AsyncProcess(
        _write_to_stdio,
        args=("child",),
        kwargs={"stderr_suffix": "error"},
    )
    async with _started_process(process):
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


@pytest.mark.anyio
async def test_process_with_no_target_exits_successfully() -> None:
    exitcode, stdout, stderr = await _run_and_collect(None)

    assert exitcode == 0
    assert stdout == b""
    assert stderr == b""


@pytest.mark.anyio
async def test_output_receivers_and_wait_are_safe_immediately_after_run_starts() -> (
    None
):
    process = AsyncProcess(
        _write_to_stdio,
        args=("immediate",),
        kwargs={"stderr_suffix": "error"},
    )
    result: tuple[int, bytes, bytes] | None = None

    async with create_task_group() as task_group:
        await task_group.start(process.run)
        try:
            result = await _collect_process_output(process)
        finally:
            await process.stop()

    assert result is not None
    exitcode, stdout, stderr = result
    assert exitcode == 0
    assert b"immediate: fd stdout\n" in stdout
    assert b"immediate: fd stderr error\n" in stderr


@pytest.mark.anyio
async def test_stop_before_run_raises() -> None:
    process = AsyncProcess(
        _write_to_stdio,
        args=("never",),
        kwargs={"stderr_suffix": "run"},
    )

    assert not process.is_alive()
    with pytest.raises(RuntimeError, match="process has not been started"):
        await process.stop()


@pytest.mark.anyio
async def test_process_run_is_one_shot() -> None:
    process = AsyncProcess(None)

    await process.run()

    with pytest.raises(RuntimeError, match="process has already been started"):
        await process.run()


@pytest.mark.anyio
async def test_process_started_with_task_group_start_can_stop_immediately() -> None:
    process = AsyncProcess(_sleep_forever)

    async with create_task_group() as task_group:
        await task_group.start(process.run)
        assert process.is_alive()
        with fail_after(2):
            await process.stop()

    assert not process.is_alive()


@pytest.mark.anyio
async def test_stdout_receiver_yields_bytes_chunks() -> None:
    process = AsyncProcess(_write_large_output)

    async with _started_process(process):
        first_stdout = await process.stdout.receive()
        exitcode, remaining_stdout, stderr = await _collect_process_output(process)

    assert exitcode == 0
    assert first_stdout + remaining_stdout == b"stdout-0123456789"
    assert stderr == b"stderr-0123456789"


@pytest.mark.anyio
async def test_output_can_be_read_after_process_exits() -> None:
    process = AsyncProcess(_write_large_output)

    async with create_task_group() as task_group:
        await task_group.start(process.run)
        assert await process.wait() == 0

    assert await process.stdout.receive() == b"stdout-0123456789"
    assert await process.stderr.receive() == b"stderr-0123456789"
    with pytest.raises(EndOfStream):
        await process.stdout.receive()
    with pytest.raises(EndOfStream):
        await process.stderr.receive()


@pytest.mark.anyio
async def test_large_stdout_and_stderr_are_not_lost() -> None:
    size = 1024 * 1024
    exitcode, stdout, stderr = await _run_and_collect(
        _write_large_exact_output,
        args=(size,),
    )

    assert exitcode == 0
    assert stdout == b"stdout:" + (b"x" * size)
    assert stderr == b"stderr:" + (b"y" * size)


@pytest.mark.anyio
async def test_child_exception_traceback_is_captured_from_stderr() -> None:
    process = AsyncProcess(_raise_after_stderr_write)

    async with _started_process(process):
        exitcode, _, stderr_bytes = await _collect_process_output(process)

    assert exitcode == 1
    stderr = stderr_bytes.decode("utf-8", errors="replace")
    assert "stderr before exception" in stderr
    assert "RuntimeError: child boom" in stderr


@pytest.mark.anyio
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


@pytest.mark.anyio
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


@pytest.mark.anyio
async def test_repeated_crashing_children_do_not_grow_parent_fd_table() -> None:
    await _run_and_collect(_exit_after_stdio_write, args=("warmup", 23))
    before = _fd_count()
    if before is None:
        pytest.skip("fd table count is not available on this platform")

    for iteration in range(20):
        exitcode, stdout, stderr = await _run_and_collect(
            _exit_after_stdio_write,
            args=(f"fd-child-{iteration}", 31),
        )

        assert exitcode == 31
        assert stdout == f"fd-child-{iteration}: stdout before _exit\n".encode()
        assert stderr == f"fd-child-{iteration}: stderr before _exit\n".encode()

    after = _fd_count()
    assert after is not None
    assert after <= before + 2


@pytest.mark.anyio
async def test_stop_allows_child_to_exit_after_sigterm() -> None:
    process = AsyncProcess(_exit_on_sigterm, args=(43,))

    async with _started_process(process):
        assert await process.stdout.receive() == b"sigterm-ready\n"

        with fail_after(2):
            await process.stop()

    assert process.exitcode == 43


@pytest.mark.anyio
async def test_stop_retries_sigterm_before_sigkill(monkeypatch: MonkeyPatch) -> None:
    monkeypatch.setattr(async_process, "_TERMINATE_GRACE_SECONDS", 0.01)
    monkeypatch.setattr(async_process, "_TERMINATE_RETRY_GRACE_SECONDS", 0.01)
    process = AsyncProcess(_exit_after_repeated_sigterm, args=(3, 44))

    async with _started_process(process):
        assert await process.stdout.receive() == b"sigterm-ready\n"

        with fail_after(2):
            await process.stop()

    assert process.exitcode == 44


@pytest.mark.anyio
async def test_stop_escalates_to_sigkill_when_child_ignores_sigterm(
    monkeypatch: MonkeyPatch,
) -> None:
    monkeypatch.setattr(async_process, "_TERMINATE_GRACE_SECONDS", 0.1)
    monkeypatch.setattr(async_process, "_TERMINATE_RETRY_GRACE_SECONDS", 0.01)
    process = AsyncProcess(_ignore_sigterm_forever)

    async with _started_process(process):
        assert await process.stdout.receive() == b"sigterm-ready\n"

        with fail_after(3):
            await process.stop()

    assert process.exitcode == -signal.SIGKILL


@pytest.mark.anyio
async def test_process_can_use_mp_channel_with_global_spawn_context() -> None:
    send, recv = mp_channel[str]()
    process = AsyncProcess(_send_over_mp_channel, args=(send,))

    async with _started_process(process):
        with fail_after(2):
            assert await recv.receive_async() == "hello from child"
            assert await process.wait() == 0

    with contextlib.suppress(Exception):
        recv.close()


@pytest.mark.anyio
@pytest.mark.skip(reason="manual MLX OOM isolation check")
async def test_death(capsys: CaptureFixture[str]) -> None:
    with capsys.disabled():
        process = AsyncProcess(_mlx_force_oom)
        stdout = b""
        stderr = b""
        async with _started_process(process):
            _, stdout, stderr = await _collect_process_output(process)

        print("PARENT: done")

        print("CHILD out:", stdout.decode("utf-8", errors="replace"))
        print("CHILD err:", stderr.decode("utf-8", errors="replace"), "hello :)")
