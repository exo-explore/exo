import contextlib
import os
from collections.abc import AsyncIterator

import anyio
import pytest
from anyio import EndOfStream, create_task_group, fail_after

from exo.utils.async_process import AsyncProcess
from exo.utils.channels import MpReceiver, MpSender, Receiver, mp_channel
from exo.utils.daemon import detach_stdio_to_devnull


def _write_before_and_after_detach() -> None:
    os.write(1, b"before stdout\n")
    os.write(2, b"before stderr\n")
    detach_stdio_to_devnull()
    os.write(1, b"after stdout\n")
    os.write(2, b"after stderr\n")


def _write_grandchild_stdio(label: str) -> None:
    os.write(1, f"{label} stdout\n".encode())
    os.write(2, f"{label} stderr\n".encode())


async def _spawn_grandchild_and_report(
    result_sender: MpSender[tuple[int, bytes, bytes]],
    label: str,
) -> None:
    result_sender.send(await _collect_spawned_child(label))
    result_sender.close()


async def _collect_spawned_child(label: str) -> tuple[int, bytes, bytes]:
    process = AsyncProcess(_write_grandchild_stdio, args=(label,))
    async with _started_process(process):
        return await _collect_process_output(process)


def _detach_stdio_then_spawn_captured_child(
    result_sender: MpSender[tuple[int, bytes, bytes]],
) -> None:
    detach_stdio_to_devnull()
    anyio.run(_spawn_grandchild_and_report, result_sender, "grandchild")


def _detach_stdio_then_spawn_captured_children_sequentially(
    result_sender: MpSender[list[tuple[int, bytes, bytes]]],
) -> None:
    async def run_children() -> list[tuple[int, bytes, bytes]]:
        results: list[tuple[int, bytes, bytes]] = []
        for index in range(5):
            results.append(await _collect_spawned_child(f"grandchild-{index}"))
        return results

    detach_stdio_to_devnull()
    result_sender.send(anyio.run(run_children))
    result_sender.close()


async def _collect_stream(stream: Receiver[bytes], output: bytearray) -> None:
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

    async with create_task_group() as collect_group:
        collect_group.start_soon(_collect_stream, process.stdout, stdout)
        collect_group.start_soon(_collect_stream, process.stderr, stderr)
        exitcodes.append(await process.wait())

    if not exitcodes:
        raise RuntimeError("process exited without a return code")
    return exitcodes[0], bytes(stdout), bytes(stderr)


@contextlib.asynccontextmanager
async def _started_process(process: AsyncProcess) -> AsyncIterator[None]:
    async with create_task_group() as task_group:
        await task_group.start(process.run)
        try:
            yield
        finally:
            await process.stop()


async def _run_process_and_receive[T](
    process: AsyncProcess,
    recv: MpReceiver[T],
    *,
    timeout: float,
) -> tuple[int, T]:
    async with _started_process(process):
        with fail_after(timeout):
            result = await recv.receive_async()
            exitcode = await process.wait()

    return exitcode, result


@pytest.mark.anyio
async def test_detach_stdio_to_devnull_redirects_stdio_away_from_capture() -> None:
    process = AsyncProcess(_write_before_and_after_detach)

    async with _started_process(process):
        exitcode, stdout, stderr = await _collect_process_output(process)

    assert exitcode == 0
    assert stdout == b"before stdout\n"
    assert stderr == b"before stderr\n"


@pytest.mark.anyio
async def test_detached_stdio_process_can_spawn_and_capture_child_stdio() -> None:
    send, recv = mp_channel[tuple[int, bytes, bytes]]()
    process = AsyncProcess(_detach_stdio_then_spawn_captured_child, args=(send,))

    try:
        daemonized_parent_exitcode, result = await _run_process_and_receive(
            process, recv, timeout=5
        )
    finally:
        recv.close()

    child_exitcode, child_stdout, child_stderr = result

    assert daemonized_parent_exitcode == 0
    assert child_exitcode == 0
    assert child_stdout == b"grandchild stdout\n"
    assert child_stderr == b"grandchild stderr\n"


@pytest.mark.anyio
async def test_detached_stdio_process_can_spawn_captured_children_sequentially() -> (
    None
):
    send, recv = mp_channel[list[tuple[int, bytes, bytes]]]()
    process = AsyncProcess(
        _detach_stdio_then_spawn_captured_children_sequentially,
        args=(send,),
    )

    try:
        daemonized_parent_exitcode, results = await _run_process_and_receive(
            process, recv, timeout=10
        )
    finally:
        recv.close()

    assert daemonized_parent_exitcode == 0
    assert results == [
        (
            0,
            f"grandchild-{index} stdout\n".encode(),
            f"grandchild-{index} stderr\n".encode(),
        )
        for index in range(5)
    ]
