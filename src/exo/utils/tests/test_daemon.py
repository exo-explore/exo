import os

import anyio
import pytest
from anyio import EndOfStream, create_task_group, fail_after
from anyio.abc import ByteReceiveStream

from exo.utils.async_process import AsyncSpawnProcess
from exo.utils.channels import MpSender, mp_channel
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
    process = AsyncSpawnProcess(_write_grandchild_stdio, args=(label,))
    exitcode: int | None = None
    stdout = bytearray()
    stderr = bytearray()

    async with create_task_group() as task_group:
        task_group.start_soon(process.run)
        await process.wait_started()
        async with create_task_group() as collect_group:
            collect_group.start_soon(_collect_stream, process.stdout, stdout)
            collect_group.start_soon(_collect_stream, process.stderr, stderr)
            exitcode = await process.wait()

    if exitcode is None:
        raise RuntimeError("grandchild process was not collected")
    result_sender.send((exitcode, bytes(stdout), bytes(stderr)))
    result_sender.close()


async def _collect_spawned_child(label: str) -> tuple[int, bytes, bytes]:
    process = AsyncSpawnProcess(_write_grandchild_stdio, args=(label,))
    exitcode: int | None = None
    stdout = bytearray()
    stderr = bytearray()

    async with create_task_group() as task_group:
        task_group.start_soon(process.run)
        await process.wait_started()
        async with create_task_group() as collect_group:
            collect_group.start_soon(_collect_stream, process.stdout, stdout)
            collect_group.start_soon(_collect_stream, process.stderr, stderr)
            exitcode = await process.wait()

    if exitcode is None:
        raise RuntimeError("child process was not collected")
    return exitcode, bytes(stdout), bytes(stderr)


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


async def _collect_stream(stream: ByteReceiveStream, output: bytearray) -> None:
    while True:
        try:
            output.extend(await stream.receive())
        except EndOfStream:
            return


@pytest.mark.asyncio
async def test_detach_stdio_to_devnull_redirects_stdio_away_from_capture() -> None:
    process = AsyncSpawnProcess(_write_before_and_after_detach)
    stdout = bytearray()
    stderr = bytearray()

    async with create_task_group() as task_group:
        task_group.start_soon(process.run)
        await process.wait_started()
        async with create_task_group() as collect_group:
            collect_group.start_soon(_collect_stream, process.stdout, stdout)
            collect_group.start_soon(_collect_stream, process.stderr, stderr)
            assert await process.wait() == 0

    assert stdout == b"before stdout\n"
    assert stderr == b"before stderr\n"


@pytest.mark.asyncio
async def test_detached_stdio_process_can_spawn_and_capture_child_stdio() -> None:
    send, recv = mp_channel[tuple[int, bytes, bytes]](
        context=AsyncSpawnProcess.context()
    )
    process = AsyncSpawnProcess(_detach_stdio_then_spawn_captured_child, args=(send,))
    result: tuple[int, bytes, bytes] | None = None
    daemonized_parent_exitcode: int | None = None

    try:
        async with create_task_group() as task_group:
            task_group.start_soon(process.run)
            await process.wait_started()
            with fail_after(5):
                result = await recv.receive_async()
                daemonized_parent_exitcode = await process.wait()
    finally:
        recv.close()

    if result is None:
        raise RuntimeError("daemonized parent did not report grandchild result")
    child_exitcode, child_stdout, child_stderr = result

    assert daemonized_parent_exitcode == 0
    assert child_exitcode == 0
    assert child_stdout == b"grandchild stdout\n"
    assert child_stderr == b"grandchild stderr\n"


@pytest.mark.asyncio
async def test_detached_stdio_process_can_spawn_captured_children_sequentially() -> (
    None
):
    send, recv = mp_channel[list[tuple[int, bytes, bytes]]](
        context=AsyncSpawnProcess.context()
    )
    process = AsyncSpawnProcess(
        _detach_stdio_then_spawn_captured_children_sequentially,
        args=(send,),
    )
    results: list[tuple[int, bytes, bytes]] | None = None
    daemonized_parent_exitcode: int | None = None

    try:
        async with create_task_group() as task_group:
            task_group.start_soon(process.run)
            await process.wait_started()
            with fail_after(10):
                results = await recv.receive_async()
                daemonized_parent_exitcode = await process.wait()
    finally:
        recv.close()

    if results is None:
        raise RuntimeError("daemonized parent did not report child results")

    assert daemonized_parent_exitcode == 0
    assert results == [
        (
            0,
            f"grandchild-{index} stdout\n".encode(),
            f"grandchild-{index} stderr\n".encode(),
        )
        for index in range(5)
    ]
