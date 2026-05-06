import os

import pytest
from anyio import EndOfStream, create_task_group
from anyio.abc import ByteReceiveStream

from exo.utils.async_process import AsyncSpawnProcess
from exo.utils.daemon import detach_stdio_to_devnull


def _write_before_and_after_detach() -> None:
    os.write(1, b"before stdout\n")
    os.write(2, b"before stderr\n")
    detach_stdio_to_devnull()
    os.write(1, b"after stdout\n")
    os.write(2, b"after stderr\n")


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
