import asyncio
from multiprocessing.context import SpawnProcess
import os
import multiprocessing as mp

import pytest
from _pytest.capture import CaptureFixture
from exo_rs import (
    CliArgs,
    NetworkingHandle,
    Pidfile,
    FromSwarm,
)


@pytest.mark.asyncio
async def test_sleep_on_multiple_items() -> None:
    print("PYTHON: starting handle")
    h = NetworkingHandle.new(os.urandom(16).hex().lstrip("0"), "default", 52414, 52413)
    print("PYTHON: handle started")

    rt = asyncio.create_task(_await_recv(h))

    # sleep for 4 ticks
    for i in range(10):
        await asyncio.sleep(1)

        await h.gossipsub_publish("topic", b"somehting or other")


async def _await_recv(h: NetworkingHandle):
    while True:
        event = await h.recv()
        match event:
            case FromSwarm.Connection() as c:
                print(f"PYTHON: connection update: {c}")
            case FromSwarm.Message() as m:
                print(f"PYTHON: message: {m}")
            case _:
                raise Exception("logical error")


def test_pickling(capsys: CaptureFixture[str]):
    with capsys.disabled():
        p = mp.get_context("spawn").Process(
            target=run_mp, args=(CliArgs.parse_from(["exo"]),)
        )
        p.start()
        p.join()


def run_mp(args: CliArgs):
    print("it got here")


if __name__ == "__main__":
    asyncio.run(test_sleep_on_multiple_items())
