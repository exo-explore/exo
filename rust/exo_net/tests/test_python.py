import asyncio

import pytest
from _pytest.capture import CaptureFixture
from exo_pyo3_bindings import (
    Keypair,
    NetworkingHandle,
    Pidfile,
    PyFromSwarm,
)


@pytest.mark.asyncio
async def test_sleep_on_multiple_items() -> None:
    print("PYTHON: starting handle")
    h = NetworkingHandle(Keypair.generate(), [], 0)
    print("PYTHON: handle started")

    rt = asyncio.create_task(_await_recv(h))

    # sleep for 4 ticks
    for i in range(10):
        await asyncio.sleep(1)

        await h.gossipsub_publish("topic", b"somehting or other")


def test_pidfile(capsys: CaptureFixture[str]):
    with capsys.disabled():
        print("\nbefore python")
        scoped_lock_file()
        print("after python")


async def _await_recv(h: NetworkingHandle):
    while True:
        event = await h.recv()
        match event:
            case PyFromSwarm.Connection() as c:
                print(f"PYTHON: connection update: {c}")
            case PyFromSwarm.Message() as m:
                print(f"PYTHON: message: {m}")


def scoped_lock_file():
    a = Pidfile("/tmp/lock.pid", 0o0600)

if __name__ == "__main__":
    asyncio.run(test_sleep_on_multiple_items())
