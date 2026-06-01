import asyncio
from contextlib import suppress

import pytest
from _pytest.capture import CaptureFixture
from exo_rs.ident import Keypair
from exo_rs.networking import (
    FromSwarm,
    NetworkingHandle,
    NoPeersSubscribedToTopicError,
)
from exo_rs.pidfile import Pidfile


@pytest.mark.asyncio
async def test_sleep_on_multiple_items() -> None:
    print("PYTHON: starting handle")
    h = NetworkingHandle(Keypair.generate(), [], 0)

    recv_task = asyncio.create_task(_await_recv(h))

    try:
        # sleep for 4 ticks
        for _ in range(4):
            await asyncio.sleep(1)

            try:
                await h.gossipsub_publish("topic", b"somehting or other")
            except NoPeersSubscribedToTopicError as e:
                print("caught it", e)
    finally:
        recv_task.cancel()
        with suppress(asyncio.CancelledError):
            await recv_task


def test_pidfile(capsys: CaptureFixture[str]):
    with capsys.disabled():
        print("\nbefore python")
        scoped_lock_file()
        print("after python")


async def _await_recv(h: NetworkingHandle):
    while True:
        event = await h.recv()
        match event:
            case FromSwarm.Connection() as c:
                print(f"PYTHON: connection update: {c}")
            case FromSwarm.Message() as m:
                print(f"PYTHON: message: {m}")


def scoped_lock_file():
    lock_file = Pidfile("/tmp/lock.pid", 0o0600)
    lock_file.close()
