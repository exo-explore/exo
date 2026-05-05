import asyncio

import pytest
from exo_pyo3_bindings import (
    Keypair,
    NetworkingHandle,
    NoPeersSubscribedToTopicError,
    PyFromSwarm,
    UnixBlobChannel,
)


@pytest.mark.asyncio
async def test_sleep_on_multiple_items() -> None:
    print("PYTHON: starting handle")
    h = NetworkingHandle(Keypair.generate(), [], 0)

    rt = asyncio.create_task(_await_recv(h))

    # sleep for 4 ticks
    for i in range(4):
        await asyncio.sleep(1)

        try:
            await h.gossipsub_publish("topic", b"somehting or other")
        except NoPeersSubscribedToTopicError as e:
            print("caught it", e)


async def _await_recv(h: NetworkingHandle):
    while True:
        event = await h.recv()
        match event:
            case PyFromSwarm.Connection() as c:
                print(f"PYTHON: connection update: {c}")
            case PyFromSwarm.Message() as m:
                print(f"PYTHON: message: {m}")


def test_unix_blob_channel_roundtrip() -> None:
    left, right = UnixBlobChannel.pair()

    left.send(b"hello")

    assert right.recv() == b"hello"


def test_unix_blob_channel_raw_fd_handoff() -> None:
    left, right = UnixBlobChannel.pair()
    raw_fd = right.into_raw_fd()
    adopted = UnixBlobChannel.from_raw_fd(raw_fd)

    left.send(b"from raw fd")

    assert adopted.recv() == b"from raw fd"
    assert right.closed()
