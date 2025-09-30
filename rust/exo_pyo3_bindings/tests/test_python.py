import asyncio

import pytest
from exo_pyo3_bindings import Keypair, NetworkingHandle, NoPeersSubscribedToTopicError


@pytest.mark.asyncio
async def test_sleep_on_multiple_items() -> None:
    print("PYTHON: starting handle")
    h = NetworkingHandle(Keypair.generate_ed25519())

    ct = asyncio.create_task(_await_cons(h))
    mt = asyncio.create_task(_await_msg(h))

    # sleep for 4 ticks
    for i in range(4):
        await asyncio.sleep(1)

        try:
            await h.gossipsub_publish("topic", b"somehting or other")
        except NoPeersSubscribedToTopicError as e:
            print("caught it", e)


async def _await_cons(h: NetworkingHandle):
    while True:
        c = await h.connection_update_recv()
        print(f"PYTHON: connection update: {c}")


async def _await_msg(h: NetworkingHandle):
    while True:
        m = await h.gossipsub_recv()
        print(f"PYTHON: message: {m}")
