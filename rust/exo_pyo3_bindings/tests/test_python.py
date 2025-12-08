import asyncio

import pytest
from exo_pyo3_bindings import (
    Keypair,
    RustNetworkingHandle,
    RustReceiver,
    RustConnectionReceiver,
)


@pytest.mark.asyncio
async def test_sleep_on_multiple_items() -> None:
    print("PYTHON: starting handle")
    s_h = await RustNetworkingHandle.create(Keypair.generate_ed25519(), "test")
    r_h = await RustNetworkingHandle.create(Keypair.generate_ed25519(), "test")

    await asyncio.sleep(1)

    cm = await r_h.get_connection_receiver()

    _, recv = await r_h.subscribe("topic")
    send, _ = await s_h.subscribe("topic")

    ct = asyncio.create_task(_await_cons(cm))
    mt = asyncio.create_task(_await_msg(recv))

    # sleep for 4 ticks
    for i in range(4):
        await asyncio.sleep(1)

        await send.send(b"somehting or other")

    await ct
    await mt


async def _await_cons(h: RustConnectionReceiver):
    while True:
        c = await h.receive()
        print(f"PYTHON: connection update: {c}")


async def _await_msg(r: RustReceiver):
    while True:
        m = await r.receive()
        print(f"PYTHON: message: {m}")
