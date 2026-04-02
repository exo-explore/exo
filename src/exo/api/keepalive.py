from collections.abc import AsyncIterator
from typing import Final

import anyio

_DONE: Final = object()


async def with_sse_keepalive(
    generator: AsyncIterator[str],
    keepalive_message: str = ": keep-alive\n\n",
    interval: float = 10.0,
) -> AsyncIterator[str]:
    yield keepalive_message
    send, recv = anyio.create_memory_object_stream[str | object]()

    async def _consume() -> None:
        async for item in generator:
            await send.send(item)
        await send.send(_DONE)

    async with anyio.create_task_group() as tg:
        tg.start_soon(_consume)
        while True:
            item: str | object | None = None
            with anyio.move_on_after(interval):
                item = await recv.receive()
            if item is None:
                yield keepalive_message
            elif item is _DONE:
                break
            else:
                assert isinstance(item, str)
                yield item
