import ctypes
import multiprocessing as mp
import threading
import time

import pytest
from anyio import (
    BrokenResourceError,
    ClosedResourceError,
    EndOfStream,
    WouldBlock,
    fail_after,
)
from loguru import logger

from exo.utils.channels import ErrorOverride, MpReceiver, MpSender, channel, mp_channel


class CustomClosedResourceError(ClosedResourceError):
    pass


class CustomBrokenResourceError(BrokenResourceError):
    pass


class CustomEndOfStream(EndOfStream):
    pass


class CustomWouldBlock(WouldBlock):
    pass


ERROR_OVERRIDE = ErrorOverride(
    closed_resource_error=CustomClosedResourceError,
    broken_resource_error=CustomBrokenResourceError,
    end_of_stream=CustomEndOfStream,
    would_block=CustomWouldBlock,
)


def foo(recv: MpReceiver[str]):
    expected = ["hi", "hi 2", "bye"]
    with recv as r:
        for item in r:
            assert item == expected.pop(0)


def bar(send: MpSender[str]):
    logger.warning("hi")
    send.send("hi")
    time.sleep(0.1)
    logger.warning("hi 2")
    send.send("hi 2")
    time.sleep(0.1)
    logger.warning("bye")
    send.send("bye")
    time.sleep(0.1)
    send.close()


@pytest.mark.anyio
@pytest.mark.parametrize("send_immediately", [False, True])
async def test_channel_ipc(send_immediately: bool):
    with fail_after(0.5):
        s, r = mp_channel[str](send_immediately=send_immediately)
        p1 = mp.Process(target=foo, args=(r,))
        p2 = mp.Process(target=bar, args=(s,))
        p1.start()
        p2.start()
        p1.join()
        p2.join()


def send_then_hold_the_gil(send: MpSender[str]):
    # Look the function up first: loading a library releases the GIL
    hold_the_gil = ctypes.PyDLL(None).sleep
    send.send("status")
    send.send("acknowledgement")
    # As MLX's distributed init does while it waits for the other ranks
    hold_the_gil(60)


def receive(recv: MpReceiver[str], count: int, seconds: float) -> list[str]:
    received: list[str] = []
    deadline = time.monotonic() + seconds
    while len(received) < count and time.monotonic() < deadline:
        try:
            received.append(recv.receive_nowait())
        except WouldBlock:
            time.sleep(0.01)
    return received


def test_what_is_sent_immediately_leaves_before_a_call_that_holds_the_gil():
    send, recv = mp_channel[str](send_immediately=True)
    sender = mp.Process(target=send_then_hold_the_gil, args=(send,), daemon=True)
    sender.start()
    try:
        assert receive(recv, 2, 5) == ["status", "acknowledgement"]
    finally:
        sender.kill()
        sender.join()
        recv.close()


def test_closing_wakes_a_reader_of_a_channel_that_sends_immediately():
    _, recv = mp_channel[str](send_immediately=True)
    ended = threading.Event()

    def read() -> None:
        with pytest.raises((EndOfStream, ClosedResourceError)):
            recv.receive()
        ended.set()

    reader = threading.Thread(target=read, daemon=True)
    reader.start()
    time.sleep(0.1)
    recv.close()
    assert ended.wait(2)


def test_channel_error_override_replaces_sync_errors_with_subclasses():
    send, recv = channel[int](0, error_override_config=ERROR_OVERRIDE)

    with pytest.raises(CustomWouldBlock) as would_block_info:
        send.send_nowait(1)
    assert type(would_block_info.value.__cause__) is WouldBlock

    recv.close()
    with pytest.raises(CustomBrokenResourceError) as broken_resource_info:
        send.send_nowait(1)
    assert type(broken_resource_info.value.__cause__) is BrokenResourceError

    send.close()
    with pytest.raises(CustomClosedResourceError) as closed_resource_info:
        send.send_nowait(1)
    assert type(closed_resource_info.value.__cause__) is ClosedResourceError


@pytest.mark.anyio
async def test_channel_error_override_replaces_async_errors_with_subclasses():
    send, recv = channel[int](0, error_override_config=ERROR_OVERRIDE)
    recv.close()

    with pytest.raises(CustomBrokenResourceError) as broken_resource_info:
        await send.send(1)
    assert type(broken_resource_info.value.__cause__) is BrokenResourceError

    send, recv = channel[int](error_override_config=ERROR_OVERRIDE)
    send.close()
    with pytest.raises(CustomEndOfStream) as end_of_stream_info:
        await recv.receive()
    assert type(end_of_stream_info.value.__cause__) is EndOfStream


@pytest.mark.anyio
async def test_channel_error_override_is_preserved_by_clones():
    send, recv = channel[int](0, error_override_config=ERROR_OVERRIDE)
    send_clone = send.clone()
    recv.close()

    with pytest.raises(CustomBrokenResourceError):
        await send_clone.send(1)

    send, recv = channel[int](0, error_override_config=ERROR_OVERRIDE)
    cloned_send = recv.clone_sender()
    recv.close()

    with pytest.raises(CustomBrokenResourceError):
        await cloned_send.send(1)


def test_closing_a_full_channel_that_sends_immediately_does_not_block():
    send, recv = mp_channel[str](send_immediately=True)
    # Fill the pipe: nobody is reading
    with pytest.raises(WouldBlock):
        for _ in range(1_000_000):
            send.send_nowait("x" * 200)

    closer = threading.Thread(target=recv.close, daemon=True)
    closer.start()
    closer.join(2)
    assert not closer.is_alive()
    send.close()
