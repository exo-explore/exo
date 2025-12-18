import multiprocessing as mp
import time

import pytest
from anyio import fail_after
from loguru import logger

from exo.utils.channels import MpReceiver, MpSender, mp_channel


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


# not async, just want the fail_after
@pytest.mark.anyio
async def test_channel_setup():
    with fail_after(0.5):
        s, r = mp_channel[str]()
        p1 = mp.Process(target=foo, args=(r,))
        p2 = mp.Process(target=bar, args=(s,))
        p1.start()
        p2.start()
        p1.join()
        p2.join()
