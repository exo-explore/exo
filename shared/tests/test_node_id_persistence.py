from __future__ import annotations

import contextlib
import logging
import multiprocessing
import os
from multiprocessing import Event, Queue, Semaphore
from multiprocessing.process import BaseProcess
from multiprocessing.queues import Queue as QueueT
from multiprocessing.synchronize import Event as EventT
from multiprocessing.synchronize import Semaphore as SemaphoreT
from typing import Optional

from pytest import LogCaptureFixture

from shared.constants import EXO_NODE_ID_KEYPAIR
from shared.utils import get_node_id_keypair

NUM_CONCURRENT_PROCS = 10


def _get_keypair_concurrent_subprocess_task(pid: int, sem: SemaphoreT, ev: EventT, queue: QueueT[bytes]) -> None:
    try:
        # synchronise with parent process
        logging.info(msg=f"SUBPROCESS {pid}: Started")
        sem.release()

        # wait to be told to begin simultaneous read
        ev.wait()
        logging.info(msg=f"SUBPROCESS {pid}: Reading start")
        queue.put(get_node_id_keypair().to_protobuf_encoding())
        logging.info(msg=f"SUBPROCESS {pid}: Reading end")
    except Exception as e:
        logging.error(msg=f"SUBPROCESS {pid}: Error encountered: {e}")


def _get_keypair_concurrent(num_procs: int) -> bytes:
    assert num_procs > 0

    sem = Semaphore(0)
    ev = Event()
    queue: QueueT[bytes] = Queue(maxsize=num_procs)

    # make parent process wait for all subprocesses to start
    logging.info(msg=f"PARENT: Starting {num_procs} subprocesses")
    ps: list[BaseProcess] = []
    for i in range(num_procs):
        p = multiprocessing.get_context("fork").Process(target=_get_keypair_concurrent_subprocess_task,
                                                        args=(i + 1, sem, ev, queue))
        ps.append(p)
        p.start()
    for _ in range(num_procs):
        sem.acquire()

    # start all the sub processes simultaneously
    logging.info(msg="PARENT: Beginning read")
    ev.set()

    # wait until all subprocesses are done & read results
    for p in ps:
        p.join()

    # check that the input/output order match, and that
    # all subprocesses end up reading the same file
    logging.info(msg="PARENT: Checking consistency")
    keypair: Optional[bytes] = None
    qsize = 0  # cannot use Queue.qsize due to MacOS incompatibility :(
    while not queue.empty():
        qsize += 1
        temp_keypair = queue.get()
        if keypair is None:
            keypair = temp_keypair
        else:
            assert keypair == temp_keypair
    assert num_procs == qsize
    return keypair  # pyright: ignore[reportReturnType]


def _delete_if_exists(p: str | bytes | os.PathLike[str] | os.PathLike[bytes]):
    with contextlib.suppress(OSError):
        os.remove(p)


def test_node_id_fetching(caplog: LogCaptureFixture):
    reps = 10

    # delete current file and write a new one
    _delete_if_exists(EXO_NODE_ID_KEYPAIR)
    kp = _get_keypair_concurrent(NUM_CONCURRENT_PROCS)

    with caplog.at_level(logging.CRITICAL):  # supress logs
        # make sure that continuous fetches return the same value
        for _ in range(reps):
            assert kp == _get_keypair_concurrent(NUM_CONCURRENT_PROCS)

        # make sure that after deleting, we are not fetching the same value
        _delete_if_exists(EXO_NODE_ID_KEYPAIR)
        for _ in range(reps):
            assert kp != _get_keypair_concurrent(NUM_CONCURRENT_PROCS)
