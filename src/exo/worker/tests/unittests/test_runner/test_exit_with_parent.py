"""A runner exits when the process that started it (its node) is killed.

A runner holds both ends of the pipes it waits on, so it never sees an end of file when its node
dies: SIGKILLed, exo used to leave its runners running for good, each holding its model's
memory."""

import contextlib
import multiprocessing as mp
import os
import signal
import time
from multiprocessing.queues import Queue

import psutil

from exo.worker.runner.bootstrap import exit_with_parent

# The runner exits within milliseconds; this only allows for a heavily loaded machine
EXIT_BOUND = 10.0


def _runner() -> None:
    exit_with_parent()
    time.sleep(600)


def _node(pids: "Queue[int]") -> None:
    runner = mp.get_context("spawn").Process(target=_runner)
    runner.start()
    pids.put(runner.pid or 0)
    time.sleep(600)


def _running(pid: int) -> bool:
    with contextlib.suppress(psutil.Error):
        process = psutil.Process(pid)
        return process.is_running() and process.status() != psutil.STATUS_ZOMBIE
    return False


def test_a_runner_exits_when_its_node_is_killed_and_not_before() -> None:
    context = mp.get_context("spawn")
    pids: Queue[int] = context.Queue()
    node = context.Process(target=_node, args=(pids,))
    node.start()
    runner = pids.get(timeout=60)
    try:
        time.sleep(1.0)
        assert _running(runner), "the runner exited while its node was alive"

        assert node.pid is not None
        os.kill(node.pid, signal.SIGKILL)
        node.join()
        deadline = time.monotonic() + EXIT_BOUND
        while _running(runner) and time.monotonic() < deadline:
            time.sleep(0.02)
        assert not _running(runner), "the runner outlived its node"
    finally:
        # Only if it is still running: an exited runner's pid may be someone else's by now
        if _running(runner):
            with contextlib.suppress(ProcessLookupError):
                os.kill(runner, signal.SIGKILL)
