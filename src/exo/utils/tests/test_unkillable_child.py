from multiprocessing.process import BaseProcess
from typing import cast

import pytest
from anyio import fail_after

from exo.utils import async_process
from exo.utils.async_process import AsyncProcess


class _UnkillableChild:
    """Stands in for a process blocked in an uninterruptible kernel wait."""

    def __init__(self) -> None:
        self.terminate_calls = 0
        self.kill_calls = 0
        self.exitcode = None
        self.pid = 4242

    def is_alive(self) -> bool:
        return True

    def join(self, timeout: float | None = None) -> None:
        return None

    def terminate(self) -> None:
        self.terminate_calls += 1

    def kill(self) -> None:
        self.kill_calls += 1

    def close(self) -> None:
        raise ValueError("cannot close a running process")


async def test_terminate_gives_up_on_unkillable_child(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Shrink every grace period so the escalation runs in well under a second.
    for name in (
        "_JOIN_GRACE_SECONDS",
        "_TERMINATE_GRACE_SECONDS",
        "_TERMINATE_RETRY_GRACE_SECONDS",
        "_KILL_GRACE_SECONDS",
    ):
        monkeypatch.setattr(async_process, name, 0.01)

    child = _UnkillableChild()
    process = AsyncProcess()
    process._process = cast(BaseProcess, cast(object, child))  # pyright: ignore[reportPrivateUsage]
    process._pid = child.pid  # pyright: ignore[reportPrivateUsage]
    process._started.set()  # pyright: ignore[reportPrivateUsage]

    with fail_after(10):
        await process._terminate_if_still_alive()  # pyright: ignore[reportPrivateUsage]

    assert child.kill_calls == async_process._KILL_ATTEMPTS  # pyright: ignore[reportPrivateUsage]
    assert child.terminate_calls == async_process._TERMINATE_ATTEMPTS  # pyright: ignore[reportPrivateUsage]
