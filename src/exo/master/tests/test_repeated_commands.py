"""The master acts on a command only once, however often it arrives."""

from collections import deque

import pytest

import exo.master.main as master_main
from exo.master.main import Master
from exo.shared.types.common import CommandId


def fresh_master() -> Master:
    master = object.__new__(Master)
    master._processed_commands = set()  # pyright: ignore[reportPrivateUsage]
    master._processed_order = deque()  # pyright: ignore[reportPrivateUsage]
    return master


def test_a_repeated_command_is_only_processed_once() -> None:
    master = fresh_master()
    command_id = CommandId()

    assert master._first_time(command_id)  # pyright: ignore[reportPrivateUsage]
    assert not master._first_time(command_id)  # pyright: ignore[reportPrivateUsage]
    assert master._first_time(CommandId())  # pyright: ignore[reportPrivateUsage]


def test_only_recent_commands_are_remembered(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(master_main, "PROCESSED_COMMANDS_KEPT", 2)
    master = fresh_master()
    oldest, *rest = [CommandId() for _ in range(3)]
    for command_id in (oldest, *rest):
        master._first_time(command_id)  # pyright: ignore[reportPrivateUsage]

    assert len(master._processed_commands) == 2  # pyright: ignore[reportPrivateUsage]
    assert master._first_time(oldest)  # pyright: ignore[reportPrivateUsage]
