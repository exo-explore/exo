"""Custom model cards saved on a node outlive the cluster state they came from."""

from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from pathlib import Path

import anyio
import pytest

import exo.worker.main as worker_main
from exo.shared.models import model_cards
from exo.shared.models.model_cards import ModelCard, ModelTask, card_cache
from exo.shared.types.backends import Backend
from exo.shared.types.commands import (
    AddCustomModelCard,
    ForwarderCommand,
    ForwarderDownloadCommand,
)
from exo.shared.types.common import ModelId
from exo.shared.types.events import Event, IndexedEvent
from exo.shared.types.memory import Memory
from exo.shared.types.state import State
from exo.utils.channels import Receiver, channel
from exo.worker.main import Worker
from exo.worker.tests.constants import NODE_A

CARD = ModelCard(
    model_id=ModelId("someone/custom-model"),
    n_layers=1,
    storage_size=Memory.from_bytes(1),
    hidden_size=1,
    supports_tensor=True,
    tasks=[ModelTask.TextGeneration],
    backends=[Backend.MlxMetal],
    is_custom=True,
)


@pytest.fixture(autouse=True)
def custom_cards_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setattr(model_cards, "_custom_cards_dir", anyio.Path(tmp_path))
    monkeypatch.setattr(card_cache, "cc", {})
    monkeypatch.setattr(worker_main, "CUSTOM_CARD_SYNC_INTERVAL", 0.01)
    monkeypatch.setattr(worker_main, "CUSTOM_CARD_ANNOUNCE_INTERVAL", 0.2)
    return tmp_path


def saved_file(directory: Path) -> Path:
    return directory / (CARD.model_id.normalize() + ".toml")


async def wait_until(condition: Callable[[], bool]) -> None:
    with anyio.fail_after(5):
        while not condition():
            await anyio.sleep(0.01)


@dataclass
class Node:
    worker: Worker
    commands: Receiver[ForwarderCommand]
    announcements: list[ModelCard] = field(default_factory=list)

    def announced(self) -> list[ModelCard]:
        """Every card announced to the master so far."""
        self.announcements.extend(
            c.command.model_card
            for c in self.commands.collect()
            if isinstance(c.command, AddCustomModelCard)
        )
        return self.announcements


@asynccontextmanager
async def running_node() -> AsyncIterator[Node]:
    event_send, _event_recv = channel[Event]()
    _index_send, index_recv = channel[IndexedEvent]()
    command_send, command_recv = channel[ForwarderCommand]()
    download_send, _download_recv = channel[ForwarderDownloadCommand]()
    worker = Worker(
        NODE_A,
        event_receiver=index_recv,
        event_sender=event_send,
        command_sender=command_send,
        download_command_sender=download_send,
        api_port=52415,
    )
    async with anyio.create_task_group() as tg:
        tg.start_soon(worker._reconcile_custom_cards)  # pyright: ignore[reportPrivateUsage]
        yield Node(worker, command_recv)
        tg.cancel_scope.cancel()


async def test_saved_card_survives_a_fresh_state_and_is_announced(
    custom_cards_dir: Path,
) -> None:
    await CARD.save_to_custom_dir()

    # A new master starts from an empty state, e.g. after the whole cluster restarted
    async with running_node() as node:
        await wait_until(lambda: len(node.announced()) > 0)
        await anyio.sleep(0.1)

    assert saved_file(custom_cards_dir).exists()
    assert node.announced() == [CARD]


async def test_unacknowledged_announcement_is_repeated() -> None:
    await CARD.save_to_custom_dir()

    async with running_node() as node:
        await wait_until(lambda: len(node.announced()) >= 2)

    assert node.announced() == [CARD, CARD]


async def test_announcement_stops_once_the_cluster_knows_the_card() -> None:
    await CARD.save_to_custom_dir()

    async with running_node() as node:
        await wait_until(lambda: len(node.announced()) > 0)
        node.worker.state = State(custom_model_cards={CARD.model_id: CARD})
        await anyio.sleep(0.5)

    assert node.announced() == [CARD]


async def test_card_added_on_this_node_is_saved(custom_cards_dir: Path) -> None:
    # The API puts a card it adds straight into this node's cache
    card_cache.cc[CARD.model_id] = CARD

    async with running_node() as node:
        node.worker.state = State(custom_model_cards={CARD.model_id: CARD})
        await wait_until(saved_file(custom_cards_dir).exists)


async def test_card_deleted_from_the_cluster_is_deleted(
    custom_cards_dir: Path,
) -> None:
    async with running_node() as node:
        node.worker.state = State(custom_model_cards={CARD.model_id: CARD})
        await wait_until(saved_file(custom_cards_dir).exists)

        node.worker.state = State()
        await wait_until(lambda: not saved_file(custom_cards_dir).exists())

    assert card_cache.get(CARD.model_id) is None
