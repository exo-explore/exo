# pyright: reportPrivateUsage=false
"""The periodic rescan only sends statuses the cluster state does not hold yet."""

import anyio
from anyio import WouldBlock

from exo.download.coordinator import DownloadCoordinator
from exo.download.impl_shard_downloader import SingletonShardDownloader
from exo.download.tests.test_download_status_not_lost import (
    MODEL_ID,
    NODE_ID,
    FakeShardDownloader,
)
from exo.shared.types.commands import ForwarderDownloadCommand
from exo.shared.types.events import (
    Event,
    IndexedEvent,
    NodeDownloadProgress,
    NodeTimedOut,
)
from exo.shared.types.worker.downloads import DownloadPending
from exo.utils.channels import Receiver, channel

RESCAN_SECONDS = 0.05


def _drain(receiver: Receiver[Event]) -> list[Event]:
    events: list[Event] = []
    while True:
        try:
            events.append(receiver.receive_nowait())
        except WouldBlock:
            return events


def _pending(events: list[Event]) -> list[DownloadPending]:
    return [
        event.download_progress
        for event in events
        if isinstance(event, NodeDownloadProgress)
        and isinstance(event.download_progress, DownloadPending)
        and event.download_progress.shard_metadata.model_card.model_id == MODEL_ID
    ]


async def test_rescan_skips_statuses_the_state_holds() -> None:
    event_send, event_recv = channel[Event]()
    indexed_send, indexed_recv = channel[IndexedEvent]()
    _, command_recv = channel[ForwarderDownloadCommand]()
    coordinator = DownloadCoordinator(
        node_id=NODE_ID,
        shard_downloader=SingletonShardDownloader(FakeShardDownloader("not_started")),
        download_command_receiver=command_recv,
        event_sender=event_send,
        event_receiver=indexed_recv,
        rescan_interval_seconds=RESCAN_SECONDS,
    )
    next_index = 0

    async def index(event: Event) -> None:
        """Play the master: index an event and broadcast it back."""
        nonlocal next_index
        await indexed_send.send(IndexedEvent(idx=next_index, event=event))
        next_index += 1

    async with anyio.create_task_group() as tg:
        tg.start_soon(coordinator.run)

        # Until the state holds the status, every rescan sends it.
        await anyio.sleep(RESCAN_SECONDS * 3)
        first = _drain(event_recv)
        assert len(_pending(first)) >= 2

        # Once indexed, rescans stop sending it.
        await index(first[0])
        await anyio.sleep(RESCAN_SECONDS)
        _drain(event_recv)
        await anyio.sleep(RESCAN_SECONDS * 4)
        assert _pending(_drain(event_recv)) == []

        # A timeout drops this node's statuses from the state: send again.
        await index(NodeTimedOut(node_id=NODE_ID))
        await anyio.sleep(RESCAN_SECONDS * 3)
        assert len(_pending(_drain(event_recv))) >= 1

        await coordinator.shutdown()
        tg.cancel_scope.cancel()
