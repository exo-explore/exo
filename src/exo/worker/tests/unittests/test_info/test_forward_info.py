# pyright: reportPrivateUsage=false
"""Unchanged self-contained info is not re-sent while the heartbeat is fresh."""

import anyio
import pytest
from anyio import WouldBlock

from exo.shared.types.commands import ForwarderCommand, ForwarderDownloadCommand
from exo.shared.types.common import NodeId
from exo.shared.types.events import (
    Event,
    IndexedEvent,
    NodeGatheredInfo,
    NodeTimedOut,
)
from exo.utils.channels import channel
from exo.utils.info_gatherer.info_gatherer import (
    GatheredInfo,
    MacThunderboltConnections,
    MiscData,
)
from exo.worker import main as worker_main
from exo.worker.main import Worker

NODE_ID = NodeId("aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa")


async def _forwarded(
    monkeypatch: pytest.MonkeyPatch,
    steps: list[tuple[float, GatheredInfo | NodeTimedOut]],
) -> list[GatheredInfo]:
    """Feed info, or an indexed timeout event, at the given clock times and
    return the info the worker sent."""
    clock = [0.0]
    monkeypatch.setattr(worker_main, "current_time", lambda: clock[0])
    event_send, event_recv = channel[Event]()
    indexed_send, indexed_recv = channel[IndexedEvent]()
    worker = Worker(
        NODE_ID,
        event_receiver=indexed_recv,
        event_sender=event_send,
        command_sender=channel[ForwarderCommand]()[0],
        download_command_sender=channel[ForwarderDownloadCommand]()[0],
        api_port=0,
    )
    info_send, info_recv = channel[GatheredInfo]()
    next_index = 0
    async with anyio.create_task_group() as tg:
        tg.start_soon(worker._forward_info, info_recv)
        tg.start_soon(worker._event_applier)
        for when, item in steps:
            clock[0] = when
            if isinstance(item, NodeTimedOut):
                await indexed_send.send(IndexedEvent(idx=next_index, event=item))
                next_index += 1
            else:
                await info_send.send(item)
            await anyio.wait_all_tasks_blocked()
        tg.cancel_scope.cancel()

    sent: list[GatheredInfo] = []
    while True:
        try:
            event = event_recv.receive_nowait()
        except WouldBlock:
            return sent
        assert isinstance(event, NodeGatheredInfo)
        sent.append(event.info)


async def test_unchanged_info_is_skipped_while_heartbeat_is_fresh(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    name = MiscData(friendly_name="mini")
    renamed = MiscData(friendly_name="mini-2")
    connections = MacThunderboltConnections(conns=[])
    sent = await _forwarded(
        monkeypatch,
        [
            (0.0, name),
            (1.0, name),  # unchanged, heartbeat fresh: skipped
            (2.0, connections),
            (3.0, connections),  # resolves against peers: always sent
            (4.0, renamed),  # changed: sent
            (20.0, renamed),  # unchanged, but nothing sent for 16 s: sent
        ],
    )
    assert sent == [name, connections, connections, renamed, renamed]


async def test_own_timeout_resends_unchanged_info(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    name = MiscData(friendly_name="mini")
    other = NodeId("bbbbbbbb-bbbb-4bbb-8bbb-bbbbbbbbbbbb")
    sent = await _forwarded(
        monkeypatch,
        [
            (0.0, name),
            (1.0, NodeTimedOut(node_id=other)),
            (2.0, name),  # another node's timeout: still skipped
            (3.0, NodeTimedOut(node_id=NODE_ID)),
            (4.0, name),  # the state dropped it: sent again
        ],
    )
    assert sent == [name, name]
