"""Stopping a master finishes it before the caller carries on."""

import anyio
import pytest

from exo.master.main import Master
from exo.routing.router import get_node_zid
from exo.shared.types.commands import ForwarderCommand, ForwarderDownloadCommand
from exo.shared.types.common import SessionId
from exo.shared.types.events import (
    Event,
    GlobalForwarderEvent,
    LocalForwarderEvent,
    StateSnapshot,
)
from exo.utils.channels import channel


async def test_shutdown_waits_for_the_master_to_stop_and_closes_its_event_sender() -> (
    None
):
    # Catches a demoted master still running (and holding its event sender open, so the
    # event router keeps it as a producer forever) after shutdown has returned
    node_id = get_node_zid()
    event_send, event_recv = channel[Event]()
    master = Master(
        node_id,
        SessionId(master_node_id=node_id, election_clock=0),
        event_sender=event_send,
        global_event_sender=channel[GlobalForwarderEvent]()[0],
        snapshot_sender=channel[StateSnapshot]()[0],
        local_event_receiver=channel[LocalForwarderEvent]()[1],
        command_receiver=channel[ForwarderCommand]()[1],
        download_command_sender=channel[ForwarderDownloadCommand]()[0],
    )
    stopped = anyio.Event()

    async def run_master() -> None:
        await master.run()
        stopped.set()

    async with anyio.create_task_group() as tg:
        tg.start_soon(run_master)
        await anyio.sleep(0.05)
        with anyio.fail_after(5):
            await master.shutdown()
        assert stopped.is_set()
        with pytest.raises(anyio.EndOfStream):
            event_recv.receive_nowait()
