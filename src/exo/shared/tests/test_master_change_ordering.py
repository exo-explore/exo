"""A node stops its old master before it follows a new one."""

from typing import cast

import anyio

from exo.main import Node
from exo.master.main import Master
from exo.routing.event_router import EventRouter
from exo.routing.router import Router, get_node_zid
from exo.shared.election import Election, ElectionResult
from exo.shared.types.commands import ForwarderCommand, ForwarderDownloadCommand
from exo.shared.types.common import NodeId, SessionId
from exo.shared.types.events import (
    GlobalForwarderEvent,
    LocalForwarderEvent,
    StateSnapshot,
)
from exo.utils.channels import channel


class RecordingEventRouter(EventRouter):
    master_running_at_switch: list[bool]
    master_stopped: anyio.Event

    def switch_session(self, session_id: SessionId) -> None:
        self.master_running_at_switch.append(not self.master_stopped.is_set())
        super().switch_session(session_id)


async def test_a_demoted_master_stops_before_the_node_switches_session() -> None:
    # Catches the old master still producing events (which the event router would tag
    # with the new master's session) once the node has switched to the new session
    node_id = get_node_zid()
    own_session = SessionId(master_node_id=node_id, election_clock=1)
    new_session = SessionId(master_node_id=NodeId("other-master"), election_clock=2)
    event_router = RecordingEventRouter(
        own_session,
        command_sender=channel[ForwarderCommand]()[0],
        external_inbound=channel[GlobalForwarderEvent]()[1],
        external_outbound=channel[LocalForwarderEvent]()[0],
        snapshot_inbound=channel[StateSnapshot]()[1],
    )
    event_router.master_running_at_switch = []
    event_router.master_stopped = anyio.Event()
    master = Master(
        node_id,
        own_session,
        event_sender=event_router.sender(),
        global_event_sender=channel[GlobalForwarderEvent]()[0],
        snapshot_sender=channel[StateSnapshot]()[0],
        local_event_receiver=channel[LocalForwarderEvent]()[1],
        command_receiver=channel[ForwarderCommand]()[1],
        download_command_sender=channel[ForwarderDownloadCommand]()[0],
    )
    result_send, result_recv = channel[ElectionResult]()
    node = Node(
        router=cast(Router, object()),
        event_router=event_router,
        download_coordinator=None,
        worker=None,
        election=cast(Election, object()),
        election_result_receiver=result_recv,
        master=master,
        api=None,
        node_id=node_id,
        offline=False,
        _api_port=0,
    )

    async def run_master() -> None:
        await master.run()
        event_router.master_stopped.set()

    async with anyio.create_task_group() as tg:
        tg.start_soon(event_router.run)
        tg.start_soon(run_master)
        tg.start_soon(node._elect_loop)  # pyright: ignore[reportPrivateUsage]
        await anyio.sleep(0.05)

        await result_send.send(
            ElectionResult(session_id=new_session, won_clock=2, is_new_master=True)
        )
        with anyio.fail_after(5):
            while not event_router.master_running_at_switch:
                await anyio.sleep(0.01)

        assert event_router.master_running_at_switch == [False]
        assert node.master is None
        assert event_router.session_id == new_session
        result_send.close()
        tg.cancel_scope.cancel()
