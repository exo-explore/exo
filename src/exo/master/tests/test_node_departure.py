"""A node that announces it is leaving is removed, with its instances, at once."""

from datetime import datetime, timezone

import anyio
import pytest

import exo.master.main as master_main
from exo.master.main import Master
from exo.master.tests.test_requests_to_deleted_instances import deployment_of
from exo.master.tests.test_silent_nodes import instance_on
from exo.routing.tests.fake_membership import FakeMembership
from exo.shared.topology import Topology
from exo.shared.types.commands import ForwarderCommand, ForwarderDownloadCommand
from exo.shared.types.common import NodeId, SessionId, SystemId
from exo.shared.types.events import (
    Event,
    GlobalForwarderEvent,
    InstanceDeleted,
    LocalForwarderEvent,
    NodeLeft,
    StateSnapshot,
)
from exo.shared.types.state import State
from exo.utils.channels import channel

MASTER = NodeId("master-node")
LEAVING = NodeId("leaving-node")


async def test_a_departed_node_and_its_instances_are_removed_without_waiting_for_a_check(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A quitting master waits only LEAVE_FLUSH after its departure is applied: if its plan
    loop waited for its next round, the new master would see the instances lost, with no
    record of the departure, and keep them for the split hold (in a pair) or place their
    models elsewhere at once, instead of after LEAVE_GRACE."""
    # Checks so rare that only an immediate reaction to the departure can pass
    monkeypatch.setattr(master_main, "PLAN_INTERVAL", 60.0)
    session = SessionId(master_node_id=MASTER, election_clock=0)
    on_leaving, elsewhere = instance_on(LEAVING, MASTER), instance_on(MASTER)
    topology = Topology()
    topology.add_node(MASTER)
    topology.add_node(LEAVING)
    now = datetime.now(tz=timezone.utc)
    state = State(
        topology=topology,
        last_seen={MASTER: now, LEAVING: now},
        instances={i.instance_id: i for i in (on_leaving, elsewhere)},
        deployments={
            i.deployment_id: deployment_of(i) for i in (on_leaving, elsewhere)
        },
    )

    global_send, global_events = channel[GlobalForwarderEvent]()
    _command_send, commands = channel[ForwarderCommand]()
    local_send, local_events = channel[LocalForwarderEvent]()
    download_send, _downloads = channel[ForwarderDownloadCommand]()
    snapshot_send, _snapshots = channel[StateSnapshot]()
    master = Master(
        MASTER,
        session,
        global_event_sender=global_send,
        snapshot_sender=snapshot_send,
        local_event_receiver=local_events,
        command_receiver=commands,
        download_command_sender=download_send,
        membership=FakeMembership(MASTER, LEAVING),
        initial_state=state,
    )

    indexed: list[Event] = []
    async with anyio.create_task_group() as tg:
        tg.start_soon(master.run)
        # The plan loop has run its first check
        await anyio.sleep(0.1)

        # The leaving node's announcement
        await local_send.send(
            LocalForwarderEvent(
                origin=SystemId(),
                origin_idx=0,
                session=session,
                event=NodeLeft(node_id=LEAVING),
            )
        )
        with anyio.fail_after(2):
            while not any(isinstance(e, InstanceDeleted) for e in indexed):
                indexed.append((await global_events.receive()).event)
        tg.cancel_scope.cancel()

    assert [type(e) for e in indexed] == [NodeLeft, InstanceDeleted]
    deleted = next(e for e in indexed if isinstance(e, InstanceDeleted))
    assert (deleted.instance_id, deleted.reason) == (on_leaving.instance_id, "NodeLeft")
    assert LEAVING not in master.state.last_seen
    assert not master.state.topology.contains_node(LEAVING)
    assert set(master.state.instances) == {elsewhere.instance_id}
