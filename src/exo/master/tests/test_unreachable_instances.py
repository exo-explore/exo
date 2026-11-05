"""The master doesn't launch a model on a device it can't reach now, as its own membership sees
it. (Requests don't go through the master: every API checks the instance it sends one to.)"""

import anyio

from exo.master.tests.master_helpers import launch_of
from exo.master.tests.test_silent_nodes import (
    MASTER,
    PEER_A,
    PEER_B,
    instance_on,
    listed,
    master_with,
    membership,
)
from exo.shared.types.commands import ForwarderCommand
from exo.shared.types.common import SystemId
from exo.shared.types.events import (
    Event,
    InstanceCreated,
)
from exo.shared.types.memory import Memory
from exo.shared.types.profiling import MemoryUsage
from exo.utils.channels import channel


async def test_a_launch_onto_a_node_it_cant_reach_is_refused() -> None:
    """A launch with its nodes given (a placement) onto a node this master can't reach now:
    its model couldn't load there, and the instance would end once the node is removed.
    Refused, as a placement leaves such a node out (the API sends a launch again until it
    shows). One onto the nodes it reaches is launched."""
    view = membership("Majority", alive=(PEER_A, PEER_B))
    master, broadcast = master_with(listed(MASTER, PEER_A, PEER_B), membership=view)
    memory = MemoryUsage(
        ram_total=Memory.from_gb(64),
        ram_available=Memory.from_gb(64),
        swap_total=Memory(),
        swap_available=Memory(),
    )
    master.state = master.state.model_copy(
        update={"node_memory": {PEER_A: memory, PEER_B: memory}}
    )
    await view.lose(PEER_A)
    unreachable, reachable = instance_on(PEER_A), instance_on(PEER_B)
    command_sender, commands = channel[ForwarderCommand]()
    master.command_receiver = commands
    launched: list[Event] = []
    async with anyio.create_task_group() as tg:
        tg.start_soon(master._command_processor)  # pyright: ignore[reportPrivateUsage]
        for instance in (unreachable, reachable):
            await command_sender.send(
                ForwarderCommand(origin=SystemId(), command=launch_of(instance))
            )
        with anyio.fail_after(5):
            while not any(isinstance(event, InstanceCreated) for event in launched):
                launched.append((await broadcast.receive()).event)
        tg.cancel_scope.cancel()

    (event,) = [event for event in launched if isinstance(event, InstanceCreated)]
    assert event.instance.instance_id == reachable.instance_id
    assert unreachable.instance_id not in master.state.instances
    assert len(master.state.deployments) == 1
