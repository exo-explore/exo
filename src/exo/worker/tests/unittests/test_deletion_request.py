"""Once the worker gives up on an instance, it asks the master to delete it: once, not on every
pass of its planning loop until the deletion arrives."""

import anyio

from exo.shared.constants import EXO_MAX_INSTANCE_RETRIES
from exo.shared.models.model_cards import ModelId
from exo.shared.types.commands import (
    DeleteInstance,
    ForwarderCommand,
    ForwarderDownloadCommand,
)
from exo.shared.types.common import NodeId
from exo.shared.types.events import Event, IndexedEvent, InstanceCreated
from exo.shared.types.worker.instances import InstanceId
from exo.shared.types.worker.runners import RunnerId
from exo.utils.channels import Receiver, channel
from exo.utils.keyed_backoff import KeyedBackoff
from exo.worker.main import Worker
from exo.worker.tests.unittests.conftest import (
    get_mlx_ring_instance,
    get_pipeline_shard_metadata,
)

MODEL = ModelId("test/model")


async def test_asks_once_to_delete_an_instance_it_gave_up_on():
    _, event_receiver = channel[IndexedEvent]()
    event_sender, _ = channel[Event]()
    command_sender, commands = channel[ForwarderCommand]()
    download_sender, _ = channel[ForwarderDownloadCommand]()
    worker = Worker(
        NodeId(),
        event_receiver=event_receiver,
        event_sender=event_sender,
        command_sender=command_sender,
        download_command_sender=download_sender,
        api_port=52415,
    )
    instance_id, runner_id = InstanceId(), RunnerId()
    sender, receiver = channel[IndexedEvent]()
    worker.event_receiver = receiver
    await sender.send(
        IndexedEvent(
            idx=0,
            event=InstanceCreated(
                instance=get_mlx_ring_instance(
                    instance_id,
                    MODEL,
                    {worker.node_id: runner_id},
                    {runner_id: get_pipeline_shard_metadata(MODEL, 0)},
                )
            ),
        )
    )
    sender.close()
    await worker._event_applier()  # pyright: ignore[reportPrivateUsage]

    # Its runner failed to start every time, and the backoff before the next try has passed
    backoff = KeyedBackoff[InstanceId](base=0.0, cap=0.0)
    for _ in range(EXO_MAX_INSTANCE_RETRIES):
        backoff.record_attempt(instance_id)
    worker._instance_backoff = backoff  # pyright: ignore[reportPrivateUsage]

    # The master is slow to delete it: the planning loop keeps coming back to it
    with anyio.move_on_after(1.5):
        await worker.plan_step()

    assert _deletions(commands) == [instance_id]


def _deletions(commands: Receiver[ForwarderCommand]) -> list[InstanceId]:
    return [
        command.command.instance_id
        for command in commands.collect()
        if isinstance(command.command, DeleteInstance)
    ]
