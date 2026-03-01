from typing import Any

import anyio
import pytest

from exo.download.coordinator import DownloadCoordinator
from exo.download.shard_downloader import NoopShardDownloader
from exo.shared.models.model_cards import ModelCard, ModelTask
from exo.shared.types.common import ModelId, NodeId, SessionId
from exo.shared.types.events import (
    GlobalForwarderEvent,
    LocalForwarderEvent,
    NodeDownloadProgress,
)
from exo.shared.types.memory import Memory
from exo.shared.types.worker.downloads import (
    DownloadPending,
)
from exo.shared.types.worker.shards import PipelineShardMetadata
from exo.utils.channels import channel

# Use the built‑in NoopShardDownloader directly – it already implements the required abstract interface.
# No additional subclass is needed for this test.


@pytest.mark.anyio
async def test_ack_behaviour():
    # Create channels (type Any for simplicity)
    _, command_receiver = channel[Any]()
    local_sender, _ = channel[Any]()
    global_sender, global_receiver = channel[Any]()

    # Minimal identifiers
    node_id = NodeId()
    session_id = SessionId(master_node_id=node_id, election_clock=0)

    # Create a dummy model card and shard metadata
    model_id = ModelId("test/model")
    model_card = ModelCard(
        model_id=model_id,
        storage_size=Memory.from_bytes(0),
        n_layers=1,
        hidden_size=1,
        supports_tensor=True,
        tasks=[ModelTask.TextGeneration],
    )
    shard = PipelineShardMetadata(
        model_card=model_card,
        device_rank=0,
        world_size=1,
        start_layer=0,
        end_layer=1,
        n_layers=1,
    )

    # Instantiate the coordinator with the dummy downloader
    coord = DownloadCoordinator(
        node_id=node_id,
        session_id=session_id,
        shard_downloader=NoopShardDownloader(),
        download_command_receiver=command_receiver,
        local_event_sender=local_sender,
        _global_event_receiver=global_receiver,
    )

    async with anyio.create_task_group() as tg:
        # Start the forwarding and ack‑clearing loops
        tg.start_soon(coord._forward_events)  # pyright: ignore[reportPrivateUsage]
        tg.start_soon(coord._clear_ofd)  # pyright: ignore[reportPrivateUsage]

        # Send a pending download progress event via the internal event sender
        pending = DownloadPending(
            node_id=node_id,
            shard_metadata=shard,
            model_directory="/tmp/model",
        )
        await coord.event_sender.send(NodeDownloadProgress(download_progress=pending))
        # Allow the forwarder to process the event
        await anyio.sleep(0.1)

        # There should be exactly one entry awaiting ACK
        assert len(coord._out_for_delivery) == 1  # pyright: ignore[reportPrivateUsage]
        # Retrieve the stored LocalForwarderEvent
        stored_fe: LocalForwarderEvent = next(iter(coord._out_for_delivery.values()))  # pyright: ignore[reportPrivateUsage]
        # Simulate receiving a global ack for this event
        ack = GlobalForwarderEvent(
            origin_idx=0,
            origin=node_id,
            session=session_id,
            event=stored_fe.event,
        )
        await global_sender.send(ack)
        # Give the clear‑ofd task a moment to process the ack
        await anyio.sleep(0.1)
        # The out‑for‑delivery map should now be empty
        assert len(coord._out_for_delivery) == 0  # pyright: ignore[reportPrivateUsage]
        # Cancel background tasks
        tg.cancel_scope.cancel()
