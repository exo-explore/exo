from collections.abc import Awaitable, Callable
from datetime import datetime, timedelta, timezone
from itertools import count
from pathlib import Path
from typing import AsyncIterator

import anyio
import pytest

from exo.download.coordinator import DownloadCoordinator
from exo.download.shard_downloader import RepoDownloadProgress, ShardDownloader
from exo.master.main import Master
from exo.master.tests.conftest import create_node_memory
from exo.shared.models.model_cards import ModelCard, ModelTask
from exo.shared.types.commands import (
    ForwarderCommand,
    ForwarderDownloadCommand,
    StartDownload,
)
from exo.shared.types.common import ModelId, NodeId, SessionId
from exo.shared.types.events import (
    ForwarderEvent,
    NodeDownloadProgress,
    NodeGatheredInfo,
)
from exo.shared.types.memory import Memory
from exo.shared.types.worker.shards import PipelineShardMetadata, ShardMetadata
from exo.utils.channels import Receiver, Sender, channel
from exo.worker.main import Worker


def _complete_progress(shard: ShardMetadata) -> RepoDownloadProgress:
    return RepoDownloadProgress(
        repo_id=str(shard.model_card.model_id),
        repo_revision="test",
        shard=shard,
        completed_files=0,
        total_files=0,
        downloaded_bytes=Memory.from_bytes(0),
        downloaded_bytes_this_session=Memory.from_bytes(0),
        total_bytes=Memory.from_bytes(0),
        overall_speed=0,
        overall_eta=timedelta(seconds=0),
        status="complete",
    )


class _TestShardDownloader(ShardDownloader):
    """Shard downloader that reports every shard as already complete."""

    async def ensure_shard(
        self, shard: ShardMetadata, config_only: bool = False
    ) -> Path:
        return Path("/tmp/test_shard")

    def on_progress(
        self,
        callback: Callable[[ShardMetadata, RepoDownloadProgress], Awaitable[None]],
    ) -> None:
        pass

    async def get_shard_download_status(
        self,
    ) -> AsyncIterator[tuple[Path, RepoDownloadProgress]]:
        # Yield nothing — no pre-existing downloads
        return
        yield  # make this an async generator

    async def get_shard_download_status_for_shard(
        self, shard: ShardMetadata
    ) -> RepoDownloadProgress:
        return _complete_progress(shard)


def _make_heartbeat(node_id: NodeId) -> NodeGatheredInfo:
    return NodeGatheredInfo(
        node_id=node_id,
        when=str(datetime.now(tz=timezone.utc)),
        info=create_node_memory(500),
    )


class _PartitionSwitch:
    """Mutable boolean flag shared with the partition proxy coroutine."""

    def __init__(self) -> None:
        self.connected = True


async def _partition_proxy(
    source: Receiver[ForwarderEvent],
    dest: Sender[ForwarderEvent],
    switch: _PartitionSwitch,
) -> None:
    """Forward events when ``switch.connected`` is True; drop otherwise."""
    with source as events:
        async for event in events:
            if switch.connected:
                await dest.send(event)


async def _wait_until(
    predicate: Callable[[], object], *, timeout: float = 5.0, poll: float = 0.02
) -> None:
    """Poll *predicate* until truthy, raising on timeout."""
    with anyio.fail_after(timeout):
        while not predicate():
            await anyio.sleep(poll)


# ---------------------------------------------------------------------------
# Test 1 – same master: Worker + DC retry recovers lost events
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_partition_recovery_same_master() -> None:
    """Worker's out_for_delivery retry fills the Master's buffer gap after a
    partition heals, even when DownloadCoordinator events are interleaved."""

    master_node = NodeId("master-node")
    worker_node = NodeId("worker-node")
    session = SessionId(master_node_id=master_node, election_clock=1)
    switch = _PartitionSwitch()

    # --- channels --------------------------------------------------------
    # Worker → proxy → Master  (local events)
    worker_local_send, proxy_local_recv = channel[ForwarderEvent]()
    proxy_local_send, master_local_recv = channel[ForwarderEvent]()

    # Master → proxy → Worker  (global events)
    master_global_send, proxy_global_recv = channel[ForwarderEvent]()
    proxy_global_send, worker_global_recv = channel[ForwarderEvent]()

    # Commands (required by constructors)
    cmd_send, cmd_recv = channel[ForwarderCommand]()
    dl_cmd_send, dl_cmd_recv = channel[ForwarderDownloadCommand]()

    # --- components ------------------------------------------------------
    worker = Worker(
        worker_node,
        session,
        global_event_receiver=worker_global_recv,
        local_event_sender=worker_local_send,
        command_sender=cmd_send.clone(),
        download_command_sender=dl_cmd_send.clone(),
        event_index_counter=count(),
    )

    dc = DownloadCoordinator(
        node_id=worker_node,
        shard_downloader=_TestShardDownloader(),
        download_command_receiver=dl_cmd_recv,
        event_sender=worker.event_sender.clone(),
        offline=True,
    )

    master = Master(
        master_node,
        session,
        command_receiver=cmd_recv,
        local_event_receiver=master_local_recv,
        global_event_sender=master_global_send,
        download_command_sender=dl_cmd_send.clone(),
    )

    async with anyio.create_task_group() as tg:
        tg.start_soon(_partition_proxy, proxy_local_recv, proxy_local_send, switch)
        tg.start_soon(_partition_proxy, proxy_global_recv, proxy_global_send, switch)
        tg.start_soon(master.run)
        tg.start_soon(dc.run)
        tg.start_soon(worker.run)

        # 1. Pre-partition: heartbeat reaches master
        await worker.event_sender.send(_make_heartbeat(worker_node))
        await _wait_until(lambda: worker_node in master.state.last_seen)
        initial_last_seen = master.state.last_seen[worker_node]

        # 2. Partition — proxy drops everything
        switch.connected = False

        # Worker heartbeat during partition — lost at proxy, kept in
        # out_for_delivery.
        await worker.event_sender.send(_make_heartbeat(worker_node))

        # Trigger a download via DC's command channel. NoopShardDownloader
        # returns status="complete" for any shard, so _start_download emits
        # NodeDownloadProgress(DownloadPending) then
        # NodeDownloadProgress(DownloadCompleted) through worker.event_sender.
        # These go through _forward_events → proxy (dropped) → out_for_delivery.
        # Use a unique model ID so the DC doesn't skip it as already-completed
        # (it pre-emits progress for the default "noop" model at startup).
        test_shard = PipelineShardMetadata(
            model_card=ModelCard(
                model_id=ModelId("test-partition-model"),
                n_layers=1,
                storage_size=Memory.from_bytes(0),
                hidden_size=1,
                supports_tensor=False,
                tasks=[ModelTask.TextGeneration],
            ),
            device_rank=0,
            world_size=1,
            start_layer=0,
            end_layer=1,
            n_layers=1,
        )
        await dl_cmd_send.send(
            ForwarderDownloadCommand(
                origin=worker_node,
                command=StartDownload(
                    target_node_id=worker_node,
                    shard_metadata=test_shard,
                ),
            )
        )

        # Wait for DC events to flow through worker's _forward_events
        # (poll instead of sleeping a fixed duration to avoid flakiness on slow CI)
        await _wait_until(lambda: len(worker.out_for_delivery) >= 3)

        # Verify at least one is a download progress event
        has_download_event = any(
            isinstance(fe.event, NodeDownloadProgress)
            for fe in worker.out_for_delivery.values()
        )
        assert has_download_event, (
            "out_for_delivery should contain DC-originated download events"
        )

        # 3. Heal partition
        switch.connected = True

        # Worker's _resend_out_for_delivery runs every ~1-2s.
        await _wait_until(
            lambda: master.state.last_seen.get(worker_node, initial_last_seen)
            > initial_last_seen,
            timeout=8.0,
        )

        # 4. All events recovered — both worker heartbeats and DC download
        # progress events were retried and accepted by master.
        await _wait_until(lambda: len(worker.out_for_delivery) == 0, timeout=8.0)

        # Master state reflects the download
        assert worker_node in master.state.downloads

        await master.shutdown()
        worker.shutdown()
        dc.shutdown()
