"""Regression test: when the shard status probe reports a model complete
but the files are missing on disk, the coordinator's periodic scan must
emit DownloadPending, not DownloadCompleted — otherwise the indexed
state.downloads entry gets re-asserted and the model keeps surfacing as
downloaded on the cluster.
"""

import asyncio
import contextlib
from collections.abc import AsyncIterator, Awaitable
from datetime import timedelta
from pathlib import Path
from typing import Callable
from unittest.mock import patch

from exo.download.coordinator import DownloadCoordinator
from exo.download.download_utils import RepoDownloadProgress
from exo.download.shard_downloader import ShardDownloader
from exo.shared.models.model_cards import ModelCard, ModelId, ModelTask
from exo.shared.types.commands import ForwarderDownloadCommand
from exo.shared.types.common import NodeId
from exo.shared.types.events import Event, NodeDownloadProgress
from exo.shared.types.memory import Memory
from exo.shared.types.worker.downloads import DownloadCompleted, DownloadPending
from exo.shared.types.worker.shards import PipelineShardMetadata, ShardMetadata
from exo.utils.channels import Receiver, Sender, channel

NODE_ID = NodeId("aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa")
MODEL_ID = ModelId("test-org/test-model-ghost")


def _make_shard(model_id: ModelId = MODEL_ID) -> ShardMetadata:
    return PipelineShardMetadata(
        model_card=ModelCard(
            model_id=model_id,
            storage_size=Memory.from_mb(100),
            n_layers=28,
            hidden_size=1024,
            supports_tensor=False,
            tasks=[ModelTask.TextGeneration],
        ),
        device_rank=0,
        world_size=1,
        start_layer=0,
        end_layer=28,
        n_layers=28,
    )


class GhostCompleteShardDownloader(ShardDownloader):
    """Fake downloader whose status-scan yields a single model reported as
    status='complete' even though no files exist on disk."""

    def __init__(self) -> None:
        self._progress_callbacks: list[
            Callable[[ShardMetadata, RepoDownloadProgress], Awaitable[None]]
        ] = []

    def on_progress(
        self,
        callback: Callable[[ShardMetadata, RepoDownloadProgress], Awaitable[None]],
    ) -> None:
        self._progress_callbacks.append(callback)

    async def ensure_shard(
        self,
        shard: ShardMetadata,
        config_only: bool = False,  # noqa: ARG002
    ) -> Path:
        raise AssertionError("ensure_shard should not be called in this test")

    async def get_shard_download_status(
        self,
    ) -> AsyncIterator[tuple[Path, RepoDownloadProgress]]:
        yield (
            Path("/fake/models/test-model-ghost"),
            RepoDownloadProgress(
                repo_id=str(MODEL_ID),
                repo_revision="main",
                shard=_make_shard(),
                completed_files=1,
                total_files=1,
                downloaded=Memory.from_mb(100),
                downloaded_this_session=Memory.from_bytes(0),
                total=Memory.from_mb(100),
                overall_speed=0,
                overall_eta=timedelta(seconds=0),
                status="complete",
            ),
        )

    async def get_shard_download_status_for_shard(
        self,
        shard: ShardMetadata,
    ) -> RepoDownloadProgress:
        raise AssertionError(
            "get_shard_download_status_for_shard should not be called in this test"
        )


async def _next_progress_for_model(
    event_recv: Receiver[Event], model_id: ModelId, timeout: float = 2.0
) -> NodeDownloadProgress | None:
    """Drain events until we see any NodeDownloadProgress for the given model, or timeout."""
    try:
        async with asyncio.timeout(timeout):
            while True:
                event = await event_recv.receive()
                if (
                    isinstance(event, NodeDownloadProgress)
                    and event.download_progress.shard_metadata.model_card.model_id
                    == model_id
                ):
                    return event
    except TimeoutError:
        return None


async def test_ghost_completion_emits_pending_not_completed() -> None:
    """Probe says status=complete but resolve_existing_model returns None
    (files are missing). The periodic scan must emit DownloadPending so the
    indexed state.downloads entry clears, not emit a fresh DownloadCompleted
    that would re-advertise the model on the cluster."""
    cmd_send: Sender[ForwarderDownloadCommand]
    cmd_send, cmd_recv = channel[ForwarderDownloadCommand]()
    event_send, event_recv = channel[Event]()

    coordinator = DownloadCoordinator(
        node_id=NODE_ID,
        shard_downloader=GhostCompleteShardDownloader(),
        download_command_receiver=cmd_recv,
        event_sender=event_send,
    )

    with patch(
        "exo.download.coordinator.resolve_existing_model",
        return_value=None,
    ):
        coordinator_task = asyncio.create_task(coordinator.run())
        try:
            event = await _next_progress_for_model(event_recv, MODEL_ID)
            assert event is not None, (
                "coordinator should have emitted a progress event for MODEL_ID"
            )
            assert isinstance(event.download_progress, DownloadPending), (
                f"expected DownloadPending when files are gone; got "
                f"{type(event.download_progress).__name__}"
            )
            assert not isinstance(event.download_progress, DownloadCompleted), (
                "ghost DownloadCompleted regression — this is the bug we're fixing"
            )
        finally:
            await coordinator.shutdown()
            coordinator_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await coordinator_task
            cmd_send.close()
