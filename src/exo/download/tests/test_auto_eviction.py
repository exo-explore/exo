"""Tests for auto-eviction in the DownloadCoordinator.

Tests exercise _start_download (the production entry point) to verify that
storage quota checks and LRU eviction work end-to-end through the coordinator.
"""

from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import AsyncMock, patch

from anyio.streams.memory import MemoryObjectStreamState

from exo.download.coordinator import DownloadCoordinator
from exo.download.shard_downloader import NoopShardDownloader
from exo.shared.models.model_cards import ModelCard, ModelId, ModelTask
from exo.shared.types.commands import ForwarderDownloadCommand
from exo.shared.types.common import NodeId
from exo.shared.types.events import Event, IndexedEvent, NodeDownloadProgress
from exo.shared.types.memory import Memory
from exo.shared.types.storage import StorageConfig
from exo.shared.types.worker.downloads import (
    ModelNotDownloading,
    ModelReady,
    ModelRejected,
)
from exo.shared.types.worker.shards import PipelineShardMetadata, ShardMetadata
from exo.utils.channels import Receiver, Sender

MODEL_A = ModelId("org/model-a")
MODEL_B = ModelId("org/model-b")
MODEL_C = ModelId("org/model-c")
MODEL_NEW = ModelId("org/model-new")
NODE_ID = NodeId("test-node")


def _shard(model_id: ModelId, size_gb: float) -> ShardMetadata:
    return PipelineShardMetadata(
        model_card=ModelCard(
            model_id=model_id,
            storage_size=Memory.from_gb(size_gb),
            n_layers=32,
            hidden_size=1000,
            supports_tensor=True,
            tasks=[ModelTask.TextGeneration],
        ),
        device_rank=0,
        world_size=1,
        start_layer=0,
        end_layer=32,
        n_layers=32,
    )


def _completed(model_id: ModelId, size_gb: float) -> ModelReady:
    return ModelReady(
        node_id=NODE_ID,
        shard_metadata=_shard(model_id, size_gb),
        total=Memory.from_gb(size_gb),
    )


def _make_coordinator(
    storage_config: StorageConfig,
    download_status: dict[ModelId, ModelReady | ModelRejected],
    model_last_used: dict[ModelId, datetime] | None = None,
) -> tuple[DownloadCoordinator, Receiver[Event]]:
    state = MemoryObjectStreamState[Event](max_buffer_size=100)
    event_sender = Sender[Event](_state=state)
    event_receiver = Receiver[Event](_state=state)

    cmd_state: MemoryObjectStreamState[ForwarderDownloadCommand] = (
        MemoryObjectStreamState(max_buffer_size=100)
    )
    cmd_receiver: Receiver[ForwarderDownloadCommand] = Receiver(_state=cmd_state)

    idx_state: MemoryObjectStreamState[IndexedEvent] = MemoryObjectStreamState(
        max_buffer_size=100
    )
    idx_receiver: Receiver[IndexedEvent] = Receiver(_state=idx_state)

    coordinator = DownloadCoordinator(
        node_id=NODE_ID,
        shard_downloader=NoopShardDownloader(),
        download_command_receiver=cmd_receiver,
        event_receiver=idx_receiver,
        event_sender=event_sender,
        storage_config=storage_config,
    )
    coordinator.download_status = dict(download_status)
    if model_last_used is not None:
        coordinator._model_last_used = model_last_used  # pyright: ignore[reportPrivateUsage]

    return coordinator, event_receiver


async def _start_download(
    coordinator: DownloadCoordinator, shard: ShardMetadata
) -> None:
    await coordinator._start_download(shard)  # pyright: ignore[reportPrivateUsage]


class TestStartDownloadAutoEviction:
    """Tests that go through _start_download — the production entry point."""

    @patch(
        "exo.download.coordinator.delete_model",
        new_callable=AsyncMock,
        return_value=True,
    )
    @patch("exo.download.coordinator.resolve_model_in_path", return_value=None)
    async def test_evicts_oldest_model_to_fit_new_download(
        self, _mock_resolve: AsyncMock, mock_delete: AsyncMock
    ) -> None:
        """_start_download should trigger auto-eviction of the oldest model."""
        config = StorageConfig(
            max_storage=Memory.from_gb(10), storage_policy="auto-evict"
        )
        coordinator, _ = _make_coordinator(
            config,
            {MODEL_A: _completed(MODEL_A, 4), MODEL_B: _completed(MODEL_B, 4)},
            model_last_used={
                MODEL_A: datetime(2024, 1, 1, tzinfo=UTC),
                MODEL_B: datetime(2024, 6, 1, tzinfo=UTC),
            },
        )

        await _start_download(coordinator, _shard(MODEL_NEW, 5))

        # MODEL_A (oldest) should have been evicted
        mock_delete.assert_called_once_with(MODEL_A)
        assert MODEL_A not in coordinator.download_status

    @patch(
        "exo.download.coordinator.delete_model",
        new_callable=AsyncMock,
        return_value=True,
    )
    @patch("exo.download.coordinator.resolve_model_in_path", return_value=None)
    async def test_evicts_multiple_in_lru_order(
        self, _mock_resolve: AsyncMock, mock_delete: AsyncMock
    ) -> None:
        """_start_download evicts multiple models oldest-first until space is freed."""
        config = StorageConfig(
            max_storage=Memory.from_gb(10), storage_policy="auto-evict"
        )
        coordinator, _ = _make_coordinator(
            config,
            {
                MODEL_A: _completed(MODEL_A, 3),
                MODEL_B: _completed(MODEL_B, 3),
                MODEL_C: _completed(MODEL_C, 3),
            },
            model_last_used={
                MODEL_A: datetime(2024, 1, 1, tzinfo=UTC),
                MODEL_B: datetime(2024, 6, 1, tzinfo=UTC),
                MODEL_C: datetime(2024, 12, 1, tzinfo=UTC),
            },
        )

        # Need 8 GiB, have 1 GiB free — need to free 7 GiB
        await _start_download(coordinator, _shard(MODEL_NEW, 8))

        evicted = [call.args[0] for call in mock_delete.call_args_list]
        assert evicted == [MODEL_A, MODEL_B, MODEL_C]

    @patch(
        "exo.download.coordinator.delete_model",
        new_callable=AsyncMock,
        return_value=True,
    )
    @patch("exo.download.coordinator.resolve_model_in_path", return_value=None)
    async def test_rejects_when_cannot_free_enough_space(
        self, _mock_resolve: AsyncMock, mock_delete: AsyncMock
    ) -> None:
        """_start_download emits DownloadRejected when eviction can't free enough."""
        config = StorageConfig(
            max_storage=Memory.from_gb(10), storage_policy="auto-evict"
        )
        coordinator, _ = _make_coordinator(
            config,
            {MODEL_A: _completed(MODEL_A, 2)},
        )

        await _start_download(coordinator, _shard(MODEL_NEW, 20))

        mock_delete.assert_not_called()
        assert isinstance(coordinator.download_status[MODEL_NEW], ModelRejected)

    @patch(
        "exo.download.coordinator.delete_model",
        new_callable=AsyncMock,
        return_value=True,
    )
    @patch("exo.download.coordinator.resolve_model_in_path", return_value=None)
    async def test_no_eviction_when_space_available(
        self, _mock_resolve: AsyncMock, mock_delete: AsyncMock
    ) -> None:
        """_start_download proceeds without evicting when enough space exists."""
        config = StorageConfig(
            max_storage=Memory.from_gb(10), storage_policy="auto-evict"
        )
        coordinator, _ = _make_coordinator(
            config,
            {MODEL_A: _completed(MODEL_A, 2)},
        )

        await _start_download(coordinator, _shard(MODEL_NEW, 5))

        mock_delete.assert_not_called()
        # Download should have started (not rejected)
        assert MODEL_NEW in coordinator.download_status
        assert not isinstance(coordinator.download_status[MODEL_NEW], ModelRejected)

    @patch(
        "exo.download.coordinator.delete_model",
        new_callable=AsyncMock,
        return_value=True,
    )
    @patch("exo.download.coordinator.resolve_model_in_path", return_value=None)
    async def test_manual_policy_rejects_instead_of_evicting(
        self, _mock_resolve: AsyncMock, mock_delete: AsyncMock
    ) -> None:
        """With manual policy, _start_download rejects instead of auto-evicting."""
        config = StorageConfig(max_storage=Memory.from_gb(10), storage_policy="manual")
        coordinator, _ = _make_coordinator(
            config,
            {MODEL_A: _completed(MODEL_A, 4), MODEL_B: _completed(MODEL_B, 4)},
        )

        await _start_download(coordinator, _shard(MODEL_NEW, 5))

        mock_delete.assert_not_called()
        assert isinstance(coordinator.download_status[MODEL_NEW], ModelRejected)

    @patch(
        "exo.download.coordinator.delete_model",
        new_callable=AsyncMock,
        return_value=True,
    )
    @patch("exo.download.coordinator.resolve_model_in_path", return_value=None)
    async def test_eviction_emits_not_downloading_event_for_evicted_model(
        self, _mock_resolve: AsyncMock, mock_delete: AsyncMock
    ) -> None:
        """Evicted models emit ModelNotDownloading events and are removed from status."""
        config = StorageConfig(
            max_storage=Memory.from_gb(10), storage_policy="auto-evict"
        )
        coordinator, event_receiver = _make_coordinator(
            config,
            {MODEL_A: _completed(MODEL_A, 4), MODEL_B: _completed(MODEL_B, 4)},
            model_last_used={
                MODEL_A: datetime(2024, 1, 1, tzinfo=UTC),
                MODEL_B: datetime(2024, 6, 1, tzinfo=UTC),
            },
        )

        await _start_download(coordinator, _shard(MODEL_NEW, 5))

        events = event_receiver.collect()
        eviction_events = [
            e
            for e in events
            if isinstance(e, NodeDownloadProgress)
            and isinstance(e.download_progress, ModelNotDownloading)
            and e.download_progress.shard_metadata.model_card.model_id == MODEL_A
        ]
        assert len(eviction_events) == 1
        assert MODEL_A not in coordinator.download_status


class TestActiveModelProtection:
    """Tests that update_active_models protects models from eviction.

    These tests use the public API (update_active_models) to mark models as
    active, then trigger eviction through _start_download.
    """

    @patch(
        "exo.download.coordinator.delete_model",
        new_callable=AsyncMock,
        return_value=True,
    )
    @patch("exo.download.coordinator.resolve_model_in_path", return_value=None)
    async def test_active_model_not_evicted(
        self, _mock_resolve: AsyncMock, mock_delete: AsyncMock
    ) -> None:
        """A model marked active via update_active_models must not be evicted."""
        config = StorageConfig(
            max_storage=Memory.from_gb(10), storage_policy="auto-evict"
        )
        coordinator, _ = _make_coordinator(
            config,
            {MODEL_A: _completed(MODEL_A, 4), MODEL_B: _completed(MODEL_B, 4)},
            model_last_used={
                MODEL_A: datetime(2024, 1, 1, tzinfo=UTC),  # oldest
                MODEL_B: datetime(2024, 6, 1, tzinfo=UTC),
            },
        )

        # Mark MODEL_A as active through the public API
        await coordinator.update_active_models({MODEL_A})

        await _start_download(coordinator, _shard(MODEL_NEW, 5))

        # MODEL_A is active — MODEL_B should be evicted instead
        mock_delete.assert_called_once_with(MODEL_B)
        assert MODEL_A in coordinator.download_status

    @patch(
        "exo.download.coordinator.delete_model",
        new_callable=AsyncMock,
        return_value=True,
    )
    @patch("exo.download.coordinator.resolve_model_in_path", return_value=None)
    async def test_all_active_models_rejected(
        self, _mock_resolve: AsyncMock, mock_delete: AsyncMock
    ) -> None:
        """When all models are active, eviction is impossible — download is rejected."""
        config = StorageConfig(
            max_storage=Memory.from_gb(10), storage_policy="auto-evict"
        )
        coordinator, _ = _make_coordinator(
            config,
            {MODEL_A: _completed(MODEL_A, 4), MODEL_B: _completed(MODEL_B, 4)},
        )

        await coordinator.update_active_models({MODEL_A, MODEL_B})

        await _start_download(coordinator, _shard(MODEL_NEW, 5))

        mock_delete.assert_not_called()
        assert isinstance(coordinator.download_status[MODEL_NEW], ModelRejected)


class TestDiskDeleteFailure:
    """Tests that eviction fails properly when disk delete fails."""

    @patch(
        "exo.download.coordinator.delete_model",
        new_callable=AsyncMock,
        return_value=False,
    )
    @patch("exo.download.coordinator.resolve_model_in_path", return_value=None)
    async def test_eviction_rejected_on_disk_delete_failure(
        self, _mock_resolve: AsyncMock, mock_delete: AsyncMock
    ) -> None:
        """When disk delete fails, auto-eviction emits DownloadRejected."""
        config = StorageConfig(
            max_storage=Memory.from_gb(10), storage_policy="auto-evict"
        )
        coordinator, _ = _make_coordinator(
            config,
            {MODEL_A: _completed(MODEL_A, 4), MODEL_B: _completed(MODEL_B, 4)},
            model_last_used={
                MODEL_A: datetime(2024, 1, 1, tzinfo=UTC),
                MODEL_B: datetime(2024, 6, 1, tzinfo=UTC),
            },
        )

        await _start_download(coordinator, _shard(MODEL_NEW, 5))

        # Should have tried to delete oldest model and failed
        mock_delete.assert_called_once_with(MODEL_A)
        # New model should be rejected
        assert isinstance(coordinator.download_status[MODEL_NEW], ModelRejected)
        # Eviction target should still be in download_status (not removed)
        assert MODEL_A in coordinator.download_status


class TestLruPersistence:
    """Tests for _persist_model_usage and _load_model_usage round-trip."""

    async def test_persist_then_load_round_trip(self, tmp_path: Path) -> None:
        """Persisting then loading recovers the same data."""
        usage_file = tmp_path / "model_usage.json"
        coordinator, _ = _make_coordinator(StorageConfig(), {})
        coordinator._model_last_used = {  # pyright: ignore[reportPrivateUsage]
            MODEL_A: datetime(2024, 1, 15, 12, 30, 0, tzinfo=UTC),
            MODEL_B: datetime(2024, 6, 1, 0, 0, 0, tzinfo=UTC),
        }

        with patch("exo.download.coordinator.EXO_MODEL_USAGE_FILE", usage_file):
            await coordinator._persist_model_usage()  # pyright: ignore[reportPrivateUsage]

            # Create a fresh coordinator and load
            coordinator2, _ = _make_coordinator(StorageConfig(), {})
            await coordinator2._load_model_usage()  # pyright: ignore[reportPrivateUsage]

        assert coordinator2._model_last_used == coordinator._model_last_used  # pyright: ignore[reportPrivateUsage]

    async def test_load_missing_file_returns_empty(self, tmp_path: Path) -> None:
        """Loading when file doesn't exist returns empty dict."""
        usage_file = tmp_path / "nonexistent" / "model_usage.json"
        coordinator, _ = _make_coordinator(StorageConfig(), {})

        with patch("exo.download.coordinator.EXO_MODEL_USAGE_FILE", usage_file):
            await coordinator._load_model_usage()  # pyright: ignore[reportPrivateUsage]

        assert coordinator._model_last_used == {}  # pyright: ignore[reportPrivateUsage]

    async def test_load_corrupt_json_returns_empty(self, tmp_path: Path) -> None:
        """Loading corrupt JSON logs warning and returns empty dict."""
        usage_file = tmp_path / "model_usage.json"
        usage_file.write_text("not valid json {{{")
        coordinator, _ = _make_coordinator(StorageConfig(), {})

        with patch("exo.download.coordinator.EXO_MODEL_USAGE_FILE", usage_file):
            await coordinator._load_model_usage()  # pyright: ignore[reportPrivateUsage]

        assert coordinator._model_last_used == {}  # pyright: ignore[reportPrivateUsage]


class TestEvictionEvents:
    """Tests that eviction emits the correct sequence of events."""

    @patch(
        "exo.download.coordinator.delete_model",
        new_callable=AsyncMock,
        return_value=True,
    )
    @patch("exo.download.coordinator.resolve_model_in_path", return_value=None)
    async def test_eviction_emits_not_downloading_event(
        self, _mock_resolve: AsyncMock, mock_delete: AsyncMock
    ) -> None:
        """Eviction emits a ModelNotDownloading event and removes from status."""
        config = StorageConfig(
            max_storage=Memory.from_gb(10), storage_policy="auto-evict"
        )
        coordinator, event_receiver = _make_coordinator(
            config,
            {MODEL_A: _completed(MODEL_A, 4), MODEL_B: _completed(MODEL_B, 4)},
            model_last_used={
                MODEL_A: datetime(2024, 1, 1, tzinfo=UTC),
                MODEL_B: datetime(2024, 6, 1, tzinfo=UTC),
            },
        )

        await _start_download(coordinator, _shard(MODEL_NEW, 5))

        events = event_receiver.collect()
        eviction_events = [
            e
            for e in events
            if isinstance(e, NodeDownloadProgress)
            and isinstance(e.download_progress, ModelNotDownloading)
            and e.download_progress.shard_metadata.model_card.model_id == MODEL_A
        ]
        assert len(eviction_events) == 1
        assert MODEL_A not in coordinator.download_status

    @patch(
        "exo.download.coordinator.delete_model",
        new_callable=AsyncMock,
        return_value=True,
    )
    @patch("exo.download.coordinator.resolve_model_in_path", return_value=None)
    async def test_multi_eviction_emits_event_per_model(
        self, _mock_resolve: AsyncMock, _mock_delete: AsyncMock
    ) -> None:
        """Each evicted model gets its own ModelNotDownloading event."""
        config = StorageConfig(
            max_storage=Memory.from_gb(10), storage_policy="auto-evict"
        )
        coordinator, event_receiver = _make_coordinator(
            config,
            {
                MODEL_A: _completed(MODEL_A, 3),
                MODEL_B: _completed(MODEL_B, 3),
                MODEL_C: _completed(MODEL_C, 3),
            },
            model_last_used={
                MODEL_A: datetime(2024, 1, 1, tzinfo=UTC),
                MODEL_B: datetime(2024, 6, 1, tzinfo=UTC),
                MODEL_C: datetime(2024, 12, 1, tzinfo=UTC),
            },
        )

        await _start_download(coordinator, _shard(MODEL_NEW, 8))

        events = event_receiver.collect()
        eviction_events = [
            e
            for e in events
            if isinstance(e, NodeDownloadProgress)
            and isinstance(e.download_progress, ModelNotDownloading)
            and e.download_progress.shard_metadata.model_card.model_id != MODEL_NEW
        ]
        evicted_model_ids = [
            e.download_progress.shard_metadata.model_card.model_id
            for e in eviction_events
        ]
        assert evicted_model_ids == [MODEL_A, MODEL_B, MODEL_C]
        for mid in [MODEL_A, MODEL_B, MODEL_C]:
            assert mid not in coordinator.download_status


class TestClearRejections:
    """Tests for clear_rejections behavior."""

    async def test_clear_rejections_resets_rejected(self) -> None:
        """clear_rejections resets ModelRejected to ModelNotDownloading."""
        config = StorageConfig(
            max_storage=Memory.from_gb(10), storage_policy="auto-evict"
        )
        rejected = ModelRejected(
            node_id=NODE_ID,
            shard_metadata=_shard(MODEL_B, 4),
            reason="Not enough space",
            required=Memory.from_gb(4),
            available=Memory.from_gb(1),
            limit=Memory.from_gb(10),
        )
        coordinator, _ = _make_coordinator(
            config,
            {MODEL_A: _completed(MODEL_A, 4), MODEL_B: rejected},
        )

        await coordinator.clear_rejections()

        # Completed should remain unchanged
        assert isinstance(coordinator.download_status[MODEL_A], ModelReady)
        # Rejected should be cleared
        assert isinstance(coordinator.download_status[MODEL_B], ModelNotDownloading)

    async def test_clear_rejections_on_policy_only_change(self) -> None:
        """clear_rejections fires even when only the policy changes (no limit change)."""
        config = StorageConfig(max_storage=Memory.from_gb(10), storage_policy="manual")
        rejected = ModelRejected(
            node_id=NODE_ID,
            shard_metadata=_shard(MODEL_A, 4),
            reason="Manual policy",
            required=Memory.from_gb(4),
            available=Memory.from_gb(1),
            limit=Memory.from_gb(10),
        )
        coordinator, _ = _make_coordinator(
            config,
            {MODEL_A: rejected, MODEL_B: _completed(MODEL_B, 3)},
        )

        await coordinator.clear_rejections()

        # Rejected should be cleared to Pending
        assert isinstance(coordinator.download_status[MODEL_A], ModelNotDownloading)
        # Completed should be unchanged
        assert isinstance(coordinator.download_status[MODEL_B], ModelReady)
