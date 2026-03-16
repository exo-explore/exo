from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import patch

from exo.shared.models.model_cards import ModelId
from exo.shared.storage import (
    calculate_used_storage,
    check_storage_quota,
    compute_evictions_needed,
    decide_storage_action,
    get_download_rejected_events,
    get_lru_eviction_candidates,
    load_storage_config,
    persist_storage_config,
)
from exo.shared.tests.conftest import get_pipeline_shard_metadata
from exo.shared.types.common import NodeId
from exo.shared.types.events import InstanceDeleted, TaskStatusUpdated
from exo.shared.types.memory import Memory
from exo.shared.types.storage import (
    StorageAllow,
    StorageConfig,
    StorageEvict,
    StorageReject,
)
from exo.shared.types.tasks import LoadModel, TaskId, TaskStatus
from exo.shared.types.worker.downloads import (
    ModelReady,
    ModelDownloading,
    ModelNotDownloading,
    DownloadProgressData,
)
from exo.shared.types.worker.instances import InstanceId, MlxRingInstance
from exo.shared.types.worker.runners import RunnerId, ShardAssignments

MODEL_A = ModelId("org/model-a")
MODEL_B = ModelId("org/model-b")
MODEL_C = ModelId("org/model-c")
MODEL_D = ModelId("org/model-d")
NODE_ID = "node-1"


def _completed(
    model_id: ModelId, size_gb: float, read_only: bool = False
) -> ModelReady:
    shard = get_pipeline_shard_metadata(model_id, device_rank=0)
    return ModelReady(
        node_id=NODE_ID,  # type: ignore[arg-type]
        shard_metadata=shard,
        total=Memory.from_gb(size_gb),
        read_only=read_only,
    )


class TestCheckStorageQuota:
    def test_unlimited_allows(self) -> None:
        config = StorageConfig(max_storage=None)
        allowed, _reason = check_storage_quota(Memory.from_gb(10), config, [])
        assert allowed is True
        assert _reason == ""

    def test_under_limit_allows(self) -> None:
        config = StorageConfig(max_storage=Memory.from_gb(20))
        downloads = [_completed(MODEL_A, 5)]
        allowed, _reason = check_storage_quota(Memory.from_gb(10), config, downloads)
        assert allowed is True

    def test_over_limit_rejects(self) -> None:
        config = StorageConfig(max_storage=Memory.from_gb(10))
        downloads = [_completed(MODEL_A, 5)]
        allowed, reason = check_storage_quota(Memory.from_gb(8), config, downloads)
        assert allowed is False
        assert "Need" in reason
        assert "available" in reason

    def test_exact_fit_allows(self) -> None:
        config = StorageConfig(max_storage=Memory.from_gb(10))
        downloads = [_completed(MODEL_A, 5)]
        allowed, _ = check_storage_quota(Memory.from_gb(5), config, downloads)
        assert allowed is True


class TestGetLruEvictionCandidates:
    def test_excludes_active_models(self) -> None:
        downloads = [_completed(MODEL_A, 5), _completed(MODEL_B, 3)]
        last_used = {
            MODEL_A: datetime(2024, 1, 1, tzinfo=UTC),
            MODEL_B: datetime(2024, 1, 2, tzinfo=UTC),
        }
        candidates = get_lru_eviction_candidates(
            downloads, last_used, frozenset({MODEL_A})
        )
        assert len(candidates) == 1
        assert candidates[0][0] == MODEL_B

    def test_excludes_read_only(self) -> None:
        downloads = [_completed(MODEL_A, 5, read_only=True), _completed(MODEL_B, 3)]
        candidates = get_lru_eviction_candidates(downloads, {}, frozenset())
        assert len(candidates) == 1
        assert candidates[0][0] == MODEL_B

    def test_sorts_oldest_first(self) -> None:
        downloads = [_completed(MODEL_A, 5), _completed(MODEL_B, 3)]
        last_used = {
            MODEL_A: datetime(2024, 6, 1, tzinfo=UTC),
            MODEL_B: datetime(2024, 1, 1, tzinfo=UTC),
        }
        candidates = get_lru_eviction_candidates(downloads, last_used, frozenset())
        assert candidates[0][0] == MODEL_B
        assert candidates[1][0] == MODEL_A

    def test_models_without_usage_get_min(self) -> None:
        downloads = [_completed(MODEL_A, 5), _completed(MODEL_B, 3)]
        last_used = {MODEL_A: datetime(2024, 6, 1, tzinfo=UTC)}
        candidates = get_lru_eviction_candidates(downloads, last_used, frozenset())
        assert candidates[0][0] == MODEL_B  # no usage -> datetime.min


class TestComputeEvictionsNeeded:
    def test_sufficient_candidates(self) -> None:
        candidates = [
            (MODEL_A, _completed(MODEL_A, 5)),
            (MODEL_B, _completed(MODEL_B, 3)),
        ]
        result = compute_evictions_needed(
            Memory.from_gb(6), Memory.from_gb(2), candidates
        )
        assert result is not None
        assert MODEL_A in result

    def test_insufficient_candidates(self) -> None:
        candidates = [
            (MODEL_A, _completed(MODEL_A, 2)),
        ]
        result = compute_evictions_needed(
            Memory.from_gb(10), Memory.from_gb(2), candidates
        )
        assert result is None

    def test_no_eviction_needed(self) -> None:
        candidates = [(MODEL_A, _completed(MODEL_A, 5))]
        result = compute_evictions_needed(
            Memory.from_gb(3), Memory.from_gb(5), candidates
        )
        assert result == []

    def test_evicts_in_lru_order(self) -> None:
        """Eviction picks candidates in the order given (oldest first from LRU sort)."""
        candidates = [
            (MODEL_A, _completed(MODEL_A, 2)),  # oldest
            (MODEL_B, _completed(MODEL_B, 2)),  # newer
            (MODEL_C, _completed(MODEL_C, 2)),  # newest
        ]
        result = compute_evictions_needed(
            Memory.from_gb(3), Memory.from_gb(1), candidates
        )
        assert result == [MODEL_A]

    def test_evicts_multiple_until_enough_space(self) -> None:
        """When one model isn't enough, evicts multiple in LRU order."""
        candidates = [
            (MODEL_A, _completed(MODEL_A, 1)),
            (MODEL_B, _completed(MODEL_B, 1)),
            (MODEL_C, _completed(MODEL_C, 1)),
        ]
        # Need 4 GiB, have 1 GiB available — need 3 GiB freed
        result = compute_evictions_needed(
            Memory.from_gb(4), Memory.from_gb(1), candidates
        )
        assert result == [MODEL_A, MODEL_B, MODEL_C]

    def test_evicts_minimum_needed(self) -> None:
        """Stops evicting as soon as enough space is freed."""
        candidates = [
            (MODEL_A, _completed(MODEL_A, 3)),
            (MODEL_B, _completed(MODEL_B, 3)),
        ]
        # Need 5 GiB, have 2 GiB — need 3 GiB freed. Model A alone suffices.
        result = compute_evictions_needed(
            Memory.from_gb(5), Memory.from_gb(2), candidates
        )
        assert result == [MODEL_A]


class TestCalculateUsedStorage:
    def test_only_counts_completed_and_ongoing(self) -> None:
        """Pending and rejected downloads should not count toward used storage."""
        shard_a = get_pipeline_shard_metadata(MODEL_A, device_rank=0)
        shard_b = get_pipeline_shard_metadata(MODEL_B, device_rank=0)
        downloads = [
            _completed(MODEL_A, 5),
            ModelNotDownloading(
                node_id=NODE_ID,  # type: ignore[arg-type]
                shard_metadata=shard_a,
            ),
            ModelDownloading(
                node_id=NODE_ID,  # type: ignore[arg-type]
                shard_metadata=shard_b,
                download_progress=DownloadProgressData(
                    total=Memory.from_gb(10),
                    downloaded=Memory.from_gb(3),
                    downloaded_this_session=Memory.from_gb(3),
                    completed_files=1,
                    total_files=5,
                    speed=0,
                    eta_ms=0,
                    files={},
                ),
            ),
        ]
        used = calculate_used_storage(downloads)
        # 5 GiB completed + 10 GiB ongoing total = 15 GiB
        assert abs(used.in_gb - 15.0) < 0.01

    def test_empty_downloads(self) -> None:
        assert calculate_used_storage([]).in_bytes == 0


class TestGetLruEvictionCandidatesExtended:
    def test_excludes_non_completed_downloads(self) -> None:
        """Only DownloadCompleted entries are eviction candidates."""
        shard_a = get_pipeline_shard_metadata(MODEL_A, device_rank=0)
        downloads = [
            _completed(MODEL_B, 3),
            ModelNotDownloading(
                node_id=NODE_ID,  # type: ignore[arg-type]
                shard_metadata=shard_a,
            ),
        ]
        candidates = get_lru_eviction_candidates(downloads, {}, frozenset())
        assert len(candidates) == 1
        assert candidates[0][0] == MODEL_B

    def test_all_active_returns_empty(self) -> None:
        """When all completed models are active, no candidates available."""
        downloads = [_completed(MODEL_A, 5), _completed(MODEL_B, 3)]
        candidates = get_lru_eviction_candidates(
            downloads, {}, frozenset({MODEL_A, MODEL_B})
        )
        assert candidates == []

    def test_three_models_lru_order(self) -> None:
        """Three models sorted correctly: oldest used first."""
        downloads = [
            _completed(MODEL_A, 2),
            _completed(MODEL_B, 3),
            _completed(MODEL_C, 1),
        ]
        last_used = {
            MODEL_A: datetime(2024, 3, 1, tzinfo=UTC),
            MODEL_B: datetime(2024, 1, 1, tzinfo=UTC),
            MODEL_C: datetime(2024, 6, 1, tzinfo=UTC),
        }
        candidates = get_lru_eviction_candidates(downloads, last_used, frozenset())
        assert [c[0] for c in candidates] == [MODEL_B, MODEL_A, MODEL_C]

    def test_mixed_active_readonly_and_regular(self) -> None:
        """Only non-active, non-read-only completed models are candidates."""
        downloads = [
            _completed(MODEL_A, 5, read_only=True),  # excluded: read-only
            _completed(MODEL_B, 3),  # excluded: active
            _completed(MODEL_C, 2),  # candidate
            _completed(MODEL_D, 1),  # candidate
        ]
        last_used = {
            MODEL_C: datetime(2024, 6, 1, tzinfo=UTC),
            MODEL_D: datetime(2024, 1, 1, tzinfo=UTC),
        }
        candidates = get_lru_eviction_candidates(
            downloads, last_used, frozenset({MODEL_B})
        )
        assert [c[0] for c in candidates] == [MODEL_D, MODEL_C]


class TestEndToEndEvictionScenario:
    """Tests that combine LRU candidate selection with eviction computation."""

    def test_evicts_oldest_model_to_fit_new_one(self) -> None:
        """10 GiB limit, 3 completed models totaling 9 GiB,
        need 3 GiB for new model — should evict the oldest."""
        downloads = [
            _completed(MODEL_A, 3),  # oldest used
            _completed(MODEL_B, 3),
            _completed(MODEL_C, 3),  # newest used
        ]
        last_used = {
            MODEL_A: datetime(2024, 1, 1, tzinfo=UTC),
            MODEL_B: datetime(2024, 6, 1, tzinfo=UTC),
            MODEL_C: datetime(2024, 12, 1, tzinfo=UTC),
        }
        config = StorageConfig(max_storage=Memory.from_gb(10))
        new_model_size = Memory.from_gb(3)

        # Step 1: quota check fails
        allowed, _ = check_storage_quota(new_model_size, config, downloads)
        assert not allowed

        # Step 2: get candidates in LRU order
        candidates = get_lru_eviction_candidates(downloads, last_used, frozenset())
        assert candidates[0][0] == MODEL_A  # oldest

        # Step 3: compute what to evict
        used = calculate_used_storage(downloads)
        assert config.max_storage is not None
        available = config.max_storage - used
        to_evict = compute_evictions_needed(new_model_size, available, candidates)
        assert to_evict == [MODEL_A]

    def test_protects_currently_active_model(self) -> None:
        """Active model should not be evicted even if it's the oldest."""
        downloads = [
            _completed(MODEL_A, 4),  # oldest but active
            _completed(MODEL_B, 4),  # next oldest, evictable
        ]
        last_used = {
            MODEL_A: datetime(2024, 1, 1, tzinfo=UTC),
            MODEL_B: datetime(2024, 6, 1, tzinfo=UTC),
        }
        config = StorageConfig(max_storage=Memory.from_gb(10))
        new_model_size = Memory.from_gb(5)

        allowed, _ = check_storage_quota(new_model_size, config, downloads)
        assert not allowed

        # MODEL_A is active — should be excluded
        candidates = get_lru_eviction_candidates(
            downloads, last_used, frozenset({MODEL_A})
        )
        assert len(candidates) == 1
        assert candidates[0][0] == MODEL_B

        used = calculate_used_storage(downloads)
        assert config.max_storage is not None
        available = config.max_storage - used
        to_evict = compute_evictions_needed(new_model_size, available, candidates)
        assert to_evict == [MODEL_B]

    def test_cannot_evict_enough_returns_none(self) -> None:
        """When all evictable space isn't enough, returns None."""
        downloads = [
            _completed(MODEL_A, 4, read_only=True),  # can't evict
            _completed(MODEL_B, 2),  # can evict but only 2 GiB
        ]
        config = StorageConfig(max_storage=Memory.from_gb(10))
        new_model_size = Memory.from_gb(8)

        candidates = get_lru_eviction_candidates(downloads, {}, frozenset())
        used = calculate_used_storage(downloads)
        assert config.max_storage is not None
        available = config.max_storage - used
        to_evict = compute_evictions_needed(new_model_size, available, candidates)
        assert to_evict is None


class TestDecideStorageAction:
    """Tests for the decide_storage_action pure function."""

    def test_unlimited_allows(self) -> None:
        config = StorageConfig(max_storage=None)
        action = decide_storage_action(Memory.from_gb(10), config, [], {}, frozenset())
        assert isinstance(action, StorageAllow)

    def test_under_limit_allows(self) -> None:
        config = StorageConfig(max_storage=Memory.from_gb(20))
        downloads = [_completed(MODEL_A, 5)]
        action = decide_storage_action(
            Memory.from_gb(10), config, downloads, {}, frozenset()
        )
        assert isinstance(action, StorageAllow)

    def test_manual_policy_rejects(self) -> None:
        config = StorageConfig(max_storage=Memory.from_gb(10), storage_policy="manual")
        downloads = [_completed(MODEL_A, 5)]
        action = decide_storage_action(
            Memory.from_gb(8), config, downloads, {}, frozenset()
        )
        assert isinstance(action, StorageReject)
        assert "Need" in action.reason

    def test_auto_evict_returns_evict(self) -> None:
        config = StorageConfig(
            max_storage=Memory.from_gb(10), storage_policy="auto-evict"
        )
        downloads = [_completed(MODEL_A, 4), _completed(MODEL_B, 4)]
        last_used = {
            MODEL_A: datetime(2024, 1, 1, tzinfo=UTC),
            MODEL_B: datetime(2024, 6, 1, tzinfo=UTC),
        }
        action = decide_storage_action(
            Memory.from_gb(5), config, downloads, last_used, frozenset()
        )
        assert isinstance(action, StorageEvict)
        assert MODEL_A in action.model_ids

    def test_auto_evict_rejects_when_impossible(self) -> None:
        config = StorageConfig(
            max_storage=Memory.from_gb(10), storage_policy="auto-evict"
        )
        downloads = [_completed(MODEL_A, 2)]
        action = decide_storage_action(
            Memory.from_gb(20), config, downloads, {}, frozenset()
        )
        assert isinstance(action, StorageReject)
        assert "Cannot free enough" in action.reason

    def test_auto_evict_protects_active_models(self) -> None:
        config = StorageConfig(
            max_storage=Memory.from_gb(10), storage_policy="auto-evict"
        )
        downloads = [_completed(MODEL_A, 4), _completed(MODEL_B, 4)]
        last_used = {
            MODEL_A: datetime(2024, 1, 1, tzinfo=UTC),
            MODEL_B: datetime(2024, 6, 1, tzinfo=UTC),
        }
        # MODEL_A is oldest but active — should evict MODEL_B instead
        action = decide_storage_action(
            Memory.from_gb(5), config, downloads, last_used, frozenset({MODEL_A})
        )
        assert isinstance(action, StorageEvict)
        assert action.model_ids == [MODEL_B]

    def test_auto_evict_all_active_rejects(self) -> None:
        config = StorageConfig(
            max_storage=Memory.from_gb(10), storage_policy="auto-evict"
        )
        downloads = [_completed(MODEL_A, 4), _completed(MODEL_B, 4)]
        action = decide_storage_action(
            Memory.from_gb(5),
            config,
            downloads,
            {},
            frozenset({MODEL_A, MODEL_B}),
        )
        assert isinstance(action, StorageReject)


def _make_instance(
    instance_id: InstanceId,
    model_id: ModelId,
    node_id: NodeId,
) -> MlxRingInstance:
    shard = get_pipeline_shard_metadata(model_id, device_rank=0)
    runner_id = RunnerId()
    return MlxRingInstance(
        instance_id=instance_id,
        shard_assignments=ShardAssignments(
            model_id=model_id,
            runner_to_shard={runner_id: shard},
            node_to_runner={node_id: runner_id},
        ),
        hosts_by_node={node_id: []},
        ephemeral_port=0,
    )


class TestGetDownloadRejectedEvents:
    """Tests for the get_download_rejected_events pure function."""

    def test_deletes_instance_for_rejected_model(self) -> None:
        node_id = NodeId("node-1")
        instance_id = InstanceId()
        instance = _make_instance(instance_id, MODEL_A, node_id)
        events = get_download_rejected_events(
            MODEL_A, node_id, {instance_id: instance}, {}
        )
        assert len(events) == 1
        assert isinstance(events[0], InstanceDeleted)
        assert events[0].instance_id == instance_id

    def test_fails_pending_tasks_before_deleting_instance(self) -> None:
        node_id = NodeId("node-1")
        instance_id = InstanceId()
        instance = _make_instance(instance_id, MODEL_A, node_id)
        task_id = TaskId()
        task = LoadModel(
            task_id=task_id,
            instance_id=instance_id,
            task_status=TaskStatus.Pending,
        )
        events = get_download_rejected_events(
            MODEL_A, node_id, {instance_id: instance}, {task_id: task}
        )
        assert len(events) == 2
        assert isinstance(events[0], TaskStatusUpdated)
        assert events[0].task_status == TaskStatus.Failed
        assert isinstance(events[1], InstanceDeleted)

    def test_ignores_different_model(self) -> None:
        node_id = NodeId("node-1")
        instance_id = InstanceId()
        instance = _make_instance(instance_id, MODEL_B, node_id)
        events = get_download_rejected_events(
            MODEL_A, node_id, {instance_id: instance}, {}
        )
        assert events == []

    def test_ignores_different_node(self) -> None:
        node_id = NodeId("node-1")
        other_node = NodeId("node-2")
        instance_id = InstanceId()
        instance = _make_instance(instance_id, MODEL_A, other_node)
        events = get_download_rejected_events(
            MODEL_A, node_id, {instance_id: instance}, {}
        )
        assert events == []

    def test_skips_completed_tasks(self) -> None:
        node_id = NodeId("node-1")
        instance_id = InstanceId()
        instance = _make_instance(instance_id, MODEL_A, node_id)
        task_id = TaskId()
        task = LoadModel(
            task_id=task_id,
            instance_id=instance_id,
            task_status=TaskStatus.Complete,
        )
        events = get_download_rejected_events(
            MODEL_A, node_id, {instance_id: instance}, {task_id: task}
        )
        # Only InstanceDeleted, no TaskStatusUpdated for completed task
        assert len(events) == 1
        assert isinstance(events[0], InstanceDeleted)


class TestPersistStorageConfig:
    """Tests for persist_storage_config I/O."""

    async def test_round_trip(self, tmp_path: Path) -> None:
        cfg_file = tmp_path / "config.toml"
        config = StorageConfig(
            max_storage=Memory.from_gb(50), storage_policy="auto-evict"
        )
        with patch("exo.shared.storage.EXO_CONFIG_FILE", cfg_file):
            await persist_storage_config(config)
            loaded = await load_storage_config()

        assert loaded.storage_policy == "auto-evict"
        assert loaded.max_storage is not None
        assert abs(loaded.max_storage.in_gb - 50.0) < 0.1

    async def test_preserves_other_keys(self, tmp_path: Path) -> None:
        cfg_file = tmp_path / "config.toml"
        cfg_file.write_text('some_other_key = "hello"\n')
        config = StorageConfig(max_storage=Memory.from_gb(10))
        with patch("exo.shared.storage.EXO_CONFIG_FILE", cfg_file):
            await persist_storage_config(config)

        contents = cfg_file.read_text()
        assert "some_other_key" in contents
        assert "hello" in contents
        assert "max_storage_gb" in contents

    async def test_clears_max_storage_gb_when_unlimited(self, tmp_path: Path) -> None:
        cfg_file = tmp_path / "config.toml"
        cfg_file.write_text('max_storage_gb = 50\nstorage_policy = "auto-evict"\n')
        config = StorageConfig(max_storage=None, storage_policy="manual")
        with patch("exo.shared.storage.EXO_CONFIG_FILE", cfg_file):
            await persist_storage_config(config)

        contents = cfg_file.read_text()
        assert "max_storage_gb" not in contents
        assert "manual" in contents

    async def test_creates_file_if_missing(self, tmp_path: Path) -> None:
        cfg_file = tmp_path / "subdir" / "config.toml"
        config = StorageConfig(max_storage=Memory.from_gb(25))
        with patch("exo.shared.storage.EXO_CONFIG_FILE", cfg_file):
            await persist_storage_config(config)

        assert cfg_file.exists()
        assert "max_storage_gb" in cfg_file.read_text()


class TestLoadStorageConfig:
    """Tests for load_storage_config I/O."""

    async def test_defaults_when_empty_file(self, tmp_path: Path) -> None:
        cfg_file = tmp_path / "config.toml"
        cfg_file.write_text("")
        with patch("exo.shared.storage.EXO_CONFIG_FILE", cfg_file):
            config = await load_storage_config()

        assert config.max_storage is None
        assert config.storage_policy == "manual"

    async def test_reads_from_file(self, tmp_path: Path) -> None:
        cfg_file = tmp_path / "config.toml"
        cfg_file.write_text('max_storage_gb = 30.0\nstorage_policy = "auto-evict"\n')
        with patch("exo.shared.storage.EXO_CONFIG_FILE", cfg_file):
            config = await load_storage_config()

        assert config.storage_policy == "auto-evict"
        assert config.max_storage is not None
        assert abs(config.max_storage.in_gb - 30.0) < 0.1

    async def test_cli_overrides_file(self, tmp_path: Path) -> None:
        cfg_file = tmp_path / "config.toml"
        cfg_file.write_text('max_storage_gb = 30.0\nstorage_policy = "manual"\n')
        with patch("exo.shared.storage.EXO_CONFIG_FILE", cfg_file):
            config = await load_storage_config(
                max_storage_gb=100.0, storage_policy="auto-evict"
            )

        assert config.storage_policy == "auto-evict"
        assert config.max_storage is not None
        assert abs(config.max_storage.in_gb - 100.0) < 0.1

    async def test_partial_cli_override(self, tmp_path: Path) -> None:
        """CLI overrides only the fields provided, file values used for the rest."""
        cfg_file = tmp_path / "config.toml"
        cfg_file.write_text('max_storage_gb = 30.0\nstorage_policy = "auto-evict"\n')
        with patch("exo.shared.storage.EXO_CONFIG_FILE", cfg_file):
            config = await load_storage_config(max_storage_gb=50.0)

        assert config.storage_policy == "auto-evict"  # from file
        assert config.max_storage is not None
        assert abs(config.max_storage.in_gb - 50.0) < 0.1  # from CLI

    async def test_defaults_on_corrupt_file(self, tmp_path: Path) -> None:
        cfg_file = tmp_path / "config.toml"
        cfg_file.write_text("not valid toml {{{")
        with patch("exo.shared.storage.EXO_CONFIG_FILE", cfg_file):
            config = await load_storage_config()

        assert config.max_storage is None
        assert config.storage_policy == "manual"

    async def test_creates_file_if_missing(self, tmp_path: Path) -> None:
        cfg_file = tmp_path / "subdir" / "config.toml"
        with patch("exo.shared.storage.EXO_CONFIG_FILE", cfg_file):
            config = await load_storage_config()

        assert config.max_storage is None
        assert cfg_file.exists()
