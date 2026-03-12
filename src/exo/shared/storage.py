import tomllib
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime

import anyio
import tomlkit
from loguru import logger
from tomlkit.exceptions import TOMLKitError

from exo.shared.constants import EXO_CONFIG_FILE
from exo.shared.models.model_cards import ModelId
from exo.shared.types.common import NodeId
from exo.shared.types.events import InstanceDeleted, TaskStatusUpdated
from exo.shared.types.memory import Memory
from exo.shared.types.storage import (
    StorageAllow,
    StorageConfig,
    StorageDecision,
    StorageEvict,
    StoragePolicy,
    StorageReject,
)
from exo.shared.types.tasks import Task, TaskId, TaskStatus
from exo.shared.types.worker.downloads import (
    DownloadCompleted,
    DownloadOngoing,
    DownloadProgress,
)
from exo.shared.types.worker.instances import Instance, InstanceId


def calculate_used_storage(downloads: Sequence[DownloadProgress]) -> Memory:
    total = Memory()
    for dp in downloads:
        if isinstance(dp, DownloadCompleted):
            total = total + dp.total
        elif isinstance(dp, DownloadOngoing):
            total = total + dp.download_progress.total
    return total


def check_storage_quota(
    model_size: Memory,
    config: StorageConfig,
    downloads: Sequence[DownloadProgress],
) -> tuple[bool, str]:
    if config.max_storage is None:
        return True, ""

    used = calculate_used_storage(downloads)
    available = config.max_storage - used

    if model_size <= available:
        return True, ""

    return (
        False,
        f"Need {model_size.in_gb:.1f} GiB, only {max(0, available.in_gb):.1f} GiB available within {config.max_storage.in_gb:.1f} GiB limit",
    )


def get_lru_eviction_candidates(
    downloads: Sequence[DownloadProgress],
    model_last_used: Mapping[ModelId, datetime],
    active_model_ids: frozenset[ModelId],
) -> list[tuple[ModelId, DownloadCompleted]]:
    candidates: list[tuple[ModelId, DownloadCompleted]] = []
    for dp in downloads:
        if not isinstance(dp, DownloadCompleted):
            continue
        if dp.read_only:
            continue
        model_id = dp.shard_metadata.model_card.model_id
        if model_id in active_model_ids:
            continue
        candidates.append((model_id, dp))

    candidates.sort(
        key=lambda item: model_last_used.get(item[0], datetime.min.replace(tzinfo=UTC))
    )
    return candidates


def compute_evictions_needed(
    model_size: Memory,
    available: Memory,
    candidates: list[tuple[ModelId, DownloadCompleted]],
) -> list[ModelId] | None:
    if model_size <= available:
        return []

    space_needed = model_size - available
    freed = Memory()
    to_evict: list[ModelId] = []

    for model_id, completed in candidates:
        to_evict.append(model_id)
        freed = freed + completed.total
        if freed >= space_needed:
            return to_evict

    return None


def decide_storage_action(
    model_size: Memory,
    config: StorageConfig,
    downloads: Sequence[DownloadProgress],
    model_last_used: Mapping[ModelId, datetime],
    active_model_ids: frozenset[ModelId],
) -> StorageDecision:
    """Pure decision function: given storage state, decide whether to allow, evict, or reject."""
    if config.max_storage is None:
        return StorageAllow()

    allowed, reason = check_storage_quota(model_size, config, downloads)
    if allowed:
        return StorageAllow()

    used = calculate_used_storage(downloads)
    raw_available = config.max_storage - used
    available = raw_available if raw_available.in_bytes >= 0 else Memory()

    if config.storage_policy == "auto-evict":
        candidates = get_lru_eviction_candidates(
            downloads, model_last_used, active_model_ids
        )
        to_evict = compute_evictions_needed(model_size, available, candidates)

        if to_evict is not None:
            return StorageEvict(model_ids=to_evict)

        return StorageReject(
            reason="Cannot free enough space even after evicting all eligible models",
            available=available,
        )

    return StorageReject(
        reason=reason,
        available=available,
    )


def get_download_rejected_events(
    rejected_model_id: ModelId,
    rejected_node_id: NodeId,
    instances: Mapping[InstanceId, Instance],
    tasks: Mapping[TaskId, Task],
) -> list[TaskStatusUpdated | InstanceDeleted]:
    """Pure function: compute events needed to clean up after a download rejection."""
    events: list[TaskStatusUpdated | InstanceDeleted] = []
    for instance_id, instance in instances.items():
        if (
            instance.shard_assignments.model_id == rejected_model_id
            and rejected_node_id in instance.shard_assignments.node_to_runner
        ):
            for task in tasks.values():
                if task.instance_id == instance_id and task.task_status in (
                    TaskStatus.Pending,
                    TaskStatus.Running,
                ):
                    events.append(
                        TaskStatusUpdated(
                            task_id=task.task_id,
                            task_status=TaskStatus.Failed,
                        )
                    )
            events.append(InstanceDeleted(instance_id=instance_id))
    return events


async def load_storage_config(
    *,
    max_storage_gb: float | None = None,
    storage_policy: StoragePolicy | None = None,
) -> StorageConfig:
    """Load StorageConfig from config.toml, overlaying any CLI arg overrides."""
    base = StorageConfig()
    cfg_file = anyio.Path(EXO_CONFIG_FILE)
    try:
        await cfg_file.parent.mkdir(parents=True, exist_ok=True)
        await cfg_file.touch(exist_ok=True)
        raw = (await cfg_file.read_bytes()).decode("utf-8")
        if raw.strip():
            data = tomllib.loads(raw)
            base = StorageConfig.from_disk(data)
    except Exception:
        logger.warning("Failed to read storage config from config file, using defaults")

    resolved_max_storage = (
        Memory.from_gb(max_storage_gb)
        if max_storage_gb is not None
        else base.max_storage
    )
    resolved_policy = (
        storage_policy if storage_policy is not None else base.storage_policy
    )
    return StorageConfig(
        max_storage=resolved_max_storage, storage_policy=resolved_policy
    )


async def persist_storage_config(config: StorageConfig) -> None:
    """Persist StorageConfig to config.toml, preserving other config keys."""
    cfg_path = anyio.Path(EXO_CONFIG_FILE)
    await cfg_path.parent.mkdir(parents=True, exist_ok=True)

    doc = tomlkit.document()
    try:
        raw = (await cfg_path.read_bytes()).decode("utf-8")
        if raw.strip():
            doc = tomlkit.parse(raw)
    except (FileNotFoundError, TOMLKitError, UnicodeDecodeError):
        pass

    # Clear max_storage_gb so it doesn't linger when max_storage is None
    doc.pop("max_storage_gb", None)  # pyright: ignore[reportUnknownMemberType]
    doc.update(config.to_disk())  # pyright: ignore[reportUnknownMemberType]

    await cfg_path.write_text(tomlkit.dumps(doc))  # pyright: ignore[reportUnknownMemberType]
    logger.debug(f"Persisted storage config to {cfg_path}")
