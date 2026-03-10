from collections.abc import Mapping, Sequence
from datetime import UTC, datetime

from exo.shared.models.model_cards import ModelId
from exo.shared.types.memory import Memory
from exo.shared.types.storage import StorageConfig
from exo.shared.types.worker.downloads import (
    DownloadCompleted,
    DownloadOngoing,
    DownloadProgress,
)


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

    # Sort by last-used time, oldest first
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
