import time
import os
from pathlib import Path
import shutil
from typing import Callable

import pytest

from exo.shared.types.worker.shards import PipelineShardMetadata, ShardMetadata
from exo.worker.download.download_utils import RepoDownloadProgress
from exo.worker.download.impl_shard_downloader import exo_shard_downloader
from exo.worker.download.shard_downloader import ShardDownloader


@pytest.mark.asyncio
async def test_shard_downloader(
    pipeline_shard_meta: Callable[[int, int], PipelineShardMetadata],
):
    shutil.rmtree(Path(os.path.expanduser("~/.exo/models/mlx-community--Llama-3.2-1B-Instruct-4bit")))

    progress_log: list[RepoDownloadProgress] = []
    shard_downloader: ShardDownloader = exo_shard_downloader()
    def _on_progress(shard: ShardMetadata, progress: RepoDownloadProgress):
        print(f"Download progress: {progress}")
        progress_log.append(progress)
    shard_downloader.on_progress(_on_progress)

    shard_metadata = pipeline_shard_meta(1, 0)
    path = await shard_downloader.ensure_shard(shard_metadata)
    assert path.exists()

    downloaded_model_path = path.parent / "mlx-community--Llama-3.2-1B-Instruct-4bit"
    assert (downloaded_model_path / "config.json").exists()
    assert (downloaded_model_path / "model.safetensors").exists()
    assert (downloaded_model_path / "model.safetensors.index.json").exists()
    assert (downloaded_model_path / "special_tokens_map.json").exists()
    assert (downloaded_model_path / "tokenizer.json").exists()
    assert (downloaded_model_path / "tokenizer_config.json").exists()

    expected_files_and_sizes = [
        ("config.json", 1121),
        ("model.safetensors", 695283921),
        ("model.safetensors.index.json", 26159),
        ("special_tokens_map.json", 296),
        ("tokenizer.json", 17209920),
        ("tokenizer_config.json", 54558),
    ]
    for filename, expected_size in expected_files_and_sizes:
        file_path = downloaded_model_path / filename
        assert file_path.stat().st_size == expected_size, f"{filename} size mismatch"

    start_time = time.monotonic()
    path_again = await shard_downloader.ensure_shard(shard_metadata)
    duration = time.monotonic() - start_time
    assert path_again == path
    assert duration < 5, f"Second call to ensure_shard took too long: {duration:.2f}s"

    print(progress_log[-1].file_progress)

    assert len(progress_log) > 0
    assert progress_log[-1].status == "complete"
    assert progress_log[-1].completed_files == 6
    assert progress_log[-1].total_files == 6
    assert progress_log[-1].downloaded_bytes == sum(file_size for _, file_size in expected_files_and_sizes)
    assert progress_log[-1].total_bytes == sum(file_size for _, file_size in expected_files_and_sizes)
