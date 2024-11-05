import asyncio
import traceback
from pathlib import Path
from typing import Dict, List, Tuple
from exo.inference.shard import Shard
from exo.download.shard_download import ShardDownloader
from exo.download.download_progress import RepoProgressEvent
from exo.download.hf.hf_helpers import download_repo_files, RepoProgressEvent, get_weight_map, get_allow_patterns, get_repo_root
from exo.helpers import AsyncCallbackSystem, DEBUG


class HFShardDownloader(ShardDownloader):
  def __init__(self, quick_check: bool = False, max_parallel_downloads: int = 4):
    self.quick_check = quick_check
    self.max_parallel_downloads = max_parallel_downloads
    self.active_downloads: Dict[Shard, asyncio.Task] = {}
    self.completed_downloads: Dict[Shard, Path] = {}
    self._on_progress = AsyncCallbackSystem[str, Tuple[Shard, RepoProgressEvent]]()

  async def ensure_shard(self, shard: Shard) -> Path:
    """Ensure a shard is downloaded and return its path. Downloads are protected from cancellation."""
    if shard in self.completed_downloads:
      if DEBUG >= 2: print(f"Using completed download for {shard}")
      return self.completed_downloads[shard]

    if self.quick_check:
      repo_root = get_repo_root(shard.model_id)
      snapshots_dir = repo_root/"snapshots"
      if snapshots_dir.exists():
        visible_dirs = [d for d in snapshots_dir.iterdir() if not d.name.startswith('.')]
        if visible_dirs:
          most_recent_dir = max(visible_dirs, key=lambda x: x.stat().st_mtime)
          return most_recent_dir

    # Check for active download
    if shard in self.active_downloads:
      if DEBUG >= 2: print(f"Using existing download for {shard}")
      try:
        return await self.active_downloads[shard]
      except asyncio.CancelledError:
        if DEBUG >= 2: print(f"Ignoring cancellation for existing download of {shard}")
        return await self.active_downloads[shard]

    # Cancel any downloads for this model_id on a different shard
    existing_active_shards = [active_shard for active_shard in self.active_downloads.keys() if active_shard.model_id == shard.model_id]
    for active_shard in existing_active_shards:
      if DEBUG >= 2: print(f"Cancelling download for {active_shard} (replacing with {shard})")
      task = self.active_downloads[active_shard]
      task.cancel()
      try:
        await task
      except asyncio.CancelledError:
        pass  # This is expected when cancelling a task
      except Exception as e:
        if DEBUG >= 2: print(f"Error in cancelling download {active_shard}: {e}")
        traceback.print_exc()
    self.active_downloads = {active_shard: task for active_shard, task in self.active_downloads.items() if active_shard.model_id != shard.model_id}

    # Start new protected download
    event = asyncio.Event()
    result = None
    error = None
    
    async def protected_download():
      nonlocal result, error
      try:
        if DEBUG >= 2: print(f"Starting protected download for {shard}")
        result = await self._download_shard(shard)
        self.completed_downloads[shard] = result
        if DEBUG >= 2: print(f"Download completed for {shard}: {result}")
        return result
      except Exception as e:
        if DEBUG >= 2: print(f"Error in download for {shard}: {e}")
        error = e
        raise
      finally:
        event.set()

    download_task = asyncio.create_task(protected_download())
    self.active_downloads[shard] = download_task

    try:
      if DEBUG >= 2: print(f"Waiting for download to complete for {shard}")
      try:
        return await download_task
      except asyncio.CancelledError:
        if DEBUG >= 2: print(f"Ignoring cancellation and waiting for download to complete for {shard}")
        await event.wait()
        if error:
          raise error
        return result
    finally:
      if DEBUG >= 2: print(f"Cleaning up download task for {shard}")
      if shard in self.active_downloads and self.active_downloads[shard] is download_task:
        self.active_downloads.pop(shard)

  async def _download_shard(self, shard: Shard) -> Path:
    async def wrapped_progress_callback(event: RepoProgressEvent):
      self._on_progress.trigger_all(shard, event)

    weight_map = await get_weight_map(shard.model_id)
    allow_patterns = get_allow_patterns(weight_map, shard)

    return await download_repo_files(repo_id=shard.model_id, progress_callback=wrapped_progress_callback, allow_patterns=allow_patterns, max_parallel_downloads=self.max_parallel_downloads)

  @property
  def on_progress(self) -> AsyncCallbackSystem[str, Tuple[Shard, RepoProgressEvent]]:
    return self._on_progress
