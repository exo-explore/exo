import asyncio
import traceback
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
from exo.inference.shard import Shard
from exo.download.shard_download import ShardDownloader
from exo.download.download_progress import RepoProgressEvent
from exo.download.hf.hf_helpers import (
    download_repo_files, RepoProgressEvent, get_weight_map, 
    get_allow_patterns, get_repo_root, fetch_file_list, 
    get_local_snapshot_dir, get_file_download_percentage,
    filter_repo_objects
)
from exo.helpers import AsyncCallbackSystem, DEBUG
from exo.models import model_cards, get_repo
import aiohttp
from aiofiles import os as aios


class HFShardDownloader(ShardDownloader):
  def __init__(self, quick_check: bool = False, max_parallel_downloads: int = 4):
    self.quick_check = quick_check
    self.max_parallel_downloads = max_parallel_downloads
    self.active_downloads: Dict[Shard, asyncio.Task] = {}
    self.completed_downloads: Dict[Shard, Path] = {}
    self._on_progress = AsyncCallbackSystem[str, Tuple[Shard, RepoProgressEvent]]()
    self.current_shard: Optional[Shard] = None
    self.current_repo_id: Optional[str] = None
    self.revision: str = "main"

  async def ensure_shard(self, shard: Shard, inference_engine_name: str) -> Path:
    self.current_shard = shard
    self.current_repo_id = get_repo(shard.model_id, inference_engine_name)
    repo_name = get_repo(shard.model_id, inference_engine_name)
    
    # First check if we already have this shard downloaded
    if shard in self.completed_downloads:
      return self.completed_downloads[shard]

    # If a download is already in progress for this shard, wait for it
    if shard in self.active_downloads:
      if DEBUG >= 2: print(f"Download already in progress for {shard}. Waiting for it to complete.")
      try:
        path = await self.active_downloads[shard]
        self.completed_downloads[shard] = path
        return path
      except asyncio.CancelledError:
        # If the task was cancelled, remove it and continue to new download
        if DEBUG >= 2: print(f"Previous download for {shard} was cancelled. Starting new download.")
        self.active_downloads.pop(shard)

    # If quick_check is True, look for existing snapshots
    if self.quick_check:
      repo_root = get_repo_root(repo_name)
      snapshots_dir = repo_root/"snapshots"
      if snapshots_dir.exists():
        visible_dirs = [d for d in snapshots_dir.iterdir() if not d.name.startswith('.')]
        if visible_dirs:
          most_recent_dir = max(visible_dirs, key=lambda x: x.stat().st_mtime)
          
          # Verify that all required files are present
          try:
            weight_map = await get_weight_map(repo_name)
            if weight_map:
              allow_patterns = get_allow_patterns(weight_map, shard)
              async with aiohttp.ClientSession() as session:
                file_list = await fetch_file_list(session, repo_name, self.revision)
                required_files = list(filter_repo_objects(file_list, allow_patterns=allow_patterns, key=lambda x: x["path"]))
                
                # Check if all required files exist and have the correct size
                all_files_present = True
                for file_info in required_files:
                  file_path = most_recent_dir/file_info["path"]
                  if not await aios.path.exists(file_path):
                    if DEBUG >= 2: print(f"Missing required file: {file_path}")
                    all_files_present = False
                    break
                  file_size = (await aios.stat(file_path)).st_size
                  if file_size != file_info["size"]:
                    if DEBUG >= 2: print(f"File size mismatch for {file_path}: expected {file_info['size']}, got {file_size}")
                    all_files_present = False
                    break
                
                if all_files_present:
                  self.completed_downloads[shard] = most_recent_dir
                  return most_recent_dir
                elif DEBUG >= 2:
                  print("Found snapshot directory but it was incomplete or corrupted. Starting fresh download.")
          except Exception as e:
            if DEBUG >= 2: print(f"Error verifying snapshot: {e}")

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

    # Start new download
    download_task = asyncio.create_task(self._download_shard(shard, repo_name))
    self.active_downloads[shard] = download_task
    try:
      path = await download_task
      self.completed_downloads[shard] = path
      return path
    finally:
      # Ensure the task is removed even if an exception occurs
      if DEBUG >= 2: print(f"Removing download task for {shard}: {shard in self.active_downloads}")
      if shard in self.active_downloads:
        self.active_downloads.pop(shard)

  async def _download_shard(self, shard: Shard, repo_name: str) -> Path:
    async def wrapped_progress_callback(event: RepoProgressEvent):
      self._on_progress.trigger_all(shard, event)

    weight_map = await get_weight_map(repo_name)
    allow_patterns = get_allow_patterns(weight_map, shard)

    return await download_repo_files(repo_name, progress_callback=wrapped_progress_callback, allow_patterns=allow_patterns, max_parallel_downloads=self.max_parallel_downloads)

  @property
  def on_progress(self) -> AsyncCallbackSystem[str, Tuple[Shard, RepoProgressEvent]]:
    return self._on_progress

  async def get_shard_download_status(self) -> Optional[Dict[str, Union[float, int]]]:
    if not self.current_shard or not self.current_repo_id:
      if DEBUG >= 2:
        print(f"No current shard or repo_id set: {self.current_shard=} {self.current_repo_id=}")
      return None

    try:
      # If no snapshot directory exists, return None - no need to check remote files
      snapshot_dir = await get_local_snapshot_dir(self.current_repo_id, self.revision)
      if not snapshot_dir:
        if DEBUG >= 2:
          print(f"No snapshot directory found for {self.current_repo_id}")
        return None

      # Get the weight map to know what files we need
      weight_map = await get_weight_map(self.current_repo_id, self.revision)
      if not weight_map:
        if DEBUG >= 2:
          print(f"No weight map found for {self.current_repo_id}")
        return None

      # Get all files needed for this shard
      patterns = get_allow_patterns(weight_map, self.current_shard)

      # Check download status for all relevant files
      status = {}
      total_bytes = 0
      downloaded_bytes = 0

      async with aiohttp.ClientSession() as session:
        file_list = await fetch_file_list(session, self.current_repo_id, self.revision)
        relevant_files = list(
            filter_repo_objects(
                file_list, allow_patterns=patterns, key=lambda x: x["path"]))

        for file in relevant_files:
          file_size = file["size"]
          total_bytes += file_size

          percentage = await get_file_download_percentage(
              session,
              self.current_repo_id,
              self.revision,
              file["path"],
              snapshot_dir,
          )
          status[file["path"]] = percentage
          downloaded_bytes += (file_size * (percentage / 100))

        # Add overall progress weighted by file size
        if total_bytes > 0:
          status["overall"] = (downloaded_bytes / total_bytes) * 100
        else:
          status["overall"] = 0
          
        # Add total size in bytes
        status["total_size"] = total_bytes
        if status["overall"] != 100:
          status["total_downloaded"] = downloaded_bytes
        

        if DEBUG >= 2:
          print(f"Download calculation for {self.current_repo_id}:")
          print(f"Total bytes: {total_bytes}")
          print(f"Downloaded bytes: {downloaded_bytes}")
          for file in relevant_files:
            print(f"File {file['path']}: size={file['size']}, percentage={status[file['path']]}")

      return status

    except Exception as e:
      if DEBUG >= 2:
        print(f"Error getting shard download status: {e}")
        traceback.print_exc()
      return None
