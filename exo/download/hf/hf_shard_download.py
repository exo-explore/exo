import asyncio
import traceback
from pathlib import Path
from typing import Dict, List, Tuple
from exo.inference.shard import Shard
from exo.download.shard_download import ShardDownloader
from exo.download.download_progress import RepoProgressEvent
from exo.download.hf.hf_helpers import download_repo_files, RepoProgressEvent, get_repo_root, get_weight_map, extract_layer_num
from exo.helpers import AsyncCallbackSystem, DEBUG

class HFShardDownloader(ShardDownloader):
    def __init__(self):
        self.active_downloads: List[Tuple[Shard, asyncio.Task]] = []
        self._on_progress = AsyncCallbackSystem[str, Tuple[Shard, RepoProgressEvent]]()

    async def ensure_shard(self, shard: Shard) -> Path:
        # If a download on this shard is already in progress, keep that one
        for active_shard, task in self.active_downloads:
            if active_shard == shard:
                return await task

        # Cancel any downloads for this model_id on a different shard
        to_remove = [(active_shard, task) for active_shard, task in self.active_downloads if active_shard.model_id == shard.model_id]
        for active_shard, task in to_remove:
            if DEBUG >= 2: print(f"Cancelling download for {active_shard} (replacing with {shard})")
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass  # This is expected when cancelling a task
            except Exception as e:
                if DEBUG >= 2: print(f"Error in cancelling download {active_shard}: {e}")
                traceback.print_exc()
        if DEBUG >= 2: print(f"Removing cancelled downloads: {to_remove}")
        self.active_downloads = [(active_shard, task) for active_shard, task in self.active_downloads if active_shard.model_id != shard.model_id]

        # Start new download
        download_task = asyncio.create_task(self._download_shard(shard))
        self.active_downloads.append((shard, download_task))
        try:
            return await download_task
        finally:
            # Ensure the task is removed even if an exception occurs
            if (shard, download_task) in self.active_downloads:
                self.active_downloads.remove((shard, download_task))

    async def _download_shard(self, shard: Shard) -> Path:
        async def wrapped_progress_callback(event: RepoProgressEvent):
            self._on_progress.trigger_all(shard, event)

        weight_map = await get_weight_map(shard.model_id)
        allow_patterns = self._get_allow_patterns(weight_map, shard.start_layer, shard.end_layer)

        return await download_repo_files(
            repo_id=shard.model_id,
            progress_callback=wrapped_progress_callback,
            allow_patterns=allow_patterns
        )

    @staticmethod
    def _get_allow_patterns(weight_map: Dict[str, str], start_layer: int, end_layer: int) -> List[str]:
        default_patterns = [
            "*.json",
            "*.py",
            "tokenizer.model",
            "*.tiktoken",
            "*.txt",
        ]
        shard_specific_patterns = []
        if weight_map:
            for tensor_name, filename in weight_map.items():
                layer_num = extract_layer_num(tensor_name)
                if layer_num is not None and start_layer <= layer_num <= end_layer:
                    shard_specific_patterns.append(filename)
        else:
            shard_specific_patterns = ["*.safetensors"]
        return list(set(default_patterns + shard_specific_patterns))  # Remove duplicates

    @property
    def on_progress(self) -> AsyncCallbackSystem[str, Tuple[Shard, RepoProgressEvent]]:
        return self._on_progress
