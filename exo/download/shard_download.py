from abc import ABC, abstractmethod
from typing import Optional, Tuple, Dict, AsyncIterator
from pathlib import Path
from exo.inference.shard import Shard
from exo.download.download_progress import RepoProgressEvent
from exo.helpers import AsyncCallbackSystem


class ShardDownloader(ABC):
  @abstractmethod
  async def ensure_shard(self, shard: Shard, inference_engine_name: str) -> Path:
    """
        Ensures that the shard is downloaded.
        Does not allow multiple overlapping downloads at once.
        If you try to download a Shard which overlaps a Shard that is already being downloaded,
        the download will be cancelled and a new download will start.

        Args:
            shard (Shard): The shard to download.
            inference_engine_name (str): The inference engine used on the node hosting the shard
        """
    pass

  @property
  @abstractmethod
  def on_progress(self) -> AsyncCallbackSystem[str, Tuple[Shard, RepoProgressEvent]]:
    pass

  @abstractmethod
  async def get_shard_download_status(self, inference_engine_name: str) -> AsyncIterator[tuple[Path, RepoProgressEvent]]:
    """Get the download status of shards.
    
    Returns:
        Optional[Dict[str, float]]: A dictionary mapping shard IDs to their download percentage (0-100),
        or None if status cannot be determined
    """
    pass


class NoopShardDownloader(ShardDownloader):
  async def ensure_shard(self, shard: Shard, inference_engine_name: str) -> Path:
    return Path("/tmp/noop_shard")

  @property
  def on_progress(self) -> AsyncCallbackSystem[str, Tuple[Shard, RepoProgressEvent]]:
    return AsyncCallbackSystem()

  async def get_shard_download_status(self, inference_engine_name: str) -> AsyncIterator[tuple[Path, RepoProgressEvent]]:
    if False: yield
