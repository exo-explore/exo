from abc import ABC, abstractmethod
from typing import Optional, Tuple
from pathlib import Path
from exo.inference.shard import Shard
from exo.download.download_progress import RepoProgressEvent
from exo.helpers import AsyncCallbackSystem

class ShardDownloader(ABC):
    @abstractmethod
    async def ensure_shard(self, shard: Shard) -> Path:
        """
        Ensures that the shard is downloaded.
        Does not allow multiple overlapping downloads at once.
        If you try to download a Shard which overlaps a Shard that is already being downloaded,
        the download will be cancelled and a new download will start.

        Args:
            shard (Shard): The shard to download.
        """
        pass

    @property
    @abstractmethod
    def on_progress(self) -> AsyncCallbackSystem[str, Tuple[Shard, RepoProgressEvent]]:
        pass
