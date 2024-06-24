from abc import ABC, abstractmethod
from typing import Optional
import numpy as np
from inference.shard import Shard

class PeerHandle(ABC):
    def id(self) -> str:
        pass

    @abstractmethod
    async def connect(self) -> None:
        pass

    @abstractmethod
    async def disconnect(self) -> None:
        pass

    @abstractmethod
    async def send_prompt(self, shard: Shard, prompt: str) -> Optional[np.array]:
        pass

    @abstractmethod
    async def send_tensor(self, shard: Shard, tensor: np.array) -> Optional[np.array]:
        pass

    @abstractmethod
    async def reset_shard(self, shard: Shard) -> None:
        pass
