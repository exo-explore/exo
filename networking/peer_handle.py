from abc import ABC, abstractmethod
from typing import Any

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
    async def send_prompt(self, prompt: str) -> None:
        pass

    @abstractmethod
    async def send_tensor(self, tensor: Any) -> None:
        pass

    @abstractmethod
    async def reset_shard(self, shard_id: str) -> None:
        pass
