from abc import ABC, abstractmethod
from typing import List
from .peer_handle import PeerHandle


class Discovery(ABC):
  @abstractmethod
  async def start(self) -> None:
    pass

  @abstractmethod
  async def stop(self) -> None:
    pass

  @abstractmethod
  async def discover_peers(self, wait_for_peers: int = 0) -> List[PeerHandle]:
    pass
