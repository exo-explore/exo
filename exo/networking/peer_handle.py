from abc import ABC, abstractmethod
from typing import Optional, Tuple, List
import numpy as np
from exo.inference.shard import Shard
from exo.topology.device_capabilities import DeviceCapabilities
from exo.topology.topology import Topology


class PeerHandle(ABC):
  @abstractmethod
  def id(self) -> str:
    pass

  @abstractmethod
  def addr(self) -> str:
    pass

  @abstractmethod
  def description(self) -> str:
    pass

  @abstractmethod
  def device_capabilities(self) -> DeviceCapabilities:
    pass

  @abstractmethod
  async def connect(self) -> None:
    pass

  @abstractmethod
  async def is_connected(self) -> bool:
    pass

  @abstractmethod
  async def disconnect(self) -> None:
    pass

  @abstractmethod
  async def health_check(self) -> bool:
    pass

  @abstractmethod
  async def send_prompt(self, shard: Shard, prompt: str, request_id: Optional[str] = None) -> Optional[np.array]:
    pass

  @abstractmethod
  async def send_tensor(self, shard: Shard, tensor: np.array, request_id: Optional[str] = None) -> Optional[np.array]:
    pass

  @abstractmethod
  async def send_result(self, request_id: str, result: List[int], is_finished: bool) -> None:
    pass

  @abstractmethod
  async def collect_topology(self, visited: set[str], max_depth: int) -> Topology:
    pass
