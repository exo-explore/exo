from typing import Optional, Tuple, List
import numpy as np
from abc import ABC, abstractmethod
from exo.helpers import AsyncCallbackSystem
from exo.inference.shard import Shard
from exo.topology.topology import Topology


class Node(ABC):
  @abstractmethod
  async def start(self, wait_for_peers: int = 0) -> None:
    pass

  @abstractmethod
  async def stop(self) -> None:
    pass

  @abstractmethod
  async def process_prompt(self, shard: Shard, prompt: str, image_str: Optional[str] = None, request_id: Optional[str] = None, inference_state: Optional[str] = None) -> Optional[np.ndarray]:
    pass

  @abstractmethod
  async def process_tensor(self, shard: Shard, tensor: np.ndarray, request_id: Optional[str] = None, inference_state: Optional[str] = None) -> Optional[np.ndarray]:
    pass

  @abstractmethod
  async def get_inference_result(self, request_id: str) -> Tuple[Optional[np.ndarray], bool]:
    pass

  @abstractmethod
  async def collect_topology(self, visited: set[str] = set(), max_depth: int = 2) -> Topology:
    pass

  @property
  @abstractmethod
  def current_topology(self) -> Topology:
    pass

  @property
  @abstractmethod
  def on_token(self) -> AsyncCallbackSystem[str, Tuple[str, List[int], bool]]:
    pass

  @property
  @abstractmethod
  def on_opaque_status(self) -> AsyncCallbackSystem[str, Tuple[str, str]]:
    pass
