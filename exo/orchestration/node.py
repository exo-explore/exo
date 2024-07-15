from typing import Optional, Tuple
import numpy as np
from abc import ABC, abstractmethod
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
    async def process_prompt(self, shard: Shard, prompt: str) -> Optional[np.ndarray]:
        pass

    @abstractmethod
    async def process_tensor(self, shard: Shard, tensor: np.ndarray) -> Optional[np.ndarray]:
        pass

    @abstractmethod
    async def reset_shard(self, shard: Shard) -> None:
        pass

    @abstractmethod
    async def collect_topology(self, visited: set[str] = set(), max_depth: int = 2) -> Topology:
        pass

    @abstractmethod
    async def get_inference_result(self, request_id: str) -> Tuple[Optional[np.ndarray], bool]:
        pass

    @abstractmethod
    async def global_reset(self, visited: set[str], max_depth: int = 2) -> None:
        pass
