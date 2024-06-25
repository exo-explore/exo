from typing import Optional
import numpy as np
from abc import ABC, abstractmethod
from inference.shard import Shard
from topology.topology import Topology

class Node(ABC):
    @abstractmethod
    async def start(self, wait_for_peers: int = 0) -> None:
        pass

    @abstractmethod
    async def stop(self) -> None:
        pass

    @abstractmethod
    async def process_tensor(self, shard: Shard, tensor: np.ndarray) -> None:
        pass

    @abstractmethod
    async def process_prompt(self, shard: Shard, prompt: str) -> None:
        pass

    @abstractmethod
    async def reset_shard(self, shard: Shard) -> None:
        pass

    async def collect_topology(self, max_depth: int = 2) -> Topology:
        pass
