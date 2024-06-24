from typing import Optional
import numpy as np
from abc import ABC, abstractmethod
from inference.shard import Shard

class Node(ABC):
    @abstractmethod
    def start(self, wait_for_peers: int = 0) -> None:
        pass

    @abstractmethod
    def stop(self) -> None:
        pass

    @abstractmethod
    def process_tensor(self, shard: Shard, tensor: np.ndarray, target: Optional[str] = None) -> None:
        pass

    @abstractmethod
    def process_prompt(self, shard: Shard, prompt: str, target: Optional[str] = None) -> None:
        pass

    @abstractmethod
    def reset_shard(self, shard: Shard) -> None:
        pass
