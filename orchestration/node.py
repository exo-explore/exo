from typing import Optional
import numpy as np
from abc import ABC, abstractmethod

class Node(ABC):
    @abstractmethod
    def start(self) -> None:
        pass

    @abstractmethod
    def stop(self) -> None:
        pass

    @abstractmethod
    def process_tensor(self, tensor: np.ndarray, target: Optional[str] = None) -> None:
        pass

    @abstractmethod
    def process_prompt(self, prompt: str, target: Optional[str] = None) -> None:
        pass

    @abstractmethod
    def reset_shard(self, shard_id: str) -> None:
        pass
