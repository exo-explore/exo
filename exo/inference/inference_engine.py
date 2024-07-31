import numpy as np

from typing import Tuple, Optional, Callable
from abc import ABC, abstractmethod
from .shard import Shard


class InferenceEngine(ABC):
  @abstractmethod
  async def infer_prompt(self, request_id: str, shard: Shard, prompt: str, image_str: Optional[str] = None, inference_state: Optional[str] = None) -> (np.ndarray, str, bool):
    pass

  @abstractmethod
  async def infer_tensor(self, request_id: str, shard: Shard, input_data: np.ndarray, inference_state: Optional[str] = None) -> Tuple[np.ndarray, str, bool]:
    pass

  @abstractmethod
  def set_on_download_progress(self, on_download_progress: Callable[[int, int], None]):
    pass
