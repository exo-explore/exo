import numpy as np

from typing import Tuple, Optional, Callable, Coroutine, Any
from abc import ABC, abstractmethod
from .shard import Shard
from exo.inference.hf_helpers import HFRepoProgressEvent

class InferenceEngine(ABC):
  @abstractmethod
  async def infer_prompt(self, request_id: str, shard: Shard, prompt: str, image_str: Optional[str] = None, inference_state: Optional[str] = None) -> (np.ndarray, str, bool):
    pass

  @abstractmethod
  async def infer_tensor(self, request_id: str, shard: Shard, input_data: np.ndarray, inference_state: Optional[str] = None) -> Tuple[np.ndarray, str, bool]:
    pass

  @abstractmethod
  def set_progress_callback(self, progress_callback: Callable[[HFRepoProgressEvent], Coroutine[Any, Any, None]]):
    pass
