from typing import Literal
from vllm.config.utils import config as config

@config
class WeightTransferConfig:
    backend: Literal["nccl", "ipc"] = ...
