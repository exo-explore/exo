import torch
from _typeshed import Incomplete
from pydantic import ConfigDict, SkipValidation as SkipValidation
from vllm.config.utils import config as config
from vllm.utils.hashing import safe_hash as safe_hash

Device: Incomplete

@config(config=ConfigDict(arbitrary_types_allowed=True))
class DeviceConfig:
    device: SkipValidation[Device | torch.device | None] = ...
    device_type: str = ...
    def compute_hash(self) -> str: ...
    def __post_init__(self) -> None: ...
