from _typeshed import Incomplete
from collections.abc import Callable as Callable
from vllm.config.utils import config as config
from vllm.utils.hashing import safe_hash as safe_hash

MoEBackend: Incomplete

@config
class KernelConfig:
    enable_flashinfer_autotune: bool = ...
    moe_backend: MoEBackend = ...
    def compute_hash(self) -> str: ...
