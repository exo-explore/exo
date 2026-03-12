from _typeshed import Incomplete
from enum import Enum
from vllm.logger import init_logger as init_logger
from vllm.model_executor.layers.fused_moe.config import FusedMoEConfig as FusedMoEConfig

logger: Incomplete

class MxFp8MoeBackend(Enum):
    FLASHINFER_TRTLLM = "FLASHINFER_TRTLLM"

def select_mxfp8_moe_backend(config: FusedMoEConfig) -> MxFp8MoeBackend: ...
