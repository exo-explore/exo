from _typeshed import Incomplete
from tpu_inference.platforms import TpuPlatform as TpuInferencePlatform
from vllm.logger import init_logger as init_logger

logger: Incomplete
TpuPlatform = TpuInferencePlatform
USE_TPU_INFERENCE: bool
