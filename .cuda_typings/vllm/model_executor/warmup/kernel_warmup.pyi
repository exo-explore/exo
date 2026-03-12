from _typeshed import Incomplete
from vllm.logger import init_logger as init_logger
from vllm.model_executor.warmup.deep_gemm_warmup import (
    deep_gemm_warmup as deep_gemm_warmup,
)
from vllm.platforms import current_platform as current_platform
from vllm.utils.deep_gemm import is_deep_gemm_supported as is_deep_gemm_supported
from vllm.utils.flashinfer import has_flashinfer as has_flashinfer
from vllm.v1.worker.gpu_model_runner import GPUModelRunner as GPUModelRunner
from vllm.v1.worker.gpu_worker import Worker as Worker

logger: Incomplete

def kernel_warmup(worker: Worker): ...
def flashinfer_autotune(runner: GPUModelRunner) -> None: ...
