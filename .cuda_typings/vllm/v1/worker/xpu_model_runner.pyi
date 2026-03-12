import torch
from _typeshed import Incomplete
from vllm.config import VllmConfig as VllmConfig
from vllm.logger import init_logger as init_logger
from vllm.utils.torch_utils import supports_xpu_graph as supports_xpu_graph
from vllm.v1.worker.gpu.model_runner import GPUModelRunner as GPUModelRunnerV2
from vllm.v1.worker.gpu_model_runner import GPUModelRunner as GPUModelRunner

logger: Incomplete

class XPUModelRunner(GPUModelRunner):
    cascade_attn_enabled: bool
    def __init__(self, vllm_config: VllmConfig, device: torch.device) -> None: ...

class XPUModelRunnerV2(GPUModelRunnerV2):
    def __init__(self, vllm_config: VllmConfig, device: torch.device) -> None: ...
