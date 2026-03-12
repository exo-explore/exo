import torch
import torch.nn as nn
from _typeshed import Incomplete
from vllm.config import VllmConfig as VllmConfig
from vllm.logger import init_logger as init_logger
from vllm.model_executor.model_loader import get_model as get_model
from vllm.tracing import instrument as instrument
from vllm.v1.utils import CpuGpuBuffer as CpuGpuBuffer
from vllm.v1.worker.gpu_model_runner import GPUModelRunner as GPUModelRunner

logger: Incomplete

class CPUModelRunner(GPUModelRunner):
    use_cuda_graph: bool
    cascade_attn_enabled: bool
    def __init__(self, vllm_config: VllmConfig, device: torch.device) -> None: ...
    model: Incomplete
    def load_model(self, load_dummy_weights: bool = False) -> None: ...
    def get_model(self) -> nn.Module: ...
    def warming_up_model(self) -> None: ...
    def get_dp_padding(self, num_tokens: int) -> tuple[int, torch.Tensor | None]: ...
