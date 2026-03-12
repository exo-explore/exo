import torch
from _typeshed import Incomplete
from contextlib import contextmanager
from torch import nn
from vllm.config import VllmConfig as VllmConfig
from vllm.config.utils import getattr_iter as getattr_iter
from vllm.logger import init_logger as init_logger
from vllm.model_executor.layers.conv import (
    Conv2dLayer as Conv2dLayer,
    Conv3dLayer as Conv3dLayer,
)
from vllm.model_executor.layers.layernorm import (
    GemmaRMSNorm as GemmaRMSNorm,
    RMSNorm as RMSNorm,
)
from vllm.model_executor.layers.linear import (
    ColumnParallelLinear as ColumnParallelLinear,
    ReplicatedLinear as ReplicatedLinear,
    RowParallelLinear as RowParallelLinear,
)
from vllm.model_executor.layers.quantization import (
    QuantizationConfig as QuantizationConfig,
)
from vllm.transformers_utils.config import (
    is_rope_parameters_nested as is_rope_parameters_nested,
)

logger: Incomplete

@contextmanager
def init_on_device_without_buffers(device: torch.device): ...

Style: Incomplete

def replace_linear_class(
    linear: nn.Linear,
    style: Style = "replicate",
    quant_config: QuantizationConfig | None = None,
    *,
    prefix: str = "",
) -> ColumnParallelLinear | RowParallelLinear | ReplicatedLinear: ...

TorchConv: Incomplete
VllmConv = Conv2dLayer | Conv3dLayer

def replace_conv_class(conv: TorchConv) -> VllmConv | TorchConv: ...
def replace_rms_norm_class(rms_norm: nn.Module, hidden_size: int) -> RMSNorm: ...
def log_replacement(name: str, old_module: nn.Module, new_module: nn.Module): ...
def get_feature_request_tip(model: str, trust_remote_code: bool) -> str: ...
def can_enable_torch_compile(vllm_config: VllmConfig) -> bool: ...
