import torch
from vllm.model_executor.kernels.linear.mixed_precision import (
    MPLinearKernel as MPLinearKernel,
    MPLinearLayerConfig as MPLinearLayerConfig,
)
from vllm.model_executor.kernels.linear.mixed_precision.allspark import (
    AllSparkLinearKernel as AllSparkLinearKernel,
)
from vllm.model_executor.kernels.linear.mixed_precision.conch import (
    ConchLinearKernel as ConchLinearKernel,
)
from vllm.model_executor.kernels.linear.mixed_precision.cpu import (
    CPUWNA16LinearKernel as CPUWNA16LinearKernel,
)
from vllm.model_executor.kernels.linear.mixed_precision.cutlass import (
    CutlassW4A8LinearKernel as CutlassW4A8LinearKernel,
)
from vllm.model_executor.kernels.linear.mixed_precision.dynamic_4bit import (
    Dynamic4bitLinearKernel as Dynamic4bitLinearKernel,
)
from vllm.model_executor.kernels.linear.mixed_precision.exllama import (
    ExllamaLinearKernel as ExllamaLinearKernel,
)
from vllm.model_executor.kernels.linear.mixed_precision.machete import (
    MacheteLinearKernel as MacheteLinearKernel,
)
from vllm.model_executor.kernels.linear.mixed_precision.marlin import (
    MarlinLinearKernel as MarlinLinearKernel,
)
from vllm.model_executor.kernels.linear.mixed_precision.xpu import (
    XPUwNa16LinearKernel as XPUwNa16LinearKernel,
)
from vllm.model_executor.kernels.linear.scaled_mm import (
    FP8ScaledMMLinearKernel as FP8ScaledMMLinearKernel,
    FP8ScaledMMLinearLayerConfig as FP8ScaledMMLinearLayerConfig,
    Int8ScaledMMLinearKernel as Int8ScaledMMLinearKernel,
    Int8ScaledMMLinearLayerConfig as Int8ScaledMMLinearLayerConfig,
    ScaledMMLinearKernel as ScaledMMLinearKernel,
    ScaledMMLinearLayerConfig as ScaledMMLinearLayerConfig,
)
from vllm.model_executor.kernels.linear.scaled_mm.aiter import (
    AiterInt8ScaledMMLinearKernel as AiterInt8ScaledMMLinearKernel,
)
from vllm.model_executor.kernels.linear.scaled_mm.cpu import (
    CPUInt8ScaledMMLinearKernel as CPUInt8ScaledMMLinearKernel,
)
from vllm.model_executor.kernels.linear.scaled_mm.cutlass import (
    CutlassFP8ScaledMMLinearKernel as CutlassFP8ScaledMMLinearKernel,
    CutlassInt8ScaledMMLinearKernel as CutlassInt8ScaledMMLinearKernel,
)
from vllm.model_executor.kernels.linear.scaled_mm.flashinfer import (
    FlashInferFP8ScaledMMLinearKernel as FlashInferFP8ScaledMMLinearKernel,
)
from vllm.model_executor.kernels.linear.scaled_mm.pytorch import (
    ChannelWiseTorchFP8ScaledMMLinearKernel as ChannelWiseTorchFP8ScaledMMLinearKernel,
    PerTensorTorchFP8ScaledMMLinearKernel as PerTensorTorchFP8ScaledMMLinearKernel,
    RowWiseTorchFP8ScaledMMLinearKernel as RowWiseTorchFP8ScaledMMLinearKernel,
)
from vllm.model_executor.kernels.linear.scaled_mm.rocm import (
    ROCmFP8ScaledMMLinearKernel as ROCmFP8ScaledMMLinearKernel,
)
from vllm.model_executor.kernels.linear.scaled_mm.triton import (
    TritonInt8ScaledMMLinearKernel as TritonInt8ScaledMMLinearKernel,
)
from vllm.model_executor.layers.quantization.utils.quant_utils import QuantKey

__all__ = [
    "init_fp8_linear_kernel",
    "init_int8_linear_kernel",
    "choose_mp_linear_kernel",
    "FP8ScaledMMLinearKernel",
    "Int8ScaledMMLinearKernel",
    "ScaledMMLinearKernel",
    "FP8ScaledMMLinearLayerConfig",
    "Int8ScaledMMLinearLayerConfig",
    "ScaledMMLinearLayerConfig",
    "AiterInt8ScaledMMLinearKernel",
    "CPUInt8ScaledMMLinearKernel",
    "CutlassFP8ScaledMMLinearKernel",
    "CutlassInt8ScaledMMLinearKernel",
    "FlashInferFP8ScaledMMLinearKernel",
    "ChannelWiseTorchFP8ScaledMMLinearKernel",
    "PerTensorTorchFP8ScaledMMLinearKernel",
    "RowWiseTorchFP8ScaledMMLinearKernel",
    "ROCmFP8ScaledMMLinearKernel",
    "TritonInt8ScaledMMLinearKernel",
    "MPLinearKernel",
    "MPLinearLayerConfig",
    "AllSparkLinearKernel",
    "ConchLinearKernel",
    "CPUWNA16LinearKernel",
    "CutlassW4A8LinearKernel",
    "Dynamic4bitLinearKernel",
    "ExllamaLinearKernel",
    "MacheteLinearKernel",
    "MarlinLinearKernel",
    "XPUwNa16LinearKernel",
]

def init_fp8_linear_kernel(
    activation_quant_key: QuantKey,
    weight_quant_key: QuantKey,
    out_dtype: torch.dtype,
    force_kernel: type[FP8ScaledMMLinearKernel] | None = None,
    module_name: str | None = None,
) -> FP8ScaledMMLinearKernel: ...
def init_int8_linear_kernel(
    is_channelwise: bool,
    is_static_input_scheme: bool,
    input_symmetric: bool,
    module_name: str,
) -> Int8ScaledMMLinearKernel: ...
def choose_mp_linear_kernel(
    config: MPLinearLayerConfig, compute_capability: int | None = None
) -> type[MPLinearKernel]: ...
