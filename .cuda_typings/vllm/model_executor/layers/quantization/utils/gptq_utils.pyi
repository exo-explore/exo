import torch
from ..gptq import GPTQConfig as GPTQConfig
from ..gptq_marlin import GPTQMarlinConfig as GPTQMarlinConfig
from collections.abc import Mapping
from vllm.model_executor.layers.linear import (
    LinearBase as LinearBase,
    UnquantizedLinearMethod as UnquantizedLinearMethod,
)
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead as ParallelLMHead,
    UnquantizedEmbeddingMethod as UnquantizedEmbeddingMethod,
)

def override_config(config: GPTQConfig | GPTQMarlinConfig, prefix: str): ...
def get_dynamic_override(
    config: GPTQConfig | GPTQMarlinConfig,
    layer_name: str,
    key: str | None = None,
    default_value: int | bool | None = None,
) -> dict | int | bool | None: ...
def is_layer_gptq_quantized(
    prefix: str,
    quantized_layers: list[str],
    fused_mapping: Mapping[str, list[str]] = ...,
) -> bool: ...
def get_linear_quant_method(
    config: GPTQConfig | GPTQMarlinConfig,
    layer: torch.nn.Module,
    prefix: str,
    linear_method_cls: type,
): ...
