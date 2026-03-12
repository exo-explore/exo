import dataclasses
import torch
from _typeshed import Incomplete
from collections.abc import Generator
from torch import nn as nn
from vllm.config import ModelConfig as ModelConfig
from vllm.config.load import LoadConfig as LoadConfig
from vllm.logger import init_logger as init_logger
from vllm.model_executor.layers.quantization.torchao import (
    torchao_version_at_least as torchao_version_at_least,
)
from vllm.model_executor.model_loader.base_loader import (
    BaseModelLoader as BaseModelLoader,
)
from vllm.model_executor.model_loader.weight_utils import (
    download_safetensors_index_file_from_hf as download_safetensors_index_file_from_hf,
    download_weights_from_hf as download_weights_from_hf,
    fastsafetensors_weights_iterator as fastsafetensors_weights_iterator,
    filter_duplicate_safetensors_files as filter_duplicate_safetensors_files,
    filter_files_not_needed_for_inference as filter_files_not_needed_for_inference,
    get_quant_config as get_quant_config,
    maybe_download_from_modelscope as maybe_download_from_modelscope,
    multi_thread_pt_weights_iterator as multi_thread_pt_weights_iterator,
    multi_thread_safetensors_weights_iterator as multi_thread_safetensors_weights_iterator,
    np_cache_weights_iterator as np_cache_weights_iterator,
    pt_weights_iterator as pt_weights_iterator,
    safetensors_weights_iterator as safetensors_weights_iterator,
)
from vllm.tracing import instrument as instrument
from vllm.transformers_utils.repo_utils import (
    list_filtered_repo_files as list_filtered_repo_files,
)

logger: Incomplete

class DefaultModelLoader(BaseModelLoader):
    DEFAULT_NUM_THREADS: int
    @dataclasses.dataclass
    class Source:
        model_or_path: str
        revision: str | None
        prefix: str = ...
        fall_back_to_pt: bool = ...
        allow_patterns_overrides: list[str] | None = ...

    counter_before_loading_weights: float
    counter_after_loading_weights: float
    def __init__(self, load_config: LoadConfig) -> None: ...
    def get_all_weights(
        self, model_config: ModelConfig, model: nn.Module
    ) -> Generator[tuple[str, torch.Tensor], None, None]: ...
    def download_model(self, model_config: ModelConfig) -> None: ...
    def load_weights(self, model: nn.Module, model_config: ModelConfig) -> None: ...
