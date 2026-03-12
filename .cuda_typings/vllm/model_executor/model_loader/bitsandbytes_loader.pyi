from _typeshed import Incomplete
from collections.abc import Callable as Callable
from torch import nn as nn
from vllm.config import ModelConfig as ModelConfig
from vllm.config.load import LoadConfig as LoadConfig
from vllm.distributed import (
    get_tensor_model_parallel_rank as get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size as get_tensor_model_parallel_world_size,
)
from vllm.logger import init_logger as init_logger
from vllm.lora.utils import is_moe_model as is_moe_model
from vllm.model_executor.layers.fused_moe import FusedMoE as FusedMoE
from vllm.model_executor.layers.linear import (
    LinearBase as LinearBase,
    MergedColumnParallelLinear as MergedColumnParallelLinear,
    QKVParallelLinear as QKVParallelLinear,
    ReplicatedLinear as ReplicatedLinear,
    RowParallelLinear as RowParallelLinear,
)
from vllm.model_executor.model_loader.base_loader import (
    BaseModelLoader as BaseModelLoader,
)
from vllm.model_executor.model_loader.utils import ParamMapping as ParamMapping
from vllm.model_executor.model_loader.weight_utils import (
    download_safetensors_index_file_from_hf as download_safetensors_index_file_from_hf,
    download_weights_from_hf as download_weights_from_hf,
    filter_duplicate_safetensors_files as filter_duplicate_safetensors_files,
    filter_files_not_needed_for_inference as filter_files_not_needed_for_inference,
    pt_weights_iterator as pt_weights_iterator,
    safetensors_weights_iterator as safetensors_weights_iterator,
)
from vllm.model_executor.models import is_pooling_model as is_pooling_model
from vllm.model_executor.utils import (
    get_moe_expert_mapping as get_moe_expert_mapping,
    get_packed_modules_mapping as get_packed_modules_mapping,
    set_weight_attrs as set_weight_attrs,
)
from vllm.platforms import current_platform as current_platform
from vllm.utils.torch_utils import set_default_torch_dtype as set_default_torch_dtype

logger: Incomplete

class BitsAndBytesModelLoader(BaseModelLoader):
    possible_config_file_names: Incomplete
    unsharded_weights_modules: list[str]
    column_sharded_weights_modules: list[str]
    maybe_fused_weights_modules: dict[str, list[int]]
    target_modules: list[str]
    tp_disabled_modules: list[str]
    expert_params_mapping: list[tuple[str, str, int, str]]
    weight_mapper: Callable
    pre_quant: bool
    load_8bit: bool
    is_pool_model: bool
    def __init__(self, load_config: LoadConfig) -> None: ...
    def load_weights(self, model: nn.Module, model_config: ModelConfig) -> None: ...
    def download_model(self, model_config: ModelConfig) -> None: ...
