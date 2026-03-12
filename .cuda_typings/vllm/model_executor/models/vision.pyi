import abc
import torch
from _typeshed import Incomplete
from abc import ABC, abstractmethod
from collections.abc import Callable
from transformers import PretrainedConfig
from typing import Final, Generic, Literal, Protocol, TypeAlias
from vllm.config import (
    MultiModalConfig as MultiModalConfig,
    VllmConfig as VllmConfig,
    get_current_vllm_config as get_current_vllm_config,
)
from vllm.distributed import (
    get_tensor_model_parallel_rank as get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size as get_tensor_model_parallel_world_size,
    tensor_model_parallel_all_gather as tensor_model_parallel_all_gather,
)
from vllm.logger import init_logger as init_logger
from vllm.platforms import current_platform as current_platform
from vllm.v1.attention.backends.registry import (
    AttentionBackendEnum as AttentionBackendEnum,
)

logger: Incomplete

class _RootConfig(Protocol[_C]):
    vision_config: _C

class VisionEncoderInfo(ABC, Generic[_C], metaclass=abc.ABCMeta):
    hf_config: Incomplete
    vision_config: Incomplete
    def __init__(self, hf_config: _RootConfig[_C]) -> None: ...
    @abstractmethod
    def get_num_image_tokens(self, *, image_width: int, image_height: int) -> int: ...
    @abstractmethod
    def get_image_size(self) -> int: ...
    @abstractmethod
    def get_patch_size(self) -> int: ...
    @abstractmethod
    def get_patch_grid_length(self) -> int: ...

class VisionLanguageConfig(Protocol):
    vision_config: Final[PretrainedConfig]

def get_vision_encoder_info(hf_config: VisionLanguageConfig) -> VisionEncoderInfo: ...
def get_vit_attn_backend(
    head_size: int, dtype: torch.dtype
) -> AttentionBackendEnum: ...
def is_vit_use_data_parallel(): ...

VisionFeatureSelectStrategyStr: Incomplete
VisionFeatureSelectStrategy: TypeAlias

def get_num_selected_vision_tokens(
    num_vision_tokens: int, strategy: VisionFeatureSelectStrategy | str
) -> int: ...
def resolve_visual_encoder_outputs(
    encoder_outputs: torch.Tensor | list[torch.Tensor],
    post_layer_norm: torch.nn.LayerNorm | None,
    *,
    select_layers: list[int] | None = None,
    max_possible_layers: int | None = None,
    last_hs_proc: Callable[[torch.Tensor], torch.Tensor] | None = None,
    feature_select_strategy: VisionFeatureSelectStrategy | None = None,
) -> torch.Tensor: ...
def run_dp_sharded_vision_model(
    image_input: torch.Tensor, vision_model: torch.nn.Module
) -> torch.Tensor: ...
def get_load_balance_assignment(
    sizes: list[int], num_gpus: int = 2
) -> tuple[list[int], list[int], list[int]]: ...
def run_dp_sharded_mrope_vision_model(
    vision_model: torch.nn.Module,
    pixel_values: torch.Tensor,
    grid_thw_list: list[list[int]],
    *,
    rope_type: Literal["rope_3d", "rope_2d"],
) -> tuple[torch.Tensor, ...]: ...
def get_llm_pos_ids_for_vision(
    start_idx: int,
    vision_idx: int,
    spatial_merge_size: int,
    t_index: list[int],
    grid_hs: torch.Tensor,
    grid_ws: torch.Tensor,
) -> torch.Tensor: ...
