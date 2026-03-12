import torch
from dataclasses import dataclass
from typing import TypeAlias
from vllm.config.cache import MambaDType as MambaDType
from vllm.config.model import ModelDType as ModelDType
from vllm.distributed import divide as divide
from vllm.utils.torch_utils import (
    STR_DTYPE_TO_TORCH_DTYPE as STR_DTYPE_TO_TORCH_DTYPE,
    get_kv_cache_torch_dtype as get_kv_cache_torch_dtype,
)

class MambaStateDtypeCalculator:
    @classmethod
    def linear_attention_state_dtype(
        cls, model_dtype: ModelDType | torch.dtype, mamba_cache_dtype: MambaDType
    ) -> tuple[torch.dtype, ...]: ...
    @classmethod
    def mamba1_state_dtype(
        cls,
        model_dtype: ModelDType | torch.dtype,
        mamba_cache_dtype: MambaDType,
        mamba_ssm_cache_dtype: MambaDType,
    ) -> tuple[torch.dtype, ...]: ...
    @classmethod
    def mamba2_state_dtype(
        cls,
        model_dtype: ModelDType | torch.dtype,
        mamba_cache_dtype: MambaDType,
        mamba_ssm_cache_dtype: MambaDType,
    ) -> tuple[torch.dtype, ...]: ...
    @classmethod
    def short_conv_state_dtype(
        cls, model_dtype: ModelDType | torch.dtype, mamba_cache_dtype: MambaDType
    ) -> tuple[torch.dtype, ...]: ...
    @classmethod
    def gated_delta_net_state_dtype(
        cls,
        model_dtype: ModelDType | torch.dtype,
        mamba_cache_dtype: MambaDType,
        mamba_ssm_cache_dtype: MambaDType = "auto",
    ) -> tuple[torch.dtype, torch.dtype]: ...
    @classmethod
    def kda_state_dtype(
        cls, model_dtype: ModelDType | torch.dtype, mamba_cache_dtype: MambaDType
    ): ...

class MambaStateShapeCalculator:
    @classmethod
    def linear_attention_state_shape(
        cls, num_heads: int, tp_size: int, head_dim: int
    ) -> tuple[tuple[int, int, int], ...]: ...
    @classmethod
    def mamba1_state_shape(
        cls,
        tp_world_size: int,
        intermediate_size: int,
        state_size: int,
        conv_kernel: int,
    ) -> tuple[tuple[int, int], tuple[int, int]]: ...
    @classmethod
    def mamba2_state_shape(
        cls,
        tp_world_size: int,
        intermediate_size: int,
        n_groups: int,
        num_heads: int,
        head_dim: int,
        state_size: int,
        conv_kernel: int,
        num_spec: int = 0,
    ) -> tuple[tuple[int, int], tuple[int, int, int]]: ...
    @classmethod
    def short_conv_state_shape(
        cls, tp_world_size: int, intermediate_size: int, conv_kernel: int
    ) -> tuple[tuple[int, int]]: ...
    @classmethod
    def extra_groups_for_head_shards(cls, ngroups: int, tp_size: int): ...
    @classmethod
    def gated_delta_net_state_shape(
        cls,
        tp_world_size: int,
        num_k_heads: int,
        num_v_heads: int,
        head_k_dim: int,
        head_v_dim: int,
        conv_kernel_size: int,
        num_spec: int = 0,
    ): ...
    @classmethod
    def kda_state_shape(
        cls,
        tp_world_size: int,
        num_heads: int,
        head_dim: int,
        num_k_heads: int | None = None,
        head_k_dim: int | None = None,
        conv_kernel_size: int = 4,
        num_spec: int = 0,
    ) -> tuple[
        tuple[int, int], tuple[int, int], tuple[int, int], tuple[int, int, int]
    ]: ...

@dataclass
class MambaCopySpec:
    start_addr: int
    num_elements: int

MambaStateCopyFunc: TypeAlias

def get_conv_copy_spec(
    state: torch.Tensor,
    block_ids: list[int],
    cur_block_idx: int,
    num_accepted_tokens: int,
) -> MambaCopySpec: ...
def get_temporal_copy_spec(
    state: torch.Tensor,
    block_ids: list[int],
    cur_block_idx: int,
    num_accepted_tokens: int,
) -> MambaCopySpec: ...

class MambaStateCopyFuncCalculator:
    @classmethod
    def linear_attention_state_copy_func(cls): ...
    @classmethod
    def mamba1_state_copy_func(cls): ...
    @classmethod
    def mamba2_state_copy_func(cls): ...
    @classmethod
    def short_conv_state_copy_func(cls): ...
    @classmethod
    def gated_delta_net_state_copy_func(cls): ...
    @classmethod
    def kda_state_copy_func(cls): ...
