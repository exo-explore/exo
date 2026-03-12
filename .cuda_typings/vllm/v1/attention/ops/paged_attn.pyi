import torch
from vllm.platforms import current_platform as current_platform

class PagedAttention:
    @staticmethod
    def split_kv_cache(
        kv_cache: torch.Tensor, num_kv_heads: int, head_size: int
    ) -> tuple[torch.Tensor, torch.Tensor]: ...
    @staticmethod
    def write_to_paged_cache(
        key: torch.Tensor,
        value: torch.Tensor,
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
        slot_mapping: torch.Tensor,
        kv_cache_dtype: str,
        k_scale: torch.Tensor,
        v_scale: torch.Tensor,
    ) -> None: ...
