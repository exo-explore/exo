import abc
import torch
from dataclasses import dataclass
from typing import Any
from vllm.config import VllmConfig as VllmConfig
from vllm.v1.attention.backend import (
    AttentionBackend as AttentionBackend,
    CommonAttentionMetadata as CommonAttentionMetadata,
)
from vllm.v1.attention.backends.mamba_attn import (
    BaseMambaAttentionMetadata as BaseMambaAttentionMetadata,
    BaseMambaAttentionMetadataBuilder as BaseMambaAttentionMetadataBuilder,
)
from vllm.v1.kv_cache_interface import AttentionSpec as AttentionSpec

def compute_varlen_chunk_metadata(
    query_start_loc: torch.Tensor, chunk_size: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]: ...

class Mamba2AttentionBackend(AttentionBackend, metaclass=abc.ABCMeta):
    @staticmethod
    def get_name() -> str: ...
    @staticmethod
    def get_builder_cls() -> type["Mamba2AttentionMetadataBuilder"]: ...

@dataclass
class Mamba2AttentionMetadata(BaseMambaAttentionMetadata):
    prep_initial_states: bool = ...
    chunk_size: int = ...
    seq_idx_p: torch.Tensor | None = ...

class Mamba2AttentionMetadataBuilder(
    BaseMambaAttentionMetadataBuilder[Mamba2AttentionMetadata]
):
    metadata_cls = Mamba2AttentionMetadata
    chunk_size: int
    def __init__(
        self,
        kv_cache_spec: AttentionSpec,
        layer_names: list[str],
        vllm_config: VllmConfig,
        device: torch.device,
    ) -> None: ...
    def build(
        self,
        common_prefix_len: int,
        common_attn_metadata: CommonAttentionMetadata,
        fast_build: bool = False,
        **kwargs: Any,
    ) -> Mamba2AttentionMetadata: ...
