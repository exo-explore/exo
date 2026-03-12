import abc
from dataclasses import dataclass
from typing import Any
from vllm.v1.attention.backend import (
    AttentionBackend as AttentionBackend,
    CommonAttentionMetadata as CommonAttentionMetadata,
)
from vllm.v1.attention.backends.mamba_attn import (
    BaseMambaAttentionMetadata as BaseMambaAttentionMetadata,
    BaseMambaAttentionMetadataBuilder as BaseMambaAttentionMetadataBuilder,
)

class Mamba1AttentionBackend(AttentionBackend, metaclass=abc.ABCMeta):
    @staticmethod
    def get_name() -> str: ...
    @staticmethod
    def get_builder_cls() -> type["Mamba1AttentionMetadataBuilder"]: ...

@dataclass
class Mamba1AttentionMetadata(BaseMambaAttentionMetadata): ...

class Mamba1AttentionMetadataBuilder(
    BaseMambaAttentionMetadataBuilder[Mamba1AttentionMetadata]
):
    metadata_cls = Mamba1AttentionMetadata
    def build(
        self,
        common_prefix_len: int,
        common_attn_metadata: CommonAttentionMetadata,
        fast_build: bool = False,
        **kwargs: Any,
    ) -> Mamba1AttentionMetadata: ...
