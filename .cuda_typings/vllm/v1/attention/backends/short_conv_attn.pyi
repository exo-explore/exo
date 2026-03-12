import abc
from dataclasses import dataclass
from vllm.v1.attention.backend import AttentionBackend as AttentionBackend
from vllm.v1.attention.backends.mamba_attn import (
    BaseMambaAttentionMetadata as BaseMambaAttentionMetadata,
    BaseMambaAttentionMetadataBuilder as BaseMambaAttentionMetadataBuilder,
)

class ShortConvAttentionBackend(AttentionBackend, metaclass=abc.ABCMeta):
    @staticmethod
    def get_name() -> str: ...
    @staticmethod
    def get_builder_cls() -> type["ShortConvAttentionMetadataBuilder"]: ...

@dataclass
class ShortConvAttentionMetadata(BaseMambaAttentionMetadata): ...

class ShortConvAttentionMetadataBuilder(
    BaseMambaAttentionMetadataBuilder[ShortConvAttentionMetadata]
):
    metadata_cls = ShortConvAttentionMetadata
