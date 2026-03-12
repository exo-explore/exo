from ..hasher import MultiModalHasher as MultiModalHasher
from ..inputs import MultiModalHashes as MultiModalHashes
from ..parse import (
    MultiModalDataItems as MultiModalDataItems,
    MultiModalUUIDItems as MultiModalUUIDItems,
)
from collections.abc import Mapping
from dataclasses import dataclass, field

@dataclass
class ProcessorInputs:
    prompt: str | list[int]
    mm_data_items: MultiModalDataItems
    mm_uuid_items: MultiModalUUIDItems | None = ...
    hf_processor_mm_kwargs: Mapping[str, object] = field(default_factory=dict)
    tokenization_kwargs: Mapping[str, object] = field(default_factory=dict)
    def get_mm_hashes(self, model_id: str) -> MultiModalHashes: ...
