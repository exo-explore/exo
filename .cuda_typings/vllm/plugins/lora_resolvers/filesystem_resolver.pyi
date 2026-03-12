from _typeshed import Incomplete
from vllm.lora.request import LoRARequest as LoRARequest
from vllm.lora.resolver import (
    LoRAResolver as LoRAResolver,
    LoRAResolverRegistry as LoRAResolverRegistry,
)

class FilesystemResolver(LoRAResolver):
    lora_cache_dir: Incomplete
    def __init__(self, lora_cache_dir: str) -> None: ...
    async def resolve_lora(
        self, base_model_name: str, lora_name: str
    ) -> LoRARequest | None: ...

def register_filesystem_resolver() -> None: ...
