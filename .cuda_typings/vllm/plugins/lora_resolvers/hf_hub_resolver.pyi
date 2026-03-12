from _typeshed import Incomplete
from vllm.logger import init_logger as init_logger
from vllm.lora.request import LoRARequest as LoRARequest
from vllm.lora.resolver import LoRAResolverRegistry as LoRAResolverRegistry
from vllm.plugins.lora_resolvers.filesystem_resolver import (
    FilesystemResolver as FilesystemResolver,
)

logger: Incomplete

class HfHubResolver(FilesystemResolver):
    repo_list: list[str]
    adapter_dirs: dict[str, set[str]]
    def __init__(self, repo_list: list[str]) -> None: ...
    async def resolve_lora(
        self, base_model_name: str, lora_name: str
    ) -> LoRARequest | None: ...

def register_hf_hub_resolver() -> None: ...
