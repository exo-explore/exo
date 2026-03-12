import abc
from _typeshed import Incomplete
from abc import ABC, abstractmethod
from collections.abc import Set as Set
from dataclasses import dataclass, field
from vllm.logger import init_logger as init_logger
from vllm.lora.request import LoRARequest as LoRARequest

logger: Incomplete

class LoRAResolver(ABC, metaclass=abc.ABCMeta):
    @abstractmethod
    async def resolve_lora(
        self, base_model_name: str, lora_name: str
    ) -> LoRARequest | None: ...

@dataclass
class _LoRAResolverRegistry:
    resolvers: dict[str, LoRAResolver] = field(default_factory=dict)
    def get_supported_resolvers(self) -> Set[str]: ...
    def register_resolver(self, resolver_name: str, resolver: LoRAResolver) -> None: ...
    def get_resolver(self, resolver_name: str) -> LoRAResolver: ...

LoRAResolverRegistry: Incomplete
