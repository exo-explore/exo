from _typeshed import Incomplete
from asyncio import Lock
from vllm.engine.protocol import EngineClient as EngineClient
from vllm.entrypoints.openai.engine.protocol import (
    ErrorResponse as ErrorResponse,
    ModelCard as ModelCard,
    ModelList as ModelList,
    ModelPermission as ModelPermission,
)
from vllm.entrypoints.openai.models.protocol import (
    BaseModelPath as BaseModelPath,
    LoRAModulePath as LoRAModulePath,
)
from vllm.entrypoints.serve.lora.protocol import (
    LoadLoRAAdapterRequest as LoadLoRAAdapterRequest,
    UnloadLoRAAdapterRequest as UnloadLoRAAdapterRequest,
)
from vllm.entrypoints.utils import create_error_response as create_error_response
from vllm.exceptions import LoRAAdapterNotFoundError as LoRAAdapterNotFoundError
from vllm.logger import init_logger as init_logger
from vllm.lora.request import LoRARequest as LoRARequest
from vllm.lora.resolver import (
    LoRAResolver as LoRAResolver,
    LoRAResolverRegistry as LoRAResolverRegistry,
)
from vllm.utils.counter import AtomicCounter as AtomicCounter

logger: Incomplete

class OpenAIServingModels:
    engine_client: Incomplete
    base_model_paths: Incomplete
    static_lora_modules: Incomplete
    lora_requests: dict[str, LoRARequest]
    lora_id_counter: Incomplete
    lora_resolvers: list[LoRAResolver]
    lora_resolver_lock: dict[str, Lock]
    model_config: Incomplete
    renderer: Incomplete
    io_processor: Incomplete
    input_processor: Incomplete
    def __init__(
        self,
        engine_client: EngineClient,
        base_model_paths: list[BaseModelPath],
        *,
        lora_modules: list[LoRAModulePath] | None = None,
    ) -> None: ...
    async def init_static_loras(self) -> None: ...
    def is_base_model(self, model_name) -> bool: ...
    def model_name(self, lora_request: LoRARequest | None = None) -> str: ...
    async def show_available_models(self) -> ModelList: ...
    async def load_lora_adapter(
        self, request: LoadLoRAAdapterRequest, base_model_name: str | None = None
    ) -> ErrorResponse | str: ...
    async def unload_lora_adapter(
        self, request: UnloadLoRAAdapterRequest
    ) -> ErrorResponse | str: ...
    async def resolve_lora(self, lora_name: str) -> LoRARequest | ErrorResponse: ...
