from .cache import (
    BaseMultiModalProcessorCache as BaseMultiModalProcessorCache,
    BaseMultiModalReceiverCache as BaseMultiModalReceiverCache,
    MultiModalProcessorOnlyCache as MultiModalProcessorOnlyCache,
    MultiModalProcessorSenderCache as MultiModalProcessorSenderCache,
    MultiModalReceiverCache as MultiModalReceiverCache,
    ShmObjectStoreReceiverCache as ShmObjectStoreReceiverCache,
    ShmObjectStoreSenderCache as ShmObjectStoreSenderCache,
)
from .inputs import MultiModalInputs as MultiModalInputs
from .processing import (
    BaseDummyInputsBuilder as BaseDummyInputsBuilder,
    BaseMultiModalProcessor as BaseMultiModalProcessor,
    BaseProcessingInfo as BaseProcessingInfo,
    InputProcessingContext as InputProcessingContext,
    TimingContext as TimingContext,
)
from _typeshed import Incomplete
from collections.abc import Mapping
from dataclasses import dataclass
from multiprocessing.synchronize import Lock as LockType
from typing import Generic, Protocol, TypeVar
from vllm.config import (
    ModelConfig as ModelConfig,
    ObservabilityConfig as ObservabilityConfig,
    VllmConfig as VllmConfig,
)
from vllm.logger import init_logger as init_logger
from vllm.model_executor.models.interfaces import (
    SupportsMultiModal as SupportsMultiModal,
)
from vllm.tokenizers import (
    TokenizerLike as TokenizerLike,
    cached_tokenizer_from_config as cached_tokenizer_from_config,
)

logger: Incomplete
N = TypeVar("N", bound=type["SupportsMultiModal"])

class ProcessingInfoFactory(Protocol[_I_co]):
    def __call__(self, ctx: InputProcessingContext) -> _I_co: ...

class DummyInputsBuilderFactory(Protocol[_I]):
    def __call__(self, info: _I) -> BaseDummyInputsBuilder[_I]: ...

class MultiModalProcessorFactory(Protocol[_I]):
    def __call__(
        self,
        info: _I,
        dummy_inputs: BaseDummyInputsBuilder[_I],
        *,
        cache: BaseMultiModalProcessorCache | None = None,
    ) -> BaseMultiModalProcessor[_I]: ...

@dataclass(frozen=True)
class _ProcessorFactories(Generic[_I]):
    info: ProcessingInfoFactory[_I]
    processor: MultiModalProcessorFactory[_I]
    dummy_inputs: DummyInputsBuilderFactory[_I]
    def build_processor(
        self,
        ctx: InputProcessingContext,
        *,
        cache: BaseMultiModalProcessorCache | None = None,
    ): ...

class MultiModalRegistry:
    def supports_multimodal_inputs(self, model_config: ModelConfig) -> bool: ...
    def register_processor(
        self,
        processor: MultiModalProcessorFactory[_I],
        *,
        info: ProcessingInfoFactory[_I],
        dummy_inputs: DummyInputsBuilderFactory[_I],
    ): ...
    def create_processor(
        self,
        model_config: ModelConfig,
        *,
        tokenizer: TokenizerLike | None = None,
        cache: BaseMultiModalProcessorCache | None = None,
    ) -> BaseMultiModalProcessor[BaseProcessingInfo]: ...
    def get_dummy_mm_inputs(
        self,
        model_config: ModelConfig,
        mm_counts: Mapping[str, int],
        *,
        cache: BaseMultiModalProcessorCache | None = None,
        processor: BaseMultiModalProcessor | None = None,
    ) -> MultiModalInputs: ...
    def processor_cache_from_config(
        self, vllm_config: VllmConfig
    ) -> BaseMultiModalProcessorCache | None: ...
    def processor_only_cache_from_config(
        self, vllm_config: VllmConfig
    ) -> MultiModalProcessorOnlyCache | None: ...
    def engine_receiver_cache_from_config(
        self, vllm_config: VllmConfig
    ) -> BaseMultiModalReceiverCache | None: ...
    def worker_receiver_cache_from_config(
        self, vllm_config: VllmConfig, shared_worker_lock: LockType
    ) -> BaseMultiModalReceiverCache | None: ...

class MultiModalTimingRegistry:
    def __init__(self, observability_config: ObservabilityConfig | None) -> None: ...
    def get(self, request_id: str) -> TimingContext: ...
    def stat(self) -> dict[str, dict[str, float]]: ...
