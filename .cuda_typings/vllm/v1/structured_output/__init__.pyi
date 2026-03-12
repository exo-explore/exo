import numpy as np
import numpy.typing as npt
from _typeshed import Incomplete
from vllm.config import VllmConfig as VllmConfig
from vllm.logger import init_logger as init_logger
from vllm.reasoning import (
    ReasoningParser as ReasoningParser,
    ReasoningParserManager as ReasoningParserManager,
)
from vllm.tokenizers import cached_tokenizer_from_config as cached_tokenizer_from_config
from vllm.utils.import_utils import LazyLoader as LazyLoader
from vllm.v1.request import Request as Request
from vllm.v1.structured_output.backend_guidance import (
    GuidanceBackend as GuidanceBackend,
)
from vllm.v1.structured_output.backend_types import (
    StructuredOutputBackend as StructuredOutputBackend,
    StructuredOutputGrammar as StructuredOutputGrammar,
)
from vllm.v1.structured_output.backend_xgrammar import (
    XgrammarBackend as XgrammarBackend,
)

logger: Incomplete

class StructuredOutputManager:
    backend: StructuredOutputBackend | None
    reasoner: ReasoningParser | None
    vllm_config: Incomplete
    fill_bitmask_parallel_threshold: int
    fill_bitmask_parallel_batch_size: int
    executor_for_fillmask: Incomplete
    executor: Incomplete
    tokenizer: Incomplete
    enable_in_reasoning: Incomplete
    def __init__(self, vllm_config: VllmConfig) -> None: ...
    def grammar_init(self, request: Request) -> None: ...
    def grammar_bitmask(
        self,
        requests: dict[str, "Request"],
        structured_output_request_ids: list[str],
        scheduled_spec_decode_tokens: dict[str, list[int]],
    ) -> npt.NDArray[np.int32] | None: ...
    def should_fill_bitmask(self, request: Request) -> bool: ...
    def should_advance(self, request: Request) -> bool: ...
    def clear_backend(self) -> None: ...
