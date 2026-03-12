import outlines_core as oc
import torch
from _typeshed import Incomplete
from dataclasses import dataclass, field
from vllm.sampling_params import SamplingParams as SamplingParams
from vllm.utils.import_utils import LazyLoader as LazyLoader
from vllm.v1.structured_output.backend_types import (
    StructuredOutputBackend as StructuredOutputBackend,
    StructuredOutputGrammar as StructuredOutputGrammar,
    StructuredOutputOptions as StructuredOutputOptions,
)
from vllm.v1.structured_output.utils import (
    OutlinesVocabulary as OutlinesVocabulary,
    get_outlines_cache as get_outlines_cache,
    get_outlines_vocabulary as get_outlines_vocabulary,
)

sre_parse: Incomplete
sre_constants: Incomplete

@dataclass
class OutlinesBackend(StructuredOutputBackend):
    vocabulary = ...
    cache = ...
    def __post_init__(self) -> None: ...
    def compile_grammar(
        self, request_type: StructuredOutputOptions, grammar_spec: str
    ) -> StructuredOutputGrammar: ...
    def allocate_token_bitmask(self, max_num_seqs: int) -> torch.Tensor: ...
    def destroy(self) -> None: ...

@dataclass
class OutlinesGrammar(StructuredOutputGrammar):
    vocab_size: int
    guide: oc.Guide = field(hash=False)
    num_processed_tokens: int = field(
        default_factory=Incomplete, repr=False, hash=False, init=False
    )
    def accept_tokens(self, request_id: str, tokens: list[int]) -> bool: ...
    def rollback(self, num_tokens: int) -> None: ...
    def validate_tokens(self, tokens: list[int]) -> list[int]: ...
    def fill_bitmask(self, bitmask: torch.Tensor, idx: int) -> None: ...
    def is_terminated(self) -> bool: ...
    def reset(self) -> None: ...

def validate_structured_output_request_outlines(params: SamplingParams): ...
def validate_regex_is_buildable(pattern: str) -> None: ...
