import torch
import xgrammar as xgr
from _typeshed import Incomplete
from dataclasses import dataclass, field
from typing import Any
from vllm.logger import init_logger as init_logger
from vllm.sampling_params import SamplingParams as SamplingParams
from vllm.utils.import_utils import LazyLoader as LazyLoader
from vllm.utils.mistral import is_mistral_tokenizer as is_mistral_tokenizer
from vllm.v1.structured_output.backend_types import (
    StructuredOutputBackend as StructuredOutputBackend,
    StructuredOutputGrammar as StructuredOutputGrammar,
    StructuredOutputOptions as StructuredOutputOptions,
)
from vllm.v1.structured_output.utils import (
    choice_as_grammar as choice_as_grammar,
    convert_lark_to_ebnf as convert_lark_to_ebnf,
    grammar_is_likely_lark as grammar_is_likely_lark,
)

logger: Incomplete

@dataclass
class XgrammarBackend(StructuredOutputBackend):
    disable_any_whitespace = ...
    vocab_size = ...
    compiler = ...
    num_speculative_tokens = ...
    def __post_init__(self) -> None: ...
    def compile_grammar(
        self, request_type: StructuredOutputOptions, grammar_spec: str
    ) -> StructuredOutputGrammar: ...
    def allocate_token_bitmask(self, max_num_seqs: int): ...
    def destroy(self) -> None: ...

@dataclass
class XgrammarGrammar(StructuredOutputGrammar):
    vocab_size: int
    matcher: xgr.GrammarMatcher = field(hash=False)
    ctx: xgr.CompiledGrammar = field(hash=False)
    num_processed_tokens: int = field(
        default_factory=Incomplete, repr=False, hash=False, init=False
    )
    def accept_tokens(self, request_id: str, tokens: list[int]) -> bool: ...
    def validate_tokens(self, tokens: list[int]) -> list[int]: ...
    def rollback(self, num_tokens: int) -> None: ...
    def fill_bitmask(self, bitmask: torch.Tensor, idx: int) -> None: ...
    def is_terminated(self) -> bool: ...
    def reset(self) -> None: ...

STRING_SUPPORTED_FORMATS: Incomplete

def has_xgrammar_unsupported_json_features(schema: dict[str, Any]) -> bool: ...
def validate_xgrammar_grammar(sampling_params: SamplingParams) -> None: ...
