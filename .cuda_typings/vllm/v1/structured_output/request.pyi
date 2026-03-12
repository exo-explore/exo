import dataclasses
import functools
from concurrent.futures import Future
from vllm.sampling_params import (
    SamplingParams as SamplingParams,
    StructuredOutputsParams as StructuredOutputsParams,
)
from vllm.v1.structured_output.backend_types import (
    StructuredOutputGrammar as StructuredOutputGrammar,
    StructuredOutputKey as StructuredOutputKey,
    StructuredOutputOptions as StructuredOutputOptions,
)

@dataclasses.dataclass
class StructuredOutputRequest:
    params: StructuredOutputsParams
    reasoning_ended: bool | None = ...
    @staticmethod
    def from_sampling_params(
        sampling_params: SamplingParams | None,
    ) -> StructuredOutputRequest | None: ...
    @property
    def is_grammar_ready(self) -> bool: ...
    @property
    def grammar(self) -> StructuredOutputGrammar | None: ...
    @grammar.setter
    def grammar(
        self, grammar: StructuredOutputGrammar | Future[StructuredOutputGrammar]
    ) -> None: ...
    @functools.cached_property
    def structured_output_key(self) -> StructuredOutputKey: ...

def get_structured_output_key(
    params: StructuredOutputsParams,
) -> StructuredOutputKey: ...
