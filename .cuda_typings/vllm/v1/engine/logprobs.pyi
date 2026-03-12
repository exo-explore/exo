from _typeshed import Incomplete
from dataclasses import dataclass
from vllm.logger import init_logger as init_logger
from vllm.logprobs import (
    PromptLogprobs as PromptLogprobs,
    SampleLogprobs as SampleLogprobs,
    append_logprobs_for_next_position as append_logprobs_for_next_position,
    create_prompt_logprobs as create_prompt_logprobs,
    create_sample_logprobs as create_sample_logprobs,
)
from vllm.tokenizers.detokenizer_utils import (
    TokenizerLike as TokenizerLike,
    convert_ids_list_to_tokens as convert_ids_list_to_tokens,
)
from vllm.v1.engine import (
    EngineCoreOutput as EngineCoreOutput,
    EngineCoreRequest as EngineCoreRequest,
)
from vllm.v1.outputs import (
    LogprobsLists as LogprobsLists,
    LogprobsTensors as LogprobsTensors,
)

logger: Incomplete
NONES: Incomplete

@dataclass
class LogprobsProcessor:
    tokenizer: TokenizerLike | None
    logprobs: SampleLogprobs | None
    prompt_logprobs: PromptLogprobs | None
    cumulative_logprob: float | None
    num_logprobs: int | None
    num_prompt_logprobs: int | None
    @classmethod
    def from_new_request(
        cls, tokenizer: TokenizerLike | None, request: EngineCoreRequest
    ) -> LogprobsProcessor: ...
    def pop_prompt_logprobs(self) -> PromptLogprobs | None: ...
    def update_from_output(self, output: EngineCoreOutput) -> None: ...
