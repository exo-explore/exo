from dataclasses import dataclass
from vllm.inputs import (
    EncoderDecoderInputs as EncoderDecoderInputs,
    TokenInputs as TokenInputs,
    token_inputs as token_inputs,
)
from vllm.inputs.data import DecoderInputs as DecoderInputs
from vllm.logprobs import Logprob as Logprob
from vllm.lora.request import LoRARequest as LoRARequest
from vllm.multimodal.inputs import (
    MultiModalInputs as MultiModalInputs,
    mm_inputs as mm_inputs,
)

@dataclass
class BeamSearchSequence:
    orig_prompt: TokenInputs | MultiModalInputs | EncoderDecoderInputs
    tokens: list[int]
    logprobs: list[dict[int, Logprob]]
    lora_request: LoRARequest | None = ...
    cum_logprob: float = ...
    text: str | None = ...
    finish_reason: str | None = ...
    stop_reason: int | str | None = ...
    def get_prompt(self): ...

@dataclass
class BeamSearchOutput:
    sequences: list[BeamSearchSequence]

class BeamSearchInstance:
    beams: list[BeamSearchSequence]
    completed: list[BeamSearchSequence]
    def __init__(
        self,
        prompt: TokenInputs | MultiModalInputs | EncoderDecoderInputs,
        lora_request: LoRARequest | None = None,
        logprobs: list[dict[int, Logprob]] | None = None,
        **kwargs,
    ) -> None: ...

def get_beam_search_score(
    tokens: list[int],
    cumulative_logprob: float,
    eos_token_id: int,
    length_penalty: float = 1.0,
) -> float: ...
def create_sort_beams_key_function(eos_token_id: int, length_penalty: float): ...
