import torch
from collections.abc import Sequence
from typing import NamedTuple, TypeAlias, TypedDict, overload
from vllm.config import ModelConfig as ModelConfig
from vllm.entrypoints.chat_utils import (
    ChatCompletionMessageParam as ChatCompletionMessageParam,
)
from vllm.inputs import (
    EmbedsPrompt as EmbedsPrompt,
    ExplicitEncoderDecoderPrompt as ExplicitEncoderDecoderPrompt,
    ProcessorInputs as ProcessorInputs,
    PromptType as PromptType,
    SingletonPrompt as SingletonPrompt,
    TextPrompt as TextPrompt,
    TokensPrompt as TokensPrompt,
)
from vllm.utils import (
    length_from_prompt_token_ids_or_embeds as length_from_prompt_token_ids_or_embeds,
)
from vllm.utils.collection_utils import is_list_of as is_list_of

@overload
def prompt_to_seq(
    prompt_or_prompts: SingletonPrompt | bytes | Sequence[SingletonPrompt | bytes],
) -> Sequence[SingletonPrompt]: ...
@overload
def prompt_to_seq(
    prompt_or_prompts: ExplicitEncoderDecoderPrompt
    | Sequence[ExplicitEncoderDecoderPrompt],
) -> Sequence[ExplicitEncoderDecoderPrompt]: ...
@overload
def prompt_to_seq(
    prompt_or_prompts: PromptType | Sequence[PromptType],
) -> Sequence[PromptType]: ...
def conversation_to_seq(
    conversation_or_conversations: list["ChatCompletionMessageParam"]
    | Sequence[list["ChatCompletionMessageParam"]],
) -> Sequence[list["ChatCompletionMessageParam"]]: ...

DecoderOnlyDictPrompt: TypeAlias = TextPrompt | TokensPrompt | EmbedsPrompt
EncoderDictPrompt: TypeAlias = TextPrompt | TokensPrompt
DecoderDictPrompt: TypeAlias = TextPrompt | TokensPrompt

class EncoderDecoderDictPrompt(TypedDict):
    encoder_prompt: EncoderDictPrompt
    decoder_prompt: DecoderDictPrompt | None

SingletonDictPrompt: TypeAlias = (
    DecoderOnlyDictPrompt | EncoderDictPrompt | DecoderDictPrompt
)
DictPrompt: TypeAlias = DecoderOnlyDictPrompt | EncoderDecoderDictPrompt

def parse_dec_only_prompt(prompt: PromptType | object) -> DecoderOnlyDictPrompt: ...
def parse_enc_dec_prompt(prompt: PromptType | object) -> EncoderDecoderDictPrompt: ...
def parse_model_prompt(model_config: ModelConfig, prompt: object): ...

class PromptComponents(NamedTuple):
    text: str | None = ...
    token_ids: list[int] | None = ...
    embeds: torch.Tensor | None = ...

def extract_target_prompt(model_config: ModelConfig, prompt: object): ...
def extract_prompt_components(
    model_config: ModelConfig, prompt: PromptType | ProcessorInputs
) -> PromptComponents: ...
def extract_prompt_len(
    model_config: ModelConfig, prompt: PromptType | ProcessorInputs
): ...
