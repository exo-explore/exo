import torch
from typing import Any, Literal, TypeAlias
from typing_extensions import NotRequired, TypedDict
from vllm.multimodal.inputs import (
    MultiModalDataDict as MultiModalDataDict,
    MultiModalEncDecInputs as MultiModalEncDecInputs,
    MultiModalInputs as MultiModalInputs,
    MultiModalUUIDDict as MultiModalUUIDDict,
)

class _PromptOptions(TypedDict):
    multi_modal_data: NotRequired[MultiModalDataDict | None]
    mm_processor_kwargs: NotRequired[dict[str, Any] | None]
    multi_modal_uuids: NotRequired[MultiModalUUIDDict]
    cache_salt: NotRequired[str]

class TextPrompt(_PromptOptions):
    prompt: str

class TokensPrompt(_PromptOptions):
    prompt_token_ids: list[int]
    prompt: NotRequired[str]
    token_type_ids: NotRequired[list[int]]

class EmbedsPrompt(_PromptOptions):
    prompt_embeds: torch.Tensor
    prompt: NotRequired[str]

DecoderOnlyPrompt: TypeAlias = (
    str | TextPrompt | list[int] | TokensPrompt | EmbedsPrompt
)
EncoderPrompt: TypeAlias = str | TextPrompt | list[int] | TokensPrompt
DecoderPrompt: TypeAlias = str | TextPrompt | list[int] | TokensPrompt

class ExplicitEncoderDecoderPrompt(TypedDict):
    encoder_prompt: EncoderPrompt
    decoder_prompt: DecoderPrompt | None

EncoderDecoderPrompt: TypeAlias = EncoderPrompt | ExplicitEncoderDecoderPrompt
SingletonPrompt: TypeAlias = DecoderOnlyPrompt | EncoderPrompt | DecoderPrompt
PromptType: TypeAlias = DecoderOnlyPrompt | EncoderDecoderPrompt

class DataPrompt(_PromptOptions):
    data: Any
    data_format: str

class _InputOptions(TypedDict):
    arrival_time: NotRequired[float]
    cache_salt: NotRequired[str]

class TokenInputs(_InputOptions):
    type: Literal["token"]
    prompt_token_ids: list[int]
    prompt: NotRequired[str]

def token_inputs(
    prompt_token_ids: list[int],
    *,
    prompt: str | None = None,
    cache_salt: str | None = None,
) -> TokenInputs: ...

class EmbedsInputs(_InputOptions):
    type: Literal["embeds"]
    prompt_embeds: torch.Tensor
    prompt: NotRequired[str]

def embeds_inputs(
    prompt_embeds: torch.Tensor,
    *,
    prompt: str | None = None,
    cache_salt: str | None = None,
) -> EmbedsInputs: ...

DecoderOnlyInputs: TypeAlias = TokenInputs | EmbedsInputs | MultiModalInputs
EncoderInputs: TypeAlias = TokenInputs | MultiModalEncDecInputs
DecoderInputs: TypeAlias = TokenInputs | MultiModalInputs

class EncoderDecoderInputs(TypedDict):
    type: Literal["enc_dec"]
    encoder_prompt: EncoderInputs
    decoder_prompt: DecoderInputs
    arrival_time: NotRequired[float]

ProcessorInputs: TypeAlias = DecoderOnlyInputs | EncoderDecoderInputs
SingletonInputs: TypeAlias = DecoderOnlyInputs | MultiModalEncDecInputs

def build_enc_dec_inputs(
    encoder_inputs: SingletonInputs,
    decoder_inputs: SingletonInputs | None,
    decoder_start_token_id: int,
) -> EncoderDecoderInputs: ...
