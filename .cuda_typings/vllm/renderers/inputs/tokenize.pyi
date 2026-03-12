from typing import TypeAlias, TypedDict
from vllm.inputs import EmbedsPrompt as EmbedsPrompt, TokensPrompt as TokensPrompt

DecoderOnlyTokPrompt: TypeAlias = TokensPrompt | EmbedsPrompt
EncoderTokPrompt: TypeAlias = TokensPrompt
DecoderTokPrompt: TypeAlias = TokensPrompt

class EncoderDecoderTokPrompt(TypedDict):
    encoder_prompt: EncoderTokPrompt
    decoder_prompt: DecoderTokPrompt | None

SingletonTokPrompt: TypeAlias = (
    DecoderOnlyTokPrompt | EncoderTokPrompt | DecoderTokPrompt
)
TokPrompt: TypeAlias = DecoderOnlyTokPrompt | EncoderDecoderTokPrompt
