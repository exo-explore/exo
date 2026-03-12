from .data import (
    DecoderInputs as DecoderInputs,
    DecoderOnlyInputs as DecoderOnlyInputs,
    EmbedsInputs as EmbedsInputs,
    EmbedsPrompt as EmbedsPrompt,
    EncoderDecoderInputs as EncoderDecoderInputs,
    EncoderInputs as EncoderInputs,
    ProcessorInputs as ProcessorInputs,
    PromptType as PromptType,
    SingletonInputs as SingletonInputs,
    TextPrompt as TextPrompt,
    TokenInputs as TokenInputs,
    TokensPrompt as TokensPrompt,
    token_inputs as token_inputs,
)
from _typeshed import Incomplete
from typing import Any
from vllm.config import VllmConfig as VllmConfig
from vllm.inputs.data import build_enc_dec_inputs as build_enc_dec_inputs
from vllm.logger import init_logger as init_logger
from vllm.multimodal import (
    MULTIMODAL_REGISTRY as MULTIMODAL_REGISTRY,
    MultiModalRegistry as MultiModalRegistry,
)
from vllm.multimodal.inputs import (
    MultiModalDataDict as MultiModalDataDict,
    MultiModalInputs as MultiModalInputs,
    MultiModalUUIDDict as MultiModalUUIDDict,
)
from vllm.renderers import (
    BaseRenderer as BaseRenderer,
    renderer_from_config as renderer_from_config,
)
from vllm.renderers.inputs import (
    DecoderDictPrompt as DecoderDictPrompt,
    DecoderOnlyDictPrompt as DecoderOnlyDictPrompt,
    EncoderDecoderDictPrompt as EncoderDecoderDictPrompt,
    EncoderDictPrompt as EncoderDictPrompt,
    SingletonDictPrompt as SingletonDictPrompt,
)
from vllm.renderers.inputs.preprocess import (
    parse_dec_only_prompt as parse_dec_only_prompt,
    parse_enc_dec_prompt as parse_enc_dec_prompt,
)
from vllm.tokenizers import TokenizerLike as TokenizerLike

logger: Incomplete

class InputPreprocessor:
    model_config: Incomplete
    renderer: Incomplete
    mm_registry: Incomplete
    def __init__(
        self,
        vllm_config: VllmConfig,
        renderer: BaseRenderer | None = None,
        mm_registry: MultiModalRegistry = ...,
    ) -> None: ...
    @property
    def tokenizer(self) -> TokenizerLike | None: ...
    def get_tokenizer(self) -> TokenizerLike: ...
    def preprocess(
        self, prompt: PromptType, tokenization_kwargs: dict[str, Any] | None = None
    ) -> ProcessorInputs: ...
