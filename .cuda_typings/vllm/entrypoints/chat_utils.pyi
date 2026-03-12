import abc
from PIL import Image
from _typeshed import Incomplete
from abc import ABC, abstractmethod
from collections.abc import Awaitable, Callable as Callable, Iterable
from dataclasses import dataclass
from functools import cached_property as cached_property
from openai.types.chat import (
    ChatCompletionContentPartInputAudioParam,
    ChatCompletionContentPartParam as OpenAIChatCompletionContentPartParam,
    ChatCompletionContentPartRefusalParam,
    ChatCompletionFunctionToolParam as ChatCompletionFunctionToolParam,
    ChatCompletionMessageParam as OpenAIChatCompletionMessageParam,
    ChatCompletionMessageToolCallParam as ChatCompletionMessageToolCallParam,
)
from openai.types.chat.chat_completion_content_part_input_audio_param import InputAudio
from openai_harmony import Message as OpenAIHarmonyMessage
from pathlib import Path
from pydantic import BaseModel
from typing import Any, Generic, Literal, TypeAlias
from typing_extensions import Required, TypedDict
from vllm import envs as envs
from vllm.config import ModelConfig as ModelConfig
from vllm.logger import init_logger as init_logger
from vllm.model_executor.models import SupportsMultiModal as SupportsMultiModal
from vllm.multimodal import (
    MULTIMODAL_REGISTRY as MULTIMODAL_REGISTRY,
    MultiModalDataDict as MultiModalDataDict,
    MultiModalUUIDDict as MultiModalUUIDDict,
)
from vllm.multimodal.inputs import (
    MultiModalBatchedField as MultiModalBatchedField,
    MultiModalFlatField as MultiModalFlatField,
    MultiModalSharedField as MultiModalSharedField,
    VisionChunk as VisionChunk,
    VisionChunkImage as VisionChunkImage,
    VisionChunkVideo as VisionChunkVideo,
)
from vllm.multimodal.media import (
    MEDIA_CONNECTOR_REGISTRY as MEDIA_CONNECTOR_REGISTRY,
    MediaConnector as MediaConnector,
)
from vllm.multimodal.processing import (
    BaseMultiModalProcessor as BaseMultiModalProcessor,
)
from vllm.utils import random_uuid as random_uuid
from vllm.utils.collection_utils import is_list_of as is_list_of
from vllm.utils.import_utils import LazyLoader as LazyLoader

logger: Incomplete

def __getattr__(name: str): ...

class ChatTemplateResolutionError(ValueError): ...

MODALITY_PLACEHOLDERS_MAP: Incomplete

class AudioURL(TypedDict, total=False):
    url: Required[str]

class ChatCompletionContentPartAudioParam(TypedDict, total=False):
    audio_url: Required[AudioURL]
    type: Required[Literal["audio_url"]]

class ChatCompletionContentPartImageEmbedsParam(TypedDict, total=False):
    image_embeds: str | dict[str, str] | None
    type: Required[Literal["image_embeds"]]
    uuid: str | None

class ChatCompletionContentPartAudioEmbedsParam(TypedDict, total=False):
    audio_embeds: str | dict[str, str] | None
    type: Required[Literal["audio_embeds"]]
    uuid: str | None

class VideoURL(TypedDict, total=False):
    url: Required[str]

class ChatCompletionContentPartVideoParam(TypedDict, total=False):
    video_url: Required[VideoURL]
    type: Required[Literal["video_url"]]

class PILImage(BaseModel):
    image_pil: Image.Image
    model_config: Incomplete

class CustomChatCompletionContentPILImageParam(TypedDict, total=False):
    image_pil: PILImage | None
    uuid: str | None

class CustomChatCompletionContentSimpleImageParam(TypedDict, total=False):
    image_url: str | None
    uuid: str | None

class CustomChatCompletionContentSimpleAudioParam(TypedDict, total=False):
    audio_url: str | None

class CustomChatCompletionContentSimpleVideoParam(TypedDict, total=False):
    video_url: str | None
    uuid: str | None

class CustomThinkCompletionContentParam(TypedDict, total=False):
    thinking: Required[str]
    closed: bool
    type: Required[Literal["thinking"]]

ChatCompletionContentPartParam: TypeAlias = (
    OpenAIChatCompletionContentPartParam
    | ChatCompletionContentPartAudioParam
    | ChatCompletionContentPartInputAudioParam
    | ChatCompletionContentPartVideoParam
    | ChatCompletionContentPartRefusalParam
    | CustomChatCompletionContentPILImageParam
    | CustomChatCompletionContentSimpleImageParam
    | ChatCompletionContentPartImageEmbedsParam
    | ChatCompletionContentPartAudioEmbedsParam
    | CustomChatCompletionContentSimpleAudioParam
    | CustomChatCompletionContentSimpleVideoParam
    | str
    | CustomThinkCompletionContentParam
)

class CustomChatCompletionMessageParam(TypedDict, total=False):
    role: Required[str]
    content: str | list[ChatCompletionContentPartParam]
    name: str
    tool_call_id: str | None
    tool_calls: Iterable[ChatCompletionMessageToolCallParam] | None
    reasoning: str | None
    tools: list[ChatCompletionFunctionToolParam] | None

ChatCompletionMessageParam: TypeAlias = (
    OpenAIChatCompletionMessageParam
    | CustomChatCompletionMessageParam
    | OpenAIHarmonyMessage
)

class ConversationMessage(TypedDict, total=False):
    role: Required[str]
    content: str | None | list[dict[str, str]]
    tool_call_id: str | None
    name: str | None
    tool_calls: Iterable[ChatCompletionMessageToolCallParam] | None
    reasoning: str | None
    reasoning_content: str | None
    tools: list[ChatCompletionFunctionToolParam] | None

ChatTemplateContentFormatOption: Incomplete
ChatTemplateContentFormat: Incomplete
ModalityStr: Incomplete

class _BatchedSingleItemField(MultiModalSharedField): ...

class BaseMultiModalItemTracker(ABC, Generic[_T], metaclass=abc.ABCMeta):
    def __init__(
        self,
        model_config: ModelConfig,
        media_io_kwargs: dict[str, dict[str, Any]] | None = None,
    ) -> None: ...
    @cached_property
    def use_unified_vision_chunk_modality(self) -> bool: ...
    @property
    def model_config(self) -> ModelConfig: ...
    @cached_property
    def model_cls(self) -> type[SupportsMultiModal]: ...
    @property
    def media_io_kwargs(self) -> dict[str, dict[str, Any]] | None: ...
    @property
    def allowed_local_media_path(self): ...
    @property
    def allowed_media_domains(self): ...
    @property
    def mm_registry(self): ...
    @cached_property
    def mm_processor(self): ...
    def add(self, modality: ModalityStr, item: _T) -> str | None: ...
    @abstractmethod
    def create_parser(
        self, mm_processor_kwargs: dict[str, Any] | None = None
    ) -> BaseMultiModalContentParser: ...

class MultiModalItemTracker(BaseMultiModalItemTracker[tuple[object, str | None]]):
    def resolve_items(
        self,
    ) -> tuple[MultiModalDataDict | None, MultiModalUUIDDict | None]: ...
    def create_parser(
        self, mm_processor_kwargs: dict[str, Any] | None = None
    ) -> BaseMultiModalContentParser: ...

class AsyncMultiModalItemTracker(
    BaseMultiModalItemTracker[Awaitable[tuple[object, str | None]]]
):
    async def resolve_items(
        self,
    ) -> tuple[MultiModalDataDict | None, MultiModalUUIDDict | None]: ...
    def create_parser(
        self, mm_processor_kwargs: dict[str, Any] | None = None
    ) -> BaseMultiModalContentParser: ...

class BaseMultiModalContentParser(ABC, metaclass=abc.ABCMeta):
    def __init__(self) -> None: ...
    def mm_placeholder_storage(self) -> dict[str, list]: ...
    @abstractmethod
    def parse_image(self, image_url: str | None, uuid: str | None = None) -> None: ...
    @abstractmethod
    def parse_image_embeds(
        self, image_embeds: str | dict[str, str] | None, uuid: str | None = None
    ) -> None: ...
    @abstractmethod
    def parse_image_pil(
        self, image_pil: Image.Image | None, uuid: str | None = None
    ) -> None: ...
    @abstractmethod
    def parse_audio(self, audio_url: str | None, uuid: str | None = None) -> None: ...
    @abstractmethod
    def parse_input_audio(
        self, input_audio: InputAudio | None, uuid: str | None = None
    ) -> None: ...
    @abstractmethod
    def parse_audio_embeds(
        self, audio_embeds: str | dict[str, str] | None, uuid: str | None = None
    ) -> None: ...
    @abstractmethod
    def parse_video(self, video_url: str | None, uuid: str | None = None) -> None: ...

class MultiModalContentParser(BaseMultiModalContentParser):
    def __init__(
        self,
        tracker: MultiModalItemTracker,
        mm_processor_kwargs: dict[str, Any] | None = None,
    ) -> None: ...
    @property
    def model_config(self) -> ModelConfig: ...
    def parse_image(self, image_url: str | None, uuid: str | None = None) -> None: ...
    def parse_image_embeds(
        self, image_embeds: str | dict[str, str] | None, uuid: str | None = None
    ) -> None: ...
    def parse_audio_embeds(
        self, audio_embeds: str | dict[str, str] | None, uuid: str | None = None
    ) -> None: ...
    def parse_image_pil(
        self, image_pil: Image.Image | None, uuid: str | None = None
    ) -> None: ...
    def parse_audio(self, audio_url: str | None, uuid: str | None = None) -> None: ...
    def parse_input_audio(
        self, input_audio: InputAudio | None, uuid: str | None = None
    ) -> None: ...
    def parse_video(self, video_url: str | None, uuid: str | None = None) -> None: ...

class AsyncMultiModalContentParser(BaseMultiModalContentParser):
    def __init__(
        self,
        tracker: AsyncMultiModalItemTracker,
        mm_processor_kwargs: dict[str, Any] | None = None,
    ) -> None: ...
    @property
    def model_config(self) -> ModelConfig: ...
    def parse_image(self, image_url: str | None, uuid: str | None = None) -> None: ...
    def parse_image_embeds(
        self, image_embeds: str | dict[str, str] | None, uuid: str | None = None
    ) -> None: ...
    def parse_audio_embeds(
        self, audio_embeds: str | dict[str, str] | None, uuid: str | None = None
    ) -> None: ...
    def parse_image_pil(
        self, image_pil: Image.Image | None, uuid: str | None = None
    ) -> None: ...
    def parse_audio(self, audio_url: str | None, uuid: str | None = None) -> None: ...
    def parse_input_audio(
        self, input_audio: InputAudio | None, uuid: str | None = None
    ) -> None: ...
    def parse_video(self, video_url: str | None, uuid: str | None = None) -> None: ...

@dataclass
class ChatTemplateConfig:
    chat_template: str | None = ...
    chat_template_content_format: ChatTemplateContentFormatOption = ...
    trust_request_chat_template: bool = ...

def validate_chat_template(chat_template: Path | str | None): ...
def load_chat_template(
    chat_template: Path | str | None, *, is_literal: bool = False
) -> str | None: ...

MM_PARSER_MAP: dict[str, Callable[[ChatCompletionContentPartParam], _ContentPart]]
PART_TYPES_TO_SKIP_NONE_CONTENT: Incomplete

def parse_chat_messages(
    messages: list[ChatCompletionMessageParam],
    model_config: ModelConfig,
    content_format: ChatTemplateContentFormat,
    media_io_kwargs: dict[str, dict[str, Any]] | None = None,
    mm_processor_kwargs: dict[str, Any] | None = None,
) -> tuple[
    list[ConversationMessage], MultiModalDataDict | None, MultiModalUUIDDict | None
]: ...
async def parse_chat_messages_async(
    messages: list[ChatCompletionMessageParam],
    model_config: ModelConfig,
    content_format: ChatTemplateContentFormat,
    media_io_kwargs: dict[str, dict[str, Any]] | None = None,
    mm_processor_kwargs: dict[str, Any] | None = None,
) -> tuple[
    list[ConversationMessage], MultiModalDataDict | None, MultiModalUUIDDict | None
]: ...
def get_history_tool_calls_cnt(conversation: list[ConversationMessage]): ...
def make_tool_call_id(id_type: str = "random", func_name=None, idx=None): ...
