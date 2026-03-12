import abc
import torch
from ..cache import BaseMultiModalProcessorCache as BaseMultiModalProcessorCache
from ..inputs import (
    MultiModalEncDecInputs as MultiModalEncDecInputs,
    MultiModalFieldConfig as MultiModalFieldConfig,
    MultiModalHashes as MultiModalHashes,
    MultiModalInputs as MultiModalInputs,
    MultiModalKwargsItem as MultiModalKwargsItem,
    MultiModalKwargsItems as MultiModalKwargsItems,
    MultiModalKwargsOptionalItems as MultiModalKwargsOptionalItems,
    PlaceholderRange as PlaceholderRange,
    mm_enc_dec_inputs as mm_enc_dec_inputs,
    mm_inputs as mm_inputs,
)
from ..parse import (
    DictEmbeddingItems as DictEmbeddingItems,
    EmbeddingItems as EmbeddingItems,
    MultiModalDataItems as MultiModalDataItems,
    MultiModalUUIDItems as MultiModalUUIDItems,
)
from .context import (
    BaseProcessingInfo as BaseProcessingInfo,
    TimingContext as TimingContext,
)
from .dummy_inputs import BaseDummyInputsBuilder as BaseDummyInputsBuilder
from .inputs import ProcessorInputs as ProcessorInputs
from _typeshed import Incomplete
from abc import ABC, abstractmethod
from collections.abc import Callable, Generator, ItemsView, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
from transformers.feature_extraction_utils import BatchFeature as BatchFeature
from typing import Generic, NamedTuple, Protocol, TypeAlias
from vllm.logger import init_logger as init_logger
from vllm.tokenizers import TokenizerLike as TokenizerLike
from vllm.utils.collection_utils import (
    flatten_2d_lists as flatten_2d_lists,
    full_groupby as full_groupby,
)

logger: Incomplete
PromptSeq: TypeAlias = str | list[int]

class _GetMatchIndex(Protocol):
    def __call__(
        self, tokenizer: TokenizerLike | None, prompt: PromptSeq, start_idx: int = 0
    ) -> int | None: ...

@dataclass
class PromptIndex:
    get_match_index: _GetMatchIndex

class PromptIndexTargets:
    @staticmethod
    def start() -> PromptIndex: ...
    @staticmethod
    def prefix(seq: PromptSeq) -> PromptIndex: ...
    @staticmethod
    def end() -> PromptIndex: ...

UpdateTarget: TypeAlias = PromptSeq | PromptIndex
PromptUpdateTarget: TypeAlias = Callable[[int], UpdateTarget] | UpdateTarget

@dataclass
class PromptUpdateDetails(Generic[_S]):
    full: _S
    is_embed: Callable[[TokenizerLike | None, PromptSeq], torch.Tensor] | None = ...
    @staticmethod
    def from_seq(seq: _S) -> PromptUpdateDetails[_S]: ...
    @staticmethod
    def select_text(seq: _S, embed_text: str) -> PromptUpdateDetails[_S]: ...
    @staticmethod
    def select_token_id(seq: _S, embed_token_id: int) -> PromptUpdateDetails[_S]: ...
    @staticmethod
    def select_token_ids(
        seq: _S, embed_token_ids: list[int]
    ) -> PromptUpdateDetails[_S]: ...

PromptUpdateInfo: TypeAlias = PromptSeq | PromptUpdateDetails
PromptUpdateContent: TypeAlias = Callable[[int], PromptUpdateInfo] | PromptUpdateInfo

class UpdateMode(str, Enum):
    INSERT = "insert"
    REPLACE = "replace"

@dataclass
class PromptUpdate(ABC, metaclass=abc.ABCMeta):
    modality: str
    target: PromptUpdateTarget
    @property
    @abstractmethod
    def content(self) -> PromptUpdateContent: ...
    @property
    @abstractmethod
    def mode(self) -> UpdateMode: ...
    def resolve(self, item_idx: int) -> ResolvedPromptUpdate: ...

@dataclass
class PromptInsertion(PromptUpdate):
    insertion: PromptUpdateContent = field(repr=False)
    @property
    def content(self) -> PromptUpdateContent: ...
    @property
    def mode(self) -> UpdateMode: ...

@dataclass
class PromptReplacement(PromptUpdate):
    replacement: PromptUpdateContent = field(repr=False)
    @property
    def content(self) -> PromptUpdateContent: ...
    @property
    def mode(self) -> UpdateMode: ...

class _HasModalityAttr(Protocol):
    modality: str

class _HasModalityProp(Protocol):
    @property
    def modality(self) -> str: ...

def full_groupby_modality(values: Iterable[_M]) -> ItemsView[str, list[_M]]: ...

class PromptTargetMatch(NamedTuple):
    start_idx: int
    end_idx: int

@dataclass(frozen=True)
class ResolvedPromptUpdate:
    modality: str
    item_idx: int
    mode: UpdateMode
    target: UpdateTarget
    content: PromptUpdateDetails = field(repr=False)
    def iter_token_matches(
        self, prompt: list[int], tokenizer: TokenizerLike | None, *, start_idx: int = 0
    ) -> Generator[PromptTargetMatch]: ...
    def iter_text_matches(
        self, prompt: str, tokenizer: TokenizerLike | None, *, start_idx: int = 0
    ) -> Generator[PromptTargetMatch]: ...
    def iter_matches(
        self,
        prompt: list[int] | str,
        tokenizer: TokenizerLike | None,
        *,
        start_idx: int = 0,
    ) -> Generator[PromptTargetMatch]: ...
    def with_target(self, target: UpdateTarget): ...
    def with_content(self, content: PromptUpdateInfo): ...

class _TokenMatch(NamedTuple):
    start_idx: int
    end_idx: int

def iter_token_matches(
    token_ids: list[int], match_ids: list[int], *, start_idx: int = 0
) -> Generator[_TokenMatch]: ...
def replace_token_matches(
    token_ids: list[int], match_ids: list[int], new_ids: list[int]
) -> list[int]: ...
@dataclass
class PlaceholderFeaturesInfo:
    modality: str
    item_idx: int
    start_idx: int
    tokens: list[int]
    is_embed: torch.Tensor | None
    @property
    def length(self) -> int: ...
    def to_range(self) -> PlaceholderRange: ...

def apply_token_matches(
    prompt: list[int],
    mm_prompt_updates: MultiModalPromptUpdates,
    tokenizer: TokenizerLike | None,
) -> tuple[list[int], "MultiModalPromptUpdatesApplyResult"]: ...
def apply_text_matches(
    prompt: str,
    mm_prompt_updates: MultiModalPromptUpdates,
    tokenizer: TokenizerLike | None,
) -> tuple[str, "MultiModalPromptUpdatesApplyResult"]: ...
def find_mm_placeholders(
    prompt: list[int],
    mm_prompt_updates: MultiModalPromptUpdates,
    tokenizer: TokenizerLike | None,
) -> Mapping[str, list[PlaceholderFeaturesInfo]]: ...

MultiModalIsCached = dict[str, list[bool]]
MultiModalPromptUpdates = Mapping[str, list[Sequence[ResolvedPromptUpdate]]]
MultiModalPromptUpdatesApplyResult = Mapping[str, list[int | None]]

class MultiModalProcessingInfo(NamedTuple):
    kwargs: MultiModalKwargsOptionalItems
    hashes: MultiModalHashes
    prompt_updates: MultiModalPromptUpdates

class BaseMultiModalProcessor(ABC, Generic[_I], metaclass=abc.ABCMeta):
    info: Incomplete
    dummy_inputs: Incomplete
    cache: Incomplete
    data_parser: Incomplete
    def __init__(
        self,
        info: _I,
        dummy_inputs: BaseDummyInputsBuilder[_I],
        *,
        cache: BaseMultiModalProcessorCache | None = None,
    ) -> None: ...
    def __call__(
        self,
        prompt: str,
        mm_items: MultiModalDataItems,
        mm_uuid_items: MultiModalUUIDItems | None = None,
        hf_processor_mm_kwargs: Mapping[str, object] | None = None,
    ) -> MultiModalInputs: ...
    def apply(
        self, inputs: ProcessorInputs, timing_ctx: TimingContext
    ) -> MultiModalInputs: ...

class EncDecMultiModalProcessor(BaseMultiModalProcessor[_I], metaclass=abc.ABCMeta):
    @abstractmethod
    def create_encoder_prompt(
        self, prompt: str | list[int], mm_items: MultiModalDataItems
    ) -> str | list[int]: ...
    def create_decoder_prompt(
        self, prompt: str | list[int], mm_items: MultiModalDataItems
    ) -> str | list[int]: ...
    def apply(
        self, inputs: ProcessorInputs, timing_ctx: TimingContext
    ) -> MultiModalEncDecInputs: ...
