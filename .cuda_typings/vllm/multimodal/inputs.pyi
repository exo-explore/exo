import abc
import torch
import torch.types
from .media import MediaWithBytes as MediaWithBytes
from PIL.Image import Image as Image
from abc import ABC, abstractmethod
from collections import UserDict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from functools import cached_property as cached_property
from transformers.feature_extraction_utils import BatchFeature as BatchFeature
from typing import Any, Literal, TypeAlias, TypedDict
from typing_extensions import NotRequired
from vllm.inputs.data import _InputOptions
from vllm.utils.collection_utils import is_list_of as is_list_of
from vllm.utils.import_utils import LazyLoader as LazyLoader
from vllm.utils.jsontree import json_map_leaves as json_map_leaves

HfImageItem: TypeAlias
HfVideoItem: TypeAlias
HfAudioItem: TypeAlias
ImageItem: TypeAlias
VideoItem: TypeAlias
AudioItem: TypeAlias
ModalityData: TypeAlias

class VisionChunkImage(TypedDict):
    type: Literal["image"]
    image: Image
    uuid: str | None

class VisionChunkVideo(TypedDict):
    type: Literal["video_chunk"]
    video_chunk: list[Image]
    uuid: str | None
    prompt: str
    video_idx: int

VisionChunk = VisionChunkImage | VisionChunkVideo

class MultiModalDataBuiltins(TypedDict, total=False):
    image: ModalityData[ImageItem]
    video: ModalityData[VideoItem]
    audio: ModalityData[AudioItem]
    vision_chunk: ModalityData[VisionChunk]

MultiModalDataDict: TypeAlias = Mapping[str, ModalityData[Any]]
MultiModalUUIDDict: TypeAlias = Mapping[str, Sequence[str | None] | str]

@dataclass(frozen=True)
class PlaceholderRange:
    offset: int
    length: int
    is_embed: torch.Tensor | None = ...
    @cached_property
    def embeds_cumsum(self) -> torch.Tensor | None: ...
    def get_num_embeds(self) -> int: ...
    def get_embeds_indices_in_range(
        self, start_idx: int, end_idx: int
    ) -> tuple[int, int]: ...
    def extract_embeds_range(self) -> list[tuple[int, int]]: ...
    def __eq__(self, other: object) -> bool: ...

NestedTensors: TypeAlias

def nested_tensors_equal(a: NestedTensors, b: NestedTensors) -> bool: ...

BatchedTensorInputs: TypeAlias = dict[str, NestedTensors]

def batched_tensors_equal(a: BatchedTensorInputs, b: BatchedTensorInputs) -> bool: ...
@dataclass
class MultiModalFeatureSpec:
    data: MultiModalKwargsItem | None
    modality: str
    identifier: str
    mm_position: PlaceholderRange
    mm_hash: str | None = ...
    @staticmethod
    def gather_kwargs(features: list["MultiModalFeatureSpec"], keys: set[str]): ...

@dataclass
class MultiModalFieldElem:
    data: NestedTensors
    field: BaseMultiModalField
    def __eq__(self, other: object) -> bool: ...

@dataclass(frozen=True, kw_only=True)
class BaseMultiModalField(ABC, metaclass=abc.ABCMeta):
    keep_on_cpu: bool = ...
    @abstractmethod
    def build_elems(
        self, modality: str, key: str, data: NestedTensors
    ) -> Sequence[MultiModalFieldElem]: ...
    def reduce_data(
        self,
        elems: list[MultiModalFieldElem],
        *,
        device: torch.types.Device = None,
        pin_memory: bool = False,
    ) -> NestedTensors: ...

@dataclass(frozen=True, kw_only=True)
class MultiModalBatchedField(BaseMultiModalField):
    def build_elems(
        self, modality: str, key: str, data: NestedTensors
    ) -> Sequence[MultiModalFieldElem]: ...

@dataclass(frozen=True, kw_only=True)
class MultiModalFlatField(BaseMultiModalField):
    slices: Sequence[slice] | Sequence[Sequence[slice]]
    dim: int = ...
    def build_elems(
        self, modality: str, key: str, data: NestedTensors
    ) -> Sequence[MultiModalFieldElem]: ...

@dataclass(frozen=True, kw_only=True)
class MultiModalSharedField(BaseMultiModalField):
    batch_size: int
    def build_elems(
        self, modality: str, key: str, data: NestedTensors
    ) -> Sequence[MultiModalFieldElem]: ...

@dataclass(frozen=True)
class MultiModalFieldConfig:
    @staticmethod
    def batched(modality: str, *, keep_on_cpu: bool = False): ...
    @staticmethod
    def flat(
        modality: str,
        slices: Sequence[slice] | Sequence[Sequence[slice]],
        dim: int = 0,
        *,
        keep_on_cpu: bool = False,
    ): ...
    @staticmethod
    def flat_from_sizes(
        modality: str,
        size_per_item: torch.Tensor,
        dim: int = 0,
        *,
        keep_on_cpu: bool = False,
    ): ...
    @staticmethod
    def shared(modality: str, batch_size: int, *, keep_on_cpu: bool = False): ...
    field: BaseMultiModalField
    modality: str
    def build_elems(
        self, key: str, batch: NestedTensors
    ) -> Sequence[MultiModalFieldElem]: ...

class MultiModalKwargsItem(UserDict[str, MultiModalFieldElem]):
    @staticmethod
    def dummy(nbytes: int = 1): ...
    def get_data(self) -> dict[str, NestedTensors]: ...

class MultiModalKwargsItems(UserDict[str, Sequence[_I]]):
    @staticmethod
    def from_hf_inputs(
        hf_inputs: BatchFeature, config_by_key: Mapping[str, MultiModalFieldConfig]
    ): ...
    def __getitem__(self, modality: str) -> Sequence[_I]: ...
    def require_data(self) -> MultiModalKwargsItems[MultiModalKwargsItem]: ...
    def get_data(
        self, *, device: torch.types.Device = None, pin_memory: bool = False
    ) -> BatchedTensorInputs: ...

MultiModalKwargsOptionalItems: TypeAlias = (
    MultiModalKwargsItems[MultiModalKwargsItem]
    | MultiModalKwargsItems[MultiModalKwargsItem | None]
)
MultiModalHashes = dict[str, list[str]]
MultiModalPlaceholderDict: TypeAlias = Mapping[str, Sequence[PlaceholderRange]]

class MultiModalInputs(_InputOptions):
    type: Literal["multimodal"]
    prompt_token_ids: list[int]
    prompt: NotRequired[str]
    mm_kwargs: MultiModalKwargsOptionalItems
    mm_hashes: MultiModalHashes
    mm_placeholders: MultiModalPlaceholderDict

def mm_inputs(
    prompt_token_ids: list[int],
    mm_kwargs: MultiModalKwargsOptionalItems,
    mm_hashes: MultiModalHashes,
    mm_placeholders: MultiModalPlaceholderDict,
    *,
    prompt: str | None = None,
    cache_salt: str | None = None,
) -> MultiModalInputs: ...

class MultiModalEncDecInputs(MultiModalInputs):
    encoder_prompt_token_ids: list[int]
    encoder_prompt: NotRequired[str]

def mm_enc_dec_inputs(
    encoder_inputs: MultiModalInputs,
    decoder_prompt_token_ids: list[int],
    *,
    decoder_prompt: str | None = None,
) -> MultiModalEncDecInputs: ...
