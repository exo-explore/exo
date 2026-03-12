import abc
import torch
from .audio import (
    AudioResampler as AudioResampler,
    AudioSpec as AudioSpec,
    normalize_audio as normalize_audio,
)
from .inputs import (
    AudioItem as AudioItem,
    HfAudioItem as HfAudioItem,
    HfImageItem as HfImageItem,
    HfVideoItem as HfVideoItem,
    ImageItem as ImageItem,
    ModalityData as ModalityData,
    MultiModalDataDict as MultiModalDataDict,
    MultiModalFieldConfig as MultiModalFieldConfig,
    MultiModalKwargsItems as MultiModalKwargsItems,
    MultiModalUUIDDict as MultiModalUUIDDict,
    VideoItem as VideoItem,
)
from .media import MediaWithBytes as MediaWithBytes
from _typeshed import Incomplete
from abc import ABC, abstractmethod
from collections import UserDict
from collections.abc import Callable, Iterator, Mapping, Sequence, Set as Set
from typing import Any, Generic, Literal, NamedTuple, TypeAlias, TypeGuard
from vllm.utils.collection_utils import is_list_of as is_list_of
from vllm.utils.import_utils import LazyLoader as LazyLoader

class ModalityDataItems(ABC, Generic[_T, _I], metaclass=abc.ABCMeta):
    data: _T
    modality: Incomplete
    def __init__(self, data: _T, modality: str) -> None: ...
    def __len__(self) -> int: ...
    def __getitem__(self, index: int) -> _I: ...
    def __iter__(self) -> Iterator[_I]: ...
    @abstractmethod
    def get_count(self) -> int: ...
    @abstractmethod
    def get(self, index: int) -> _I: ...
    def get_all(self) -> list[_I]: ...
    def get_item_for_hash(self, index: int) -> object: ...
    def get_all_items_for_hash(self) -> list[object]: ...
    @abstractmethod
    def get_processor_data(self) -> Mapping[str, object]: ...
    @abstractmethod
    def get_passthrough_data(self) -> Mapping[str, object]: ...

class ProcessorBatchItems(ModalityDataItems[Sequence[_T], _T]):
    def get_count(self) -> int: ...
    def get(self, index: int) -> _T: ...
    def get_item_for_hash(self, index: int) -> _T | MediaWithBytes[_T]: ...
    def get_processor_data(self) -> Mapping[str, object]: ...
    def get_passthrough_data(self) -> Mapping[str, object]: ...

def validate_embedding_ndim(
    tensor: torch.Tensor, modality: str, index: int | None = None
) -> None: ...

class EmbeddingItems(
    ModalityDataItems[torch.Tensor | list[torch.Tensor], torch.Tensor]
):
    def __init__(
        self,
        data: torch.Tensor | list[torch.Tensor],
        modality: str,
        expected_hidden_size: int | None = None,
    ) -> None: ...
    def get_count(self) -> int: ...
    def get(self, index: int) -> torch.Tensor: ...
    def get_processor_data(self) -> Mapping[str, object]: ...
    def get_passthrough_data(self) -> Mapping[str, object]: ...
    def get_feature_size(self, item_idx: int) -> int: ...

class DictEmbeddingItems(
    ModalityDataItems[Mapping[str, torch.Tensor], Mapping[str, torch.Tensor]]
):
    fields_config: Incomplete
    required_fields: Incomplete
    def __init__(
        self,
        data: Mapping[str, torch.Tensor],
        modality: str,
        required_fields: set[str],
        fields_factory: Callable[
            [Mapping[str, torch.Tensor]], Mapping[str, MultiModalFieldConfig]
        ],
    ) -> None: ...
    def get_count(self) -> int: ...
    def get(self, index: int) -> Mapping[str, torch.Tensor]: ...
    def get_processor_data(self) -> Mapping[str, object]: ...
    def get_passthrough_data(self) -> Mapping[str, object]: ...

class AudioProcessorItems(ProcessorBatchItems[HfAudioItem | None]):
    def __init__(self, data: Sequence[HfAudioItem | None]) -> None: ...
    def get_audio_length(self, item_idx: int) -> int: ...

class AudioEmbeddingItems(EmbeddingItems):
    def __init__(
        self,
        data: torch.Tensor | list[torch.Tensor],
        expected_hidden_size: int | None = None,
    ) -> None: ...

class ImageSize(NamedTuple):
    width: int
    height: int

class ImageProcessorItems(ProcessorBatchItems[HfImageItem | None]):
    def __init__(self, data: Sequence[HfImageItem | None]) -> None: ...
    def get_image_size(self, item_idx: int) -> ImageSize: ...

class ImageEmbeddingItems(EmbeddingItems):
    def __init__(
        self,
        data: torch.Tensor | list[torch.Tensor],
        expected_hidden_size: int | None = None,
    ) -> None: ...

class VideoProcessorItems(ProcessorBatchItems[HfVideoItem | None]):
    metadata: Incomplete
    def __init__(
        self,
        data: Sequence[HfVideoItem | None],
        metadata: dict[str, Any] | list[dict[str, Any] | None] | None = None,
    ) -> None: ...
    def get_num_frames(self, item_idx: int) -> int: ...
    def get_frame_size(self, item_idx: int) -> ImageSize: ...

class VideoEmbeddingItems(EmbeddingItems):
    def __init__(
        self,
        data: torch.Tensor | list[torch.Tensor],
        expected_hidden_size: int | None = None,
    ) -> None: ...

class VisionChunkProcessorItems(ProcessorBatchItems[Any]):
    def __init__(self, data: Sequence[Any]) -> None: ...

class MultiModalDataItems(UserDict[str, ModalityDataItems[Any, Any]]):
    def select(self, modalities: Set[str]): ...
    def get_count(self, modality: str, *, strict: bool = True) -> int: ...
    def get_all_counts(self) -> Mapping[str, int]: ...
    def get_items(self, modality: str, typ: type[_D] | tuple[type[_D], ...]) -> _D: ...

ModalityDataParser: TypeAlias = Callable[
    [ModalityData[Any]], ModalityDataItems[Any, Any] | None
]

class MultiModalDataParser:
    audio_resampler: Incomplete
    target_channels: Incomplete
    video_needs_metadata: Incomplete
    expected_hidden_size: Incomplete
    def __init__(
        self,
        *,
        target_sr: float | None = None,
        target_channels: int | None = None,
        audio_resample_method: Literal["librosa", "scipy"] = "librosa",
        video_needs_metadata: bool = False,
        expected_hidden_size: int | None = None,
    ) -> None: ...
    @classmethod
    def is_embeddings(
        cls, data: object
    ) -> TypeGuard[torch.Tensor | list[torch.Tensor]]: ...
    def parse_mm_data(self, mm_data: MultiModalDataDict) -> MultiModalDataItems: ...

MultiModalUUIDItems: TypeAlias = dict[str, Sequence[str | None]]

def parse_mm_uuids(mm_uuids: MultiModalUUIDDict | None) -> MultiModalUUIDItems: ...
