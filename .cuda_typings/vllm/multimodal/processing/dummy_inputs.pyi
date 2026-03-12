import abc
from ..inputs import MultiModalDataDict as MultiModalDataDict
from .context import BaseProcessingInfo as BaseProcessingInfo
from .inputs import ProcessorInputs as ProcessorInputs
from _typeshed import Incomplete
from abc import ABC, abstractmethod
from collections.abc import Mapping
from typing import Generic
from vllm.config.multimodal import (
    AudioDummyOptions as AudioDummyOptions,
    BaseDummyOptions as BaseDummyOptions,
    ImageDummyOptions as ImageDummyOptions,
    VideoDummyOptions as VideoDummyOptions,
)
from vllm.logger import init_logger as init_logger

logger: Incomplete

class BaseDummyInputsBuilder(ABC, Generic[_I], metaclass=abc.ABCMeta):
    info: Incomplete
    def __init__(self, info: _I) -> None: ...
    @abstractmethod
    def get_dummy_text(self, mm_counts: Mapping[str, int]) -> str: ...
    @abstractmethod
    def get_dummy_mm_data(
        self,
        seq_len: int,
        mm_counts: Mapping[str, int],
        mm_options: Mapping[str, BaseDummyOptions],
    ) -> MultiModalDataDict: ...
    def get_dummy_processor_inputs(
        self,
        seq_len: int,
        mm_counts: Mapping[str, int],
        mm_options: Mapping[str, BaseDummyOptions],
    ) -> ProcessorInputs: ...
