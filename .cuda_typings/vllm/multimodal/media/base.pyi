import abc
import numpy as np
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Generic

@dataclass
class MediaWithBytes(Generic[_T]):
    media: _T
    original_bytes: bytes = field(repr=False)
    def __array__(self, *args, **kwargs) -> np.ndarray: ...
    def __getattr__(self, name: str): ...

class MediaIO(ABC, Generic[_T], metaclass=abc.ABCMeta):
    @classmethod
    def merge_kwargs(
        cls,
        default_kwargs: dict[str, Any] | None,
        runtime_kwargs: dict[str, Any] | None,
    ) -> dict[str, Any]: ...
    @abstractmethod
    def load_bytes(self, data: bytes) -> _T: ...
    @abstractmethod
    def load_base64(self, media_type: str, data: str) -> _T: ...
    @abstractmethod
    def load_file(self, filepath: Path) -> _T: ...
