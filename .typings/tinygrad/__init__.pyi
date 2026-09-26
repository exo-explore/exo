"""Type stub for the optional tinygrad package."""

from tinygrad.dtype import DType, dtypes
from tinygrad.tensor import Tensor

class Device:
    DEFAULT: str

__all__ = ["Device", "DType", "Tensor", "dtypes"]
