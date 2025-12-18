from math import ceil
from typing import Self

from exo.utils.pydantic_ext import CamelCaseModel


class Memory(CamelCaseModel):
    in_bytes: int = 0

    @classmethod
    def from_bytes(cls, val: int) -> Self:
        """Construct a new Memory object from a number of bytes"""
        return cls(in_bytes=val)

    @property
    def in_kb(self) -> int:
        """The approximate kilobytes this memory represents, rounded up. Setting this property rounds to the nearest byte."""
        return ceil(self.in_bytes / 1024)

    @in_kb.setter
    def in_kb(self, val: int):
        """Set this memorys value in kilobytes."""
        self.in_bytes = val * 1024

    @classmethod
    def from_kb(cls, val: int) -> Self:
        """Construct a new Memory object from a number of kilobytes"""
        return cls(in_bytes=val * 1024)

    @classmethod
    def from_float_kb(cls, val: float) -> Self:
        """Construct a new Memory object from a number of kilobytes, rounding where appropriate"""
        return cls(in_bytes=round(val * 1024))

    @property
    def in_mb(self) -> float:
        """The approximate megabytes this memory represents. Setting this property rounds to the nearest byte."""
        return self.in_bytes / (1024**2)

    @in_mb.setter
    def in_mb(self, val: float):
        """Set the megabytes for this memory, rounded to the nearest byte."""
        self.in_bytes = round(val * (1024**2))

    @classmethod
    def from_mb(cls, val: float) -> Self:
        """Construct a new Memory object from a number of megabytes"""
        return cls(in_bytes=round(val * (1024**2)))

    @classmethod
    def from_gb(cls, val: float) -> Self:
        """Construct a new Memory object from a number of megabytes"""
        return cls(in_bytes=round(val * (1024**3)))

    @property
    def in_gb(self) -> float:
        """The approximate gigabytes this memory represents."""
        return self.in_bytes / (1024**3)

    def __add__(self, other: "Memory") -> "Memory":
        return Memory.from_bytes(self.in_bytes + other.in_bytes)

    def __lt__(self, other: Self) -> bool:
        return self.in_bytes < other.in_bytes

    def __le__(self, other: Self) -> bool:
        return self.in_bytes <= other.in_bytes

    def __gt__(self, other: Self) -> bool:
        return self.in_bytes > other.in_bytes

    def __ge__(self, other: Self) -> bool:
        return self.in_bytes >= other.in_bytes
