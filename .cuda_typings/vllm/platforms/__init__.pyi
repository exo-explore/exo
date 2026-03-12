from .interface import (
    CpuArchEnum as CpuArchEnum,
    Platform as Platform,
    PlatformEnum as PlatformEnum,
)

__all__ = ["Platform", "PlatformEnum", "current_platform", "CpuArchEnum", "_init_trace"]

_init_trace: str
current_platform: Platform
