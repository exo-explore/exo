from _typeshed import Incomplete
from typing import Any

class VLLMValidationError(ValueError):
    parameter: Incomplete
    value: Incomplete
    def __init__(
        self, message: str, *, parameter: str | None = None, value: Any = None
    ) -> None: ...

class VLLMNotFoundError(Exception): ...

class LoRAAdapterNotFoundError(VLLMNotFoundError):
    message: str
    def __init__(self, lora_name: str, lora_path: str) -> None: ...
