from collections.abc import Callable as Callable
from typing import Any

class lazy:
    def __init__(self, factory: Callable[[], Any]) -> None: ...
