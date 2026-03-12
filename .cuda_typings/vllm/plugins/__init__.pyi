from _typeshed import Incomplete
from collections.abc import Callable
from typing import Any

logger: Incomplete
DEFAULT_PLUGINS_GROUP: str
IO_PROCESSOR_PLUGINS_GROUP: str
PLATFORM_PLUGINS_GROUP: str
STAT_LOGGER_PLUGINS_GROUP: str
plugins_loaded: bool

def load_plugins_by_group(group: str) -> dict[str, Callable[[], Any]]: ...
def load_general_plugins() -> None: ...
