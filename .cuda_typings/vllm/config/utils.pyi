from _typeshed import DataclassInstance, Incomplete
from collections.abc import Callable, Sequence
from pydantic import ConfigDict
from pydantic.fields import Field as PydanticField
from typing import Any, Protocol, TypeVar
from typing_extensions import dataclass_transform
from vllm.logger import init_logger as init_logger

logger: Incomplete
ConfigType = type[DataclassInstance]
ConfigT = TypeVar("ConfigT", bound=DataclassInstance)

@dataclass_transform(field_specifiers=(PydanticField,))
def config(
    cls=None, *, config: ConfigDict | None = None, **kwargs: Any
) -> type[ConfigT] | Callable[[type[ConfigT]], type[ConfigT]]: ...
def get_field(cls, name: str) -> Any: ...
def is_init_field(cls, name: str) -> bool: ...
def replace(dataclass_instance: ConfigT, /, **kwargs) -> ConfigT: ...
def getattr_iter(
    object: object,
    names: Sequence[str],
    default: Any | None = None,
    default_factory: Callable[[], Any] | None = None,
    warn: bool = False,
) -> Any: ...
def get_attr_docs(cls) -> dict[str, str]: ...

class SupportsHash(Protocol):
    def compute_hash(self) -> str: ...

class SupportsMetricsInfo(Protocol):
    def metrics_info(self) -> dict[str, str]: ...

def update_config(config: ConfigT, overrides: dict[str, Any]) -> ConfigT: ...
def normalize_value(x): ...
def get_hash_factors(
    config: ConfigT, ignored_factors: set[str]
) -> dict[str, object]: ...
def hash_factors(items: dict[str, object]) -> str: ...

class Range:
    start: int
    end: int
    def is_single_size(self) -> bool: ...
    def __contains__(self, size: int) -> bool: ...
    def __eq__(self, other: object) -> bool: ...
    def __hash__(self) -> int: ...

def handle_deprecated(
    config: ConfigT,
    old_name: str,
    new_name_or_names: str | list[str],
    removal_version: str,
) -> None: ...
def get_from_deprecated_env_if_set(
    env_name: str, removal_version: str, field_name: str | None = None
) -> str | None: ...
def set_from_deprecated_env_if_set(
    config: ConfigT,
    env_name: str,
    removal_version: str,
    field_name: str,
    to_bool: bool = False,
    to_int: bool = False,
) -> None: ...
