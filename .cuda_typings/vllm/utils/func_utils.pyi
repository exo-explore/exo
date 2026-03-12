from _typeshed import Incomplete
from collections.abc import Callable, Mapping
from functools import lru_cache
from typing import Any, TypeVar
from typing_extensions import ParamSpec
from vllm.logger import init_logger as init_logger

logger: Incomplete
P = ParamSpec("P")
T = TypeVar("T")
F = TypeVar("F", bound=Callable[..., Any])

def identity(value: T, **kwargs) -> T: ...
def run_once(f: Callable[P, None]) -> Callable[P, None]: ...
def deprecate_args(
    start_index: int,
    is_deprecated: bool | Callable[[], bool] = True,
    additional_message: str | None = None,
) -> Callable[[F], F]: ...
def deprecate_kwargs(
    *kws: str,
    is_deprecated: bool | Callable[[], bool] = True,
    additional_message: str | None = None,
) -> Callable[[F], F]: ...
@lru_cache
def supports_kw(
    callable: Callable[..., object],
    kw_name: str,
    *,
    requires_kw_only: bool = False,
    allow_var_kwargs: bool = True,
) -> bool: ...
def get_allowed_kwarg_only_overrides(
    callable: Callable[..., object],
    overrides: Mapping[str, object] | None,
    *,
    requires_kw_only: bool = True,
    allow_var_kwargs: bool = False,
) -> dict[str, Any]: ...
