import ast
from collections.abc import Callable

from exo.security_gate.issue import Issue

from .async_hazards import check_async_hazards
from .dangerous_calls import check_dangerous_calls
from .dict_mutation import check_dict_mutation
from .event_sourcing import check_event_sourcing
from .network_exposure import check_network_exposure
from .pydantic_mutation import check_pydantic_mutation
from .secrets import check_secrets

__all__ = ["Issue", "ALL_CHECKS"]

ALL_CHECKS: list[Callable[[str, str, ast.Module], list[Issue]]] = [
    check_secrets,
    check_dangerous_calls,
    check_async_hazards,
    check_pydantic_mutation,
    check_event_sourcing,
    check_network_exposure,
    check_dict_mutation,
]
