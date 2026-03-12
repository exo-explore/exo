from _typeshed import Incomplete
from vllm.logger import init_logger as init_logger

logger: Incomplete
CONFIG_HOME: Incomplete
RAY_NON_CARRY_OVER_ENV_VARS_FILE: Incomplete
RAY_NON_CARRY_OVER_ENV_VARS: Incomplete
DEFAULT_ENV_VAR_PREFIXES: set[str]
DEFAULT_EXTRA_ENV_VARS: set[str]

def get_env_vars_to_copy(
    exclude_vars: set[str] | None = None,
    additional_vars: set[str] | None = None,
    destination: str | None = None,
) -> set[str]: ...
