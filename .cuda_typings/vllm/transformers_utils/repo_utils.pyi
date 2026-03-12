from _typeshed import Incomplete
from collections.abc import Callable as Callable
from functools import cache
from pathlib import Path
from vllm import envs as envs
from vllm.logger import init_logger as init_logger

logger: Incomplete

def with_retry(
    func: Callable[[], _R], log_msg: str, max_retries: int = 2, retry_delay: int = 2
) -> _R: ...
@cache
def list_repo_files(
    repo_id: str,
    *,
    revision: str | None = None,
    repo_type: str | None = None,
    token: str | bool | None = None,
) -> list[str]: ...
def list_filtered_repo_files(
    model_name_or_path: str,
    allow_patterns: list[str],
    revision: str | None = None,
    repo_type: str | None = None,
    token: str | bool | None = None,
) -> list[str]: ...
def any_pattern_in_repo_files(
    model_name_or_path: str,
    allow_patterns: list[str],
    revision: str | None = None,
    repo_type: str | None = None,
    token: str | bool | None = None,
): ...
def is_mistral_model_repo(
    model_name_or_path: str,
    revision: str | None = None,
    repo_type: str | None = None,
    token: str | bool | None = None,
) -> bool: ...
def file_exists(
    repo_id: str,
    file_name: str,
    *,
    repo_type: str | None = None,
    revision: str | None = None,
    token: str | bool | None = None,
) -> bool: ...
def file_or_path_exists(
    model: str | Path, config_name: str, revision: str | None
) -> bool: ...
def get_model_path(model: str | Path, revision: str | None = None): ...
def get_hf_file_bytes(
    file_name: str, model: str | Path, revision: str | None = "main"
) -> bytes | None: ...
def try_get_local_file(
    model: str | Path, file_name: str, revision: str | None = "main"
) -> Path | None: ...
def get_hf_file_to_dict(
    file_name: str, model: str | Path, revision: str | None = "main"
): ...
