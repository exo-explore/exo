import os
from _typeshed import Incomplete
from vllm.logger import init_logger as init_logger

logger: Incomplete

def try_get_class_from_dynamic_module(
    class_reference: str,
    pretrained_model_name_or_path: str,
    trust_remote_code: bool,
    cache_dir: str | os.PathLike | None = None,
    force_download: bool = False,
    resume_download: bool | None = None,
    proxies: dict[str, str] | None = None,
    token: bool | str | None = None,
    revision: str | None = None,
    local_files_only: bool = False,
    repo_type: str | None = None,
    code_revision: str | None = None,
    warn_on_fail: bool = True,
    **kwargs,
) -> type | None: ...
