from _typeshed import Incomplete
from vllm import envs as envs
from vllm.assets.base import get_cache_dir as get_cache_dir
from vllm.logger import init_logger as init_logger
from vllm.utils.import_utils import PlaceholderModule as PlaceholderModule

logger: Incomplete
SUPPORTED_SCHEMES: Incomplete
runai_model_streamer: Incomplete

def list_safetensors(path: str = "") -> list[str]: ...
def is_runai_obj_uri(model_or_path: str) -> bool: ...

class ObjectStorageModel:
    dir: Incomplete
    def __init__(self, url: str) -> None: ...
    def pull_files(
        self,
        model_path: str = "",
        allow_pattern: list[str] | None = None,
        ignore_pattern: list[str] | None = None,
    ) -> None: ...
