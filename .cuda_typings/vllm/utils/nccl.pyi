from _typeshed import Incomplete
from vllm.logger import init_logger as init_logger

logger: Incomplete

def find_nccl_library() -> str: ...
def find_nccl_include_paths() -> list[str] | None: ...
