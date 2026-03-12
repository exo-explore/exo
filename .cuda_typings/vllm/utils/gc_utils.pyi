from _typeshed import Incomplete
from vllm.logger import init_logger as init_logger

logger: Incomplete

class GCDebugConfig:
    enabled: bool
    top_objects: int
    def __init__(self, gc_debug_conf: str | None = None) -> None: ...

class GCDebugger:
    config: Incomplete
    start_time_ns: int
    num_objects: int
    gc_top_collected_objects: str
    def __init__(self, config: GCDebugConfig) -> None: ...
    def handle(self, phase: str, info: dict[str, int]) -> None: ...

def freeze_gc_heap() -> None: ...
def maybe_attach_gc_debug_callback() -> None: ...
