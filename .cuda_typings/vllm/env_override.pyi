from _typeshed import Incomplete
from vllm.logger import init_logger as init_logger
from vllm.utils.torch_utils import is_torch_equal as is_torch_equal

logger: Incomplete

def memory_plan_reuse_patched(self): ...
def get_graph_partition_signature_patched(
    self, partitions, skip_cudagraphs: list[bool]
): ...
def should_partition_patched(self, node, should_log: bool = False) -> bool: ...
