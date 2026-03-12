from ..fx_utils import is_func as is_func
from ..vllm_inductor_pass import VllmInductorPass as VllmInductorPass
from _typeshed import Incomplete
from torch import fx as fx
from vllm.logger import init_logger as init_logger

logger: Incomplete

class SplitCoalescingPass(VllmInductorPass):
    @VllmInductorPass.time_and_log
    def __call__(self, graph: fx.Graph) -> None: ...
