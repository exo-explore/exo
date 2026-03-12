from ..vllm_inductor_pass import VllmInductorPass as VllmInductorPass
from torch import fx as fx

class PostCleanupPass(VllmInductorPass):
    @VllmInductorPass.time_and_log
    def __call__(self, graph: fx.Graph) -> None: ...
